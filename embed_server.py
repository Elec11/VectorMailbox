#!/usr/bin/env python3
"""
embed_server.py - Multimodal-capable embedding endpoint for EmbeddingGemma 2.

Drop-in replacement for a text-only wrapper around
SentenceTransformer("google/embeddinggemma-2"). Serves the OpenAI-compatible
contract that mailbox.py speaks:

  POST /v1/embeddings
  Authorization: Bearer <key>        (only checked when --api-key is set)
  {"input": <string | array of (string | object)>}
  -> {"data": [{"embedding": [ ... 768 floats ... ]}]}

Object items (multimodal, decoded from data: URLs):
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64,..."}}
  {"type": "text", "text": "..."}

An item may carry both a media key and a "text" key; text and media are
then combined into a single embedding (the sentence-transformers multimodal
dict from Google's EmbeddingGemma 2 multimodal guide).

Setup:
  pip install -U "sentence-transformers[image,audio,video]" transformers fastapi uvicorn
  pip install librosa        # optional, mp3/m4a decoding fallback
  python embed_server.py --port 1234 [--api-key sk-...] [--device auto|cuda|cpu]

Notes:
  - The device is selected explicitly and logged at load time: by default
    it uses CUDA when available, else CPU (pass --device to force). It
    runs in bfloat16 on GPU / float32 on CPU. Do NOT force float16:
    EmbeddingGemma 2 activations are incompatible with it. If it lands on
    CPU, torch.cuda.is_available() is False - check that torch is a CUDA
    build (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121).
  - Audio is decoded to a float32 waveform; the model resamples to 16 kHz.
  - Images are passed through as PIL images (mailbox.py already resizes
    them to <=1024 px). The vision token budget can be tuned per call via
    processing_kwargs={"image": {"max_soft_tokens": 70|140|280|560|1120}}.
  - Video is not wired up (mail has no video attachments); the
    sentence-transformers[image,audio,video] extras install torchcodec
    anyway, so a "video_url" branch is easy to add later.
  - Task prompts: the current client sends them as text ('title: … | text: …',
    'task: search result | query: …') and those strings are encoded exactly as
    received. Only the older 'Document: ' / 'SearchQuery: ' prefixes a
    --profile legacy client sends are mapped onto sentence-transformers'
    prompt_name (v6.1.0+); older versions fall back to the raw strings.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import tempfile
import threading

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

log = logging.getLogger("embed_server")

MODEL_ID = "google/embeddinggemma-2"

_model = None
_model_lock = threading.Lock()
_device = "auto"  # "auto" | "cuda" | "cpu"; set by main()


def get_model():
    """Load the model once (it is ~740M params; loading takes a while).

    The device is chosen explicitly and logged, so a CPU-only torch build
    (torch.cuda.is_available() == False) is visible instead of silently
    running on CPU. bfloat16 on GPU (float16 is incompatible with
    EmbeddingGemma 2 activations); float32 on CPU.
    """
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            cuda_available = torch.cuda.is_available()
            dev = _device
            if dev == "auto":
                dev = "cuda" if cuda_available else "cpu"
            log.info(
                "loading %s (requested=%s, cuda_available=%s, device=%s)",
                MODEL_ID, _device, cuda_available, dev,
            )
            dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
            _model = SentenceTransformer(
                MODEL_ID,
                device=dev,
                model_kwargs={"torch_dtype": dtype},
            )
            log.info("model ready (device=%s, dtype=%s)", _model.device, dtype)
        return _model


def decode_data_url(url: str) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise HTTPException(status_code=400, detail="only data: URLs are supported")
    _, _, b64 = url.partition(",")
    try:
        return base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base64 in data URL: {e}")


def decode_image(raw: bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB") if img.mode not in ("RGB", "L") else img


def decode_audio(raw: bytes) -> np.ndarray:
    """Decode audio bytes to a float32 waveform (any sample rate; the model
    resamples to 16 kHz). Tries soundfile first (wav/flac/ogg), then
    librosa (mp3/m4a/... via audioread)."""
    try:
        import soundfile as sf
        arr, _ = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr
    except Exception:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".audio") as f:
            f.write(raw)
            f.flush()
            arr, _ = librosa.load(f.name, sr=None, mono=True)
        return arr


def item_to_encode_input(item) -> object:
    """Convert one request item to what model.encode() understands: a plain
    string, or a multimodal dict with text/image/audio keys."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise HTTPException(
            status_code=400, detail=f"unsupported input item: {type(item)!r}"
        )
    t = item.get("type")
    payload: dict = {}
    if t == "image_url":
        try:
            payload["image"] = decode_image(decode_data_url(item["image_url"]["url"]))
        except KeyError:
            raise HTTPException(status_code=400, detail="image_url.url is missing")
    elif t == "audio_url":
        try:
            payload["audio"] = decode_audio(decode_data_url(item["audio_url"]["url"]))
        except KeyError:
            raise HTTPException(status_code=400, detail="audio_url.url is missing")
    elif t == "text":
        return item.get("text", "")
    else:
        raise HTTPException(status_code=400, detail=f"unknown item type: {t!r}")
    if item.get("text"):
        # Combined multimodal dict: one embedding for text + media.
        payload["text"] = item["text"]
    return payload


def encode_inputs(model, inputs: list) -> object:
    """Encode a batch.

    The current client sends the prompt *as text*, which is what EmbeddingGemma
    2 was trained on: 'title: <subject> | text: <content>' for a document and
    'task: search result | query: <query>' for a query. Those strings are
    passed through untouched - stripping them, or letting sentence-transformers
    add its own prompt on top, moves the vectors into a different part of the
    space than the one mailbox.py compares against.

    The older 'Document: ' / 'SearchQuery: ' prefixes are still mapped to
    sentence-transformers' prompt_name (v6.1.0+), which is how that generation
    of the model received task prompts; a --profile legacy client sends those.
    Falls back to the raw strings if the installed version has no prompt_name
    or if a batch mixes both prompts."""
    prompts = []
    for i in inputs:
        p = None
        if isinstance(i, str):
            for name in ("Document", "SearchQuery"):
                if i.startswith(name + ": "):
                    p = name
                    break
        prompts.append(p)
    uniq = {p for p in prompts if p}
    prompt = uniq.pop() if len(uniq) == 1 else None
    stripped = [
        i[len(p) + 2:] if (p is not None and prompt is not None) else i
        for i, p in zip(inputs, prompts)
    ]
    if prompt is not None:
        try:
            return model.encode(stripped, prompt_name=prompt, convert_to_numpy=True)
        except TypeError:
            pass
    return model.encode(stripped, convert_to_numpy=True)


class EmbedRequest(BaseModel):
    input: object  # str | list[str | dict]
    model: str | None = None
    encoding_format: str | None = None


def make_app(api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="EmbeddingGemma 2 multimodal embeddings")

    @app.post("/v1/embeddings")
    def embeddings(req: EmbedRequest, authorization: str | None = Header(None)):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid api key")
        items = req.input if isinstance(req.input, list) else [req.input]
        if not items:
            raise HTTPException(status_code=400, detail="empty input")
        model = get_model()
        inputs = [item_to_encode_input(i) for i in items]
        vecs = encode_inputs(model, inputs)
        return {
            "data": [
                {"embedding": np.asarray(v, dtype=np.float32).tolist()}
                for v in vecs
            ]
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_ID}

    return app


def main() -> None:
    global _device
    ap = argparse.ArgumentParser(
        description="Multimodal EmbeddingGemma 2 embedding server"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument(
        "--api-key", default=None, help="require this Bearer key (off by default)"
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="device to load the model on (default: auto = cuda if available, else cpu)",
    )
    args = ap.parse_args()
    _device = args.device

    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run(make_app(args.api_key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
