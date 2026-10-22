#!/usr/bin/env python3
"""
test_embed_media.py - does your embedding endpoint really embed images?

mailbox.py sends media as ONE combined object and quietly downgrades to the
attachment's filename if the endpoint rejects that shape (EmbeddingClient
.embed_media / MultimodalRejected). That fallback is what makes a broken image
setup invisible, so this script probes the endpoint explicitly and tells you
which case you are in:

  A. images really change the vector           -> attachment content searchable
  B. image input is accepted but the vector is unchanged
                                               -> you only search filenames
  C. image input is rejected                   -> mailbox.py uses text labels

Cases B and C look identical from `index` output, which is why this exists.
Every media item shape in use is tried, and each one's server error is printed
verbatim; if one is accepted, that is the value for `media_item_shape`:

  media_item_shape="image_url"  {"type": "image_url",
                                 "image_url": {"url": "data:..." ...}, "text": ...}
  media_item_shape="content"    {"content": [{"type": "image_url", ...},
                                             {"type": "text", ...}]}
  native /api/embed             {"input": "text", "images": ["base64..."]}

If your config points at Ollama's native endpoint, the same host's
/v1/embeddings is probed too, so you can see what switching would buy.

  python test_embed_media.py
  python test_embed_media.py --endpoint http://localhost:11434/v1/embeddings
  python test_embed_media.py --image C:\\path\\to\\shot.png --query "bar chart"

Exit code: 0 = case A, 1 = case B or C (no real image embedding),
2 = endpoint unreachable or reply unusable.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests

import mailbox as mb


def make_png(kind: str) -> bytes:
    """Synthetic, deterministic 'attachments' that a mail folder would hold."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    if kind == "dashboard":
        d.rectangle([0, 0, 640, 60], fill=(30, 60, 120))
        d.text((16, 22), "DASHBOARD  revenue 42,000", fill="white")
        for i, h in enumerate((120, 210, 90, 260, 170)):
            d.rectangle([40 + i * 110, 340 - h, 120 + i * 110, 340], fill=(40, 140, 70))
        d.text((40, 355), "Q1 Q2 Q3 Q4 Q5", fill="black")
    elif kind == "circle":
        d.ellipse([120, 60, 520, 360], fill=(200, 40, 40), outline="black", width=4)
        d.text((300, 200), "STOP", fill="white")
    else:  # 'table'
        d.text((20, 20), "INVOICE 2026-0041", fill="black")
        for row in range(8):
            y = 60 + row * 36
            d.line([20, y, 620, y], fill=(180, 180, 180))
            d.text((24, y + 8), f"item {row}   amount {row * 137}.00", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def cos(a, b) -> float:
    return float(unit(a) @ unit(b))


def media_chunk(name: str, data: bytes, text: str | None, mime="image/png") -> dict:
    """The chunk dict EmbeddingClient.embed_media expects (text=None means an
    image-only item, with no text part in the request object)."""
    return {"kind": "attachment", "src": f"attachment: {name}", "tier": mb.MEDIA_TIER,
            "label": f"attachment: {name}", "text": text or "",
            "image_b64": b64(data), "mime": mime}


def image_url_part(data: bytes, mime: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64(data)}"}}


def sibling_url(url: str, path: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, path, "", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe image embedding support")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--endpoint")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--dim", type=int)
    ap.add_argument("--image", help="also probe this image file")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds per request (3 attempts are made; a cold model "
                         "load can take longer than the first reply deserves)")
    ap.add_argument("--query", default="screenshot of a dashboard with revenue bars")
    ap.add_argument("--dump", action="store_true", help="print raw reply keys")
    args = ap.parse_args()

    cfg = dict(mb.load_config(args.config))
    for key, val in (("embed_endpoint", args.endpoint), ("embed_model", args.model),
                     ("embed_api_key", args.api_key), ("dim", args.dim)):
        if val:
            cfg[key] = val
    cfg["embed_timeout"] = args.timeout
    if not cfg.get("embed_endpoint"):
        print("no embed_endpoint (set it in config.json or pass --endpoint)")
        return 2

    client = mb.EmbeddingClient(cfg)
    dialect = "Ollama native /api/embed" if client.native else "OpenAI-compatible"
    print(f"endpoint: {cfg['embed_endpoint']}")
    print(f"dialect:  {dialect}   (media_item_shape={cfg.get('media_item_shape')!r})")
    print(f"model:    {cfg.get('embed_model') or '(omitted)'}")
    print(f"dim:      {client.dim}   timeout: {args.timeout:g}s")
    print(f"prompts:  {cfg.get('query_prompt')!r} / {cfg.get('doc_prompt')!r}\n")

    images = {"dashboard.png": make_png("dashboard"),
              "circle.png": make_png("circle"),
              "invoice-table.png": make_png("table")}
    mime_of = {k: "image/png" for k in images}
    if args.image:
        try:
            with open(args.image, "rb") as fh:
                images["yours.jpg"] = fh.read()
            mime_of["yours.jpg"] = "image/jpeg"
            print(f"probing your file: {args.image} "
                  f"({len(images['yours.jpg']) / 1024:.0f} KB)")
        except OSError as e:
            print(f"could not read --image {args.image}: {e}")

    # 1. text baseline -----------------------------------------------------
    try:
        tv = client.embed_texts(["screenshot of a dashboard with revenue bars",
                                 "invoice 2026-0041 for the flexa project"])
    except Exception as e:
        print(f"[1] TEXT BASELINE FAILED: {type(e).__name__}: {e}")
        print("    Fix the endpoint first: nothing below can work until plain "
              "text embedding answers.")
        return 2
    print(f"[1] text baseline OK: {tv.shape[0]} vectors of dim {tv.shape[1]}"
          + ("" if tv.shape[1] == client.dim else
             f"  WARNING: != configured dim {client.dim}"))

    label = "attachment: dashboard.png"
    raw = images["dashboard.png"]

    # 2. every media item shape this endpoint's dialect knows --------------
    print(f"\n[2] media item shapes ({dialect})")
    accepted_shape, raw_vec = None, None

    def try_item(name, item):
        """POST one media item and report the server's own words."""
        nonlocal accepted_shape, raw_vec
        try:
            data = client._post(client._body([item]), multimodal=True)
            if args.dump:
                print(f"      reply keys: {sorted(data)}")
            vec = client._vectors(data)[0]
            print(f"    {name:34s} ACCEPTED")
            if accepted_shape is None:
                accepted_shape, raw_vec = name, vec
            return vec
        except mb.MultimodalRejected as e:
            print(f"    {name:34s} REJECTED  {e}")
        except Exception as e:
            print(f"    {name:34s} FAILED    {type(e).__name__}: {e}")
        return None

    if client.native:
        body = {"input": label, "images": [b64(raw)]}
        if client.model:
            body["model"] = client.model
        try:
            data = client._post(body, multimodal=True)
            raw_vec = client._vectors(data)[0]
            accepted_shape = "native_images"
            print(f"    {'/api/embed + images[]':34s} ACCEPTED")
        except Exception as e:
            print(f"    {'/api/embed + images[]':34s} FAILED    {e}")
    else:
        for shape in ("image_url", "content"):
            client.media_shape = shape
            part = image_url_part(raw, "image/png")
            item = ({"content": [part, {"type": "text",
                                        "text": f"{cfg['doc_prompt']}: {label}"}]}
                    if shape == "content" else
                    {**part, "text": f"{cfg['doc_prompt']}: {label}"})
            try_item(f"media_item_shape={shape!r}", item)
        client.media_shape = cfg.get("media_item_shape", "image_url")

    # If the config uses Ollama's native endpoint, show what the OpenAI-style
    # endpoint on the SAME host accepts - that is the switch people try next.
    if client.native:
        alt = sibling_url(cfg["embed_endpoint"], "/v1/embeddings")
        print(f"\n    same host, OpenAI shape: {alt}")
        for shape in ("image_url", "content"):
            part = image_url_part(raw, "image/png")
            body = {"input": [{"content": [part, {"type": "text", "text": label}]}
                              if shape == "content" else
                              {**part, "text": label}]}
            if client.model:
                body["model"] = client.model
            try:
                r = requests.post(alt, json=body, timeout=args.timeout)
                note = "ACCEPTED" if r.status_code < 400 else \
                    f"REJECTED  HTTP {r.status_code}: {r.text[:120]}"
                print(f"    {'media_item_shape=' + repr(shape):34s} {note}")
            except Exception as e:
                print(f"    {'media_item_shape=' + repr(shape):34s} FAILED    {e}")

    # Steps 3-5 must measure the shape that was actually accepted, not the one
    # config.json happens to name - otherwise a working endpoint would be
    # judged by a rejected envelope.
    if accepted_shape and accepted_shape.startswith("media_item_shape"):
        client.media_shape = accepted_shape.split("=")[-1].strip("'")
        client.multimodal_ok = None
        print(f"    -> steps 3-5 use media_item_shape={client.media_shape!r}")

    # 3. is the image ACTUALLY used? Same text, different image, no image --
    #    Byte-identical vectors prove the bytes went nowhere.
    print("\n[3] does the image change the vector?")
    image_is_used = False
    if accepted_shape == "native_images":
        base = {"input": label}
        blue = {"input": label, "images": [b64(make_png("circle"))]}
        if client.model:
            base["model"] = blue["model"] = client.model
        plain = client._vectors(client._post(base))[0]
        other = client._vectors(client._post(blue))[0]
        print(f"    cos(dashboard vs no image) = {cos(raw_vec, plain):.6f}")
        print(f"    cos(dashboard vs circle)   = {cos(raw_vec, other):.6f}")
        image_is_used = not (np.array_equal(raw_vec, plain)
                             or np.array_equal(raw_vec, other))
        if not image_is_used:
            print("    -> byte-identical: the endpoint IGNORES the image bytes.")
    elif accepted_shape:
        img_only = client.embed_media(media_chunk("dashboard.png", raw, None))
        combined = client.embed_media(media_chunk("dashboard.png", raw, label))
        filename_only = client.embed_texts([f"{cfg['doc_prompt']}: {label}"])[0]
        print(f"    cos(image only, image+text)  = {cos(img_only, combined):.4f}")
        print(f"    cos(image+text, filename)    = {cos(combined, filename_only):.6f}")
        image_is_used = cos(combined, filename_only) < 0.999
        if not image_is_used:
            print("    -> the reply equals the filename-only vector: pixels ignored.")
    else:
        print("    skipped: no shape carrying image input was accepted.")

    # 4. the exact call index/search make ----------------------------------
    print("\n[4] EmbeddingClient.embed_media(), the call indexing uses")
    combined = client.embed_media(media_chunk("dashboard.png", raw, label))
    print(f"    multimodal_ok = {client.multimodal_ok} "
          f"({'image content embedded' if client.multimodal_ok else 'TEXT LABELS'}"
          + (", native endpoint" if client.native else "") + ")")

    # 5. cross-modal retrieval --------------------------------------------
    print(f"\n[5] cross-modal retrieval, query: {args.query!r}")
    vecs = {name: client.embed_media(media_chunk(name, data, f"attachment: {name}",
                                                 mime_of[name]))
            for name, data in images.items()}
    qv = client.embed_query(args.query)
    for i, (name, v) in enumerate(sorted(vecs.items(), key=lambda kv: -cos(qv, kv[1])), 1):
        print(f"    {i}. {cos(qv, v):+.4f}  {name}")
    if not image_is_used:
        print("    (this ranking is filename similarity, not image content)")

    # 6. verdict -----------------------------------------------------------
    print("\n--- verdict ---")
    if image_is_used:
        print("OK: images really are embedded. Attachment content is searchable; "
              "media chunks get tier 'media' and are reachable through the dense "
              "leg of the search.")
        if accepted_shape and accepted_shape != cfg.get("media_item_shape"):
            print(f"Set \"media_item_shape\": \"{accepted_shape.split('=')[-1].strip(chr(39))}\" "
                  f"in config.json to use the shape this endpoint accepted.")
        return 0
    if not accepted_shape:
        print("IMAGES ARE NOT EMBEDDED - no media item shape was accepted; the "
              "server's own error messages are printed above, verbatim.")
        print("mailbox.py indexes attachment names/types only. To search image "
              "content, serve a multimodal model that accepts image input: "
              "embed_server.py, then point embed_endpoint at its /v1/embeddings.")
    else:
        print("IMAGES ARE NOT EMBEDDED - the endpoint accepts image input and "
              "ignores it (identical vectors with, without and with a different "
              "image).")
        if client.native:
            print("This is Ollama's /api/embed: its `images` field is not applied "
                  "to embedding models. Run this script with "
                  "--endpoint .../v1/embeddings to see what the OpenAI shape on "
                  "the same host accepts; mailbox.py detects all of this and "
                  "indexes attachments as text rather than storing filename "
                  "vectors under a tier that implies otherwise.")
        print("Use embed_server.py (sentence-transformers multimodal encode with "
              "a PIL image) for real attachment-content search.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
