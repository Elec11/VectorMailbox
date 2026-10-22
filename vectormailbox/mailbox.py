#!/usr/bin/env python3
"""
mailbox.py - Hybrid-searchable Gmail inbox (dense + BM25, multi-resolution).

Indexes Gmail (IMAP, app password) into a local SQLite + numpy vector store
using EmbeddingGemma 2 via a remote embedding endpoint. Bodies, metadata
(sender / recipients / attachment extensions), attachments (images, PDFs,
Office files, optionally audio) and thread context (prior mails) are all
indexed and searchable. Per batch, all message text (body + metadata) is
embedded before any attachment is processed. No LLM is required.

Two things make retrieval strong, and both are index-side:

  1. MULTI-RESOLUTION CHUNKING - every source text (a body, or the text of
     one attachment) is indexed at three granularities: `small` (a paragraph
     window: finds the exact sentence), `large` (a section: the model's sweet
     spot) and `xlarge` (the whole source: what the mail is about). A tier is
     only written when it adds information, so a short mail is still one
     chunk and a long mail is a coarse chunk plus fine windows.

  2. HYBRID SEARCH - the dense cosine leg is fused with a lexical BM25 leg
     (SQLite FTS5, porter stemmer) over chunk text plus message metadata
     (From / To / Cc / Subject / attachment filenames / date). The two legs
     are merged with Reciprocal Rank Fusion, which needs no score
     normalisation: FTS5 bm25 magnitudes and cosine similarities are not
     comparable, their ranks are. Exact strings (invoice numbers, surnames,
     acronyms, filenames) come from BM25; paraphrase and images come from
     the vectors.

Commands:
  python mailbox.py index [--full] [--folders INBOX,SENT] [--limit N]
  python mailbox.py search "query" [--mode hybrid|semantic|lexical]
      [--top 10] [--time-weight] [--alpha 0.3] [--tiers small,large]
      [--pool 200] [--w-dense 0.5] [--w-lex 0.5] [--rrf-k 60] [--agg-k 3]
      [--explain] [--json]
  python mailbox.py reindex [--tiers small,large] [--rebuild-fts] [--re-embed]
      [--retier] [--dry-run] [--limit N]
  python mailbox.py stats

`reindex` re-derives chunk tiers and/or the FTS index from the text already
in the database - no IMAP round-trip. `--rebuild-fts` costs zero embeddings,
so hybrid search can be switched on for an existing index instantly.
`--re-embed` rebuilds the stored text vectors after a change of embedding
model; media vectors cannot be rebuilt offline (their bytes are not stored),
so those chunks wait for `index --full` to re-fetch them.

Config: config.json (see config.example.json) or environment variables
GMAIL_USER, GMAIL_APP_PASSWORD, EMBED_ENDPOINT, EMBED_API_KEY, EMBED_MODEL.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

import numpy as np
import requests
import tqdm

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    # Optional by design: 'from:flexa factuur van vorige week' is better with
    # its filters parsed out, but search must work if the module is missing.
    from . import queryparse
except ImportError:  # pragma: no cover
    queryparse = None

log = logging.getLogger("mailbox")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "gmail_user": "",
    "gmail_app_password": "",
    "imap_host": "imap.gmail.com",
    "imap_port": 993,
    "imap_timeout": 90,             # seconds one IMAP read may take. Per read,
                                    # not per fetch: a 40 MB batch over a slow
                                    # link keeps resetting it, while a Gmail that
                                    # has gone quiet mid-command stops costing a
                                    # request that never finishes (0: no limit)
    "smtp_host": "smtp.gmail.com",   # stored for future use; v1 is read-only
    "smtp_port": 465,
    "embed_endpoint": "",            # e.g. https://your-host/v1/embeddings
    "embed_api_key": "",
    "embed_model": "",               # model id some hosts require (e.g. Ollama); "" omits it
    "embed_timeout": 120,            # seconds per endpoint request (lower it to
                                     # fail fast when the host is unreachable)
    "db_path": "mailbox.db",
    "folders": ["INBOX"],
    "batch_size": 32,               # embeddings per endpoint request
    "fetch_batch_size": 10,         # messages per IMAP FETCH (Gmail rejects large batches)
    # --- reconciliation: keeping the index in step with mail you deleted ----
    # Nothing in this program deletes mail, and nothing ever will: the IMAP
    # verbs it sends are listed on GmailFetcher. `prune_deleted` removes local
    # rows for messages the folder no longer holds - the mail is already gone
    # from Gmail, and what is dropped is the index's description of it.
    "prune_deleted": True,          # `index` reconciles after fetching (off: `--no-prune`)
    "prune_max_fraction": 0.25,     # refuse when more than this share of one
                                    # folder vanished in a single run: a listing
                                    # that came back truncated would otherwise
                                    # take the whole index with it
    "prune_restore": True,          # a message that came back to the folder
                                    # (un-deleted from the bin) is re-fetched by
                                    # UID even though it is older than the
                                    # incremental watermark
    "reconcile_batch_size": 500,    # UIDs per command in the reconcile sweep. It
                                    # asks for headers only (no bodies), so the
                                    # 500-message ceiling that applies to a
                                    # RFC822 fetch does not apply here
    "restore_probe_max": 2000,      # cap on the header lookup of live UIDs the
                                    # index has no row for: on a partly indexed
                                    # mailbox those are simply mail that was
                                    # never indexed, not mail that came back
    "dim": 768,                     # EmbeddingGemma 2 full dim (MRL: 512/256/128)
    "query_prompt": "SearchQuery",  # only read by prompt_scheme "legacy"
    "doc_prompt": "Document",       # ("" disables prefixing)
    "prompt_scheme": "gemma",       # gemma = EmbeddingGemma 2's trained format
                                    # (`task: … | query:` / `title: … | text:`);
                                    # legacy = the literal "SearchQuery: " /
                                    # "Document: " strings, which llama.cpp
                                    # embeds as *content* (measured: cos of the
                                    # same text with and without that prefix is
                                    # 0.89-0.91, not 1.0, so the prefix is not
                                    # stripped and was pure out-of-training junk)
    "query_task": "search result",  # gemma instruction for the query side
    "doc_title_from_subject": True, # put Subject / attachment filename in `title:`
    "thread_context_depth": 2,      # how many prior mails to include
    "thread_context_chars": 200,    # chars per prior mail
    "max_attachment_mb": 10,
    "embed_audio": False,           # embed raw audio attachments (endpoint must support)
    "pdf_pages_as_images": False,   # render visual-only PDFs to page images (needs PyMuPDF)
    "embed_media_multimodal": True, # try the multimodal object form first; set false for text-only endpoints to skip the probe request
    "media_item_shape": "image_url",  # how a media item is sent to an OpenAI-style
                                    # endpoint: "image_url" -> {"type": "image_url",
                                    # "image_url": {"url": "data:..."}, "text": ...}
                                    # (OpenAI/vLLM/TEI/embed_server.py), or
                                    # "content" -> {"content": [<part>, ...]}
                                    # (chat-style parts nested per input item)

    # --- local previews: the thumbnail sidecar -----------------------------
    # Opening an image used to mean a fresh IMAP login and a whole-message FETCH
    # per click, because the index keeps text and never file bytes. Thumbnails
    # are the one exception: a 320 px copy of each indexed image, written when
    # the indexer already has the pixels in hand, in a *sidecar* file - so
    # previews are instant and offline, and full files still come from Gmail.
    "store_thumbs": True,           # write previews at index time (off: no sidecar writes)
    "thumb_max_side": 320,          # px on the longest side of a preview
    "media_cache_mb": 512,          # sidecar budget; least-recently-seen rows go
    "media_db_path": "",            # "" = "<db_path>.media.db", beside the index

    # --- reading a mail as mail: the HTML pane ------------------------------
    # The index keeps the *text* of a message; showing its HTML is a different
    # thing and costs a fetch, so it happens on click and is cached as a bounded
    # copy in the same sidecar budget the previews share. Off, the pane stays the
    # text it always was — nothing else about this app depends on these keys.
    "render_html": True,            # offer "as mail" for a message's HTML part
    "render_html_max_bytes": 2_000_000,  # refuse to fetch or render a bigger part
    "show_remote_images": False,    # Gmail's default, and this one's: a mail's
                                    # own "display images" click overrides it

    # --- multi-resolution chunking (3 tiers, small -> very large) ---------
    # Window size in characters per tier. `xlarge` deliberately equals
    # max_source_chars, so the very-large tier is exactly "the whole source".
    "chunk_sizes": {"small": 320, "large": 1200, "xlarge": 6000},
    "chunk_overlap_frac": 0.15,     # overlap between windows of a tier, as a
                                    # fraction of that tier's size (0 disables)
    "max_source_chars": 6000,       # whole-message budget per source text
    "max_windows_per_source": 24,   # hard cap on windows per tier per source.
                                    # ceil(max_source_chars / (small*(1-overlap)))
                                    # + 1 = 24 covers a full source at 320 chars;
                                    # lower it to trade tail coverage for cost
    "index_tiers": ["small", "large", "xlarge"],  # derived tiers to build at
                                    # index time; the canonical (whole-source)
                                    # chunk is always written regardless
    "tier_weights": {},             # optional per-tier multiplier on the dense
                                    # leg, e.g. {"small": 1.0, "xlarge": 0.9}

    # --- hybrid search (BM25 + dense, fused with RRF) --------------------
    # No porter: it is an *English* stemmer, so on a Dutch mailbox it splits
    # word forms apart instead of joining them (measured on a real index:
    # 'factuur' matched 4 477 chunks and 'facturen' only 1 111, 'domein' 163 vs
    # 'domeinen' 35). Morphology is handled at query time instead, by
    # expand_morphology, which can join forms of both languages.
    "fts_tokenizer": 'unicode61 remove_diacritics 2',
    "fts_meta_weight": 1.5,         # bm25 weight of the metadata column
    "search_mode": "hybrid",        # hybrid | semantic | lexical
    "rrf_k": 20,                    # RRF damping: 60 flattens a few hundred
                                    # candidates into one indistinguishable
                                    # band (rank 1 fused 0.89, rank 8 fused 0.86)
    "agg_k": 3,                     # per-message score = mean of the top-k chunks
    "lexical_pool": 200,            # chunks fetched from bm25 before fusion
    "dense_pool": 0,                # 0 = score every chunk (brute force)
    "fusion_pool": 200,             # per-leg message cap *before* fusion: RRF
                                    # adds w/(k+rank) per leg, so a leg ranking
                                    # all 4 452 messages against one capped at
                                    # 200 chunks compares ranks on different
                                    # scales (0.5/(k+4000) vs 0.5/(k+1))
    "w_dense": 0.5,                 # RRF weight of the vector leg
    "w_lex": 0.5,                   # RRF weight of the bm25 leg

    # --- query-side lexical handling -------------------------------------
    "fts_stopwords": True,          # drop NL/EN stopwords from the MATCH
                                    # expression: FTS5 has no stoplist, and a
                                    # required 'van'/'de' silently zeroed recall
                                    # ('domein verhuizen authorisatiecode'
                                    # matched 0 chunks with the AND form)
    "fts_or_first": True,           # rank the OR form with bm25 (IDF already
                                    # demotes common words) and use the AND form
                                    # only as a bonus; AND-first lost every long
                                    # query, then OR'd every word as a fallback
    "expand_morphology": True,      # join Dutch/English word forms at query
                                    # time (factuur~facturen, domein~domeinen)
    "stopwords_file": "",           # optional extra stopword list, one per line
    "synonyms_file": "",            # optional alias list, 'a => b, c' per line
    "fts_quoted_weight": 0.6,       # bm25 weight of the quoted-history column:
                                    # findable, but never out-ranks the text the
                                    # sender actually wrote
    "fts_full_match_bonus": 1.25,   # multiply the bm25 score of messages holding
                                    # every query term (the AND form no longer
                                    # ranks, so it rewards instead of excluding)

    # --- message-level ranking -------------------------------------------
    "tier_norm": True,              # per-tier z-score before mixing: the cosine
                                    # of a 236-char window and of a 3.4k-char
                                    # body are not on one scale, so a raw mean
                                    # of the top-3 silently favours one tier
    "window_dedupe": 0.92,          # collapse windows of one message whose
                                    # vectors are this similar before the top-k
                                    # mean: 20 near-identical windows of one
                                    # marketing mail beat one short relevant mail
    "collapse": True,               # one result per thread, no duplicate bodies
    "collapse_sim": 0.97,           # cosine above which two messages' bodies
                                    # count as the same text
    "recency_half_life_days": 240,  # 0 disables the recency prior
    "recency_weight": 0.35,         # prior = (1-w) + w * 0.5 ** (age/half_life)
    "demote_transactional": 1.0,    # multiplier for noreply/newsletter mail
                                    # (1.0 disables). It was 0.55, because 1 865
                                    # of 4 452 messages filled the top of nearly
                                    # every query - but 47 % of a real inbox *is*
                                    # transactional, and the mail people search
                                    # for ("ssl certificaat vernieuwen", "dmarc",
                                    # "aanmaning") comes from exactly those
                                    # senders. Measured on 84 queries the 0.55
                                    # cost 0.037 recall@5 [0.010, 0.072] and
                                    # 0.045 MRR@10 [0.009, 0.085], and P@5 rose
                                    # without it, so it was not buying precision
                                    # either. Set 0.55 to try it again.
    "transactional_senders": [],    # extra address substrings to demote
    "min_relevance": 0.30,          # below it, say so instead of returning ten
                                    # confident irrelevancies. Measured with
                                    # floor_sweep.py: unanswerable queries top
                                    # out at 0.38-0.76 and real answers at
                                    # 0.33-0.98, so no number separates the two
                                    # and this default only fires on filters that
                                    # match nothing. Raw cosines cannot floor
                                    # either: wrong results scored 0.78-0.84.
    "snippet": True,                # show the bm25-highlighted passage

    # --- reranking (opt-in; only pays when stage 1 is weak, so measure) ---
    "rerank": "off",                # off | endpoint | llm
    "rerank_endpoint": "",          # llama-server --reranking: http://host/v1/rerank
    "rerank_api_key": "",
    "rerank_model": "",             # sent as "model" when the host needs it
    "rerank_candidates": 40,        # messages re-scored per query
    "rerank_blend": 0.7,            # rerank vs fused score
    "rerank_timeout": 120,

    # --- document text hygiene (index-time; reindex --clean applies it) ---
    "clean_text": True,             # strip quoted history, signatures,
                                    # unsubscribe/legal footers and HTML junk
    "keep_quoted_in_fts": True,     # keep the stripped-away text findable by
                                    # bm25 without embedding it (quoted column)
    "decode_mime_words": True,      # RFC 2047 subjects (12.6 % of a real inbox
                                    # were still '=?UTF-8?B?...?=' = junk in
                                    # both legs)
    "index_dedupe_threads": True,   # do not embed a window already embedded for
                                    # the same thread (up to 29 messages of one
                                    # thread shared an identical body)

    # --- email intent filters -------------------------------------------
    "parse_filters": True,          # from:/van:/met bijlage/vorige week ->
                                    # SQL pre-filters instead of vector noise

    "profile": "v2",                # 'legacy' restores the pre-tuning pipeline
}

# `profile: legacy` (or `mailbox.py search --profile legacy`) turns every tuning
# knob added after the first release back off, so today's ranking stays
# reproducible on an existing index: it is the A/B baseline, and the rollback.
LEGACY_PROFILE = {
    "prompt_scheme": "legacy", "query_prompt": "SearchQuery", "doc_prompt": "Document",
    "clean_text": False, "keep_quoted_in_fts": False, "decode_mime_words": False,
    "index_dedupe_threads": False,
    "fts_tokenizer": "porter unicode61 remove_diacritics 2",
    "fts_stopwords": False, "fts_or_first": False, "expand_morphology": False,
    "fusion_pool": 0, "rrf_k": 60, "tier_norm": False, "window_dedupe": 0.0,
    "collapse": False, "recency_half_life_days": 0, "recency_weight": 0.0,
    "demote_transactional": 1.0, "min_relevance": 0.0, "snippet": False,
    "rerank": "off", "parse_filters": False,
}

# EmbeddingGemma 2 is trained with these instruction prefixes; the model card
# says using them "improves quality" and that documents with a real title must
# be written `title: {title} | text: {content}`. A mailbox has a title for every
# chunk (the Subject, or the attachment filename), so `title: none` - what a
# plain "Document: " prefix compiles to - throws that signal away.
GEMMA_DOC = "title: {title} | text: {text}"
GEMMA_QUERY = "task: {task} | query: {query}"
_NO_TITLE = "none"


def load_config(path: str = "config.json", profile: str = "") -> dict:
    """Config = DEFAULTS, then config.json, then environment, then profile.

    `profile` (also settable as the `profile` key) selects a set of ranking
    knobs: 'v2' (the default) and 'legacy', the pre-tuning pipeline, used as the
    A/B baseline and as the one-key rollback.

    Naming a profile applies that profile's whole knob set, over config.json.
    A file that lists every knob (any copy of config.example.json does) must not
    turn the switch into a no-op, and `--profile v2` must be able to undo a
    `"profile": "legacy"` left in the file - both directions failing silently is
    the worse failure, because the result looks like a measurement of the old
    pipeline. A value set in the shell environment outranks a profile; a value
    config.json set that the profile overwrites is reported. To keep a value
    through a profile switch, set it in the environment or pass the flag.
    """
    cfg = dict(DEFAULTS)
    from_file: set[str] = set()
    from_env: set[str] = set()
    file_profile = ""
    p = Path(path)
    if p.exists():
        user = json.loads(p.read_text(encoding="utf-8"))
        from_file |= set(user)
        # Only a profile a human wrote counts as named: DEFAULTS carries
        # profile='v2' as the default, and reading that back would rewrite
        # every hand-tuned config on load.
        file_profile = str(user.get("profile") or "").strip()
        cfg.update(user)
    env_overrides = {
        "GMAIL_USER": "gmail_user",
        "GMAIL_APP_PASSWORD": "gmail_app_password",
        "EMBED_ENDPOINT": "embed_endpoint",
        "EMBED_API_KEY": "embed_api_key",
        "EMBED_MODEL": "embed_model",
    }
    for env_name, key in env_overrides.items():
        val = os.environ.get(env_name)
        if val:
            cfg[key] = val
            from_env.add(key)
    named = str(profile or file_profile or "").strip().lower()
    want = named or "v2"
    if want in ("legacy", "v1", "old"):
        target, name = LEGACY_PROFILE, "legacy"
    elif want == "v2":
        target = {k: DEFAULTS[k] for k in LEGACY_PROFILE if k in DEFAULTS}
        name = "v2"
    else:
        log.warning("unknown profile %r - using the tuned defaults", want)
        target, name = {}, "v2"
    overridden = []
    if named:
        for key, val in target.items():
            if key in from_env:
                continue
            if key in from_file and cfg.get(key) != val:
                overridden.append(key)
            cfg[key] = val
    cfg["profile"] = name
    if overridden:
        log.warning("profile %s overrides %d value(s) from %s: %s",
                    name, len(overridden), path,
                    ", ".join(sorted(overridden)[:8]))
    # Where this config came from, so a later re-load (bench --variant
    # profile=legacy) reads the same file instead of a default path.
    cfg["config_path"] = str(path)
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Attachment:
    name: str
    content_type: str
    data: bytes


@dataclass
class HtmlBody:
    """A message's HTML body, decoded, plus what it took to get it.

    Kept apart from `Attachment` because it is not one: this is the mail itself,
    and the reading pane has to be able to say which of the two it is showing —
    the text that was indexed and the HTML that was only ever rendered are
    different parts of the same message, and do not always say the same thing.

    `charset` is the encoding actually used, which is not always the one the
    header claimed; when they differ, that difference is exactly what a person
    would see as mojibake, so it stays askable.
    """
    text: str
    charset: str
    section: str = ""


#: Reserved keys in the sidecar. The section memory and the byte cache are keyed
#: by *filename*, and an HTML body is not a file — so it is filed under a name no
#: file can have (`/` is illegal in one on every system this app runs on, and a
#: preview row is only ever written for pictures anyway). A real attachment whose
#: name came through as exactly this string would at worst cost one wasted
#: section read: every path below re-checks what the part says it is.
HTML_PART = "#text/html"
HTML_CIDS = "#text/html+cid"


@dataclass
class Message:
    message_id: str
    folder: str
    date: datetime
    sender: str
    to: str
    cc: str
    subject: str
    body_text: str
    attachments: list = field(default_factory=list)
    thread_id: str = ""
    references: list = field(default_factory=list)
    gmail_url: str = ""
    uid: int = 0            # IMAP UID it was fetched under, inside `folder`


def parse_date(value) -> datetime:
    if isinstance(value, datetime):
        d = value
    elif value:
        s = str(value)
        try:
            d = datetime.fromisoformat(s)
        except ValueError:
            try:
                d = parsedate_to_datetime(s)
            except Exception:
                return datetime.now(timezone.utc)
    else:
        return datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


GMAIL_SLOT = re.compile(r"(mail\.google\.com/mail/u/)\d+(/)")


def gmail_index(cfg: dict) -> int:
    """Which signed-in Google account this mailbox lives in: the `N` of
    `mail.google.com/mail/u/N/`.

    Work and private accounts in one browser sit in different slots, and a Gmail
    web link only resolves in the slot it belongs to: the same message id opens
    different mail, or nothing, from another one.
    """
    try:
        n = int(cfg.get("gmail_account_index", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def gmail_message_url(gm_msgid: str, slot: int) -> str:
    """Gmail's own web link for one message, from Gmail's own number for it.

    `X-GM-MSGID` is decimal and Gmail's URL wants it hex without the `0x`, which
    is the entire trick of a link that opens *this* message instead of a search
    for its subject. Empty for anything that is not a number, because a link that
    opens the wrong mail is worse than no link.
    """
    text = str(gm_msgid or "").strip()
    if not text.isdigit():
        return ""
    return f"https://mail.google.com/mail/u/{int(slot)}/#all/{hex(int(text))[2:]}"


def gmail_account_url(url: str, index: int) -> str:
    """Point a stored Gmail link at the configured account slot.

    `X-GM-MSGID` gives the message id but says nothing about which browser slot
    it belongs to, and the stored column predates this setting. Normalising on
    the way out is what makes changing `gmail_account_index` fix every existing
    link at once: the id was already right, and re-embedding a whole mailbox to
    move one digit in a URL is an absurd price. A link that is not Gmail's own is
    left exactly as it was.
    """
    if not url or "mail.google.com" not in url:
        return url
    return GMAIL_SLOT.sub(lambda m: m.group(1) + str(max(0, int(index))) + m.group(2),
                          url, count=1)


def decode_part(part) -> str:
    raw = part.get_payload(decode=True)
    if raw is None:
        return ""
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _normalise_lines(text: str) -> str:
    """Collapse whitespace inside lines and blank runs between them.

    Line structure is what the hygiene rules below match on, so it has to
    survive HTML extraction to be usable at all.
    """
    out = []
    for line in (text or "").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\xa0]+", " ", line).strip()
        if line:
            out.append(line)
        elif out and out[-1] != "":
            out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


# Invisible or non-content markup. `get_text()` on a whole document returns the
# *text* of <style> and <script> too, so CSS used to end up in the embeddings.
_HTML_DROP = ("script", "style", "noscript", "template", "head", "iframe", "svg")
# Tags after which a paragraph break is meant.
_HTML_BLOCK = ("p", "div", "tr", "li", "table", "ul", "ol", "h1", "h2", "h3",
               "h4", "h5", "h6", "section", "article", "blockquote", "pre",
               "footer", "header", "nav")


def html_to_text(html: str) -> str:
    """HTML -> text, keeping paragraph structure and dropping invisible junk.

    The old one-liner produced a single endless line with no paragraph breaks
    at all: windows could not end on a block boundary, a signature or an
    unsubscribe block was not a line anything could match, and every HTML mail
    lost the structure that makes chunking work.
    """
    if not html:
        return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            soup = None
        if soup is not None:
            for tag in soup.find_all(list(_HTML_DROP)):
                tag.decompose()
            for br in soup.find_all("br"):
                br.replace_with("\n")
            for tag in soup.find_all(list(_HTML_BLOCK)):
                tag.append("\n")
            return _normalise_lines(soup.get_text(" "))
    body = re.sub(r"(?is)<(script|style|noscript|template|head)[^>]*>.*?</\1>",
                  " ", html)
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    body = re.sub(r"(?i)</(p|div|tr|li|h[1-6]|table|ul|ol|blockquote|pre)>",
                  "\n\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    return _normalise_lines(body)


def pick_attachment(raw, want: str) -> Optional[Attachment]:
    """The file behind one filename in a fetched message, or None.

    `raw` is the RFC822 bytes as IMAP hands them over. Matching is
    case-insensitive on the RFC 2047-decoded filename, because that is what the
    reading pane shows and what a click sends back: the pane got its name from
    `_parse` decoding the same header, so the two sides agree by construction.
    """
    import email as emaillib
    msg = emaillib.message_from_bytes(raw) if isinstance(raw, (bytes, bytearray)) \
        else raw
    want = (want or "").strip().lower()
    if not want:
        return None
    for part in msg.walk():
        fn = part.get_filename()
        if not fn:
            continue
        fn = decode_mime_words(fn)
        if fn.strip().lower() != want:
            continue
        data = part.get_payload(decode=True) or b""
        return Attachment(fn.strip(),
                          part.get_content_type() or "application/octet-stream",
                          bytes(data))
    return None


#: A declared charset is an instruction to a browser, not a hint: it decodes with
#: it or it shows replacement characters, and it does not quietly try another one
#: because that is how a French mail arrives as `Ã©`. `iso-8859-1` becomes
#: `windows-1252` because that is what the HTML specification says a reader should
#: hear when a document claims ISO-8859-1 — the two differ in 27 code points, all
#: of them the punctuation mail actually uses (’ “ ” €).
_CHARSET_ALIAS = {"iso-8859-1": "windows-1252", "iso8859-1": "windows-1252",
                 "latin1": "windows-1252", "us-ascii": "utf-8", "ascii": "utf-8"}


def decode_text_part(part) -> tuple:
    """Decode one text part as a reader would see it: `(text, charset_used)`.

    `decode_part` is what the indexer has always used — UTF-8, else latin-1, the
    header ignored — and for text that is *searched* that rule stays exactly
    where it is, because changing it would move search results under an index
    people have already run. For text that is *shown*, believing nothing is what
    produces mojibake, so the render path asks the part what it is and decodes
    with that, replacement characters and all.
    """
    raw = part.get_payload(decode=True)
    if raw is None:
        return "", ""
    declared = (part.get_content_charset() or "").strip().lower().replace("_", "-")
    declared = _CHARSET_ALIAS.get(declared, declared)
    if declared:
        try:
            return raw.decode(declared), declared
        except LookupError:
            pass                       # a charset this Python has never heard of
        except UnicodeDecodeError:
            return raw.decode(declared, errors="replace"), declared
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    return raw.decode("windows-1252", errors="replace"), "windows-1252"


def _is_html_body(part) -> bool:
    """Whether this part *is* the mail's HTML, rather than a file shaped like it.

    A part with a filename or an `attachment` disposition is a document hanging
    off the message, and this app keeps documents download-only: rendering one in
    the pane would show a stranger's HTML while the reading pane's own word for
    that is "attachment". So the check is on both fields, and it is the same check
    on the section path and the whole-message fallback.
    """
    if part is None or part.get_content_maintype() == "multipart":
        return False
    if (part.get_content_type() or "").lower() != "text/html":
        return False
    if part.get_filename():
        return False
    return (part.get_content_disposition() or "").lower() != "attachment"


def pick_html_part(raw) -> Optional[HtmlBody]:
    """The message's own HTML body, out of a fetched RFC822 — or None.

    The first part that *is* the mail (see `_is_html_body`), in the order the
    message lists them, which is the order a mail client takes too. This is the
    fallback for when a part cannot be located by section; the bytes of the whole
    message are already in hand by then, so the walk costs nothing extra.
    """
    import email as emaillib
    msg = emaillib.message_from_bytes(raw) if isinstance(raw, (bytes, bytearray)) \
        else raw
    try:
        parts = [msg] if not msg.is_multipart() else list(msg.walk())
    except Exception:
        return None
    for part in parts:
        if not _is_html_body(part):
            continue
        text, charset = decode_text_part(part)
        if text.strip():
            return HtmlBody(text, charset, "")
    return None


# ---------------------------------------------------------------------------
# Multi-resolution chunking: one source text -> up to 3 granularities
# ---------------------------------------------------------------------------

MEDIA_TIER = "media"  # image/audio chunks: indivisible, never re-derived

_PARA_BREAK = re.compile(r"\n\s*\n+")
_SENT_BREAK = re.compile(r"(?<=[.!?])\s+")
# FTS5 keywords: a bare token equal to one of these would be parsed as an
# operator instead of a word, so such tokens get quoted in a MATCH string.
_FTS_KEYWORDS = {"and", "or", "not", "near"}


def hard_split(s: str, max_size: int) -> list[str]:
    """Split an oversized run on whitespace. A single word longer than
    max_size stays its own (oversized) window: never lose content."""
    out: list[str] = []
    cur = ""
    for w in s.split(" "):
        if cur and len(cur) + 1 + len(w) > max_size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def split_blocks(text: str, max_size: int) -> list[str]:
    """Smallest sensible units for windowing: paragraphs, else sentences,
    else whitespace runs. Blocks are whitespace-normalised, never truncated."""
    blocks: list[str] = []
    for para in _PARA_BREAK.split(text or ""):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_size:
            blocks.append(para)
            continue
        for sent in _SENT_BREAK.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_size:
                blocks.append(sent)
            else:
                blocks.extend(hard_split(sent, max_size))
    return blocks


def window_text(text: str, size: int, overlap: int = 0,
                max_windows: int = 8) -> list[str]:
    """Pack blocks into windows of at most `size` chars.

    The overlap budget carries whole trailing *blocks* into the next window
    (rather than slicing characters), so windows always start and end on a
    block boundary - no mid-word or mid-sentence cuts.
    """
    blocks = split_blocks(text, size)
    if not blocks:
        return []
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for b in blocks:
        add = len(b) + (1 if cur else 0)
        if cur and cur_len + add > size:
            windows.append(" ".join(cur))
            keep: list[str] = []
            keep_len = 0
            for x in reversed(cur):
                extra = len(x) + (1 if keep else 0)
                if keep_len + extra > overlap:
                    break
                keep.insert(0, x)
                keep_len += extra
            cur, cur_len = keep, keep_len
            add = len(b) + (1 if cur else 0)
            if cur and cur_len + add > size:   # overlap would overflow anyway
                cur, cur_len, add = [], 0, len(b)
        cur.append(b)
        cur_len += add
    if cur:
        windows.append(" ".join(cur))
    if max_windows and len(windows) > max_windows:
        _note_truncation(size, len(windows) - max_windows)
        windows = windows[:max_windows]
    return [w for w in windows if w.strip()]


# Window caps happen per source per tier, so on a real mailbox they would print
# thousands of identical warnings; count them and report once at the end.
_TRUNCATIONS: dict = {"sources": 0, "windows": 0}


def _note_truncation(size: int, dropped: int) -> None:
    first = _TRUNCATIONS["sources"] == 0
    _TRUNCATIONS["sources"] += 1
    _TRUNCATIONS["windows"] += dropped
    if first:
        log.warning("sources longer than max_windows_per_source windows are cut "
                    "short (%d-char windows); further occurrences are counted "
                    "and reported at the end of the run", size)


def truncation_note() -> str:
    """One line for the end of a run, or '' when nothing was cut."""
    if not _TRUNCATIONS["sources"]:
        return ""
    return (f"note: {_TRUNCATIONS['sources']} source(s) exceeded "
            f"max_windows_per_source, dropping {_TRUNCATIONS['windows']} window(s); "
            f"the canonical chunk still holds each complete capped text, and bm25 "
            f"still indexes it, but the dense leg misses those tails - raise "
            f"max_windows_per_source for full coverage")


def canonical_tier(length: int, sizes: dict) -> tuple[str, int]:
    """Tier that holds the whole source: the smallest one that fits it."""
    order = sorted(((int(v), k) for k, v in sizes.items()))
    for size, name in order:
        if length <= size:
            return name, size
    return order[-1][1], order[-1][0]


def tiered_chunks(text: str, sizes: dict, overlap_frac: float = 0.15,
                  max_source_chars: int = 6000, max_windows: int = 8) -> list[dict]:
    """Split one source text across the tiers.

    Rules (so nothing is ever embedded twice with identical text):
      - the text is capped at `max_source_chars`: that is the whole-message
        budget, and it is also what the `xlarge` tier holds;
      - the *canonical* tier is the smallest tier that fits the capped text;
        it contributes exactly one chunk, containing the complete capped text
        (`nwin == 1`) - which is what `reindex` re-derives finer tiers from
        later, without touching IMAP;
      - every *smaller* tier contributes its windows, minus any window whose
        text already went in (the whole source itself, or an earlier window).
        Quoted footers and repeated signature blocks make duplicate windows
        common, and each one would otherwise cost an embedding.

    Returns {tier, text, pos, nwin, canonical} dicts.
    """
    capped = (text or "").strip()[:int(max_source_chars)]
    if not capped:
        return []
    canon, canon_size = canonical_tier(len(capped), sizes)
    out = [{"tier": canon, "text": capped, "pos": 0, "nwin": 1, "canonical": True}]
    seen = {capped}
    for name, size in sorted(((k, int(v)) for k, v in sizes.items()),
                             key=lambda kv: kv[1]):
        if size >= canon_size:
            continue                     # this tier, or coarser: canonical covers it
        wins: list[str] = []
        for w in window_text(capped, size, int(size * overlap_frac), max_windows):
            if w in seen:                      # already indexed at a finer tier
                continue
            seen.add(w)                        # inside the loop: windows repeat
            wins.append(w)                     # a lot in quoted mail threads
        for i, w in enumerate(wins):
            out.append({"tier": name, "text": w, "pos": i, "nwin": len(wins),
                        "canonical": False})
    return out


def chunk_label(src: str, tier: str, pos: int, nwin: int,
                canonical: bool = False) -> str:
    """Chunk label: the bare source key for a whole-source chunk (the same
    shape pre-tier databases used), 'body [small 3/9]' for a window.

    Only a canonical chunk gets the bare label: a window that collapsed to a
    single window (`nwin == 1`) still says so, so a label alone can never be
    confused with the complete source text.
    """
    if canonical:
        return src
    return f"{src} [{tier} {pos + 1}/{max(1, nwin)}]"


# ---------------------------------------------------------------------------
# Mail text hygiene
# ---------------------------------------------------------------------------

# Quoted history, matched against a whole line: the separator that starts the
# dragged-along text. 777 of 4 452 bodies in a real inbox carried one, and each
# one repeated a message the thread already indexes (one body appeared verbatim
# in 29 messages).
_QUOTED_RE = re.compile(
    r"^\s*(?:On [^\n]{0,120}\bwrote\s*:|Op [^\n]{0,120}\bschreef[^\n]{0,20}:"
    r"|Am [^\n]{0,120}\bschrieb:)\s*$", re.IGNORECASE)
_QUOTED_HEADER_RE = re.compile(
    r"^\s*[-=*_]{2,}\s*(?:original\s+message|oorspronkelijk(?:e)?\s+(?:bericht|mail)"
    r"|forwarded\s+message|doorgestuurd\s+bericht|begin\s+van\s+bericht"
    r"|start\s+of\s+forwarded\s+message)", re.IGNORECASE)
_QUOTED_ARROW_RE = re.compile(r"^\s*(?:>|»|｜|\|)\s*\S")

# Signature / device footer: from the first match down, everything is boilerplate.
_SIGNATURE_RE = re.compile(
    r"^\s*(?:-{2,}\s*$|Met\s+vriendelijke\s+groeten?|Vriendelijke\s+groeten?"
    r"|(?:Beste|Groete|Groeten|Mvg|Met\s+groeten)[,!.]?\s*$"
    r"|(?:Kind\s+)?(?:regards|greetings)\b|With\s+(?:kind\s+)?(?:regards|best\s+wishes)"
    r"|Sent\s+from\s+my\s+(?:iPhone|iPad|Android|Galaxy|Outlook)"
    r"|Verzonden\s+vanaf\s+(?:mijn|de\s+)|Get\s+Outlook\s+for\s+(?:iOS|Android|Windows)"
    r"|This\s+(?:e-?mail|bericht).{0,20}was\s+sent\s+automatically"
    r"|Dit\s+(?:e-?mail|bericht)\s+werd\s+automatisch\s+verzonden)",
    re.IGNORECASE)
# Platform boilerplate, removed wherever it appears (forwarded mail stacks
# several footers, so "from the first one down" is not enough on its own).
_BOILERPLATE_RE = re.compile(
    r"unsubscribe|afmelden|uitschrijven|opt\s+out\s+of\s+these|"
    r"this\s+(?:e-?mail|message|bericht)\s+was\s+(?:sent|generated)\s+automatically|"
    r"dit\s+(?:e-?mail|bericht)\s+(?:is|werd)\s+automatisch|"
    r"do\s+not\s+reply\s+to\s+this\s+(?:e-?mail|message)|"
    r"reageer\s+niet\s+op\s+deze\s+(?:e-?mail|melding|bericht)|"
    r"privacy\s+(?:policy|verklaring)|algemene\s+voorwaarden|cookie\s+(?:policy|instellingen)|"
    r"no\s+reply\s+(?:e-?mail|address)|noreply@|donotreply@", re.IGNORECASE)
_LEGAL_RE = re.compile(
    r"(?:this|deze|dit)\s+(?:e-?mail|message|bericht|mail)\b[^\n]{0,90}"
    r"(?:confidential|vertrouwelijk|geheim|privileged|interne\b)", re.IGNORECASE)
_LEGAL_TAIL_RE = re.compile(
    r"^\s*(?:if\s+you\s+are\s+not\s+the\s+(?:intended|addressee)|"
    r"indien\s+u\s+niet\s+de\s+(?:bedoelde|ontvanger))", re.IGNORECASE)


def clean_text(text: str, strip_quoted: bool = True,
               strip_footers: bool = True) -> tuple[str, str]:
    """Split a body into (what the sender just wrote, what they dragged along).

    Returns `(clean, dropped)`. Only the clean half is embedded; the dropped
    half can still be BM25-indexed, so an old quote stays findable without its
    text polluting the vector of every reply that quoted it.

    Measured on a real inbox: median body 1 838 chars -> 1 067 after this; 1 463
    of 4 444 bodies lost more than 30 % of their characters to quoted history
    and footers. That padding is why one short relevant mail had to out-score
    eleven windows of the same quoted contract.
    """
    if not text or not text.strip():
        return "", ""
    lines = _normalise_lines(text).split("\n")
    kept: list[str] = []
    dropped: list[str] = []
    in_quoted = False
    for line in lines:
        if in_quoted:
            dropped.append(line)
            continue
        if not line:
            kept.append(line)
            continue
        if strip_quoted and (_QUOTED_RE.match(line)
                             or _QUOTED_HEADER_RE.match(line)
                             or _QUOTED_ARROW_RE.match(line)):
            in_quoted = True
            dropped.append(line)
            continue
        if strip_footers and (_BOILERPLATE_RE.search(line)
                              or _LEGAL_RE.search(line)
                              or _LEGAL_TAIL_RE.match(line)):
            dropped.append(line)
            continue
        kept.append(line)
    if strip_footers:
        for i, line in enumerate(kept):
            if line and _SIGNATURE_RE.match(line):
                dropped.extend(kept[i:])
                kept = kept[:i]
                break
    clean = _normalise_lines("\n".join(kept))
    # A bare "+1" reply genuinely has no new text: keep it empty rather than
    # embedding the quoted text a second time under this message's own id.
    return clean, _normalise_lines("\n".join(dropped))


_MIME_WORD_RE = re.compile(r"=\?([^?]*)\?([BbQq])\?([^?]*)\?=")


def decode_mime_words(value: str) -> str:
    """Decode RFC 2047 encoded-words in a header value.

    `msg.get('Subject')` returns the *raw* header, so a quoted-printable or
    base64 subject was indexed verbatim: 563 of 4 452 messages in a real inbox,
    e.g. `=?UTF-8?Q?10%=20korting=20op=20alle=20WOODIES=20in=20oktober?=`. Both
    legs then searched a string no human ever types.
    """
    if not value or "=?" not in value:
        return (value or "").strip()
    try:
        parts = []
        for text, enc in decode_header(value):
            parts.append(text.decode(enc or "utf-8", "replace")
                         if isinstance(text, bytes) else text)
        decoded = "".join(parts)
    except Exception:
        def _one(m):
            charset, kind, data = m.group(1) or "utf-8", m.group(2).upper(), m.group(3)
            try:
                if kind == "B":
                    pad = "=" * (-len(data) % 4)
                    return base64.b64decode(data + pad).decode(charset, "replace")
                return unquote(data.replace("_", " "))
            except Exception:
                return m.group(0)
        decoded = _MIME_WORD_RE.sub(_one, value)
    decoded = re.sub(r"\s+", " ", decoded).strip()
    return decoded or (value or "").strip()


TRANSACTIONAL_SENDERS = (
    "noreply", "no-reply", "no_reply", "noreply-", "donotreply", "do-not-reply",
    "notificatie", "notification", "newsletter", "mailing", "mailer-daemon",
    "automated", "auto-reply", "autoreply", "alerts@", "updates@", "status@",
    "postmaster", "bounce", "mailer@", "e-mailing", "marketing", "campaign",
    "deals@", "support@", "servicedesk@", "info@", "informatie@", "admin@",
    "webmaster", "updatetime@", "no-reply@", "mail@", "abonnements",
)


def is_transactional(sender: str, extra=()) -> bool:
    """Does this address belong to a system rather than to a person?

    1 865 of 4 452 messages in a real inbox came from addresses like this (the
    busiest single sender was a monitoring bot with 287 messages) and they
    filled the top of nearly every query: 'screenshot van de nieuwe website'
    returned four parcel-tracking mails as ranks 1-4.
    """
    s = (sender or "").lower()
    if not s:
        return False
    return any(n and n in s for n in tuple(TRANSACTIONAL_SENDERS) + tuple(extra))


# ---------------------------------------------------------------------------
# Query-side lexical handling (stopwords, NL/EN morphology, snippets)
# ---------------------------------------------------------------------------

# FTS5 has no stoplist and the old expression AND-ed every token, so one Dutch
# function word silently dropped a query to nothing: 'domein verhuizen
# authorisatiecode' matched 0 chunks, 'wat moest er van de domein verhuizing'
# matched 0, and the OR fallback then matched 33 366 chunks of noise.
STOPWORDS = frozenset("""
 de het een dat van der den des met en of voor op aan in bij om te ze hij zij
 deze dit die daar waar wanneer hoe wat wie wel niet ook nog ge gij u jij ik
 wij we men is was zijn ben waren heb heeft had hebben hebt wordt worden werd
 werden zal zullen zou zouden kan kunnen moest moeten mag mogen zoals als maar
 want omdat toen voordat daarna dus zelfs enkel alleen elke ieder alle alles
 iets niets iemand niemand hier daar onder boven naar over tegen zonder binnen
 buiten tussen per via tijdens sedert vanaf sinds tot t/m zeer meer meest veel
 weinig nogmaals echter derhalve tenzij wijder
 your you the and or for with from to of in on at by it its this that what
 which who whom whose when where how why not no yes can could should would
 will shall do does did have has had was were are been being a an i me my we
 our they them their he she his her there here then than so such very just
 also too more most other some any each both few much many about above below
""".split())

# Dutch and English morphology, generated on the *query* side. An English
# stemmer cannot join Dutch forms (it split 'factuur' 4 477 chunks from
# 'facturen' 1 111), so documents are left unstemmed and the query is widened.
_SUFFIX_RULES = (
    ("heden", "heid"), ("eringen", "ering"), ("ingen", "ing"), ("aties", "atie"),
    ("ities", "ity"), ("ties", "tie"), ("ures", "uur"), ("etjes", "etje"),
    ("jes", "je"), ("eren", "er"), ("en", ""), ("ies", "y"), ("es", ""),
    ("s", ""), ("ing", ""), ("ed", ""), ("ly", ""), ("ness", ""),
)
# Irregular and domain-plausible pairs the suffix rules cannot reach. Tunable,
# and `synonyms_file` adds more without touching code.
_IRREGULAR_FORMS = {
    "factuur": ["facturen", "facturatie", "gefactureerd"],
    "facturen": ["factuur", "facturatie"],
    "facturatie": ["factuur"],
    "domein": ["domeinen", "domeinnaam"], "domeinen": ["domein"],
    "wachtwoord": ["wachtwoorden", "wachtwoordherstel"],
    "wachtwoorden": ["wachtwoord"],
    "website": ["websites", "webshop"], "websites": ["website"],
    "offerte": ["offertes"], "offertes": ["offerte"],
    "order": ["orders", "orderbevestiging"], "orders": ["order"],
    "account": ["accounts"], "accounts": ["account"],
    "hosting": ["host", "hosts", "hosted"], "host": ["hosts", "hosting"],
    "invoice": ["invoices", "invoicing"], "invoices": ["invoice"],
    "payment": ["payments", "paymentterms"], "payments": ["payment"],
    "login": ["logins", "inloggen", "aanmelden"],
    "inloggen": ["login", "wachtwoord", "aanmelden"],
    "verhuizing": ["verhuizen", "verhuisd"], "verhuizen": ["verhuizing"],
    "overzetten": ["overzet", "overzetting"],
    "betaald": ["betalen", "betaling"], "betaling": ["betalen", "betaald"],
    "betalen": ["betaling", "betaald"],
    "screenshot": ["screenshots", "schermafbeelding"],
    "werknemers": ["werknemer"], "medewerker": ["medewerkers"],
    "update": ["updates", "bijwerken"], "updates": ["update"],
}


@lru_cache(maxsize=4)
def synonym_map(path: str) -> dict:
    """Optional `synonyms_file`: `factuur => facturen, declaratie` per line."""
    out: dict = {}
    if not path:
        return out
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=>" not in line:
                continue
            left, right = line.split("=>", 1)
            forms = [f.strip().lower() for f in [left] + right.split(",") if f.strip()]
            for f in forms:
                out.setdefault(f.lower(), [])
                out[f.lower()] += [g for g in forms if g != f and g not in out[f.lower()]]
    except OSError as e:
        log.warning("synonyms_file %s unreadable: %s", path, e)
    return out


def word_variants(token: str, synonyms: dict | None = None,
                  max_variants: int = 5) -> list[str]:
    """Morphological variants of one token, English and Dutch alike.

    The token itself always comes first so a prefix form of it stays available.
    """
    t = (token or "").strip()
    if len(t) < 4:
        return [t] if t else []
    low = t.lower()
    out = [t]
    if synonyms:
        out.extend(synonyms.get(low, []))
    out.extend(_IRREGULAR_FORMS.get(low, []))
    for pat, rep in _SUFFIX_RULES:
        if low.endswith(pat):
            cand = low[: len(low) - len(pat)] + rep
            if cand != low and len(cand) >= 3:
                out.append(cand)
    if not low.endswith("s"):
        out += [low + "en", low + "s"]        # Dutch plural is the common case
    elif low.endswith("en"):
        out.append(low[:-2])
    elif low.endswith("s"):
        out.append(low[:-1])
    seen, res = {low}, []
    for v in out:
        v = (v or "").strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            res.append(v)
    return [t] + res[: max(0, max_variants - 1)]


def query_tokens(query: str, drop_stopwords: bool = True,
                 limit: int = 12) -> list[str]:
    """Meaningful query tokens: word chars, no FTS keywords, no stopwords."""
    toks = [t for t in re.findall(r"\w+", query or "", re.UNICODE) if t]
    if drop_stopwords:
        kept = [t for t in toks
                if t.lower() not in STOPWORDS and t.lower() not in _FTS_KEYWORDS]
        toks = kept or toks        # stopword removal must never empty a query
    return toks[:limit]


def build_fts_query(query: str, prefix: bool = True, join: str = " AND ",
                    drop_stopwords: bool = True, variants: bool = True,
                    synonyms: dict | None = None) -> str:
    """Turn a user query into a safe FTS5 MATCH expression.

    Tokens are word-characters only, so no FTS operator can appear in them; a
    token equal to an FTS keyword gets quoted. Each token becomes one
    parenthesised group of its morphological variants, with a `*` prefix form
    for longer tokens (so 'invoi' still finds 'invoice'), and `join` says how
    the groups combine.
    """
    toks = query_tokens(query, drop_stopwords=drop_stopwords)
    if not toks:
        return ""
    groups = []
    for i, t in enumerate(toks):
        if t.lower() in _FTS_KEYWORDS:
            groups.append(f'"{t}"')
            continue
        alts = word_variants(t, synonyms) if variants else [t]
        alts = [a for a in alts if a and a.rstrip("*").lower() not in _FTS_KEYWORDS]
        if prefix and (len(t) >= 5 or i == len(toks) - 1) and t not in alts:
            alts.append(t)
        if prefix and len(t) >= 5:
            alts.append(f"{t}*")
        alts = list(dict.fromkeys(alts))[:5] or [t]
        safe = [f'"{a}"' if a.rstrip("*").lower() in _FTS_KEYWORDS else a
                for a in alts]
        groups.append(f"({' OR '.join(safe)})" if len(safe) > 1 else safe[0])
    return join.join(groups)


def keyword_snippet(text: str, query: str, width: int = 260) -> str:
    """A window of `text` around the first query term that actually appears.

    The result used to be labelled with the chunk's *name* ('body [small 3/9]')
    and its first 240 characters, so the user could not see why a hit hit.
    """
    body = (text or "").strip()
    if not body:
        return ""
    terms = [t for t in query_tokens(query, drop_stopwords=False) if len(t) > 2]
    low = body.lower()
    pos = -1
    hit = ""
    for t in sorted(set(terms), key=len, reverse=True):
        p = low.find(t.lower())
        if p >= 0 and (pos < 0 or p < pos):
            pos, hit = p, t
    if pos < 0:
        start = 0
    else:
        start = max(0, pos - width // 3)
    out = body[start:start + width]
    if start > 0:
        out = "…" + out
    if start + width < len(body):
        out += "…"
    if hit:
        out = re.sub(rf"(?<!\w){re.escape(hit)}", f"[{hit}]", out, count=1,
                     flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Gmail ingestion (IMAP)
# ---------------------------------------------------------------------------

#: `* 4158 (UID 5173 RFC822 {49449}` - the UID of the message the literal after
#: this fragment belongs to. Gmail puts it in the same fragment imaplib hands
#: alongside the body, so a fetch can be told which UID it answered without a
#: second round-trip.
_UID_RE = re.compile(rb"\bUID (\d+)", re.I)
#: the `Message-ID:` line of a `HEADER.FIELDS (MESSAGE-ID)` reply, brackets and
#: all - which is exactly how `_parse` stores it, so the two sides can be
#: compared without inventing a normalisation of their own.
_MSGID_RE = re.compile(rb"^Message-ID:\s*(.+?)\s*$", re.I | re.M)
#: a folded header value (`\r\n` + whitespace) joined back into one line first
_FOLD_RE = re.compile(rb"\r?\n[ \t]+")
#: Gmail's own message id, as it appears in the *prefix* of a FETCH reply when
#: `X-GM-MSGID` was asked for as a fetch item. It is not an in-band header of the
#: RFC822 body, which is why a mailbox indexed with `(UID RFC822)` alone has no
#: deep Gmail links at all: the number exists, it was just never asked for.
_GM_MSGID_RE = re.compile(rb"\bX-GM-MSGID\s+(\d+)", re.I)
#: which MIME section a reply belongs to, and whether it is the part's headers
#: (`BODY[2.MIME]`) or its bytes (`BODY.PEEK[2.1]`).
_SECTION_RE = re.compile(rb"BODY\[(?:PEEK\[)?([0-9]+(?:\.[0-9]+)*)(\.MIME)?",
                         re.I)


def _gm_msgid(prefix) -> str:
    """Gmail's numeric message id out of a FETCH reply prefix, '' when absent."""
    try:
        found = _GM_MSGID_RE.search(bytes(prefix or b""))
    except (TypeError, ValueError):
        return ""
    return found.group(1).decode() if found else ""


def _mime_part(raw) -> "object":
    """A MIME part's bytes read as a message, or None when they are not one.

    Used for a *single part* — its headers alone (a `BODY[k.MIME]` answer) or
    its headers plus its encoded body — so part inspection never assumes a whole
    mail. `None` is a fact about those bytes, and every caller treats it as
    "this section is not the file": a truncated reply, a server that answered
    something else, an empty part.
    """
    import email as emaillib
    try:
        msg = emaillib.message_from_bytes(bytes(raw or b""))
    except Exception:
        return None
    return msg if msg is not None and (msg.items() or msg.get_payload()) else None


def _section_key(section: str) -> tuple:
    """`"2.10"` sorts after `"2.9"`: sections in the order a reader expects.

    Only affects which of two same-named parts is tried first, so that it is the
    first one in the message — the one `pick_attachment` would have handed over.
    """
    parts = tuple(int(p) for p in str(section).split(".") if p.isdigit())
    return parts or (10 ** 9,)


#: What Gmail answers when a FETCH names a MIME section the message does not
#: have. The whole command fails and nothing in it comes back: a probe that asks
#: for sixteen sections of a three-part mail is not answered with the three that
#: exist, it is answered with *nothing*. Every section search in this file is
#: sized to the message because of that one reply.
FETCH_ITEM_REJECTED = b"could not be fetched"

def _structure_list(text) -> Optional[list]:
    """The first s-expression in a `BODYSTRUCTURE` reply, as nested lists.

    `NIL` becomes None, every other token stays bytes. Only the shapes RFC 3501
    allows are read: parentheses, quoted strings, `{n}` literals, bare atoms.
    Anything malformed — an unbalanced tree, a literal whose bytes are not in the
    reply, a paren-less token where a part should be — answers `None`, which sends
    the caller back to probing sections one command at a time. That is the whole
    safety argument for reading this field at all: it may be misread, and the
    worst a misreading can do is cost a slower path, because the file handed over
    is still the one whose *own headers* say it is the file that was asked for.
    """
    data = bytes(text or b"")
    stack: list = [[]]
    i = 0
    while i < len(data):
        ch = data[i:i + 1]
        if ch in b" \t\r\n":
            i += 1
            continue
        if len(stack) == 1 and not stack[0] and ch != b"(":
            # A part list starts with a parenthesis and nothing else does. What
            # sits in front of one — a reply line's own `4152 (UID 5163
            # BODYSTRUCTURE` — is not a part list, and reading it as one would
            # number every part after the wrong thing.
            return None
        if ch == b"(":
            stack.append([])
            i += 1
            continue
        if ch == b")":
            if len(stack) < 2:
                return None
            done = stack.pop()
            stack[-1].append(done)
            if len(stack) == 1:
                return done          # the tree is complete; the reply's own
            i += 1                   # trailing `)` is not part of it
            continue
        if ch == b'"':
            end = data.find(b'"', i + 1)
            if end < 0:
                return None
            stack[-1].append(data[i + 1:end])
            i = end + 1
            continue
        if ch == b"{":
            end = data.find(b"}", i)
            if end < 0:
                return None
            try:
                size = int(data[i + 1:end])
            except ValueError:
                return None
            start = end + 1
            if start + size > len(data):
                return None          # a literal that is not (all) there yet
            stack[-1].append(data[start:start + size])
            i = start + size
            continue
        end = i
        while end < len(data) and data[end:end + 1] not in b" \t\r\n()\"":
            end += 1
        stack[-1].append(None if data[i:end].upper() == b"NIL" else data[i:end])
        i = end
    return None                      # never closed: not a structure


def _structure_parts(node, prefix: str = "") -> Optional[list]:
    """`[(section, filename, content_type, octets)]` for the leaf parts of a
    parsed BODYSTRUCTURE, in section order.

    IMAP numbers a multipart's children 1..N in the order they appear, so a
    leaf's section is its path down the tree: `2`, `1.3`, `1.2.4`. The root of a
    mail that is not a multipart has no numbered section at all, so a singlepart
    yields nothing and the caller falls back — which is also the answer for a
    tree whose fields are not where RFC 3501 puts them.

    Filenames are read the way a mail client reads them: the disposition's
    `FILENAME` first (what an attachment is *called*), then the type's `NAME`,
    then the description. Sizes are the encoded octets, the number a server
    reports before transferring anything, which is what makes a "this file is
    60 MB" decision possible *before* downloading 60 MB.
    """
    if not isinstance(node, list) or not node:
        return None
    if not isinstance(node[0], list):
        if not prefix:
            return []                # a singlepart mail has no numbered sections
        ctype = b"/".join(x for x in node[0:2] if x).lower().decode("latin-1")
        params = node[2] if len(node) > 2 and isinstance(node[2], list) else []
        octets = node[6] if len(node) > 6 else None
        try:
            size = int(octets) if octets else 0
        except (TypeError, ValueError):
            size = 0
        disp_params = node[9] if len(node) > 9 and isinstance(node[9], list) else []
        name = None
        for group, key in ((disp_params, b"filename"), (params, b"name")):
            for k, v in _pairs(group):
                if k == key and v:
                    name = v.decode("utf-8", "replace")
                    break
            if name:
                break
        if not name and len(node) > 4 and node[4]:
            name = node[4].decode("utf-8", "replace")
        return [(prefix, name, ctype, size)]
    leaves: list = []
    depth = 0
    while depth < len(node) and isinstance(node[depth], list):
        found = _structure_parts(node[depth],
                                 str(depth + 1) if not prefix
                                 else f"{prefix}.{depth + 1}")
        if found is None:
            return None
        leaves.extend(found)
        depth += 1
    return leaves


def _pairs(fields) -> list:
    """A BODYSTRUCTURE parameter list as `[(key, value)]`, keys lowercased."""
    out = []
    for i in range(0, len(fields) - 1, 2):
        key, value = fields[i], fields[i + 1]
        if isinstance(key, bytes):
            out.append((key.lower(), value))
    return out


class FolderUnavailable(Exception):
    """An IMAP folder could not be examined, so nothing is known about it.

    Raised rather than swallowed: a folder that could not be listed must never
    be treated as an empty one, because "empty" is the one answer that would
    justify deleting the rows that reference it.
    """


@dataclass
class FolderSnapshot:
    """What one IMAP folder holds right now, as far as a read-only look sees.

    `uids` is the whole truth about *this folder*: a message that left it -
    deleted, archived, moved - is simply not in the set, whichever other folder
    it went to. `uidvalidity` says whether those numbers mean what the last
    run's numbers meant (see `reconcile_deletions`).
    """
    folder: str
    uidvalidity: int
    uids: set


class GmailFetcher:
    """Fetches messages from Gmail over IMAP using an app password.

    **Read-only, always.** The verbs this class sends are `LOGIN`, `LIST`,
    `EXAMINE`, `UID SEARCH`, `UID FETCH` (with `BODY.PEEK`, so nothing is even
    marked seen) and `LOGOUT`. It never sends `SELECT` read-write, `STORE`,
    `UID STORE`, `COPY`, `MOVE`, `APPEND`, `EXPUNGE`, `DELETEMESSAGE`, `CREATE`
    or `RENAME`: this program does not change anybody's mailbox, and the only
    deletion it is capable of is a local row describing a message the user
    already deleted in Gmail. `test_reconciliation_never_writes` fails the
    suite if a run ever issues one of the other verbs.
    """

    #: every IMAP command this class is allowed to issue (see above)
    READ_ONLY_COMMANDS = ("LOGIN", "LIST", "NAMESPACE", "EXAMINE", "SEARCH",
                          "FETCH", "NOOP", "CHECK", "LOGOUT", "CAPABILITY")

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _connect(self, debug: bool = False):
        """Create a logged-in IMAP connection.

        `imap_timeout` goes on the socket, where it bounds *one read*: a big FETCH
        over a slow link keeps arriving and keeps resetting the clock, so it is
        never cut off, while a server that has gone silent is. Gmail does go
        silent - on a socket it decided to stop serving, occasionally in the
        middle of a part - and without a bound the read simply never returns: the
        command hangs, the thread with it, and in the GUI one stalled file per
        media worker is all it takes for attachment buttons to stop answering.
        With a bound the timeout surfaces as `TimeoutError`, an `OSError`, which
        is exactly what `GmailSession.call` already reads as "this connection is
        gone" (retry once, on a fresh login) and what a backfill counts as one
        file it could not read and moves past.
        """
        import imaplib
        conn = imaplib.IMAP4_SSL(self.cfg["imap_host"], self.cfg["imap_port"],
                                 timeout=self._timeout())
        if debug:
            conn.debug = 4  # log every byte sent/received to stderr
        conn.login(self.cfg["gmail_user"], self.cfg["gmail_app_password"])
        return conn

    def _timeout(self) -> Optional[float]:
        """`imap_timeout` as a socket timeout: seconds, or None for no limit."""
        try:
            seconds = float(self.cfg.get("imap_timeout",
                                         DEFAULTS["imap_timeout"]))
        except (TypeError, ValueError):
            log.warning("imap_timeout %r is not a number; using %s s",
                        self.cfg.get("imap_timeout"),
                        DEFAULTS["imap_timeout"])
            seconds = float(DEFAULTS["imap_timeout"])
        return seconds if seconds > 0 else None

    @contextlib.contextmanager
    def session(self, debug: bool = False):
        """One logged-in connection, logged out on exit.

        Reconciliation wants several commands against one account (a listing
        per folder, maybe a sweep, maybe a targeted fetch), and a fresh login
        per folder is both slower and the kind of thing Gmail rate-limits.
        """
        conn = self._connect(debug)
        try:
            yield conn
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    # -- IMAP argument shapes ------------------------------------------------
    # imaplib wraps *every* str argument in double quotes on the wire, which is
    # right for a literal and wrong for everything else:
    #   * `"[Gmail]/All Mail"` as one quoted string makes Gmail answer
    #     `BAD Could not parse command`, so folders with a space in them are
    #     unselectable through imaplib as text;
    #   * `"SINCE 01-Oct-2026"` is a text search for that string, not a date
    #     search. imaplib happens to pass a lone SINCE/BEFORE/ON through
    #     unquoted (its own `_internal_date_search_pattern`), which is the only
    #     reason the incremental fetch works - two criteria together, or Gmail's
    #     `X-GM-*` keys, have to be bytes to survive at all.
    # Bytes go out untouched, so quoting is done here, deliberately, per kind.

    @staticmethod
    def _bname(name: str) -> bytes:
        """A mailbox name as IMAP should receive it: always quoted, once."""
        safe = str(name).replace("\\", "\\\\").replace('"', '\\"')
        return b'"%s"' % safe.encode("utf-8")

    @staticmethod
    def _bytes(text: str) -> bytes:
        """A search criterion or UID set, sent exactly as written."""
        return str(text).encode("utf-8")

    @staticmethod
    def _joined(data) -> bytes:
        """The body of an IMAP response: it can span several lines."""
        return b" ".join(d for d in (data or []) if isinstance(d, (bytes, bytearray)))

    # -- reading a folder without touching it --------------------------------
    def select_readonly(self, conn, name: str) -> int:
        """EXAMINE a folder and return its UIDVALIDITY (0 when not reported).

        EXAMINE and not SELECT: SELECT makes the connection a writer - it is
        the folder a later `STORE` or `EXPUNGE` would act on, and some servers
        finalise pending deletions when a read-write selection ends. Neither is
        wanted here, and the reply is the same either way.
        """
        typ, _data = conn.select(self._bname(name), readonly=True)
        if typ != "OK":
            raise FolderUnavailable(f"EXAMINE {name} -> {typ}")
        vals = (getattr(conn, "untagged_responses", None) or {}).get("UIDVALIDITY")
        try:
            return int(vals[-1])                      # imaplib gives [b'1']
        except (AttributeError, IndexError, TypeError, ValueError):
            return 0

    def snapshot(self, conn, name: str) -> FolderSnapshot:
        """The folder's live UIDs, in one cheap command.

        `UID SEARCH ALL` answers with numbers only: ~27 KB and well under a
        second for 4 200 messages, against the ~340 KB and 8 s a header sweep
        over the same folder costs. That difference is why a refresh reconciles
        by UID and only sweeps when it has to.
        """
        uidvalidity = self.select_readonly(conn, name)
        _typ, data = conn.uid("SEARCH", None, self._bytes("ALL"))
        return FolderSnapshot(folder=name, uidvalidity=uidvalidity,
                              uids={int(x) for x in self._joined(data).split()
                                    if x.isdigit()})

    def id_map(self, conn, name: str, uids=None) -> dict:
        """`{lower(Message-ID): uid}` for a folder, headers only.

        This is the join between Gmail and the index: `messages.message_id` *is*
        the `Message-ID` header the fetcher stored, so asking the server for
        exactly that header of every message says which stored rows still have
        a home. `BODY.PEEK` so no message is marked seen; `UID` in the items so
        each reply says which message it was.

        Keys are lowercased on both sides: Gmail hands back some senders' ids
        lowercased, and a row stored as `<ABC@x>` has to meet `<abc@x>`.
        """
        self.select_readonly(conn, name)
        ids = sorted(int(u) for u in (uids or ()))
        if not ids:
            _typ, data = conn.uid("SEARCH", None, self._bytes("ALL"))
            ids = sorted(int(x) for x in self._joined(data).split() if x.isdigit())
        size = max(1, int(self.cfg.get("reconcile_batch_size",
                                      DEFAULTS["reconcile_batch_size"])))
        items = self._bytes("(UID BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
        out: dict = {}
        for i in range(0, len(ids), size):
            page = self._bytes(",".join(str(u) for u in ids[i:i + size]))
            _typ, msgs = conn.uid("FETCH", page, items)
            for item in msgs or []:
                if not (isinstance(item, tuple) and len(item) == 2
                        and isinstance(item[1], (bytes, bytearray))):
                    continue
                uid = _UID_RE.search(bytes(item[0] or b""))
                head = _FOLD_RE.sub(b" ", bytes(item[1]))
                mid = _MSGID_RE.search(head)
                if uid and mid:
                    out[mid.group(1).decode("utf-8", "replace").strip().lower()] \
                        = int(uid.group(1))
            if i + size < len(ids):
                time.sleep(1.0)          # polite to IMAP rate limits, as in fetch()
        return out

    def fetch_uids(self, conn, name: str, uids, debug: bool = False) -> list:
        """The messages behind specific UIDs.

        For mail that came back: an un-deleted message is older than
        `last_indexed_date`, so `fetch()` would not offer it again for years,
        while one targeted FETCH is the whole repair.
        """
        self.select_readonly(conn, name)
        ids = [str(u) for u in (uids or ())]
        size = max(1, int(self.cfg.get("fetch_batch_size", 10)))
        out = []
        for i in range(0, len(ids), size):
            out.extend(self._fetch_page(conn, [i2.encode() for i2 in
                                               ids[i:i + size]], name, debug=debug))
        return out

    # -- reading one message back in full ------------------------------------
    # The index stores text, never file bytes, so an attachment that is to be
    # opened has to come from the folder it was indexed from. `BODY.PEEK[]` is
    # the whole message without the side effect of marking it seen - the same
    # rule the reconcile holds to: every verb here stays inside
    # READ_ONLY_COMMANDS, and opening a PDF cannot change anybody's mail.

    def fetch_message(self, conn, name: str, uid: int,
                      debug: bool = False) -> Optional[bytes]:
        """The full RFC822 bytes of one UID, read-only (`BODY.PEEK[]`)."""
        self.select_readonly(conn, name)
        _typ, msgs = conn.uid("FETCH", self._bytes(str(int(uid))),
                              self._bytes("(UID BODY.PEEK[])"))
        for item in msgs or []:
            if (isinstance(item, tuple) and len(item) == 2
                    and isinstance(item[1], (bytes, bytearray))
                    and bytes(item[1]).strip()):
                return bytes(item[1])
        return None

    def fetch_by_message_id(self, conn, name: str, message_id: str,
                            debug: bool = False) -> Optional[bytes]:
        """The same bytes found by `Message-ID` instead of by UID.

        For rows indexed before UIDs were recorded (`uid = 0`), and as the
        honest answer when Gmail renumbered the folder under a new
        UIDVALIDITY: the header is the identity this index keys on, so it can
        always be asked for. A HEADER search reads nothing but headers until
        the one hit is fetched with a PEEK.
        """
        if not message_id:
            return None
        self.select_readonly(conn, name)
        mid = str(message_id).replace("\\", "").replace('"', "").strip()
        crit = self._bytes(f'(HEADER Message-ID "{mid}")')
        _typ, data = conn.uid("SEARCH", None, crit)
        for tok in self._joined(data).split():
            if not tok.isdigit():
                continue
            raw = self.fetch_message(conn, name, int(tok), debug=debug)
            if raw:
                return raw
        return None

    # -- reading ONE file without reading the message ------------------------
    #: How many top-level MIME sections a click asks about by name. A
    #: `multipart/mixed` with a body and its attachments numbers those 1..N, so
    #: this covers the ordinary mail; a file one level deeper (an inline image
    #: inside a `multipart/related` inside the mixed part) is chased one level,
    #: and anything past that falls back to reading the whole message — slower,
    #: and always right.
    PART_PROBE_MAX = 16
    #: How many of those sections one command asks about. Small, because a run
    #: containing one section the message does not have is refused whole and has
    #: to be asked again in pieces: four is wide enough that a plain
    #: text-plus-attachments mail is answered in a single command, and narrow
    #: enough that a mail which overshoots loses one small command doing so.
    PROBE_CHUNK = 4

    @staticmethod
    def _sections(msgs, mime: bool = False) -> dict:
        """`{section: bytes}` out of a FETCH reply, keyed by MIME section number.

        `mime=True` keeps only the `BODY[k.MIME]` answers (a part's headers),
        the default only its bytes. One command can ask for both kinds, and they
        arrive with the *same* section number, so which kind an answer is has to
        be read off the prefix too — which is also the only place the number
        appears at all (`77 (UID 77 BODY[2.MIME] {97}`). A command that asked
        about sixteen sections needs that prefix to know which answer is which;
        a reply whose prefix names no section (a bare `UID`, a flag list) is not
        an answer about a part and is skipped.
        """
        out: dict = {}
        for item in msgs or []:
            if not (isinstance(item, tuple) and len(item) == 2
                    and isinstance(item[1], (bytes, bytearray))):
                continue
            try:
                prefix = bytes(item[0] or b"")
            except (TypeError, ValueError):
                continue
            found = _SECTION_RE.search(prefix)
            if found and bool(found.group(2)) == bool(mime):
                out[found.group(1).decode()] = bytes(item[1])
        return out

    def part_headers(self, conn, name: str, uid: int, sections=None) -> dict:
        """`{section: headers}` for the sections of one message that answer.

        `BODYSTRUCTURE` is deliberately not consulted on this path: it is the one
        for a server whose structure could not be read, and it asks the server
        itself what each part is. The asking has to be *sized*, though, because
        Gmail fails a FETCH outright when a single named section does not exist —
        a blind probe of sixteen sections on a three-part mail comes back with
        nothing at all, which is precisely how this path once stopped working in
        production while every test stayed green on a fake that answered each
        section it had and kept silent about the rest.

        So the sections are asked in short runs from the bottom up, a run that is
        refused is split, and the search ends at the first section that does not
        answer on its own: numbering is handed out in order, so a missing section
        is the end of the message rather than a gap in it. Most mail is answered
        in two or three commands this way, and none of them is ever a command the
        server refused on principle.
        """
        secs = [str(s) for s in (sections
                                 or range(1, self.PART_PROBE_MAX + 1))]
        out: dict = {}
        for i in range(0, len(secs), self.PROBE_CHUNK):
            found, past_the_end = self._ask(conn, uid,
                                            secs[i:i + self.PROBE_CHUNK], 0)
            out.update(found)
            if past_the_end:
                break
        return out

    def _ask(self, conn, uid: int, secs: list, depth: int) -> tuple:
        """`({section: headers}, ran_past_the_end)` for one run of sections.

        The second answer is the one that makes this cheap: a section that is not
        there at all means the message stopped below it, so the caller stops
        asking instead of walking up to the probe limit one refused command at a
        time. Halving only ever splits a run the server refused as a whole, and
        never goes below one section, so the work is bounded by twice the number
        of parts, whatever a caller passed in as the limit.
        """
        if not secs:
            return {}, False
        items = "(" + " ".join(f"BODY.PEEK[{s}.MIME]" for s in secs) + ")"
        try:
            typ, msgs = conn.uid("FETCH", self._bytes(str(int(uid))),
                                 self._bytes(items))
        except Exception as e:
            log.debug("section probe could not be sent: %s", e)
            return {}, True
        found = self._sections(msgs, mime=True)
        if found or str(typ) == "OK":
            return found, False      # answered, or answered nothing on its own terms
        if len(secs) == 1:
            return {}, True          # this section does not exist: neither does any above
        if depth >= 4:
            return {}, False
        middle = len(secs) // 2
        out, stop = self._ask(conn, uid, secs[:middle], depth + 1)
        if stop:
            return out, True
        rest, stop = self._ask(conn, uid, secs[middle:], depth + 1)
        out.update(rest)
        return out, stop

    def part_structure(self, conn, name: str, uid: int) -> Optional[list]:
        """The leaf parts of one message from `BODYSTRUCTURE`, or None.

        One small command names every part before a byte of any part is
        transferred: which section it lives in, what the file is called, how big
        it is. A file then costs two commands — this and the part itself —
        however many parts the message carries, where the header probe needs one
        more command for every range it has to guess about.

        None of it is taken on trust. A section read here is only a place to
        look: `_read_part` re-reads it *with its headers* and compares the
        filename there with the one that was asked for, so a server that
        describes its parts oddly, or a parser meeting a field in a shape it did
        not expect, costs a slower path and never the wrong file. `None` — no
        answer, an answer that will not parse, a message that is not a multipart
        and so has no numbered parts — tells the caller to probe instead.
        """
        try:
            typ, msgs = conn.uid("FETCH", self._bytes(str(int(uid))),
                                 self._bytes("(BODYSTRUCTURE)"))
        except Exception as e:
            log.debug("no BODYSTRUCTURE: %s", e)
            return None
        if str(typ) != "OK":
            return None
        for item in msgs or []:
            chunks = ([item[0]] if isinstance(item, tuple) and item
                      else [item] if isinstance(item, (bytes, bytearray)) else [])
            for chunk in chunks:
                try:
                    chunk = bytes(chunk)
                except (TypeError, ValueError):
                    continue
                at = chunk.upper().find(b"BODYSTRUCTURE")
                if at < 0:
                    continue
                tree = _structure_list(chunk[at + len(b"BODYSTRUCTURE"):])
                parts = _structure_parts(tree) if tree is not None else None
                if parts:
                    return parts
        return None

    def fetch_attachment(self, conn, name: str, uid: int, want: str,
                         parts=None, max_bytes: Optional[int] = None):
        """One named file, over the wire as one MIME part instead of one message.

        Three ways to find the part, tried in that order, each verified the same
        way: the section this program *remembered* for this file (one command);
        `BODYSTRUCTURE`, which names every part and its size before any of them is
        transferred (two commands for the file, whatever the message holds); and a
        header probe, sized to the message because Gmail refuses a FETCH that names
        a section it does not have. A 40 KB logo inside a 5 MB mail stops costing
        5 MB either way, which is the whole point: what made opening a picture slow
        was never the picture, it was the message around it.

        Nothing here is taken on trust. The section that matched is re-read,
        decoded and checked against `want` with the same filename rule
        `pick_attachment` uses — the same decoder, the same RFC 2047 decoding,
        the same case-insensitivity — and any doubt at all (a server that will
        not answer a `.MIME` probe, a part that will not decode, a name that
        does not match) answers `None` so the caller reads the whole message the
        way this path used to be the only path. Being slower than the fallback
        is acceptable; being wrong about which file someone clicked is not.

        `parts` is the optional section memory (`MediaCache.part` /
        `.remember_part`): a file opened before in this folder at this UID is
        read from the section remembered then, probe-free — and still verified,
        because mail gets edited and folders get renumbered under a stored UID.
        A structure that was read is filed into it for *every* part it named, so
        the ninth picture of a mailing is one command and not another search.

        `max_bytes` is for a reader with a budget rather than a person who asked:
        a job cutting thumbnails has no use for a 60 MB original, and the
        structure reports a part's size *before* it is transferred, which is the
        only moment that transfer can still be avoided. Nobody clicking wants a
        limit, so the GUI passes none.

        `BODY.PEEK` everywhere, as in the rest of this class: opening a file
        cannot even mark the mail as read.
        """
        want_l = (want or "").strip().lower()
        if not want_l or not uid:
            return None
        def over_budget(size: int) -> bool:
            """One budget check for both ways of finding the part, so a job with
            a limit cannot get past it by having learned the section before."""
            if max_bytes and int(size or 0) > max_bytes:
                log.debug("%s#%s %s: %s bytes, over the %s byte limit",
                          name, uid, want, size, max_bytes)
                return True
            return False

        def remember(section: str, filename: str, size: int) -> None:
            """Every part the structure named, filed under the name a later click
            will ask it by — the decoded name, because that is what the index
            stored and what the attachment chip shows."""
            clean = decode_mime_words(filename or "").strip()
            if parts is None or not clean:
                return
            try:
                parts.remember_part(name, uid, clean, section, size)
            except Exception as e:
                log.debug("section %s not remembered: %s", section, e)

        known, known_octets = "", 0
        if parts is not None:
            try:
                known, known_octets = parts.part(name, uid, want) or ("", 0)
            except Exception as e:
                log.debug("no remembered section: %s", e)
            known = str(known or "")
        if known:
            if over_budget(known_octets):
                return None      # already known to be too big; nothing to ask for
            found = self._read_part(conn, name, uid, known, want_l)
            if found is not None:
                return found

        # Two passes, and the order matters: everything the structure named is
        # filed *before* any part is read, so a message read once is a message
        # whose remaining files are each one command away. Reading on the fly
        # would stop at the file that was asked for and leave the ones after it
        # as mysterious as they were, which is the opposite of what a structure
        # is for.
        wanted: list = []
        for section, filename, _ctype, size in (
                self.part_structure(conn, name, uid) or ()):
            remember(section, filename, size)
            if decode_mime_words(filename or "").strip().lower() == want_l:
                wanted.append((section, size))
        for section, size in wanted:
            if over_budget(size):
                return None
            found = self._read_part(conn, name, uid, section, want_l)
            if found is not None:
                return found

        def scan(heads: dict, parents: list):
            """The first section that says it is `want_l`, read and verified."""
            for section in sorted(heads, key=_section_key):
                part = _mime_part(heads[section])
                if part is None:
                    continue
                if part.get_content_maintype() == "multipart":
                    # A container, not a file: its children are the next round.
                    parents.extend(f"{section}.{i}"
                                   for i in range(1, self.PART_PROBE_MAX + 1))
                    continue
                fn = part.get_filename()
                if not fn or decode_mime_words(fn).strip().lower() != want_l:
                    continue
                found = self._read_part(conn, name, uid, section, want_l,
                                        headers=part)
                if found is None:
                    continue
                if parts is not None:
                    try:
                        parts.remember_part(name, uid, want, section,
                                            len(found.data))
                    except Exception as e:
                        log.debug("section %s not remembered: %s", section, e)
                return found
            return None

        try:
            heads = self.part_headers(conn, name, uid)
        except Exception as e:
            log.debug("no MIME section probe: %s", e)
            return None
        parents: list = []
        found = scan(heads, parents)
        if found is not None or not parents:
            return found
        # One nesting level, and only when the first round found a container:
        # the inline image inside a `multipart/related` inside a `mixed` part is
        # common enough in HTML mail to be worth a second small command, and
        # deeper than that the whole-message fallback is the honest answer.
        try:
            heads = self.part_headers(conn, name, uid, sections=parents)
        except Exception as e:
            log.debug("no nested MIME section probe: %s", e)
            return None
        return scan(heads, [])

    def _read_part(self, conn, name: str, uid: int, section: str,
                   want_l: str, headers=None):
        """One section's bytes, checked that they answer to `want_l`.

        `BODY[k]` is the part with its transfer encoding intact, so the headers
        go back in front of them and `email` decodes the part exactly as a
        whole-message parse would have: same base64, same charset handling, no
        second implementation of either. When the headers were not probed (a
        remembered section) they are fetched in the same command as the bytes,
        so the check never costs a round trip of its own.

        The filename is checked one last time on the reconstructed part. What
        comes back is the file that was asked for, or None.
        """
        try:
            items = (f"(BODY.PEEK[{section}])" if headers is not None else
                     f"(BODY.PEEK[{section}.MIME] BODY.PEEK[{section}])")
            _typ, msgs = conn.uid("FETCH", self._bytes(str(int(uid))),
                                  self._bytes(items))
        except Exception as e:
            log.debug("section %s could not be read: %s", section, e)
            return None
        body = self._sections(msgs).get(section)
        if not body:
            return None
        if headers is None:
            head = self._sections(msgs, mime=True).get(section) or b""
        else:
            head = headers.as_bytes().split(b"\r\n\r\n", 1)[0]
        try:
            part = _mime_part(head + b"\r\n\r\n" + body)
            data = part.get_payload(decode=True) if part is not None else None
        except Exception as e:
            log.debug("section %s did not decode: %s", section, e)
            return None
        if not data:
            return None
        fn = decode_mime_words(part.get_filename() or "").strip()
        if fn.lower() != want_l:
            return None
        return Attachment(fn, part.get_content_type() or "application/octet-stream",
                          bytes(data))

    # -- reading the mail's own HTML body --------------------------------------

    def fetch_html_part(self, conn, name: str, uid: int, parts=None,
                        max_bytes: Optional[int] = None) -> Optional[HtmlBody]:
        """The message's HTML body, without reading the message to get it.

        Same three ways as `fetch_attachment`, in the same order, and for the same
        reason: a remembered section is one command, `BODYSTRUCTURE` is two, and a
        probe is a few more — where the whole message is everything. The thing
        being read here is usually most of the mail, so the third path is not the
        last resort it would be for a 40 KB logo in a 5 MB message; it is the
        honest fallback when no way of naming the part worked.

        Verification is the part that makes any of this safe to try. A section is
        only a place to look: what comes back is re-read with its headers and must
        *say* it is a `text/html` part that is the mail itself — not a file (see
        `_is_html_body`), because this app shows bodies and hands out attachments,
        and confusing the two would render a stranger's document while calling it
        the message.

        `parts` is the section memory again (`HTML_PART` as its filename key), so
        the second look at a mail is one command; the body itself is cached
        separately by the caller, which is what makes a re-render free.
        """
        if not uid:
            return None

        def remember(section: str, octets: int) -> None:
            if parts is None or not section:
                return
            try:
                parts.remember_part(name, uid, HTML_PART, section, octets)
            except Exception as e:
                log.debug("html section %s not remembered: %s", section, e)

        def over_budget(size: int) -> bool:
            if max_bytes and int(size or 0) > max_bytes:
                log.debug("%s#%s html: %s bytes, over the %s byte limit",
                          name, uid, size, max_bytes)
                return True
            return False

        known = ""
        if parts is not None:
            try:
                known = str((parts.part(name, uid, HTML_PART) or ("", 0))[0] or "")
            except Exception as e:
                log.debug("no remembered html section: %s", e)
        if known:
            found = self._read_html_part(conn, name, uid, known)
            if found is not None:
                return found

        candidates: list = []
        for section, filename, ctype, size in (
                self.part_structure(conn, name, uid) or ()):
            if (ctype or "").lower() != "text/html" or filename:
                continue
            candidates.append((section, size))
        for section, size in candidates:
            if over_budget(size):
                return None
            found = self._read_html_part(conn, name, uid, section)
            if found is not None:
                remember(section, len(found.text.encode("utf-8", "replace")))
                return found

        # Nothing named it, so ask the parts themselves — one nesting level, the
        # same bounded probe `fetch_attachment` uses, then the whole message.
        parents: list = []

        def scan(heads: dict):
            for section in sorted(heads, key=_section_key):
                part = _mime_part(heads[section])
                if part is None:
                    continue
                if part.get_content_maintype() == "multipart":
                    parents.extend(f"{section}.{i}"
                                   for i in range(1, self.PART_PROBE_MAX + 1))
                    continue
                if not _is_html_body(part):
                    continue
                found = self._read_html_part(conn, name, uid, section, headers=part)
                if found is not None:
                    remember(section, len(found.text.encode("utf-8", "replace")))
                    return found
            return None

        try:
            heads = self.part_headers(conn, name, uid)
        except Exception as e:
            log.debug("no MIME section probe for the html body: %s", e)
            heads = {}
        found = scan(heads)
        if found is None and parents:
            try:
                found = scan(self.part_headers(conn, name, uid, sections=parents))
            except Exception as e:
                log.debug("no nested MIME probe for the html body: %s", e)
        if found is not None:
            return found
        # The root of a mail that is not a multipart has no numbered section at
        # all, so for the common single-file HTML newsletter *this* is the cheap
        # path: the message and its body are the same bytes.
        raw = self.fetch_message(conn, name, uid)
        return pick_html_part(raw) if raw else None

    def _read_html_part(self, conn, name: str, uid: int, section: str,
                        headers=None) -> Optional[HtmlBody]:
        """One section's bytes, taken only if they say they are the HTML body.

        `BODY[k]` keeps the transfer encoding on, so the headers go back in front
        and `email` decodes the part exactly as a whole-message parse would: the
        same base64, the same charset rules, no second implementation of either —
        and then `_is_html_body` re-checks the reconstructed part, which is the
        one place a server describing its sections oddly cannot slip a file past.
        """
        try:
            items = (f"(BODY.PEEK[{section}])" if headers is not None else
                     f"(BODY.PEEK[{section}.MIME] BODY.PEEK[{section}])")
            _typ, msgs = conn.uid("FETCH", self._bytes(str(int(uid))),
                                  self._bytes(items))
        except Exception as e:
            log.debug("html section %s could not be read: %s", section, e)
            return None
        body = self._sections(msgs).get(section)
        if not body:
            return None
        if headers is None:
            head = self._sections(msgs, mime=True).get(section) or b""
        else:
            head = headers.as_bytes().split(b"\r\n\r\n", 1)[0]
        try:
            part = _mime_part(head + b"\r\n\r\n" + body)
        except Exception as e:
            log.debug("html section %s did not parse: %s", section, e)
            return None
        if not _is_html_body(part):
            return None
        text, charset = decode_text_part(part)
        if not text.strip():
            return None
        return HtmlBody(text, charset, section)

    def content_ids(self, conn, name: str, uid: int) -> dict:
        """`{content-id: filename}` for one message's parts.

        An inline image is addressed from the HTML as `cid:…`, and the only thing
        that ties such a reference to a file is the part's own `Content-ID`
        header: `BODYSTRUCTURE` does not carry it, so this is the section probe,
        which asks the server what each part says it is. Two or three small
        commands, which is why the caller runs it only for a mail whose HTML
        actually mentions a cid, and caches the answer.
        """
        out: dict = {}
        try:
            heads = self.part_headers(conn, name, uid)
        except Exception as e:
            log.debug("no section headers to read content-ids from: %s", e)
            return out
        for _section, head in (heads or {}).items():
            part = _mime_part(head)
            if part is None:
                continue
            cid = (part.get("Content-ID") or "").strip().strip("<>").lower()
            fn = (part.get_filename() or "").strip()
            if cid and fn:
                out[decode_mime_words(cid).strip().lower()] = \
                    decode_mime_words(fn).strip()
        return out

    def gm_msgid_map(self, conn, name: str, uids, pause: float = 1.0,
                     debug: bool = False) -> dict:
        """`{uid: X-GM-MSGID}` for a folder: Gmail's own ids, no bodies read.

        The link `mail.google.com/mail/u/N/#all/<hex>` is made from that number,
        and mail indexed before it was asked for has no way to know it. One
        batched FETCH of a Gmail item per page says it for a whole folder, at
        the same cost and with the same politeness (`reconcile_batch_size`,
        then a pause) as the header sweep reconciliation already runs.
        """
        self.select_readonly(conn, name)
        ids = sorted(int(u) for u in (uids or ()) if u)
        if not ids:
            return {}
        size = max(1, int(self.cfg.get("reconcile_batch_size",
                                      DEFAULTS["reconcile_batch_size"])))
        items = self._bytes("(UID X-GM-MSGID)")
        out: dict = {}
        for i in range(0, len(ids), size):
            page = self._bytes(",".join(str(u) for u in ids[i:i + size]))
            _typ, msgs = conn.uid("FETCH", page, items)
            for item in msgs or []:
                # The shape matters more than it looks. imaplib hands back a
                # fetch item that carries no literal as one plain line of bytes,
                # and only an item with a literal as a `(prefix, literal)` pair -
                # and `X-GM-MSGID` carries no literal, so Gmail's answer is the
                # line. Reading only the pair form answers nothing at all against
                # the real server while a test double that always hands back pairs
                # stays green, which is how every one of these links came to be
                # written empty.
                try:
                    line = bytes(item[0] if isinstance(item, tuple) and item
                                 else item or b"")
                except (TypeError, ValueError):
                    continue
                uid, gm = _UID_RE.search(line), _gm_msgid(line)
                if uid and gm:
                    out[int(uid.group(1))] = gm
            if i + size < len(ids):
                time.sleep(pause)          # polite to IMAP rate limits, as in fetch()
        return out

    def count_messages(self, since: Optional[str], folders: list,
                       limit: Optional[int] = None,
                       before: Optional[str] = None) -> int:
        """Count how many messages fetch() will return (for a progress bar).

        Mirrors fetch()'s per-folder logic (SINCE/BEFORE criteria + last-N
        limit) but only runs a cheap UID SEARCH - no message data is
        transferred.
        """
        conn = self._connect()
        total = 0
        try:
            for folder in folders:
                try:
                    self.select_readonly(conn, folder)
                except Exception:
                    continue
                if before:
                    criteria = f"BEFORE {before}"
                elif since:
                    criteria = f"SINCE {since}"
                else:
                    criteria = "ALL"
                _, data = conn.uid("SEARCH", None, self._bytes(criteria))
                n = len(self._joined(data).split())
                if limit is not None:
                    n = min(n, limit)
                total += n
        finally:
            try:
                conn.logout()
            except Exception:
                pass
        return total

    def fetch(self, since: Optional[str], folders: list, limit: Optional[int] = None,
              before: Optional[str] = None, debug: bool = False):
        """Yield Message objects, most recent first.

        `since` is an IMAP date like '01-Oct-2026' (SINCE), `before` is an
        IMAP date for a BEFORE search (used to resume a --full run), or None
        for ALL.
        """
        conn = self._connect(debug)
        try:
            batch = int(self.cfg.get("fetch_batch_size", 10))
            for folder in folders:
                try:
                    self.select_readonly(conn, folder)
                except Exception as e:
                    log.warning("folder %s unavailable: %s", folder, e)
                    continue
                if before:
                    criteria = f"BEFORE {before}"
                elif since:
                    criteria = f"SINCE {since}"
                else:
                    criteria = "ALL"
                _, data = conn.uid("SEARCH", None, self._bytes(criteria))
                # The SEARCH response can span multiple lines; join them all.
                uids = self._joined(data).split()
                if limit is not None and uids:
                    uids = uids[-limit:]  # most recent
                uids = uids[::-1]  # newest first: work back in time
                for i in range(0, len(uids), batch):
                    page = uids[i:i + batch]
                    yield from self._fetch_page(conn, page, folder)
                    time.sleep(1.0)  # be polite to IMAP rate limits
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    def _fetch_page(self, conn, page: list, folder: str, debug: bool = False):
        """Fetch a page of UIDs; on a BAD parse error, retry one UID at a time.

        Note: the message set is sent COMMA-separated. Gmail rejects the
        standard space-separated multi-UID form with 'Could not parse command'
        but accepts comma (and range) form.

        `(UID RFC822)` and not plain `RFC822`: the UID sitting in the response
        prefix is what ties a stored row back to the folder it was fetched
        from, which is how `reconcile_deletions` can check a folder by UID
        instead of re-reading its headers. Reading it out of a reply that was
        on its way anyway costs nothing.

        `X-GM-MSGID` rides along for the same reason: it is the number Gmail's
        own web UI uses (`mail.google.com/mail/u/0/#all/<hex>`), and it is a
        *fetch item*, not a header inside the body — so asking for it in the
        command that was already running is the only cheap way to learn it, and
        the one thing that makes "open in Gmail" a link to this message instead
        of a search for its subject.
        """
        import imaplib

        items = self._bytes("(UID X-GM-MSGID RFC822)")
        try:
            _, msgs = conn.uid("FETCH", b",".join(page), items)
        except imaplib.IMAP4.error as e:
            if "BAD" in str(e):
                log.warning("batch UID FETCH failed (%s); retrying one by one", e)
                for uid in page:
                    _, msgs = conn.uid("FETCH", uid, items)
                    for item in msgs or []:
                        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (bytes, bytearray)):
                            import email as emaillib
                            # here the UID is known: it is the one we asked for
                            yield self._parse(emaillib.message_from_bytes(
                                bytes(item[1])), folder, uid=_as_int(uid),
                                gm_msgid=_gm_msgid(item[0]))
                return
            raise
        for item in msgs or []:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (bytes, bytearray)):
                import email as emaillib
                prefix = bytes(item[0] or b"")
                found = _UID_RE.search(prefix)
                yield self._parse(emaillib.message_from_bytes(bytes(item[1])), folder,
                                  uid=int(found.group(1)) if found else 0,
                                  gm_msgid=_gm_msgid(prefix))

    def _hdr(self, value) -> str:
        """Header value as a human would type it (RFC 2047 decoded)."""
        value = (value or "").strip()
        if self.cfg.get("decode_mime_words", True):
            return decode_mime_words(value)
        return value

    def _parse(self, msg, folder: str, uid: int = 0,
               gm_msgid: str = "") -> Message:
        """One RFC822 message into a `Message` row.

        `gm_msgid` is Gmail's own id as read out of the FETCH reply prefix: the
        body does not carry it, so a row indexed from `(UID RFC822)` alone could
        never link back into Gmail's web UI.
        """
        gm_msgid = str(gm_msgid or "").strip() or (msg.get("X-GM-MSGID") or "").strip()
        mid = (msg.get("Message-ID") or "").strip()
        if not mid:
            mid = f"{folder}:{gm_msgid or 'unknown'}"
        date = parse_date(msg.get("Date"))
        sender = parseaddr(msg.get("From") or "")[1]
        to = self._addr_list(msg, "To")
        cc = self._addr_list(msg, "Cc")
        # Decoded, not raw: 563 of 4 452 subjects in a real mailbox were stored
        # as their MIME encoded-words, which both retrieval legs then searched.
        subject = self._hdr(msg.get("Subject"))

        plain, html = "", ""
        attachments: list[Attachment] = []
        for part in msg.walk():
            if part.get_content_disposition() == "attachment" or part.get_filename():
                fn = part.get_filename()
                if fn:
                    fn = self._hdr(fn)
                    data = part.get_payload(decode=True) or b""
                    attachments.append(
                        Attachment(fn, part.get_content_type() or "application/octet-stream", data)
                    )
                continue
            ct = part.get_content_type()
            if ct == "text/plain" and not plain:
                plain = decode_part(part)
            elif ct == "text/html" and not html:
                html = decode_part(part)
        body_text = plain if plain.strip() else html_to_text(html)

        refs = []
        for h in msg.get_all("References", []) or []:
            refs.extend(h.split())
        irt = (msg.get("In-Reply-To") or "").strip()
        if irt:
            refs.append(irt)

        # Direct Gmail web URL from Gmail's own id (decimal -> hex). No extra
        # IMAP round-trip: `X-GM-MSGID` was asked for in the fetch that was
        # already running. Empty when it was not offered or is not numeric.
        gmail_url = gmail_message_url(gm_msgid, gmail_index(self.cfg))

        return Message(
            message_id=mid,
            folder=folder,
            date=date,
            sender=sender,
            to=to,
            cc=cc,
            subject=subject,
            body_text=body_text,
            attachments=attachments,
            thread_id=(msg.get("X-GM-THRID") or "").strip(),
            references=refs,
            gmail_url=gmail_url,
            # the UID this message was fetched under, from the reply prefix:
            # what lets the next reconcile check a folder by UID instead of
            # reading its headers again.
            uid=uid,
        )

    @staticmethod
    def _addr_list(msg, name: str) -> str:
        out = []
        for h in msg.get_all(name, []) or []:
            for part in h.split(","):
                addr = parseaddr(part)[1]
                if addr:
                    out.append(addr)
        return ", ".join(out)


# ---------------------------------------------------------------------------
# One IMAP login, kept for as long as it works
# ---------------------------------------------------------------------------

class GmailSession:
    """A logged-in, read-only IMAP connection reused across clicks.

    Fetching one attachment used to be: connect, TLS handshake, login, EXAMINE,
    FETCH, LOGOUT. Gmail answers that handshake in well over a second, and for a
    40 KB file it is the *whole* cost of the click — which is what made opening a
    picture feel slow. The login is the expensive part, so this holds on to it:
    `call(folder, fn)` hands `fn` a connection with `folder` examined, logging in
    only when there is none and reconnecting once, transparently, when Gmail has
    since closed an idle socket.

    It is still a reader and nothing else: the connection comes from
    `GmailFetcher._connect`, the folder is entered with `select_readonly`
    (EXAMINE), and the verbs used through it are the read-only ones that class
    lists. Holding a connection open does not make this program able to change
    anybody's mailbox.

    The lock lets the GUI's two media workers share one login: a second cold
    fetch waits a moment for the first rather than opening a parallel login Gmail
    would rather not see — and a thumbnail read never waits here at all, because
    it is served from the local sidecar and does not come through this class.

    A folder is re-examined only when the folder *changes*: the connection
    remembers where it is, and repeated clicks in one thread stay one FETCH deep.
    """

    #: Idle seconds after which the next call says NOOP first. It is cheaper to
    #: discover a dropped socket here, in a place that can retry, than in the
    #: middle of a FETCH whose bytes are already half-read.
    KEEPALIVE_AFTER = 60.0

    def __init__(self, fetcher, keepalive_after: Optional[float] = None):
        self.fetcher = fetcher
        self.keepalive_after = float(
            self.KEEPALIVE_AFTER if keepalive_after is None
            else keepalive_after)
        self._conn = None
        self._folder = None
        self._used = 0.0
        self._lock = threading.Lock()
        self.reconnects = 0

    def _fresh(self):
        conn = self.fetcher._connect()
        self._conn = conn
        self._folder = None
        self._used = time.time()
        return conn

    def _drop(self):
        """Log the held connection out, quietly: it may already be gone."""
        conn, self._conn, self._folder = self._conn, None, None
        if conn is not None:
            try:
                conn.logout()
            except Exception:
                pass

    def _alive(self, conn) -> bool:
        """True when the held connection still answers to NOOP."""
        if time.time() - self._used <= self.keepalive_after:
            return True
        try:
            conn.noop()
            self._used = time.time()
            return True
        except Exception as e:
            log.debug("idle IMAP connection did not answer to NOOP: %s", e)
            return False

    def call(self, folder: str, fn):
        """`fn(conn)` over the shared login, with `folder` examined.

        One retry, and only for a connection that looks *gone* (`OSError`, TLS
        error, or imaplib's `abort`): a folder that does not exist, or a message
        Gmail does not have, is an answer and not a broken pipe, so it is passed
        to the caller instead of being fetched a second time.
        """
        import imaplib
        dead = (OSError, EOFError, imaplib.IMAP4.abort)
        with self._lock:
            for attempt in (1, 2):
                try:
                    conn = self._conn or self._fresh()
                    if not self._alive(conn):
                        self._drop()
                        conn = self._fresh()
                    if self._folder != folder:
                        self.fetcher.select_readonly(conn, folder)
                        self._folder = folder
                    self._used = time.time()
                    return fn(conn)
                except dead as e:
                    if attempt == 2:
                        raise
                    log.debug("reopening the IMAP connection: %s", e)
                    self.reconnects += 1
                    self._drop()
            return None                      # unreachable: attempt 2 raises

    def close(self) -> None:
        with self._lock:
            self._drop()


# ---------------------------------------------------------------------------
# Attachment text extraction
# ---------------------------------------------------------------------------

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def resize_image_b64(data: bytes, mime: str, max_side: int = 1024) -> tuple[str, str]:
    """Return (base64, mime). Resizes to keep payloads small; falls back to raw bytes."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), "image/png"
    except Exception:
        return base64.b64encode(data).decode(), mime


# ---------------------------------------------------------------------------
# Local previews: the thumbnail sidecar
# ---------------------------------------------------------------------------

#: Longest side of a stored preview, in pixels. Small enough that every indexed
#: image of a real mailbox (4 000 of them) is a few tens of MB, large enough to
#: read a screenshot without opening it.
THUMB_MAX_SIDE = 320
THUMB_JPEG_QUALITY = 82


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _uses_alpha(img) -> bool:
    """Whether this image's alpha channel is actually doing something.

    Plenty of screenshots and mail merged pictures are saved as RGBA with an
    alpha channel that is opaque everywhere: the encoder wrote one because the
    format allows it. Keeping PNG for those costs about ten times what the
    picture needs (a live mailbox's largest previews were 190 KB PNGs of a 320 px
    image that could not show a single transparent pixel), and asking the pixels
    is the only way to tell the two apart. A palette image whose transparency
    lives in `info` rather than a band is alpha by definition, and a check that
    cannot run keeps the format that cannot be wrong.
    """
    if img.mode == "P" and "transparency" in img.info:
        return True
    if "A" not in img.getbands():
        return False
    try:
        return img.getchannel("A").getextrema()[0] < 255
    except Exception:
        return True


def make_thumb(data: bytes, mime: str = "",
               max_side: int = 0) -> Optional[tuple]:
    """A small copy of an image, for the reading pane: `(bytes, mime, w, h)`.

    The indexer decodes and resizes every image attachment anyway
    (`resize_image_b64`), so the pixels are in hand exactly once per message and
    a preview is nearly free to keep next to them. That is the whole reason this
    exists: opening a picture used to cost an IMAP login plus the *whole*
    message, and a 40 KB logo inside a 5 MB mail paid for the 5 MB.

    JPEG unless the picture shows alpha — a transparent screenshot goes muddy
    behind a JPEG, and that is the one case worth PNG's bytes; an alpha channel
    that is opaque everywhere is not that case, and `_uses_alpha` is what tells
    them apart. An animated GIF previews as its first frame; the file itself is
    never touched, only copied.

    Bytes PIL cannot read answer `None`, and the chip falls back to the
    extension icon: an unreadable attachment still opens as itself.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        side = int(max_side or THUMB_MAX_SIDE)
        if side > 0 and max(img.size) > side:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            img.thumbnail((side, side), resample)
        # Asked here, on the pixels about to be saved rather than the ones that
        # arrived: on a 4,000 px screenshot that is the difference between
        # reading a channel of the preview and one of the original.
        alpha = _uses_alpha(img)
        buf = io.BytesIO()
        if alpha:
            img.save(buf, format="PNG", optimize=True)
            out_mime = "image/png"
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=THUMB_JPEG_QUALITY,
                     optimize=True, progressive=True)
            out_mime = "image/jpeg"
        return buf.getvalue(), out_mime, int(img.size[0]), int(img.size[1])
    except Exception as e:
        # A picture PIL cannot open is a fact about one file, not an error: the
        # reading pane loses its thumbnail and keeps the file.
        log.debug("no thumbnail for this attachment: %s", e)
        return None


def media_db_path(cfg: dict) -> str:
    """Where previews live: beside the index, unless `media_db_path` says otherwise.

    Derived from `db_path` rather than fixed, so `--db baseline.db` measures a
    frozen copy without reaching into the live mailbox's previews — the same
    isolation every other `--db` override gives you.
    """
    p = str((cfg or {}).get("media_db_path") or "").strip()
    if p:
        return p
    db = Path(str((cfg or {}).get("db_path") or DEFAULTS["db_path"]))
    return str(db.with_suffix(".media.db"))


class MediaCache:
    """Previews of indexed images, in a sidecar file — not in `mailbox.db`.

    Opening an image attachment means asking Gmail for bytes this index
    deliberately never stored. That is right for the file and wrong for a
    preview: a 320 px copy answers "which one is this" with no network round
    trip, and it is the same picture the embedding endpoint was given. So
    previews are kept, and the full file still comes from Gmail on click.

    Not a table in `mailbox.db`, on purpose: `Store` warms ~146 MB of vectors on
    open, a baseline copy is made by copying that one file, and image BLOBs
    belong in neither. `VACUUM`, `--db` copies and `bench` are untouched.

    Connections are per thread (`threading.local`): the GUI's media pool has two
    workers and a sqlite3 connection belongs to the thread that opened it, the
    same affinity rule `Store` follows. WAL and a busy timeout are needed here
    and nowhere else, because the indexer subprocess writes this file *while*
    the GUI reads it — unlike the index, which the GUI only re-opens once the
    child has finished.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS media(
            message_id TEXT NOT NULL, filename TEXT NOT NULL,
            mime TEXT, width INTEGER, height INTEGER,
            bytes BLOB NOT NULL, source_bytes INTEGER DEFAULT 0,
            created_at TEXT, accessed_at TEXT,
            PRIMARY KEY (message_id, filename));
        CREATE INDEX IF NOT EXISTS media_msg ON media(message_id);
        -- Which MIME section a file sat in, per folder+UID: a file opened twice
        -- is then read from the section learned the first time, without the
        -- header probe. Still verified, because folders renumber under a UID.
        CREATE TABLE IF NOT EXISTS parts(
            folder TEXT NOT NULL, uid INTEGER NOT NULL, filename TEXT NOT NULL,
            section TEXT, octets INTEGER, PRIMARY KEY (folder, uid, filename));
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
        """

    def __init__(self, path: str, cfg: Optional[dict] = None):
        self.path = str(path)
        self.cfg = dict(cfg or {})
        self.budget_bytes = max(1, int(self.cfg.get(
            "media_cache_mb", DEFAULTS["media_cache_mb"]))) * 1_000_000
        self._local = threading.local()
        Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        seed = sqlite3.connect(self.path)
        try:
            seed.execute("PRAGMA journal_mode=WAL")
            seed.executescript(self.SCHEMA)
            seed.commit()
        finally:
            seed.close()

    def _db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # -- previews ------------------------------------------------------------
    def put(self, message_id: str, filename: str, data: bytes, mime: str,
            width: int = 0, height: int = 0, source_bytes: int = 0) -> None:
        now = _now_iso()
        self._db().execute(
            "INSERT OR REPLACE INTO media (message_id, filename, mime, width,"
            " height, bytes, source_bytes, created_at, accessed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (message_id, filename, mime, int(width), int(height),
             sqlite3.Binary(data), int(source_bytes), now, now))
        self._db().commit()

    def put_attachment(self, message_id: str, filename: str, data: bytes,
                       mime: str = "") -> Optional[dict]:
        """Preview one attachment, when it is a picture PIL can read."""
        thumb = make_thumb(data, mime, int(self.cfg.get(
            "thumb_max_side", DEFAULTS["thumb_max_side"])))
        if thumb is None:
            return None
        blob, t_mime, w, h = thumb
        self.put(message_id, filename, blob, t_mime, w, h, len(data or b""))
        return {"filename": filename, "mime": t_mime, "width": w, "height": h}

    def get(self, message_id: str, filename: str) -> Optional[dict]:
        row = self._db().execute(
            "SELECT mime, bytes, width, height FROM media"
            " WHERE message_id = ? AND filename = ?",
            (message_id, filename)).fetchone()
        if row is None:
            return None
        self._db().execute("UPDATE media SET accessed_at = ?"
                           " WHERE message_id = ? AND filename = ?",
                           (_now_iso(), message_id, filename))
        self._db().commit()
        return {"mime": row[0] or "image/jpeg", "bytes": bytes(row[1]),
                "width": row[2] or 0, "height": row[3] or 0}

    def for_message(self, message_id: str) -> dict:
        """`{filename: {width, height}}` for the previews this message has."""
        return {r[0]: {"width": r[1] or 0, "height": r[2] or 0}
                for r in self._db().execute(
                    "SELECT filename, width, height FROM media"
                    " WHERE message_id = ?", (message_id,))}

    def first_for_messages(self, message_ids) -> dict:
        """`{message_id: filename}` — one preview per listed message.

        One query for a whole page: the list pane wants a single thumbnail per
        row, and asking per row would be fifty queries per scroll.
        """
        ids = [m for m in (message_ids or ()) if m]
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        out: dict = {}
        for mid, name in self._db().execute(
                f"SELECT message_id, filename FROM media"
                f" WHERE message_id IN ({marks})"
                f" ORDER BY message_id, filename", ids):
            out.setdefault(mid, name)
        return out

    def thumb_message_ids(self) -> set:
        """Every message with at least one preview (what `media --backfill` skips)."""
        return {r[0] for r in self._db().execute(
            "SELECT DISTINCT message_id FROM media")}

    def has(self, message_id: str, filename: str) -> bool:
        return self._db().execute(
            "SELECT 1 FROM media WHERE message_id = ? AND filename = ?",
            (message_id, filename)).fetchone() is not None

    def delete_messages(self, message_ids) -> int:
        """Drop previews of messages this index no longer holds.

        The delete mirror promises no orphans: message row, chunks, vectors, FTS
        entries — and now the preview, or a dropped message would keep showing a
        picture of itself in the list.
        """
        ids = [m for m in (message_ids or ()) if m]
        if not ids:
            return 0
        done = 0
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            done += self._db().execute(
                f"DELETE FROM media WHERE message_id IN ({marks})",
                chunk).rowcount
        self._db().commit()
        return done

    def prune_to_budget(self) -> int:
        """Drop least-recently-seen previews until the sidecar fits its budget."""
        db = self._db()
        total = int(db.execute("SELECT COALESCE(SUM(length(bytes)),0) FROM media"
                               ).fetchone()[0] or 0)
        if total <= self.budget_bytes:
            return 0
        gone = 0
        for mid, name, size in db.execute(
                "SELECT message_id, filename, length(bytes) FROM media"
                " ORDER BY COALESCE(accessed_at, created_at), message_id,"
                " filename").fetchall():
            if total <= self.budget_bytes:
                break
            db.execute("DELETE FROM media WHERE message_id = ? AND filename = ?",
                       (mid, name))
            total -= int(size or 0)
            gone += 1
        db.commit()
        if gone:
            log.info("media cache over budget: dropped %d preview(s)", gone)
        return gone

    def clear(self) -> int:
        n = int(self._db().execute("SELECT COUNT(1) FROM media").fetchone()[0])
        self._db().execute("DELETE FROM media")
        self._db().commit()
        return n

    def count(self) -> int:
        """How many previews exist.

        `stats()` also measures their bytes, which means reading every blob in
        the file: right for `media --stats`, wrong for a status poll that runs
        every few seconds.
        """
        return int(self._db().execute("SELECT COUNT(1) FROM media").fetchone()[0])

    def part_count(self) -> int:
        """How many MIME sections this sidecar remembers the way to.

        Counted apart from the previews on purpose: a remembered section costs a
        couple of bytes, survives a `clear()`, and saves the probing on every
        future open of that file — preview or no preview.
        """
        return int(self._db().execute("SELECT COUNT(1) FROM parts").fetchone()[0])

    def stats(self) -> dict:
        row = self._db().execute(
            "SELECT COUNT(1), COALESCE(SUM(length(bytes)),0),"
            " COALESCE(SUM(source_bytes),0) FROM media").fetchone()
        return {"path": self.path, "rows": int(row[0] or 0),
                "bytes": int(row[1] or 0), "source_bytes": int(row[2] or 0),
                "budget_bytes": self.budget_bytes,
                "file_bytes": (Path(self.path).stat().st_size
                               if Path(self.path).exists() else 0)}

    # -- MIME section memory -------------------------------------------------
    def part(self, folder: str, uid: int, filename: str) -> Optional[tuple]:
        row = self._db().execute(
            "SELECT section, octets FROM parts WHERE folder = ? AND uid = ?"
            " AND filename = ?", (str(folder), int(uid or 0),
                                  str(filename))).fetchone()
        return (row[0], int(row[1] or 0)) if row else None

    def remember_part(self, folder: str, uid: int, filename: str,
                      section: str, octets: int) -> None:
        self._db().execute(
            "INSERT OR REPLACE INTO parts (folder, uid, filename, section,"
            " octets) VALUES (?,?,?,?,?)",
            (str(folder), int(uid or 0), str(filename), str(section),
             int(octets)))
        self._db().commit()

    # -- run state -----------------------------------------------------------
    def get_state(self, key: str, default: str = "") -> str:
        row = self._db().execute("SELECT value FROM state WHERE key = ?",
                                 (key,)).fetchone()
        return row[0] if row and row[0] is not None else default

    def set_state(self, key: str, value: str) -> None:
        self._db().execute("INSERT OR REPLACE INTO state (key, value)"
                           " VALUES (?,?)", (key, str(value)))
        self._db().commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


def open_media_cache(cfg: dict) -> Optional[MediaCache]:
    """The sidecar, or None when previews are off or the file cannot be opened.

    Never fatal: previews are a convenience, and a read-only share or a full
    disk must cost the user a thumbnail, not their search box.
    """
    if not cfg.get("store_thumbs", DEFAULTS["store_thumbs"]):
        return None
    path = media_db_path(cfg)
    try:
        return MediaCache(path, cfg)
    except Exception as e:
        log.warning("thumbnails are off: %s could not be opened (%s)", path, e)
        return None


def pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        log.warning("pdf text extraction failed: %s", e)
        return ""


def pdf_page_images(data: bytes) -> list[tuple[bytes, str]]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.warning("pdf_pages_as_images requires PyMuPDF (pip install PyMuPDF)")
        return []
    out = []
    try:
        doc = fitz.open(stream=data)
        for page in doc:
            pix = page.get_pixmap(dpi=100)
            out.append((pix.tobytes("png"), "image/png"))
    except Exception as e:
        log.warning("pdf page rendering failed: %s", e)
    return out


def docx_text(data: bytes) -> str:
    try:
        from docx import Document
        d = Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    except Exception as e:
        log.warning("docx extraction failed: %s", e)
        return ""


def xlsx_text(data: bytes) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                vals = [str(v) for v in row if v is not None]
                if vals:
                    out.append(" | ".join(vals))
        return "\n".join(out)
    except Exception as e:
        log.warning("xlsx extraction failed: %s", e)
        return ""


def pptx_text(data: bytes) -> str:
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(data))
        out = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    out.append(shape.text_frame.text)
        return "\n".join(out)
    except Exception as e:
        log.warning("pptx extraction failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Content extraction (message -> chunks)
# ---------------------------------------------------------------------------

class ContentExtractor:
    """Turns a Message into index chunks, at every configured granularity.

    Chunk dict shape:
      {kind, src, tier, pos, nwin, canonical, label, text [, image_b64, mime]}
      kind   body | meta | attachment     (unchanged from pre-tier databases)
      src    source key: 'body' | 'meta' | 'attachment: <name>'
      tier   small | large | xlarge for text, 'media' for image/audio
      pos/nwin  window index within its tier (0/1 for a whole-source chunk)
      canonical  holds the complete capped source text (nwin is then 1)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sizes = chunk_sizes_of(cfg)
        self.overlap_frac = float(cfg.get("chunk_overlap_frac", 0.15))
        self.max_source = int(cfg.get("max_source_chars", 6000))
        self.max_windows = int(cfg.get("max_windows_per_source", 8))
        self.index_tiers = {t for t in (cfg.get("index_tiers") or [])}

    # -- tiering helpers --------------------------------------------------
    def _tiered(self, kind: str, src: str, text: str) -> list[dict]:
        """One source text -> chunk dicts for every tier that adds value.

        The canonical (whole-source) chunk is always kept, even when its tier
        is missing from index_tiers: it is the row `reindex` re-derives the
        finer tiers from, so dropping it would break re-indexing later.
        """
        out = []
        for r in tiered_chunks(text, self.sizes, self.overlap_frac,
                               self.max_source, self.max_windows):
            if (not r["canonical"] and self.index_tiers
                    and r["tier"] not in self.index_tiers):
                continue
            out.append(self._chunk(kind, src, r))
        return out

    @staticmethod
    def _chunk(kind: str, src: str, r: dict) -> dict:
        return {
            "kind": kind, "src": src, "tier": r["tier"], "pos": r["pos"],
            "nwin": r["nwin"], "canonical": bool(r["canonical"]),
            "label": chunk_label(src, r["tier"], r["pos"], r["nwin"],
                                 bool(r["canonical"])),
            "text": r["text"],
        }

    @staticmethod
    def _media(src: str, text: str, image_b64: str, mime: str) -> dict:
        """A media chunk: one indivisible item, so it is its own canonical row
        at the coarsest granularity (tier 'media', never re-derived)."""
        return {"kind": "attachment", "src": src, "tier": MEDIA_TIER, "pos": 0,
                "nwin": 1, "canonical": True, "label": src, "text": text,
                "image_b64": image_b64, "mime": mime}

    def extract(self, msg: Message, thread_context: str) -> list[dict]:
        """Return chunk dicts for one message (body, meta, then attachments).

        The body is split into what the sender wrote and what they quoted
        (`clean_text`): only the first half is embedded, the second stays
        available to bm25 through the `quoted` column. Every chunk also carries
        a `title` - the Subject for body/meta, the filename for attachments -
        because EmbeddingGemma 2 is trained on `title: … | text: …` and a
        mailbox always has one, so `title: none` would waste the signal.
        """
        chunks: list[dict] = []

        body = msg.body_text.strip()
        quoted = ""
        if body and self.cfg.get("clean_text", True):
            body, quoted = clean_text(body)
        if body:
            chunks += self._tiered("body", "body", body)

        # Meta chunk: sender, recipients, attachment extensions, date, subject,
        # plus thread context so the conversation itself is searchable.
        exts = ", ".join(
            Path(a.name).suffix.lower().lstrip(".") for a in msg.attachments
        ) or "none"
        meta_lines = [
            f"From: {msg.sender}",
            f"To: {msg.to}",
        ]
        if msg.cc:
            meta_lines.append(f"Cc: {msg.cc}")
        meta_lines += [
            f"Date: {msg.date:%Y-%m-%d}",
            f"Subject: {msg.subject}",
            f"Attachments: {exts}",
        ]
        if thread_context:
            meta_lines.append(f"Thread: {thread_context}")
        chunks += self._tiered("meta", "meta", "\n".join(meta_lines))

        for att in msg.attachments:
            if len(att.data) > self.cfg["max_attachment_mb"] * 1024 * 1024:
                log.warning("skip oversized attachment %s", att.name)
                continue
            chunks += self._attachment_chunks(att)

        # One carrier row for the stripped text: the canonical body chunk, or
        # the meta chunk for a reply that was nothing but a quote.
        if quoted and self.cfg.get("keep_quoted_in_fts", True):
            carrier = next((c for c in chunks
                            if c["kind"] == "body" and c["canonical"]), None)
            if carrier is None:
                carrier = next((c for c in chunks if c["canonical"]), None)
            if carrier is not None:
                carrier["quoted"] = quoted
        for c in chunks:
            c.setdefault("title", title_of(c, msg.subject))
        return chunks

    def _attachment_chunks(self, att: Attachment) -> list[dict]:
        ct = (att.content_type or "").lower()
        name = att.name or ""
        lower = name.lower()
        src = f"attachment: {name}"

        if ct in IMAGE_TYPES or lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            b64, mime = resize_image_b64(att.data, att.content_type or "image/png")
            return [self._media(src, src, b64, mime)]

        if ct == "application/pdf" or lower.endswith(".pdf"):
            text = pdf_text(att.data)
            if len(text.strip()) < 100 and self.cfg["pdf_pages_as_images"]:
                out = []
                for i, (raw, mime) in enumerate(pdf_page_images(att.data), 1):
                    b64 = base64.b64encode(raw).decode()
                    page_src = f"attachment: {name} (page {i})"
                    out.append(self._media(page_src, page_src, b64, mime))
                if out:
                    return out
            return self._tiered("attachment", src, text) if text.strip() else []

        if ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or lower.endswith(".docx"):
            return self._tiered("attachment", src, docx_text(att.data))

        if ct == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or lower.endswith(".xlsx"):
            return self._tiered("attachment", src, xlsx_text(att.data))

        if ct == "application/vnd.openxmlformats-officedocument.presentationml.presentation" or lower.endswith(".pptx"):
            return self._tiered("attachment", src, pptx_text(att.data))

        if self.cfg["embed_audio"]:
            return [self._media(src, src, base64.b64encode(att.data).decode(), ct)] \
                if ct.startswith("audio/") else []

        return []  # other types: metadata only (extension is in the meta chunk)


# ---------------------------------------------------------------------------
# Index metadata helpers (shared by the extractor and the store)
# ---------------------------------------------------------------------------

# Added to `chunks` by the multi-resolution upgrade. All nullable or defaulted,
# so a pre-tier database opens, reads and searches unchanged: every legacy row
# is a whole-source chunk, which is exactly what `canonical DEFAULT 1` says.
#
# `text_hash` is the fingerprint of the normalised text: it makes duplicate
# windows detectable without comparing strings (one body in a real inbox was
# embedded 29 times, once per message of its thread). `quoted` holds the text
# hygiene stripped out of the chunk, so it is BM25-searchable but never
# embedded - which is the whole point of cleaning the body.
CHUNK_COLUMNS = (
    ("tier", "TEXT"),
    ("src", "TEXT"),
    ("pos", "INTEGER DEFAULT 0"),
    ("nwin", "INTEGER DEFAULT 1"),
    ("canonical", "INTEGER NOT NULL DEFAULT 1"),
    ("search_meta", "TEXT"),
    ("text_hash", "TEXT"),
    ("quoted", "TEXT"),
)

ATTACHMENT_PREFIX = "attachment: "
# BM25 columns. `quoted` is indexed but never embedded, so the text hygiene
# removes from a vector stays reachable by keyword search.
FTS_COLUMNS = ("meta", "text", "quoted")


def title_of(chunk: dict, subject: str = "") -> str:
    """The `title:` half of this chunk's embedding input.

    Attachments are titled by filename (what a user types to find one), body
    and meta chunks by Subject. A paginated PDF page keeps its page marker:
    'report.pdf (page 3)' is a better title than the bare filename.
    """
    src = chunk.get("src") or chunk.get("label") or ""
    if src.startswith(ATTACHMENT_PREFIX):
        return src[len(ATTACHMENT_PREFIX):].strip() or (subject or _NO_TITLE)
    return (subject or "").strip() or _NO_TITLE


def normalise_for_hash(text: str) -> str:
    """Whitespace-and-case-folded text: what two windows are compared on."""
    return " ".join((text or "").split()).lower()


def media_label(chunk: dict) -> tuple[str, str]:
    """(title, text) for the text half of a media embedding request.

    The old label was the raw src string, so an image was embedded as the
    sentence 'attachment: Screenshot 2025-10-01.png' - text describing a
    filename, which is why image search returned whatever shared the word
    'screenshot' with the query. The filename belongs in `title:`; the text
    half says what the item is.
    """
    title = title_of(chunk)
    mime = (chunk.get("mime") or "").lower()
    src = chunk.get("src") or ""
    if "page " in src:
        return title, "scanned pdf page"
    if mime.startswith("image/"):
        return title, "image attachment"
    if mime.startswith("audio/"):
        return title, "voice recording"
    return title, "attachment"


#: What a query image is allowed to be: the same set the indexer embeds, so an
#: accepted upload is also comparable against what is in the index. Anything
#: else is refused before it costs an endpoint call.
QUERY_IMAGE_TYPES = frozenset(IMAGE_TYPES)


def query_image_chunk(image_b64: str, mime: str = "image/png") -> dict:
    """A query image as one media item, with no text half.

    Same shape an attachment chunk has, so it goes through the same envelope
    negotiation - minus the label: `text` is empty, which makes `_media_item`
    send the image on its own instead of pairing it with a sentence about a
    filename nobody typed.
    """
    return {"kind": "attachment", "src": "", "tier": MEDIA_TIER, "pos": 0,
            "nwin": 1, "canonical": True, "label": "", "text": "",
            "image_b64": image_b64, "mime": mime or "image/png"}


def text_hash(text: str) -> str:
    return hashlib.sha1(normalise_for_hash(text).encode("utf-8", "replace")).hexdigest()

# FTS5 tokenizer strings come from config and go into CREATE VIRTUAL TABLE,
# which takes no bound parameters - so the character set is restricted.
_TOKENIZER_OK = re.compile(r"^[A-Za-z0-9_ .,()\"-]+$")


def _host_of(url: str) -> str:
    """`scheme://host[:port]` of an endpoint, for the index fingerprint.

    The port stays in (a second server on the same host can serve a different
    build), path and query do not (they change with the dialect, not the model).
    """
    u = str(url or "").strip()
    if not u:
        return ""
    try:
        p = urlsplit(u)
        if p.netloc:
            return f"{p.scheme or 'http'}://{p.netloc}"
    except ValueError:
        pass
    return u


def safe_tokenizer(name: str) -> str:
    """Validate a configured FTS5 tokenizer string (falls back to the default)."""
    name = (name or "").strip()
    if not name or not _TOKENIZER_OK.match(name):
        if name:
            log.warning("ignoring unsafe fts_tokenizer %r (using the default)", name)
        return DEFAULTS["fts_tokenizer"]
    return name


def chunk_sizes_of(cfg: dict) -> dict:
    """The configured tier ladder as {name: chars}, falling back to DEFAULTS."""
    sizes = {k: int(v) for k, v in dict(cfg.get("chunk_sizes") or {}).items()
             if int(v) > 0}
    return sizes or dict(DEFAULTS["chunk_sizes"])


def chunk_search_meta(msg: Message, exts: str, kind: str, src: str) -> str:
    """The metadata half of a chunk's search document.

    FTS indexes this next to the chunk text, which is what makes a chunk
    findable by sender, recipient, subject, date and attachment filename -
    the exact strings BM25 handles well and embeddings handle badly.
    """
    parts = []
    if msg.sender:
        parts.append(f"From: {msg.sender}")
    if msg.to:
        parts.append(f"To: {msg.to}")
    if msg.cc:
        parts.append(f"Cc: {msg.cc}")
    if msg.subject:
        parts.append(f"Subject: {msg.subject}")
    parts.append(f"Date: {msg.date:%Y-%m-%d}")
    if kind == "attachment":
        parts.append(src)                      # 'attachment: invoice-2026-114.pdf'
    elif exts:
        parts.append(f"Attachments: {exts}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Thread context
# ---------------------------------------------------------------------------

def assign_thread_ids(msgs: list[Message]) -> None:
    """Fill in thread_id for messages without X-GM-THRID via References union-find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m in msgs:
        if not m.thread_id:
            for r in m.references:
                union(m.message_id, r)
            m.thread_id = find(m.message_id)


def build_thread_contexts(new_msgs: list[Message], store: "Store",
                         depth: int, chars: int) -> dict[str, str]:
    """For each new message, a compact context block of the prior mails in its thread."""
    known = store.load_context_map()  # message_id -> (date, sender, context_text)
    for m in new_msgs:
        known[m.message_id] = (m.date, m.sender, m.body_text.strip()[:chars])

    groups: dict[str, list[Message]] = {}
    for m in new_msgs:
        groups.setdefault(m.thread_id or m.message_id, []).append(m)

    out: dict[str, str] = {}
    for msgs in groups.values():
        msgs.sort(key=lambda m: m.date)
        for i, m in enumerate(msgs):
            prior = msgs[max(0, i - depth):i]
            if not prior:
                continue
            lines = []
            for p in reversed(prior):  # most recent first
                d, s, t = known.get(p.message_id, (None, "", ""))
                if t:
                    lines.append(f"{d:%Y-%m-%d} {s}: {t[:chars]}" if d else f"{s}: {t[:chars]}")
            if lines:
                out[m.message_id] = f"{len(prior)} prior message(s): " + " | ".join(lines)
    return out


# ---------------------------------------------------------------------------
# Embedding client (remote endpoint, batched)
# ---------------------------------------------------------------------------

class MultimodalRejected(Exception):
    """The endpoint rejected the multimodal (image/audio) input shape."""


class EmbeddingClient:
    """
    Thin client for the embedding endpoint.

    Default adapter speaks an OpenAI-compatible shape:
      POST {endpoint}  {"model": "...", "input": [...]}  ->  {"data": [{"embedding": [...]}]}
    "model" is sent only when embed_model is set (required by hosts such as Ollama).
    Text items are strings (task-prompt prefixed). Media items are one
    combined object, media + its text label in the same dict, so a
    multimodal endpoint returns a single combined embedding:
      {"type": "image_url",
       "image_url": {"url": "data:<mime>;base64,..."},
       "text": "Document: attachment: <name>"}
    (audio uses type "audio_url"). If your endpoint only accepts strings
    (it rejects the object form with "'input' field must be a string or an
    array of strings"), media chunks automatically fall back to embedding
    their text label, so the run still completes - attachments stay
    searchable by name/type, just not by image content. If your endpoint
    differs, replace the _post / _vec bodies only - the rest of the
    script is unaffected.
    """

    def __init__(self, cfg: dict):
        self.url = cfg["embed_endpoint"]
        self.key = cfg["embed_api_key"]
        self.model = (cfg.get("embed_model") or "").strip()
        self.dim = int(cfg["dim"])
        self.doc_prompt = cfg["doc_prompt"]
        self.query_prompt = cfg["query_prompt"]
        # Which instruction format the endpoint's model was trained with. The
        # literal "Document: " / "SearchQuery: " strings are *content* to
        # EmbeddingGemma 2 (cos of a text with and without that prefix is
        # 0.89-0.91, not 1.0), and its real contract is
        # `title: … | text: …` / `task: … | query: …`.
        scheme = str(cfg.get("prompt_scheme", "gemma")).strip().lower()
        if scheme not in ("gemma", "legacy"):
            log.warning("unknown prompt_scheme %r, using 'gemma'", scheme)
            scheme = "gemma"
        self.scheme = scheme
        self.query_task = str(cfg.get("query_task") or "search result")
        self.use_titles = bool(cfg.get("doc_title_from_subject", True))
        self.batch = int(cfg["batch_size"])
        self.timeout = float(cfg.get("embed_timeout", DEFAULTS["embed_timeout"]))
        self.media_multimodal = bool(cfg.get("embed_media_multimodal", True))
        shape = str(cfg.get("media_item_shape", DEFAULTS["media_item_shape"])).strip()
        if shape not in ("image_url", "content"):
            log.warning("unknown media_item_shape %r, using 'image_url'", shape)
            shape = "image_url"
        self.media_shape = shape
        self.multimodal_ok = None  # None = unknown yet, True/False = proven
        # Ollama's native endpoint ({"embeddings": [...]}) instead of the
        # OpenAI-compatible /v1/embeddings. Both are read; only the OpenAI one
        # can carry image content.
        self.native = "/api/embed" in self.url.lower()
        self._native_media_warned = False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def _body(self, item) -> dict:
        body = {"input": item}
        if self.model:
            body["model"] = self.model
        return body

    def _vectors(self, data: dict) -> list[np.ndarray]:
        """Vectors from either reply shape: OpenAI compatible
        `{"data": [{"embedding": [...]}]}` or Ollama native
        `{"embeddings": [[...], ...]}` (what /api/embed returns)."""
        if isinstance(data.get("data"), list):
            return [self._vec(d) for d in data["data"]]
        if isinstance(data.get("embeddings"), list):
            return [self._vec({"embedding": v}) for v in data["embeddings"]]
        raise RuntimeError(
            f"unexpected embedding reply, expected 'data' or 'embeddings', got "
            f"keys {sorted(data) if isinstance(data, dict) else type(data).__name__}")

    def _post(self, payload: dict, multimodal: bool = False) -> dict:
        last = "unknown error"
        for attempt in range(3):
            try:
                r = requests.post(self.url, json=payload, headers=self._headers(),
                                  timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    time.sleep(2 ** (attempt + 1))
                    continue
                if multimodal and r.status_code in (400, 422):
                    # Validation rejection of the object form: this endpoint
                    # only accepts strings. Not retriable - let the caller
                    # fall back to text.
                    raise MultimodalRejected(
                        f"HTTP {r.status_code}: {r.text[:200]}"
                    )
                if 400 <= r.status_code < 500:
                    # Other client errors (bad key, bad path, ...): no point
                    # in retrying.
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                return r.json()
            except (MultimodalRejected, RuntimeError):
                raise
            except requests.RequestException as e:
                last = str(e)
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"embedding endpoint failed after retries: {last}")

    def _vec(self, item: dict) -> np.ndarray:
        v = np.asarray(item["embedding"], dtype=np.float32)[:self.dim]
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def doc_input(self, text: str, title: str = "") -> str:
        """The exact string sent for one document chunk (never stored as text).

        Under the gemma scheme a chunk with no real title gets `title: none`,
        which is what the model was trained on for untitled documents; the
        subject/filename is the title a mailbox naturally provides.
        """
        if self.scheme == "gemma":
            t = " ".join((title or "").split()) if self.use_titles else ""
            return GEMMA_DOC.format(title=t or _NO_TITLE, text=text)
        return f"{self.doc_prompt}: {text}" if self.doc_prompt else text

    def query_input(self, text: str) -> str:
        if self.scheme == "gemma":
            return GEMMA_QUERY.format(task=self.query_task, query=text)
        return f"{self.query_prompt}: {text}" if self.query_prompt else text

    def embed_texts(self, texts: list[str],
                    titles: Optional[list[str]] = None) -> np.ndarray:
        out = []
        if titles is None:
            titles = [""] * len(texts)
        prefixed = [self.doc_input(t, ti) for t, ti in zip(texts, titles)]
        for i in range(0, len(prefixed), self.batch):
            chunk = prefixed[i:i + self.batch]
            data = self._post(self._body(chunk))
            out.extend(self._vectors(data))
        return np.stack(out)

    def embed_query(self, text: str) -> np.ndarray:
        data = self._post(self._body([self.query_input(text)]))
        return self._vectors(data)[0]

    def _media_item(self, chunk: dict) -> dict:
        """One media item, in the configured request shape.

        Two dialects are in the wild against OpenAI-style endpoints: the flat
        part object (OpenAI / vLLM / TEI / embed_server.py) and the same parts
        nested per input item under "content" (chat-style). Same bytes, same
        label - only the envelope differs, so `media_item_shape` picks one and
        the multimodal probe reports which one the endpoint accepted.
        """
        kind = "image_url" if chunk["mime"].startswith("image/") else "audio_url"
        part = {"type": kind,
                kind: {"url": f"data:{chunk['mime']};base64,{chunk['image_b64']}"}}
        t = chunk.get("text")
        # Google's card is explicit that prefixes apply to *text* only: the
        # image itself goes unprefixed, while its text half stays a document -
        # titled with the filename, which is what a user types to find it.
        if self.scheme == "gemma":
            title, what = media_label(chunk)
            labelled = self.doc_input(what, title) if t else None
        else:
            labelled = f"{self.doc_prompt}: {t}" if (t and self.doc_prompt) else t
        if self.media_shape == "content":
            # The envelope stays nested even with no text part, otherwise an
            # image-only item silently degrades to the other dialect and every
            # shape looks rejected.
            parts = [part] if not labelled else [part,
                                                 {"type": "text", "text": labelled}]
            return {"content": parts}
        return part if not labelled else {**part, "text": labelled}

    def _media_vector(self, chunk: dict) -> tuple[Optional[np.ndarray], list[str]]:
        """One media item to the endpoint, in whichever envelope it accepts.

        Returns `(vector, errors)`: the vector is None when every shape was
        rejected. Endpoints disagree on the envelope (llama.cpp wants nested
        "content" parts, OpenAI/vLLM/TEI a flat part object), so the configured
        shape is tried first and the other one once as a fallback - negotiated
        per run, logged, never guessed silently.
        """
        shapes = [self.media_shape] + [s for s in ("image_url", "content")
                                       if s != self.media_shape]
        errors: list[str] = []
        for shape in shapes:
            self.media_shape = shape
            try:
                data = self._post(self._body([self._media_item(chunk)]),
                                  multimodal=True)
            except MultimodalRejected as e:
                errors.append(f"media_item_shape={shape!r}: {e}")
                continue
            if errors:
                log.info("endpoint accepts image input as media_item_shape=%r "
                         "(set it in config.json to skip the probe)", shape)
            return self._vectors(data)[0], errors
        return None, errors

    def embed_query_image(self, data: bytes, mime: str = "image/png",
                          name: str = "") -> np.ndarray:
        """One image, as a query vector.

        The document side labels an attachment
        (`title: Screenshot.png | text: image attachment`) because a filename is
        how a person finds a file. A *query* image has no such half: its name
        comes from a screenshot folder and says nothing about tulips, and
        EmbeddingGemma's multimodal contract puts instruction prefixes on **text
        only** - the picture goes in bare. So this sends the pixels alone, which
        is what lets them meet the attachment vectors in the same space.

        Unlike `embed_media` there is no text fallback, and that is the point:
        the fallback for an attachment is a *document* labelled by its filename,
        while a fallback here would embed the sentence "image attachment" and
        answer a picture query with every mail that shares those words. A
        dead-end said plainly beats a ranking that never looked at the image.
        """
        if self.native:
            raise SystemExit(
                "this embed_endpoint is Ollama's native /api/embed, which accepts "
                "image input and ignores it for embedding models, so a picture "
                "cannot be searched against it. Serve a multimodal model over the "
                "OpenAI shape (embed_server.py, or a multimodal GGUF under "
                "llama.cpp) and point embed_endpoint at it.")
        if not self.media_multimodal:
            raise SystemExit(
                "embed_media_multimodal is off, so no image bytes are ever sent "
                "to the endpoint - turn it on to search by image, or describe the "
                "picture in words")
        b64, mime = resize_image_b64(data, mime or "image/png")
        vec, errors = self._media_vector(query_image_chunk(b64, mime))
        if vec is None:
            self.multimodal_ok = False
            raise SystemExit(
                "the endpoint rejected image input (%s), so it cannot search by "
                "image content - it stores attachments by name and type only. "
                "Describe the picture in words instead, or serve a "
                "pixel-capable multimodal endpoint (embed_server.py, or a "
                "multimodal GGUF under llama.cpp); test_embed_media.py prints "
                "what this endpoint does." % "; ".join(errors))
        self.multimodal_ok = True
        return vec

    def embed_media(self, chunk: dict) -> np.ndarray:
        """Embed one media chunk; falls back to its text label when the
        endpoint rejects multimodal input (text-only servers) or when
        embed_media_multimodal is disabled."""
        if not self.media_multimodal or self.multimodal_ok is False:
            return self._embed_media_text_fallback(chunk)
        if self.native:
            # Ollama's native /api/embed takes a sibling `images` array but
            # ignores it for embedding models: the reply is byte-identical
            # with and without the image (verified on 0.40.6 + embeddinggemma2,
            # see test_embed_media.py). Sending it anyway would store filename
            # vectors under a 'media' tier and look like real image search, so
            # index the label and say why, once.
            if not self._native_media_warned:
                self._native_media_warned = True
                log.warning(
                    "native /api/embed endpoint: Ollama does not embed image "
                    "input for embedding models, so attachments are indexed by "
                    "name/type only. Serve a multimodal model over the OpenAI "
                    "shape (embed_server.py) and point embed_endpoint at it to "
                    "search attachment content; run test_embed_media.py to check."
                )
            return self._embed_media_text_fallback(chunk)
        vec, errors = self._media_vector(chunk)
        if vec is None:
            if self.multimodal_ok is None:
                log.warning(
                    "endpoint rejected image input (%s); falling back to text-only "
                    "embedding for image/audio attachments for the rest of this run "
                    "- they stay searchable by name/type, not by content. Serve a "
                    "multimodal model over the OpenAI shape (embed_server.py, or a "
                    "multimodal GGUF under llama.cpp) to embed media content; "
                    "test_embed_media.py prints what each shape does.",
                    "; ".join(errors),
                )
            self.multimodal_ok = False
            return self._embed_media_text_fallback(chunk)
        self.multimodal_ok = True
        return vec

    def _embed_media_text_fallback(self, chunk: dict) -> np.ndarray:
        """Text-only stand-in for a media chunk: its title and what it is."""
        if self.scheme == "gemma":
            title, what = media_label(chunk)
            q = self.doc_input(what, title)
        else:
            t = chunk.get("text") or f"attachment: {chunk.get('mime', '')}"
            q = f"{self.doc_prompt}: {t}" if self.doc_prompt else t
        data = self._post(self._body([q]))
        return self._vectors(data)[0]


# ---------------------------------------------------------------------------
# Storage (SQLite + numpy)
# ---------------------------------------------------------------------------

@dataclass
class IndexArrays:
    """The whole index in memory, one row per chunk (parallel lists)."""
    ids: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    senders: list = field(default_factory=list)
    subjects: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    tiers: list = field(default_factory=list)
    cids: list = field(default_factory=list)
    urls: list = field(default_factory=list)
    threads: list = field(default_factory=list)
    kinds: list = field(default_factory=list)
    hashes: list = field(default_factory=list)
    vecs: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.ids)


class Store:
    """SQLite state: messages, tiered chunks, vectors and the BM25 index.

    `chunks_fts` is a plain (not external-content) FTS5 table keyed by
    chunk id: it duplicates the chunk text so rows can be deleted normally
    when a message is re-indexed, which external-content tables cannot do.

    `cache_index` keeps the in-memory `IndexArrays` of `load_index` instead of
    rebuilding it per call. It is opt-in because a cached snapshot is only
    correct while nothing writes: every mutating method here calls
    `invalidate_index()`, and a process that writes through a *different*
    connection to the same file (another `index` run) cannot be seen, so only
    long-lived readers that own their database should turn it on.
    """

    def __init__(self, path: str, cfg: Optional[dict] = None,
                 cache_index: bool = False):
        self.cfg = cfg or {}
        self.cache_index = bool(cache_index)
        self._idx_cache: dict = {}
        self.sizes = chunk_sizes_of(self.cfg)
        self.gmail_index = gmail_index(self.cfg)
        self.tokenizer = safe_tokenizer(
            self.cfg.get("fts_tokenizer", DEFAULTS["fts_tokenizer"]))
        self.meta_weight = float(
            self.cfg.get("fts_meta_weight", DEFAULTS["fts_meta_weight"]))
        self.conn = sqlite3.connect(path)
        #: the thumbnail sidecar, opened on first use (`media_cache`)
        self._media: Optional[MediaCache] = None
        self.conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS messages(
                message_id TEXT PRIMARY KEY, folder TEXT, date TEXT,
                sender TEXT, to_addrs TEXT, cc TEXT, subject TEXT,
                context_text TEXT, thread_id TEXT, attachment_exts TEXT,
                gmail_url TEXT, uid INTEGER);
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT, kind TEXT, label TEXT, text TEXT);
            CREATE TABLE IF NOT EXISTS embeddings(
                chunk_id INTEGER PRIMARY KEY, vec BLOB, dim INTEGER);
            CREATE TABLE IF NOT EXISTS index_state(key TEXT PRIMARY KEY, value TEXT);
            -- What the index used to hold and no longer does, one row per
            -- message the folders stopped listing. Nothing reads it for search;
            -- it exists so a removal can be reviewed, so a reopened message can
            -- be explained, and so mail that comes back is recognised as a
            -- message to re-index rather than as mail that was never there.
            CREATE TABLE IF NOT EXISTS removed_messages(
                message_id TEXT PRIMARY KEY, folder TEXT, uid INTEGER,
                date TEXT, sender TEXT, subject TEXT, removed_at TEXT,
                reason TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                meta, text, quoted, tokenize="{self.tokenizer}");
            """
        )
        # Idempotent migrations for DBs created before a column existed: add
        # it once. Guarded so re-opening a migrated DB is a no-op.
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(messages)")]
        if "gmail_url" not in cols:
            self.conn.execute("ALTER TABLE messages ADD COLUMN gmail_url TEXT")
        if "uid" not in cols:
            # The IMAP UID each row was fetched under, which is what lets a
            # later run ask "is this message still in the folder?" with a
            # listing instead of a header re-read. 0 means "never recorded", and
            # the first reconcile of such a database fills it in once.
            self.conn.execute("ALTER TABLE messages ADD COLUMN uid INTEGER"
                              " DEFAULT 0")
        ccols = [r[1] for r in self.conn.execute("PRAGMA table_info(chunks)")]
        for name, decl in CHUNK_COLUMNS:
            if name not in ccols:
                self.conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {decl}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_hash ON"
                          " chunks(message_id, text_hash)")
        self.conn.commit()
        self.ensure_fts_schema()
        self.backfill_chunk_meta()
        self.backfill_text_hash()
        self.sync_fts()

    # -- migrations -------------------------------------------------------
    def ensure_fts_schema(self) -> bool:
        """Recreate `chunks_fts` when its columns or tokenizer no longer match.

        Both are free operations (BM25 is rebuilt from stored text, no
        embeddings), so it happens on open instead of waiting for
        `reindex --rebuild-fts`. Two real cases need it: adding the `quoted`
        column, and the tokenizer switch away from porter - which is an English
        stemmer silently splitting Dutch word forms apart in every existing
        index. The refill itself is `sync_fts`, which indexes every chunk that
        has no FTS row; after a drop that is all of them.
        """
        want = f"{self.tokenizer}|{','.join(FTS_COLUMNS)}"
        if self.get_state("fts_schema") == want:
            return False
        log.info("rebuilding the BM25 index: schema or tokenizer changed to %s",
                 want)
        self.conn.execute("DROP TABLE IF EXISTS chunks_fts")
        self.conn.execute(
            f'CREATE VIRTUAL TABLE chunks_fts USING fts5'
            f'({", ".join(FTS_COLUMNS)}, tokenize="{self.tokenizer}")')
        self.set_state("fts_schema", want)
        self.set_state("fts_tokenizer", self.tokenizer)
        return True

    def backfill_text_hash(self) -> int:
        """Fingerprint existing chunk text once, so duplicates become visible.

        Only NULL rows are touched: the hash of a chunk never changes while its
        text does not.
        """
        rows = self.conn.execute(
            "SELECT id, COALESCE(text,'') FROM chunks WHERE text_hash IS NULL"
        ).fetchall()
        if rows:
            self.conn.executemany(
                "UPDATE chunks SET text_hash=? WHERE id=?",
                [(text_hash(t), cid) for cid, t in rows])
            self.conn.commit()
            self.invalidate_index()
        return len(rows)

    def clean_stored_text(self, dry_run: bool = False) -> dict:
        """Re-run the text hygiene over stored chunk text: no IMAP involved.

        What `reindex --clean` is for. The corpus was written by the old
        extractor, so every stored body still carries the quote it replied to
        and the legal footer underneath it; hygiene decides what belongs in a
        vector at index time, so re-deriving it offline is the only way to fix
        an already indexed mailbox. The removed text is not lost - it moves to
        `quoted`, which BM25 still searches, and the chunk's FTS row is rewritten
        in the same statement so the lexical leg cannot go stale.

        Returns {changed, chars, ids}: ids are the chunks whose vector is now
        stale and have to be re-embedded.

        Only canonical rows. A derived window is a cut of a canonical text, and
        the tier rebuild in the same `reindex` run re-derives it from the repaired
        source; cleaning windows here made the report lie, because a window's cut
        boundary can look like a quoted block however clean its source is. A
        mailbox with nothing left to repair then still reported 59 chunks to fix,
        on every run, forever.
        """
        rows = self.conn.execute(
            "SELECT id, message_id, COALESCE(search_meta,''), COALESCE(text,''),"
            " COALESCE(quoted,'') FROM chunks"
            " WHERE tier != ? AND canonical = 1", (MEDIA_TIER,)
        ).fetchall()
        changed: list[int] = []
        chars = 0
        cur = self.conn.cursor()
        for cid, _mid, smeta, text, quoted in rows:
            clean, dropped = clean_text(text)
            if clean == text and not dropped:
                continue
            chars += max(0, len(text) - len(clean))
            changed.append(cid)
            if dry_run:
                continue
            new_quoted = "\n".join(x for x in (quoted, dropped) if x).strip()
            cur.execute("UPDATE chunks SET text=?, text_hash=?, quoted=? WHERE id=?",
                        (clean, text_hash(clean), new_quoted or None, cid))
            cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
            cur.execute("INSERT INTO chunks_fts(rowid, meta, text, quoted)"
                        " VALUES (?,?,?,?)", (cid, smeta, clean, new_quoted))
        if not dry_run:
            self.conn.commit()
            self.invalidate_index()
        return {"changed": len(changed), "chars": chars, "ids": changed,
                "scanned": len(rows)}

    def decode_stored_subjects(self, dry_run: bool = False) -> dict:
        """Repair subjects that were stored as their RFC 2047 encoded-words.

        565 of 4 452 messages in a real inbox were indexed before the fetcher
        decoded headers, so both retrieval legs have been searching base64 ever
        since: BM25 cannot match `Contact` inside `=?UTF-8?B?Q29udGFjdA==?=`,
        and the title that the vector of that message is built from is that same
        string. The repair runs the decoder the fetcher runs, so a fixed row is
        indistinguishable from a freshly fetched one.

        The subject reaches three places: the row, the metadata block kept in
        `search_meta` (and as the text of a meta chunk), and the title of the
        embedding prompt. The last one needs no rewrite - the prompt is rebuilt
        from `messages.subject` at embed time - so the chunks this touches are
        returned for the caller to re-embed.

        Returns {changed, ids, scanned}.
        """
        rows = self.conn.execute(
            "SELECT message_id, subject FROM messages WHERE subject LIKE '%=?%'"
        ).fetchall()
        fixed = [(mid, subj, decode_mime_words(subj)) for mid, subj in rows]
        fixed = [t for t in fixed if t[2] != t[1]]
        ids: list[int] = []
        cur, reads = self.conn.cursor(), self.conn.cursor()
        for mid, old, new in fixed:
            if dry_run:
                continue
            cur.execute("UPDATE messages SET subject=? WHERE message_id=?", (new, mid))
            for cid, smeta, text, quoted in reads.execute(
                    "SELECT id, COALESCE(search_meta,''), COALESCE(text,''),"
                    " COALESCE(quoted,'') FROM chunks"
                    " WHERE message_id = ? AND tier != ?", (mid, MEDIA_TIER)):
                new_meta = smeta.replace(old, new)
                new_text = text.replace(old, new)
                if new_meta == smeta and new_text == text:
                    continue
                ids.append(cid)
                cur.execute("UPDATE chunks SET text=?, text_hash=?, search_meta=?"
                            " WHERE id=?", (new_text, text_hash(new_text),
                                            new_meta or None, cid))
                cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
                cur.execute("INSERT INTO chunks_fts(rowid, meta, text, quoted)"
                            " VALUES (?,?,?,?)", (cid, new_meta, new_text, quoted))
        if not dry_run:
            self.conn.commit()
            self.invalidate_index()
        return {"changed": len(fixed), "ids": ids, "scanned": len(rows)}

    def prune_orphan_vectors(self) -> int:
        """Delete embeddings whose chunk row is gone, and say how many.

        A tier rebuild replaces every derived window of the tiers it builds, and
        the vectors of the chunks it removed stay in `embeddings`: no search ever
        reads them (the index is loaded through `chunks`), but a real mailbox
        picked up 22 926 of them, which is 70 MB of file and of copy time for
        rows that answer nothing.
        """
        n = self.conn.execute(
            "DELETE FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)"
        ).rowcount
        self.conn.commit()
        self.invalidate_index()
        return max(0, int(n))

    # -- reconciliation: reflecting mail that is gone ------------------------
    def folder_counts(self) -> dict:
        """Stored rows per IMAP folder — what reconcile has to say about."""
        return {r[0] or "": int(r[1]) for r in self.conn.execute(
            "SELECT folder, COUNT(*) FROM messages GROUP BY folder")}

    def folder_rows(self, folder: str) -> list:
        """`(message_id, uid)` for every stored row of one folder.

        A `uid` of 0 means *unverified*: a row written before UIDs were stored,
        or by a fetch whose reply carried none. Such a row can only be matched
        by Message-ID, which is what makes the first reconcile of an old index
        sweep its headers once.
        """
        return [(r[0], int(r[1] or 0)) for r in self.conn.execute(
            "SELECT message_id, uid FROM messages WHERE folder = ?", (folder,))]

    def set_uid(self, message_id: str, uid: int) -> None:
        self.conn.execute("UPDATE messages SET uid=? WHERE message_id=?",
                          (int(uid or 0), message_id))

    # -- thumbnails -------------------------------------------------------
    def media_cache(self, only_if_exists: bool = False) -> Optional[MediaCache]:
        """The preview sidecar, opened on first use, or None when previews are off.

        `only_if_exists` is for the delete path: a prune over a mailbox that has
        no sidecar yet has nothing to clean, and it must not answer by creating
        an empty one next to the index.
        """
        if not self.cfg.get("store_thumbs", DEFAULTS["store_thumbs"]):
            return None
        path = media_db_path(self.cfg)
        if self._media is not None and self._media.path == path:
            return self._media
        if only_if_exists and not Path(path).exists():
            return None
        try:
            self._media = MediaCache(path, self.cfg)
        except Exception as e:
            log.warning("thumbnails are off: %s could not be opened (%s)",
                        path, e)
            self._media = None
        return self._media

    def drop_thumbnails(self, message_ids) -> int:
        """Previews of messages this index no longer holds, gone with them.

        The delete mirror promises no orphans — message row, chunks, vectors,
        FTS entries together. A preview left behind would be the worst kind: a
        dropped message still showing its picture in the list.
        """
        ids = [m for m in (message_ids or ()) if m]
        media = self.media_cache(only_if_exists=True)
        if not ids or media is None:
            return 0
        try:
            return media.delete_messages(ids)
        except Exception as e:
            log.warning("could not drop previews of %d removed message(s): %s",
                        len(ids), e)
            return 0

    def remove_messages(self, message_ids, reason: str = "gone",
                        dry_run: bool = False) -> list:
        """Drop messages the folder no longer holds, and remember what was dropped.

        This is the only deletion in this program, and it never deletes mail:
        the message is already gone from the IMAP folder — the user deleted it
        in Gmail — and what goes here is the index's own description of it, so
        that search stops answering with something that no longer exists.

        Every id is written to `removed_messages` first. That is what makes the
        act reviewable (*what did last night's refresh drop?*), what lets the
        GUI say *that one was removed on 21 Oct* instead of *no such message*,
        and what recognises a message that comes back to the folder so it can
        be re-indexed instead of staying invisible under the watermark.

        Returns the rows removed (or, with `dry_run`, the rows that would go).
        """
        ids = [m for m in (message_ids or ()) if m]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            "SELECT message_id, folder, uid, date, sender, subject FROM messages"
            f" WHERE message_id IN ({marks})", ids).fetchall()
        if dry_run:
            return [dict(zip(("message_id", "folder", "uid", "date", "sender",
                              "subject"), r)) for r in rows]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for mid, folder, uid, date, sender, subject in rows:
            self.conn.execute(
                "INSERT OR REPLACE INTO removed_messages (message_id, folder,"
                " uid, date, sender, subject, removed_at, reason)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (mid, folder, int(uid or 0), date, sender, subject, now, reason))
            self._drop_message_chunks(mid)
            self.conn.execute("DELETE FROM messages WHERE message_id = ?", (mid,))
        self.conn.commit()
        self.drop_thumbnails([r[0] for r in rows])
        self.invalidate_index()
        return [dict(zip(("message_id", "folder", "uid", "date", "sender",
                          "subject"), r)) for r in rows]

    def tombstone(self, message_id: str) -> Optional[dict]:
        """What was removed under this id, or None if nothing ever was."""
        row = self.conn.execute(
            "SELECT message_id, folder, uid, date, sender, subject, removed_at,"
            " reason FROM removed_messages WHERE message_id = ?",
            (message_id,)).fetchone()
        if not row:
            return None
        return dict(zip(("message_id", "folder", "uid", "date", "sender",
                         "subject", "removed_at", "reason"), row))

    def tombstones(self) -> dict:
        """`{message_id: row}` for everything ever removed."""
        out = {}
        for row in self.conn.execute(
                "SELECT message_id, folder, uid, date, sender, subject, removed_at,"
                " reason FROM removed_messages"):
            out[row[0]] = dict(zip(("message_id", "folder", "uid", "date", "sender",
                                    "subject", "removed_at", "reason"), row))
        return out

    def clear_tombstone(self, message_id: str) -> None:
        self.conn.execute("DELETE FROM removed_messages WHERE message_id = ?",
                          (message_id,))

    def removed_count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM removed_messages").fetchone()[0])

    def reconcile_state(self) -> dict:
        """The last reconcile, as `reconcile_deletions` recorded it.

        `last_removal_at` is read from the tombstones, not from a state key: it
        means *when did mail last actually leave*, and a reconcile that removed
        nothing must not move it. The status line says "last dropped on", so a
        quiet run borrowing the timestamp of the loud one would be a lie about a
        deletion — the one thing this feature may never print.
        """
        raw = self.get_state("last_reconcile")
        try:
            data = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            data = {}
        newest = self.conn.execute(
            "SELECT MAX(removed_at) FROM removed_messages").fetchone()[0] or ""
        return {"last_reconcile": data,
                "last_reconcile_at": self.get_state("last_reconcile_at") or "",
                "last_removal_at": newest,
                "removed_total": self.removed_count()}


    def backfill_chunk_meta(self) -> int:
        """Give pre-tier chunks the columns the tier scheme relies on.

        A legacy chunk is a single whole-source window whose label already is
        the source key, i.e. exactly what the tier scheme calls a *canonical*
        chunk. Media chunks are recognisable because they store their label
        as their text. Idempotent: it only ever touches NULL rows.
        """
        todo = self.conn.execute(
            """SELECT c.id, c.kind, c.label, COALESCE(c.text,''), c.tier, c.src,
                      c.search_meta, m.sender, m.to_addrs, m.cc, m.subject,
                      m.attachment_exts, m.date
               FROM chunks c LEFT JOIN messages m ON m.message_id = c.message_id
               WHERE c.tier IS NULL OR c.src IS NULL OR c.search_meta IS NULL"""
        ).fetchall()
        if not todo:
            return 0
        n = 0
        for cid, kind, label, text, tier, src, smeta, sender, to, cc, subj, exts, date in todo:
            kind = kind or "body"
            label = label or "body"
            if src is None or src == "":
                src = label
            if tier is None:
                is_media = kind == "attachment" and text.strip() == label.strip()
                tier = MEDIA_TIER if is_media else canonical_tier(len(text.strip()), self.sizes)[0]
            if smeta is None:
                msg = Message(
                    message_id="", folder="", date=parse_date(date),
                    sender=sender or "", to=to or "", cc=cc or "",
                    subject=subj or "", body_text="",
                )
                smeta = chunk_search_meta(msg, exts or "", kind, src)
            self.conn.execute(
                "UPDATE chunks SET tier=?, src=?, pos=COALESCE(pos,0),"
                " nwin=COALESCE(NULLIF(nwin,0),1), search_meta=? WHERE id=?",
                (tier, src, smeta, cid),
            )
            n += 1
        self.conn.commit()
        self.invalidate_index()
        return n

    def sync_fts(self) -> int:
        """Index every chunk that has no FTS row yet (first run after the FTS
        upgrade, or rows written by an older build)."""
        rows = self.conn.execute(
            """SELECT c.id, COALESCE(c.search_meta,''), COALESCE(c.text,''),
                      COALESCE(c.quoted,'')
               FROM chunks c LEFT JOIN chunks_fts f ON f.rowid = c.id
               WHERE f.rowid IS NULL"""
        ).fetchall()
        if rows:
            self.conn.executemany(
                "INSERT INTO chunks_fts(rowid, meta, text, quoted) VALUES (?,?,?,?)",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def rebuild_fts(self) -> int:
        """Re-index all chunk text into FTS from what is stored: costs no
        embeddings, and is what changing fts_tokenizer requires."""
        self.conn.execute("DELETE FROM chunks_fts")
        rows = self.conn.execute(
            "SELECT id, COALESCE(search_meta,''), COALESCE(text,''),"
            " COALESCE(quoted,'') FROM chunks").fetchall()
        self.conn.executemany(
            "INSERT INTO chunks_fts(rowid, meta, text, quoted) VALUES (?,?,?,?)",
            rows)
        self.conn.commit()
        self.set_state("fts_tokenizer", self.tokenizer)
        return len(rows)

    def retier_canonical(self) -> int:
        """Re-stamp canonical chunks with the tier their length implies.

        Free (metadata only, no re-embedding). The migration stamped every
        legacy chunk 'xlarge', which would make --tiers filtering wrong for
        short bodies; after this, tier labels match the configured ladder.
        """
        rows = self.conn.execute(
            "SELECT id, text, tier FROM chunks WHERE canonical = 1 AND tier != ?",
            (MEDIA_TIER,),
        ).fetchall()
        n = 0
        for cid, text, tier in rows:
            want = canonical_tier(len((text or "").strip()), self.sizes)[0]
            if want != tier:
                self.conn.execute("UPDATE chunks SET tier=? WHERE id=?", (want, cid))
                n += 1
        self.conn.commit()
        self.invalidate_index()
        return n

    # -- index metadata (so a stale index says so instead of confusing) ----
    def save_index_meta(self) -> None:
        self.set_state("chunk_sizes", json.dumps(self.sizes, sort_keys=True))
        self.set_state("chunk_overlap_frac", str(self.cfg.get("chunk_overlap_frac",
                                                              DEFAULTS["chunk_overlap_frac"])))
        self.set_state("max_source_chars", str(self.cfg.get("max_source_chars",
                                                            DEFAULTS["max_source_chars"])))
        self.set_state("fts_tokenizer", self.tokenizer)
        # What produced the *vectors*, so a later change of model or server says
        # so instead of quietly mixing two cosine spaces (same dim, different
        # build: nothing else in the index notices).
        self.set_state("embed_model", str(self.cfg.get("embed_model", "") or ""))
        self.set_state("dim", str(self.cfg.get("dim", DEFAULTS["dim"])))
        self.set_state("embed_host", _host_of(str(self.cfg.get("embed_endpoint", ""))))

    def warn_if_stale(self) -> None:
        """Warn (never fail) when the index was built with a different tier
        ladder, tokenizer or embedding model than the config now asks for."""
        got = self.get_state("chunk_sizes")
        want = json.dumps(self.sizes, sort_keys=True)
        if got and got != want:
            log.warning("index built with chunk_sizes %s, config says %s - "
                        "run: python mailbox.py reindex", got, want)
        got = self.get_state("fts_tokenizer")
        if got and got != self.tokenizer:
            log.warning("index built with fts_tokenizer %r, config says %r - "
                        "run: python mailbox.py reindex --rebuild-fts",
                        got, self.tokenizer)
        for key, label in (("embed_model", "embed_model"), ("dim", "dim")):
            want = str(self.cfg.get(key, DEFAULTS.get(key, "")) or "")
            if key == "dim":
                want = str(self.cfg.get("dim", DEFAULTS["dim"]))
            got = self.get_state(key)
            if got and want and got != want:
                log.warning("vectors were embedded with %s=%r, config says %r - old "
                            "and new vectors are different spaces, so scores mix; "
                            "run: python mailbox.py reindex --re-embed",
                            label, got, want)
        got, want = self.get_state("embed_host"), _host_of(
            str(self.cfg.get("embed_endpoint", "")))
        if got and want and got != want:
            log.warning("vectors were embedded by the server at %s, config now "
                        "points at %s - the same model name on another server is "
                        "not necessarily the same build or quantisation; if it "
                        "differs, run: python mailbox.py reindex --re-embed",
                        got, want)

    # -- writes -----------------------------------------------------------
    def has(self, mid: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM messages WHERE message_id = ?", (mid,)
        ).fetchone() is not None

    def _drop_message_chunks(self, mid: str) -> None:
        """Delete a message's chunks with their vectors and FTS rows.

        Re-indexing a message now replaces several rows per source, so the
        old append-only behaviour would duplicate them.
        """
        self.conn.execute(
            "DELETE FROM embeddings WHERE chunk_id IN"
            " (SELECT id FROM chunks WHERE message_id=?)", (mid,))
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN"
            " (SELECT id FROM chunks WHERE message_id=?)", (mid,))
        self.conn.execute("DELETE FROM chunks WHERE message_id=?", (mid,))
        self.invalidate_index()

    def set_embedding(self, chunk_id: int, vec) -> None:
        """Replace one chunk's vector, leaving its text and FTS row alone.

        Used by `reindex --re-embed`: after a change of embedding model the
        stored cosines are no longer comparable with new ones, but the text
        they came from still is - so the vectors are what gets rebuilt.
        """
        arr = np.asarray(vec, dtype=np.float32)
        self.conn.execute("DELETE FROM embeddings WHERE chunk_id=?", (chunk_id,))
        self.conn.execute("INSERT INTO embeddings (chunk_id, vec, dim) VALUES (?,?,?)",
                          (chunk_id, arr.tobytes(), len(arr)))
        self.invalidate_index()

    def add_chunk(self, cur, message_id: str, chunk: dict, vec,
                  search_meta: str) -> int:
        """Insert one chunk row + its vector + its FTS row. Returns the id."""
        kind = chunk.get("kind", "body")
        text = chunk.get("text")
        quoted = chunk.get("quoted") or ""
        src = chunk.get("src") or chunk["label"]
        tier = chunk.get("tier") or canonical_tier(len((text or "").strip()), self.sizes)[0]
        canonical = 1 if chunk.get("canonical", True) else 0
        cur.execute(
            "INSERT INTO chunks (message_id, kind, label, text, tier, src, pos,"
            " nwin, canonical, search_meta, text_hash, quoted)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (message_id, kind, chunk["label"], text, tier, src,
             int(chunk.get("pos", 0)), int(chunk.get("nwin", 1)), canonical,
             search_meta, text_hash(text or ""), quoted),
        )
        cid = cur.lastrowid
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32)
            cur.execute(
                "INSERT INTO embeddings (chunk_id, vec, dim) VALUES (?,?,?)",
                (cid, arr.tobytes(), len(arr)),
            )
        cur.execute(
            "INSERT INTO chunks_fts(rowid, meta, text, quoted) VALUES (?,?,?,?)",
            (cid, search_meta, text or "", quoted),
        )
        return cid

    def commit(self, msg: Message, chunks: list[dict], vecs: list[np.ndarray]) -> None:
        exts = ",".join(Path(a.name).suffix.lower().lstrip(".") for a in msg.attachments)
        cur = self.conn.cursor()
        self._drop_message_chunks(msg.message_id)
        # The preview keeps the *new* text of the message, not the quote it
        # dragged along: thread context is built from these 300 characters, so
        # an unclean preview repeats one old message down a whole thread.
        preview = msg.body_text.strip()
        if self.cfg.get("clean_text", True):
            preview = clean_text(preview)[0] or preview
        cur.execute(
            "INSERT OR REPLACE INTO messages (message_id, folder, date, sender,"
            " to_addrs, cc, subject, context_text, thread_id, attachment_exts,"
            " gmail_url, uid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                msg.message_id, msg.folder, msg.date.isoformat(),
                msg.sender, msg.to, msg.cc, msg.subject,
                preview[:300], msg.thread_id, exts,
                msg.gmail_url, self._keep_uid(msg),
            ),
        )
        for c, v in zip(chunks, vecs):
            smeta = chunk_search_meta(msg, exts, c.get("kind", "body"),
                                      c.get("src") or c["label"])
            self.add_chunk(cur, msg.message_id, c, v, smeta)
        self.conn.commit()
        self.invalidate_index()

    def _keep_uid(self, msg: Message) -> int:
        """The UID to store for `msg`: the one it was fetched under, or the one
        it is already known by.

        `INSERT OR REPLACE` rebuilds the row, so a re-index that came in without
        a UID (`--full` on a server that did not report one, a hand-built
        `Message`) would otherwise drop a perfectly good UID and make the row
        look unverified to the next reconcile.
        """
        uid = int(getattr(msg, "uid", 0) or 0)
        if uid:
            return uid
        row = self.conn.execute(
            "SELECT uid FROM messages WHERE message_id = ?",
            (msg.message_id,)).fetchone()
        return int(row[0] or 0) if row and row[0] else 0

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO index_state VALUES (?,?)", (key, value)
        )
        self.conn.commit()

    def get_state(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM index_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def delete_state(self, key: str) -> None:
        self.conn.execute("DELETE FROM index_state WHERE key = ?", (key,))
        self.conn.commit()

    def invalidate_index(self) -> None:
        """Forget the cached snapshot, so the next `load_index` reads the file.

        Every method here that changes what `load_index` returns calls it. Being
        over-eager costs one re-read; being wrong costs a search that silently
        answers from a corpus that no longer exists.
        """
        self._idx_cache.clear()

    def load_index(self, with_vectors: bool = True) -> IndexArrays:
        """The index, from cache when `cache_index` is on.

        The vector snapshot is a superset of the text-only one, so a warm hybrid
        cache also answers a lexical query rather than holding the same chunk
        text a second time (~60 MB of Python strings). Two snapshots coexist only
        when a lexical query arrives before the first hybrid one.
        """
        if not self.cache_index:
            return self._read_index(with_vectors)
        key = bool(with_vectors)
        if not key:
            warm = self._idx_cache.get(True)
            if warm is not None:
                return warm
        idx = self._idx_cache.get(key)
        if idx is None:
            idx = self._read_index(with_vectors)
            self._idx_cache[key] = idx
        return idx

    def _url(self, url: str) -> str:
        """A stored Gmail link, in the account slot this config says to use."""
        return gmail_account_url(url or "", self.gmail_index)

    def _read_index(self, with_vectors: bool = True) -> IndexArrays:
        """Load every chunk as parallel lists (one entry per chunk).

        Thread id, kind and text hash come along because message-level ranking
        needs them: one result per conversation, and near-duplicate windows of
        one body are only recognisable as such when you know which message (and
        which thread) a window came from.
        """
        cols = ("m.message_id, m.date, m.sender, m.subject, c.label, c.text,"
                " %s, m.gmail_url, c.id, COALESCE(c.tier,''),"
                " COALESCE(m.thread_id,''), c.kind, COALESCE(c.text_hash,'')")
        if with_vectors:
            rows = self.conn.execute(
                "SELECT " + cols % "e.vec" + """
                FROM messages m
                JOIN chunks c ON c.message_id = m.message_id
                JOIN embeddings e ON e.chunk_id = c.id
                """
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT " + cols % "NULL" + """
                FROM messages m
                JOIN chunks c ON c.message_id = m.message_id
                """
            ).fetchall()
        out = IndexArrays()
        if not rows:
            return out
        out.ids = [r[0] for r in rows]
        out.dates = [parse_date(r[1]) for r in rows]
        out.senders = [r[2] for r in rows]
        out.subjects = [r[3] for r in rows]
        out.labels = [r[4] for r in rows]
        out.texts = [r[5] or "" for r in rows]
        out.urls = [self._url(r[7] or "") for r in rows]
        out.cids = [r[8] for r in rows]
        out.tiers = [r[9] for r in rows]
        out.threads = [r[10] or r[0] for r in rows]
        out.kinds = [r[11] or "body" for r in rows]
        out.hashes = [r[12] for r in rows]
        if with_vectors:
            out.vecs = np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
        return out

    def load_context_map(self) -> dict:
        rows = self.conn.execute(
            "SELECT message_id, date, sender, context_text FROM messages"
        ).fetchall()
        return {
            r[0]: (parse_date(r[1]), r[2] or "", r[3] or "") for r in rows
        }

    def thread_counts(self) -> dict:
        """{thread_id: message count} - how big the conversation behind a hit is."""
        return {r[0]: int(r[1]) for r in self.conn.execute(
            "SELECT COALESCE(NULLIF(thread_id,''), message_id), COUNT(*)"
            " FROM messages GROUP BY 1")}

    def canonical_body_hashes(self, message_ids) -> dict:
        """{message_id: fingerprint of the whole body} for the collapse step.

        The chunk that made a message a hit is often a window, and a window's
        fingerprint only covers the part of the letter it holds. Two messages
        carrying the same letter, reached through different windows of different
        lengths, then look like two answers. The canonical chunk of a message is
        the whole body, so its fingerprint is the key that recognises them as
        one - which is what the reader sees when a query returns the same
        newsletter text three times.
        """
        ids = [m for m in message_ids if m]
        out: dict[str, str] = {}
        for i in range(0, len(ids), 900):            # keep the IN list bounded
            part = ids[i:i + 900]
            for mid, h in self.conn.execute(
                    "SELECT message_id, text_hash FROM chunks"
                    " WHERE canonical = 1 AND kind = 'body' AND message_id IN (%s)"
                    % ",".join("?" * len(part)), part):
                if h:
                    out[mid] = h
        return out

    def embedded_thread_hashes(self, threads) -> set:
        """{(thread key, text hash)} for chunks of these threads that already
        have a vector - what `index_dedupe_threads` skips new duplicates of."""
        keys = [t for t in threads if t]
        if not keys:
            return set()
        out: set[tuple[str, str]] = set()
        for i in range(0, len(keys), 500):          # keep the IN list bounded
            part = keys[i:i + 500]
            rows = self.conn.execute(
                "SELECT COALESCE(NULLIF(m.thread_id,''), m.message_id), c.text_hash"
                " FROM chunks c JOIN messages m ON m.message_id = c.message_id"
                " JOIN embeddings e ON e.chunk_id = c.id"
                " WHERE COALESCE(NULLIF(m.thread_id,''), m.message_id) IN (%s)"
                % ",".join("?" * len(part)), part)
            out.update((r[0], r[1]) for r in rows if r[1])
        return out

    def _filter_clauses(self, f) -> tuple[list[str], list]:
        """SQL WHERE fragments (+ args) for one `queryparse.Filters`.

        One builder for two readers: the search pre-filter
        (`filter_message_ids`) and the GUI's inbox browsing (`page_messages`).
        Duplicating these predicates is how a hand-clicked sender chip and the
        words `van flexa` end up meaning two different things in one product.
        """
        clauses: list[str] = []
        args: list = []

        def add_many(attr: str, cols: tuple) -> None:
            vals = [str(v).strip().lower() for v in (getattr(f, attr, None) or [])
                    if str(v).strip()]
            if not vals:
                return
            ors = []
            for v in vals:
                ors.append(" OR ".join(f"lower(COALESCE({c},'')) LIKE ?" for c in cols))
            clauses.append("(" + " OR ".join(ors) + ")")
            # extend(), not `args += [...]`: inside a nested function that
            # reads the enclosing list, `+=` is an assignment and makes `args`
            # a local of add_many, so the first filter with a value would
            # crash the whole search instead of filtering anything.
            args.extend(f"%{v}%" for v in vals for _ in cols)

        add_many("senders", ("sender",))
        add_many("recipients", ("to_addrs", "cc"))
        add_many("subjects", ("subject",))
        since = getattr(f, "since", None)
        until = getattr(f, "until", None)
        if since is not None:
            clauses.append("date >= ?")
            args.append(parse_date(since).isoformat())
        if until is not None:
            clauses.append("date < ?")
            args.append(parse_date(until).isoformat())
        has_att = getattr(f, "has_attachment", None)
        if has_att is not None:
            # 'none' is what the indexer writes for a message without files.
            op = "NOT IN" if has_att else "IN"
            clauses.append(f"COALESCE(attachment_exts,'') {op} ('', 'none')")
        if getattr(f, "unread", None) is not None:
            # Not storable today: the IMAP fetch never asked for \Seen, so there
            # is no flag in the schema. Saying so beats silently ignoring it.
            log.warning("is:unread / ongelezen is not supported: the index does "
                        "not store IMAP flags, so this part of the query is "
                        "ignored")
        return clauses, args

    def filter_message_ids(self, f) -> set:
        """Message ids matching the structured email filters (see queryparse).

        Filters are applied *before* similarity search, as SQL, and never as
        query text: 'from:flexa factuur' embedded as one sentence makes the
        vector drift toward the word 'from' instead of restricting the search,
        which is the mistake every NL2filter-free setup makes.
        """
        clauses, args = self._filter_clauses(f)
        if not clauses:
            return {r[0] for r in self.conn.execute("SELECT message_id FROM messages")}
        rows = self.conn.execute(
            "SELECT message_id FROM messages WHERE " + " AND ".join(clauses), args)
        return {r[0] for r in rows}

    # -- browsing (what a GUI reads; no search engine involved) -------------
    def page_messages(self, before: Optional[str] = None, limit: int = 50,
                      filters=None, text: Optional[str] = None,
                      folder: Optional[str] = None) -> list[dict]:
        """One page of the mailbox, newest first, for a list pane.

        Keyset pagination (`date < ?`) rather than OFFSET: a mailbox grows at the
        top while you page through it, and an offset walks the same message twice
        or skips one. `before` is exclusive, so the caller feeds back the last
        date it was given.

        `text` is a LIKE over subject, sender and the stored preview - not the
        FTS index. Browsing must stay a different thing from searching: a list
        pane that quietly runs BM25 surprises you with 2019 mail when you typed
        a sender's name.
        """
        clauses, args = self._filter_clauses(filters) if filters is not None \
            else ([], [])
        if before:
            clauses.append("date < ?")
            args.append(str(before))
        if folder:
            clauses.append("folder = ?")
            args.append(folder)
        if text:
            like = f"%{str(text).strip().lower()}%"
            clauses.append("(lower(COALESCE(subject,'')) LIKE ?"
                           " OR lower(COALESCE(sender,'')) LIKE ?"
                           " OR lower(COALESCE(context_text,'')) LIKE ?)")
            args.extend([like, like, like])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = ("SELECT message_id, folder, date, sender, to_addrs, subject,"
               " COALESCE(context_text,''), COALESCE(attachment_exts,''),"
               " COALESCE(thread_id,''), COALESCE(gmail_url,''),"
               "  (SELECT COUNT(*) FROM messages t"
               "    WHERE t.thread_id = messages.thread_id"
               "      AND COALESCE(messages.thread_id,'') <> '')"
               " FROM messages" + where +
               " ORDER BY date DESC, message_id DESC LIMIT ?")
        keys = ("message_id", "folder", "date", "sender", "to_addrs", "subject",
                "preview", "attachment_exts", "thread_id", "gmail_url",
                "thread_len")
        rows = self.conn.execute(sql, list(args) + [max(1, int(limit))]).fetchall()
        out = [dict(zip(keys, r)) for r in rows]
        for d in out:
            d["gmail_url"] = self._url(d["gmail_url"])
        return out

    def count_messages(self, filters=None, text: Optional[str] = None,
                       folder: Optional[str] = None) -> int:
        """How many messages a `page_messages` filter matches, all pages."""
        clauses, args = self._filter_clauses(filters) if filters is not None \
            else ([], [])
        if folder:
            clauses.append("folder = ?")
            args.append(folder)
        if text:
            like = f"%{str(text).strip().lower()}%"
            clauses.append("(lower(COALESCE(subject,'')) LIKE ?"
                           " OR lower(COALESCE(sender,'')) LIKE ?"
                           " OR lower(COALESCE(context_text,'')) LIKE ?)")
            args.extend([like, like, like])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM messages" + where, args).fetchone()[0])

    def message_detail(self, message_id: str) -> Optional[dict]:
        """Everything the index knows about one message, including its whole body.

        A search hit is a *window* - often a 320-character slice - so a reading
        pane built from `Hit.text` shows the reader a paragraph out of a message
        they cannot see. The canonical body chunk holds the complete indexed text
        (capped at `max_source_chars`), which is what this returns; `preview`
        covers the few messages that have no canonical body row at all.
        """
        head = self.conn.execute(
            "SELECT message_id, folder, date, sender, to_addrs, cc, subject,"
            " COALESCE(context_text,''), COALESCE(thread_id,''),"
            " COALESCE(attachment_exts,''), COALESCE(gmail_url,'')"
            " FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        if head is None:
            return None
        keys = ("message_id", "folder", "date", "sender", "to_addrs", "cc",
                "subject", "preview", "thread_id", "attachment_exts", "gmail_url")
        out = dict(zip(keys, head))
        out["gmail_url"] = self._url(out["gmail_url"])
        body = self.conn.execute(
            "SELECT COALESCE(text,''), COALESCE(quoted,''), tier, length(text)"
            " FROM chunks WHERE message_id = ? AND canonical = 1 AND kind = 'body'"
            " ORDER BY length(COALESCE(text,'')) DESC LIMIT 1",
            (message_id,)).fetchone()
        if body:
            out["body"] = body[0] or ""
            out["quoted"] = body[1] or ""
            out["tier"] = body[2] or ""
            # `max_source_chars` truncates the indexed text: saying so beats a
            # reader mistaking the end of the indexed text for the end of the mail.
            cap = int(self.cfg.get("max_source_chars", DEFAULTS["max_source_chars"]))
            out["truncated"] = len(out["body"]) >= cap - 40
            out["body_missing"] = False
        else:
            out["body"] = out["preview"]
            out["quoted"] = ""
            out["tier"] = ""
            out["truncated"] = False
            out["body_missing"] = True
        # Names come from chunk labels, the only place an attachment's own
        # filename is stored. A PDF rendered page by page writes one chunk per
        # page (`name.pdf (page 3)`), so the suffix goes and the file appears
        # once: a reading pane listing four copies of one invoice reads as a bug.
        names: dict[str, None] = {}
        for r in self.conn.execute(
                "SELECT DISTINCT label FROM chunks"
                " WHERE message_id = ? AND kind = 'attachment'"
                "   AND COALESCE(label,'') LIKE ?",
                (message_id, ATTACHMENT_PREFIX + "%")):
            name = (r[0] or "")[len(ATTACHMENT_PREFIX):].strip()
            page = re.match(r"^(?P<name>.+?) \(page \d+\)$", name)
            names[(page.group("name") if page else name).strip()] = None
        out["attachments"] = [a for a in names if a]
        return out

    def message_ref(self, message_id: str) -> Optional[tuple]:
        """Where this message still lives at the source: `(folder, uid)`.

        The index never keeps file bytes, so anything that has to re-read a
        message (opening an attachment, say) starts here. `uid` of 0 means
        "indexed before UIDs were recorded" — the folder is still the truth,
        and `fetch_by_message_id` covers the gap. None means the row itself
        is gone, which is a different answer from "Gmail is unreachable".
        """
        row = self.conn.execute(
            "SELECT COALESCE(folder, ''), COALESCE(uid, 0) FROM messages"
            " WHERE message_id = ?", (message_id,)).fetchone()
        return (str(row[0]), int(row[1])) if row else None

    def thread_messages(self, thread_id: str) -> list[dict]:
        """The conversation a message belongs to, oldest first."""
        if not thread_id:
            return []
        rows = self.conn.execute(
            "SELECT message_id, date, sender, subject, COALESCE(context_text,'')"
            " FROM messages WHERE thread_id = ?"
            " ORDER BY date ASC, message_id ASC", (thread_id,)).fetchall()
        keys = ("message_id", "date", "sender", "subject", "preview")
        return [dict(zip(keys, r)) for r in rows]

    def sender_facets(self, limit: int = 30) -> list[dict]:
        """Who sends the most mail, for the list pane's sidebar."""
        rows = self.conn.execute(
            "SELECT COALESCE(sender,'') AS s, COUNT(*) AS n FROM messages"
            " GROUP BY lower(s) ORDER BY n DESC, s ASC LIMIT ?",
            (max(1, int(limit)),)).fetchall()
        return [{"sender": s, "count": int(n)} for s, n in rows]

    def facet_totals(self) -> dict:
        """Headline counts for the rail: total, with files, date range."""
        total, oldest, newest = self.conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM messages").fetchone()
        with_files = self.conn.execute(
            "SELECT COUNT(*) FROM messages"
            " WHERE COALESCE(attachment_exts,'') NOT IN ('', 'none')").fetchone()[0]
        threads = self.conn.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM messages"
            " WHERE COALESCE(thread_id,'') <> ''").fetchone()[0]
        return {"messages": int(total or 0), "with_attachments": int(with_files),
                "threads": int(threads), "oldest": oldest, "newest": newest}

    # -- lexical (BM25) leg ------------------------------------------------
    # -- lexical (BM25) leg ------------------------------------------------
    def _fts_match(self, expr: str, tiers: Optional[list], pool: int,
                   meta_weight: float) -> list[tuple[int, float]]:
        """One MATCH expression -> [(chunk_id, score)], best first.

        bm25() is negative and *lower* is better, so the score is negated:
        both search legs then read 'higher is better'. The third bm25 weight
        belongs to `quoted`: text the sender dragged along is evidence, but
        weaker than what they just wrote.
        """
        qw = float(self.cfg.get("fts_quoted_weight", 0.6))
        sql = ("SELECT f.rowid, bm25(chunks_fts, ?, 1.0, ?) AS s FROM chunks_fts f"
               " JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ?")
        args: list = [float(meta_weight), qw, expr]
        if tiers:
            sql += " AND c.tier IN (%s)" % ",".join("?" * len(tiers))
            args += list(tiers)
        sql += " ORDER BY s LIMIT ?"
        args.append(int(pool))
        try:
            return [(r[0], -float(r[1])) for r in self.conn.execute(sql, args)]
        except sqlite3.OperationalError as e:
            log.warning("skipping the lexical leg (fts query failed: %s)", e)
            return []

    def lexical_search(self, query: str, tiers: Optional[list] = None,
                       pool: Optional[int] = None,
                       meta_weight: Optional[float] = None,
                       prefix: bool = True,
                       or_first: Optional[bool] = None) -> list[tuple[int, float]]:
        """BM25 over the FTS index -> [(chunk_id, score)], best first.

        Two shapes of expression are tried, and which one *ranks* is the whole
        question. The old order - strict AND first, OR only when the AND could
        not fill the pool - quietly decided that a query had to contain every
        word, which a natural-language query almost never does: 'domein
        verhuizen authorisatiecode' matched 0 chunks, then fell back to an OR of
        the same words matching 33 366. So the OR form ranks, and bm25's IDF
        does what the AND used to pretend to do (a rare term outweighs a common
        one); matching *every* term is a bonus multiplier instead of a gate.
        """
        pool = int(pool or DEFAULTS["lexical_pool"])
        w = float(self.meta_weight if meta_weight is None else meta_weight)
        or_first = (bool(self.cfg.get("fts_or_first", True))
                    if or_first is None else bool(or_first))
        drop_sw = bool(self.cfg.get("fts_stopwords", True))
        variants = bool(self.cfg.get("expand_morphology", True))
        syn = synonym_map(str(self.cfg.get("synonyms_file") or ""))
        bonus = float(self.cfg.get("fts_full_match_bonus", 1.25))
        joins = (" OR ", " AND ") if or_first else (" AND ", " OR ")
        best: dict[int, float] = {}
        seen: set[str] = set()
        for i, join in enumerate(joins):
            expr = build_fts_query(query, prefix=prefix, join=join,
                                   drop_stopwords=drop_sw, variants=variants,
                                   synonyms=syn)
            if not expr or expr in seen:
                continue
            seen.add(expr)
            strict = join.strip() == "AND"
            for cid, s in self._fts_match(expr, tiers, pool, w):
                s = s * bonus if (strict and or_first) else s
                if cid not in best or s > best[cid]:
                    best[cid] = s
            if not or_first and (len(best) >= max(1, pool // 2)
                                 or i == len(joins) - 1):
                break                      # pool filled, or nothing left to try
        return sorted(best.items(), key=lambda kv: -kv[1])[:pool]

    # -- reindexing support ------------------------------------------------
    def canonical_sources(self, limit: Optional[int] = None) -> list[dict]:
        """Canonical (whole-source) chunks, excluding media: what tiers are
        re-derived from, so `reindex` needs no IMAP round-trip.

        `canonical` is a column, not `nwin == 1`: deduplication can collapse a
        tier to a single window, and reading the index back from that window
        would quietly shrink the indexed text of the message.
        """
        min_len = min(self.sizes.values())
        sql = (
            "SELECT c.id, c.message_id, c.kind, c.src, c.label, c.tier, c.text,"
            " m.sender, m.to_addrs, m.cc, m.subject, m.attachment_exts, m.date"
            " FROM chunks c JOIN messages m ON m.message_id = c.message_id"
            " WHERE c.canonical = 1 AND c.tier != ?"
            "   AND length(COALESCE(c.text,'')) > ?"
            " ORDER BY c.message_id, c.id"
        )
        args: list = [MEDIA_TIER, min_len]
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        keys = ("id", "message_id", "kind", "src", "label", "tier", "text",
                "sender", "to_addrs", "cc", "subject", "attachment_exts", "date")
        return [dict(zip(keys, r)) for r in self.conn.execute(sql, args)]

    def delete_derived(self, tiers: list[str],
                       message_ids: Optional[list] = None) -> int:
        """Drop the window chunks of the given tiers, with their vectors and
        FTS rows, optionally only for those messages. Canonical and media
        chunks are never touched: they hold the source text and its vector."""
        if not tiers:
            return 0
        removed = 0
        # An empty `message_ids` means "every message", not "none of them".
        # Chunked so a mailbox-wide sweep does not build one IN list per id.
        if message_ids:
            ids = list(message_ids)
            groups = [ids[i:i + 900] for i in range(0, len(ids), 900)]
        else:
            groups = [None]
        for group in groups:
            cond = "canonical = 0 AND tier IN (%s)" % ",".join("?" * len(tiers))
            args: list = list(tiers)
            if group:
                cond += " AND message_id IN (%s)" % ",".join("?" * len(group))
                args += list(group)
            self.conn.execute(
                f"DELETE FROM embeddings WHERE chunk_id IN"
                f" (SELECT id FROM chunks WHERE {cond})", args)
            self.conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN"
                f" (SELECT id FROM chunks WHERE {cond})", args)
            removed += self.conn.execute(
                f"DELETE FROM chunks WHERE {cond}", args).rowcount
        self.conn.commit()
        self.invalidate_index()
        return removed

    def fts_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    def stats(self) -> dict:
        m = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        c = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        k = self.conn.execute(
            "SELECT kind, COUNT(*) FROM chunks GROUP BY kind"
        ).fetchall()
        t = self.conn.execute(
            "SELECT COALESCE(tier,'?'), COUNT(*), AVG(LENGTH(text))"
            " FROM chunks GROUP BY tier"
        ).fetchall()
        derived = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE canonical = 0").fetchone()[0]
        dim = self.conn.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()
        dim = int(dim[0]) if dim else 0
        return {
            "messages": m,
            "chunks": c,
            "by_kind": {k0: n for k0, n in k},
            "by_tier": {t0: n for t0, n, _ in t},
            "avg_len_by_tier": {t0: int(avg or 0) for t0, _, avg in t},
            "derived_chunks": derived,
            "fts_docs": self.fts_count(),
            "fts_tokenizer": self.get_state("fts_tokenizer") or self.tokenizer,
            "chunk_sizes": self.get_state("chunk_sizes"),
            "vectors": self.conn.execute(
                "SELECT COUNT(*) FROM embeddings").fetchone()[0],
            "vector_bytes": c * dim * 4,
            "dim": dim,
            "removed": self.removed_count(),
            "last_reconcile_at": self.get_state("last_reconcile_at") or "",
            "last_removal_at": self.conn.execute(
                "SELECT MAX(removed_at) FROM removed_messages").fetchone()[0] or "",
        }


# ---------------------------------------------------------------------------
# Hybrid search: dense cosine + BM25, fused with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class RerankClient:
    """Cross-encoder reranker over the Jina/llama.cpp `/v1/rerank` shape.

    `llama-server -m bge-reranker-v2-m3.gguf --rerank` answers
    POST /v1/rerank {"query","documents"} -> {"results":[{"index","relevance_score"}]},
    which is also what the Jina/Cohere APIs answer, so one client covers a
    local model and a hosted one.

    Off by default, and it should stay that way unless `bench` proves it: a
    cross-encoder re-reads every candidate against the query, so on a corpus
    this small it mostly re-decides the order stage 1 already got wrong - and
    the field's own reports (and this project's measurements) include runs where
    adding it made quality worse. Enable it only with a measured gain.
    """

    def __init__(self, cfg: dict):
        self.url = str(cfg.get("rerank_endpoint") or "").strip()
        self.key = str(cfg.get("rerank_api_key") or "").strip()
        self.model = str(cfg.get("rerank_model") or "").strip()
        self.timeout = float(cfg.get("rerank_timeout", 120))
        self.warned = False

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Scores in the order of `docs`, or [] when the endpoint is unusable."""
        if not self.url or not docs:
            if not self.warned:
                log.warning("rerank is enabled but rerank_endpoint is empty - "
                            "using the fused order")
                self.warned = True
            return []
        body = {"query": query, "documents": docs}
        if self.model:
            body["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        try:
            r = requests.post(self.url, json=body, headers=headers,
                              timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("rerank endpoint failed (%s) - keeping the fused order", e)
            self.url = ""                # one warning per process, not per query
            return []
        out = [0.0] * len(docs)
        results = data.get("results") or data.get("scores") or []
        for item in results:
            if isinstance(item, dict) and "index" in item:
                i = int(item["index"])
                if 0 <= i < len(out):
                    out[i] = float(item.get("relevance_score", item.get("score", 0.0)))
            elif isinstance(item, (int, float)):
                pass                     # positional form handled below
        if not any(out):
            scores = data.get("scores")
            if isinstance(scores, list) and len(scores) == len(docs):
                out = [float(s) for s in scores]
        top = max(out) if out else 0.0
        if top > 1.0:                    # some hosts return unbounded logits
            out = [s / top for s in out]
        return out


@dataclass
class SearchParams:
    """Everything about one query that is not the query text."""
    mode: str = "hybrid"            # hybrid | semantic | lexical
    top: int = 10
    pool: int = DEFAULTS["lexical_pool"]
    dense_pool: int = 0             # 0 = score every chunk
    rrf_k: float = DEFAULTS["rrf_k"]
    agg_k: int = DEFAULTS["agg_k"]
    w_dense: float = DEFAULTS["w_dense"]
    w_lex: float = DEFAULTS["w_lex"]
    tiers: Optional[list] = None    # None = every tier
    prefix: bool = True
    time_weight: bool = False
    alpha: float = 0.3
    meta_weight: Optional[float] = None
    tier_weights: dict = field(default_factory=dict)
    fusion_pool: int = DEFAULTS["fusion_pool"]
    tier_norm: bool = DEFAULTS["tier_norm"]
    window_dedupe: float = DEFAULTS["window_dedupe"]
    collapse: bool = DEFAULTS["collapse"]
    collapse_sim: float = DEFAULTS["collapse_sim"]
    recency_half_life_days: float = DEFAULTS["recency_half_life_days"]
    recency_weight: float = DEFAULTS["recency_weight"]
    demote_transactional: float = DEFAULTS["demote_transactional"]
    transactional_senders: tuple = ()
    min_relevance: float = DEFAULTS["min_relevance"]
    snippet: bool = DEFAULTS["snippet"]
    rerank: str = DEFAULTS["rerank"]
    rerank_candidates: int = DEFAULTS["rerank_candidates"]
    rerank_blend: float = DEFAULTS["rerank_blend"]
    filters: object = None          # queryparse.Filters, applied as a pre-filter
    explain: bool = False


@dataclass
class Hit:
    """One message-level result: fused score plus what each leg thought."""
    message_id: str
    score: float
    fused: float
    dense: Optional[float]
    dense_rank: Optional[int]
    lex: Optional[float]
    lex_rank: Optional[int]
    date: datetime
    sender: str
    subject: str
    label: str
    tier: str
    text: str
    url: str
    days: Optional[float] = None
    prior: float = 1.0
    hash: str = ""
    thread_id: str = ""
    also: int = 0                   # messages collapsed into this one
    thread_len: int = 1
    rerank: Optional[float] = None
    snippet: str = ""

    def as_dict(self) -> dict:
        return {
            "message_id": self.message_id, "score": round(self.score, 6),
            "fused": round(self.fused, 6), "prior": round(self.prior, 4),
            "dense": None if self.dense is None else round(self.dense, 6),
            "dense_rank": self.dense_rank,
            "bm25": None if self.lex is None else round(self.lex, 6),
            "bm25_rank": self.lex_rank,
            "rerank": None if self.rerank is None else round(self.rerank, 6),
            "date": self.date.isoformat(), "sender": self.sender,
            "subject": self.subject, "label": self.label, "tier": self.tier,
            "text": self.text, "snippet": self.snippet,
            "gmail_url": self.url, "thread_id": self.thread_id,
            "also_in_thread": self.also, "thread_len": self.thread_len,
            "age_days": None if self.days is None else round(self.days, 2),
        }


def rank_map(agg: dict) -> dict:
    """{key: (score, ...)} -> {key: 1-based rank}, best score first."""
    return {k: i + 1 for i, (k, _) in enumerate(
        sorted(agg.items(), key=lambda kv: -kv[1][0]))}


def tier_stats(idx: IndexArrays, rows: list, tiers: Optional[list]) -> dict:
    """{tier: (mean, std)} over the chunk scores one leg produced.

    A cosine over a 236-character window and a cosine over a 3 400-character
    body do not live on the same scale (short windows sit higher: less topic
    dilution), so averaging the raw top-3 of a message silently preferred
    whichever tier its chunks happened to fall in. Standardising per tier
    inside a leg makes the mean mean something.
    """
    acc: dict[str, list] = {}
    for i, s in rows:
        t = idx.tiers[i]
        if tiers and t not in tiers:
            continue
        acc.setdefault(t, []).append(float(s))
    out = {}
    for t, vals in acc.items():
        a = np.asarray(vals, dtype=np.float64)
        out[t] = (float(a.mean()), float(a.std()) or 1.0)
    return out


def params_from_cfg(cfg: dict, **overrides) -> "SearchParams":
    """SearchParams from config, with explicit overrides applied last.

    One place, so adding a knob means adding a DEFAULTS entry - it is then
    settable from config.json, from the CLI, and from eval.py without touching
    three call sites.
    """
    p = SearchParams(
        mode=str(cfg.get("search_mode", "hybrid")),
        pool=int(cfg.get("lexical_pool", DEFAULTS["lexical_pool"])),
        dense_pool=int(cfg.get("dense_pool", 0)),
        rrf_k=float(cfg.get("rrf_k", DEFAULTS["rrf_k"])),
        agg_k=int(cfg.get("agg_k", DEFAULTS["agg_k"])),
        w_dense=float(cfg.get("w_dense", DEFAULTS["w_dense"])),
        w_lex=float(cfg.get("w_lex", DEFAULTS["w_lex"])),
        tier_weights=dict(cfg.get("tier_weights") or {}),
        fusion_pool=int(cfg.get("fusion_pool", DEFAULTS["fusion_pool"])),
        tier_norm=bool(cfg.get("tier_norm", DEFAULTS["tier_norm"])),
        window_dedupe=float(cfg.get("window_dedupe", DEFAULTS["window_dedupe"])),
        collapse=bool(cfg.get("collapse", DEFAULTS["collapse"])),
        collapse_sim=float(cfg.get("collapse_sim", DEFAULTS["collapse_sim"])),
        recency_half_life_days=float(
            cfg.get("recency_half_life_days", DEFAULTS["recency_half_life_days"])),
        recency_weight=float(cfg.get("recency_weight", DEFAULTS["recency_weight"])),
        demote_transactional=float(
            cfg.get("demote_transactional", DEFAULTS["demote_transactional"])),
        transactional_senders=tuple(cfg.get("transactional_senders") or ()),
        min_relevance=float(cfg.get("min_relevance", DEFAULTS["min_relevance"])),
        snippet=bool(cfg.get("snippet", DEFAULTS["snippet"])),
        rerank=str(cfg.get("rerank", DEFAULTS["rerank"])),
        rerank_candidates=int(cfg.get("rerank_candidates",
                                      DEFAULTS["rerank_candidates"])),
        rerank_blend=float(cfg.get("rerank_blend", DEFAULTS["rerank_blend"])),
    )
    for key, val in overrides.items():
        if val is None:
            continue
        if not hasattr(p, key):
            raise TypeError(f"unknown search option {key!r}")
        setattr(p, key, val)
    return p


def apply_filters(cfg: dict, query: str) -> tuple[str, object]:
    """Split a query into (text worth embedding, structured filters).

    Returns the query unchanged when the parser is absent or found nothing: a
    missing queryparse.py must degrade search quality by nothing at all.
    """
    if queryparse is None or not cfg.get("parse_filters", True):
        return query, None
    try:
        f = queryparse.parse(query)
    except Exception as e:                # a parser bug must not kill search
        log.warning("query filter parsing failed (%s) - searching as typed", e)
        return query, None
    if f.is_empty():
        return query, None
    # 'van flexa' with nothing else left would embed an empty string: keep the
    # original text in that case, the filter is the only thing the user meant.
    return (f.residual or query), f


class Searcher:
    """Turns the two retrieval legs into one message-level ranking.

    Legs:
      dense     cosine(query, chunk) over every chunk (EmbeddingGemma 2);
      lexical   bm25 over chunk text + metadata + quoted text, via SQLite FTS5.

    Each leg is rolled up from chunks to messages as the **mean of its top-k
    windows**, after dropping exact-duplicate windows of the same message and
    (for the shortlist) near-duplicates: a plain max lets a long message win on
    lottery tickets, a plain mean buries one exact hit in its own filler, and
    twenty near-identical windows of one marketing mail outvote one short
    relevant mail. Windows are standardised per tier first (`tier_stats`).

    The two rankings are merged with RRF, `w/(k + rank)` per leg - ranks, not
    scores, because a cosine of 0.62 and a bm25 of -3.1e-06 are not on the same
    scale, while 'rank 1' means the same thing in both. Each leg is capped at
    `fusion_pool` messages first: adding 0.5/(k+4000) to a bm25 term that only
    ever reaches rank 200 is not a weak signal, it is a different scale.

    The fused value is divided by its analytic maximum (rank 1 in both legs),
    so 1.0 always means 'top of both legs' - which makes it a comparable
    *relevance* across queries, and therefore usable as an abstention floor.

    Two bounded priors then reorder inside that ranking (multiplicative, max
    1.0, so they can never promote something the legs disliked): a recency
    half-life, and a penalty for system senders. Finally one result is kept per
    conversation (thread id, identical body, or bodies whose vectors are
    `collapse_sim` apart), because ten copies of one PostNL tracking mail is
    not ten answers.
    """

    def __init__(self, cfg: dict, store: Store, client: Optional[EmbeddingClient] = None):
        self.cfg = cfg
        self.store = store
        self.client = client

    def params(self, **overrides) -> "SearchParams":
        """SearchParams from this searcher's config, with overrides applied."""
        return params_from_cfg(self.cfg, **overrides)

    def _allow_set(self, p: "SearchParams") -> Optional[set]:
        """Message ids surviving the `from:/met bijlage/vorige week` filters."""
        f = p.filters
        if f is None or getattr(f, "is_empty", lambda: True)():
            return None
        return self.store.filter_message_ids(f)

    def _aggregate(self, idx: IndexArrays, rows: list, tiers: Optional[list],
                   p: SearchParams, allow: Optional[set] = None,
                   stats: Optional[dict] = None) -> dict:
        """[(row, score)] -> {message_id: (score, best row, top windows)}.

        The third element keeps the windows that produced the score, so the
        shortlist can be re-scored with near-duplicate windows collapsed
        (`_dedupe_windows`) without another pass over the index.
        """
        per: dict[str, list] = {}
        for i, s in rows:
            tier = idx.tiers[i]
            if tiers and tier not in tiers:
                continue
            if allow is not None and idx.ids[i] not in allow:
                continue
            if stats:
                mu, sd = stats.get(tier, (0.0, 1.0))
                s = (s - mu) / (sd or 1.0)
            w = float(p.tier_weights.get(tier, 1.0)) if p.tier_weights else 1.0
            per.setdefault(idx.ids[i], []).append((s * w, i))
        k = max(1, int(p.agg_k))
        out: dict[str, tuple] = {}
        for mid, vals in per.items():
            vals.sort(key=lambda t: -t[0])
            keep, seen = [], set()
            for s, i in vals[: max(k, 12)]:
                h = idx.hashes[i] if idx.hashes else ""
                if h and h in seen:
                    continue         # the same text twice is not twice the evidence
                if h:
                    seen.add(h)
                keep.append((s, i))
            out[mid] = (float(np.mean([v for v, _ in keep[:k]])),
                        max(keep, key=lambda t: (t[0], -len(idx.texts[t[1]])))[1],
                        keep)
        return out

    def _dedupe_windows(self, idx: IndexArrays, agg: dict, p: "SearchParams",
                        limit: int) -> None:
        """Collapse near-identical windows of one message before the top-k mean.

        Chunking with 15 % overlap over a 20-window body produces windows that
        are 90 % the same text; each one is a free extra vote for that message.
        Only the shortlist is touched - a message outside the top `limit` is
        not going to be reordered by this.
        """
        thr = float(p.window_dedupe)
        if thr <= 0 or idx.vecs.size == 0:
            return
        k = max(1, int(p.agg_k))
        for mid, _ in sorted(agg.items(), key=lambda kv: -kv[1][0])[:limit or None]:
            score, best, keep = agg[mid]
            if len(keep) <= k:
                continue
            kept: list = []
            for s, i in keep:
                v = idx.vecs[i]
                if any(float(np.dot(v, idx.vecs[j])) >= thr for _, j in kept):
                    continue
                kept.append((s, i))
                if len(kept) >= k:
                    break
            if len(kept) < len(keep[:k]):
                agg[mid] = (float(np.mean([v for v, _ in kept])),
                            max(kept, key=lambda t: (t[0], -len(idx.texts[t[1]])))[1],
                            kept)

    @staticmethod
    def _cap(agg: dict, limit: int) -> dict:
        """Keep the top `limit` messages of one leg, best first.

        RRF adds `w/(k+rank)` per leg. A leg that ranks all 4 452 messages
        against one capped at 200 chunks does not contribute a weak opinion, it
        contributes a different scale: 0.5/(60+4000) is not 'slightly worse than
        rank 200', it is noise added to a term that only goes to 200.
        """
        if not limit or len(agg) <= limit:
            return agg
        return dict(sorted(agg.items(), key=lambda kv: -kv[1][0])[: int(limit)])

    def _prior(self, p: SearchParams, sender: str, days: float) -> float:
        """Bounded [0..1] multiplier from recency and sender type.

        Multiplicative and capped at 1.0 on purpose: a prior may reorder two
        messages the legs already liked, it must never lift an irrelevant one
        into the top 3 (an additive time term did exactly that - the old
        --time-weight blended 30 % of 'how new is it' into the score).
        """
        prior = 1.0
        hl = float(p.recency_half_life_days)
        w = min(1.0, max(0.0, float(p.recency_weight)))
        if hl > 0 and w > 0:
            prior *= (1.0 - w) + w * (0.5 ** (days / hl))
        dem = float(p.demote_transactional)
        if dem < 1.0 and is_transactional(sender, p.transactional_senders):
            prior *= dem
        return prior

    def _collapse(self, idx: IndexArrays, hits: list, p: SearchParams,
                  rep_row: dict, canon: Optional[dict] = None) -> tuple[list, int]:
        """One result per conversation: same thread, same body, same picture.

        Returns (kept, dropped). Without this the top 10 of a query about a
        recurring sender was that one sender ten times over.

        `canon` maps a message to the fingerprint of its whole body: a window
        hit's own fingerprint is only part of the text, and the same letter
        reached through two windows is one answer to the reader, not two.
        """
        canon = canon or {}
        kept: list[Hit] = []
        seen_threads: set[str] = set()
        seen_hashes: set[str] = set()
        reps: list = []
        dropped = 0
        have_vecs = idx.vecs.size > 0
        for h in hits:
            key = h.thread_id or h.message_id
            if key in seen_threads:
                dropped += 1
                continue
            body = canon.get(h.message_id) or h.hash
            if body and body in seen_hashes:
                dropped += 1
                continue
            row = rep_row.get(h.message_id, -1)
            v = idx.vecs[row] if (have_vecs and row >= 0) else None
            if v is not None and reps:
                if max(float(np.dot(v, r)) for r in reps) >= float(p.collapse_sim):
                    dropped += 1
                    continue
            seen_threads.add(key)
            if body:
                seen_hashes.add(body)
            if v is not None:
                reps.append(v)
            kept.append(h)
        return kept, dropped

    def _rerank(self, hits: list, query: str, p: SearchParams) -> None:
        """Cross-encode the shortlist and blend it into the score, in place."""
        if not hits or p.rerank == "off":
            return
        client = getattr(self, "_reranker", None)
        if client is None:
            client = self._reranker = RerankClient(self.cfg)
        docs = [f"{h.subject}\n{(h.snippet or h.text)[:1500]}" for h in hits]
        scores = client.rerank(query, docs)
        if not scores:
            p.rerank = "off"          # endpoint dead: do not retry per query
            return
        blend = min(1.0, max(0.0, float(p.rerank_blend)))
        for h, s in zip(hits, scores):
            h.rerank = s
            h.score = blend * s + (1.0 - blend) * h.score

    def run(self, query: str,
            p: Optional[SearchParams] = None,
            image: Optional[np.ndarray] = None,
            image_name: str = "") -> tuple[list[Hit], dict]:
        """Rank the mailbox for one query, which may be words, a picture, or both.

        `image` is a query vector for an image (see
        `EmbeddingClient.embed_query_image`), already embedded by the caller:
        this method stays the ranking, so it can be tested without an endpoint.
        """
        p = p if p is not None else self.params()
        need_vecs = p.mode in ("hybrid", "semantic")
        if image is not None and p.mode == "lexical":
            raise SystemExit(
                "an image can only be searched by the vector leg, and mode "
                "'lexical' is keywords only - switch to hybrid or semantic")
        # 'facturen van flexa van vorige week' is two different questions asked
        # of two different engines: sender+date are SQL constraints, 'facturen'
        # is the thing to find. Embedding the whole sentence asks the vector for
        # both and satisfies neither.
        text, filters = apply_filters(self.cfg, query)
        parsed = filters is not None and p.filters is None
        if filters is not None and p.filters is None:
            p.filters = filters
        w_d = float(p.w_dense) if p.mode != "lexical" else 0.0
        w_l = float(p.w_lex) if p.mode != "semantic" else 0.0
        image_only = image is not None and not text.strip()
        if image_only:
            # Nothing was typed, so the keyword leg has no words to run. A
            # picture is a vector-leg-only input: reporting `hybrid` with a bm25
            # weight here would promise a second opinion that cannot exist.
            w_l = 0.0
        if w_d <= 0 and w_l <= 0:
            raise SystemExit("both search legs are weighted 0 "
                             "(--w-dense / --w-lex)")

        allow = self._allow_set(p)
        if allow is not None and not allow:
            if parsed:
                # A filter read out of the sentence can be wrong about the
                # sentence: 'Afgeleverd: je pakket van Lyon' holds no sender
                # called Lyon, and 'Statusupdate van je order' no sender called
                # je. An empty allow set then says the parse was wrong, not that
                # the mailbox has no answer, so the query is searched as typed -
                # an empty list is never the useful answer for a guess.
                log.debug("parsed filters %s match no message, searching as typed",
                          _filter_text(filters))
                p.filters, text, allow = None, query, None
            else:
                # The filters alone are decisive: no message can match, and there
                # is no point embedding the query to find nothing.
                return [], {"mode": p.mode, "query": query, "abstained": True,
                            "relevance": 0.0, "filters": _filter_text(p.filters),
                            "filters_struct": _filters_dict(p.filters),
                            "image": image_name or "", "image_only": image_only,
                            "dense_candidates": 0, "lexical_candidates": 0,
                            "chunks_loaded": 0, "collapsed": 0,
                            "note": "no message matches the from:/date/attachment filters"}

        idx = self.store.load_index(with_vectors=need_vecs)
        if len(idx) == 0:
            raise SystemExit("index is empty - run 'python mailbox.py index' first")
        if need_vecs and self.store.stats()["vectors"] == 0:
            raise SystemExit("no vectors in the index - run "
                             "'python mailbox.py index --full' first")

        # Leg 1: dense.
        dense_rows: list[tuple[int, float]] = []
        if need_vecs:
            if self.client is None:
                raise SystemExit("missing config: embed_endpoint")
            sims = None
            if image is not None:
                im = idx.vecs @ image
                # Either side may carry the match, so a chunk is scored by its
                # better cosine, not by the mean of the two. A mean asks one
                # chunk to resemble the picture *and* the words at once, which
                # is the case where you needed neither: the screenshot of the
                # invoice you are describing is found by the words, the photo of
                # the part you cannot name by the pixels - averaged, both sink.
                sims = im
            if text.strip() or image is None:
                # `image is None` keeps an empty query embedding as the empty
                # query, which is what every caller without a picture always got.
                tx = idx.vecs @ self.client.embed_query(text)
                sims = tx if sims is None else np.maximum(sims, tx)
            order = np.argsort(-sims)
            if p.dense_pool:
                order = order[: int(p.dense_pool)]
            dense_rows = [(int(i), float(sims[i])) for i in order]

        # Leg 2: lexical (chunk ids mapped back onto the loaded rows).
        lex_rows: list[tuple[int, float]] = []
        fts_docs = self.store.fts_count()
        if w_l > 0:
            if fts_docs == 0:
                log.warning("FTS index is empty - the lexical leg finds nothing; "
                            "run: python mailbox.py reindex --rebuild-fts")
            else:
                row_of = {cid: i for i, cid in enumerate(idx.cids)}
                for cid, s in self.store.lexical_search(
                        text, tiers=p.tiers, pool=p.pool,
                        meta_weight=p.meta_weight, prefix=p.prefix):
                    i = row_of.get(cid)
                    if i is not None:
                        lex_rows.append((i, s))

        st_d = tier_stats(idx, dense_rows, p.tiers) \
            if (p.tier_norm and dense_rows) else None
        st_l = tier_stats(idx, lex_rows, p.tiers) \
            if (p.tier_norm and lex_rows) else None
        dense_agg = self._aggregate(idx, dense_rows, p.tiers, p, allow, st_d) \
            if w_d > 0 else {}
        lex_agg = self._aggregate(idx, lex_rows, p.tiers, p, allow, st_l) \
            if w_l > 0 else {}
        dense_agg = self._cap(dense_agg, p.fusion_pool)
        lex_agg = self._cap(lex_agg, max(p.fusion_pool, p.pool))
        self._dedupe_windows(idx, dense_agg, p, p.fusion_pool or len(dense_agg))
        rd = rank_map(dense_agg) if w_d > 0 else {}
        rl = rank_map(lex_agg) if w_l > 0 else {}

        # RRF, normalised so that 'rank 1 in both active legs' is exactly 1.0.
        ceiling = ((w_d / (p.rrf_k + 1.0)) if w_d > 0 else 0.0) + \
                  ((w_l / (p.rrf_k + 1.0)) if w_l > 0 else 0.0)
        now = datetime.now(timezone.utc)
        thread_len = self.store.thread_counts()
        rep_row: dict[str, int] = {}
        hits: list[Hit] = []
        for mid in set(rd) | set(rl):
            fused = 0.0
            if mid in rd:
                fused += w_d / (p.rrf_k + rd[mid])
            if mid in rl:
                fused += w_l / (p.rrf_k + rl[mid])
            norm = fused / ceiling if ceiling else 0.0
            # Show the chunk from the leg that ranked this message best.
            if mid in rd and (mid not in rl or rd[mid] <= rl[mid]):
                i = dense_agg[mid][1]
            else:
                i = lex_agg[mid][1]
            rep_row[mid] = i
            d = idx.dates[i]
            days = max(0.0, (now - d).total_seconds() / 86400.0)
            prior = self._prior(p, idx.senders[i], days)
            score = norm * prior
            if p.time_weight:
                # The explicit switch stays an explicit, additive blend.
                score = (1 - p.alpha) * norm + p.alpha * float(np.exp(-days / 30.0))
            hits.append(Hit(
                message_id=mid, score=score, fused=norm,
                dense=dense_agg.get(mid, (None,))[0], dense_rank=rd.get(mid),
                lex=lex_agg.get(mid, (None,))[0], lex_rank=rl.get(mid),
                date=d, sender=idx.senders[i], subject=idx.subjects[i],
                label=idx.labels[i], tier=idx.tiers[i], text=idx.texts[i],
                url=idx.urls[i], days=days, prior=prior,
                thread_id=idx.threads[i] or mid,
                hash=idx.hashes[i] if idx.hashes else "",
                thread_len=int(thread_len.get(idx.threads[i] or mid, 1) or 1),
            ))
        hits.sort(key=lambda h: (-h.score, h.message_id))

        collapsed = 0
        if p.collapse:
            # One read for the whole candidate set, not one per hit.
            canon = self.store.canonical_body_hashes(set(rd) | set(rl))
            hits, collapsed = self._collapse(idx, hits, p, rep_row, canon)

        if p.rerank != "off" and text.strip():
            # A cross-encoder compares two texts. With an image and no words
            # there is nothing for it to compare against, and re-ordering the
            # list on an empty query would undo the vector ranking that found
            # these hits in the first place.
            self._rerank(hits[: max(1, int(p.rerank_candidates))], text, p)
            hits.sort(key=lambda h: (-h.score, h.message_id))

        top_n = max(1, int(p.top))
        relevance = hits[0].score if hits else 0.0
        abstained = relevance < float(p.min_relevance)
        if p.snippet:
            for h in hits[:top_n]:
                h.snippet = keyword_snippet(h.text, text)
                if p.explain:
                    h.label = f"{h.label} (fused {h.fused:.3f}, prior {h.prior:.2f})"

        meta = {
            "mode": p.mode, "w_dense": w_d, "w_lex": w_l, "rrf_k": p.rrf_k,
            "agg_k": p.agg_k, "tiers": p.tiers, "pool": p.pool,
            "fusion_pool": p.fusion_pool, "tier_norm": p.tier_norm,
            "chunks_loaded": len(idx), "chunks_matched": len(set(
                [idx.ids[i] for i, _ in dense_rows] +
                [idx.ids[i] for i, _ in lex_rows])),
            "dense_candidates": len(rd), "lexical_candidates": len(rl),
            "fts_docs": fts_docs, "indexed_messages": len(set(idx.ids)),
            "time_weight": p.time_weight, "alpha": p.alpha,
            "collapsed": collapsed, "abstained": abstained,
            "relevance": round(relevance, 4),
            "min_relevance": float(p.min_relevance),
            "filters": _filter_text(p.filters),
            # The same thing in structure, so a front-end can render each filter
            # as a chip the user can drop and re-search. `filters` alone is prose:
            # it can be read but not taken apart.
            "filters_struct": _filters_dict(p.filters),
            "rerank": p.rerank,
            "query": query,
            # The picture, when one was given: a name for the UI to show back,
            # and whether it was the only thing searched by (which is why the
            # bm25 weight above reads 0 in that case).
            "image": image_name or "",
            "image_only": image_only,
            "embed_text": text if text != query else "",
        }
        return hits[:top_n], meta


def _filters_dict(filters) -> Optional[dict]:
    """The applied filters as plain JSON-able data, or None when there are none.

    `describe()` gives a sentence for a terminal; a GUI needs the fields back to
    draw one removable chip each. Fields are copied by name rather than by
    `asdict()` so a Filters object never leaks an attribute the web layer did not
    ask for.
    """
    if filters is None or queryparse is None:
        return None
    out: dict = {}
    for key in ("senders", "recipients", "subjects", "since", "until",
                "has_attachment", "residual", "notes"):
        try:
            val = getattr(filters, key)
        except AttributeError:
            continue
        if val not in (None, "", [], ()):
            out[key] = val
    return out or None


def _filter_text(filters) -> str:
    """One-line summary of the applied email-intent filters ('' when none)."""
    if filters is None or queryparse is None:
        return ""
    try:
        return queryparse.describe(filters)
    except Exception:
        return ""


def search_query(cfg: dict, query: str, store: Optional[Store] = None,
                 client: Optional[EmbeddingClient] = None,
                 **overrides) -> tuple[list[str], dict]:
    """One-shot search: the stable entry point for eval.py and scripts.

    `overrides` are SearchParams fields (mode, top, tiers, rrf_k, collapse,
    rerank, min_relevance, ...), so adding an option never breaks a caller.
    One override is not a SearchParams field: `profile` reloads the config under
    that knob set, which is how `bench --variant old=profile=legacy` measures
    the old pipeline against the new one in one run.
    Returns (message_ids best-first, meta) - ids rather than Hit objects,
    because that is all a metric needs.
    """
    profile = str(overrides.pop("profile", "") or "")
    if profile and profile != cfg.get("profile"):
        cfg = load_config(cfg.get("config_path", "config.json"), profile)
    store = store or Store(cfg["db_path"], cfg)
    if client is None and cfg.get("embed_endpoint"):
        client = EmbeddingClient(cfg)
    searcher = Searcher(cfg, store, client)
    p = searcher.params(**overrides)
    hits, meta = searcher.run(query, p)
    return [h.message_id for h in hits], meta


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _store_thumbs(media, msg) -> int:
    """Previews for one freshly indexed message, written after it committed.

    Cheap because it is not a second decode pass for the sake of a picture: the
    indexer has these bytes in hand for the embedding call anyway. What is
    skipped mirrors what the index skips — types it cannot read as pixels, and
    files over `max_attachment_mb` — so a message never has a preview for a
    file it would not have indexed.

    First name match wins, like `pick_attachment`: a message carrying two parts
    called `logo.png` gets one preview, and the click resolves to the same part
    the preview shows.
    """
    if media is None:
        return 0
    cap = int(media.cfg.get("max_attachment_mb",
                            DEFAULTS["max_attachment_mb"])) * 1024 * 1024
    n = 0
    seen: set = set()
    for att in msg.attachments:
        ct = (att.content_type or "").lower()
        name = (att.name or "").strip()
        if ct not in IMAGE_TYPES or not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        if len(att.data or b"") > cap:
            continue
        try:
            if media.put_attachment(msg.message_id, name, att.data, ct):
                n += 1
        except Exception as e:
            log.debug("preview failed for %s: %s", name, e)
    return n


def _index_batch(batch, store, extractor, client, depth, chars,
                 media=None) -> int:
    """Index one batch: thread ids, thread context, extract, embed, commit.

    `media` is the optional thumbnail sidecar: previews are written *after* a
    message commits, so an interrupted batch leaves no preview behind for a
    message the index does not have.

    Embedding is two-phase per batch: ALL message text (body + meta) is
    embedded first, and only then are the attachments handled (text-extracted
    ones batched, media ones one at a time). This keeps text indexed before
    any attachment work. Returns the number of messages committed.

    Windows whose normalised text was already embedded for the *same thread*
    are stored without a vector: they stay findable by bm25, but a reply that
    quotes the contract does not get its own copy of that text in the vector
    space - which is how one message ended up as 29 votes for one sentence.
    """
    assign_thread_ids(batch)
    ctx_map = build_thread_contexts(batch, store, depth, chars)

    # Extract every message's chunks up front (order: body, meta, attachments).
    plan = []
    for m in batch:
        chunks = extractor.extract(m, ctx_map.get(m.message_id, ""))
        if chunks:
            plan.append((m, chunks))
    if not plan:
        return 0

    dedupe = bool(store.cfg.get("index_dedupe_threads", True))
    seen: set[tuple[str, str]] = set()
    if dedupe:
        threads = {(m.thread_id or m.message_id) for m, _ in plan}
        seen |= store.embedded_thread_hashes(threads)

    def fresh(m, c) -> bool:
        if not dedupe:
            return True
        h = c.get("text_hash") or text_hash(c.get("text") or "")
        key = (m.thread_id or m.message_id, h)
        if key in seen:
            return False
        seen.add(key)
        return True

    # Per-message vector slots, kept in chunk order so store.commit()'s
    # zip(chunks, vecs) stays aligned.
    vecs_by_plan = [[None] * len(chunks) for _, chunks in plan]

    # Phase 1: embed ALL text (body + meta) for the whole batch first.
    text_items = []  # (plan_idx, chunk_idx, chunk)
    for pi, (m, chunks) in enumerate(plan):
        for ci, c in enumerate(chunks):
            if c["kind"] in ("body", "meta") and fresh(m, c):
                text_items.append((pi, ci, c))
    if text_items:
        tv = client.embed_texts([c["text"] for _, _, c in text_items],
                                titles=[c.get("title", "") for _, _, c in text_items])
        for (pi, ci, _), v in zip(text_items, tv):
            vecs_by_plan[pi][ci] = v

    # Phase 2: only now handle the attachments.
    # 2a: text-extracted attachments (PDF/DOCX/XLSX/PPTX) - batched.
    att_text_items = []
    for pi, (m, chunks) in enumerate(plan):
        for ci, c in enumerate(chunks):
            if c["kind"] == "attachment" and not c.get("image_b64") and fresh(m, c):
                att_text_items.append((pi, ci, c))
    if att_text_items:
        av = client.embed_texts([c["text"] for _, _, c in att_text_items],
                                titles=[c.get("title", "") for _, _, c in att_text_items])
        for (pi, ci, _), v in zip(att_text_items, av):
            vecs_by_plan[pi][ci] = v
    # 2b: media attachments (image/audio) - one at a time (probe + fallback).
    for pi, (m, chunks) in enumerate(plan):
        for ci, c in enumerate(chunks):
            if c.get("image_b64"):
                vecs_by_plan[pi][ci] = client.embed_media(c)

    # Commit (single commit per message; preserves dedup + resume).
    done = 0
    for pi, (m, chunks) in enumerate(plan):
        store.commit(m, chunks, vecs_by_plan[pi])
        _store_thumbs(media, m)
        done += 1
    return done


# ---------------------------------------------------------------------------
# Reconciliation: reflecting mail that was deleted in Gmail
# ---------------------------------------------------------------------------

def _as_int(value) -> int:
    """An IMAP sequence number or UID as an int, 0 when it is not one."""
    try:
        return int(bytes(value).decode("ascii", "replace"))
    except (TypeError, ValueError):
        return 0


def _reconcile_folder(conn, fetcher, store, cfg, folder, *, dry_run=False,
                      force=False, verbose=False) -> dict:
    """Bring one folder's rows in line with what the folder actually holds.

    The folder is the authority, and one `UID SEARCH ALL` says what it holds.
    A stored row is alive when the folder still lists its UID, or - if that UID
    was never recorded - when the folder's headers still carry its Message-ID.
    Neither is true means the message left the folder, and by the config's
    choice that is what leaves the index too: this mirrors the indexed folders
    rather than shadowing the whole account.

    Every way out of here that is not the last one removes nothing, and says
    why in `skipped` - because the one answer that would justify mass deletion
    is "this folder is empty", and that answer is far more often a server,
    login or folder-name problem than a user who cleaned house.
    """
    rows = store.folder_rows(folder)
    res = {"stored": len(rows), "live": 0, "removed": 0, "restored": 0,
           "swept": False, "skipped": "", "uids_backfilled": 0,
           "unverified": 0, "missing": 0, "gone": []}
    try:
        snap = fetcher.snapshot(conn, folder)
    except Exception as e:
        res["skipped"] = ("the folder could not be listed "
                          f"({type(e).__name__}: {e})")
        return res
    res["live"] = len(snap.uids)

    key = "imap_uidvalidity:" + folder
    prev = store.get_state(key) or ""
    cur = str(snap.uidvalidity or "")
    if res["stored"] and not snap.uids:
        res["skipped"] = (f"the folder listed 0 messages while the index holds "
                          f"{res['stored']} - nothing was removed; check the "
                          "folder name and the account before trusting that")
        return res
    if prev and cur and prev != cur:
        # A folder re-created under a new UIDVALIDITY numbers everything again,
        # so every stored UID would look stale at once. Re-map, remove nothing.
        res["swept"] = True
        res["skipped"] = (f"UIDVALIDITY changed {prev} -> {cur}: these UIDs are "
                          "a different numbering than the stored rows, so this "
                          "run re-maps them and removes nothing")
        res["uids_backfilled"] = _reconcile_uids(conn, fetcher, store, folder,
                                                 rows, dry_run=dry_run)
        if not dry_run:
            store.set_state(key, cur)
        return res
    if cur and prev != cur and not dry_run:
        store.set_state(key, cur)

    live_mids: dict = {}
    if any(not uid for _mid, uid in rows):
        res["swept"] = True                       # headers are the only join left
        live_mids = fetcher.id_map(conn, folder)
        res["uids_backfilled"] = _reconcile_uids(
            conn, fetcher, store, folder, rows, live_mids, dry_run=dry_run)
        if not dry_run:
            # with the UIDs now recorded, judge the rows by them as well
            rows = [(mid, live_mids.get(str(mid).lower(), uid) or uid)
                    for mid, uid in rows]

    def alive(mid: str, uid: int) -> bool:
        """The folder lists this row's UID, or its Message-ID is in the folder.

        Either proof is enough. A UID the folder does not list is only evidence
        of absence once the headers have been read too: without that, a message
        whose id Gmail spells differently would be deleted on a technicality.
        """
        if bool(uid) and uid in snap.uids:
            return True
        return bool(live_mids) and str(mid).lower() in live_mids

    gone = [mid for mid, uid in rows if not alive(mid, uid)]
    res["gone"] = gone

    dropped_ids: set = set()
    if gone:
        cap = float(cfg.get("prune_max_fraction", DEFAULTS["prune_max_fraction"]))
        limit = max(1, int(res["stored"] * cap)) if res["stored"] else 0
        if not force and limit and len(gone) > limit:
            res["skipped"] = (f"{len(gone)} of {res['stored']} messages of "
                              f"{folder} are no longer in the folder, more than "
                              f"prune_max_fraction ({cap:g}) - nothing was "
                              "removed; check Gmail, then re-run with "
                              "--prune-force if it is really what you did")
            return res
        dropped = store.remove_messages(gone, dry_run=dry_run)
        dropped_ids = {d["message_id"] for d in dropped}
        res["removed"] = len(dropped)
        res["gone"] = [d["message_id"] for d in dropped]
        if verbose or dry_run:
            # say WHICH, whenever mail is at risk of going: a bare count from a
            # dry run is a claim the user cannot check
            res["shown"] = [f"{(d.get('date') or '?')[:10]}  "
                            f"{d.get('sender') or '?'}  —  "
                            f"{d.get('subject') or '(no subject)'}"
                            for d in dropped]

    # Rows the next run still cannot judge by UID: no UID, no header match, and
    # not one of the rows that are leaving anyway. Nearly always 0; worth a line
    # when it is not, because it says the folder is reporting something odd.
    res["unverified"] = sum(1 for mid, uid in rows
                            if not uid and mid not in dropped_ids
                            and str(mid).lower() not in live_mids)

    # The other direction, which the same two lists give for free: UIDs the
    # folder holds that no row of this index is filed under. Those were never
    # fetched at all - a gap in some old run, mail the watermark walked past -
    # and this run cannot fix that (it is a check, not an index), but naming the
    # number is what makes "the folders are mirrored" an honest sentence: it is
    # a mirror with a known gap, not a mirror that quietly lost the reflection.
    if not dry_run:
        placed = snap.uids & {uid for _mid, uid in rows if uid}
        res["missing"] = len(snap.uids) - len(placed)

    if bool(cfg.get("prune_restore", DEFAULTS["prune_restore"])):
        res["restored"] = _restore_returned(conn, fetcher, store, cfg, folder,
                                           snap, rows, dry_run=dry_run,
                                           verbose=verbose)
    return res


def _reconcile_uids(conn, fetcher, store, folder, rows, live_mids=None,
                    dry_run: bool = False) -> int:
    """Record the UID of every stored row of `folder` the headers could place.

    One-time cost per folder: an index written before UIDs were stored has no
    way to answer "is this message still there?" except by Message-ID, and that
    needs the header sweep. After this, the cheap UID listing is enough.

    A dry run counts and writes nothing: `prune --dry-run` is what a user runs
    to find out whether they dare run the real thing, so it must leave the
    index exactly as it found it.
    """
    ids = live_mids if live_mids is not None else fetcher.id_map(conn, folder)
    n = 0
    for mid, uid in rows:
        found = ids.get(str(mid).lower())
        if found and found != uid:
            if not dry_run:
                store.set_uid(mid, found)
            n += 1
    if n and not dry_run:
        store.conn.commit()
    return n


def _restore_returned(conn, fetcher, store, cfg, folder, snap, rows, *,
                      dry_run=False, verbose=False) -> int:
    """Re-index mail that came back to the folder, by UID.

    A message un-deleted from the bin is older than `last_indexed_date`, so the
    incremental fetch would not offer it again for years. The folder already
    named the UID it lives at now, so one targeted FETCH is the whole repair -
    and `removed_messages` is what makes this bounded: only ids this index used
    to have and no longer has are candidates, so a partly indexed mailbox (where
    every un-indexed message is a "new" UID) does not turn a refresh into a
    re-index of the folder.
    """
    tombstones = {mid: row for mid, row in store.tombstones().items()
                  if (row.get("folder") or folder) == folder}
    if not tombstones:
        return 0
    known = {uid for _mid, uid in rows if uid}
    unknown = sorted(snap.uids - known)
    if not unknown:
        return 0
    cap = int(cfg.get("restore_probe_max", DEFAULTS["restore_probe_max"]))
    if len(unknown) > cap:
        if verbose:
            log.info("%s: %d live UIDs have no row - too many to probe, and most "
                     "were simply never indexed", folder, len(unknown))
        return 0
    by_mid = fetcher.id_map(conn, folder, uids=unknown)
    wanted = {mid: by_mid[mid] for mid in tombstones
              if mid.lower() in by_mid}
    if not wanted:
        return 0
    if dry_run:
        if verbose:
            log.info("%s: %d removed message(s) are back in the folder (dry run)",
                     folder, len(wanted))
        return 0
    msgs = [m for m in fetcher.fetch_uids(conn, folder, list(wanted.values()))
            if not store.has(m.message_id)]
    if not msgs:
        return 0
    extractor = ContentExtractor(cfg)
    client = EmbeddingClient(cfg)
    done = _index_batch(msgs, store, extractor, client,
                        int(cfg["thread_context_depth"]),
                        int(cfg["thread_context_chars"]),
                        media=store.media_cache())
    for mid in wanted:
        if store.has(mid):
            store.clear_tombstone(mid)
    store.conn.commit()
    return done


def reconcile_deletions(cfg: dict, store: "Store", folders: list, *,
                        fetcher=None, dry_run: bool = False,
                        force: bool = False, verbose: bool = False,
                        quiet: bool = False) -> dict:
    """Reflect the mailbox's deletions in the index. Read-only over IMAP.

    Called by `index` (so the GUI's refresh button does it too) and by
    `mailbox.py prune`. Returns a summary, which is also stored in
    `index_state` so the GUI can say what the last refresh did without keeping
    a log file of its own.
    """
    folders = [f for f in (folders or []) if f]
    out: dict = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "folders": {}, "removed": 0, "restored": 0, "dry_run": bool(dry_run)}
    fetcher = fetcher or GmailFetcher(cfg)
    try:
        with fetcher.session() as conn:
            for folder in folders:
                res = _reconcile_folder(conn, fetcher, store, cfg, folder,
                                        dry_run=dry_run, force=force,
                                        verbose=verbose)
                out["folders"][folder] = res
                out["removed"] += int(res.get("removed") or 0)
                out["restored"] += int(res.get("restored") or 0)
    except Exception as e:
        # Login or connection trouble is never a reason to remove anything:
        # keep whatever the index has and say the check did not happen.
        out["error"] = f"{type(e).__name__}: {e}"
    if not quiet:
        print_reconciliation(out)
    if not dry_run:
        # The GUI reads these two keys to describe the last refresh; the ids
        # themselves stay out of the state (they are in `removed_messages`,
        # where they belong and where they can be queried).
        brief = {"removed": out["removed"], "restored": out["restored"],
                 "folders": {f: {k: v for k, v in r.items()
                                 if k in ("stored", "live", "removed", "restored",
                                          "swept", "skipped", "uids_backfilled",
                                          "unverified", "missing")}
                             for f, r in out["folders"].items()}}
        if out.get("error"):
            brief["error"] = out["error"]
        store.set_state("last_reconcile_at", out["at"])
        store.set_state("last_reconcile", json.dumps(brief))
    return out


def print_reconciliation(out: dict) -> None:
    """One line per folder, always - including the line that says nothing changed.

    A reconcile that printed nothing when it removed nothing would leave "did it
    even look?" unanswerable, and that is the first question a user asks after
    deleting something.
    """
    if out.get("error"):
        print(f"reconcile: the mailbox could not be checked ({out['error']}) - "
              "nothing was removed")
    for folder, res in (out.get("folders") or {}).items():
        gone_word = "would be removed" if out.get("dry_run") else "removed"
        bits = [f"{res.get('live', 0)} live", f"{res.get('stored', 0)} stored"]
        if res.get("removed"):
            bits.append(f"{res['removed']} {gone_word}")
        if res.get("restored"):
            bits.append(f"{res['restored']} came back and were re-indexed")
        if res.get("uids_backfilled"):
            bits.append(f"{res['uids_backfilled']} uids "
                        + ("to record" if out.get("dry_run") else "recorded"))
        if res.get("swept"):
            bits.append("headers swept")
        if res.get("unverified"):
            bits.append(f"{res['unverified']} rows still without a uid")
        if res.get("missing"):
            bits.append(f"{res['missing']} in the folder were never indexed")
        if res.get("skipped"):
            bits.append(f"skipped: {res['skipped']}")
        print(f"reconciled {folder}: " + ", ".join(bits))
        for line in (res.get("shown") or [])[:25]:
            print(f"    - {line}")
        if res.get("missing"):
            print("      (mail the folder holds and this index never fetched: "
                  "an incremental run only looks past its watermark, so "
                  "`index --full` is what picks it up)")
        left = len(res.get("shown") or []) - 25
        if left > 0:
            print(f"    ... and {left} more")


def cmd_prune(args) -> None:
    """Report (and by default perform) what Gmail no longer has."""
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    for key in ("gmail_user", "gmail_app_password"):
        if not cfg[key]:
            raise SystemExit(f"missing config: {key} (set it in {args.config})")
    store = Store(cfg["db_path"], cfg)
    folders = (args.folders.split(",") if args.folders else list(cfg["folders"]))
    out = reconcile_deletions(cfg, store, folders, dry_run=args.dry_run,
                              force=args.prune_force, verbose=args.verbose,
                              quiet=True)
    print_reconciliation(out)
    if args.dry_run:
        print("\ndry run: nothing was removed. Re-run without --dry-run to apply.")
    s = store.stats()
    print(f"store now: {s['messages']} messages, {s['chunks']} chunks, "
          f"{s['fts_docs']} fts docs.")
    note = truncation_note()
    if note:
        print(note)


def cmd_index(args) -> None:
    cfg = load_config(args.config, getattr(args, "profile", ""))
    for key in ("gmail_user", "gmail_app_password", "embed_endpoint"):
        if not cfg[key]:
            raise SystemExit(
                f"missing config: {key} (set it in {args.config} or via environment)"
            )
    store = Store(cfg["db_path"], cfg)
    store.warn_if_stale()
    # --limit N (without --full) = a "test run": fetch the N most recent
    # regardless of last_indexed_date, and don't update it, so test runs
    # don't disturb the incremental state.
    test_run = bool(args.limit) and not args.full
    since = None
    before = None
    if args.full:
        # Resume: if a previous --full run was interrupted, continue from the
        # watermark. BEFORE <watermark + 1 day> re-includes the watermark day
        # (catches any partially-processed messages there); dedup handles overlap.
        wm = store.get_state("full_run_watermark")
        if wm:
            before = (datetime.fromisoformat(wm) + timedelta(days=1)).strftime("%d-%b-%Y")
            log.info("resuming --full from watermark %s (BEFORE %s)", wm, before)
    elif not test_run:
        since = store.get_state("last_indexed_date")
    if since:
        since = datetime.fromisoformat(since).strftime("%d-%b-%Y")
    if test_run:
        log.info(
            "test run: fetching the %d most recent (last_indexed_date ignored "
            "and not updated)", args.limit
        )
    folders = args.folders.split(",") if args.folders else cfg["folders"]

    fetcher = GmailFetcher(cfg)
    total = fetcher.count_messages(since, folders, args.limit, before)
    extractor = ContentExtractor(cfg)
    client = EmbeddingClient(cfg)
    # Previews ride along: the bytes are already in hand per message, so an
    # index run is the only moment they are free.
    media = store.media_cache()
    depth = int(cfg["thread_context_depth"])
    chars = int(cfg["thread_context_chars"])
    batch_of = 50

    done = 0
    newest_covered = None
    interrupted = False
    with tqdm.tqdm(total=total if total > 0 else None, desc="indexing") as bar:
        buf = []

        def flush():
            nonlocal done
            if not buf:
                return
            done += _index_batch(buf, store, extractor, client, depth, chars,
                                 media=media)
            if args.full:
                store.set_state(
                    "full_run_watermark",
                    min(m.date for m in buf).date().isoformat(),
                )

        try:
            for m in fetcher.fetch(since, folders, args.limit, before,
                                  debug=args.debug):
                bar.update(1)  # one message processed (indexed or deduped)
                if store.has(m.message_id):
                    continue  # already indexed
                buf.append(m)
                if newest_covered is None or m.date > newest_covered:
                    newest_covered = m.date
                if len(buf) == batch_of:
                    flush()
                    buf = []
            flush()  # final partial batch
        except KeyboardInterrupt:
            interrupted = True
            log.warning("interrupted; %d messages committed so far", done)

    if interrupted:
        wm = store.get_state("full_run_watermark") if args.full else None
        print(
            f"\nInterrupted. Re-run the same command to resume"
            + (f" (watermark {wm})." if wm else ".")
        )
        return

    if args.full:
        store.delete_state("full_run_watermark")  # run complete
        if newest_covered:
            store.set_state("last_indexed_date", newest_covered.isoformat())
    elif not test_run and newest_covered:
        store.set_state("last_indexed_date", newest_covered.isoformat())

    # Now that new mail is in, look at what is no longer in the folders. This
    # runs on every ordinary `index`, including one that indexed nothing -
    # deleting mail produces no new mail, and "press refresh, index follows
    # Gmail" only holds if the check is not gated on having fetched something.
    reconciled = None
    reason = _prune_skipped(cfg, args, test_run)
    if reason is None:
        reconciled = reconcile_deletions(cfg, store, folders, fetcher=fetcher,
                                        quiet=True)
        print_reconciliation(reconciled)

    if done == 0 and not (reconciled and reconciled["removed"]):
        print("Nothing to index - store is up to date."
              + ("" if reconciled else " (reconcile off: " + reason + ")"))
        return
    store.save_index_meta()
    s = store.stats()
    tiers = ", ".join(f"{k}:{v}" for k, v in sorted(s["by_tier"].items()))
    print(f"Indexed {done} new messages. Store now: {s['messages']} messages, "
          f"{s['chunks']} chunks ({tiers}), {s['fts_docs']} fts docs.")
    note = truncation_note()
    if note:
        print(note)


def _prune_skipped(cfg: dict, args, test_run: bool) -> Optional[str]:
    """Why this `index` run must not reconcile, or None when it may.

    Three situations where removing rows would be wrong rather than late: a
    test run (`--limit`) only ever saw part of the folder, so "not in this
    page" means nothing; a database named on the command line that is not the
    one this config indexes; and a config that switched the feature off.
    """
    if not bool(cfg.get("prune_deleted", DEFAULTS["prune_deleted"])):
        return "prune_deleted is false in the config"
    if getattr(args, "no_prune", False):
        return "--no-prune"
    if test_run:
        return "--limit only fetched part of the mailbox"
    other = getattr(args, "db", None)
    if other and Path(other).resolve() != Path(cfg["db_path"]).resolve():
        return f"--db {other} is not the database this config indexes"
    return None


def _resolve_params(args, cfg: dict, mode: str) -> SearchParams:
    """CLI flags win over config, config wins over DEFAULTS.

    Every flag is optional on the parser (`bench` reuses this), so each one is
    read with getattr: adding a knob to one subcommand cannot break another.
    """
    def pick(flag, key, cast=int):
        val = getattr(args, flag, None)
        return cast(val) if val is not None else cast(cfg.get(key, DEFAULTS[key]))

    over: dict = {
        "mode": mode,
        "top": int(args.top),
        "pool": pick("pool", "lexical_pool"),
        "dense_pool": pick("dense_pool", "dense_pool"),
        "rrf_k": pick("rrf_k", "rrf_k", float),
        "agg_k": pick("agg_k", "agg_k"),
        "w_dense": pick("w_dense", "w_dense", float),
        "w_lex": pick("w_lex", "w_lex", float),
        "prefix": not getattr(args, "no_prefix", False),
        "time_weight": bool(getattr(args, "time_weight", False)),
        "alpha": float(getattr(args, "alpha", 0.3) or 0.3),
        "meta_weight": getattr(args, "meta_weight", None),
    }
    tiers = getattr(args, "tiers", None)
    if tiers:
        over["tiers"] = [t.strip() for t in tiers.split(",") if t.strip()]
    flags = [
        ("fusion_pool", "fusion_pool", int),
        ("collapse_sim", "collapse_sim", float),
        ("window_dedupe", "window_dedupe", float),
        ("min_relevance", "min_relevance", float),
        ("half_life", "recency_half_life_days", float),
        ("recency_weight", "recency_weight", float),
        ("demote_transactional", "demote_transactional", float),
        ("rerank_candidates", "rerank_candidates", int),
        ("rerank", "rerank", str),
    ]
    for flag, key, cast in flags:
        val = getattr(args, flag, None)
        if val is not None:
            over[key] = cast(val)
    if getattr(args, "no_collapse", False):
        over["collapse"] = False
    if getattr(args, "no_tier_norm", False):
        over["tier_norm"] = False
    if getattr(args, "no_snippet", False):
        over["snippet"] = False
    if getattr(args, "no_filters", False):
        over["filters"] = _NO_FILTERS          # sentinel: do not parse this query
    if getattr(args, "explain", False):
        over["explain"] = True
    return params_from_cfg(cfg, **over)


class _NoFilters:
    """Sentinel for --no-filters: an empty filter set means 'no constraint',
    which is not the same as 'do not parse'; this says the latter."""

    is_empty = staticmethod(lambda: True)


_NO_FILTERS = _NoFilters()


def _fmt_rrf(h: Hit, meta: dict) -> str:
    """Show the RRF arithmetic behind a rank, for --explain."""
    k = meta["rrf_k"]
    parts = []
    if meta["w_dense"] > 0 and h.dense_rank:
        parts.append(f"dense {meta['w_dense']:g}/({k:g}+{h.dense_rank})")
    if meta["w_lex"] > 0 and h.lex_rank:
        parts.append(f"bm25 {meta['w_lex']:g}/({k:g}+{h.lex_rank})")
    raw = 0.0
    if meta["w_dense"] > 0 and h.dense_rank:
        raw += meta["w_dense"] / (k + h.dense_rank)
    if meta["w_lex"] > 0 and h.lex_rank:
        raw += meta["w_lex"] / (k + h.lex_rank)
    term = " + ".join(parts) if parts else "-"
    extra = f"  age {h.days:.0f}d blended by alpha={meta['alpha']:g}" if h.days is not None else ""
    return f"   rrf:     {term} = {raw:.5f} -> {h.fused:.3f}{extra}"


def _apply_db(args, cfg: dict) -> dict:
    """`--db` on the command line wins over the config's db_path: measuring a
    frozen copy of an index must not need a second config file."""
    db = getattr(args, "db", None)
    if db:
        cfg["db_path"] = str(db)
    return cfg


def cmd_search(args) -> None:
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    mode = (args.mode or cfg.get("search_mode") or DEFAULTS["search_mode"]).strip().lower()
    if mode not in ("hybrid", "semantic", "lexical"):
        raise SystemExit(f"unknown --mode {mode!r} (use hybrid | semantic | lexical)")
    store = Store(cfg["db_path"], cfg)
    store.warn_if_stale()
    client = EmbeddingClient(cfg) if cfg["embed_endpoint"] else None
    params = _resolve_params(args, cfg, mode)
    hits, meta = Searcher(cfg, store, client).run(args.query, params)

    if args.log:
        _log_query(args.log, meta, hits)

    if args.json:
        print(json.dumps({"meta": meta, "results": [h.as_dict() for h in hits]},
                         ensure_ascii=False, indent=2))
        return

    legs = []
    if meta["w_dense"] > 0:
        legs.append(f"dense {meta['w_dense']:g}")
    if meta["w_lex"] > 0:
        legs.append(f"bm25 {meta['w_lex']:g}")
    desc = " + ".join(legs)
    print(f"Query: {args.query!r}")
    if meta.get("embed_text"):
        print(f"Searching: {meta['embed_text']!r}")
    if meta.get("filters"):
        print(f"Filters:   {meta['filters']}")
    print(f"Top {len(hits)} of {meta['chunks_matched']} matched messages "
          f"[{mode}: {desc}]\n")
    detail = (f"chunks: {meta['chunks_loaded']} scored ({meta['dense_candidates']} "
              f"messages in the fusion pool)" if meta["w_dense"] > 0
              else "chunks: cosine leg off")
    detail += (f", bm25 pool {meta['pool']} -> {meta['lexical_candidates']} messages "
               f"from {meta['fts_docs']} fts docs" if meta["w_lex"] > 0 else "")
    detail += f", tiers: {','.join(params.tiers) if params.tiers else 'all'}"
    detail += f", per-message top-{meta['agg_k']} mean"
    if meta.get("collapsed"):
        detail += f", {meta['collapsed']} collapsed into other conversations"
    print(detail + "\n")

    if meta.get("abstained"):
        # Saying 'nothing here' is a result. A fixed score cut that always
        # returns 10 hits is how a search tool turns a miss into a lie.
        top = meta.get("relevance", 0.0)
        print(f"No result above the relevance floor "
              f"(best {top:.3f} < --min-relevance {meta['min_relevance']:g}).")
        if hits:
            print("Closest, for reference only:\n")
    for rank, h in enumerate(hits, 1):
        cols = [f"score={h.score:.3f}"]
        if meta["w_dense"] > 0:
            cols.append(f"cos={h.dense:.3f}" if h.dense is not None else "cos=-")
            if meta["mode"] == "hybrid":
                cols[-1] += f"(#{h.dense_rank})" if h.dense_rank else "(#-)"
        if meta["w_lex"] > 0:
            cols.append(f"bm25={h.lex:.2f}" if h.lex is not None else "bm25=-")
            if meta["mode"] == "hybrid":
                cols[-1] += f"(#{h.lex_rank})" if h.lex_rank else "(#-)"
        if h.rerank is not None:
            cols.append(f"rerank={h.rerank:.3f}")
        extra = f"  (+{h.also} in thread)" if h.also else ""
        print(f"{rank}. {'  '.join(cols)}  {h.date:%Y-%m-%d}  {h.sender}{extra}")
        print(f"   subject: {h.subject}")
        if h.url:
            print(f"   gmail:   {h.url}")
        print(f"   matched: {h.label}")
        snippet = h.snippet or " ".join((h.text or "").split())
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        if snippet:
            print(f"   text:    {snippet}")
        if args.explain:
            print(_fmt_rrf(h, meta))
        print()


def _log_query(path: str, meta: dict, hits: list) -> None:
    """Append one JSONL line per search: the raw material for an eval set.

    Queries people actually typed are the only honest source of test cases, and
    they are unrecoverable unless something writes them down.
    """
    row = dict(meta)
    row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["results"] = [{"message_id": h.message_id, "score": round(h.score, 5),
                       "subject": h.subject[:120], "date": h.date.date().isoformat(),
                       "dense_rank": h.dense_rank, "bm25_rank": h.lex_rank}
                      for h in hits]
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("could not write %s: %s", path, e)


def cmd_reindex(args) -> None:
    """Rebuild chunk tiers and/or the BM25 index from the text already stored.

    The whole-source (canonical) chunk of every source holds the complete
    indexed text, so finer tiers can be re-derived without IMAP. Rebuilding
    the FTS index costs no embeddings at all.
    """
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    store = Store(cfg["db_path"], cfg)
    sizes = chunk_sizes_of(cfg)
    overlap = float(cfg.get("chunk_overlap_frac", DEFAULTS["chunk_overlap_frac"]))
    max_src = int(cfg.get("max_source_chars", DEFAULTS["max_source_chars"]))
    max_win = int(cfg.get("max_windows_per_source", DEFAULTS["max_windows_per_source"]))
    ladder = sorted(sizes, key=lambda k: sizes[k])

    # `--tiers ""` deliberately means "no text tiers": a pure --rebuild-fts run.
    tiers = ([t.strip() for t in args.tiers.split(",") if t.strip()]
             if args.tiers is not None else list(ladder))
    tiers = [t for t in tiers if t != MEDIA_TIER]
    unknown = [t for t in tiers if t not in sizes]
    if unknown:
        raise SystemExit(f"unknown tier(s) {unknown}; configured ladder: {ladder}")
    if not tiers and not args.rebuild_fts and not args.re_embed \
            and not getattr(args, "clean", False):
        raise SystemExit("nothing to do (no tiers selected, no --rebuild-fts, "
                         "no --re-embed, no --clean)")

    cleaned = {"changed": 0, "chars": 0, "ids": [], "scanned": 0}
    if getattr(args, "clean", False):
        # Fingerprints first: without them the duplicate suppression at index
        # and search time has nothing to compare, and this is the pass that can
        # write them without re-embedding a single vector.
        store.backfill_text_hash()
        cleaned = store.clean_stored_text(dry_run=args.dry_run)
        print(f"text hygiene:                {cleaned['changed']}/{cleaned['scanned']}"
              f" chunks cleaned, {cleaned['chars']:,} characters moved from the"
              f" vectors into quoted text"
              + (" [--dry-run: nothing written]" if args.dry_run else ""))
        subj = store.decode_stored_subjects(dry_run=args.dry_run)
        cleaned["ids"] = sorted(set(cleaned["ids"]) | set(subj["ids"]))
        print(f"subject repair:                {subj['changed']}/{subj['scanned']}"
              f" messages were stored as an RFC 2047 encoded-word; rewritten, "
              f"{len(subj['ids'])} chunks queued for re-embedding"
              + (" [--dry-run: nothing written]" if args.dry_run else ""))
        if args.dry_run:
            print("                             (chunk counts below are the"
                  " pre-cleaning numbers)")

    sources = store.canonical_sources(args.limit)
    plan: list[tuple[dict, list]] = []
    new_chunks = 0
    retier = 0
    for row in sources:
        want = []
        for r in tiered_chunks(row["text"], sizes, overlap, max_src, max_win):
            if r["canonical"] or r["tier"] not in tiers:
                continue          # the canonical row already holds this text
            want.append({
                "kind": row["kind"], "src": row["src"], "tier": r["tier"],
                "pos": r["pos"], "nwin": r["nwin"], "canonical": False,
                "label": chunk_label(row["src"], r["tier"], r["pos"], r["nwin"]),
                "text": r["text"],
                "title": title_of({"src": row["src"], "kind": row["kind"]},
                                  row["subject"] or ""),
            })
        # Every source is in the plan, even one that now derives no windows: a
        # letter whose text got shorter must also lose the windows the old, longer
        # text produced. Skipping those left 6 710 stale windows in a real index,
        # still searchable, still competing in the results, and no reindex would
        # ever remove them.
        plan.append((row, want))
        new_chunks += len(want)
        if canonical_tier(len((row["text"] or "").strip()), sizes)[0] != row["tier"]:
            retier += 1
    if tiers:
        ph = ",".join("?" * len(tiers))
        now_derived = store.conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE canonical = 0 AND tier IN ({ph})",
            list(tiers)).fetchone()[0]
    else:
        now_derived = 0
    # A canonical chunk that got shorter than the smallest window is not a
    # source any more, so it never reaches the loop above, and the windows its
    # longer text produced stay indexed forever: in FTS, with a vector, competing
    # for the top ten. A repair that shortens bodies (Phase 2 cleaning) makes
    # thousands of such messages, so they are swept explicitly here.
    stale_mids: list = []
    if tiers:
        planned = {row["message_id"] for row, _ in plan}
        stale_mids = [r[0] for r in store.conn.execute(
            f"SELECT DISTINCT message_id FROM chunks WHERE canonical = 0"
            f" AND tier IN ({ph})", list(tiers)).fetchall()
            if r[0] not in planned]
        if stale_mids:
            print(f"stale windows:               {len(stale_mids)} messages hold "
                  f"windows their current text no longer derives")
    batch = max(1, int(cfg["batch_size"]))
    re_rows, media_stale = [], 0
    if args.re_embed or cleaned["ids"]:
        # --re-embed re-reads every text chunk; --re-embed is implied by --clean
        # only for the chunks whose text actually changed, which is the whole
        # point of cleaning offline instead of re-fetching the mailbox.
        sel = ("SELECT c.id, c.text, c.src, c.kind, COALESCE(m.subject,'')"
               " FROM chunks c JOIN messages m ON m.message_id = c.message_id"
               " WHERE c.tier != ? AND c.text IS NOT NULL AND TRIM(c.text) != ''")
        # The tier replacement below re-embeds every window it writes, and the
        # windows it drops take their row ids with them. Re-embedding windows in
        # this pass would therefore be double work whose vectors land on rows
        # that no longer exist - that is where 22 830 orphan vectors came from.
        # Only the rows a rebuild never touches (the canonical ones) need a pass
        # here, and a rebuild that writes nothing needs the full list.
        if new_chunks:
            sel += " AND c.canonical = 1"
        if args.re_embed:
            rows = store.conn.execute(sel + " ORDER BY c.id", (MEDIA_TIER,)).fetchall()
        else:
            rows = []
            ids = sorted(cleaned["ids"])
            for i in range(0, len(ids), 900):        # bound the IN list
                part = ids[i:i + 900]
                rows += store.conn.execute(
                    sel + " AND c.id IN (%s) ORDER BY c.id" % ",".join("?" * len(part)),
                    [MEDIA_TIER] + part).fetchall()
        re_rows = [{"id": r[0], "text": r[1], "src": r[2], "kind": r[3],
                    "title": title_of({"src": r[2], "kind": r[3]}, r[4] or "")}
                   for r in rows]
    if args.re_embed:
        media_stale = store.conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN embeddings e ON e.chunk_id = c.id"
            " WHERE c.tier = ?", (MEDIA_TIER,)).fetchone()[0]
    total = new_chunks + len(re_rows)
    calls = -(-total // batch)  # ceil

    print(f"sources (whole-source chunks): {len(sources)}"
          + (" (limited)" if args.limit else ""))
    print(f"tiers to build:                {', '.join(tiers) or '-'} "
          f"(ladder {', '.join(f'{k}={sizes[k]}' for k in ladder)})")
    print(f"derived chunks:                {now_derived} -> {new_chunks}")
    print(f"embeddings to compute:         {total} "
          f"({new_chunks} new"
          + (f" + {len(re_rows)} re-embedded" if re_rows else "")
          + f", ~{calls} endpoint calls at batch {batch})")
    if args.re_embed:
        print(f"media chunks kept as-is:       {media_stale} (image/audio bytes are "
              f"not stored in the db; 'index --full' re-embeds them from IMAP)")
    print(f"canonical chunks to re-stamp:  {retier} (metadata only, no embedding)")
    print(f"fts docs now:                  {store.fts_count()} "
          f"(tokenizer {store.tokenizer!r})"
          + (" -> full rebuild" if args.rebuild_fts else ""))
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        note = truncation_note()
        if note:
            print(note)
        return
    if args.rebuild_fts and not cfg["embed_endpoint"]:
        log.info("no embed_endpoint configured: rebuilding the fts index only")

    client = EmbeddingClient(cfg) if cfg["embed_endpoint"] else None
    if (new_chunks or re_rows) and client is None:
        raise SystemExit("missing config: embed_endpoint (needed to embed chunks)")

    if not args.no_retier:
        n = store.retier_canonical()
        if n:
            print(f"re-stamped {n} canonical chunks to their length-implied tier")

    if plan or stale_mids:
        # First the messages the loop below cannot reach (their source no longer
        # derives a window, so nothing has to be embedded for them), then the
        # per-source replace of everything that still derives windows.
        removed = store.delete_derived(tiers, stale_mids) if stale_mids else 0
        inserted = 0
        texts = [c["text"] for _, chunks in plan for c in chunks]
        with tqdm.tqdm(total=len(texts), desc="re-chunking") as bar:
            for start in range(0, len(plan), 200):       # bounded memory, resumable
                group = plan[start:start + 200]
                mids = sorted({row["message_id"] for row, _ in group})
                removed += store.delete_derived(tiers, mids)
                items = [(row, c) for row, chunks in group for c in chunks]
                vecs = client.embed_texts([c["text"] for _, c in items],
                                          titles=[c.get("title", "") for _, c in items]) \
                    if items else []
                cur = store.conn.cursor()
                for (row, c), v in zip(items, vecs):
                    msg = Message(
                        message_id=row["message_id"], folder="", date=parse_date(row["date"]),
                        sender=row["sender"] or "", to=row["to_addrs"] or "", cc=row["cc"] or "",
                        subject=row["subject"] or "", body_text="",
                    )
                    smeta = chunk_search_meta(msg, row["attachment_exts"] or "",
                                              c["kind"], c["src"])
                    store.add_chunk(cur, row["message_id"], c, v, smeta)
                    inserted += 1
                store.conn.commit()
                bar.update(len(items))
        print(f"replaced {removed} derived chunks with {inserted} new ones")

    if re_rows:
        # Only text: image/audio bytes are not stored, so media vectors cannot
        # be rebuilt from the db. They are reported, not silently mixed in.
        for i in tqdm.tqdm(range(0, len(re_rows), batch), desc="re-embedding"):
            group = re_rows[i:i + batch]
            vecs = client.embed_texts([r["text"] for r in group],
                                      titles=[r["title"] for r in group])
            for r, v in zip(group, vecs):
                store.set_embedding(r["id"], v)
            store.conn.commit()
        print(f"re-embedded {len(re_rows)} stored text chunks")
        if media_stale:
            print(f"note: {media_stale} media chunks still hold vectors from the "
                  f"previous model (their bytes are not in the db) - "
                  f"'index --full' re-embeds those from IMAP")

    if args.rebuild_fts:
        n = store.rebuild_fts()
        print(f"rebuilt the fts index: {n} docs")

    if tiers:
        # The tier replacement above leaves the old windows' vectors behind.
        orphans = store.prune_orphan_vectors()
        if orphans:
            print(f"pruned {orphans} vectors of chunks this run replaced "
                  f"(~{orphans * int(cfg['dim']) * 4 / 1e6:.0f} MB of rows; the "
                  f"file itself only shrinks on VACUUM)")

    store.save_index_meta()
    s = store.stats()
    print(f"store now: {s['messages']} messages, {s['chunks']} chunks "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(s['by_tier'].items()))}), "
          f"{s['fts_docs']} fts docs")
    note = truncation_note()
    if note:
        print(note)


def cmd_shell(args) -> None:
    """Interactive search loop: type a query, get results, no re-loading.

    The 215 MB of vectors is loaded once and reused; a `search` per question
    pays that again, which is what makes trying things feel expensive.
    """
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    store = Store(cfg["db_path"], cfg, cache_index=True)
    store.warn_if_stale()
    client = EmbeddingClient(cfg) if cfg["embed_endpoint"] else None
    searcher = Searcher(cfg, store, client)
    opts: dict = {}
    print("mailbox shell - one query per line. "
          ":top 20 | :mode lexical | :tiers small | :explain on | :json on | "
          ":profile notes | :quit")
    while True:
        try:
            line = input("mbox> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, val = line[1:].partition(" ")
            val = val.strip()
            if cmd in ("quit", "q", "exit"):
                return
            if cmd in ("top", "mode", "tiers", "min_relevance", "rerank"):
                if not val:
                    opts.pop(cmd, None)
                    print(f"{cmd}: back to config default")
                    continue
                opts[cmd] = int(val) if cmd == "top" else val
                print(f"{cmd} = {opts[cmd]}")
            elif cmd in ("explain", "json", "collapse", "filters"):
                on = val.lower() in ("on", "yes", "1", "true", "")
                if cmd == "collapse":
                    opts["collapse"] = on
                elif cmd == "filters":
                    opts["filters"] = _NO_FILTERS if not on else None
                else:
                    opts[cmd] = on
                print(f"{cmd} = {on}")
            else:
                print("known: top mode tiers min_relevance rerank explain json "
                      "collapse filters quit")
            continue
        over = dict(opts)
        explain = over.pop("explain", False)
        as_json = over.pop("json", False)
        if "tiers" in over and isinstance(over["tiers"], str):
            over["tiers"] = [t.strip() for t in over["tiers"].split(",") if t.strip()]
        p = params_from_cfg(cfg, top=int(cfg.get("top", 10)), **over)
        hits, meta = searcher.run(line, p)
        if as_json:
            print(json.dumps({"meta": meta,
                              "results": [h.as_dict() for h in hits]},
                             ensure_ascii=False, indent=2))
            continue
        if meta.get("filters"):
            print(f"   filters: {meta['filters']}")
        if meta.get("abstained"):
            print(f"   nothing above the relevance floor "
                  f"(best {meta.get('relevance', 0):.3f})")
        for rank, h in enumerate(hits, 1):
            print(f"{rank:2}. {h.score:.3f}  {h.date:%Y-%m-%d}  {h.sender[:28]:28} "
                  f"{h.subject[:60]}")
            if explain:
                print(_fmt_rrf(h, meta))
            elif h.snippet:
                print(f"       {h.snippet[:150]}")


def cmd_stats(args) -> None:
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    store = Store(cfg["db_path"], cfg)
    store.warn_if_stale()
    s = store.stats()
    print(f"messages: {s['messages']}")
    print(f"chunks:   {s['chunks']}  ({s['derived_chunks']} of them tier windows)")
    for kind, n in s["by_kind"].items():
        print(f"  kind {kind}: {n}")
    for tier, n in sorted(s["by_tier"].items(), key=lambda kv: -kv[1]):
        avg = s["avg_len_by_tier"].get(tier, 0)
        print(f"  tier {tier}: {n} (avg {avg} chars)")
    print(f"fts docs: {s['fts_docs']}  (tokenizer: {s['fts_tokenizer']})")
    print(f"built with chunk_sizes: {s['chunk_sizes'] or 'unknown (pre-tier index)'}")
    print(f"vectors:  {s['vectors']} (dim {s['dim']}, "
          f"~{s['vector_bytes'] / 1e6:.0f} MB loaded per search)")
    last = store.get_state("last_indexed_date")
    print(f"last indexed: {last or 'never'}")
    rec = store.reconcile_state()
    print(f"removed from the index (Gmail no longer has them): {s['removed']}")
    # two different clocks on purpose: the last time the folders were checked,
    # and the last time that check actually took something out
    print(f"last reconciled: {rec['last_reconcile_at'] or 'never'}")
    if rec["removed_total"]:
        print(f"last dropped:    {rec['last_removal_at'] or 'unknown'}")
    for folder, res in (rec["last_reconcile"].get("folders") or {}).items():
        note = f" - {res['skipped']}" if res.get("skipped") else ""
        back = f", {res.get('restored', 0)} back" if res.get("restored") else ""
        miss = (f", {res['missing']} not indexed" if res.get("missing") else "")
        print(f"  {folder}: {res.get('live', 0)} live, {res.get('stored', 0)} stored, "
              f"{res.get('removed', 0)} removed{back}{miss}{note}")
    if s["removed"]:
        print("\nremoved, newest first:")
        rows = store.conn.execute(
            "SELECT removed_at, date, sender, subject FROM removed_messages"
            " ORDER BY removed_at DESC LIMIT 11").fetchall()
        for row in rows[:10]:
            print(f"  {row[0][:10]}  {(row[1] or '?')[:10]}  {row[2] or '?'}"
                  f"  —  {row[3] or '(no subject)'}")
        if len(rows) > 10:
            print(f"  ... and {s['removed'] - 10} more")


#: File types the preview sweep will ask Gmail for. `IMAGE_TYPES` is keyed on the
#: MIME type a part declares; the index kept only a file's *extension*, so this
#: is the list a stored row can actually be matched against. Anything outside it
#: (tiff, heic, ico) has no preview by design: Pillow may read it, but the GUI
#: would then be rendering a type a browser might not.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

#: Sidecar state keys. The watermark is the date of the oldest message the last
#: `media --backfill` reached, so an interrupted sweep says where it stopped
#: instead of being a number nobody can check.
MEDIA_WATERMARK = "media_watermark"
MEDIA_BACKFILL_AT = "media_backfilled_at"

#: How often a running `media --backfill` says where it is, in seconds. The
#: sweep's own pause comes every `reconcile_batch_size` messages — 500, which at
#: a MIME part per file is the better part of an hour — and a job that says
#: nothing for an hour while its numbers are still moving looks exactly like a
#: job that stopped, which is how one of these was killed mid-flight.
MEDIA_REPORT_SECONDS = 20


def _has_image_ext(exts: str) -> bool:
    """Whether a stored `attachment_exts` list names a file a preview can cut."""
    return any(("." + e.strip().lower()) in IMAGE_EXTS
               for e in str(exts or "").split(",") if e.strip())


def preview_names(store: "Store", message_id: str, exts: str) -> list:
    """Which of one indexed message's attachments should get a preview.

    Two halves joined by hand, because that is all the index has: `attachment_exts`
    says what *kind* of files a message carried (cheap, one column, good enough to
    skip 90 % of the mailbox) and the chunk labels say what they were *called*
    (needed to ask for one by name). A name whose extension Pillow and every
    browser both read is asked for; a PDF or a .txt has no preview by design.
    """
    if not _has_image_ext(exts):
        return []
    detail = store.message_detail(message_id) or {}
    return [n for n in detail.get("attachments") or ()
            if os.path.splitext(str(n))[1].lower() in IMAGE_EXTS]


def cmd_media(args) -> None:
    """The preview sidecar: what it holds, what it is missing, how to undo it.

    Previews are cut at index time, so a mailbox indexed before they were kept has
    thousands of pictures the reading pane cannot show until somebody reads those
    files again. `--backfill` does that — one login, one MIME part at a time,
    newest first — and `--stats` says how far it got. `--links` does the same
    favour for the other thing mail indexed early is missing: the Gmail web link,
    which only Gmail's own message number can give.

    None of it is precious. A preview is a copy of a file Gmail still has, so
    `--clear` costs disk space back and nothing else, and the sweep can cut the
    copies again whenever they are wanted.
    """
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    store = Store(cfg["db_path"], cfg)
    store.warn_if_stale()
    media = store.media_cache()
    if media is None:
        print(f"no previews: previews are off (store_thumbs) or "
              f"{media_db_path(cfg)} could not be opened")
        return
    if args.clear:
        if args.dry_run:
            n = media.count()
            print(f"would delete {n} preview(s) from {media.path}")
            print("dry run: nothing was removed. Re-run without --dry-run.")
            return
        print(f"cleared {media.clear()} preview(s) from {media.path}")
        print("the index kept every word; `media --backfill` cuts them again.")
        return
    if args.prune_orphans:
        _media_orphans(store, media, dry_run=args.dry_run)
        return
    if args.links:
        _media_links(cfg, store, args)
        return
    if args.backfill:
        _media_backfill(cfg, store, media, args)
    _media_stats(store, media)


def _media_stats(store, media) -> None:
    s = media.stats()
    print(f"sidecar: {s['path']}")
    print(f"previews: {s['rows']}  -  {s['bytes'] / 1e6:.1f} MB kept of "
          f"{s['source_bytes'] / 1e6:.0f} MB of originals "
          f"(budget {s['budget_bytes'] / 1e6:.0f} MB, "
          f"{s['file_bytes'] / 1e6:.1f} MB on disk)")
    print(f"sections remembered: {media.part_count()}")
    done = media.thumb_message_ids()
    missing = no_uid = 0
    for mid, exts, uid in store.conn.execute(
            "SELECT message_id, COALESCE(attachment_exts,''), COALESCE(uid, 0)"
            " FROM messages"):
        if mid in done or not _has_image_ext(exts):
            continue
        missing += 1
        if not uid:
            no_uid += 1
    print(f"messages with an image and no preview: {missing}")
    if missing:
        print("  -> python mailbox.py media --backfill")
        if no_uid:
            print(f"  ({no_uid} of them carry no UID: their preview comes with "
                  "the next index run)")
    mark = media.get_state(MEDIA_WATERMARK, "")
    if mark:
        print(f"last backfill reached {mark[:10]}"
              f" on {media.get_state(MEDIA_BACKFILL_AT, '')[:10]}")
    live = {r[0] for r in store.conn.execute("SELECT message_id FROM messages")}
    gone = [m for m in sorted(done) if m not in live]
    if gone:
        print(f"previews of messages no longer indexed: {len(gone)}"
              "  -> python mailbox.py media --prune-orphans")


def _media_orphans(store, media, dry_run: bool = False) -> None:
    """Drop previews whose message is not in the index any more.

    Deletion already mirrors itself — `Store.remove_messages` takes the previews
    with the row — so an orphan means a database was copied or restored without
    its sidecar. It costs disk and it shows the picture of mail nobody can find.
    """
    live = {r[0] for r in store.conn.execute("SELECT message_id FROM messages")}
    orphans = [m for m in sorted(media.thumb_message_ids()) if m not in live]
    if not orphans:
        print("no orphaned previews: every preview belongs to an indexed message")
        return
    rows = sum(1 for m in orphans for _ in media.for_message(m))
    if dry_run:
        print(f"would delete {rows} preview(s) of {len(orphans)} message(s) "
              "that are no longer indexed")
        print("dry run: nothing was removed. Re-run without --dry-run.")
        return
    print(f"dropped {media.delete_messages(orphans)} preview(s) of "
          f"{len(orphans)} message(s) that are no longer indexed")


def _media_links(cfg: dict, store, args) -> None:
    """Fill in the Gmail web links that mail was indexed without.

    A link is made from `X-GM-MSGID`, which indexing asks for in the fetch it was
    doing anyway — so mail indexed before that has no way to invent one afterwards.
    One batched FETCH per folder says it for the whole folder, and it transfers no
    message bodies at all: the number lives in the response's prefix. Rows that
    already have a link are not touched, so this is safe to run twice.
    """
    for key in ("gmail_user", "gmail_app_password"):
        if not cfg[key]:
            raise SystemExit(f"missing config: {key} (set it in {args.config})")
    want: dict = {}
    for folder, uid in store.conn.execute(
            "SELECT folder, uid FROM messages WHERE COALESCE(gmail_url,'') = ''"
            "   AND COALESCE(uid, 0) != 0"):
        want.setdefault(str(folder), []).append(int(uid))
    if not want:
        print("every indexed message with a UID already has its Gmail link")
        return
    asked = sum(len(v) for v in want.values())
    print(f"{asked} message(s) in {len(want)} folder(s) have no Gmail link")
    fetcher = GmailFetcher(cfg)
    session = GmailSession(fetcher)
    slot = gmail_index(cfg)
    filled = 0
    try:
        for folder in sorted(want):
            uids = want[folder]

            def ids(conn, folder=folder, uids=uids):
                return fetcher.gm_msgid_map(conn, folder, uids)

            got = session.call(folder, ids)
            n = 0
            for uid, gm in sorted(got.items()):
                url = gmail_message_url(gm, slot)
                if url:
                    n += store.conn.execute(
                        "UPDATE messages SET gmail_url = ? WHERE folder = ?"
                        "   AND uid = ? AND COALESCE(gmail_url,'') = ''",
                        (url, folder, uid)).rowcount
            store.conn.commit()
            filled += n
            print(f"  {folder}: Gmail answered for {len(got)} of {len(uids)}, "
                  f"{n} link(s) written")
    finally:
        session.close()
    quiet = asked - filled
    print(f"gmail links: {filled} written")
    if quiet:
        print(f"  ({quiet} message(s) Gmail gave no id for, or the folder was "
              "unreachable: their rows keep an empty link rather than a wrong one)")


def _media_backfill(cfg: dict, store, media, args) -> None:
    """Cut previews for images that were indexed before previews were kept.

    One login for the whole sweep (a preview used to cost a fresh IMAP handshake
    apiece), and one MIME part per file rather than one message: the reason a
    40 KB logo inside a 5 MB mail used to take a while is precisely the 5 MB, and
    reading the message to get the logo would repeat that mistake 4 000 times.
    """
    try:
        import PIL                                    # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is needed to cut previews (pip install Pillow)")
    for key in ("gmail_user", "gmail_app_password"):
        if not cfg[key]:
            raise SystemExit(f"missing config: {key} (set it in {args.config})")
    done = media.thumb_message_ids()
    rows = []
    no_uid = 0
    for mid, folder, uid, exts, date in store.conn.execute(
            "SELECT message_id, folder, COALESCE(uid, 0),"
            " COALESCE(attachment_exts,''), date FROM messages"
            " ORDER BY date DESC, rowid DESC"):
        if mid in done or not _has_image_ext(exts):
            continue
        if not uid:
            no_uid += 1            # only reading the whole message back would work
            continue
        rows.append((str(mid), str(folder), int(uid), str(exts), str(date or "")))
    print(f"{len(rows)} message(s) carry an image and have no preview yet")
    if no_uid:
        print(f"  ({no_uid} more carry no UID; the next index run gives them one)")
    if not rows:
        return
    limit = max(0, int(args.limit or 0))
    if args.dry_run:
        for mid, folder, uid, exts, date in rows[:min(20, limit or 20)]:
            print(f"  {(date or '?')[:10]}  {folder}#{uid}  "
                  f"{', '.join(preview_names(store, mid, exts)) or '(no name)'}")
        rest = len(rows) - min(20, limit or 20)
        if rest > 0:
            print(f"  ... and {rest} more")
        print("dry run: Gmail was not asked for anything and nothing was written")
        return
    fetcher = GmailFetcher(cfg)
    session = GmailSession(fetcher)
    batch = max(1, int(cfg.get("reconcile_batch_size",
                               DEFAULTS["reconcile_batch_size"])))
    # A preview never needs a mail's biggest files. Whatever the indexer itself
    # skipped (`max_attachment_mb`) it never embedded, so nothing about it is
    # waiting for a thumbnail either — and the structure states a part's *encoded*
    # size, about a third larger than the file, so the limit is widened to mean
    # here what it means at index time. `0` means the job will read anything.
    cap_bytes = int(float(cfg.get("max_attachment_mb",
                                  DEFAULTS["max_attachment_mb"]) or 0) * 4e6 // 3)
    made = missed = reached = 0
    started = time.time()

    def report() -> None:
        """Where the sweep got to, flushed: piped into a log it would otherwise
        sit in a buffer for a thousand messages."""
        print(f"  {reached}/{len(rows)} messages read, {made} previews, "
              f"{missed} file(s) Gmail would not hand over, "
              f"{(time.time() - started) / 60:.1f} min", flush=True)

    since = started
    try:
        for i, (mid, folder, uid, exts, date) in enumerate(rows):
            if limit and reached >= limit:
                print(f"stopped after {limit} message(s); the next run continues "
                      f"from {(date or '?')[:10]}")
                break
            reached += 1
            for name in preview_names(store, mid, exts):
                def read(conn, folder=folder, uid=uid, name=name):
                    return fetcher.fetch_attachment(conn, folder, uid, name,
                                                    parts=media,
                                                    max_bytes=cap_bytes)

                try:
                    att = session.call(folder, read)
                except FolderUnavailable as e:
                    print(f"  {folder}: {e}")
                    break
                except Exception as e:
                    log.debug("%s#%s %s: %s", folder, uid, name, e)
                    missed += 1
                    continue
                if att is None:
                    missed += 1        # the part is not there any more, or unreadable
                    continue
                if media.put_attachment(mid, att.name or name, att.data):
                    made += 1
                else:
                    missed += 1        # Pillow could not read it: the icon chip it is
            if reached % batch == 0:
                report()
                time.sleep(1.0)        # the same pause the reconcile sweep keeps
                media.set_state(MEDIA_WATERMARK, date)
                media.set_state(MEDIA_BACKFILL_AT, _now_iso())
                since = time.time()
            elif time.time() - since >= MEDIA_REPORT_SECONDS:
                # Between those pauses: a line saying the numbers moved, so the
                # person watching can tell slow from stuck.
                report()
                since = time.time()
    finally:
        session.close()
    media.set_state(MEDIA_WATERMARK, rows[min(reached, len(rows)) - 1][4]
                    if reached else "")
    media.set_state(MEDIA_BACKFILL_AT, _now_iso())
    dropped = media.prune_to_budget()
    s = media.stats()
    print(f"previews: {made} cut, {missed} file(s) Gmail would not hand over "
          f"(or could not be read)")
    if dropped:
        print(f"  {dropped} least-recently-opened preview(s) went over the "
              f"{media.budget_bytes / 1e6:.0f} MB budget")
    print(f"sidecar now: {s['rows']} previews, {s['bytes'] / 1e6:.1f} MB")


def cmd_gui(args) -> None:
    """Serve the mailbox UI in a browser.

    Imported here rather than at the top so the search core keeps no dependency
    on a web stack: `search`, `index` and the eval tools work on a machine that
    never installed fastapi.
    """
    try:
        from . import gui
    except ImportError as e:
        raise SystemExit("the gui needs fastapi and uvicorn:  "
                         "pip install fastapi uvicorn"
                         f"\n(import failed: {e})")
    cfg = _apply_db(args, load_config(args.config, getattr(args, "profile", "")))
    gui.run(args, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # A Windows console is often cp1252/cp850, and mail is full of emoji: one
    # undisplayable character must not lose you the whole result page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(
        description="Hybrid-searchable Gmail inbox (dense + BM25, multi-resolution chunks)")
    p.add_argument("--config", default="config.json")
    p.add_argument("--db", help="override db_path from the config "
                                "(e.g. a frozen copy to measure)")
    p.add_argument("--profile", default="",
                   help="knob set to start from: 'v2' (default, tuned) or "
                        "'legacy' (the pre-tuning behaviour, for A/B and rollback)")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="index mail (incremental by default)")
    pi.add_argument("--full", action="store_true",
                    help="index everything, ignoring last_indexed_date")
    pi.add_argument("--folders", help="comma-separated IMAP folders, e.g. INBOX,SENT")
    pi.add_argument("--limit", type=int,
                    help="test run: index the N most recent regardless of "
                         "last_indexed_date (and without updating it)")
    pi.add_argument("--debug", action="store_true",
                    help="log every IMAP byte sent/received (for diagnosing server errors)")
    pi.add_argument("--no-prune", action="store_true",
                    help="skip the after-index check for mail the folders no "
                         "longer hold (see `prune`); indexing is unaffected")
    pi.set_defaults(func=cmd_index)

    pp = sub.add_parser("prune", help="drop (or report) indexed mail that Gmail "
                                      "no longer has in the indexed folders")
    pp.add_argument("--dry-run", action="store_true",
                    help="list what would be removed and remove nothing")
    pp.add_argument("--folders", help="comma-separated IMAP folders to reconcile "
                                      "(default: the config's `folders`)")
    pp.add_argument("--prune-force", action="store_true",
                    help="allow a run that would remove more than "
                         "prune_max_fraction of one folder")
    pp.add_argument("--verbose", action="store_true",
                    help="name every message considered, not just the count")
    pp.set_defaults(func=cmd_prune)

    ps = sub.add_parser("search", help="hybrid (dense + bm25) search over the index")
    ps.add_argument("query")
    ps.add_argument("--mode", choices=["hybrid", "semantic", "lexical"],
                    help="search legs to use (default: config search_mode, else hybrid); "
                         "lexical needs no embedding endpoint")
    ps.add_argument("--top", type=int, default=10)
    ps.add_argument("--time-weight", action="store_true",
                    help="blend recency into the score (the old additive form; "
                         "the default recency prior is bounded and usually better)")
    ps.add_argument("--alpha", type=float, default=0.3,
                    help="recency weight when --time-weight is on (default 0.3)")
    ps.add_argument("--tiers", help="restrict to these chunk tiers, "
                                    "e.g. 'small' or 'large,xlarge' (default: all)")
    ps.add_argument("--pool", type=int, default=None,
                    help="bm25 candidate pool size (default: config lexical_pool)")
    ps.add_argument("--dense-pool", type=int, default=None,
                    help="keep only the N best cosine chunks before fusion (0 = all)")
    ps.add_argument("--w-dense", type=float, default=None,
                    help="RRF weight of the vector leg (default 0.5)")
    ps.add_argument("--w-lex", type=float, default=None,
                    help="RRF weight of the bm25 leg (default 0.5)")
    ps.add_argument("--rrf-k", type=float, default=None,
                    help="RRF damping constant (default 20)")
    ps.add_argument("--agg-k", type=int, default=None,
                    help="per-message score = mean of its top-k chunks (default 3)")
    ps.add_argument("--meta-weight", type=float, default=None,
                    help="bm25 weight of the metadata column (default 1.5)")
    ps.add_argument("--no-prefix", action="store_true",
                    help="disable the trailing-term prefix match (invoi -> invoice)")
    ps.add_argument("--fusion-pool", type=int, default=None,
                    help="messages each leg contributes to RRF (default 200; "
                         "0 = every message, which mixes two rank scales)")
    ps.add_argument("--no-tier-norm", action="store_true",
                    help="compare raw cosine across chunk tiers (default: standardise per tier)")
    ps.add_argument("--window-dedupe", type=float, default=None,
                    help="cosine above which two windows of one message count once "
                         "(default 0.92; 0 = off)")
    ps.add_argument("--no-collapse", action="store_true",
                    help="allow several messages of the same conversation in the top N")
    ps.add_argument("--collapse-sim", type=float, default=None,
                    help="cosine above which two results are the same content (default 0.97)")
    ps.add_argument("--min-relevance", type=float, default=None,
                    help="score floor below which the answer is 'nothing found' "
                         "(default 0.30; 0 = always return --top hits)")
    ps.add_argument("--half-life", type=float, default=None,
                    help="recency half-life in days (default 240; 0 = off)")
    ps.add_argument("--recency-weight", type=float, default=None,
                    help="how much of the score recency may move (default 0.35)")
    ps.add_argument("--demote-transactional", type=float, default=None,
                    help="multiplier for system senders like newsletters (default 0.55; 1 = off)")
    ps.add_argument("--rerank", choices=["off", "endpoint"], default=None,
                    help="cross-encode the shortlist via rerank_endpoint (off by "
                         "default: prove the gain with bench first)")
    ps.add_argument("--rerank-candidates", type=int, default=None,
                    help="how many top messages to rerank (default 40)")
    ps.add_argument("--no-filters", action="store_true",
                    help="do not read from:/van:/met bijlage/dates out of the query")
    ps.add_argument("--no-snippet", action="store_true",
                    help="print the raw matched chunk instead of a keyword window")
    ps.add_argument("--log", metavar="PATH",
                    help="append this query and its results to a JSONL file "
                         "(queries people typed are the honest eval set)")
    ps.add_argument("--explain", action="store_true",
                    help="show the per-hit leg ranks and the RRF arithmetic")
    ps.add_argument("--json", action="store_true", help="machine-readable output")
    ps.set_defaults(func=cmd_search)

    psh = sub.add_parser("shell", help="interactive search (loads the index once)")
    psh.set_defaults(func=cmd_shell)

    pr = sub.add_parser(
        "reindex", help="re-derive chunk tiers and/or rebuild the bm25 index "
                        "from the text already in the database (no IMAP)")
    pr.add_argument("--tiers", help="comma-separated tiers to rebuild "
                                    "(default: every configured tier except media)")
    pr.add_argument("--rebuild-fts", action="store_true",
                    help="re-index all chunk text into FTS (costs no embeddings; "
                         "needed after changing fts_tokenizer)")
    pr.add_argument("--clean", action="store_true",
                    help="re-run text hygiene over the stored text (quotes and "
                         "footers out of the vectors, into quoted text) and "
                         "re-embed only the chunks whose text changed")
    pr.add_argument("--re-embed", action="store_true",
                    help="re-embed every stored text chunk from the text already "
                         "in the database - what a change of embedding model "
                         "needs; media chunks keep their old vectors")
    pr.add_argument("--no-retier", action="store_true",
                    help="skip re-stamping canonical chunks with their length-implied tier")
    pr.add_argument("--limit", type=int, help="only re-derive from N sources (try it first)")
    pr.add_argument("--dry-run", action="store_true",
                    help="report what would change (and how many embeddings it costs) "
                         "and write nothing")
    pr.set_defaults(func=cmd_reindex)

    pt = sub.add_parser("stats", help="show index statistics")
    pt.set_defaults(func=cmd_stats)

    pm = sub.add_parser("media", help="local previews of attachments: what is "
                                      "kept, what is missing, how to undo it")
    pm.add_argument("--stats", action="store_true",
                    help="what the sidecar holds and what it still lacks "
                         "(what runs when you ask for nothing)")
    pm.add_argument("--backfill", action="store_true",
                    help="read the indexed images that have no preview yet and "
                         "cut one: one login, one MIME part per file, newest first")
    pm.add_argument("--limit", type=int, default=0,
                    help="stop after N messages (--backfill). An interrupted "
                         "sweep is not lost: previews already cut are kept")
    pm.add_argument("--dry-run", action="store_true",
                    help="report what would be read or removed and do neither")
    pm.add_argument("--links", action="store_true",
                    help="write the Gmail web links of mail indexed before "
                         "Gmail's message id was asked for (batched, no bodies)")
    pm.add_argument("--prune-orphans", action="store_true",
                    help="drop previews of messages that are no longer indexed")
    pm.add_argument("--clear", action="store_true",
                    help="delete every stored preview. The index keeps every "
                         "word; only the pictures go")
    pm.set_defaults(func=cmd_media)

    pg = sub.add_parser("gui", help="mailbox-style search UI in the browser "
                                    "(needs fastapi + uvicorn)")
    pg.add_argument("--port", type=int, default=8765)
    pg.add_argument("--host", default="127.0.0.1",
                    help="bind address. The default is loopback only: this "
                         "serves your whole mailbox with no login, so binding "
                         "to 0.0.0.0 hands it to your network")
    pg.add_argument("--top", type=int, help="results per search (default: config)")
    pg.add_argument("--no-browser", action="store_true",
                    help="print the URL instead of opening a browser tab")
    pg.set_defaults(func=cmd_gui)

    try:                                    # bench / evalgen / label / audit
        from . import eval as _eval_cli
    except ImportError:
        _eval_cli = None
    if _eval_cli is not None:
        _eval_cli.add_subcommands(sub)
    else:
        log.debug("eval.py not importable: bench/evalgen/label/audit unavailable")

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
