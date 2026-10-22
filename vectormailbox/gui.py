"""Mailbox-style web UI over the hybrid index: search, read, browse.

`python gui.py` (or `python mailbox.py gui`) starts a local server and opens your
browser. Everything is read-only against `mailbox.db` except the Refresh button,
which runs `mailbox.py index` as a *subprocess* — this process never writes the
index, so a UI bug cannot damage an index that took an afternoon to build.

Three things shape the design, and all three come from the engine underneath:

**The index is loaded once, not per query.** One `Store(cache_index=True)` holds
~146 MB of vectors for a 47 720-chunk mailbox. Re-reading it per keystroke would
be the whole latency budget, so the `Engine` owns one snapshot and answers query
after query from it, exactly like `mailbox.py shell`.

**One worker thread for the index, on purpose.** Every database and search call
runs on a single executor thread. That keeps one SQLite connection on the thread
that created it (no `check_same_thread` fudging), and it serialises searches so
two browser tabs cannot each pull a 146 MB matrix into RAM at once. Requests
queue; `/api/status` says so, and the front-end debounces and discards stale
answers.

**Reading a file is not a database read, so it does not share that thread.** A
click on an attachment used to occupy the one worker that answers searches for as
long as Gmail took to hand the message over. Previews and file bytes now run on
a second, two-worker *media pool*: a search cannot be stuck behind a fetch, a
fetch cannot be stuck behind a search, and `queued` — which counts searches —
stays honest. The sidecar the previews live in is opened per thread, so this
costs nothing the index does not already pay.

**A dead embedding endpoint must not hang or lie.** `EmbeddingClient._post`
retries three times with 2/4/8 s sleeps before raising, which is ~14 s of
nothing. So the engine probes once at startup with a short timeout and, while the
endpoint is down, serves keyword-only results and says that is what it is doing —
rather than showing a spinner and then an error for a query that BM25 can answer.
That trade only exists for words: a query **image** is readable by the vector leg
alone, so with the endpoint down (or an endpoint that rejects image input) an
image query is refused with a sentence instead of quietly answered by BM25.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import hashlib
import json
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import mailbox
from .mailbox import (DEFAULTS, EmbeddingClient, GmailFetcher, GmailSession,
                     HTML_CIDS, HTML_PART, MediaCache, QUERY_IMAGE_TYPES,
                     Searcher, Store, gmail_index, load_config,
                     open_media_cache, pick_attachment, pick_html_part)

try:
    from . import queryparse
except ImportError:                                  # filters then stay unparsed
    queryparse = None

try:
    from . import mailhtml
except ImportError as e:
    # Without the sanitizer there is no safe way to show a mail's HTML, but there
    # is still a perfectly good text pane — so this is one feature turned off
    # with a reason, not a server that refuses to start.
    mailhtml = None
    MAILHTML_WHY = str(e)

log = logging.getLogger("gui")

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
REFRESH_LOG = ROOT / "gui-index.log"

#: Re-probe a downed endpoint at most this often, so a dead llama.cpp is one
#: slow request per minute instead of one per query.
REPROBE_SECONDS = 60.0
#: A health probe must be quick or it is not a health probe.
PROBE_TIMEOUT = 5.0
#: `run()` refuses to answer past this many queued search requests: a browser
#: holding down the enter key should not build an unbounded backlog.
MAX_QUEUED = 8
#: Previews and attachment bytes get their own two workers, so that neither
#: queues behind a search nor a search behind a Gmail fetch. One worker would
#: serialise two images opening together; more would be IMAP connections nobody
#: asked for — the shared login inside `GmailSession` is the real limit anyway.
MEDIA_WORKERS = 2
#: A preview is a re-encode of bytes this index already saw, keyed to the message
#: it came from, so it cannot change under the same URL: cache it for a week and
#: let the browser re-ask rather than re-transfer.
THUMB_CACHE = "private, max-age=604800, immutable"
#: A full file belongs to Gmail and could be edited at the source, so this is
#: short: long enough that reopening a preview costs nothing, short enough that
#: a corrected invoice shows up today rather than next week.
FILE_CACHE = "private, max-age=3600"

DEFAULT_PORT = 8765
MODES = ("hybrid", "semantic", "lexical")


class EngineError(Exception):
    """A search could not be answered, with a message safe to show a user."""


class NothingToRender(EngineError):
    """This message has no HTML to show, or showing it was refused.

    A different answer from "Gmail is not answering", and the reading pane shows
    the two differently: one is a property of the mail ("this one is plain text,
    so text is what you get"), the other is a property of the connection and is
    worth retrying after a Refresh.

    `reason` carries that difference across the wire as a word instead of forcing
    the page to recognize a sentence: a UI that greps error text to decide what a
    button may do breaks the first time the sentence is improved.
    """

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason


#: A search image travels from the browser as base64 in a JSON body. The cap is
#: on the decoded bytes: `resize_image_b64` re-encodes whatever arrives as a PNG
#: of at most 1024 px a side, so a 20 MB phone photo is a few hundred thousand
#: characters by the time the model sees it. The cap is not there to protect the
#: endpoint; it is there so a search box cannot be used to post a disk image.
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def sniff_image(data: bytes) -> str:
    """The type these bytes really are, from their magic, or '' if unsupported.

    What a browser reports for a dropped file is a guess it made from the
    extension; the bytes are the fact, and the bytes are what goes to the
    embedding endpoint. Naming the type here turns "a PDF in an image field"
    into a sentence about the file instead of an opaque HTTP 400 downstream.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def decode_query_image(raw, name: str = "") -> tuple[bytes, str, str]:
    """The `image` field of a search body -> `(bytes, mime, name)`.

    Two forms are accepted: a `data:<mime>;base64,…` URL, which is what a
    browser's FileReader hands out, and bare base64. Everything is checked here,
    because this is the one place the GUI takes a block of user bytes and hands
    it to another process.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise EngineError("the image is empty")
    text = raw.strip()
    claimed = ""
    if text[:5].lower() == "data:":
        head, _, text = text.partition(",")
        claimed = head[5:].split(";")[0].strip().lower()
    try:
        data = base64.b64decode(text)
    except Exception:
        raise EngineError("the image is not valid base64") from None
    if not data:
        raise EngineError("the image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise EngineError(
            f"the image is {len(data) / 1048576:.1f} MB and a search image is "
            f"capped at {MAX_IMAGE_BYTES // 1048576} MB — it is resized to 1024 px "
            "on the way to the model, so a smaller copy finds the same mail")
    mime = sniff_image(data)
    if not mime or mime not in QUERY_IMAGE_TYPES:
        raise EngineError(
            "that is not a picture this index can search by"
            + (f" (the file says it is {claimed})" if claimed else "")
            + f" — PNG, JPEG, WebP or GIF up to {MAX_IMAGE_BYTES // 1048576} MB")
    return data, mime, (str(name or "").strip()[:120] or "the attached image")


class Engine:
    """One warm search engine behind one worker thread.

    The `Store`, its SQLite connection and the cached `IndexArrays` are created
    *on* the executor thread and only ever touched from it, which is why no
    sqlite3 thread-affinity flag is set anywhere: there is nothing to forgive.
    """

    def __init__(self, cfg: dict, top: Optional[int] = None):
        self.cfg = dict(cfg)
        self.top = int(top) if top else int(cfg.get("top", 10) or 10)
        self.db_path = str(cfg.get("db_path", "mailbox.db"))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mbox")
        #: previews and file bytes, away from the thread that owns the index
        self._media_pool = ThreadPoolExecutor(max_workers=MEDIA_WORKERS,
                                              thread_name_prefix="mbox-media")
        #: Local previews of indexed images, in the sidecar beside the index —
        #: None when `store_thumbs` is off or the file could not be opened. Opened
        #: here, on the main thread, but never *used* here: `MediaCache` hands
        #: every thread its own connection, so the media workers and the index
        #: worker each read it without sharing a sqlite3 handle.
        self.media: Optional[MediaCache] = open_media_cache(self.cfg)
        #: One IMAP login for every click, reused until it stops working.
        #: Connecting is lazy: nothing here talks to Gmail until a file does.
        self.imap = GmailSession(GmailFetcher(self.cfg))
        self._queue = 0
        self._queue_lock = threading.Lock()
        #: thread id of the single worker, recorded the first time it runs work
        self.worker_ident = 0
        self.store: Optional[Store] = None
        self.searcher: Optional[Searcher] = None
        self.client: Optional[EmbeddingClient] = None
        self.ready = False
        self.loading = False
        self.open_error = ""
        # Endpoint health is a cached judgement, not a per-query experiment: the
        # cost of finding out the hard way is 14 seconds of retrying.
        self.endpoint_ok: Optional[bool] = None
        self.endpoint_error = ""
        self.endpoint_checked = 0.0
        self.refresh: dict = {"running": False, "started_at": None,
                              "finished_at": None, "exit_code": None,
                              "log": "", "cmd": ""}
        self._refresh_lock = threading.Lock()

    # -- threading ----------------------------------------------------------
    def submit(self, fn: Callable, *a, **kw):
        """Run `fn` on the engine thread. Returns a concurrent Future.

        Never call this from a function already running on the engine thread:
        with one worker, the outer task holds the only thread while waiting for
        the inner one, and the request hangs forever. On-thread code calls
        `_open`/`rebuild` directly.
        """
        if self.worker_ident and threading.get_ident() == self.worker_ident:
            raise RuntimeError(
                f"submit() from the engine thread would deadlock the single "
                f"worker; call {getattr(fn, '__name__', fn)}() directly instead")
        return self._pool.submit(self._queued, fn, a, kw)

    async def acall(self, fn: Callable, *a, **kw):
        """Await `fn` on the engine thread, from an `async def` route."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._queued, fn, a, kw)

    async def amedia(self, fn: Callable, *a):
        """Await `fn` on a media worker: previews and file bytes, not index work.

        Not counted in `queued`, which reports searches. A picture opening in the
        background must not make the page say "3 searches waiting", and a search
        backlog must not delay the preview that is already computed.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._media_pool,
                                          functools.partial(fn, *a))

    def _queued(self, fn: Callable, a: tuple, kw: dict):
        self.worker_ident = threading.get_ident()
        with self._queue_lock:
            self._queue += 1
            if self._queue > MAX_QUEUED:
                self._queue -= 1
                raise EngineError("too many searches waiting; try again in a moment")
        try:
            return fn(*a, **kw)
        finally:
            with self._queue_lock:
                self._queue -= 1

    @property
    def queued(self) -> int:
        return self._queue

    # -- lifecycle ----------------------------------------------------------
    def open(self, timeout: float = 600.0) -> dict:
        """Open the store and warm the index *on the engine thread*, and wait.

        This is the only opening entry point, and it is a method rather than a
        call to `_open` for one reason: a sqlite3 connection belongs to the
        thread that created it. Creating it anywhere else gives you a perfectly
        healthy open and a `ProgrammingError` on the first request that reads —
        which is precisely what the first run of the test suite did.
        """
        return self.submit(self._open).result(timeout=timeout)

    def _open(self) -> dict:
        """Runs on the engine thread."""
        self.loading = True
        self.open_error = ""
        try:
            self._teardown()
            self.store = Store(self.db_path, self.cfg, cache_index=True)
            self.store.warn_if_stale()
            if self.cfg.get("embed_endpoint"):
                self.client = EmbeddingClient(self.cfg)
            self.searcher = Searcher(self.cfg, self.store, self.client)
            # Warm the snapshot now so the first human query is not also the
            # slowest one: this is the 146 MB read, paid once at startup.
            self.store.load_index(with_vectors=self.client is not None)
            self.ready = True
            self._probe_endpoint(force=True)
        except SystemExit as e:            # mailbox.py signals with SystemExit
            self.open_error = str(e) or "the index could not be opened"
            log.warning("engine open failed: %s", self.open_error)
        except Exception as e:
            self.open_error = f"{type(e).__name__}: {e}"
            log.exception("engine open failed")
        finally:
            self.loading = False
        return self.status_sync()

    def rebuild(self) -> dict:
        """Re-open after an index run: the snapshot must not outlive the data.

        On-thread like `_open`; the refresh watcher wraps this in a `submit`.
        """
        self.ready = False
        return self._open()

    def _teardown(self) -> None:
        if self.store is not None:
            try:
                self.store.conn.close()
            except Exception:
                pass
        self.store = self.searcher = self.client = None

    def close(self, timeout: float = 30.0) -> None:
        """Close on the engine thread too — same affinity rule as opening."""
        try:
            self.submit(self._teardown).result(timeout=timeout)
        except Exception:
            pass
        self._pool.shutdown(wait=False)
        # The login is the one resource here Gmail keeps score of, so it is
        # logged out instead of left to a socket timeout. Not waited for: Ctrl-C
        # should stop the server, and a fetch still in flight has nowhere to
        # report an error to at this point anyway.
        self._media_pool.shutdown(wait=False)
        self.imap.close()

    # -- endpoint health ----------------------------------------------------
    def _probe(self) -> bool:
        """Ask the endpoint for one 768-dim vector, quickly."""
        if self.client is None:
            self.endpoint_error = "no embed_endpoint configured"
            return False
        probe = EmbeddingClient({**self.cfg, "embed_timeout": PROBE_TIMEOUT})
        try:
            vec = probe.embed_query("health check")
            ok = bool(getattr(vec, "size", 0))
            self.endpoint_error = "" if ok else "empty vector"
            return ok
        except Exception as e:
            self.endpoint_error = f"{type(e).__name__}: {e}"
            return False

    def _probe_endpoint(self, force: bool = False) -> Optional[bool]:
        now = time.time()
        if not force and self.endpoint_ok and (now - self.endpoint_checked) < 5:
            return True
        if (not force and self.endpoint_ok is False
                and (now - self.endpoint_checked) < REPROBE_SECONDS):
            return False
        self.endpoint_ok = self._probe()
        self.endpoint_checked = now
        if not self.endpoint_ok:
            log.warning("embedding endpoint unavailable: %s", self.endpoint_error)
        return self.endpoint_ok

    # -- search -------------------------------------------------------------
    def _overrides(self, body: dict, mode: str) -> dict:
        over: dict = {"mode": mode, "top": self.top}
        # Every knob the CLI has, so the UI never silently searches with a
        # different ranking than `search` would.
        for key, cast in (("top", int), ("pool", int), ("dense_pool", int),
                          ("rrf_k", float), ("agg_k", int), ("w_dense", float),
                          ("w_lex", float), ("fusion_pool", int),
                          ("window_dedupe", float), ("collapse_sim", float),
                          ("recency_half_life_days", float),
                          ("recency_weight", float),
                          ("demote_transactional", float),
                          ("min_relevance", float), ("rerank_candidates", int)):
            if body.get(key) is not None:
                over[key] = cast(body[key])
        for key in ("collapse", "tier_norm", "snippet", "explain", "time_weight",
                    "prefix"):
            if body.get(key) is not None:
                over[key] = bool(body[key])
        if body.get("tiers"):
            over["tiers"] = [str(t).strip() for t in str(body["tiers"]).split(",")
                             if str(t).strip()]
        if body.get("rerank") in ("off", "endpoint"):
            over["rerank"] = body["rerank"]
        return over

    def _explicit_filters(self, raw: dict):
        """Rebuild a `queryparse.Filters` from the chips the user left standing.

        The client sends back what `meta.filters_struct` gave it, minus the chip
        it removed. Passing these as `SearchParams.filters` makes them decisive
        (`run()` does not second-guess a filter a human chose), which is the
        documented difference between a guessed filter and a typed one.
        """
        if queryparse is None or not isinstance(raw, dict):
            return None
        f = queryparse.Filters()
        for key in ("senders", "recipients", "subjects"):
            val = raw.get(key)
            if isinstance(val, (list, tuple)):
                setattr(f, key, [str(v) for v in val])
        for key in ("since", "until"):
            if raw.get(key):
                setattr(f, key, str(raw[key]))
        if raw.get("has_attachment") is not None:
            f.has_attachment = bool(raw["has_attachment"])
        return None if f.is_empty() else f

    def search(self, body: dict) -> dict:
        """One search, with the engine's own reasoning attached."""
        if self.open_error:
            raise EngineError(self.open_error)
        if self.store is None:
            raise EngineError("the index is still loading")
        query = str(body.get("q") or "").strip()
        raw_image = body.get("image")
        image_bytes: Optional[bytes] = None
        image_mime = ""
        image_name = ""
        if raw_image not in (None, "", {}):
            image_bytes, image_mime, image_name = decode_query_image(
                raw_image, str(body.get("image_name") or ""))
        if not query and not image_bytes:
            raise EngineError("no query")

        want = str(body.get("mode") or self.cfg.get("search_mode")
                   or DEFAULTS["search_mode"]).strip().lower()
        if want not in MODES:
            raise EngineError(f"unknown mode {want!r}")
        degraded = ""
        qimg = None
        if image_bytes is not None:
            # A picture is embedded before anything else, because exactly one
            # leg can read it. Downgrading to keywords here would answer a
            # question about a photo with mail that shares no feature with it,
            # and hand it over as a ranked list — so an endpoint that is down,
            # absent, or unable to embed pixels is refused in a sentence.
            if want == "lexical":
                raise EngineError(
                    "a picture is found by the vector leg only — "
                    "use hybrid or semantic")
            if self.client is None:
                raise EngineError(
                    "no embed endpoint is configured, and only the vector leg "
                    "can read a picture — search by words instead")
            if not self._probe_endpoint():
                raise EngineError(
                    f"the embedding endpoint is unreachable "
                    f"({self.endpoint_error or 'unavailable'}) — "
                    "search by words until it answers")
            try:
                qimg = self.client.embed_query_image(image_bytes, image_mime,
                                                     image_name)
            except SystemExit as e:
                # mailbox.py signals "this endpoint cannot embed images" with
                # the sentence that explains it, which is the user's answer.
                raise EngineError(str(e) or "this endpoint cannot embed images") \
                    from None
        elif want != "lexical" and not self._probe_endpoint():
            # Keyword-only is a real answer, not a failure: say which it is.
            want, degraded = "lexical", (
                "embedding endpoint unreachable, so these results come from "
                "keyword search only: " + (self.endpoint_error or "unavailable"))

        over = self._overrides(body, want)
        chips = body.get("filters")
        text = query
        if isinstance(chips, dict):
            # The chips the user left standing are the authority, so this search
            # runs on exactly the text they sent with exactly those filters, and
            # the parser is switched off for it. Without `parse_filters: False`
            # the sentence is re-read and a dropped chip silently comes back:
            # `apply_filters` would still cut `van flexa` out of the embedded
            # text, so removing the sender filter would lose the word's effect
            # without gaining the restriction's removal.
            searcher = Searcher({**self.cfg, "parse_filters": False},
                                self.store, self.client)
            text = str(body.get("text") or query).strip() or query
            p = searcher.params(**over)
            p.filters = self._explicit_filters(chips)
        else:
            searcher = self.searcher
            p = searcher.params(**over)
        started = time.perf_counter()
        try:
            hits, meta = searcher.run(text, p, image=qimg, image_name=image_name)
        except SystemExit as e:
            # run() raises SystemExit for an empty index, no vectors, a missing
            # endpoint and both legs weighted 0. In a server that is a 409 with
            # a sentence in it, never a dead process.
            raise EngineError(str(e) or "the search could not run") from None
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"{type(e).__name__}: {e}") from None
        took = (time.perf_counter() - started) * 1000.0

        results = []
        for rank, h in enumerate(hits, 1):
            row = h.as_dict()
            row["rank"] = rank
            results.append(row)
        # A search list is where a picture is most worth seeing: the row that
        # matched on a screenshot's text can show the screenshot.
        self._add_thumbs(results)
        out = {"query": query, "took_ms": round(took, 1), "meta": meta,
               "results": results}
        if degraded:
            out["degraded"] = degraded
        return out

    # -- browsing -----------------------------------------------------------
    def _browse_filters(self, sender: str, since: str, until: str,
                        has_attachment: Optional[bool]):
        """Hand-picked browse filters, in the same shape search uses."""
        if queryparse is None:
            return None
        f = queryparse.Filters()
        if sender:
            f.senders = [sender]
        if since:
            f.since = since
        if until:
            f.until = until
        if has_attachment is not None:
            f.has_attachment = has_attachment
        return None if f.is_empty() else f

    def messages(self, before: str = "", limit: int = 50, sender: str = "",
                 since: str = "", until: str = "", text: str = "",
                 has_attachment: Optional[bool] = None) -> dict:
        if self.store is None:
            raise EngineError("the index is still loading")
        limit = max(1, min(200, int(limit)))
        f = self._browse_filters(sender, since, until, has_attachment)
        items = self.store.page_messages(before=before, limit=limit + 1,
                                         filters=f, text=text or None)
        more = len(items) > limit
        items = items[:limit]
        # One query for the page's previews, not one per row: a list of fifty
        # rows asking which of its attachments has a picture is the difference
        # between a scroll that feels instant and fifty round trips to the
        # sidecar. `first_for_messages` answers for every listed message at once.
        self._add_thumbs(items)
        return {"items": items,
                "next_cursor": items[-1]["date"] if (more and items) else "",
                "total": self.store.count_messages(filters=f, text=text or None)}

    def _add_thumbs(self, rows: list) -> None:
        """Give each row a `thumb`: the filename of its first local preview.

        Shared by the list and the search list, because both are pages of rows a
        person scrolls. `""` means "this row has no picture" rather than a missing
        key, so the front end draws the extension chip it always drew instead of
        guessing at an absent field. A sidecar that cannot be read costs the
        thumbnails and nothing else: a list of mail is still a list of mail.
        """
        if self.media is None or not rows:
            return
        try:
            thumbs = self.media.first_for_messages(
                [r.get("message_id") for r in rows])
        except Exception:
            log.exception("previews could not be listed for a page of rows")
            return
        for r in rows:
            r["thumb"] = thumbs.get(r.get("message_id"), "")

    def message(self, message_id: str) -> dict:
        if self.store is None:
            raise EngineError("the index is still loading")
        detail = self.store.message_detail(message_id)
        if detail is None:
            raise EngineError(self._missing_message(message_id))
        detail["thread"] = self.store.thread_messages(detail.get("thread_id") or "")
        # Which of this message's files have a local preview, and how big it is:
        # the reading pane draws a chip with a picture in it instead of one with
        # an extension in it, and sizes the box from the real aspect ratio.
        detail["media"] = (self.media.for_message(detail.get("message_id")
                                                  or message_id)
                           if self.media is not None else {})
        return detail

    def _missing_message(self, message_id: str) -> str:
        """Why an id the URL named is not here, in the words that fit.

        Two different situations, which the one old message conflated: a link
        to mail that was never indexed, and a link to mail this index dropped
        because the folder stopped holding it. For the second there is a real
        answer to give - when it went, and that this app did not delete it,
        because this app cannot delete anything.
        """
        try:
            gone = self.store.tombstone(message_id)
        except Exception:                       # a missing row stays a missing row
            gone = None
        if not gone:
            return f"no message {message_id!r} in the index"
        when = (gone.get("removed_at") or "")[:10] or "an earlier run"
        where = f" {gone['folder']}" if gone.get("folder") else ""
        return (f"removed from the index on {when}: the folder{where} no longer "
                "holds this message, so the last refresh dropped it. Nothing in "
                "this app deletes mail — look in Gmail.")

    # -- attachments ---------------------------------------------------------
    #: Types a browser may open in its own tab from this origin. SVG and HTML
    #: are deliberately *not* on it: they execute, and an attachment is
    #: attacker-authored text served from the same origin as the app itself.
    INLINE_TYPES = ("image/", "audio/", "video/", "application/pdf", "text/plain")

    def attachment_ref(self, message_id: str) -> dict:
        """Where a message's files live at Gmail: `{message_id, folder, uid}`.

        The index keeps text, not files, so opening a file means going back to
        the folder and UID its row recorded. That lookup is an index read, so it
        runs on the engine thread — and it is the *only* part of a click that
        does. Handing the folder and UID straight back to the route is what lets
        `read_attachment` do the slow part somewhere else: the worker that owns
        the SQLite connection answers in milliseconds and goes back to searches,
        instead of sitting idle for the seconds a 5 MB message takes to arrive.
        """
        if self.store is None:
            raise EngineError("the index is still loading")
        if not (self.cfg.get("gmail_user") and self.cfg.get("gmail_app_password")):
            raise EngineError(
                "opening attachments needs gmail_user and gmail_app_password in "
                "config.json (or GMAIL_USER / GMAIL_APP_PASSWORD in the "
                "environment); browsing, searching and their previews do not")
        ref = self.store.message_ref(message_id)
        if ref is None:
            raise EngineError(self._missing_message(message_id))
        folder, uid = ref
        if not folder:
            raise EngineError("this row records no IMAP folder, so its files "
                              "cannot be traced back to Gmail")
        return {"message_id": message_id, "folder": folder, "uid": uid}

    def read_attachment(self, ref: dict, filename: str) -> tuple:
        """One file's bytes, fetched from Gmail on a media worker.

        `EXAMINE` and `BODY.PEEK`, nothing outside `READ_ONLY_COMMANDS`, over one
        login this process keeps open — so opening a PDF cannot even mark the
        mail read, and does not pay for a fresh TLS handshake and login to do it.

        Two ways to get the bytes, and only one promise about them. The fast one
        reads the single MIME part that carries `filename`; the other reads the
        whole message and picks the part out of it. The fast one is taken only
        when the part it read *says* it is that file — the same filename rule
        `pick_attachment` applies — and any doubt at all falls back to the whole
        message. Returns `(bytes, content_type, filename)`.

        The bytes are handed to the requester and stored nowhere here: the index
        keeps holding text only, and the browser's own cache is what makes the
        second open free. A picture's preview is a separate route and a separate
        thing — a re-encode of this file, not this file.
        """
        folder = str(ref.get("folder") or "")
        uid = int(ref.get("uid") or 0)
        mid = str(ref.get("message_id") or "")
        fetcher = self.imap.fetcher

        def read(conn):
            att = None
            if uid:
                att = fetcher.fetch_attachment(conn, folder, uid, filename,
                                               parts=self.media)
            if att is not None:
                return att
            raw = (fetcher.fetch_message(conn, folder, uid) if uid else
                   # Indexed before UIDs were recorded: the Message-ID header is
                   # the identity that still finds the message in the folder.
                   fetcher.fetch_by_message_id(conn, folder, mid))
            if raw is None:
                raise EngineError(
                    f"Gmail no longer answers for this message in {folder} (uid "
                    f"{uid or 'unknown'}): it may have moved or been renumbered "
                    "there. A Refresh re-reads the folders")
            return pick_attachment(raw, filename)

        try:
            att = self.imap.call(folder, read)
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Gmail could not be reached for this file: "
                              f"{type(e).__name__}: {e}") from None
        if att is None:
            raise EngineError(
                f"no file named {filename!r} on that message in {folder}; it "
                "may have been renamed at the source")
        return att.data, att.content_type or "application/octet-stream", att.name

    # -- reading the mail as mail --------------------------------------------

    def render_html(self, ref: dict, images: Optional[bool] = None) -> dict:
        """One message's own HTML part, sanitized and wrapped for a sandboxed frame.

        What makes this different from opening a file: the bytes are the *mail*,
        so they go through `mailhtml` — a sanitizer with an email profile — before
        anything sees them, and they arrive in a frame with no script permission.
        Nothing here fetches anything on the mail's behalf: the frame's own policy
        decides what may load, and remote images stay hidden until this reader asks
        for them, as they do in Gmail.

        One fetch per message, then a cache. The index keeps the message's *text*
        and always will; the HTML is asked for over IMAP on the first look, filed
        in the sidecar's byte budget as a cache (least-recently-seen rows are the
        ones that go, same as a preview), and read from there after that. The
        second reason it is a cache and not a store is honesty: what is rendered
        is the `text/html` part, which is *not* the part that was searched, so the
        pane says which part it is showing rather than implying the two agree.
        """
        folder = str(ref.get("folder") or "")
        uid = int(ref.get("uid") or 0)
        mid = str(ref.get("message_id") or "")
        if mailhtml is None:
            raise NothingToRender(MAILHTML_WHY)
        if not self.cfg.get("render_html", DEFAULTS["render_html"]):
            raise NothingToRender("rendering a message's HTML is turned off "
                                  "(config: render_html); its text is still here")
        limit = int(self.cfg.get("render_html_max_bytes",
                                 DEFAULTS["render_html_max_bytes"]) or 0)
        want_images = (bool(self.cfg.get("show_remote_images",
                                         DEFAULTS["show_remote_images"]))
                       if images is None else bool(images))

        raw, charset, section, cached = "", "", "", False
        hit = self.media.get(mid, HTML_PART) if self.media is not None else None
        if hit is not None:
            raw = (hit["bytes"] or b"").decode("utf-8", "replace")
            cached = True
            # Which section the part sat in is something the sidecar still knows,
            # so the strip keeps naming it on a second look. Which charset the
            # *message* declared is not: the cache holds UTF-8 by construction,
            # and a pane claiming "read as windows-1252" about bytes it decoded as
            # UTF-8 would be repeating a claim it did not just check. That fact
            # goes quiet here, and `cached` says why.
            if uid and self.media is not None:
                known = self.media.part(folder, uid, HTML_PART)
                section = known[0] if known else ""
        if not raw.strip():
            fetcher = self.imap.fetcher

            def read(conn):
                if uid:
                    part = fetcher.fetch_html_part(conn, folder, uid,
                                                   parts=self.media,
                                                   max_bytes=limit)
                    if part is not None:
                        return part
                body = (fetcher.fetch_message(conn, folder, uid) if uid else
                        # Indexed before UIDs were recorded: the Message-ID
                        # header still finds the message in the folder.
                        fetcher.fetch_by_message_id(conn, folder, mid))
                if body is None:
                    raise NothingToRender(
                        f"Gmail no longer answers for this message in {folder} "
                        f"(uid {uid or 'unknown'}): it may have moved or been "
                        "renumbered there. A Refresh re-reads the folders")
                return pick_html_part(body)

            try:
                found = self.imap.call(folder, read)
            except EngineError:
                raise
            except Exception as e:
                raise EngineError(
                    f"Gmail could not be reached for this message's HTML: "
                    f"{type(e).__name__}: {e}") from None
            if found is None or not (found.text or "").strip():
                raise NothingToRender(
                    "this message has no HTML part to show — what is here is its "
                    "text, which is what the index searched", reason="no_html")
            raw, charset, section = found.text, found.charset, found.section
            if self.media is not None:
                try:
                    data = raw.encode("utf-8")
                    self.media.put(mid, HTML_PART, data,
                                   "text/html; charset=utf-8", source_bytes=len(data))
                    self.media.prune_to_budget()
                except Exception as e:
                    log.debug("rendered body not cached for %s: %s", mid, e)

        cid_map = self._cid_map(folder, uid, mid) if "cid:" in raw.lower() else {}
        try:
            done = mailhtml.render_email_html(
                raw, message_url=f"/api/message/{quote(mid)}/attachment",
                cid_map=cid_map, images_allowed=want_images,
                max_bytes=limit or mailhtml.MAX_BYTES)
        except mailhtml.MailHtmlError as e:
            # The guards refuse *before* parsing, because markup nested a hundred
            # thousand deep costs the sanitizer a minute on a worker whose job is
            # to answer a click in a second. The reader gets the reason, and the
            # pane keeps its text view.
            raise NothingToRender(f"{e} — shown as text instead") from None
        return {"message_id": mid,
                "html": mailhtml.frame_document(done["html"],
                                                images_allowed=want_images),
                "shown": "text/html", "charset": charset,
                "section": section, "cached": cached,
                "images_allowed": want_images,
                "images_blocked": done["images_blocked"],
                "images_shown": done["images_shown"],
                "inline_images": done["inline"],
                "empty": not done["html"].strip(),
                "notes": done["notes"]}

    def _cid_map(self, folder: str, uid: int, mid: str) -> dict:
        """`{content-id: filename}` for this message, from Gmail once.

        An inline logo is addressed as `cid:…` in the HTML, and the only thing
        that resolves it is a header the structure does not carry — so this is one
        small section probe per message, cached beside the body it belongs to, and
        asked for only by a mail that actually mentions a cid. The filenames it
        returns are the ones the attachment route already serves, which is the
        whole trick: an inline image becomes an ordinary click on a file this
        program is allowed to open.
        """
        if self.media is not None:
            hit = self.media.get(mid, HTML_CIDS)
            if hit is not None:
                try:
                    return dict(json.loads((hit["bytes"] or b"{}").decode("utf-8")))
                except (ValueError, TypeError):
                    pass
        if not uid:
            return {}
        try:
            found = self.imap.call(folder,
                                   lambda conn: self.imap.fetcher.content_ids(
                                       conn, folder, uid))
        except Exception as e:
            log.debug("no content-ids for %s: %s", mid, e)
            return {}
        if self.media is not None:
            try:
                self.media.put(mid, HTML_CIDS, json.dumps(found).encode("utf-8"),
                               "application/json", source_bytes=len(found) * 64)
                self.media.prune_to_budget()
            except Exception as e:
                log.debug("content-ids not cached for %s: %s", mid, e)
        return found

    def thumb(self, message_id: str, filename: str) -> Optional[dict]:
        """A message's stored preview of one file, or None when there is none.

        Reads the sidecar only, on a media worker: no Gmail, no index, no reason
        to wait for either. That is the whole point of keeping previews — the
        reading pane answers "which picture is this" from disk, and Gmail is
        asked only for the file itself, when it is opened.

        Previews exist only for images PIL could read at index time, so "no
        preview" is the normal answer for a PDF and for any message indexed
        before this existed. `media --backfill` fills in the latter.
        """
        if self.media is None:
            return None
        got = self.media.get(message_id, filename)
        if got and not str(got.get("mime", "")).startswith("image/"):
            # Cannot happen from this program — a preview is a PIL re-encode, so
            # it is a PNG or a JPEG — but a sidecar is a file, and this origin
            # serves it inline. Only pictures get that.
            log.warning("preview for %s/%s is %s, not an image: not served",
                        message_id, filename, got.get("mime"))
            return None
        return got

    def facets(self) -> dict:
        if self.store is None:
            raise EngineError("the index is still loading")
        return {"senders": self.store.sender_facets(30),
                "totals": self.store.facet_totals()}

    # -- status -------------------------------------------------------------
    def status_sync(self) -> dict:
        # How many previews the sidecar holds. `None` means previews are off
        # (`store_thumbs` false, or the file would not open), which is a different
        # sentence from "0": one is a setting, the other a backlog to sweep.
        try:
            previews = self.media.count() if self.media is not None else None
        except Exception as e:
            log.debug("preview count failed: %s", e)
            previews = None
        st: dict = {"ready": self.ready, "loading": self.loading,
                    "error": self.open_error, "db_path": self.db_path,
                    "top": self.top, "previews": previews,
                    "mode": self.cfg.get("search_mode", DEFAULTS["search_mode"]),
                    "endpoint_configured": bool(self.cfg.get("embed_endpoint")),
                    "endpoint_ok": self.endpoint_ok,
                    "endpoint_error": self.endpoint_error,
                    # Whether this endpoint takes image input at all: None until
                    # an image has been tried against it - a startup probe would
                    # cost a vision encode on every boot, for a leg most queries
                    # never use - then True/False, so the page can say "this
                    # endpoint cannot search by picture" instead of letting the
                    # user wonder why an image found nothing.
                    "endpoint_images": None if self.client is None
                    else self.client.multimodal_ok,
                    "min_relevance": self.cfg.get("min_relevance",
                                                  DEFAULTS["min_relevance"]),
                    "collapse": bool(self.cfg.get("collapse", True)),
                    "tier_norm": bool(self.cfg.get("tier_norm", True)),
                    "parse_filters": bool(self.cfg.get("parse_filters", True)),
                     "gmail_account_index": gmail_index(self.cfg),
                     "queued": self._queue}
        if self.store is not None:
            try:
                s = self.store.stats()
                st.update({"messages": s["messages"], "chunks": s["chunks"],
                           "vectors": s["vectors"], "fts_docs": s["fts_docs"],
                           "dim": s["dim"],
                           "vector_mb": round(s["vector_bytes"] / 1e6, 1),
                           "by_tier": s["by_tier"]})
                st["last_indexed_date"] = self.store.get_state("last_indexed_date")
                # What the last refresh did about mail that is gone from Gmail.
                # The GUI has no delete button and never will, so this is the
                # only place it can say where those messages went: the pill's
                # tooltip and the refresh toast read these two keys.
                rec = self.store.reconcile_state()
                st["removed_total"] = rec["removed_total"]
                st["last_reconcile_at"] = rec["last_reconcile_at"]
                st["last_removal_at"] = rec["last_removal_at"]
                st["last_reconcile"] = rec["last_reconcile"]
            except Exception as e:                       # status never breaks
                st["stats_error"] = str(e)
        return st

    # -- index refresh ------------------------------------------------------
    def start_refresh(self) -> dict:
        """Run `mailbox.py index` as a child process and tail its log.

        A subprocess, not a thread: indexing wants to write the database this
        process has open, and `index` owns its own progress bar, watermark and
        IMAP session. Piped stdio is also the one thing a confined Windows
        executor reliably refuses, so the child writes straight to a file.
        """
        with self._refresh_lock:
            if self.refresh["running"]:
                raise EngineError("an index run is already going")
            cmd = [sys.executable, str(ROOT / "mailbox.py"), "index"]
            self.refresh.update({"running": True, "started_at": time.time(),
                                 "finished_at": None, "exit_code": None,
                                 "log": "", "cmd": " ".join(cmd)})
            try:
                with open(REFRESH_LOG, "a", encoding="utf-8",
                          errors="replace") as fh:
                    fh.write(f"\n===== refresh start "
                             f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    fh.flush()
                    # One handle, inherited by the child and closed here after:
                    # opening a second one per run would leak a file handle each
                    # time, and the child does not need ours once it starts.
                    proc = subprocess.Popen(
                        cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                        stdout=fh, stderr=subprocess.STDOUT)
            except Exception as e:
                self.refresh.update({"running": False, "exit_code": -1})
                raise EngineError(f"could not start the index run: {e}") from None
        threading.Thread(target=self._watch_refresh, args=(proc,),
                         daemon=True, name="mbox-refresh").start()
        return {"started": True, "cmd": self.refresh["cmd"]}

    def _watch_refresh(self, proc) -> None:
        code = proc.wait()
        self.refresh.update({"running": False, "exit_code": code,
                             "finished_at": time.time()})
        if code == 0:
            # New mail means the cached snapshot answers from yesterday's corpus.
            try:
                self.submit(self.rebuild).result(timeout=600)
            except Exception as e:
                log.warning("re-open after index failed: %s", e)

    def refresh_status(self) -> dict:
        out = dict(self.refresh)
        out["log"] = tail_log(REFRESH_LOG)
        return out


def tail_log(path: Path, lines: int = 40) -> str:
    """The last `lines` of the index run, carriage returns turned into rows.

    `tqdm` redraws one line with `\\r`; written to a file that is one enormous
    line, so splitting on it is what makes a progress bar readable in a panel.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    rows = [r.strip() for r in raw.replace("\r", "\n").split("\n") if r.strip()]
    return "\n".join(rows[-lines:])


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _quoted(value: str) -> str:
    """A filename inside a `Content-Disposition`, quoted the way the header needs.

    The name comes from a MIME part somebody else wrote, and an unescaped `"` or
    a newline in it would either truncate the header or be a header injection
    attempt served from this origin.
    """
    safe = str(value or "").replace("\\", "\\\\").replace('"', '\\"')
    return safe.replace("\r", "").replace("\n", "")


def _etag_of(data: bytes) -> str:
    """A weak validator for bytes this server can hand over again.

    Weak (`W/`) is the honest marking: it says "these bytes look the same to me",
    which is what a cache actually needs here — a preview is a re-encode this
    program makes, and a file at Gmail can be replaced without this index knowing.
    """
    return 'W/"%s"' % hashlib.sha1(data).hexdigest()[:16]


def _served(data: bytes, ctype: str, name: str, request: Request, *,
            inline: bool, cache: str) -> Response:
    """A file this origin hands over: cached, never sniffed, inline only when it
    cannot execute.

    `If-None-Match` is answered with an empty 304 instead of the bytes again, so
    a lightbox reopened or the same invoice opened twice costs a header round
    trip and not a file. That is what makes "open it again" free without this
    program keeping a copy of anybody's attachment on disk.
    """
    etag = _etag_of(data)
    sent = request.headers.get("if-none-match") or ""
    if etag in sent or etag[3:-1] in sent:
        return Response(status_code=304,
                        headers={"ETag": etag, "Cache-Control": cache})
    head = {"Content-Disposition":
            f'{"inline" if inline else "attachment"}; filename="{_quoted(name)}"',
            "X-Content-Type-Options": "nosniff",
            "ETag": etag, "Cache-Control": cache}
    return Response(content=data, media_type=ctype, headers=head)


def _fail(exc: Exception) -> JSONResponse:
    """Errors as data, because a spinner that never stops is worse than a 409."""
    status = 409 if isinstance(exc, EngineError) else 500
    detail = str(getattr(exc, "args", [str(exc)])[0]) if isinstance(exc, EngineError) \
        else f"{type(exc).__name__}: {exc}"
    return JSONResponse({"error": detail}, status_code=status)


def _body_filters(raw: Any):
    """Whether a search body carries explicit chips (as opposed to NL parsing)."""
    return isinstance(raw, dict) and bool(raw)


def create_app(engine: Engine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warm in the background rather than blocking startup: /api/status can
        # then say "loading" while the 146 MB read is still happening, instead
        # of the browser sitting on a connection that refuses to answer.
        task = asyncio.create_task(app.state.engine.acall(app.state.engine._open))
        app.state.warm = task
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="vectormailbox", docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.state.engine = engine

    @app.get("/")
    async def index():
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/api/status")
    async def status():
        eng: Engine = app.state.engine
        try:
            st = await eng.acall(eng.status_sync)
        except Exception as e:
            st = {"ready": False, "loading": eng.loading, "error": str(e),
                  "db_path": eng.db_path}
        st["queued"] = eng.queued
        st["refresh_running"] = bool(eng.refresh["running"])
        return st

    @app.post("/api/search")
    async def search(body: dict = Body(default_factory=dict)):
        eng: Engine = app.state.engine
        try:
            return await eng.acall(eng.search, body)
        except EngineError as e:
            return _fail(e)
        except Exception as e:
            log.exception("search failed")
            return _fail(e)

    @app.get("/api/messages")
    async def messages(before: str = "", limit: int = 50, sender: str = "",
                       since: str = "", until: str = "", text: str = "",
                       has_attachment: Optional[bool] = None):
        eng: Engine = app.state.engine
        try:
            return await eng.acall(eng.messages, before, limit, sender, since,
                                   until, text, has_attachment)
        except EngineError as e:
            return _fail(e)
        except Exception as e:
            log.exception("browse failed")
            return _fail(e)

    @app.get("/api/message/{message_id}")
    async def message(message_id: str):
        eng: Engine = app.state.engine
        try:
            return await eng.acall(eng.message, message_id)
        except EngineError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            log.exception("message failed")
            return _fail(e)

    @app.get("/api/message/{message_id}/attachment/{filename}")
    async def attachment(message_id: str, filename: str, request: Request):
        """One file, fetched from Gmail as it is opened and kept nowhere.

        Two executors rather than one, because one would be the bottleneck: the
        index read that says *where* the file lives goes to the engine thread,
        the Gmail read that fetches it goes to a media worker. A search is
        therefore never waiting behind a mailbox download, `queued` keeps
        meaning "searches waiting", and the download itself no longer pays for a
        fresh login or for the parts of the message nobody asked for.

        Images/PDF/text open in the tab (`inline`); anything that could
        execute from this origin — HTML, SVG, scripts — is download-only.
        """
        eng: Engine = app.state.engine
        try:
            ref = await eng.acall(eng.attachment_ref, message_id)
            data, ctype, name = await eng.amedia(eng.read_attachment, ref,
                                                 filename)
        except EngineError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            log.exception("attachment failed")
            return _fail(e)
        return _served(data, ctype, name, request,
                       inline=ctype.lower().startswith(Engine.INLINE_TYPES),
                       cache=FILE_CACHE)

    @app.get("/api/message/{message_id}/html")
    async def message_html(message_id: str, images: Optional[bool] = None):
        """This message's own HTML, as data — never as a document on this origin.

        The answer is JSON: the sanitized body wrapped in a frame document this
        program writes, in an envelope whose media type says JSON even though HTML
        is inside it. That is not pedantry. Served as `text/html` from this origin
        the bytes would *be* this app's document — same origin, same API, same
        cookies — and a stranger's markup would be first-party code. As JSON it is
        text the page hands to a sandboxed frame that has no script permission and
        no origin of its own, which is the same shape Gmail uses: the mail is
        rendered *at* a different origin, not merely escaped into this one.

        It is also why an `.html` **attachment** stays download-only while a mail's
        HTML **body** can be read as mail: one is a file this app would be handing
        out as itself, the other is content this app shapes, sanitizes and cages.

        `images` is this reader asking for a message's remote images (or not). The
        answer decides the frame's own policy, written into the document before the
        mail's markup is spliced in — the sender never negotiates with it, and
        until it is set no pixel of a mail leaves this machine.
        """
        eng: Engine = app.state.engine
        try:
            ref = await eng.acall(eng.attachment_ref, message_id)
        except EngineError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            log.exception("html source lookup failed")
            return _fail(e)
        try:
            return await eng.amedia(eng.render_html, ref, images)
        except NothingToRender as e:
            return JSONResponse({"error": str(e), "reason": e.reason},
                                status_code=404)
        except EngineError as e:
            return _fail(e)
        except Exception as e:
            log.exception("html render failed")
            return _fail(e)

    @app.get("/api/message/{message_id}/media/{filename}")
    async def media(message_id: str, filename: str, request: Request):
        """A message's own preview of one of its files, read off local disk.

        No Gmail, no index read: the preview was made from the same bytes the
        indexer already saw, and kept beside the index for exactly this request.
        On a mailbox where opening a picture meant logging into IMAP and pulling
        the whole message across for it, this is the part that answers at once.

        Images only — stricter than `INLINE_TYPES`, because a preview is
        something this page renders *inside itself* rather than in a tab.
        """
        eng: Engine = app.state.engine
        try:
            thumb = await eng.amedia(eng.thumb, message_id, filename)
        except Exception as e:
            log.exception("preview failed")
            return _fail(e)
        if not thumb:
            return JSONResponse(
                {"error": "no preview of this file is stored here — "
                          "python mailbox.py media --backfill keeps them"},
                status_code=404)
        return _served(thumb["bytes"], thumb["mime"], filename, request,
                       inline=True, cache=THUMB_CACHE)

    @app.get("/api/facets")
    async def facets():
        eng: Engine = app.state.engine
        try:
            return await eng.acall(eng.facets)
        except Exception as e:
            return _fail(e)

    @app.post("/api/index/refresh")
    async def refresh_start():
        eng: Engine = app.state.engine
        try:
            return await asyncio.to_thread(eng.start_refresh)
        except EngineError as e:
            return _fail(e)
        except Exception as e:
            log.exception("refresh start failed")
            return _fail(e)

    @app.get("/api/index/refresh")
    async def refresh_status():
        eng: Engine = app.state.engine
        out = dict(eng.refresh)
        out["log"] = tail_log(REFRESH_LOG)
        return out

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="mailbox search UI (hybrid search, reading pane, inbox browse)")
    p.add_argument("--config", default="config.json")
    p.add_argument("--db", help="override db_path from the config")
    p.add_argument("--profile", default="",
                   help="knob set to start from: 'v2' (default) or 'legacy'")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address. The default is loopback only: this serves "
                        "your whole mailbox with no login, so 0.0.0.0 hands it to "
                        "your network")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--top", type=int, help="results per search (default: config)")
    p.add_argument("--no-browser", action="store_true", help="do not open a tab")
    return p


def run(args, cfg: dict) -> None:
    """Serve. `mailbox.py gui` calls this with its already-loaded config."""
    import uvicorn

    engine = Engine(cfg, top=getattr(args, "top", None))
    app = create_app(engine)
    url = f"http://{args.host}:{args.port}/"
    print(f"vectormailbox  {cfg['db_path']}")
    print(f"  open {url}   (Ctrl-C to stop)")
    if engine.cfg.get("embed_endpoint"):
        print(f"  embedding endpoint {engine.cfg['embed_endpoint']}"
              f"{'  (model ' + str(engine.cfg.get('embed_model')) + ')' if engine.cfg.get('embed_model') else ''}")
    if not getattr(args, "no_browser", False):
        threading.Timer(1.0, _open_browser, args=(url,)).start()
    try:
        uvicorn.run(app, host=args.host, port=int(args.port), log_level="warning")
    finally:
        engine.close()


def _open_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass            # a headless box just prints the URL, which is the point


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.profile)
    if getattr(args, "db", None):
        cfg["db_path"] = str(args.db)
    run(args, cfg)


if __name__ == "__main__":
    main()
