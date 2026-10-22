#!/usr/bin/env python3
"""
Tests for the mailbox GUI: the cached store, the read-only browsing queries it
runs, and the HTTP contract in front of them.

No network and no real index: the embedding endpoint stays stubbed by
test_mailbox's hashed bag-of-words transport, so a search here exercises the
same fusion code the CLI runs. The parts worth testing are exactly the parts a
screenshot cannot show — a cache that goes stale silently, a `SystemExit` that
would kill the server, and a subject line that is really HTML.

  python test_gui.py
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from urllib.parse import quote

import mailbox as mb
import gui
import queryparse as qp
from test_mailbox import (FakeIMAP, Image, NEEDS_PIL, StubbedEndpoint,
                          index_messages, make_cfg, make_msg, png_bytes)

try:
    from fastapi.testclient import TestClient
except ImportError:                                       # pragma: no cover
    TestClient = None

NEEDS_CLIENT = "fastapi + httpx are needed for the gui tests"


def pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """A real, valid PDF, written by hand.

    The indexer only stores attachment text it can actually extract, so a
    fixture made of `b"x"` produces no chunk and tests the wrong thing — and
    neither generator at hand works here: PyMuPDF is not installed on this
    interpreter (requirements.txt lists it; `index` falls back to text-only),
    and pypdf reads PDFs but cannot write text into one. So the file is
    assembled here, with the xref offsets computed rather than guessed.

    Object layout, so the numbers stay readable: 1 catalog, 2 page tree,
    3.. page objects, then the content streams, then the one font.
    """
    n = len(pages)
    first_page, first_stream, font = 3, 3 + n, 3 + 2 * n
    objs = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Kids ["
            + " ".join(f"{first_page + i} 0 R" for i in range(n))
            + f"] /Count {n} >>"),
    }
    for i, lines in enumerate(pages):
        stream = ("BT /F1 11 Tf 56 740 Td 14 TL\n"
                  + "".join(f"({pdf_escape(line)}) Tj T*\n" for line in lines)
                  + "ET").encode("latin-1", "replace")
        objs[first_page + i] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Resources << /Font << /F1 {font} 0 R >> >>"
            f" /Contents {first_stream + i} 0 R >>")
        objs[first_stream + i] = (f"<< /Length {len(stream)} >>\nstream\n"
                                 f"{stream.decode('latin-1')}\nendstream")
    objs[font] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num in range(1, font + 1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n{objs[num]}\nendobj\n".encode("latin-1", "replace")
    xref_at = len(out)
    out += f"xref\n0 {font + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {font + 1} /Root 1 0 R >>\nstartxref\n"
            f"{xref_at}\n%%EOF\n").encode()
    return bytes(out)


def invoice_pdf(text: str, pages: int = 1) -> bytes:
    """One page with `text` on it, or `pages` pages carrying one line each."""
    if pages == 1:
        return make_pdf([[text]])
    return make_pdf([[f"{text} (pagina {i + 1})"] for i in range(pages)])


def msg_set():
    """Three messages, one of them written by somebody who means you harm."""
    return [
        make_msg("bob", sender="bob@flexa.nl", subject="factuur domein",
                 body="hierbij de factuur voor het domein van het kantoor",
                 date=datetime(2026, 3, 4, tzinfo=timezone.utc),
                 # indexed in the first browser slot; the config below says the
                 # mailbox actually lives in the fourth one
                 gmail_url="https://mail.google.com/mail/u/0/#all/1ab23c",
                 atts=[mb.Attachment("factuur.pdf", "application/pdf",
                                     invoice_pdf(
                                         "Factuur 2026-014 voor het domein van "
                                         "het kantoor, te betalen binnen veertien "
                                         "dagen. Bedrag 480,00 euro exclusief btw "
                                         "voor de levering van het jaar.")),
                       # plain text is metadata-only for the extractor, so this
                       # one contributes an extension and no chunk
                       mb.Attachment("notities.txt", "text/plain",
                                     b"de factuur is betaald")]),
        make_msg("shop", sender="shop@bol.nl", subject="domein verlengd",
                 body="het domein is verlengd tot volgend jaar",
                 cc="carol@example.com",
                 date=datetime(2026, 5, 6, tzinfo=timezone.utc)),
        make_msg("xss", sender="x@example.com",
                 subject="<img src=x onerror=alert(1)>",
                 body="kwetsbaarheid <script>alert(2)</script> in het formulier",
                 date=datetime(2026, 6, 1, tzinfo=timezone.utc)),
    ]


class Fixture(StubbedEndpoint):
    """A stubbed endpoint, a three-message index, and an open engine."""

    def setUp(self):
        super().setUp()
        index_messages(self.store, self.cfg, msg_set())
        cfg = make_cfg(self.db)
        # The mailbox was indexed in slot 0 and the config now says slot 3: every
        # Gmail link the API hands out has to follow the config, not the row.
        cfg["gmail_account_index"] = 3
        self.engine = gui.Engine(cfg)
        self.addCleanup(self.engine.close)
        self.engine.open()
        # No `with`: entering the context manager would run the lifespan warm-up
        # a second time. The one test that wants that path builds its own app.
        self.app = gui.create_app(self.engine)
        self.client = TestClient(self.app)

    def get_json(self, path):
        r = self.client.get(path)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def post_search(self, body):
        r = self.client.post("/api/search", json=body)
        return r


def ids_of(payload):
    return [h["message_id"] for h in payload.get("results", [])]


# ---------------------------------------------------------------------------
# the cache the whole UI rests on
# ---------------------------------------------------------------------------

class TestIndexCache(StubbedEndpoint):
    def test_cache_off_by_default(self):
        """Default off, because a cached snapshot is wrong while another
        connection is writing — the shell and the GUI opt in, nothing else."""
        store = mb.Store(self.db, self.cfg)
        self.stores.append(store)
        self.assertIsNot(store.load_index(), store.load_index())

    def test_warm_snapshot_is_reused(self):
        store = mb.Store(self.db, self.cfg, cache_index=True)
        self.stores.append(store)
        first = store.load_index()
        self.assertIs(first, store.load_index())

    def test_warm_vector_snapshot_answers_a_text_only_call(self):
        """Answering with_vectors=False from the warm vector snapshot costs
        nothing, while building a second one would duplicate ~60 MB of text."""
        store = mb.Store(self.db, self.cfg, cache_index=True)
        self.stores.append(store)
        with_vectors = store.load_index(with_vectors=True)
        self.assertIs(with_vectors, store.load_index(with_vectors=False))

    def test_committing_a_message_invalidates_the_cache(self):
        store = mb.Store(self.db, self.cfg, cache_index=True)
        self.stores.append(store)
        index_messages(store, self.cfg, msg_set())
        warm = store.load_index()
        self.assertEqual(len(warm.ids), len(warm.vecs))
        index_messages(store, self.cfg, [
            make_msg("nieuw", body="nieuwe mail over het park",
                     subject="park", date=datetime(2026, 7, 1, tzinfo=timezone.utc))])
        fresh = store.load_index()
        self.assertIsNot(warm, fresh)
        self.assertIn("nieuw", fresh.ids)


# ---------------------------------------------------------------------------
# one filter builder, two readers
# ---------------------------------------------------------------------------

class TestFilterParity(StubbedEndpoint):
    def setUp(self):
        super().setUp()
        # Parity on an empty mailbox is trivially true, so this class needs the
        # same three messages the other readers see: one with attachments, one
        # with a cc, dates a month apart.
        index_messages(self.store, self.cfg, msg_set())

    def test_browsing_and_search_agree_on_the_same_filters(self):
        """`page_messages` and `filter_message_ids` share `_filter_clauses`;
        this is the test that keeps a clicked chip and the words `van bob` from
        meaning two different things."""
        cases = [
            qp.Filters(senders=["bob@flexa.nl"]),
            qp.Filters(senders=["flexa"], has_attachment=True),
            qp.Filters(recipients=["carol@example.com"]),
            qp.Filters(since="2026-04-01"),
            qp.Filters(until="2026-04-01"),
            qp.Filters(has_attachment=False),
            qp.Filters(senders=["bol"], subjects=["domein"]),
        ]
        for f in cases:
            with self.subTest(filters=f):
                want = self.store.filter_message_ids(f)
                got = {r["message_id"] for r in self.store.page_messages(limit=99,
                                                                         filters=f)}
                self.assertEqual(want, got)
                self.assertEqual(len(got), self.store.count_messages(filters=f))

    def test_empty_filter_set_is_no_filter(self):
        self.assertEqual(self.store.count_messages(), 3)
        self.assertEqual(self.store.count_messages(filters=qp.Filters()), 3)


# ---------------------------------------------------------------------------
# browsing
# ---------------------------------------------------------------------------

class TestBrowsing(StubbedEndpoint):
    def setUp(self):
        super().setUp()
        index_messages(self.store, self.cfg, msg_set())

    def test_pages_newest_first(self):
        page = self.store.page_messages(limit=50)
        self.assertEqual([m["message_id"] for m in page], ["xss", "shop", "bob"])

    def test_cursor_is_exclusive(self):
        first = self.store.page_messages(limit=2)
        self.assertEqual(len(first), 2)
        second = self.store.page_messages(before=first[-1]["date"], limit=2)
        self.assertEqual([m["message_id"] for m in second], ["bob"])
        seen = {m["message_id"] for m in first} | {m["message_id"] for m in second}
        self.assertEqual(len(seen), 3)                  # nothing walked twice

    def test_total_counts_every_page(self):
        self.assertEqual(self.store.count_messages(), 3)
        self.assertEqual(len(self.store.page_messages(limit=2)), 2)

    def test_text_is_a_like_not_the_fts_index(self):
        """Browsing by typing a sender must not turn into BM25 and start
        surfacing 2019 mail."""
        hits = self.store.page_messages(text="flexa")
        self.assertEqual([m["message_id"] for m in hits], ["bob"])
        self.assertEqual(self.store.count_messages(text="factuur"), 1)

    def test_thread_len_counts_the_conversation(self):
        self.store.conn.execute(
            "UPDATE messages SET thread_id = 'T1' WHERE message_id IN ('bob','shop')")
        self.store.conn.commit()
        page = {m["message_id"]: m for m in self.store.page_messages(limit=50)}
        self.assertEqual(page["bob"]["thread_len"], 2)
        # A message with no Gmail thread is filed under its own id (`m.thread_id
        # or m.message_id`, all over the indexer), so a lone mail counts as a
        # thread of one. `> 1` is what means "there is more of this conversation"
        # — which is what the front-end asks.
        self.assertEqual(page["xss"]["thread_len"], 1)

    def test_sender_facets_are_ordered_by_volume(self):
        facets = self.store.sender_facets(10)
        self.assertEqual([f["count"] for f in facets], [1, 1, 1])
        self.assertEqual({f["sender"] for f in facets},
                         {"bob@flexa.nl", "shop@bol.nl", "x@example.com"})

    def test_facet_totals(self):
        tot = self.store.facet_totals()
        self.assertEqual(tot["messages"], 3)
        self.assertEqual(tot["with_attachments"], 1)
        self.assertTrue(str(tot["newest"]).startswith("2026-06"))


# ---------------------------------------------------------------------------
# the reading pane
# ---------------------------------------------------------------------------

class TestMessageDetail(StubbedEndpoint):
    def setUp(self):
        super().setUp()
        index_messages(self.store, self.cfg, msg_set())

    def test_body_beats_the_preview(self):
        """A search hit is a window; the pane must show the whole indexed body,
        not the 300-character preview the list pane uses."""
        d = self.store.message_detail("bob")
        self.assertFalse(d["body_missing"])
        self.assertIn("factuur", d["body"])
        self.assertGreaterEqual(len(d["body"]), len(d["preview"]))

    def test_missing_body_falls_back_and_says_so(self):
        """8 of 4 452 real messages have no canonical body chunk; the pane has
        to say "this is all the index kept" instead of showing a lie."""
        self.store.conn.execute(
            "DELETE FROM chunks WHERE message_id = 'bob'"
            " AND canonical = 1 AND kind = 'body'")
        self.store.conn.commit()
        d = self.store.message_detail("bob")
        self.assertTrue(d["body_missing"])
        self.assertEqual(d["body"], d["preview"])

    def test_attachment_names_lose_their_prefix(self):
        """Names come back from the chunk labels, without the `attachment: `
        prefix the indexer writes. A plain-text attachment has no chunk (the
        extractor indexes its extension only), so it is absent here and present
        in `attachment_exts` — which is why the pane shows both."""
        d = self.store.message_detail("bob")
        self.assertEqual(d["attachments"], ["factuur.pdf"])
        self.assertEqual(d["attachment_exts"], "pdf,txt")

    def test_truncation_follows_the_config(self):
        cfg = make_cfg(self.db, max_source_chars=6)
        store = mb.Store(self.db, cfg, cache_index=True)
        self.stores.append(store)
        self.assertTrue(store.message_detail("bob")["truncated"])
        self.assertFalse(self.store.message_detail("xss")["truncated"])

    def test_unknown_message_is_none(self):
        self.assertIsNone(self.store.message_detail("eris"))

    def test_thread_messages_are_oldest_first(self):
        self.store.conn.execute("UPDATE messages SET thread_id = 'T1'"
                                " WHERE message_id IN ('bob','shop','xss')")
        self.store.conn.commit()
        th = self.store.thread_messages("T1")
        self.assertEqual([m["message_id"] for m in th], ["bob", "shop", "xss"])
        self.assertEqual(self.store.thread_messages(""), [])


class TestAttachmentNames(StubbedEndpoint):
    """An attachment rendered page by page writes one chunk per page, all
    labelled with the same file; a reading pane that lists four copies of one
    invoice reads as a bug, so the names are collapsed here.

    It also pins the two ways a name is missing, because they need different
    words in the UI: a type the extractor does not read at all, and a message
    whose indexed text was never a body.
    """

    def test_pages_of_one_pdf_appear_as_one_name(self):
        """The labels are written by the indexer; here the page-per-chunk case
        is reproduced with copies, because this interpreter has no PyMuPDF and
        so cannot render pages — which is exactly the path that writes those
        labels. The name recovery must not care either way."""
        index_messages(self.store, self.cfg, [make_msg(
            "scan", subject="scan van bon", body="zie scan",
            atts=[mb.Attachment("scan.pdf", "application/pdf",
                                invoice_pdf("scan van bon", pages=3))])])
        rows = [r[0] for r in self.store.conn.execute(
            "SELECT id FROM chunks WHERE message_id='scan' AND kind='attachment'")]
        self.assertEqual(len(rows), 1)              # text-only extraction here
        first = rows[0]
        for page in (2, 3):
            self.store.conn.execute(
                "INSERT INTO chunks (message_id, kind, label, text)"
                " SELECT message_id, kind, ?, text FROM chunks WHERE id = ?",
                (f"attachment: scan.pdf (page {page})", first))
        self.store.conn.commit()
        labels = [r[0] for r in self.store.conn.execute(
            "SELECT DISTINCT label FROM chunks"
            " WHERE message_id='scan' AND kind='attachment'")]
        self.assertEqual(len(labels), 3)            # three chunks, one file
        d = self.store.message_detail("scan")
        self.assertEqual(d["attachments"], ["scan.pdf"])
        self.assertEqual(d["attachment_exts"], "pdf")

    def test_an_image_attachment_keeps_its_name(self):
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        index_messages(self.store, self.cfg, [make_msg(
            "plot", subject="plaatje", body="zie bijlage",
            atts=[mb.Attachment("shot.png", "image/png", png)])])
        d = self.store.message_detail("plot")
        self.assertEqual(d["attachments"], ["shot.png"])


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestStatusEndpoint(Fixture):
    def test_status_reports_the_index(self):
        st = self.get_json("/api/status")
        self.assertTrue(st["ready"])
        self.assertEqual(st["messages"], 3)
        self.assertTrue(st["vectors"])
        self.assertTrue(st["fts_docs"])
        self.assertEqual(st["db_path"], self.db)

    def test_index_html_and_assets_are_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("vectormailbox", r.text)
        for path in ("/static/app.css", "/static/app.js"):
            a = self.client.get(path)
            self.assertEqual(a.status_code, 200, path)
            # no CDN: the page must not reach outside this machine
            for banned in ("//cdn.", "unpkg.com", "googleapis.com", "jsdelivr"):
                self.assertNotIn(banned, a.text)

    def test_lifespan_warms_before_the_first_answer(self):
        """The startup task loads the index on the engine thread; the first
        request is queued behind it, so it cannot answer "not ready"."""
        eng = gui.Engine(make_cfg(self.db))
        self.addCleanup(eng.close)
        with TestClient(gui.create_app(eng)) as client:
            self.assertTrue(client.get("/api/status").json()["ready"])


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestRemovedMail(Fixture):
    """Mail deleted (or archived away) in Gmail leaves the index at the next
    refresh. This interface has no delete button and never gets one, so what it
    owes the user instead is the truth about where those messages went."""

    def drop_one(self):
        """What a refresh does when the folder stops holding a message, done by
        another connection — which is exactly how the GUI sees it."""
        self.store.remove_messages(["bob"])
        self.store.set_state("last_reconcile_at", "2026-10-21T10:18:37+00:00")
        self.store.set_state("last_reconcile", json.dumps(
            {"removed": 1, "restored": 0,
             "folders": {"INBOX": {"stored": 3, "live": 2, "removed": 1,
                                   "restored": 0}}}))
        # the same call the refresh watcher makes, and on the same thread: the
        # engine's sqlite connection belongs to the engine thread, not this one
        self.engine.submit(self.engine.rebuild).result(timeout=30)

    def test_status_says_what_the_last_refresh_dropped(self):
        self.assertEqual(self.get_json("/api/status")["removed_total"], 0)
        self.drop_one()
        st = self.get_json("/api/status")
        self.assertEqual(st["messages"], 2)
        self.assertEqual(st["removed_total"], 1)
        self.assertEqual(st["last_reconcile_at"], "2026-10-21T10:18:37+00:00")
        self.assertEqual(st["last_reconcile"]["folders"]["INBOX"]["removed"], 1)
        # the pill says "last dropped on": that is when the newest tombstone was
        # written, which is not the same moment as the last check
        self.assertTrue(st["last_removal_at"].startswith("2026-"))
        self.assertNotEqual(st["last_removal_at"], st["last_reconcile_at"])

    def test_a_dropped_message_says_where_it_went(self):
        """'no such message' sends people hunting for a bug; the index knows this
        one was dropped, when, and that it was not this app that deleted it."""
        self.drop_one()
        r = self.client.get("/api/message/bob")
        self.assertEqual(r.status_code, 404)
        self.assertIn("removed from the index on 2026-10-21", r.json()["error"])
        self.assertIn("no longer holds this message", r.json()["error"])

    def test_mail_that_was_never_indexed_still_says_so(self):
        r = self.client.get("/api/message/nope")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["error"], "no message 'nope' in the index")

    def test_a_dropped_message_stops_answering_searches(self):
        query = {"q": "factuur", "mode": "lexical", "top": 10}
        self.assertIn("bob", ids_of(self.post_search(query).json()))
        self.drop_one()
        self.assertNotIn("bob", ids_of(self.post_search(query).json()))

    def test_the_toast_and_the_run_it_reports_share_one_clock(self):
        """The toast only claims a drop if *this* run reconciled. `started_at` is
        a Python `time.time()` in seconds and `Date.parse` returns milliseconds,
        so a comparison that forgets the units is true for every run forever:
        the page would announce dropped mail after a refresh that dropped
        nothing. No browser here to catch that, so pin the conversion."""
        js = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("at / 1000", js)
        self.assertNotIn("at >= r.started_at", js)

    def test_the_tooltip_reads_the_removal_clock_not_the_check_clock(self):
        """"Last dropped" is the newest tombstone, not the last check — a quiet
        refresh would otherwise date a deletion it did not make — and it goes
        through the page's own date label, because the stored stamp is UTC and a
        sliced string of it prints an hour this page disagrees with."""
        js = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        fn = js.split("function droppedNote")[1].split("\n}")[0]
        self.assertIn("s.last_removal_at", fn)
        self.assertNotIn("s.last_reconcile_at", fn)
        self.assertIn("whenLabel(asDate(", fn)

    def test_the_only_thing_the_page_can_ask_for_is_a_re_read(self):
        """Read-only is a property of the whole surface, not of one query: two
        POSTs, one that searches and one that re-runs the indexer. Anything that
        could delete mail would have to start by appearing here."""
        src = (Path(__file__).with_name("gui.py")).read_text(encoding="utf-8")
        self.assertEqual(sorted(set(re.findall(r'@app\.post\("([^"]+)"', src))),
                         ["/api/index/refresh", "/api/search"])


def raw_mime(mid, filename, data, content_type="application/pdf"):
    """The message as it sits in the folder: one body plus one named file."""
    from email.message import EmailMessage
    m = EmailMessage()
    m["Message-ID"] = mid
    m["From"] = "bob@flexa.nl"
    m["Subject"] = "factuur domein"
    m.set_content("hierbij de factuur")
    maintype, _, subtype = content_type.partition("/")
    m.add_attachment(data, maintype=maintype, subtype=subtype or "octet-stream",
                     filename=filename)
    return bytes(m)


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestAttachmentEndpoint(Fixture):
    """Opening an attachment: the bytes are not in the index, so the server
    re-reads the message from Gmail at click time — read-only, inline only for
    types that cannot execute from this origin, and never written to this disk."""

    PDF = b"%PDF-1.4 factuur 2026-014"
    PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 24

    def setUp(self):
        super().setUp()
        from unittest import mock
        # bob's row carries uid 0 (indexed before UIDs were recorded), so the
        # whole click runs over the Message-ID fallback — the join a real old
        # mailbox actually uses. ada's row carries the UID it was indexed under,
        # which is the shape mail indexed since has: the click can go straight
        # to that message, and from there to one part of it.
        ada = make_msg("ada", sender="ada@example.com",
                       subject="logo voor de site",
                       atts=[mb.Attachment("logo.png", "image/png", self.PNG)])
        ada.uid = 33
        index_messages(self.store, self.cfg, [ada])
        self.conn = FakeIMAP({"INBOX": {
            77: raw_mime("bob", "factuur.pdf", self.PDF),
            33: raw_mime("ada", "logo.png", self.PNG, "image/png"),
        }})
        # `_connect` is an instance method: the replacement must close over the
        # *test*, not over the fetcher, or `self` here would be the GmailFetcher.
        patch = mock.patch.object(gui.GmailFetcher, "_connect",
                                  lambda _fetcher, debug=False: self.conn)
        patch.start()
        self.addCleanup(patch.stop)

    def url(self, mid, name):
        return (f"/api/message/{mid}/attachment/"
                + quote(name, safe=""))

    def test_a_named_file_is_fetched_from_gmail_and_answered_inline(self):
        r = self.client.get(self.url("bob", "factuur.pdf"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.content, self.PDF)
        self.assertEqual(r.headers["content-type"].split(";")[0],
                         "application/pdf")
        self.assertIn("inline", r.headers["content-disposition"])
        self.assertIn("factuur.pdf", r.headers["content-disposition"])
        self.assertIn(("UID SEARCH", (None, b'(HEADER Message-ID "bob")')),
                      self.conn.commands,
                      "uid 0: the message is traced by its Message-ID header")
        self.assertIn("UID FETCH", [v for v, _a in self.conn.commands])

    def test_a_type_that_can_execute_from_this_origin_is_download_only(self):
        self.conn = FakeIMAP({"INBOX": {
            77: raw_mime("bob", "factuur.html", b"<script>alert(1)</script>",
                         "text/html"),
        }})
        r = self.client.get(self.url("bob", "factuur.html"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("attachment;", r.headers["content-disposition"])
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")

    def test_a_file_the_message_does_not_carry_names_the_answer(self):
        r = self.client.get(self.url("bob", "ontbrekend.pdf"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("no file named", r.json()["error"])

    def test_a_message_the_index_dropped_says_where_it_went(self):
        self.store.remove_messages(["bob"])
        self.engine.submit(self.engine.rebuild).result(timeout=30)
        r = self.client.get(self.url("bob", "factuur.pdf"))
        self.assertEqual(r.status_code, 404)
        # the same sentence the reading pane gives, not a bare "no such file"
        self.assertIn("removed from the index", r.json()["error"])

    def test_gmail_that_does_not_answer_the_header_is_said_so(self):
        from test_mailbox import FakeIMAP
        self.conn = FakeIMAP({"INBOX": {}})          # the folder lost the mail
        r = self.client.get(self.url("bob", "factuur.pdf"))
        self.assertEqual(r.status_code, 404)
        err = r.json()["error"]
        self.assertIn("Gmail no longer answers", err)
        self.assertIn("Refresh", err)

    def test_no_app_password_is_a_sentence_not_a_stack_trace(self):
        self.engine.cfg = {**self.engine.cfg, "gmail_app_password": ""}
        r = self.client.get(self.url("bob", "factuur.pdf"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("gmail_app_password", r.json()["error"])

    def test_a_file_in_a_message_that_knows_its_uid_costs_one_part(self):
        """The click names one file; the wire should carry that file. Reading the
        whole message to get a 40 KB logo out of a 5 MB mail is what made opening
        a picture feel like dial-up."""
        r = self.client.get(self.url("ada", "logo.png"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.content, self.PNG)
        self.assertEqual(r.headers["content-type"].split(";")[0], "image/png")
        asked = [args[-1] for v, args in self.conn.commands if v == "UID FETCH"]
        self.assertTrue(asked, "the click did read something")
        whole = [i for i in asked if b"RFC822" in i or b"BODY.PEEK[]" in i]
        self.assertEqual(whole, [], "no part of the message that was not asked for")

    def test_a_file_is_served_by_a_media_worker_so_searches_keep_their_queue(self):
        """`queued` means *searches waiting*, and it has to keep meaning that:
        a mailbox download that ran on the engine thread would hide inside that
        number and hold the search box behind somebody else's attachment."""
        seen = []
        real = gui.Engine.read_attachment

        def spy(self_, ref, filename):
            seen.append(threading.current_thread().name)
            return real(self_, ref, filename)

        with mock.patch.object(gui.Engine, "read_attachment", spy):
            r = self.client.get(self.url("ada", "logo.png"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith("mbox-media"), seen[0])
        self.assertEqual(self.get_json("/api/status")["queued"], 0)

    def test_the_only_verbs_asked_of_gmail_are_reads(self):
        self.client.get(self.url("bob", "factuur.pdf"))
        self.client.get(self.url("ada", "logo.png"))
        allowed = {"EXAMINE", "UID SEARCH", "UID FETCH", "LOGOUT"}
        bad = [v for v, _a in self.conn.commands if v not in allowed]
        self.assertEqual(bad, [], "opening a file may only ever read the mailbox")


def html_mail(mid, html, *, text=None, inline=(), alternative=True,
              from_addr="news@acme.example", subject="nieuwsbrief"):
    """A folder's raw bytes for a mail whose HTML body is `html`.

    Multipart/alternative with the text part first, because that is what an HTML
    mail with a plain-text fallback looks like on the wire — and the shape is the
    point: the part the index searched and the part the pane renders are then two
    different sections of one message, which is the difference the reading pane
    has to be able to name. `alternative=False` is the other common shape, a
    singlepart `text/html` mail where the message *is* its body, and `inline` adds
    images with a Content-ID for the HTML to point at.
    """
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    top = MIMEMultipart("alternative" if alternative else "related")
    top["Message-ID"] = mid
    top["From"] = from_addr
    top["Subject"] = subject
    if alternative:
        top.attach(MIMEText(text if text is not None
                            else "deze mail heeft ook een html-versie",
                            "plain", "utf-8"))
    top.attach(MIMEText(html, "html", "utf-8"))
    for cid, name, data in inline:
        img = MIMEImage(data, "png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=name)
        top.attach(img)
    return bytes(top)


def plain_mail(mid, text="deze mail is pure tekst"):
    """A mail with no HTML anywhere in it, which most mail still is."""
    from email.mime.text import MIMEText
    m = MIMEText(text, "plain", "utf-8")
    m["Message-ID"] = mid
    m["From"] = "ada@example.com"
    m["Subject"] = "status"
    return bytes(m)


NEWS = ('<html><head><title>acme nieuws</title>'
        '<style>h1{color:#c00}body{background:url(http://evil/bg.png)}</style>'
        '</head><body bgcolor="#ffffff">'
        '<table cellpadding="6" align="center"><tr>'
        '<td bgcolor="#f4f4f4"><h1>Bedankt voor je bestelling</h1>'
        '<p>Factuur <a href="http://evil.example/f/1">bekijk hier</a></p>'
        '<img src="http://evil.example/x.png" width="600" height="80">'
        '<img src="cid:logo" width="140" height="32">'
        '<script>alert(1)</script>'
        '<p onclick="steal()">mee doen?</p></td></tr></table></body></html>')

PNG_ATOM = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 24


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestHtmlBodyEndpoint(Fixture):
    """Reading a mail as mail: its HTML part, sanitized, as data, in a frame with
    no scripts — and nothing about the text pane, the attachment rules, or the
    mailbox's read state changed on the way.

    The claims worth testing here are not "the sanitizer works" (that is
    `test_mailhtml.py`, sixty-odd cases against the markup mailboxes actually
    contain). They are the ones that only hold at this seam: the answer is JSON so
    a mail can never be this origin's document; the part shown is named because it
    is not the part searched; a second look asks Gmail for nothing; and none of it
    marks a message read.
    """

    def setUp(self):
        super().setUp()
        from unittest import mock
        rows = [
            make_msg("news", sender="news@acme.example", subject="nieuwsbrief",
                     body="deze mail heeft ook een html-versie"),
            make_msg("plain", sender="ada@example.com", subject="status",
                     body="deze mail is pure tekst"),
            make_msg("single", sender="news@acme.example", subject="banner"),
            make_msg("related", sender="news@acme.example", subject="met logo"),
            make_msg("deep", sender="news@acme.example", subject="nestje"),
            make_msg("accents", sender="fr@acme.example", subject="bonjour"),
        ]
        uids = {"news": 41, "plain": 42, "single": 43, "related": 44,
                "deep": 45, "accents": 46}
        for row in rows:
            row.uid = uids[row.message_id]
        index_messages(self.store, self.cfg, rows)
        # A mail whose whole body is one `text/html` part: for this shape the
        # message and the body are the same bytes, so the cheap path *is* the
        # whole-message read, and there is no section number to remember.
        single = (b"Message-ID: <single>\r\nFrom: news@acme.example\r\n"
                  b"Subject: banner\r\n"
                  b"Content-Type: text/html; charset=\"utf-8\"\r\n\r\n"
                  b"<h1>Een enkele html-part</h1><p>hallo</p>")
        # Declared ISO-8859-1 with a byte that only means something in CP1252:
        # the difference between "l’hôtel" and an invisible control character is
        # decided by whether the reader believes the header or guesses. Quoted-
        # printable rather than 8-bit because that is what European mail of this
        # charset actually is on the wire — and because a section FETCH returns
        # the part's *encoded* bytes, so the decoding has to happen here, from the
        # charset, with nobody having quietly re-encoded it on the way.
        accents = (b"Message-ID: <accents>\r\nFrom: fr@acme.example\r\n"
                   b"Subject: bonjour\r\n"
                   b"Content-Type: text/html; charset=iso-8859-1\r\n"
                   b"Content-Transfer-Encoding: quoted-printable\r\n\r\n"
                   b"Bienvenue =E0 l=92h=F4tel")
        self.conn = FakeIMAP({"INBOX": {
            41: html_mail("news", NEWS),
            42: plain_mail("plain"),
            43: single,
            44: html_mail("related", '<p>hallo</p><img src="cid:logo">',
                          text=None, alternative=False,
                          inline=[("logo", "logo.png", PNG_ATOM)]),
            45: html_mail("deep", "<div>" * 900 + "diep" + "</div>" * 900),
            46: accents,
        }})
        patch = mock.patch.object(gui.GmailFetcher, "_connect",
                                  lambda _fetcher, debug=False: self.conn)
        patch.start()
        self.addCleanup(patch.stop)
        self.assertTrue(self.engine.media is not None,
                        "the sidecar is where a fetched body is kept")

    def url(self, mid, images=None):
        return f"/api/message/{mid}/html" + (f"?images={images}"
                                             if images is not None else "")

    def html_of(self, mid, images=None):
        r = self.client.get(self.url(mid, images))
        self.assertEqual(r.status_code, 200, r.text)
        return r, r.json()

    # -- what the answer is ---------------------------------------------------

    def test_the_body_arrives_as_data_and_never_as_a_document(self):
        """The whole security claim of this endpoint in one header: HTML inside,
        JSON outside. As `text/html` these bytes would *be* this app — same
        origin, same API, same cookies — with a stranger's markup first-party in
        it; as JSON they are text a page hands to a frame that has no scripts and
        no origin, which is the shape Gmail renders mail at."""
        r, data = self.html_of("news")
        self.assertEqual(r.headers["content-type"].split(";")[0],
                         "application/json")
        self.assertTrue(data["html"].startswith("<!DOCTYPE html>"))
        self.assertIn("img-src 'none'", data["html"])
        self.assertIn("default-src 'none'", data["html"])

    def test_the_mail_keeps_looking_like_mail(self):
        """Sanitizing that leaves a newsletter looking like an apology is not
        safer, it is broken: the table, its `cellpadding`, the sender's
        `bgcolor`, the `<font>` and the head stylesheet are how mail is built."""
        _r, data = self.html_of("news")
        body = data["html"]
        self.assertIn("<h1>Bedankt voor je bestelling</h1>", body)
        self.assertIn('cellpadding="6"', body)
        self.assertIn('bgcolor="#f4f4f4"', body)
        self.assertIn('align="center"', body)
        self.assertIn("h1{color:#c00}", body.replace(" ", "").replace("\n", ""))
        self.assertIn('target="_blank"', body)
        self.assertIn('rel="noopener noreferrer"', body)
        # The mail's own `<body>` is unwrapped rather than copied: it is the frame
        # that decides the page's background and font, so a mail cannot repaint
        # the reading pane by declaring itself dark.
        self.assertNotIn("<body ", body)

    def test_what_cannot_be_allowed_is_gone_rather_than_escaped(self):
        _r, data = self.html_of("news")
        body = data["html"]
        self.assertNotIn("<script", body)
        self.assertNotIn("alert(1)", body)
        self.assertNotIn("onclick", body)
        self.assertNotIn("<title>", body)      # and its text, not left on the page
        self.assertNotIn("acme nieuws", body)

    def test_the_part_on_screen_is_the_one_the_pane_names(self):
        """Rendering the HTML part is not rendering the message: the text the
        index searched is another section, and may even say something else. So the
        answer says which part this is, and the plain alternative is demonstrably
        not what is shown."""
        _r, data = self.html_of("news")
        self.assertEqual(data["shown"], "text/html")
        self.assertEqual(data["section"], "2")
        self.assertFalse(data["cached"])
        self.assertNotIn("deze mail heeft ook een html-versie", data["html"])

    def test_a_body_that_is_the_whole_message_still_has_a_part(self):
        """The singlepart shape, which most HTML mail from a form or a shop *is*:
        no multipart wrapper, so no numbered sections in the structure at all.
        The body still arrives, and the section it names is either the one the
        server was willing to address the message by — RFC 3501 lets `BODY[1]`
        mean a singlepart whole message — or nothing, when it would not answer for
        `BODY[1.MIME]` and this app read the message itself instead. Both are a
        named part; neither is a claim about a section nobody read."""
        _r, data = self.html_of("single")
        self.assertIn("<h1>Een enkele html-part</h1>", data["html"])
        self.assertEqual(data["shown"], "text/html")
        self.assertIn(data["section"], ("", "1"))

    # -- images ---------------------------------------------------------------

    def test_a_pixel_of_the_senders_server_never_leaves_unasked(self):
        _r, blocked = self.html_of("news")
        self.assertEqual(blocked["images_blocked"], 1)
        self.assertNotIn("http://evil.example/x.png", blocked["html"])
        self.assertIn("data:image/gif", blocked["html"])   # holds the layout
        self.assertIn("1 remote image hidden", " ".join(blocked["notes"]))
        _r, shown = self.html_of("news", 1)
        self.assertIn("http://evil.example/x.png", shown["html"])
        self.assertIn("img-src http:", shown["html"])
        self.assertEqual(shown["images_shown"], 1)

    def test_an_inline_image_is_a_request_for_this_messages_own_file(self):
        """A `cid:` reference is a pointer into the same message, and the only
        thing that resolves it is a header the structure does not carry. The
        answer is an ordinary attachment URL: the frame loads a file this app is
        already allowed to hand out, rather than the mail being given a way to
        read its own bytes."""
        _r, data = self.html_of("related")
        self.assertIn("/api/message/related/attachment/logo.png", data["html"])
        self.assertEqual(data["inline_images"], 1)
        self.assertEqual(
            [n for n in data["notes"] if "cid:" in n], [],
            "the reference resolved, so there is nothing to report")

    # -- when there is nothing to show ----------------------------------------

    def test_a_mail_without_an_html_part_says_which_part_there_is(self):
        r = self.client.get(self.url("plain"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("no HTML part", r.json()["error"])
        # A word, not a sentence to recognize: the page disables the switch on
        # this, and would stop working the day the wording is improved.
        self.assertEqual(r.json()["reason"], "no_html")

    def test_markup_nested_past_the_limit_is_refused_before_it_costs_seconds(self):
        """900 nested divs is not a newsletter, and parsing it is the sanitizer's
        problem for a minute; refusing it is this app's answer in a second, with
        the reason in the pane that still has its text."""
        r = self.client.get(self.url("deep"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("nests its markup", r.json()["error"])
        self.assertIn("shown as text instead", r.json()["error"])

    def test_gmail_that_raises_is_a_connection_problem_not_an_empty_mail(self):
        from unittest import mock
        with mock.patch.object(gui.GmailFetcher, "fetch_message",
                               side_effect=OSError("connection reset")):
            r = self.client.get(self.url("plain"))
        self.assertEqual(r.status_code, 409)
        self.assertIn("could not be reached", r.json()["error"])

    def test_html_the_page_could_have_been_handed_is_still_download_only(self):
        """The body of a mail renders because it is sanitized and caged. A *file*
        this origin serves is a different thing, and `.html` stays exactly as
        download-only as it was before this endpoint existed: one is content this
        app shapes, the other would be this app's own document. The attachment
        suite carries the byte-level proof; this is the pin that nothing about the
        new route loosened the inline list."""
        self.assertNotIn("text/html", gui.Engine.INLINE_TYPES)
        self.assertNotIn("image/svg", gui.Engine.INLINE_TYPES)

    # -- what it costs, and what it must never do -----------------------------

    def test_the_second_look_asks_gmail_for_nothing(self):
        """A fetched body is kept in the sidecar's byte budget, as a cache: the
        first look pays for a fetch, every look after that reads a local file, and
        the budget the previews live in is what bounds it.

        The strip under the frame is why the two answers are compared here and not
        just the byte counts. A second look may go quiet about a fact it did not
        re-check — the charset the message declared, since the cache holds UTF-8 by
        construction — but it may not lose the part it is quoting, and it may not
        fill either slot with a phrase about itself: this once printed "read as the
        cached copy" where a charset belongs."""
        _r, first = self.html_of("news")
        asked = len(self.conn.commands)
        self.assertGreater(asked, 0, "the first look did go to Gmail")
        _r, data = self.html_of("news")
        self.assertEqual(len(self.conn.commands), asked,
                         "a cached body costs Gmail no commands")
        self.assertTrue(data["cached"])
        self.assertEqual(first["charset"], "utf-8")
        self.assertEqual(data["charset"], "")
        self.assertEqual(data["shown"], first["shown"])
        self.assertEqual(data["section"], first["section"])

    def test_the_only_verbs_asked_are_reads_and_nothing_marks_the_mail_seen(self):
        """`EXAMINE`, and never a body specifier that pulls the text: a FETCH of
        `RFC822` or a bare `BODY[]` sets `\\Seen` on the spot, which would mean
        that looking at a mail in this app marked it read in Gmail. `BODYSTRUCTURE`
        is the one item that reads a message without fetching its text, so it may
        appear; the text itself may only come through `BODY.PEEK[`."""
        for mid in ("news", "single", "related", "plain"):
            self.client.get(self.url(mid))
        verbs = [v for v, _a in self.conn.commands]
        self.assertNotIn("SELECT", verbs, "EXAMINE, or the flag changes")
        self.assertEqual([v for v in verbs
                          if v not in ("EXAMINE", "UID SEARCH", "UID FETCH",
                                       "LOGOUT")], [])
        fetched = [args[-1] for v, args in self.conn.commands if v == "UID FETCH"]
        self.assertTrue(fetched, "a body was fetched")
        self.assertEqual([i for i in fetched if b"RFC822" in i], [],
                         "RFC822 sets \\Seen")
        self.assertEqual([i for i in fetched
                          if b"BODY[" in i and b"BODY.PEEK[" not in i], [],
                         "a bare BODY[] sets \\Seen just the same")

    def test_the_render_runs_on_a_media_worker_so_searches_keep_their_queue(self):
        """Same two executors as an attachment, for the same reason: an IMAP read
        that ran on the engine thread would hide inside `queued` and hold the
        search box behind somebody else's newsletter."""
        from unittest import mock
        seen = []
        real = gui.Engine.render_html

        def spy(self_, ref, images=None):
            seen.append(threading.current_thread().name)
            return real(self_, ref, images)

        with mock.patch.object(gui.Engine, "render_html", spy):
            r = self.client.get(self.url("news"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith("mbox-media"), seen[0])
        self.assertEqual(self.get_json("/api/status")["queued"], 0)

    def test_a_mail_is_decoded_by_the_charset_it_declares(self):
        """The indexer's rule — UTF-8, else latin-1, header ignored — stays, or
        search results would move under an index people already ran. For shown
        text the header is the difference between an accent and a control
        character, so the render path believes it, as a browser does."""
        _r, data = self.html_of("accents")
        self.assertEqual(data["charset"], "windows-1252")
        self.assertIn("l’hôtel", data["html"])
        self.assertNotIn("\x92", data["html"])

    def test_a_mail_rendering_is_refused_when_the_config_says_so(self):
        """`render_html` off has to leave the rest of the app alone: this is the
        text pane it always was, with a reason instead of a button that hangs."""
        self.engine.cfg = {**self.engine.cfg, "render_html": False}
        r = self.client.get(self.url("news"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("render_html", r.json()["error"])


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
@unittest.skipIf(Image is None, NEEDS_PIL)
class TestLocalPreviews(Fixture):
    """A picture used to cost a login, a whole message across the wire, and a
    spinner. Its preview is now cut at index time from bytes the indexer already
    held, kept beside the index, and read off disk — so the first thing the user
    sees is instant, and Gmail is not asked for it at all."""

    def setUp(self):
        super().setUp()
        self.conn = FakeIMAP({"INBOX": {}})
        patch = mock.patch.object(gui.GmailFetcher, "_connect",
                                  lambda _fetcher, debug=False: self.conn)
        patch.start()
        self.addCleanup(patch.stop)
        shot = make_msg("pic", subject="screenshot van de banner",
                        atts=[mb.Attachment("screenshot.png", "image/png",
                                            png_bytes(900))])
        index_messages(self.store, self.cfg, [shot],
                       media=self.store.media_cache())
        self.engine.submit(self.engine.rebuild).result(timeout=30)

    def media_url(self, mid="pic", name="screenshot.png"):
        return f"/api/message/{mid}/media/" + quote(name, safe="")

    def test_a_preview_is_read_off_disk_and_gmail_is_never_asked(self):
        r = self.client.get(self.media_url())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"].split(";")[0], "image/jpeg")
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.conn.commands, [],
                         "a preview that logs in is the old bug, retold")

    def test_a_preview_is_small_and_still_the_same_picture(self):
        from io import BytesIO
        r = self.client.get(self.media_url())
        w, h = Image.open(BytesIO(r.content)).size
        self.assertLessEqual(max(w, h), mb.THUMB_MAX_SIDE)
        self.assertLess(len(r.content), 60_000, "a preview is not a second file")
        row = self.engine.media.get("pic", "screenshot.png")
        self.assertEqual((w, h), (row["width"], row["height"]),
                         "the reading pane sizes its box from these same numbers")

    def test_the_browser_is_told_to_keep_it(self):
        first = self.client.get(self.media_url())
        etag = first.headers["etag"]
        self.assertTrue(etag.startswith('W/"'), etag)
        self.assertIn("immutable", first.headers["cache-control"])
        again = self.client.get(self.media_url(),
                                headers={"if-none-match": etag})
        self.assertEqual(again.status_code, 304)
        self.assertEqual(again.content, b"",
                         "a revalidation answers empty, or it saved nothing")

    def test_a_file_that_is_not_a_picture_says_so_with_the_command_that_fixes_it(self):
        r = self.client.get(self.media_url("bob", "factuur.pdf"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("media --backfill", r.json()["error"])
        self.assertEqual(self.conn.commands, [])

    def test_the_status_line_says_how_many_previews_are_kept(self):
        st = self.get_json("/api/status")
        self.assertEqual(st["previews"], 1,
                         "the picture, and not bob's PDF: only images get one")

    def test_the_list_carries_one_preview_per_row_and_the_pane_carries_all(self):
        rows = {it["message_id"]: it
                for it in self.get_json("/api/messages?limit=50")["items"]}
        self.assertEqual(rows["pic"]["thumb"], "screenshot.png")
        self.assertEqual(rows["bob"]["thumb"], "",
                         "a row without a picture gets no thumbnail slot")
        detail = self.get_json("/api/message/pic")
        self.assertIn("screenshot.png", detail["media"])
        self.assertGreater(detail["media"]["screenshot.png"]["width"], 0,
                           "the pane sizes its box from the real preview")

    def test_a_search_result_carries_its_picture_as_well(self):
        """The list of matches is where a preview earns its keep: a row that
        matched on what a screenshot says can already show the screenshot. The
        search pays for that with one sidecar query for the whole page, which is
        why it is filled in here and not fetched per row by the browser."""
        data = self.post_search({"q": "banner", "mode": "lexical",
                                 "top": 5}).json()
        hits = {h["message_id"]: h for h in data["results"]}
        self.assertIn("pic", hits, "the screenshot should match its own subject")
        self.assertEqual(hits["pic"]["thumb"], "screenshot.png")
        self.assertEqual(self.conn.commands, [],
                         "a preview that logs into Gmail is not a local one")

    def test_a_dropped_message_loses_its_preview_with_its_row(self):
        """A preview outliving its message is the worst orphan: a deleted mail
        still showing its picture in the list."""
        self.assertEqual(len(self.store.remove_messages(["pic"])), 1)
        self.engine.submit(self.engine.rebuild).result(timeout=30)
        self.assertEqual(self.client.get(self.media_url()).status_code, 404)
        self.assertEqual(self.engine.media.count(), 0)
        self.assertEqual(self.engine.thumb("pic", "screenshot.png"), None)


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestSearchEndpoint(Fixture):
    def test_search_returns_ranked_hits_with_explain_fields(self):
        r = self.post_search({"q": "factuur", "mode": "lexical", "top": 5})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertIn("bob", ids_of(data))
        hit = data["results"][0]
        for key in ("score", "fused", "prior", "dense_rank", "bm25_rank", "tier",
                    "snippet", "date", "sender", "rank", "label"):
            self.assertIn(key, hit)
        self.assertEqual(hit["rank"], 1)
        self.assertLess(hit["score"], 1.01)

    def test_empty_query_is_an_error_not_a_crash(self):
        self.assertEqual(self.post_search({"q": "   "}).status_code, 409)

    def test_unknown_mode_is_rejected(self):
        r = self.post_search({"q": "factuur", "mode": "vibes"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("vibes", r.json()["error"])

    def test_systemexit_becomes_a_409(self):
        """`run()` signals a dead end with SystemExit — the way a CLI does it.
        Uncought that is a SystemExit inside a request handler, which takes the
        whole server down over one empty index."""
        with mock.patch.object(mb.Searcher, "run",
                               side_effect=SystemExit("the index is empty")):
            r = self.post_search({"q": "factuur", "mode": "lexical"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["error"], "the index is empty")
        self.assertTrue(self.get_json("/api/status")["ready"])   # still serving

    def test_a_nested_submit_is_refused_instead_of_hanging(self):
        """One worker thread: submitting from inside a task would wait on the
        thread it is occupying. A clear error beats a request that never ends."""
        def nested():
            return self.engine.submit(lambda: 1)

        with self.assertRaises(RuntimeError) as cm:
            self.engine.submit(nested).result(timeout=10)
        self.assertIn("deadlock", str(cm.exception))
        self.assertTrue(self.get_json("/api/status")["ready"])   # still serving

    def test_unreachable_endpoint_degrades_to_keywords_and_says_so(self):
        """Not a spinner for 14 seconds and then an error: BM25 can answer this,
        and the user is told which engine answered."""
        self.engine.endpoint_ok = False
        self.engine.endpoint_error = "connection refused"
        self.engine.endpoint_checked = time.time()
        r = self.post_search({"q": "factuur", "mode": "hybrid"})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertIn("degraded", data)
        self.assertEqual(data["meta"]["mode"], "lexical")
        self.assertIn("bob", ids_of(data))

    def test_semantic_mode_does_not_downgrade_when_the_endpoint_is_up(self):
        r = self.post_search({"q": "factuur", "mode": "semantic"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("degraded", r.json())
        self.assertEqual(r.json()["meta"]["mode"], "semantic")

    def test_knobs_reach_the_engine(self):
        r = self.post_search({"q": "domein", "mode": "lexical", "top": 1,
                              "tier_norm": False, "agg_k": 1, "min_relevance": 0})
        self.assertEqual(len(r.json()["results"]), 1)
        meta = r.json()["meta"]
        self.assertFalse(meta["tier_norm"])
        self.assertEqual(meta["agg_k"], 1)

    def test_tier_selection_reaches_the_engine(self):
        r = self.post_search({"q": "domein", "mode": "lexical",
                              "tiers": "small,large", "min_relevance": 0})
        self.assertEqual(r.json()["meta"]["tiers"], ["small", "large"])


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestImageQueryEndpoint(Fixture):
    """`POST /api/search` with an `image`.

    The arithmetic of a picture in a hybrid search is tested in test_mailbox;
    what belongs here is the HTTP contract — what is refused, in what words, and
    what is never sent back out.
    """

    PNG = b"\x89PNG\r\n\x1a\n" + b"pixels " * 24

    def data_url(self, data=None, mime="image/png"):
        return "data:" + mime + ";base64," + base64.b64encode(
            data if data is not None else self.PNG).decode()

    def test_an_image_alone_is_a_search(self):
        r = self.post_search({"image": self.data_url(),
                              "image_name": "screenshot.png", "mode": "semantic"})
        self.assertEqual(r.status_code, 200, r.text)
        meta = r.json()["meta"]
        self.assertEqual(meta["image"], "screenshot.png")
        self.assertTrue(meta["image_only"])
        self.assertEqual(meta["w_lex"], 0, "keywords cannot read a picture, so "
                                           "the bm25 leg must be off, not half on")

    def test_image_and_words_may_share_one_query(self):
        r = self.post_search({"q": "factuur", "image": self.data_url()})
        self.assertEqual(r.status_code, 200, r.text)
        meta = r.json()["meta"]
        self.assertEqual(meta["image"], "the attached image")
        self.assertFalse(meta["image_only"])
        self.assertGreater(meta["w_lex"], 0, "the words still run through bm25")

    def test_the_image_bytes_are_never_sent_back(self):
        """The thumbnail lives in the browser; the API answers with scores. A
        search endpoint that echoes 5 MB of base64 per request turns a mail
        reader into a memory leak, and there is no reason for it to."""
        body = self.data_url()
        r = self.post_search({"q": "factuur", "image": body})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn(body[22:222], r.text)
        self.assertLess(len(r.text), 200000)

    def test_bytes_that_are_not_an_image_are_refused(self):
        r = self.post_search({"q": "factuur",
                              "image": base64.b64encode(make_pdf([["pdf"]])).decode()})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("PNG", r.json()["error"])

    def test_a_claimed_mime_is_not_believed(self):
        """A browser's `data:application/pdf` on an image field is a guess; the
        magic bytes are the fact, and the bytes are what the endpoint gets."""
        r = self.post_search({"q": "factuur",
                              "image": self.data_url(make_pdf([["pdf"]]),
                                                     mime="image/png")})
        self.assertEqual(r.status_code, 409, r.text)

    def test_an_oversized_image_is_refused_before_the_endpoint_is_called(self):
        big = self.PNG + b"x" * gui.MAX_IMAGE_BYTES
        r = self.post_search({"image": self.data_url(big)})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("MB", r.json()["error"])

    def test_garbage_in_the_image_field_is_refused(self):
        for bad in ("!!!!", "   ", "data:image/png;base64,", 1234, ["x"]):
            r = self.post_search({"q": "factuur", "image": bad})
            self.assertEqual(r.status_code, 409, repr(bad))

    def test_no_words_and_no_image_is_still_an_error(self):
        self.assertEqual(self.post_search({"q": "  ", "image": None}).status_code,
                         409)

    def test_a_picture_is_not_downgraded_to_keywords(self):
        """The degrade path exists because BM25 can answer *words*. A picture is
        not words: with the endpoint down the honest answer is a refusal, not a
        ranked list of mail that shares nothing with the photo."""
        self.engine.endpoint_ok = False
        self.engine.endpoint_error = "connection refused"
        self.engine.endpoint_checked = time.time()
        r = self.post_search({"q": "factuur", "image": self.data_url(),
                              "mode": "hybrid"})
        self.assertEqual(r.status_code, 409, r.text)
        # The sentence names the move, not the engine that could not help.
        self.assertIn("search by words until it answers", r.json()["error"])
        self.assertNotIn("degraded", r.json())
        self.assertTrue(self.get_json("/api/status")["ready"])      # still serving

    def test_lexical_mode_refuses_a_picture(self):
        r = self.post_search({"q": "factuur", "image": self.data_url(),
                              "mode": "lexical"})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("use hybrid or semantic", r.json()["error"])

    def test_an_endpoint_that_rejects_images_says_so(self):
        with mock.patch.object(mb.EmbeddingClient, "embed_query_image",
                               side_effect=SystemExit("this endpoint cannot "
                                                      "embed images")):
            r = self.post_search({"q": "factuur", "image": self.data_url()})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(r.json()["error"], "this endpoint cannot embed images")
        self.assertTrue(self.get_json("/api/status")["ready"])

    def test_status_says_whether_pictures_can_be_searched(self):
        """None until an image has been tried: the GUI does not spend a vision
        encode at startup on a leg most queries never use."""
        self.assertIsNone(self.get_json("/api/status")["endpoint_images"])
        self.post_search({"q": "factuur", "image": self.data_url()})
        self.assertIs(self.get_json("/api/status")["endpoint_images"], True)


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestChipRoundTrip(Fixture):
    """The chips the UI shows are the filters the server applied, and removing
    one has to actually remove it."""

    def base(self):
        return {"q": "factuur van bob@flexa.nl", "mode": "lexical",
                "min_relevance": 0, "top": 10}

    def test_query_filters_come_back_as_structured_chips(self):
        data = self.post_search(self.base()).json()
        struct = data["meta"]["filters_struct"]
        self.assertEqual(struct["senders"], ["bob@flexa.nl"])
        self.assertEqual(ids_of(data), ["bob"])

    def test_dropping_the_last_chip_removes_the_restriction(self):
        """Re-parsing the sentence after the chip is gone would cut `van bob`
        out of the embedded text and quietly keep the filter's effect: with the
        parser off for chip searches, the words come back and the restriction
        goes."""
        restricted = self.post_search({
            "q": "domein van bob@flexa.nl", "mode": "lexical", "min_relevance": 0,
            "top": 10, "filters": {"senders": ["bob@flexa.nl"]},
            "text": "domein"}).json()
        self.assertEqual(ids_of(restricted), ["bob"])
        dropped = self.post_search({
            "q": "domein van bob@flexa.nl", "mode": "lexical", "min_relevance": 0,
            "top": 10, "filters": {}, "text": "domein"}).json()
        self.assertTrue({"bob", "shop"} <= set(ids_of(dropped)))
        self.assertIsNone(dropped["meta"]["filters_struct"])

    def test_explicit_chips_survive_a_query_that_would_parse_differently(self):
        data = self.post_search({"q": "domein van shop@bol.nl", "mode": "lexical",
                                 "min_relevance": 0,
                                 "filters": {"senders": ["bob@flexa.nl"]},
                                 "text": "domein"}).json()
        self.assertEqual(data["meta"]["filters_struct"]["senders"],
                         ["bob@flexa.nl"])
        self.assertEqual(ids_of(data), ["bob"])

    def test_attachment_chip_filters_browsably(self):
        data = self.post_search({"q": "factuur", "mode": "lexical",
                                 "min_relevance": 0,
                                 "filters": {"has_attachment": True},
                                 "text": "factuur"}).json()
        self.assertEqual(ids_of(data), ["bob"])
        self.assertIs(data["meta"]["filters_struct"]["has_attachment"], True)


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestBrowseEndpoints(Fixture):
    def setUp(self):
        super().setUp()
        self.store.conn.execute(
            "UPDATE messages SET thread_id = 'T1' WHERE message_id IN ('bob','shop')")
        self.store.conn.commit()

    def test_messages_page_and_cursor(self):
        page = self.get_json("/api/messages?limit=2")
        self.assertEqual([m["message_id"] for m in page["items"]], ["xss", "shop"])
        self.assertTrue(page["next_cursor"])
        self.assertEqual(page["total"], 3)
        nxt = self.get_json("/api/messages?limit=2&before=" + page["next_cursor"])
        self.assertEqual([m["message_id"] for m in nxt["items"]], ["bob"])
        self.assertFalse(nxt["next_cursor"])

    def test_sender_and_attachment_filters(self):
        self.assertEqual(
            [m["message_id"] for m in
             self.get_json("/api/messages?sender=flexa")["items"]], ["bob"])
        self.assertEqual(
            [m["message_id"] for m in
             self.get_json("/api/messages?has_attachment=true")["items"]], ["bob"])
        self.assertEqual(
            len(self.get_json("/api/messages?has_attachment=false")["items"]), 2)

    def test_thread_len_shows_in_the_list(self):
        page = {m["message_id"]: m
                for m in self.get_json("/api/messages")["items"]}
        self.assertEqual(page["shop"]["thread_len"], 2)

    def test_message_detail_endpoint_carries_the_thread(self):
        d = self.get_json("/api/message/bob")
        self.assertIn("factuur", d["body"])
        self.assertEqual([m["message_id"] for m in d["thread"]], ["bob", "shop"])
        self.assertEqual(d["attachments"], ["factuur.pdf"])

    def test_unknown_message_is_a_404(self):
        self.assertEqual(self.client.get("/api/message/eris").status_code, 404)

    def test_facets_endpoint(self):
        f = self.get_json("/api/facets")
        self.assertEqual(f["totals"]["messages"], 3)
        self.assertEqual(len(f["senders"]), 3)


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestMailIsNeverMarkup(Fixture):
    """The one class of bug that turns a mail reader into an XSS hole, and the
    reason `app.js` has no innerHTML anywhere in it."""

    def test_a_hostile_subject_survives_as_text(self):
        page = self.get_json("/api/messages?text=onerror")
        self.assertEqual([m["message_id"] for m in page["items"]], ["xss"])
        self.assertEqual(page["items"][0]["subject"],
                         "<img src=x onerror=alert(1)>")

    def test_the_hostile_body_is_returned_as_data(self):
        r = self.client.get("/api/message/xss")
        self.assertEqual(r.headers["content-type"].split(";")[0], "application/json")
        self.assertEqual(json.loads(r.text)["body"].count("<script>"), 1)

    def test_the_script_is_valid_javascript(self):
        """Text-node discipline means nothing if the file does not parse. One
        stray parenthesis left the entire page a static shell — no results, no
        status, no error message — while every Python-side assertion here still
        passed, because nothing on this side of the wire executes the script.
        Both scripts are checked, since a page that loads two of them dies on
        whichever one is broken.
        Skipped when Node is absent or may not be spawned."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        for name in ("app.js", "mailhtml.js"):
            path = gui.WEB_DIR / name
            try:
                done = subprocess.run([node, "--check", str(path)],
                                      capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as e:
                self.skipTest(f"cannot run node here: {e}")
            self.assertEqual(done.returncode, 0, f"{name}: {done.stderr[:900]}")

    def test_the_front_end_never_writes_html_from_data(self):
        """Neither script may hand an HTML string to a parser of this document,
        and the second file is the one that has to be checked hardest: it is the
        only place mail markup arrives at all, and it passes it by *assignment*
        (`srcdoc`) instead of by parsing, so the markup is never this page's.

        Lexical on purpose, comments included: it is a grep a reviewer can repeat
        by hand, which is why those four words appear nowhere in these two files
        even in prose."""
        for name in ("app.js", "mailhtml.js"):
            src = (gui.WEB_DIR / name).read_text(encoding="utf-8")
            for banned in ("innerHTML", "outerHTML", "document.write",
                           "insertAdjacentHTML"):
                self.assertNotIn(banned, src, f"{name}: {banned}")
        self.assertIn("textContent", (gui.WEB_DIR / "app.js").read_text(encoding="utf-8"))
        mail = (gui.WEB_DIR / "mailhtml.js").read_text(encoding="utf-8")
        self.assertIn("f.srcdoc = data.html", mail,
                      "the ban is only meaningful while the frame is still filled")

    def test_the_page_loads_the_frame_script_before_the_page_that_uses_it(self):
        """Classic scripts, in order, because there is no bundler here to sort it
        out: `app.js` reaches for `window.MailHtml` the first time a message is
        opened, and a reordered line in this file is a dead reading pane."""
        page = (gui.WEB_DIR / "index.html").read_text(encoding="utf-8")
        self.assertLess(page.index('src="/static/mailhtml.js"'),
                        page.index('src="/static/app.js"'))


class TestMailFrameContract(unittest.TestCase):
    """The one file in this app that puts mail-authored bytes on screen, executed
    for real in Node against a stubbed DOM.

    Reading it is not enough, because everything that matters about it is a
    property of what the browser ends up holding: which permissions the frame has,
    in what order they were set, and whether the document arrived as an assigned
    `srcdoc` rather than as parsed markup. A stubbed `iframe` records the writes
    in the order they happen, so the assertions are about the shipped code's
    effect on the page and not about a copy of it.
    """

    SCRIPT = r"""
    const fs = require('fs');
    const src = fs.readFileSync(process.argv[2], 'utf8');
    const fail = (msg) => { console.error(msg); process.exit(1); };
    const eq = (what, got, want) => {
      if (String(got) !== String(want)) fail(what + ': got ' + got + ', want ' + want);
    };

    // One recording iframe: attribute writes and the srcdoc assignment are logged
    // in the order the code performed them.
    const writes = [];
    const element = (tag) => {
      const el = {tagName: tag, className: '', attrs: {}};
      el.setAttribute = (k, v) => { writes.push(k); el.attrs[k] = v; };
      Object.defineProperty(el, 'srcdoc', {
        set(v) { writes.push('srcdoc'); el._srcdoc = v; },
        get() { return el._srcdoc; },
      });
      return el;
    };
    const requests = [];
    let answer = {};
    const doc = {createElement: element};
    const win = {};
    new Function('document', 'window', 'fetch', src)(doc, win,
      (url) => { requests.push(url); return Promise.resolve(answer); });
    const MailHtml = win.MailHtml;
    if (!MailHtml) fail('web/mailhtml.js does not define window.MailHtml');

    // --- the frame -----------------------------------------------------------
    eq('the sandbox, spelled', MailHtml.SANDBOX,
       'allow-popups allow-popups-to-escape-sandbox');
    const f = MailHtml.frame({html: '<p>hallo</p>'});
    eq('element', f.tagName, 'iframe');
    eq('styled as the reading pane', f.className, 'mail-frame');
    eq('sandbox', f.attrs.sandbox, MailHtml.SANDBOX);
    if (/allow-scripts|allow-same-origin/.test(f.attrs.sandbox)) {
      fail('a frame that runs scripts, or has this origin, is the bug');
    }
    eq('a counted load is not a read of this page', f.attrs.referrerpolicy,
       'no-referrer');
    eq('a mail says what it is', f.attrs.title,
       "the message's own HTML, with scripts disabled");
    eq('the document, assigned', f._srcdoc, '<p>hallo</p>');
    eq('permissions exist before the document does',
       writes.join(','), 'title,sandbox,referrerpolicy,srcdoc');

    // --- the answer it is built from ----------------------------------------
    const reply = (body, ok, status) => ({
      ok: ok !== false, status: status || 200, statusText: 'Conflict',
      json: () => Promise.resolve(body),
    });
    let n = 0;
    answer = reply({html: '<p>ja</p>'});
    Promise.resolve()
      .then(() => MailHtml.load('m/1').then((data) => {
        eq('the body is the servers, not a copy', data.html, '<p>ja</p>');
        eq('one message, one address', requests[0], '/api/message/m%2F1/html');
        n = requests.length;
        return MailHtml.load('m/1').then(() =>
          eq('a second look at the same view is not a request', requests.length, n));
      }))
      .then(() => {
        n = requests.length;
        answer = reply({html: '<p>met plaatjes</p>'});
        return MailHtml.load('m/1', true).then(() =>
          eq('images are asked for, not assumed',
             requests[n], '/api/message/m%2F1/html?images=1'));
      })
      .then(() => {
        // A message id is not a path segment: nothing in it may reach the route.
        n = requests.length;
        answer = reply({error: 'deze mail heeft geen html-part', reason: 'no_html'},
                       false, 404);
        return MailHtml.load('../x').then(
          () => fail('a refusal must not resolve'),
          (err) => {
            eq('address of a hostile id', requests[n], '/api/message/..%2Fx/html');
            eq('the sentence a person can read', err.message,
               'deze mail heeft geen html-part');
            // The switch is disabled on this word, so it has to survive the wire.
            eq('and the reason, as a field', err.reason, 'no_html');
          });
      })
      .then(() => {
        // A server that answers nothing parseable still has to say something.
        answer = {ok: false, status: 409, statusText: 'Conflict',
                  json: () => Promise.reject(new Error('not json'))};
        return MailHtml.load('r').then(
          () => fail('a refusal must not resolve'),
          (err) => {
            eq('a body that is not JSON still has a status', err.message,
               '409 Conflict');
            eq('and no reason invented', err.reason, '');
          });
      })
      .then(() => {
        // Bounded: the pane holds the last few bodies, not every mail opened
        // since the page loaded, and it is the oldest one that goes.
        const n0 = requests.length;
        const ids = [];
        for (let i = 0; i < 9; i++) ids.push('m' + i);
        answer = reply({html: '<p>x</p>'});
        return ids.reduce((p, id) => p.then(() => MailHtml.load(id)),
                          Promise.resolve())
          .then(() => eq('nine mails, nine bodies', requests.length, n0 + 9))
          .then(() => MailHtml.load('m8'))
          .then(() => eq('the newest body is still there',
                         requests.length, n0 + 9))
          .then(() => MailHtml.load('m0'))
          .then(() => eq('the oldest was let go', requests.length, n0 + 10));
      })
      .then(() => console.log('OK'), fail);
    """

    def test_the_frame_the_browser_ends_up_with(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        script = gui.WEB_DIR / "mailhtml.js"
        with tempfile.TemporaryDirectory() as td:
            runner = Path(td) / "frame.js"
            runner.write_text(self.SCRIPT, encoding="utf-8")
            try:
                done = subprocess.run([node, str(runner), str(script)],
                                      capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as e:
                self.skipTest(f"cannot run node here: {e}")
        self.assertEqual(done.returncode, 0, done.stderr[:1500])
        self.assertEqual(done.stdout.strip(), "OK")



class TestImageShrinkMath(unittest.TestCase):
    """The arithmetic behind "your photo was too big, so we drew it smaller".

    A 24 MB phone photo used to be refused with a message explaining that it
    would have been resized to 1024 px anyway — a promise nothing kept, and the
    reason a large pick looked like a dead search bar. The browser now does that
    reduction itself, so the two numbers it computes are worth pinning: the size
    it reduces to, and the byte count it judges the cap against.

    Node runs the *real* functions, lifted out of `web/app.js`, so this is the
    shipped arithmetic under test and not a copy of it.
    """

    SCRIPT = r"""
    const src = require('fs').readFileSync(process.argv[2], 'utf8');
    const grab = (name) => {
      const m = src.match(new RegExp('function ' + name + '\\s*\\([^)]*\\)\\s*\\{[\\s\\S]*?\\n\\}'));
      if (!m) { console.error('web/app.js no longer defines ' + name); process.exit(1); }
      return new Function(m[0] + '; return ' + name + ';')();
    };
    const fitSide = grab('fitSide'), dataBytes = grab('dataBytes');
    const eq = (what, got, want) => {
      if (String(got) !== String(want)) {
        console.error(what + ': got ' + got + ', want ' + want);
        process.exit(1);
      }
    };
    eq('landscape 4032x3024', fitSide(4032, 3024, 1024), '1024,768');
    eq('portrait 3024x4032', fitSide(3024, 4032, 1024), '768,1024');
    eq('square 2000x2000', fitSide(2000, 2000, 1024), '1024,1024');
    eq('already small, never up-scaled', fitSide(800, 600, 1024), '800,600');
    eq('exactly the cap', fitSide(1024, 1024, 1024), '1024,1024');
    eq('one pixel', fitSide(1, 1, 1024), '1,1');
    eq('nothing to draw', fitSide(0, 0, 1024), '1,1');
    eq('a long thin strip', fitSide(9000, 300, 1024), '1024,34');
    eq('bytes behind 5 bytes of base64',
       dataBytes('data:image/png;base64,' + Buffer.from('hello').toString('base64')), 5);
    eq('an empty payload', dataBytes('data:image/png;base64,'), 0);
    eq('96 kB of pixels',
       dataBytes('data:image/jpeg;base64,'
                 + Buffer.alloc(96 * 1024, 7).toString('base64')), 96 * 1024);
    console.log('OK');
    """

    def test_the_reduction_arithmetic(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        app = gui.WEB_DIR / "app.js"
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "fit.js"
            script.write_text(self.SCRIPT, encoding="utf-8")
            try:
                done = subprocess.run([node, str(script), str(app)],
                                      capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as e:
                self.skipTest(f"cannot run node here: {e}")
        self.assertEqual(done.returncode, 0, done.stderr[:900])
        self.assertEqual(done.stdout.strip(), "OK")

    def test_an_oversized_pick_is_reduced_and_not_refused(self):
        """The refusal this replaced told the user the image was going to be
        resized anyway. Either do it or do not say it; here it is done."""
        src = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("it is resized to 1024 px on the way to the model anyway",
                         src, "a toast promising a resize the front end never did")
        self.assertIn("const MAX_IMAGE_SIDE = 1024;", src,
                      "the client must reduce to the side mailbox.resize_image_b64 uses")
        self.assertIn("await shrinkImage(file)", src, "setImage must do the reduction")
        self.assertIn("cv.toBlob", src)
        self.assertIn("is-busy", src, "the chip has to look busy while it works")

    def test_the_reduced_image_is_the_one_that_is_sent(self):
        """Silently searching with a different picture than the file the user
        picked is how a search tool loses trust, so the chip says what it sent."""
        src = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("reduced from", src)
        self.assertIn("KB sent", src)


class TestNoSpeculativeHints(unittest.TestCase):
    """A limitation is said where it bites, and a failure ends in a move.

    The reader pane used to carry a paragraph under every attachment list saying
    that attachment text is searchable, that only some files have a preview, and
    that opening one needs the Gmail account reachable: three answers to
    questions nobody had asked yet, printed on every mail, in the same pane where
    the real failure is reported once it actually happens. Same species as the
    refusal at `test_an_oversized_pick_is_reduced_and_not_refused` — a sentence
    about a state the reader is not in is not information, it is noise that reads
    as a warning. These pin the shipped front end to the rule, in both
    directions: nothing speculative, and the failures that matter still spoken.
    """

    @classmethod
    def setUpClass(cls):
        cls.js = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        cls.py = Path(gui.__file__).read_text(encoding="utf-8")

    def test_an_ordinary_mail_explains_nothing(self):
        for gone in (
            "needs the account reachable",       # a failure that has not happened
            "indexed and searchable",            # the row tag already says this
            "media --backfill",                  # a maintenance command in a mail
            "the 300-character preview",         # the heading already reads preview
            "max_source_chars",                  # a config key, not an answer
            "searchable, never embedded",        # index design used as a heading
            "not searchable",                    # a badge apologising for the index
            "kept nowhere here",
            "the only place the whole file is asked for",
            "set gmail_account_index in config.json",
        ):
            self.assertNotIn(gone, self.js, f"the reader still explains: {gone}")

    def test_no_sentence_sends_the_reader_elsewhere(self):
        for gone in ("the status pill says",
                     "Raise the floor in the knobs",
                     "bm25 has no words to match a photo",
                     "is not the problem here",
                     "the log says why"):
            self.assertNotIn(gone, self.js,
                             f"a sentence that hands the reader an errand: {gone}")

    def test_the_failures_that_are_real_still_speak(self):
        for kept in ("this message could not be read",     # the mail itself is gone
                     "could not open ",                    # the file did not arrive
                     "the search did not run",
                     "the list did not load",
                     "search by picture takes PNG",        # refused, with the set
                     "a search image may be",              # still over after shrinking
                     "cannot read a picture",              # endpoint said no, at the pick
                     "the closest mail, not an answer"):   # abstention, stated
            self.assertIn(kept, self.js,
                          f"a real failure lost its sentence: {kept}")

    def test_a_server_refusal_ends_in_a_move(self):
        """Each doomed search is refused in one sentence with one thing to do;
        the second half of the old sentences explained the other leg instead."""
        for want in ("use hybrid or semantic",
                     "search by words instead",
                     "search by words until it answers",
                     "media --backfill keeps them"):
            self.assertIn(want, self.py, f"a refusal no longer offers a way on: {want}")
        for gone in ("the keyword leg that works ",
                     "a keyword search would ignore it",
                     "it is either not an image this index could read",
                     "is keywords alone"):
            self.assertNotIn(gone, self.py, f"the refusal lectures again: {gone}")


class TestGmailAccountSlot(Fixture):
    """`#all/<id>` opens in whichever signed-in account the browser is showing,
    so every link leaving this API has to carry the configured
    `gmail_account_index` — links indexed before it was set included, which is
    why the slot is rewritten on the way out instead of at index time."""

    WANT = "https://mail.google.com/mail/u/3/#all/1ab23c"

    def test_browsing_links_carry_the_configured_slot(self):
        items = self.get_json("/api/messages?limit=50")["items"]
        by_id = {m["message_id"]: m["gmail_url"] for m in items}
        self.assertEqual(by_id["bob"], self.WANT)
        self.assertEqual(by_id["shop"], "")          # nothing stored, nothing invented

    def test_the_detail_link_carries_it(self):
        r = self.client.get("/api/message/bob")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["gmail_url"], self.WANT)

    def test_a_search_hit_carries_it(self):
        """Search reads its links out of the in-memory index arrays, not the row
        readers the panes use, so it needs its own assertion to be covered."""
        r = self.post_search({"q": "factuur", "min_relevance": 0})
        self.assertEqual(r.status_code, 200, r.text)
        hits = [h for h in r.json()["results"] if h["message_id"] == "bob"]
        self.assertTrue(hits, r.json())
        self.assertEqual(hits[0]["gmail_url"], self.WANT)

    def test_the_status_tells_the_front_end_which_slot_to_write(self):
        self.assertEqual(self.get_json("/api/status")["gmail_account_index"], 3)

    def test_the_front_end_takes_the_number_from_the_server(self):
        js = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("gmail_account_index", js)
        self.assertNotIn("mail/u/0/", js)            # no slot baked into the page


class TestHiddenIsAuthoritative(unittest.TestCase):
    """The other bug class no test here can see: nothing in this file opens a
    browser, so a rule the browser applies *incorrectly* looks exactly like a
    rule it applies correctly. This one is worth pinning because it is a
    property of CSS rather than of the code: the browser's own
    `[hidden] { display: none }` sits in the user-agent sheet, and any author
    rule outranks that sheet regardless of selector weight, so a class setting
    `display` silently un-hides every element `app.js` hides with `.hidden`."""

    def test_hidden_beats_every_display_rule_it_meets(self):
        js = (gui.WEB_DIR / "app.js").read_text(encoding="utf-8")
        css = (gui.WEB_DIR / "app.css").read_text(encoding="utf-8")
        html = (gui.WEB_DIR / "index.html").read_text(encoding="utf-8")

        # Which elements does the script hide? Both `$('id').hidden` and the
        # aliased form (`const t = $('tuning'); t.hidden = …`).
        alias = dict(re.findall(r"(?:const|let)\s+(\w+)\s*=\s*\$\(['\"]([\w-]+)['\"]\)", js))
        toggled: set[str] = set()
        for stmt in re.split(r"[;\n]", js):
            if ".hidden" not in stmt:
                continue
            toggled |= set(re.findall(r"\$\(['\"]([\w-]+)['\"]\)\.hidden", stmt))
            for name, el_id in alias.items():
                if name + ".hidden" in stmt:
                    toggled.add(el_id)
        self.assertIn("reader-empty", toggled)
        self.assertIn("tuning", toggled)

        # Of those, which carry a class that sets `display`? Those are the ones
        # the browser will refuse to hide without the rule below — the empty
        # state stayed over the opened message, and the knobs panel could only
        # be opened, for exactly this reason.
        blocked = []
        for el_id in sorted(toggled):
            for tag in re.findall(r"<[^>]*\bid=\"%s\"[^>]*>" % re.escape(el_id), html):
                for cls in " ".join(re.findall(r'class="([^"]*)"', tag)).split():
                    for block in re.findall(r"\.%s\b[^{]*\{([^}]*)\}" % re.escape(cls), css):
                        if re.search(r"^\s*display:", block, re.M):
                            blocked.append(".%s (#%s)" % (cls, el_id))
        self.assertIn(".reader-empty (#reader-empty)", blocked, blocked)
        self.assertIn(".tuning (#tuning)", blocked, blocked)
        self.assertRegex(css, r"\[hidden\]\s*\{\s*display:\s*none[^}]*!important",
                         "no rule making `hidden` outrank the sheet's display rules")


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------

class FakeProc:
    def __init__(self, gate=None, code=0):
        self.gate = gate
        self.code = code

    def wait(self):
        if self.gate is not None:
            self.gate.wait(10)
        return self.code


@unittest.skipIf(TestClient is None, NEEDS_CLIENT)
class TestRefresh(Fixture):
    def setUp(self):
        super().setUp()
        self.log = Path(self.tmp.name) / "gui-index.log"
        patcher = mock.patch.object(gui, "REFRESH_LOG", self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def wait_until_stopped(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.client.get("/api/index/refresh").json()["running"]:
                return self.client.get("/api/index/refresh").json()
            time.sleep(0.05)
        self.fail("the fake index run never finished")

    def test_second_refresh_is_refused_while_one_runs(self):
        gate = threading.Event()
        with mock.patch.object(gui.subprocess, "Popen",
                               lambda *a, **k: FakeProc(gate)):
            started = self.client.post("/api/index/refresh")
            self.assertEqual(started.status_code, 200, started.text)
            self.assertTrue(started.json()["started"])
            again = self.client.post("/api/index/refresh")
            self.assertEqual(again.status_code, 409)
            self.assertIn("already", again.json()["error"])
            self.assertTrue(self.client.get("/api/index/refresh").json()["running"])
            gate.set()
        self.wait_until_stopped()

    def test_log_is_tailed_from_disk_and_the_engine_reopens(self):
        """The subprocess writes its own progress; the panel reads the file, and
        a successful run must re-open the engine — a warm snapshot of yesterday's
        corpus would otherwise answer every query after the refresh."""
        def fake_popen(cmd, **kwargs):
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write("fetching mail\rindexed 3 messages\n")
            return FakeProc(code=0)

        with mock.patch.object(gui.subprocess, "Popen", fake_popen):
            self.client.post("/api/index/refresh")
        done = self.wait_until_stopped()
        self.assertEqual(done["exit_code"], 0)
        self.assertIn("indexed 3 messages", done["log"])
        self.assertTrue(self.engine.ready)
        st = self.get_json("/api/status")
        self.assertEqual(st["messages"], 3)

    def test_a_failing_run_is_reported_not_swallowed(self):
        with mock.patch.object(gui.subprocess, "Popen", lambda *a, **k: FakeProc(code=2)):
            self.client.post("/api/index/refresh")
        done = self.wait_until_stopped()
        self.assertEqual(done["exit_code"], 2)
        self.assertTrue(self.engine.ready)               # old snapshot still valid

    def test_unstartable_child_is_a_409(self):
        def boom(*a, **k):
            raise OSError("no python here")
        with mock.patch.object(gui.subprocess, "Popen", boom):
            r = self.client.post("/api/index/refresh")
        self.assertEqual(r.status_code, 409)
        self.assertIn("could not start", r.json()["error"])
        self.assertFalse(self.client.get("/api/index/refresh").json()["running"])


class TestTailLog(unittest.TestCase):
    def test_carriage_returns_become_rows_and_only_the_tail_survives(self):
        """tqdm redraws one line with \\r; in a file that is one enormous line."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            p = Path(d) / "log"
            p.write_text("start\n" + "\r".join(f"step {i}" for i in range(200)) + "\n",
                         encoding="utf-8")
            out = gui.tail_log(p, lines=5)
            self.assertEqual(out.split("\n")[-1], "step 199")
            self.assertEqual(len(out.split("\n")), 5)
            self.assertNotIn("start", out)

    def test_a_missing_log_is_empty_not_an_error(self):
        self.assertEqual(gui.tail_log(Path(tempfile.gettempdir()) / "nope-42.log"), "")


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

class TestCliWiring(StubbedEndpoint):
    make_store = False

    def test_gui_subcommand_is_registered(self):
        """`--help` in-process: registration is what breaks when a subparser is
        added in the wrong place, and argparse exits before any config loads."""
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["mailbox.py", "gui", "--help"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(buf):
                mb.main()
        self.assertEqual(cm.exception.code, 0)
        help_text = buf.getvalue()
        self.assertIn("--port", help_text)
        self.assertIn("--no-browser", help_text)

    def test_cmd_gui_hands_the_loaded_config_to_the_server(self):
        seen = {}
        args = argparse.Namespace(config=self.cfg_path, profile="", db=self.db,
                                  port=8765, host="127.0.0.1", top=None,
                                  no_browser=True)

        def fake_run(run_args, cfg):
            seen["db"] = run_args.db
            seen["cfg"] = cfg

        with mock.patch.object(gui, "run", fake_run):
            mb.cmd_gui(args)
        self.assertEqual(seen["cfg"]["db_path"], self.db)
        self.assertEqual(seen["db"], self.db)

    def test_engine_survives_an_unopenable_database(self):
        """`Store` and `run()` signal with SystemExit — the way a CLI does it.
        Opening must not take the process with it, so /api/status can say what
        is wrong instead of a browser showing a dead tab.

        The warning is asserted, not just tolerated: a failure that is neither
        raised nor logged is invisible, and an unreadable mailbox is the one
        state where the user needs the reason on screen.
        """
        cfg = make_cfg(self.tmp.name)               # a directory is not a database
        eng = gui.Engine(cfg)
        self.addCleanup(eng.close)
        with self.assertLogs("gui", level="WARNING") as logged:
            status = eng.open()
        self.assertFalse(status["ready"])
        self.assertIn("unable to open database file", status["error"])
        self.assertIn("unable to open database file", "\n".join(logged.output))
        if TestClient is None:
            self.skipTest(NEEDS_CLIENT)
        client = TestClient(gui.create_app(eng))
        r = client.post("/api/search", json={"q": "x"})
        self.assertEqual(r.status_code, 409)
        self.assertTrue(r.json()["error"])
        self.assertIn("unable to open database file", r.json()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
