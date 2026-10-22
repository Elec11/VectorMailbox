#!/usr/bin/env python3
"""
Tests for the hybrid + multi-resolution indexer.

Stdlib unittest only (no pytest in requirements.txt), and no network: the
embedding endpoint is stubbed by replacing EmbeddingClient._post with a
deterministic bag-of-words vector, so chunking, the FTS index, the fusion
math and both commands are exercised end to end without a model.

  python -m unittest test_mailbox -v
"""
from __future__ import annotations

import base64
import contextlib
import imaplib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

import mailbox as mb
import queryparse as qp

DIM = 96
_STUB_TOKENS = re.compile(r"\w+", re.UNICODE)


def fake_vec(text: str) -> np.ndarray:
    """Cheap stand-in for EmbeddingGemma: a hashed bag of words.

    Shared words -> shared direction -> cosine similarity behaves the way the
    real model behaves for keyword overlap, which is all the fusion tests need.
    """
    v = np.zeros(DIM, dtype=np.float32)
    for tok in _STUB_TOKENS.findall((text or "").lower()):
        v[sum(tok.encode("utf-8")) % DIM] += 1.0
    return v


def fake_post(self, payload, multimodal=False):
    out = []
    for item in payload["input"]:
        if isinstance(item, str):
            text = item
        else:  # media item: its text label (or its type as a last resort)
            text = item.get("text") or item.get("type", "media")
        out.append({"embedding": fake_vec(text).tolist()})
    return {"data": out}


def image_url_of(item) -> str:
    """The data URL inside a media item, in either envelope, or '' if none."""
    parts = [item] + list(item.get("content") or [])
    for p in parts:
        url = (p.get("image_url") or {}).get("url")
        if url:
            return url
    return ""


#: Width of `pixel_vec`. Deliberately not the 96 of `fake_vec`: a base64 blob is
#: one long token, and in 96 dimensions it lands on the same dimension as an
#: ordinary word often enough to give an unrelated body a higher cosine than the
#: picture it has nothing in common with. 768 is the real model's width anyway.
PIXEL_DIM = 768


def pixel_vec(text: str) -> np.ndarray:
    v = np.zeros(PIXEL_DIM, dtype=np.float32)
    for tok in _STUB_TOKENS.findall((text or "").lower()):
        v[sum(tok.encode("utf-8")) % PIXEL_DIM] += 1.0
    return v


def pixel_post(self, payload, multimodal=False):
    """Stub of a *pixel-capable* endpoint: a media item is embedded from its
    image, not from its filename.

    `fake_post` above is a text-only endpoint, and no claim about image search
    can be tested with a stub that ignores the pixels. Words keep hashing as
    before, so this one stub answers both legs of a hybrid search: two files
    with the same bytes get the same vector, and nothing else does.
    """
    out = []
    for item in payload["input"]:
        if isinstance(item, str):
            text = item
        else:
            url = image_url_of(item)
            if url:
                # Only what follows `base64,` is pixels. The envelope
                # (`data:image/png;base64`) is the same for every image, and a
                # stub that hashes it puts every two pictures at cos 0.8.
                text = url.rsplit(",", 1)[-1]
            else:
                text = item.get("text") or "media"
        out.append({"embedding": pixel_vec(text).tolist()})
    return {"data": out}


def make_cfg(db_path, **over):
    cfg = dict(mb.DEFAULTS)
    cfg.update({
        "db_path": db_path,
        "embed_endpoint": "http://stub.invalid/v1/embeddings",
        "embed_api_key": "",
        "embed_model": "",
        "batch_size": 8,
        "gmail_user": "u@example.com",
        "gmail_app_password": "x",
    })
    cfg.update(over)
    return cfg


def make_msg(mid="m1", body="hello world", sender="ada@example.com",
             subject="weekly status", date=None, to="me@example.com", cc="",
             atts=(), gmail_url=""):
    return mb.Message(
        message_id=mid, folder="INBOX",
        date=date or datetime(2026, 5, 4, tzinfo=timezone.utc),
        sender=sender, to=to, cc=cc, subject=subject, body_text=body,
        attachments=list(atts), gmail_url=gmail_url,
    )


def index_messages(store, cfg, msgs, media=None):
    extractor = mb.ContentExtractor(cfg)
    client = mb.EmbeddingClient(cfg)
    return mb._index_batch(list(msgs), store, extractor, client,
                           int(cfg["thread_context_depth"]),
                           int(cfg["thread_context_chars"]), media=media)


def search(store, cfg, query, **over):
    params = mb.SearchParams(**over)
    searcher = mb.Searcher(cfg, store, mb.EmbeddingClient(cfg))
    return searcher.run(query, params)


class StubbedEndpoint(unittest.TestCase):
    """Base class that swaps the endpoint transport for the hashed stub."""

    #: subclasses that need a pre-existing (legacy) database skip this
    make_store = True

    @classmethod
    def setUpClass(cls):
        cls._real_post = mb.EmbeddingClient._post
        mb.EmbeddingClient._post = fake_post

    @classmethod
    def tearDownClass(cls):
        mb.EmbeddingClient._post = cls._real_post

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.stores = []
        self.db = os.path.join(self.tmp.name, "test.db")
        self.cfg = make_cfg(self.db)
        self.cfg_path = os.path.join(self.tmp.name, "config.json")
        with open(self.cfg_path, "w", encoding="utf-8") as fh:
            json.dump(self.cfg, fh)
        self.store = self.open_store() if self.make_store else None

    def open_store(self, cfg=None):
        """A Store, tracked so its sqlite connection is closed on teardown -
        Windows refuses to delete a temp dir holding an open file."""
        store = mb.Store(self.db, cfg or self.cfg)
        self.stores.append(store)
        return store

    def tearDown(self):
        for store in self.stores:
            store.conn.close()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking(unittest.TestCase):

    def test_windows_stay_within_size_and_lose_nothing(self):
        text = " ".join(f"sentence number {i} talks about the flexa project"
                        for i in range(40))
        wins = mb.window_text(text, 320, 48)
        self.assertGreater(len(wins), 2)
        for w in wins:
            self.assertLessEqual(len(w), 320 + 64, "window oversized")
        joined = " ".join(wins)
        for i in (0, 13, 39):                      # first, middle, last
            self.assertIn(f"sentence number {i}", joined)

    def test_windows_start_on_block_boundaries(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        wins = mb.window_text(text, 20, 0)
        for w in wins:
            self.assertTrue(w == w.strip())
            self.assertFalse(w.startswith(" "), "window starts mid-whitespace")

    def test_hard_split_keeps_every_word(self):
        s = "a" * 500 + " tail"
        parts = mb.hard_split(s, 100)
        self.assertTrue(any(p.endswith("tail") for p in parts))

    def test_short_text_is_one_chunk_not_three(self):
        rows = mb.tiered_chunks("a short note", {"small": 320, "large": 1200,
                                                 "xlarge": 6000})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tier"], "small")
        self.assertTrue(rows[0]["canonical"])

    def test_tiers_grow_with_length(self):
        sizes = {"small": 320, "large": 1200, "xlarge": 6000}
        one = mb.tiered_chunks("x" * 300, sizes)
        self.assertEqual([r["tier"] for r in one], ["small"])

        two = mb.tiered_chunks(("word " * 200).strip(), sizes)      # ~1000 chars
        self.assertEqual({r["tier"] for r in two}, {"small", "large"})
        canon = [r for r in two if r["canonical"]]
        self.assertEqual(len(canon), 1)
        self.assertEqual(canon[0]["tier"], "large")
        self.assertEqual(canon[0]["nwin"], 1)

        three = mb.tiered_chunks(("word " * 600).strip(), sizes)    # ~3000 chars
        self.assertEqual({r["tier"] for r in three}, {"small", "large", "xlarge"})
        self.assertEqual([r for r in three if r["canonical"]][0]["tier"], "xlarge")

    def test_no_window_duplicates_another_chunk(self):
        sizes = {"small": 320, "large": 1200, "xlarge": 6000}
        # Deliberately repetitive, like a mailed report with a repeated footer:
        # identical windows must be dropped, not embedded twice.
        text = ("alpha beta gamma delta. " * 90).strip()            # ~2430 chars
        rows = mb.tiered_chunks(text, sizes)
        texts = [r["text"] for r in rows]
        self.assertEqual(len(texts), len(set(texts)), "a chunk is stored twice")
        canon = [r for r in rows if r["canonical"]][0]["text"]
        for r in rows:
            if not r["canonical"]:
                self.assertLess(len(r["text"]), len(canon))

    def test_repetitive_source_costs_fewer_chunks(self):
        sizes = {"small": 320, "large": 1200, "xlarge": 6000}
        repetitive = mb.tiered_chunks(("same old footer. " * 200).strip(), sizes)
        varied = mb.tiered_chunks(
            " ".join(f"line {i} says something different about item {i * 7}."
                     for i in range(120)), sizes)
        self.assertLess(len(repetitive), len(varied))

    def test_positions_are_contiguous_per_tier(self):
        sizes = {"small": 320, "large": 1200, "xlarge": 6000}
        text = " ".join(f"sentence {i} describes thing {i * 3}." for i in range(200))
        rows = mb.tiered_chunks(text, sizes)
        for tier in {r["tier"] for r in rows}:
            wins = sorted([r for r in rows if r["tier"] == tier], key=lambda r: r["pos"])
            self.assertEqual([r["pos"] for r in wins], list(range(len(wins))))
            self.assertEqual({r["nwin"] for r in wins}, {len(wins)})

    def test_canonical_holds_the_complete_capped_source(self):
        sizes = {"small": 320, "large": 1200, "xlarge": 6000}
        text = ("lorem ipsum dolor sit amet. " * 400).strip()      # > 6000
        rows = mb.tiered_chunks(text, sizes, max_source_chars=6000)
        canon = [r for r in rows if r["canonical"]][0]
        self.assertEqual(canon["nwin"], 1)
        self.assertEqual(len(canon["text"]), 6000)
        self.assertTrue(text.startswith(canon["text"][:200]))

    def test_empty_text_yields_nothing(self):
        self.assertEqual(mb.tiered_chunks("   \n  ", {"small": 320}), [])

    def test_overlap_repeats_context_without_oversizing(self):
        text = ("one two three four five six seven eight nine ten. " * 30).strip()
        no_ov = mb.window_text(text, 200, 0)
        with_ov = mb.window_text(text, 200, 60)
        self.assertGreater(sum(len(w) for w in with_ov),
                           sum(len(w) for w in no_ov))

    def test_label_says_whether_it_is_the_whole_source(self):
        # Bare label == canonical row == complete indexed text (the pre-tier
        # shape). A window says so even when it collapsed to a single window.
        self.assertEqual(mb.chunk_label("body", "xlarge", 0, 1, canonical=True), "body")
        self.assertEqual(mb.chunk_label("body", "small", 2, 9), "body [small 3/9]")
        self.assertEqual(mb.chunk_label("body", "small", 0, 1), "body [small 1/1]")


class TestFtsQueryBuilder(unittest.TestCase):

    def test_prefix_forms_apply_to_long_tokens(self):
        """A prefix match is per token, not 'last token only': a typo in the
        middle of a query is just as likely, and a 3-letter token expanding to
        a prefix match only adds noise."""
        self.assertEqual(mb.build_fts_query("flexa invoice", variants=False),
                         "(flexa OR flexa*) AND (invoice OR invoice*)")
        self.assertEqual(mb.build_fts_query("flexa invoice", prefix=False,
                                            variants=False),
                         "flexa AND invoice")
        self.assertEqual(mb.build_fts_query("ab cd extended", join=" OR ",
                                            variants=False),
                         "ab OR cd OR (extended OR extended*)")

    def test_word_forms_are_joined_at_query_time(self):
        """FTS5 has no stemmer here (porter was dropped: it only stems English,
        so it split factuur/facturen instead of joining them), so the query
        carries its own variants."""
        q = mb.build_fts_query("facturen")
        self.assertIn("factuur", q)
        self.assertIn(" OR ", q)
        q2 = mb.build_fts_query("domeinen")
        self.assertIn("domein", q2)

    def test_stopwords_do_not_become_required_terms(self):
        """No FTS stoplist exists, so a required 'van'/'de' silently zeroed
        recall for Dutch queries."""
        self.assertNotIn("van", mb.build_fts_query("domein van de verhuizing"))
        self.assertNotIn(" de ", mb.build_fts_query("domein van de verhuizing"))
        # a query of nothing but stopwords still searches for something
        all_sw = mb.build_fts_query("van de het", variants=False)
        self.assertEqual(all_sw, "van AND de AND het")

    def test_operators_cannot_leak_into_the_expression(self):
        for nasty in ('a" OR b', "inv(oice)*", "NOT x", "a: b -c ^d", "NEAR(a b)"):
            q = mb.build_fts_query(nasty)
            bare = re.sub(r'"[^\"]*"', "", q)          # quoted keywords are fine
            bare = re.sub(r"\b(AND|OR)\b", "", bare)   # explicit joins are fine
            bare = re.sub(r"\s*[*()]\s*", "", bare)    # grouping + one prefix op
            self.assertFalse(re.search(r"""[":^']""", bare),
                             f"an fts operator leaked: {nasty!r} -> {q!r}")
            conn = sqlite3.connect(":memory:")         # and fts5 must accept it
            conn.execute("create virtual table t using fts5(x)")
            conn.execute("insert into t(x) values('a b invoice')")
            conn.execute("select rowid from t where t match ?", (q,)).fetchall()
            conn.close()

    def test_keyword_tokens_get_quoted(self):
        self.assertEqual(mb.build_fts_query("and", variants=False), '"and"')
        self.assertEqual(mb.build_fts_query("or near", variants=False),
                         '"or" AND "near"')

    def test_punctuation_only_query_is_empty(self):
        self.assertEqual(mb.build_fts_query("!!! ???"), "")
        self.assertEqual(mb.build_fts_query(""), "")

    def test_long_queries_are_capped(self):
        q = mb.build_fts_query(" ".join(f"w{i}" for i in range(50)))
        self.assertEqual(len(q.split(" AND ")), 12)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtractor(StubbedEndpoint):

    def test_body_gets_three_tiers_when_long(self):
        cfg = make_cfg(self.db)
        ex = mb.ContentExtractor(cfg)
        body = ("the flexa project invoice is due next week. " * 70).strip()
        chunks = ex.extract(make_msg(body=body), "")
        kinds = {c["kind"] for c in chunks}
        self.assertEqual(kinds, {"body", "meta"})
        self.assertEqual({c["tier"] for c in chunks if c["kind"] == "body"},
                         {"small", "large", "xlarge"})
        meta = [c for c in chunks if c["kind"] == "meta"]
        self.assertEqual(len(meta), 1, "short meta must not be windowed")

    def test_short_body_is_a_single_chunk(self):
        ex = mb.ContentExtractor(make_cfg(self.db))
        chunks = ex.extract(make_msg(body="ping"), "")
        self.assertEqual([c["label"] for c in chunks], ["body", "meta"])
        self.assertEqual([c["tier"] for c in chunks], ["small", "small"])

    def test_media_chunk_is_its_own_tier(self):
        ex = mb.ContentExtractor(make_cfg(self.db))
        png = (b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        chunks = ex.extract(make_msg(body="see image", atts=[
            mb.Attachment("shot.png", "image/png", png)]), "")
        media = [c for c in chunks if c["kind"] == "attachment"]
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0]["tier"], mb.MEDIA_TIER)
        self.assertEqual(media[0]["nwin"], 1)
        self.assertTrue(media[0]["canonical"])
        self.assertIn("image_b64", media[0])

    def test_index_tiers_config_skips_derived_tiers(self):
        cfg = make_cfg(self.db, index_tiers=["large"])
        ex = mb.ContentExtractor(cfg)
        body = ("some longish body text with several sentences. " * 60).strip()
        chunks = ex.extract(make_msg(body=body), "")
        body_tiers = {c["tier"] for c in chunks if c["kind"] == "body"}
        self.assertIn("xlarge", body_tiers, "canonical row must survive filtering")
        self.assertIn("large", body_tiers)
        self.assertNotIn("small", body_tiers)


# ---------------------------------------------------------------------------
# Storage + FTS
# ---------------------------------------------------------------------------

class TestStore(StubbedEndpoint):

    def test_commit_writes_tiers_meta_and_fts(self):
        index_messages(self.store, self.cfg,
                       [make_msg(body="the invoice for the flexa project")])
        s = self.store.stats()
        self.assertEqual(s["messages"], 1)
        self.assertGreaterEqual(s["chunks"], 2)
        self.assertEqual(s["fts_docs"], s["chunks"], "every chunk needs an fts row")
        self.assertGreaterEqual(s["vectors"], s["chunks"])

    def test_bm25_finds_the_exact_string_the_vectors_might_miss(self):
        msgs = [
            make_msg("m1", body="please find the invoice 2026-0041 attached thanks",
                     subject="invoice 2026-0041", sender="finance@flexa.nl"),
            make_msg("m2", body="lunch tomorrow? somewhere nice maybe",
                     subject="lunch", sender="pim@example.com"),
            make_msg("m3", body="the quarterly numbers look fine to me overall",
                     subject="numbers", sender="ana@example.com"),
        ]
        index_messages(self.store, self.cfg, msgs)
        hits = self.store.lexical_search("2026-0041")
        self.assertTrue(hits, "bm25 found nothing")
        cid = hits[0][0]
        mid = self.store.conn.execute(
            "SELECT message_id FROM chunks WHERE id=?", (cid,)).fetchone()[0]
        self.assertEqual(mid, "m1")

    def test_metadata_column_is_searchable(self):
        index_messages(self.store, self.cfg, [
            make_msg("m1", body="as agreed, see attached", sender=" accountant@flexa.nl",
                     subject="August retention"),
            make_msg("m2", body="as agreed, see attached", sender="pim@example.com",
                     subject="nothing related here"),
        ])
        hits = self.store.lexical_search("retention")
        ids = {self.store.conn.execute("SELECT message_id FROM chunks WHERE id=?",
                                       (cid,)).fetchone()[0] for cid, _ in hits}
        self.assertEqual(ids, {"m1"})

    def test_tier_filter_applies_to_the_lexical_leg(self):
        body = ("the invoice for the flexa project is due. " * 60).strip()
        index_messages(self.store, self.cfg, [make_msg("m1", body=body)])
        all_hits = self.store.lexical_search("flexa")
        small_only = self.store.lexical_search("flexa", tiers=["small"])
        self.assertTrue(all_hits)
        self.assertTrue(small_only)
        self.assertLessEqual(len(small_only), len(all_hits))
        tiers = {self.store.conn.execute("SELECT tier FROM chunks WHERE id=?",
                                         (cid,)).fetchone()[0] for cid, _ in small_only}
        self.assertEqual(tiers, {"small"})

    def test_prefix_match_is_what_finds_a_partial_word(self):
        index_messages(self.store, self.cfg,
                       [make_msg("m1", body="we will invoice the client on friday")])
        # 'invoi' is not in the index (porter stems 'invoice' to 'invoic'), so
        # only the trailing prefix term can find it - that is what --no-prefix
        # gives up.
        self.assertTrue(self.store.lexical_search("invoi"))
        self.assertEqual(self.store.lexical_search("invoi", prefix=False), [])
        self.assertTrue(self.store.lexical_search("invoice"))     # stemmed match

    def test_hostile_query_text_does_not_raise(self):
        index_messages(self.store, self.cfg, [make_msg("m1", body="a normal note")])
        for q in ('"unbalanced', "a AND", "NEAR(x y)", "*)(", "", "   ", "AND OR NOT"):
            self.assertIsInstance(self.store.lexical_search(q), list)

    def test_reindexing_a_message_replaces_its_chunks(self):
        msg = make_msg("m1", body="first version of the story about the shed")
        index_messages(self.store, self.cfg, [msg])
        first = self.store.stats()["chunks"]
        msg2 = make_msg("m1", body="second version of the story about the "
                                   "shed, now with many more words and several "
                                   "sentences so that it is chunked into more "
                                   "pieces than before " * 6)
        index_messages(self.store, self.cfg, [msg2])
        s = self.store.stats()
        self.assertEqual(s["messages"], 1)
        self.assertGreater(s["chunks"], first)
        rows = self.store.conn.execute(
            "SELECT COUNT(DISTINCT text) FROM chunks").fetchone()[0]
        total = self.store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertEqual(rows, total, "old chunks survived the re-index")
        self.assertEqual(s["fts_docs"], total)
        self.assertEqual(self.store.lexical_search("version"),
                         self.store.lexical_search("version"))
        ids = {self.store.conn.execute("SELECT message_id FROM chunks WHERE id=?",
                                       (cid,)).fetchone()[0]
               for cid, _ in self.store.lexical_search("second")}
        self.assertEqual(ids, {"m1"}, "the replaced text is still findable")

    def test_empty_fts_index_is_reported_not_fatal(self):
        index_messages(self.store, self.cfg, [make_msg("m1", body="hello there")])
        self.store.conn.execute("DELETE FROM chunks_fts")
        self.store.conn.commit()
        hits, meta = search(self.store, self.cfg, "hello there")
        self.assertEqual(meta["fts_docs"], 0)
        self.assertTrue(hits, "the dense leg alone must still answer")

    def test_clean_repairs_quotes_and_encoded_subjects_from_the_old_index(self):
        """`reindex --clean` exists for a corpus the old extractor wrote: bodies
        that still carry the quote they replied to, and subjects stored as their
        RFC 2047 encoded-word. Both hurt retrieval the same way - text that is
        not the message, indexed as if it were - and both are repairable offline.
        """
        index_messages(self.store, self.cfg, [
            make_msg("old", body="Bedankt, ik ga akkoord.",
                     subject="Alternatif nodigt uit", sender="relatie@alt.nl")])
        subject = "Alternatif nodigt uit"
        # The shape that broke retrieval, built rather than typed out so the
        # test cannot drift from what a real header looks like.
        import base64
        enc = "=?utf-8?b?" + base64.b64encode(subject.encode("utf-8")).decode() + "?="
        dragged = "\n> Op 1 jan schreef Kim: de overeenkomst is bijgevoegd.\n" \
                  "-- \nautomatische mail, reageer niet"
        for cid, text, smeta in self.store.conn.execute(
                "SELECT id, text, search_meta FROM chunks").fetchall():
            # The old extractor wrote the encoded header everywhere the subject
            # is kept, so both columns carry it - and the body carries the
            # whole reply chain, footer included.
            self.store.conn.execute(
                "UPDATE chunks SET text=?, search_meta=? WHERE id=?",
                ((text or "").replace(subject, enc) + dragged,
                 (smeta or "").replace(subject, enc), cid))
        self.store.conn.execute("UPDATE messages SET subject=? WHERE message_id='old'",
                               (enc,))
        self.store.conn.commit()
        self.store.conn.execute("DELETE FROM chunks_fts")   # rebuild from that text
        self.store.conn.commit()
        self.store.sync_fts()
        self.assertFalse(self.store.lexical_search('"nodigt"'),
                         "setup: 'nodigt' sits inside the base64, not searchable")

        hygiene = self.store.clean_stored_text()
        self.assertGreaterEqual(hygiene["changed"], 1)
        self.assertGreater(hygiene["chars"], 0, "the dragged text had to move")
        fixed = self.store.decode_stored_subjects()
        self.assertEqual(fixed["changed"], 1)
        self.assertTrue(fixed["ids"], "a chunk whose title changed must be re-embedded")

        self.assertEqual(self.store.conn.execute(
            "SELECT subject FROM messages WHERE message_id='old'").fetchone()[0],
            "Alternatif nodigt uit")
        # Two things the user can notice: the subject words are searchable, and
        # the quote is still findable in the column it moved to. Cleaning moves
        # text, it does not delete it.
        self.assertTrue(self.store.lexical_search('"nodigt"'))
        self.assertTrue(self.store.lexical_search('"overeenkomst"'))
        text, quoted = self.store.conn.execute(
            "SELECT text, quoted FROM chunks WHERE canonical=1 AND kind='body'"
            " AND message_id='old'").fetchone()
        self.assertNotIn("automatische mail", text)
        self.assertIn("overeenkomst", quoted or "")

        # A second pass has nothing to do: `--clean` can be run again without
        # wondering what it will decide to touch this time.
        again = self.store.clean_stored_text()
        self.assertEqual(again["changed"], 0, "cleaning is not a fixed point")
        # And it only looks at canonical rows: a window is a cut of one of them,
        # so the tier rebuild of the same reindex run re-derives it from the
        # repaired source, and a cut boundary can look like a quote however clean
        # its source is - which is what made the report claim work that was gone.
        canonical = self.store.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE canonical = 1 AND tier != 'media'"
        ).fetchone()[0]
        self.assertEqual(again["scanned"], canonical,
                         "windows are the tier rebuild's job, not the cleaner's")

    def test_a_dry_run_repairs_nothing(self):
        index_messages(self.store, self.cfg, [
            make_msg("old", body="hé daar", subject="x", sender="r@alt.nl")])
        self.store.conn.execute("UPDATE messages SET subject=? WHERE message_id='old'",
                               ("=?UTF-8?Q?vroeg?=",))
        self.store.conn.commit()
        res = self.store.decode_stored_subjects(dry_run=True)
        self.assertEqual(res["changed"], 1, "the dry run still counts what it found")
        self.assertEqual(res["ids"], [])
        self.assertEqual(self.store.conn.execute(
            "SELECT subject FROM messages").fetchone()[0], "=?UTF-8?Q?vroeg?=")

    def test_prune_orphan_vectors_removes_rows_without_a_chunk(self):
        """A tier rebuild replaces derived chunks and leaves their vectors in
        `embeddings`. Nothing reads those rows - the index is loaded through
        `chunks` - but they are 70 MB of file per real mailbox, and a growing
        file with no growing benefit is the kind of thing nobody notices until
        the disk does."""
        index_messages(self.store, self.cfg, [make_msg("m1", body="alive text")])
        live = [r[0] for r in self.store.conn.execute("SELECT id FROM chunks")]
        self.store.conn.execute("INSERT INTO embeddings (chunk_id, vec, dim)"
                                " VALUES (999999, ?, 4)", (b"\x00" * 16,))
        self.store.conn.commit()

        self.assertEqual(self.store.prune_orphan_vectors(), 1)
        left = [r[0] for r in self.store.conn.execute(
            "SELECT chunk_id FROM embeddings ORDER BY chunk_id")]
        self.assertEqual(left, sorted(live), "a real vector was removed")
        self.assertEqual(self.store.prune_orphan_vectors(), 0, "idempotent")


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class TestGmailAccountSlot(StubbedEndpoint):
    """`X-GM-MSGID` gives a message id, but a Gmail web link also names *which
    signed-in account* it belongs to, and only opens in that browser slot: from
    `u/0` the id of a `u/1` message is somebody else's mail or an empty pane."""

    URL = "https://mail.google.com/mail/u/0/#all/1ab23c"

    def test_the_first_account_is_the_default(self):
        self.assertEqual(mb.gmail_index({}), 0)
        self.assertEqual(mb.gmail_index({"gmail_account_index": "1"}), 1)

    def test_nonsense_does_not_move_the_slot(self):
        """A typo must not silently point every link at another account."""
        self.assertEqual(mb.gmail_index({"gmail_account_index": "second"}), 0)
        self.assertEqual(mb.gmail_index({"gmail_account_index": None}), 0)
        self.assertEqual(mb.gmail_index({"gmail_account_index": -3}), 0)

    def test_only_the_slot_moves_and_the_id_stays(self):
        self.assertEqual(
            mb.gmail_account_url(self.URL, 2),
            "https://mail.google.com/mail/u/2/#all/1ab23c")
        self.assertEqual(mb.gmail_account_url(self.URL, 0), self.URL)

    def test_a_link_that_is_not_gmails_is_left_alone(self):
        other = "https://outlook.office.com/mail/u/0/item/AAMK"
        self.assertEqual(mb.gmail_account_url(other, 4), other)
        self.assertEqual(mb.gmail_account_url("", 4), "")

    def test_the_fetcher_writes_the_configured_slot(self):
        """The header is read in-band, so this is the one place the slot is
        *decided*; everything else only rewrites what this produced."""
        import email as emaillib
        fetcher = mb.GmailFetcher.__new__(mb.GmailFetcher)     # no IMAP connection
        fetcher.cfg = {"gmail_account_index": 2, "decode_mime_words": True}
        raw = emaillib.message_from_string(
            "From: ada@example.com\r\nSubject: x\r\n"
            "Date: Mon, 4 May 2026 10:00:00 +0000\r\n"
            "Message-ID: <m1@example.com>\r\n"
            "X-GM-MSGID: 1234567890123\r\n\r\nbody\r\n")
        msg = fetcher._parse(raw, "INBOX")
        self.assertEqual(msg.gmail_url,
                         "https://mail.google.com/mail/u/2/#all/"
                         + hex(1234567890123)[2:])

    def test_an_old_row_reads_back_at_the_new_slot(self):
        """Changing the setting has to fix the links that already exist: the id
        in them is right, and re-embedding a mailbox to move one digit in a URL
        is not a price anyone should pay."""
        stored = "https://mail.google.com/mail/u/0/#all/1ab23c"
        want = "https://mail.google.com/mail/u/3/#all/1ab23c"
        index_messages(self.store, self.cfg, [make_msg("m1", gmail_url=stored)])
        rows = list(self.store.conn.execute("SELECT gmail_url FROM messages"))
        self.assertEqual([r[0] for r in rows], [stored])   # the column keeps what it saw

        moved = self.open_store(make_cfg(self.db, gmail_account_index=3))
        self.assertEqual(moved.message_detail("m1")["gmail_url"], want)
        self.assertEqual(moved.page_messages(limit=5)[0]["gmail_url"], want)
        self.assertEqual(moved.load_index(with_vectors=False).urls[0], want)


class TestFusion(StubbedEndpoint):

    def setUp(self):
        super().setUp()
        self.msgs = [
            # exact-string match: bm25 loves it, cosine is lukewarm
            make_msg("exact", body="reference 8812-KX for the September payment",
                     subject="reference 8812-KX", sender="billing@flexa.nl",
                     date=datetime(2026, 1, 10, tzinfo=timezone.utc)),
            # paraphrase: cosine loves it
            make_msg("para", body="the dashboard screenshot you asked for is "
                                  "here, showing the new charts",
                     subject="dashboard", sender="design@example.com",
                     date=datetime(2026, 3, 10, tzinfo=timezone.utc)),
            # unrelated filler
            make_msg("noise", body="unsubscribe from the weekly newsletter list",
                     subject="newsletter", sender="news@example.com",
                     date=datetime(2026, 5, 10, tzinfo=timezone.utc)),
        ]
        index_messages(self.store, self.cfg, self.msgs)

    def test_hybrid_puts_the_exact_match_first(self):
        hits, _ = search(self.store, self.cfg, "8812-KX")
        self.assertEqual(hits[0].message_id, "exact")
        self.assertEqual(hits[0].lex_rank, 1)
        self.assertIsNotNone(hits[0].dense)

    def test_semantic_mode_ignores_the_lexical_leg(self):
        hits, meta = search(self.store, self.cfg, "8812-KX", mode="semantic")
        self.assertEqual(meta["w_lex"], 0.0)
        self.assertEqual(meta["lexical_candidates"], 0)
        for h in hits:
            self.assertIsNone(h.lex_rank)

    def test_lexical_mode_needs_no_vectors(self):
        hits, meta = search(self.store, self.cfg, "8812-KX", mode="lexical")
        self.assertEqual(meta["w_dense"], 0.0)
        self.assertEqual([h.message_id for h in hits][0], "exact")

    def test_weights_choose_the_leg(self):
        hits, _ = search(self.store, self.cfg, "dashboard screenshot charts",
                         w_dense=0.0, w_lex=1.0)
        self.assertEqual(hits[0].message_id, "para")
        hits, _ = search(self.store, self.cfg, "dashboard screenshot charts",
                         w_dense=1.0, w_lex=0.0)
        self.assertEqual(hits[0].message_id, "para")

    def test_top_score_is_one_when_both_legs_agree(self):
        hits, meta = search(self.store, self.cfg, "reference 8812-KX september payment")
        self.assertAlmostEqual(hits[0].fused, 1.0, places=6)
        self.assertEqual(hits[0].dense_rank, 1)
        self.assertEqual(hits[0].lex_rank, 1)

    def test_rrf_arithmetic_is_as_documented(self):
        agg_d = {"a": (0.9, 0), "b": (0.5, 1)}
        agg_l = {"b": (4.0, 1), "a": (1.0, 0)}
        rd, rl = mb.rank_map(agg_d), mb.rank_map(agg_l)
        self.assertEqual((rd["a"], rd["b"]), (1, 2))
        self.assertEqual((rl["b"], rl["a"]), (1, 2))
        k, w = 60.0, 0.5
        fused_b = w / (k + 2) + w / (k + 1)
        fused_a = w / (k + 1) + w / (k + 2)
        self.assertAlmostEqual(fused_a, fused_b, places=12,
                               msg="RRF must be symmetric in the legs")

    def test_topk_mean_does_not_reward_volume(self):
        """12 mediocre chunks must not outvote 1 precise one."""
        many = "alpha beta gamma delta epsilon zeta eta theta. " * 24
        one = "zeta eta theta sigma targetword appears once here"
        index_messages(self.store, self.cfg, [
            make_msg("many", body=many, subject="long thread", sender="a@x.nl"),
            make_msg("one", body=one, subject="short", sender="b@x.nl"),
        ])
        hits, _ = search(self.store, self.cfg, "targetword", agg_k=1)
        self.assertEqual(hits[0].message_id, "one")
        # a mean over the top 3 dilutes, but the unique hit still leads
        hits, _ = search(self.store, self.cfg, "targetword", agg_k=3)
        self.assertEqual(hits[0].message_id, "one")

    def test_tier_filter_restricts_candidates(self):
        _, all_meta = search(self.store, self.cfg, "dashboard", tiers=None)
        _, meta = search(self.store, self.cfg, "dashboard", tiers=["xlarge"])
        self.assertLessEqual(meta["chunks_loaded"], all_meta["chunks_loaded"])
        for h in search(self.store, self.cfg, "dashboard", tiers=["media"])[0]:
            self.assertEqual(h.tier, "media")

    def test_time_weight_replaces_the_recency_prior(self):
        """--time-weight is the explicit knob: it *replaces* the bounded recency
        prior rather than stacking on it, so alpha=0 is exactly pure fusion."""
        base, _ = search(self.store, self.cfg, "dashboard")
        same, _ = search(self.store, self.cfg, "dashboard", time_weight=True, alpha=0.0)
        for h in same:
            self.assertAlmostEqual(h.score, h.fused, places=9)
        self.assertTrue(any(abs(h.score - h.fused) > 1e-9 for h in base),
                        "the configured prior should move scores")

    def test_recency_prior_is_bounded(self):
        """A prior may demote an old match, never promote an irrelevant one."""
        hits_on, _ = search(self.store, self.cfg, "dashboard",
                            recency_weight=0.9, recency_half_life_days=30)
        hits_off, _ = search(self.store, self.cfg, "dashboard", recency_weight=0.0)
        for h in hits_on:
            self.assertLessEqual(h.prior, 1.0 + 1e-9)
            self.assertGreaterEqual(h.prior, 1.0 - 0.9 - 1e-9)
        self.assertEqual({h.message_id for h in hits_on},
                         {h.message_id for h in hits_off},
                         "a prior re-orders the same candidates, it does not filter")

    def test_time_weight_prefers_recent_when_alpha_high(self):
        index_messages(self.store, self.cfg, [
            make_msg("old", body="the dashboard is slow today", subject="dashboard",
                     sender="a@x.nl", date=datetime(2020, 1, 1, tzinfo=timezone.utc)),
            make_msg("new", body="the dashboard is slow today", subject="dashboard",
                     sender="b@x.nl", date=datetime.now(timezone.utc)),
        ])
        hits, _ = search(self.store, self.cfg, "dashboard slow", time_weight=True,
                         alpha=0.9)
        self.assertEqual(hits[0].message_id, "new")

    def test_hit_serialisation_is_json_safe(self):
        hits, meta = search(self.store, self.cfg, "dashboard")
        json.dumps({"meta": meta, "results": [h.as_dict() for h in hits]})


# ---------------------------------------------------------------------------
# Migration and reindex
# ---------------------------------------------------------------------------

LEGACY_SCHEMA = """
CREATE TABLE messages(message_id TEXT PRIMARY KEY, folder TEXT, date TEXT,
    sender TEXT, to_addrs TEXT, cc TEXT, subject TEXT, context_text TEXT,
    thread_id TEXT, attachment_exts TEXT, gmail_url TEXT);
CREATE TABLE chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT,
    kind TEXT, label TEXT, text TEXT);
CREATE TABLE embeddings(chunk_id INTEGER PRIMARY KEY, vec BLOB, dim INTEGER);
CREATE TABLE index_state(key TEXT PRIMARY KEY, value TEXT);
"""


class TestMigrationAndReindex(StubbedEndpoint):

    #: these tests build the database themselves, before Store is opened
    make_store = False

    def make_legacy_db(self):
        conn = sqlite3.connect(self.db)
        conn.executescript(LEGACY_SCHEMA)
        bodies = {
            "old1": "short note about the garage",
            "old2": ("the flexa project invoice discussion continues below. "
                     * 90).strip(),                       # ~4600 chars
        }
        for mid, body in bodies.items():
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (mid, "INBOX", "2024-02-02T00:00:00+00:00", "ada@example.com",
                          "me@example.com", "", "old subject", body[:300], "", "pdf", ""))
            for kind, text in (("body", body), ("meta", "From: ada@example.com")):
                conn.execute("INSERT INTO chunks (message_id,kind,label,text)"
                             " VALUES (?,?,?,?)", (mid, kind, kind, text))
                cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute("INSERT INTO embeddings VALUES (?,?,?)",
                             (cid, fake_vec(text).astype(np.float32).tobytes(), DIM))
        conn.commit()
        conn.close()

    def test_legacy_db_upgrades_in_place_and_stays_searchable(self):
        self.make_legacy_db()
        store = self.open_store()
        s = store.stats()
        self.assertEqual(s["messages"], 2)
        self.assertEqual(s["fts_docs"], s["chunks"], "fts backfilled from stored text")
        self.assertNotIn(None, [r[0] for r in store.conn.execute("SELECT tier FROM chunks")])
        hits, _ = search(store, self.cfg, "garage")
        self.assertEqual(hits[0].message_id, "old1")
        # reopening is a no-op
        again = mb.Store(self.db, self.cfg)
        self.assertEqual(again.stats()["fts_docs"], s["chunks"])
        self.assertEqual(again.stats()["chunks"], s["chunks"])

    def test_legacy_bm25_works_before_any_reindex(self):
        self.make_legacy_db()
        store = self.open_store()
        hits = store.lexical_search("flexa")
        self.assertTrue(hits)
        mid = store.conn.execute("SELECT message_id FROM chunks WHERE id=?",
                                 (hits[0][0],)).fetchone()[0]
        self.assertEqual(mid, "old2")

    def test_reindex_derives_tiers_and_leaves_canonical_untouched(self):
        self.make_legacy_db()
        store = self.open_store()
        before = store.conn.execute(
            "SELECT id, text FROM chunks WHERE label='body' ORDER BY id").fetchall()
        vectors_before = store.conn.execute(
            "SELECT chunk_id, vec FROM embeddings ORDER BY chunk_id").fetchall()

        args = _args(config=self.cfg_path, tiers="small,large", rebuild_fts=True,
                     dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(args)
        self.assertEqual(store.stats()["chunks"], 4, "--dry-run wrote nothing")

        args = _args(config=self.cfg_path, tiers="small,large", rebuild_fts=True)
        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(args)

        s = store.stats()
        self.assertIn("small", s["by_tier"])
        self.assertIn("large", s["by_tier"])
        self.assertGreater(s["derived_chunks"], 0)
        self.assertEqual(s["fts_docs"], s["chunks"])
        after = store.conn.execute(
            "SELECT id, text FROM chunks WHERE label='body' ORDER BY id").fetchall()
        self.assertEqual(after, before, "canonical text must not change")
        kept = dict(store.conn.execute(
            "SELECT chunk_id, vec FROM embeddings WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE nwin=1) ORDER BY chunk_id").fetchall())
        for cid, vec in vectors_before:
            self.assertEqual(kept[cid], vec, "an existing embedding was rewritten")

        # idempotent: a second run must not grow the index
        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(args)
        self.assertEqual(store.stats()["chunks"], s["chunks"], "reindex is not idempotent")
        self.assertEqual(store.stats()["fts_docs"], store.stats()["chunks"])

    def test_reindex_removes_windows_of_a_source_that_shrank(self):
        """Repairing a body can drop a message from three windows to one. The
        windows the old, longer text produced have to go with it: they stay in
        FTS, they keep their vector, they keep competing for the top ten, and a
        plan that only rebuilds sources that still derive windows never deletes
        them again. A real index carried 6 710 of these."""
        self.make_legacy_db()
        store = self.open_store()
        args = _args(config=self.cfg_path, tiers="small,large")
        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(args)
        mid, n_before = store.conn.execute(
            "SELECT message_id, COUNT(*) FROM chunks WHERE canonical=0"
            " GROUP BY message_id ORDER BY COUNT(*) DESC, message_id LIMIT 1"
        ).fetchone()
        self.assertGreater(n_before, 0, "the fixture needs windows before it shrinks")

        store.conn.execute("UPDATE chunks SET text='Kort. Zie bijlage.'"
                           " WHERE message_id=? AND canonical=1", (mid,))
        store.conn.commit()
        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(args)

        left = store.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE canonical=0 AND message_id=?",
            (mid,)).fetchone()[0]
        self.assertEqual(left, 0,
                         "windows of a source that no longer derives any survived")
        s = store.stats()
        self.assertEqual(s["fts_docs"], s["chunks"],
                         "an FTS row of a removed window survived")
        self.assertEqual(store.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE chunk_id NOT IN"
            " (SELECT id FROM chunks)").fetchone()[0], 0,
            "the vector of a removed window survived")

    def test_reindex_improves_recall_for_an_exact_string(self):
        self.make_legacy_db()
        store = self.open_store()
        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(_args(config=self.cfg_path, tiers="small,large"))
        hits, _ = search(store, self.cfg, "flexa invoice 9021")
        self.assertEqual(hits[0].message_id, "old2")

    def test_rebuild_fts_costs_no_embeddings(self):
        store = self.open_store()
        index_messages(store, self.cfg, [make_msg("m1", body="alpha beta")])
        calls = {"n": 0}
        real = mb.EmbeddingClient.embed_texts

        def counting(self, texts):
            calls["n"] += 1
            return real(self, texts)

        mb.EmbeddingClient.embed_texts = counting
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mb.cmd_reindex(_args(config=self.cfg_path, tiers="", rebuild_fts=True))
        finally:
            mb.EmbeddingClient.embed_texts = real
        self.assertEqual(calls["n"], 0, "--rebuild-fts must not embed anything")
        self.assertEqual(store.fts_count(), store.stats()["chunks"])

    def test_stale_index_warns_instead_of_failing(self):
        store = self.open_store()
        index_messages(store, self.cfg, [make_msg("m1", body="alpha")])
        store.save_index_meta()
        cfg2 = make_cfg(self.db, chunk_sizes={"small": 100, "large": 500,
                                             "xlarge": 3000})
        store2 = self.open_store(cfg2)
        with self.assertLogs("mailbox", level="WARNING") as logs:
            store2.warn_if_stale()
        self.assertTrue(any("reindex" in m for m in logs.output))

    def test_re_embed_replaces_text_vectors_only(self):
        """A change of embedding model makes stored cosines incomparable with
        new ones: --re-embed rebuilds the text vectors from the text already in
        the db, and leaves media vectors alone because their bytes are not."""
        self.make_legacy_db()
        store = self.open_store()
        bad = np.zeros(DIM, dtype=np.float32)
        bad[3] = 1.0
        cid = store.conn.execute("SELECT id FROM chunks WHERE kind='body'"
                                 " LIMIT 1").fetchone()[0]
        mid = store.conn.execute("SELECT message_id FROM chunks WHERE id=?",
                                 (cid,)).fetchone()[0]
        store.set_embedding(cid, bad)
        store.conn.execute("INSERT INTO chunks (message_id,kind,label,text,tier)"
                           " VALUES (?,?,?,?,?)",
                           (mid, "attachment", "attachment: a.png",
                            "attachment: a.png", mb.MEDIA_TIER))
        mcid = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        store.set_embedding(mcid, bad)
        store.conn.commit()
        n_before = store.stats()["chunks"]

        with contextlib.redirect_stdout(io.StringIO()):
            mb.cmd_reindex(_args(config=self.cfg_path, tiers="", re_embed=True))

        text = store.conn.execute("SELECT text FROM chunks WHERE id=?",
                                  (cid,)).fetchone()[0]
        subj = store.conn.execute("SELECT subject FROM messages WHERE message_id=?",
                                  (mid,)).fetchone()[0]
        # re-embedding goes through the same prompt builder as indexing, which
        # puts the Subject in `title:` - comparing against a prefix-less embed
        # would fail for the prompt, not for the migration.
        want = mb.EmbeddingClient(self.cfg).embed_texts([text], titles=[subj])[0]
        got = np.frombuffer(store.conn.execute(
            "SELECT vec FROM embeddings WHERE chunk_id=?", (cid,)).fetchone()[0],
            dtype=np.float32)
        self.assertTrue(np.allclose(got, want), "text chunk re-embedded from its text")
        self.assertEqual(store.stats()["chunks"], n_before, "no chunks added")
        still = np.frombuffer(store.conn.execute(
            "SELECT vec FROM embeddings WHERE chunk_id=?", (mcid,)).fetchone()[0],
            dtype=np.float32)
        self.assertTrue(np.allclose(still, bad), "media vector must not be touched")

    def test_change_of_embedding_model_is_warned_about(self):
        """Same dim, different model or server: nothing in a cosine search looks
        wrong, so the fingerprint recorded at index time has to say so."""
        store = self.open_store(make_cfg(self.db, embed_model="model-a",
                                         embed_endpoint="http://a/v1/embeddings"))
        store.save_index_meta()
        other_model = make_cfg(self.db, embed_model="model-b",
                               embed_endpoint="http://a/v1/embeddings")
        with self.assertLogs("mailbox", level="WARNING") as logs:
            self.open_store(other_model).warn_if_stale()
        self.assertTrue(any("--re-embed" in m for m in logs.output), logs.output)

        other_server = make_cfg(self.db, embed_model="model-b",
                                embed_endpoint="http://b:9999/v1/embeddings")
        with self.assertLogs("mailbox", level="WARNING") as logs:
            self.open_store(other_server).warn_if_stale()
        self.assertTrue(any("build or quantisation" in m for m in logs.output),
                        logs.output)


def _args(config, **over):
    """Stand-in for the argparse namespace cmd_reindex reads."""
    ns = {"config": config, "tiers": None, "rebuild_fts": False, "no_retier": False,
          "re_embed": False, "limit": None, "dry_run": False}
    ns.update(over)
    return type("A", (), ns)()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

class TestEndpointDialects(unittest.TestCase):
    """The client reads both reply shapes and never claims image support it
    cannot deliver (a wrong claim silently stores filename vectors as 'media')."""

    VEC = [1.0] + [0.0] * 767
    CHUNK = {"kind": "attachment", "src": "attachment: a.png",
             "tier": mb.MEDIA_TIER, "label": "attachment: a.png",
             "text": "attachment: a.png", "image_b64": "AAEC", "mime": "image/png"}

    def client(self, url):
        cfg = make_cfg(":memory:", embed_endpoint=url)
        return mb.EmbeddingClient(cfg)

    def test_openai_and_ollama_native_replies_parse_identically(self):
        c = self.client("http://x/v1/embeddings")
        openai = {"data": [{"embedding": self.VEC}]}
        native = {"embeddings": [self.VEC]}
        self.assertTrue(np.array_equal(c._vectors(openai)[0], c._vectors(native)[0]))

    def test_unexpected_reply_names_the_keys_it_got(self):
        c = self.client("http://x/v1/embeddings")
        with self.assertRaises(RuntimeError) as cm:
            c._vectors({"vector": [1, 2]})
        self.assertIn("vector", str(cm.exception))

    def test_dialect_is_detected_from_the_url(self):
        self.assertTrue(self.client("http://h:11434/api/embed").native)
        self.assertFalse(self.client("http://h:11434/v1/embeddings").native)

    def test_native_endpoint_never_claims_multimodal(self):
        sent = []

        def fake_post(payload, multimodal=False):
            sent.append((payload, multimodal))
            return {"embeddings": [self.VEC]}

        c = self.client("http://h:11434/api/embed")
        c._post = fake_post
        c.embed_media(self.CHUNK)
        self.assertEqual([m for _, m in sent], [False],
                         "a native endpoint must not be sent image input")
        self.assertIsNone(c.multimodal_ok)
        self.assertIn("title: a.png | text: image attachment", sent[0][0]["input"])

    def test_both_media_item_shapes_are_producible(self):
        """Endpoints disagree on the envelope: a flat part object, or chat-style
        parts nested per input item. Same bytes either way."""
        c = self.client("http://x/v1/embeddings")
        c.media_shape = "image_url"
        flat = c._media_item(self.CHUNK)
        self.assertEqual(flat["type"], "image_url")
        self.assertEqual(flat["image_url"]["url"], "data:image/png;base64,AAEC")
        self.assertTrue(flat["text"].startswith("title: a.png | text: "),
                        "an image's filename belongs in title:, not in a sentence "
                        "about the word 'attachment'")

        c.media_shape = "content"
        nested = c._media_item(self.CHUNK)
        self.assertEqual([p["type"] for p in nested["content"]], ["image_url", "text"])
        self.assertEqual(nested["content"][0]["image_url"], flat["image_url"])

    def test_configured_shape_is_the_one_sent(self):
        sent = []

        def fake_post(payload, multimodal=False):
            sent.append(payload)
            return {"data": [{"embedding": self.VEC}]}

        c = self.client("http://x/v1/embeddings")
        c._post = fake_post
        c.media_shape = "content"
        c.embed_media(self.CHUNK)
        item = sent[0]["input"][0]
        self.assertIn("content", item)
        self.assertEqual(item["content"][1]["text"],
                         "title: a.png | text: image attachment")
        self.assertTrue(c.multimodal_ok)

    def test_legacy_profile_keeps_the_old_document_prefix(self):
        """`--profile legacy` must reproduce the pre-tuning strings byte for
        byte, or the A/B baseline is not the old pipeline."""
        cfg = make_cfg(":memory:", embed_endpoint="http://x/v1/embeddings",
                       prompt_scheme="legacy")
        c = mb.EmbeddingClient(cfg)
        self.assertEqual(c.doc_input("body text", title="a.png"), "Document: body text")
        self.assertEqual(c.query_input("factuur"), "SearchQuery: factuur")
        self.assertTrue(c._media_item(dict(self.CHUNK))["text"]
                        .startswith("Document: "))

    def test_media_shape_is_negotiated_on_rejection(self):
        """llama.cpp wants nested content parts, OpenAI/vLLM/TEI a flat part
        object: a rejection of the configured shape retries the other one
        instead of giving up on image content for the whole run."""
        seen = []

        def fake_post(payload, multimodal=False):
            item = payload["input"][0]
            seen.append(item)
            if "content" not in item:
                raise mb.MultimodalRejected('HTTP 400: "prompt" elements must be')
            return {"data": [{"embedding": self.VEC}]}

        c = self.client("http://x/v1/embeddings")
        c.media_shape = "image_url"
        c._post = fake_post
        with self.assertLogs("mailbox", level="INFO"):
            c.embed_media(self.CHUNK)
        self.assertTrue(c.multimodal_ok)
        self.assertEqual(c.media_shape, "content")
        self.assertEqual(len(seen), 2, "one probe per shape, no more")

    def test_image_only_item_keeps_its_envelope(self):
        """No text label must not silently switch dialects: an endpoint that
        only accepts nested parts would then reject every shape."""
        c = self.client("http://x/v1/embeddings")
        c.media_shape = "content"
        item = c._media_item({**self.CHUNK, "text": ""})
        self.assertEqual([p["type"] for p in item["content"]], ["image_url"])
        c.media_shape = "image_url"
        self.assertEqual(c._media_item({**self.CHUNK, "text": ""}),
                         {"type": "image_url",
                          "image_url": {"url": "data:image/png;base64,AAEC"}})

    def test_unknown_shape_warns_and_falls_back(self):
        with self.assertLogs("mailbox", level="WARNING"):
            c = mb.EmbeddingClient(make_cfg(":memory:",
                                            embed_endpoint="http://x/v1/embeddings",
                                            media_item_shape="chatml"))
        self.assertEqual(c.media_shape, "image_url")


class TestQueryImage(StubbedEndpoint):
    """Searching *with* a picture: the query side of image input.

    The endpoint stub here is `pixel_post`, not the shared `fake_post`: a stub
    that reads an image item's filename cannot express "these two files look
    alike", which is the one property every image-search claim depends on.
    """

    PNG = b"\x89PNG\r\n\x1a\n" + b" pixels " * 12
    OTHER = b"\x89PNG\r\n\x1a\n" + b" other pixels " * 12

    @classmethod
    def setUpClass(cls):
        cls._real_post = mb.EmbeddingClient._post
        mb.EmbeddingClient._post = pixel_post

    def client(self, **over):
        return mb.EmbeddingClient(make_cfg(self.db, **over) if over else self.cfg)

    def index_pair(self):
        """One mail with an image attachment, one mail of plain words."""
        index_messages(self.store, self.cfg, [
            make_msg(mid="shot", subject="onderdeel besteld",
                     body="de foto van het onderdeel dat besteld is",
                     atts=[mb.Attachment("onderdeel.png", "image/png", self.PNG)]),
            make_msg(mid="nota", subject="notulen",
                     body="de nota van de vergadering van dinsdag"),
        ])

    def search_with(self, text="", image=None, image_name="", cfg=None, **over):
        p = mb.SearchParams(mode="semantic", top=10, agg_k=1, tier_norm=False,
                            collapse=False, rerank="off", **over)
        cfg = cfg or self.cfg
        return mb.Searcher(cfg, self.store, mb.EmbeddingClient(cfg)).run(
            text, p, image=image, image_name=image_name)

    def top(self, text="", image=None, **over):
        return self.search_with(text, image, **over)[0][0]

    # -- what goes over the wire -------------------------------------------

    def test_a_query_image_is_sent_bare(self):
        """No filename, no `title:` prefix, no "image attachment" label.

        The document side labels an attachment because a filename is how people
        find files; a query image's name comes from a screenshot folder and says
        nothing about what is in it, and EmbeddingGemma puts instruction prefixes
        on text only. The picture goes alone, or it does not meet the attachment
        vectors in the space they live in.
        """
        seen = []

        def spy(payload, multimodal=False):
            seen.append(payload["input"][0])
            return pixel_post(None, payload, multimodal)

        c = self.client()
        c._post = spy
        vec = c.embed_query_image(self.PNG, "image/png", "IMG_0042.png")
        self.assertEqual(vec.shape, (PIXEL_DIM,))
        self.assertTrue(c.multimodal_ok)
        self.assertEqual(len(seen), 1, "one image, one request")
        item = seen[0]
        self.assertEqual(image_url_of(item),
                         "data:image/png;base64,"
                         + base64.b64encode(self.PNG).decode())
        self.assertNotIn("IMG_0042", json.dumps(item))
        self.assertNotIn("title:", json.dumps(item))
        self.assertNotIn("text", item, "a query image travels without a text part")

    def test_nested_envelope_also_carries_the_image_alone(self):
        c = self.client()
        c.media_shape = "content"
        item = c._media_item(mb.query_image_chunk("AAEC", "image/png"))
        self.assertEqual([p["type"] for p in item["content"]], ["image_url"])

    def test_both_sides_use_one_vector_for_one_picture(self):
        """The property the feature rests on: the photo you attach and the photo
        somebody mailed are one point in the space."""
        c = self.client()
        doc = c.embed_media({"kind": "attachment", "src": "attachment: shot.png",
                             "tier": mb.MEDIA_TIER, "label": "attachment: shot.png",
                             "text": "attachment: shot.png",
                             "image_b64": base64.b64encode(self.PNG).decode(),
                             "mime": "image/png"})
        q = c.embed_query_image(self.PNG, "image/png", "screenshot.png")
        other = c.embed_query_image(self.OTHER, "image/png", "other.png")
        self.assertAlmostEqual(float(doc @ q), 1.0, places=5)
        self.assertLess(float(doc @ other), 0.1)

    # -- what the engine does with it ---------------------------------------

    def test_the_picture_finds_the_mail_it_came_from(self):
        self.index_pair()
        vec = self.client().embed_query_image(self.PNG, "image/png", "IMG_1.png")
        hits, meta = self.search_with(image=vec, image_name="IMG_1.png")
        self.assertEqual(hits[0].message_id, "shot")
        self.assertEqual(meta["image"], "IMG_1.png")
        self.assertTrue(meta["image_only"])
        self.assertEqual(meta["w_lex"], 0,
                         "a picture has no keywords; bm25 cannot be one leg of it")

    def test_each_side_can_carry_the_match_on_its_own(self):
        """The contract the user chose: *either* may match.

        One mail is in the index because of this picture, another because of
        these words; a query carrying both must bring both back, and the perfect
        pixel match outranks a partial word match. A fusion that required both
        sides to agree would return neither.
        """
        self.index_pair()
        vec = self.client().embed_query_image(self.PNG, "image/png")
        hits, meta = self.search_with("vergadering dinsdag", image=vec,
                                      image_name="IMG_1.png")
        self.assertEqual([h.message_id for h in hits[:2]], ["shot", "nota"])
        self.assertFalse(meta["image_only"])

    def test_words_that_match_nothing_do_not_dilute_the_picture(self):
        """`max(cos(image), cos(text))`, not a mean.

        A mean halves every one-sided match, so a chunk that resembles the
        picture a little *and* the words a little outranks a chunk that *is* the
        picture - which is the query the feature exists for.
        """
        self.index_pair()
        vec = self.client().embed_query_image(self.PNG, "image/png")
        only = self.top(image=vec)
        both = self.top("zzzqqq", image=vec)
        self.assertEqual(both.message_id, "shot")
        self.assertAlmostEqual(both.dense, only.dense, places=5,
                               msg="the words added nothing and the score moved")

    def test_a_picture_that_matches_nothing_does_not_dilute_the_words(self):
        self.index_pair()
        vec = self.client().embed_query_image(self.OTHER, "image/png")
        only = self.top("vergadering dinsdag")
        both = self.top("vergadering dinsdag", image=vec)
        self.assertEqual(both.message_id, "nota")
        self.assertAlmostEqual(both.dense, only.dense, places=5)

    def test_an_image_cannot_be_searched_in_lexical_mode(self):
        self.index_pair()
        vec = self.client().embed_query_image(self.PNG, "image/png")
        p = mb.SearchParams(mode="lexical")
        with self.assertRaises(SystemExit) as cm:
            mb.Searcher(self.cfg, self.store, self.client()).run("", p, image=vec)
        self.assertIn("lexical", str(cm.exception))

    # -- when the endpoint cannot do it --------------------------------------

    def test_a_native_endpoint_is_refused_before_the_request(self):
        sent = []

        def spy(payload, multimodal=False):
            sent.append(payload)
            return {"embeddings": [[0.0] * PIXEL_DIM]}

        c = self.client(embed_endpoint="http://h:11434/api/embed")
        c._post = spy
        with self.assertRaises(SystemExit) as cm:
            c.embed_query_image(self.PNG, "image/png")
        self.assertIn("/api/embed", str(cm.exception))
        self.assertEqual(sent, [], "a native endpoint ignores image input, so "
                                   "asking it would return a text vector for a photo")

    def test_multimodal_off_is_refused(self):
        c = self.client(embed_media_multimodal=False)
        with self.assertRaises(SystemExit) as cm:
            c.embed_query_image(self.PNG, "image/png")
        self.assertIn("embed_media_multimodal", str(cm.exception))

    def test_every_shape_rejected_says_so_and_remembers_it(self):
        """Refuse in a sentence, and let the index side stop probing."""
        def reject(payload, multimodal=False):
            raise mb.MultimodalRejected("HTTP 400: image input not supported")

        c = self.client()
        c._post = reject
        with self.assertRaises(SystemExit) as cm:
            c.embed_query_image(self.PNG, "image/png")
        self.assertIn("test_embed_media.py", str(cm.exception))
        self.assertIs(c.multimodal_ok, False)

    def test_a_query_image_never_falls_back_to_a_text_label(self):
        """`embed_media` answers a rejection with the filename vector, which is
        right for a document. Here that same fallback would embed the sentence
        "image attachment" and rank every mail holding those words for a photo."""
        def reject(payload, multimodal=False):
            raise mb.MultimodalRejected("HTTP 400: no image input")

        c = self.client()
        c._post = reject
        with self.assertRaises(SystemExit):
            c.embed_query_image(self.PNG, "image/png")
        self.assertIs(c.multimodal_ok, False)


class TestCli(StubbedEndpoint):

    def setUp(self):
        super().setUp()
        index_messages(self.store, self.cfg, [
            make_msg("m1", body="reference 8812-KX for september",
                     subject="reference 8812-KX", sender="billing@flexa.nl"),
            make_msg("m2", body="the dashboard screenshot is attached",
                     subject="dashboard", sender="design@example.com"),
        ])

    def _run(self, argv):
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["mailbox.py"] + argv
        try:
            with contextlib.redirect_stdout(out):
                mb.main()
        finally:
            sys.argv = old
        return out.getvalue()

    def test_search_json_output(self):
        text = self._run(["--config", self.cfg_path, "search", "8812-KX",
                          "--json", "--top", "5"])
        payload = json.loads(text)
        self.assertEqual(payload["meta"]["mode"], "hybrid")
        self.assertEqual(payload["results"][0]["subject"], "reference 8812-KX")
        self.assertIn("bm25_rank", payload["results"][0])

    def test_search_explain_and_mode(self):
        text = self._run(["--config", self.cfg_path, "search", "8812-KX",
                          "--explain", "--mode", "hybrid"])
        self.assertIn("rrf:", text)
        self.assertIn("bm25", text)
        text = self._run(["--config", self.cfg_path, "search", "dashboard",
                          "--mode", "semantic"])
        self.assertIn("dense", text)

    def test_stats_reports_tiers_and_fts(self):
        text = self._run(["--config", self.cfg_path, "stats"])
        self.assertIn("fts docs", text)
        self.assertIn("tier ", text)
        self.assertIn("chunk_sizes", text)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._run(["--config", self.cfg_path, "search", "x", "--mode", "vibes"])


class TestProfilePrecedence(unittest.TestCase):
    """`--profile legacy` is the A/B baseline and the rollback. A switch that
    silently does nothing is worse than no switch: the bench number it produces
    looks like a measurement of the old pipeline."""

    def cfgfile(self, tmp, full=False, **over):
        """A config file: either the shape of config.example.json (every knob)
        or a short hand-written one."""
        cfg = dict(mb.DEFAULTS) if full else {}
        cfg.update(over)
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return str(path)

    def test_a_full_config_file_still_switches(self):
        """config.example.json lists every knob. Copying it must not turn
        --profile legacy into a no-op because each key 'looks deliberate'."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.cfgfile(tmp, full=True, profile="v2")
            cfg = mb.load_config(path, "legacy")
            self.assertEqual(cfg["profile"], "legacy")
            self.assertEqual(cfg["rrf_k"], mb.LEGACY_PROFILE["rrf_k"])
            self.assertFalse(cfg["parse_filters"])
            self.assertEqual(cfg["prompt_scheme"], "legacy")

    def test_a_profile_named_in_the_file_switches_too(self):
        """Copying config.example.json and writing "profile": "legacy" in it is
        the documented rollback: the file lists every knob, so 'the file said it
        first' would leave the rollback inert."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.cfgfile(tmp, profile="legacy", rrf_k=42)
            cfg = mb.load_config(path)
            self.assertEqual(cfg["profile"], "legacy")
            self.assertEqual(cfg["rrf_k"], mb.LEGACY_PROFILE["rrf_k"])
            self.assertFalse(cfg["parse_filters"])

    def test_no_profile_named_leaves_the_file_alone(self):
        """The common case: a config.json tuned by hand and no profile anywhere
        must not have its knobs reset to DEFAULTS on load."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.cfgfile(tmp, rrf_k=25, min_relevance=0.5)
            cfg = mb.load_config(path)
            self.assertEqual(cfg["rrf_k"], 25)
            self.assertEqual(cfg["min_relevance"], 0.5)
            self.assertEqual(cfg["profile"], "v2")

    def test_switching_back_is_a_switch_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.cfgfile(tmp, profile="legacy", rrf_k=42)
            cfg = mb.load_config(path, "v2")
            self.assertEqual(cfg["profile"], "v2")
            self.assertEqual(cfg["rrf_k"], mb.DEFAULTS["rrf_k"])
            self.assertTrue(cfg["parse_filters"])

    def test_a_shell_variable_outranks_the_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.cfgfile(tmp)
            with mock.patch.dict(os.environ, {"EMBED_MODEL": "from-env"}):
                cfg = mb.load_config(path, "legacy")
            self.assertEqual(cfg["embed_model"], "from-env")

    def test_every_legacy_key_has_a_tuned_counterpart(self):
        """LEGACY_PROFILE is a diff against DEFAULTS; a key that exists only
        in the legacy set can never be switched back by --profile v2."""
        self.assertFalse(set(mb.LEGACY_PROFILE) - set(mb.DEFAULTS))


class TestQueryParse(unittest.TestCase):
    """Phase 4: 'mail van flexa sinds maart' is two questions - which mails
    from whom, and which of those match the words. Only the second one is an
    embedding job; the first is SQL, and applying it as query text is what
    makes the vector drift toward the word 'van'.

    `today` is pinned everywhere: a relative date is a claim about a calendar,
    and a test that reads the real clock passes or fails by the date.
    """

    TODAY = date(2026, 6, 15)        # a Monday

    def p(self, q):
        return qp.parse(q, today=self.TODAY)

    def test_sender_phrases_become_a_filter(self):
        f = self.p("factuur van bob@flexa.nl")
        self.assertEqual(f.senders, ["bob@flexa.nl"])
        self.assertEqual(f.residual, "factuur",
                         "the words still have to be searched for")
        self.assertEqual(self.p("mail van jan jansen").senders, ["jan jansen"])

    def test_a_month_after_van_is_not_a_person(self):
        """'van maart', 'per maand': what follows 'van' is Dutch grammar, not a
        name. A sender filter built from it matches nobody and the query that
        looked specific returns nothing at all."""
        for q in ("factuur van maart", "wat kost het per maand",
                  "scan van het document", "factuur van de techniek"):
            self.assertFalse(self.p(q).senders, q)
            self.assertFalse(self.p(q).recipients, q)

    def test_recipient_phrases(self):
        self.assertEqual(self.p("naar kim@voorbeeld.nl").recipients,
                         ["kim@voorbeeld.nl"])
        self.assertEqual(self.p("aan kim").recipients, ["kim"])

    def test_relative_date_phrases(self):
        self.assertEqual((self.p("vorige week").since,
                          self.p("vorige week").until),
                         ("2026-06-08", "2026-06-15"))
        self.assertEqual((self.p("previous month").since,
                          self.p("previous month").until),
                         ("2026-05-01", "2026-06-01"))
        self.assertEqual((self.p("laatste kwartaal").since,
                          self.p("laatste kwartaal").until),
                         ("2026-01-01", "2026-04-01"))
        self.assertEqual((self.p("gisteren").since, self.p("gisteren").until),
                         ("2026-06-14", "2026-06-15"))
        older = self.p("ouder dan 2 weken")
        self.assertIsNone(older.since)
        self.assertEqual(older.until, "2026-06-01")

    def test_explicit_date_phrases(self):
        since = self.p("sinds 2025-01-01")
        self.assertEqual(since.since, "2025-01-01")
        self.assertIsNone(since.until, "'sinds' has no upper bound")
        self.assertEqual(since.residual, "", "'sinds' must not leak into the text")
        span = self.p("tussen 2024-03-01 en 2024-03-31")
        self.assertEqual((span.since, span.until), ("2024-03-01", "2024-04-01"))
        day = self.p("mail van 2024-09-23")
        self.assertEqual((day.since, day.until), ("2024-09-23", "2024-09-24"))

    def test_a_period_ends_the_day_before_the_next_one(self):
        """since/until is [since, until). Q1 2025 ends 31 March, so until is
        1 April; one day more and 'Q1' quietly answers with 1 April mails."""
        self.assertEqual(self.p("Q1 2025").until, "2025-04-01")
        self.assertEqual(self.p("Q1 2025").since, "2025-01-01")
        self.assertEqual(self.p("mail van maart 2024").until, "2024-04-01")
        self.assertEqual(self.p("2024").until, "2025-01-01")
        # 'tot <date>' reads as 'tot en met' (up to and including), so the
        # exclusive bound is the day after the one that was typed.
        self.assertEqual(self.p("tot 2026-03-05").until, "2026-03-06")
        self.assertIsNone(self.p("tot 2026-03-05").since)

    def test_a_day_is_not_a_month(self):
        """'5 mei 2024' read as 'mei 2024' leaves the 5 behind in the embedded
        text and turns one day into a month."""
        f = self.p("5 mei 2024")
        self.assertEqual((f.since, f.until), ("2024-05-05", "2024-05-06"))
        self.assertNotIn("5", f.residual.split())

    def test_an_ordinary_query_parses_to_nothing(self):
        for q in ("hoe gaat het", "factuur", "dashboard screenshot", "8812-KX"):
            self.assertTrue(self.p(q).is_empty(), q)

    def test_filters_off_leaves_the_query_alone(self):
        cfg = make_cfg("unused.db", parse_filters=False)
        text, f = mb.apply_filters(cfg, "factuur van bob@flexa.nl")
        self.assertIsNone(f)
        self.assertEqual(text, "factuur van bob@flexa.nl")

    def test_a_filter_only_query_keeps_its_text(self):
        """'van flexa' is all filter: embedding the empty residual would give
        every mail the same score, so the original text is kept."""
        cfg = make_cfg("unused.db")
        text, f = mb.apply_filters(cfg, "van bob@flexa.nl")
        self.assertIsNotNone(f)
        self.assertEqual(text, "van bob@flexa.nl")

    def test_a_dutch_pronoun_after_van_is_not_a_sender(self):
        """'Statusupdate van je order' is a subject, not a request for mail from
        someone called `je`. Each wrong guess costs the whole result set, so a
        pronoun ends the phrase instead of becoming a filter."""
        for q in ("Statusupdate van je order - week 14",
                  "Nieuws van onze leverancier",
                  "Herinnering van hun sideproject",
                  "Rapport van mijn stage"):
            self.assertEqual(qp.parse(q).senders, [], q)

    def test_the_english_to_of_up_to_date_is_not_a_recipient(self):
        f = qp.parse("Is your business name up to date?")
        self.assertEqual((f.senders, f.recipients), ([], []), f.as_dict())


class TestFilterPreFilter(StubbedEndpoint):
    """Filters are a SQL pre-filter, applied before either leg scores."""

    TODAY = date(2026, 6, 15)

    def setUp(self):
        super().setUp()
        index_messages(self.store, self.cfg, [
            make_msg("bob", body="hierbij de factuur voor het domein",
                     subject="factuur domein", sender="bob@flexa.nl",
                     date=datetime(2026, 3, 4, tzinfo=timezone.utc),
                     atts=[mb.Attachment("factuur.pdf", "application/pdf", b"x")]),
            make_msg("shop", body="je bestelling is onderweg",
                     subject="bestelling", sender="shop@bol.nl",
                     cc="carol@example.com",
                     date=datetime(2026, 5, 6, tzinfo=timezone.utc)),
        ])

    def ids(self, q):
        return self.store.filter_message_ids(qp.parse(q, today=self.TODAY))

    def test_sender_filter_needs_the_words_nowhere(self):
        """Regression: building the argument list with `args += [...]` inside
        the nested add() made `args` a local of the closure, so the first query
        with a sender crashed the whole search instead of filtering it."""
        self.assertEqual(self.ids("glurp van bob@flexa.nl"), {"bob"})

    def test_recipient_filter_looks_at_cc_too(self):
        self.assertEqual(self.ids("naar carol@example.com"), {"shop"})

    def test_date_filter_bounds_are_respected(self):
        self.assertEqual(self.ids("mail van 2026-03-04"), {"bob"})
        self.assertEqual(self.ids("sinds 2026-03-05"), {"shop"})
        self.assertEqual(self.ids("voor 2026-03-05"), {"bob"})

    def test_attachment_filter(self):
        self.assertEqual(self.ids("met bijlage"), {"bob"})

    def test_two_filters_intersect_and_the_words_are_not_ones(self):
        """'factuur' is not a filter: the pre-filter restricts *who/when*, the
        words are still scored by the two legs afterwards."""
        self.assertEqual(self.ids("factuur van bob@flexa.nl"), {"bob"})
        self.assertEqual(self.ids("van shop@bol.nl met bijlage"), set())
        self.assertEqual(self.ids("van bob@flexa.nl met bijlage"), {"bob"})

    def test_the_search_itself_honours_the_filter(self):
        hits, meta = search(self.store, self.cfg, "glurp van bob@flexa.nl")
        self.assertEqual([h.message_id for h in hits], ["bob"])
        self.assertIn("bob@flexa.nl", meta["filters"])

    def test_a_guessed_sender_that_matches_nothing_does_not_empty_the_search(self):
        """'van' is a Dutch preposition first and a sender second, so a parser
        guess that no message can satisfy may not turn a searchable mailbox into
        an empty page: the query is searched as typed, as if it had kept quiet."""
        query = "factuur van iemanddieernstigbestaat.nl"
        guessed, meta = search(self.store, self.cfg, query)
        quiet, _ = search(self.store, dict(self.cfg, parse_filters=False), query)
        self.assertEqual([h.message_id for h in guessed],
                         [h.message_id for h in quiet])
        self.assertEqual(guessed[0].message_id, "bob")
        self.assertEqual(meta["filters"], "", "a dropped guess is not a filter")

    def test_a_filter_the_user_typed_stays_decisive(self):
        """The difference between a guess and an instruction: `from:nobody` means
        'mail from that sender', and the honest answer to that is nothing."""
        sp = mb.Searcher(self.cfg, self.store, mb.EmbeddingClient(self.cfg))
        p = sp.params()
        p.filters = qp.parse("van iemanddieernstigbestaat.nl")
        hits, meta = sp.run("factuur", p)
        self.assertEqual([h.message_id for h in hits], [])
        self.assertTrue(meta.get("abstained"))
        self.assertIn("no message matches", meta.get("note", ""))


class TestSearchContract(StubbedEndpoint):
    """What eval.py and cmd_search rely on: the shape of the return value,
    loud typos in options, and what collapsing promises to the user."""

    def setUp(self):
        super().setUp()
        same = ("Beste customer, uw domein example.nl verloopt over 14 dagen. "
                "Graag de transfer code aanvragen via het portaal.")
        index_messages(self.store, self.cfg, [
            make_msg("exact", body="reference 8812-KX for the September payment",
                     subject="reference 8812-KX", sender="billing@flexa.nl",
                     date=datetime(2026, 1, 10, tzinfo=timezone.utc)),
            make_msg("noise", body="unsubscribe from the weekly newsletter list",
                     subject="newsletter", sender="news@example.com",
                     date=datetime(2026, 5, 10, tzinfo=timezone.utc)),
            # one conversation, two mails, one answer each
            make_msg("t1", body=same, subject="domein verloopt",
                     sender="hosting@example.nl",
                     date=datetime(2026, 4, 1, tzinfo=timezone.utc)),
            make_msg("t2", body=same, subject="domein verloopt",
                     sender="support@example.nl",
                     date=datetime(2026, 4, 2, tzinfo=timezone.utc)),
        ])
        for mid in ("t1", "t2"):     # same thread, decided after indexing
            self.store.conn.execute("UPDATE messages SET thread_id=? "
                                    "WHERE message_id=?", ("TH1", mid))
        self.store.conn.commit()

    def test_search_query_returns_ids_and_meta(self):
        ids, meta = mb.search_query(self.cfg, "8812-KX", store=self.store,
                                    client=mb.EmbeddingClient(self.cfg))
        self.assertEqual(ids[0], "exact")
        self.assertTrue(all(isinstance(i, str) for i in ids))
        for key in ("abstained", "relevance", "filters", "query", "collapsed"):
            self.assertIn(key, meta)

    def test_unknown_option_is_a_type_error_not_a_silent_default(self):
        with self.assertRaises(TypeError) as cm:
            mb.params_from_cfg(self.cfg, min_relevnce=0.4)
        self.assertIn("unknown search option", str(cm.exception))

    def test_one_result_per_thread(self):
        """Two mails that repeat the same sentence are one answer. Without
        this the top of the list is the same paragraph twice and every other
        result is pushed out of a ten-slot list."""
        hits, meta = search(self.store, self.cfg, "transfer code domein verloopt",
                            top=10)
        ids = [h.message_id for h in hits]
        self.assertTrue(set(ids) & {"t1", "t2"}, "the thread should be found")
        self.assertFalse({"t1", "t2"} <= set(ids),
                         "both mails of one thread are one answer")
        self.assertGreaterEqual(meta["collapsed"], 1)

    @staticmethod
    def _hit(mid, digest, thread=""):
        return mb.Hit(message_id=mid, score=1.0, fused=1.0, dense=None,
                      dense_rank=None, lex=1.0, lex_rank=1,
                      date=datetime(2026, 1, 1, tzinfo=timezone.utc),
                      sender="a@example.nl", subject="s", label="body",
                      tier="small", text="t", url="", hash=digest,
                      thread_id=thread or mid)

    def test_the_collapse_key_is_the_whole_body_not_the_window_that_matched(self):
        """The chunk that made a message a hit is often a window, and a window's
        fingerprint covers only part of the letter. Two messages carrying the
        same text then carry two fingerprints, and the reader gets the same
        paragraph twice while the collapse believes it stopped exactly that."""
        searcher = mb.Searcher(self.cfg, self.store, mb.EmbeddingClient(self.cfg))
        params = mb.SearchParams(top=10, collapse=True, collapse_sim=0.97)
        idx = mb.IndexArrays(vecs=np.zeros((0, 4)))
        hits = [self._hit("m1", "window-hash-a"), self._hit("m2", "window-hash-b")]

        kept, dropped = searcher._collapse(idx, hits, params, {},
                                           {"m1": "same-letter", "m2": "same-letter"})
        self.assertEqual([h.message_id for h in kept], ["m1"])
        self.assertEqual(dropped, 1)

        kept2, dropped2 = searcher._collapse(idx, hits, params, {})
        self.assertEqual(len(kept2), 2,
                         "without the whole-body map the window fingerprints "
                         "decide, and two windows of one letter differ")
        self.assertEqual(dropped2, 0)

    def test_canonical_body_hashes_covers_the_messages_asked_for(self):
        got = self.store.canonical_body_hashes(["exact", "t1", "not-a-message"])
        self.assertEqual(set(got), {"exact", "t1"},
                         "a message with no body chunk has no body fingerprint")
        self.assertNotEqual(got["exact"], got["t1"])

    def test_abstention_is_a_flag_the_caller_can_act_on(self):
        """Below the floor the list is not thrown away - the caller says 'no
        result above the relevance floor' and shows the closest as reference -
        but it must say so, because an empty-looking inbox and a wrong one are
        different problems for the user."""
        hits, meta = search(self.store, self.cfg, "zwangerschapscursus buurt",
                            min_relevance=0.95)
        self.assertTrue(meta["abstained"])
        self.assertLess(meta["relevance"], 0.95)
        self.assertEqual(meta["min_relevance"], 0.95)
        loose, meta2 = search(self.store, self.cfg, "zwangerschapscursus buurt",
                              min_relevance=0.0)
        self.assertTrue(loose, "the floor is the only reason for the abstention")
        self.assertFalse(meta2["abstained"])

    def test_clean_text_splits_the_new_words_from_the_dragged_along(self):
        body = ("Ik heb de factuur ontvangen, dank u.\n"
                "> Op 12 maart schreef Kim:\n"
                "> De betaling is nog niet ontvangen, herinnering.\n"
                "-- \n"
                "Voorbeeld Media - this mail is automated - do not reply.")
        clean, dropped = mb.clean_text(body)
        self.assertIn("factuur ontvangen", clean)
        self.assertNotIn("herinnering", clean)
        self.assertIn("herinnering", dropped)
        self.assertNotIn("do not reply", clean)


class FakeIMAP:
    """Enough of imaplib to serve the read-only surface out of a dict.

    `boxes` maps a folder name to `{uid: raw RFC822 bytes}`. Every command is
    recorded as it arrives, and anything that could change the mailbox raises on
    the spot: a `SELECT` without readonly, any `UID` verb that is not SEARCH or
    FETCH, and any attribute not implemented here. So a reconcile that finishes
    without tripping any of those is a reconcile that only read - which is the
    claim worth testing, and not something a look at the source proves.
    """

    def __init__(self, boxes, uidvalidity=None, missing=(), gmids=None,
                 no_structure=False):
        self.boxes = boxes
        self.uidvalidity = {name: (uidvalidity or {}).get(name, 1)
                            for name in boxes}
        self.missing = set(missing)          # folders that refuse to be examined
        #: Gmail's own id per UID, for the `X-GM-MSGID` fetch item. Absent UIDs
        #: get a derived one, so a test only names the ids it cares about.
        self.gmids = {int(k): str(v) for k, v in (gmids or {}).items()}
        #: a server that answers nothing for `BODYSTRUCTURE`. Not paranoia: the
        #: section probe is the path that stays when the structure is missing,
        #: and a test can only reach it by turning the short path off.
        self.no_structure = no_structure
        self.commands = []
        self.selected = None
        self.untagged_responses = {}

    # -- the imaplib surface the fetcher uses -------------------------------
    def logout(self):
        self.commands.append(("LOGOUT", ()))
        return ("BYE", [b""])

    def noop(self):
        """Keep-alive: the one command that asks for nothing and proves the
        socket still answers."""
        self.commands.append(("NOOP", ()))
        return ("OK", [b""])

    def select(self, box, readonly=False):
        name = self._name(box)
        self.commands.append(("EXAMINE" if readonly else "SELECT", (box, readonly)))
        if not readonly:
            raise AssertionError("a read-write SELECT would let this connection write")
        if name not in self.boxes or name in self.missing:
            self.untagged_responses = {}
            return ("NO", [b"[NONEXISTENT] no such folder"])
        self.selected = name
        self.untagged_responses = {
            "UIDVALIDITY": [str(self.uidvalidity[name]).encode()],
            "EXISTS": [str(len(self.boxes[name])).encode()],
        }
        return ("OK", [str(len(self.boxes[name])).encode()])

    def uid(self, command, *args):
        self.commands.append((f"UID {command}".upper(), args))
        cmd = command.upper()
        if cmd not in ("SEARCH", "FETCH"):
            raise AssertionError(f"UID {cmd} would change the mailbox")
        if self.selected is None:
            raise AssertionError("a UID command with no folder examined")
        msgs = self.boxes[self.selected]
        if cmd == "SEARCH":
            want = b" ".join(a for a in args if isinstance(a, (bytes, bytearray)))
            hdr = re.search(rb'HEADER Message-ID "(.*?)"', want, re.I)
            uids = sorted(msgs)
            if hdr:                             # the attachment fallback's join
                needle = hdr.group(1).lower()

                def _mid(u):
                    found = re.search(rb"Message-ID:\s*(\S+)", msgs[u], re.I)
                    return found.group(1).lower() if found else b""

                uids = [u for u in uids if needle in _mid(u)]
            return ("OK", [b" ".join(str(u).encode() for u in uids)])
        want, items = bytes(args[0]), bytes(args[-1])
        up = items.upper()
        headers = b"HEADER.FIELDS" in up
        sections = re.findall(rb"BODY\.PEEK\[([0-9.]+)(\.MIME)?\]", up)
        if sections:
            # A per-part fetch: the answer is that MIME part's headers or its
            # bytes, tagged with the section it belongs to — which is the only
            # way a command about sixteen sections can be read back at all.
            out, missing = [], []
            for uid in self._set(want):
                raw = msgs.get(uid)
                if raw is None:
                    continue
                parts = self._parts(raw)
                for sec, kind in sections:
                    part = parts.get(sec.decode())
                    if part is None:
                        missing.append(sec)
                        continue
                    # Headers and body separately, the way a server sends them:
                    # `as_bytes()` would join them with a newline this fake would
                    # have to split again, and line endings in the test message
                    # are not guaranteed to be the ones the wire uses.
                    head = b"".join(
                        b"%s: %s\r\n" % (str(k).encode("utf-8", "surrogateescape"),
                                         str(v).encode("utf-8", "surrogateescape"))
                        for k, v in part.items())
                    raw_body = part.get_payload()
                    # A container's payload is a list of children, so its plain
                    # section answers empty here. Nothing in this program reads
                    # one: `BODY[k.MIME]` is what says "look inside", and the
                    # bytes are always read from a leaf section.
                    body = (raw_body.encode("utf-8", "surrogateescape")
                            if isinstance(raw_body, str) else b"")
                    if kind:
                        out.append((f"{uid} (UID {uid} BODY[{sec.decode()}"
                                    f".MIME] {{{len(head)}}}".encode(), head))
                    else:
                        out.append((f"{uid} (UID {uid} BODY[{sec.decode()}"
                                    f"] {{{len(body)}}}".encode(), body))
            if missing:
                # Gmail's answer to a FETCH naming a section the message does not
                # have, and the reason this fake used to be wrong: it handed over
                # the sections that *did* exist and said nothing about the ones
                # that did not, so a blind probe of sixteen sections on a
                # three-part mail looked like a working probe here and came back
                # empty on the real server. One absent item fails the command.
                return ("NO", [b"Some messages could not be FETCHed (Failure)"])
            return ("OK", out)
        if b"BODYSTRUCTURE" in up:
            if self.no_structure:
                return ("OK", [])
            out = []
            for uid in self._set(want):
                raw = msgs.get(uid)
                if raw is None:
                    continue
                tag = (f"{uid} (UID {uid} BODYSTRUCTURE "
                       + self._bodystructure(raw).decode("latin-1"))
                out.append((tag.encode("latin-1"), b""))
            return ("OK", out)
        gm = b"X-GM-MSGID" in up
        out = []
        for uid in self._set(want):
            raw = msgs.get(uid)
            if raw is None:
                continue
            if headers:
                found = re.search(rb"Message-ID:\s*(\S+)", raw, re.I)
                payload = b"Message-ID: %s\r\n\r\n" % (found.group(1) if found else b"")
                tag = (f"{uid} (UID {uid} BODY[HEADER.FIELDS (MESSAGE-ID)]"
                       f" {{{len(payload)}}}")
            elif gm and b"RFC822" not in up:
                # Gmail's own id, in the response line and nowhere in the body —
                # and that line comes back from imaplib *on its own*, not as a
                # `(prefix, literal)` pair, because this item carries no literal.
                # Handing back pairs here is what let `gm_msgid_map` read nothing
                # from Gmail's answer for two years while every test stayed green.
                if self.gm_of(uid) is None:
                    continue
                out.append(f"{uid} (X-GM-MSGID {self.gm_of(uid)} UID {uid})"
                           .encode("latin-1"))
                continue
            else:
                payload, gm_id = raw, self.gm_of(uid) if gm else None
                tag = (f"{uid} (UID {uid}"
                       + (f" X-GM-MSGID {gm_id}" if gm_id else "")
                       + f" RFC822 {{{len(raw)}}}")
            out.append((tag.encode(), payload))
        return ("OK", out)

    def gm_of(self, uid):
        """Gmail's numeric id for a UID, or None when this folder has no id for it.

        With no `gmids` named, every UID gets a derived one — stable and
        predictable, so a paging test can assert the hex link it should produce.
        Naming any id means *only* the named UIDs have one, which is how a test
        shows a UID Gmail answered no id for.
        """
        if self.gmids:
            return self.gmids.get(int(uid))
        return str(1_000_000_000_000_000_000 + int(uid) * 7919)

    @staticmethod
    def _bodystructure(raw) -> bytes:
        """A BODYSTRUCTURE for a raw message, spelled the way RFC 3501 spells it.

        Built from the message rather than written out per test, so the section
        numbers in it are the same ones `_parts` serves on the FETCH that follows.
        A fake that disagreed with itself about which section holds which file
        would test nothing about a real server — and this fake's disagreement is
        exactly where the production bug its new `NO` answer now reproduces was
        sleeping.
        """
        import email as emaillib

        def q(value):
            if value in (None, ""):
                return b"NIL"
            text = str(value).replace('"', "").replace("\\", "")
            return b'"' + text.encode("utf-8", "surrogateescape") + b'"'

        def params(part, keep_boundary=False):
            out = []
            for key, value in (part.get_params() or []):
                if key.lower() == "boundary" and not keep_boundary:
                    continue
                out += [q(key), q(value)]
            return out

        def disposition(part):
            raw_disp = part.get("Content-Disposition")
            if not raw_disp:
                return b"NIL", b"NIL"
            pieces = raw_disp.split(";")
            out = [q(pieces[0].strip())]
            for piece in pieces[1:]:
                key, _, value = piece.partition("=")
                out += [q(key.strip()), q(value.strip().strip('"'))]
            return out[0], b"(" + b" ".join(out[1:]) + b")"

        def leaf(part):
            maintype, _, subtype = part.get_content_type().partition("/")
            body = part.get_payload(decode=False)
            size = (len(body.encode("utf-8", "surrogateescape"))
                    if isinstance(body, str) else 0)
            disp, disp_params = disposition(part)
            return b"(" + b" ".join([
                q(maintype.upper()), q(subtype.upper()),
                b"(" + b" ".join(params(part)) + b")",
                b"NIL", b"NIL", q(part.get("Content-Transfer-Encoding") or "7BIT"),
                str(size).encode(), b"NIL", disp, disp_params,
                b"NIL", b"NIL"]) + b")"

        def walk(part):
            if not part.is_multipart():
                return leaf(part)
            kids = b" ".join(walk(sub) for sub in part.get_payload())
            _main, _, subtype = part.get_content_type().partition("/")
            disp, disp_params = disposition(part)
            return b"(" + kids + b" " + b" ".join([
                q(subtype.upper()), b"(" + b" ".join(params(part)) + b")",
                b"NIL", b"NIL", disp, b'"8BIT"', b"0"]) + b")"

        return walk(emaillib.message_from_bytes(raw))

    @staticmethod
    def _parts(raw):
        """`{IMAP section: part}` for one message, numbered the way RFC 3501 does.

        The top-level children of a multipart are 1..N and a nested part is
        `parent.child`, which is what makes a section number fetched from a fake
        mean the same thing as one fetched from Gmail.

        A container gets a section of its own, because Gmail answers
        `BODY[n.MIME]` for a `multipart` part with its `Content-Type` header —
        and that header is exactly how a probe learns to look one level deeper.
        """
        import email as emaillib
        msg = emaillib.message_from_bytes(raw)
        out = {}

        def walk(m, prefix):
            if m.is_multipart():
                if prefix:
                    out[prefix] = m
                for i, sub in enumerate(m.get_payload(), 1):
                    walk(sub, str(i) if not prefix else f"{prefix}.{i}")
            else:
                out[prefix or "1"] = m

        walk(msg, "")
        return out

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _name(box):
        text = box.decode() if isinstance(box, (bytes, bytearray)) else str(box)
        return text[1:-1] if text[:1] == '"' and text[-1:] == '"' else text

    @staticmethod
    def _set(want):
        ids = []
        for part in want.decode().split(","):
            if ":" in part:
                lo, hi = part.split(":")
                ids.extend(range(int(lo), int(hi) + 1))
            elif part.strip():
                ids.append(int(part))
        return ids

    def verbs(self):
        return [c[0] for c in self.commands]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AssertionError(f"{name} would change the mailbox")


class FakeFetcher(mb.GmailFetcher):
    """The real read-only fetching surface, over a FakeIMAP instead of Gmail."""

    def __init__(self, cfg, conn):
        super().__init__(cfg)
        self.conn = conn

    def _connect(self, debug=False):
        return self.conn


def raw_msg(mid, body="hello world", sender="ada@example.com",
            subject="weekly status", date="Mon, 4 May 2026 10:00:00 +0000"):
    """A whole RFC822 message, the way IMAP hands one over."""
    return (f"Message-ID: {mid}\r\nFrom: {sender}\r\nTo: bob@example.com\r\n"
            f"Date: {date}\r\nSubject: {subject}\r\n\r\n{body}\r\n").encode()


class TestReconciliation(StubbedEndpoint):
    """The index mirrors the folders: what leaves a folder leaves the index,
    what comes back to one comes back, and none of it writes to the mailbox."""

    def seed(self, *specs, folder="INBOX"):
        """Index messages at recorded UIDs: `seed((4, "<a@x>"), ...)`."""
        msgs = []
        for uid, mid in specs:
            m = make_msg(mid, body=f"bespreekbare tekst over {mid}")
            m.folder = folder
            m.uid = uid
            msgs.append(m)
        index_messages(self.store, self.cfg, msgs)

    def reconcile(self, boxes, folders=("INBOX",), uidvalidity=None, **over):
        conn = FakeIMAP(boxes, uidvalidity=uidvalidity)
        out = mb.reconcile_deletions(self.cfg, self.store, list(folders),
                                     fetcher=FakeFetcher(self.cfg, conn),
                                     quiet=True, **over)
        return out, conn

    def count(self, sql):
        return self.store.conn.execute(sql).fetchone()[0]

    # -- the behaviour the user asked for -----------------------------------
    def test_mail_the_folder_no_longer_has_leaves_the_index(self):
        self.seed((4, "<keep@x>"), (5, "<gone@x>"))
        out, _ = self.reconcile({"INBOX": {4: raw_msg("<keep@x>")}})
        self.assertEqual(out["removed"], 1)
        self.assertFalse(self.store.has("<gone@x>"))
        self.assertTrue(self.store.has("<keep@x>"))
        # and nothing of the removed one is left to be searched
        self.assertEqual(
            self.count("SELECT COUNT(*) FROM chunks WHERE message_id='<gone@x>'"), 0)
        self.assertEqual(
            self.count("SELECT COUNT(*) FROM embeddings WHERE chunk_id NOT IN"
                       " (SELECT id FROM chunks)"), 0)
        self.assertEqual(
            self.count("SELECT COUNT(*) FROM chunks_fts WHERE rowid NOT IN"
                       " (SELECT id FROM chunks)"), 0)
        hits, _ = search(self.store, self.cfg, "tekst over gone", min_relevance=0.0)
        ids = [h.message_id for h in hits]
        self.assertNotIn("<gone@x>", ids)
        self.assertIn("<keep@x>", ids)

    def test_what_was_dropped_is_written_down_and_said(self):
        self.seed((4, "<keep@x>"), (5, "<gone@x>"))
        out, _ = self.reconcile({"INBOX": {4: raw_msg("<keep@x>")}})
        tom = self.store.tombstone("<gone@x>")
        self.assertIsNotNone(tom, "the index must be able to say what it dropped")
        self.assertEqual(tom["uid"], 5)
        self.assertEqual(tom["folder"], "INBOX")
        self.assertTrue(tom["removed_at"])
        self.assertEqual(self.store.stats()["removed"], 1)
        state = self.store.reconcile_state()
        self.assertEqual(state["removed_total"], 1)
        self.assertTrue(state["last_reconcile_at"])
        self.assertEqual(state["last_reconcile"]["folders"]["INBOX"]["removed"], 1)
        self.assertNotIn("gone", state["last_reconcile"]["folders"]["INBOX"],
                         "ids belong in removed_messages, not in the state blob")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            mb.print_reconciliation(out)
        self.assertIn("reconciled INBOX", printed.getvalue())
        self.assertIn("1 removed", printed.getvalue())

    def test_a_quiet_run_does_not_borrow_the_last_drop(self):
        """`last_removal_at` answers "when did mail last actually leave", so a
        later run that removed nothing must not move it: the status line prints
        it as *last dropped on*, which is a claim about a deletion."""
        self.seed((4, "<keep@x>"), (5, "<gone@x>"))
        self.reconcile({"INBOX": {4: raw_msg("<keep@x>")}})
        dropped = self.store.reconcile_state()["last_removal_at"]
        self.assertEqual(dropped, self.store.tombstone("<gone@x>")["removed_at"])
        self.reconcile({"INBOX": {4: raw_msg("<keep@x>")}})        # nothing to do
        after = self.store.reconcile_state()
        self.assertEqual(after["last_removal_at"], dropped)
        self.assertGreaterEqual(after["last_reconcile_at"], dropped)
        self.assertEqual(self.store.stats()["last_removal_at"], dropped)

    def test_mail_the_folder_holds_and_the_index_never_fetched_is_named(self):
        """"The folders are mirrored" has to be true in both directions, and this
        run only owns one of them: a UID the folder lists with no row here was
        never fetched, and an incremental run will not walk back for it. The
        mirror says so instead of quietly being smaller."""
        self.seed((4, "<a@x>"), (5, "<b@x>"))
        out, _ = self.reconcile({"INBOX": {4: raw_msg("<a@x>"),
                                           5: raw_msg("<b@x>"),
                                           9: raw_msg("<never-fetched@x>")}})
        self.assertEqual(out["removed"], 0, "mail in the folder is not mail gone")
        self.assertEqual(out["folders"]["INBOX"]["missing"], 1)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            mb.print_reconciliation(out)
        self.assertIn("1 in the folder were never indexed", printed.getvalue())
        self.assertIn("index --full", printed.getvalue())
        self.assertEqual(self.store.reconcile_state()["last_reconcile"]
                         ["folders"]["INBOX"]["missing"], 1,
                         "the GUI reads the same brief and should see the gap")

    def test_a_folder_that_was_not_asked_about_keeps_its_rows(self):
        self.seed((4, "<inbox@x>"))
        self.seed((77, "<sent@x>"), folder="[Gmail]/Sent Mail")
        out, _ = self.reconcile({"INBOX": {4: raw_msg("<inbox@x>")}})
        self.assertEqual(out["removed"], 0)
        self.assertTrue(self.store.has("<sent@x>"))

    def test_mail_that_came_back_is_reindexed(self):
        """Un-deleting in Gmail has to reach search again, even though the
        incremental fetch is watermarked by date and would never offer it."""
        self.seed((4, "<a@x>"), (5, "<b@x>"))
        self.reconcile({"INBOX": {4: raw_msg("<a@x>")}})
        self.assertFalse(self.store.has("<b@x>"))
        self.store.set_state("last_indexed_date", "2027-01-01T00:00:00+00:00")
        out, conn = self.reconcile({"INBOX": {4: raw_msg("<a@x>"),
                                              9: raw_msg("<b@x>")}})
        self.assertEqual(out["restored"], 1)
        self.assertTrue(self.store.has("<b@x>"))
        self.assertIsNone(self.store.tombstone("<b@x>"),
                          "a message that is back is no longer a removal")
        self.assertEqual(self.store.folder_rows("INBOX"), [(("<a@x>"), 4), ("<b@x>", 9)])
        self.assertIn("UID FETCH", conn.verbs(), "the repair is one targeted FETCH")

    def test_restore_is_bounded_when_the_folder_holds_undiscovered_mail(self):
        """A partly indexed mailbox has thousands of live UIDs with no row: probing
        them all would turn a refresh into a re-index of the folder."""
        self.seed((4, "<a@x>"), (5, "<b@x>"))
        self.reconcile({"INBOX": {4: raw_msg("<a@x>")}})
        self.cfg["restore_probe_max"] = 2
        live = {4: raw_msg("<a@x>")}
        live.update({20 + i: raw_msg(f"<new{i}@x>") for i in range(6)})
        out, conn = self.reconcile({"INBOX": live})
        self.assertEqual(out["restored"], 0)
        self.assertFalse(self.store.has("<b@x>"))

    # -- the guards: refuse rather than half-do it ---------------------------
    def test_a_folder_that_lists_nothing_is_not_believed(self):
        self.seed((4, "<a@x>"))
        out, _ = self.reconcile({"INBOX": {}})
        self.assertEqual(out["removed"], 0)
        self.assertTrue(self.store.has("<a@x>"))
        self.assertIn("0 messages", out["folders"]["INBOX"]["skipped"])

    def test_a_folder_that_cannot_be_examined_removes_nothing(self):
        self.seed((4, "<a@x>"))
        conn = FakeIMAP({"INBOX": {}}, missing=["INBOX"])
        out = mb.reconcile_deletions(self.cfg, self.store, ["INBOX"],
                                     fetcher=FakeFetcher(self.cfg, conn), quiet=True)
        self.assertEqual(out["removed"], 0)
        self.assertTrue(self.store.has("<a@x>"))
        self.assertIn("could not be listed", out["folders"]["INBOX"]["skipped"])

    def test_a_run_that_would_eat_the_index_needs_to_be_forced(self):
        self.seed(*[(uid, f"<m{uid}@x>") for uid in range(1, 9)])
        out, _ = self.reconcile({"INBOX": {1: raw_msg("<m1@x>")}})
        self.assertEqual(out["removed"], 0)
        self.assertIn("prune_max_fraction", out["folders"]["INBOX"]["skipped"])
        self.assertEqual(self.count("SELECT COUNT(*) FROM messages"), 8)
        out2, _ = self.reconcile({"INBOX": {1: raw_msg("<m1@x>")}}, force=True)
        self.assertEqual(out2["removed"], 7)

    def test_a_changed_uidvalidity_remaps_rather_than_deletes(self):
        """A folder re-created under a new UIDVALIDITY numbers everything again;
        taken at face value that is 'every message is gone'."""
        self.seed((900, "<a@x>"), (901, "<b@x>"))
        self.store.set_state("imap_uidvalidity:INBOX", "1")
        out, _ = self.reconcile({"INBOX": {7: raw_msg("<a@x>"), 8: raw_msg("<b@x>")}},
                                uidvalidity={"INBOX": 2})
        self.assertEqual(out["removed"], 0)
        self.assertIn("UIDVALIDITY changed 1 -> 2", out["folders"]["INBOX"]["skipped"])
        self.assertEqual(self.store.folder_rows("INBOX"), [("<a@x>", 7), ("<b@x>", 8)])
        self.assertEqual(self.store.get_state("imap_uidvalidity:INBOX"), "2")

    def test_dry_run_says_it_without_doing_it(self):
        self.seed((4, "<a@x>"), (5, "<b@x>"))
        out, _ = self.reconcile({"INBOX": {4: raw_msg("<a@x>")}}, dry_run=True)
        self.assertEqual(out["folders"]["INBOX"]["removed"], 1)
        self.assertTrue(self.store.has("<b@x>"))
        self.assertIsNone(self.store.tombstone("<b@x>"))
        self.assertEqual(self.store.removed_count(), 0)
        self.assertIsNone(self.store.get_state("last_reconcile_at"))
        self.assertIn("<b@x>", out["folders"]["INBOX"]["gone"])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            mb.print_reconciliation(out)
        self.assertIn("would be removed", printed.getvalue())

    def test_a_dry_run_leaves_the_index_exactly_as_it_found_it(self):
        """Including the UIDs it could have recorded and the UIDVALIDITY it could
        have stored: `--dry-run` is what gets run to decide whether to trust the
        real run, so it may not be the thing that changes the index."""
        self.seed((0, "<a@x>"), (0, "<b@x>"))
        out, _ = self.reconcile({"INBOX": {4: raw_msg("<a@x>")}}, dry_run=True)
        self.assertTrue(out["folders"]["INBOX"]["swept"])
        self.assertEqual(out["folders"]["INBOX"]["uids_backfilled"], 1)
        self.assertEqual(self.store.folder_rows("INBOX"), [("<a@x>", 0), ("<b@x>", 0)])
        self.assertIsNone(self.store.get_state("imap_uidvalidity:INBOX"))
        self.assertEqual(self.store.stats()["messages"], 2)

    # -- matching, and the one-time cost of an index without UIDs ------------
    def test_an_index_without_uids_sweeps_headers_once(self):
        self.seed((0, "<a@x>"), (0, "<b@x>"))
        out, conn = self.reconcile({"INBOX": {4: raw_msg("<a@x>")}})
        self.assertTrue(out["folders"]["INBOX"]["swept"])
        self.assertEqual(out["removed"], 1)
        self.assertEqual(self.store.folder_rows("INBOX"), [("<a@x>", 4)])
        self.assertIn("UID FETCH", conn.verbs())
        # and the second look needs no headers at all
        out2, conn2 = self.reconcile({"INBOX": {4: raw_msg("<a@x>")}})
        self.assertFalse(out2["folders"]["INBOX"]["swept"])
        self.assertEqual(out2["removed"], 0)
        self.assertNotIn("UID FETCH", conn2.verbs())

    def test_a_uid_is_not_evidence_of_absence_by_itself(self):
        """While headers are being swept a row counts as alive on either proof:
        a message whose id Gmail reports differently must not be deleted on a
        technicality."""
        self.seed((0, "<other@x>"), (9, "<odd@x>"))
        out, _ = self.reconcile({"INBOX": {9: raw_msg("<different@x>"),
                                           4: raw_msg("<other@x>")}})
        self.assertEqual(out["removed"], 0, "uid 9 is live, whatever its header says")
        self.assertTrue(self.store.has("<odd@x>"))

    def test_case_in_the_message_id_is_not_a_deletion(self):
        """Gmail reports Message-ID lowercase; the header usually is not."""
        self.seed((0, "<ABC@Example.com>"))
        out, _ = self.reconcile({"INBOX": {7: raw_msg("<abc@example.com>")}})
        self.assertEqual(out["removed"], 0)
        self.assertTrue(self.store.has("<ABC@Example.com>"))
        self.assertEqual(self.store.folder_rows("INBOX"), [("<ABC@Example.com>", 7)])

    # -- the wire, and the one rule that holds everywhere ---------------------
    def test_reconciling_never_asks_to_change_the_mailbox(self):
        self.seed((0, "<a@x>"), (0, "<b@x>"))
        out, conn = self.reconcile({"INBOX": {4: raw_msg("<a@x>")}})
        self.assertEqual(out["removed"], 1)
        allowed = {v.split()[-1] for v in conn.verbs()}     # "UID FETCH" -> "FETCH"
        self.assertEqual(allowed - set(mb.GmailFetcher.READ_ONLY_COMMANDS), set(),
                         "the index may only ever read the mailbox")
        # the tripwire itself, so the assertion above means something:
        with self.assertRaises(AssertionError):
            conn.select(b'"INBOX"')                       # read-write SELECT
        with self.assertRaises(AssertionError):
            conn.uid("STORE", b"4", b"+FLAGS (\\Deleted)")
        with self.assertRaises(AssertionError):
            getattr(conn, "expunge")

    def test_folder_names_and_criteria_reach_the_server_as_meant(self):
        """imaplib quotes every `str` argument, which is why the mailbox name
        must be quoted by hand and the criteria must not be quoted at all."""
        f = mb.GmailFetcher(self.cfg)
        self.assertEqual(f._bname("INBOX"), b'"INBOX"')
        self.assertEqual(f._bname("[Gmail]/All Mail"), b'"[Gmail]/All Mail"')
        self.assertEqual(f._bytes("SINCE 01-Oct-2026"), b"SINCE 01-Oct-2026")
        box = {"[Gmail]/Sent Mail": {4: raw_msg("<a@x>")}}
        conn = FakeIMAP(box)
        fetcher = FakeFetcher(self.cfg, conn)
        with fetcher.session() as c:
            self.assertEqual(fetcher.select_readonly(c, "[Gmail]/Sent Mail"), 1)
            self.assertEqual(fetcher.snapshot(c, "[Gmail]/Sent Mail").uids, {4})
        self.assertEqual(conn.verbs()[0], "EXAMINE")
        self.assertEqual(conn.commands[0],
                         ("EXAMINE", (b'"[Gmail]/Sent Mail"', True)),
                         "the folder name reaches the server quoted by us, in bytes")
        self.assertIn(("UID SEARCH", (None, b"ALL")), conn.commands)

    def test_a_fetched_message_carries_the_uid_it_arrived_on(self):
        conn = FakeIMAP({"INBOX": {9: raw_msg("<a@x>")}})
        fetcher = FakeFetcher(self.cfg, conn)
        with fetcher.session() as c:
            fetcher.select_readonly(c, "INBOX")
            got = list(fetcher._fetch_page(c, [b"9"], "INBOX"))
        self.assertEqual([m.uid for m in got], [9])

    def test_reindexing_without_a_uid_keeps_the_one_on_record(self):
        self.seed((5, "<a@x>"))
        index_messages(self.store, self.cfg, [make_msg("<a@x>", body="opnieuw")])
        self.assertEqual(self.store.folder_rows("INBOX"), [("<a@x>", 5)])

    def test_the_prune_is_skipped_where_it_would_be_wrong(self):
        def ns(**over):
            base = {"no_prune": False, "db": None}
            base.update(over)
            return type("A", (), base)()

        self.assertIsNone(mb._prune_skipped(self.cfg, ns(), False))
        self.assertIn("--limit", mb._prune_skipped(self.cfg, ns(), True))
        self.assertIn("--no-prune", mb._prune_skipped(self.cfg, ns(no_prune=True), False))
        self.assertIn("not the database this config indexes",
                      mb._prune_skipped(self.cfg, ns(db="elsewhere.db"), False))
        off = make_cfg(self.db, prune_deleted=False)
        self.assertIn("prune_deleted", mb._prune_skipped(off, ns(), False))


def raw_with_file(mid, filename, data, content_type="application/pdf"):
    """A real MIME message carrying one file part, the way Gmail sends one."""
    from email.message import EmailMessage
    m = EmailMessage()
    m["Message-ID"] = mid
    m["From"] = "ada@example.com"
    m["Subject"] = "factuur"
    m.set_content("hierbij de bijlage")
    maintype, _, subtype = content_type.partition("/")
    m.add_attachment(data, maintype=maintype, subtype=subtype or "octet-stream",
                     filename=filename)
    return bytes(m)


class TestOpeningAttachments(StubbedEndpoint):
    """The index keeps text, not files: opening an attachment re-reads the
    message from its folder, read-only, and picks the named part out of it."""

    def fetcher(self, boxes, **kw):
        conn = FakeIMAP(boxes, **kw)
        return FakeFetcher(self.cfg, conn), conn

    def test_fetch_message_reads_the_whole_message_without_marking_it_seen(self):
        raw = raw_msg("<a@x>")
        fetcher, conn = self.fetcher({"INBOX": {7: raw}})
        with fetcher.session() as c:
            got = fetcher.fetch_message(c, "INBOX", 7)
        self.assertEqual(got, raw)
        self.assertIn("EXAMINE", conn.verbs())
        asked = [a for v, a in conn.commands if v == "UID FETCH"]
        self.assertEqual(asked, [(b"7", b"(UID BODY.PEEK[])")],
                         "BODY.PEEK[]: the full message, never marking it seen")

    def test_a_uid_that_is_not_there_answers_none_not_an_error(self):
        fetcher, _ = self.fetcher({"INBOX": {7: raw_msg("<a@x>")}})
        with fetcher.session() as c:
            self.assertIsNone(fetcher.fetch_message(c, "INBOX", 9))

    def test_a_row_without_a_uid_is_found_by_its_message_id(self):
        raw = raw_msg("<gone@x>", body="oud")
        fetcher, conn = self.fetcher({"INBOX": {41: raw}})
        with fetcher.session() as c:
            got = fetcher.fetch_by_message_id(c, "INBOX", "<gone@x>")
        self.assertEqual(got, raw)
        self.assertIn(("UID SEARCH", (None, b'(HEADER Message-ID "<gone@x>")')),
                      conn.commands)
        with fetcher.session() as c:
            self.assertIsNone(fetcher.fetch_by_message_id(c, "INBOX", "<nope@x>"))

    def test_reading_a_file_back_issues_nothing_but_reads(self):
        raw = raw_with_file("<a@x>", "factuur.pdf", b"%PDF-1.4 test")
        fetcher, conn = self.fetcher({"INBOX": {7: raw}})
        with fetcher.session() as c:
            fetcher.fetch_message(c, "INBOX", 7)
            fetcher.fetch_by_message_id(c, "INBOX", "<a@x>")
        allowed = {v.split()[-1] for v in conn.verbs()}
        self.assertEqual(allowed - set(mb.GmailFetcher.READ_ONLY_COMMANDS), set())

    def test_message_ref_names_the_folder_and_uid_or_says_the_row_is_gone(self):
        m = make_msg("<a@x>")
        m.uid = 12
        index_messages(self.store, self.cfg, [m])
        self.assertEqual(self.store.message_ref("<a@x>"), ("INBOX", 12))
        self.assertIsNone(self.store.message_ref("<never@x>"))


class TestPickAttachment(unittest.TestCase):
    """The click sends back the name the pane showed; the part that answers
    has to be matched on the same decoded name, not on a raw header."""

    def test_a_named_file_part_is_returned_with_its_bytes(self):
        raw = raw_with_file("<a@x>", "Factuur 2026.pdf", b"%PDF-1.4 factuur")
        att = mb.pick_attachment(raw, "Factuur 2026.pdf")
        self.assertIsNotNone(att)
        self.assertEqual(att.data, b"%PDF-1.4 factuur")
        self.assertEqual(att.content_type, "application/pdf")

    def test_the_match_is_case_insensitive_like_the_pane_display(self):
        raw = raw_with_file("<a@x>", "Factuur.pdf", b"x")
        self.assertIsNotNone(mb.pick_attachment(raw, "factuur.PDF"))

    def test_an_encoded_filename_meets_the_decoded_name_the_pane_shows(self):
        raw = (b"Subject: x\r\nMessage-ID: <a@x>\r\n"
               b"Content-Type: application/pdf; name=\"x.pdf\"\r\n"
               b"Content-Disposition: attachment;\r\n"
               b" filename=\"=?UTF-8?Q?factuur_2026-01=2Epdf?=\"\r\n"
               b"Content-Transfer-Encoding: base64\r\n\r\n"
               b"UERGLTEuNCBiYXNlNjQgYnl0ZXM=\r\n")
        att = mb.pick_attachment(raw, "factuur 2026-01.pdf")
        self.assertIsNotNone(att, "the pane shows the decoded name and sends it back")
        self.assertEqual(att.data, b"PDF-1.4 base64 bytes")

    def test_no_part_no_name(self):
        raw = raw_msg("<a@x>")
        self.assertIsNone(mb.pick_attachment(raw, "factuur.pdf"))
        self.assertIsNone(mb.pick_attachment(raw_msg("<a@x>"), ""))


# ---------------------------------------------------------------------------
# Previews, one part at a time, one login
# ---------------------------------------------------------------------------

NEEDS_PIL = "a preview is a Pillow re-encode, and Pillow is not installed here"
try:
    from PIL import Image
except ImportError:
    Image = None


def png_bytes(side=240, mode="RGB", color=(9, 130, 200), transparent=False):
    """A decodable PNG. `make_thumb` looks at pixels, so the magic-plus-junk
    bytes other tests hand round are not a picture to Pillow.

    `transparent=True` puts a real see-through half in it. An RGBA image made by
    `Image.new` alone has an alpha channel that is opaque everywhere, which is
    what most mail attachments look like in practice, and is not transparency.
    """
    img = Image.new(mode, (side, side), color)
    if transparent and "A" in img.getbands():
        alpha = Image.new("L", (side, side), 255)
        alpha.paste(Image.new("L", (side, side // 2), 0), (0, 0))
        img.putalpha(alpha)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def raw_mixed(mid, files, body="hierbij de bijlage"):
    """A real `multipart/mixed`: one body part, then each `(name, mime, bytes)`.

    Built with the stdlib rather than written out, because the section numbers
    this exercises are the ones `email` and IMAP both agree on.
    """
    from email.message import EmailMessage
    m = EmailMessage()
    m["Message-ID"] = mid
    m["From"] = "ada@example.com"
    m["Subject"] = "plaatjes"
    m.set_content(body)
    for name, mime, data in files:
        maintype, _, subtype = mime.partition("/")
        m.add_attachment(data, maintype=maintype,
                         subtype=subtype or "octet-stream", filename=name)
    return bytes(m)


def raw_related(mid, filename, data, mime="image/png"):
    """A picture one level down: `mixed` → `related` → (html, the picture).

    The shape HTML mail really uses, and the reason a probe that stops at the
    top level would answer "no such file" for an attachment that is visible.
    """
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    subtype = mime.partition("/")[2] or "png"
    mixed = MIMEMultipart("mixed")
    mixed["Message-ID"] = mid
    mixed.attach(MIMEText("hierbij de bijlage"))
    related = MIMEMultipart("related")
    related.attach(MIMEText("<p>hoi</p>", "html"))
    img = MIMEImage(data, _subtype=subtype or "png")
    img.add_header("Content-Disposition", "attachment", filename=filename)
    related.attach(img)
    mixed.attach(related)
    return mixed.as_bytes()


@unittest.skipIf(Image is None, NEEDS_PIL)
class TestMakeThumb(unittest.TestCase):
    """A preview comes out of the bytes the indexer is holding anyway, and one
    it cannot read is a missing preview — never a failed index run."""

    def test_a_big_picture_comes_back_small_and_smaller(self):
        raw = png_bytes(1200)
        data, mime, w, h = mb.make_thumb(raw)
        self.assertEqual(mime, "image/jpeg")
        self.assertLessEqual(max(w, h), mb.THUMB_MAX_SIDE)
        self.assertLess(len(data), len(raw))

    def test_a_transparent_image_stays_a_png(self):
        _data, mime, _w, _h = mb.make_thumb(png_bytes(600, "RGBA",
                                                      transparent=True))
        self.assertEqual(mime, "image/png",
                         "alpha flattened to black is a ruined screenshot")

    def test_an_alpha_channel_hiding_nothing_is_no_reason_for_a_png(self):
        """An RGBA picture with every pixel opaque is what a screenshot exported
        by half of Windows looks like. Keeping PNG for it costs the mailbox its
        largest previews: 190 KB for a 320 px picture that had nothing to show
        through."""
        noisy = Image.frombytes("RGB", (480, 480), os.urandom(480 * 480 * 3))
        opaque = noisy.convert("RGBA")          # a channel, and nothing to hide
        src = io.BytesIO()
        opaque.save(src, format="PNG")
        data, mime, _w, _h = mb.make_thumb(src.getvalue())
        self.assertEqual(mime, "image/jpeg",
                         "a channel that is opaque everywhere is not transparency")
        same = Image.open(io.BytesIO(src.getvalue()))
        same.thumbnail((mb.THUMB_MAX_SIDE, mb.THUMB_MAX_SIDE))
        flat = io.BytesIO()
        same.save(flat, format="PNG", optimize=True)
        self.assertLess(len(data), len(flat.getvalue()),
                        "the JPEG is the smaller copy of the same pixels")

    def test_the_size_asked_for_is_the_size_made(self):
        _data, _mime, w, h = mb.make_thumb(png_bytes(1200), max_side=64)
        self.assertLessEqual(max(w, h), 64)

    def test_bytes_that_are_not_a_picture_have_no_preview(self):
        self.assertIsNone(mb.make_thumb(b"\x89PNG\r\n\x1a\n" + b"0" * 64))
        self.assertIsNone(mb.make_thumb(b""))


@unittest.skipIf(Image is None, NEEDS_PIL)
class TestMediaCache(unittest.TestCase):
    """Previews live in a sidecar: derived location, a budget it keeps itself,
    and no row outliving the message it was cut from."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(str(Path(self.tmp.name) / "mailbox.db"),
                            media_cache_mb=1)
        self.mc = mb.MediaCache(mb.media_db_path(self.cfg), self.cfg)
        self.addCleanup(self.mc.close)

    def test_the_sidecar_sits_beside_the_index_it_belongs_to(self):
        self.assertEqual(Path(mb.media_db_path(self.cfg)).name,
                         "mailbox.media.db")
        self.assertEqual(Path(mb.media_db_path({"db_path": "x/baseline.db"})),
                         Path("x") / "baseline.media.db")
        self.assertEqual(
            mb.media_db_path({"db_path": "x/mailbox.db",
                              "media_db_path": "other/m.db"}), "other/m.db")

    def test_a_preview_round_trips_with_the_size_the_pane_needs(self):
        made = self.mc.put_attachment("m1", "shot.png", png_bytes(900))
        self.assertEqual(made["mime"], "image/jpeg")
        got = self.mc.get("m1", "shot.png")
        self.assertEqual((got["width"], got["height"]),
                         (made["width"], made["height"]))
        self.assertTrue(got["bytes"].startswith(b"\xff\xd8"))   # JPEG magic

    def test_a_file_that_is_not_a_picture_stores_nothing(self):
        self.assertIsNone(self.mc.put_attachment("m1", "factuur.pdf",
                                                 b"%PDF-1.4 x"))
        self.assertIsNone(self.mc.get("m1", "factuur.pdf"))
        self.assertEqual(self.mc.count(), 0)

    def test_a_page_asks_for_its_previews_in_one_query(self):
        self.mc.put_attachment("m1", "a.png", png_bytes())
        self.mc.put_attachment("m1", "b.png", png_bytes())
        self.mc.put_attachment("m2", "c.png", png_bytes())
        self.assertEqual(sorted(self.mc.for_message("m1")), ["a.png", "b.png"])
        self.assertEqual(self.mc.first_for_messages(["m1", "m2", "verwijderd"]),
                         {"m1": "a.png", "m2": "c.png"})
        self.assertEqual(self.mc.first_for_messages([]), {})

    def test_dropping_messages_drops_their_previews(self):
        self.mc.put_attachment("m1", "a.png", png_bytes())
        self.mc.put_attachment("m2", "b.png", png_bytes())
        self.assertEqual(self.mc.delete_messages(["m1", ""]), 1)
        self.assertEqual(self.mc.thumb_message_ids(), {"m2"})
        self.assertEqual(self.mc.delete_messages([]), 0)

    def test_the_sweep_drops_what_nobody_opened_first(self):
        """Least-recently-opened goes first — and since the clock that stamps
        "opened" is second-precise, the test has to move that clock to show the
        order at all."""
        stamps = [f"2026-01-0{ i + 1}T00:00:00+00:00" for i in range(6)]
        blob = b"x" * 300_000
        with mock.patch.object(mb, "_now_iso", lambda: stamps.pop(0)):
            for i in range(4):
                self.mc.put(f"m{i}", "p.png", blob, "image/png")
            self.mc.get("m3", "p.png")             # the one still being looked at
        self.assertEqual(self.mc.prune_to_budget(), 1)
        self.assertFalse(self.mc.has("m0", "p.png"))
        self.assertTrue(self.mc.has("m3", "p.png"))
        self.assertLessEqual(self.mc.stats()["bytes"], self.mc.budget_bytes)

    def test_a_burst_over_budget_comes_back_under_it(self):
        for i in range(6):
            self.mc.put(f"m{i}", "p.png", b"x" * 300_000, "image/png")
        gone = self.mc.prune_to_budget()
        self.assertGreaterEqual(gone, 3)
        self.assertEqual(self.mc.count(), 6 - gone)
        self.assertLessEqual(self.mc.stats()["bytes"], self.mc.budget_bytes)

    def test_a_section_is_remembered_per_folder_and_uid(self):
        self.assertIsNone(self.mc.part("INBOX", 7, "a.png"))
        self.mc.remember_part("INBOX", 7, "a.png", "2.1", 4242)
        self.assertEqual(self.mc.part("INBOX", 7, "a.png"), ("2.1", 4242))
        self.assertIsNone(self.mc.part("Verzonden", 7, "a.png"),
                          "the same UID in another folder is another message")
        self.mc.remember_part("INBOX", 7, "a.png", "3", 99)
        self.assertEqual(self.mc.part("INBOX", 7, "a.png"), ("3", 99))

    def test_clearing_loses_previews_and_keeps_the_learned_sections(self):
        self.mc.put_attachment("m1", "a.png", png_bytes())
        self.mc.remember_part("INBOX", 7, "a.png", "2", 10)
        self.assertEqual(self.mc.clear(), 1)
        self.assertEqual(self.mc.count(), 0)
        self.assertEqual(self.mc.part("INBOX", 7, "a.png"), ("2", 10))

    def test_the_state_table_holds_a_resume_point(self):
        self.assertEqual(self.mc.get_state("media_watermark", "0"), "0")
        self.mc.set_state("media_watermark", "4100")
        self.assertEqual(self.mc.get_state("media_watermark"), "4100")

    def test_two_threads_get_their_own_connection_and_the_same_rows(self):
        """The GUI reads the sidecar from its media workers while the indexer
        writes from its own thread, so a single shared sqlite handle is not an
        option and per-thread connections are. `close()` closes the caller's,
        which is also what lets a worker's temp directory be cleaned up."""
        import threading
        self.mc.put_attachment("m1", "a.png", png_bytes())
        seen = []

        def reader():
            seen.append(self.mc.get("m1", "a.png") is not None)
            self.mc.close()                    # this thread's connection only

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=10)
        self.assertEqual(seen, [True])
        self.assertTrue(self.mc.has("m1", "a.png"), "the writer's own view is intact")


@unittest.skipIf(Image is None, NEEDS_PIL)
class TestPreviewsDuringIndexing(StubbedEndpoint):
    """Previews are cut while indexing, from the bytes the embedding call is
    already given, and they leave when the message leaves."""

    def setUp(self):
        super().setUp()
        self.media = self.store.media_cache()
        self.addCleanup(lambda: self.media and self.media.close())

    def test_an_indexed_image_gets_a_preview(self):
        made = index_messages(self.store, self.cfg, [
            make_msg("m1", atts=[mb.Attachment("logo.png", "image/png",
                                               png_bytes(700))])],
            media=self.media)
        self.assertEqual(made, 1)
        self.assertEqual(sorted(self.media.for_message("m1")), ["logo.png"])

    def test_a_preview_is_never_cut_for_a_file_the_index_skipped(self):
        """The indexer caps attachment size and only reads four image types;
        a preview for a file it would not have indexed would promise a click
        that cannot land."""
        over = make_msg("big", atts=[mb.Attachment(
            "groot.png", "image/png",
            png_bytes(700) + b"0" * (1024 * 1024 * int(
                self.cfg["max_attachment_mb"]) + 1))])
        tiff = make_msg("tif", atts=[mb.Attachment("scan.tiff", "image/tiff",
                                                   png_bytes(700))])
        index_messages(self.store, self.cfg, [over, tiff], media=self.media)
        self.assertEqual(self.media.for_message("big"), {})
        self.assertEqual(self.media.for_message("tif"), {})

    def test_two_parts_with_one_name_get_one_preview_of_the_right_part(self):
        first, second = png_bytes(700, "RGB", (200, 9, 9)), png_bytes(700, "RGB", (9, 9, 200))
        raw = raw_mixed("<dup@x>", [("x.png", "image/png", first),
                                    ("x.png", "image/png", second)])
        msg = make_msg("dup", atts=[mb.Attachment("x.png", "image/png", first),
                                    mb.Attachment("x.png", "image/png", second)])
        index_messages(self.store, self.cfg, [msg], media=self.media)
        self.assertEqual(len(self.media.for_message("dup")), 1)
        self.assertTrue(self.media.has("dup", "x.png"))
        self.assertEqual(mb.pick_attachment(raw, "x.png").data, first)

    def test_a_dropped_message_leaves_no_preview_behind(self):
        index_messages(self.store, self.cfg, [
            make_msg("m1", atts=[mb.Attachment("a.png", "image/png",
                                               png_bytes())])], media=self.media)
        self.assertTrue(self.media.has("m1", "a.png"))
        self.assertEqual(len(self.store.remove_messages(["m1"])), 1)
        self.assertEqual(self.media.count(), 0)


class TestMediaCacheToggle(StubbedEndpoint):
    """Previews are a choice, and switching them off has to undo the local
    storage completely — including not creating the file by looking for it."""

    def setUp(self):
        super().setUp()
        self.sidecar = Path(mb.media_db_path(self.cfg))

    def test_no_sidecar_is_opened_when_previews_are_off(self):
        store = self.open_store(make_cfg(self.db, store_thumbs=False))
        self.assertIsNone(store.media_cache())
        self.assertFalse(self.sidecar.exists())
        self.assertEqual(store.drop_thumbnails(["m1"]), 0)
        self.assertFalse(self.sidecar.exists(),
                         "cleaning up must not create the file it cleans")

    def test_a_prune_that_found_nothing_leaves_no_file_behind(self):
        store = self.open_store()
        opened = []
        self.addCleanup(lambda: [m.close() for m in opened])
        self.assertIsNone(store.media_cache(only_if_exists=True))
        self.assertFalse(self.sidecar.exists())
        opened.append(store.media_cache())
        self.assertTrue(self.sidecar.exists())


class TestReadingThePartList(unittest.TestCase):
    """The one line a server says about a message's structure, read without
    trusting it: every section it names is re-read and checked before a file is
    served, so what these tests have to get right is the numbering, the names and
    the sizes — and the refusal to invent any of the three."""

    def leaves(self, text):
        tree = mb._structure_list(text)
        return None if tree is None else mb._structure_parts(tree)

    def test_the_leaves_are_numbered_the_way_imap_numbers_them(self):
        got = self.leaves(
            b'((("TEXT" "PLAIN" ("text/plain" NIL "charset" "us-ascii") NIL NIL '
            b'"7bit" 18 NIL NIL NIL NIL NIL) ("IMAGE" "PNG" ("image/png" NIL) NIL'
            b' NIL "base64" 48472 NIL "attachment" ("filename" '
            b'"image001.png") NIL NIL) "RELATED" ("multipart/related" NIL) NIL '
            b'NIL NIL "8BIT" 0) ("APPLICATION" "PDF" NIL NIL NIL "base64" 2048 '
            b'NIL "attachment" ("filename" "factuur.pdf") NIL NIL) "MIXED" '
            b'("multipart/mixed" NIL) NIL NIL NIL "8BIT" 0)')
        self.assertEqual([(s, n, o) for s, n, _t, o in got],
                         [("1.1", None, 18), ("1.2", "image001.png", 48472),
                          ("2", "factuur.pdf", 2048)],
                         "a container is a place to look, never a file")

    def test_the_replys_own_closing_parenthesis_is_not_part_of_the_tree(self):
        """imaplib hands over the whole untagged line, and the tree is only the
        first s-expression in it: what closes the *response* after it, and
        anything a server appends (`BODY=8BITTEXT`), is not a part."""
        got = self.leaves(
            b'(("IMAGE" "JPEG" NIL NIL NIL "base64" 11 NIL "inline" '
            b'("filename" "a.jpg") NIL NIL) "RELATED" NIL NIL NIL NIL "8BIT" 0)'
            b') BODY=8BITTEXT')
        self.assertEqual([(s, n) for s, n, _t, _o in got], [("1", "a.jpg")])

    def test_a_name_carried_by_the_type_instead_of_the_disposition_is_found(self):
        got = self.leaves(b'(("IMAGE" "GIF" ("name" "logo.gif") NIL NIL "base64" '
                          b'64 NIL NIL NIL NIL NIL) "MIXED" NIL NIL NIL NIL '
                          b'"8BIT" 0)')
        self.assertEqual([(s, n) for s, n, _t, _o in got], [("1", "logo.gif")])

    def test_a_description_is_the_last_place_a_name_can_hide(self):
        got = self.leaves(b'(("APPLICATION" "OCTET-STREAM" NIL "id" "diag.dat" '
                          b'"binary" 12 NIL NIL NIL NIL NIL) "MIXED" NIL NIL NIL '
                          b'NIL "8BIT" 0)')
        self.assertEqual([(s, n) for s, n, _t, _o in got], [("1", "diag.dat")])

    def test_a_message_that_is_one_part_has_no_numbered_parts(self):
        """No container, so no section numbers — and no guessing at `BODY[1]`,
        which one server means as the text and another as nothing. `[]` here is
        the caller's cue to probe or to read the whole message."""
        self.assertEqual(self.leaves(b'("TEXT" "PLAIN" NIL NIL NIL "7bit" 12 '
                                     b'NIL NIL NIL NIL NIL)'), [])

    def test_a_tree_that_does_not_parse_is_answered_with_none(self):
        for broken in (b'(("TEXT" "PLAIN" NIL NIL NIL "7bit" 1',
                       b'("TEXT" "PLAIN ("charset" "utf-8"',
                       b'4152 (UID 5163 BODYSTRUCTURE (("IMAGE" "PNG" NIL NIL NIL'
                       b' "7bit" 1 NIL NIL NIL NIL NIL) "MIXED" NIL NIL NIL NIL '
                       b'"8BIT" 0)',
                       b'("A" "B" NIL NIL NIL "7bit" 1 NIL NIL {9} short)',
                       b'something that is not a tree at all',
                       b''):
            self.assertIsNone(mb._structure_list(broken), broken)

    def test_a_literal_string_is_one_value_however_it_is_spelled(self):
        """`{n}` means the n bytes after the brace, spaces and punctuation free:
        a filename carrying a comma is sent that way by some servers."""
        got = self.leaves(b'(("IMAGE" "PNG" NIL NIL NIL "base64" 5 NIL '
                          b'"attachment" ("filename" {14}a cat, two.png) NIL NIL)'
                          b' "MIXED" NIL NIL NIL NIL "8BIT" 0)')
        self.assertEqual([(s, n) for s, n, _t, _o in got], [("1", "a cat, two.png")])

    def test_nil_in_every_optional_field_still_lists_the_parts(self):
        got = self.leaves(b'(("TEXT" "HTML" NIL NIL NIL "8bit" 90 NIL NIL NIL NIL'
                          b' NIL) ("IMAGE" "PNG" NIL NIL NIL NIL NIL NIL NIL NIL '
                          b'NIL NIL) "RELATED" NIL NIL NIL NIL NIL 0)')
        self.assertEqual([(s, n, o) for s, n, _t, o in got],
                         [("1", None, 90), ("2", None, 0)],
                         "a part with no name and no size is still a part, named"
                         " by nothing and sized at nothing")

    def test_a_part_counted_in_a_way_that_is_not_a_number_is_counted_as_nothing(self):
        got = self.leaves(b'(("IMAGE" "PNG" NIL NIL NIL "base64" NIL NIL "inline"'
                          b' ("filename" "x.png") NIL NIL) "MIXED" NIL NIL NIL NIL'
                          b' "8BIT" 0)')
        self.assertEqual([(s, n, o) for s, n, _t, o in got], [("1", "x.png", 0)],
                         "a size that cannot be read must not stop the file being"
                         " found, and must not pretend to be big")


class TestReadingOneFileInsteadOfTheMessage(StubbedEndpoint):
    """The click asks for one file. Reading the whole message to hand it over is
    what made a logo inside a big mail cost the whole mail, so the part is read
    on its own — and only when it can be *proved* to be the named file."""

    PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 24
    PDF = b"%PDF-1.4 factuur 2026-014"

    def fetcher(self, boxes, **kw):
        conn = FakeIMAP(boxes, **kw)
        return FakeFetcher(self.cfg, conn), conn

    @contextlib.contextmanager
    def opened(self, fetcher, folder="INBOX"):
        """A connection with the folder examined.

        `fetch_attachment` does not select — choosing the folder is the
        connection's job (`GmailSession` in the GUI, `session()`+`select_readonly`
        in the indexer). Asking for a part with nothing examined is a protocol
        error, and this is where a test says the protocol was followed.
        """
        with fetcher.session() as conn:
            fetcher.select_readonly(conn, folder)
            yield conn

    def sidecar(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        cfg = make_cfg(str(Path(tmp.name) / "m.db"))
        media = mb.MediaCache(mb.media_db_path(cfg), cfg)
        self.addCleanup(media.close)
        return media

    def boxes(self):
        return {"INBOX": {7: raw_mixed("<a@x>", [
            ("logo.png", "image/png", self.PNG),
            ("factuur.pdf", "application/pdf", self.PDF)])}}

    def items(self, conn):
        return [a[-1] for v, a in conn.commands if v == "UID FETCH"]

    def test_a_named_file_is_read_as_one_part(self):
        fetcher, _conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            att = fetcher.fetch_attachment(c, "INBOX", 7, "logo.png")
        self.assertIsNotNone(att)
        self.assertEqual(att.data, self.PNG)
        self.assertEqual(att.content_type, "image/png")
        self.assertEqual(att.name, "logo.png")

    def test_the_whole_message_never_crosses_the_wire(self):
        fetcher, conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            fetcher.fetch_attachment(c, "INBOX", 7, "factuur.pdf")
        whole = [i for i in self.items(conn)
                 if b"RFC822" in i or b"BODY.PEEK[]" in i or b"BODY[]" in i]
        self.assertEqual(whole, [], "the point of the exercise")

    def test_it_answers_exactly_what_reading_the_message_would_have(self):
        """The part read is not allowed to be a different answer from the whole
        message: same part, same name, same type, same bytes."""
        raw = self.boxes()["INBOX"][7]
        fetcher, _ = self.fetcher(self.boxes())
        for name in ("logo.png", "factuur.pdf"):
            with self.opened(fetcher) as c:
                fast = fetcher.fetch_attachment(c, "INBOX", 7, name)
            slow = mb.pick_attachment(raw, name)
            self.assertEqual((fast.name, fast.content_type, fast.data),
                             (slow.name, slow.content_type, slow.data), name)

    def test_a_file_the_parts_do_not_carry_is_not_guessed_at(self):
        """`None`, so the caller reads the whole message: a section probe that
        finds nothing is not evidence that the file is not there."""
        fetcher, _ = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            self.assertIsNone(fetcher.fetch_attachment(c, "INBOX", 7,
                                                       "ontbrekend.pdf"))
            self.assertIsNone(fetcher.fetch_attachment(c, "INBOX", 7, ""))
            self.assertIsNone(fetcher.fetch_attachment(c, "INBOX", 0, "x.png"))

    def test_a_picture_one_level_down_is_still_found(self):
        data = self.PNG
        fetcher, conn = self.fetcher({"INBOX": {
            9: raw_related("<n@x>", "inline.png", data)}})
        with self.opened(fetcher) as c:
            att = fetcher.fetch_attachment(c, "INBOX", 9, "inline.png")
        self.assertIsNotNone(att, "HTML mail hides its pictures one level down")
        self.assertEqual(att.data, data)
        read = [i for i in self.items(conn)
                if b"BODY.PEEK[2.2.MIME] BODY.PEEK[2.2]" in i]
        self.assertTrue(read, "the tree is read at once, not probed level by level")

    def test_a_refused_probe_still_finds_the_file(self):
        """The regression this path exists to keep, with the short path turned off.

        Gmail's answer to a FETCH naming a MIME section the message does not have
        is `NO ... Some messages could not be FETCHed (Failure)`, and nothing else
        in that command comes back. A blind probe of sixteen sections on a
        three-part mail therefore looks exactly like a probe that found no
        matching part — so the file was never read by section and every click paid
        for the whole message, while the suite stayed green because this fake used
        to hand over the sections it had and keep silent about the ones it had
        not. The fake now refuses the command like the server does.

        So the wide probe is asked, refused, asked again in pieces, and the file
        still arrives without the message ever crossing the wire.
        """
        fetcher, conn = self.fetcher(self.boxes(), no_structure=True)
        with self.opened(fetcher) as c:
            att = fetcher.fetch_attachment(c, "INBOX", 7, "factuur.pdf")
        self.assertIsNotNone(att, "a refused batch is not an answer about the file")
        self.assertEqual(att.data, self.PDF)
        items = self.items(conn)
        self.assertIn(b"(BODY.PEEK[1.MIME] BODY.PEEK[2.MIME] BODY.PEEK[3.MIME]"
                      b" BODY.PEEK[4.MIME])", items, "the first ask is a run of four")
        self.assertIn(b"(BODY.PEEK[3.MIME] BODY.PEEK[4.MIME])", items,
                      "a refused run is asked again in halves")
        self.assertFalse([i for i in items if b"BODY.PEEK[5.MIME]" in i
                          or b"BODY.PEEK[9.MIME]" in i],
                         "a section that is simply not there ends the search")
        self.assertEqual([i for i in items
                          if b"RFC822" in i or b"BODY.PEEK[]" in i], [],
                         "reading the whole message is not how this one worked")

    def test_a_picture_one_level_down_is_found_when_the_structure_is_refused(self):
        """The nesting chase belongs to the probe path, where a container's own
        headers are what says to look inside it: a server that answers nothing for
        `BODYSTRUCTURE` still gets the inline picture out of a `multipart/related`
        one level down."""
        data = self.PNG
        fetcher, conn = self.fetcher({"INBOX": {
            9: raw_related("<n@x>", "inline.png", data)}}, no_structure=True)
        with self.opened(fetcher) as c:
            att = fetcher.fetch_attachment(c, "INBOX", 9, "inline.png")
        self.assertIsNotNone(att)
        self.assertEqual(att.data, data)
        nested = [i for i in self.items(conn) if b"BODY.PEEK[2.1.MIME]" in i]
        self.assertTrue(nested, "the second round goes where a container pointed")

    def test_the_second_click_on_a_file_is_one_command(self):
        """The first click learns which section the file lives in; the next one
        asks for that section and its headers together, without even asking what
        the message is made of."""
        media = self.sidecar()
        fetcher, conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            self.assertTrue(fetcher.fetch_attachment(c, "INBOX", 7, "logo.png",
                                                     parts=media))
            before = len(self.items(conn))
            again = fetcher.fetch_attachment(c, "INBOX", 7, "logo.png",
                                             parts=media)
        self.assertIsNotNone(again)
        section, octets = media.part("INBOX", 7, "logo.png")
        self.assertEqual(section, "2")
        self.assertGreaterEqual(octets, len(self.PNG),
                                "the size kept is the size the server stated, the "
                                "transfer-encoded one, which is the only number "
                                "that can still stop a download before it starts")
        self.assertEqual(self.items(conn)[before:],
                         [b"(BODY.PEEK[2.MIME] BODY.PEEK[2])"],
                         "the part and its headers, in one command, no probe")

    def test_reading_one_file_files_every_file_in_the_message(self):
        """The structure names all of them, so all of them are remembered: the
        second attachment of a mail a person is reading, or the ninth picture of
        a mailing a backfill is working through, starts one command in rather
        than with another search for it."""
        media = self.sidecar()
        fetcher, conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            self.assertTrue(fetcher.fetch_attachment(c, "INBOX", 7, "logo.png",
                                                     parts=media))
            self.assertEqual(media.part("INBOX", 7, "factuur.pdf")[0], "3")
            before = len(self.items(conn))
            pdf = fetcher.fetch_attachment(c, "INBOX", 7, "factuur.pdf",
                                           parts=media)
        self.assertIsNotNone(pdf)
        self.assertEqual(pdf.data, self.PDF)
        self.assertEqual(self.items(conn)[before:],
                         [b"(BODY.PEEK[3.MIME] BODY.PEEK[3])"])

    def test_a_file_bigger_than_the_budget_is_not_downloaded_to_be_refused(self):
        """A thumbnail job has no use for a 60 MB original, and the structure is
        the only moment its size is known before the bytes arrive. A person who
        clicked asks for the file and gets it: no limit is passed on that path."""
        media = self.sidecar()
        fetcher, conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            self.assertIsNotNone(fetcher.fetch_attachment(
                c, "INBOX", 7, "logo.png", parts=media, max_bytes=10 ** 6))
            self.assertIsNone(fetcher.fetch_attachment(
                c, "INBOX", 7, "factuur.pdf", parts=media, max_bytes=1))
        self.assertEqual(media.part("INBOX", 7, "factuur.pdf")[0], "3",
                         "refusing to download it does not unlearn where it is")


    def test_a_remembered_section_that_no_longer_matches_is_reprobed(self):
        """A folder renumbers under a stored UID and a message gets edited, so a
        remembered section is a hint and never a fact: the part it names is read
        and checked, and when it is not the wanted file the structure is probed
        again and the real part is served."""
        media = self.sidecar()
        media.remember_part("INBOX", 7, "logo.png", "1", 10)   # the text part
        fetcher, _conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            att = fetcher.fetch_attachment(c, "INBOX", 7, "logo.png", parts=media)
        self.assertIsNotNone(att, "a stale hint costs a probe, not the file")
        self.assertEqual(att.data, self.PNG)
        self.assertEqual(media.part("INBOX", 7, "logo.png")[0], "2",
                         "and the section is corrected for next time")

    def test_reading_one_part_asks_for_nothing_but_reads(self):
        fetcher, conn = self.fetcher(self.boxes())
        with self.opened(fetcher) as c:
            fetcher.fetch_attachment(c, "INBOX", 7, "logo.png")
            fetcher.fetch_attachment(c, "INBOX", 7, "geen.png")
        allowed = {v.split()[-1] for v in conn.verbs()}
        self.assertEqual(allowed - set(mb.GmailFetcher.READ_ONLY_COMMANDS), set())


class DeadIMAP:
    """A connection Gmail already closed. Everything asked of it aborts, which
    is the one failure a reused login is allowed to answer by reconnecting."""

    def __init__(self):
        self.commands = []

    def select(self, *a, **k):
        raise imaplib.IMAP4.abort("connection dropped")

    def uid(self, *a, **k):
        raise imaplib.IMAP4.abort("connection dropped")

    def noop(self):
        raise imaplib.IMAP4.abort("connection dropped")

    def logout(self):
        self.commands.append(("LOGOUT", ()))
        return ("BYE", [b""])


class CountingFetcher(FakeFetcher):
    """`FakeFetcher` with a tally of how often a *new* connection was asked for,
    which is the only way to see that one is being reused."""

    def __init__(self, cfg, conns):
        super().__init__(cfg, conns[0])
        self.conns = list(conns)
        self.connects = 0

    def _connect(self, debug=False):
        conn = self.conns[min(self.connects, len(self.conns) - 1)]
        self.connects += 1
        return conn


class TestReusedLogin(unittest.TestCase):
    """Opening a file used to log in, read, and log out — per click. The session
    keeps the one login, examines a folder only when it changes, and reconnects
    exactly once when Gmail has already dropped the socket."""

    def session(self, *conns, **over):
        fetcher = CountingFetcher(dict(mb.DEFAULTS), list(conns))
        sess = mb.GmailSession(fetcher, **over)
        self.addCleanup(sess.close)
        return sess, fetcher

    def test_two_files_in_one_folder_cost_one_folder_selection(self):
        conn = FakeIMAP({"INBOX": {7: raw_msg("<a@x>"), 8: raw_msg("<b@x>")}})
        sess, fetcher = self.session(conn)
        for uid in (7, 8):
            sess.call("INBOX", lambda c, u=uid:
                      c.uid("FETCH", b"%d" % u, b"(UID BODY.PEEK[])"))
        self.assertEqual([FakeIMAP._name(a[0])
                          for v, a in conn.commands if v == "EXAMINE"],
                         ["INBOX"], "one selection, two reads")
        self.assertEqual([v for v, _a in conn.commands if v == "UID FETCH"],
                         ["UID FETCH"] * 2)
        self.assertEqual(fetcher.connects, 1, "one login for both clicks")

    def test_another_folder_on_the_same_login_is_examined_first(self):
        conn = FakeIMAP({"INBOX": {}, "Verzonden": {}})
        sess, fetcher = self.session(conn)
        for folder in ("INBOX", "Verzonden", "INBOX"):
            sess.call(folder, lambda c: None)
        self.assertEqual([FakeIMAP._name(a[0])
                          for v, a in conn.commands if v == "EXAMINE"],
                         ["INBOX", "Verzonden", "INBOX"],
                         "the connection is asked where it is going, not assumed")
        self.assertEqual(fetcher.connects, 1, "still one login")

    def test_a_connection_gmail_already_closed_is_reopened_once(self):
        box = {"INBOX": {7: raw_msg("<a@x>")}}
        good = FakeIMAP(box)
        sess, fetcher = self.session(DeadIMAP(), good)
        got = sess.call("INBOX", lambda c: c.uid("FETCH", b"7", b"(UID BODY.PEEK[])"))
        self.assertEqual(got[0], "OK")
        self.assertEqual(fetcher.connects, 2, "dropped, so opened again")
        self.assertEqual(sess.reconnects, 1, "and not quietly: it is counted")

    def test_a_connection_that_dies_twice_is_not_milked(self):
        sess, fetcher = self.session(DeadIMAP(), DeadIMAP())
        with self.assertRaises(imaplib.IMAP4.abort):
            sess.call("INBOX", lambda c: None)
        self.assertEqual(fetcher.connects, 2, "one retry, then the real error")

    def test_an_idle_connection_is_woken_before_it_is_used(self):
        """A NOOP in a place that can reconnect is cheaper than a FETCH that
        dies halfway through its bytes."""
        box = {"INBOX": {7: raw_msg("<a@x>")}}
        conn = FakeIMAP(box)
        sess, fetcher = self.session(conn, keepalive_after=-1.0)
        sess.call("INBOX", lambda c: None)
        sess.call("INBOX", lambda c: None)
        self.assertIn("NOOP", [v for v, _a in conn.commands])
        self.assertEqual(fetcher.connects, 1, "woken, not replaced")

    def test_a_folder_that_is_not_there_is_an_answer_not_a_broken_pipe(self):
        conn = FakeIMAP({"INBOX": {}}, missing={"Gisteren"})
        sess, fetcher = self.session(conn)
        with self.assertRaises(mb.FolderUnavailable):
            sess.call("Gisteren", lambda c: None)
        self.assertEqual(fetcher.connects, 1, "a missing folder is not a dead socket")

    def test_closing_logs_out_and_the_next_call_opens_fresh(self):
        conn = FakeIMAP({"INBOX": {}})
        sess, fetcher = self.session(conn)
        sess.call("INBOX", lambda c: None)
        sess.close()
        self.assertIn("LOGOUT", [v for v, _a in conn.commands])
        sess.call("INBOX", lambda c: None)
        self.assertEqual(fetcher.connects, 2)


class StalledIMAP:
    """A connection Gmail has stopped answering: the command went out, and the
    read for its reply was cut short by the socket's own timeout. `TimeoutError`
    *is* what `socket.timeout` has been since Python 3.10, so this is the shape
    the real thing arrives in."""

    def __init__(self):
        self.commands = []

    def select(self, *a, **k):
        raise TimeoutError("read timed out")

    def uid(self, *a, **k):
        raise TimeoutError("read timed out")

    def noop(self):
        raise TimeoutError("read timed out")

    def logout(self):
        self.commands.append(("LOGOUT", ()))
        return ("BYE", [b""])


class TestReadTimeout(unittest.TestCase):
    """A Gmail that has gone quiet is not a request to wait forever for. The
    budget belongs on the socket, where a slow but moving transfer keeps resetting
    it and only a stall ends it — and that stall has to arrive as an error the
    reused login already reads as "replace this connection", because a bound that
    only produces a stranger exception has changed nothing about the hang."""

    def connect(self, **over):
        seen = {}

        class Recorder:
            """Stands in for `imaplib.IMAP4_SSL` and remembers what it was given."""

            def __init__(self, host, port, timeout=None):
                seen.update(host=host, port=port, timeout=timeout)

            def login(self, user, password):
                return ("OK", [b""])

        cfg = dict(mb.DEFAULTS, imap_host="imap.x", imap_port=1234, **over)
        orig = imaplib.IMAP4_SSL
        imaplib.IMAP4_SSL = Recorder
        try:
            mb.GmailFetcher(cfg)._connect()
        finally:
            imaplib.IMAP4_SSL = orig
        return seen

    def test_the_read_budget_goes_on_the_socket_of_the_connection(self):
        seen = self.connect()
        self.assertEqual(seen["host"], "imap.x")
        self.assertEqual(seen["timeout"], 90.0, "90 seconds per read, out of the box")
        self.assertEqual(self.connect(imap_timeout=5)["timeout"], 5.0,
                         "a setting, not a constant buried in the connect call")

    def test_zero_is_the_way_to_ask_for_no_limit(self):
        self.assertIsNone(self.connect(imap_timeout=0)["timeout"])
        self.assertIsNone(self.connect(imap_timeout=-1)["timeout"])

    def test_a_number_that_is_not_a_number_becomes_the_default_and_a_warning(self):
        with self.assertLogs(mb.log.name, level="WARNING"):
            seen = self.connect(imap_timeout="next tuesday")
        self.assertEqual(seen["timeout"], 90.0,
                         "a typo in a config is not a reason to hang again")

    def test_a_stalled_read_is_answered_with_a_fresh_login(self):
        """The whole point of the bound: a timeout is an `OSError`, so the one
        retry a dropped socket already gets covers a server that stopped talking,
        and the file is read on the connection after it."""
        good = FakeIMAP({"INBOX": {7: raw_msg("<a@x>")}})
        fetcher = CountingFetcher(dict(mb.DEFAULTS), [StalledIMAP(), good])
        sess = mb.GmailSession(fetcher)
        self.addCleanup(sess.close)
        got = sess.call("INBOX", lambda c:
                        c.uid("FETCH", b"7", b"(UID BODY.PEEK[])"))
        self.assertEqual(got[0], "OK")
        self.assertEqual(fetcher.connects, 2, "stalled, so opened again")
        self.assertEqual(sess.reconnects, 1, "counted, not silent")


class TestGmailMessageIds(StubbedEndpoint):
    """The web link is built from Gmail's own number for a message. Mail indexed
    before that number was asked for cannot invent one, so it is fetched for a
    whole folder in batched reads that transfer no bodies at all."""

    def fetcher(self, boxes, gmids=None, **over):
        conn = FakeIMAP(boxes, gmids=gmids)
        return FakeFetcher(dict(self.cfg, **over), conn), conn

    @staticmethod
    def items(conn):
        return [a[-1] for v, a in conn.commands if v == "UID FETCH"]

    def test_the_id_arrives_in_the_fetch_prefix_not_in_the_body(self):
        """Gmail answers `X-GM-MSGID` as a fetch item: it is in the response's
        prefix line, where no header parser would think to look."""
        fetcher, conn = self.fetcher({"INBOX": {7: raw_msg("<a@x>")}},
                                     gmids={7: "1234567890123"})
        with fetcher.session() as c:
            fetcher.select_readonly(c, "INBOX")
            msgs = list(fetcher._fetch_page(c, [b"7"], "INBOX"))
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].gmail_url,
                         "https://mail.google.com/mail/u/0/#all/"
                         + hex(1234567890123)[2:])
        self.assertIn(b"(UID X-GM-MSGID RFC822)", self.items(conn),
                      "the id is asked for with the message, in one command")

    def test_the_parser_accepts_the_id_it_is_handed(self):
        fetcher = mb.GmailFetcher.__new__(mb.GmailFetcher)     # no IMAP connection
        fetcher.cfg = {"gmail_account_index": 1}
        import email as emaillib
        raw = emaillib.message_from_string(
            "From: ada@example.com\r\nSubject: x\r\n"
            "Date: Mon, 4 May 2026 10:00:00 +0000\r\n"
            "Message-ID: <m1@example.com>\r\n\r\nbody\r\n")
        self.assertEqual(
            fetcher._parse(raw, "INBOX", uid=9, gm_msgid="999").gmail_url,
            "https://mail.google.com/mail/u/1/#all/" + hex(999)[2:])
        self.assertEqual(fetcher._parse(raw, "INBOX", uid=9).gmail_url, "",
                         "no id, no link: an empty string, never a wrong one")

    def test_a_whole_folders_ids_come_back_in_one_batched_read(self):
        box = {u: raw_msg(f"<m{u}@x>") for u in (1, 2, 3)}
        fetcher, conn = self.fetcher({"INBOX": box}, gmids={1: "11", 3: "33"})
        with fetcher.session() as c:
            got = fetcher.gm_msgid_map(c, "INBOX", [3, 1, 2, 0], pause=0)
        self.assertEqual(got, {1: "11", 3: "33"},
                         "a UID Gmail gives no id for is simply not in the map")
        asked = [(a[0], a[1]) for v, a in conn.commands if v == "UID FETCH"]
        self.assertEqual(asked, [(b"1,2,3", b"(UID X-GM-MSGID)")],
                         "one command, comma-separated: Gmail rejects spaces")

    def test_a_big_id_list_is_paged_and_paused_like_every_other_sweep(self):
        box = {u: raw_msg(f"<m{u}@x>") for u in range(1, 6)}
        fetcher, conn = self.fetcher({"INBOX": box}, reconcile_batch_size=2)
        with mock.patch.object(mb.time, "sleep") as slept:
            with fetcher.session() as c:
                got = fetcher.gm_msgid_map(c, "INBOX", list(range(1, 6)),
                                           pause=1.5)
        self.assertEqual(len(got), 5)
        self.assertEqual([a[0] for v, a in conn.commands if v == "UID FETCH"],
                         [b"1,2", b"3,4", b"5"],
                         "the same batch size the reconcile sweep uses")
        self.assertEqual(slept.call_count, 2, "a pause between pages, not after the last")


class TestMediaCommand(unittest.TestCase):
    """`media` repairs the two things mail indexed before previews existed is
    missing: the picture the reading pane shows, and the link that opens that
    mail at Gmail. Both are repairs and not migrations - no text, no vector and
    no row is rewritten, and every one of them can be undone with `--clear`.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.db = os.path.join(self.dir.name, "mailbox.db")
        self.cfg = make_cfg(self.db, reconcile_batch_size=2)
        self.cfg_path = os.path.join(self.dir.name, "config.json")
        with open(self.cfg_path, "w", encoding="utf-8") as fh:
            json.dump(self.cfg, fh)
        patch = mock.patch.object(mb.GmailFetcher, "_connect",
                                  lambda _f, debug=False: self.conn)
        patch.start()
        self.addCleanup(patch.stop)
        patch2 = mock.patch.object(mb.EmbeddingClient, "_post", fake_post)
        patch2.start()
        self.addCleanup(patch2.stop)
        self.conn = FakeIMAP({})

    def args(self, **over):
        ns = {"config": self.cfg_path, "db": None, "profile": "", "stats": False,
              "backfill": False, "limit": 0, "dry_run": False, "links": False,
              "prune_orphans": False, "clear": False}
        ns.update(over)
        return type("A", (), ns)()

    def run_media(self, **over):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mb.cmd_media(self.args(**over))
        return out.getvalue()

    def open_store(self):
        store = mb.Store(self.db, self.cfg)
        self.addCleanup(store.conn.close)
        return store

    def previews(self):
        media = self.open_store().media_cache()
        self.addCleanup(media.close)
        return media.count(), media.part_count()

    def add_image(self, mid="old1", uid=11, thumb=False, gmail_url=""):
        """One indexed mail with a 700 px picture: what a mailbox indexed before
        previews looked like (`thumb=True` for one indexed after)."""
        png = png_bytes(700)
        msg = make_msg(mid, atts=[mb.Attachment("banner.png", "image/png", png)],
                       gmail_url=gmail_url)
        msg.uid = uid
        store = mb.Store(self.db, self.cfg)
        index_messages(store, self.cfg, [msg],
                       media=store.media_cache() if thumb else None)
        store.conn.close()
        self.conn = FakeIMAP({"INBOX": {uid: raw_related(mid, "banner.png", png)}})
        return png

    def test_stats_names_the_gap_and_the_command_that_closes_it(self):
        self.add_image()
        out = self.run_media()
        self.assertIn("messages with an image and no preview: 1", out)
        self.assertIn("media --backfill", out,
                      "a count without the way to fix it is a shrug")

    @unittest.skipIf(Image is None, NEEDS_PIL)
    def test_backfill_cuts_the_preview_and_remembers_where_the_file_lives(self):
        from io import BytesIO
        self.add_image()
        out = self.run_media(backfill=True)
        self.assertIn("previews: 1 cut", out)
        media = self.open_store().media_cache()
        self.addCleanup(media.close)
        row = media.get("old1", "banner.png")
        self.assertIsNotNone(row, "the picture is there now without being opened")
        w, h = Image.open(BytesIO(row["bytes"])).size
        self.assertLessEqual(max(w, h), mb.THUMB_MAX_SIDE)
        self.assertEqual(media.part_count(), 1,
                         "the section is remembered: the next open asks for it "
                         "by number instead of probing the message")
        asked = [a[-1] for v, a in self.conn.commands if v == "UID FETCH"]
        self.assertTrue(asked)
        self.assertFalse([a for a in asked if b"RFC822" in a],
                         "reading a picture back must not read the mail around it")

    @unittest.skipIf(Image is None, NEEDS_PIL)
    def test_a_sweep_that_runs_a_while_says_so_even_between_its_pauses(self):
        """The sweep pauses and marks its place every `reconcile_batch_size`
        messages — 500 by default, which on mail carrying eight pictures each is
        the better part of an hour of silence while the numbers are still moving.
        Silence is what gets a running sweep killed for a stuck one, so the line
        comes out on a clock as well."""
        self.add_image()
        with mock.patch.object(mb, "MEDIA_REPORT_SECONDS", 0):
            out = self.run_media(backfill=True)
        self.assertIn("messages read", out,
                      "a line saying it is still working, not only that it ended")
        self.assertIn("Gmail would not hand over", out,
                      "the count that separates a slow sweep from a losing one")

    def test_a_dry_run_asks_gmail_for_nothing_at_all(self):
        self.add_image()
        out = self.run_media(backfill=True, dry_run=True)
        self.assertIn("banner.png", out, "the point of a dry run is to be checkable")
        self.assertEqual(self.conn.commands, [],
                         "a report that costs a login is not a report")
        self.assertEqual(self.previews(), (0, 0))

    def test_a_message_indexed_without_a_uid_is_left_to_the_index_and_said_so(self):
        """Reaching such a message means reading its whole body back, which is
        the cost this command exists to avoid - so it waits for the index instead,
        and says so rather than looking finished."""
        self.add_image(uid=0)
        out = self.run_media(backfill=True)
        self.assertIn("carry no UID", out)
        self.assertEqual(self.conn.commands, [])
        self.assertEqual(self.previews(), (0, 0))

    @unittest.skipIf(Image is None, NEEDS_PIL)
    def test_a_preview_of_mail_that_is_gone_is_shown_then_dropped(self):
        self.add_image(thumb=True)
        store = self.open_store()
        store.conn.execute("DELETE FROM messages WHERE message_id = 'old1'")
        store.conn.commit()
        out = self.run_media()
        self.assertIn("no longer indexed", out)
        self.assertIn("--prune-orphans", out)
        out = self.run_media(prune_orphans=True, dry_run=True)
        self.assertIn("would delete 1 preview", out)
        self.assertEqual(self.previews(), (1, 0), "a dry run deletes nothing")
        self.run_media(prune_orphans=True)
        self.assertEqual(self.previews(), (0, 0))

    @unittest.skipIf(Image is None, NEEDS_PIL)
    def test_clear_gives_the_disk_back_and_keeps_every_word(self):
        self.add_image(thumb=True)
        self.assertEqual(self.previews()[0], 1)
        out = self.run_media(clear=True, dry_run=True)
        self.assertIn("dry run", out)
        self.assertEqual(self.previews()[0], 1)
        self.run_media(clear=True)
        self.assertEqual(self.previews()[0], 0)
        self.assertEqual(self.open_store().stats()["messages"], 1,
                         "previews are copies; the index never held the file")

    def test_links_writes_only_rows_that_have_no_link(self):
        """A Gmail link can only come from Gmail's own message id, which early
        indexing did not ask for. Asking for it costs a batched FETCH of numbers
        and no message bodies - and a row that already has a link keeps it, so
        running this twice is the same as running it once."""
        png = png_bytes(60)
        first = make_msg("old1")
        first.uid = 11
        kept = make_msg("old2", subject="another mail",
                        atts=[mb.Attachment("x.png", "image/png", png)],
                        gmail_url="KEEP")
        kept.uid = 12
        store = mb.Store(self.db, self.cfg)
        index_messages(store, self.cfg, [first, kept])
        store.conn.close()
        ids = {11: "1234567890123", 12: "987654321"}
        self.conn = FakeIMAP({"INBOX": {11: raw_msg("<old1@x>"),
                                        12: raw_msg("<old2@x>")}}, gmids=ids)
        out = self.run_media(links=True)
        self.assertIn("1 link(s) written", out)
        rows = dict(self.open_store().conn.execute(
            "SELECT message_id, gmail_url FROM messages").fetchall())
        self.assertEqual(rows["old2"], "KEEP")
        want = mb.gmail_message_url(ids[11], mb.gmail_index(self.cfg))
        self.assertEqual(rows["old1"], want)
        self.assertTrue(rows["old1"].endswith(hex(int(ids[11]))[2:]),
                        "Gmail's number goes into the link the way Gmail's own "
                        "web UI writes it: decimal in, hex out")
        verbs = {v for v, _a in self.conn.commands}
        self.assertTrue(verbs <= {"EXAMINE", "UID FETCH", "LOGOUT"}, verbs)
        out = self.run_media(links=True)
        self.assertIn("already has its Gmail link", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
