#!/usr/bin/env python3
"""
Tests for the retrieval-quality harness in eval.py.

Stdlib unittest only (no pytest in requirements.txt), and no network and no
mailbox.py: the metric arithmetic, the bootstrap, the buckets and the eval-file
I/O are all pure, and the two commands that do read an index (evalgen, and
bench --offline) are driven against a throwaway in-memory / temp SQLite file.
Keeping mailbox out of the picture is deliberate - eval.py must stay testable
while mailbox.py is being edited under it.

  python test_eval.py
  python -m unittest test_eval -v
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import eval as ev


def row(rid, ranked, relevant, **over):
    """A per-query row, the shape every metric function reads."""
    data = {"id": rid, "query": rid, "ranked": list(ranked),
            "relevant": list(relevant), "forbidden": [],
            "out_of_corpus": False}
    data.update(over)
    return data


def namespace(**over):
    """An argparse namespace carrying exactly the attributes the cmd_*
    functions are documented to expect; each command reads only what it needs."""
    args = {
        "config": "config.json", "eval_file": None, "variant": None,
        "baseline": None, "save": None, "offline": None, "top": 10,
        "iters": 200, "seed": 0, "n": 20, "out": None, "query": None,
        "rank": None, "db": None,
    }
    args.update(over)
    return type("Args", (), args)()


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_recall_is_found_over_labelled(self):
        self.assertEqual(ev.recall_at_k(["a", "b"], {"a", "c"}, 2), 0.5)
        self.assertEqual(ev.recall_at_k(["c", "a", "b"], ["a", "b", "c"], 3), 1.0)
        self.assertEqual(ev.recall_at_k(["a", "b"], {"x"}, 2), 0.0)

    def test_recall_ignores_relevance_beyond_k(self):
        self.assertEqual(ev.recall_at_k(["a", "b", "z"], {"a", "z"}, 2), 0.5)

    def test_precision_denominator_is_k_not_the_result_count(self):
        self.assertAlmostEqual(ev.precision_at_k(["a", "b", "c"], {"a"}, 3), 1 / 3)
        self.assertEqual(ev.precision_at_k(["a"], {"a"}, 5), 0.2)
        self.assertEqual(ev.precision_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_hit_at_k(self):
        self.assertEqual(ev.hit_at_k(["a", "b", "c"], {"c"}, 3), 1.0)
        self.assertEqual(ev.hit_at_k(["a", "b", "c"], {"c"}, 2), 0.0)

    def test_reciprocal_rank_uses_the_first_hit(self):
        self.assertEqual(ev.reciprocal_rank(["x", "a", "b"], {"a", "b"}), 0.5)
        self.assertEqual(ev.reciprocal_rank(["x", "y"], {"a"}), 0.0)

    def test_mrr_at_k_cuts_off_at_k(self):
        self.assertAlmostEqual(ev.mrr_at_k(["a", "b", "c"], {"c"}, 3), 1 / 3)
        self.assertEqual(ev.mrr_at_k(["a", "b", "c"], {"c"}, 2), 0.0)

    def test_ndcg_known_case(self):
        # One relevant mail at position 3: DCG 1/log2(4) = 0.5, IDCG 1/log2(2)
        # = 1, so exactly half the ideal.
        self.assertAlmostEqual(ev.ndcg_at_k(["a", "b", "c"], {"c"}, 3), 0.5)

    def test_ndcg_two_relevant_out_of_place(self):
        # ranked a,x,b,c with a,b relevant in a window of 4:
        #   DCG  = 1/log2(2) + 1/log2(4)          = 1 + 0.5
        #   IDCG = 1/log2(2) + 1/log2(3)          = 1 + 0.630929...
        got = ev.ndcg_at_k(["a", "x", "b", "c"], {"a", "b"}, 4)
        self.assertAlmostEqual(got, 1.5 / (1 + 1 / 1.584962500721156), places=6)

    def test_ndcg_is_one_for_the_ideal_ordering(self):
        self.assertAlmostEqual(ev.ndcg_at_k(["b", "a", "x"], {"a", "b"}, 3), 1.0)

    def test_ndcg_never_exceeds_one(self):
        self.assertLessEqual(ev.ndcg_at_k(["a", "b", "c"], {"a"}, 3), 1.0)

    def test_empty_and_unlabelled_edges(self):
        for fn in (ev.recall_at_k, ev.precision_at_k, ev.hit_at_k, ev.ndcg_at_k):
            self.assertEqual(fn([], {"a"}, 5), 0.0, fn.__name__)
            self.assertEqual(fn(["a"], set(), 5), 0.0, fn.__name__)
            self.assertEqual(fn(["a"], {"a"}, 0), 0.0, fn.__name__)
        self.assertEqual(ev.reciprocal_rank([], {"a"}), 0.0)
        self.assertEqual(ev.mrr_at_k(["a"], None, 3), 0.0)
        self.assertEqual(ev.summarise([])["n"], 0.0)

    def test_relevant_accepts_set_list_or_single_id(self):
        ranked, k = ["x", "a", "y"], 3
        for rel in ({"a"}, ["a"], ("a",), "a"):
            self.assertAlmostEqual(ev.recall_at_k(ranked, rel, k), 1.0)

    def test_duplicate_ids_do_not_steal_a_rank(self):
        # A repeated id in a hand-edited list must not push 'b' from rank 2 to 3.
        self.assertEqual(ev.mrr_at_k(["a", "a", "b"], {"b"}, 3), 0.5)
        self.assertEqual(ev.recall_at_k(["a", "a"], {"a"}, 2), 1.0)

    def test_forbidden_at_k(self):
        self.assertEqual(ev.forbidden_at_k(["a", "bad"], {"bad"}, 2), 1.0)
        self.assertEqual(ev.forbidden_at_k(["a", "bad"], {"bad"}, 1), 0.0)
        self.assertEqual(ev.forbidden_at_k(["a"], [], 3), 0.0)

    def test_row_metrics_reports_exactly_the_metric_names(self):
        metrics = ev.row_metrics(row("q", ["a"], ["a"]))
        self.assertEqual(set(metrics), set(ev.METRICS) | {"abstained"})
        self.assertEqual(metrics["recall@1"], 1.0)


# ---------------------------------------------------------------------------
# summarise()
# ---------------------------------------------------------------------------

class TestSummarise(unittest.TestCase):

    def test_means_are_over_queries(self):
        rows = [row("a", ["x", "rel"], ["rel"]), row("b", ["x", "y"], ["rel"])]
        s = ev.summarise(rows)
        self.assertEqual(s["recall@5"], 0.5)             # 1.0 and 0.0
        self.assertAlmostEqual(s["mrr@5"], 0.25)         # 0.5 and 0.0
        self.assertEqual(s["n"], 2.0)
        self.assertEqual(s["n_scored"], 2.0)

    def test_abstention_rate(self):
        rows = [row("a", ["rel"], ["rel"], abstained=True),
                row("b", ["rel"], ["rel"], abstained=False)]
        self.assertEqual(ev.summarise(rows)["abstain_rate"], 0.5)

    def test_abstention_can_be_derived_from_the_relevance_floor(self):
        rows = [row("a", ["rel"], ["rel"], relevance=0.21, min_relevance=0.30),
                row("b", ["rel"], ["rel"], relevance=0.80, min_relevance=0.30)]
        s = ev.summarise(rows)
        self.assertEqual(s["abstain_rate"], 0.5)
        self.assertEqual(ev.bucket_errors(rows)["ok"], ["b"])
        self.assertEqual(ev.bucket_errors(rows)["below_threshold"], ["a"])

    def test_out_of_corpus_rows_do_not_drag_the_recall_mean(self):
        rows = [row("a", ["rel"], ["rel"]),
                row("ood", ["junk"], [], out_of_corpus=True, abstained=True)]
        s = ev.summarise(rows)
        self.assertEqual(s["n"], 2.0)
        self.assertEqual(s["n_scored"], 1.0)
        self.assertEqual(s["n_out_of_corpus"], 1.0)
        self.assertEqual(s["recall@5"], 1.0, "the probe must not count here")
        self.assertEqual(s["abstain_rate"], 0.0, "abstaining on a real query "
                                                 "is what abstain_rate is for")
        self.assertEqual(s["abstain_rate_out_of_corpus"], 1.0)

    def test_explicit_retrievable_flag_wins(self):
        self.assertEqual(ev.summarise([row("a", [], [], retrievable=False)])
                         ["n_scored"], 0.0)
        self.assertEqual(ev.summarise([row("a", [], [], out_of_corpus=True,
                                           retrievable=True)])["n_scored"], 1.0)

    def test_empty_input_is_printable(self):
        s = ev.summarise([])
        self.assertEqual(s["n"], 0.0)
        for name in ev.METRICS:
            self.assertEqual(s[name], 0.0)


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------

def _pair_fixture(a_hits, b_hits, n=10):
    """Two runs over n queries: A answers the queries in a_hits, B the ones in
    b_hits - so the direction of the delta is known by construction."""
    rows_a, rows_b = [], []
    for i in range(n):
        rel = f"rel{i}"
        rows_a.append(row(f"q{i}", [rel] if i in a_hits else ["junk"], [rel]))
        rows_b.append(row(f"q{i}", [rel] if i in b_hits else ["junk"], [rel]))
    return rows_a, rows_b


class TestBootstrap(unittest.TestCase):

    def test_delta_direction_and_ci_exclude_zero(self):
        rows_a, rows_b = _pair_fixture({0, 1, 2, 3, 4, 5, 6, 7}, {0, 1})
        res = ev.paired_bootstrap_ci(rows_a, rows_b, "recall@5", iters=500, seed=7)
        delta, lo, hi = res                      # unpacks as a triple
        self.assertAlmostEqual(delta, 0.6)
        self.assertGreater(lo, 0.0, "A clearly beats B: the interval must sit "
                                    "above zero")
        self.assertGreater(res.hi, 0.5, "six of ten queries flipped: the upper "
                                        "bound has to stay well above zero")
        self.assertTrue(res.excludes_zero)
        self.assertEqual(res.n, 10)

    def test_deterministic_for_a_seed(self):
        rows_a, rows_b = _pair_fixture({0, 2, 4, 6, 8}, {1, 3})
        one = ev.paired_bootstrap_ci(rows_a, rows_b, "mrr@10", iters=300, seed=3)
        two = ev.paired_bootstrap_ci(rows_a, rows_b, "mrr@10", iters=300, seed=3)
        self.assertEqual((one.delta, one.lo, one.hi), (two.delta, two.lo, two.hi))

    def test_pairs_by_id_not_by_position(self):
        rows_a, rows_b = _pair_fixture({0, 1, 2, 3}, {4})
        straight = ev.paired_bootstrap_ci(rows_a, rows_b, "recall@5", seed=1)
        shuffled = ev.paired_bootstrap_ci(rows_a, list(reversed(rows_b)),
                                          "recall@5", seed=1)
        self.assertEqual(straight.delta, shuffled.delta)
        self.assertEqual((straight.lo, straight.hi), (shuffled.lo, shuffled.hi))

    def test_identical_runs_give_a_zero_interval(self):
        rows_a, rows_b = _pair_fixture({0, 1, 2}, {0, 1, 2})
        res = ev.paired_bootstrap_ci(rows_a, rows_b, "ndcg@10", seed=0)
        self.assertEqual((res.delta, res.lo, res.hi), (0.0, 0.0, 0.0))
        self.assertFalse(res.excludes_zero)

    def test_abstained_is_a_valid_metric(self):
        rows_a = [row("a", ["rel"], ["rel"], abstained=True)]
        rows_b = [row("a", ["rel"], ["rel"], abstained=False)]
        res = ev.paired_bootstrap_ci(rows_a, rows_b, "abstained", seed=0)
        self.assertEqual(res.delta, 1.0)

    def test_fewer_than_two_queries_is_explicitly_insufficient(self):
        rows_a, rows_b = _pair_fixture({0}, {1}, n=1)
        res = ev.paired_bootstrap_ci(rows_a, rows_b, "recall@5")
        self.assertTrue(res.insufficient)
        self.assertFalse(res.excludes_zero)
        self.assertEqual(res.n, 1)
        self.assertEqual((res.delta, res.lo, res.hi), (1.0, 1.0, 1.0))
        self.assertIn("insufficient", str(res))
        empty = ev.paired_bootstrap_ci([], [], "recall@5")
        self.assertTrue(empty.insufficient)
        self.assertEqual(empty.delta, 0.0)

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(KeyError):
            ev.paired_bootstrap_ci([], [], "vibes")

    def test_unshared_queries_are_dropped_not_padded(self):
        rows_a = [row("a", ["rel"], ["rel"]), row("b", ["rel"], ["rel"])]
        rows_b = [row("a", [], ["rel"])]
        res = ev.paired_bootstrap_ci(rows_a, rows_b, "recall@5")
        self.assertEqual(res.n, 1)
        self.assertTrue(res.insufficient)


# ---------------------------------------------------------------------------
# bucket_errors()
# ---------------------------------------------------------------------------

class TestBuckets(unittest.TestCase):

    def test_all_five_buckets(self):
        noise = [f"n{i}" for i in range(60)]
        rows = [
            row("ok", ["rel", "x", "y"], ["rel"]),
            row("buried", noise[:9] + ["rel"], ["rel"]),
            row("missing", noise + ["rel"], ["rel"]),
            row("threshold", ["rel", "x"], ["rel"], abstained=True),
            row("ood", ["junk"], [], out_of_corpus=True),
        ]
        buckets = ev.bucket_errors(rows)
        self.assertEqual(buckets["ok"], ["ok"])
        self.assertEqual(buckets["buried"], ["buried"])
        self.assertEqual(buckets["missing_from_pool"], ["missing"])
        self.assertEqual(buckets["below_threshold"], ["threshold"])
        self.assertEqual(buckets["not_in_corpus"], ["ood"])
        self.assertEqual(sum(ev.bucket_counts(buckets).values()), len(rows))

    def test_depth_boundaries(self):
        # Position 3 is reachable, position 4 is already buried.
        self.assertEqual(ev.bucket_errors([row("a", ["x", "y", "rel"], ["rel"])])
                         ["ok"], ["a"])
        self.assertEqual(ev.bucket_errors([row("a", ["x", "y", "z", "rel"],
                                               ["rel"])])["buried"], ["a"])

    def test_the_first_cause_wins(self):
        # Relevant at rank 9 and the run abstained: 'buried' is the honest
        # cause, so 'below_threshold' must not also claim it.
        noise = [f"n{i}" for i in range(8)]
        buckets = ev.bucket_errors(
            [row("a", noise + ["rel"], ["rel"], abstained=True)])
        self.assertEqual(buckets["buried"], ["a"])
        self.assertEqual(buckets["below_threshold"], [])

    def test_unlabelled_scorable_row_is_not_in_corpus(self):
        buckets = ev.bucket_errors([row("a", ["x", "y"], [])])
        self.assertEqual(buckets["not_in_corpus"], ["a"])
        self.assertEqual(buckets["missing_from_pool"], [])

    def test_custom_pool_and_depth(self):
        # A rank-2 hit is buried once the human only reaches 1, and outside
        # the pool once the retriever only kept 1 candidate.
        self.assertEqual(ev.bucket_errors([row("a", ["x", "rel"], ["rel"])],
                                          pool=3, depth=1)["buried"], ["a"])
        self.assertEqual(ev.bucket_errors([row("a", ["x", "rel"], ["rel"])],
                                          pool=1,
                                          depth=1)["missing_from_pool"], ["a"])

    def test_bucket_counts_are_ordered_and_complete(self):
        counts = ev.bucket_counts(ev.bucket_errors([]))
        self.assertEqual(list(counts), list(ev.BUCKETS))
        self.assertEqual(sum(counts.values()), 0)

    def test_rows_without_id_fall_back_to_the_query(self):
        buckets = ev.bucket_errors([{"query": "kaas", "ranked": [],
                                     "out_of_corpus": True}])
        self.assertEqual(buckets["not_in_corpus"], ["kaas"])


# ---------------------------------------------------------------------------
# Eval-file I/O
# ---------------------------------------------------------------------------

class TestEvalFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "eval.jsonl")

    def write(self, text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def read(self):
        return ev.load_rows(self.path)

    def test_blank_lines_and_comments_are_skipped(self):
        self.write("# a note\n\n"
                   '{"id": "a", "query": "q"}\n'
                   "\n   \n# another\n")
        rows, errors = self.read()
        self.assertEqual(len(rows), 1)
        self.assertEqual(errors, [])

    def test_bad_json_reports_its_line_number(self):
        self.write('{"id": "a", "query": "q"}\n'
                   '{"id": "b", "query": "oops"\n'
                   '{"id": "c", "query": "r"}\n')
        rows, errors = self.read()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["line"], 2)
        self.assertIn("JSON", errors[0]["error"])

    def test_non_object_and_missing_query_are_reported(self):
        self.write('["not", "an", "object"]\n{"id": "x"}\n{"query": "   "}\n')
        rows, errors = self.read()
        self.assertEqual(rows, [])
        self.assertEqual([e["line"] for e in errors], [1, 2, 3])
        self.assertIn("object", errors[0]["error"])
        self.assertIn("query", errors[1]["error"])

    def test_missing_file_is_an_error_not_a_crash(self):
        rows, errors = ev.load_rows(os.path.join(self.tmp.name, "nope.jsonl"))
        self.assertEqual(rows, [])
        self.assertEqual(errors[0]["line"], 0)
        self.assertIn("cannot read", errors[0]["error"])

    def test_unknown_keys_are_preserved(self):
        self.write('{"id": "a", "query": "q", "intent": "factuur", "score": 3}\n')
        rows, _ = self.read()
        self.assertEqual(rows[0]["intent"], "factuur")
        self.assertEqual(rows[0]["score"], 3)

    def test_defaults_are_filled_in(self):
        self.write('{"query": "factuur", "relevant": "<m1@x>"}\n'
                   '{"query": "domein", "relevant": ["<m2@x>", "<m3@x>"]}\n')
        rows, errors = self.read()
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["id"], "l1", "a missing id falls back to the "
                                             "line number so it stays fixable")
        self.assertEqual(rows[0]["relevant"], ["<m1@x>"], "one id as a string "
                                                          "is still one id")
        self.assertEqual(rows[1]["relevant"], ["<m2@x>", "<m3@x>"])
        self.assertEqual(rows[0]["forbidden"], [])
        self.assertFalse(rows[0]["out_of_corpus"])

    def test_out_of_corpus_string_is_read_as_a_bool(self):
        self.write('{"query": "kaas", "out_of_corpus": "true"}\n')
        rows, _ = self.read()
        self.assertIs(rows[0]["out_of_corpus"], True)
        self.assertIn("out_of_corpus", rows[0]["note"])

    def test_duplicate_ids_are_reported_and_dropped(self):
        self.write('{"id": "same", "query": "a"}\n{"id": "same", "query": "b"}\n')
        rows, errors = self.read()
        self.assertEqual([r["id"] for r in rows], ["same"])
        self.assertEqual(errors[0]["line"], 2)
        self.assertIn("duplicate", errors[0]["error"])

    def test_save_round_trip_keeps_labels_and_order(self):
        rows = [{"id": "a", "query": "facturen", "relevant": ["<m1>"],
                 "forbidden": ["<m2>"], "out_of_corpus": False,
                 "note": "hand labelled", "extra": 1}]
        self.assertEqual(ev.save_rows(rows, self.path), 1)
        back, errors = self.read()
        self.assertEqual(errors, [])
        self.assertEqual(back, rows)
        with open(self.path, encoding="utf-8") as fh:
            first_key = list(json.loads(fh.read().strip()))[0]
        self.assertEqual(first_key, "id", "the canonical key order is what "
                                          "makes the file diffable")

    def test_save_creates_the_directory(self):
        nested = os.path.join(self.tmp.name, "deeper", "eval.jsonl")
        ev.save_rows([{"id": "a", "query": "q"}], nested)
        self.assertTrue(os.path.exists(nested))

    def test_shipped_example_file_loads_cleanly(self):
        example = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "eval", "eval.example.jsonl")
        if not os.path.exists(example):
            self.skipTest("eval/eval.example.jsonl not next to this file")
        rows, errors = ev.load_rows(example)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(rows), 4)
        self.assertTrue(any(r["out_of_corpus"] for r in rows))
        self.assertTrue(all(r["query"] and r["id"] for r in rows))

    def test_row_id_is_stable_and_distinct(self):
        self.assertEqual(ev.row_id_for("van marc"), ev.row_id_for("van marc"))
        self.assertNotEqual(ev.row_id_for("van marc"), ev.row_id_for("Van Marc"))
        self.assertTrue(ev.row_id_for("").startswith("query-"))
        self.assertTrue(ev.row_id_for("factuur 8812-KX").startswith(
            "factuur-8812-kx-"), "readable slug first, so the file stays greppable")


# ---------------------------------------------------------------------------
# Variant parsing
# ---------------------------------------------------------------------------

class TestVariants(unittest.TestCase):

    def test_bare_name_is_the_defaults(self):
        variant = ev.parse_variant("huidig")
        self.assertEqual((variant.name, variant.search, variant.config),
                         ("huidig", {}, {}))
        self.assertEqual(variant.describe(), "config defaults")

    def test_keys_types_and_the_cfg_prefix(self):
        variant = ev.parse_variant(
            "lex=mode=lexical,agg_k=5,rrf_k=1.5,prefix=false,dense_pool=none,"
            "tiers=small|large,cfg.w_lex=0.8")
        self.assertEqual(variant.name, "lex")
        self.assertEqual(variant.search,
                         {"mode": "lexical", "agg_k": 5, "rrf_k": 1.5,
                          "prefix": False, "dense_pool": None,
                          "tiers": ["small", "large"]})
        self.assertEqual(variant.config, {"w_lex": 0.8})
        self.assertIn("cfg.w_lex", variant.describe())

    def test_defaults_when_no_flag(self):
        variants = ev.parse_variants(None)
        self.assertEqual([v.name for v in variants],
                         [s.split("=")[0] for s in ev.DEFAULT_VARIANTS])
        self.assertEqual([v.search for v in variants], [{}, {"mode": "lexical"}],
                         "the fallback always includes a lexical-only run: it "
                         "needs no embedding endpoint")

    def test_bad_specs_exit_with_their_own_message(self):
        with self.assertRaises(SystemExit):
            ev.parse_variant("name=broken")
        with self.assertRaises(SystemExit):
            ev.parse_variant("=mode=lexical")
        with self.assertRaises(SystemExit):
            ev.parse_variant("")
        with self.assertRaises(SystemExit):
            ev.parse_variants(["a=mode=lexical", "a=mode=semantic"])


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------

class _Hit:
    """Stand-in for mailbox.Hit: only message_id and score are read."""

    def __init__(self, mid, score):
        self.message_id = mid
        self.score = score


class TestRunRecords(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_normalise_result_handles_every_shape(self):
        ids, scores, meta = ev._normalise_result(
            (["a", "b"], {"abstained": True}))
        self.assertEqual((ids, scores, meta["abstained"]),
                         (["a", "b"], [None, None], True))
        ids, scores, _ = ev._normalise_result(
            ([_Hit("a", 0.9), {"message_id": "b", "score": 0.4}], {}))
        self.assertEqual((ids, scores), (["a", "b"], [0.9, 0.4]))
        ids, scores, meta = ev._normalise_result(["a"])
        self.assertEqual((ids, meta), (["a"], {}))

    def test_run_variant_asks_for_the_pool_depth(self):
        seen = []

        class StubRunner:
            cfg = {"db_path": "x.db"}

            def run(self, query, overrides, top=None):
                seen.append((query, overrides, top))
                return ["m1"], [1.0], {"abstained": False, "relevance": 0.9}

        run = ev.run_variant(StubRunner(), [row("q1", [], ["m1"])],
                             ev.Variant("v", {"mode": "lexical"}), top=10)
        self.assertEqual(seen[0][2], max(10, ev.POOL_K),
                         "the pool bucket needs 50 ids, not 10")
        self.assertEqual(seen[0][1], {"mode": "lexical"})
        self.assertEqual(run["name"], "v")
        self.assertEqual(run["per_query"][0]["ranked"], ["m1"])
        self.assertFalse(run["per_query"][0]["abstained"])
        self.assertEqual(run["per_query"][0]["relevance"], 0.9)

    def test_save_and_load_runs_round_trip(self):
        runs = [{"name": "a", "config": {"k": 1},
                 "per_query": [{"id": "q1", "ranked": ["m1"]}]},
                {"name": "b", "config": {}, "per_query": []}]
        path = os.path.join(self.tmp.name, "run.json")
        written = ev.save_runs(runs, path)
        self.assertEqual(len(written), 2, "one file per variant, or all but "
                                          "the last is lost")
        self.assertTrue(all(os.path.exists(p) for p in written))
        self.assertEqual({os.path.basename(p) for p in written},
                         {"run-a.json", "run-b.json"})
        self.assertEqual(ev.load_runs(written[0])[0]["name"], "a")
        single = os.path.join(self.tmp.name, "single.json")
        ev.save_runs([runs[0]], single)
        self.assertEqual(ev.load_runs(single), [runs[0]])

    def test_save_runs_substitutes_the_name_placeholder(self):
        runs = [{"name": "a", "config": {}, "per_query": []},
                {"name": "b", "config": {}, "per_query": []}]
        written = ev.save_runs(
            runs, os.path.join(self.tmp.name, "bench-{name}.json"))
        self.assertEqual([os.path.basename(p) for p in written],
                         ["bench-a.json", "bench-b.json"])

    def test_load_runs_accepts_a_runs_bundle_and_a_bare_list(self):
        path = os.path.join(self.tmp.name, "bundle.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"runs": [{"name": "a", "per_query": []},
                                {"name": "b", "per_query": []}]}, fh)
        self.assertEqual([r["name"] for r in ev.load_runs(path)], ["a", "b"])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([{"per_query": []}], fh)
        self.assertEqual(len(ev.load_runs(path)), 1)

    def test_load_runs_rejects_a_file_it_cannot_use(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaises(SystemExit):
            ev.load_runs(bad)
        empty = os.path.join(self.tmp.name, "empty.json")
        with open(empty, "w", encoding="utf-8") as fh:
            json.dump({"nope": 1}, fh)
        with self.assertRaises(SystemExit):
            ev.load_runs(empty)

    def test_label_run_joins_labels_by_id(self):
        labels = [row("q1", [], ["m1"], note="rule=x"), row("q2", [], ["m2"])]
        run = {"name": "v", "per_query": [
            {"id": "q1", "ranked": ["m1"], "abstained": True, "took_ms": 5},
            {"id": "gone", "ranked": ["m9"]}]}
        joined = ev.label_run(run, labels)
        self.assertEqual([r["id"] for r in joined], ["q1"], "a query that left "
                                                            "the eval set drops")
        self.assertEqual(joined[0]["note"], "rule=x")
        self.assertEqual(joined[0]["ranked"], ["m1"])
        self.assertTrue(joined[0]["abstained"])

    def test_best_rank_and_disagreements(self):
        a = row("q", ["x", "rel"], ["rel"])
        b = row("q", ["x", "y", "z"], ["rel"])
        self.assertEqual(ev.best_rank(a, 5), 2)
        self.assertEqual(ev.best_rank(b, 5), 0)
        hits = ev.disagreements([a], [b], 5)
        self.assertEqual([h["id"] for h in hits], ["q"])
        self.assertEqual(ev.disagreements([b], [a], 5), [],
                         "the direction matters: B hits nowhere")


# ---------------------------------------------------------------------------
# evalgen: the deterministic SQL rules, against a throwaway schema
# ---------------------------------------------------------------------------

def make_index_db(path):
    """The two tables evalgen reads, with only the columns its queries use."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE messages(message_id TEXT PRIMARY KEY, sender TEXT,
                              date TEXT, subject TEXT);
        CREATE TABLE chunks(message_id TEXT, kind TEXT, canonical INTEGER,
                            label TEXT);
    """)
    people = [
        ("<m01@x>", "marc@flexa.nl", "2026-05-14T09:00:00+00:00",
         "Offerte aanvraag webshop"),
        ("<m02@x>", "marc@flexa.nl", "2026-05-14T11:00:00+00:00",
         "Re: Offerte aanvraag webshop"),
        ("<m03@x>", "marc@flexa.nl", "2026-06-02T09:00:00+00:00",
         "Factuur 8812 betaalen"),
        ("<m04@x>", "noreply@trello.com", "2026-06-02T09:30:00+00:00",
         "Niels B has requested to join"),
        ("<m05@x>", "noreply@trello.com", "2026-06-02T10:30:00+00:00",
         "Lisa P mentioned you"),
        ("<m06@x>", "lauw@flexa.nl", "2026-07-01T08:00:00+00:00",
         "=?utf-8?Q?Vraag?= over het domein"),
    ]
    con.executemany("INSERT INTO messages VALUES (?,?,?,?)", people)
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?)", [
        ("<m01@x>", "body", 1, "body"),
        ("<m03@x>", "attachment", 1,
         "attachment: Voorbeeld Factuur V26.pdf"),
        ("<m03@x>", "attachment", 0,
         "attachment: Voorbeeld Factuur V26.pdf [small 1/2]"),
        ("<m04@x>", "attachment", 1, "attachment: screenshot.png"),
    ])
    con.commit()
    return con


class TestDiversity(unittest.TestCase):
    """Recall says the answer is in the list. It does not say the list is one
    conversation copied ten times, which is the failure a person notices first."""

    def test_a_thread_is_counted_once(self):
        ident = {"a": ("T1", "h1"), "b": ("T1", "h2"), "c": ("T2", "h3")}
        d = ev.diversity([{"ranked": ["a", "b", "c"]}], ident, k=10)
        self.assertEqual(d["threads_mean"], 2.0)
        self.assertEqual(d["threads_min"], 2)
        self.assertEqual(d["dup_bodies"], 0)

    def test_a_repeated_body_is_reported(self):
        ident = {"a": ("T1", "same"), "b": ("T2", "same")}
        d = ev.diversity([{"ranked": ["a", "b"]}], ident, k=10)
        self.assertEqual(d["dup_bodies"], 1)
        self.assertEqual(d["threads_mean"], 2.0,
                         "two different mails carrying the same text: the "
                         "thread collapse cannot catch that one")

    def test_an_unthreaded_message_is_its_own_thread(self):
        ident = {"a": ("a", "h1"), "b": ("b", "h2")}
        self.assertEqual(ev.diversity([{"ranked": ["a", "b"]}], ident)["threads_mean"], 2.0)

    def test_only_the_scored_depth_counts(self):
        ident = {"a": ("T1", "h1"), "b": ("T1", "h2"), "c": ("T2", "h3")}
        d = ev.diversity([{"ranked": ["a", "b", "c"]}], ident, k=2)
        self.assertEqual(d["threads_mean"], 1.0, "the second thread sits at 3")
        self.assertEqual(ev.diversity([{"ranked": ["a", "b", "c"]}], ident,
                                      k=3)["threads_mean"], 2.0)
        self.assertEqual(d["dup_bodies"], 0)

    def test_it_reads_threads_and_hashes_from_the_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "m.db")
            con = sqlite3.connect(db)
            con.execute("create table messages (message_id text, thread_id text)")
            con.execute("create table chunks (message_id text, canonical int,"
                        " text_hash text)")
            con.execute("insert into messages values ('a', 'T1'), ('b', null)")
            con.execute("insert into chunks values ('a', 1, 'abc')")
            con.commit()
            con.close()
            ident = ev.message_identity(db)
            self.assertEqual(ident["a"], ("T1", "abc"))
            self.assertEqual(ident["b"], ("b", ""))

    def test_an_unreadable_index_disables_the_check(self):
        """A run measured against an index that is gone cannot be scored for
        diversity; reporting 10.0 threads would be an invented number."""
        self.assertEqual(ev.message_identity(""), {})
        self.assertEqual(ev.message_identity(os.path.join(tempfile.gettempdir(),
                                                         "no-such-mailbox.db")), {})


class TestEvalgen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = os.path.join(cls.tmp.name, "index.db")
        cls.con = make_index_db(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_subject_rule_is_exact_and_records_itself(self):
        rows = ev.gen_subject_rows(self.con, 10, __import__("random").Random(0))
        subject = next(r for r in rows if "Offerte" in r["query"])
        self.assertEqual(subject["relevant"], ["<m01@x>", "<m02@x>"],
                         "Re: collapses onto the same subject, so both count")
        self.assertIn("rule=subject_exact", subject["note"])
        self.assertIn("WEAK", subject["note"])

    def test_subject_rule_skips_encoded_subjects(self):
        rows = ev.gen_subject_rows(self.con, 10, __import__("random").Random(0))
        self.assertFalse([r for r in rows if "=?utf-8" in r["query"]])

    def test_sender_rule_skips_transactional_senders(self):
        rows = ev.gen_sender_rows(self.con, 10, __import__("random").Random(0))
        self.assertEqual([r["query"] for r in rows], ["van marc"])
        self.assertEqual(rows[0]["relevant"], ["<m01@x>", "<m02@x>", "<m03@x>"])
        self.assertIn("rule=sender_exact marc@flexa.nl", rows[0]["note"])

    def test_date_rule_only_uses_days_it_can_measure(self):
        rows = ev.gen_date_rows(self.con, 10, __import__("random").Random(0))
        # 2026-06-02 has 3 mails, 2026-05-14 has 2: both are in range, and a
        # day with one mail would prove nothing so it is never offered.
        self.assertEqual({r["query"] for r in rows},
                         {"mail van 2026-05-14", "mail van 2026-06-02"})
        by_day = {r["query"]: r for r in rows}
        self.assertEqual(by_day["mail van 2026-05-14"]["relevant"],
                         ["<m01@x>", "<m02@x>"])
        self.assertIn("rule=date_exact", by_day["mail van 2026-06-02"]["note"])

    def test_attachment_rule_uses_canonical_filenames_only(self):
        rows = ev.gen_attachment_rows(self.con, 10, __import__("random").Random(0))
        self.assertEqual({r["query"] for r in rows},
                         {"voorbeeld factuur v26.pdf",
                          "screenshot.png"})
        fact = next(r for r in rows if "factuur" in r["query"])
        self.assertEqual(fact["relevant"], ["<m03@x>"])
        self.assertIn("rule=attachment_file", fact["note"])

    def test_ood_rows_are_empty_of_relevance_and_full_of_probes(self):
        rows = ev.gen_ood_rows(20, __import__("random").Random(0))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r["out_of_corpus"] and not r["relevant"]
                            for r in rows))
        self.assertEqual(len({r["id"] for r in rows}), 20,
                         "random-word probes must not collide")
        self.assertTrue(all("out_of_corpus" in r["note"] for r in rows))

    def test_generation_is_reproducible_for_a_seed(self):
        rng_a = __import__("random").Random(5)
        rng_b = __import__("random").Random(5)
        self.assertEqual(ev.gen_subject_rows(self.con, 4, rng_a),
                         ev.gen_subject_rows(self.con, 4, rng_b))

    def test_rule_of_survives_a_colon_in_the_subject(self):
        # Subjects are full of colons, and the note embeds them: the rule name
        # is the first word, not the text before the first colon.
        subject = {"note": "rule=subject_exact 'Factuur: betaald' (1 mail) - "
                           "WEAK label: x", "out_of_corpus": False}
        self.assertEqual(ev._rule_of(subject), "subject_exact")
        self.assertEqual(ev._rule_of({"note": "out_of_corpus: fixed probe",
                                      "out_of_corpus": True}), "out_of_corpus")

    def test_shares_add_up_to_n(self):
        for n in (1, 5, 10, 40, 41):
            shares = ev._shares(n)
            self.assertEqual(sum(shares.values()), n)
        self.assertEqual(ev._shares(40)["subject"], 12)

    def test_cmd_evalgen_writes_an_editable_draft(self):
        out = os.path.join(self.tmp.name, "draft.jsonl")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            ev.cmd_evalgen(namespace(db=self.db, config="no-such.json",
                                     n=10, out=out, seed=0))
        self.assertIn("how to edit", buf.getvalue())
        rows, errors = ev.load_rows(out)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(rows), 5)
        rules = {ev._rule_of(r) for r in rows}
        self.assertTrue(rules <= {"subject_exact", "sender_exact", "date_exact",
                                  "attachment_file", "out_of_corpus"}, rules)
        self.assertTrue(any(r["out_of_corpus"] for r in rows))
        self.assertIn("subject_exact", rules)
        first = open(out, encoding="utf-8").readline()
        self.assertTrue(first.startswith("#"), "the header must announce that "
                                               "these are weak labels")

    def test_cmd_evalgen_rejects_a_missing_database(self):
        with self.assertRaises(SystemExit):
            ev.cmd_evalgen(namespace(db=os.path.join(self.tmp.name, "gone.db"),
                                     config="no-such.json", n=5,
                                     out=os.path.join(self.tmp.name, "x.jsonl")))


# ---------------------------------------------------------------------------
# cmd_bench in offline mode: no mailbox, no endpoint, no index
# ---------------------------------------------------------------------------

class TestBenchOffline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.eval_file = os.path.join(self.tmp.name, "eval.jsonl")
        ev.save_rows([
            row("hit", ["rel-a"], ["rel-a"]),
            row("miss", ["junk"], ["rel-b"]),
            row("probe", ["junk"], [], out_of_corpus=True, abstained=True),
        ], self.eval_file)
        self.baseline = os.path.join(self.tmp.name, "good.json")
        self.offline = os.path.join(self.tmp.name, "weak.json")
        ev.save_runs([{"name": "good", "config": {}, "per_query": [
            {"id": "hit", "ranked": ["rel-a"], "abstained": False},
            {"id": "miss", "ranked": ["rel-b"], "abstained": False},
            {"id": "probe", "ranked": ["junk"], "abstained": False}]}],
            self.baseline)
        ev.save_runs([{"name": "weak", "config": {}, "per_query": [
            {"id": "hit", "ranked": ["junk"], "abstained": False},
            {"id": "miss", "ranked": ["junk"], "abstained": False},
            {"id": "probe", "ranked": ["junk"], "abstained": True}]}],
            self.offline)

    def run_bench(self, **over):
        args = namespace(eval_file=self.eval_file, baseline=self.baseline,
                         offline=self.offline, iters=200, seed=1, **over)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ev.cmd_bench(args)
        return out.getvalue()

    def test_it_prints_the_table_buckets_and_pairs(self):
        text = self.run_bench()
        self.assertIn("variant", text)
        for header in ("R@5", "R@10", "P@5", "MRR@10", "nDCG@10", "abstain"):
            self.assertIn(header, text)
        self.assertIn("error buckets", text)
        for bucket in ev.BUCKETS:
            self.assertIn(bucket, text)
        self.assertIn("paired bootstrap", text)
        self.assertIn("recall@5", text)
        self.assertIn("disagreements", text)

    def test_the_numbers_are_the_hand_computed_ones(self):
        text = self.run_bench()
        # good: hit 1.0, miss 1.0, probe excluded -> recall@5 = 1.0
        # weak: hit 0.0, miss 0.0                          -> recall@5 = 0.0
        line = next(l for l in text.splitlines() if l.startswith("good"))
        self.assertIn("1.000", line)
        line = next(l for l in text.splitlines() if l.startswith("weak"))
        self.assertIn("0.000", line)
        self.assertIn("good hits, weak misses", text)
        self.assertIn("hit", text)

    def test_offline_says_so_and_has_nothing_to_save(self):
        text = self.run_bench()
        self.assertIn("offline", text)
        text = self.run_bench(save=os.path.join(self.tmp.name, "out.json"))
        self.assertIn("nothing fresh to save", text)

    def test_missing_eval_file_is_fatal(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            ev.cmd_bench(namespace(eval_file=os.path.join(self.tmp.name, "x"),
                                   offline=self.offline))
        self.assertIn("cannot read", out.getvalue())

    def test_offline_bench_does_not_import_mailbox(self):
        # The whole point of importing mailbox inside the command: comparing
        # saved runs must work while mailbox.py does not even parse, and must
        # not pay for loading it at all.
        before = set(sys.modules)
        self.run_bench()
        self.assertNotIn("mailbox", set(sys.modules) - before)


class TestImportIsolation(unittest.TestCase):

    def test_importing_eval_does_not_import_mailbox(self):
        code = ("import sys, eval;"
                "print('mailbox' in sys.modules)")
        try:
            proc = subprocess.run([sys.executable, "-c", code],
                                  capture_output=True, text=True, timeout=60,
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
        except (OSError, subprocess.SubprocessError):
            self.skipTest("cannot spawn a subprocess here")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False",
                         "eval.py must not import mailbox at module level")


class TestCliWiring(unittest.TestCase):
    """mailbox.py's main() registers these commands on its own subparsers, so
    the flags have to land on a namespace where the parent's global --config is
    not overwritten by a subparser default - and every default a cmd_* reads has
    to exist, because these commands are also called from a hand-built
    namespace."""

    @staticmethod
    def parse(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config.json")
        ev.add_subcommands(parser.add_subparsers(dest="command", required=True))
        return parser.parse_args(argv)

    def test_global_config_survives_the_subparser(self):
        args = self.parse(["--config", "other.json", "bench"])
        self.assertEqual(args.config, "other.json", "a subparser writes its "
                                                    "defaults into the same "
                                                    "namespace")
        self.assertIs(args.func, ev.cmd_bench)

    def test_a_later_config_wins(self):
        args = self.parse(["--config", "global.json", "audit", "--config",
                           "local.json"])
        self.assertEqual(args.config, "local.json")

    def test_each_command_carries_the_defaults_its_function_reads(self):
        bench = self.parse(["bench"])
        self.assertEqual((bench.eval_file, bench.top, bench.iters, bench.seed),
                         (ev.DEFAULT_EVAL_FILE, 10, 10000, 0))
        self.assertIsNone(bench.baseline)
        self.assertIsNone(bench.save)
        self.assertIsNone(bench.variant, "None means DEFAULT_VARIANTS")
        gen = self.parse(["evalgen"])
        self.assertIs(gen.func, ev.cmd_evalgen)
        self.assertEqual((gen.n, gen.out, gen.seed),
                         (40, ev.DEFAULT_PROPOSED_FILE, 0))
        lab = self.parse(["label", "van marc", "2"])
        self.assertIs(lab.func, ev.cmd_label)
        self.assertEqual((lab.query, lab.rank, lab.top), ("van marc", 2, 10))
        aud = self.parse(["audit"])
        self.assertIs(aud.func, ev.cmd_audit)

    def test_variant_flags_reach_the_parser_untouched(self):
        args = self.parse(["bench", "--variant", "lex=mode=lexical",
                           "--variant", "wide=cfg.w_lex=0.8,tiers=small|large"])
        self.assertEqual(args.variant, ["lex=mode=lexical",
                                        "wide=cfg.w_lex=0.8,tiers=small|large"])
        names = [v.name for v in ev.parse_variants(args.variant)]
        self.assertEqual(names, ["lex", "wide"])

    def test_standalone_parser_exposes_the_same_four_commands(self):
        parser = ev.build_parser()
        args = parser.parse_args(["bench", "--offline", "run.json"])
        self.assertEqual(args.offline, "run.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            parser.parse_args([])          # a command is required, not optional


if __name__ == "__main__":
    unittest.main(verbosity=2)
