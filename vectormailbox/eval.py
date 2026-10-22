#!/usr/bin/env python3
"""
eval.py - retrieval-quality harness for mailbox.py (it never writes the index).

Why this exists: a hybrid retriever otherwise gets tuned by feel, and "the new
reranker looks better" is not a measurement. This module turns the search API
into numbers - recall / precision / MRR / nDCG per query, a paired bootstrap
interval for A versus B, and a bucket that says *why* a query failed (nothing
to find, not in the pool, buried below the fold, thrown away by the relevance
floor) - so a change is accepted or rejected on the same fixed set of queries.

Commands (registered by mailbox.py's argparse through `add_subcommands()`, or
run standalone with `python eval.py ...`):

  python eval.py bench   [--eval-file eval/eval.jsonl] [--variant NAME=k=v,...]
                         [--baseline FILE] [--save FILE] [--offline FILE]
                         [--top 10] [--iters 10000] [--seed 0] [--db PATH]
  python eval.py evalgen [--n 40] [--out eval/eval.proposed.jsonl] [--seed 0]
                         [--db PATH]
  python eval.py label   "query" 3 [--eval-file eval/eval.jsonl] [--db PATH]
  python eval.py audit   [--db mailbox.db]

`bench` hits the (slow, networked) embedding endpoint once per query per
variant, so it can save the per-query ranked lists to a file; `--baseline` and
`--offline` then compare saved runs without touching the endpoint again. A save
holds only what a fresh run produces (the ranked lists) - the labels stay in
the eval file and are joined back by row `id`, so a re-run can never silently
score against a stale copy of the ground truth.

Eval set: JSONL, one query per line - see eval/eval.example.jsonl. Blank lines
and `#` comments are skipped so the file can be annotated in place, unknown
keys are preserved, and a broken line is reported with its line number instead
of losing the other 199 labels.

`evalgen` proposes rows from the live index by deterministic SQL rules (subject
match, sender match, day match, attachment filename) plus fixed out-of-corpus
probes. No LLM, no network, and every row says in `note` which rule made it,
because a rule-derived label is a weak label.

Design note: the metric functions are pure - no I/O, no numpy, no mailbox
import - so test_eval.py can pin the arithmetic with hand-computed examples.
`import mailbox` happens inside the command functions only, which keeps
`import eval` working while mailbox.py itself is being edited.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Metric primitives (pure: lists of ids in, floats out)
# ---------------------------------------------------------------------------

#: k values the harness reports, in table order. These names are the stable
#: contract: summarise(), paired_bootstrap_ci() and the printed tables all key
#: on them, so renaming one is a breaking change for saved baselines.
METRICS = (
    "recall@1", "recall@3", "recall@5", "recall@10",
    "precision@5", "precision@10",
    "mrr@5", "mrr@10",
    "ndcg@5", "ndcg@10",
    "hit@5", "forbidden@10",
)

POOL_K = 50      # bucket_errors: "the retriever saw it at all" depth
TOP_K = 3        # bucket_errors: "a human scrolling the list reaches it" depth


def _id_list(ids: Any) -> list[str]:
    """Coerce a ranked list to de-duplicated strings.

    Ranks are positional, so a repeated id would silently steal a rank from the
    message behind it. Message-level search never emits duplicates, but a
    hand-edited saved run can.
    """
    if ids is None:
        return []
    if isinstance(ids, str):
        ids = [ids]
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        s = str(raw)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _id_set(rel: Any) -> set[str]:
    """Coerce a relevance set (set / list / single id / None) to a set."""
    if rel is None:
        return set()
    if isinstance(rel, str):
        return {rel} if rel else set()
    return {str(x) for x in rel if str(x)}


def recall_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """Share of the relevant set that appears in the top k.

    0.0 when nothing is labelled: an unlabelled query cannot earn recall, and
    scoring it 1.0 would reward a missing label.
    """
    rel = _id_set(relevant_ids)
    if not rel or k <= 0:
        return 0.0
    return len(rel.intersection(_id_list(ranked_ids)[:k])) / float(len(rel))


def precision_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """Share of the top k that is relevant.

    The denominator is k, not the number of results returned: returning less
    than you were allowed to is also a failure to retrieve.
    """
    rel = _id_set(relevant_ids)
    if not rel or k <= 0:
        return 0.0
    return sum(1 for mid in _id_list(ranked_ids)[:k] if mid in rel) / float(k)


def hit_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """1.0 when at least one relevant id is in the top k, else 0.0."""
    rel = _id_set(relevant_ids)
    if not rel or k <= 0:
        return 0.0
    return 1.0 if rel.intersection(_id_list(ranked_ids)[:k]) else 0.0


def reciprocal_rank(ranked_ids, relevant_ids) -> float:
    """1 / rank of the first relevant hit over the whole list, 0.0 if absent."""
    rel = _id_set(relevant_ids)
    if not rel:
        return 0.0
    for i, mid in enumerate(_id_list(ranked_ids), 1):
        if mid in rel:
            return 1.0 / i
    return 0.0


def mrr_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """reciprocal_rank() cut off at k: a first hit outside the top k is worth
    nothing here - that is the point of capping it."""
    rel = _id_set(relevant_ids)
    if not rel or k <= 0:
        return 0.0
    for i, mid in enumerate(_id_list(ranked_ids)[:k], 1):
        if mid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """Binary-relevance nDCG: gain 1 per relevant hit, discount 1/log2(i+1) at
    1-based position i, divided by the ideal ordering of the same labels.

    Ranking the only relevant mail at position 3 scores (1/log2(4)) / 1 = 0.5:
    half as much as putting it first, which is the discount a user scrolling
    the list actually pays.
    """
    rel = _id_set(relevant_ids)
    if not rel or k <= 0:
        return 0.0
    ranked = _id_list(ranked_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 1)
              for i, mid in enumerate(ranked, 1) if mid in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0.0 else 0.0


def forbidden_at_k(ranked_ids, forbidden_ids, k: int) -> float:
    """1.0 when a forbidden id shows up in the top k.

    Deliberately not folded into the other metrics: showing the mail the user
    said to ignore is a different kind of wrong than missing the right one, and
    averaging the two would hide both.
    """
    forb = _id_set(forbidden_ids)
    if not forb or k <= 0:
        return 0.0
    return 1.0 if forb.intersection(_id_list(ranked_ids)[:k]) else 0.0


def _is_abstained(row: dict) -> bool:
    """Abstention as the run decided it, or as the floor says when a row only
    carries the raw relevance number."""
    val = row.get("abstained")
    if val is not None:
        return bool(val)
    rel, floor = row.get("relevance"), row.get("min_relevance")
    if rel is not None and floor is not None:
        try:
            return float(rel) < float(floor)
        except (TypeError, ValueError):
            return False
    return False


def _is_retrievable(row: dict) -> bool:
    """Whether any message in this corpus could answer the query at all.

    The caller may set `retrievable` explicitly; otherwise `out_of_corpus`
    implies it. Only a *proven* impossibility counts, so an absent flag means
    'yes' - assuming a query unanswerable would silently delete hard cases.
    """
    val = row.get("retrievable")
    if val is not None:
        return bool(val)
    return not bool(row.get("out_of_corpus"))


def row_metrics(row: dict) -> dict:
    """Every METRIC for one per-query row: {id, query, ranked, relevant,
    forbidden, abstained, ...}.

    Rows are plain dicts so a saved run, a fresh run and a hand-written test
    fixture are all acceptable input, and a row may carry the extra fields
    (`relevance`, `took_ms`) the commands add.
    """
    ranked = _id_list(row.get("ranked"))
    rel = _id_set(row.get("relevant"))
    forb = _id_set(row.get("forbidden"))
    out: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = recall_at_k(ranked, rel, k)
    for k in (5, 10):
        out[f"precision@{k}"] = precision_at_k(ranked, rel, k)
        out[f"mrr@{k}"] = mrr_at_k(ranked, rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
    out["hit@5"] = hit_at_k(ranked, rel, 5)
    out["forbidden@10"] = forbidden_at_k(ranked, forb, 10)
    out["abstained"] = 1.0 if _is_abstained(row) else 0.0
    return out


def summarise(rows: Iterable[dict]) -> dict:
    """Mean of every metric plus the abstention rates.

    Out-of-corpus rows are kept out of the retrieval means on purpose: recall
    against an empty `relevant` list is 0 by definition, so counting them would
    punish a run for not answering an unanswerable query. Their abstention is
    reported separately - there, abstaining *is* the correct answer.
    """
    rows = list(rows)
    scored = [r for r in rows if _is_retrievable(r)]
    ood = [r for r in rows if not _is_retrievable(r)]
    out: dict[str, float] = {
        "n": float(len(rows)),
        "n_scored": float(len(scored)),
        "n_out_of_corpus": float(len(ood)),
    }
    for name in METRICS:
        out[name] = (statistics.fmean(row_metrics(r)[name] for r in scored)
                     if scored else 0.0)
    out["abstain_rate"] = (statistics.fmean(1.0 if _is_abstained(r) else 0.0
                                            for r in scored)
                           if scored else 0.0)
    out["abstain_rate_out_of_corpus"] = (
        statistics.fmean(1.0 if _is_abstained(r) else 0.0 for r in ood)
        if ood else 0.0)
    return out


@dataclass
class BootstrapResult:
    """Paired bootstrap outcome.

    Iterating yields (delta, lo, hi), so `delta, lo, hi = paired_bootstrap_ci(...)`
    stays readable while `.insufficient` remains available for the case of too
    few queries to say anything at all.
    """
    delta: float
    lo: float
    hi: float
    n: int
    iters: int
    insufficient: bool = False
    note: str = ""

    def __iter__(self):
        return iter((self.delta, self.lo, self.hi))

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval sits on one side of 0: on this eval set
        the difference is not explainable by resampling luck."""
        if self.insufficient:
            return False
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        if self.insufficient:
            return f"{self.delta:+.4f}  (insufficient data: {self.note})"
        return f"{self.delta:+.4f}  [{self.lo:+.4f}, {self.hi:+.4f}]"


def _pair_rows(rows_a: list[dict], rows_b: list[dict]) -> list[tuple[dict, dict]]:
    """Pair the two runs per query: by `id` when both sides carry matching ids
    (a saved run comes back in a different order than a fresh one), else by
    position. Unpaired queries drop out - there is no per-query difference to
    resample for them.
    """
    rows_a, rows_b = list(rows_a), list(rows_b)
    by_id_b = {str(r["id"]): r for r in rows_b if r.get("id") is not None}
    if by_id_b:
        pairs = [(a, by_id_b[str(a["id"])]) for a in rows_a
                 if a.get("id") is not None and str(a.get("id")) in by_id_b]
        if pairs:
            return pairs
    return list(zip(rows_a, rows_b))


def paired_bootstrap_ci(rows_a, rows_b, metric: str = "recall@10",
                        iters: int = 10000, seed: int = 0) -> BootstrapResult:
    """Paired bootstrap over the per-query differences of one metric.

    Paired, not two independent intervals: the queries are the same in both
    runs, so the variance that matters is the difference per query, and an
    unpaired test would spend its whole budget on how hard the eval set is.
    Percentile interval on resampled means; deterministic for a given seed and
    resample count, which is what makes a printed table reproducible next week.
    """
    if metric != "abstained" and metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; use one of "
                       f"{', '.join(METRICS)}, or 'abstained'")

    def one(row: dict) -> float:
        return (1.0 if _is_abstained(row) else 0.0) if metric == "abstained" \
            else row_metrics(row)[metric]

    pairs = _pair_rows(list(rows_a), list(rows_b))
    diffs = [one(a) - one(b) for a, b in pairs]
    n = len(diffs)
    if n < 2:
        # One query cannot carry a confidence interval: say so instead of
        # printing [x, x] as if it meant something.
        delta = float(diffs[0]) if diffs else 0.0
        return BootstrapResult(delta, delta, delta, n, int(iters), True,
                               f"need >= 2 paired queries, got {n}")
    delta = float(statistics.fmean(diffs))
    rng = random.Random(int(seed))
    means = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n))
                   for _ in range(max(1, int(iters))))
    top = len(means) - 1
    lo = float(means[int(0.025 * top)])
    hi = float(means[int(0.975 * top)])
    return BootstrapResult(delta, lo, hi, n, int(iters))


#: bucket_errors() output order, worst cause first. Each query lands in exactly
#: one bucket: the first cause that explains it.
BUCKETS = ("not_in_corpus", "missing_from_pool", "buried", "below_threshold", "ok")


def bucket_errors(rows: Iterable[dict], pool: int = POOL_K,
                  depth: int = TOP_K) -> dict[str, list[str]]:
    """Split the eval set into the four failure buckets plus `ok`.

    Why buckets and not just recall@10: the four causes need four different
    fixes - index more, widen the candidate pool, change the fusion, change the
    relevance threshold - and one number cannot tell them apart.
    """
    out: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for row in rows:
        rid = str(row.get("id") if row.get("id") is not None
                  else row.get("query") or "?")
        if not _is_retrievable(row):
            out["not_in_corpus"].append(rid)
            continue
        ranked = _id_list(row.get("ranked"))
        rel = _id_set(row.get("relevant"))
        best = next((i for i, mid in enumerate(ranked[:max(pool, depth)], 1)
                     if mid in rel), 0)
        if not rel or not best:
            # An empty `relevant` list is the same failure as nothing being
            # retrievable: no ranking can satisfy it. Label it or drop it.
            out["missing_from_pool" if rel else "not_in_corpus"].append(rid)
        elif best > depth:
            out["buried"].append(rid)
        elif _is_abstained(row):
            out["below_threshold"].append(rid)
        else:
            out["ok"].append(rid)
    return out


def bucket_counts(buckets: dict[str, list[str]]) -> dict[str, int]:
    """Sizes per bucket, in BUCKETS order (a missing bucket reads 0)."""
    return {b: len(buckets.get(b) or []) for b in BUCKETS}


# ---------------------------------------------------------------------------
# Eval-set file I/O
# ---------------------------------------------------------------------------

DEFAULT_EVAL_FILE = os.path.join("eval", "eval.jsonl")
DEFAULT_PROPOSED_FILE = os.path.join("eval", "eval.proposed.jsonl")

#: canonical key order on disk; anything else a human added is kept as-is.
ROW_KEYS = ("id", "query", "relevant", "forbidden", "out_of_corpus", "note")


def _as_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "ja")
    return bool(val)


def _as_ids(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (str, int)):
        val = [val]
    return [str(x) for x in val if str(x)]


def _normalise_row(obj: dict, lineno: int) -> tuple[Optional[dict], str]:
    """One JSON object -> a row dict, or (None, why it was rejected)."""
    query = str(obj.get("query") or "").strip()
    if not query:
        return None, "missing or empty 'query'"
    row: dict[str, Any] = {
        "id": (str(obj.get("id") or "").strip() or f"l{lineno}"),
        "query": query,
        "relevant": _as_ids(obj.get("relevant")),
        "forbidden": _as_ids(obj.get("forbidden")),
        "out_of_corpus": _as_bool(obj.get("out_of_corpus")),
        "note": str(obj.get("note") or ""),
    }
    for key, val in obj.items():               # tolerate unknown keys
        if key not in ROW_KEYS:
            row[key] = val
    if row["out_of_corpus"] and not row["note"]:
        row["note"] = "out_of_corpus: nothing should match; abstain is correct"
    return row, ""


def load_rows(path: str) -> tuple[list[dict], list[dict]]:
    """Read an eval JSONL file -> (rows, errors).

    Errors are returned, not raised: one broken line in a hand-edited file must
    not throw away the other 199 labels. `errors` holds `{"line": int,
    "error": str}` with 1-based line numbers (line 0 means the file itself).
    Blank lines and `#` comments are skipped. A duplicate `id` is reported and
    the second row dropped, because every comparison joins runs on `id`.
    """
    rows: list[dict] = []
    errors: list[dict] = []
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError as exc:
        return [], [{"line": 0, "error": f"cannot read {path}: {exc}"}]
    seen: set[str] = set()
    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                errors.append({"line": lineno, "error": f"not valid JSON: {exc}"})
                continue
            if not isinstance(obj, dict):
                errors.append({"line": lineno,
                               "error": f"expected a JSON object, got "
                                        f"{type(obj).__name__}"})
                continue
            row, why = _normalise_row(obj, lineno)
            if row is None:
                errors.append({"line": lineno, "error": why})
                continue
            if row["id"] in seen:
                errors.append({"line": lineno,
                               "error": f"duplicate id {row['id']!r}: ids must "
                                        f"stay unique and stable - dropped"})
                continue
            seen.add(row["id"])
            rows.append(row)
    return rows, errors


def save_rows(rows: Iterable[dict], path: str) -> int:
    """Write rows as JSONL (canonical key order, unknown keys last) and return
    the count. Pure serialisation - no comment header - so a load/save round
    trip is byte-stable for the data rows.
    """
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            ordered = {k: row[k] for k in ROW_KEYS if k in row}
            ordered.update({k: v for k, v in row.items() if k not in ROW_KEYS})
            fh.write(json.dumps(ordered, ensure_ascii=False) + "\n")
            n += 1
    return n


def row_id_for(query: str) -> str:
    """Stable row id from the query text: a readable slug plus a short hash.

    The hash keeps two queries that slug alike apart, and the slug keeps the
    file greppable. Ids are what saved runs are joined on, so they must not
    shift when a row is re-labelled - hence derived from the query, not from a
    row counter.
    """
    words = re.findall(r"\w+", (query or "").lower(), re.UNICODE)
    slug = "-".join(words[:4])[:32].strip("-") or "query"
    digest = hashlib.sha1((query or "").encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}"


# ---------------------------------------------------------------------------
# Running queries: the one place that knows mailbox.py's shape
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """One thing to bench: a name, search-API overrides, config overrides."""
    name: str
    search: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.search.items())]
        parts += [f"cfg.{k}={v}" for k, v in sorted(self.config.items())]
        return ", ".join(parts) if parts else "config defaults"


#: used when --variant is absent: the config default, plus a lexical-only run
#: that needs no embedding endpoint (so bench still works when the host is
#: down, and so a dense win is always shown against a bm25-only baseline).
DEFAULT_VARIANTS = ("baseline", "lexical=mode=lexical")


def _coerce(raw: str) -> Any:
    """Flag text -> bool / None / int / float / list / str.

    Lists use `|` because `,` already separates the key=value pairs of one
    variant, and `--variant x=tiers=small|large` has to survive both splits.
    """
    val = raw.strip()
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if "|" in val:
        return [p.strip() for p in val.split("|") if p.strip()]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def parse_variant(spec: str) -> Variant:
    """`NAME=key=val,key=val` -> Variant.

    A bare NAME means "the defaults, called NAME" - enough to name a baseline.
    Keys prefixed `cfg.` (or `config.`) go to the config overrides instead of
    the search overrides, because some knobs only exist as config.
    """
    text = (spec or "").strip()
    if not text:
        raise SystemExit("empty --variant")
    if "=" not in text:
        return Variant(text)
    name, _, body = text.partition("=")
    variant = Variant(name.strip())
    if not variant.name:
        raise SystemExit(f"bad --variant {spec!r}: missing name before '='")
    for chunk in body.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"bad --variant {spec!r}: {chunk!r} is not key=value")
        key, _, raw = chunk.partition("=")
        key = key.strip()
        if key.startswith("cfg.") or key.startswith("config."):
            variant.config[key.split(".", 1)[1]] = _coerce(raw)
        else:
            variant.search[key] = _coerce(raw)
    return variant


def parse_variants(specs: Optional[list]) -> list[Variant]:
    """Parse --variant (repeated), or fall back to DEFAULT_VARIANTS."""
    specs = list(specs) if specs else list(DEFAULT_VARIANTS)
    out: list[Variant] = []
    for spec in specs:
        variant = parse_variant(spec)
        if any(v.name == variant.name for v in out):
            raise SystemExit(f"duplicate --variant name {variant.name!r}")
        out.append(variant)
    return out


def _normalise_result(res: Any) -> tuple[list[str], list, dict]:
    """Whatever mailbox hands back -> (ids, scores, meta).

    Tolerates both shapes the search API has had (message-id strings, or Hit
    objects) plus a bare list with no meta: this harness is written next to a
    mailbox.py that is being edited, and a shape change should cost a print
    note, not the whole benchmark.
    """
    if isinstance(res, tuple):
        payload = res[0] if res else []
        meta = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
    else:
        payload, meta = res, {}
    ids: list[str] = []
    scores: list[Any] = []
    for item in payload or []:
        if isinstance(item, str):
            ids.append(item)
            scores.append(None)
        elif isinstance(item, dict):
            ids.append(str(item.get("message_id")))
            scores.append(item.get("score"))
        else:
            ids.append(str(getattr(item, "message_id")))
            scores.append(getattr(item, "score", None))
    return ids, scores, meta


class Runner:
    """`ids, scores, meta = run(query, overrides)` over whatever mailbox.py
    exposes today.

    `search_query(cfg, query, **overrides)` is the stable door and is what this
    uses; it builds the params itself, so new search options need no change
    here. The Searcher fallback exists only so the harness stays runnable in
    the window before that wrapper lands - it reads the SearchParams field
    names off the dataclass rather than hard-coding them, and reports any
    override this mailbox.py does not know about instead of failing silently.
    """

    def __init__(self, cfg: dict):
        from . import mailbox as mb            # inside: mailbox.py may be mid-edit
        self.mb = mb
        self.cfg = dict(cfg)
        self.store = None
        self.client = None

    def close(self) -> None:
        if self.store is not None:
            try:
                self.store.conn.close()
            except Exception:
                pass                    # teardown: nothing useful to do
            self.store = None

    def _open_store(self):
        if self.store is None:
            self.store = self.mb.Store(self.cfg.get("db_path", "mailbox.db"),
                                       self.cfg)
        return self.store

    def _open_client(self):
        if self.client is None and self.cfg.get("embed_endpoint"):
            self.client = self.mb.EmbeddingClient(self.cfg)
        return self.client

    @staticmethod
    def _accepts(fn, name: str) -> bool:
        """Whether a callable takes this keyword. The wrapper is allowed to
        grow (or shrink) its signature without breaking a benchmark: passing
        `store` keeps a whole bench at one load of the vector arrays instead of
        one load per query, which on a few hundred MB index is the difference
        between seconds and minutes."""
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(p.kind == p.VAR_KEYWORD
                                     for p in params.values())

    def run(self, query: str, overrides: Optional[dict] = None,
            top: Optional[int] = None) -> tuple[list[str], list, dict]:
        ov = dict(overrides or {})
        if top:
            ov.setdefault("top", int(top))
        wrapper = getattr(self.mb, "search_query", None)
        if callable(wrapper):
            kwargs = dict(ov)
            if self._accepts(wrapper, "store"):
                kwargs["store"] = self._open_store()
            if self._accepts(wrapper, "client"):
                kwargs["client"] = self._open_client()
            return _normalise_result(wrapper(self.cfg, query, **kwargs))

        params_cls = getattr(self.mb, "SearchParams", None)
        searcher_cls = getattr(self.mb, "Searcher", None)
        if params_cls is None or searcher_cls is None:
            raise SystemExit("mailbox.py exposes neither search_query() nor "
                             "Searcher/SearchParams - eval.py cannot run")
        known = {f.name for f in dataclasses.fields(params_cls)}
        dropped = sorted(set(ov) - known)
        if dropped:
            print(f"  note: this mailbox.py has no SearchParams field for "
                  f"{', '.join(dropped)} - ignored for this variant")
        params = params_cls(**{k: v for k, v in ov.items() if k in known})
        return _normalise_result(searcher_cls(self.cfg, self._open_store(),
                                              self._open_client()).run(query,
                                                                       params))


def _ro_conn(db_path: str) -> sqlite3.Connection:
    """Read-only connection: `mode=ro` makes an accidental write impossible
    rather than merely unplanned, which is the whole promise of `audit`."""
    if not os.path.exists(db_path):
        raise SystemExit(f"no such database: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _arg(args, name: str, default: Any = None) -> Any:
    """Read an optional argparse attribute, treating 'not defined' and None
    alike: these commands are called from mailbox.py's parser, which may not
    define every flag, and from a hand-built namespace in tests."""
    val = getattr(args, name, None)
    return default if val is None else val


def _config_of(args) -> dict:
    """Config as plain JSON, without importing mailbox.

    evalgen and audit only ever need db_path, and reading the file directly
    keeps them working (and importable) while mailbox.py is mid-edit; bench and
    label use mailbox.load_config because they need the env-var overrides too.
    """
    path = _arg(args, "config", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _db_path(args, cfg: Optional[dict] = None) -> str:
    override = _arg(args, "db")
    if override:
        return str(override)
    if cfg:
        return str(cfg.get("db_path", "mailbox.db"))
    return "mailbox.db"


def run_variant(runner: Runner, rows: list[dict], variant: Variant,
                top: int) -> dict:
    """One variant over the whole eval set -> a saveable run record.

    `top` is asked for as max(top, POOL_K): the pool bucket needs to know
    whether the answer sat at position 37, which a 10-wide list cannot show.
    """
    per_query: list[dict] = []
    total_ms = 0.0
    for row in rows:
        start = time.perf_counter()
        ids, scores, meta = runner.run(row["query"], variant.search,
                                       top=max(int(top), POOL_K))
        took = (time.perf_counter() - start) * 1000.0
        total_ms += took
        rec = {
            "id": row["id"],
            "query": row["query"],
            "ranked": ids,
            "scores": scores,
            "abstained": bool(meta.get("abstained", False)),
            "took_ms": round(took, 1),
        }
        if meta.get("relevance") is not None:
            rec["relevance"] = meta.get("relevance")
        per_query.append(rec)
    return {
        "name": variant.name,
        "config": {
            "variant": variant.describe(),
            "db_path": runner.cfg.get("db_path"),
            "top_asked": int(top),
            "fetched_to": max(int(top), POOL_K),
            "total_ms": round(total_ms, 1),
        },
        "per_query": per_query,
    }


def load_runs(path: str) -> list[dict]:
    """Read a saved-runs file: one run object, `{"runs": [...]}`, or a list."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from None
    except ValueError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from None
    if isinstance(data, dict):
        runs = data.get("runs") or [data]
    elif isinstance(data, list):
        runs = data
    else:
        raise SystemExit(f"{path}: expected a run object or a list of them")
    out = []
    for i, run in enumerate(runs):
        if not isinstance(run, dict) or "per_query" not in run:
            raise SystemExit(f"{path}: entry {i} has no 'per_query' list")
        run.setdefault("name", f"{os.path.basename(path)}#{i}")
        run.setdefault("config", {})
        out.append(run)
    return out


def save_runs(runs: list[dict], path: str) -> list[str]:
    """Write run records and return the paths written.

    One run -> `path` as given. Several -> one file per run (`{name}` in the
    path is substituted, else the name is added to the stem), because a run is
    worth keeping per variant: overwriting four variants into one file keeps
    only the last one.
    """
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if len(runs) <= 1 and "{name}" not in path:
        target = str(path)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(runs[0] if runs else {}, fh, ensure_ascii=False, indent=1)
        return [target]
    written = []
    for run in runs:
        name = str(run.get("name", "run"))
        if "{name}" in path:
            target = path.replace("{name}", name)
        else:
            stem, ext = os.path.splitext(path)
            target = f"{stem}-{name}{ext or '.json'}"
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(run, fh, ensure_ascii=False, indent=1)
        written.append(target)
    return written


def label_run(run: dict, rows: list[dict]) -> list[dict]:
    """Join a saved run's ranked lists with the eval file's labels by `id`.

    The saved file deliberately holds no labels: the eval set is the thing a
    human edits, and a copy of the labels inside every run file is how a
    re-benchmark silently scores against last month's ground truth.
    """
    by_id = {str(r["id"]): r for r in rows}
    out: list[dict] = []
    for rec in run.get("per_query") or []:
        label = by_id.get(str(rec.get("id")))
        if label is None:
            continue                     # query no longer in the eval set
        merged = dict(label)
        merged.update({k: v for k, v in rec.items() if k != "id"})
        out.append(merged)
    return out


def best_rank(row: dict, k: int) -> int:
    """1-based rank of the first relevant id within the top k, 0 if absent."""
    rel = _id_set(row.get("relevant"))
    for i, mid in enumerate(_id_list(row.get("ranked"))[:k], 1):
        if mid in rel:
            return i
    return 0


def disagreements(rows_a: list[dict], rows_b: list[dict], k: int) -> list[dict]:
    """Queries where A hits within the top k and B does not.

    This direction only, on purpose: the symmetric count is dominated by
    queries both runs miss, and it is the queries one variant gets right that
    the other throws away which decide whether a change is worth its cost.
    """
    out = []
    for a, b in _pair_rows(rows_a, rows_b):
        ra, rb = best_rank(a, k), best_rank(b, k)
        if ra and not rb:
            out.append({"id": a.get("id"), "query": a.get("query"),
                        "a_rank": ra, "b_rank": rb})
    return out


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

#: metrics the A-vs-B bootstrap table prints (the rest are in the summary).
PAIR_METRICS = ("recall@5", "recall@10", "mrr@10", "ndcg@10", "precision@5")

_SUM_COLS = (("R@5", "recall@5"), ("R@10", "recall@10"), ("P@5", "precision@5"),
             ("MRR@10", "mrr@10"), ("nDCG@10", "ndcg@10"),
             ("forb@10", "forbidden@10"), ("abstain", "abstain_rate"))

#: bucket_errors() column width in the bucket table: the longest name
#: ("missing_from_pool") plus two spaces, or the headers run into each other.
_BUCKET_COL = max(len(b) for b in BUCKETS) + 2


def _print_label_errors(errors: list[dict]) -> None:
    if not errors:
        return
    print(f"eval file problems ({len(errors)}):")
    for err in errors:
        where = f"line {err['line']}" if err["line"] else "file"
        print(f"  {where}: {err['error']}")
    print()


def _mean_ms(per_query: list[dict]) -> float:
    vals = [float(r.get("took_ms") or 0.0) for r in per_query]
    return statistics.fmean(vals) if vals else 0.0


def _print_summary(runs: list[tuple[dict, list[dict]]], top: int) -> dict:
    print(f"{'variant':<18}{'n':>4}" + "".join(f"{h:>10}" for h, _ in _SUM_COLS))
    summaries = {}
    for run, per_query in runs:
        s = summarise(per_query)
        summaries[str(run["name"])] = s
        line = f"{str(run['name'])[:18]:<18}{int(s['n_scored']):>4}"
        line += "".join(f"{s[key]:>10.3f}" for _, key in _SUM_COLS)
        print(line)
    print()
    for run, per_query in runs:
        s = summaries[str(run["name"])]
        tail = (f", out-of-corpus {int(s['n_out_of_corpus'])} "
                f"(abstained {s['abstain_rate_out_of_corpus']:.0%})"
                if s["n_out_of_corpus"] else "")
        print(f"  {str(run['name'])[:18]:<18} mean {_mean_ms(per_query):.0f} ms per "
              f"query{tail}")
    print(f"  hit cutoff for the disagreement list: top {top}; out-of-corpus "
          f"queries are excluded from the metric means")
    return summaries


def message_identity(db_path: str) -> dict:
    """`message_id -> (thread_id, canonical body hash)` for the diversity check.

    Opened read-only: a bench run scores an index and must not migrate or
    re-sync one it is only measuring. Anything unreadable gives `{}`, and the
    report then says the check could not run instead of claiming diversity.
    """
    if not db_path or not os.path.exists(db_path):
        return {}
    uri = "file:" + os.path.abspath(db_path).replace(os.sep, "/") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        try:
            rows = con.execute(
                "select m.message_id, m.thread_id,"
                " (select c.text_hash from chunks c"
                "   where c.message_id = m.message_id and c.canonical = 1"
                "     and c.text_hash is not null limit 1)"
                " from messages m").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return {}
    # A message with no thread id is its own thread: folding it together with
    # every other unthreaded message would invent a conversation.
    return {mid: (thr or mid, h or "") for mid, thr, h in rows}


def diversity(per_query: list[dict], ident: dict, k: int = 10) -> dict:
    """How many separate answers the top `k` really holds, per query.

    Recall says the right mail is in the list. It does not say the list is the
    same paragraph four times, which is what a user notices first.
    """
    threads, dupes, checked = [], 0, 0
    for rec in per_query:
        ids = _id_list(rec.get("ranked"))[:k]
        if not ids:
            continue
        checked += 1
        threads.append(len({ident.get(m, (m, ""))[0] for m in ids}))
        hashes = [ident[m][1] for m in ids if m in ident and ident[m][1]]
        if len(hashes) != len(set(hashes)):
            dupes += 1
    return {"n": checked,
            "threads_mean": (sum(threads) / len(threads)) if threads else 0.0,
            "threads_min": min(threads) if threads else 0,
            "dup_bodies": dupes}


def _print_diversity(runs: list[tuple[dict, list[dict]]], top: int) -> None:
    """Distinct threads and repeated bodies, resolved from each run's own index."""
    print(f"\ndiversity in the top {top} (what the list holds, not what it found):")
    print(f"{'variant':<18}{'threads mean':>14}{'min':>6}{'repeated body':>16}")
    cache: dict = {}
    for run, per_query in runs:
        name = str(run["name"])[:18]
        db = str((run.get("config") or {}).get("db_path") or "")
        if db not in cache:
            cache[db] = message_identity(db)
        ident = cache[db]
        if not ident:
            print(f"{name:<18}{'-':>14}{'-':>6}   no readable index at "
                  f"{db or '(none recorded)'}")
            continue
        d = diversity(per_query, ident, top)
        print(f"{name:<18}{d['threads_mean']:>14.1f}{d['threads_min']:>6}"
              f"{str(d['dup_bodies']) + ' of ' + str(d['n']):>16}")
    print("  `repeated body` counts queries whose top list holds two results "
          "with the same normalised body; `min` is the worst query, where one "
          "conversation repeating itself is most visible.")


def _print_buckets(runs: list[tuple[dict, list[dict]]], shortest: int) -> None:
    print(f"\nerror buckets (first cause that applies, pool={POOL_K}, "
          f"depth={TOP_K}):")
    print(f"{'variant':<18}" + "".join(f"{b:>{_BUCKET_COL}}" for b in BUCKETS))
    for run, per_query in runs:
        counts = bucket_counts(bucket_errors(per_query))
        print(f"{str(run['name'])[:18]:<18}"
              + "".join(f"{counts[b]:>{_BUCKET_COL}}" for b in BUCKETS))
    if shortest < POOL_K:
        print(f"  NOTE: the shortest saved list holds {shortest} ids, so "
              f"'buried' and 'missing_from_pool' are split at {shortest}, not "
              f"{POOL_K}, for runs that short")
    for run, per_query in runs:
        buckets = bucket_errors(per_query)
        named = [(b, ids) for b, ids in buckets.items() if b != "ok" and ids]
        if not named:
            continue
        print(f"\n  {run['name']}:")
        for bucket, ids in named:
            shown = ", ".join(ids[:10]) + (f" +{len(ids) - 10} more"
                                           if len(ids) > 10 else "")
            print(f"    {bucket:<16} {shown}")


def _print_pairs(runs: list[tuple[dict, list[dict]]], iters: int, seed: int) -> None:
    if len(runs) < 2:
        return
    print(f"\npaired bootstrap, A minus B ({iters} resamples, seed {seed}; "
          f"'*' marks an interval that excludes 0):")
    for i in range(len(runs) - 1):
        for j in range(i + 1, len(runs)):
            (run_a, per_a), (run_b, per_b) = runs[i], runs[j]
            n_pairs = len(_pair_rows(per_a, per_b))
            print(f"\n  A = {run_a['name']}   B = {run_b['name']}   "
                  f"({n_pairs} shared queries)")
            if n_pairs < 2:
                print(f"    insufficient data: {n_pairs} shared query, no "
                      f"interval can be drawn")
                continue
            print(f"    {'metric':<13}{'delta':>9}  {'95% CI':<24}"
                  f"{'call':<7}sig")
            for metric in PAIR_METRICS:
                res = paired_bootstrap_ci(per_a, per_b, metric,
                                          iters=iters, seed=seed)
                verdict = "tie"
                if res.excludes_zero:
                    verdict = "A>B" if res.delta > 0 else "A<B"
                ci = f"[{res.lo:+.4f}, {res.hi:+.4f}]"
                print(f"    {metric:<13}{res.delta:>+9.4f}  {ci:<24}"
                      f"{verdict:<7}{'*' if res.excludes_zero else ''}")


def _print_disagreements(runs: list[tuple[dict, list[dict]]], top: int) -> None:
    if len(runs) < 2:
        return
    print(f"\ndisagreements (A hits within top {top} while B does not - the "
          f"queries that actually decide a change):")
    said = False
    for i in range(len(runs) - 1):
        for j in range(i + 1, len(runs)):
            (run_a, per_a), (run_b, per_b) = runs[i], runs[j]
            for name_x, rows_x, name_y, rows_y in (
                    (run_a["name"], per_a, run_b["name"], per_b),
                    (run_b["name"], per_b, run_a["name"], per_a)):
                hits = disagreements(rows_x, rows_y, top)
                if not hits:
                    continue
                said = True
                print(f"\n  {name_x} hits, {name_y} misses ({len(hits)}):")
                for hit in hits:
                    print(f"    {str(hit['id'])[:34]:<34}#{hit['a_rank']:<3}"
                          f"{str(hit['query'])[:66]}")
    if not said:
        print("  none - every query in this set is a hit for both, or a miss "
              "for both")


def cmd_bench(args) -> None:
    """Score one or more search variants over the eval set.

    A variant is (name, search overrides, config overrides); each is run over
    every query, then the runs are compared: metric table, error buckets, a
    paired bootstrap per pair, and the disagreement list.
    """
    eval_file = str(_arg(args, "eval_file", DEFAULT_EVAL_FILE))
    top = int(_arg(args, "top", 10))
    rows, errors = load_rows(eval_file)
    _print_label_errors(errors)
    if not rows:
        raise SystemExit(f"no usable queries in {eval_file}")

    offline = bool(_arg(args, "offline"))
    saved: list[dict] = []
    for flag in ("baseline", "offline"):
        path = _arg(args, flag)
        if path:
            saved.extend(load_runs(str(path)))
    runs: list[tuple[dict, list[dict]]] = []
    for run in saved:
        run["config"]["from_file"] = True    # loaded, so never re-saved
        runs.append((run, label_run(run, rows)))
    if offline:
        names = ", ".join(str(run["name"]) for run, _ in runs) or "none"
        print(f"offline: scoring saved runs only ({names})")
    else:
        from . import mailbox as mb           # inside: only when we must touch the index
        # The harness inherits the CLI's --profile, so `vectormailbox --profile
        # legacy bench` measures the old pipeline as it was configured - store
        # included, which is what decides the BM25 tokenizer on an old index.
        cfg = mb.load_config(str(_arg(args, "config", "config.json")),
                             str(getattr(args, "profile", "") or ""))
        cfg["db_path"] = _db_path(args, cfg)
        variants = parse_variants(_arg(args, "variant"))
        print(f"db: {cfg['db_path']}   eval: {eval_file}   queries: {len(rows)}"
              f"   scored to top {top} (lists fetched to {max(top, POOL_K)} "
              f"for the pool bucket)")
        for variant in variants:
            print(f"\nrunning {variant.name}: {variant.describe()}")
            vcfg = dict(cfg)
            vcfg.update(variant.config)
            runner = Runner(vcfg)
            try:
                run = run_variant(runner, rows, variant, top)
            except Exception as exc:
                runner.close()
                print(f"  variant {variant.name} failed: {exc}")
                if isinstance(exc, TypeError):
                    # A signature mismatch is not a network problem: it means
                    # mailbox.py changed shape under this harness, and the next
                    # reader must not go looking for a dead endpoint.
                    print("  (skipped - mailbox.py does not accept what this "
                          "harness passed: check search_query() and "
                          "SearchParams against eval.py)")
                else:
                    print("  (skipped - an unreachable embedding endpoint is "
                          "the usual cause; bench a lexical variant or use "
                          "--offline)")
                continue
            runner.close()
            print(f"  {len(run['per_query'])} queries in "
                  f"{run['config']['total_ms'] / 1000.0:.1f} s")
            runs.append((run, label_run(run, rows)))

    if not runs:
        raise SystemExit("nothing to score: every variant failed and no "
                         "--baseline/--offline file was given")
    names = [str(r["name"]) for r, _ in runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate run names {names}: rename one variant "
                         f"(--variant NAME=...) or the saved file")

    print()
    _print_summary(runs, top)
    shortest = min((len(rec.get("ranked") or [])
                    for _, per_query in runs for rec in per_query), default=0)
    _print_buckets(runs, shortest)
    _print_diversity(runs, top)
    _print_pairs(runs, int(_arg(args, "iters", 10000)),
                 int(_arg(args, "seed", 0)))
    _print_disagreements(runs, top)

    save_path = _arg(args, "save")
    if save_path:
        fresh = [run for run, _ in runs if not run["config"].get("from_file")]
        if not fresh:
            print("\n--save: nothing fresh to save (offline run)")
        else:
            written = save_runs(fresh, str(save_path))
            print(f"\nsaved {len(written)} run file(s): {', '.join(written)}")


# ---------------------------------------------------------------------------
# evalgen: propose an eval set from the live index (SQLite only, no network)
# ---------------------------------------------------------------------------

#: strategy shares of --n; they sum to 1.0 and the largest remainder goes to
#: `subject`, whose labels are the most worth having.
EVALGEN_SHARES = (("subject", 0.30), ("sender", 0.20), ("date", 0.15),
                  ("attachment", 0.15), ("out_of_corpus", 0.20))

#: Queries nothing in a business mailbox should answer. A run that scores them
#: above the floor has a scoring problem, not a coverage problem.
OOD_SENTENCES = (
    "waar koop ik de beste kaas",
    "recept voor stamppot met rookworst",
    "hoe vaak moet ik de kat voederen",
    "kaartjes voor de eredivisie",
    "keuringsdatum van de schoorsteen",
    "zwangerschapscursus in de buurt",
    "hoeveel lichtjaar is het naar de maan",
    "handleiding van de stoomreiniger",
    "breuken optellen uitleg groep 7",
    "blu ray collector editie oorlog der werelden",
    "fietsroute langs de veluwemeer",
    "tuinverlichting met bewegingsmelder",
)

#: Random-word probes: words a business mailbox has no reason to contain.
OOD_WORDS = ("kaas", "fiets", "duin", "trap", "wolk", "ananas", "sneeuwbal",
             "marmer", "kompas", "giraffe", "zandloper", "ventilator",
             "bloemkool", "kanarie", "tandenborstel", "parasol")

#: Machine-generated senders. 'van noreply' is not a question a human asks, so
#: these are useless as sender queries - and they dominate a real inbox (see
#: `audit`'s transactional count).
TRANSACTIONAL = re.compile(
    r"(noreply|no[-_]?reply|donotreply|do-not-reply|notificat|newsletter|"
    r"nieuwsbrief|mailing|mailer-daemon|support|info@|notification|"
    r"alerts?|bounces?|automatisch|auto-)", re.I)

#: A weak label longer than this is not a label, it is the whole inbox; the row
#: says so in `note` rather than pretending recall@10 means something.
RELEVANT_CAP = 20


def _shares(total: int) -> dict[str, int]:
    """Largest-remainder split of `total` over EVALGEN_SHARES."""
    raw = {name: total * frac for name, frac in EVALGEN_SHARES}
    base = {name: int(val) for name, val in raw.items()}
    left = total - sum(base.values())
    for name, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]),
                          reverse=True):
        if left <= 0:
            break
        base[name] += 1
        left -= 1
    return base


def _norm_subject(subject: str) -> str:
    """The subject as a query, and as the key of the weak-label rule: no
    Re:/Fwd: prefixes (a reply is the same conversation, so it must carry the
    same label), one space per run of whitespace, no trailing punctuation."""
    text = " ".join((subject or "").split())
    while True:
        stripped = re.sub(r"^(re|fw|fwd|antwoord|doorgestuurd|aw)\s*:\s*",
                          "", text, flags=re.I)
        if stripped == text:
            break
        text = stripped
    return text.strip(" -–—:,.")


def _localpart(sender: str) -> str:
    return (sender or "").split("@")[0].strip()


def _cap(ids: list[str], limit: int) -> tuple[list[str], bool]:
    """Cap a weak-label list, saying so: a truncated `relevant` list silently
    understates recall, so the row has to admit it was cut."""
    return ids[:limit], len(ids) > limit


def _mails(n: int) -> str:
    return f"{n} {'mail' if n == 1 else 'mails'}"


def _rule_of(row: dict) -> str:
    """The rule name a row came from, read back off its `note`.

    Parsing the first word rather than splitting on ':' because the note
    carries subject text, and a subject full of colons must not eat the label.
    """
    note = str(row.get("note") or "")
    if note.startswith("rule="):
        return note.split()[0][len("rule="):]
    return "out_of_corpus" if row.get("out_of_corpus") else "other"


def gen_subject_rows(con, want: int, rng: random.Random) -> list[dict]:
    """Query = the subject line; relevant = every message with the same
    normalised subject. An exact-string rule, so it is deterministic - and
    weak: two mails with the same subject are not both the answer."""
    groups: dict[str, list[str]] = {}
    for mid, subject in con.execute(
            "SELECT message_id, subject FROM messages"
            " WHERE subject IS NOT NULL AND length(trim(subject)) >= 12"
            "   AND subject NOT LIKE '=%?%' ORDER BY message_id"):
        norm = _norm_subject(subject)
        if len(norm) >= 12:
            groups.setdefault(norm, []).append(str(mid))
    cands = sorted((n for n in groups if len(n.split()) >= 3), key=str.lower)
    out = []
    for norm in (rng.sample(cands, min(want, len(cands))) if cands else []):
        ids = sorted(groups[norm])
        shown, cut = _cap(ids, RELEVANT_CAP)
        out.append({
            "id": row_id_for(norm), "query": norm, "relevant": shown,
            "forbidden": [], "out_of_corpus": False,
            "note": f"rule=subject_exact '{norm}' ({_mails(len(ids))}"
                    f"{', capped at ' + str(RELEVANT_CAP) if cut else ''}) - "
                    f"WEAK label: same subject is not the same answer",
        })
    return out


def gen_sender_rows(con, want: int, rng: random.Random) -> list[dict]:
    """Query = 'van <sender localpart>' (how a Dutch speaker actually looks for
    a person); relevant = every message from that exact address."""
    cands = [(s, int(n)) for s, n in con.execute(
        "SELECT sender, COUNT(*) FROM messages WHERE sender IS NOT NULL"
        " AND sender LIKE '%@%' GROUP BY sender"
        " HAVING COUNT(*) BETWEEN 2 AND 60 ORDER BY sender")
        if not TRANSACTIONAL.search(s or "") and len(_localpart(s)) >= 3]
    out = []
    for sender, n in (rng.sample(cands, min(want, len(cands))) if cands else []):
        ids = [str(r[0]) for r in con.execute(
            "SELECT message_id FROM messages WHERE sender = ?"
            " ORDER BY message_id", (sender,))]
        shown, cut = _cap(ids, RELEVANT_CAP)
        query = f"van {_localpart(sender)}"
        out.append({
            "id": row_id_for(query), "query": query, "relevant": shown,
            "forbidden": [], "out_of_corpus": False,
            "note": f"rule=sender_exact {sender} ({_mails(n)}"
                    f"{', capped at ' + str(RELEVANT_CAP) if cut else ''}) - "
                    f"WEAK label: everything from one address, not one answer",
        })
    return out


def gen_date_rows(con, want: int, rng: random.Random) -> list[dict]:
    """Query = 'mail van <YYYY-MM-DD>'; relevant = every message stamped that
    UTC day. Days with 2..6 messages only: on a day with 40 mails, recall@10
    measures nothing and every run scores badly."""
    days = [d for d, _ in con.execute(
        "SELECT substr(date, 1, 10), COUNT(*) FROM messages"
        " WHERE date IS NOT NULL GROUP BY substr(date, 1, 10)"
        " HAVING COUNT(*) BETWEEN 2 AND 6 ORDER BY substr(date, 1, 10)")]
    out = []
    for day in (rng.sample(days, min(want, len(days))) if days else []):
        ids = [str(r[0]) for r in con.execute(
            "SELECT message_id FROM messages WHERE substr(date,1,10) = ?"
            " ORDER BY message_id", (day,))]
        shown, cut = _cap(ids, RELEVANT_CAP)
        query = f"mail van {day}"
        out.append({
            "id": row_id_for(query), "query": query, "relevant": shown,
            "forbidden": [], "out_of_corpus": False,
            "note": f"rule=date_exact {day} ({_mails(len(ids))}"
                    f"{', capped at ' + str(RELEVANT_CAP) if cut else ''}) - "
                    f"WEAK label: the date is indexed in the meta chunk",
        })
    return out


def gen_attachment_rows(con, want: int, rng: random.Random) -> list[dict]:
    """Query = the attachment filename; relevant = every message carrying an
    attachment chunk with that exact filename. Filenames are the strongest
    exact-string signal a hybrid retriever has, so these are the rows a lexical
    change must never lose."""
    groups: dict[str, list[str]] = {}
    for mid, label in con.execute(
            "SELECT message_id, label FROM chunks"
            " WHERE kind = 'attachment' AND canonical = 1"
            # Ordered by the label, not by the rowid: the id column is an
            # implementation detail, and these ids get sorted anyway.
            " ORDER BY message_id, label"):
        name = re.sub(r"^attachment:\s*", "", (label or "")).strip()
        stem, ext = os.path.splitext(name)
        if not ext or len(ext) > 6 or len(stem) < 3 or len(name) > 70:
            continue
        groups.setdefault(name.lower(), []).append(str(mid))
    cands = sorted((n for n in groups if os.path.splitext(n)[1]), key=str.lower)
    out = []
    for name in (rng.sample(cands, min(want, len(cands))) if cands else []):
        ids = sorted(set(groups[name]))
        shown, cut = _cap(ids, min(RELEVANT_CAP, 10))
        out.append({
            "id": row_id_for(name), "query": name, "relevant": shown,
            "forbidden": [], "out_of_corpus": False,
            "note": f"rule=attachment_file '{name}' ({_mails(len(ids))}"
                    f"{', capped at 10' if cut else ''}) - exact filename from "
                    f"the chunk label",
        })
    return out


def gen_ood_rows(want: int, rng: random.Random) -> list[dict]:
    """Out-of-corpus probes: fixed nonsense / out-of-domain Dutch sentences,
    then random-word queries.

    `relevant` is empty and `out_of_corpus` is true, so `summarise()` keeps
    them out of the recall means and `bucket_errors()` files them under
    not_in_corpus; what they measure is the abstention rate.
    """
    if want <= 0:
        return []
    sentences = list(OOD_SENTENCES)
    rng.shuffle(sentences)
    queries = sentences[:want]
    while len(queries) < want:
        queries.append(" ".join(rng.sample(OOD_WORDS, 3)))
    return [{
        "id": row_id_for(q), "query": q, "relevant": [], "forbidden": [],
        "out_of_corpus": True,
        "note": "out_of_corpus: fixed probe, nothing should match - scoring it "
                "above the relevance floor is a false positive",
    } for q in queries]


HOW_TO_EDIT = """\
how to edit {path}:
  One line per query. `id` must stay unique and stable - saved runs are joined
  to labels on it. Delete any row you would never really ask.
  `relevant` is a RULE-DERIVED weak label (see `note` on each row):
    - drop ids that are not really an answer: same subject, same sender or
      same day is not the same as "this is what I wanted"
    - add the ones the rule missed: `vectormailbox search "<query>"
      --json` prints message_id and gmail_url per hit, paste those ids in
    - a list capped by the rule says so in `note`; uncap it if you know better
  `forbidden` = ids that must never rank for that query (the 40th newsletter
  from a sender you asked about by name, for instance).
  `out_of_corpus: true` = nothing in the mailbox can answer it, so the run is
  scored on abstaining; that is what the probes at the end are for.
  Keep this file separate from eval.jsonl: this is the draft, eval.jsonl is the
  set you compare versions against.
  When you re-label by hand: python eval.py label "<query>" <rank> --eval-file
  {path}"""


def cmd_evalgen(args) -> None:
    """Propose candidate eval rows from the live index for a human to edit into
    a real eval set.

    No LLM, no network, no writes: every label comes from a deterministic SQL
    rule, and the seed fixes which candidates are picked, so two runs on the
    same index give the same draft. The rules are deliberately visible in
    `note`: a generated label is a starting point, never ground truth.
    """
    cfg = _config_of(args)
    db_path = _db_path(args, cfg)
    n = int(_arg(args, "n", 40))
    if n <= 0:
        raise SystemExit("--n must be positive")
    out_path = str(_arg(args, "out", DEFAULT_PROPOSED_FILE))
    seed = int(_arg(args, "seed", 0))
    rng = random.Random(seed)
    want = _shares(n)
    con = _ro_conn(db_path)
    try:
        rows = (gen_subject_rows(con, want["subject"], rng)
                + gen_sender_rows(con, want["sender"], rng)
                + gen_date_rows(con, want["date"], rng)
                + gen_attachment_rows(con, want["attachment"], rng)
                + gen_ood_rows(want["out_of_corpus"], rng))
    finally:
        con.close()
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["id"] in seen:
            continue          # two rules proposed the same query text
        seen.add(row["id"])
        unique.append(row)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# proposed eval rows from {db_path} (seed {seed}): weak "
                 f"labels, edit before using as eval.jsonl\n")
        for row in unique:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    got: dict[str, int] = {}
    for row in unique:
        key = _rule_of(row)
        got[key] = got.get(key, 0) + 1
    dropped = len(rows) - len(unique)
    print(f"proposed {len(unique)} rows -> {out_path}   (db {db_path}, "
          f"seed {seed}, read-only)")
    print(f"{'rule':<17}{'wanted':>7}{'got':>5}")
    for strategy, frac in EVALGEN_SHARES:
        rule = {"subject": "subject_exact", "sender": "sender_exact",
                "date": "date_exact", "attachment": "attachment_file",
                "out_of_corpus": "out_of_corpus"}[strategy]
        print(f"{rule:<17}{want[strategy]:>7}{got.get(rule, 0):>5}")
    if dropped:
        print(f"  ({dropped} duplicate query text(s) collapsed: two rules "
              "proposed the same query)")
    print("(fewer wanted than got nothing = that rule found fewer candidates; "
          "raise --n or loosen the rule's filter)\n")
    print(HOW_TO_EDIT.format(path=out_path))


# ---------------------------------------------------------------------------
# label: one query, one rank, appended to the eval file
# ---------------------------------------------------------------------------

def cmd_label(args) -> None:
    """`eval.py label "<query>" <rank>`: run one search, print the list, mark
    that rank's message id relevant for that query.

    Positional rank and no prompt on purpose: a human can then label thirty
    queries from a shell loop, and a mis-clicked command fails with an exit
    code instead of waiting for input forever in a cron job.
    """
    from . import mailbox as mb               # inside: mailbox.py may be mid-edit
    query = str(_arg(args, "query", "") or "").strip()
    if not query:
        raise SystemExit("label needs a query: eval.py label \"<query>\" <rank>")
    rank = int(_arg(args, "rank", 0))
    if rank < 1:
        raise SystemExit("rank is 1-based (which result to mark relevant)")
    eval_file = str(_arg(args, "eval_file", DEFAULT_EVAL_FILE))
    top = max(int(_arg(args, "top", 10)), rank)

    cfg = mb.load_config(str(_arg(args, "config", "config.json")))
    cfg["db_path"] = _db_path(args, cfg)
    runner = Runner(cfg)
    try:
        ids, scores, meta = runner.run(query, {"top": top})
    finally:
        runner.close()
    if not ids:
        raise SystemExit("the search returned nothing to label")
    if rank > len(ids):
        raise SystemExit(f"rank {rank} is past the end of {len(ids)} results")

    con = _ro_conn(cfg["db_path"])
    try:
        marks = ",".join("?" * len(ids))
        info = {r[0]: r for r in con.execute(
            f"SELECT message_id, date, sender, subject FROM messages"
            f" WHERE message_id IN ({marks})", ids)}
    finally:
        con.close()

    print(f"Query: {query!r}   db: {cfg['db_path']}"
          + ("   (the run abstained)" if meta.get("abstained") else ""))
    print(f"Marking rank {rank} relevant in {eval_file}\n")
    for i, mid in enumerate(ids, 1):
        rec = info.get(mid)
        date, sender, subject = rec[1:4] if rec else ("?", "?", "(not in messages)")
        score = scores[i - 1] if i - 1 < len(scores) else None
        shown = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        print(f"{'>' if i == rank else ' '} {i:>3}  score {shown}  "
              f"{str(date)[:10]}  {str(sender)[:28]:<28} {str(subject)[:58]}")
        print(f"        {mid}")

    rows, errors = load_rows(eval_file)
    _print_label_errors(errors)
    chosen = str(ids[rank - 1])
    target = next((r for r in rows
                   if r["id"] == row_id_for(query)
                   or r["query"].casefold() == query.casefold()), None)
    if target is None:
        target = {"id": row_id_for(query), "query": query, "relevant": [],
                  "forbidden": [], "out_of_corpus": False, "note": ""}
        rows.append(target)
    added = chosen not in target["relevant"]
    if added:
        target["relevant"].append(chosen)
    target["out_of_corpus"] = False
    target["note"] = (f"rule=manual rank={rank} of {len(ids)} on "
                      f"{time.strftime('%Y-%m-%d %H:%M')} - hand labelled")
    save_rows(rows, eval_file)
    print(f"\n{'added ' if added else 'already had '}"
          f"{chosen} -> {target['id']}: {len(target['relevant'])} relevant "
          f"id(s) for {query!r}")


# ---------------------------------------------------------------------------
# audit: read-only index diagnostics
# ---------------------------------------------------------------------------

#: Quoted reply text is indexed like real content: it fills the bm25 pool with
#: words the sender never wrote, and it inflates chunk counts per message.
QUOTE_MARKERS = (
    ("quoted: Op ... schreef:", re.compile(r"(?im)^\s*op .{0,90}schreef[!:]")),
    ("quoted: On ... wrote:", re.compile(r"(?im)^\s*on .{0,120}wrote[!:]")),
    ("quoted: Original Message",
     re.compile(r"(?i)-{4,}\s*original message\s*-{4,}")),
    ("quoted: '>' block", re.compile(r"(?m)^\s{0,3}>")),
    ("quoted: forwarded header",
     re.compile(r"(?im)^\s*-{6,}\s*(forwarded message|doorgestuurd"
                r" bericht)")),
)

#: Signatures and legal footers: pure noise for retrieval, and they are the
#: reason one sender's every mail matches a query about "afmelden".
FOOTER_MARKERS = (
    ("footer: unsubscribe", re.compile(r"(?i)unsubscribe")),
    ("footer: afmelden", re.compile(r"(?i)afmelden|uitschrijven")),
    ("footer: confidentiality", re.compile(
        r"(?i)confidentiality|vertrouwelijk|this e-?mail is intended only")),
    ("footer: virus disclaimer", re.compile(
        r"(?i)this message.{0,40}(virus|scanned)|kontroleer op virussen")),
)

RFC2047 = re.compile(r"=\?[^?\s]+\?[BbQq]\?[^?]*\?=")

#: Dutch probe pairs: the porter stemmer folds singular and plural together, so
#: these two counts say whether that helps or hurts the exact word a user
#: typed. (fts > LIKE: it also matches words you did not type. fts < LIKE: the
#: tokenizer dropped tokens it should have kept.)
FTS_PROBES = (("factuur", "facturen"), ("domein", "domeinen"),
              ("wachtwoord", "wachtwoorden"))


def _percentile(sorted_vals: list[float], frac: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round(frac * (len(sorted_vals) - 1)))
    return float(sorted_vals[max(0, min(len(sorted_vals) - 1, idx))])


def _scan_bodies(con) -> dict:
    """One pass over the canonical body chunks.

    Lengths, exact duplicates, quoted replies and footers all come out of the
    same scan on purpose: this is the biggest table in the index, and four
    scans would make `audit` the slowest thing in the project.
    """
    out = {
        "bodies": 0, "lengths": [], "dupes": {}, "examples": {},
        "quoted": {}, "any_quote": 0, "footers": {}, "any_footer": 0,
    }
    for text in con.execute("SELECT COALESCE(text,'') FROM chunks"
                            " WHERE canonical = 1 AND kind = 'body'"):
        body = text[0]
        out["bodies"] += 1
        out["lengths"].append(len(body))
        norm = " ".join(body.split()).casefold()
        key = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        out["dupes"][key] = out["dupes"].get(key, 0) + 1
        if out["dupes"][key] == 2:
            out["examples"][key] = " ".join(body.split())[:70]
        for name, rx in QUOTE_MARKERS:
            if rx.search(body):
                out["quoted"][name] = out["quoted"].get(name, 0) + 1
                out["any_quote"] += 1
                break
        for name, rx in FOOTER_MARKERS:
            if rx.search(body):
                out["footers"][name] = out["footers"].get(name, 0) + 1
                out["any_footer"] += 1
                break
    out["lengths"].sort()
    return out


def cmd_audit(args) -> None:
    """Read-only diagnostics on the configured index: how much is in there, and
    which corpus smells (duplicate bodies, quoted replies, footers, undecoded
    subjects, transactional flood) quietly eat the top-k slots.

    Every statement runs over `mode=ro`: an audit that can write is an audit
    that can break the thing it measures.
    """
    cfg = _config_of(args)
    db_path = _db_path(args, cfg)
    con = _ro_conn(db_path)
    print(f"index audit: {db_path}  ({os.path.getsize(db_path) / 1e6:.0f} MB, "
          f"opened mode=ro - nothing written)")

    msgs = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vecs = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    fts = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    dim_row = con.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()
    dim = int(dim_row[0]) if dim_row else 0
    print(f"\nmessages {msgs}   chunks {chunks}   vectors {vecs}   "
          f"fts docs {fts}")
    if fts != chunks:
        print(f"  WARNING: {abs(chunks - fts)} chunks have no FTS row, so the "
              f"lexical leg cannot see them: vectormailbox reindex "
              f"--rebuild-fts")
    if vecs != chunks:
        # Two different problems hide behind one unequal pair: chunks the dense
        # leg cannot reach, and vectors whose chunk is gone. They need opposite
        # actions, so they get separate lines.
        unembedded = con.execute(
            "SELECT COUNT(*) FROM chunks c LEFT JOIN embeddings e"
            " ON e.chunk_id = c.id WHERE e.chunk_id IS NULL").fetchone()[0]
        if unembedded:
            print(f"  NOTE: {unembedded} chunks have no vector (windows past "
                  f"max_windows_per_source, or media the endpoint refused)")
        orphan = vecs - (chunks - unembedded)
        if orphan > 0:
            print(f"  NOTE: {orphan} vectors belong to chunks that no longer "
                  f"exist - a reindex replaced their tier. They cost disk, not "
                  f"recall: 'reindex' prunes them")
    print(f"  vector dim {dim}  (~{chunks * dim * 4 / 1e6:.0f} MB of vectors "
          f"loaded per search)")

    print("\nchunks per tier:")
    for tier, canon, n in con.execute(
            "SELECT COALESCE(tier,'?'), COALESCE(canonical,0), COUNT(*)"
            " FROM chunks GROUP BY tier, canonical ORDER BY tier, canonical"):
        print(f"  {tier:<7} {'canonical' if canon else 'derived  '}: {n}")
    print("chunks per kind:")
    for kind, n in con.execute("SELECT COALESCE(kind,'?'), COUNT(*) FROM chunks"
                               " GROUP BY kind ORDER BY COUNT(*) DESC"):
        print(f"  {kind:<11} {n}")

    scan = _scan_bodies(con)
    bodies = scan["bodies"]
    lens = scan["lengths"]
    print(f"\ncanonical body chunks: {bodies}")
    print(f"  normalised-body length: median {_percentile(lens, 0.5):.0f} "
          f"chars, p90 {_percentile(lens, 0.9):.0f} chars, "
          f"max {lens[-1] if lens else 0} chars")

    groups = {k: v for k, v in scan["dupes"].items() if v > 1}
    redundant = sum(v - 1 for v in groups.values())
    print(f"exact-duplicate normalised bodies: {len(groups)} groups, "
          f"{redundant} redundant chunks"
          + (f" ({redundant / bodies:.1%} of bodies)" if bodies else "")
          + f", largest group {max(groups.values()) if groups else 0}")
    for key, n in sorted(groups.items(), key=lambda kv: -kv[1])[:3]:
        print(f"  x{n}: {scan['examples'].get(key, '?')}")

    def pct(n: int) -> str:
        return f"{(n / bodies if bodies else 0):.1%}"

    print(f"bodies carrying quoted-reply markers: {scan['any_quote']} "
          f"({pct(scan['any_quote'])})")
    for name, _ in QUOTE_MARKERS:
        if scan["quoted"].get(name):
            print(f"  {name:<26} {scan['quoted'][name]:>6}  "
                  f"({pct(scan['quoted'][name])})")
    print(f"bodies carrying footer / unsubscribe / confidentiality markers: "
          f"{scan['any_footer']} ({pct(scan['any_footer'])})")
    for name, _ in FOOTER_MARKERS:
        if scan["footers"].get(name):
            print(f"  {name:<26} {scan['footers'][name]:>6}  "
                  f"({pct(scan['footers'][name])})")
    print("  (quoted text and footers are indexed as if the sender wrote them: "
          "they fill the bm25 pool and the top-k)")

    per_msg = sorted(n for (n,) in con.execute(
        "SELECT COUNT(*) FROM chunks GROUP BY message_id"))
    print(f"\nmessages with >20 chunks: {sum(1 for n in per_msg if n > 20)}   "
          f">40 chunks: {sum(1 for n in per_msg if n > 40)}")
    print(f"  chunks per message: median {_percentile(per_msg, 0.5):.0f}, "
          f"p90 {_percentile(per_msg, 0.9):.0f}, "
          f"max {per_msg[-1] if per_msg else 0} "
          f"(a message with 40 chunks gets 40 chances per leg to reach the top)")

    still = [(m, s) for m, s in con.execute(
        "SELECT message_id, subject FROM messages WHERE subject IS NOT NULL")
        if RFC2047.search(s or "")]
    print(f"\nsubjects still RFC 2047-encoded: {len(still)}"
          + (f" ({len(still) / msgs:.1%} of messages)" if msgs else "")
          + " - these index as '=?utf-8?B?...?=', not as words")
    for _mid, subject in still[:3]:
        print(f"  {(subject or '')[:88]}")

    print("top senders:")
    for sender, n in con.execute(
            "SELECT sender, COUNT(*) FROM messages GROUP BY sender"
            " ORDER BY COUNT(*) DESC, sender LIMIT 10"):
        print(f"  {n:>5}  {sender}")
    senders = con.execute("SELECT COALESCE(sender,''), COUNT(*) FROM messages"
                          " GROUP BY sender").fetchall()
    trans = sum(int(n) for s, n in senders if TRANSACTIONAL.search(s))
    trans_senders = sum(1 for s, _ in senders if TRANSACTIONAL.search(s))
    print(f"transactional-looking senders: {trans_senders} addresses, {trans} "
          f"messages" + (f" ({trans / msgs:.1%})" if msgs else "")
          + " - the bulk of the corpus, which is why recall on 'real' queries "
            "moves slowly")

    print("\nFTS probes (the porter stemmer and the trailing prefix match "
          "decide these):")
    print(f"  {'form':<14}{'fts chunks':>12}{'fts prefix':>12}{'LIKE text':>12}")
    for singular, plural in FTS_PROBES:
        for form in (singular, plural):
            try:
                fts_n = con.execute("SELECT COUNT(*) FROM chunks_fts"
                                    " WHERE chunks_fts MATCH ?",
                                    (f'"{form}"',)).fetchone()[0]
                pre_n = con.execute("SELECT COUNT(*) FROM chunks_fts"
                                    " WHERE chunks_fts MATCH ?",
                                    (f"{form}*",)).fetchone()[0]
            except sqlite3.OperationalError as exc:
                print(f"  probe {form!r} failed: {exc}")
                fts_n = pre_n = 0
            like_n = con.execute("SELECT COUNT(*) FROM chunks"
                                 " WHERE COALESCE(text,'') LIKE ?",
                                 (f"%{form}%",)).fetchone()[0]
            print(f"  {form:<14}{fts_n:>12}{pre_n:>12}{like_n:>12}")
        print("    (fts > LIKE: the stemmer also matches words you did not "
              "type; fts < LIKE: tokens the tokenizer dropped)")
    con.close()
    print("\naudit opened the database read-only; it changed nothing.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    """Flags every eval subcommand shares.

    `--config` defaults to SUPPRESS: mailbox.py has its own global --config,
    and a subparser writes its defaults into the same namespace, so a plain
    default here would silently override the value the user gave before the
    subcommand.
    """
    p.add_argument("--config", default=argparse.SUPPRESS,
                   help="config file for db_path / endpoint (default: the "
                        "global --config, else config.json)")
    p.add_argument("--db", help="override db_path from the config")


def add_subcommands(sub) -> None:
    """Register bench / evalgen / label / audit on an existing subparsers object.

    mailbox.py calls this from main() and gets the four commands wired to the
    cmd_* functions above; the attributes below are exactly what they read.
    """
    pb = sub.add_parser(
        "bench", help="score search variants over eval/eval.jsonl: "
                      "recall/precision/MRR/nDCG, error buckets, paired "
                      "bootstrap, disagreements")
    _add_common(pb)
    pb.add_argument("--eval-file", dest="eval_file", default=DEFAULT_EVAL_FILE,
                    help=f"JSONL eval set (default: {DEFAULT_EVAL_FILE})")
    pb.add_argument("--variant", action="append",
                    help="NAME=key=val,key=val (repeatable). Keys are search "
                         "options; prefix cfg. for config keys; '|' makes a "
                         "list (tiers=small|large). Repeat to compare "
                         f"(default: {', '.join(DEFAULT_VARIANTS)})")
    pb.add_argument("--baseline", metavar="FILE",
                    help="add a saved run record to the comparison without "
                         "re-running the embedder")
    pb.add_argument("--save", metavar="FILE",
                    help="write this run's per-query ranked lists (several "
                         "variants: one file each, {name} is substituted)")
    pb.add_argument("--offline", metavar="FILE",
                    help="score saved run records only - no endpoint, no "
                         "index")
    pb.add_argument("--top", type=int, default=10,
                    help="cut-off to score (lists are fetched to 50 for the "
                         "pool bucket; default 10)")
    pb.add_argument("--iters", type=int, default=10000,
                    help="bootstrap resamples per pair (default 10000)")
    pb.add_argument("--seed", type=int, default=0,
                    help="bootstrap seed (default 0; deterministic)")
    pb.set_defaults(func=cmd_bench)

    pg = sub.add_parser(
        "evalgen", help="propose eval rows from the index (SQL rules only, no "
                        "network, no writes) for a human to edit")
    _add_common(pg)
    pg.add_argument("--n", type=int, default=40,
                    help="rows to propose (default 40), split subject 30%%, "
                         "sender 20%%, date 15%%, attachment 15%%, "
                         "out-of-corpus 20%%")
    pg.add_argument("--out", default=DEFAULT_PROPOSED_FILE,
                    help=f"where to write (default: {DEFAULT_PROPOSED_FILE})")
    pg.add_argument("--seed", type=int, default=0,
                    help="candidate-selection seed (default 0; deterministic)")
    pg.set_defaults(func=cmd_evalgen)

    pl = sub.add_parser(
        "label", help="run one query, print the list, mark result <rank> "
                      "relevant in the eval file (scriptable, no prompt)")
    _add_common(pl)
    pl.add_argument("query")
    pl.add_argument("rank", type=int, help="1-based position to mark relevant")
    pl.add_argument("--eval-file", dest="eval_file", default=DEFAULT_EVAL_FILE,
                    help=f"eval set to update (default: {DEFAULT_EVAL_FILE})")
    pl.add_argument("--top", type=int, default=10,
                    help="results to fetch (default 10; raised to the rank if "
                         "needed)")
    pl.set_defaults(func=cmd_label)

    pa = sub.add_parser(
        "audit", help="read-only index diagnostics: duplicates, quoted "
                      "replies, footers, undecoded subjects, FTS probes")
    _add_common(pa)
    pa.set_defaults(func=cmd_audit)


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser, so `python eval.py audit` works without going through
    mailbox.py (which is the real entry point in normal use)."""
    p = argparse.ArgumentParser(
        prog="eval.py",
        description="Retrieval-quality harness for mailbox.py (read-only on "
                    "the index)")
    p.add_argument("--config", default="config.json", help="config file")
    sub = p.add_subparsers(dest="command", required=True)
    add_subcommands(sub)
    return p


def main(argv: Optional[list] = None) -> None:
    # A Windows console is cp1252/cp850 and mail is full of emoji: one
    # undisplayable character must not lose the whole audit output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
