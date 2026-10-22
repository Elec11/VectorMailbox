"""Turn eval/eval.proposed.jsonl into eval/eval.jsonl, the set versions are
compared against.

Two honest corrections, both recorded in each row's `note`:

1. A rule that matched more mails than the tool can show (a sender with 38
   messages, a subject line the shop sends daily) makes recall@10 <= 10/N no
   matter how good the ranking is. Those label lists are capped to the 10 most
   recent matches: the metric then asks "are the ten you would want the first
   ten you get", which is the question a top-10 list can actually answer.

2. evalgen's rules are all *metadata* (subject/sender/date/filename). Real
   queries are about content, so this adds rows from exact phrases: a mail
   containing 'authorisatiecode' verbatim is an answer to a search for
   'authorisatiecode' by the same rule that makes an exact subject an answer.

Read-only: the mailbox db is opened mode=ro, only the eval files are written.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

MAX_LABELS = 10

#: content queries, phrased the way someone would type them - including a few
#: whole-sentence ones, because the words around the phrase are exactly what
#: drowns a vector query. A row is only emitted when the phrase matches between
#: 1 and 8 messages: fewer than one is nothing to score, more than eight is a
#: topic, not a query.
PHRASES = [
    ("opdrachtbevestiging", "opdrachtbevestiging"),
    ("ssl certificaat vernieuwen", "SSL certificaat"),
    ("email account aanmaken", "account aanmaken"),
    ("factuur betalen", "factuur betalen"),
    ("abonnement opzeggen", "opzeggen"),
    ("transfer code", "transfer code"),
    ("hoe vraag ik een transfer code op", "transfer code"),
    ("aanmaning", "aanmaning"),
    ("oplevering", "oplevering"),
    ("afgekeurd", "afgekeurd"),
    ("dmarc", "dmarc"),
    ("nameservers aanpassen", "nameservers"),
    ("tweestapsverificatie", "tweestapsverificatie"),
    ("storing vanochtend", "storing"),
    ("migratie van de server", "migratie"),
]


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectormailbox import eval as ev    # noqa: E402  (pure module: no mailbox import)


def conn_ro(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise SystemExit(f"no such database: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append(json.loads(s))
    return rows


#: the harness's own id function: saved runs are joined to labels on the id,
#: so a second spelling of it would silently orphan every row this adds.
row_id = ev.row_id_for


def cap_labels(con: sqlite3.Connection, row: dict) -> str:
    """Keep the newest MAX_LABELS ids of a rule-derived list; say so in note."""
    ids = list(row.get("relevant") or [])
    if len(ids) <= MAX_LABELS:
        return ""
    marks = ",".join("?" * len(ids))
    got = con.execute(
        f"select message_id from messages where message_id in ({marks}) "
        f"order by date desc limit ?", (*ids, MAX_LABELS)).fetchall()
    kept = [r[0] for r in got]
    if len(kept) < MAX_LABELS:
        return ""
    row["relevant"] = kept
    return (f" labels capped to the {MAX_LABELS} most recent "
            f"(was {len(ids)}: a top-10 list cannot answer more)")


def phrase_rows(con: sqlite3.Connection, wanted: list[str]) -> list[dict]:
    out, seen = [], set()
    for query, phrase in PHRASES:
        if wanted and query not in wanted:
            continue
        like = f"%{phrase}%"
        # The whole message lives in the canonical chunk, not in `messages`.
        hits = con.execute(
            "select distinct m.message_id from messages m "
            "join chunks c on c.message_id = m.message_id "
            "where c.canonical = 1 and (lower(c.text) like lower(?) "
            "   or lower(m.subject) like lower(?)) "
            "order by m.date desc limit 40", (like, like)).fetchall()
        ids = [r[0] for r in hits]
        if not (1 <= len(ids) <= 8):
            print(f"  skip {query!r}: phrase {phrase!r} matched {len(ids)} "
                  f"(needs 1-8)")
            continue
        rid = row_id(query)
        if rid in seen:
            continue
        seen.add(rid)
        out.append({
            "id": rid,
            "query": query,
            "relevant": ids[:MAX_LABELS],
            "forbidden": [],
            "note": f"rule=phrase {phrase!r} ({len(ids)} mails contain it "
                    f"verbatim) - weak label: the phrase is the evidence, "
                    f"delete the row if you would not really ask this",
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", default="eval/eval.proposed.jsonl")
    ap.add_argument("--out", default="eval/eval.jsonl")
    ap.add_argument("--db", default="mailbox.db")
    ap.add_argument("--phrase", action="append",
                    help="only add these content queries (repeatable)")
    ap.add_argument("--no-phrases", action="store_true")
    args = ap.parse_args(argv)

    rows = load_rows(args.src)
    con = conn_ro(args.db)
    try:
        capped = 0
        for row in rows:
            note = cap_labels(con, row)
            if note:
                row["note"] = (row.get("note", "") + ";").rstrip(";") + note
                capped += 1
        extra = [] if args.no_phrases else phrase_rows(
            con, args.phrase or [])
    finally:
        con.close()

    have = {r["id"] for r in rows}
    rows.extend(r for r in extra if r["id"] not in have)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# scored eval set - produced by eval/curation.py from "
                 f"{args.src} against {args.db}\n")
        fh.write("# delete rows you would never ask; add ids with: "
                 "python eval.py label \"<query>\" <rank> --eval-file "
                 f"{args.out}\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = [r for r in rows if not r.get("out_of_corpus")]
    abstain = [r for r in rows if r.get("out_of_corpus")]
    print(f"\n{args.out}: {len(scored)} in-corpus queries "
          f"(+{len(abstain)} out-of-corpus probes), {capped} label lists capped, "
          f"{len(extra)} content rows added")
    short = [r for r in scored if len(r.get("relevant") or []) < 2]
    if short:
        print(f"note: {len(short)} rows have a single labelled answer - MRR "
              f"carries those, recall@10 is 1 or 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
