"""Fill only the window tails `max_windows_per_source` dropped — no IMAP.

`reindex` rebuilds every derived window of the tiers it builds, so recovering a
few hundred missing tails would re-embed the whole mailbox again. This script
compares, per source, what the *current* config derives against what the
database actually holds, and re-embeds only the messages that are short.

    python reindex_tails.py --dry-run            # what it would do + cost
    python reindex_tails.py                      # asks, then does it
    python reindex_tails.py --tiers small,large  # default: small only

Same guarantees as `reindex`: it reads the canonical whole-source chunks, so it
never touches Gmail, never touches canonical or media chunks, and a second run
finds nothing left to do.
"""
import argparse
import sys

import tqdm

import mailbox as mb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--tiers", default="small",
                    help="comma-separated tiers to rebuild (default small: the "
                         "only tier the cap can bite in practice)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    cfg = mb.load_config(args.config)
    store = mb.Store(cfg["db_path"], cfg)
    sizes = mb.chunk_sizes_of(cfg)
    overlap = float(cfg.get("chunk_overlap_frac", mb.DEFAULTS["chunk_overlap_frac"]))
    max_src = int(cfg.get("max_source_chars", mb.DEFAULTS["max_source_chars"]))
    max_win = int(cfg.get("max_windows_per_source",
                          mb.DEFAULTS["max_windows_per_source"]))
    ladder = sorted(sizes, key=lambda k: sizes[k])
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    bad = [t for t in tiers if t not in sizes]
    if bad:
        raise SystemExit(f"unknown tier(s) {bad}; configured ladder: {ladder}")

    def derive(row):
        """Derived chunks the current config wants for this canonical source."""
        out = []
        for r in mb.tiered_chunks(row["text"], sizes, overlap, max_src, max_win):
            if r["canonical"] or r["tier"] not in tiers:
                continue
            out.append({"kind": row["kind"], "src": row["src"], "tier": r["tier"],
                        "pos": r["pos"], "nwin": r["nwin"], "canonical": False,
                        "label": mb.chunk_label(row["src"], r["tier"], r["pos"],
                                                r["nwin"]),
                        "text": r["text"]})
        return out

    stored: dict = {}
    ph = ",".join("?" * len(tiers))
    for mid, src, text in store.conn.execute(
            f"SELECT message_id, src, text FROM chunks "
            f"WHERE canonical = 0 AND tier IN ({ph})", list(tiers)):
        # keyed without the tier: tiered_chunks never stores the same text at
        # two tiers of one source, so "is this text there?" is per (src).
        stored.setdefault((mid, src), set()).add(text)

    sources = store.canonical_sources(None)
    by_mid: dict = {}
    short: set = set()
    for row in sources:
        by_mid.setdefault(row["message_id"], []).append(row)
        have = stored.get((row["message_id"], row["src"]), set())
        if any(c["text"] not in have for c in derive(row)):
            short.add(row["message_id"])

    plan = []
    for mid in sorted(short):
        chunks = [c for row in by_mid[mid] for c in derive(row)]
        if chunks:
            plan.append((by_mid[mid][0], chunks))
    new_chunks = sum(len(c) for _, c in plan)
    batch = max(1, int(cfg["batch_size"]))
    calls = -(-new_chunks // batch)

    print(f"sources scanned:      {len(sources)}")
    print(f"messages short:       {len(short)}")
    print(f"tiers to rebuild:     {', '.join(tiers)} (cap {max_win}, ladder "
          f"{', '.join(f'{k}={sizes[k]}' for k in ladder)})")
    print(f"chunks to embed:      {new_chunks} (~{calls} endpoint calls at "
          f"batch {batch})")
    s = store.stats()
    print(f"store now:            {s['messages']} messages, {s['chunks']} chunks "
          f"({s['derived_chunks']} derived)")
    if not short:
        print("nothing missing: every source already holds the windows the "
              "current config derives.")
        return
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not cfg["embed_endpoint"]:
        raise SystemExit("missing config: embed_endpoint (needed to embed chunks)")
    if not args.yes:
        answer = input(f"\nembed {new_chunks} chunks ({calls} calls)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    client = mb.EmbeddingClient(cfg)
    removed = inserted = 0
    with tqdm.tqdm(total=new_chunks, desc="filling tails") as bar:
        for start in range(0, len(plan), 200):           # bounded memory, resumable
            group = plan[start:start + 200]
            mids = sorted({row["message_id"] for row, _ in group})
            removed += store.delete_derived(tiers, mids)
            items = [(row, c) for row, chunks in group for c in chunks]
            vecs = client.embed_texts([c["text"] for _, c in items])
            cur = store.conn.cursor()
            for (row, c), v in zip(items, vecs):
                msg = mb.Message(
                    message_id=row["message_id"], folder="",
                    date=mb.parse_date(row["date"]), sender=row["sender"] or "",
                    to=row["to_addrs"] or "", cc=row["cc"] or "",
                    subject=row["subject"] or "", body_text="",
                )
                smeta = mb.chunk_search_meta(msg, row["attachment_exts"] or "",
                                             c["kind"], c["src"])
                store.add_chunk(cur, row["message_id"], c, v, smeta)
                inserted += 1
            store.conn.commit()
            bar.update(len(items))
    print(f"replaced {removed} derived chunks with {inserted} new ones")
    store.save_index_meta()
    s = store.stats()
    print(f"store now: {s['messages']} messages, {s['chunks']} chunks "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(s['by_tier'].items()))}), "
          f"{s['fts_docs']} fts docs")
    note = mb.truncation_note()
    if note:
        print(note)
    else:
        print("no truncation this run: the cap now covers every source.")


if __name__ == "__main__":
    sys.exit(main())
