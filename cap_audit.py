"""Read-only audit of the `max_windows_per_source` cap on an existing index.

Answers three questions without writing anything (SQLite mode=ro):
  1. which cap value covers every source in this mailbox,
  2. how many window texts are missing from the dense leg right now,
  3. what a `reindex` of each tier selection would cost in embeddings.

    python cap_audit.py [mailbox.db]
"""
import sqlite3
import statistics
import sys

from vectormailbox import mailbox as mb

DB = sys.argv[1] if len(sys.argv) > 1 else "mailbox.db"
cfg = mb.load_config("config.json")
sizes = mb.chunk_sizes_of(cfg)
overlap = float(cfg.get("chunk_overlap_frac", mb.DEFAULTS["chunk_overlap_frac"]))
max_src = int(cfg.get("max_source_chars", mb.DEFAULTS["max_source_chars"]))
cap = int(cfg.get("max_windows_per_source", mb.DEFAULTS["max_windows_per_source"]))
ladder = sorted(sizes, key=lambda k: sizes[k])

print(f"config: sizes={sizes} overlap={overlap} max_source_chars={max_src} "
      f"max_windows_per_source={cap}")

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
print("\nindex metadata stored in the db (note: the window cap is NOT recorded):")
for k, v in con.execute("SELECT key, value FROM index_state ORDER BY key"):
    print(f"  {k} = {v}")

print("\nchunks by tier:")
for tier, canon, n in con.execute(
        "SELECT COALESCE(tier,'?'), canonical, COUNT(*) FROM chunks "
        "GROUP BY tier, canonical ORDER BY tier, canonical"):
    print(f"  {tier:7s} {'canonical' if canon else 'derived '}: {n}")

# Walk every canonical (whole-source) text: what the cap keeps vs what fits.
need_max = {t: 0 for t in ladder}
cut_src = {t: 0 for t in ladder}
missing = {t: 0 for t in ladder}
full = {t: 0 for t in ladder}
sources = 0
for (text,) in con.execute(
        "SELECT text FROM chunks WHERE canonical=1 AND tier!='media' "
        "  AND LENGTH(COALESCE(text,''))>?", (min(sizes.values()),)):
    sources += 1
    at_cap, uncapped = {}, {}
    for capped, bucket in ((cap, at_cap), (10 ** 9, uncapped)):
        for r in mb.tiered_chunks(text, sizes, overlap, max_src, capped):
            if not r["canonical"]:
                bucket[r["tier"]] = bucket.get(r["tier"], 0) + 1
    for t in ladder:
        full[t] += uncapped.get(t, 0)
        need_max[t] = max(need_max[t], uncapped.get(t, 0))
        if uncapped.get(t, 0) > at_cap.get(t, 0):
            cut_src[t] += 1
            missing[t] += uncapped[t] - at_cap[t]

print(f"\ncanonical text sources: {sources}")
print(f"{'tier':8s}{'worst case':>12s}{'sources cut':>13s}{'windows missing':>17s}"
      f"{'full set':>10s}")
for t in ladder:
    print(f"{t:8s}{need_max[t]:12d}{cut_src[t]:13d}{missing[t]:17d}{full[t]:10d}")
print(f"\ntotal dense-coverage gap at cap {cap}: {sum(missing.values())} window "
      f"texts over {sum(cut_src.values())} source/tier cases")

need = sorted(
    sum(1 for r in mb.tiered_chunks(t, sizes, overlap, max_src, 10 ** 9)
        if not r["canonical"] and r["tier"] == "small")
    for (t,) in con.execute(
        "SELECT text FROM chunks WHERE canonical=1 AND tier NOT IN ('media','small') "
        "  AND LENGTH(COALESCE(text,''))>?", (sizes['small'],)))
if need:
    print(f"small-tier windows a source needs here: median {statistics.median(need)}, "
          f"p99 {need[int(len(need) * 0.99) - 1]}, max {need[-1]} "
          f"-> a cap of {need[-1]} covers this mailbox; "
          f"{need[-1] + 4} leaves headroom")

batch = int(cfg.get("batch_size", 32))
print("\nreindex cost (reindex re-embeds every derived window of the tiers it "
      "builds, not just the missing ones):")
for label, sel in [(t, [t]) for t in ladder] + [("all", ladder)]:
    n = sum(full[t] for t in sel)
    print(f"  reindex --tiers {label:17s} -> {n:6d} chunks, "
          f"~{-(-n // batch)} endpoint calls at batch {batch}")
con.close()
