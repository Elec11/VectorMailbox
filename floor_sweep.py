"""Where the relevance floor belongs, measured from a saved bench run.

Reads the top relevance of every query in a run file (`--save run.json` writes
one per variant) and splits it by the `out_of_corpus` flag in the eval file, then
sweeps candidate floors: how many unanswerable queries a floor would catch,
against how many answerable ones it would silence. On a real inbox the two
distributions overlap almost completely and no number wins, which is why
`min_relevance` is a guard against an empty result set and not a feature.

    python floor_sweep.py [final-new.json] [eval/eval.jsonl]
"""
import json
import sys

run_file = sys.argv[1] if len(sys.argv) > 1 else "final-new.json"
eval_file = sys.argv[2] if len(sys.argv) > 2 else "eval/eval.jsonl"

labels = {}
with open(eval_file, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#"):
            row = json.loads(line)
            labels[row["id"]] = bool(row.get("out_of_corpus"))

run = json.load(open(run_file, encoding="utf-8"))
tops = {}
for rec in run["per_query"]:
    top = rec.get("relevance")
    if rec["id"] in labels and top is not None:
        tops[rec["id"]] = top

ood = sorted(t for i, t in tops.items() if labels[i])
inc = sorted(t for i, t in tops.items() if not labels[i])


def stats(name, vals):
    if not vals:
        return f"{name}: none"
    mid = vals[len(vals) // 2]
    return (f"{name}: n={len(vals)} min={vals[0]:.3f} "
            f"p25={vals[len(vals)//4]:.3f} med={mid:.3f} max={vals[-1]:.3f}")


print(stats("out-of-corpus ", ood))
print(stats("in-corpus     ", inc))
print()
print("floor  caught-ood  wrongly-silenced-good-queries")
for floor in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
    caught = sum(1 for t in ood if t < floor)
    wrong = sum(1 for t in inc if t < floor)
    print(f"{floor:5.2f}  {caught:3d}/{len(ood):<3d}          {wrong:2d}/{len(inc)}")
print()
print("lowest in-corpus tops (these would be silenced first):",
      [round(t, 3) for t in inc[:6]])
print("lowest out-of-corpus tops:", [round(t, 3) for t in ood[:6]])
