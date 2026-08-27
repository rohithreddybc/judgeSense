#!/usr/bin/env bash
# Progress of the multi-vendor sweep: rows landed per cell, against the target.
set -u
cd "$(dirname "$0")/.."
python - <<'PY'
import glob, io, os, collections

RAW = os.path.join("data", "results_v2", "raw")
TARGET = {"factuality": 250, "coherence": 250, "relevance": 500, "preference": 260}
CLAUDE = ("claude-haiku", "claude-sonnet", "claude-opus-4-7")

rows = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
    base = os.path.basename(f)[:-6]
    judge, _, task = base.rpartition("_")
    n = sum(1 for line in io.open(f, encoding="utf-8") if line.strip())
    rows[judge][task] = n

done = todo = 0
print(f"  {'judge':22} " + " ".join(f"{t[:5]:>11}" for t in TARGET) + "   total")
print("  " + "-" * 78)
for judge in sorted(rows):
    cells, tot, tgt = [], 0, 0
    for task, target in TARGET.items():
        n = rows[judge].get(task, 0)
        tot += n
        tgt += target
        mark = "ok" if n >= target else " "
        cells.append(f"{n:>5}/{target:<4}{mark}")
    pct = 100 * tot / tgt if tgt else 0
    done += tot
    todo += tgt
    tag = " (prior run)" if judge in CLAUDE else ""
    print(f"  {judge:22} " + " ".join(cells) + f"  {pct:5.1f}%{tag}")
print("  " + "-" * 78)
print(f"  overall rows: {done:,} / {todo:,}  ({100*done/todo:.1f}%)")

for pid in sorted(glob.glob(os.path.join("logs", "sweep", "*.pid"))):
    tag = os.path.basename(pid)[:-4]
    log = pid[:-4] + ".log"
    last = ""
    if os.path.exists(log):
        lines = [l.rstrip() for l in io.open(log, encoding="utf-8", errors="replace") if l.strip()]
        last = lines[-1][:88] if lines else ""
    print(f"  [{tag}] {last}")
PY
