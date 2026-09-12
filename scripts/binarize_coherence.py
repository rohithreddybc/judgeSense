"""Is coherence's large effect an artefact of its five-point scale?

REVIEWER WjHn W1

Coherence is the only ordinal task and it carries by far the largest effect. A
judge with five admissible answers has more ways to disagree with itself than a
judge with two, so the comparison against the binary tasks is confounded with
the size of the decision space. W1 asked for the control: collapse the scale to
binary and see whether the effect survives.

This collapses each rating to `high` (4 or 5) versus `low` (1, 2 or 3), which is
the split SummEval's own usable/not-usable reading implies, and recomputes the
endpoint on the collapsed labels. Both arms and both repeats are collapsed the
same way, so the comparison is like for like.

If the effect is an artefact of scale width, binarising should shrink it toward
the binary tasks' range. If it survives, the width of the decision space is not
the explanation.

    python scripts/binarize_coherence.py
"""
from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SNAP = REPO / "data" / "results_v2" / "_snapshot"
MULT = REPO / "data" / "results_v2" / "multiplicity.json"
OUT = REPO / "data" / "results_v2" / "binarized_coherence.json"

ARMS = ("decision_a", "decision_b", "decision_a_repeat")
HIGH = {"4", "5"}


def collapse(v):
    """1-5 -> low/high; anything outside the scale stays itself, so a
    malformed answer never silently becomes a valid one."""
    if v is None:
        return None
    s = str(v).strip()
    if s in {"1", "2", "3", "4", "5"}:
        return "high" if s in HIGH else "low"
    return s


def cell_delta(path, binarise):
    para_n = para_k = rep_n = rep_k = 0
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        a, b, ar = (r.get(k) for k in ARMS)
        if binarise:
            a, b, ar = collapse(a), collapse(b), collapse(ar)
        if a is not None and b is not None:
            para_n += 1
            para_k += (a == b)
        if a is not None and ar is not None:
            rep_n += 1
            rep_k += (a == ar)
    if not para_n or not rep_n:
        return None
    return para_k / para_n - rep_k / rep_n


def main() -> int:
    readable = {(c["judge"], c["task"])
                for c in json.load(io.open(MULT, encoding="utf-8"))
                ["readable_cells_bh"].values()}

    raw, binned, rows = [], [], []
    for p in sorted(SNAP.glob("*_coherence.jsonl")):
        judge = p.stem.rsplit("_", 1)[0]
        if (judge, "coherence") not in readable:
            continue
        d0, d1 = cell_delta(p, False), cell_delta(p, True)
        if d0 is None or d1 is None:
            continue
        raw.append(d0)
        binned.append(d1)
        rows.append({"judge": judge, "five_point": round(d0, 4),
                     "binarised": round(d1, 4),
                     "shrinkage": round(1 - abs(d1) / abs(d0), 3) if d0 else None})

    res = {
        "n_judges": len(rows),
        "split": "high = {4,5}, low = {1,2,3}",
        "mean_five_point": round(statistics.fmean(raw), 4),
        "mean_binarised": round(statistics.fmean(binned), 4),
        "mean_shrinkage": round(1 - abs(statistics.fmean(binned))
                                / abs(statistics.fmean(raw)), 3),
        "per_judge": rows,
    }
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))

    print(f"  coherence, {len(rows)} readable judges, {res['split']}")
    print(f"  mean dJSS five-point : {res['mean_five_point']:+.4f}")
    print(f"  mean dJSS binarised  : {res['mean_binarised']:+.4f}")
    print(f"  shrinkage            : {100 * res['mean_shrinkage']:.1f}%")
    print(f"  for comparison, the binary tasks pool at -0.057, -0.045, -0.034")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
