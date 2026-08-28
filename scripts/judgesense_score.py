"""Score somebody else's judge on JudgeSense.

The benchmark is only worth as much as the number of people who can compute it
on their own judge. This takes a predictions file and returns JSS, the repeat
ceiling, and ΔJSS with a clustered interval -- no API keys, no provider clients,
no dependency on how the predictions were produced.

INPUT: JSON Lines, one object per benchmark row.

    {"pair_id": "fact_v2_0001",
     "decision_a": "YES", "decision_b": "NO",
     "decision_a_repeat": "YES", "decision_b_repeat": "NO"}

  pair_id            must match the shipped data/v2/*.jsonl
  decision_a/_b      the judge's answer to prompt_a / prompt_b
  *_repeat           OPTIONAL, and the reason to bother: without them only raw
                     JSS can be reported, and raw JSS cannot separate wording
                     sensitivity from a judge that simply disagrees with itself.
                     Issue each prompt a SECOND time, unchanged, to fill these.

Anything the task's label set does not contain is treated as malformed and
charged as disagreement, matching the paper. Nothing is coerced to a label.

    python scripts/judgesense_score.py preds.jsonl
    python scripts/judgesense_score.py preds.jsonl --task factuality --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from metrics_v2 import cluster_bootstrap_ci, jss  # noqa: E402
from multiplicity import (  # noqa: E402
    bootstrap_p_value, minimum_detectable_effect, practically_meaningful,
)
from structural_variants import LABEL_SETS, UNCLEAR  # noqa: E402

TASKS = ("factuality", "coherence", "relevance", "preference")
DATA = REPO / "data" / "v2"


def _load_benchmark() -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for task in TASKS:
        path = DATA / f"{task}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                row = json.loads(line)
                row["_task"] = task
                index[str(row["pair_id"])] = row
    return index


def _norm(value, task: str) -> str:
    """Map a prediction onto the task's label set, or UNCLEAR. Never coerces."""
    if value is None:
        return UNCLEAR
    text = str(value).strip().upper()
    for label in LABEL_SETS[task]:
        if text == str(label).upper():
            return str(label)
    return UNCLEAR


def score(pred_path: Path, only_task: Optional[str] = None,
          n_bootstrap: int = 2000) -> dict:
    bench = _load_benchmark()
    if not bench:
        raise SystemExit("no benchmark data found under data/v2/")

    per_task: Dict[str, List[dict]] = defaultdict(list)
    unknown = 0
    seen = set()
    for line in pred_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        pid = str(p.get("pair_id"))
        row = bench.get(pid)
        if row is None:
            unknown += 1
            continue
        if pid in seen:
            continue                        # last-write-wins would need a rule; be strict
        seen.add(pid)
        task = row["_task"]
        if only_task and task != only_task:
            continue
        rec = {
            "pair_id": pid,
            "item_id": row.get("item_id", pid),
            "prompt_pair_id": row.get("prompt_pair_id", pid),
            "decision_a": _norm(p.get("decision_a"), task),
            "decision_b": _norm(p.get("decision_b"), task),
        }
        if p.get("decision_a_repeat") is not None:
            rec["decision_a_repeat"] = _norm(p.get("decision_a_repeat"), task)
        if p.get("decision_b_repeat") is not None:
            rec["decision_b_repeat"] = _norm(p.get("decision_b_repeat"), task)
        per_task[task].append(rec)

    out: Dict[str, dict] = {"tasks": {}, "unknown_pair_ids": unknown}
    for task, recs in sorted(per_task.items()):
        expected = sum(1 for r in bench.values() if r["_task"] == task)
        para = cluster_bootstrap_ci(recs, jss, "item", n_bootstrap=n_bootstrap)

        # The ceiling uses BOTH arms' repeats where they are supplied, not just
        # arm A. A ceiling measured under one template cannot absorb the noise
        # the other template generates, so that noise would be charged to
        # paraphrasing and ΔJSS would overstate the effect.
        rep_recs = []
        for r in recs:
            if r.get("decision_a_repeat") is not None:
                rep_recs.append({**r, "decision_a": r["decision_a"],
                                 "decision_b": r["decision_a_repeat"]})
            if r.get("decision_b_repeat") is not None:
                rep_recs.append({**r, "decision_a": r["decision_b"],
                                 "decision_b": r["decision_b_repeat"]})
        entry = {
            "n_rows": len(recs),
            "n_expected": expected,
            "coverage": round(len(recs) / expected, 4) if expected else None,
            "jss": round(para["estimate"], 4),
            "jss_ci": [round(para["ci_lower"], 4), round(para["ci_upper"], 4)],
            "n_clusters": para["n_clusters"],
            "malformed_rate": round(
                sum(1 for r in recs
                    for k in ("decision_a", "decision_b") if r[k] == UNCLEAR)
                / max(1, 2 * len(recs)), 4),
        }
        if rep_recs:
            rep = cluster_bootstrap_ci(rep_recs, jss, "item", n_bootstrap=n_bootstrap)
            delta = para["estimate"] - rep["estimate"]
            entry.update({
                "jss_repeat": round(rep["estimate"], 4),
                "delta_jss": round(delta, 4),
                "practically_meaningful": practically_meaningful(delta),
            })
        else:
            entry["delta_jss"] = None
            entry["note"] = ("no repeat arms supplied; ΔJSS is undefined and raw "
                             "JSS cannot separate wording sensitivity from "
                             "decoding noise")
        out["tasks"][task] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score a judge on JudgeSense from a predictions file.")
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--task", choices=TASKS, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--json", type=Path, default=None, help="also write JSON here")
    args = ap.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"no such file: {args.predictions}")
    result = score(args.predictions, args.task, args.bootstrap)

    print(f"\n  {'task':12} {'n':>6} {'cov':>6} {'JSS':>7} {'ceiling':>8} "
          f"{'dJSS':>8}  {'95% CI':>18}  {'malformed':>9}")
    print("  " + "-" * 84)
    for task, e in result["tasks"].items():
        d = e.get("delta_jss")
        d_txt = f"{d:+.4f}" if d is not None else "    --  "
        ceil = f"{e['jss_repeat']:.4f}" if "jss_repeat" in e else "   --   "
        ci = f"[{e['jss_ci'][0]:.3f}, {e['jss_ci'][1]:.3f}]"
        print(f"  {task:12} {e['n_rows']:>6} {e['coverage'] or 0:>6.2f} "
              f"{e['jss']:>7.4f} {ceil:>8} {d_txt:>8}  {ci:>18}  "
              f"{e['malformed_rate']:>9.3f}")

    if result["unknown_pair_ids"]:
        print(f"\n  {result['unknown_pair_ids']} prediction(s) had a pair_id not in "
              f"the benchmark and were ignored.")
    missing = [t for t, e in result["tasks"].items() if e.get("delta_jss") is None]
    if missing:
        print(f"\n  No repeat arms for: {', '.join(missing)}.")
        print("  ΔJSS is the endpoint this benchmark is built around. Without a")
        print("  repeat arm you cannot tell a judge that is sensitive to wording")
        print("  from one that simply disagrees with itself. Re-issue each prompt")
        print("  once more, unchanged, and supply decision_a_repeat/_b_repeat.")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
