"""
Recompute JSS on the common support: items EVERY compared judge answered on both
arms.

Conditioning JSS on verdict pairs (see PREREGISTRATION.md) restricts a refusing
judge to the items it agreed to judge, which invites the obvious objection that
the judges are then being compared on different subsets. This script answers it
with a measurement instead of an argument: intersect the answered items across
judges and recompute, so every judge in a task is scored on identical items.

Reports, per task: the common-support size, each judge's JSS on it, and the
shift from its full-support JSS. A judge whose ranking depends on which subset
it is scored over is exactly what a reviewer would want flagged.

Usage:
    python scripts/common_support.py
    python scripts/common_support.py --judges claude-haiku claude-sonnet

The pair-class comparison uses the module constant rather than a string
literal. This script compared against "both_verdict" after that class was
renamed to "both_answered", so the intersection was empty and every
common-support figure came back null -- the same defect that silently voided
the Manski bounds, in a second file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "results_v2" / "raw"
OUT = REPO / "data" / "results_v2" / "common_support.json"


def _regen():
    spec = importlib.util.spec_from_file_location(
        "regen", REPO / "scripts" / "regenerate_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", nargs="*", default=None,
                    help="restrict to these judges (default: every judge present)")
    ap.add_argument("--raw", default=str(RAW))
    args = ap.parse_args(argv)

    regen = _regen()
    raw = Path(args.raw)
    files = sorted(raw.glob("*_*.jsonl"))
    if not files:
        print(f"No raw outputs in {raw}. Run src/run_v2.py first.")
        return 1

    by_task: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)
    for f in files:
        judge, task = f.stem.rsplit("_", 1)
        if args.judges and judge not in args.judges:
            continue
        by_task[task][judge] = regen._records(f)

    report: Dict[str, dict] = {}
    for task, per_judge in sorted(by_task.items()):
        if len(per_judge) < 2:
            continue  # a common support needs at least two judges to be common
        answered = {
            judge: {r["item_id"] for r in recs if regen._pair_class(r) == regen.PAIR_BOTH_ANSWERED}
            for judge, recs in per_judge.items()
        }
        common = set.intersection(*answered.values())
        union = set.union(*answered.values())
        entry = {
            "n_common_items": len(common),
            "n_union_items": len(union),
            "judges": {},
        }
        for judge, recs in sorted(per_judge.items()):
            full = regen.metrics_for_cell(recs, task)
            subset = [r for r in recs if r["item_id"] in common]
            restricted = regen.metrics_for_cell(subset, task) if len(subset) >= 2 else None
            entry["judges"][judge] = {
                "n_answered": len(answered[judge]),
                "jss_full_support": full.get("jss_strict"),
                "jss_common_support": restricted.get("jss_strict") if restricted else None,
                "shift": (
                    round(restricted["jss_strict"] - full["jss_strict"], 4)
                    if restricted and restricted.get("jss_strict") is not None
                    and full.get("jss_strict") is not None else None
                ),
            }
        report[task] = entry

    if not report:
        print("Nothing to compare: a common support needs at least two judges per task.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    for task, entry in report.items():
        print(f"\n{task}: common support {entry['n_common_items']} of "
              f"{entry['n_union_items']} answered items")
        for judge, m in entry["judges"].items():
            print(f"  {judge:<20} answered={m['n_answered']:<5} "
                  f"full={m['jss_full_support']} common={m['jss_common_support']} "
                  f"shift={m['shift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
