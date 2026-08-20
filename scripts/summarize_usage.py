"""
Aggregate token spend from a v2 run into data/results_v2/usage.json.

Reads EVERY line of the raw outputs, deliberately without the last-write-wins
dedup that `regenerate_results.py` applies. The two readers answer different
questions and must differ:

  regenerate_results  -> "what did the judge decide?"  -> one record per pair
  summarize_usage     -> "what did this cost?"          -> every attempt counts

A row that errored and was retried was PAID FOR TWICE, so the superseded record
is real spend and is summed here even though it is discarded for scoring.

Cost policy: prices are not in this repository and are never invented. Supply a
price table with --prices to get cost; without one, cost is null. Where any call
in a group returned no usage, the group's cost is marked `lower_bound: true`,
because a partial token sum can only understate spend.

Price file format (per million tokens):
    {"gpt-4o": {"input_per_mtok": 2.50, "output_per_mtok": 10.00}, ...}

Usage:
    python scripts/summarize_usage.py
    python scripts/summarize_usage.py --prices prices.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "results_v2" / "raw"
OUT = REPO / "data" / "results_v2" / "usage.json"
ARMS = ("usage_a", "usage_b", "usage_a_repeat")


def _blank() -> Dict:
    return {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "calls_missing_usage": 0, "retries": 0, "errors": 0,
        "wall_clock_ms": 0, "rows": 0, "superseded_records": 0,
    }


def _fold(acc: Dict, usage: Optional[dict]) -> None:
    if usage is None:
        # The call happened (the arm exists) but the provider reported nothing,
        # or the seam was stubbed. Counted as a call with unknown tokens rather
        # than dropped, so the summary shows how partial it is.
        acc["calls"] += 1
        acc["calls_missing_usage"] += 1
        return
    acc["calls"] += 1
    ti, to = usage.get("input_tokens"), usage.get("output_tokens")
    if ti is None and to is None:
        acc["calls_missing_usage"] += 1
    acc["input_tokens"] += ti or 0
    acc["output_tokens"] += to or 0
    acc["retries"] += max(0, int(usage.get("attempts") or 1) - 1)
    acc["wall_clock_ms"] += int(usage.get("latency_ms") or 0)
    if usage.get("error"):
        acc["errors"] += 1


def _cost(group: Dict, price: Optional[dict]) -> Optional[dict]:
    if not price:
        return None
    inp = price.get("input_per_mtok")
    out = price.get("output_per_mtok")
    if inp is None or out is None:
        return None
    usd = group["input_tokens"] / 1e6 * inp + group["output_tokens"] / 1e6 * out
    return {
        "usd": round(usd, 4),
        "lower_bound": group["calls_missing_usage"] > 0,
        "basis": "known tokens only; excludes calls that returned no usage",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise token spend from a v2 run.")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--prices", default=None,
                    help="JSON: {judge: {input_per_mtok, output_per_mtok}}. "
                         "Without it, cost is reported as null rather than guessed.")
    args = ap.parse_args(argv)

    prices = json.loads(Path(args.prices).read_text(encoding="utf-8")) if args.prices else {}
    raw = Path(args.raw)
    files = sorted(raw.glob("*_*.jsonl"))
    if not files:
        print(f"No raw outputs in {raw}. Run src/run_v2.py first.")
        return 1

    per_cell: Dict[tuple, Dict] = defaultdict(_blank)
    per_judge: Dict[str, Dict] = defaultdict(_blank)
    overall = _blank()

    for f in files:
        judge, task = f.stem.rsplit("_", 1)
        seen = set()
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a killed run
            for acc in (per_cell[(judge, task)], per_judge[judge], overall):
                acc["rows"] += 1
                if rec.get("pair_id") in seen:
                    acc["superseded_records"] += 1
                for arm in ARMS:
                    if arm in rec:
                        _fold(acc, rec.get(arm))
            seen.add(rec.get("pair_id"))

    summary = {
        "note": ("every attempt is counted, including superseded records from "
                 "retried rows, because those calls were paid for. Scoring uses "
                 "last-write-wins and will show fewer records."),
        "overall": {**overall, "cost": None},
        "per_judge": {},
        "per_cell": {},
    }
    for judge, g in sorted(per_judge.items()):
        summary["per_judge"][judge] = {**g, "cost": _cost(g, prices.get(judge))}
    for (judge, task), g in sorted(per_cell.items()):
        summary["per_cell"][f"{judge}/{task}"] = {**g, "cost": _cost(g, prices.get(judge))}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    o = overall
    print(f"Wrote {OUT}")
    print(f"  calls={o['calls']}  in={o['input_tokens']}  out={o['output_tokens']}  "
          f"missing_usage={o['calls_missing_usage']}  retries={o['retries']}  "
          f"errors={o['errors']}  superseded={o['superseded_records']}")
    if not prices:
        print("  cost: null (no --prices supplied; costs are never estimated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
