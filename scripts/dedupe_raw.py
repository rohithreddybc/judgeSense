"""Compact raw judge output by collapsing repeated pair_ids.

This is HOUSEKEEPING, not a correctness fix, and the distinction matters.

Repeated pair_ids are normal and expected: the runner appends and never
rewrites, so an errored row that is later retried leaves both records behind.
regenerate_results._records already collapses them last-write-wins before any
metric is computed, and run_v2._completed_pair_ids decides resume the same way.
The reported numbers are therefore unaffected by duplicates, and running this
script is optional.

What duplicates do cost is calls and wall-clock, when they come from two
processes over one (judge, task) rather than from retries. Raw files are
append-only and unlocked, and the completed set is read once at cell start, so
two writers each work the whole backlog and both pay. That is what
run_v2._acquire_cell now prevents.

Use this to shrink files and make row counts read as item counts. It applies the
same last-write-wins rule as the reader, so it can never change a result.

Run:  python scripts/dedupe_raw.py [--apply]     (default is a dry run)
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "results_v2" / "raw"
ARMS = ("usage_a", "usage_b", "usage_a_repeat", "usage_b_repeat")


def dedupe_file(path: Path, apply: bool) -> tuple[int, int]:
    """Collapse repeated pair_ids, KEEPING THE LAST record written.

    Last-write-wins is not a choice made here; it is the rule the rest of the
    pipeline already applies. regenerate_results._records reads the same way,
    and run_v2._completed_pair_ids decides resume on the final record per
    pair_id. Anything else -- "keep the best-looking record", say -- would mark a
    row done that the runner still intends to retry, and would make this script
    disagree with the reader about which attempt counts.
    """
    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    last: dict[str, dict] = {}
    order: list[str] = []
    for rec in rows:
        pid = rec.get("pair_id")
        if pid not in last:
            order.append(pid)
        last[pid] = rec              # later record supersedes earlier
    best = last
    removed = len(rows) - len(order)
    if removed and apply:
        tmp = path.with_suffix(".jsonl.tmp")
        with io.open(tmp, "w", encoding="utf-8", newline="") as fh:
            for pid in order:
                fh.write(json.dumps(best[pid], ensure_ascii=False) + "\n")
        tmp.replace(path)
    return len(rows), removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite files; default only reports")
    args = ap.parse_args()

    total_removed = 0
    for path in sorted(RAW.glob("*.jsonl")):
        n, removed = dedupe_file(path, args.apply)
        if removed:
            total_removed += removed
            verb = "removed" if args.apply else "would remove"
            print(f"  {path.name:44} {n:5} rows, {verb} {removed}")
    if total_removed == 0:
        print("  no duplicate pair_ids found")
    elif not args.apply:
        print(f"\n  dry run: {total_removed} duplicate rows. Re-run with --apply.")
    else:
        print(f"\n  removed {total_removed} duplicate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
