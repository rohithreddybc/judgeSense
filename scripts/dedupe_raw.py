"""Remove duplicate pair_ids from raw judge output.

Raw files are append-only and unlocked. Two processes over the same
(judge, task) each read the completed set once at cell start, so both work the
same backlog and both append -- the runner warns about this and nothing
enforced it. Restarting a provider's process while its predecessor was still
draining produced 73 duplicated rows across four cells.

Duplicates are not merely wasteful: metrics resample at the item, so an item
present twice is weighted twice, and a cell whose duplicates are concentrated in
one arm would shift the endpoint.

Resolution keeps ONE record per pair_id, preferring the one that actually
carries evidence: most arms answered, then fewest errored arms, then the later
timestamp. A successful retry therefore beats the errored attempt it replaced,
which is the outcome resume was trying to produce.

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


def _quality(rec: dict) -> tuple:
    answered = errored = 0
    for a in ARMS:
        u = rec.get(a)
        if isinstance(u, dict):
            if u.get("error"):
                errored += 1
            elif u.get("output_tokens") is not None:
                answered += 1
    decided = sum(1 for k in ("decision_a", "decision_b")
                  if rec.get(k) not in (None, "UNCLEAR"))
    return (answered, decided, -errored, str(rec.get("ts") or ""))


def dedupe_file(path: Path, apply: bool) -> tuple[int, int]:
    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    best: dict[str, dict] = {}
    order: list[str] = []
    for rec in rows:
        pid = rec.get("pair_id")
        if pid not in best:
            best[pid] = rec
            order.append(pid)
        elif _quality(rec) > _quality(best[pid]):
            best[pid] = rec
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
