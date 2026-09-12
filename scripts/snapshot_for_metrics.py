"""Build a deduplicated, judge-filtered snapshot of raw/ for the metrics pass.

TWO REASONS THIS EXISTS RATHER THAN RUNNING regenerate_results.py ON raw/.

Retired judges. The daily Groq job moves its three cells INTO raw/ while it
runs and back out afterwards, so for several hours a day raw/ holds partial
cells for judges that are not in the reported roster. regenerate_results globs
raw/ with no registry filter and would read those as finished judges.

Live writers. Those same files are being appended to by a running process.
Rewriting them in place -- which is what a dedupe does -- races that writer.

So the snapshot copies only the judges asked for, collapsing duplicate
pair_ids the way the results loader does (last write wins), and never writes
inside raw/. regenerate_results.py --raw <snapshot> then sees exactly the
roster intended and nothing else.

    python scripts/snapshot_for_metrics.py --out data/results_v2/_snapshot
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "results_v2" / "raw"
sys.path.insert(0, str(REPO / "src"))

from judge_registry import JUDGES, is_retired  # noqa: E402


def reported_judges() -> list[str]:
    """Every verified, non-retired judge -- the roster the paper reports."""
    return sorted(n for n, s in JUDGES.items()
                  if s["verified"] and not is_retired(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "results_v2" / "_snapshot"))
    ap.add_argument("--judges", nargs="*", default=None)
    args = ap.parse_args()

    keep = set(args.judges) if args.judges else set(reported_judges())
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    kept = skipped = collapsed = 0
    for src in sorted(RAW.glob("*_*.jsonl")):
        judge = src.stem.rsplit("_", 1)[0]
        if judge not in keep:
            skipped += 1
            continue
        rows: "OrderedDict[str, str]" = OrderedDict()
        dupes = 0
        with io.open(src, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    pid = str(json.loads(line)["pair_id"])
                except (ValueError, KeyError):
                    continue
                if pid in rows:
                    dupes += 1
                rows[pid] = line          # last write wins, as the loader does
        collapsed += dupes
        with io.open(out / src.name, "w", encoding="utf-8", newline="\n") as fh:
            for line in rows.values():
                fh.write(line + "\n")
        kept += 1

    print(f"  snapshot: {out}")
    print(f"  cells kept {kept}, cells skipped {skipped} (retired or unlisted)")
    print(f"  duplicate rows collapsed: {collapsed}")
    print(f"  judges: {len(keep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
