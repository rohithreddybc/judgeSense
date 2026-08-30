"""Spend one day's Groq free-tier quota on the three retired Groq judges.

Groq caps the free tier at 200,000 tokens per day PER MODEL. That is not enough
to finish a judge in one sitting, but it is enough to finish one in about a
week if every day's reset is actually consumed. This is the job that consumes
it: run it once a day and the three cells fill in on their own.

WHY A WRAPPER AND NOT `run_v2 --judges ...` DIRECTLY

`scripts/regenerate_results.py` globs `data/results_v2/raw/*_*.jsonl` with no
registry filter, so a partial Groq cell sitting in `raw/` would be read into the
metrics pipeline as though it were a finished judge. The rows therefore live in
`data/results_v2/retired_groq/`, outside the glob.

`run_v2` writes to `raw/` and has no output override, and its resume logic reads
completed rows from the file it is about to append to -- so the rows have to be
in `raw/` while it runs or every completed row would be paid for a second time.
This moves them in, runs, and moves them back out. The move-back is in a
`finally`, so an interrupted run still leaves `raw/` clean.

    python scripts/groq_daily.py            # run one day's quota
    python scripts/groq_daily.py --status   # report progress, run nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "results_v2" / "raw"
PARK = REPO / "data" / "results_v2" / "retired_groq"
LOG = REPO / "logs" / "groq_daily.log"

JUDGES = ("gpt-oss-20b", "gpt-oss-120b", "qwen3.8-27b")
ROWS_PER_JUDGE = 1260


def _cells(directory: Path) -> list[Path]:
    return [p for j in JUDGES for p in sorted(directory.glob(f"{j}_*.jsonl"))]


def _rows(directory: Path) -> dict[str, int]:
    counts = {j: 0 for j in JUDGES}
    for p in _cells(directory):
        judge = p.name.rsplit("_", 1)[0]
        with io.open(p, encoding="utf-8") as fh:
            counts[judge] += sum(1 for line in fh if line.strip())
    return counts


def _report(counts: dict[str, int]) -> str:
    parts = [f"{j} {counts[j]}/{ROWS_PER_JUDGE}" for j in JUDGES]
    done = sum(counts.values())
    return f"{' | '.join(parts)}  total {done}/{ROWS_PER_JUDGE * len(JUDGES)}"


def _move(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in _cells(src):
        target = dst / p.name
        if target.exists():
            # Should not happen; refuse rather than clobber paid rows.
            raise SystemExit(f"refusing to overwrite existing {target}")
        shutil.move(str(p), str(target))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true",
                    help="report progress and exit without running anything")
    ap.add_argument("--max-minutes", type=int, default=180,
                    help="stop the run after this long (default 180). Once the "
                         "day's quota is gone run_v2 does not stop: it walks "
                         "the rest of the backlog recording 429s, which would "
                         "run past the next day's job. Killing it is safe -- "
                         "the runner is resumable and errored rows are re-run.")
    args = ap.parse_args()

    before = _rows(PARK)
    if args.status:
        print(_report(before))
        return 0

    if all(before[j] >= ROWS_PER_JUDGE for j in JUDGES):
        print("all three Groq cells are complete; nothing to do")
        return 0

    stamp = dt.datetime.now().isoformat(timespec="seconds")
    _move(PARK, RAW)
    rc, tail = None, []
    try:
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-m", "src.run_v2",
                 "--judges", *JUDGES,
                 "--budget-policy", "matched", "--repeat-baseline",
                 "--skip-preflight", "--yes"],
                cwd=str(REPO), capture_output=True, text=True,
                timeout=args.max_minutes * 60,
            )
            rc = proc.returncode
            tail = (proc.stdout or "").strip().splitlines()[-12:]
        except subprocess.TimeoutExpired as exc:
            rc = "timeout"
            out = exc.stdout or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            tail = out.strip().splitlines()[-12:]
    finally:
        after_raw = _rows(RAW)
        _move(RAW, PARK)
        # A killed child leaves its cell lock behind; the next run would refuse
        # that cell for _LOCK_STALE_SECONDS. Clear the ones whose owner is gone.
        subprocess.run([sys.executable, str(REPO / "scripts" / "clear_dead_locks.py"),
                        "--remove"], cwd=str(REPO), capture_output=True, text=True)

    gained = {j: after_raw[j] - before[j] for j in JUDGES}
    line = (f"[{stamp}] exit={rc} "
            f"gained {sum(gained.values())} rows "
            f"({', '.join(f'{j} +{gained[j]}' for j in JUDGES)}) :: "
            f"{_report(after_raw)}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    print(line)
    for t in tail:
        print("   ", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
