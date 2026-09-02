"""One tick of the sweep: run every unfinished provider group, then exit.

WHY THIS EXISTS

Nothing launched from an interactive session survives in this environment.
`nohup`, PowerShell `Start-Process` and harness-managed background tasks were
all tried and all reaped -- silently, mid-cell, with empty stderr. What does
survive is the Windows Task Scheduler, which launches its own process outside
any session. So the sweep is driven by a scheduled task that calls this script
on a repeating interval.

Each tick is additive and safe to repeat. `run_v2` is resumable: raw output is
append-only and completed rows are skipped at cell start, so a tick that is
killed halfway costs time and nothing else. The per-cell lock means two ticks
overlapping refuse each other's cells rather than paying twice for them.

A pidfile guards against the scheduler starting a second tick while the first
is still working, which would otherwise pile up processes every interval.

    python scripts/sweep_tick.py                  # run one tick
    python scripts/sweep_tick.py --status         # report and exit
    python scripts/sweep_tick.py --max-minutes 25 # bound the tick
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "results_v2" / "raw"
LOG = REPO / "logs" / "sweep_tick.log"
PIDFILE = REPO / "logs" / "sweep_tick.pid"

ROWS_PER_JUDGE = 1260

# One group per provider connection, split further where a provider tolerates
# it. Groups are disjoint, so their cells never collide.
GROUPS = {
    "hf3":     ["gemma-4-31b", "qwen3-8b"],
    "hf4":     ["qwen3-14b", "qwen3-32b"],
    "hf5":     ["qwen3.8-27b-hf"],
    "mistral": ["mistral-small", "magistral-small", "mistral-medium"],
    "novita":  ["qwen", "deepseek-v4-flash"],
    "ds1":     ["qwen-3.6-flash", "qwen3.7-flash"],
    "ds2":     ["deepseek-v4-flash-ds", "glm-5.2"],
    "ds3":     ["kimi-k3", "deepseek-v4-pro"],
    "google":  ["gemini-flash", "gemini-3.7-flash"],
    "hf1":     ["llama3-8b", "llama-3.3-70b"],
    "hf2":     ["llama-4-scout", "llama-4-maverick"],
}


def rows_for(judge: str) -> int:
    """Count DISTINCT pair_ids, not lines.

    Raw output is append-only, so a row re-run after an error leaves the failed
    record in place and appends the good one. Counting lines therefore counts
    those superseded records too, and a cell can read as over-target while
    still missing items -- which is exactly what happened: kimi-k3 and
    qwen3.8-27b-hf both reported past 1260 with 39 preference items never run,
    and the sweep declared itself finished. Reading uniques matches how the
    results loader collapses the file (last write wins).
    """
    seen: set[tuple[str, str]] = set()
    for p in RAW.glob(f"{judge}_*.jsonl"):
        task = p.stem.rsplit("_", 1)[1]
        with io.open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add((task, str(json.loads(line)["pair_id"])))
                except (ValueError, KeyError):
                    continue
    return len(seen)


def unfinished() -> dict[str, list[str]]:
    out = {}
    for tag, judges in GROUPS.items():
        todo = [j for j in judges if rows_for(j) < ROWS_PER_JUDGE]
        if todo:
            out[tag] = todo
    return out


def _pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return True
    return str(pid) in out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=55,
                    help="bound the tick so it finishes before the next one "
                         "is due (default 55)")
    args = ap.parse_args()

    todo = unfinished()

    if args.status:
        total = sum(rows_for(j) for g in GROUPS.values() for j in g)
        target = ROWS_PER_JUDGE * sum(len(g) for g in GROUPS.values())
        for tag, judges in sorted(GROUPS.items()):
            marks = " ".join(f"{j}={rows_for(j)}" for j in judges)
            print(f"  {tag:9} {marks}")
        print(f"  TOTAL {total}/{target} ({100 * total / target:.1f}%)")
        print(f"  groups with work left: {sorted(todo)}")
        return 0

    if not todo:
        print("sweep complete; nothing to do")
        return 0

    if PIDFILE.exists():
        try:
            prev = int(PIDFILE.read_text().strip())
        except (ValueError, OSError):
            prev = -1
        if prev > 0 and prev != os.getpid() and _pid_alive(prev):
            print(f"tick {prev} is still running; skipping this interval")
            return 0

    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")

    stamp = dt.datetime.now().isoformat(timespec="seconds")
    before = {j: rows_for(j) for g in GROUPS.values() for j in g}

    procs = {}
    try:
        for tag, judges in todo.items():
            log = REPO / "logs" / "sweep" / f"{tag}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = io.open(log, "a", encoding="utf-8")
            procs[tag] = (subprocess.Popen(
                [sys.executable, "-u", "-m", "src.run_v2", "--judges", *judges,
                 "--budget-policy", "matched", "--repeat-baseline",
                 "--skip-preflight", "--yes"],
                cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
            ), fh)

        deadline = args.max_minutes * 60
        for tag, (proc, fh) in procs.items():
            try:
                proc.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        for tag, (proc, fh) in procs.items():
            if proc.poll() is None:
                proc.kill()
            try:
                fh.close()
            except Exception:
                pass
        subprocess.run([sys.executable, str(REPO / "scripts" / "clear_dead_locks.py"),
                        "--remove"], cwd=str(REPO), capture_output=True, text=True)
        try:
            PIDFILE.unlink()
        except OSError:
            pass

    gained = sum(rows_for(j) - before[j] for j in before)
    total = sum(rows_for(j) for j in before)
    target = ROWS_PER_JUDGE * len(before)
    line = (f"[{stamp}] groups={sorted(todo)} gained={gained} "
            f"total={total}/{target} ({100 * total / target:.1f}%)")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
