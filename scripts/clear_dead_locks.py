"""Remove cell locks whose owning process is gone.

A cell lock is released in a `finally`, so an orderly exit leaves nothing
behind. A process killed outright does, and the lock then blocks that cell for
`_LOCK_STALE_SECONDS` even though nothing is writing it. Waiting the timeout out
is correct but slow, and a resume that starts inside the window loses the cell
for the whole run.

This checks the recorded pid rather than the age, so a lock held by a LIVE
worker is never removed no matter how long that worker has been on the cell.
Locks whose pid is still running are left strictly alone.

    python scripts/clear_dead_locks.py            # report only
    python scripts/clear_dead_locks.py --remove   # delete the dead ones
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCKS = REPO / "data" / "results_v2" / "raw" / ".locks"


def _alive(pid: int) -> bool:
    """Is this pid running? Errs toward True, so an unknown pid is left alone."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        except Exception:
            return True
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true",
                    help="delete the dead locks instead of only reporting them")
    args = ap.parse_args()

    if not LOCKS.exists():
        print("no lock directory; nothing to do")
        return 0

    dead, live, unreadable = [], [], []
    for path in sorted(LOCKS.glob("*.lock")):
        age = time.time() - path.stat().st_mtime
        try:
            pid = int(path.read_text().strip())
        except (ValueError, OSError):
            unreadable.append((path, age))
            continue
        (live if _alive(pid) else dead).append((path, pid, age))

    for path, pid, age in live:
        print(f"  LIVE  {path.name:44} pid {pid} ({age:.0f}s)")
    for path, age in unreadable:
        print(f"  ??    {path.name:44} unreadable pid ({age:.0f}s) -- left in place")
    for path, pid, age in dead:
        print(f"  DEAD  {path.name:44} pid {pid} ({age:.0f}s)")
        if args.remove:
            path.unlink()

    print(f"\n{len(live)} live, {len(dead)} dead, {len(unreadable)} unreadable")
    if dead and not args.remove:
        print("re-run with --remove to delete the dead ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
