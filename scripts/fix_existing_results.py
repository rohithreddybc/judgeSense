"""
Backfill and re-normalize existing result JSONL files.

Handles both old-schema records (prompt_a_decision_raw / run_number)
and new-schema records (prompt_a_raw / run).

Fixes:
  - Maps prompt_a_decision_raw -> prompt_a_raw (old schema only)
  - Re-applies normalize_decision() from the actual raw text
  - Adds: normalized_a, normalized_b, flipped, error, run

Safe to rerun — idempotent.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.models import normalize_decision

RESULTS_DIR = REPO_ROOT / "data" / "results" / "raw_outputs"


def backfill_file(path: Path) -> int:
    """Backfill and re-normalize one JSONL file. Returns number of records updated."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    updated = 0
    for rec in records:
        task_type = rec.get("task_type", "")
        changed = False

        # ── Migrate old raw field names to new schema ──────────────────────
        if "prompt_a_raw" not in rec or not rec.get("prompt_a_raw"):
            rec["prompt_a_raw"] = rec.get("prompt_a_decision_raw", "")
            changed = True
        if "prompt_b_raw" not in rec or not rec.get("prompt_b_raw"):
            rec["prompt_b_raw"] = rec.get("prompt_b_decision_raw", "")
            changed = True

        # ── Migrate run field ──────────────────────────────────────────────
        if "run" not in rec:
            rec["run"] = rec.get("run_number", 1)
            changed = True

        # ── Re-normalize from actual raw text ──────────────────────────────
        raw_a = rec.get("prompt_a_raw", "")
        raw_b = rec.get("prompt_b_raw", "")
        err_a = isinstance(raw_a, str) and raw_a.startswith("ERROR:")
        err_b = isinstance(raw_b, str) and raw_b.startswith("ERROR:")

        new_a = "UNCLEAR" if err_a else normalize_decision(raw_a, task_type)
        new_b = "UNCLEAR" if err_b else normalize_decision(raw_b, task_type)
        new_flip = new_a != new_b
        new_err = (raw_a if err_a else raw_b) if (err_a or err_b) else None

        if (rec.get("normalized_a") != new_a or rec.get("normalized_b") != new_b
                or rec.get("flipped") != new_flip or rec.get("error") != new_err):
            rec["normalized_a"]      = new_a
            rec["normalized_b"]      = new_b
            rec["prompt_a_decision"] = new_a
            rec["prompt_b_decision"] = new_b
            rec["flipped"]           = new_flip
            rec["error"]             = new_err
            changed = True

        if changed:
            updated += 1

    if updated > 0:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    return updated


def main():
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    total_updated = 0
    for f in files:
        n = backfill_file(f)
        status = f"{n} records updated" if n > 0 else "already up-to-date"
        print(f"  {f.name}: {status}")
        total_updated += n

    print(f"\nDone. {total_updated} records backfilled across {len(files)} files.")


if __name__ == "__main__":
    main()
