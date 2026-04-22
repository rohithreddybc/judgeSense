"""
Backfill missing fields in existing gpt-4o-mini result JSONL files.

Adds: normalized_a, normalized_b, flipped, error, run
from existing: prompt_a_decision, prompt_b_decision, run_number

Safe to rerun — only writes files with changes.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "raw_outputs"

FIELDS_TO_ADD = {"normalized_a", "normalized_b", "flipped", "error", "run"}


def backfill_file(path: Path) -> int:
    """Backfill one JSONL file. Returns number of records updated."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    updated = 0
    for rec in records:
        missing = FIELDS_TO_ADD - rec.keys()
        if not missing:
            continue

        if "normalized_a" in missing:
            rec["normalized_a"] = rec.get("prompt_a_decision", "UNCLEAR")
        if "normalized_b" in missing:
            rec["normalized_b"] = rec.get("prompt_b_decision", "UNCLEAR")
        if "flipped" in missing:
            rec["flipped"] = rec.get("normalized_a") != rec.get("normalized_b")
        if "error" in missing:
            raw_a = rec.get("prompt_a_decision_raw", "")
            raw_b = rec.get("prompt_b_decision_raw", "")
            err_a = raw_a.startswith("ERROR:") if isinstance(raw_a, str) else False
            err_b = raw_b.startswith("ERROR:") if isinstance(raw_b, str) else False
            rec["error"] = (raw_a if err_a else raw_b) if (err_a or err_b) else None
        if "run" in missing:
            rec["run"] = rec.get("run_number", 1)

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
