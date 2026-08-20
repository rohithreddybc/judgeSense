"""
Freeze / verify the dataset the judge run will consume.

The builder is deterministic in CONTENT but not byte-identical across runs: each
record's `source.retrieved_at` records when the item was fetched, so two builds
of identical data differ in that field alone. A raw file checksum would therefore
always mismatch and be useless as a guarantee.

This computes a CONTENT checksum instead — every field except `retrieved_at`,
canonicalised and hashed in row order. Two builds of the same data produce the
same content hash; any change to an item, label, prompt, ordering, or provenance
id changes it.

Usage:
    python scripts/freeze_dataset.py --write     # record data/v2/FROZEN.json
    python scripts/freeze_dataset.py             # verify against the record

Run `--write` once the dataset is final, and plain `verify` immediately before
spending a judge run: it proves the run's inputs are the audited ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "v2"
FROZEN = DATA / "FROZEN.json"
TASKS = ("factuality", "coherence", "relevance", "preference")
VOLATILE = ("retrieved_at",)


def content_hash(path: Path) -> Dict[str, object]:
    """SHA256 over canonicalised rows, excluding volatile provenance fields."""
    h = hashlib.sha256()
    rows = 0
    items = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        src = rec.get("source")
        if isinstance(src, dict):
            rec = {**rec, "source": {k: v for k, v in src.items() if k not in VOLATILE}}
        h.update(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
        rows += 1
        if rec.get("item_id"):
            items.add(rec["item_id"])
    return {"sha256": h.hexdigest(), "rows": rows, "unique_items": len(items)}


def compute() -> Dict[str, Dict[str, object]]:
    out = {}
    for task in TASKS:
        p = DATA / f"{task}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing split: {p}")
        out[task] = content_hash(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze/verify the v2 dataset content.")
    ap.add_argument("--write", action="store_true", help="record the current content hashes")
    args = ap.parse_args(argv)

    current = compute()

    if args.write:
        FROZEN.write_text(json.dumps(
            {"note": "content hashes exclude source.retrieved_at, which is volatile by design",
             "splits": current}, indent=2), encoding="utf-8")
        print(f"Wrote {FROZEN}")
        for t, v in current.items():
            print(f"  {t:<12} {v['sha256'][:16]}  rows={v['rows']:<4} items={v['unique_items']}")
        return 0

    if not FROZEN.exists():
        print(f"No frozen record at {FROZEN}. Run with --write once the dataset is final.")
        return 1

    recorded = json.loads(FROZEN.read_text(encoding="utf-8"))["splits"]
    ok = True
    for task in TASKS:
        cur, rec = current[task], recorded.get(task, {})
        match = cur["sha256"] == rec.get("sha256")
        ok &= match
        print(f"  {'[ ok ]' if match else '[FAIL]'} {task:<12} rows={cur['rows']:<4} "
              f"items={cur['unique_items']:<4} {cur['sha256'][:16]}"
              + ("" if match else f"  != recorded {str(rec.get('sha256'))[:16]}"))
    print("\nDataset matches the frozen record." if ok else
          "\nDATASET DIFFERS from the frozen record — do NOT start a paid run until this is explained.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
