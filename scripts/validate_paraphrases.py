"""
Validate prompt pair semantic equivalence using GPT-4o-mini.

Usage — run 4 terminals in parallel for maximum speed (from judgeSense/):

    python scripts/validate_paraphrases.py --task factuality --workers 20
    python scripts/validate_paraphrases.py --task coherence  --workers 20
    python scripts/validate_paraphrases.py --task preference --workers 20
    python scripts/validate_paraphrases.py --task relevance  --workers 20

Then summarize all results:

    python scripts/validate_paraphrases.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set


# ── .env loading ───────────────────────────────────────────────────────────────

def _load_env():
    try:
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

_load_env()


# ── Constants ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR  = _REPO_ROOT / "data" / "prompt_pairs"
_VAL_DIR   = _REPO_ROOT / "data" / "validation"

_TASK_FILES = {
    "factuality": _DATA_DIR / "factuality.jsonl",
    "coherence":  _DATA_DIR / "coherence.jsonl",
    "relevance":  _DATA_DIR / "relevance.jsonl",
    "preference": _DATA_DIR / "preference.jsonl",
}

_VALIDATION_PROMPT = (
    "Are these two evaluation prompts semantically equivalent "
    "— do they ask a judge to do the same thing, just worded "
    "differently? Answer YES or NO only.\n\n"
    "Prompt A: {prompt_a}\n"
    "Prompt B: {prompt_b}"
)

_MODEL      = "gpt-4o-mini-2024-07-18"
_MAX_TOKENS = 5
_TEMP       = 0.0


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _completed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {
        rec["pair_id"]
        for rec in _load_jsonl(path)
        if rec.get("validation_decision") in ("YES", "NO")
    }


def _append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ── API call ──────────────────────────────────────────────────────────────────

def _call(client, pair: dict) -> dict:
    content = _VALIDATION_PROMPT.format(
        prompt_a=pair["prompt_a"],
        prompt_b=pair["prompt_b"],
    )
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=_TEMP,
                max_tokens=_MAX_TOKENS,
                timeout=20,
            )
            raw = resp.choices[0].message.content.strip().upper()
            decision = "YES" if "YES" in raw else "NO" if "NO" in raw else "UNCLEAR"
            return {
                "pair_id":             pair["pair_id"],
                "task_type":           pair.get("task_type", "unknown"),
                "validation_decision": decision,
                "raw_response":        raw,
                "error":               None,
            }
        except Exception as exc:
            if attempt == 0:
                time.sleep(3)
            else:
                return {
                    "pair_id":             pair["pair_id"],
                    "task_type":           pair.get("task_type", "unknown"),
                    "validation_decision": "ERROR",
                    "raw_response":        "",
                    "error":               str(exc),
                }


# ── Validation runner ─────────────────────────────────────────────────────────

def run_validation(task: str, workers: int = 20) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set in .env or environment")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] pip install openai")
        sys.exit(1)

    client      = OpenAI(api_key=api_key)
    input_path  = _TASK_FILES[task]
    output_path = _VAL_DIR / f"{task}_paraphrase.jsonl"

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return

    pairs = _load_jsonl(input_path)
    done  = _completed_ids(output_path)
    todo  = [p for p in pairs if p["pair_id"] not in done]

    print(f"[{task}] {len(pairs)} pairs total | {len(done)} already done | {len(todo)} to validate")
    if not todo:
        print(f"[{task}] Nothing to do.")
        return

    n_yes, n_no, n_err = 0, 0, 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_call, client, pair): pair for pair in todo}
        for i, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            _append_jsonl(rec, output_path)
            d = rec["validation_decision"]
            if d == "YES":   n_yes += 1
            elif d == "NO":  n_no  += 1
            else:            n_err += 1
            if i % 25 == 0 or i == len(todo):
                print(f"[{task}] {i}/{len(todo)}  YES={n_yes}  NO={n_no}  err={n_err}")

    print(f"[{task}] Done — YES={n_yes}, NO={n_no}, errors={n_err}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    all_records = []
    for task in _TASK_FILES:
        path = _VAL_DIR / f"{task}_paraphrase.jsonl"
        if path.exists():
            all_records.extend(_load_jsonl(path))

    if not all_records:
        print("No validation results found. Run --task first.")
        return

    valid = [r for r in all_records if r["validation_decision"] in ("YES", "NO")]
    yes_  = [r for r in valid if r["validation_decision"] == "YES"]
    no_   = [r for r in valid if r["validation_decision"] == "NO"]
    total = len(valid)

    sep = "=" * 62
    print(f"\n{sep}")
    print("Paraphrase Validation Summary")
    print(sep)
    print(f"Total pairs validated  : {total}")
    if total:
        print(f"Equivalent     (YES)   : {len(yes_):>4}  ({100*len(yes_)/total:.1f}%)")
        print(f"NOT equivalent (NO)    : {len(no_):>4}  ({100*len(no_)/total:.1f}%)")

    print(f"\n{'Task':<15} {'Validated':>10} {'YES':>6} {'NO':>6} {'YES%':>7}")
    print("-" * 45)
    for task in sorted(_TASK_FILES):
        recs  = [r for r in valid if r.get("task_type") == task]
        if not recs:
            print(f"{task:<15} {'(no data)':>10}")
            continue
        n_yes = sum(1 for r in recs if r["validation_decision"] == "YES")
        n_no  = len(recs) - n_yes
        print(f"{task:<15} {len(recs):>10} {n_yes:>6} {n_no:>6} {100*n_yes/len(recs):>6.1f}%")

    if no_:
        print(f"\nPairs flagged NOT equivalent ({len(no_)}):")
        for r in sorted(no_, key=lambda x: x["pair_id"]):
            print(f"  {r['pair_id']:<15}  ({r['task_type']})")
    else:
        print("\nNo pairs flagged as NOT equivalent.")

    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="validate_paraphrases",
        description="Validate prompt pair semantic equivalence via GPT-4o-mini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Fastest usage — paste in 4 separate terminals:\n"
            "  python scripts/validate_paraphrases.py --task factuality --workers 20\n"
            "  python scripts/validate_paraphrases.py --task coherence  --workers 20\n"
            "  python scripts/validate_paraphrases.py --task preference --workers 20\n"
            "  python scripts/validate_paraphrases.py --task relevance  --workers 20\n\n"
            "Then summarize:\n"
            "  python scripts/validate_paraphrases.py --summarize"
        ),
    )
    parser.add_argument(
        "--task",
        choices=list(_TASK_FILES.keys()) + ["all"],
        help="Task dataset to validate, or 'all' to run every task sequentially.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        metavar="N",
        help="Concurrent API calls per task (default: 20).",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print summary table from completed validation files.",
    )
    args = parser.parse_args(argv)

    if args.summarize:
        print_summary()
        return

    if not args.task:
        parser.print_help()
        return

    tasks = list(_TASK_FILES.keys()) if args.task == "all" else [args.task]
    for task in tasks:
        run_validation(task, workers=args.workers)

    if len(tasks) > 1:
        print_summary()


if __name__ == "__main__":
    main()
