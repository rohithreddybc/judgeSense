"""
Manual paraphrase-equivalence review CLI.

For each pair in data/prompt_pairs/{task}.jsonl, shows prompt_a and prompt_b
side-by-side and asks the human reviewer for a YES / NO / UNSURE label and an
optional one-line note. Progress is checkpointed to
data/validation/manual/{task}_manual.jsonl after every decision so the user
can stop and resume at any time (Ctrl+C is safe).

Output format (per record, JSONL):
    {
      "pair_id": "fact_001",
      "task_type": "factuality",
      "manual_label": "YES" | "NO" | "UNSURE",
      "note": "free-text",
      "reviewer": "rohit",
      "timestamp": "2026-04-30T..."
    }

Usage:
    python scripts/manual_review.py --task factuality
    python scripts/manual_review.py --task all
    python scripts/manual_review.py --task coherence --reviewer rohit
    python scripts/manual_review.py --summarize           # agreement vs gpt-4o-mini

Criteria (record these in the paper appendix verbatim):
    YES   = both prompts ask the judge to make the same evaluative decision
            on the same item; differences are limited to surface phrasing.
            Polarity-inverted templates (e.g. "does this contain errors?") count
            as YES if the underlying decision is the same once labels are remapped.
    NO    = the prompts ask for different evaluative decisions, or the same
            prompt would defensibly produce different correct answers.
    UNSURE = borderline; flag for adjudication. Treat as NO for headline JSS
             but report the count separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_PAIRS_DIR  = _REPO_ROOT / "data" / "prompt_pairs"
_MANUAL_DIR = _REPO_ROOT / "data" / "validation" / "manual"
_AUTO_DIR   = _REPO_ROOT / "data" / "validation"

_TASKS = ["factuality", "coherence", "relevance", "preference"]


def _load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _completed_ids(path: Path) -> Set[str]:
    return {rec["pair_id"] for rec in _load_jsonl(path) if "pair_id" in rec}


def _prompt_label() -> Optional[str]:
    """Read y/n/u/q from the user. Returns 'YES'/'NO'/'UNSURE' or None to quit."""
    while True:
        try:
            ans = input("  Equivalent? [y]es / [n]o / [u]nsure / [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if ans in ("y", "yes"):
            return "YES"
        if ans in ("n", "no"):
            return "NO"
        if ans in ("u", "unsure"):
            return "UNSURE"
        if ans in ("q", "quit", "exit"):
            return None
        print("  (please enter y, n, u, or q)")


def review_task(task: str, reviewer: str) -> None:
    pairs_path  = _PAIRS_DIR / f"{task}.jsonl"
    output_path = _MANUAL_DIR / f"{task}_manual.jsonl"

    if not pairs_path.exists():
        print(f"[ERROR] No pairs file at {pairs_path}", file=sys.stderr)
        return

    pairs = _load_jsonl(pairs_path)
    done  = _completed_ids(output_path)
    todo  = [p for p in pairs if p.get("pair_id") not in done]

    print(f"\n=== Task: {task} ({len(done)}/{len(pairs)} done, {len(todo)} remaining) ===")
    if not todo:
        print("All pairs already reviewed.")
        return

    for i, pair in enumerate(todo, 1):
        print("\n" + "=" * 78)
        print(f"  {pair['pair_id']}  ({i}/{len(todo)} remaining in this session)")
        print("=" * 78)
        print("\n  PROMPT A:")
        print("    " + pair["prompt_a"].replace("\n", "\n    "))
        print("\n  PROMPT B:")
        print("    " + pair["prompt_b"].replace("\n", "\n    "))
        print()

        label = _prompt_label()
        if label is None:
            print("\n[saved progress; exiting]")
            return

        try:
            note = input("  Note (optional, ENTER to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""
            print()

        record = {
            "pair_id":      pair["pair_id"],
            "task_type":    pair.get("task_type", task),
            "manual_label": label,
            "note":         note,
            "reviewer":     reviewer,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(record, output_path)


def summarize() -> None:
    """Compare manual labels against gpt-4o-mini classifier labels per task."""
    print("\n=== Manual review summary ===\n")
    grand_yes = grand_no = grand_unsure = grand_total = 0
    grand_agree = grand_disagree = grand_compared = 0

    for task in _TASKS:
        manual_path = _MANUAL_DIR / f"{task}_manual.jsonl"
        auto_path   = _AUTO_DIR / f"{task}_paraphrase.jsonl"

        manual = {r["pair_id"]: r for r in _load_jsonl(manual_path)}
        auto   = {r["pair_id"]: r for r in _load_jsonl(auto_path)}

        if not manual:
            print(f"  {task:11s}  (no manual labels yet)")
            continue

        n_yes    = sum(1 for r in manual.values() if r["manual_label"] == "YES")
        n_no     = sum(1 for r in manual.values() if r["manual_label"] == "NO")
        n_unsure = sum(1 for r in manual.values() if r["manual_label"] == "UNSURE")
        n_total  = len(manual)

        # Compare with classifier where both exist
        compared = agree = 0
        for pid, m_rec in manual.items():
            a_rec = auto.get(pid)
            if a_rec is None:
                continue
            # classifier output field name varies; look for common keys
            classifier_label = (
                a_rec.get("validation_decision")
                or a_rec.get("classifier_label")
                or a_rec.get("validation_label")
                or a_rec.get("label")
                or a_rec.get("equivalent")
                or ""
            )
            if isinstance(classifier_label, bool):
                classifier_label = "YES" if classifier_label else "NO"
            classifier_label = str(classifier_label).strip().upper()

            if classifier_label in ("YES", "NO"):
                compared += 1
                # treat manual UNSURE as NO for agreement scoring
                m_label = m_rec["manual_label"]
                m_norm  = "NO" if m_label == "UNSURE" else m_label
                if m_norm == classifier_label:
                    agree += 1

        agreement = (agree / compared * 100) if compared else 0.0
        print(
            f"  {task:11s}  YES={n_yes:3d}  NO={n_no:3d}  UNSURE={n_unsure:3d}  "
            f"(total {n_total:3d})  |  classifier-agreement {agree}/{compared} = {agreement:.1f}%"
        )

        grand_yes += n_yes
        grand_no += n_no
        grand_unsure += n_unsure
        grand_total += n_total
        grand_agree += agree
        grand_compared += compared

    print()
    print(
        f"  TOTAL        YES={grand_yes:3d}  NO={grand_no:3d}  UNSURE={grand_unsure:3d}  "
        f"(total {grand_total:3d})"
    )
    if grand_compared:
        print(
            f"  Overall classifier-vs-human agreement: "
            f"{grand_agree}/{grand_compared} = {grand_agree / grand_compared * 100:.1f}%"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="manual_review")
    parser.add_argument(
        "--task",
        choices=_TASKS + ["all"],
        help="Which task to review (or 'all').",
    )
    parser.add_argument(
        "--reviewer",
        default="rohit",
        help="Annotator name to record (default: rohit).",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print per-task counts and classifier-vs-human agreement; do not start a review.",
    )
    args = parser.parse_args(argv)

    if args.summarize:
        summarize()
        return

    if not args.task:
        parser.error("--task is required (or use --summarize)")

    tasks = _TASKS if args.task == "all" else [args.task]
    for task in tasks:
        review_task(task, args.reviewer)

    print("\nAll selected tasks done. Run with --summarize for stats.")


if __name__ == "__main__":
    main()
