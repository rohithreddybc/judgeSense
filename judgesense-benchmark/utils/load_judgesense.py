"""
load_judgesense.py — minimal JSONL loader for the JudgeSense benchmark.

Usage:
    from utils.load_judgesense import load_task, load_all
    pairs = load_task("factuality")
    all_data = load_all()
"""

import json
from pathlib import Path

TASKS = ["factuality", "coherence", "preference", "relevance"]


def load_task(task: str, data_dir: str | Path = "data") -> list[dict]:
    """Load a single JudgeSense task file.

    Args:
        task: One of 'factuality', 'coherence', 'preference', 'relevance'.
        data_dir: Path to the data/ directory (default: 'data').

    Returns:
        List of record dicts with keys: pair_id, task_type, source_benchmark,
        prompt_a, prompt_b, response_being_judged, ground_truth_label,
        semantic_equivalence_score.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {TASKS}")
    path = Path(data_dir) / f"{task}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all(data_dir: str | Path = "data") -> dict[str, list[dict]]:
    """Load all four task files.

    Returns:
        Dict keyed by task name, each value is a list of record dicts.
    """
    return {task: load_task(task, data_dir) for task in TASKS}


if __name__ == "__main__":
    all_data = load_all()
    total = sum(len(v) for v in all_data.values())
    print(f"Loaded {total} total records across {len(all_data)} tasks:")
    for task, records in all_data.items():
        print(f"  {task}: {len(records)} pairs")
