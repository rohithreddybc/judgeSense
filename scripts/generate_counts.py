"""
Machine-generated source of truth for every reported count (reviewer qkzU).

Reads the data files actually on disk plus the code registries
(SUPPORTED_MODELS, EXCLUDED_PAIRS) and writes data/counts.json. Documents
must consume that file instead of restating numbers by hand — the 9-vs-13
judges and 494-vs-500 pairs drift between the paper body and the checklist
is exactly the failure mode this removes (500 rows - 6 excluded pairs = 494;
9 judges was the pass-1 roster, 13 is the current registry).

Usage:
    python scripts/generate_counts.py [--output data/counts.json]

CI re-runs this script and fails if the committed file is stale.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.metrics import EXCLUDED_PAIRS  # noqa: E402
from src.models import SUPPORTED_MODELS  # noqa: E402

TASKS = ("factuality", "coherence", "relevance", "preference")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dataset_counts(data_dir: Path) -> dict:
    out = {}
    for task in TASKS:
        path = data_dir / f"{task}.jsonl"
        if not path.exists():
            out[task] = None
            continue
        records = load_jsonl(path)
        out[task] = {
            "rows": len(records),
            "unique_items": len({r.get("item_id", r.get("response_being_judged"))
                                 for r in records}),
            "unique_prompt_pairs": len({
                r.get("prompt_pair_id", (r.get("prompt_a"), r.get("prompt_b")))
                for r in records
            }),
            "unique_prompt_texts": len({p for r in records
                                        for p in (r.get("prompt_a"), r.get("prompt_b"))}),
            "label_histogram": _histogram(records, "ground_truth_label"),
            "excluded_pairs": sorted(
                r["pair_id"] for r in records if r.get("pair_id") in EXCLUDED_PAIRS
            ),
        }
    present = [c for c in out.values() if c]
    out["_totals"] = {
        "rows": sum(c["rows"] for c in present),
        "unique_items": sum(c["unique_items"] for c in present),
        "unique_prompt_pairs": sum(c["unique_prompt_pairs"] for c in present),
        "rows_after_exclusions": sum(c["rows"] for c in present) - len(EXCLUDED_PAIRS),
        "n_excluded_pairs": len(EXCLUDED_PAIRS),
    }
    return out


def _histogram(records: list[dict], key: str) -> dict:
    hist: dict = {}
    for rec in records:
        label = str(rec.get(key))
        hist[label] = hist.get(label, 0) + 1
    return dict(sorted(hist.items()))


def validation_counts(validation_dir: Path) -> dict:
    out = {}
    manual_dir = validation_dir / "manual"
    for path in sorted(manual_dir.glob("*.jsonl")) if manual_dir.exists() else []:
        records = load_jsonl(path)
        stamps = sorted(datetime.fromisoformat(r["timestamp"])
                        for r in records if r.get("timestamp"))
        gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        out[path.name] = {
            "decisions": len(records),
            "label_histogram": _histogram(records, "manual_label"),
            "median_seconds_between_decisions":
                round(statistics.median(gaps), 3) if gaps else None,
        }
    for path in sorted(validation_dir.glob("*_paraphrase.jsonl")) if validation_dir.exists() else []:
        records = load_jsonl(path)
        out[path.name] = {
            "decisions": len(records),
            "decision_histogram": _histogram(records, "validation_decision"),
        }
    return out


def evaluation_design_counts(dataset: dict) ->  dict:
    n_judges = len(SUPPORTED_MODELS)
    rows = dataset["_totals"]["rows"]
    rows_after = dataset["_totals"]["rows_after_exclusions"]
    per_run_calls = rows * 2  # two prompt variants per row
    return {
        "judges_in_registry": n_judges,
        "judge_names": sorted(SUPPORTED_MODELS.keys()),
        "runs_per_task_reported": 3,
        "api_calls_per_run_all_rows": per_run_calls,
        "api_calls_3_runs_all_rows_1_judge": per_run_calls * 3,
        "api_calls_3_runs_all_rows_all_judges": per_run_calls * 3 * n_judges,
        "rows_used_in_metrics_after_exclusions": rows_after,
        "note": (
            "The 494-vs-500 discrepancy is rows minus the "
            f"{dataset['_totals']['n_excluded_pairs']} pairs in "
            f"src.metrics.EXCLUDED_PAIRS. judges_in_registry counts "
            f"src.models.SUPPORTED_MODELS entries as committed ({n_judges}); "
            "documents citing a different judge count (9 or 13) must "
            "reconcile against this registry, which is the only "
            "machine-checkable roster in the repository."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data/counts.json")
    parser.add_argument("--output", default="data/counts.json")
    args = parser.parse_args()

    dataset = dataset_counts(REPO / "data" / "prompt_pairs")
    counts = {
        "generated_by": "scripts/generate_counts.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_v1": dataset,
        "dataset_v2": (
            dataset_counts(REPO / "data" / "v2")
            if (REPO / "data" / "v2").exists() else
            "not built (data/v2/ absent); build with src/dataset_builder_v2.py"
        ),
        "human_validation": validation_counts(REPO / "data" / "validation"),
        "evaluation_design": evaluation_design_counts(dataset),
    }

    out_path = REPO / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts["dataset_v1"]["_totals"], indent=2))
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
