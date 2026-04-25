"""
JudgeSense metrics — sensitivity and consistency statistics for LLM judges.

Implements the Judge Sensitivity Score (JSS) and related measures
as defined in the JudgeSense paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


# Pairs excluded from analysis (non-equivalent prompt variants identified by GPT-4o-mini)
EXCLUDED_PAIRS = {
    "fact_040", "fact_045", "fact_050",
    "fact_090", "fact_095", "fact_100",
}


def judge_sensitivity_score(
    decisions_a: Sequence[str],
    decisions_b: Sequence[str],
) -> float:
    """
    Judge Sensitivity Score (JSS): fraction of pairs where both prompt
    variants elicit the same decision from the judge.

    JSS = mean(decisions_a[i] == decisions_b[i])

    Args:
        decisions_a: Decisions from prompt variant A (list of strings).
        decisions_b: Decisions from prompt variant B (list of strings).

    Returns:
        JSS in [0, 1]. Higher means more consistent across prompt variants.

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    if len(decisions_a) != len(decisions_b):
        raise ValueError(
            f"Length mismatch: decisions_a has {len(decisions_a)} items, "
            f"decisions_b has {len(decisions_b)}."
        )
    if len(decisions_a) == 0:
        raise ValueError("decisions_a and decisions_b must not be empty.")

    matches = sum(a == b for a, b in zip(decisions_a, decisions_b))
    return matches / len(decisions_a)


def decision_flip_rate(
    decisions_a: Sequence[str],
    decisions_b: Sequence[str],
) -> float:
    """
    Decision Flip Rate: fraction of pairs where the judge changes its
    decision between prompt variants A and B.

    flip_rate = 1 - JSS
    """
    return 1.0 - judge_sensitivity_score(decisions_a, decisions_b)


def cohens_kappa(
    decisions_a: Sequence[str],
    decisions_b: Sequence[str],
) -> float:
    """
    Cohen's kappa: inter-rater agreement corrected for chance agreement.

    Returns 1.0 when sklearn returns NaN (all labels identical — perfect agreement).

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    if len(decisions_a) != len(decisions_b):
        raise ValueError(
            f"Length mismatch: {len(decisions_a)} vs {len(decisions_b)}."
        )
    if len(decisions_a) == 0:
        raise ValueError("Inputs must not be empty.")

    try:
        import math
        import warnings
        from sklearn.metrics import cohen_kappa_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(decisions_a, decisions_b)
        return 1.0 if math.isnan(kappa) else float(kappa)
    except ImportError:
        pass

    # Manual fallback when sklearn is unavailable
    n = len(decisions_a)
    p_o = sum(a == b for a, b in zip(decisions_a, decisions_b)) / n

    labels = set(decisions_a) | set(decisions_b)
    p_e = sum(
        (sum(d == label for d in decisions_a) / n)
        * (sum(d == label for d in decisions_b) / n)
        for label in labels
    )

    if abs(1.0 - p_e) < 1e-12:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_confidence_interval(
    decisions_a: Sequence[str],
    decisions_b: Sequence[str],
    metric_fn: Callable[[Sequence[str], Sequence[str]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for any scalar metric.

    Resamples (decisions_a[i], decisions_b[i]) pairs with replacement
    n_bootstrap times, applies metric_fn to each resample, and returns
    the percentile-based confidence interval.
    """
    pairs = list(zip(decisions_a, decisions_b))
    n = len(pairs)

    rng = np.random.default_rng(seed=42)

    boot_stats = []
    for _ in range(n_bootstrap):
        idxs = rng.integers(0, n, size=n)
        boot_a = [pairs[i][0] for i in idxs]
        boot_b = [pairs[i][1] for i in idxs]
        boot_stats.append(metric_fn(boot_a, boot_b))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_all_metrics(
    decisions_a: Sequence[str],
    decisions_b: Sequence[str],
) -> dict:
    """
    Compute the full JudgeSense metric suite.

    Returns dict with keys: jss, flip_rate, cohens_kappa, ci_lower, ci_upper,
    n_pairs, n_flips.
    """
    jss   = judge_sensitivity_score(decisions_a, decisions_b)
    flip  = decision_flip_rate(decisions_a, decisions_b)
    kappa = cohens_kappa(decisions_a, decisions_b)
    ci_lower, ci_upper = bootstrap_confidence_interval(
        decisions_a, decisions_b, judge_sensitivity_score
    )
    n_pairs = len(decisions_a)
    n_flips = round(flip * n_pairs)

    return {
        "jss":          round(jss,   4),
        "flip_rate":    round(flip,  4),
        "cohens_kappa": round(kappa, 4),
        "ci_lower":     round(ci_lower, 4),
        "ci_upper":     round(ci_upper, 4),
        "n_pairs":      n_pairs,
        "n_flips":      n_flips,
    }


def compute_results_summary(results_dir: str | Path) -> Dict:
    """
    Read all JSONL from results_dir, compute JSS per (model, task),
    save metrics_summary.json alongside results, and print a formatted table.

    Records with error != None or UNCLEAR decisions are excluded from metrics.

    Returns the summary dict.
    """
    results_dir = Path(results_dir)
    summary: Dict[str, Dict[str, dict]] = {}

    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        groups: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}
        with open(jsonl_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("pair_id") in EXCLUDED_PAIRS:
                    continue

                if rec.get("error") is not None:
                    continue

                model     = rec.get("model", "unknown")
                task_type = rec.get("task_type", "unknown")
                norm_a    = rec.get("normalized_a") or rec.get("prompt_a_decision", "")
                norm_b    = rec.get("normalized_b") or rec.get("prompt_b_decision", "")

                if norm_a == "UNCLEAR" or norm_b == "UNCLEAR":
                    continue

                key = (model, task_type)
                if key not in groups:
                    groups[key] = ([], [])
                groups[key][0].append(norm_a)
                groups[key][1].append(norm_b)

        for (model, task_type), (da, db) in groups.items():
            if len(da) < 2:
                continue
            metrics = compute_all_metrics(da, db)
            if model not in summary:
                summary[model] = {}
            summary[model][task_type] = metrics

    # Save
    out_path = results_dir.parent / "metrics_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Print table
    header = f"{'Model':<20} {'Task':<12} {'JSS':>6} {'Flip':>6} {'Kappa':>7} {'CI 95%':>14} {'N':>6}"
    print("\n" + "=" * len(header))
    print("JudgeSense Metrics Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model in sorted(summary):
        for task in sorted(summary[model]):
            m = summary[model][task]
            ci = f"[{m['ci_lower']:.3f},{m['ci_upper']:.3f}]"
            print(
                f"{model:<20} {task:<12} {m['jss']:>6.3f} {m['flip_rate']:>6.3f} "
                f"{m['cohens_kappa']:>7.3f} {ci:>14} {m['n_pairs']:>6}"
            )
    print("=" * len(header))
    print(f"Saved to: {out_path}\n")

    return summary


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="JudgeSense metrics")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Compute metrics summary from data/results/raw_outputs/ and print table.",
    )
    args = parser.parse_args()

    if args.summarize:
        compute_results_summary("data/results/raw_outputs/")
    else:
        print("=== JudgeSense Metrics Self-Test ===\n")

        a1 = ["YES", "YES", "NO",  "YES", "NO",  "YES", "YES", "NO",  "YES", "NO"]
        b1 = ["YES", "NO",  "NO",  "YES", "NO",  "YES", "YES", "NO",  "YES", "YES"]
        print("Scenario 1: High consistency (8/10 agree, 2 flips)")
        pprint.pprint(compute_all_metrics(a1, b1))
        print()

        a2 = ["YES", "YES", "YES", "YES", "YES"]
        b2 = ["NO",  "YES", "NO",  "YES", "NO"]
        print("Scenario 2: Low consistency (3/5 agree, 2 flips)")
        pprint.pprint(compute_all_metrics(a2, b2))
        print()

        a3 = ["YES"] * 10
        b3 = ["YES"] * 10
        print("Scenario 3: Perfect consistency (10/10 agree, 0 flips)")
        pprint.pprint(compute_all_metrics(a3, b3))
        print()

        a4 = ["YES"] * 6
        b4 = ["NO"]  * 6
        print("Scenario 4: Zero consistency (0/6 agree, all flips)")
        pprint.pprint(compute_all_metrics(a4, b4))
