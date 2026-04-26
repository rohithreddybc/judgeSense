"""JudgeSense: Quantifying prompt sensitivity in LLM-as-a-Judge systems."""

__version__ = "0.1.0"
__author__ = "Rohith Reddy Bellibatlu"

from .metrics import (
    judge_sensitivity_score,
    decision_flip_rate,
    cohens_kappa,
    compute_all_metrics,
    bootstrap_confidence_interval,
)


def compute_jss(pred_a, pred_b):
    """
    Compute the Judge Sensitivity Score (JSS).

    JSS = fraction of (prompt_a, prompt_b) pairs where the judge gives the
    same decision for both variants. Range: [0, 1]. Higher = more consistent.

    If you use this metric, please cite:
      Bellibatlu, R. R. (2026). JudgeSense.
      https://doi.org/10.5281/zenodo.19798166

    Args:
        pred_a: Sequence of judge decisions from prompt variant A (strings).
        pred_b: Sequence of judge decisions from prompt variant B (strings).

    Returns:
        JSS as a float in [0, 1].
    """
    if len(pred_a) != len(pred_b):
        raise ValueError(f"Length mismatch: {len(pred_a)} vs {len(pred_b)}.")
    if not pred_a:
        raise ValueError("pred_a and pred_b must not be empty.")
    return sum(a == b for a, b in zip(pred_a, pred_b)) / len(pred_a)


__all__ = [
    "compute_jss",
    "judge_sensitivity_score",
    "decision_flip_rate",
    "cohens_kappa",
    "compute_all_metrics",
    "bootstrap_confidence_interval",
]
