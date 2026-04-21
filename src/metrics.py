"""
JudgeSense metrics — sensitivity and consistency statistics for LLM judges.

Implements the Judge Sensitivity Score (JSS) and related measures
as defined in the JudgeSense paper.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


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

    Args:
        decisions_a: Decisions from prompt variant A.
        decisions_b: Decisions from prompt variant B.

    Returns:
        Flip rate in [0, 1]. Higher means more sensitive to prompt wording.
    """
    return 1.0 - judge_sensitivity_score(decisions_a, decisions_b)


def cohens_kappa(
    decisions_a: Sequence[str],
    decisions_b: Sequence[str],
) -> float:
    """
    Cohen's kappa: inter-rater agreement corrected for chance agreement.

    kappa = (p_o - p_e) / (1 - p_e)

    where p_o is observed agreement and p_e is expected agreement by chance.
    Uses sklearn.metrics.cohen_kappa_score when available; falls back to a
    manual computation otherwise.

    Args:
        decisions_a: Decisions from prompt variant A.
        decisions_b: Decisions from prompt variant B.

    Returns:
        Cohen's kappa in [-1, 1]:
            < 0.0  : less than chance agreement
            0–0.20 : slight
            0.20–0.40 : fair
            0.40–0.60 : moderate
            0.60–0.80 : substantial
            0.80–1.0  : near-perfect

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
        # sklearn warns + returns NaN when only one label is present; we handle that
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

    # Guard against degenerate case where all decisions are identical
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

    Args:
        decisions_a: Decisions from prompt variant A.
        decisions_b: Decisions from prompt variant B.
        metric_fn: A callable (decisions_a, decisions_b) -> float.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        confidence: Coverage probability, e.g. 0.95 for a 95% CI.

    Returns:
        (lower, upper): Lower and upper confidence interval bounds.
    """
    pairs = list(zip(decisions_a, decisions_b))
    n = len(pairs)

    # Fixed seed for reproducibility across runs
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

    Args:
        decisions_a: Decisions from prompt variant A.
        decisions_b: Decisions from prompt variant B.

    Returns:
        Dictionary with keys:
            jss          — Judge Sensitivity Score
            flip_rate    — Decision Flip Rate (1 - JSS)
            cohens_kappa — Cohen's kappa
            ci_lower     — 95% bootstrap CI lower bound for JSS
            ci_upper     — 95% bootstrap CI upper bound for JSS
            n_pairs      — total number of evaluated pairs
            n_flips      — number of pairs where decisions differed
    """
    jss = judge_sensitivity_score(decisions_a, decisions_b)
    flip = decision_flip_rate(decisions_a, decisions_b)
    kappa = cohens_kappa(decisions_a, decisions_b)
    ci_lower, ci_upper = bootstrap_confidence_interval(
        decisions_a, decisions_b, judge_sensitivity_score
    )
    n_pairs = len(decisions_a)
    n_flips = round(flip * n_pairs)

    return {
        "jss": round(jss, 4),
        "flip_rate": round(flip, 4),
        "cohens_kappa": round(kappa, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_pairs": n_pairs,
        "n_flips": n_flips,
    }


if __name__ == "__main__":
    import pprint

    print("=== JudgeSense Metrics Self-Test ===\n")

    # Scenario 1: High consistency — 8 out of 10 agree
    a1 = ["YES", "YES", "NO",  "YES", "NO",  "YES", "YES", "NO",  "YES", "NO"]
    b1 = ["YES", "NO",  "NO",  "YES", "NO",  "YES", "YES", "NO",  "YES", "YES"]
    print("Scenario 1: High consistency (8/10 agree, 2 flips)")
    pprint.pprint(compute_all_metrics(a1, b1))
    print()

    # Scenario 2: Low consistency — 3 out of 5 agree
    a2 = ["YES", "YES", "YES", "YES", "YES"]
    b2 = ["NO",  "YES", "NO",  "YES", "NO"]
    print("Scenario 2: Low consistency (3/5 agree, 2 flips)")
    pprint.pprint(compute_all_metrics(a2, b2))
    print()

    # Scenario 3: Perfect consistency
    a3 = ["YES"] * 10
    b3 = ["YES"] * 10
    print("Scenario 3: Perfect consistency (10/10 agree, 0 flips)")
    pprint.pprint(compute_all_metrics(a3, b3))
    print()

    # Scenario 4: Zero consistency
    a4 = ["YES"] * 6
    b4 = ["NO"]  * 6
    print("Scenario 4: Zero consistency (0/6 agree, all flips)")
    pprint.pprint(compute_all_metrics(a4, b4))
