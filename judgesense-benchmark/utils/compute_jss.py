"""
compute_jss.py — Judge Sensitivity Score (JSS) for the JudgeSense benchmark.

JSS measures how often a judge gives the same decision when presented with
two semantically equivalent but differently phrased prompts.

    JSS = mean(decisions_a[i] == decisions_b[i])

Higher JSS (→ 1.0) means the judge is consistent across prompt variants.
Lower JSS (→ 0.0) means the judge is highly sensitive to prompt phrasing.
"""

from __future__ import annotations


def compute_jss(
    decisions_a: list[str],
    decisions_b: list[str],
) -> float:
    """Compute the Judge Sensitivity Score (JSS).

    Args:
        decisions_a: Judge decisions elicited by prompt variant A.
        decisions_b: Judge decisions elicited by prompt variant B.
            Must be the same length as decisions_a.

    Returns:
        JSS in [0.0, 1.0].

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    if len(decisions_a) != len(decisions_b):
        raise ValueError(
            f"Length mismatch: decisions_a has {len(decisions_a)} items, "
            f"decisions_b has {len(decisions_b)}."
        )
    if not decisions_a:
        raise ValueError("decisions_a and decisions_b must not be empty.")

    matches = sum(a == b for a, b in zip(decisions_a, decisions_b))
    return matches / len(decisions_a)


def flip_rate(decisions_a: list[str], decisions_b: list[str]) -> float:
    """Decision Flip Rate = 1 - JSS."""
    return 1.0 - compute_jss(decisions_a, decisions_b)


if __name__ == "__main__":
    a = ["YES", "YES", "NO", "YES", "NO", "YES", "YES", "NO", "YES", "NO"]
    b = ["YES", "NO",  "NO", "YES", "NO", "YES", "YES", "NO", "YES", "YES"]
    jss = compute_jss(a, b)
    print(f"JSS: {jss:.3f}  |  Flip rate: {flip_rate(a, b):.3f}")
