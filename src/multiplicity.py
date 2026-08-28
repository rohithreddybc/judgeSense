"""The multiplicity plan, implemented.

PREREGISTRATION.md commits to three things and marks all three
"NOT YET IMPLEMENTED in code; applied at write-up". A plan that is only ever
applied by hand at write-up is not a pre-registration, it is an intention, and
the paper currently has to say so:

    The multiplicity plan added at 82ea522 is not applied in this paper.

At twelve cells that was defensible. The multi-vendor sweep takes it to a
hundred (25 judges x 4 tasks), where an unadjusted "CI excludes zero" rule
yields roughly five spurious findings under a global null -- and the cells that
clear are exactly the ones a narrative foregrounds. So it is implemented here.

What the plan says, and what this module provides:

1. ONE PRIMARY CONTRAST. The confirmatory claim is ΔJSS pooled across judges
   within a task: four intervals, Holm-corrected across the four.
   -> pooled_delta_by_task, holm

2. EVERYTHING PER-CELL IS EXPLORATORY. Cell intervals are reported in full with
   Benjamini-Hochberg values at a 10% false discovery rate.
   -> benjamini_hochberg

3. A SMALLEST EFFECT OF INTEREST. |ΔJSS| < 0.02 is not practically meaningful
   whatever its interval does.
   -> practically_meaningful

Plus the power statement the plan promises alongside the results, so a null is
distinguishable from an underpowered test:
   -> minimum_detectable_effect

Pooling detail that matters: every judge answers the SAME items, so judges are
correlated and pooling them as independent would understate uncertainty. One
item-level resample is drawn per bootstrap iteration and shared across judges,
and the pooled statistic is recomputed inside that resample. The correlation is
therefore carried by the resampling scheme rather than assumed away.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Two-sided normal quantiles, hardcoded so the module has no scipy dependency.
_Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
_Z_POWER = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def bootstrap_p_value(draws: Sequence[float], null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for H0: theta == null.

    Computed from the resample distribution rather than from a normal
    approximation, because ΔJSS is a bounded difference of proportions and is
    skewed near the ceiling -- exactly where most of these cells sit.

    The (1 + count) / (1 + n) form never returns 0: a p-value of zero would
    claim more resolution than 2,000 resamples can provide, and would sort to
    the top of any adjustment as if it were infinitely significant.
    """
    d = np.asarray(list(draws), dtype=float)
    if d.size == 0:
        return float("nan")
    n = d.size
    p_low = (1.0 + np.sum(d <= null)) / (1.0 + n)
    p_high = (1.0 + np.sum(d >= null)) / (1.0 + n)
    return float(min(1.0, 2.0 * min(p_low, p_high)))


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Holm-Bonferroni step-down adjustment.

    Controls the family-wise error rate: the probability of ANY false rejection
    across the family. That is the right guarantee for the confirmatory tier,
    where one spurious task-level claim would be quoted as a finding.

    Adjusted values are enforced monotone non-decreasing down the sorted order,
    so a later hypothesis can never appear more significant than an earlier one
    it was tested after.
    """
    items = [(k, v) for k, v in pvalues.items() if v == v]      # drop NaN
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kv: kv[1])
    out: Dict[str, dict] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)                             # monotone
        out[key] = {"p_raw": p, "p_adj": running,
                    "reject": running <= alpha, "rank": i + 1, "n_tests": m}
    for key, p in pvalues.items():
        if p != p:
            out[key] = {"p_raw": float("nan"), "p_adj": float("nan"),
                        "reject": False, "rank": None, "n_tests": m}
    return out


def benjamini_hochberg(pvalues: Dict[str, float], fdr: float = 0.10) -> Dict[str, dict]:
    """Benjamini-Hochberg step-up adjustment at a declared false discovery rate.

    Controls the expected PROPORTION of false rejections among rejections, which
    is the appropriate guarantee for an exploratory tier that is reported in full
    rather than mined for the significant rows.

    Adjusted values are enforced monotone from the largest p downward, the
    standard correction that stops a small p-value inheriting a larger
    neighbour's adjustment.
    """
    items = [(k, v) for k, v in pvalues.items() if v == v]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kv: kv[1])
    adj: List[float] = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = min(1.0, items[i][1] * m / (i + 1))
        prev = min(prev, val)
        adj[i] = prev
    out: Dict[str, dict] = {}
    for i, (key, p) in enumerate(items):
        out[key] = {"p_raw": p, "p_adj": adj[i], "reject": adj[i] <= fdr,
                    "rank": i + 1, "n_tests": m, "fdr": fdr}
    for key, p in pvalues.items():
        if p != p:
            out[key] = {"p_raw": float("nan"), "p_adj": float("nan"),
                        "reject": False, "rank": None, "n_tests": m, "fdr": fdr}
    return out


def practically_meaningful(delta: Optional[float], sesoi: float = 0.02) -> Optional[bool]:
    """|delta| >= the smallest effect declared of interest.

    Kept separate from significance on purpose: an interval excluding zero at a
    magnitude below the SESOI is a statement about sample size, not about judges.
    """
    if delta is None:
        return None
    return abs(delta) >= sesoi


def minimum_detectable_effect(sd: float, n_clusters: int, alpha: float = 0.05,
                              power: float = 0.80) -> Optional[float]:
    """Smallest |ΔJSS| this cell could have detected, at its actual cluster count.

    Reported so a null is distinguishable from an underpowered test, which the
    plan promises and the pipeline never emitted. `sd` is the bootstrap standard
    error of the estimate, so the cluster correlation is already inside it and
    n_clusters is used only to flag a cell too small to interpret.
    """
    if not sd or sd != sd or n_clusters < 2:
        return None
    z_a = _Z.get(round(alpha, 3), _Z[0.05])
    z_b = _Z_POWER.get(round(power, 2), _Z_POWER[0.80])
    return float((z_a + z_b) * sd)


def pooled_delta_by_task(
    per_judge_records: Dict[str, dict],
    delta_fn: Callable[[Sequence[dict], Sequence[dict], Sequence[str]], float],
    cluster_ids: Sequence[str],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """ΔJSS pooled across judges within one task -- the confirmatory contrast.

    per_judge_records maps judge -> {"paraphrase": [...], "repeat": [...]}.
    delta_fn computes one judge's delta restricted to a set of cluster ids.

    ONE item resample is drawn per iteration and applied to every judge, because
    the judges answered the same items. Resampling each judge independently
    would treat shared items as independent evidence and shrink the interval
    that the pooled claim rests on.
    """
    judges = sorted(per_judge_records)
    ids = list(cluster_ids)
    if not judges or not ids:
        return {"pooled_delta": None, "n_judges": len(judges), "n_clusters": len(ids)}

    point = []
    for j in judges:
        d = delta_fn(per_judge_records[j]["paraphrase"],
                     per_judge_records[j]["repeat"], ids)
        if d is not None and d == d:
            point.append(d)
    if not point:
        return {"pooled_delta": None, "n_judges": len(judges), "n_clusters": len(ids)}

    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(n_bootstrap):
        sample = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        per = []
        for j in judges:
            d = delta_fn(per_judge_records[j]["paraphrase"],
                         per_judge_records[j]["repeat"], sample)
            if d is not None and d == d:
                per.append(d)
        if per:
            draws.append(float(np.mean(per)))
    if not draws:
        return {"pooled_delta": float(np.mean(point)), "n_judges": len(judges),
                "n_clusters": len(ids)}

    a = 1.0 - confidence
    arr = np.asarray(draws)
    sd = float(np.std(arr, ddof=1))
    return {
        "pooled_delta": float(np.mean(point)),
        "ci_lower": float(np.percentile(arr, 100 * a / 2)),
        "ci_upper": float(np.percentile(arr, 100 * (1 - a / 2))),
        "p_value": bootstrap_p_value(arr),
        "se": sd,
        "mde_80": minimum_detectable_effect(sd, len(ids)),
        "n_judges": len(judges),
        "n_clusters": len(ids),
        "n_bootstrap": len(draws),
        "confidence": confidence,
    }


# ── the discrimination ceiling ───────────────────────────────────────────────
# The other half of the pre-registered decision rule, and the other item that
# was marked NOT YET IMPLEMENTED: "these thresholds are not constants anywhere
# and no verdict is emitted; this is a rule binding the write-up."
#
# A benchmark no heuristic can beat is worthless if no competent judge can beat
# it either. A prior version of this work was withdrawn partly for tasks that
# did not discriminate, so the ceiling is pre-committed exactly as the floor is,
# and the verdict is computed rather than asserted in prose.
#
# Thresholds are position-corrected accuracy for the BEST judge on the task,
# taken verbatim from PREREGISTRATION.md.
DISCRIMINATION_THRESHOLDS = {
    "factuality": 0.75,
    "coherence": 0.40,      # exact-match on the 1-5 scale
    "relevance": 0.70,
    "preference": 0.65,
}

# Reported alongside, because a constant-answer judge already achieves this and
# quoting uniform chance would overstate every margin.
MAJORITY_CLASS_RATE = {
    "factuality": 0.500,
    "coherence": 0.348,     # gold skew {4:87, 3:71, 2:55, 5:32, 1:5}
    "relevance": 0.500,
    "preference": 0.500,
}


def discrimination_verdict(task: str, best_accuracy: Optional[float]) -> dict:
    """Does this task discriminate, by the rule declared before the data?

    Returns the threshold, the margin over a constant-answer judge, and a
    verdict of "discriminating" / "not_discriminating" / "undetermined". A task
    that fails carries no judge ranking, which is a constraint on the write-up
    that this makes checkable.
    """
    threshold = DISCRIMINATION_THRESHOLDS.get(task)
    majority = MAJORITY_CLASS_RATE.get(task)
    if threshold is None:
        raise KeyError(f"no declared discrimination threshold for task {task!r}")
    if best_accuracy is None or best_accuracy != best_accuracy:
        return {"task": task, "threshold": threshold, "majority_class": majority,
                "best_accuracy": None, "margin_over_majority": None,
                "verdict": "undetermined", "ranking_permitted": False}
    passes = best_accuracy >= threshold
    return {
        "task": task,
        "threshold": threshold,
        "majority_class": majority,
        "best_accuracy": float(best_accuracy),
        "margin_over_majority": (float(best_accuracy - majority)
                                 if majority is not None else None),
        "verdict": "discriminating" if passes else "not_discriminating",
        "ranking_permitted": passes,
    }
