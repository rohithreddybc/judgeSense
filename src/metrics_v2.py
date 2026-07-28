"""
JudgeSense v2 metrics — cluster-aware, chance-corrected, ordinal-aware.

`src/metrics.py` (v1) is intentionally left untouched so published numbers
remain reproducible; this module supersedes it for all new analysis.

Key contracts (docs/V2_ARCHITECTURE.md §2):
- Confidence intervals require an EXPLICIT `cluster_unit` ("row",
  "prompt_pair", "item"); there is no default that silently assumes
  independent rows, and the declared unit is echoed in every result.
- All agreement metrics take an `unclear_policy`: "drop" reproduces v1
  behavior; "disagree" (strict) counts malformed/UNCLEAR output as
  disagreement and is the v2 headline mode.
- Chance-corrected JSS is Cohen's kappa over the two variants' decisions:
  a judge that always outputs one label scores 0, not 1.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

UNCLEAR = "UNCLEAR"
CLUSTER_UNITS = ("row", "prompt_pair", "item")

# ── Record helpers ───────────────────────────────────────────────────────────
# A "record" is a dict with at least: decision_a, decision_b, and (for
# clustered CIs) item_id and prompt_pair_id.


def _apply_unclear_policy(
    records: Sequence[dict], unclear_policy: str
) -> Tuple[List[str], List[str], List[bool]]:
    """
    Returns (decisions_a, decisions_b, valid_mask) after applying the policy.
    "drop": UNCLEAR rows removed. "disagree": rows kept; agreement functions
    must score any row containing UNCLEAR as a disagreement.
    """
    if unclear_policy not in ("drop", "disagree"):
        raise ValueError(f"unclear_policy must be 'drop' or 'disagree', got {unclear_policy!r}")
    da, db, valid = [], [], []
    for rec in records:
        a, b = rec["decision_a"], rec["decision_b"]
        is_unclear = (a == UNCLEAR) or (b == UNCLEAR) or a is None or b is None
        if unclear_policy == "drop" and is_unclear:
            continue
        da.append(a if a is not None else UNCLEAR)
        db.append(b if b is not None else UNCLEAR)
        valid.append(not is_unclear)
    return da, db, valid


def jss(records: Sequence[dict], unclear_policy: str = "disagree") -> float:
    """
    Judge Sensitivity Score: fraction of prompt pairs with matching decisions.
    Under "disagree", a row with any UNCLEAR decision counts as a mismatch
    (even if both sides are UNCLEAR — an unparseable answer is not evidence
    of a consistent judgment).
    """
    da, db, valid = _apply_unclear_policy(records, unclear_policy)
    if not da:
        raise ValueError("No records to score (all dropped or empty input).")
    matches = sum(1 for a, b, v in zip(da, db, valid) if v and a == b)
    return matches / len(da)


def chance_corrected_jss(records: Sequence[dict], unclear_policy: str = "disagree") -> float:
    """
    Cohen's kappa between variant-A and variant-B decisions.

    Corrects the failure mode where raw JSS rewards judges that compress
    their output distribution (v1 audit: corr(JSS, entropy) = -0.484): the
    expected agreement of a degenerate always-one-label judge is 1.0, so its
    kappa is 0 by convention here (p_e == 1).
    """
    da, db, valid = _apply_unclear_policy(records, unclear_policy)
    if not da:
        raise ValueError("No records to score (all dropped or empty input).")
    n = len(da)
    p_o = sum(1 for a, b, v in zip(da, db, valid) if v and a == b) / n
    labels = set(da) | set(db)
    p_e = sum(
        (sum(d == lab for d in da) / n) * (sum(d == lab for d in db) / n)
        for lab in labels
    )
    if abs(1.0 - p_e) < 1e-12:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ── Ordinal (Likert) agreement ───────────────────────────────────────────────

def quadratic_weighted_kappa(
    records: Sequence[dict],
    categories: Sequence[str] = ("1", "2", "3", "4", "5"),
    unclear_policy: str = "drop",
) -> float:
    """
    Quadratic-weighted kappa between variant-A and variant-B Likert decisions,
    so a 3<->4 flip is penalized far less than a 1<->5 flip.

    UNCLEAR handling: "drop" removes such rows; "disagree" maps UNCLEAR to
    the maximally distant category from the other decision (conservative:
    an unparseable answer is treated as a worst-case flip).
    """
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    def _idx_pair(a: str, b: str) -> Optional[Tuple[int, int]]:
        ia, ib = cat_index.get(a), cat_index.get(b)
        if ia is not None and ib is not None:
            return ia, ib
        if unclear_policy == "drop":
            return None
        if ia is None and ib is None:
            return 0, k - 1  # both unparseable: maximal disagreement
        known = ia if ia is not None else ib
        far = 0 if known >= k // 2 else k - 1
        return (known, far) if ia is not None else (far, known)

    pairs = []
    for rec in records:
        p = _idx_pair(str(rec["decision_a"]), str(rec["decision_b"]))
        if p is not None:
            pairs.append(p)
    if not pairs:
        raise ValueError("No scorable Likert records.")

    observed = np.zeros((k, k))
    for ia, ib in pairs:
        observed[ia, ib] += 1
    n = observed.sum()
    weights = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    hist_a, hist_b = observed.sum(axis=1), observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / n
    denom = (weights * expected).sum()
    if denom < 1e-12:
        return 0.0
    return float(1.0 - (weights * observed).sum() / denom)


def mean_absolute_flip(
    records: Sequence[dict],
    categories: Sequence[str] = ("1", "2", "3", "4", "5"),
    unclear_policy: str = "drop",
) -> float:
    """Mean |decision_a - decision_b| on the Likert scale; UNCLEAR under
    "disagree" is charged the maximum distance (k-1)."""
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    dists = []
    for rec in records:
        ia = cat_index.get(str(rec["decision_a"]))
        ib = cat_index.get(str(rec["decision_b"]))
        if ia is None or ib is None:
            if unclear_policy == "drop":
                continue
            dists.append(k - 1)
        else:
            dists.append(abs(ia - ib))
    if not dists:
        raise ValueError("No scorable Likert records.")
    return float(np.mean(dists))


# ── Cluster bootstrap ────────────────────────────────────────────────────────

def cluster_bootstrap_ci(
    records: Sequence[dict],
    metric_fn: Callable[[Sequence[dict]], float],
    cluster_unit: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Percentile bootstrap CI resampling CLUSTERS with replacement.

    cluster_unit: "row" (each record its own cluster — reproduces the naive
    bootstrap), "prompt_pair" (cluster on prompt_pair_id), or "item"
    (cluster on item_id). Records must carry the corresponding key.

    Returns {"estimate", "ci_lower", "ci_upper", "cluster_unit",
    "n_clusters", "n_rows", "n_bootstrap", "confidence"} — the declared unit
    travels with the result so tables cannot silently drop it.
    """
    if cluster_unit not in CLUSTER_UNITS:
        raise ValueError(f"cluster_unit must be one of {CLUSTER_UNITS}, got {cluster_unit!r}")
    records = list(records)
    if not records:
        raise ValueError("No records.")

    if cluster_unit == "row":
        clusters: List[List[dict]] = [[r] for r in records]
    else:
        key = "prompt_pair_id" if cluster_unit == "prompt_pair" else "item_id"
        grouped: Dict[str, List[dict]] = {}
        for rec in records:
            if key not in rec:
                raise KeyError(
                    f"Record missing '{key}' required for cluster_unit={cluster_unit!r}"
                )
            grouped.setdefault(rec[key], []).append(rec)
        clusters = list(grouped.values())

    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    stats = []
    for _ in range(n_bootstrap):
        idxs = rng.integers(0, n_clusters, size=n_clusters)
        sample: List[dict] = []
        for i in idxs:
            sample.extend(clusters[i])
        stats.append(metric_fn(sample))

    alpha = 1.0 - confidence
    return {
        "estimate": metric_fn(records),
        "ci_lower": float(np.percentile(stats, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(stats, 100 * (1 - alpha / 2))),
        "cluster_unit": cluster_unit,
        "n_clusters": n_clusters,
        "n_rows": len(records),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


# ── Distribution diagnostics ─────────────────────────────────────────────────

def label_histogram(records: Sequence[dict]) -> Dict[str, int]:
    """Pooled histogram over both variants' decisions (UNCLEAR included)."""
    counts: Counter = Counter()
    for rec in records:
        counts[str(rec["decision_a"])] += 1
        counts[str(rec["decision_b"])] += 1
    return dict(counts)


def decision_entropy(records: Sequence[dict]) -> float:
    """Shannon entropy (bits) of the pooled decision distribution."""
    counts = label_histogram(records)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No decisions.")
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def jss_entropy_correlation(per_judge: Dict[str, Sequence[dict]]) -> dict:
    """
    Across judges, correlate JSS with decision entropy — the diagnostic that
    exposed raw JSS rewarding output-distribution compression (r = -0.484 on
    v1 results). Reports Pearson and Spearman, with per-judge values.
    """
    from scipy import stats as sps

    judges = sorted(per_judge.keys())
    if len(judges) < 3:
        raise ValueError("Need at least 3 judges for a meaningful correlation.")
    jss_vals = [jss(per_judge[j], unclear_policy="disagree") for j in judges]
    ent_vals = [decision_entropy(per_judge[j]) for j in judges]
    pearson = sps.pearsonr(jss_vals, ent_vals)
    spearman = sps.spearmanr(jss_vals, ent_vals)
    return {
        "judges": judges,
        "jss": dict(zip(judges, jss_vals)),
        "entropy_bits": dict(zip(judges, ent_vals)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


# ── Convenience: full suite ──────────────────────────────────────────────────

def compute_all_metrics_v2(
    records: Sequence[dict],
    cluster_unit: str,
    likert: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Full v2 metric suite for one (judge, task) slice of records."""
    strict = cluster_bootstrap_ci(
        records, lambda r: jss(r, "disagree"), cluster_unit,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    out = {
        "jss_strict": strict,
        "jss_drop": jss(records, "drop"),
        "chance_corrected_jss": chance_corrected_jss(records, "disagree"),
        "label_histogram": label_histogram(records),
        "decision_entropy_bits": decision_entropy(records),
        "cluster_unit": cluster_unit,
    }
    if likert:
        out["quadratic_weighted_kappa"] = quadratic_weighted_kappa(records, unclear_policy="disagree")
        out["mean_absolute_flip"] = mean_absolute_flip(records, unclear_policy="disagree")
    return out
