"""
JudgeSense v2 metrics — cluster-aware, chance-corrected, ordinal-aware.

`src/metrics.py` (v1) is intentionally left untouched so published numbers
remain reproducible; this module supersedes it for all new analysis.

Key contracts (docs/V2_ARCHITECTURE.md §2):
- Confidence intervals require an EXPLICIT `cluster_unit` ("row",
  "structural_pair", "prompt_pair", "item"); there is no default that
  silently assumes independent rows, and the declared unit is echoed in
  every result. On the structural axis "item" is mandatory, not merely
  conservative: all five structural pairs for an item share its single S0
  arm (docs/V2_1_STRUCTURAL_AXIS.md §4).
- Class N (intervention) variants must never be scored as JSS; see
  `assert_jss_eligible` and `structural_shift_rate`.
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
# Nesting: item contains either prompt_pair (instruction axis) or
# structural_pair (structural axis), each of which contains rows (A/B orderings).
CLUSTER_UNITS = ("row", "structural_pair", "prompt_pair", "item")

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

    # Kappa is computed over PARSEABLE decisions only, under both policies.
    #
    # Rationale: observed agreement never credits an UNCLEAR row (an
    # unparseable answer is not evidence of a consistent judgment). If UNCLEAR
    # were nonetheless left in the marginals, expected agreement would carry
    # UNCLEAR x UNCLEAR mass that observed agreement can never realise, so
    # kappa would be depressed in proportion to a judge's malformed-output
    # rate rather than its inconsistency — worst exactly where the rate is
    # highest (v1: Mistral-7B at 15.7%). Scoring both terms on the same
    # support keeps the correction measuring what it claims to measure.
    #
    # The malformed rate is a real quality signal, but it is reported by
    # `jss(..., "disagree")` and `label_histogram`, not folded into kappa.
    pa = [a for a, v in zip(da, valid) if v]
    pb = [b for b, v in zip(db, valid) if v]
    n = len(pa)
    if n == 0:
        raise ValueError("No parseable decision pairs to chance-correct.")

    p_o = sum(1 for a, b in zip(pa, pb) if a == b) / n
    labels = set(pa) | set(pb)
    p_e = sum(
        (sum(d == lab for d in pa) / n) * (sum(d == lab for d in pb) / n)
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
        key = {
            "prompt_pair": "prompt_pair_id",
            "structural_pair": "structural_pair_id",
            "item": "item_id",
        }[cluster_unit]
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
    # The chance-corrected score is the headline answer to the "JSS rewards
    # compressed output distributions" critique, so it carries a clustered CI
    # too — a point estimate without uncertainty is what the v1 review faulted.
    try:
        chance = cluster_bootstrap_ci(
            records, lambda r: chance_corrected_jss(r, "disagree"), cluster_unit,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    except ValueError:
        # No parseable decision pairs (in the full set or in some resample);
        # chance correction is undefined rather than zero.
        chance = None
    # `jss(..., "drop")` raises when every row is UNCLEAR (e.g. a judge that
    # never emits a parseable decision). That is a legitimate, reportable
    # state, not a reason to abort the whole suite.
    try:
        jss_drop = jss(records, "drop")
    except ValueError:
        jss_drop = None
    out = {
        "jss_strict": strict,
        "jss_drop": jss_drop,
        "chance_corrected_jss": chance,
        "label_histogram": label_histogram(records),
        "decision_entropy_bits": decision_entropy(records),
        "cluster_unit": cluster_unit,
    }
    if likert:
        out["quadratic_weighted_kappa"] = quadratic_weighted_kappa(records, unclear_policy="disagree")
        out["mean_absolute_flip"] = mean_absolute_flip(records, unclear_policy="disagree")
    return out


# ── Structural axis: Class N interventions ───────────────────────────────────
# Chain-of-thought and expert-persona arms plausibly change the judgment
# process rather than merely rewording the request, so disagreement between
# them and S0 is NOT judge instability. Reporting it as JSS would be measuring
# that CoT is a different task. These arms get Structural Shift Rate instead.
# See docs/V2_1_STRUCTURAL_AXIS.md section 1.


class MetricContractError(ValueError):
    """Raised when a metric is applied to a variant class it must not describe."""


def _variant_classes(records: Sequence[dict]) -> set:
    return {r.get("variant_class") for r in records if r.get("variant_class") is not None}


def assert_jss_eligible(records: Sequence[dict]) -> None:
    """
    Refuse to compute JSS over Class N records.

    This is the one pooling the structural design forbids, so it is enforced in
    code rather than left to reviewer discipline. Records with no
    `variant_class` (the instruction axis) are unaffected.
    """
    classes = _variant_classes(records)
    forbidden = classes & {"N"}
    if forbidden:
        raise MetricContractError(
            "JSS is not defined over Class N (intervention) variants: "
            f"found {sorted(classes)}. Use structural_shift_rate() instead — "
            "disagreement between an intervention arm and S0 is a shift in "
            "verdicts, not judge instability."
        )


def structural_shift_rate(
    records: Sequence[dict],
    unclear_policy: str = "disagree",
) -> dict:
    """
    Structural Shift Rate for a Class N arm: how far the intervention moves
    verdicts relative to the S0 baseline, with directional decomposition.

    `records` carry `decision_a` (S0) and `decision_b` (the variant arm).
    Returns the shift rate, the transition counts S0-label -> variant-label,
    and the net per-label flow, so a result can say which direction the
    intervention pushes rather than only how much it perturbs.
    """
    da, db, valid = _apply_unclear_policy(records, unclear_policy)
    if not da:
        raise ValueError("No records to score (all dropped or empty input).")

    transitions: Counter = Counter()
    shifted = 0
    for a, b, v in zip(da, db, valid):
        transitions[(a, b)] += 1
        if not v or a != b:
            shifted += 1

    labels = sorted(set(da) | set(db))
    net: Dict[str, int] = {}
    for label in labels:
        gained = sum(n for (a, b), n in transitions.items() if b == label and a != label)
        lost = sum(n for (a, b), n in transitions.items() if a == label and b != label)
        net[label] = gained - lost

    return {
        "structural_shift_rate": shifted / len(da),
        "n_rows": len(da),
        "n_shifted": shifted,
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
        "net_flow": net,
        "unclear_policy": unclear_policy,
    }


def mean_likert_shift(
    records: Sequence[dict],
    categories: Sequence[str] = ("1", "2", "3", "4", "5"),
) -> dict:
    """
    Signed mean shift on the Likert scale for a Class N arm (variant minus S0).

    Direction is the point: a persona that systematically rates one point
    harsher is a different finding from one that adds symmetric noise, and an
    unsigned rate cannot tell them apart. Unparseable rows are excluded and
    counted rather than charged a distance, since a signed magnitude has no
    defensible value for them.
    """
    index = {c: i for i, c in enumerate(categories)}
    deltas: List[int] = []
    n_unparseable = 0
    for rec in records:
        ia = index.get(str(rec["decision_a"]))
        ib = index.get(str(rec["decision_b"]))
        if ia is None or ib is None:
            n_unparseable += 1
            continue
        deltas.append(ib - ia)
    if not deltas:
        raise ValueError("No parseable Likert pairs.")
    return {
        "mean_shift": float(np.mean(deltas)),
        "n_scored": len(deltas),
        "n_unparseable": n_unparseable,
        "harsher": sum(1 for d in deltas if d < 0),
        "more_lenient": sum(1 for d in deltas if d > 0),
        "unchanged": sum(1 for d in deltas if d == 0),
    }


def repeat_baseline_jss(records: Sequence[dict], unclear_policy: str = "disagree") -> float:
    """
    Noise ceiling: JSS between two calls of the SAME prompt (S0 issued twice).

    Every structural and instruction-axis result should be read as a delta from
    this. v1 had no such control, so its JSS numbers conflate prompt
    sensitivity with ordinary decoding variance and no published figure can
    separate the two.

    `records` here are repeat-pair decision records — see
    `src/repeat_baseline.py` for the call-record contract and
    `build_repeat_pairs()`, which turns per-call runner output into the
    `decision_a`/`decision_b` shape this function (and `jss_repeat_delta`
    below) expect.
    """
    return jss(records, unclear_policy=unclear_policy)


def jss_repeat_delta(
    paraphrase_records: Sequence[dict],
    repeat_records: Sequence[dict],
    cluster_unit: str,
    unclear_policy: str = "disagree",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    JSS reported as a DELTA from the repeat-baseline ceiling: JSS - JSS_rep.

    A raw JSS number cannot say whether a judge is sensitive to rewording or
    just noisy at temperature 0: JSS=0.90 means nothing on its own. Paired
    against JSS_rep=0.90 it means "no measurable paraphrase sensitivity";
    paired against JSS_rep=1.00 it means the judge is a perfect self-repeater
    that nonetheless moves 10% of the time under rewording — real sensitivity.

    `paraphrase_records` (decision_a/decision_b = the two paraphrase arms,
    e.g. P1/P2 on the instruction axis or S0/Sk on the structural axis) and
    `repeat_records` (decision_a/decision_b = call 1/call 2 of the SAME S0
    prompt, from `build_repeat_pairs`) both carry `item_id`. Every item in
    this suite contributes exactly one paraphrase-arm comparison AND one
    repeat-arm comparison from a SHARED S0 call context, so the two are not
    independent: pooling them into two separate bootstraps and subtracting
    the two point estimates (or worse, their CIs) would silently assume
    independence between arms that share an item. This function instead
    resamples ITEMS jointly, one draw per bootstrap replicate, and computes
    both JSS and JSS_rep from that single draw before taking the delta.

    `cluster_unit` must be passed explicitly (no default), echoing the
    module-wide contract that `CLUSTER_UNITS` = ("row", "structural_pair",
    "prompt_pair", "item") is never assumed. For THIS paired computation only
    "item" is accepted: the repeat arm exists at one row per (judge, item)
    (docs/V2_1_STRUCTURAL_AXIS.md §3, "S0 is issued twice per item"), so any
    finer declared unit has no corresponding repeat-side granularity to pair
    against — resampling at "row"/"prompt_pair"/"structural_pair" would have
    to either duplicate the single repeat row across sub-clusters (fabricating
    within-item independence it doesn't have) or drop the pairing entirely.
    Item is the coarsest node in the nesting hierarchy the module already
    treats as mandatory on the structural axis (§4); this generalizes that
    same mandate to any repeat-baseline delta.

    Only items present in BOTH inputs contribute; the counts of
    paraphrase-only and repeat-only items are reported so a caller can see if
    the two record sets are mismatched.

    Returns {"jss", "jss_rep", "delta", "ci_lower", "ci_upper", "cluster_unit",
    "n_clusters", "n_items_paraphrase_only", "n_items_repeat_only",
    "n_bootstrap", "confidence", "unclear_policy"}.
    """
    if cluster_unit not in CLUSTER_UNITS:
        raise ValueError(f"cluster_unit must be one of {CLUSTER_UNITS}, got {cluster_unit!r}")
    if cluster_unit != "item":
        raise ValueError(
            "jss_repeat_delta requires cluster_unit='item': the repeat arm "
            "exists at one row per (judge, item), so pairing it against a "
            "finer declared unit "
            f"({[u for u in CLUSTER_UNITS if u != 'item']}) would either "
            "fabricate within-item independence the repeat arm doesn't have, "
            "or silently drop the pairing. Item is the coarsest shared node "
            "and is always a valid resampling unit for data nested beneath it."
        )

    para_by_item: Dict[str, List[dict]] = {}
    for r in paraphrase_records:
        if "item_id" not in r:
            raise KeyError("paraphrase record missing 'item_id' required for jss_repeat_delta")
        para_by_item.setdefault(r["item_id"], []).append(r)
    rep_by_item: Dict[str, List[dict]] = {}
    for r in repeat_records:
        if "item_id" not in r:
            raise KeyError("repeat record missing 'item_id' required for jss_repeat_delta")
        rep_by_item.setdefault(r["item_id"], []).append(r)

    shared_items = sorted(set(para_by_item) & set(rep_by_item))
    if not shared_items:
        raise ValueError(
            "No items with both a paraphrase-arm record and a repeat-arm "
            "record; the delta is undefined without a shared S0 context."
        )
    n_para_only = len(set(para_by_item) - set(rep_by_item))
    n_rep_only = len(set(rep_by_item) - set(para_by_item))

    def _point(items: Sequence[str]) -> Tuple[float, float]:
        para = [r for it in items for r in para_by_item[it]]
        rep = [r for it in items for r in rep_by_item[it]]
        return (
            jss(para, unclear_policy=unclear_policy),
            jss(rep, unclear_policy=unclear_policy),
        )

    jss_full, jss_rep_full = _point(shared_items)
    delta_full = jss_full - jss_rep_full

    rng = np.random.default_rng(seed)
    n_items = len(shared_items)
    deltas = []
    for _ in range(n_bootstrap):
        idxs = rng.integers(0, n_items, size=n_items)
        sampled_items = [shared_items[i] for i in idxs]
        j_p, j_r = _point(sampled_items)
        deltas.append(j_p - j_r)

    alpha = 1.0 - confidence
    return {
        "jss": jss_full,
        "jss_rep": jss_rep_full,
        "delta": delta_full,
        "ci_lower": float(np.percentile(deltas, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        "cluster_unit": cluster_unit,
        "n_clusters": n_items,
        "n_items_paraphrase_only": n_para_only,
        "n_items_repeat_only": n_rep_only,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "unclear_policy": unclear_policy,
    }


def format_failure_rate(records: Sequence[dict], side: str = "b") -> dict:
    """
    Share of arm outputs that failed to parse — a first-class result, not a
    dropped row.

    S1 (JSON) and S4 (FINAL: marker) impose format contracts a judge can break
    while still "answering". v1 excluded unparseable output from JSS entirely,
    which hides prompt-induced failure exactly where it matters most (Mistral-7B
    reached 15.7%).
    """
    key = f"decision_{side}"
    total = len(records)
    if total == 0:
        raise ValueError("No records.")
    failed = sum(1 for r in records if r.get(key) in (None, UNCLEAR))
    return {"format_failure_rate": failed / total, "n_failed": failed, "n_rows": total}
