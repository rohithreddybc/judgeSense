"""Tests for src/metrics_v2.py (all record inputs are test fixtures)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics_v2 import (  # noqa: E402
    chance_corrected_jss,
    cluster_bootstrap_ci,
    compute_all_metrics_v2,
    decision_entropy,
    jss,
    jss_entropy_correlation,
    label_histogram,
    mean_absolute_flip,
    quadratic_weighted_kappa,
)


def rec(a, b, item="i0", pair="p0"):
    return {"decision_a": a, "decision_b": b, "item_id": item, "prompt_pair_id": pair}


# ── JSS and UNCLEAR policy ───────────────────────────────────────────────────

def test_jss_basic_agreement():
    records = [rec("YES", "YES"), rec("YES", "NO"), rec("NO", "NO"), rec("NO", "NO")]
    assert jss(records) == pytest.approx(0.75)


def test_unclear_drop_vs_disagree():
    records = [rec("YES", "YES"), rec("YES", "UNCLEAR"), rec("UNCLEAR", "UNCLEAR")]
    # v1 behavior: dropping UNCLEAR inflates agreement to 1.0
    assert jss(records, "drop") == pytest.approx(1.0)
    # strict: both UNCLEAR rows are disagreements (even the matching one)
    assert jss(records, "disagree") == pytest.approx(1 / 3)


def test_jss_empty_after_drop_raises():
    with pytest.raises(ValueError):
        jss([rec("UNCLEAR", "UNCLEAR")], "drop")


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        jss([rec("YES", "YES")], "ignore")


# ── Chance correction ────────────────────────────────────────────────────────

def test_always_one_label_judge_scores_zero_not_one():
    records = [rec("A", "A") for _ in range(50)]
    assert jss(records) == 1.0                      # raw JSS is fooled
    assert chance_corrected_jss(records) == 0.0     # corrected JSS is not


def test_chance_corrected_matches_kappa_by_hand():
    # 2x2 case: p_o = 0.8, marginals a: 0.6/0.4, b: 0.6/0.4 -> p_e = 0.52
    records = (
        [rec("YES", "YES")] * 5 + [rec("NO", "NO")] * 3
        + [rec("YES", "NO")] * 1 + [rec("NO", "YES")] * 1
    )
    expected = (0.8 - 0.52) / (1 - 0.52)
    assert chance_corrected_jss(records) == pytest.approx(expected)


# ── Ordinal metrics ──────────────────────────────────────────────────────────

def test_qwk_perfect_agreement():
    records = [rec(str(i % 5 + 1), str(i % 5 + 1)) for i in range(25)]
    assert quadratic_weighted_kappa(records) == pytest.approx(1.0)


def test_qwk_penalizes_far_flips_more_than_near_flips():
    base = [rec(str(i % 5 + 1), str(i % 5 + 1)) for i in range(20)]
    near = base + [rec("3", "4")] * 5
    far = base + [rec("1", "5")] * 5
    assert quadratic_weighted_kappa(near) > quadratic_weighted_kappa(far)


def test_mean_absolute_flip_values():
    records = [rec("3", "4"), rec("1", "5"), rec("2", "2")]
    assert mean_absolute_flip(records) == pytest.approx((1 + 4 + 0) / 3)


def test_ordinal_unclear_disagree_charges_max_distance():
    records = [rec("3", "3"), rec("5", "UNCLEAR")]
    assert mean_absolute_flip(records, unclear_policy="drop") == pytest.approx(0.0)
    assert mean_absolute_flip(records, unclear_policy="disagree") == pytest.approx(2.0)
    # QWK: UNCLEAR maps to the category farthest from the known decision
    qwk = quadratic_weighted_kappa(records, unclear_policy="disagree")
    assert qwk < 1.0


# ── Cluster bootstrap ────────────────────────────────────────────────────────

def test_cluster_unit_is_mandatory_and_validated():
    records = [rec("YES", "YES")]
    with pytest.raises(TypeError):
        cluster_bootstrap_ci(records, jss)  # no unit -> refuse
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(records, jss, "independent_rows")


def test_missing_cluster_key_raises():
    records = [{"decision_a": "YES", "decision_b": "YES"}]
    with pytest.raises(KeyError):
        cluster_bootstrap_ci(records, jss, "item")


def test_row_unit_reproduces_naive_bootstrap():
    rng = np.random.default_rng(7)
    records = [
        rec("YES", "YES" if rng.random() < 0.7 else "NO", item=f"i{k}", pair=f"p{k}")
        for k in range(60)
    ]
    row_ci = cluster_bootstrap_ci(records, jss, "row", n_bootstrap=500, seed=3)
    pair_ci = cluster_bootstrap_ci(records, jss, "prompt_pair", n_bootstrap=500, seed=3)
    # every row is its own prompt_pair here, so the two must coincide exactly
    assert row_ci["ci_lower"] == pair_ci["ci_lower"]
    assert row_ci["ci_upper"] == pair_ci["ci_upper"]
    assert row_ci["cluster_unit"] == "row"
    assert pair_ci["n_clusters"] == 60


def test_item_clustering_widens_ci_under_within_cluster_correlation():
    # 10 items x 12 rows; each item is entirely-agree or entirely-disagree,
    # so rows are perfectly correlated within items. The row bootstrap
    # pretends n=120 independent observations; the item bootstrap knows n=10.
    records = []
    for k in range(10):
        outcome = "YES" if k < 5 else "NO"
        for j in range(12):
            records.append(rec("YES", outcome, item=f"i{k}", pair=f"i{k}_p{j % 3}"))
    row_ci = cluster_bootstrap_ci(records, jss, "row", n_bootstrap=800, seed=1)
    item_ci = cluster_bootstrap_ci(records, jss, "item", n_bootstrap=800, seed=1)
    row_width = row_ci["ci_upper"] - row_ci["ci_lower"]
    item_width = item_ci["ci_upper"] - item_ci["ci_lower"]
    assert item_ci["n_clusters"] == 10
    assert item_width > 1.5 * row_width


def test_ci_result_carries_declared_unit_and_counts():
    records = [rec("YES", "YES", item=f"i{k // 2}", pair=f"p{k}") for k in range(10)]
    out = cluster_bootstrap_ci(records, jss, "item", n_bootstrap=50)
    assert out["cluster_unit"] == "item"
    assert out["n_clusters"] == 5
    assert out["n_rows"] == 10
    assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]


# ── Distribution diagnostics ─────────────────────────────────────────────────

def test_label_histogram_and_entropy():
    records = [rec("YES", "NO"), rec("YES", "YES")]
    assert label_histogram(records) == {"YES": 3, "NO": 1}
    # H(0.75, 0.25) = 0.8113 bits
    assert decision_entropy(records) == pytest.approx(0.8113, abs=1e-3)


def test_degenerate_distribution_has_zero_entropy():
    assert decision_entropy([rec("A", "A")] * 10) == pytest.approx(0.0)


def test_jss_entropy_correlation_detects_compression_reward():
    # Judge that compresses output (always A) has high raw JSS, low entropy;
    # diverse honest judges have lower JSS, higher entropy -> negative r.
    per_judge = {
        "compressor": [rec("A", "A") for _ in range(40)],
        "honest_1": [rec("A" if i % 2 else "B", "A" if (i + i // 7) % 2 else "B")
                     for i in range(40)],
        "honest_2": [rec("A" if i % 3 else "B", "B" if (i + 1) % 3 else "A")
                     for i in range(40)],
        "honest_3": [rec("B" if i % 2 else "A", "A" if (i * 3) % 5 < 2 else "B")
                     for i in range(40)],
    }
    out = jss_entropy_correlation(per_judge)
    assert out["pearson_r"] < 0
    assert out["jss"]["compressor"] == 1.0
    assert out["entropy_bits"]["compressor"] == 0.0


def test_correlation_needs_three_judges():
    with pytest.raises(ValueError):
        jss_entropy_correlation({"a": [rec("A", "A")], "b": [rec("A", "A")]})


# ── Full suite ───────────────────────────────────────────────────────────────

def test_compute_all_metrics_v2_shape():
    records = [rec(str(i % 5 + 1), str((i + i // 9) % 5 + 1),
                   item=f"i{i // 3}", pair=f"p{i}") for i in range(30)]
    out = compute_all_metrics_v2(records, cluster_unit="item", likert=True,
                                 n_bootstrap=100)
    assert out["cluster_unit"] == "item"
    assert out["jss_strict"]["cluster_unit"] == "item"
    assert "quadratic_weighted_kappa" in out
    assert "mean_absolute_flip" in out
    assert "decision_entropy_bits" in out


# ── kappa: UNCLEAR must not enter the chance baseline ────────────────────────
# Observed agreement never credits an UNCLEAR row, so leaving UNCLEAR in the
# marginals would inflate expected agreement with mass the observed term can
# never realise — depressing kappa in proportion to a judge's malformed rate
# rather than its inconsistency.

def test_kappa_ignores_unclear_rows_in_marginals():
    base = [rec("A", "A", f"i{i}") for i in range(10)] + \
           [rec("B", "B", f"i{i}") for i in range(10, 20)]
    k_clean = chance_corrected_jss(base, "disagree")
    padded = base + [rec("UNCLEAR", "UNCLEAR", f"u{i}") for i in range(10)]
    k_padded = chance_corrected_jss(padded, "disagree")
    assert k_clean == pytest.approx(k_padded), (
        "adding unparseable rows changed kappa; UNCLEAR is leaking into the "
        "chance baseline"
    )


def test_kappa_matches_between_policies_on_parseable_support():
    recs = [rec("A", "A", f"i{i}") for i in range(8)] + \
           [rec("B", "A", f"i{i}") for i in range(8, 12)] + \
           [rec("UNCLEAR", "B", f"u{i}") for i in range(4)]
    assert chance_corrected_jss(recs, "disagree") == pytest.approx(
        chance_corrected_jss(recs, "drop")
    )


def test_degenerate_single_label_judge_scores_zero_kappa():
    recs = [rec("A", "A", f"i{i}") for i in range(30)]
    assert chance_corrected_jss(recs, "disagree") == 0.0


# ── suite robustness ─────────────────────────────────────────────────────────

def test_compute_all_metrics_survives_all_unclear():
    recs = [rec("UNCLEAR", "UNCLEAR", f"i{i}") for i in range(12)]
    out = compute_all_metrics_v2(recs, cluster_unit="item", n_bootstrap=50)
    assert out["jss_drop"] is None
    assert out["jss_strict"]["estimate"] == 0.0


def test_chance_corrected_jss_carries_a_clustered_ci():
    recs = [rec("A", "A", f"i{i}") for i in range(10)] + \
           [rec("A", "B", f"i{i}") for i in range(10, 20)]
    out = compute_all_metrics_v2(recs, cluster_unit="item", n_bootstrap=200)
    cc = out["chance_corrected_jss"]
    assert isinstance(cc, dict), "chance-corrected score must ship with uncertainty"
    assert cc["cluster_unit"] == "item"
    assert cc["ci_lower"] <= cc["estimate"] <= cc["ci_upper"]
