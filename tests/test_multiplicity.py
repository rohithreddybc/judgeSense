"""The multiplicity plan must actually control what it claims to control.

PREREGISTRATION.md commits to Holm on four pooled task contrasts and
Benjamini-Hochberg at 10% FDR on the per-cell tier, and marked both
"NOT YET IMPLEMENTED". These tests are what make the commitment real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multiplicity import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_p_value,
    holm,
    minimum_detectable_effect,
    practically_meaningful,
)


# ── bootstrap p-values ───────────────────────────────────────────────────────

def test_p_value_is_never_zero():
    """Zero would claim more resolution than the resample count provides, and
    would sort to the top of any adjustment as if infinitely significant."""
    draws = [-1.0] * 2000
    assert bootstrap_p_value(draws) > 0


def test_p_value_is_small_when_the_distribution_excludes_the_null():
    assert bootstrap_p_value([-0.2] * 2000) < 0.01


def test_p_value_is_large_when_the_null_sits_mid_distribution():
    d = list(np.linspace(-1, 1, 2001))
    assert bootstrap_p_value(d) > 0.5


# ── Holm: family-wise error rate ─────────────────────────────────────────────

def test_holm_is_more_conservative_than_raw():
    raw = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
    out = holm(raw)
    for k in raw:
        assert out[k]["p_adj"] >= raw[k]


def test_holm_adjusted_values_never_decrease_down_the_order():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5})
    ordered = sorted(out.values(), key=lambda r: r["rank"])
    adj = [r["p_adj"] for r in ordered]
    assert adj == sorted(adj), f"non-monotone: {adj}"


def test_holm_rejects_nothing_when_every_test_is_null():
    out = holm({k: 0.6 for k in "abcd"})
    assert not any(r["reject"] for r in out.values())


def test_holm_controls_fwer_under_a_global_null():
    """The guarantee, checked by simulation rather than asserted: across many
    families of pure noise, the share of families with ANY rejection stays at
    or below alpha. This is the property that makes a task-level claim safe to
    quote."""
    rng = np.random.default_rng(0)
    families, alpha = 2000, 0.05
    any_reject = 0
    for _ in range(families):
        p = {f"t{i}": float(rng.uniform()) for i in range(4)}
        if any(r["reject"] for r in holm(p, alpha=alpha).values()):
            any_reject += 1
    assert any_reject / families <= alpha * 1.3, any_reject / families


# ── Benjamini-Hochberg: false discovery rate ─────────────────────────────────

def test_bh_is_less_conservative_than_holm():
    """Which is the point of using it for the exploratory tier."""
    raw = {f"c{i}": 0.001 * (i + 1) for i in range(100)}
    b = benjamini_hochberg(raw, fdr=0.10)
    h = holm(raw)
    assert sum(r["reject"] for r in b.values()) >= sum(r["reject"] for r in h.values())


def test_bh_adjusted_values_are_monotone():
    rng = np.random.default_rng(1)
    raw = {f"c{i}": float(rng.uniform()) for i in range(100)}
    out = benjamini_hochberg(raw)
    ordered = sorted(out.values(), key=lambda r: r["rank"])
    adj = [r["p_adj"] for r in ordered]
    assert adj == sorted(adj)


def test_bh_holds_fdr_under_a_global_null():
    rng = np.random.default_rng(2)
    fdr, runs, rates = 0.10, 400, []
    for _ in range(runs):
        raw = {f"c{i}": float(rng.uniform()) for i in range(100)}
        rejects = sum(r["reject"] for r in benjamini_hochberg(raw, fdr=fdr).values())
        rates.append(rejects / 100)
    assert float(np.mean(rates)) <= fdr * 1.2, float(np.mean(rates))


def test_a_hundred_cells_of_noise_would_otherwise_yield_false_findings():
    """The reason the plan exists, stated as the expectation it actually is.

    An unadjusted 0.05 rule over 100 null cells yields ~5 'findings' ON AVERAGE;
    any single draw can land anywhere, so averaging over many families is the
    honest check. 25 judges x 4 tasks is exactly this situation.
    """
    rng = np.random.default_rng(3)
    unadj, adj = [], []
    for _ in range(300):
        raw = {f"c{i}": float(rng.uniform()) for i in range(100)}
        unadj.append(sum(1 for p in raw.values() if p <= 0.05))
        adj.append(sum(r["reject"] for r in benjamini_hochberg(raw, fdr=0.10).values()))
    mean_unadj, mean_adj = float(np.mean(unadj)), float(np.mean(adj))
    assert 3.5 <= mean_unadj <= 6.5, mean_unadj      # ~5 by construction
    assert mean_adj < 1.0, mean_adj                  # BH removes essentially all
    assert mean_adj < mean_unadj


# ── smallest effect of interest, and power ───────────────────────────────────

def test_sesoi_is_independent_of_significance():
    assert practically_meaningful(0.019) is False
    assert practically_meaningful(-0.05) is True
    assert practically_meaningful(None) is None


def test_mde_grows_with_uncertainty_and_is_none_when_undefined():
    assert minimum_detectable_effect(0.02, 250) < minimum_detectable_effect(0.05, 250)
    assert minimum_detectable_effect(0.0, 250) is None
    assert minimum_detectable_effect(0.02, 1) is None


def test_nan_pvalues_do_not_silently_become_rejections():
    """A cell below the support floor has no p-value; it must not be counted as
    a test, nor reported as a finding."""
    out = holm({"a": 0.001, "below_floor": float("nan")})
    assert out["below_floor"]["reject"] is False
    assert out["a"]["n_tests"] == 1, "a NaN cell must not inflate the family size"


# ── discrimination ceiling ───────────────────────────────────────────────────

def test_discrimination_thresholds_match_the_preregistration():
    """The numbers are a pre-commitment; drifting them would be unfalsifiable."""
    from multiplicity import DISCRIMINATION_THRESHOLDS as T
    assert T == {"factuality": 0.75, "coherence": 0.40,
                 "relevance": 0.70, "preference": 0.65}


def test_a_task_below_its_threshold_permits_no_ranking():
    from multiplicity import discrimination_verdict
    v = discrimination_verdict("coherence", 0.30)
    assert v["verdict"] == "not_discriminating"
    assert v["ranking_permitted"] is False


def test_coherence_is_scored_against_the_majority_class_not_uniform_chance():
    """A constant-'4' judge scores 0.348 on the skewed gold, so quoting the 0.20
    a five-point scale suggests would overstate the margin by 15 points."""
    from multiplicity import discrimination_verdict
    v = discrimination_verdict("coherence", 0.452)
    assert v["majority_class"] == pytest.approx(0.348)
    assert v["margin_over_majority"] == pytest.approx(0.104, abs=1e-3)
    assert v["verdict"] == "discriminating"


def test_missing_accuracy_is_undetermined_not_a_pass():
    from multiplicity import discrimination_verdict
    v = discrimination_verdict("relevance", None)
    assert v["verdict"] == "undetermined"
    assert v["ranking_permitted"] is False


# ── transport control ────────────────────────────────────────────────────────

def test_transport_contrast_is_judged_against_the_sesoi_not_zero():
    """Two transports never agree exactly. An interval excluding zero below the
    smallest effect of interest says something about sample size, not about
    transports."""
    from multiplicity import transport_contrast
    close = transport_contrast(-0.206, -0.213)
    assert close["within_sesoi"] is True
    far = transport_contrast(-0.206, -0.260)
    assert far["within_sesoi"] is False
    assert "must be stated" in far["verdict"]


def test_transport_contrast_reports_the_shift_with_sign():
    from multiplicity import transport_contrast
    out = transport_contrast(-0.20, -0.25)
    assert out["transport_shift"] == pytest.approx(-0.05)


def test_transport_contrast_refuses_when_a_side_is_missing():
    from multiplicity import transport_contrast
    assert transport_contrast(None, -0.2)["comparable"] is False
    assert transport_contrast(-0.2, None)["comparable"] is False
