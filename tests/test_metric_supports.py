"""
Statistics printed side by side must describe the same rows, and an estimand
must not be published under a design that cannot identify it.

Each test below corresponds to a defect that produced a wrong published number
while the suite was green. The common shape: two quantities computed on
different supports, printed adjacently with no support column, so the reader
performs an arithmetic check that cannot succeed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.structural_variants import UNCLEAR  # noqa: E402


def _regen():
    spec = importlib.util.spec_from_file_location(
        "regen", ROOT / "scripts" / "regenerate_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _u(finish="end_turn", error=None):
    return {"input_tokens": 10, "output_tokens": 1, "finish_reason": finish,
            "attempts": 1, "error": error, "latency_ms": 5}


def _rec(i, a="YES", b="YES", ua=None, ub=None, task="factuality", **kw):
    r = {"pair_id": f"p{i}", "item_id": f"i{i}", "task_type": task,
         "decision_a": a, "decision_b": b, "ground_truth_label": "accurate",
         "ground_truth_position": "A", "error": None, "ab_order": None,
         "usage_a": ua or _u(), "usage_b": ub or _u()}
    r.update(kw)
    return r


# ── a dead call is not a format failure ─────────────────────────────────────

def test_transport_failure_is_its_own_outcome_not_malformed_output():
    """A call that never completed is published as the judge's malformed rate
    and charged as paraphrase disagreement. A 529 storm would appear in the
    paper as a format-following defect."""
    regen = _regen()
    dead = _u(finish=None, error="connection reset")
    recs = [_rec(i, UNCLEAR, UNCLEAR, ua=dead, ub=dead) for i in range(10)]
    assert regen._pair_class(recs[0]) == "transport_error"
    out = regen.metrics_for_cell(recs, "factuality")
    assert out["n_pairs_transport_error"] == 10
    assert out["n_pairs_both_answered"] == 0


def test_a_missing_finish_reason_is_not_silently_a_verdict():
    regen = _regen()
    rec = _rec(0, UNCLEAR, UNCLEAR, ua=_u(finish=None), ub=_u(finish=None))
    assert regen._pair_class(rec) == "transport_error"


# ── kappa must correct a number that is actually shown ──────────────────────

def test_kappa_publishes_its_own_support():
    """Cohen's kappa drops pairs containing UNCLEAR; jss(..., "disagree") keeps
    them as mismatches. Printed adjacently, a kappa of 0.82 sat beside a JSS of
    0.51 and read as "almost all agreement is non-chance" -- the opposite of
    what happened."""
    regen = _regen()
    recs = [_rec(i, "YES", "YES") for i in range(5)]
    recs += [_rec(50 + i, "NO", "NO") for i in range(5)]
    recs += [_rec(100 + i, UNCLEAR, UNCLEAR) for i in range(90)]
    out = regen.metrics_for_cell(recs, "factuality")
    assert out["chance_corrected_jss_n"] == 10, "kappa saw only the parseable pairs"
    assert out["n_pairs_both_answered"] == 100
    assert out["jss_strict"] == pytest.approx(0.10, abs=0.01)
    assert out["jss_on_parseable_pairs"] == pytest.approx(1.0), (
        "the JSS on kappa's own support must be published beside it")


def test_the_printed_histogram_reproduces_the_printed_entropy():
    """Computed on different supports, the histogram did not reproduce the
    entropy, and a reader checking the arithmetic found a discrepancy with no
    explanation available in the artifact."""
    import math
    regen = _regen()
    recs = [_rec(i, "YES", "YES") for i in range(30)]
    recs += [_rec(100 + i, "NO", "NO") for i in range(10)]
    out = regen.metrics_for_cell(recs, "factuality")
    hist = out["label_histogram"]
    total = sum(hist.values())
    expected = -sum((v / total) * math.log2(v / total) for v in hist.values() if v)
    assert out["decision_entropy_bits"] == pytest.approx(expected, abs=0.01)


# ── per-arm rates must share the pooled rate's definition ───────────────────

def test_per_arm_malformed_rates_exclude_refusals_like_the_pooled_rate():
    """Three keys sharing a prefix carried two definitions; on one cell arm A
    read 0.57 against a pooled 0.29, doubling that template's apparent
    format-failure rate."""
    regen = _regen()
    refused = _u(finish="refusal")
    recs = [_rec(i, UNCLEAR, "YES", ua=refused) for i in range(10)]
    recs += [_rec(50 + i, "YES", "YES") for i in range(10)]
    out = regen.metrics_for_cell(recs, "factuality")
    assert out["malformed_rate_arm_a"] == 0.0, "refused arms are not format failures"
    assert out["unclear_rate_arm_a"] > 0.0, "the raw UNCLEAR rate stays available"


# ── the endpoint must not be published under a design that cannot identify it ─

def test_delta_is_withheld_when_the_ceiling_covers_only_one_arm():
    """A ceiling measured under one template cannot separate paraphrase
    sensitivity from template-specific decoding noise. The pooling fix landed in
    code while every shipped record still carried arm A only, so the confound
    was published silently."""
    regen = _regen()
    recs = [_rec(i, "YES", "YES", decision_a_repeat="YES") for i in range(150)]
    out = regen.metrics_for_cell(recs, "factuality")["jss_repeat_delta"]
    assert out.get("delta") is None
    assert out.get("ceiling_single_arm") is True
    assert "one prompt arm only" in out["delta_withheld_reason"]


def test_delta_is_emitted_once_both_arms_are_present():
    regen = _regen()
    recs = [_rec(i, "YES", "YES", decision_a_repeat="YES", decision_b_repeat="YES")
            for i in range(150)]
    out = regen.metrics_for_cell(recs, "factuality")["jss_repeat_delta"]
    assert out.get("delta") is not None
    assert set(out["repeat_agreement_by_arm"]) == {"a", "b"}
    assert out["arm_ceiling_gap"] == pytest.approx(0.0)


def test_a_refused_repeat_call_is_not_scored_as_self_disagreement():
    """A perfectly stable judge with half its repeats refused reported a delta
    of +0.50 with an interval excluding zero."""
    regen = _regen()
    recs = []
    for i in range(150):
        refused = i % 2 == 0
        recs.append(_rec(
            i, "YES", "YES",
            decision_a_repeat=UNCLEAR if refused else "YES",
            decision_b_repeat="YES",
            usage_a_repeat=_u(finish="refusal") if refused else _u(),
            usage_b_repeat=_u(),
        ))
    out = regen.metrics_for_cell(recs, "factuality")["jss_repeat_delta"]
    assert out.get("jss_rep") == pytest.approx(1.0), (
        "refused repeats must be dropped, not counted as the judge disagreeing "
        "with itself")


def test_item_loss_is_measured_against_the_cell_the_judge_started_with():
    """Measured against the post-filter set, refusal loss was invisible and the
    preregistered 50% ceiling could never fire."""
    regen = _regen()
    refused = _u(finish="refusal")
    recs = [_rec(i, "YES", "YES", decision_a_repeat="YES", decision_b_repeat="YES")
            for i in range(120)]
    recs += [_rec(500 + i, UNCLEAR, UNCLEAR, ua=refused, ub=refused)
             for i in range(200)]
    out = regen.metrics_for_cell(recs, "factuality")["jss_repeat_delta"]
    assert out.get("delta") is None
    assert "lack a usable pair" in (out.get("reason") or ""), out
