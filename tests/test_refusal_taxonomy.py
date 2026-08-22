"""
JSS must be computed on its proper support, and refusal reported as its own
construct rather than folded into disagreement.

A refusal is upstream of any judgement: the provider halted before the model
rendered a verdict. Scoring it as paraphrase disagreement asserts the judge
produced two conflicting judgements, which it did not. Scoring it as a third
label is worse -- it awards JSS 1.0 to a judge that refuses everything.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _regen():
    spec = importlib.util.spec_from_file_location(
        "regen", ROOT / "scripts" / "regenerate_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _u(finish):
    return {"input_tokens": 10, "output_tokens": 1, "finish_reason": finish,
            "attempts": 1, "error": None, "latency_ms": 5}


def _rec(i, a="YES", b="YES", fa="end_turn", fb="end_turn"):
    return {"pair_id": f"p{i}", "item_id": f"i{i}", "task_type": "factuality",
            "decision_a": a, "decision_b": b, "ground_truth_label": "accurate",
            "error": None, "usage_a": _u(fa), "usage_b": _u(fb)}


def test_a_judge_that_refuses_everything_does_not_score_perfect_agreement():
    """The degenerate case the taxonomy exists to prevent."""
    recs = [_rec(i, "UNCLEAR", "UNCLEAR", "refusal", "refusal") for i in range(20)]
    out = _regen().metrics_for_cell(recs, "factuality")
    assert out["n_pairs_both_answered"] == 0
    assert out["consistent_refusal_rate"] == 1.0
    assert out["jss_strict"] != 1.0, "refusing everything must not read as perfect stability"


def test_jss_is_computed_over_verdict_pairs_only():
    # 10 clean agreeing pairs + 10 pairs where one arm was refused
    recs = [_rec(i) for i in range(10)]
    recs += [_rec(100 + i, "YES", "UNCLEAR", "end_turn", "refusal") for i in range(10)]
    out = _regen().metrics_for_cell(recs, "factuality")
    assert out["n_pairs_both_answered"] == 10
    assert out["jss_support"] == "verdict_pairs"
    assert out["jss_strict"] == 1.0, "the ten answered pairs all agree"
    assert out["refusal_discordance_rate"] == 0.5


def test_refusal_inclusive_sensitivity_is_reported_and_is_more_punitive():
    recs = [_rec(i) for i in range(10)]
    recs += [_rec(100 + i, "YES", "UNCLEAR", "end_turn", "refusal") for i in range(10)]
    out = _regen().metrics_for_cell(recs, "factuality")
    assert out["jss_strict_refusal_inclusive"] == pytest.approx(0.5)
    assert out["jss_strict_refusal_inclusive"] < out["jss_strict"], (
        "the inclusive figure must be visible so the conditioning can be checked")


def test_discordant_refusal_is_counted_as_a_sensitivity_signal():
    """One arm refused, the other judged: the rewording changed willingness to
    judge at all, which is the finding, not an inconvenience."""
    recs = [_rec(i, "YES", "UNCLEAR", "end_turn", "refusal") for i in range(4)]
    recs += [_rec(50 + i) for i in range(6)]
    out = _regen().metrics_for_cell(recs, "factuality")
    assert out["refusal_discordance_rate"] == pytest.approx(0.4)
    assert out["consistent_refusal_rate"] == 0.0


def test_malformed_output_is_not_reclassified_as_refusal():
    """A completed but unparseable answer stays malformed; only a
    provider-flagged decline counts as refusal."""
    recs = [_rec(i, "UNCLEAR", "UNCLEAR", "end_turn", "end_turn") for i in range(10)]
    out = _regen().metrics_for_cell(recs, "factuality")
    assert out["refusal_rate"] == 0.0
    assert out["n_pairs_both_answered"] == 10
    assert out["malformed_rate"] == 1.0
    assert out["jss_strict_refusal_inclusive"] is None, "no refusals: no separate figure needed"


def test_runs_without_usage_metadata_behave_exactly_as_before():
    recs = [{"pair_id": f"p{i}", "item_id": f"i{i}", "decision_a": "YES",
             "decision_b": "YES", "ground_truth_label": "accurate"} for i in range(10)]
    out = _regen().metrics_for_cell(recs, "factuality")
    assert out["n_pairs_both_answered"] == 10
    assert out["refusal_rate"] is None, "absent metering is unknown, not zero refusals"
    assert out["jss_strict"] == 1.0


# ── regression: a decline must be recognised on EVERY provider ───────────────
# Matching only Anthropic's "refusal" made an OpenAI-compatible decline classify
# as a verdict; its empty content parsed to UNCLEAR and was then charged as
# paraphrase disagreement, which is precisely what the taxonomy exists to
# prevent. Those judges would have reported refusal_rate 0.0 regardless.

@pytest.mark.parametrize("reason", [
    "refusal", "content_filter", "safety", "recitation", "blocklist",
    "SAFETY", "Content_Filter", "  refusal  ",
])
def test_every_provider_decline_spelling_counts_as_refusal(reason):
    regen = _regen()
    rec = _rec(0, "UNCLEAR", "UNCLEAR", reason, reason)
    assert regen._arm_refused(rec, "a"), f"{reason!r} must count as a refusal"
    assert regen._pair_class(rec) == "both_refused"


@pytest.mark.parametrize("reason", ["end_turn", "stop", "max_tokens", "length", None, 123])
def test_normal_terminations_are_not_refusals(reason):
    regen = _regen()
    assert not regen._arm_refused(_rec(0, fa=reason), "a")


def test_openai_decline_is_not_charged_as_paraphrase_disagreement():
    """The end-to-end consequence of the bug this guards."""
    regen = _regen()
    recs = [_rec(i, "UNCLEAR", "UNCLEAR", "content_filter", "content_filter")
            for i in range(10)]
    out = regen.metrics_for_cell(recs, "factuality")
    assert out["n_pairs_both_answered"] == 0, "declines must not be scored as verdicts"
    assert out["refusal_rate"] == 1.0
    assert out["consistent_refusal_rate"] == 1.0
