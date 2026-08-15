"""
Tests for src/polarity.py — reviewer xmQT W3 (handle polarity-inverted
templates by remapping labels instead of dropping them).

Decision strings here are TEST FIXTURES, never model outputs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics_v2 import jss  # noqa: E402
from src.polarity import (  # noqa: E402
    CANONICAL_LABELS,
    DIRECT,
    INVERTED,
    INVERTED_TEMPLATES,
    UNCLEAR,
    PolarityError,
    assert_remap_is_bijective,
    canonicalize,
    canonicalize_records,
    has_inverted_arm,
    split_by_polarity,
    template_remap,
)


def rec(a, b, task="factuality", pa=DIRECT, pb=DIRECT, item="i0"):
    return {
        "task_type": task, "decision_a": a, "decision_b": b,
        "polarity_a": pa, "polarity_b": pb,
        "template_a_id": None if pa == DIRECT else "T_INV_1",
        "template_b_id": None if pb == DIRECT else "T_INV_1",
        "item_id": item,
    }


# ── remap contracts ──────────────────────────────────────────────────────────

def test_all_remaps_are_bijections_onto_the_canonical_space():
    for task in CANONICAL_LABELS:
        assert_remap_is_bijective(task)


def test_inverted_factuality_flips_yes_and_no():
    assert canonicalize("factuality", "YES", INVERTED, "T_INV_1") == "inaccurate"
    assert canonicalize("factuality", "NO", INVERTED, "T_INV_1") == "accurate"
    assert canonicalize("factuality", "YES", DIRECT) == "accurate"
    assert canonicalize("factuality", "NO", DIRECT) == "inaccurate"


def test_inverted_coherence_reverses_the_scale():
    assert canonicalize("coherence", "1", INVERTED, "T_INV_1") == "5"
    assert canonicalize("coherence", "5", INVERTED, "T_INV_1") == "1"
    assert canonicalize("coherence", "3", INVERTED, "T_INV_1") == "3"


def test_unclear_passes_through_unconverted():
    # An unparseable answer has no polarity to correct; converting it would
    # manufacture a decision that was never made.
    assert canonicalize("factuality", UNCLEAR, INVERTED, "T_INV_1") == UNCLEAR
    assert canonicalize("factuality", None, DIRECT) == UNCLEAR


def test_out_of_space_answer_is_unclear_not_guessed():
    assert canonicalize("factuality", "MAYBE", DIRECT) == UNCLEAR


def test_unknown_task_or_polarity_raises():
    with pytest.raises(PolarityError):
        template_remap("relevance", DIRECT)
    with pytest.raises(PolarityError):
        template_remap("factuality", "sideways")


# ── the actual reviewer point ────────────────────────────────────────────────

def test_remapping_rescues_a_consistent_judge_that_raw_scoring_calls_unstable():
    """
    A judge answering both arms correctly should score JSS 1.0.

    Raw, it scores 0.0 — YES to "is this accurate?" and NO to "does this contain
    errors?" are the SAME judgment expressed in opposite label conventions. This
    is exactly why v1's inverted template was a defect, and why dropping those
    pairings (v1's fix) inflates measured consistency instead of testing it.
    """
    raw = [rec("YES", "NO", pa=DIRECT, pb=INVERTED, item=f"i{i}") for i in range(20)]
    assert jss(raw, "disagree") == 0.0

    canon = canonicalize_records(raw)
    assert jss(canon, "disagree") == 1.0
    assert canon[0]["raw_decision_a"] == "YES"      # raw preserved for audit
    assert canon[0]["decision_a"] == "accurate"


def test_remapping_does_not_hide_a_genuinely_inconsistent_judge():
    # Judge says accurate on the direct arm and ALSO "has errors" on the
    # inverted arm: a real contradiction, which must survive remapping.
    raw = [rec("YES", "YES", pa=DIRECT, pb=INVERTED, item=f"i{i}") for i in range(20)]
    canon = canonicalize_records(raw)
    assert jss(canon, "disagree") == 0.0


def test_direct_only_pairings_are_unaffected_by_canonicalization():
    raw = [rec("YES", "YES", item=f"i{i}") for i in range(10)]
    raw += [rec("YES", "NO", item=f"j{i}") for i in range(10)]
    before = jss(raw, "disagree")
    after = jss(canonicalize_records(raw), "disagree")
    assert before == after == 0.5


# ── reporting contract ───────────────────────────────────────────────────────

def test_split_by_polarity_separates_the_harder_pairings():
    records = [rec("YES", "NO", item=f"d{i}") for i in range(6)]
    records += [rec("YES", "NO", pb=INVERTED, item=f"x{i}") for i in range(4)]
    parts = split_by_polarity(records)
    assert len(parts["direct_only"]) == 6
    assert len(parts["inverted_involving"]) == 4
    assert len(parts["all"]) == 10
    assert all(not has_inverted_arm(r) for r in parts["direct_only"])


def test_dropping_inverted_pairings_raises_measured_consistency():
    """
    The concrete form of xmQT's suspicion: excluding the inverted arms can only
    move JSS up, because they are the pairings a judge is most likely to fail.
    """
    records = [rec("YES", "YES", item=f"d{i}") for i in range(10)]          # agree
    records += [rec("YES", "YES", pb=INVERTED, item=f"x{i}") for i in range(10)]  # contradict
    canon = canonicalize_records(records)
    parts = split_by_polarity(canon)
    jss_direct_only = jss(parts["direct_only"], "disagree")
    jss_all = jss(parts["all"], "disagree")
    assert jss_direct_only == 1.0
    assert jss_all == 0.5
    assert jss_direct_only > jss_all


# ── template hygiene ─────────────────────────────────────────────────────────

def test_inverted_templates_declare_polarity_and_ship_their_remap():
    # v1's defect was an inverted template existing while its remap did not.
    for task, specs in INVERTED_TEMPLATES.items():
        for spec in specs:
            assert spec["polarity"] == INVERTED
            assert spec["remap"], f"{task}/{spec['template_id']} has no remap"
            assert "{text}" in spec["text"]
            assert set(spec["remap"].values()) == set(CANONICAL_LABELS[task])
