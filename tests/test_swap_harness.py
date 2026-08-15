"""Tests for src/swap_harness.py (all judged records are test fixtures)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.swap_harness import (  # noqa: E402
    POSITION_INCONSISTENT,
    collapse_presentations,
    content_decision,
    position_bias_corrected_jss,
    position_bias_rate,
)

ORIG_MAP = {"A": "candidate_1", "B": "candidate_2"}
SWAP_MAP = {"A": "candidate_2", "B": "candidate_1"}


def presentations(pair_id, da_orig, db_orig, da_swap, db_swap):
    """Both presentations of one prompt pair with given positional answers."""
    return [
        {"prompt_pair_id": pair_id, "item_id": f"item_{pair_id}",
         "ab_order": "original", "candidate_map": ORIG_MAP,
         "decision_a": da_orig, "decision_b": db_orig},
        {"prompt_pair_id": pair_id, "item_id": f"item_{pair_id}",
         "ab_order": "swapped", "candidate_map": SWAP_MAP,
         "decision_a": da_swap, "decision_b": db_swap},
    ]


# ── content_decision ─────────────────────────────────────────────────────────

def test_faithful_judge_maps_to_content():
    # Judge prefers candidate_1: answers A in original order, B in swapped.
    assert content_decision("A", "B", ORIG_MAP, SWAP_MAP) == "candidate_1"
    assert content_decision("B", "A", ORIG_MAP, SWAP_MAP) == "candidate_2"


def test_position_follower_is_inconsistent():
    # Always-A: picks candidate_1 in original order but candidate_2 swapped.
    assert content_decision("A", "A", ORIG_MAP, SWAP_MAP) == POSITION_INCONSISTENT


def test_unparseable_positional_answer_is_inconsistent():
    assert content_decision("UNCLEAR", "B", ORIG_MAP, SWAP_MAP) == POSITION_INCONSISTENT
    assert content_decision(None, "B", ORIG_MAP, SWAP_MAP) == POSITION_INCONSISTENT


# ── collapse + corrected JSS ─────────────────────────────────────────────────

def test_always_a_judge_gets_zero_corrected_jss():
    judged = []
    for k in range(20):
        judged += presentations(f"p{k}", "A", "A", "A", "A")
    # Raw positional agreement would be perfect; corrected JSS is 0.
    assert position_bias_corrected_jss(judged) == 0.0
    diag = position_bias_rate(judged)
    assert diag["first_position_rate"] == 1.0
    assert diag["inconsistency_rate"] == 1.0


def test_faithful_consistent_judge_gets_perfect_corrected_jss():
    judged = []
    for k in range(20):
        # Prefers candidate_1 under both templates, tracks it across orderings.
        judged += presentations(f"p{k}", "A", "A", "B", "B")
    assert position_bias_corrected_jss(judged) == 1.0
    diag = position_bias_rate(judged)
    assert diag["inconsistency_rate"] == 0.0
    assert diag["first_position_rate"] == 0.5  # A half the time, by design


def test_template_disagreement_counts_against_corrected_jss():
    judged = []
    for k in range(10):
        # Template A picks candidate_1, template B picks candidate_2 —
        # both position-consistent, but the templates disagree.
        judged += presentations(f"p{k}", "A", "B", "B", "A")
    assert position_bias_corrected_jss(judged) == 0.0
    assert position_bias_rate(judged)["inconsistency_rate"] == 0.0


def test_mixed_population():
    judged = []
    judged += presentations("p1", "A", "A", "B", "B")  # faithful agree
    judged += presentations("p2", "A", "A", "A", "A")  # position follower
    judged += presentations("p3", "A", "B", "B", "A")  # template disagreement
    judged += presentations("p4", "B", "B", "A", "A")  # faithful agree on cand_2
    assert position_bias_corrected_jss(judged) == pytest.approx(0.5)


def test_missing_ordering_is_an_error_not_a_skip():
    judged = presentations("p1", "A", "A", "B", "B")[:1]  # original only
    with pytest.raises(ValueError, match="missing ordering"):
        collapse_presentations(judged)


def test_collapsed_records_keep_cluster_keys():
    judged = presentations("p9", "A", "A", "B", "B")
    collapsed = collapse_presentations(judged)
    assert collapsed[0]["item_id"] == "item_p9"
    assert collapsed[0]["prompt_pair_id"] == "p9"
    assert collapsed[0]["decision_a"] == "candidate_1"
