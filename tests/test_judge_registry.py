"""Tests for src/judge_registry.py (reviewer points xmQT W1, WjHn W5/Q4, qkzU Q3)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge_registry import (  # noqa: E402
    JUDGES,
    MAIN_AXIS_TOTAL_ITEMS,
    MAIN_AXIS_TOTAL_ROWS,
    MATCHED_BUDGET_TOKENS,
    STRUCTURAL_AXIS_JUDGES,
    RegistryError,
    family_ladders,
    main_axis_run_plan,
    max_tokens_for,
    purpose_built_judges,
    reasoning_judges,
    run_plan,
    select_judges,
)


# ── matched token budget (xmQT W1, qkzU Q3) ──────────────────────────────────

def test_native_policy_reproduces_the_v1_budget_asymmetry():
    # The confound the reviewers identified, preserved so it can be compared
    # against rather than quietly erased.
    assert max_tokens_for("gpt-4o", "native") == 20
    assert max_tokens_for("deepseek", "native") == 1024


def test_matched_policy_equalizes_every_judge():
    budgets = {max_tokens_for(j, "matched") for j in JUDGES}
    assert budgets == {MATCHED_BUDGET_TOKENS}


def test_unknown_judge_or_policy_raises():
    with pytest.raises(RegistryError):
        max_tokens_for("nope", "native")
    with pytest.raises(RegistryError):
        max_tokens_for("gpt-4o", "unlimited")


# ── same-family ladders (WjHn W5/Q4) ─────────────────────────────────────────

def test_ladders_are_within_family_and_size_ordered():
    ladders = family_ladders()
    assert ladders, "no within-family ladder available; scale claims unsupportable"
    assert "llama-3.1" in ladders
    assert ladders["llama-3.1"] == ["llama3-8b", "llama3-70b"]
    for members in ladders.values():
        families = {JUDGES[m]["family"] for m in members}
        assert len(families) == 1
        sizes = [JUDGES[m]["size_b"] for m in members]
        assert sizes == sorted(sizes)


def test_ladders_exclude_judges_without_a_declared_size():
    for members in family_ladders().values():
        assert all(JUDGES[m]["size_b"] is not None for m in members)


# ── reasoning models (WjHn W3/Q3) ────────────────────────────────────────────

def test_multiple_reasoning_judges_are_available():
    # The v1 claim rested on DeepSeek-R1 alone; that was a scoping failure, not
    # a shortage of models.
    judges = reasoning_judges()
    assert len(judges) >= 3
    assert "deepseek" in judges


# ── purpose-built judges (xmQT Limitations) ──────────────────────────────────

def test_purpose_built_judges_are_registered_but_unverified():
    all_pb = purpose_built_judges(verified_only=False)
    assert {"prometheus-2-7b", "nemotron-70b"} <= set(all_pb)
    # None have been exercised against a provider yet, so none may be selected
    # silently.
    assert purpose_built_judges(verified_only=True) == []


def test_unverified_judges_are_rejected_at_selection_time():
    with pytest.raises(RegistryError, match="unverified"):
        select_judges(["prometheus-2-7b"])
    # ...but can be run deliberately.
    assert select_judges(["prometheus-2-7b"], allow_unverified=True) == ["prometheus-2-7b"]


def test_unknown_judge_rejected():
    with pytest.raises(RegistryError, match="unknown judge"):
        select_judges(["gpt-4o", "not-a-model"])


def test_default_selection_is_verified_only():
    assert all(JUDGES[j]["verified"] for j in select_judges())


# ── pre-registered structural subset ─────────────────────────────────────────

def test_structural_subset_is_six_judges_across_disjoint_families():
    assert len(STRUCTURAL_AXIS_JUDGES) == 6
    families = [JUDGES[j]["family"] for j in STRUCTURAL_AXIS_JUDGES]
    assert len(set(families)) == len(families), f"families repeat: {families}"
    select_judges(list(STRUCTURAL_AXIS_JUDGES))  # must all be valid and verified


# ── budget stated before it is spent ─────────────────────────────────────────

def test_run_plan_reports_total_calls_and_budgets():
    plan = run_plan(4200, list(STRUCTURAL_AXIS_JUDGES), budget_policy="matched")
    assert plan["n_judges"] == 6
    assert plan["total_calls"] == 25200          # docs/V2_1_STRUCTURAL_AXIS.md section 5
    assert set(plan["max_tokens"].values()) == {MATCHED_BUDGET_TOKENS}


# ── repeat-baseline cost stated in advance (docs/V2_1_STRUCTURAL_AXIS.md §7) ─

def test_run_plan_repeat_cost_defaults_to_zero_and_preserves_prior_behavior():
    # No repeat_calls_per_judge passed -> byte-identical to the pre-existing
    # contract: calls_per_judge/total_calls unaffected by the new parameter.
    plan = run_plan(4200, list(STRUCTURAL_AXIS_JUDGES), budget_policy="matched")
    assert plan["repeat_calls_per_judge"] == 0
    assert plan["calls_per_judge_with_repeat"] == plan["calls_per_judge"] == 4200
    assert plan["total_calls_with_repeat"] == plan["total_calls"] == 25200


def test_run_plan_states_repeat_arm_cost_explicitly():
    plan = run_plan(3000, ["gpt-4o", "claude-haiku"], repeat_calls_per_judge=1000)
    assert plan["repeat_calls_per_judge"] == 1000
    assert plan["calls_per_judge"] == 3000                   # base unchanged
    assert plan["total_calls"] == 6000                       # base unchanged
    assert plan["calls_per_judge_with_repeat"] == 4000
    assert plan["total_calls_with_repeat"] == 8000


def test_main_axis_dataset_shape_matches_documented_composition():
    # 250 factuality + 250 coherence + 500 relevance + 452 preference = 1452
    # rows; pairwise tasks fold 2 orderings into 1 item -> 976 unique items.
    # Preference is 226 items, not 250: the split is held at an exact 50/50
    # winner-longer balance and the pool's smaller bucket has only 113 pairs.
    assert MAIN_AXIS_TOTAL_ROWS == 1452
    assert MAIN_AXIS_TOTAL_ITEMS == 976


def test_main_axis_run_plan_without_repeat_matches_rows_times_arms():
    plan = main_axis_run_plan(judges=["gpt-4o"], include_repeat_baseline=False)
    assert plan["calls_per_judge"] == 2904        # 1452 rows x 2 prompt arms
    assert plan["repeat_calls_per_judge"] == 0
    assert plan["calls_per_judge_with_repeat"] == 2904
    assert plan["dataset"]["include_repeat_baseline"] is False


def test_main_axis_run_plan_adds_two_repeat_calls_per_item():
    """Both templates are repeated, not one. A ceiling measured under a single
    template cannot absorb noise the other generates, so that noise would be
    charged to paraphrasing; and a stale factor here understates the printed
    budget, which is the number approved before any money is spent."""
    plan = main_axis_run_plan(judges=["gpt-4o"], include_repeat_baseline=True)
    assert plan["calls_per_judge"] == 2904
    assert plan["repeat_calls_per_judge"] == 1952   # 976 items x 2 arms
    assert plan["calls_per_judge_with_repeat"] == 4856
    assert plan["total_calls_with_repeat"] == 4856  # single judge here


def test_main_axis_run_plan_scales_across_default_verified_judges():
    plan = main_axis_run_plan(include_repeat_baseline=True)
    n = plan["n_judges"]
    assert plan["total_calls"] == 2904 * n
    assert plan["total_calls_with_repeat"] == 4856 * n
    assert plan["total_calls_with_repeat"] - plan["total_calls"] == 1952 * n
