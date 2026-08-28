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
    # The ladder was llama-3.1 (8B/70B) until 2026-08-25, when the Groq key
    # started returning 403 on every request and llama3-70b was demoted to
    # unverified, taking the 70B rung with it. qwen-3 (8B/14B/32B, dense, all on
    # HuggingFace) replaces it and is a stronger ladder: three rungs rather than
    # two, one family, parameter count the only difference between them.
    # Restore the llama-3.1 assertion if that key is reissued.
    ladders = family_ladders()
    assert ladders, "no within-family ladder available; scale claims unsupportable"
    assert "qwen-3" in ladders
    assert ladders["qwen-3"] == ["qwen3-8b", "qwen3-14b", "qwen3-32b"]
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
    #
    # The assertion is on the FAMILY, not on the "deepseek" alias. That alias was
    # demoted to unverified on 2026-08-25: it emits an unterminated <think> trace
    # and never reaches a label inside the matched budget, so every call parses
    # to UNCLEAR. Pinning the test to one alias would either force a known-broken
    # judge back into selection or fail for the wrong reason.
    judges = reasoning_judges()
    assert len(judges) >= 3
    families = {JUDGES[j]["family"] for j in judges}
    assert any(f.startswith("deepseek") for f in families), (
        f"no DeepSeek-family reasoning judge is selectable; families={families}"
    )


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
    # 250 factuality + 250 coherence + 500 relevance + 212 preference = 1260
    # rows; pairwise tasks fold 2 orderings into 1 item -> 880 unique items.
    # Preference is 106 items, not 250: the decisive-vote label rule, the
    # contradictory-gold drop and the exact 50/50 winner-longer balance each
    # cut the human-labelled pool, and the loader reports every drop rather than
    # padding. See load_preference_items in src/data_sources.py.
    assert MAIN_AXIS_TOTAL_ROWS == 1260
    assert MAIN_AXIS_TOTAL_ITEMS == 880


def test_main_axis_run_plan_without_repeat_matches_rows_times_arms():
    plan = main_axis_run_plan(judges=["gpt-4o"], include_repeat_baseline=False)
    assert plan["calls_per_judge"] == 2520        # 1260 rows x 2 prompt arms
    assert plan["repeat_calls_per_judge"] == 0
    assert plan["calls_per_judge_with_repeat"] == 2520
    assert plan["dataset"]["include_repeat_baseline"] is False


def test_main_axis_run_plan_adds_two_repeat_calls_per_item():
    """Both templates are repeated, not one. A ceiling measured under a single
    template cannot absorb noise the other generates, so that noise would be
    charged to paraphrasing; and a stale factor here understates the printed
    budget, which is the number approved before any money is spent."""
    plan = main_axis_run_plan(judges=["gpt-4o"], include_repeat_baseline=True)
    assert plan["calls_per_judge"] == 2520
    assert plan["repeat_calls_per_judge"] == 1760   # 880 items x 2 arms
    assert plan["calls_per_judge_with_repeat"] == 4280
    assert plan["total_calls_with_repeat"] == 4280  # single judge here


def test_main_axis_run_plan_scales_across_default_verified_judges():
    plan = main_axis_run_plan(include_repeat_baseline=True)
    n = plan["n_judges"]
    assert plan["total_calls"] == 2520 * n
    assert plan["total_calls_with_repeat"] == 4280 * n
    assert plan["total_calls_with_repeat"] - plan["total_calls"] == 1760 * n


def test_equal_sized_entries_do_not_form_a_ladder():
    """qwen3.8-27b (Groq) and qwen3.8-27b-hf (HuggingFace) are the same 27B
    weights on two hosts. That is a provider contrast, not a scale contrast, and
    a scale claim must not be able to rest on it."""
    for family, members in family_ladders().items():
        sizes = [JUDGES[m]["size_b"] for m in members]
        assert len(set(sizes)) >= 2, (
            f"{family} is a degenerate ladder: sizes={sizes}, members={members}")
