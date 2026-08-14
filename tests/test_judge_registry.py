"""Tests for src/judge_registry.py (reviewer points xmQT W1, WjHn W5/Q4, qkzU Q3)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge_registry import (  # noqa: E402
    JUDGES,
    MATCHED_BUDGET_TOKENS,
    STRUCTURAL_AXIS_JUDGES,
    RegistryError,
    family_ladders,
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
