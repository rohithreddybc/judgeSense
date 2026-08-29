"""Batching a judge is only safe if the batch cannot leak or misalign.

The Anthropic API key has no credit, so Claude judges are reachable only through
Claude Code subagents. One label per subagent costs 43,115 tokens against ~413
via the API; the overhead is fixed per subagent, so batching amortises it
(measured: 41,017 tokens for twenty items -- essentially the same as for one).

Batching buys that at the price of three risks. These tests are what make the
price payable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claude_code_judge import (  # noqa: E402
    BatchResponseError, build_batch_prompt, make_batches, parse_batch_response,
    provenance, units_for_task,
)


def _rows(n=40):
    return [{"pair_id": f"p{i:04d}", "prompt_a": f"A prompt {i}",
             "prompt_b": f"B prompt {i}"} for i in range(n)]


# ── the contamination invariant ──────────────────────────────────────────────

def test_no_batch_ever_holds_both_arms_of_one_item():
    """The whole measurement is agreement between an item's two arms. A judge
    that sees both in one context could reconcile them, which would manufacture
    agreement that the API path could never produce."""
    batches = make_batches(units_for_task(_rows(60)), batch_size=12, seed=7)
    for i, batch in enumerate(batches):
        pids = [u["pair_id"] for u in batch]
        assert len(pids) == len(set(pids)), f"batch {i} holds a repeated pair_id"


def test_every_unit_is_placed_exactly_once():
    units = units_for_task(_rows(50))
    batches = make_batches(units, batch_size=16, seed=3)
    placed = [u["id"] for b in batches for u in b]
    assert sorted(placed) == sorted(u["id"] for u in units)
    assert len(placed) == len(set(placed)), "a unit was duplicated across batches"


def test_batches_respect_the_size_cap():
    for b in make_batches(units_for_task(_rows(70)), batch_size=10, seed=1):
        assert len(b) <= 10


def test_position_within_a_batch_is_recorded():
    """So a position effect is measurable afterwards rather than assumed absent."""
    for b in make_batches(units_for_task(_rows(30)), batch_size=8, seed=5):
        for j, u in enumerate(b):
            assert u["batch_position"] == j
            assert isinstance(u["batch_index"], int)


def test_batching_is_deterministic_for_a_seed():
    a = make_batches(units_for_task(_rows(40)), batch_size=9, seed=11)
    b = make_batches(units_for_task(_rows(40)), batch_size=9, seed=11)
    assert [[u["id"] for u in x] for x in a] == [[u["id"] for u in y] for y in b]


# ── alignment: the failure that would corrupt a cell silently ────────────────

def test_a_short_response_raises_instead_of_shifting_labels():
    """Trusting order would assign item 2's answer to item 1 and shift every
    later label by one. Every subsequent verdict in the cell would be wrong and
    nothing would look broken."""
    ids = [f"p{i}#a" for i in range(5)]
    text = "\n".join(json.dumps({"id": i, "answer": "YES"}) for i in ids[:4])
    with pytest.raises(BatchResponseError, match="missing"):
        parse_batch_response(text, ids)


def test_unexpected_ids_raise():
    ids = ["p1#a", "p2#a"]
    text = "\n".join(json.dumps({"id": i, "answer": "NO"})
                     for i in ["p1#a", "p2#a", "p99#a"])
    with pytest.raises(BatchResponseError):
        parse_batch_response(text, ids)


def test_answers_are_keyed_by_id_not_by_order():
    ids = ["p1#a", "p2#a", "p3#a"]
    shuffled = [{"id": "p3#a", "answer": "C"}, {"id": "p1#a", "answer": "A"},
                {"id": "p2#a", "answer": "B"}]
    got = parse_batch_response("\n".join(json.dumps(o) for o in shuffled), ids)
    assert got == {"p1#a": "A", "p2#a": "B", "p3#a": "C"}


def test_surrounding_prose_and_code_fences_are_tolerated():
    """Agents wrap output. Being strict about ids while lenient about framing is
    the combination that survives real responses."""
    ids = ["p1#a", "p2#a"]
    text = ("Here are the results:\n```\n"
            + json.dumps({"id": "p1#a", "answer": "YES"}) + "\n"
            + json.dumps({"id": "p2#a", "answer": "NO"}) + "\n```\nDone.")
    assert parse_batch_response(text, ids) == {"p1#a": "YES", "p2#a": "NO"}


# ── the prompt carries what the judge needs ──────────────────────────────────

def test_prompt_names_every_id_and_demands_independence():
    batch = make_batches(units_for_task(_rows(6)), batch_size=6, seed=2)[0]
    p = build_batch_prompt(batch)
    for u in batch:
        assert u["id"] in p
    assert "ENTIRELY ON ITS OWN" in p
    assert str(len(batch)) in p


# ── provenance must not impersonate the API path ─────────────────────────────

def test_provenance_declares_itself_incomparable_to_api_judges():
    """Temperature is not exposed and the system prompt is the harness's. A
    record shaped like an API usage record would make two different
    measurements indistinguishable downstream."""
    pr = provenance("claude-opus-5", 50, 42)
    assert pr["transport"] == "claude_code_subagent"
    assert pr["comparable_to_api_judges"] is False
    assert pr["temperature"] is None
    assert pr["system_prompt_sha"] is None
    assert pr["system_prompt_sent"] is False


# ── the bug that would have deleted the endpoint ─────────────────────────────

def test_base_arm_strips_a_suffix_not_a_character_set():
    """str.rstrip('_repeat') removes any of {_,r,e,p,a,t} from the right, so
    'a_repeat' becomes '' and 'a' becomes ''. The prompt lookup then read
    row['prompt_'], every repeat unit was silently dropped, and the run would
    have produced no ceiling and therefore no dJSS at all."""
    from claude_code_judge import base_arm
    assert base_arm("a") == "a"
    assert base_arm("b") == "b"
    assert base_arm("a_repeat") == "a"
    assert base_arm("b_repeat") == "b"
    assert "a_repeat".rstrip("_repeat") == "", "the original bug, pinned"


def test_repeat_units_actually_get_built():
    units = units_for_task(_rows(5), arms=("a", "b", "a_repeat", "b_repeat"))
    assert len(units) == 20, f"expected 4 arms x 5 rows, got {len(units)}"
    assert sum(1 for u in units if u["arm"].endswith("_repeat")) == 10


def test_a_repeat_unit_carries_the_identical_prompt():
    """A repeat that re-issues a DIFFERENT string is not a repeat; the ceiling
    would then measure wording sensitivity too, and dJSS would understate."""
    units = {u["id"]: u for u in
             units_for_task(_rows(3), arms=("a", "a_repeat"))}
    for pid in ("p0000", "p0001", "p0002"):
        assert units[f"{pid}#a"]["prompt"] == units[f"{pid}#a_repeat"]["prompt"]


# ── the cancellation claim, made true ────────────────────────────────────────

def test_repeat_batches_mirror_arm_batches_exactly():
    """dJSS subtracts JSS_rep from JSS_para, so batch context cancels ONLY if a
    repeat sits among the same neighbours, in the same position, as the unit it
    baselines. Independently shuffled repeats would leave that difference inside
    the endpoint -- the very confound the repeat arm exists to remove."""
    from claude_code_judge import mirror_repeat_batches
    arm_batches = make_batches(units_for_task(_rows(40)), batch_size=10, seed=4)
    rep_batches = mirror_repeat_batches(arm_batches)

    assert len(rep_batches) == len(arm_batches)
    for arm_b, rep_b in zip(arm_batches, rep_batches):
        assert len(arm_b) == len(rep_b)
        for a, r in zip(arm_b, rep_b):
            assert r["pair_id"] == a["pair_id"]
            assert r["batch_position"] == a["batch_position"]
            assert r["batch_index"] == a["batch_index"]
            assert r["prompt"] == a["prompt"]
            assert r["mirrors"] == a["id"]


def test_repeat_batches_inherit_the_no_both_arms_invariant():
    from claude_code_judge import mirror_repeat_batches
    for batch in mirror_repeat_batches(
            make_batches(units_for_task(_rows(50)), batch_size=12, seed=9)):
        pids = [u["pair_id"] for u in batch]
        assert len(pids) == len(set(pids))
