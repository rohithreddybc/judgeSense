"""Tests for src/repeat_baseline.py (all call/record inputs are fixtures)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.repeat_baseline import (  # noqa: E402
    REPEAT_ARM,
    RepeatBaselineError,
    build_repeat_pairs,
    jss_repeat_delta,
    repeat_baseline_jss,
)


def call(item, idx, decision="YES", arm=REPEAT_ARM, judge="gpt-4o"):
    return {"judge": judge, "item_id": item, "task": "factuality",
            "arm": arm, "repeat_index": idx, "decision": decision}


# ── build_repeat_pairs: happy path ───────────────────────────────────────────

def test_build_repeat_pairs_basic_shape():
    calls = [call("i0", 1, "YES"), call("i0", 2, "NO")]
    pairs = build_repeat_pairs(calls)
    assert pairs == [{
        "decision_a": "YES",
        "decision_b": "NO",
        "item_id": "i0",
        "repeat_pair_id": "i0",
        "arm_a": REPEAT_ARM,
        "arm_b": REPEAT_ARM,
    }]


def test_build_repeat_pairs_ignores_non_repeat_calls():
    # A runner's full call log includes single-shot arms (P1/P2, S1-S5) that
    # never set repeat_index; those must pass through untouched.
    calls = [
        {"judge": "gpt-4o", "item_id": "i0", "arm": "P1", "decision": "YES"},
        {"judge": "gpt-4o", "item_id": "i0", "arm": "P2", "decision": "YES",
         "repeat_index": None},
        call("i0", 1, "YES"),
        call("i0", 2, "YES"),
    ]
    pairs = build_repeat_pairs(calls)
    assert len(pairs) == 1
    assert pairs[0]["item_id"] == "i0"


def test_build_repeat_pairs_multiple_items():
    calls = [call("i0", 1, "YES"), call("i0", 2, "NO"),
              call("i1", 1, "NO"), call("i1", 2, "NO")]
    pairs = build_repeat_pairs(calls)
    by_item = {p["item_id"]: p for p in pairs}
    assert by_item["i0"]["decision_a"] == "YES" and by_item["i0"]["decision_b"] == "NO"
    assert by_item["i1"]["decision_a"] == "NO" and by_item["i1"]["decision_b"] == "NO"


def test_build_repeat_pairs_consumable_by_repeat_baseline_jss():
    calls = [call(f"i{k}", 1, "YES") for k in range(10)] + \
            [call(f"i{k}", 2, "YES" if k < 8 else "NO") for k in range(10)]
    pairs = build_repeat_pairs(calls)
    assert repeat_baseline_jss(pairs) == pytest.approx(0.8)


# ── build_repeat_pairs: contract violations ─────────────────────────────────

def test_rejects_repeat_index_on_non_s0_arm():
    calls = [call("i0", 1, arm="P1"), call("i0", 2, arm="P1")]
    with pytest.raises(RepeatBaselineError, match="S0"):
        build_repeat_pairs(calls)


def test_rejects_invalid_repeat_index():
    calls = [call("i0", 1), call("i0", 3)]
    with pytest.raises(RepeatBaselineError, match="repeat_index"):
        build_repeat_pairs(calls)


def test_rejects_duplicate_repeat_index_for_same_item():
    calls = [call("i0", 1, "YES"), call("i0", 1, "NO")]
    with pytest.raises(RepeatBaselineError, match="duplicate"):
        build_repeat_pairs(calls)


def test_rejects_incomplete_pair():
    calls = [call("i0", 1, "YES")]  # missing repeat_index=2
    with pytest.raises(RepeatBaselineError, match="missing"):
        build_repeat_pairs(calls)


def test_rejects_call_missing_item_id():
    calls = [{"arm": REPEAT_ARM, "repeat_index": 1, "decision": "YES"}]
    with pytest.raises(RepeatBaselineError, match="item_id"):
        build_repeat_pairs(calls)


def test_empty_call_log_yields_no_pairs():
    assert build_repeat_pairs([]) == []


# ── contract re-exports from src.metrics_v2 are wired correctly ────────────

def test_repeat_baseline_module_reexports_delta_computation():
    para = [{"decision_a": "A", "decision_b": "A", "item_id": f"i{k}"} for k in range(5)]
    repeat_calls = [call(f"i{k}", 1, "A") for k in range(5)] + \
                   [call(f"i{k}", 2, "A") for k in range(5)]
    repeat_pairs = build_repeat_pairs(repeat_calls)
    out = jss_repeat_delta(para, repeat_pairs, cluster_unit="item", n_bootstrap=50)
    assert out["jss"] == pytest.approx(1.0)
    assert out["jss_rep"] == pytest.approx(1.0)
    assert out["delta"] == pytest.approx(0.0)
