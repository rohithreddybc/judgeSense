"""
A provider refusal must be recorded as data, never crash and never masquerade
as a transport error or a format-following failure.

Observed live: claude-sonnet returns stop_reason="refusal" with an EMPTY content
list on 30% of the TREC-COVID relevance items, while claude-haiku and
claude-opus-4-7 return normal answers for the same prompts. Indexing that empty
list raised IndexError ("list index out of range"), which the retry wrapper
reported as an API error -- discarding the usage the provider had returned,
spending a second call on a deterministic outcome, and leaving the row
indistinguishable from a network failure.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import usage_meter as um  # noqa: E402


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_anthropic_refusal_yields_empty_text_not_an_exception(monkeypatch):
    refusal = _Obj(content=[], stop_reason="refusal",
                   usage=_Obj(input_tokens=1078, output_tokens=0))
    monkeypatch.setattr(um, "_request", lambda *a: (um._first_text(refusal.content), refusal))
    um.clear_last_meta()
    out = um.metered_call("anthropic", None, "m", "p", 20)
    meta = um.take_last_meta()
    assert out == "", "a refusal is empty output, not an ERROR: string"
    assert meta["error"] is None, "a refusal is not a transport failure"
    assert meta["attempts"] == 1, "a deterministic refusal must not be retried"
    assert meta["finish_reason"] == "refusal"
    assert meta["empty_content"] is True
    assert meta["input_tokens"] == 1078, "usage the provider reported must be kept"


def test_first_text_handles_empty_and_missing_content():
    assert um._first_text([]) == ""
    assert um._first_text(None) == ""
    assert um._first_text([_Obj(text="  A  ")]) == "A"
    assert um._first_text([_Obj(text=None)]) == ""


def test_openai_shaped_empty_choices_does_not_raise():
    assert um.extract_finish_reason("openai", _Obj(choices=[])) is None
    assert um.extract_finish_reason("anthropic", _Obj(stop_reason="refusal")) == "refusal"
    assert um.extract_finish_reason("openai", None) is None


def _regen():
    spec = importlib.util.spec_from_file_location(
        "regen", ROOT / "scripts" / "regenerate_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(i, finish_a):
    usage = lambda fr: {"input_tokens": 10, "output_tokens": 1, "finish_reason": fr,
                        "attempts": 1, "error": None, "latency_ms": 5}
    return {"pair_id": f"p{i}", "item_id": f"i{i}", "task_type": "relevance",
            "decision_a": "UNCLEAR" if finish_a == "refusal" else "A",
            "decision_b": "A", "ground_truth_position": "A",
            "ground_truth_label": "candidate_relevant", "error": None,
            "usage_a": usage(finish_a), "usage_b": usage("end_turn")}


def test_refusals_are_reported_separately_from_malformed_output():
    """Both collapse to UNCLEAR, so without this a safety behaviour is published
    as a format-following failure."""
    recs = [_rec(i, "refusal" if i < 5 else "end_turn") for i in range(10)]
    out = _regen().metrics_for_cell(recs, "relevance")
    assert out["n_refusals"] == 5
    assert out["n_metered_arms"] == 20
    assert out["refusal_rate"] == pytest.approx(0.25)
    assert out["malformed_rate"] > 0, "refused arms still count as malformed"


def test_refusal_rate_is_null_when_no_arm_carried_usage():
    """Runs predating usage metering must report null, not a confident zero."""
    recs = [{"pair_id": f"p{i}", "item_id": f"i{i}", "decision_a": "A",
             "decision_b": "A", "ground_truth_position": "A",
             "ground_truth_label": "candidate_relevant"} for i in range(4)]
    out = _regen().metrics_for_cell(recs, "relevance")
    assert out["refusal_rate"] is None
    assert out["n_metered_arms"] == 0


def test_reader_carries_usage_through_to_the_metrics(tmp_path):
    """The reader projects raw rows to a reduced schema; dropping usage there
    silently nulled the refusal rate even though the raw files had it."""
    import json
    p = tmp_path / "j_relevance.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps(_rec(i, "refusal")) + "\n")
    recs = _regen()._records(p)
    assert "usage_a" in recs[0] and recs[0]["usage_a"]["finish_reason"] == "refusal"
