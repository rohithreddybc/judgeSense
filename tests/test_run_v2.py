"""
Tests for the v2 runner (src/run_v2.py) — no network, no API keys.

Every judge call is monkeypatched, so these exercise the run/resume/parse/write
logic that must be correct BEFORE any paid run: resumability (a crash re-runs at
most one row), output schema, strict parsing, and error recording.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.run_v2 as run_v2  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """A 3-row factuality file and an isolated output dir."""
    data = tmp_path / "v2"
    data.mkdir()
    rows = [
        {"pair_id": "f1", "item_id": "i1", "prompt_pair_id": "i1#T1-T2",
         "task_type": "factuality", "ab_order": None, "ground_truth_label": "accurate",
         "prompt_a": "A? YES or NO", "prompt_b": "A'? YES or NO"},
        {"pair_id": "f2", "item_id": "i2", "prompt_pair_id": "i2#T1-T2",
         "task_type": "factuality", "ab_order": None, "ground_truth_label": "inaccurate",
         "prompt_a": "B? YES or NO", "prompt_b": "B'? YES or NO"},
        {"pair_id": "f3", "item_id": "i3", "prompt_pair_id": "i3#T1-T2",
         "task_type": "factuality", "ab_order": None, "ground_truth_label": "accurate",
         "prompt_a": "C? YES or NO", "prompt_b": "C'? YES or NO"},
    ]
    with open(data / "factuality.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    monkeypatch.setattr(run_v2, "_DATA_DIR", data)
    monkeypatch.setattr(run_v2, "_OUT_DIR", tmp_path / "out")
    # deterministic fake client + call
    monkeypatch.setattr(run_v2, "_resolve_client", lambda judge: ("CLIENT", "model-x", "openai"))
    monkeypatch.setattr(run_v2, "max_tokens_for", lambda judge, policy: 20)
    return tmp_path


def _answers(seq):
    """Return a _call stand-in yielding the given raw strings in order."""
    it = iter(seq)
    def fake(provider, client, model_id, prompt, max_tokens):
        return next(it)
    return fake


def test_run_cell_writes_one_record_per_row_with_decisions(tiny_dataset, monkeypatch):
    # a,b for each of 3 rows: agree, agree, disagree
    monkeypatch.setattr(run_v2, "_call",
                        _answers(["YES", "YES", "NO", "NO", "YES", "NO"]))
    stats = run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=None)
    assert stats == {"resumed": 0, "new": 3, "errors": 0, "total": 3}
    recs = [json.loads(l) for l in open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8")]
    assert [r["pair_id"] for r in recs] == ["f1", "f2", "f3"]
    assert (recs[0]["decision_a"], recs[0]["decision_b"]) == ("YES", "YES")
    assert (recs[2]["decision_a"], recs[2]["decision_b"]) == ("YES", "NO")
    assert all(r["error"] is None for r in recs)
    assert recs[0]["model"] == "gpt-4o" and recs[0]["max_tokens"] == 20


def test_resume_skips_completed_rows_and_only_runs_the_rest(tiny_dataset, monkeypatch):
    # first run: only f1 (limit via short answer stream would error; use limit=1)
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "YES"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=1)
    # second run over the full file: f1 must be skipped, only f2+f3 issue calls
    monkeypatch.setattr(run_v2, "_call", _answers(["NO", "NO", "YES", "NO"]))
    stats = run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=None)
    assert stats["resumed"] == 1 and stats["new"] == 2
    recs = [json.loads(l) for l in open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8")]
    assert [r["pair_id"] for r in recs] == ["f1", "f2", "f3"]  # f1 once, not duplicated


def test_api_error_is_recorded_not_raised(tiny_dataset, monkeypatch):
    monkeypatch.setattr(run_v2, "_call",
                        _answers(["YES", "ERROR:rate limit", "YES", "YES", "NO", "NO"]))
    stats = run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=None)
    assert stats["errors"] == 1 and stats["new"] == 3
    recs = [json.loads(l) for l in open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8")]
    assert recs[0]["error"] == "rate limit"
    assert recs[0]["decision_b"] == run_v2.UNCLEAR  # errored arm parses to UNCLEAR


def test_errored_row_is_reattempted_on_resume(tiny_dataset, monkeypatch):
    # f1 errors on arm b -> written with error -> NOT counted as completed
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "ERROR:x"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=1)
    # resume: f1 must be retried because its prior record carried an error
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "YES", "NO", "NO", "YES", "YES"]))
    stats = run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=None)
    assert stats["resumed"] == 0        # errored f1 is not a completed row
    assert stats["new"] == 3


def test_repeat_baseline_adds_a_third_arm(tiny_dataset, monkeypatch):
    # a,b,repeat per row x1 row
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "YES", "NO"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=True, limit=1)
    rec = json.loads(open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8").readline())
    assert rec["decision_a"] == "YES" and rec["decision_a_repeat"] == "NO"


def test_unparseable_output_is_unclear_not_a_crash(tiny_dataset, monkeypatch):
    monkeypatch.setattr(run_v2, "_call",
                        _answers(["I think probably yes and no", "YES", "NO", "NO", "YES", "YES"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=None)
    rec = json.loads(open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8").readline())
    assert rec["decision_a"] == run_v2.UNCLEAR   # ambiguous -> UNCLEAR, raw retained
    assert rec["error"] is None                  # UNCLEAR is a decision, not an error
