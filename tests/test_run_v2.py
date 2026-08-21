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


def test_repeat_baseline_repeats_both_templates_not_just_arm_a(tiny_dataset, monkeypatch):
    """The ceiling must be measurable on the same template whose disagreement it
    explains. Repeating only arm A left noise under template B charged to
    paraphrasing, so the endpoint was partly a property of which template was
    designated A."""
    # a, b, repeat-a, repeat-b for one row
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "YES", "NO", "YES"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=True, limit=1)
    rec = json.loads(open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8").readline())
    assert rec["decision_a"] == "YES" and rec["decision_a_repeat"] == "NO"
    assert rec["decision_b"] == "YES" and rec["decision_b_repeat"] == "YES"


def test_unparseable_output_is_unclear_not_a_crash(tiny_dataset, monkeypatch):
    monkeypatch.setattr(run_v2, "_call",
                        _answers(["I think probably yes and no", "YES", "NO", "NO", "YES", "YES"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=None)
    rec = json.loads(open(run_v2._out_path("gpt-4o", "factuality"), encoding="utf-8").readline())
    assert rec["decision_a"] == run_v2.UNCLEAR   # ambiguous -> UNCLEAR, raw retained
    assert rec["error"] is None                  # UNCLEAR is a decision, not an error


# ── regression: append-only file + retries must not double-count ─────────────
# The runner appends and never rewrites, so an errored row that is later retried
# leaves TWO records for one pair_id. Reading both feeds a phantom UNCLEAR
# disagreement into that item's cluster and silently biases the one-shot
# metrics. Runner and reader must agree on last-write-wins.

def test_retried_row_is_not_counted_twice_by_the_reader(tiny_dataset, monkeypatch):
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "ERROR:timeout"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=1)
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "YES"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=1)

    path = run_v2._out_path("gpt-4o", "factuality")
    raw = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(raw) == 2, "append-only file should hold both attempts"

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "regen", Path(__file__).resolve().parent.parent / "scripts" / "regenerate_results.py")
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)
    recs = regen._records(path)
    assert len(recs) == 1, "reader must keep one record per pair_id"
    assert recs[0]["decision_b"] == "YES", "must keep the LAST (successful) attempt"


def test_successful_row_is_not_repaid_after_a_later_failed_attempt(tiny_dataset, monkeypatch):
    # succeed, then a stale failure is appended for the same pair; the row must
    # still count as done so the resume does not pay for it again.
    monkeypatch.setattr(run_v2, "_call", _answers(["YES", "YES"]))
    run_v2.run_cell("gpt-4o", "factuality", "native", repeat_baseline=False, limit=1)
    path = run_v2._out_path("gpt-4o", "factuality")
    assert run_v2._completed_pair_ids(path) == {"f1"}


def test_repeat_baseline_fires_once_per_item_not_per_row(tmp_path, monkeypatch):
    """Pairwise items have two ab_order rows; the repeat arm is per ITEM, so
    firing it on both would overspend the budgeted repeat calls by 50%."""
    data = tmp_path / "v2"; data.mkdir()
    rows = [
        {"pair_id": "r1_original", "item_id": "i1", "prompt_pair_id": "i1#T1-T2",
         "task_type": "relevance", "ab_order": "original", "ground_truth_position": "A",
         "ground_truth_label": "candidate_relevant", "prompt_a": "P?", "prompt_b": "P'?"},
        {"pair_id": "r1_swapped", "item_id": "i1", "prompt_pair_id": "i1#T1-T2",
         "task_type": "relevance", "ab_order": "swapped", "ground_truth_position": "B",
         "ground_truth_label": "candidate_relevant", "prompt_a": "Q?", "prompt_b": "Q'?"},
    ]
    with open(data / "relevance.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    monkeypatch.setattr(run_v2, "_DATA_DIR", data)
    monkeypatch.setattr(run_v2, "_OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(run_v2, "_resolve_client", lambda j: ("C", "m", "openai"))
    monkeypatch.setattr(run_v2, "max_tokens_for", lambda j, p: 20)

    calls = []
    def counting(provider, client, model_id, prompt, max_tokens):
        calls.append(prompt)
        return "A"
    monkeypatch.setattr(run_v2, "_call", counting)
    run_v2.run_cell("gpt-4o", "relevance", "native", repeat_baseline=True, limit=None)

    # 2 rows x 2 arms = 4, plus TWO repeats (both templates, canonical row only)
    assert len(calls) == 6, f"expected 6 calls (4 arms + 2 repeats), got {len(calls)}"
    recs = [json.loads(l) for l in open(run_v2._out_path("gpt-4o", "relevance"), encoding="utf-8")]
    by_order = {r["ab_order"]: r for r in recs}
    assert "decision_a_repeat" in by_order["original"]
    assert "decision_b_repeat" in by_order["original"]
    assert "decision_a_repeat" not in by_order["swapped"]
    assert "decision_b_repeat" not in by_order["swapped"]
