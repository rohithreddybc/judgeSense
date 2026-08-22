"""
Preconditions that must hold before a paid sweep starts.

Every one of these guards a failure that costs money rather than correctness,
and every one was missing while the suite was green. The runner's own tests
monkeypatch `_resolve_client`, so a judge that existed in the selection registry
but not in the client registry passed every test and would have aborted the
sweep after seven judges had been paid for.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import _append_jsonl, _load_jsonl, _RATE_LIMIT  # noqa: E402
from src.judge_registry import JUDGES, select_judges  # noqa: E402
from src.models import SUPPORTED_MODELS  # noqa: E402
import src.run_v2 as run_v2  # noqa: E402


# ── the two registries must agree, or the sweep dies mid-run ────────────────

def test_every_selectable_judge_is_resolvable_to_a_client():
    """run_v2 selects from JUDGES but resolves through SUPPORTED_MODELS. A name
    in the first and not the second raises KeyError inside the loop, after
    earlier judges have already been billed."""
    missing = sorted(set(select_judges(None)) - set(SUPPORTED_MODELS))
    assert not missing, f"selectable but unresolvable: {missing}"


@pytest.mark.parametrize("judge", sorted(set(select_judges(None)) & set(SUPPORTED_MODELS)))
def test_the_two_registries_describe_the_same_model(judge):
    reg, sup = JUDGES[judge], SUPPORTED_MODELS[judge]
    assert reg["provider"] == sup["provider"]
    assert reg["model_id"] == sup["model_id"]
    assert reg["key"] == sup["key"]


# ── the approved number must describe the approved run ──────────────────────

def test_planned_calls_tracks_tasks_and_limit():
    """The printed figure is the approval gate. Built from module constants it
    overstated a single-task run several-fold and ignored --limit entirely."""
    one = run_v2._planned_calls(["gpt-4o"], ["factuality"], None, False)
    assert one["total"] == 250 * 2

    limited = run_v2._planned_calls(["gpt-4o"], ["factuality"], 5, False)
    assert limited["total"] == 5 * 2

    full = run_v2._planned_calls(["gpt-4o"], list(run_v2.TASKS), None, True)
    assert full["total"] == 4280, "1260 rows x 2 arms + 880 items x 2 repeats"


def test_planned_calls_scales_with_judges():
    one = run_v2._planned_calls(["gpt-4o"], list(run_v2.TASKS), None, True)
    three = run_v2._planned_calls(["gpt-4o", "claude-haiku", "gemini-flash"],
                                  list(run_v2.TASKS), None, True)
    assert three["total"] == 3 * one["total"]


def test_planned_calls_equals_what_run_cell_actually_issues(tmp_path, monkeypatch):
    """Derived-versus-actual, on the real dataset files."""
    calls = {"n": 0}

    def counting(provider, client, model_id, prompt, max_tokens):
        calls["n"] += 1
        return "YES"

    monkeypatch.setattr(run_v2, "_OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(run_v2, "_resolve_client", lambda j: ("C", "m", "openai"))
    monkeypatch.setattr(run_v2, "_call", counting)
    monkeypatch.setattr(run_v2.time, "sleep", lambda s: None)

    predicted = run_v2._planned_calls(["gpt-4o"], ["factuality"], 20, True)
    run_v2.run_cell("gpt-4o", "factuality", "matched", repeat_baseline=True, limit=20)
    assert calls["n"] == predicted["total"]


# ── a broken cell must cost that cell, not the sweep ────────────────────────

def test_a_cell_that_keeps_erroring_aborts_instead_of_paying_through(tmp_path, monkeypatch):
    calls = {"n": 0}

    def always_failing(provider, client, model_id, prompt, max_tokens):
        calls["n"] += 1
        return "ERROR:429 rate limit exceeded"

    monkeypatch.setattr(run_v2, "_OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(run_v2, "_resolve_client", lambda j: ("C", "m", "groq"))
    monkeypatch.setattr(run_v2, "_call", always_failing)
    monkeypatch.setattr(run_v2.time, "sleep", lambda s: None)

    with pytest.raises(run_v2.RunAborted):
        run_v2.run_cell("gpt-4o", "factuality", "matched", repeat_baseline=False, limit=None)
    # 250 rows are available; the breaker must stop far short of paying for them
    assert calls["n"] <= run_v2.MAX_CONSECUTIVE_ERRORS * 2 + 2
    assert calls["n"] < 250, "aborted before working the whole cell"


def test_rate_limited_providers_are_paced():
    """Groq, Novita and DashScope free tiers 429 without spacing."""
    for provider in ("groq", "novita", "dashscope"):
        assert _RATE_LIMIT.get(provider, 0) > 0, f"{provider} has no pacing"


# ── paid records must survive a hard kill ───────────────────────────────────

def test_a_torn_final_line_does_not_swallow_the_next_record(tmp_path):
    """A process killed mid-write leaves a record with no trailing newline.
    Appending blindly fuses the next record onto it, so the reader drops BOTH:
    the interrupted row and a row that was actually paid for."""
    path = tmp_path / "out.jsonl"
    _append_jsonl({"pair_id": "a", "v": 1}, path)
    _append_jsonl({"pair_id": "b", "v": 2}, path)

    raw = path.read_bytes()
    path.write_bytes(raw[:-12])          # simulate SIGKILL mid-write

    _append_jsonl({"pair_id": "c", "v": 3}, path)

    recs = _load_jsonl(path)
    ids = [r.get("pair_id") for r in recs]
    assert "a" in ids, "the intact earlier record must survive"
    assert "c" in ids, "the new record must not be fused onto the torn line"


def test_appends_are_flushed_to_disk(tmp_path):
    path = tmp_path / "out.jsonl"
    _append_jsonl({"pair_id": "a"}, path)
    assert path.exists() and path.stat().st_size > 0


# ── reproducibility evidence ────────────────────────────────────────────────

def test_every_judge_declares_whether_it_is_version_pinned():
    for name, spec in JUDGES.items():
        assert "pinned" in spec, f"{name} does not declare pinned"


@pytest.mark.parametrize("judge", sorted(JUDGES))
def test_pinned_matches_the_model_id(judge):
    spec = JUDGES[judge]
    looks_pinned = (bool(re.search(r"\d{4}-\d{2}-\d{2}|-\d{8}\b", spec["model_id"]))
                    and "latest" not in spec["model_id"])
    assert spec["pinned"] == looks_pinned, (
        f"{judge}: pinned={spec['pinned']} but model_id={spec['model_id']!r}")
