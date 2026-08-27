"""
Tests for token/cost instrumentation (src/usage_meter.py, scripts/summarize_usage.py).

No network, no API keys: provider responses are canned stubs. These prove the
properties that matter for a one-shot paid run — usage is captured, never
fabricated, never attributed to the wrong call, and failed/retried calls are
still counted as spend.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import usage_meter as um  # noqa: E402


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _openai_like(pt, ct, text="YES"):
    return _Obj(choices=[_Obj(message=_Obj(content=text))],
                usage=_Obj(prompt_tokens=pt, completion_tokens=ct))


# ── per-provider extraction; absent usage must yield None, never a guess ─────

@pytest.mark.parametrize("provider", ["openai", "novita", "dashscope", "groq", "mistral", "huggingface"])
def test_openai_shaped_providers(provider):
    assert um.extract_usage(provider, _openai_like(11, 3)) == (11, 3)


def test_anthropic_usage_fields():
    r = _Obj(usage=_Obj(input_tokens=42, output_tokens=7))
    assert um.extract_usage("anthropic", r) == (42, 7)


def test_google_usage_metadata_and_reasoning_tokens():
    r = _Obj(usage_metadata=_Obj(prompt_token_count=30, candidates_token_count=5,
                                 thoughts_token_count=12))
    # reasoning tokens are billed as output, so they are added rather than dropped
    assert um.extract_usage("google", r) == (30, 17)


def test_missing_usage_is_none_not_zero():
    # A provider that returns no usage must not be recorded as "0 tokens" —
    # zero is a measurement, None is an absence, and the summary distinguishes them.
    assert um.extract_usage("openai", _Obj(choices=[], usage=None)) == (None, None)
    assert um.extract_usage("anthropic", _Obj()) == (None, None)
    assert um.extract_usage("google", _Obj()) == (None, None)
    assert um.extract_usage("openai", None) == (None, None)


# ── metered_call writes meta on success and on failure ──────────────────────

def test_metered_call_records_usage_on_success(monkeypatch):
    monkeypatch.setattr(um, "_request", lambda *a: ("YES", _openai_like(100, 4)))
    um.clear_last_meta()
    out = um.metered_call("openai", None, "m", "p", 20)
    assert out == "YES"
    meta = um.take_last_meta()
    assert (meta["input_tokens"], meta["output_tokens"], meta["total_tokens"]) == (100, 4, 104)
    assert meta["attempts"] == 1 and meta["error"] is None
    assert meta["latency_ms"] >= 0


def test_failed_call_still_records_spend(monkeypatch):
    calls = {"n": 0}
    def boom(*a):
        calls["n"] += 1
        raise RuntimeError("rate limit")
    monkeypatch.setattr(um, "_request", boom)
    monkeypatch.setattr(um.time, "sleep", lambda s: None)  # don't wait in tests
    um.clear_last_meta()
    out = um.metered_call("openai", None, "m", "p", 20)
    assert out.startswith("ERROR:")
    meta = um.take_last_meta()
    # every attempt was made and paid for in wall-clock; tokens are unknowable
    assert meta["attempts"] == um._MAX_ATTEMPTS and calls["n"] == um._MAX_ATTEMPTS
    assert meta["input_tokens"] is None and meta["output_tokens"] is None
    assert "rate limit" in meta["error"]


def test_rate_limits_back_off_further_each_time_and_others_do_not():
    """A 429 is a request to wait, not a failure.

    Retrying it on the same flat timer as a transport error burns the row: a
    single live sweep recorded 118 rate-limit errors from Mistral and 92 from
    Novita, each an errored arm a later resume pass had to pay for again. The
    wait has to actually grow, and only for quota rejections.
    """
    limit = RuntimeError("Error code: 429 - rate limit reached")
    other = RuntimeError("Connection reset by peer")

    assert um._is_rate_limit(limit)
    assert not um._is_rate_limit(other)

    waits = [um._backoff_seconds(limit, i) for i in range(um._MAX_ATTEMPTS)]
    assert waits == sorted(waits), f"rate-limit back-off must not shrink: {waits}"
    assert waits[0] < waits[-1], f"back-off never grows: {waits}"

    flat = {um._backoff_seconds(other, i) for i in range(um._MAX_ATTEMPTS)}
    assert len(flat) == 1, f"transport retry should stay flat, got {flat}"


def test_rate_limit_detection_spans_provider_spellings():
    """Each SDK words it differently; missing one silently reverts that
    provider to the flat retry."""
    for msg in ("Error code: 429 - rate limit reached",
                'API error occurred: Status 429. Body: {"message":"Rate limit exceeded"}',
                "RATE_LIMIT_EXCEEDED",
                "Too Many Requests"):
        assert um._is_rate_limit(RuntimeError(msg)), msg
    for msg in ("500 MODEL_NOT_AVAILABLE", "Connection reset by peer",
                "invalid_request_error"):
        assert not um._is_rate_limit(RuntimeError(msg)), msg


def test_take_consumes_so_meta_cannot_be_reused(monkeypatch):
    monkeypatch.setattr(um, "_request", lambda *a: ("YES", _openai_like(5, 1)))
    um.metered_call("openai", None, "m", "p", 20)
    assert um.take_last_meta() is not None
    assert um.take_last_meta() is None, "second read must not re-report the same call"


def test_stubbed_call_seam_yields_null_usage_not_stale_usage(monkeypatch):
    """The runner's tests patch the call seam with a string-returning stub. The
    clear-before-call must ensure that yields None rather than silently
    attributing the PREVIOUS real call's tokens to this one."""
    monkeypatch.setattr(um, "_request", lambda *a: ("YES", _openai_like(999, 999)))
    um.metered_call("openai", None, "m", "p", 20)   # a real call leaves meta set
    um.LAST_CALL_META = {"input_tokens": 999}       # simulate a stale slot
    um.clear_last_meta()
    assert um.take_last_meta() is None


# ── aggregator: spend counts every attempt, scoring does not ────────────────

def test_summary_counts_superseded_records_as_spend(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    u = lambda i, o: {"input_tokens": i, "output_tokens": o, "total_tokens": i + o,
                      "latency_ms": 10, "attempts": 1, "error": None}
    rows = [
        # first attempt errored, then the same pair was retried successfully:
        # BOTH were paid for, so both must appear in the spend total.
        {"pair_id": "p1", "model": "j", "task_type": "factuality",
         "usage_a": u(10, 1), "usage_b": u(10, 1), "error": "boom"},
        {"pair_id": "p1", "model": "j", "task_type": "factuality",
         "usage_a": u(10, 1), "usage_b": u(10, 1), "error": None},
    ]
    with open(raw / "j_factuality.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "su", Path(__file__).resolve().parent.parent / "scripts" / "summarize_usage.py")
    su = importlib.util.module_from_spec(spec); spec.loader.exec_module(su)
    su.OUT = tmp_path / "usage.json"
    su.main(["--raw", str(raw)])

    s = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    o = s["overall"]
    assert o["calls"] == 4, "all four arm-calls were paid for"
    assert o["input_tokens"] == 40
    assert o["superseded_records"] == 1, "the retried row leaves one shadowed record"
    assert o["cost"] is None, "no price table supplied -> cost must be null, never guessed"


def test_cost_requires_a_price_table_and_flags_partial_data(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    rows = [
        {"pair_id": "p1", "model": "j", "task_type": "factuality",
         "usage_a": {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                     "latency_ms": 5, "attempts": 1, "error": None},
         "usage_b": None,          # provider returned no usage
         "error": None},
    ]
    with open(raw / "j_factuality.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"j": {"input_per_mtok": 2.0, "output_per_mtok": 10.0}}),
                      encoding="utf-8")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "su", Path(__file__).resolve().parent.parent / "scripts" / "summarize_usage.py")
    su = importlib.util.module_from_spec(spec); spec.loader.exec_module(su)
    su.OUT = tmp_path / "usage.json"
    su.main(["--raw", str(raw), "--prices", str(prices)])

    s = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    cost = s["per_judge"]["j"]["cost"]
    assert cost["usd"] == pytest.approx(12.0)      # 1M in @2 + 1M out @10
    assert cost["lower_bound"] is True, "a call with unknown usage means this understates spend"
    assert s["per_judge"]["j"]["calls_missing_usage"] == 1
