"""
Per-call token/latency metering for the v2 judge run.

`src/evaluate.py` (the v1 runner) discards the provider response object and
returns only the answer text, so token usage is unrecoverable after a run
without re-querying — and re-querying is exactly what this project cannot
afford. Rather than change v1's call path (its behaviour is pinned so published
runs stay reproducible), this module provides a parallel METERED call that makes
the same SDK request and additionally captures usage.

Contract, and the reason for it:

- `metered_call(...)` returns a bare string, exactly like `evaluate._call`, and
  writes the usage/timing for that call into the module slot `LAST_CALL_META`.
  Keeping the return type identical means `run_v2`'s tests, which monkeypatch
  the call seam with a string-returning stub, keep working untouched.
- Callers `clear_last_meta()` BEFORE the call and `take_last_meta()` after. The
  clear matters: with a patched call the slot is never written, so the caller
  reads `None` and records usage as null instead of silently attributing the
  PREVIOUS call's tokens to this one.
- A token count is NEVER estimated. If a provider does not return usage, the
  field is `None`. A plausible-looking fabricated number would be worse than a
  missing one, since it would silently enter the reported cost.

THREAD-SAFETY INVARIANT: `LAST_CALL_META` is process-global and NOT thread-safe.
It is correct only because the v2 runner is strictly sequential within a
process; parallelism is by disjoint (judge, task) sets in SEPARATE processes,
which do not share this slot. If the call loop is ever threaded inside one
process, usage will silently attach to the wrong arm. Do not thread it without
replacing this slot with a thread-local or an explicit return value.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

try:
    from .evaluate import _SYSTEM_PROMPT, _TIMEOUT, _openai_token_param
except ImportError:  # `python src/run_v2.py` invocation style
    from evaluate import _SYSTEM_PROMPT, _TIMEOUT, _openai_token_param  # type: ignore

LAST_CALL_META: Optional[Dict[str, Any]] = None


def clear_last_meta() -> None:
    global LAST_CALL_META
    LAST_CALL_META = None


def take_last_meta() -> Optional[Dict[str, Any]]:
    """Read and consume the slot, so meta can never be reused for a later call."""
    global LAST_CALL_META
    meta, LAST_CALL_META = LAST_CALL_META, None
    return meta


def _g(obj: Any, *path: str) -> Optional[Any]:
    """Walk an attribute path, returning None if any hop is missing."""
    cur = obj
    for name in path:
        cur = getattr(cur, name, None)
        if cur is None:
            return None
    return cur


def extract_usage(provider: str, response: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    (input_tokens, output_tokens) from a provider response, or (None, None).

    Each provider names these differently, and several return no usage at all on
    some backends (notably HuggingFace inference endpoints and some
    OpenAI-compatible gateways). Absent usage yields None, never a guess.
    """
    if response is None:
        return None, None
    if provider == "anthropic":
        return _g(response, "usage", "input_tokens"), _g(response, "usage", "output_tokens")
    if provider == "google":
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return None, None
        out = getattr(um, "candidates_token_count", None)
        thoughts = getattr(um, "thoughts_token_count", None)
        if out is not None and thoughts:
            out = out + thoughts  # reasoning tokens are billed output
        return getattr(um, "prompt_token_count", None), out
    # openai and every OpenAI-compatible gateway (novita, dashscope, groq),
    # plus mistral and huggingface chat.completions
    return _g(response, "usage", "prompt_tokens"), _g(response, "usage", "completion_tokens")


def extract_finish_reason(provider: str, response: Any) -> Optional[str]:
    """Why the provider stopped generating, or None if it did not say.

    Recorded because a judge that DECLINES an item is a different measurement
    from one whose answer failed to parse, and the two are indistinguishable
    once both have collapsed to UNCLEAR. Anthropic reports this as
    stop_reason="refusal" with an empty content list; OpenAI-shaped APIs use
    choices[0].finish_reason.
    """
    if response is None:
        return None
    if provider == "anthropic":
        return getattr(response, "stop_reason", None)
    if provider == "google":
        cands = getattr(response, "candidates", None) or []
        return str(getattr(cands[0], "finish_reason", None)) if cands else None
    choices = getattr(response, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


def _first_text(parts) -> str:
    """Text of the first content block, or "" when the provider returned none.

    An empty content list is a real response, not a malformed one: indexing it
    raised IndexError ("list index out of range"), which the retry wrapper then
    reported as an API error. That lost the usage the provider HAD returned,
    burned a second call on a deterministic outcome, and mislabelled a refusal
    as a transport failure.
    """
    if not parts:
        return ""
    text = getattr(parts[0], "text", None)
    return (text or "").strip()


def _request(provider: str, client, model_id: str, prompt: str, max_tokens: int) -> Tuple[str, Any]:
    """Make one SDK request; return (text, raw_response). Mirrors evaluate's
    per-provider calls exactly so the metered path is not a different
    experiment from the one the v1 code documents."""
    if provider in ("openai", "novita", "dashscope", "groq"):
        kwargs = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "timeout": _TIMEOUT,
            _openai_token_param(model_id): max_tokens,
        }
        if not model_id.lower().startswith("gpt-5"):
            kwargs["temperature"] = 0.0
        r = client.chat.completions.create(**kwargs)
        if not getattr(r, "choices", None):
            return "", r
        return (r.choices[0].message.content or "").strip(), r
    if provider == "anthropic":
        r = client.messages.create(
            model=model_id, max_tokens=max_tokens, system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _first_text(getattr(r, "content", None)), r
    if provider == "huggingface":
        r = client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.01,
        )
        if not getattr(r, "choices", None):
            return "", r
        return (r.choices[0].message.content or "").strip(), r
    if provider == "google":
        from google.genai import types
        r = client.models.generate_content(
            model=model_id, contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens, temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (getattr(r, "text", None) or "").strip(), r
    if provider == "mistral":
        r = client.chat.complete(
            model=model_id, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max_tokens,
        )
        if not getattr(r, "choices", None):
            return "", r
        return (r.choices[0].message.content or "").strip(), r
    raise ValueError(f"unknown provider {provider!r}")


def metered_call(provider: str, client, model_id: str, prompt: str, max_tokens: int) -> str:
    """
    One judge call with one retry, recording usage/timing into LAST_CALL_META.

    Returns the answer text, or "ERROR:<detail>" if both attempts failed —
    identical to `evaluate._call`, so this is a drop-in at the same seam. Meta is
    written on BOTH paths: a failed call still consumed time and possibly tokens,
    and omitting it would understate the spend.
    """
    global LAST_CALL_META
    t0 = time.time()
    attempts = 0
    last_exc: Optional[BaseException] = None
    for attempt in range(2):
        attempts += 1
        try:
            text, response = _request(provider, client, model_id, prompt, max_tokens)
            tin, tout = extract_usage(provider, response)
            LAST_CALL_META = {
                "input_tokens": tin,
                "output_tokens": tout,
                "finish_reason": extract_finish_reason(provider, response),
                "empty_content": text == "",
                "total_tokens": (tin + tout) if (tin is not None and tout is not None) else None,
                "latency_ms": int((time.time() - t0) * 1000),
                "attempts": attempts,
                "error": None,
            }
            return text
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            last_exc = exc
            if attempt == 0:
                time.sleep(5)
    LAST_CALL_META = {
        # Tokens burned by an attempt that raised before returning a response are
        # unrecoverable; they are represented by `attempts`, never fabricated.
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "empty_content": None,
        "latency_ms": int((time.time() - t0) * 1000),
        "attempts": attempts,
        "error": str(last_exc),
    }
    return f"ERROR:{last_exc}"
