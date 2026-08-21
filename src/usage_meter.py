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

import hashlib
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


def extract_served_model(provider: str, response: Any) -> Optional[str]:
    """The model string the PROVIDER echoed back, not the one we asked for.

    Most judge ids in the registry are floating aliases: an alias resolves to
    whatever snapshot the provider currently serves, so a replication a year
    from now can silently run a different model under the same name. Where the
    provider echoes a resolved id, recording it makes that drift detectable
    after the fact even though it cannot be prevented at request time.
    """
    if response is None:
        return None
    for attr in ("model", "model_version", "modelVersion"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


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


# The decoding configuration is ONE constant applied to every provider, because
# a judge-class comparison is uninterpretable if the classes were sampled
# differently. The earlier per-branch construction diverged on two axes at once:
# the Anthropic branch passed no temperature at all (defaulting to 1.0, which
# produced a 13.6% self-disagreement rate on byte-identical prompts and was very
# nearly read as a decoding-noise ceiling), HuggingFace used 0.01, and three of
# the five branches sent no system prompt -- so those judges were never given the
# instruction that suppresses the preamble the strict parser rejects, and their
# malformed output was charged as paraphrase disagreement.
#
# Every resolved parameter is recorded on the call record; nothing about the
# configuration is inferred from this source at analysis time.
TEMPERATURE = 0.0

# Models that reject an explicit temperature. The parameter is omitted and the
# omission is RECORDED, so the provider default is visible in the data rather
# than silently mixed in with the matched judges.
_NO_TEMPERATURE_PREFIXES = ("gpt-5",)


def accepts_temperature(model_id: str) -> bool:
    return not model_id.lower().startswith(_NO_TEMPERATURE_PREFIXES)


def decoding_config(model_id: str, max_tokens: int) -> Dict[str, Any]:
    """What was actually requested, for the record. Never re-derived later."""
    return {
        "temperature": TEMPERATURE if accepts_temperature(model_id) else None,
        "temperature_omitted_provider_default": not accepts_temperature(model_id),
        "max_tokens": max_tokens,
        "system_prompt_sent": True,
        "system_prompt_sha": hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest()[:12],
    }


def _request(provider: str, client, model_id: str, prompt: str, max_tokens: int) -> Tuple[str, Any]:
    """One SDK request; returns (text, raw_response).

    Every branch sends the same system prompt and the same temperature. Where a
    provider expresses those differently (Google takes a system instruction on
    the config object; the chat APIs take a system message), the transport
    differs but the request does not.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if provider in ("openai", "novita", "dashscope", "groq"):
        kwargs = {
            "model": model_id,
            "messages": messages,
            "timeout": _TIMEOUT,
            _openai_token_param(model_id): max_tokens,
        }
        if accepts_temperature(model_id):
            kwargs["temperature"] = TEMPERATURE
        r = client.chat.completions.create(**kwargs)
        if not getattr(r, "choices", None):
            return "", r
        return (r.choices[0].message.content or "").strip(), r
    if provider == "anthropic":
        r = client.messages.create(
            model=model_id, max_tokens=max_tokens, system=_SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return _first_text(getattr(r, "content", None)), r
    if provider == "huggingface":
        r = client.chat.completions.create(
            model=model_id, messages=messages,
            max_tokens=max_tokens, temperature=TEMPERATURE,
        )
        if not getattr(r, "choices", None):
            return "", r
        return (r.choices[0].message.content or "").strip(), r
    if provider == "google":
        from google.genai import types
        r = client.models.generate_content(
            model=model_id, contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens, temperature=TEMPERATURE,
                system_instruction=_SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (getattr(r, "text", None) or "").strip(), r
    if provider == "mistral":
        r = client.chat.complete(
            model=model_id, messages=messages,
            temperature=TEMPERATURE, max_tokens=max_tokens,
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
                "decoding": decoding_config(model_id, max_tokens),
                "model_id": model_id,
                "model_served": extract_served_model(provider, response),
                "provider": provider,
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
        "decoding": decoding_config(model_id, max_tokens),
        "model_id": model_id,
        "provider": provider,
        "finish_reason": None,
        "empty_content": None,
        "latency_ms": int((time.time() - t0) * 1000),
        "attempts": attempts,
        "error": str(last_exc),
    }
    return f"ERROR:{last_exc}"
