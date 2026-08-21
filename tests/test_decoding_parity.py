"""
Every judge must be sampled the same way, or a judge-class comparison is
uninterpretable.

The earlier per-branch construction diverged on two axes simultaneously: the
Anthropic branch passed no temperature (defaulting to 1.0 -- a 13.6%
self-disagreement rate on byte-identical prompts, which was very nearly reported
as a decoding-noise ceiling), HuggingFace used 0.01, and three of five branches
sent no system prompt at all, so those judges never received the instruction
that suppresses the preamble the strict parser rejects.
"""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import usage_meter as um  # noqa: E402

PROVIDERS = ["openai", "novita", "dashscope", "groq", "anthropic",
             "huggingface", "google", "mistral"]


class _Rec:
    """Captures the kwargs a branch would send, without any network call."""
    def __init__(self):
        self.kwargs = None

    def _capture(self, **kw):
        self.kwargs = kw
        return _Resp()


class _Resp:
    def __init__(self):
        self.choices = []
        self.content = []
        self.text = ""
        self.usage = None


def _client_for(provider, rec):
    class C:
        pass
    c = C()
    if provider == "anthropic":
        c.messages = type("M", (), {"create": staticmethod(rec._capture)})()
    elif provider == "google":
        c.models = type("M", (), {"generate_content": staticmethod(rec._capture)})()
    elif provider == "mistral":
        c.chat = type("Ch", (), {"complete": staticmethod(rec._capture)})()
    else:
        c.chat = type("Ch", (), {
            "completions": type("Co", (), {"create": staticmethod(rec._capture)})()
        })()
    return c


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != "google"])
def test_every_provider_sends_the_same_temperature(provider, monkeypatch):
    rec = _Rec()
    um._request(provider, _client_for(provider, rec), "some-model", "p", 20)
    assert rec.kwargs.get("temperature") == um.TEMPERATURE, (
        f"{provider} must sample at {um.TEMPERATURE}, got {rec.kwargs.get('temperature')!r}")


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != "google"])
def test_every_provider_sends_the_system_prompt(provider):
    rec = _Rec()
    um._request(provider, _client_for(provider, rec), "some-model", "p", 20)
    if provider == "anthropic":
        assert rec.kwargs.get("system") == um._SYSTEM_PROMPT
    else:
        msgs = rec.kwargs.get("messages") or []
        roles = [m.get("role") for m in msgs]
        assert "system" in roles, f"{provider} sent no system prompt; roles={roles}"
        system = next(m["content"] for m in msgs if m["role"] == "system")
        assert system == um._SYSTEM_PROMPT


def test_google_sends_both_through_its_own_config_object():
    pytest.importorskip("google.genai")
    rec = _Rec()
    um._request("google", _client_for("google", rec), "some-model", "p", 20)
    config = rec.kwargs.get("config")
    assert getattr(config, "temperature", None) == um.TEMPERATURE
    assert getattr(config, "system_instruction", None) == um._SYSTEM_PROMPT


def test_models_that_reject_temperature_are_recorded_not_silently_defaulted():
    """gpt-5 rejects the parameter. Omitting it is acceptable; hiding the
    omission is not, because that judge then runs at an undocumented provider
    default while the others are matched."""
    assert not um.accepts_temperature("gpt-5.5")
    cfg = um.decoding_config("gpt-5.5", 1024)
    assert cfg["temperature"] is None
    assert cfg["temperature_omitted_provider_default"] is True

    cfg = um.decoding_config("claude-haiku-4-5-20251001", 1024)
    assert cfg["temperature"] == um.TEMPERATURE
    assert cfg["temperature_omitted_provider_default"] is False


def test_no_branch_hardcodes_a_divergent_temperature():
    """Guards the specific regression: a literal temperature in a branch."""
    src = inspect.getsource(um._request)
    for bad in ("temperature=0.01", "temperature=1.0", "temperature=0.7"):
        assert bad not in src, f"{bad} hardcoded in a provider branch"
