"""
JudgeSense model wrappers — unified interface for judge LLMs.

Supports:
  - OpenAI (GPT-4o-mini, GPT-4o)
  - Anthropic (Claude Haiku, Claude Sonnet)
  - HuggingFace (Llama 3.1 8B/70B, Qwen, DeepSeek)
  - Mistral (Mistral-7B)
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


def _load_env():
    """Load .env file manually — avoids python-dotenv AssertionError on Windows."""
    try:
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

_load_env()


# Supported models registry.
# `max_tokens` is per-model: non-reasoning instruction-tuned judges keep the
# original 20-token cap (single-token answers); reasoning-tuned judges
# (deepseek-r1, GPT-5.5, Claude Opus 4.7, Qwen 3.6 Flash, DeepSeek-V4 Flash)
# get 1024 tokens so their internal chain can complete before emitting the answer.
SUPPORTED_MODELS = {
    # ── Existing 8 non-reasoning judges (unchanged, max_tokens=20) ──
    "gpt-4o-mini":   {"provider": "openai",      "model_id": "gpt-4o-mini-2024-07-18",            "key": "OPENAI_API_KEY",    "max_tokens": 20},
    "gpt-4o":        {"provider": "openai",      "model_id": "gpt-4o-2024-08-06",                 "key": "OPENAI_API_KEY",    "max_tokens": 20},
    "claude-haiku":  {"provider": "anthropic",   "model_id": "claude-haiku-4-5-20251001",         "key": "ANTHROPIC_API_KEY", "max_tokens": 20},
    "claude-sonnet": {"provider": "anthropic",   "model_id": "claude-sonnet-4-5",                 "key": "ANTHROPIC_API_KEY", "max_tokens": 20},
    "gemini-flash":  {"provider": "google",      "model_id": "gemini-2.5-flash",                  "key": "GOOGLE_API_KEY",    "max_tokens": 20},
    "llama3-8b":     {"provider": "huggingface", "model_id": "meta-llama/Llama-3.1-8B-Instruct",  "key": "HF_TOKEN",          "max_tokens": 20},
    "llama3-70b":    {"provider": "groq",        "model_id": "llama-3.1-70b-versatile",           "key": "GROQ_API_KEY",      "max_tokens": 20},  # HF endpoint broken; Novita lacks 3.1 70B; using Groq
    "mistral-7b":    {"provider": "mistral",     "model_id": "mistral-small-latest",              "key": "MISTRAL_API_KEY",   "max_tokens": 20},
    # v2 renamed this judge to "mistral-small": the id resolves to
    # mistral-small-latest, which is not a 7B checkpoint, so the old key
    # asserted a parameter count the model does not have. Both names are kept
    # here so v1's published runs stay reproducible under the name they were
    # published with, while the v2 registry's name resolves too. Do not delete
    # either: src/evaluate.py subscripts this dict directly, and a name present
    # only in judge_registry KeyErrors mid-run, after earlier judges are paid for.
    "mistral-small": {"provider": "mistral",     "model_id": "mistral-small-latest",              "key": "MISTRAL_API_KEY",   "max_tokens": 20},
    "qwen":          {"provider": "novita",      "model_id": "qwen/qwen-2.5-72b-instruct",        "key": "NOVITA_API_KEY",    "max_tokens": 20},

    # ── Re-run at 1024 to retire the truncation caveat ──
    "deepseek":      {"provider": "novita",      "model_id": "deepseek/deepseek-r1",              "key": "NOVITA_API_KEY",    "max_tokens": 1024},

    # ── 4 new judges added in revision pass 2 (April 2026) ──
    "gpt-5.5":            {"provider": "openai",    "model_id": "gpt-5.5",                        "key": "OPENAI_API_KEY",    "max_tokens": 1024},
    "claude-opus-4-7":    {"provider": "anthropic", "model_id": "claude-opus-4-7",                "key": "ANTHROPIC_API_KEY", "max_tokens": 1024},
    "qwen-3.6-flash":     {"provider": "dashscope", "model_id": "qwen3.6-35b-a3b",                "key": "DASHSCOPE_API_KEY", "max_tokens": 1024},
    "deepseek-v4-flash":  {"provider": "novita",    "model_id": "deepseek/deepseek-v4-flash",     "key": "NOVITA_API_KEY",    "max_tokens": 1024},

    # ── multi-vendor expansion (2026-08-25) ──
    "kimi-k3":             {"provider": "dashscope",   "model_id": "kimi-k3",                                           "key": "DASHSCOPE_API_KEY", "max_tokens": 1024},
    "deepseek-v4-pro":     {"provider": "dashscope",   "model_id": "deepseek-v4-pro-0813",                              "key": "DASHSCOPE_API_KEY", "max_tokens": 1024},
    "mistral-medium":      {"provider": "mistral",     "model_id": "mistral-medium-2604",                               "key": "MISTRAL_API_KEY",   "max_tokens": 1024},
    "qwen3.8-27b-hf":      {"provider": "huggingface", "model_id": "Qwen/Qwen3.8-27B",                                  "key": "HF_TOKEN",          "max_tokens": 1024},
    "gemini-3.1-pro":      {"provider": "google",      "model_id": "gemini-3.1-pro-preview",                            "key": "GOOGLE_API_KEY",    "max_tokens": 1024},
    "gpt-oss-20b":         {"provider": "groq",        "model_id": "openai/gpt-oss-20b",                                "key": "GROQ_API_KEY",      "max_tokens": 1024},
    "gpt-oss-120b":        {"provider": "groq",        "model_id": "openai/gpt-oss-120b",                               "key": "GROQ_API_KEY",      "max_tokens": 1024},
    "qwen3.8-27b":         {"provider": "groq",        "model_id": "qwen/qwen3.8-27b",                                  "key": "GROQ_API_KEY",      "max_tokens": 1024},
    "qwen3-8b":            {"provider": "huggingface", "model_id": "Qwen/Qwen3-8B",                                     "key": "HF_TOKEN",          "max_tokens": 1024},
    "qwen3-14b":           {"provider": "huggingface", "model_id": "Qwen/Qwen3-14B",                                    "key": "HF_TOKEN",          "max_tokens": 1024},
    "qwen3-32b":           {"provider": "huggingface", "model_id": "Qwen/Qwen3-32B",                                    "key": "HF_TOKEN",          "max_tokens": 1024},
    # These MUST mirror judge_registry.JUDGES: run_v2 selects judges from the
    # registry but resolves clients through this dict, so a name present in one
    # and absent from the other fails at client-build time, mid-sweep.
    # tests/test_registry_parity.py enforces the correspondence.
    "llama-3.3-70b":       {"provider": "huggingface", "model_id": "meta-llama/Llama-3.3-70B-Instruct",                  "key": "HF_TOKEN",          "max_tokens": 1024},
    "llama-4-scout":       {"provider": "huggingface", "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",          "key": "HF_TOKEN",          "max_tokens": 1024},
    "llama-4-maverick":    {"provider": "huggingface", "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  "key": "HF_TOKEN",          "max_tokens": 1024},
    "gemma-4-31b":         {"provider": "huggingface", "model_id": "google/gemma-4-31B-it",                              "key": "HF_TOKEN",          "max_tokens": 1024},
    "gemini-3.7-flash":    {"provider": "google",      "model_id": "gemini-3.7-flash",                                   "key": "GOOGLE_API_KEY",    "max_tokens": 1024},
    "qwen3.7-flash":       {"provider": "dashscope",   "model_id": "qwen3.7-flash-2026-07-15",                           "key": "DASHSCOPE_API_KEY", "max_tokens": 1024},
    "deepseek-r1-0528":    {"provider": "huggingface", "model_id": "deepseek-ai/DeepSeek-R1-0528",                       "key": "HF_TOKEN",          "max_tokens": 1024},
    "qwen3-235b-thinking": {"provider": "huggingface", "model_id": "Qwen/Qwen3-235B-A22B-Thinking-2507",                 "key": "HF_TOKEN",          "max_tokens": 1024},
    "deepseek-v4-flash-ds":{"provider": "dashscope",   "model_id": "deepseek-v4-flash-0731",                             "key": "DASHSCOPE_API_KEY", "max_tokens": 1024},
    "glm-5.2":             {"provider": "dashscope",   "model_id": "glm-5.2",                                            "key": "DASHSCOPE_API_KEY", "max_tokens": 1024},
    "magistral-small":     {"provider": "mistral",     "model_id": "magistral-small-latest",                             "key": "MISTRAL_API_KEY",   "max_tokens": 1024},
}


class JudgeModel(ABC):
    """Abstract base class for judge models."""

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    @abstractmethod
    def evaluate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class GPT4oMiniJudge(JudgeModel):
    """Judge using OpenAI's GPT-4o-mini model."""

    def __init__(self, api_key: str, temperature: float = 0.0):
        super().__init__(temperature)
        self.api_key = api_key
        self.model_name = "gpt-4o-mini-2024-07-18"
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

    def evaluate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an evaluation assistant. Give only the requested answer with no explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def __repr__(self) -> str:
        return f"GPT4oMiniJudge(model={self.model_name}, temperature={self.temperature})"


class LlamaJudge(JudgeModel):
    """Judge using Meta's Llama 3 models via HuggingFace."""

    def __init__(self, hf_token: str, temperature: float = 0.0):
        super().__init__(temperature)
        self.hf_token = hf_token
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(api_key=hf_token)
        except ImportError:
            raise ImportError(
                "huggingface-hub is required. Install with: pip install huggingface-hub"
            )

    def evaluate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=max(self.temperature, 0.01),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def __repr__(self) -> str:
        return f"LlamaJudge(model={self.model_name}, temperature={self.temperature})"


class MistralJudge(JudgeModel):
    """Judge using Mistral's models."""

    def __init__(self, api_key: str, temperature: float = 0.0):
        super().__init__(temperature)
        self.api_key = api_key
        self.model_name = "mistral-small-latest"
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key)
        except ImportError:
            raise ImportError(
                "mistralai is required. Install with: pip install mistralai"
            )

    def evaluate(self, prompt: str) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def __repr__(self) -> str:
        return f"MistralJudge(model={self.model_name}, temperature={self.temperature})"


class GeminiJudge(JudgeModel):
    """Judge using Google's Gemini Flash model."""

    def __init__(self, api_key: str, temperature: float = 0.0):
        super().__init__(temperature)
        self.api_key = api_key
        self.model_name = "gemini-2.5-flash"
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )

    def evaluate(self, prompt: str) -> str:
        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=self.temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def __repr__(self) -> str:
        return f"GeminiJudge(model={self.model_name}, temperature={self.temperature})"


def normalize_decision(raw: str, task_type: str) -> str:
    """
    Extract clean decision from raw model output.

    Args:
        raw: Raw model response text.
        task_type: One of "factuality", "coherence", "relevance", "preference".

    Returns:
        Normalized decision token (YES/NO, 1-5, A/B, or UNCLEAR).
    """
    raw = raw.strip().upper()

    if task_type == "factuality":
        if "YES" in raw:
            return "YES"
        if "NO" in raw:
            return "NO"
        return "UNCLEAR"

    elif task_type in ["relevance", "preference"]:
        import re
        cleaned = re.sub(r'[*_]+', '', raw).strip()
        m = re.search(r'\b([AB])\b', cleaned)
        if m:
            return m.group(1)
        return "UNCLEAR"

    elif task_type == "coherence":
        for char in raw:
            if char in "12345":
                return char
        return "UNCLEAR"

    return raw[:10]


def create_judge(model_name: str, api_key: str, temperature: float = 0.0) -> JudgeModel:
    """
    Factory function to create a judge model.

    Args:
        model_name: One of "gpt-4o-mini", "llama3", "mistral".
        api_key: API key for the model.
        temperature: Sampling temperature.

    Returns:
        JudgeModel instance.
    """
    if model_name.lower() in ["gpt-4o-mini", "gpt4o-mini", "openai"]:
        return GPT4oMiniJudge(api_key=api_key, temperature=temperature)
    elif model_name.lower() in ["llama3", "llama"]:
        return LlamaJudge(hf_token=api_key, temperature=temperature)
    elif model_name.lower() in ["mistral"]:
        return MistralJudge(api_key=api_key, temperature=temperature)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: gpt-4o-mini, llama3, mistral"
        )


__all__ = [
    "SUPPORTED_MODELS",
    "JudgeModel",
    "GPT4oMiniJudge",
    "LlamaJudge",
    "MistralJudge",
    "GeminiJudge",
    "create_judge",
    "normalize_decision",
]
