"""
JudgeSense model wrappers — unified interface for judge LLMs.

Supports:
  - OpenAI (GPT-4o-mini, GPT-4o)
  - Anthropic (Claude Haiku, Claude Sonnet)
  - Google (Gemini Flash)
  - HuggingFace (Llama 3.1 8B/70B, Qwen, DeepSeek)
  - Mistral (Mistral-7B)
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import json

# Supported models registry with exact model IDs
SUPPORTED_MODELS = {
    "gpt-4o-mini":   {"provider": "openai",      "model_id": "gpt-4o-mini-2024-07-18",        "key": "OPENAI_API_KEY"},
    "gpt-4o":        {"provider": "openai",      "model_id": "gpt-4o-2024-08-06",             "key": "OPENAI_API_KEY"},
    "claude-haiku":  {"provider": "anthropic",   "model_id": "claude-haiku-4-5-20251001",     "key": "ANTHROPIC_API_KEY"},
    "claude-sonnet": {"provider": "anthropic",   "model_id": "claude-sonnet-4-5",             "key": "ANTHROPIC_API_KEY"},
    "gemini-flash":  {"provider": "google",      "model_id": "gemini-2.0-flash",              "key": "GOOGLE_API_KEY"},
    "llama3-8b":     {"provider": "huggingface", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "key": "HF_TOKEN"},
    "llama3-70b":    {"provider": "huggingface", "model_id": "meta-llama/Llama-3.1-70B-Instruct", "key": "HF_TOKEN"},
    "mistral-7b":    {"provider": "mistral",     "model_id": "mistral-small-latest",          "key": "MISTRAL_API_KEY"},
    "qwen":          {"provider": "huggingface", "model_id": "Qwen/Qwen2.5-7B-Instruct",      "key": "HF_TOKEN"},
    "deepseek":      {"provider": "huggingface", "model_id": "deepseek-ai/DeepSeek-V3",       "key": "HF_TOKEN"},
}


class JudgeModel(ABC):
    """Abstract base class for judge models."""

    def __init__(self, temperature: float = 0.0):
        """
        Initialize judge model.
        
        Args:
            temperature: Sampling temperature (0 = deterministic, 1 = random).
        """
        self.temperature = temperature

    @abstractmethod
    def evaluate(self, prompt: str) -> str:
        """
        Evaluate a prompt and return judge's decision.
        
        Args:
            prompt: The evaluation prompt.
            
        Returns:
            Judge's decision (string).
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class GPT4oMiniJudge(JudgeModel):
    """Judge using OpenAI's GPT-4o-mini model."""

    def __init__(self, api_key: str, temperature: float = 0.0):
        """
        Initialize GPT-4o-mini judge.
        
        Args:
            api_key: OpenAI API key.
            temperature: Sampling temperature.
        """
        super().__init__(temperature)
        self.api_key = api_key
        self.model_name = "gpt-4o-mini-2024-07-18"
        
        # Lazy import to avoid requiring openai if not used
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

    def evaluate(self, prompt: str) -> str:
        """
        Evaluate using GPT-4o-mini.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Model's response text.
        """
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
    """Judge using Meta's Llama 3 8B model via HuggingFace."""

    def __init__(self, hf_token: str, temperature: float = 0.0):
        """
        Initialize Llama 3 8B judge.
        
        Args:
            hf_token: HuggingFace API token.
            temperature: Sampling temperature.
        """
        super().__init__(temperature)
        self.hf_token = hf_token
        self.model_name = "meta-llama/Llama-2-7b-hf"  # or Llama-3-8B-Instruct
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(api_key=hf_token)
        except ImportError:
            raise ImportError(
                "huggingface-hub is required. Install with: pip install huggingface-hub"
            )

    def evaluate(self, prompt: str) -> str:
        """
        Evaluate using Llama 3 8B via HuggingFace Inference API.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Model's response text.
        """
        try:
            system = "You are an evaluation assistant. Give only the requested answer with no explanation."
            full_prompt = f"{system}\n{prompt}"
            response = self.client.text_generation(
                prompt=full_prompt,
                max_new_tokens=20,
                temperature=self.temperature,
                top_p=0.95
            )
            return response.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def __repr__(self) -> str:
        return f"LlamaJudge(model={self.model_name}, temperature={self.temperature})"


class MistralJudge(JudgeModel):
    """Judge using Mistral's Mistral-7B model."""

    def __init__(self, api_key: str, temperature: float = 0.0):
        """
        Initialize Mistral-7B judge.
        
        Args:
            api_key: Mistral API key.
            temperature: Sampling temperature.
        """
        super().__init__(temperature)
        self.api_key = api_key
        self.model_name = "mistral-7b-instruct"
        
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key)
        except ImportError:
            raise ImportError(
                "mistralai is required. Install with: pip install mistralai"
            )

    def evaluate(self, prompt: str) -> str:
        """
        Evaluate using Mistral-7B.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Model's response text.
        """
        try:
            system = "You are an evaluation assistant. Give only the requested answer with no explanation."
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": f"{system}\n{prompt}"}
                ],
                temperature=self.temperature,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def __repr__(self) -> str:
        return f"MistralJudge(model={self.model_name}, temperature={self.temperature})"


def normalize_decision(raw: str, task_type: str) -> str:
    """
    Extract clean decision from raw model output.

    Args:
        raw: Raw model response text.
        task_type: One of "factuality", "coherence", "relevance", "preference".

    Returns:
        Normalized decision token (YES/NO, 1-5, A/B, etc.).
    """
    raw = raw.strip().upper()

    if task_type == "factuality":
        if "YES" in raw:
            return "YES"
        if "NO" in raw:
            return "NO"
        return "UNCLEAR"

    elif task_type in ["relevance", "preference"]:
        if raw.startswith("A"):
            return "A"
        if raw.startswith("B"):
            return "B"
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
        
    Raises:
        ValueError: If model_name is not recognized.
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
    "create_judge",
    "normalize_decision",
]
