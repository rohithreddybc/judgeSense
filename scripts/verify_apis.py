"""
JudgeSense API Verification Script

Tests each LLM API with a minimal prompt ("Reply with the word OK only.")
and reports PASS/FAIL status for each model.

Usage:
    python scripts/verify_apis.py

Note: Requires .env file with API keys for each model being tested.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Manually load .env (python-dotenv has issues with find_dotenv())
def _load_env_manual():
    """Load .env file manually"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()
    except FileNotFoundError:
        pass

_load_env_manual()


def _openai_token_param(model_name: str) -> str:
    """GPT-5.x and o-series require 'max_completion_tokens'; older models use 'max_tokens'."""
    m = model_name.lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "max_completion_tokens"
    return "max_tokens"


def test_openai(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test OpenAI model."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Reply with only the word OK."},
                {"role": "user", "content": "Reply with the word OK only."},
            ],
            _openai_token_param(model_name): 1024,
        }
        # GPT-5.x rejects custom temperature; only pass it for older models
        if not model_name.lower().startswith("gpt-5"):
            kwargs["temperature"] = 0.0
        response = client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "").strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


def test_anthropic(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test Anthropic model (Claude)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=20,
            system="Reply with only the word OK.",
            messages=[
                {"role": "user", "content": "Reply with the word OK only."}
            ]
        )
        text = response.content[0].text.strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


def test_google(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test Google Gemini model via the current google.genai SDK."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with the word OK only.",
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (response.text or "").strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


def test_huggingface(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test HuggingFace model."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Reply with the word OK only."}
            ],
            max_tokens=20,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


def test_mistral(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test Mistral model."""
    try:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=model_name,
            messages=[
                {"role": "user", "content": "Reply with the word OK only."}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        text = response.choices[0].message.content.strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


def test_novita(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test Novita-hosted model (OpenAI-compatible)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.novita.ai/v3/openai")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Reply with the word OK only."}
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        text = response.choices[0].message.content.strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


def test_dashscope(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test Alibaba Cloud Model Studio (DashScope) — OpenAI-compatible mode."""
    try:
        from openai import OpenAI
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Reply with the word OK only."}
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        text = response.choices[0].message.content.strip()
        return True, text
    except Exception as exc:
        return False, str(exc)


# Model registry with exact model IDs from src/models.py
MODELS = {
    "gpt-4o-mini":      ("gpt-4o-mini-2024-07-18",            "OPENAI_API_KEY",    test_openai),
    "gpt-4o":           ("gpt-4o-2024-08-06",                 "OPENAI_API_KEY",    test_openai),
    "claude-haiku":     ("claude-haiku-4-5-20251001",         "ANTHROPIC_API_KEY", test_anthropic),
    "claude-sonnet":    ("claude-sonnet-4-5",                 "ANTHROPIC_API_KEY", test_anthropic),
    "gemini-flash":     ("gemini-2.5-flash",                  "GOOGLE_API_KEY",    test_google),
    "llama3-8b":        ("meta-llama/Llama-3.1-8B-Instruct",  "HF_TOKEN",          test_huggingface),
    "llama3-70b":       ("meta-llama/Llama-3.1-70B-Instruct", "HF_TOKEN",          test_huggingface),
    "mistral-7b":       ("mistral-small-latest",              "MISTRAL_API_KEY",   test_mistral),
    "qwen":             ("Qwen/Qwen2.5-7B-Instruct",          "HF_TOKEN",          test_huggingface),
    "deepseek":         ("deepseek-ai/DeepSeek-V3",           "HF_TOKEN",          test_huggingface),

    # ── Pass-2 additions (April 2026): confirm exact SKUs against provider docs ──
    "gpt-5.5":            ("gpt-5.5",                         "OPENAI_API_KEY",    test_openai),
    "claude-opus-4-7":    ("claude-opus-4-7",                 "ANTHROPIC_API_KEY", test_anthropic),
    "qwen-3.6-flash":     ("qwen3.6-35b-a3b",                 "DASHSCOPE_API_KEY", test_dashscope),
    "deepseek-v4-flash":  ("deepseek/deepseek-v4-flash",      "NOVITA_API_KEY",    test_novita),
}


def main():
    print("=" * 70)
    print("JudgeSense API Verification")
    print("=" * 70)
    print()

    results = {}
    passed = 0
    failed = 0
    skipped = 0

    for model_name, (model_id, env_var, test_fn) in MODELS.items():
        api_key = os.environ.get(env_var)

        if not api_key:
            print(f"Testing {model_name:<20} [--] SKIP  (missing {env_var})")
            results[model_name] = {
                "status": "SKIP",
                "response": None,
                "error": f"Missing API key: {env_var}",
            }
            skipped += 1
            continue

        print(f"Testing {model_name:<20} ", end="", flush=True)

        success, output = test_fn(model_id, api_key)

        if success:
            print(f"[OK] PASS  (response: \"{output}\")")
            results[model_name] = {
                "status": "PASS",
                "response": output,
                "error": None,
            }
            passed += 1
        else:
            print(f"[XX] FAIL  (error: {output})")
            results[model_name] = {
                "status": "FAIL",
                "response": None,
                "error": output,
            }
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed}/{len(MODELS)} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        failed_models = [m for m, r in results.items() if r["status"] == "FAIL"]
        print(f"Failed models: {', '.join(failed_models)}")
    print()

    # Save results to JSON
    output_dir = Path(__file__).parent
    output_path = output_dir / "verify_results.json"
    results["summary"] = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": len(MODELS),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
