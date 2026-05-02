"""
JudgeSense evaluation runner — calls judge LLMs on prompt pairs and records decisions.

Usage examples:
    python src/evaluate.py --model gpt-4o-mini --task factuality --runs 3
    python src/evaluate.py --model claude-haiku --task all --runs 1 --dry-run
    python src/evaluate.py --model all --task all --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple


# ── .env loading ──────────────────────────────────────────────────────────────

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


# ── Model registry ────────────────────────────────────────────────────────────

# Import after env loading so keys are available.
# Support both invocation styles:
#   python -m src.evaluate ...     (relative import works)
#   python src/evaluate.py ...     (relative import fails; fall back to absolute)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root  -> 'src.models'
sys.path.insert(0, str(Path(__file__).resolve().parent))         # src dir    -> 'models'
try:
    from .models import SUPPORTED_MODELS, normalize_decision
except ImportError:
    from models import SUPPORTED_MODELS, normalize_decision  # type: ignore

_ALL_MODELS = list(SUPPORTED_MODELS.keys())  # 9 models

# Rate limits in seconds per API call (between calls, not per minute)
_RATE_LIMIT = {
    "openai":       0.5,
    "anthropic":    0.5,
    "google":       1.0,
    "huggingface":  1.0,
    "mistral":      0.5,
    "novita":       0.5,
    "dashscope":    0.5,
}

_TIMEOUT          = 60   # seconds per API call (raised from 30 to accommodate reasoning models)
_DEFAULT_MAX_TOKENS = 20  # fallback if SUPPORTED_MODELS entry omits max_tokens

# Cost per 1K tokens (input + output blended estimate) — 0 for free tiers
_COST_PER_1K = {
    "gpt-4o-mini":      0.000150,
    "gpt-4o":           0.002500,
    "claude-haiku":     0.000800,
    "claude-sonnet":    0.003000,
    "gemini-flash":     0.0,
    "llama3-8b":        0.0,
    "llama3-70b":       0.0,
    "mistral-7b":       0.0,
    "qwen":             0.0,
    "deepseek":         0.005000,  # reasoning model — output-heavy
    "gpt-5.5":            0.012000,  # placeholder; update from OpenAI pricing
    "claude-opus-4-7":    0.015000,  # placeholder; update from Anthropic pricing
    "qwen-3.6-flash":     0.0,        # update from Alibaba Model Studio pricing
    "deepseek-v4-flash":  0.001000,  # placeholder; update from Novita pricing
}

_SYSTEM_PROMPT = "You are an evaluation assistant. Give only the requested answer with no explanation."


# ── Path constants ─────────────────────────────────────────────────────────────

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_DATA_DIR    = _REPO_ROOT / "data" / "prompt_pairs"
_RESULTS_DIR = _REPO_ROOT / "data" / "results" / "raw_outputs"

_TASK_FILES = {
    "factuality": _DATA_DIR / "factuality.jsonl",
    "coherence":  _DATA_DIR / "coherence.jsonl",
    "relevance":  _DATA_DIR / "relevance.jsonl",
    "preference": _DATA_DIR / "preference.jsonl",
}


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed line {lineno} in {path}")
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _completed_keys(path: Path) -> Set[Tuple[str, int]]:
    """Return set of (pair_id, run) tuples successfully completed (no error)."""
    if not path.exists():
        return set()
    keys: Set[Tuple[str, int]] = set()
    for rec in _load_jsonl(path):
        pid = rec.get("pair_id")
        run = rec.get("run")
        if pid is not None and run is not None and rec.get("error") is None:
            keys.add((pid, int(run)))
    return keys


# ── Provider call functions ───────────────────────────────────────────────────

def _openai_token_param(model_id: str) -> str:
    """GPT-5.x / o-series require 'max_completion_tokens'; older models use 'max_tokens'."""
    m = model_id.lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "max_completion_tokens"
    return "max_tokens"


def _call_openai(client, model_id: str, prompt: str, max_tokens: int) -> str:
    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "timeout": _TIMEOUT,
        _openai_token_param(model_id): max_tokens,
    }
    # GPT-5.x rejects custom temperature; only pass it for older models
    if not model_id.lower().startswith("gpt-5"):
        kwargs["temperature"] = 0.0
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _call_anthropic(client, model_id: str, prompt: str, max_tokens: int) -> str:
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_huggingface(client, model_id: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.01,  # some HF endpoints reject 0.0
    )
    return response.choices[0].message.content.strip()


def _call_google(client, model_id: str, prompt: str, max_tokens: int) -> str:
    from google.genai import types
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


def _call_mistral(client, model_id: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.complete(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Client factory ────────────────────────────────────────────────────────────

def _build_client(model_name: str):
    """Build SDK client for model_name. Returns (client, model_id, provider)."""
    cfg      = SUPPORTED_MODELS[model_name]
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    api_key  = os.environ.get(cfg["key"], "")

    if not api_key:
        raise RuntimeError(f"Missing env var: {cfg['key']} (required for {model_name})")

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        return OpenAI(api_key=api_key), model_id, provider

    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        return anthropic.Anthropic(api_key=api_key), model_id, provider

    elif provider == "google":
        try:
            from google import genai
        except ImportError:
            raise ImportError("pip install google-genai")
        return genai.Client(api_key=api_key), model_id, provider

    elif provider == "huggingface":
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("pip install huggingface-hub")
        return InferenceClient(api_key=api_key), model_id, provider

    elif provider == "mistral":
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("pip install mistralai")
        return Mistral(api_key=api_key), model_id, provider

    elif provider == "novita":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.novita.ai/v3/openai",
        ), model_id, provider

    elif provider == "dashscope":
        # Alibaba Cloud Model Studio (DashScope) — OpenAI-compatible mode.
        # Defaults to the international endpoint; override with DASHSCOPE_BASE_URL
        # to use the China endpoint instead.
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        return OpenAI(api_key=api_key, base_url=base_url), model_id, provider

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Single call with one retry ────────────────────────────────────────────────

def _call(provider: str, client, model_id: str, prompt: str, max_tokens: int) -> str:
    """Route to correct provider function; retry once on failure."""
    dispatch = {
        "openai":      _call_openai,
        "anthropic":   _call_anthropic,
        "google":      _call_google,
        "huggingface": _call_huggingface,
        "mistral":     _call_mistral,
        "novita":      _call_openai,  # OpenAI-compatible
        "dashscope":   _call_openai,  # OpenAI-compatible
    }
    fn = dispatch[provider]
    try:
        return fn(client, model_id, prompt, max_tokens)
    except Exception as exc:
        time.sleep(5)
        try:
            return fn(client, model_id, prompt, max_tokens)
        except Exception as exc2:
            return f"ERROR:{exc2}"


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    model_name: str,
    task: str,
    pairs: List[dict],
    client,
    model_id: str,
    provider: str,
    run_number: int,
    runs_total: int,
    output_path: Path,
    done: Set[Tuple[str, int]],
) -> Tuple[int, float]:
    """
    Evaluate all pairs for one run. Skips pairs already in `done`.
    Returns (n_ok, estimated_cost).
    """
    rate_sec   = _RATE_LIMIT.get(provider, 0.5)
    cost_1k    = _COST_PER_1K.get(model_name, 0.0)
    max_tokens = SUPPORTED_MODELS[model_name].get("max_tokens", _DEFAULT_MAX_TOKENS)
    total_tok  = 0
    n_ok       = 0

    for i, pair in enumerate(pairs, 1):
        pair_id   = pair.get("pair_id", f"unknown_{i}")
        task_type = pair.get("task_type", task)
        prompt_a  = pair.get("prompt_a", "")
        prompt_b  = pair.get("prompt_b", "")

        if (pair_id, run_number) in done:
            continue

        print(f"{model_name} | {task} | pair {i}/{len(pairs)} | run {run_number}/{runs_total} | max_tokens={max_tokens}")

        # Call A
        raw_a = _call(provider, client, model_id, prompt_a, max_tokens)
        time.sleep(rate_sec)

        # Call B
        raw_b = _call(provider, client, model_id, prompt_b, max_tokens)
        time.sleep(rate_sec)

        error_a = raw_a.startswith("ERROR:") if isinstance(raw_a, str) else False
        error_b = raw_b.startswith("ERROR:") if isinstance(raw_b, str) else False
        has_error = error_a or error_b

        norm_a = normalize_decision(raw_a, task_type) if not error_a else "UNCLEAR"
        norm_b = normalize_decision(raw_b, task_type) if not error_b else "UNCLEAR"

        record = {
            "pair_id":          pair_id,
            "task_type":        task_type,
            "model":            model_name,
            "run":              run_number,
            "prompt_a_decision": norm_a,
            "prompt_b_decision": norm_b,
            "normalized_a":     norm_a,
            "normalized_b":     norm_b,
            "flipped":          norm_a != norm_b,
            "prompt_a_raw":     raw_a,
            "prompt_b_raw":     raw_b,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "error":            (raw_a if error_a else raw_b) if has_error else None,
        }
        _append_jsonl(record, output_path)
        done.add((pair_id, run_number))

        if not has_error:
            n_ok += 1
            total_tok += (50 + max_tokens) * 2

    estimated_cost = (total_tok / 1000.0) * cost_1k
    return n_ok, estimated_cost


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="JudgeSense evaluation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=_ALL_MODELS + ["all"],
        help="Judge LLM to use, or 'all' to run every model.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(_TASK_FILES.keys()) + ["all"],
        help="Task dataset to evaluate, or 'all' for every task.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of evaluation runs per task (default: 3).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DATA_DIR,
        metavar="DIR",
        help=f"Directory with prompt pair JSONL files (default: {_DATA_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_RESULTS_DIR,
        metavar="DIR",
        help=f"Output directory for raw results (default: {_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 5 pairs per task (for quick testing).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    models = _ALL_MODELS if args.model == "all" else [args.model]
    tasks  = list(_TASK_FILES.keys()) if args.task == "all" else [args.task]

    total_cost = 0.0

    for model_name in models:
        print(f"\n=== Model: {model_name} ===")

        try:
            client, model_id, provider = _build_client(model_name)
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[SKIP] {model_name}: {exc}")
            continue

        for task in tasks:
            input_path = Path(args.input) / f"{task}.jsonl"

            if not input_path.exists():
                print(f"[WARN] Input file not found, skipping: {input_path}")
                continue

            pairs = _load_jsonl(input_path)
            if not pairs:
                print(f"[WARN] No valid pairs for task={task}")
                continue

            if args.dry_run:
                pairs = pairs[:5]
                print(f"[dry-run] Limiting to {len(pairs)} pairs for task={task}")

            output_path = Path(args.output) / f"{model_name}_{task}.jsonl"
            done = _completed_keys(output_path)

            for run_num in range(1, args.runs + 1):
                n_ok, cost = run_evaluation(
                    model_name=model_name,
                    task=task,
                    pairs=pairs,
                    client=client,
                    model_id=model_id,
                    provider=provider,
                    run_number=run_num,
                    runs_total=args.runs,
                    output_path=output_path,
                    done=done,
                )
                total_cost += cost
                print(f"  run {run_num}/{args.runs} done: {n_ok}/{len(pairs)} ok, cost=${cost:.4f}")

    print(f"\nTotal estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
