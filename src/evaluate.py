"""
JudgeSense evaluation runner — calls judge LLMs on prompt pairs and records decisions.

For each prompt pair (A, B) in a task JSONL, this script submits both prompts
to the chosen judge model and records both decisions, enabling downstream
computation of the Judge Sensitivity Score (JSS) and related metrics.

Usage examples:
    python src/evaluate.py --model gpt-4o-mini --task factuality --runs 3
    python src/evaluate.py --model gpt-4o-mini --task all --runs 1 --dry-run
    python src/evaluate.py --model mistral --task coherence --runs 5 \
        --input data/prompt_pairs/ --output data/results/raw_outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_OK = True
except ImportError:
    _DOTENV_OK = False

try:
    from tqdm import tqdm as _tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("judgeSense.evaluate")

if not _DOTENV_OK:
    log.warning(
        "python-dotenv is not installed — .env file will not be loaded. "
        "API keys must be set as environment variables."
    )


# ── Repository path constants ──────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "prompt_pairs"
_RESULTS_DIR = _REPO_ROOT / "data" / "results" / "raw_outputs"

_TASK_FILES = {
    "factuality": _DATA_DIR / "factuality.jsonl",
    "coherence":  _DATA_DIR / "coherence.jsonl",
    "relevance":  _DATA_DIR / "relevance.jsonl",
    "preference": _DATA_DIR / "preference.jsonl",
}

# Maps model names to the environment variable that holds their API key
_KEY_ENV_VARS = {
    "gpt-4o-mini": "OPENAI_API_KEY",
    "llama3":      "HF_TOKEN",
    "mistral":     "MISTRAL_API_KEY",
}

# Rate limiting (seconds between API calls)
_RATE_LIMIT = {
    "gpt-4o-mini": 0.5,
    "llama3":      1.0,
    "mistral":     0.5,
}

# Cost per 1K tokens (input + output average)
_COST_PER_1K_TOKENS = {
    "gpt-4o-mini": 0.000150,
    "gpt-4o":      0.002500,
    "claude-haiku": 0.000800,
    "claude-sonnet": 0.003000,
    "gemini-flash": 0.000000,
    "llama3":      0.000000,
    "mistral":     0.000000,
}


# ── JSONL I/O helpers ──────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file; skip blank lines and warn on malformed lines."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    """Append one record to a JSONL file; creates file and parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ── API client factories ───────────────────────────────────────────────────────

def _build_client(model: str, api_key: str):
    """
    Instantiate the SDK client for the given model.

    Raises:
        ImportError: If the required SDK package is not installed.
        ValueError: If the model name is unrecognised.
    """
    if model == "gpt-4o-mini":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for gpt-4o-mini. "
                "Install with: pip install openai"
            )
        return OpenAI(api_key=api_key)

    elif model == "llama3":
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for llama3. "
                "Install with: pip install huggingface-hub"
            )
        return InferenceClient(api_key=api_key)

    elif model == "mistral":
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "mistralai is required for mistral. "
                "Install with: pip install mistralai"
            )
        return Mistral(api_key=api_key)

    else:
        raise ValueError(f"Unknown model: {model!r}. Choose from: gpt-4o-mini, llama3, mistral")


# ── Per-model call functions ───────────────────────────────────────────────────

def _call_openai(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": "You are an expert evaluator. Provide clear, concise judgments.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def _call_llama(client, prompt: str) -> str:
    # HuggingFace Inference API — use chat completions endpoint
    result = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.01,  # some HF endpoints require temperature > 0
    )
    return result.choices[0].message.content.strip()


def _call_mistral(client, prompt: str) -> str:
    result = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    return result.choices[0].message.content.strip()


def _call_model(model: str, client, prompt: str) -> str:
    """Route a prompt to the correct SDK call based on model name."""
    if model == "gpt-4o-mini":
        return _call_openai(client, prompt)
    elif model == "llama3":
        return _call_llama(client, prompt)
    elif model == "mistral":
        return _call_mistral(client, prompt)
    else:
        raise ValueError(f"Unknown model: {model!r}")


# ── Helpers for efficiency improvements ────────────────────────────────────────

def _call_model_with_retry(model: str, client, prompt: str) -> str:
    """Call model with single retry on failure (waits 5 seconds before retry)."""
    try:
        return _call_model(model, client, prompt)
    except Exception as exc:
        log.warning("Initial call failed for %s, retrying in 5s: %s", model, exc)
        time.sleep(5)
        try:
            return _call_model(model, client, prompt)
        except Exception as exc2:
            return f"ERROR: {exc2}"


def _normalize_and_record(raw: str, task_type: str) -> tuple[str, str]:
    """Normalize decision and return both raw and normalized forms."""
    try:
        from src.models import normalize_decision
        normalized = normalize_decision(raw, task_type)
    except ImportError:
        # Fallback if module import fails
        normalized = raw[:20]
    return raw, normalized


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    model: str,
    task: str,
    pairs: List[dict],
    client,
    run_number: int,
    output_path: Path,
) -> tuple[int, float]:
    """
    Evaluate prompt pairs and stream results to output_path.

    Calls prompt_a then prompt_b for each pair, with rate limiting and retry logic.
    Records both raw and normalized decisions.

    Returns:
        (n_ok, estimated_cost): Pairs successfully evaluated and estimated USD cost.
    """
    iterator = pairs
    if _TQDM_OK:
        iterator = _tqdm(
            pairs,
            desc=f"run {run_number} | {model} | {task}",
            unit="pair",
            leave=True,
        )

    rate_limit_sec = _RATE_LIMIT.get(model, 0.5)
    cost_per_1k = _COST_PER_1K_TOKENS.get(model, 0.0)
    total_tokens = 0
    n_ok = 0

    for pair in iterator:
        pair_id   = pair.get("pair_id", "unknown")
        task_type = pair.get("task_type", task)
        prompt_a  = pair.get("prompt_a", "")
        prompt_b  = pair.get("prompt_b", "")

        # Call prompt A (with rate limiting and retry)
        raw_a = _call_model_with_retry(model, client, prompt_a)
        raw_a, norm_a = _normalize_and_record(raw_a, task_type)
        time.sleep(rate_limit_sec)

        # Call prompt B (with rate limiting and retry)
        raw_b = _call_model_with_retry(model, client, prompt_b)
        raw_b, norm_b = _normalize_and_record(raw_b, task_type)
        time.sleep(rate_limit_sec)

        # Estimate token usage: prompt ~50 tokens, response ~20 tokens
        total_tokens += (50 + 20) * 2  # for both A and B

        record = {
            "pair_id":              pair_id,
            "task_type":            task_type,
            "model":                model,
            "prompt_a_decision_raw": raw_a,
            "prompt_a_decision":    norm_a,
            "prompt_b_decision_raw": raw_b,
            "prompt_b_decision":    norm_b,
            "run_number":           run_number,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(record, output_path)

        if not raw_a.startswith("ERROR") and not raw_b.startswith("ERROR"):
            n_ok += 1

    estimated_cost = (total_tokens / 1000.0) * cost_per_1k

    if not _TQDM_OK:
        log.info(
            "Completed run %d — %d/%d pairs evaluated successfully. "
            "Estimated cost: ${:.4f}",
            run_number, n_ok, len(pairs), estimated_cost,
        )
    return n_ok, estimated_cost


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "JudgeSense evaluation runner.\n"
            "Submits prompt pairs (A, B) to a judge LLM and records both decisions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["gpt-4o-mini", "llama3", "mistral"],
        help="Judge LLM to use.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["factuality", "coherence", "relevance", "preference", "all"],
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
        help=f"Directory containing prompt pair JSONL files (default: {_DATA_DIR}).",
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

    # Resolve tasks list
    tasks = list(_TASK_FILES.keys()) if args.task == "all" else [args.task]

    # Validate API key before doing any work — fail with a clean message
    env_var = _KEY_ENV_VARS[args.model]
    api_key = os.environ.get(env_var)
    if not api_key:
        log.error(
            "Missing API key: environment variable %s is not set.\n"
            "  Add it to your .env file:  %s=your_key_here\n"
            "  Or export it in the shell: export %s=your_key_here",
            env_var, env_var, env_var,
        )
        sys.exit(1)

    # Build SDK client once; reuse across all tasks and runs
    try:
        client = _build_client(args.model, api_key)
    except (ImportError, ValueError) as exc:
        log.error("Cannot initialise client for %s: %s", args.model, exc)
        sys.exit(1)

    log.info(
        "Evaluation started | model=%s | tasks=%s | runs=%d | dry_run=%s",
        args.model, tasks, args.runs, args.dry_run,
    )

    total_cost = 0.0

    for task in tasks:
        input_path = Path(args.input) / f"{task}.jsonl"

        if not input_path.exists():
            log.warning("Input file not found, skipping: %s", input_path)
            continue

        pairs = _load_jsonl(input_path)
        if not pairs:
            log.warning("No valid pairs loaded for task=%s from %s", task, input_path)
            continue

        if args.dry_run:
            pairs = pairs[:5]
            log.info("[dry-run] Limiting to %d pairs for task=%s", len(pairs), task)

        output_path = Path(args.output) / f"{args.model}_{task}.jsonl"

        for run_num in range(1, args.runs + 1):
            log.info(
                "Run %d/%d | task=%s | pairs=%d | output=%s",
                run_num, args.runs, task, len(pairs), output_path,
            )
            _, cost = run_evaluation(
                model=args.model,
                task=task,
                pairs=pairs,
                client=client,
                run_number=run_num,
                output_path=output_path,
            )
            total_cost += cost
            log.info("Task %s run %d cost: $%.4f (total: $%.4f)", task, run_num, cost, total_cost)

    log.info("Evaluation complete. Total estimated cost: $%.4f", total_cost)


if __name__ == "__main__":
    main()
