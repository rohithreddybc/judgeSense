#!/usr/bin/env python3
"""
Task 4 — Pair-level flip overlap.

A pair "flips" for a model if that model's T4-corrected decisions are NOT
all identical across all runs (i.e., the model is inconsistent on that pair).

Output: outputs/factuality_pair_flip_overlap.csv
Columns: pair_id, flip_count (0-9), flipping_models (comma-separated)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data/results/raw_outputs"
PAIRS_FILE  = ROOT / "data/prompt_pairs/factuality.jsonl"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.metrics import EXCLUDED_PAIRS

TEMPLATES = {
    "T1": "Is this factually correct? Answer YES or NO only.",
    "T2": "Fact-check this response. Reply YES (correct) or NO (incorrect).",
    "T3": "Are the facts in this response accurate? YES or NO.",
    "T4": "Does this response contain factual errors? Answer NO (accurate) or YES (has errors).",
    "T5": "Evaluate factual accuracy. One word answer - YES or NO.",
}

MODELS = [
    "claude-haiku", "claude-sonnet", "deepseek", "gemini-flash",
    "gpt-4o-mini", "gpt-4o", "llama3-70b", "mistral-7b", "qwen",
]


def identify_template(prompt_text: str) -> str:
    part = prompt_text.split("\n\nResponse:")[0] if "\n\nResponse:" in prompt_text else prompt_text
    return next((k for k, v in TEMPLATES.items() if v == part), "UNKNOWN")


def load_pair_templates() -> dict:
    out = {}
    with open(PAIRS_FILE) as fh:
        for line in fh:
            p = json.loads(line)
            out[p["pair_id"]] = (
                identify_template(p["prompt_a"]),
                identify_template(p["prompt_b"]),
            )
    return out


def model_flips_on_pair(model: str, pid: str) -> bool:
    """
    Returns True if ANY run for this (model, pair) has raw decision_a != decision_b.
    No polarity correction — raw normalized_a vs normalized_b directly.
    """
    log = RESULTS_DIR / f"{model}_factuality.jsonl"
    with open(log) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["pair_id"] != pid or rec.get("error"):
                continue
            da, db = rec.get("normalized_a"), rec.get("normalized_b")
            if not da or not db or "UNCLEAR" in (da, db):
                continue
            if da != db:
                return True
    return False


def main():
    print("=" * 70)
    print("TASK 4 — Pair-Level Flip Overlap")
    print("=" * 70)

    pair_templates = load_pair_templates()
    valid_pairs = sorted(
        pid for pid in pair_templates if pid not in EXCLUDED_PAIRS
    )
    assert len(valid_pairs) == 119, f"Expected 119 valid pairs, got {len(valid_pairs)}"

    rows = []
    for pid in valid_pairs:
        flipping = [m for m in MODELS if model_flips_on_pair(m, pid)]
        rows.append({
            "pair_id":         pid,
            "flip_count":      len(flipping),
            "flipping_models": ",".join(flipping),
        })

    df = pd.DataFrame(rows).sort_values("flip_count", ascending=False)
    df["has_T4"] = df["pair_id"].map(
        lambda p: 1 if "T4" in pair_templates[p] else 0
    )
    df.drop(columns="has_T4").to_csv(
        OUTPUT_DIR / "factuality_pair_flip_overlap.csv", index=False
    )

    # Console output
    n_all  = (df["flip_count"] == 9).sum()
    n_none = (df["flip_count"] == 0).sum()
    print(f"\nPairs where ALL 9 models flip: {n_all}")
    print(f"Pairs where NO model flips: {n_none}")

    print(f"\nTop 10 pairs by flip count:")
    print(f"{'pair_id':<12} {'flip_count':>10}  {'has_T4':>6}")
    print("-" * 36)
    for _, r in df.head(10).iterrows():
        print(f"{r['pair_id']:<12} {int(r['flip_count']):>10}  {int(r['has_T4']):>6}")

    # T4 vs non-T4 mean flip count
    t4_mean    = df[df["has_T4"] == 1]["flip_count"].mean()
    non_t4_mean = df[df["has_T4"] == 0]["flip_count"].mean()
    print(f"\nT4-involved pairs mean flip_count: {t4_mean:.2f}")
    print(f"Non-T4 pairs mean flip_count: {non_t4_mean:.2f}")

    # Concentration check
    top_10 = df.head(10)
    t4_in_top10 = top_10["has_T4"].sum()
    concentrated = "YES" if t4_in_top10 >= 6 else "NO"
    print(f"\nAre high-flip pairs concentrated in T4 pairs? {concentrated} "
          f"({t4_in_top10}/10 top pairs involve T4)")

    # Verdict
    if t4_mean >= 8 and non_t4_mean <= 1:
        print("\nOVERLAP BUG FIXED")
    else:
        print(f"\nResult inconsistent: T4_mean={t4_mean:.2f} (expected ~9), "
              f"non_T4_mean={non_t4_mean:.2f} (expected ~0)")

    print(f"\nSaved: {OUTPUT_DIR / 'factuality_pair_flip_overlap.csv'}")


if __name__ == "__main__":
    main()
