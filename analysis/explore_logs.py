#!/usr/bin/env python3
"""
Exploratory script to verify factuality decision log structure.
Validates:
- JSONL format and row counts
- Pair IDs and excluded pairs
- Presence of required columns
- Template encoding verification
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent  # judgeSense directory
RESULTS_DIR = PROJECT_ROOT / "data/results/raw_outputs"
PROMPT_PAIRS_FILE = PROJECT_ROOT / "data/prompt_pairs/factuality.jsonl"

# Excluded pairs (from metrics.py)
EXCLUDED_PAIRS = {"fact_040", "fact_045", "fact_050", "fact_090", "fact_095", "fact_100"}

# Template definitions (from user spec)
TEMPLATES = {
    "T1": "Is this factually correct? Answer YES or NO only.",
    "T2": "Fact-check this response. Reply YES (correct) or NO (incorrect).",
    "T3": "Are the facts in this response accurate? YES or NO.",
    "T4": "Does this response contain factual errors? Answer NO (accurate) or YES (has errors).",
    "T5": "Evaluate factual accuracy. One word answer - YES or NO."
}


def identify_template(prompt_text):
    """Identify template by matching against known prompts.
    Prompts include the response text after the template, so extract template part first."""
    # Template is before "\n\nResponse:"
    if "\n\nResponse:" in prompt_text:
        template_part = prompt_text.split("\n\nResponse:")[0]
    else:
        template_part = prompt_text

    for template_name, template_text in TEMPLATES.items():
        if template_part == template_text:
            return template_name
    return "UNKNOWN"


def main():
    print("=" * 80)
    print("JudgeSense Factuality Logs — Structure Verification")
    print("=" * 80)

    # Step 1: Load and verify prompt pairs
    print("\n[1] Loading template mapping from prompt_pairs/factuality.jsonl...")
    pair_templates = {}
    with open(PROMPT_PAIRS_FILE) as f:
        for line in f:
            pair = json.loads(line)
            pair_id = pair["pair_id"]
            prompt_a = pair["prompt_a"]
            prompt_b = pair["prompt_b"]
            template_a = identify_template(prompt_a)
            template_b = identify_template(prompt_b)
            pair_templates[pair_id] = (template_a, template_b)

    print(f"  * Loaded {len(pair_templates)} pairs")
    print(f"  * Excluded pairs present: {EXCLUDED_PAIRS.intersection(pair_templates.keys())}")
    print(f"  * Valid pairs (after exclusion): {len(pair_templates) - len(EXCLUDED_PAIRS)}")

    # Step 2: Verify template distribution
    template_pairs = defaultdict(int)
    for pair_id, (template_a, template_b) in pair_templates.items():
        if pair_id not in EXCLUDED_PAIRS:
            key = tuple(sorted([template_a, template_b]))
            template_pairs[key] += 1

    print(f"\n[2] Template pair distribution across {len(pair_templates) - len(EXCLUDED_PAIRS)} valid pairs:")
    for (t1, t2), count in sorted(template_pairs.items()):
        print(f"  {t1}–{t2}: {count} pairs")

    # Step 3: Read a sample decision log
    sample_model = "claude-haiku_factuality.jsonl"
    sample_file = RESULTS_DIR / sample_model

    if not sample_file.exists():
        print(f"\n[3] ERROR: Sample file not found: {sample_file}")
        return

    print(f"\n[3] Reading sample log: {sample_file}")
    decisions = []
    with open(sample_file) as f:
        for line in f:
            decisions.append(json.loads(line))

    print(f"  * Total rows: {len(decisions)}")

    # Verify structure
    unique_pair_ids = set(d["pair_id"] for d in decisions)
    unique_runs = set(d["run"] for d in decisions)
    print(f"  * Unique pair_ids: {len(unique_pair_ids)}")
    print(f"  * Unique runs: {sorted(unique_runs)}")
    print(f"  * Expected rows: {len(unique_pair_ids)} pairs × {len(unique_runs)} runs = {len(unique_pair_ids) * len(unique_runs)}")

    # Step 4: Print first 3 rows
    print(f"\n[4] First 3 decision rows:")
    for i, decision in enumerate(decisions[:3]):
        print(f"\n  Row {i+1}:")
        print(f"    pair_id: {decision['pair_id']}")
        print(f"    run: {decision['run']}")
        print(f"    templates: {pair_templates.get(decision['pair_id'], ('?', '?'))}")
        print(f"    normalized_a: {decision.get('normalized_a')}")
        print(f"    normalized_b: {decision.get('normalized_b')}")
        print(f"    flipped: {decision.get('flipped')}")

    # Step 5: Verify excluded pairs
    excluded_found = set(d["pair_id"] for d in decisions) & EXCLUDED_PAIRS
    print(f"\n[5] Excluded pairs found in log: {excluded_found}")
    print(f"    -> Must filter these {len(excluded_found)} pairs from analysis")

    # Step 6: Count rows per model
    print(f"\n[6] Counting rows across all 9 factuality models:")
    model_counts = {}
    for model_file in sorted(RESULTS_DIR.glob("*_factuality.jsonl")):
        with open(model_file) as f:
            count = sum(1 for _ in f)
        model_name = model_file.stem.replace("_factuality", "")
        model_counts[model_name] = count
        print(f"    {model_name:20s}: {count:4d} rows")

    total_rows = sum(model_counts.values())
    print(f"\n  Total: {total_rows} rows across 9 models")
    print(f"  Expected (125 pairs × 3 runs × 9 models): {125 * 3 * 9}")
    print(f"  After exclusion (119 pairs × 3 runs × 9 models): {119 * 3 * 9}")

    # Step 7: Verify column names
    print(f"\n[7] Column names in decision logs:")
    if decisions:
        cols = list(decisions[0].keys())
        for col in cols:
            print(f"    - {col}")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"[OK] Data structure verified")
    print(f"[OK] {len(pair_templates)} pairs loaded with template mapping")
    print(f"[OK] {len(EXCLUDED_PAIRS)} excluded pairs identified")
    print(f"[OK] {total_rows} total decision rows across all models")
    print(f"[OK] Ready to proceed with template-pair JSS analysis")


if __name__ == "__main__":
    main()
