#!/usr/bin/env python3
"""
Task 2 — Recompute factuality JSS with T4 polarity correction.

JSS_original: raw normalized_a vs normalized_b (no inversion)
JSS_fixed   : T4 decisions inverted before comparison
delta       : JSS_fixed - JSS_original
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
from src.metrics import judge_sensitivity_score, EXCLUDED_PAIRS

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


def _identify_template(prompt_text: str) -> str:
    part = prompt_text.split("\n\nResponse:")[0] if "\n\nResponse:" in prompt_text else prompt_text
    return next((k for k, v in TEMPLATES.items() if v == part), "UNKNOWN")


def load_pair_templates() -> dict:
    out = {}
    with open(PAIRS_FILE) as fh:
        for line in fh:
            p = json.loads(line)
            out[p["pair_id"]] = (
                _identify_template(p["prompt_a"]),
                _identify_template(p["prompt_b"]),
            )
    return out


def _invert(decision: str) -> str:
    return "NO" if decision == "YES" else "YES"


def compute_model_jss(model: str, pair_templates: dict) -> dict:
    raw_a, raw_b, fix_a, fix_b = [], [], [], []

    log = RESULTS_DIR / f"{model}_factuality.jsonl"
    with open(log) as fh:
        for line in fh:
            rec = json.loads(line)
            pid = rec["pair_id"]
            if pid in EXCLUDED_PAIRS:
                continue
            if rec.get("error") is not None:
                continue
            da = rec.get("normalized_a")
            db = rec.get("normalized_b")
            if not da or not db or da == "UNCLEAR" or db == "UNCLEAR":
                continue

            ta, tb = pair_templates.get(pid, ("UNKNOWN", "UNKNOWN"))

            raw_a.append(da)
            raw_b.append(db)

            fix_a.append(_invert(da) if ta == "T4" else da)
            fix_b.append(_invert(db) if tb == "T4" else db)

    jss_orig  = judge_sensitivity_score(raw_a, raw_b)
    jss_fixed = judge_sensitivity_score(fix_a, fix_b)
    return {
        "model":        model,
        "JSS_fixed":    round(jss_fixed, 4),
        "JSS_original": round(jss_orig,  4),
        "delta":        round(jss_fixed - jss_orig, 4),
        "N":            len(raw_a),
    }


def main():
    print("=" * 70)
    print("TASK 2 — Factuality JSS (Bug-Fixed)")
    print("=" * 70)

    pair_templates = load_pair_templates()
    rows = [compute_model_jss(m, pair_templates) for m in MODELS]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "factuality_jss_fixed.csv", index=False)

    print(f"\n{'Model':<20} {'JSS_fixed':>10} {'JSS_orig':>10} {'delta':>8} {'N':>5}")
    print("-" * 58)
    for _, r in df.iterrows():
        flag = "  ***" if abs(r["delta"]) > 0.05 else ""
        print(f"{r['model']:<20} {r['JSS_fixed']:>10.4f} {r['JSS_original']:>10.4f} "
              f"{r['delta']:>8.4f} {int(r['N']):>5}{flag}")

    if (df["delta"].abs() > 0.05).any():
        print("\nBUG IMPACT CONFIRMED -- T4 polarity inversion changed JSS materially")

    print(f"\nSaved: {OUTPUT_DIR / 'factuality_jss_fixed.csv'}")


if __name__ == "__main__":
    main()
