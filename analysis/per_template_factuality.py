#!/usr/bin/env python3
"""
Task 3 — Per-template factuality JSS analysis.

GROUP A (no T4): (T1,T2),(T1,T3),(T1,T5),(T2,T3),(T2,T5),(T3,T5)
GROUP B (T4 corrected): (T1,T4),(T2,T4),(T3,T4),(T4,T5)

Only pairs that actually exist in the dataset are included.
T4 decisions are always inverted before comparison.

Outputs:
  outputs/factuality_T4_vs_noT4.csv
  outputs/factuality_per_template_pair_JSS.csv
"""
import json
import sys
from itertools import combinations
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

GROUP_A = {frozenset(p) for p in [
    ("T1","T2"),("T1","T3"),("T1","T5"),
    ("T2","T3"),("T2","T5"),("T3","T5"),
]}
GROUP_B = {frozenset(p) for p in [
    ("T1","T4"),("T2","T4"),("T3","T4"),("T4","T5"),
]}


def identify_template(prompt_text: str) -> str:
    part = prompt_text.split("\n\nResponse:")[0] if "\n\nResponse:" in prompt_text else prompt_text
    return next((k for k, v in TEMPLATES.items() if v == part), "UNKNOWN")


def _invert(decision: str) -> str:
    return "NO" if decision == "YES" else "YES"


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


def load_decisions(model: str) -> dict:
    """Returns {(pair_id, run): (norm_a, norm_b)}."""
    out = {}
    with open(RESULTS_DIR / f"{model}_factuality.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("error") is not None:
                continue
            da, db = rec.get("normalized_a"), rec.get("normalized_b")
            if not da or not db or "UNCLEAR" in (da, db):
                continue
            out[(rec["pair_id"], rec["run"])] = (da, db)
    return out


def compute_pair_jss(pair_templates, decisions, ta_want, tb_want):
    """Compute T4-corrected JSS for one (ta_want, tb_want) template combination."""
    a_list, b_list = [], []
    for pid, (ta, tb) in pair_templates.items():
        if pid in EXCLUDED_PAIRS:
            continue
        # Match either direction
        if not ({ta, tb} == {ta_want, tb_want}):
            continue
        for run in (1, 2, 3):
            entry = decisions.get((pid, run))
            if entry is None:
                continue
            da, db = entry
            # Orient so ta side is 'a'
            if ta != ta_want:
                da, db = db, da
                ta, tb = tb, ta
            da_f = _invert(da) if ta == "T4" else da
            db_f = _invert(db) if tb == "T4" else db
            a_list.append(da_f)
            b_list.append(db_f)
    if not a_list:
        return None, 0
    return judge_sensitivity_score(a_list, b_list), len(a_list)


def main():
    print("=" * 70)
    print("TASK 3 — Per-Template Factuality Analysis")
    print("=" * 70)

    pair_templates = load_pair_templates()

    # Which template pairs actually exist in the dataset
    existing_pairs = set()
    for pid, (ta, tb) in pair_templates.items():
        if pid not in EXCLUDED_PAIRS:
            existing_pairs.add(frozenset([ta, tb]))

    t4_rows, tp_rows = [], []

    for model in MODELS:
        decisions = load_decisions(model)

        grp_a_vals, grp_b_vals = [], []
        grp_a_n, grp_b_n = 0, 0

        # Per-template-pair JSS
        all_tnames = list(TEMPLATES.keys())
        for i, ta in enumerate(all_tnames):
            for tb in all_tnames[i+1:]:
                if frozenset([ta, tb]) not in existing_pairs:
                    continue
                jss, n = compute_pair_jss(pair_templates, decisions, ta, tb)
                if jss is None:
                    continue
                label = f"{ta}-{tb}"
                tp_rows.append({"model": model, "template_pair": label,
                                "JSS": round(jss, 4), "N": n})

                fs = frozenset([ta, tb])
                if fs in GROUP_A:
                    grp_a_vals.append(jss); grp_a_n += n
                elif fs in GROUP_B:
                    grp_b_vals.append(jss); grp_b_n += n

        jss_a = sum(grp_a_vals) / len(grp_a_vals) if grp_a_vals else None
        jss_b = sum(grp_b_vals) / len(grp_b_vals) if grp_b_vals else None
        delta = round(jss_a - jss_b, 4) if (jss_a and jss_b) else None
        t4_rows.append({
            "model": model,
            "JSS_no_T4":             round(jss_a, 4) if jss_a else None,
            "JSS_with_T4_corrected": round(jss_b, 4) if jss_b else None,
            "delta": delta,
            "N_no_T4":   grp_a_n,
            "N_with_T4": grp_b_n,
        })

    # Write outputs
    df_t4 = pd.DataFrame(t4_rows)
    df_tp = pd.DataFrame(tp_rows).sort_values(["template_pair", "JSS"], ascending=[True, False])

    df_t4.to_csv(OUTPUT_DIR / "factuality_T4_vs_noT4.csv", index=False)
    df_tp.to_csv(OUTPUT_DIR / "factuality_per_template_pair_JSS.csv", index=False)

    # Console summary
    print(f"\n{'Model':<20} {'JSS_no_T4':>10} {'JSS+T4_corr':>13} {'delta':>8}")
    print("-" * 56)
    for _, r in df_t4.iterrows():
        flag = "  ***" if r["delta"] and abs(r["delta"]) > 0.05 else ""
        print(f"{r['model']:<20} {r['JSS_no_T4']:>10.4f} "
              f"{r['JSS_with_T4_corrected']:>13.4f} {r['delta']:>8.4f}{flag}")

    print(f"\nSaved:")
    print(f"  {OUTPUT_DIR / 'factuality_T4_vs_noT4.csv'}")
    print(f"  {OUTPUT_DIR / 'factuality_per_template_pair_JSS.csv'}")


if __name__ == "__main__":
    main()
