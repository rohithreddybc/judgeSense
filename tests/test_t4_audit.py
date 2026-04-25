"""
Task 1 — T4 polarity audit.

Verifies that T4 decisions are stored as raw model output (inverted polarity)
and that applying inversion improves agreement with T3.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data/results/raw_outputs"
PAIRS_FILE  = ROOT / "data/prompt_pairs/factuality.jsonl"

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


def _identify_template(prompt_text: str) -> str:
    part = prompt_text.split("\n\nResponse:")[0] if "\n\nResponse:" in prompt_text else prompt_text
    return next((k for k, v in TEMPLATES.items() if v == part), "UNKNOWN")


def _load_pair_templates():
    out = {}
    with open(PAIRS_FILE) as fh:
        for line in fh:
            p = json.loads(line)
            out[p["pair_id"]] = (
                _identify_template(p["prompt_a"]),
                _identify_template(p["prompt_b"]),
            )
    return out


def _load_model_decisions(model: str):
    decisions = {}
    with open(RESULTS_DIR / f"{model}_factuality.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            decisions[(rec["pair_id"], rec["run"])] = (
                rec.get("normalized_a"), rec.get("normalized_b")
            )
    return decisions


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pair_templates():
    return _load_pair_templates()


@pytest.fixture(scope="module")
def t3t4_pairs(pair_templates):
    return [
        pid for pid, (ta, tb) in pair_templates.items()
        if pid not in EXCLUDED_PAIRS and {"T3", "T4"} == {ta, tb}
    ]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_t4_pairs_identified(pair_templates, t3t4_pairs):
    """T4 pairs must exist and be correctly mapped."""
    assert len(t3t4_pairs) > 0, "No T3-T4 pairs found — check template mapping"
    # T4-T5 pairs also exist
    t4t5 = [p for p, (a, b) in pair_templates.items()
             if p not in EXCLUDED_PAIRS and {"T4", "T5"} == {a, b}]
    assert len(t4t5) > 0, "No T4-T5 pairs found"


def test_t4_stored_raw_not_corrected(pair_templates, t3t4_pairs):
    """
    For T3-T4 pairs, raw disagreement rate must be > 0.5 across all models.
    This proves T4 is stored with inverted polarity (raw), not pre-corrected.
    If T4 were pre-corrected, T3 and T4 would mostly agree in the raw data.
    """
    all_raw_disagree = 0
    all_total = 0

    for model in MODELS:
        decisions = _load_model_decisions(model)
        for pid in t3t4_pairs:
            ta, tb = pair_templates[pid]
            for run in (1, 2, 3):
                key = (pid, run)
                if key not in decisions:
                    continue
                da, db = decisions[key]
                if da is None or db is None:
                    continue
                # Orient so T3 is always 'a'
                if tb == "T3":
                    da, db = db, da
                all_total += 1
                if da != db:
                    all_raw_disagree += 1

    assert all_total > 0
    raw_disagree_rate = all_raw_disagree / all_total
    assert raw_disagree_rate > 0.5, (
        f"Raw disagreement rate for T3-T4 = {raw_disagree_rate:.3f}; "
        f"expected > 0.50 if T4 is stored with inverted polarity"
    )


def test_inversion_improves_t3t4_agreement(pair_templates, t3t4_pairs):
    """
    After inverting T4 decisions, agreement rate for T3-T4 pairs must be
    higher than the raw agreement rate.
    """
    raw_agree = raw_total = fixed_agree = fixed_total = 0

    for model in MODELS:
        decisions = _load_model_decisions(model)
        for pid in t3t4_pairs:
            ta, tb = pair_templates[pid]
            for run in (1, 2, 3):
                key = (pid, run)
                if key not in decisions:
                    continue
                da, db = decisions[key]
                if da is None or db is None:
                    continue

                raw_total += 1
                if da == db:
                    raw_agree += 1

                # Invert whichever side is T4
                da_f = ("NO" if da == "YES" else "YES") if ta == "T4" else da
                db_f = ("NO" if db == "YES" else "YES") if tb == "T4" else db
                fixed_total += 1
                if da_f == db_f:
                    fixed_agree += 1

    assert raw_total > 0
    raw_jss   = raw_agree   / raw_total
    fixed_jss = fixed_agree / fixed_total
    assert fixed_jss > raw_jss, (
        f"Inversion did not improve agreement: raw={raw_jss:.3f} fixed={fixed_jss:.3f}"
    )


def test_no_unknown_templates(pair_templates):
    """Every pair must resolve to known template names."""
    unknowns = [
        pid for pid, (ta, tb) in pair_templates.items()
        if "UNKNOWN" in (ta, tb)
    ]
    assert unknowns == [], f"Unknown templates for pairs: {unknowns}"
