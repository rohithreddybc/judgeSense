"""Task 3 — tests for per-template factuality outputs."""
import re
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
T4_CSV = ROOT / "outputs" / "factuality_T4_vs_noT4.csv"
TP_CSV = ROOT / "outputs" / "factuality_per_template_pair_JSS.csv"

MODELS = [
    "claude-haiku", "claude-sonnet", "deepseek", "gemini-flash",
    "gpt-4o-mini", "gpt-4o", "llama3-70b", "mistral-7b", "qwen",
]
PAIR_RE = re.compile(r"^T[1-5]-T[1-5]$")


@pytest.fixture(scope="module")
def df_t4():
    assert T4_CSV.exists(), f"Missing: {T4_CSV}"
    return pd.read_csv(T4_CSV)


@pytest.fixture(scope="module")
def df_tp():
    assert TP_CSV.exists(), f"Missing: {TP_CSV}"
    return pd.read_csv(TP_CSV)


def test_t4_csv_columns(df_t4):
    expected = {"model","JSS_no_T4","JSS_with_T4_corrected","delta","N_no_T4","N_with_T4"}
    assert expected.issubset(df_t4.columns)


def test_t4_all_models(df_t4):
    assert set(df_t4["model"]) == set(MODELS)


def test_t4_jss_in_range(df_t4):
    for col in ("JSS_no_T4", "JSS_with_T4_corrected"):
        assert (df_t4[col] >= 0).all() and (df_t4[col] <= 1).all()


def test_tp_csv_columns(df_tp):
    assert {"model","template_pair","JSS","N"}.issubset(df_tp.columns)


def test_tp_jss_in_range(df_tp):
    assert (df_tp["JSS"] >= 0).all() and (df_tp["JSS"] <= 1).all()


def test_tp_pair_format(df_tp):
    bad = [p for p in df_tp["template_pair"] if not PAIR_RE.match(p)]
    assert bad == [], f"Invalid template_pair values: {bad}"


def test_tp_pair_ordered(df_tp):
    """First template index must be less than second (T1-T2, not T2-T1)."""
    for pair in df_tp["template_pair"].unique():
        a, b = pair.split("-")
        assert a < b, f"Pair {pair} is not ordered (expected {b}-{a} → {a}-{b})"


def test_tp_only_existing_pairs(df_tp):
    existing = {"T1-T2","T1-T5","T2-T3","T3-T4","T4-T5"}
    found = set(df_tp["template_pair"].unique())
    assert found.issubset(existing), f"Unexpected pairs: {found - existing}"
