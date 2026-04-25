"""Task 2 — tests for factuality_jss_fixed.csv."""
import pandas as pd
import pytest
from pathlib import Path

CSV = Path(__file__).parent.parent / "outputs" / "factuality_jss_fixed.csv"
MODELS = [
    "claude-haiku", "claude-sonnet", "deepseek", "gemini-flash",
    "gpt-4o-mini", "gpt-4o", "llama3-70b", "mistral-7b", "qwen",
]


@pytest.fixture(scope="module")
def df():
    assert CSV.exists(), f"Missing output: {CSV}"
    return pd.read_csv(CSV)


def test_all_models_present(df):
    assert set(df["model"]) == set(MODELS)


def test_jss_fixed_in_range(df):
    assert (df["JSS_fixed"] >= 0).all() and (df["JSS_fixed"] <= 1).all()


def test_jss_original_in_range(df):
    assert (df["JSS_original"] >= 0).all() and (df["JSS_original"] <= 1).all()


def test_n_at_least_100(df):
    assert (df["N"] >= 100).all(), df[df["N"] < 100][["model", "N"]]


def test_delta_equals_difference(df):
    computed = (df["JSS_fixed"] - df["JSS_original"]).round(4)
    assert (computed == df["delta"]).all()


def test_fixed_jss_geq_original(df):
    """T4 inversion should always improve or maintain JSS."""
    assert (df["JSS_fixed"] >= df["JSS_original"]).all()
