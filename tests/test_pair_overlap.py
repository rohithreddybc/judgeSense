"""Task 4 — tests for factuality_pair_flip_overlap.csv."""
import pandas as pd
import pytest
from pathlib import Path

CSV = Path(__file__).parent.parent / "outputs" / "factuality_pair_flip_overlap.csv"


@pytest.fixture(scope="module")
def df():
    assert CSV.exists(), f"Missing: {CSV}"
    return pd.read_csv(CSV)


def test_row_count(df):
    assert len(df) == 119, f"Expected 119 rows, got {len(df)}"


def test_flip_count_in_range(df):
    assert (df["flip_count"] >= 0).all() and (df["flip_count"] <= 9).all()


def test_required_columns(df):
    assert {"pair_id", "flip_count", "flipping_models"}.issubset(df.columns)


def test_flipping_models_count_matches(df):
    """flip_count must equal number of models listed in flipping_models."""
    for _, row in df.iterrows():
        models_str = str(row["flipping_models"]) if pd.notna(row["flipping_models"]) else ""
        listed = len(models_str.split(",")) if models_str else 0
        assert row["flip_count"] == listed, (
            f"{row['pair_id']}: flip_count={row['flip_count']} "
            f"but {listed} models listed"
        )


def test_pair_ids_are_factuality(df):
    assert df["pair_id"].str.startswith("fact_").all()
