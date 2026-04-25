"""Task 5 — tests for generated figure PDFs."""
from pathlib import Path
import pytest

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

FIGURES = [
    "fig1_coherence_jss_bar.pdf",
    "fig2_jss_heatmap.pdf",
    "fig4_factuality_raw_vs_corrected.pdf",
]


@pytest.mark.parametrize("filename", FIGURES)
def test_figure_exists(filename):
    path = OUTPUT_DIR / filename
    assert path.exists(), f"Missing figure: {path}"


@pytest.mark.parametrize("filename", FIGURES)
def test_figure_size(filename):
    path = OUTPUT_DIR / filename
    assert path.stat().st_size > 10_000, (
        f"{filename} is suspiciously small: {path.stat().st_size} bytes"
    )
