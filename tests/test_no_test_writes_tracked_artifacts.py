"""
The test suite must not write into files the paper ships.

A test that called regenerate_results.main() redirected RAW and OUT_JSON but not
OUT_TEX, so every run overwrote tables/main_results_v2.tex with its synthetic
fixture. The suite runs before committing, so the table that reached the
repository read "goodjudge & factuality & 10 & 1.000" while the Results prose
discussed the real measurements -- and the manuscript \input's that file, so
Table 1 would have rendered the smoke-test row in the submitted PDF.

Nothing in the pipeline detected it: every test passed, the paper compiled, and
the reproducibility statement claiming each table is regenerated from committed
raw outputs was, at that moment, false.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Files the manuscript renders or the release ships. A test run must leave every
# one of these byte-identical.
TRACKED_ARTIFACTS = [
    "tables/main_results_v2.tex",
    "data/results_v2/metrics_summary.json",
    "data/results_v2/shortcut_controls.json",
    "data/v2/factuality.jsonl",
    "data/v2/coherence.jsonl",
    "data/v2/relevance.jsonl",
    "data/v2/preference.jsonl",
    "data/v2/FROZEN.json",
]


def _digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.mark.parametrize("relpath", TRACKED_ARTIFACTS)
def test_artifact_is_not_modified_by_the_test_suite(relpath):
    """Runs the rest of the suite in a subprocess and checks this file is intact.

    Deselects itself to avoid unbounded recursion. Skips when the artifact is
    absent, so a fresh clone without a completed run does not fail.
    """
    path = ROOT / relpath
    if not path.exists():
        pytest.skip(f"{relpath} not present in this checkout")

    before = _digest(path)
    subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "-x",
         "--deselect", str(Path(__file__)), "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, timeout=1800,
    )
    after = _digest(path)
    assert before == after, (
        f"{relpath} was modified by running the test suite. A test is writing to "
        f"a tracked artifact -- redirect the module-level output path with "
        f"monkeypatch, as tests/test_undefined_metrics.py does for OUT_JSON and "
        f"OUT_TEX."
    )
