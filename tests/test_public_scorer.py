"""The public scorer must agree with the pipeline that produced the paper.

scripts/judgesense_score.py exists so a third party can compute ΔJSS on their
own judge without our provider clients, our keys, or our runner. It is a second
implementation of the same measurement, which makes it worth exactly as much as
its agreement with the first: a scorer that quietly disagrees with the published
numbers would have every outside user reporting figures incomparable to ours.

This runs the shipped scorer over committed judge output and checks it lands on
the committed metrics.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "judgesense_score", ROOT / "scripts" / "judgesense_score.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

JUDGE = "claude-haiku"
SUMMARY = ROOT / "data" / "results_v2" / "metrics_summary.json"


def _predictions(tmp_path: Path, tasks) -> Path:
    rows = []
    for task in tasks:
        raw = ROOT / "data" / "results_v2" / "raw" / f"{JUDGE}_{task}.jsonl"
        if not raw.exists():
            pytest.skip(f"{raw.name} not present")
        for line in raw.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "pair_id": r["pair_id"],
                "decision_a": r.get("decision_a"),
                "decision_b": r.get("decision_b"),
                "decision_a_repeat": r.get("decision_a_repeat"),
                "decision_b_repeat": r.get("decision_b_repeat"),
            })
    path = tmp_path / "preds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("task", ["factuality", "coherence"])
def test_scorer_reproduces_the_committed_metrics(tmp_path, task):
    if not SUMMARY.exists():
        pytest.skip("metrics_summary.json not present")
    committed = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cell = committed.get(JUDGE, {}).get(task)
    if not cell or not (cell.get("jss_repeat_delta") or {}).get("delta"):
        pytest.skip(f"no committed delta for {JUDGE}/{task}")

    got = scorer.score(_predictions(tmp_path, [task]), only_task=task,
                       n_bootstrap=200)["tasks"][task]
    exp = cell["jss_repeat_delta"]

    assert got["jss"] == pytest.approx(cell["jss_strict"], abs=1e-4)
    assert got["jss_repeat"] == pytest.approx(exp["jss_rep"], abs=1e-4)
    assert got["delta_jss"] == pytest.approx(exp["delta"], abs=1e-4)


def test_missing_repeat_arms_yield_no_delta_and_say_why(tmp_path):
    """Raw JSS without a ceiling cannot separate wording sensitivity from a
    judge that disagrees with itself. The tool must refuse to imply otherwise."""
    path = tmp_path / "p.jsonl"
    raw = ROOT / "data" / "results_v2" / "raw" / f"{JUDGE}_factuality.jsonl"
    if not raw.exists():
        pytest.skip("no committed output")
    rows = []
    for line in raw.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows.append({"pair_id": r["pair_id"], "decision_a": r.get("decision_a"),
                         "decision_b": r.get("decision_b")})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    got = scorer.score(path, only_task="factuality", n_bootstrap=100)["tasks"]["factuality"]
    assert got["delta_jss"] is None
    assert "repeat" in got["note"].lower()


def test_unrecognised_labels_are_malformed_not_coerced(tmp_path):
    """The paper's parser never rounds an unmappable answer to the nearest
    label; neither may the public scorer, or outside numbers would be inflated
    relative to ours."""
    bench = scorer._load_benchmark()
    pid = next(p for p, r in bench.items() if r["_task"] == "factuality")
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(
        {"pair_id": pid, "decision_a": "probably yes", "decision_b": "YES"}),
        encoding="utf-8")
    got = scorer.score(path, only_task="factuality", n_bootstrap=50)["tasks"]["factuality"]
    assert got["malformed_rate"] == pytest.approx(0.5)
