"""
Regression: a judge whose every output is unparseable must be reportable, not fatal.

`chance_corrected_jss`, `decision_entropy` and `quadratic_weighted_kappa` are
mathematically undefined when no record carries a parseable decision, and they
raise ValueError to say so. `metrics_for_cell` used to call them unguarded and
`main` looped over raw files with no exception handling, so a single
100%-malformed judge aborted metric regeneration for EVERY judge in a run that
had already been paid for. A cell that is entirely malformed is a real result
about that judge and must be reported as malformed_rate 1.0 with the undefined
metrics null.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _regen():
    spec = importlib.util.spec_from_file_location(
        "regen", ROOT / "scripts" / "regenerate_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unclear_recs(task, n=250):
    return [{"pair_id": f"p{i}", "item_id": f"i{i}", "task_type": task,
             "decision_a": "UNCLEAR", "decision_b": "UNCLEAR",
             "ground_truth_label": "accurate", "ground_truth_position": "A",
             "error": None}
            for i in range(n)]


@pytest.mark.parametrize("task", ["factuality", "coherence", "relevance", "preference"])
def test_all_unclear_cell_is_reported_not_raised(task):
    out = _regen().metrics_for_cell(_unclear_recs(task), task)
    assert out["malformed_rate"] == 1.0
    assert out["chance_corrected_jss"] is None, "undefined must be null, not a number"
    assert out["n_rows"] == 250


@pytest.mark.parametrize("task", ["factuality", "coherence", "relevance", "preference"])
def test_undefined_metrics_never_serialise_as_negative_zero(task):
    import json
    out = _regen().metrics_for_cell(_unclear_recs(task), task)
    assert "-0.0" not in json.dumps(out)


def test_pairwise_accuracy_is_null_when_nothing_is_scorable():
    out = _regen().metrics_for_cell(_unclear_recs("relevance"), "relevance")
    assert out["pairwise"] == {"accuracy": None, "n": 0}


def test_one_broken_cell_does_not_void_the_other_cells(tmp_path, monkeypatch, capsys):
    """main() must record the failure for the offending cell and keep going."""
    import json
    regen = _regen()
    raw = tmp_path / "raw"
    raw.mkdir()
    good = [{"pair_id": f"g{i}", "item_id": f"i{i}", "task_type": "factuality",
             "decision_a": "YES", "decision_b": "YES", "ground_truth_label": "accurate",
             "error": None} for i in range(10)]
    with open(raw / "goodjudge_factuality.jsonl", "w", encoding="utf-8") as fh:
        for r in good:
            fh.write(json.dumps(r) + "\n")
    with open(raw / "badjudge_factuality.jsonl", "w", encoding="utf-8") as fh:
        for r in _unclear_recs("factuality", n=10):
            fh.write(json.dumps(r) + "\n")

    # force a hard failure for the bad cell only, to prove main() isolates it
    real = regen.metrics_for_cell
    def flaky(recs, task):
        if any(r["decision_a"] == "UNCLEAR" for r in recs):
            raise RuntimeError("synthetic cell failure")
        return real(recs, task)
    monkeypatch.setattr(regen, "metrics_for_cell", flaky)
    monkeypatch.setattr(regen, "RAW", raw)
    monkeypatch.setattr(regen, "OUT_JSON", tmp_path / "metrics.json")

    regen.main([]) if regen.main.__code__.co_argcount else regen.main()
    summary = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "jss_strict" in summary["goodjudge"]["factuality"], "good cell must survive"
    assert "error" in summary["badjudge"]["factuality"], "bad cell must be recorded"
