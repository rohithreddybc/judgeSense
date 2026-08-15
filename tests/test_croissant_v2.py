"""Tests for scripts/generate_croissant_v2.py (fixture data in tmp_path only)."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "generate_croissant_v2", REPO / "scripts" / "generate_croissant_v2.py"
)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def write_fixture_split(data_dir, task, n=4):
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / f"{task}.jsonl", "w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "pair_id": f"{task}_{i}",
                "item_id": f"{task}_item_{i}",
                "ground_truth_label": "accurate" if i % 2 else "inaccurate",
                "source": {"source_dataset": f"fixture_source_{task}",
                           "source_split": "test",
                           "source_record_id": f"test[{i}]"},
            }) + "\n")


def test_refuses_when_dataset_not_built(tmp_path):
    with pytest.raises(gen.DatasetNotBuiltError, match="not built"):
        gen.build_croissant(tmp_path)


def test_refuses_partial_build(tmp_path):
    write_fixture_split(tmp_path, "factuality")
    with pytest.raises(gen.DatasetNotBuiltError):
        gen.build_croissant(tmp_path)


def test_refuses_empty_split(tmp_path):
    for task in gen.TASKS:
        write_fixture_split(tmp_path, task)
    (tmp_path / "coherence.jsonl").write_text("")
    with pytest.raises(gen.DatasetNotBuiltError, match="empty"):
        gen.build_croissant(tmp_path)


def test_metadata_is_computed_from_data(tmp_path):
    for task in gen.TASKS:
        write_fixture_split(tmp_path, task, n=6)
    croissant = gen.build_croissant(tmp_path)

    assert croissant["rai:hasSyntheticData"] is True
    assert "syntheticDataExplanation" in json.dumps(croissant)
    assert len(croissant["distribution"]) == 4
    for file_obj in croissant["distribution"]:
        assert len(file_obj["sha256"]) == 64
        assert "6 records over 6 unique" in file_obj["description"]
    # source datasets listed in the explanation come from the data itself
    assert "fixture_source_factuality" in croissant["rai:syntheticDataExplanation"]
    # anonymity: no author names anywhere in the generated metadata
    assert "Anonymous" in croissant["creator"]["name"]
