"""The Claude Code ingest path must not be able to contaminate real results.

Table 1 once shipped "goodjudge & factuality & 10 & 1.000" because a test wrote
a synthetic cell into a tracked artifact. This pipeline writes judge cells, so
it is the same hazard with a new name: a dry run of ingest wrote a fabricated
cell straight into data/results_v2/raw/, where nothing distinguishes it from a
paid run.
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
    "cc_run", ROOT / "scripts" / "claude_code_run.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "WORK", tmp_path / "work")
    monkeypatch.setattr(cc, "RAW", tmp_path / "raw")
    monkeypatch.setattr(cc, "STAGING", tmp_path / "staged")
    return tmp_path


def _prepare(sandbox, judge="cc-test", task="factuality", batch=10):
    cc.prepare(judge, task, batch, seed=1)
    manifest = json.loads(
        (cc.WORK / judge / task / "manifest.json").read_text(encoding="utf-8"))
    return manifest


def _answer_all(manifest, answer="YES"):
    for entry in manifest["batches"]:
        Path(entry["answer_file"]).write_text(
            "\n".join(json.dumps({"id": i, "answer": answer})
                      for i in entry["ids"]), encoding="utf-8")


def test_ingest_stages_and_does_not_touch_results_by_default(sandbox):
    m = _prepare(sandbox)
    _answer_all(m)
    assert cc.ingest("cc-test", "factuality", allow_partial=False) == 0
    assert (cc.STAGING / "cc-test_factuality.jsonl").exists()
    assert not (cc.RAW / "cc-test_factuality.jsonl").exists(), (
        "ingest wrote into the results directory without --publish")


def test_publish_refuses_to_overwrite_a_paid_cell(sandbox):
    m = _prepare(sandbox)
    _answer_all(m)
    cc.RAW.mkdir(parents=True, exist_ok=True)
    existing = cc.RAW / "cc-test_factuality.jsonl"
    existing.write_text('{"pair_id": "already paid for"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cc.ingest("cc-test", "factuality", allow_partial=False, publish=True)
    assert "already paid for" in existing.read_text(encoding="utf-8")


def test_unanswered_batches_block_the_write(sandbox):
    _prepare(sandbox)                      # no answers written
    assert cc.ingest("cc-test", "factuality", allow_partial=False) == 2
    assert not (cc.STAGING / "cc-test_factuality.jsonl").exists()


def test_a_misaligned_batch_blocks_the_write(sandbox):
    m = _prepare(sandbox)
    _answer_all(m)
    first = m["batches"][0]
    lines = [json.dumps({"id": i, "answer": "YES"}) for i in first["ids"][:-1]]
    Path(first["answer_file"]).write_text("\n".join(lines), encoding="utf-8")
    assert cc.ingest("cc-test", "factuality", allow_partial=False) == 2


def test_records_are_written_under_their_own_budget_policy(sandbox):
    """Not 'matched'. The harness exposes no max_tokens and no temperature, so
    claiming the matched-budget control would be false, and regenerate_results
    filters on this field -- mislabelling would pool an uncontrolled judge with
    the controlled ones and nothing downstream would notice."""
    m = _prepare(sandbox)
    _answer_all(m)
    cc.ingest("cc-test", "factuality", allow_partial=False)
    rows = [json.loads(l) for l in
            (cc.STAGING / "cc-test_factuality.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
    assert rows, "nothing written"
    for r in rows:
        assert r["budget_policy"] == "claude_code_batched"
        assert r["transport"] == "claude_code_subagent"
        assert r["decoding"]["comparable_to_api_judges"] is False
        assert r["decoding"]["temperature"] is None


def test_every_row_carries_the_cluster_unit(sandbox):
    """item_id is the mandatory resampling unit; a row without it cannot be
    clustered and would silently fall back to a looser interval."""
    m = _prepare(sandbox)
    _answer_all(m)
    cc.ingest("cc-test", "factuality", allow_partial=False)
    for line in (cc.STAGING / "cc-test_factuality.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            assert json.loads(line).get("item_id")
