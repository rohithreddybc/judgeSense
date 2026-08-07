"""
Tests for scripts/data_audit.py.

Datasets written here are TEST FIXTURES living only in tmp_path; they exist
to prove each audit check fires (and passes) correctly and are never placed
under data/.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("data_audit", REPO / "scripts" / "data_audit.py")
data_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_audit)

THRESHOLDS = {
    "min_unique_items": 10,
    "max_duplicate_row_ratio": 0.05,
    "max_label_share": 0.9,
    "min_distinct_labels": 2,
    "cluster_unit": "item",
    "min_clusters": 10,
    "min_seconds_per_decision": 2.0,
}


def good_record(i, task="factuality"):
    return {
        "pair_id": f"{task}_{i}",
        "task_type": task,
        "item_id": f"item_{i}",
        "prompt_pair_id": f"item_{i}#T1-T2",
        "source_benchmark": "truthful_qa",
        "source": {
            "source_dataset": "truthful_qa",
            "source_split": "validation",
            "source_record_id": f"validation[{i}]",
        },
        "prompt_a": f"variant A {i}",
        "prompt_b": f"variant B {i}",
        "response_being_judged": f"unique item text {i}",
        "ground_truth_label": "accurate" if i % 2 else "inaccurate",
    }


def write_split(tmp_path, name, records):
    path = tmp_path / f"{name}.jsonl"
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def run_audit(tmp_path, records, thresholds=None, validation_dir=None, split="factuality"):
    write_split(tmp_path, split, records)
    cfg = {
        "dataset_dir": str(tmp_path.relative_to(REPO)) if str(tmp_path).startswith(str(REPO)) else str(tmp_path),
        "splits": [split],
        "thresholds": thresholds or THRESHOLDS,
    }
    if validation_dir is not None:
        cfg["validation_dir"] = str(validation_dir)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    # audit() resolves dataset_dir against REPO; use absolute paths in tests
    cfg["dataset_dir"] = str(tmp_path)
    cfg_path.write_text(json.dumps(cfg))
    return data_audit.audit(cfg_path)


def by_check(report, name):
    return [r for r in report["results"] if r["check"] == name]


# ── Happy path ───────────────────────────────────────────────────────────────

def test_clean_dataset_passes_all_checks(tmp_path):
    report = run_audit(tmp_path, [good_record(i) for i in range(20)])
    assert report["passed"], report["results"]


# ── unique_items ─────────────────────────────────────────────────────────────

def test_low_unique_items_fails(tmp_path):
    # 20 rows but only 3 unique judged items (the v1 coherence pattern)
    records = [good_record(i) for i in range(20)]
    for i, rec in enumerate(records):
        rec["response_being_judged"] = f"item text {i % 3}"
    report = run_audit(tmp_path, records)
    check = by_check(report, "unique_items")[0]
    assert not check["passed"]
    assert check["observed"] == 3


# ── duplicate_rows ───────────────────────────────────────────────────────────

def test_duplicated_rows_fail_even_with_fresh_pair_ids(tmp_path):
    records = [good_record(i) for i in range(10)]
    clones = []
    for i in range(10):
        clone = dict(records[i % 2])
        clone["pair_id"] = f"clone_{i}"  # new id must not launder a dup row
        clones.append(clone)
    report = run_audit(tmp_path, records + clones)
    check = by_check(report, "duplicate_rows")[0]
    assert not check["passed"]
    assert check["observed"] >= 0.4


# ── provenance ───────────────────────────────────────────────────────────────

def test_bare_benchmark_name_without_record_id_fails(tmp_path):
    records = [good_record(i) for i in range(20)]
    for rec in records[:5]:
        rec["source"] = {}  # v1 shape: a claim with nothing behind it
    report = run_audit(tmp_path, records)
    check = by_check(report, "provenance")[0]
    assert not check["passed"]
    assert check["observed"] == 5


def test_record_id_equal_to_benchmark_name_does_not_count(tmp_path):
    records = [good_record(i) for i in range(20)]
    records[0]["source"]["source_record_id"] = "truthful_qa"  # a name, not an id
    report = run_audit(tmp_path, records)
    assert not by_check(report, "provenance")[0]["passed"]


# ── label_degeneracy ─────────────────────────────────────────────────────────

def test_single_label_split_fails(tmp_path):
    records = [good_record(i) for i in range(20)]
    for rec in records:
        rec["ground_truth_label"] = "A"  # v1 pre-swap pairwise shape
    report = run_audit(tmp_path, records)
    check = by_check(report, "label_degeneracy")[0]
    assert not check["passed"]
    assert check["observed"]["distinct"] == 1


def test_dominant_label_fails_share_threshold(tmp_path):
    records = [good_record(i) for i in range(20)]
    for rec in records[:19]:
        rec["ground_truth_label"] = "accurate"
    report = run_audit(tmp_path, records)
    assert not by_check(report, "label_degeneracy")[0]["passed"]


# ── effective_sample_size ────────────────────────────────────────────────────

def test_few_clusters_fail_despite_many_rows(tmp_path):
    records = [good_record(i) for i in range(40)]
    for i, rec in enumerate(records):
        rec["item_id"] = f"item_{i % 4}"  # 40 rows, 4 items
    report = run_audit(tmp_path, records)
    check = by_check(report, "effective_sample_size")[0]
    assert not check["passed"]
    assert check["observed"] == 4


def test_legacy_records_without_item_id_cluster_on_judged_text(tmp_path):
    records = [good_record(i) for i in range(20)]
    for i, rec in enumerate(records):
        del rec["item_id"]
        rec["response_being_judged"] = f"text {i % 5}"
    report = run_audit(tmp_path, records)
    check = by_check(report, "effective_sample_size")[0]
    assert check["observed"] == 5
    assert not check["passed"]


# ── annotation_timing ────────────────────────────────────────────────────────

def make_validation_file(directory, name, gap_seconds, n=30):
    directory.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    with open(directory / name, "w") as fh:
        for i in range(n):
            stamp = (t0 + timedelta(seconds=i * gap_seconds)).isoformat()
            fh.write(json.dumps({"pair_id": f"p{i}", "manual_label": "YES",
                                 "timestamp": stamp}) + "\n")


def test_subsecond_annotation_timing_fails(tmp_path):
    vdir = tmp_path / "manual"
    make_validation_file(vdir, "relevance_manual.jsonl", gap_seconds=0.7)
    report = run_audit(tmp_path, [good_record(i) for i in range(20)],
                       validation_dir=vdir)
    checks = [r for r in by_check(report, "annotation_timing") if r["split"]]
    assert len(checks) == 1
    assert not checks[0]["passed"]
    assert checks[0]["observed"] == pytest.approx(0.7, abs=0.01)


def test_plausible_annotation_timing_passes(tmp_path):
    vdir = tmp_path / "manual"
    make_validation_file(vdir, "factuality_manual.jsonl", gap_seconds=6.0)
    report = run_audit(tmp_path, [good_record(i) for i in range(20)],
                       validation_dir=vdir)
    checks = [r for r in by_check(report, "annotation_timing") if r["split"]]
    assert checks[0]["passed"]


# ── structural ───────────────────────────────────────────────────────────────

def test_missing_split_file_fails_loudly(tmp_path):
    cfg = {"dataset_dir": str(tmp_path), "splits": ["factuality", "ghost"],
           "thresholds": THRESHOLDS}
    write_split(tmp_path, "factuality", [good_record(i) for i in range(20)])
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    report = data_audit.audit(cfg_path)
    assert not report["passed"]
    assert any(r["check"] == "split_present" and not r["passed"]
               for r in report["results"])


def test_gate_fails_on_shipped_v1_data():
    # The real config against the real v1 data: the gate MUST fail, because
    # the defects it was built for are present in the shipped dataset.
    report = data_audit.audit(REPO / "data" / "audit_config.json")
    assert not report["passed"]
    failing = {r["check"] for r in report["results"] if not r["passed"]}
    assert {"unique_items", "duplicate_rows", "provenance",
            "effective_sample_size", "annotation_timing"} <= failing


# ── evidence-absence must fail, not skip ─────────────────────────────────────
# A declared validation_dir is the claim that human validation happened. If the
# evidence is missing or untimestamped the gate cannot verify that claim, and an
# unverifiable claim must fail — otherwise deleting the records turns CI green.

def test_declared_validation_dir_with_no_records_fails(tmp_path):
    vdir = tmp_path / "manual"
    vdir.mkdir()
    report = run_audit(tmp_path, [good_record(i) for i in range(20)],
                       validation_dir=vdir)
    checks = by_check(report, "annotation_timing")
    assert len(checks) == 1
    assert not checks[0]["passed"], "empty validation dir must fail, not skip"
    assert not report["passed"]


def test_declared_validation_dir_that_does_not_exist_fails(tmp_path):
    report = run_audit(tmp_path, [good_record(i) for i in range(20)],
                       validation_dir=tmp_path / "does_not_exist")
    checks = by_check(report, "annotation_timing")
    assert len(checks) == 1
    assert not checks[0]["passed"]
    assert not report["passed"]


def test_validation_records_without_timestamps_fail(tmp_path):
    vdir = tmp_path / "manual"
    vdir.mkdir()
    with open(vdir / "coherence_manual.jsonl", "w") as fh:
        for i in range(25):
            fh.write(json.dumps({"pair_id": f"p{i}", "manual_label": "YES"}) + "\n")
    report = run_audit(tmp_path, [good_record(i) for i in range(20)],
                       validation_dir=vdir)
    checks = [r for r in by_check(report, "annotation_timing") if r["split"]]
    assert len(checks) == 1
    assert not checks[0]["passed"], "stripping timestamps must not turn the check green"
    assert checks[0]["observed"] is None


# ── label degeneracy scores the JUDGE'S decision space ───────────────────────
# For a swap-design pairwise task the content label is constant by construction
# (in relevance the qrels-positive document is always the relevant one), while
# the scored decision is positional and balanced. Scoring the content label
# would fail a sound dataset; scoring position must still catch a real one.

def pairwise_record(i, position):
    rec = good_record(i)
    rec["task_type"] = "relevance"
    rec["ground_truth_label"] = "candidate_relevant"   # constant by construction
    rec["ground_truth_position"] = position
    return rec


def test_balanced_position_passes_despite_constant_content_label(tmp_path):
    records = [pairwise_record(i, "A" if i % 2 == 0 else "B") for i in range(20)]
    report = run_audit(tmp_path, records, split="relevance")
    checks = by_check(report, "label_degeneracy")
    assert checks[0]["passed"], checks[0]
    assert checks[0]["observed"]["scored_field"] == "ground_truth_position"


def test_degenerate_position_still_fails(tmp_path):
    records = [pairwise_record(i, "A") for i in range(20)]
    report = run_audit(tmp_path, records, split="relevance")
    checks = by_check(report, "label_degeneracy")
    assert not checks[0]["passed"], "always-A ground truth must still be caught"
    assert checks[0]["observed"]["scored_field"] == "ground_truth_position"


def test_pointwise_still_scored_on_content_label(tmp_path):
    report = run_audit(tmp_path, [good_record(i) for i in range(20)])
    checks = by_check(report, "label_degeneracy")
    assert checks[0]["observed"]["scored_field"] == "ground_truth_label"


def test_degenerate_pointwise_label_fails(tmp_path):
    records = [good_record(i) for i in range(20)]
    for r in records:
        r["ground_truth_label"] = "accurate"
    report = run_audit(tmp_path, records)
    assert not by_check(report, "label_degeneracy")[0]["passed"]


# ── ground-truth consistency ─────────────────────────────────────────────────
# Identical judged content with different correct answers is unsatisfiable: a
# judge answering the same displayed text the same way is scored right once and
# wrong once. Both real instances came from upstream data, not from our code.

def test_contradictory_ground_truth_fails(tmp_path):
    records = [good_record(i) for i in range(20)]
    records[1]["response_being_judged"] = records[0]["response_being_judged"]
    records[1]["ground_truth_label"] = "inaccurate"
    records[0]["ground_truth_label"] = "accurate"
    report = run_audit(tmp_path, records)
    check = by_check(report, "ground_truth_consistency")[0]
    assert not check["passed"]
    assert check["observed"]["n_contradictions"] == 1


def test_duplicated_content_fails_even_when_labels_agree(tmp_path):
    records = [good_record(i) for i in range(20)]
    records[1]["response_being_judged"] = records[0]["response_being_judged"]
    records[1]["ground_truth_label"] = records[0]["ground_truth_label"]
    report = run_audit(tmp_path, records)
    check = by_check(report, "ground_truth_consistency")[0]
    assert not check["passed"]
    assert check["observed"]["n_contradictions"] == 0
    assert check["observed"]["n_duplicated_content"] == 1


def test_distinct_content_passes(tmp_path):
    report = run_audit(tmp_path, [good_record(i) for i in range(20)])
    assert by_check(report, "ground_truth_consistency")[0]["passed"]
