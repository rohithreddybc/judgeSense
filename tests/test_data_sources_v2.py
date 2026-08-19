"""
Tests for src/data_sources.py.

All dataset payloads in this file are TEST FIXTURES that exist only to
exercise loader logic (provenance propagation, schema validation, failure
behavior). They are never written under data/ and must never be — the
loaders' whole point is that real items come only from real sources.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_sources import (  # noqa: E402
    DataSourceSchemaError,
    DataSourceUnavailableError,
    load_coherence_items,
    load_factuality_items,
    load_preference_items,
    load_relevance_items,
)


class FakeSplit:
    """Minimal stand-in for a datasets.Dataset split (test fixture only)."""

    def __init__(self, rows):
        self.rows = rows

    @property
    def column_names(self):
        return sorted({k for row in self.rows for k in row})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def __iter__(self):
        return iter(self.rows)


def make_loader(splits):
    """Return a fake _loader keyed on (dataset_id, config, split)."""

    def _loader(dataset_id, config, split):
        key = (dataset_id, config, split)
        if key not in splits:
            raise DataSourceUnavailableError(dataset_id, f"fixture has no {key}")
        return splits[key]

    return _loader


# ── Failure policy ───────────────────────────────────────────────────────────

def test_unavailable_source_raises_and_never_falls_back():
    def failing_loader(dataset_id, config, split):
        raise DataSourceUnavailableError(dataset_id, "connection refused (fixture)")

    with pytest.raises(DataSourceUnavailableError) as exc:
        load_factuality_items(n_items=1, _loader=failing_loader)
    assert "refuses to substitute" in str(exc.value)


def test_schema_drift_raises():
    loader = make_loader({
        ("truthful_qa", "generation", "validation"): FakeSplit(
            [{"question": "q", "renamed_answer": "a"}]
        )
    })
    with pytest.raises(DataSourceSchemaError):
        load_factuality_items(n_items=1, _loader=loader)


def test_insufficient_items_raises_instead_of_padding():
    loader = make_loader({
        ("truthful_qa", "generation", "validation"): FakeSplit([
            {"question": "Q1?", "best_answer": "A1", "incorrect_answers": ["W1"]},
        ])
    })
    with pytest.raises(DataSourceSchemaError, match="refusing to pad"):
        load_factuality_items(n_items=50, _loader=loader)


# ── Provenance chain ─────────────────────────────────────────────────────────

def test_factuality_items_carry_real_provenance_and_labels():
    rows = [
        {"question": f"Q{i}?", "best_answer": f"A{i}", "incorrect_answers": [f"W{i}"]}
        for i in range(10)
    ]
    loader = make_loader({("truthful_qa", "generation", "validation"): FakeSplit(rows)})
    items = load_factuality_items(n_items=10, _loader=loader)

    assert len(items) == 10
    labels = {i.ground_truth_label for i in items}
    assert labels == {"accurate", "inaccurate"}
    for item in items:
        src = item.source
        assert src.source_dataset == "truthful_qa"
        assert src.source_split == "validation"
        # record id must point at a specific row, not just name a benchmark
        assert src.source_record_id.startswith("validation[")
        row_idx = int(src.source_record_id.split("[")[1].rstrip("]"))
        assert item.text == f"Q: Q{row_idx}?\nA: " + (
            f"A{row_idx}" if item.ground_truth_label == "accurate" else f"W{row_idx}"
        )


def test_coherence_labels_come_from_expert_scores_not_loop_index():
    rows = [{
        "text": "doc",
        "machine_summaries": ["s0", "s1", "s2"],
        "coherence": [4.67, 1.33, 3.0],
    }]
    loader = make_loader({("mteb/summeval", None, "test"): FakeSplit(rows)})
    items = load_coherence_items(n_items=3, _loader=loader)

    by_sum = {i.source.source_record_id: i for i in items}
    assert by_sum["test[0].machine_summaries[0]"].ground_truth_label == "5"
    assert by_sum["test[0].machine_summaries[1]"].ground_truth_label == "1"
    assert by_sum["test[0].machine_summaries[2]"].ground_truth_label == "3"
    for i in items:
        assert "coherence_raw" in i.extra


def test_coherence_misaligned_scores_raise():
    rows = [{"text": "doc", "machine_summaries": ["s0", "s1"], "coherence": [4.0]}]
    loader = make_loader({("mteb/summeval", None, "test"): FakeSplit(rows)})
    with pytest.raises(DataSourceSchemaError):
        load_coherence_items(n_items=1, _loader=loader)


def test_relevance_uses_human_judged_positive_and_explicit_negative():
    # TREC-COVID graded qrels: positive = score 2 (fully relevant), negative =
    # score 0 (explicitly non-relevant). Score 1 (partial) is used for neither.
    corpus = FakeSplit([{"_id": f"d{i}", "text": f"passage about topic {i}"} for i in range(20)])
    queries = FakeSplit([{"_id": "q1", "text": "the query about topic"}])
    qrels = FakeSplit([
        {"query-id": "q1", "corpus-id": "d3", "score": 2},   # human: fully relevant
        {"query-id": "q1", "corpus-id": "d7", "score": 0},   # human: non-relevant
        {"query-id": "q1", "corpus-id": "d9", "score": 0},   # human: non-relevant
        {"query-id": "q1", "corpus-id": "d5", "score": 1},   # partial: must not be used
    ])
    loader = make_loader({
        ("BeIR/trec-covid", "corpus", "corpus"): corpus,
        ("BeIR/trec-covid", "queries", "queries"): queries,
        ("BeIR/trec-covid-qrels", None, "test"): qrels,
    })
    items = load_relevance_items(n_items=1, _loader=loader)
    item = items[0]
    sf = item.source.source_fields
    assert sf["relevant_doc_id"] == "d3"
    assert sf["relevant_human_grade"].startswith("2")
    assert sf["nonrelevant_doc_id"] in {"d7", "d9"}          # an explicit negative
    assert sf["nonrelevant_human_grade"].startswith("0")
    assert "d5" not in (sf["relevant_doc_id"], sf["nonrelevant_doc_id"])  # partial excluded
    assert item.extra["candidate_relevant"] == "passage about topic 3"


def test_preference_majority_vote_and_tie_exclusion():
    def row(qid, winner):
        return {
            "question_id": qid, "model_a": "m1", "model_b": "m2",
            "winner": winner, "turn": 1,
            "conversation_a": [
                {"role": "user", "content": f"question {qid}"},
                {"role": "assistant", "content": f"answer-a {qid}"},
            ],
            "conversation_b": [
                {"role": "user", "content": f"question {qid}"},
                {"role": "assistant", "content": f"answer-b {qid}"},
            ],
        }

    rows = [
        row(1, "model_a"), row(1, "model_a"), row(1, "model_b"),  # 3 votes, majority a
        row(2, "model_b"), row(2, "model_b"),                     # 2 votes, majority b
        row(3, "model_a"),                                        # single vote -> excluded (min_votes=2)
        row(4, "model_a"), row(4, "model_b"),                     # 2 votes, tie -> excluded
        row(5, "tie"), row(5, "tie"),                             # tie label -> excluded
    ]
    loader = make_loader({("lmsys/mt_bench_human_judgments", None, "human"): FakeSplit(rows)})
    items = load_preference_items(n_items=2, _loader=loader)

    by_qid = {i.source.source_record_id.split(";")[0]: i for i in items}
    assert set(by_qid) == {"question_id=1", "question_id=2"}   # single-vote q3 excluded by default
    assert by_qid["question_id=1"].ground_truth_label == "candidate_1"
    assert by_qid["question_id=2"].ground_truth_label == "candidate_2"
    for i in items:
        assert int(i.source.source_fields["total_votes"]) >= 2  # no single-annotator labels

    # Only two multi-vote majority items exist; asking for three must fail loud.
    with pytest.raises(DataSourceSchemaError, match="refusing to pad"):
        load_preference_items(n_items=3, _loader=loader)


def test_preference_min_votes_one_readmits_single_annotator_items():
    def row(qid, winner):
        return {
            "question_id": qid, "model_a": "m1", "model_b": "m2",
            "winner": winner, "turn": 1,
            "conversation_a": [{"role": "user", "content": f"q{qid}"},
                               {"role": "assistant", "content": f"a{qid}"}],
            "conversation_b": [{"role": "user", "content": f"q{qid}"},
                               {"role": "assistant", "content": f"b{qid}"}],
        }
    rows = [row(1, "model_a"), row(2, "model_b")]  # two single-vote items
    loader = make_loader({("lmsys/mt_bench_human_judgments", None, "human"): FakeSplit(rows)})
    # default min_votes=2 rejects both single-vote items
    with pytest.raises(DataSourceSchemaError):
        load_preference_items(n_items=1, _loader=loader)
    # min_votes=1 restores the prior single-vote behaviour
    items = load_preference_items(n_items=1, min_votes=1, _loader=loader)
    assert len(items) == 1
