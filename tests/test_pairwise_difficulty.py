"""
Tests for the difficulty-aware pairwise selection in src/data_sources.py.

Context: on the previous run, 11/13 judges answered the relevance and
preference tasks at accuracy exactly 1.000, because the relevance negative
was a random (topically unrelated) corpus doc and the preference selection
was dominated by unanimously-decided comparisons. The loaders now select
HARD pairings by default — top-BM25 non-qrels negatives, most-contested
human votes — and record per-item difficulty evidence in `source_fields`.

All dataset payloads here are TEST FIXTURES exercising selection logic only;
they are never written under data/ and must never be.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_sources import (  # noqa: E402
    _BM25Index,
    load_preference_items,
    load_relevance_items,
)
from test_data_sources_v2 import FakeSplit, make_loader  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

def relevance_splits():
    """A corpus with one on-topic distractor and many off-topic fillers."""
    docs = [
        # qrels-relevant doc for q1
        {"_id": "d_pos", "text": "machine learning algorithms improve prediction accuracy"},
        # on-topic distractor: shares most query terms, NOT in qrels
        {"_id": "d_hard", "text": "prediction accuracy of machine learning classifiers"},
        # second qrels-relevant doc: lexically close but must never be the negative
        {"_id": "d_pos2", "text": "machine learning improves prediction in clinical models"},
    ] + [
        {"_id": f"d_fill{i:02d}", "text": f"the capital of country {i} is city {i}"}
        for i in range(30)
    ]
    corpus = FakeSplit(docs)
    queries = FakeSplit(
        [{"_id": "q1", "text": "machine learning prediction accuracy"}]
    )
    qrels = FakeSplit([
        {"query-id": "q1", "corpus-id": "d_pos", "score": 1},
        {"query-id": "q1", "corpus-id": "d_pos2", "score": 1},
    ])
    return {
        ("BeIR/scifact", "corpus", "corpus"): corpus,
        ("BeIR/scifact", "queries", "queries"): queries,
        ("BeIR/scifact-qrels", None, "train"): qrels,
    }


def preference_rows():
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

    return [
        # qid 1: unanimous 3-0 for model_a          -> margin 3, ratio 1.0
        row(1, "model_a"), row(1, "model_a"), row(1, "model_a"),
        # qid 2: contested 2-1 for model_b          -> margin 1, ratio 0.3333
        row(2, "model_b"), row(2, "model_b"), row(2, "model_a"),
        # qid 3: 2-0 for model_a plus one tie vote  -> margin 2, ratio 0.6667
        row(3, "model_a"), row(3, "model_a"), row(3, "tie"),
    ]


def preference_loader():
    return make_loader(
        {("lmsys/mt_bench_human_judgments", None, "human"): FakeSplit(preference_rows())}
    )


# ── Relevance: hard negatives ────────────────────────────────────────────────

def test_hard_negative_is_top_bm25_doc_outside_qrels():
    items = load_relevance_items(
        n_items=1, difficulty="hard", _loader=make_loader(relevance_splits())
    )
    fields = items[0].source.source_fields
    # The on-topic distractor wins; both qrels docs are excluded even though
    # they outscore it lexically.
    assert fields["nonrelevant_doc_id"] == "d_hard"
    assert fields["nonrelevant_doc_id"] not in {"d_pos", "d_pos2"}
    assert items[0].extra["candidate_nonrelevant"] == (
        "prediction accuracy of machine learning classifiers"
    )
    # Label authority unchanged: still qrels, still the same candidate key.
    assert items[0].ground_truth_label == "candidate_relevant"


def test_relevance_difficulty_fields_are_populated_and_consistent():
    for difficulty in ("hard", "easy"):
        items = load_relevance_items(
            n_items=1, difficulty=difficulty, _loader=make_loader(relevance_splits())
        )
        fields = items[0].source.source_fields
        assert fields["difficulty"] == difficulty
        neg_score = float(fields["neg_bm25_score"])
        pos_score = float(fields["pos_bm25_score"])
        gap = float(fields["bm25_score_gap"])
        assert gap == pytest.approx(pos_score - neg_score, abs=1e-3)
        assert int(fields["neg_bm25_rank"]) >= 1
        # existing provenance keys survive
        for key in ("relevant_doc_id", "nonrelevant_doc_id", "qrels_dataset", "pairing"):
            assert key in fields


def test_hard_negative_scores_at_least_as_high_as_easy_negative():
    """Hard is measurably harder: its distractor's lexical proximity to the
    query is maximal over the eligible pool, so it can never score below
    whatever the easy (random) pairing drew for the same query."""
    hard = load_relevance_items(
        n_items=1, difficulty="hard", _loader=make_loader(relevance_splits())
    )[0]
    easy = load_relevance_items(
        n_items=1, difficulty="easy", _loader=make_loader(relevance_splits())
    )[0]
    hard_neg = float(hard.source.source_fields["neg_bm25_score"])
    easy_neg = float(easy.source.source_fields["neg_bm25_score"])
    assert hard_neg >= easy_neg
    # In this fixture the fillers share no query term at all, so strictly harder.
    assert hard_neg > easy_neg
    assert easy.source.source_fields["nonrelevant_doc_id"].startswith("d_fill")


def test_easy_negative_still_excluded_from_qrels():
    items = load_relevance_items(
        n_items=1, difficulty="easy", _loader=make_loader(relevance_splits())
    )
    assert items[0].source.source_fields["nonrelevant_doc_id"] not in {"d_pos", "d_pos2"}


def test_bm25_ranking_is_deterministic_with_zero_overlap():
    index = _BM25Index({"b": "alpha beta", "a": "alpha beta"})
    ranked = index.ranked("nothing shared")
    assert ranked == [("a", 0.0), ("b", 0.0)]  # doc-id tiebreak, all zero


# ── Preference: contested comparisons ────────────────────────────────────────

def test_hard_preference_prefers_narrow_margins():
    items = load_preference_items(
        n_items=2, difficulty="hard", _loader=preference_loader()
    )
    qids = {i.source.source_record_id.split(";")[0] for i in items}
    # The two most contested comparisons: 2-1 (qid 2) and 2-0-1tie (qid 3).
    assert qids == {"question_id=2", "question_id=3"}
    # Labels remain the recorded human majority.
    by_qid = {i.source.source_record_id.split(";")[0]: i for i in items}
    assert by_qid["question_id=2"].ground_truth_label == "candidate_2"
    assert by_qid["question_id=3"].ground_truth_label == "candidate_1"


def test_easy_preference_prefers_unanimous():
    items = load_preference_items(
        n_items=2, difficulty="easy", _loader=preference_loader()
    )
    qids = {i.source.source_record_id.split(";")[0] for i in items}
    assert qids == {"question_id=1", "question_id=3"}


def test_preference_margin_fields_are_populated():
    items = load_preference_items(
        n_items=3, difficulty="hard", _loader=preference_loader()
    )
    by_qid = {i.source.source_record_id.split(";")[0]: i.source.source_fields for i in items}

    f2 = by_qid["question_id=2"]
    assert f2["difficulty"] == "hard"
    assert f2["vote_margin"] == "1"
    assert f2["total_votes"] == "3"
    assert float(f2["vote_margin_ratio"]) == pytest.approx(1 / 3, abs=1e-3)

    f3 = by_qid["question_id=3"]
    assert f3["vote_margin"] == "2"
    assert f3["total_votes"] == "3"
    assert float(f3["vote_margin_ratio"]) == pytest.approx(2 / 3, abs=1e-3)

    f1 = by_qid["question_id=1"]
    assert f1["vote_margin"] == "3"
    assert float(f1["vote_margin_ratio"]) == pytest.approx(1.0)

    for fields in by_qid.values():
        for key in ("first_seen_row", "vote_tally", "label_rule"):
            assert key in fields  # existing provenance keys survive


def test_hard_split_mean_margin_ratio_below_easy():
    """Difficulty metric check at split level: the hard selection's mean
    vote-margin ratio must be strictly below the easy selection's."""

    def mean_ratio(difficulty):
        items = load_preference_items(
            n_items=2, difficulty=difficulty, _loader=preference_loader()
        )
        return sum(
            float(i.source.source_fields["vote_margin_ratio"]) for i in items
        ) / len(items)

    assert mean_ratio("hard") < mean_ratio("easy")


# ── Duplicate suppression (audit-gate parity) ───────────────────────────────
#
# The audit gate keys judged content on the DISPLAYED candidates only — the
# query/question is not part of `response_being_judged`, and the A/B swap
# design displays both orders of every pair. Hard selection concentrates
# lexically-similar queries on the same documents, so the loaders must ship
# each unordered content pair at most once.

def test_relevance_same_doc_pair_ships_once():
    """Two claims about the same qrels doc that mine the same distractor must
    not produce two items with identical displayed candidates."""
    splits = relevance_splits()
    splits[("BeIR/scifact", "queries", "queries")] = FakeSplit([
        {"_id": "q1", "text": "machine learning prediction accuracy"},
        {"_id": "q2", "text": "prediction accuracy machine learning"},
    ])
    splits[("BeIR/scifact-qrels", None, "train")] = FakeSplit([
        {"query-id": "q1", "corpus-id": "d_pos", "score": 1},
        {"query-id": "q2", "corpus-id": "d_pos", "score": 1},
    ])
    loader = make_loader(splits)

    items = load_relevance_items(n_items=1, difficulty="hard", _loader=loader)
    assert len(items) == 1

    from src.data_sources import DataSourceSchemaError
    with pytest.raises(DataSourceSchemaError, match="duplicate document pairs"):
        load_relevance_items(n_items=2, difficulty="hard", _loader=loader)


def test_preference_role_reversed_pair_ships_once():
    """The same two responses appearing in reversed roles (even on another
    question) would contradict themselves under the swap design; only the
    first may ship."""

    def row(qid, ra, rb, winner):
        return {
            "question_id": qid, "model_a": "m1", "model_b": "m2",
            "winner": winner, "turn": 1,
            "conversation_a": [{"role": "user", "content": f"question {qid}"},
                               {"role": "assistant", "content": ra}],
            "conversation_b": [{"role": "user", "content": f"question {qid}"},
                               {"role": "assistant", "content": rb}],
        }

    rows = [
        row(1, "resp X", "resp Y", "model_a"), row(1, "resp X", "resp Y", "model_a"),
        # same responses, roles reversed, different question -> duplicate
        row(2, "resp Y", "resp X", "model_a"),
        # degenerate: both candidates identical -> never shippable
        row(3, "same text", "same text", "model_a"),
    ]
    loader = make_loader(
        {("lmsys/mt_bench_human_judgments", None, "human"): FakeSplit(rows)}
    )
    items = load_preference_items(n_items=1, difficulty="hard", _loader=loader)
    assert len(items) == 1

    from src.data_sources import DataSourceSchemaError
    with pytest.raises(DataSourceSchemaError, match="refusing to pad"):
        load_preference_items(n_items=2, difficulty="hard", _loader=loader)


# ── Guard rails ──────────────────────────────────────────────────────────────

def test_invalid_difficulty_raises():
    with pytest.raises(ValueError, match="difficulty"):
        load_relevance_items(n_items=1, difficulty="medium",
                             _loader=make_loader(relevance_splits()))
    with pytest.raises(ValueError, match="difficulty"):
        load_preference_items(n_items=1, difficulty="brutal",
                              _loader=preference_loader())


def test_ties_still_excluded_in_both_difficulties():
    rows = preference_rows() + [
        {"question_id": 4, "model_a": "m1", "model_b": "m2", "winner": "model_a",
         "turn": 1,
         "conversation_a": [{"role": "user", "content": "question 4"},
                            {"role": "assistant", "content": "answer-a 4"}],
         "conversation_b": [{"role": "user", "content": "question 4"},
                            {"role": "assistant", "content": "answer-b 4"}]},
        {"question_id": 4, "model_a": "m1", "model_b": "m2", "winner": "model_b",
         "turn": 1,
         "conversation_a": [{"role": "user", "content": "question 4"},
                            {"role": "assistant", "content": "answer-a 4"}],
         "conversation_b": [{"role": "user", "content": "question 4"},
                            {"role": "assistant", "content": "answer-b 4"}]},
    ]
    loader = make_loader(
        {("lmsys/mt_bench_human_judgments", None, "human"): FakeSplit(rows)}
    )
    for difficulty in ("hard", "easy"):
        items = load_preference_items(n_items=3, difficulty=difficulty, _loader=loader)
        qids = {i.source.source_record_id.split(";")[0] for i in items}
        assert "question_id=4" not in qids  # 1-1 tie can never ship
