"""
Tests for src/dataset_builder_v2.py.

SourceItem inputs here are TEST FIXTURES exercising builder structure only;
they are never written under data/.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_sources import SourceItem, SourceRecord  # noqa: E402
from src.dataset_builder_v2 import (  # noqa: E402
    PAIRWISE_TASKS,
    TEMPLATES,
    build_task_records,
)


def make_item(task, idx, **extra_kwargs):
    extras = {
        "factuality": {},
        "coherence": {"coherence_raw": 3.4},
        "relevance": {
            "candidate_relevant": f"relevant passage {idx}",
            "candidate_nonrelevant": f"offtopic passage {idx}",
        },
        "preference": {
            "candidate_1": f"good answer {idx}",
            "candidate_2": f"weak answer {idx}",
        },
    }[task]
    labels = {
        "factuality": "accurate",
        "coherence": "3",
        "relevance": "relevant_candidate",
        "preference": "candidate_1",
    }
    return SourceItem(
        item_id=f"{task}_item_{idx}",
        task_type=task,
        text=f"fixture text {idx}",
        ground_truth_label=labels[task],
        source=SourceRecord(
            source_dataset="fixture_dataset",
            source_config=None,
            source_split="test",
            source_record_id=f"test[{idx}]",
            source_fields={},
            retrieved_at="2026-01-01T00:00:00+00:00",
        ),
        extra={**extras, **extra_kwargs},
    )


def test_templates_do_not_invert_polarity():
    # The v1 factuality T4 artifact: a template asking the OPPOSITE question
    # ("does this contain errors? YES = bad") while sharing a label space.
    for template in TEMPLATES["factuality"]:
        lowered = template.lower()
        assert "error" not in lowered and "incorrect?" not in lowered
        assert "yes" in lowered and "no" in lowered
    for template in TEMPLATES["coherence"]:
        assert "1" in template and "5" in template
        assert "incoherent" in template.lower()  # same anchor direction in all


def test_pointwise_records_one_row_per_item_with_provenance():
    items = [make_item("factuality", i) for i in range(30)]
    records = build_task_records("factuality", items)

    assert len(records) == 30
    assert len({r["item_id"] for r in records}) == 30          # no duplication
    assert len({r["prompt_pair_id"] for r in records}) == 30
    for rec in records:
        assert rec["source"]["source_record_id"].startswith("test[")
        assert rec["source_benchmark"] == "fixture_dataset"
        assert rec["prompt_a"] != rec["prompt_b"]
        assert "fixture text" in rec["prompt_a"]
        assert "semantic_equivalence_score" not in rec  # never hardcoded


def test_template_combinations_rotate_across_items():
    items = [make_item("coherence", i) for i in range(20)]
    records = build_task_records("coherence", items)
    combos = {(r["template_a"], r["template_b"]) for r in records}
    assert len(combos) == 10  # all C(5,2) combinations used across 20 items


def test_pairwise_emits_both_orderings_with_consistent_maps():
    items = [make_item("preference", i) for i in range(5)]
    records = build_task_records("preference", items)

    assert len(records) == 10  # 2 orderings per item
    by_pair = {}
    for rec in records:
        by_pair.setdefault(rec["prompt_pair_id"], {})[rec["ab_order"]] = rec
    for pair_id, orders in by_pair.items():
        assert set(orders) == {"original", "swapped"}
        orig, swap = orders["original"], orders["swapped"]
        assert orig["candidate_map"] == {"A": "candidate_1", "B": "candidate_2"}
        assert swap["candidate_map"] == {"A": "candidate_2", "B": "candidate_1"}
        # ground truth stays content-level; positional label flips with order
        assert orig["ground_truth_label"] == swap["ground_truth_label"] == "candidate_1"
        assert orig["ground_truth_position"] == "A"
        assert swap["ground_truth_position"] == "B"
        # the actual candidate text swaps position in the rendered prompt
        assert orig["prompt_a"].index("good answer") < orig["prompt_a"].index("weak answer")
        assert swap["prompt_a"].index("weak answer") < swap["prompt_a"].index("good answer")


def test_pairwise_tasks_constant():
    assert PAIRWISE_TASKS == {"relevance", "preference"}


def test_coherence_carries_raw_expert_score():
    records = build_task_records("coherence", [make_item("coherence", 0)])
    assert records[0]["ground_truth_raw"] == pytest.approx(3.4)
    assert records[0]["ground_truth_label"] == "3"
