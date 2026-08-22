"""
The printed call budget must match the dataset actually on disk.

`src/judge_registry.py` hardcodes the row counts that `main_axis_run_plan` turns
into "Planned calls: N" — the number the operator confirms before spending. If a
rebuild changes the dataset and the constants are not updated, that number is
wrong in whichever direction nobody notices, and the run either overspends or
silently under-covers. These tests bind the constants to the files.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge_registry import (  # noqa: E402
    MAIN_AXIS_PAIRWISE_TASKS,
    MAIN_AXIS_REPEAT_ARMS_PER_ITEM,
    MAIN_AXIS_ROWS_PER_TASK,
    MAIN_AXIS_TOTAL_ITEMS,
    MAIN_AXIS_TOTAL_ROWS,
    main_axis_run_plan,
)

DATA = ROOT / "data" / "v2"


def _rows(task):
    return [json.loads(line) for line in (DATA / f"{task}.jsonl").open(encoding="utf-8")]


@pytest.mark.parametrize("task", sorted(MAIN_AXIS_ROWS_PER_TASK))
def test_declared_row_count_matches_the_file(task):
    assert len(_rows(task)) == MAIN_AXIS_ROWS_PER_TASK[task]


@pytest.mark.parametrize("task", sorted(MAIN_AXIS_ROWS_PER_TASK))
def test_pairwise_tasks_carry_exactly_two_orderings_per_item(task):
    rows = _rows(task)
    items = {r["item_id"] for r in rows}
    expected = 2 if task in MAIN_AXIS_PAIRWISE_TASKS else 1
    assert len(rows) == expected * len(items)


def test_totals_match_the_dataset():
    total_rows = sum(len(_rows(t)) for t in MAIN_AXIS_ROWS_PER_TASK)
    total_items = sum(len({r["item_id"] for r in _rows(t)}) for t in MAIN_AXIS_ROWS_PER_TASK)
    assert MAIN_AXIS_TOTAL_ROWS == total_rows
    assert MAIN_AXIS_TOTAL_ITEMS == total_items


def test_planned_calls_equal_what_the_runner_will_actually_issue():
    """Budget arithmetic against the files: 2 arms per row, plus one repeat per
    item (the repeat fires only on the canonical ordering, never per row)."""
    plan = main_axis_run_plan(judges=["gpt-4o"], include_repeat_baseline=True)
    arm_calls = sum(len(_rows(t)) for t in MAIN_AXIS_ROWS_PER_TASK) * 2
    # TWO repeat calls per item: the ceiling is measured under both templates,
    # because one measured under a single template cannot absorb noise the other
    # generates. A stale factor here understates the printed budget, which is
    # the number the operator approves before spending.
    repeat_calls = MAIN_AXIS_REPEAT_ARMS_PER_ITEM * sum(
        len({r["item_id"] for r in _rows(t)}) for t in MAIN_AXIS_ROWS_PER_TASK
    )
    assert plan["calls_per_judge"] == arm_calls
    assert plan["repeat_calls_per_judge"] == repeat_calls
    assert plan["calls_per_judge_with_repeat"] == arm_calls + repeat_calls
