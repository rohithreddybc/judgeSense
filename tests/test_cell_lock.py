"""Two processes must not write the same (judge, task).

Raw output is append-only with no lock, and the completed set is read ONCE at
cell start. Two processes over one cell therefore each work the whole remaining
backlog and both append. Restarting a provider's process while its predecessor
was still draining duplicated 77 rows across four cells; metrics resample at the
item, so a duplicated item is weighted twice.

The runner printed a warning about this. A warning is not an enforcement.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_v2  # noqa: E402


@pytest.fixture
def cell(tmp_path, monkeypatch):
    monkeypatch.setattr(run_v2, "_OUT_DIR", tmp_path)
    judge, task = "somejudge", "factuality"
    yield judge, task
    run_v2._lock_path(judge, task).unlink(missing_ok=True)


def test_second_writer_is_refused(cell):
    judge, task = cell
    run_v2._acquire_cell(judge, task)
    with pytest.raises(run_v2.CellBusy):
        run_v2._acquire_cell(judge, task)


def test_refusal_names_the_cell_and_says_what_to_do(cell):
    judge, task = cell
    run_v2._acquire_cell(judge, task)
    with pytest.raises(run_v2.CellBusy) as exc:
        run_v2._acquire_cell(judge, task)
    msg = str(exc.value)
    assert judge in msg and task in msg
    assert "--judges" in msg, "the message must say how to split the work"


def test_a_lock_left_by_a_killed_process_goes_stale(cell):
    """A crashed holder must not block its cell forever, or a killed sweep can
    never be resumed."""
    judge, task = cell
    path = run_v2._acquire_cell(judge, task)
    old = time.time() - run_v2._LOCK_STALE_SECONDS - 10
    os.utime(path, (old, old))
    run_v2._acquire_cell(judge, task)  # reclaimed, no raise


def test_lock_is_released_even_when_the_cell_raises(cell, monkeypatch):
    """A failing cell must not poison every later resume of that cell."""
    judge, task = cell

    def boom(*a, **k):
        raise RuntimeError("cell blew up")

    monkeypatch.setattr(run_v2, "_run_cell_locked", boom)
    with pytest.raises(RuntimeError, match="cell blew up"):
        run_v2.run_cell(judge, task, "matched", True, None)
    assert not run_v2._lock_path(judge, task).exists(), "lock outlived the failure"
