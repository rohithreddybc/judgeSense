"""Retirement is separate from verification, and both gates must hold.

`verified` says the checkpoint was confirmed against the provider. `retired`
says the judge will not finish inside its provider's free-tier daily cap. The
first was briefly used to express the second, which broke the daily quota job:
`select_judges` refuses an unverified checkpoint, and rightly so.

These pin the distinction so it does not get collapsed again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_registry import (  # noqa: E402
    JUDGES, RegistryError, family_ladders, is_retired, select_judges,
)

RETIRED = ("gpt-oss-20b", "gpt-oss-120b", "qwen3.8-27b", "gemini-3.1-pro")


@pytest.mark.parametrize("name", RETIRED)
def test_retired_judges_stay_verified(name: str) -> None:
    """Their checkpoints are confirmed and their answers parse.

    Nothing about these four failed a quality check, so flipping `verified`
    would assert something untrue about the model id -- and would make
    select_judges refuse them even when a caller names them deliberately.
    """
    assert JUDGES[name]["verified"] is True
    assert is_retired(name)


@pytest.mark.parametrize("name", RETIRED)
def test_retired_judges_carry_their_reason(name: str) -> None:
    """The value is the measurement, not a bare flag, so the reason travels
    with the entry rather than living only in a commit message."""
    reason = JUDGES[name]["retired"]
    assert isinstance(reason, str) and reason.strip()
    assert "day" in reason.lower()


def test_retired_judges_are_absent_from_the_default_roster() -> None:
    """A default sweep must not spend hours discovering a daily cap."""
    default = select_judges()
    assert not [n for n in default if is_retired(n)]


@pytest.mark.parametrize("name", RETIRED)
def test_retired_judges_run_when_named_explicitly(name: str) -> None:
    """The daily quota job depends on this: it names the judges it wants and
    must not be refused for naming retired ones."""
    assert select_judges([name]) == [name]


def test_unverified_is_still_refused() -> None:
    """The verification gate must survive the addition of the retirement gate."""
    unverified = [n for n, s in JUDGES.items() if not s["verified"]]
    if not unverified:
        pytest.skip("no unverified judges in the registry")
    with pytest.raises(RegistryError):
        select_judges([unverified[0]])


def test_no_ladder_rests_on_a_retired_judge() -> None:
    """A size ladder whose rungs produce no rows is a scale claim with nothing
    behind it. gpt-oss 20B/120B is exactly that case."""
    for family, members in family_ladders().items():
        retired = [m for m in members if is_retired(m)]
        assert not retired, f"ladder {family} rests on retired judges: {retired}"


def test_every_judge_answers_the_retired_question() -> None:
    """`is_retired` must not raise on an entry that predates the field."""
    for name in JUDGES:
        assert is_retired(name) in (True, False)
