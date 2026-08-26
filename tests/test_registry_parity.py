"""The two judge dicts must agree.

run_v2 selects judges from judge_registry.JUDGES but resolves provider clients
through models.SUPPORTED_MODELS. A judge present in one and missing from the
other builds fine, passes selection, and then dies at client-build time --
partway through a paid sweep, after money has been spent on earlier cells.

The failure is cheap to prevent and expensive to discover, so it is a test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_registry import JUDGES  # noqa: E402
from models import SUPPORTED_MODELS  # noqa: E402


def _selectable() -> list[str]:
    """Judges the runner is allowed to pick without --allow-unverified."""
    return [name for name, spec in JUDGES.items() if spec.get("verified")]


@pytest.mark.parametrize("judge", _selectable())
def test_every_verified_judge_can_resolve_a_client(judge: str) -> None:
    assert judge in SUPPORTED_MODELS, (
        f"{judge!r} is verified in judge_registry.JUDGES but absent from "
        f"models.SUPPORTED_MODELS, so run_v2 would fail at client build. "
        f"Add it to both dicts."
    )


@pytest.mark.parametrize("judge", _selectable())
def test_provider_and_model_id_agree(judge: str) -> None:
    reg, sup = JUDGES[judge], SUPPORTED_MODELS[judge]
    assert reg["provider"] == sup["provider"], (
        f"{judge!r}: registry says provider={reg['provider']!r}, "
        f"SUPPORTED_MODELS says {sup['provider']!r}"
    )
    assert reg["model_id"] == sup["model_id"], (
        f"{judge!r}: registry says model_id={reg['model_id']!r}, "
        f"SUPPORTED_MODELS says {sup['model_id']!r} -- the run would evaluate a "
        f"different checkpoint than the one the paper reports."
    )
    assert reg["key"] == sup["key"], (
        f"{judge!r}: registry reads {reg['key']!r}, SUPPORTED_MODELS reads "
        f"{sup['key']!r}"
    )


def test_no_verified_judge_lacks_a_family() -> None:
    """`family` groups the scale ladders; an unset one silently drops a judge
    out of every within-family comparison."""
    missing = [n for n in _selectable() if not JUDGES[n].get("family")]
    assert not missing, f"verified judges with no family: {missing}"


def test_registry_names_are_unique_case_insensitively() -> None:
    """Two judges differing only in case would collide in the raw filename
    (`{judge}_{task}.jsonl`) and silently overwrite each other's results."""
    lowered = [n.lower() for n in JUDGES]
    dupes = {n for n in lowered if lowered.count(n) > 1}
    assert not dupes, f"case-colliding judge names would share a raw file: {dupes}"


def test_raw_filenames_do_not_collide_on_the_underscore_split() -> None:
    """metrics/regeneration recover the judge from `name.split('_')[0]`, so a
    judge whose name contains '_' would be attributed to the wrong judge."""
    bad = [n for n in JUDGES if "_" in n]
    assert not bad, (
        f"judge names must not contain '_': {bad} -- the results loader splits "
        f"raw filenames on '_' to recover the judge name."
    )
