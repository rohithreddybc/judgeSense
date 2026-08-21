"""
Shortcut properties asserted against the SHIPPED files, not the loader.

The loaders are unit-tested on synthetic fixtures, which cannot show what the
real build actually produced. These read data/v2/*.jsonl and fail if a heuristic
that ignores the intended construct scores above chance -- the class of defect
that caused the v1 withdrawal, and which every label-level check passed.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "v2"
PAIRWISE = ("relevance", "preference")


def _rows(task):
    return [json.loads(line) for line in (DATA / f"{task}.jsonl").open(encoding="utf-8")]


def _candidates(prompt):
    """The A and B candidate texts as rendered to the judge."""
    lines = prompt.split("\n")
    starts = {}
    for i, line in enumerate(lines):
        for marker in ("A:", "B:"):
            if line.startswith(marker) and marker not in starts:
                starts[marker] = i
    if "A:" not in starts or "B:" not in starts:
        return None, None
    a = "\n".join(lines[starts["A:"]:starts["B:"]])[2:].strip()
    b = "\n".join(lines[starts["B:"]:])[2:].strip()
    return a, b


@pytest.mark.parametrize("task", PAIRWISE)
def test_position_only_judge_scores_chance(task):
    """The A/B swap must make "always answer A" worth exactly 50%. If this
    drifts, the position-bias analysis measures the imbalance, not the judge."""
    rows = [r for r in _rows(task) if r.get("ground_truth_position") in ("A", "B")]
    a_share = sum(1 for r in rows if r["ground_truth_position"] == "A") / len(rows)
    assert a_share == pytest.approx(0.5, abs=0.01), f"always-A scores {a_share:.3f}"


@pytest.mark.parametrize("task", PAIRWISE)
def test_length_only_judge_scores_chance(task):
    """Two-sided: reliably picking the SHORTER candidate is as cheap a shortcut
    as picking the longer one."""
    rows = [r for r in _rows(task) if r.get("ground_truth_position") in ("A", "B")]
    wins = seen = 0
    for r in rows:
        a, b = _candidates(r["prompt_a"])
        if a is None or len(a) == len(b):
            continue
        seen += 1
        wins += (("A" if len(a) > len(b) else "B") == r["ground_truth_position"])
    assert seen > 0, "no candidate pair could be read; the check must not vacuously pass"
    share = wins / seen
    assert 0.44 <= share <= 0.56, f"length-only judge scores {share:.3f} on {seen} rows"


def test_preference_length_buckets_are_exactly_balanced():
    """The build holds winner-longer at exactly 50%; the smaller bucket caps the
    split size. Drift here means the trim was bypassed."""
    rows = [r for r in _rows("preference") if r.get("ab_order") == "original"]
    longer = sum(1 for r in rows
                 if r["source"]["source_fields"].get("winner_is_longer") == "yes")
    assert longer * 2 == len(rows), f"{longer} winner-longer of {len(rows)} items"


def test_relevance_lexical_overlap_judge_scores_chance():
    """Negatives are BM25-MATCHED to the positive, not maximised: an earlier
    build made distractors keyword-denser than the true positive, so
    "pick the passage with LOWER query overlap" scored 75-92%."""
    import re
    rows = [r for r in _rows("relevance") if r.get("ground_truth_position") in ("A", "B")]
    wins = seen = 0
    for r in rows:
        a, b = _candidates(r["prompt_a"])
        q = re.search(r'query "(.*?)"', r["prompt_a"])
        if a is None or not q:
            continue
        terms = set(re.findall(r"\w+", q.group(1).lower()))
        sa = len(terms & set(re.findall(r"\w+", a.lower())))
        sb = len(terms & set(re.findall(r"\w+", b.lower())))
        if sa == sb:
            continue
        seen += 1
        wins += (("A" if sa > sb else "B") == r["ground_truth_position"])
    assert seen > 0
    share = wins / seen
    assert 0.44 <= share <= 0.56, f"overlap-only judge scores {share:.3f} on {seen} rows"


@pytest.mark.parametrize("task", PAIRWISE)
def test_no_candidate_is_a_placeholder(task):
    """A candidate whose body is missing is answered against by noticing it is
    empty. One shipped in the first v2 build (relevance, candidate "Unknown")."""
    import re
    placeholder = re.compile(r"^(unknown|none|n/?a|null|nan|no abstract)$", re.I)
    bad = []
    for r in _rows(task):
        if r.get("ab_order") != "original":
            continue
        a, b = _candidates(r["prompt_a"])
        for text in (a, b):
            if text is not None and placeholder.match(text.strip()):
                bad.append((r["pair_id"], text[:40]))
    assert not bad, f"placeholder candidates: {bad[:5]}"
