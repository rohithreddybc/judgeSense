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


# ── Preference: the length buckets must be matched on ANNOTATOR AGREEMENT ────
#
# Holding winner-longer at exactly 50% removes length as a label signal, but the
# trim that achieves it selects only from the larger bucket: MT-Bench supplies
# 272 winner-longer comparisons against 113 winner-shorter ones, so the shorter
# bucket ships whole while the longer bucket is cut to 113. An earlier build cut
# it by taking the MOST CONTESTED 113, which left the two buckets matched on
# length but far apart on agreement (mean vote-margin ratio 0.567 vs 0.745,
# 16% vs 54% unanimous) -- one confound traded for another, and a judge that is
# good on close calls reads as length-biased. These assert against the SHIPPED
# file that the buckets are matched on margin too.

_MARGIN_TOL = 0.05          # absolute, on the mean margin ratio
_SHARE_TOL = 0.05           # absolute, on the unanimous share
_CDF_TOL = 0.05             # max gap between the two buckets' margin CDFs


def _preference_margin_buckets():
    """(winner-longer ratios, winner-shorter ratios) over shipped items."""
    seen, longer, shorter = set(), [], []
    for r in _rows("preference"):
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        sf = r["source"]["source_fields"]
        ratio = float(sf["vote_margin_ratio"])
        (longer if sf["winner_is_longer"] == "yes" else shorter).append(ratio)
    assert longer and shorter, "preference split has an empty length bucket"
    return longer, shorter


def test_preference_margin_mean_matches_across_length_buckets():
    """Mean vote-margin ratio must not differ by more than 0.05 between the
    winner-longer and winner-shorter buckets."""
    longer, shorter = _preference_margin_buckets()
    ml = sum(longer) / len(longer)
    ms = sum(shorter) / len(shorter)
    assert abs(ml - ms) <= _MARGIN_TOL, (
        f"mean vote_margin_ratio {ml:.4f} (winner-longer, n={len(longer)}) vs "
        f"{ms:.4f} (winner-shorter, n={len(shorter)}); delta {abs(ml - ms):.4f} "
        f"exceeds {_MARGIN_TOL}"
    )


def test_preference_margin_median_matches_across_length_buckets():
    """Same for the median, which the mean can hide: the defective build had
    medians 0.500 and 1.000 -- the buckets did not even overlap at the centre."""
    import statistics
    longer, shorter = _preference_margin_buckets()
    dl, ds_ = statistics.median(longer), statistics.median(shorter)
    assert abs(dl - ds_) <= _MARGIN_TOL, (
        f"median vote_margin_ratio {dl:.4f} (winner-longer) vs {ds_:.4f} "
        f"(winner-shorter); delta {abs(dl - ds_):.4f} exceeds {_MARGIN_TOL}"
    )


def test_preference_unanimous_share_matches_across_length_buckets():
    """The share of comparisons the annotators agreed on unanimously must be the
    same in both buckets. This is the coarse, readable form of the same
    property: 18/113 vs 61/113 was the defect."""
    longer, shorter = _preference_margin_buckets()
    ul = sum(1 for x in longer if x >= 1.0) / len(longer)
    us = sum(1 for x in shorter if x >= 1.0) / len(shorter)
    assert abs(ul - us) <= _SHARE_TOL, (
        f"unanimous share {ul:.4f} (winner-longer) vs {us:.4f} (winner-shorter); "
        f"delta {abs(ul - us):.4f} exceeds {_SHARE_TOL}"
    )


def test_preference_margin_distributions_match_across_length_buckets():
    """Whole-distribution check, not just two summary statistics: the largest
    gap between the buckets' empirical CDFs of vote_margin_ratio."""
    longer, shorter = _preference_margin_buckets()
    grid = sorted(set(longer) | set(shorter))

    def cdf(sample, x):
        return sum(1 for v in sample if v <= x) / len(sample)

    gap = max(abs(cdf(longer, x) - cdf(shorter, x)) for x in grid)
    assert gap <= _CDF_TOL, (
        f"max CDF gap between the length buckets' vote_margin_ratio "
        f"distributions is {gap:.4f}, above {_CDF_TOL}"
    )


def test_preference_is_not_all_coin_flips():
    """Selecting the most-contested comparisons minimises the mutual information
    between EVERY feature and the label -- length and lexical overlap, but also
    genuine answer quality. A split built that way passes the shortcut controls
    by construction while losing the ability to separate a competent judge from
    an incompetent one, which is the non-discriminating-task defect that
    withdrew v1. Require that a substantial share of shipped comparisons carry a
    decisive human majority."""
    longer, shorter = _preference_margin_buckets()
    ratios = longer + shorter
    unanimous = sum(1 for x in ratios if x >= 1.0) / len(ratios)
    assert unanimous >= 0.30, (
        f"only {unanimous:.3f} of shipped preference items have a unanimous "
        f"human majority; the split may be too contested to discriminate"
    )
