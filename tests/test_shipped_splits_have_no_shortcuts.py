"""
Shortcut properties asserted against the SHIPPED files, not the loader.

The loaders are unit-tested on synthetic fixtures, which cannot show what the
real build actually produced. These read data/v2/*.jsonl and fail if a heuristic
that ignores the intended construct scores above chance -- the class of defect
that caused the v1 withdrawal, and which every label-level check passed.
"""

import json
import sys
from collections import Counter
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


# -- Factuality: nothing about WHERE an item sits may predict its label -------
#
# The v2 build assigned template pairs with `combos[idx % 10]` while the loader
# emitted best_answer/incorrect_answers[0] in a strict 2-cycle. The two cycles
# ran in lockstep, so every one of the 10 template pairs mapped to exactly one
# label (250/250), the pooled per-template balance was T2 75/25 and T4 25/75,
# and file line parity was an exact oracle as well. A constant-YES judge scored
# 0.75 on T2 and 0.25 on T4 from item assignment alone -- and T4 carries the
# polarity narrative, so every per-template claim was confounded.
#
# These read the SHIPPED file. The loader now shuffles under the seed and the
# builder assigns template pairs stratified on the label.

_CHANCE_TOL = 0.10          # absolute, around 0.5


def _factuality_rows():
    rows = _rows("factuality")
    assert rows, "factuality split is empty; the check must not vacuously pass"
    return rows


def test_template_pair_does_not_predict_the_factuality_label():
    """Accuracy of the BEST possible template-pair -> label rule. Assigning
    each template pair its own majority label is the strongest such rule that
    exists, so if that is near chance no weaker one can do better."""
    rows = _factuality_rows()
    by_combo = {}
    for r in rows:
        combo = (r["template_a"], r["template_b"])
        by_combo.setdefault(combo, []).append(r["ground_truth_label"])
    best = sum(max(Counter(labels).values()) for labels in by_combo.values())
    share = best / len(rows)
    majority = max(Counter(r["ground_truth_label"] for r in rows).values()) / len(rows)
    assert share <= majority + _CHANCE_TOL, (
        f"the best template-pair -> label rule scores {share:.4f} over "
        f"{len(by_combo)} template pairs on {len(rows)} items; the "
        f"label-only baseline is {majority:.4f}. Template assignment is "
        f"leaking the label."
    )


def test_each_template_sees_both_factuality_labels_in_balance():
    """Pooled over both arms of every pair it appears in, each template must see
    accurate and inaccurate about equally. This is the form the defect actually
    took in the per-template results table."""
    rows = _factuality_rows()
    per_template = {}
    for r in rows:
        for template in (r["template_a"], r["template_b"]):
            per_template.setdefault(template, Counter())[r["ground_truth_label"]] += 1
    assert len(per_template) == 5, f"expected T1-T5, found {sorted(per_template)}"
    offenders = {}
    for template, counts in per_template.items():
        n = sum(counts.values())
        share = counts["accurate"] / n
        if not 0.45 <= share <= 0.55:
            offenders[template] = f"{counts['accurate']}/{n} = {share:.4f}"
    assert not offenders, (
        f"per-template pooled 'accurate' share outside [0.45, 0.55]: {offenders}"
    )


def test_file_line_parity_does_not_predict_the_factuality_label():
    """Row order must carry no label information either: the alternating
    emission made "line index even -> accurate" exact at 250/250."""
    rows = _factuality_rows()
    even_accurate = sum(1 for i, r in enumerate(rows)
                        if (i % 2 == 0) == (r["ground_truth_label"] == "accurate"))
    share = max(even_accurate, len(rows) - even_accurate) / len(rows)
    assert share <= 0.5 + _CHANCE_TOL, (
        f"line parity predicts the label on {share:.4f} of {len(rows)} rows"
    )


def test_item_id_order_does_not_predict_the_factuality_label():
    """Same property on the identifier a consumer sorts on. item_id was assigned
    in emission order, so its parity was a perfect oracle independently of how
    the file happened to be ordered."""
    rows = sorted(_factuality_rows(), key=lambda r: r["item_id"])
    even_accurate = sum(1 for i, r in enumerate(rows)
                        if (i % 2 == 0) == (r["ground_truth_label"] == "accurate"))
    share = max(even_accurate, len(rows) - even_accurate) / len(rows)
    assert share <= 0.5 + _CHANCE_TOL, (
        f"item_id parity predicts the label on {share:.4f} of {len(rows)} items"
    )


# -- Preference: the shipped label rule must hold on the shipped items --------


def test_no_preference_item_violates_its_own_label_rule():
    """Every preference row carries a `label_rule` string. It was enforced
    against the TOTAL vote count, which includes tie votes, so 73/226 shipped
    items (32.3%) had a winner resting on fewer than two decisive votes or on a
    decisive count no greater than the tie count -- items where the modal human
    judgement is "neither is better", and a judge reproducing the modal human is
    scored wrong. Re-derive both counts from the raw tally and check the rule."""
    import ast
    import re

    seen, violations = set(), []
    for r in _rows("preference"):
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        sf = r["source"]["source_fields"]
        rule = sf["label_rule"]
        floor = int(re.search(r">=\s*(\d+)", rule).group(1))
        tally = ast.literal_eval(sf["vote_tally"])
        decisive = {k: v for k, v in tally.items() if k in ("model_a", "model_b")}
        ties = sum(v for k, v in tally.items() if k not in ("model_a", "model_b"))
        winner_votes = max(decisive.values()) if decisive else 0
        runner_up = sum(decisive.values()) - winner_votes
        if winner_votes < floor or winner_votes <= ties or winner_votes <= runner_up:
            violations.append((r["item_id"], tally))
    assert seen, "preference split is empty; the check must not vacuously pass"
    assert not violations, (
        f"{len(violations)}/{len(seen)} preference items violate the label rule "
        f"shipped inside them, e.g. {violations[:3]}"
    )


def test_shipped_label_rule_text_names_decisive_votes():
    """The rule string must describe what is enforced. The old text promised a
    "majority of >= 2 human votes" while the code counted tie votes toward that
    majority; a rule that does not say what it checks cannot be audited."""
    rows = _rows("preference")
    rules = {r["source"]["source_fields"]["label_rule"] for r in rows}
    assert len(rules) == 1, f"preference split ships {len(rules)} different label rules"
    rule = rules.pop()
    assert "decisive" in rule.lower() and "tie" in rule.lower(), (
        f"label_rule does not state the decisive-vote and tie conditions: {rule!r}"
    )


# -- Pairwise: no candidate text may carry contradictory gold ----------------


def _item_candidates(task):
    """(query key, winner text, loser text) per item of a pairwise split."""
    out = {}
    for r in _rows(task):
        if r["item_id"] in out or r.get("ground_truth_position") not in ("A", "B"):
            continue
        a, b = _candidates(r["prompt_a"])
        assert a is not None, f"{task}: candidates unreadable for {r['pair_id']}"
        winner, loser = (a, b) if r["ground_truth_position"] == "A" else (b, a)
        qkey = r["source"]["source_record_id"]
        for sep in ("#", ";"):
            if sep in qkey:
                qkey = qkey.split(sep, 1)[0]
        out[r["item_id"]] = (qkey, winner, loser)
    assert out, f"{task}: no items read; the check must not vacuously pass"
    return out


@pytest.mark.parametrize("task", PAIRWISE)
def test_no_candidate_text_carries_contradictory_gold(task):
    """A response that is gold-correct in one item and gold-wrong in another on
    the SAME query makes a consistent, correct judge wrong on at least one of
    them. The v2 preference split had 52 such texts across 108/226 items, from
    only 78 MT-Bench questions. The audit gate's ground_truth_consistency check
    cannot see this: it keys on `response_being_judged`, the concatenated
    "A: ... | B: ..." block, which is unique per row by construction.

    Role changes across DIFFERENT queries are legitimate (a TREC-COVID document
    is relevant to one query and not another) and are not asserted on."""
    per_item = _item_candidates(task)
    roles = {}
    for _iid, (qkey, winner, loser) in per_item.items():
        roles.setdefault((qkey, winner), set()).add("correct")
        roles.setdefault((qkey, loser), set()).add("wrong")
    bad = {k for k, v in roles.items() if len(v) > 1}
    affected = [iid for iid, (q, w, l) in per_item.items()
                if (q, w) in bad or (q, l) in bad]
    assert not bad, (
        f"{task}: {len(bad)} candidate text(s) are gold-correct in one item and "
        f"gold-wrong in another on the same query, affecting "
        f"{len(affected)}/{len(per_item)} items; e.g. {sorted(bad)[0][1][:80]!r}"
    )
