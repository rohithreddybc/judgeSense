"""
Regenerate v2 results from raw judge outputs — the run->paper loop.

Reads data/results_v2/raw/{judge}_{task}.jsonl (produced by src/run_v2.py) and
emits, per (judge, task):
  - strict JSS and its item-clustered 95% bootstrap CI,
  - chance-corrected JSS (kappa),
  - decision-entropy and label histogram,
  - malformed-output rate,
  - quadratic-weighted kappa for coherence,
  - position-corrected accuracy for the pairwise tasks,
  - JSS-vs-repeat delta where a repeat baseline was collected.

Outputs data/results_v2/metrics_summary.json and a ready-to-\\input LaTeX table
(tables/main_results_v2.tex). Every number is derived here from committed raw
outputs, so the paper's results are reproducible by re-running this one script;
nothing is transcribed by hand.

Clustering unit is ALWAYS "item": the two ab_order rows of a pairwise item share
an item_id, and repeated arms nest within it, so item-level resampling is the
only unit that does not understate uncertainty. This is enforced, not optional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_REPO))

from src.metrics_v2 import (  # noqa: E402
    cluster_bootstrap_ci, jss, chance_corrected_jss, decision_entropy,
    label_histogram, quadratic_weighted_kappa, format_failure_rate,
    jss_repeat_delta,
)
from src.structural_variants import UNCLEAR  # noqa: E402

RAW = _REPO / "data" / "results_v2" / "raw"
OUT_JSON = _REPO / "data" / "results_v2" / "metrics_summary.json"
OUT_TEX = _REPO / "tables" / "main_results_v2.tex"
POINTWISE = {"factuality", "coherence"}


class MixedConfigurationError(RuntimeError):
    """A cell contains rows produced under more than one decoding budget."""


def _records(path: Path, budget_policy: Optional[str] = None) -> List[dict]:
    """Raw rows -> metric records ({decision_a, decision_b, item_id, ...}).

    Deduplicates by pair_id, KEEPING THE LAST record written. The runner appends
    and never rewrites, so a row that errored and was later retried leaves two
    records: the stale failure and the good retry. Reading both would feed a
    phantom UNCLEAR disagreement into that item's cluster and silently bias the
    metrics. Last-write-wins matches the runner's own resume semantics, under
    which an errored row is not "done" and is re-executed.
    """
    by_pair: Dict[str, dict] = {}
    order: List[str] = []
    n_superseded = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write can leave a truncated final line; it is
            # incomplete data, not a decision, so it is dropped rather than
            # guessed at.
            continue
        pid = str(r.get("pair_id"))
        if pid in by_pair:
            n_superseded += 1
        else:
            order.append(pid)
        by_pair[pid] = r
    if n_superseded:
        print(f"  [{path.name}] {n_superseded} superseded record(s) ignored "
              f"(retried rows); using the last write per pair_id")

    # Rows produced under different decoding budgets are different
    # measurements and must never be pooled into one cell. Scoring the mixture
    # silently reported a judge as run at one budget while part of its data came
    # from another.
    policies = {r.get("budget_policy") for r in by_pair.values()}
    policies.discard(None)
    if budget_policy is not None and policies:
        # A file whose records carry NO policy at all predates the field; it is
        # legacy, not a mismatch, and is scored as-is. Filtering only applies
        # once at least one record declares a policy, which is what makes a
        # mixed cell detectable.
        by_pair = {k: v for k, v in by_pair.items()
                   if v.get("budget_policy") in (budget_policy, None)}
        order = [p for p in order if p in by_pair]
    elif len(policies) > 1:
        raise MixedConfigurationError(
            f"{path.name} mixes decoding budgets {sorted(policies)}. "
            f"Re-run the cell under one policy, or pass budget_policy= to select. "
            f"Pooling them would report the cell as run at a budget it was not."
        )

    recs = []
    for pid in order:
        r = by_pair[pid]
        recs.append({
            "decision_a": r.get("decision_a", UNCLEAR),
            "decision_b": r.get("decision_b", UNCLEAR),
            "item_id": r.get("item_id"),
            "prompt_pair_id": r.get("prompt_pair_id"),
            "ground_truth_label": r.get("ground_truth_label"),
            "ground_truth_position": r.get("ground_truth_position"),
            "decision_a_repeat": r.get("decision_a_repeat"),
            # Arm B's repeat is required for the pooled ceiling. Omitting it
            # here would have left the ceiling silently template-A-only even
            # after the runner began collecting both.
            "decision_b_repeat": r.get("decision_b_repeat"),
            # Required by the repeat-delta support filter. Omitting it made
            # every row look canonical, which silently restored the very
            # canonical-vs-swapped contamination the filter exists to remove.
            "ab_order": r.get("ab_order"),
            "budget_policy": r.get("budget_policy"),
            # Per-call metadata is carried through because a provider-reported
            # refusal is a distinct outcome from an unparseable answer, and the
            # two are indistinguishable once both are UNCLEAR. Absent on runs
            # made before usage metering existed, which _refusal_stats reports
            # as null rather than as zero refusals.
            **{k: r[k] for k in
               ("usage_a", "usage_b", "usage_a_repeat", "usage_b_repeat") if k in r},
        })
    return recs


def _accuracy(recs: List[dict], task: str) -> Dict:
    """Position-corrected accuracy for pairwise tasks: did arm A pick the side
    carrying the ground truth? Reported so a ceiling (all correct) or a
    position-anchored judge (chance) is visible directly, per reviewer p5cJ W2."""
    scored = [r for r in recs if r.get("ground_truth_position") in ("A", "B")
              and r["decision_a"] in ("A", "B")]
    if not scored:
        return {"accuracy": None, "n": 0}
    correct = sum(1 for r in scored if r["decision_a"] == r["ground_truth_position"])
    a_rate = sum(1 for r in scored if r["decision_a"] == "A") / len(scored)
    return {"accuracy": correct / len(scored), "answer_A_rate": a_rate, "n": len(scored)}


# Gold labels are stored in the source corpus's vocabulary; the judge answers in
# the task's answer space. Mapping them here rather than at the call site keeps
# the correspondence in one place.
_POINTWISE_LABEL = {"accurate": "YES", "inaccurate": "NO"}


def _pointwise_accuracy(recs: List[dict], task: str) -> Dict:
    """Exact-match accuracy on arm A for the pointwise tasks.

    Computed in the pipeline because the paper's discrimination check depends on
    it, and the two figures it needed were previously hand-derived in a shell --
    beneath a reproducibility statement asserting that no number in the paper is
    transcribed by hand.

    Also reports the majority-class rate, which is the honest floor: a constant
    judge scores that, not the uniform 1/k a five-point scale suggests. On
    coherence the gold distribution is skewed enough that the two differ by 15
    points, and quoting uniform chance overstates the discrimination margin.
    """
    import collections
    scored, correct = 0, 0
    gold = collections.Counter()
    for rec in recs:
        label = rec.get("ground_truth_label")
        decision = rec.get("decision_a")
        if label is None or decision in (None, UNCLEAR):
            continue
        gold[str(label)] += 1
        scored += 1
        correct += str(decision) == _POINTWISE_LABEL.get(str(label), str(label))
    if not scored:
        return {"accuracy": None, "n": 0, "majority_class_rate": None}
    return {
        "accuracy": round(correct / scored, 4),
        "n": scored,
        "majority_class_rate": round(max(gold.values()) / scored, 4),
        "arm": "a",
    }


def _defined(fn, *args, **kwargs):
    """A metric's value, or None where it is mathematically undefined.

    A judge that never emits a parseable decision has no chance-correctable
    pairs, no scorable Likert records, and no decision distribution, so
    `chance_corrected_jss`, `quadratic_weighted_kappa` and `decision_entropy`
    all raise ValueError on it. That is not a defect in the run: a cell whose
    output is 100% malformed is a real and publishable result about that judge,
    reported here as a malformed_rate of 1.0 with the undefined metrics null.
    Allowing the exception to propagate aborted `main`'s loop over raw files, so
    one unparseable judge destroyed the metrics for every other judge in a run
    that had already been paid for.
    """
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return None


def _round4(value):
    # `+ 0.0` normalises negative zero: a degenerate distribution rounds to
    # -0.0, which serialises into JSON as "-0.0" and reads as a defect.
    return None if value is None else round(value, 4) + 0.0


# Providers spell "I declined to answer" differently, and the raw string is
# stored verbatim so the record keeps what the provider actually said. Matching
# only Anthropic's "refusal" silently misclassified every OpenAI-compatible
# decline as a verdict: its empty content parses to UNCLEAR, and under the
# disagree policy that is then CHARGED AS PARAPHRASE DISAGREEMENT -- the exact
# outcome the taxonomy exists to prevent. Those judges' refusal rates would have
# read 0.0 no matter what happened.
REFUSAL_REASONS = frozenset({
    "refusal",          # anthropic
    "content_filter",   # openai and OpenAI-compatible gateways
    "safety",           # google
    "recitation",       # google, blocked output
    "blocklist",        # google
    "prohibited_content",
})


def _is_refusal_reason(reason) -> bool:
    """The one place a provider decline is recognised.

    Both the pair classifier and the per-call rate route through this, so the
    two can no longer disagree about what a refusal is -- they did, and the
    per-call rate kept matching Anthropic's spelling after the classifier was
    generalised.
    """
    return isinstance(reason, str) and reason.strip().lower() in REFUSAL_REASONS


def _arm_refused(rec: dict, arm: str) -> bool:
    """Whether the provider flagged this arm as declined.

    Read from the per-call metadata the runner records, never inferred from the
    text: a judge that writes "I cannot help with that" in parseable prose is
    malformed output, not a provider-flagged refusal, and the two must not be
    silently merged.

    Matching is case-insensitive because providers are inconsistent about case
    (google returns enum names such as SAFETY).
    """
    usage = rec.get(f"usage_{arm}") or {}
    return _is_refusal_reason(usage.get("finish_reason"))


def _arm_transport_failed(rec: dict, arm: str) -> bool:
    """The call did not complete, so the judge never had a chance to answer.

    A dead call yields UNCLEAR exactly like an unparseable answer, and without
    this it is published as the judge's malformed-output rate AND charged as
    paraphrase disagreement. A 529 storm during a paid run would appear in the
    paper as a format-following defect.
    """
    usage = rec.get(f"usage_{arm}")
    if usage is None:
        return False
    return bool(usage.get("error")) or usage.get("finish_reason") is None


def _arm_malformed_rate(recs: List[dict], arm: str) -> Optional[float]:
    """Unparseable answers on one arm, excluding refusals and dead calls.

    Those are not format failures: one is a decline, the other never reached the
    model. Counting them here attributes a safety behaviour, or a network fault,
    to the judge's instruction-following.
    """
    considered = malformed = 0
    for rec in recs:
        if _arm_transport_failed(rec, arm) or _arm_refused(rec, arm):
            continue
        considered += 1
        malformed += rec.get(f"decision_{arm}") == UNCLEAR
    return (malformed / considered) if considered else None


# Pair-class labels are constants, not literals. These were literals, the class
# was renamed from "both_verdict" to "both_answered", and one comparison in
# _refusal_bounds kept the old spelling. It never matched, so the Manski bounds
# were silently None in every cell -- while two sections of the paper claimed to
# report them and one described their width. A typo in a string literal produced
# a false claim in a manuscript, and no test caught it because the function
# returned a well-formed dict either way.
PAIR_BOTH_ANSWERED = "both_answered"
PAIR_ONE_REFUSED = "one_refused"
PAIR_BOTH_REFUSED = "both_refused"
PAIR_TRANSPORT_ERROR = "transport_error"


def _pair_class(rec: dict) -> str:
    """One of: both_verdict, one_refused, both_refused.

    Records with no usage metadata at all (runs predating metering) cannot carry
    a refusal, so they classify as both_verdict and behave exactly as before.
    """
    if _arm_transport_failed(rec, "a") or _arm_transport_failed(rec, "b"):
        return PAIR_TRANSPORT_ERROR
    a, b = _arm_refused(rec, "a"), _arm_refused(rec, "b")
    if a and b:
        return PAIR_BOTH_REFUSED
    if a or b:
        return PAIR_ONE_REFUSED
    return PAIR_BOTH_ANSWERED


def _malformed_rate(recs: List[dict]) -> Optional[float]:
    """Unparseable COMPLETED responses, over arms that were not refused.

    Both a refusal and a malformed answer end as UNCLEAR, so counting UNCLEAR
    directly double-counts every refusal and attributes a safety behaviour to
    format-following. The denominator excludes refused arms too: a refused arm
    had no opportunity to be well-formed, so including it deflates the rate.
    """
    failed = arms = 0
    for r in recs:
        for arm in ("a", "b"):
            if _arm_refused(r, arm):
                continue
            arms += 1
            failed += r.get(f"decision_{arm}") in (None, UNCLEAR)
    return failed / arms if arms else None


def _outcome_partition(recs: List[dict]) -> Dict:
    """The three outcome categories, as counts that must sum to the arm total.

    Reported as an explicit partition because the paper claims each call ends in
    exactly one of three outcomes; an assertion is cheaper than a reader
    checking it by hand.
    """
    verdict = refused = malformed = 0
    for r in recs:
        for arm in ("a", "b"):
            if _arm_refused(r, arm):
                refused += 1
            elif r.get(f"decision_{arm}") in (None, UNCLEAR):
                malformed += 1
            else:
                verdict += 1
    total = verdict + refused + malformed
    return {
        "n_arm_calls": total,
        "n_verdict_arms": verdict,
        "n_refused_arms": refused,
        "n_malformed_arms": malformed,
        "partitions": total == 2 * len(recs),
    }


def _refusal_taxonomy(recs: List[dict]) -> Dict:
    """Refusal as an outcome category, decomposed.

    A refusal is upstream of any judgement: the provider halted before the model
    rendered a verdict. Scoring it as paraphrase DISAGREEMENT asserts the judge
    produced two conflicting judgements, which it did not; scoring it as a third
    LABEL would award JSS 1.0 to a judge that refuses everything. So JSS is
    computed over pairs where both arms returned verdicts, and refusal is
    reported separately.

    The discordance rate is itself a sensitivity statistic, and the most
    interesting quantity here: a pair where one arm was refused and the other
    judged means a meaning-preserving rewording changed whether the judge was
    willing to judge at all.
    """
    # Metering is the presence of a usage record, not the presence of a
    # finish_reason inside it. Keyed on finish_reason, a cell whose every call
    # DIED -- usage present, error set, finish_reason absent -- was reported as
    # never metered, and took an early return carrying different keys from the
    # normal path.
    metered = any(r.get(f"usage_{arm}") is not None
                  for r in recs for arm in ("a", "b"))
    if not metered:
        # Same rule as refusal_rate: absent metering is unknown, not zero.
        # Reporting RDR 0.0 for an unmetered cell states that no rewording
        # changed the judge's willingness to answer, which was never measured.
        return {"n_pairs_both_answered": len(recs),
                "n_pairs_transport_error": 0,
                "refusal_discordance_rate": None,
                "refusal_discordance_ci95": None,
                "consistent_refusal_rate": None}
    classes = [_pair_class(r) for r in recs]
    n = len(recs) or 1
    # RDR is called the most interesting of these statistics, so it carries an
    # interval on the same clustering unit as everything else. Reporting it as a
    # bare proportion over ROWS also double-counted every pairwise item.
    rdr_ci = _defined(
        cluster_bootstrap_ci,
        recs, lambda rs: sum(1 for r in rs if _pair_class(r) == PAIR_ONE_REFUSED) / len(rs),
        "item", n_bootstrap=2000,
    )
    return {
        # Named for what it is. Malformed arms REMAIN in this support and are
        # scored as disagreement -- a completed response the parser could not
        # read is judge behaviour, unlike a refusal or a dead call. Calling it
        # "verdict pairs" implied the preregistered both-arms-parsed support,
        # which is a different and smaller set.
        "n_pairs_both_answered": classes.count(PAIR_BOTH_ANSWERED),
        "n_pairs_transport_error": classes.count(PAIR_TRANSPORT_ERROR),
        "refusal_discordance_rate": round(classes.count(PAIR_ONE_REFUSED) / n, 4),
        "refusal_discordance_ci95": (
            [round(rdr_ci["ci_lower"], 4), round(rdr_ci["ci_upper"], 4)]
            if rdr_ci else None
        ),
        "consistent_refusal_rate": round(classes.count(PAIR_BOTH_REFUSED) / n, 4),
    }


def _refusal_stats(recs: List[dict]) -> Dict:
    """Share of arm-calls the provider reported as a refusal.

    Read from the per-call usage metadata the runner records
    (a provider decline reason in REFUSAL_REASONS), so it reflects what the
    provider said rather
    than an inference from empty output. Null where no arm carried usage at all,
    which is the case for runs made before usage metering existed.
    """
    refused = arms = 0
    for r in recs:
        for key in ("usage_a", "usage_b", "usage_a_repeat", "usage_b_repeat"):
            if key not in r:
                continue
            usage = r.get(key) or {}
            if not usage:
                continue
            arms += 1
            refused += _is_refusal_reason(usage.get("finish_reason"))
    if not arms:
        return {"refusal_rate": None, "n_refusals": 0, "n_metered_arms": 0}
    return {
        "refusal_rate": round(refused / arms, 4),
        "n_refusals": refused,
        "n_metered_arms": arms,
    }


def metrics_for_cell(recs: List[dict], task: str) -> Dict:
    likert = task == "coherence"
    # The sensitivity construct is measured on its proper support: pairs where
    # the judge actually rendered a verdict on both phrasings. The
    # refusal-inclusive figure is reported below as a sensitivity analysis, so
    # nothing is hidden by the conditioning.
    verdict = [r for r in recs if _pair_class(r) == PAIR_BOTH_ANSWERED]
    scored = verdict or recs
    strict = cluster_bootstrap_ci(scored, lambda r: jss(r, "disagree"), "item", n_bootstrap=2000)
    out = {
        "n_rows": len(recs),
        # n_items counts every item in the cell; n_items_analysed is the cluster
        # count the interval was actually computed over, which is smaller
        # wherever pairs were dropped from the support. Printing only the first
        # beside a CI overstated the analysed support by 43% on one cell.
        "n_items": len({r["item_id"] for r in recs}),
        "n_items_analysed": strict.get("n_clusters"),
        "jss_strict": round(strict["estimate"], 4),
        "ci95": [round(strict["ci_lower"], 4), round(strict["ci_upper"], 4)],
        "cluster_unit": "item",
        "chance_corrected_jss": _round4(_defined(chance_corrected_jss, scored, "disagree")),
        # Cohen's kappa restricts to pairs where BOTH arms parsed, while
        # jss(..., "disagree") keeps an unparseable arm in the denominator as a
        # mismatch. On one shipped cell that was n=69 against n=124, and a kappa
        # of 0.82 sat beside a JSS of 0.51 -- reading as "almost all of the
        # agreement is non-chance", the opposite of what happened. The support
        # is now published, and the JSS computed on kappa's own support is
        # printed next to it so the correction has something to correct.
        "chance_corrected_jss_n": sum(
            1 for r in scored
            if r["decision_a"] != UNCLEAR and r["decision_b"] != UNCLEAR
        ),
        "jss_on_parseable_pairs": _round4(_defined(jss, scored, "drop")),
        "decision_entropy_bits": _round4(_defined(decision_entropy, scored)),
        # Same support as decision_entropy_bits above. Computed on `recs` while
        # the entropy used `scored`, the printed histogram did not reproduce the
        # printed entropy (1.5000 against 1.5786 on one cell) and a reader
        # checking the arithmetic found a discrepancy with no explanation.
        "label_histogram": label_histogram(scored),
        "label_histogram_all_rows": label_histogram(recs),
        # Malformed output is counted over BOTH arms: a judge can fail to parse
        # on either phrasing, and reporting one side under-states the rate that
        # the strict-mode JSS is charging for.
        # Refusals are SUBTRACTED here. A refused arm yields empty content,
        # which parses to UNCLEAR and would otherwise be counted a second time
        # as malformed output -- so the two "distinct outcomes" did not
        # partition, and a judge's safety behaviour was reported as a
        # format-following failure. Measured on one cell before the fix:
        # malformed 0.500 of which 0.298 was refusal.
        "malformed_rate": _round4(_malformed_rate(recs)),
        # Refusal-excluded, like malformed_rate. format_failure_rate has no
        # refusal awareness, so these two keys carried a different definition
        # from the pooled key sharing their prefix: on one cell arm A read
        # 0.5722 against a pooled 0.2857, inflating that template's apparent
        # format-failure rate twofold. The per-arm split is exactly what a
        # reader consults to blame one template.
        "malformed_rate_arm_a": _round4(_arm_malformed_rate(recs, "a")),
        "malformed_rate_arm_b": _round4(_arm_malformed_rate(recs, "b")),
        "unclear_rate_arm_a": round(format_failure_rate(recs, "a")["format_failure_rate"], 4),
        "unclear_rate_arm_b": round(format_failure_rate(recs, "b")["format_failure_rate"], 4),
        # A judge that DECLINES an item is not the same measurement as one whose
        # answer failed to parse, but both collapse to UNCLEAR and would be
        # reported identically. claude-sonnet refuses 30% of the TREC-COVID
        # relevance items while claude-haiku and claude-opus-4-7 refuse none of
        # the same prompts, so a malformed_rate that silently folds the two
        # together would attribute a safety behaviour to format-following.
        **_refusal_stats(recs),
        **_refusal_taxonomy(recs),
        "outcome_partition": _outcome_partition(recs),
        **_refusal_bounds(recs),
        # Sensitivity analysis: every refused arm counted as disagreement, the
        # most punitive reading. Reported so a reviewer can see what the
        # conditioning above is worth rather than having to take it on trust.
        "jss_strict_refusal_inclusive": _round4(
            _defined(lambda rs: jss(rs, "disagree"), recs)) if len(verdict) != len(recs) else None,
        "jss_support": "verdict_pairs" if verdict else "all_rows",
    }
    if likert:
        # "drop", not "disagree". Under the disagree policy an unparseable
        # Likert answer is imputed to whichever extreme is FARTHEST from the
        # other arm -- an UNCLEAR/UNCLEAR pair becomes (1, 5). That is maximal
        # coercion, in a section that states parsing never coerces. It is also
        # computed on `scored`, so kappa and the JSS printed beside it share one
        # support; previously they did not, and kappa was not a correction of
        # the number it sat next to.
        out["quadratic_weighted_kappa"] = _round4(
            _defined(quadratic_weighted_kappa, scored, unclear_policy="drop"))
        out["quadratic_weighted_kappa_policy"] = "drop"
        out["quadratic_weighted_kappa_punitive"] = _round4(
            _defined(quadratic_weighted_kappa, scored, unclear_policy="disagree"))
    if task not in POINTWISE:
        out["pairwise"] = _accuracy(recs, task)
    else:
        out["pointwise"] = _pointwise_accuracy(scored, task)
    if any(r.get("decision_a_repeat") is not None for r in recs):
        out["jss_repeat_delta"] = _repeat_delta(verdict or recs, all_recs=recs)
    return out


# A cell whose primary endpoint rests on a handful of items is not a result. The
# floor is declared here, in code, rather than left to a reader to notice: one
# committed cell carried a delta computed on THREE clusters with a zero-width
# 95% interval and no warning attached.
MIN_DELTA_CLUSTERS = 100
MAX_ITEM_LOSS_FRACTION = 0.5


def _rule_of_three_upper(n_trials: int) -> Optional[float]:
    """Upper 95% bound on an event rate after n trials with zero events.

    When a judge never self-disagrees, every bootstrap replicate returns a
    repeat agreement of exactly 1.000, the ceiling contributes ZERO variance to
    the delta, and the percentile interval collapses onto the paraphrase term
    alone. But one repeat call per item bounds the true self-disagreement rate
    only at 3/n -- on 250 items that is 1.2%, the same order as the effects
    being claimed. Reporting it stops a zero-width ceiling from reading as a
    known quantity.
    """
    return (3.0 / n_trials) if n_trials else None


def _refusal_bounds(recs: List[dict]) -> Dict:
    """Worst-case bounds on JSS under the refusal conditioning.

    Conditioning on both-arms-answered is selection on a POST-TREATMENT
    variable: the paper's own RDR construct concedes that rewording can change
    whether a judge answers at all, which makes refusal an outcome rather than a
    nuisance. The conditioned estimate is therefore point-identified only under
    an assumption nobody can check.

    These are the Manski bounds: every refused pair counted as agreement gives
    the upper edge, every one as disagreement the lower. The truth lies inside
    regardless of why the judge declined, so the width states how much the
    conditioning is actually worth.
    """
    classes = [_pair_class(r) for r in recs]
    verdict = [r for r, c in zip(recs, classes) if c == PAIR_BOTH_ANSWERED]
    n_refused = sum(1 for c in classes if c != PAIR_BOTH_ANSWERED)
    n = len(recs)
    if not n or not verdict:
        return {"jss_bounds": None}
    agree = sum(1 for r in verdict if r["decision_a"] == r["decision_b"])
    return {
        "jss_bounds": {
            "lower": round(agree / n, 4),
            "upper": round((agree + n_refused) / n, 4),
            "width": round(n_refused / n, 4),
            "n_unidentified_pairs": n_refused,
            "basis": "Manski worst case: refused pairs counted as all-disagree "
                     "then all-agree; no assumption about why the judge declined",
        }
    }


def _repeat_delta(recs: List[dict], all_recs: Optional[List[dict]] = None) -> Dict:
    """ΔJSS on a support both terms actually share.

    Two defects made the earlier call uninterpretable.

    First, the repeat arm is issued only on the canonical ordering, while the
    paraphrase term averaged canonical AND position-swapped rows. Swapped rows
    are by design the harder half -- they exist to defeat a position-anchored
    judge -- so the difference absorbed a canonical-versus-swapped contrast that
    has nothing to do with wording, biasing the endpoint toward the hypothesised
    sign by construction. Both terms are now restricted to the canonical rows.

    Second, the paraphrase term was computed refusal-inclusive while the JSS
    reported beside it was computed over verdict pairs, so one cell carried
    three different JSS values under one estimation rule. The caller now passes
    the same support used for jss_strict.
    """
    canonical = [r for r in recs if r.get("ab_order") in (None, "original")]
    # The ceiling is estimated from BOTH arms' repeats, not arm A's alone.
    # If template B is intrinsically higher-entropy -- longer, likelier to draw a
    # preamble the strict parser rejects -- then a ceiling measured only under A
    # cannot absorb noise generated under B, and that noise is charged to
    # paraphrasing. Pooling both repeats makes the ceiling a property of the
    # template PAIR, which is what the paraphrase term is computed over.
    rep = []
    for r in canonical:
        for arm in ("a", "b"):
            other = r.get(f"decision_{arm}_repeat")
            if other is None:
                continue
            # The repeat arm gets the SAME refusal and transport rules as the
            # paraphrase arm. Without this a provider-refused repeat scored as
            # self-disagreement, deflating the ceiling and inflating the delta:
            # a perfectly stable judge with half its repeats refused reported
            # delta +0.50 with an interval excluding zero.
            usage = r.get(f"usage_{arm}_repeat") or {}
            if _is_refusal_reason(usage.get("finish_reason")):
                continue
            if usage and (usage.get("error") or usage.get("finish_reason") is None):
                continue
            rep.append({"decision_a": r[f"decision_{arm}"],
                        "decision_b": other,
                        "item_id": r["item_id"]})
    if not rep:
        return {"delta": None, "reason": "no repeat arm on the canonical ordering"}

    para_items = {r["item_id"] for r in canonical}
    rep_items = {r["item_id"] for r in rep}
    common = para_items & rep_items
    # Measured against every canonical item the cell STARTED with, not against
    # the post-refusal-filter set. Computed the old way, a judge refusing two
    # thirds of its items reported item_loss_fraction 0.0, and the preregistered
    # 50% ceiling was unreachable by construction.
    base = all_recs if all_recs is not None else recs
    all_canonical = {r["item_id"] for r in base if r.get("ab_order") in (None, "original")}
    denom = all_canonical or para_items
    loss = 1.0 - (len(common) / len(denom)) if denom else 1.0
    if len(common) < MIN_DELTA_CLUSTERS:
        return {"delta": None, "n_clusters": len(common),
                "reason": f"support {len(common)} below the declared floor of "
                          f"{MIN_DELTA_CLUSTERS} clusters"}
    if loss > MAX_ITEM_LOSS_FRACTION:
        return {"delta": None, "n_clusters": len(common),
                "item_loss_fraction": round(loss, 4),
                "reason": f"{loss:.0%} of items lack a usable pair on one side; "
                          f"above the declared {MAX_ITEM_LOSS_FRACTION:.0%} ceiling"}

    out = jss_repeat_delta(canonical, rep, "item", n_bootstrap=2000)
    out["support"] = "canonical ordering, verdict pairs"
    # Arm-specific ceilings, reported as a diagnostic: a large gap between them
    # is evidence the two templates are not exchangeable, which would undermine
    # the paraphrase design itself rather than merely widen an interval.
    per_arm = {}
    for arm in ("a", "b"):
        pairs = [r for r in canonical if r.get(f"decision_{arm}_repeat") is not None]
        if pairs:
            # Same unclear_policy as jss_rep. Raw string equality credited
            # UNCLEAR == UNCLEAR as agreement while jss_rep charged it as
            # disagreement, so the guard that exists to detect non-exchangeable
            # templates reported a 0.032 gap on a cell whose true gap is 0.209
            # and whose arms differ 2.7x in parse-failure rate.
            paired = [{"decision_a": r[f"decision_{arm}"],
                       "decision_b": r[f"decision_{arm}_repeat"],
                       "item_id": r["item_id"]} for r in pairs]
            value = _defined(jss, paired, "disagree")
            if value is not None:
                per_arm[arm] = round(value, 4)
    out["repeat_agreement_by_arm"] = per_arm or None
    if len(per_arm) == 2:
        out["arm_ceiling_gap"] = round(abs(per_arm["a"] - per_arm["b"]), 4)
    else:
        # A ceiling measured under one template cannot separate "the judge is
        # destabilised by rewording" from "template B is intrinsically noisier
        # or harder to parse". Emitting delta anyway published that confound
        # silently, because the pooling fix landed in code while every shipped
        # record still carried arm A only.
        out["ceiling_single_arm"] = True
        out["ceiling_arms_present"] = sorted(per_arm)
        out["delta_withheld_reason"] = (
            "repeat baseline collected on one prompt arm only; delta cannot "
            "separate paraphrase sensitivity from template-specific decoding "
            "noise. Re-collect with --repeat-baseline on the current runner."
        )
        out["delta_single_arm_ceiling"] = out.pop("delta", None)
    out["item_loss_fraction"] = round(loss, 4)
    # At a ceiling of exactly 1.000 the percentile bootstrap reports zero
    # uncertainty for a quantity estimated from one draw per item. State the
    # rule-of-three bound so the ceiling is not read as known.
    if out.get("jss_rep") is not None and out["jss_rep"] >= 1.0:
        bound = _rule_of_three_upper(len(rep))
        out["ceiling_at_boundary"] = True
        out["ceiling_disagreement_upper_95"] = round(bound, 4) if bound else None
        out["ceiling_note"] = (
            "repeat agreement is exactly 1.000, so the bootstrap attributes no "
            "variance to the ceiling; with one repeat per item the true "
            "self-disagreement rate is bounded above by 3/n, not zero"
        )
    return out


# The manuscript is a single self-contained file, so the generated table is
# spliced INTO it between sentinels rather than \input from ../tables/. An
# \input makes paper/main.tex unbuildable anywhere the tables directory is
# absent -- Overleaf, an arXiv upload, a coauthor's copy -- and the workaround
# was a second main.tex plus a comment telling people not to use the first,
# which is a trap rather than a design.
#
# Splicing keeps the property that matters: regenerating the results rewrites
# the paper's table, so the two cannot drift. tables/main_results_v2.tex is
# still written, for anything that consumes it directly.
MANUSCRIPT = _REPO / "paper" / "main.tex"
TABLE_BEGIN = "% <<< GENERATED TABLE: regenerate_results.py -- do not edit by hand"
TABLE_END = "% >>> END GENERATED TABLE"


def _splice_into_manuscript(table_tex: str) -> None:
    if not MANUSCRIPT.exists():
        return
    text = MANUSCRIPT.read_text(encoding="utf-8")
    start, end = text.find(TABLE_BEGIN), text.find(TABLE_END)
    if start == -1 or end == -1:
        print(f"  [warn] {MANUSCRIPT.name} has no generated-table sentinels; "
              f"the table was not spliced and the paper may now disagree with "
              f"the data. Restore the sentinels around the table to re-link it.")
        return
    block = f"{TABLE_BEGIN}\n{table_tex.rstrip()}\n{TABLE_END}"
    updated = text[:start] + block + text[end + len(TABLE_END):]
    if updated != text:
        MANUSCRIPT.write_text(updated, encoding="utf-8")
        print(f"  spliced the results table into paper/{MANUSCRIPT.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate v2 results from raw outputs.")
    ap.add_argument("--raw", default=str(RAW))
    # The main results are declared to use the matched budget; selecting it here
    # means a cell polluted by a smoke test at another policy is excluded rather
    # than silently pooled. Pass --budget-policy "" to score every row and let
    # the mixed-configuration guard fire instead.
    ap.add_argument("--budget-policy", default="matched",
                    help="score only rows produced under this decoding budget "
                         "(default: matched; empty string disables the filter)")
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    files = sorted(raw.glob("*_*.jsonl"))
    if not files:
        print(f"No raw outputs in {raw}. Run src/run_v2.py first.")
        return 1

    summary: Dict[str, Dict] = {}
    for f in files:
        judge, task = f.stem.rsplit("_", 1)
        recs = _records(f, args.budget_policy or None)
        if len(recs) < 2:
            continue
        try:
            summary.setdefault(judge, {})[task] = metrics_for_cell(recs, task)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not void the rest
            print(f"  [skip] {judge}/{task}: {type(exc).__name__}: {exc}")
            summary.setdefault(judge, {})[task] = {
                "error": f"{type(exc).__name__}: {exc}", "n_rows": len(recs),
            }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The paper's main table. It carries the PRE-REGISTERED ENDPOINT, not raw
    # JSS: the earlier version emitted coherence-only agreement with no delta,
    # no caption and no label, so \ref{tab:main} in the results section resolved
    # to nothing and the table the paper cited was not the table the pipeline
    # produced. Anything a reader sees here is regenerated from raw outputs.
    def _fmt(value, places=3):
        return "---" if value is None else f"{value:.{places}f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paraphrase sensitivity per judge and task. "
        r"$\Delta\mathrm{JSS} = \mathrm{JSS}_{\text{para}} - "
        r"\mathrm{JSS}_{\text{rep}}$ is the pre-registered endpoint; a negative "
        r"value means rewording costs more agreement than re-issuing the "
        r"identical prompt. Intervals are 95\% item-clustered bootstrap over "
        r"2{,}000 resamples. Cells below the declared support floor of 100 "
        r"clusters report no endpoint rather than one computed from too few. "
        r"$r$ is the provider-reported refusal rate and $m$ the malformed-output "
        r"rate over the arms the judge attempted, so the two do not overlap. The JSS column is computed over "
        r"all rows of the cell; $\mathrm{JSS}_{\text{rep}}$ and "
        r"$\Delta\mathrm{JSS}$ are computed over the canonical ordering only, "
        r"so the two columns do not subtract to the printed delta where a cell "
        r"carries refusals or a swapped-ordering imbalance.}",
        r"\label{tab:main}",
        r"\begin{tabular}{llrrrlrr}",
        r"\toprule",
        r"Judge & Task & $n$ & JSS & $\mathrm{JSS}_{\text{rep}}$ & "
        r"$\Delta\mathrm{JSS}$ [95\% CI] & $r$ & $m$ \\",
        r"\midrule",
    ]
    for judge in sorted(summary):
        first = True
        for task in ("factuality", "coherence", "relevance", "preference"):
            cell = summary[judge].get(task)
            if not cell or "error" in cell:
                continue
            delta = cell.get("jss_repeat_delta") or {}
            if delta.get("delta") is None:
                # State WHY the endpoint is absent; a blank cell reads as an
                # oversight, and the reason is itself a reportable result.
                d_txt = r"\emph{support below floor}"
            else:
                d_txt = (f"${delta['delta']:+.3f}$ "
                         f"[{delta['ci_lower']:.3f}, {delta['ci_upper']:.3f}]")
            lines.append(
                f"{judge if first else ''} & {task} & {cell.get('n_rows', 0)} & "
                f"{_fmt(cell.get('jss_strict'))} & {_fmt(delta.get('jss_rep'))} & "
                f"{d_txt} & {_fmt(cell.get('refusal_rate'))} & "
                f"{_fmt(cell.get('malformed_rate'))} \\\\"
            )
            first = False
        if not first:
            lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    table_tex = "\n".join(lines) + "\n"
    OUT_TEX.write_text(table_tex, encoding="utf-8")
    _splice_into_manuscript(table_tex)

    print(f"Wrote {OUT_JSON} ({len(summary)} judges) and {OUT_TEX}")
    print("Every reported number is now derived from committed raw outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
