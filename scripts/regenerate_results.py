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
from typing import Dict, List

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


def _records(path: Path) -> List[dict]:
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
            # Per-call metadata is carried through because a provider-reported
            # refusal is a distinct outcome from an unparseable answer, and the
            # two are indistinguishable once both are UNCLEAR. Absent on runs
            # made before usage metering existed, which _refusal_stats reports
            # as null rather than as zero refusals.
            **{k: r[k] for k in ("usage_a", "usage_b", "usage_a_repeat") if k in r},
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


REFUSAL = "refusal"


def _arm_refused(rec: dict, arm: str) -> bool:
    """Whether the provider flagged this arm as declined.

    Read from the per-call metadata the runner records, never inferred from the
    text: a judge that writes "I cannot help with that" in parseable prose is
    malformed output, not a provider-flagged refusal, and the two must not be
    silently merged.
    """
    usage = rec.get(f"usage_{arm}") or {}
    return usage.get("finish_reason") == REFUSAL


def _pair_class(rec: dict) -> str:
    """One of: both_verdict, one_refused, both_refused.

    Records with no usage metadata at all (runs predating metering) cannot carry
    a refusal, so they classify as both_verdict and behave exactly as before.
    """
    a, b = _arm_refused(rec, "a"), _arm_refused(rec, "b")
    if a and b:
        return "both_refused"
    if a or b:
        return "one_refused"
    return "both_verdict"


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
    classes = [_pair_class(r) for r in recs]
    n = len(recs) or 1
    return {
        "n_verdict_pairs": classes.count("both_verdict"),
        "refusal_discordance_rate": round(classes.count("one_refused") / n, 4),
        "consistent_refusal_rate": round(classes.count("both_refused") / n, 4),
    }


def _refusal_stats(recs: List[dict]) -> Dict:
    """Share of arm-calls the provider reported as a refusal.

    Read from the per-call usage metadata the runner records
    (`finish_reason == "refusal"`), so it reflects what the provider said rather
    than an inference from empty output. Null where no arm carried usage at all,
    which is the case for runs made before usage metering existed.
    """
    refused = arms = 0
    for r in recs:
        for key in ("usage_a", "usage_b", "usage_a_repeat"):
            if key not in r:
                continue
            usage = r.get(key) or {}
            if not usage:
                continue
            arms += 1
            refused += usage.get("finish_reason") == "refusal"
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
    verdict = [r for r in recs if _pair_class(r) == "both_verdict"]
    scored = verdict or recs
    strict = cluster_bootstrap_ci(scored, lambda r: jss(r, "disagree"), "item", n_bootstrap=2000)
    out = {
        "n_rows": len(recs),
        "n_items": len({r["item_id"] for r in recs}),
        "jss_strict": round(strict["estimate"], 4),
        "ci95": [round(strict["ci_lower"], 4), round(strict["ci_upper"], 4)],
        "cluster_unit": "item",
        "chance_corrected_jss": _round4(_defined(chance_corrected_jss, scored, "disagree")),
        "decision_entropy_bits": _round4(_defined(decision_entropy, scored)),
        "label_histogram": label_histogram(recs),
        # Malformed output is counted over BOTH arms: a judge can fail to parse
        # on either phrasing, and reporting one side under-states the rate that
        # the strict-mode JSS is charging for.
        "malformed_rate": round(
            (format_failure_rate(recs, "a")["n_failed"]
             + format_failure_rate(recs, "b")["n_failed"]) / (2 * len(recs)), 4),
        "malformed_rate_arm_a": round(format_failure_rate(recs, "a")["format_failure_rate"], 4),
        "malformed_rate_arm_b": round(format_failure_rate(recs, "b")["format_failure_rate"], 4),
        # A judge that DECLINES an item is not the same measurement as one whose
        # answer failed to parse, but both collapse to UNCLEAR and would be
        # reported identically. claude-sonnet refuses 30% of the TREC-COVID
        # relevance items while claude-haiku and claude-opus-4-7 refuse none of
        # the same prompts, so a malformed_rate that silently folds the two
        # together would attribute a safety behaviour to format-following.
        **_refusal_stats(recs),
        **_refusal_taxonomy(recs),
        # Sensitivity analysis: every refused arm counted as disagreement, the
        # most punitive reading. Reported so a reviewer can see what the
        # conditioning above is worth rather than having to take it on trust.
        "jss_strict_refusal_inclusive": _round4(
            _defined(lambda rs: jss(rs, "disagree"), recs)) if len(verdict) != len(recs) else None,
        "jss_support": "verdict_pairs" if verdict else "all_rows",
    }
    if likert:
        out["quadratic_weighted_kappa"] = _round4(
            _defined(quadratic_weighted_kappa, recs, unclear_policy="disagree"))
    if task not in POINTWISE:
        out["pairwise"] = _accuracy(recs, task)
    if any(r.get("decision_a_repeat") is not None for r in recs):
        rep = [{"decision_a": r["decision_a"], "decision_b": r["decision_a_repeat"],
                "item_id": r["item_id"]} for r in recs if r.get("decision_a_repeat")]
        out["jss_repeat_delta"] = jss_repeat_delta(recs, rep, "item", n_bootstrap=2000)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate v2 results from raw outputs.")
    ap.add_argument("--raw", default=str(RAW))
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    files = sorted(raw.glob("*_*.jsonl"))
    if not files:
        print(f"No raw outputs in {raw}. Run src/run_v2.py first.")
        return 1

    summary: Dict[str, Dict] = {}
    for f in files:
        judge, task = f.stem.rsplit("_", 1)
        recs = _records(f)
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

    # Coherence table, sorted by JSS — the discriminating task, mirrors the paper.
    rows = []
    for judge, tasks in summary.items():
        if "coherence" in tasks:
            c = tasks["coherence"]
            rows.append((judge, c["jss_strict"], c["chance_corrected_jss"],
                         c["ci95"], c["n_items"]))
    rows.sort(key=lambda r: -r[1])
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Judge & JSS & $\kappa$ & 95\% CI (item) & items \\", r"\midrule"]
    for judge, j, k, ci, n in rows:
        lines.append(f"{judge} & {j:.3f} & {k:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_JSON} ({len(summary)} judges) and {OUT_TEX}")
    print("Every reported number is now derived from committed raw outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
