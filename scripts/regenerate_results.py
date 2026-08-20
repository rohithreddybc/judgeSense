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


def metrics_for_cell(recs: List[dict], task: str) -> Dict:
    likert = task == "coherence"
    strict = cluster_bootstrap_ci(recs, lambda r: jss(r, "disagree"), "item", n_bootstrap=2000)
    out = {
        "n_rows": len(recs),
        "n_items": len({r["item_id"] for r in recs}),
        "jss_strict": round(strict["estimate"], 4),
        "ci95": [round(strict["ci_lower"], 4), round(strict["ci_upper"], 4)],
        "cluster_unit": "item",
        "chance_corrected_jss": round(chance_corrected_jss(recs, "disagree"), 4),
        "decision_entropy_bits": round(decision_entropy(recs), 4),
        "label_histogram": label_histogram(recs),
        # Malformed output is counted over BOTH arms: a judge can fail to parse
        # on either phrasing, and reporting one side under-states the rate that
        # the strict-mode JSS is charging for.
        "malformed_rate": round(
            (format_failure_rate(recs, "a")["n_failed"]
             + format_failure_rate(recs, "b")["n_failed"]) / (2 * len(recs)), 4),
        "malformed_rate_arm_a": round(format_failure_rate(recs, "a")["format_failure_rate"], 4),
        "malformed_rate_arm_b": round(format_failure_rate(recs, "b")["format_failure_rate"], 4),
    }
    if likert:
        out["quadratic_weighted_kappa"] = round(
            quadratic_weighted_kappa(recs, unclear_policy="disagree"), 4)
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
        summary.setdefault(judge, {})[task] = metrics_for_cell(recs, task)

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
