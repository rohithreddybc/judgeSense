"""
JudgeSense CI data-audit gate.

Fails the build (exit code 1) when the dataset the repository ships violates
integrity requirements. This gate exists because v1 shipped hardcoded items
labeled as benchmark-sourced; it makes that class of defect structurally
unable to pass CI again.

Checks (each configurable via the audit config JSON):
  unique_items          unique judged items per split >= min_unique_items
  duplicate_rows        duplicate-row ratio <= max_duplicate_row_ratio
  provenance            every record's source_benchmark claim is backed by a
                        per-item source record id (+ split); a bare benchmark
                        name with no record id fails
  label_degeneracy      no label exceeds max_label_share; at least
                        min_distinct_labels distinct labels per split
  effective_sample_size clusters at the declared clustering unit >= floor
  annotation_timing     median seconds between consecutive human-validation
                        decisions >= min_seconds_per_decision (when timing
                        data exists)

Usage:
    python scripts/data_audit.py --config data/audit_config.json
    python scripts/data_audit.py --config ... --json-out audit_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def check_unique_items(records: List[dict], cfg: dict) -> dict:
    key = cfg.get("item_key", "response_being_judged")
    fallback = "response_being_judged"
    items = {r.get(key, r.get(fallback)) for r in records}
    items.discard(None)
    floor = cfg["min_unique_items"]
    return {
        "check": "unique_items",
        "observed": len(items),
        "threshold": floor,
        "passed": len(items) >= floor,
        "detail": f"{len(items)} unique items (key={key!r}); floor is {floor}",
    }


def check_duplicate_rows(records: List[dict], cfg: dict) -> dict:
    ignore = {"pair_id", "timestamp", "retrieved_at"}
    seen = set()
    dups = 0
    for rec in records:
        fingerprint = json.dumps(
            {k: v for k, v in sorted(rec.items()) if k not in ignore},
            sort_keys=True, default=str,
        )
        if fingerprint in seen:
            dups += 1
        seen.add(fingerprint)
    ratio = dups / len(records) if records else 0.0
    ceiling = cfg["max_duplicate_row_ratio"]
    return {
        "check": "duplicate_rows",
        "observed": round(ratio, 4),
        "threshold": ceiling,
        "passed": ratio <= ceiling,
        "detail": f"{dups}/{len(records)} rows duplicate another row "
                  f"(ignoring ids/timestamps); ceiling is {ceiling}",
    }


def check_provenance(records: List[dict], cfg: dict) -> dict:
    unbacked = 0
    for rec in records:
        claim = rec.get("source_benchmark") or rec.get("source", {}).get("source_dataset")
        source = rec.get("source") or {}
        record_id = source.get("source_record_id", "")
        split = source.get("source_split", "")
        backed = bool(claim) and bool(record_id) and bool(split) \
            and record_id.strip().lower() != str(claim).strip().lower()
        if not backed:
            unbacked += 1
    return {
        "check": "provenance",
        "observed": unbacked,
        "threshold": 0,
        "passed": unbacked == 0,
        "detail": f"{unbacked}/{len(records)} records claim a source "
                  "benchmark without a per-item source record id + split "
                  "(v1 shipped 500/500 such records)",
    }


def check_label_degeneracy(records: List[dict], cfg: dict) -> dict:
    """
    Guards the question "could a judge score well by always answering the same
    thing?" — so it must inspect the label in the JUDGE'S DECISION SPACE.

    For a pairwise task built with the A/B swap design, that is
    `ground_truth_position` (which side the correct candidate is displayed on),
    not `ground_truth_label` (which candidate is correct by content). The
    content label is constant there by construction — in relevance the
    qrels-positive document is always the relevant one, there is no alternative
    — while position is balanced, so an always-"A" judge scores ~50%, not 100%.
    Scoring the content label would fail a sound dataset.

    Where no positional field exists (factuality, coherence) the content label
    IS the decision, and it is used directly. The content histogram is always
    reported so a genuinely degenerate construction stays visible.
    """
    positional = [r for r in records if r.get("ground_truth_position") is not None]
    use_position = len(positional) == len(records) and records
    field = "ground_truth_position" if use_position else "ground_truth_label"

    labels: Dict[str, int] = {}
    content: Dict[str, int] = {}
    for rec in records:
        lab = str(rec.get(field))
        labels[lab] = labels.get(lab, 0) + 1
        clab = str(rec.get("ground_truth_label"))
        content[clab] = content.get(clab, 0) + 1

    total = sum(labels.values())
    top_share = max(labels.values()) / total if total else 1.0
    distinct = len(labels)
    max_share = cfg["max_label_share"]
    min_distinct = cfg["min_distinct_labels"]
    passed = top_share <= max_share and distinct >= min_distinct
    detail = f"scored on '{field}'; histogram {labels}"
    if use_position:
        detail += f"; content-label histogram {content} (constant by construction is expected here)"
    return {
        "check": "label_degeneracy",
        "observed": {
            "top_share": round(top_share, 4),
            "distinct": distinct,
            "scored_field": field,
        },
        "threshold": {"max_label_share": max_share, "min_distinct_labels": min_distinct},
        "passed": passed,
        "detail": detail,
    }


def check_length_shortcut(records: List[dict], cfg: dict) -> dict:
    """On a pairwise task, the correct candidate must not be identifiable by
    length. If the ground-truth answer is the longer candidate far more than
    half the time, "always pick the longer response" beats chance and the task
    measures verbosity preference rather than the intended construct (a known
    LLM-judge failure mode). Non-pairwise tasks pass trivially.

    'Longer wins' is read from the winner_chars/loser_chars provenance where
    present (preference), else measured from the displayed candidates.
    """
    positional = [r for r in records if r.get("ground_truth_position") in ("A", "B")]
    if not positional:
        return {"check": "length_shortcut", "observed": None, "threshold": None,
                "passed": True, "detail": "not a pairwise task; length cannot label."}

    max_share = cfg.get("max_longer_wins_share", 0.62)
    longer_wins = seen = 0
    for r in positional:
        sf = r.get("source", {}).get("source_fields", {}) if isinstance(r.get("source"), dict) else {}
        wl = sf.get("winner_is_longer")
        if wl in ("yes", "no"):
            longer_wins += (wl == "yes"); seen += 1
            continue
        # relevance and any pairwise task without the field: compare candidate texts
        cmap = r.get("candidate_map") or {}
        pos = r.get("ground_truth_position")
        gt_text = r.get(f"candidate_relevant") if pos else None
        # fall back to the displayed prompt split
        m = re.search(r"\bA:\s*(.*?)\n\s*B:\s*(.*)$", str(r.get("response_being_judged", "")), re.S)
        if not m:
            continue
        a_txt, b_txt = m.group(1), m.group(2)
        gt_txt = a_txt if pos == "A" else b_txt
        other = b_txt if pos == "A" else a_txt
        if len(gt_txt) == len(other):
            continue
        longer_wins += len(gt_txt) > len(other); seen += 1
    share = (longer_wins / seen) if seen else 0.5
    passed = share <= max_share
    return {
        "check": "length_shortcut",
        "observed": {"longer_wins_share": round(share, 4), "n": seen},
        "threshold": {"max_longer_wins_share": max_share},
        "passed": passed,
        "detail": (f"correct answer is the longer candidate in {share:.1%} of {seen} "
                   f"pairwise items; a length-only baseline scores this. "
                   f"cap {max_share:.0%}."),
    }


def check_ground_truth_consistency(records: List[dict], cfg: dict) -> dict:
    """
    No two records may present identical judged content with different correct
    answers, and no content may appear twice at all.

    A contradiction is unsatisfiable: a judge answering the same displayed text
    the same way is scored right once and wrong once, so the item measures
    nothing. Real instances found in the v2 build before this check existed:

      - SummEval ships byte-identical machine summaries carrying different
        expert coherence ratings (labels 4 and 5 for one identical summary).
      - MT-Bench contains both (model_a=X, model_b=Y) and the swapped row for
        the same question; keyed by ordered pair they became two items whose
        positional ground truth pointed opposite ways on identical text.

    Plain duplication without contradiction is also failed: it inflates the item
    count without adding information, which is the v1 defect in miniature.
    """
    by_content: Dict[str, set] = {}
    counts: Dict[str, int] = {}
    for rec in records:
        content = str(rec.get("response_being_judged"))
        answer = rec.get("ground_truth_position") or rec.get("ground_truth_label")
        by_content.setdefault(content, set()).add(str(answer))
        counts[content] = counts.get(content, 0) + 1

    # Pairwise tasks legitimately emit each content block once per A/B ordering,
    # and the two orderings render different text, so any exact repeat is still
    # a duplicate regardless of task shape.
    contradictions = sorted(c for c, answers in by_content.items() if len(answers) > 1)
    duplicates = sorted(c for c, n in counts.items() if n > 1)

    passed = not contradictions and not duplicates
    detail = "no duplicated or contradictory judged content"
    if contradictions:
        detail = (
            f"{len(contradictions)} judged text(s) carry more than one correct "
            f"answer, e.g. {contradictions[0][:70]!r}"
        )
    elif duplicates:
        detail = (
            f"{len(duplicates)} judged text(s) appear more than once, e.g. "
            f"{duplicates[0][:70]!r}"
        )
    return {
        "check": "ground_truth_consistency",
        "observed": {
            "n_contradictions": len(contradictions),
            "n_duplicated_content": len(duplicates),
        },
        "threshold": {"n_contradictions": 0, "n_duplicated_content": 0},
        "passed": passed,
        "detail": detail,
    }


def check_effective_sample_size(records: List[dict], cfg: dict) -> dict:
    unit = cfg["cluster_unit"]
    key = {"item": "item_id", "prompt_pair": "prompt_pair_id", "row": None}.get(unit, unit)
    if key is None:
        clusters = len(records)
    else:
        cluster_ids = {r.get(key) for r in records}
        # Legacy v1 data has no explicit cluster keys; fall back to the
        # judged text for item-level clustering so v1 cannot dodge the check.
        if cluster_ids == {None} and unit == "item":
            cluster_ids = {r.get("response_being_judged") for r in records}
        cluster_ids.discard(None)
        clusters = len(cluster_ids)
    floor = cfg["min_clusters"]
    return {
        "check": "effective_sample_size",
        "observed": clusters,
        "threshold": floor,
        "passed": clusters >= floor,
        "detail": f"{clusters} clusters at declared unit {unit!r}; floor is "
                  f"{floor}. CIs computed on more rows than clusters must "
                  "resample clusters (src/metrics_v2.py).",
    }


def check_annotation_timing(validation_dir: Path, cfg: dict) -> List[dict]:
    results = []
    floor = cfg["min_seconds_per_decision"]
    files = sorted(validation_dir.glob("*.jsonl")) if validation_dir.exists() else []
    if not files:
        # A declared validation_dir IS the claim that human validation happened.
        # If the records are absent, that claim is unverifiable, and an
        # unverifiable claim must fail the gate rather than pass it silently.
        # (Deleting or moving the records is otherwise a way to turn CI green.)
        return [{
            "check": "annotation_timing",
            "split": None,
            "observed": None,
            "threshold": floor,
            "passed": False,
            "detail": f"config declares validation_dir={validation_dir} but no "
                      "*.jsonl human-validation records were found there. The "
                      "gate cannot verify a validation claim with no evidence; "
                      "remove validation_dir from the config if no human "
                      "validation is being claimed.",
        }]
    for f in files:
        records = load_jsonl(f)
        stamps = sorted(
            datetime.fromisoformat(r["timestamp"])
            for r in records if r.get("timestamp")
        )
        gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        median_gap = statistics.median(gaps) if gaps else None
        if median_gap is None:
            # Records exist but carry no usable timestamps: timing is
            # unverifiable. Same principle as above — fail, do not skip,
            # otherwise stripping the timestamp field turns the check green.
            results.append({
                "check": "annotation_timing",
                "split": f.name,
                "observed": None,
                "threshold": floor,
                "passed": False,
                "detail": f"{len(records)} records in {f.name} but fewer than two "
                          "carry a parseable 'timestamp'; per-decision timing "
                          "cannot be verified.",
            })
            continue
        results.append({
            "check": "annotation_timing",
            "split": f.name,
            "observed": round(median_gap, 3),
            "threshold": floor,
            "passed": median_gap >= floor,
            "detail": f"median inter-decision gap {median_gap!r}s over "
                      f"{len(records)} decisions; floor is {floor}s "
                      "(a human reading and judging two full prompts cannot "
                      "sustain sub-second decisions)",
        })
    return results


def audit(config_path: Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    data_dir = REPO / cfg["dataset_dir"]
    splits = cfg["splits"]

    all_results: List[dict] = []
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            all_results.append({
                "check": "split_present",
                "split": split,
                "observed": None,
                "threshold": None,
                "passed": False,
                "detail": f"declared split file missing: {path}",
            })
            continue
        records = load_jsonl(path)
        split_cfg = {**cfg["thresholds"], **cfg.get("split_overrides", {}).get(split, {})}
        for result in (
            check_unique_items(records, split_cfg),
            check_duplicate_rows(records, split_cfg),
            check_provenance(records, split_cfg),
            check_label_degeneracy(records, split_cfg),
            check_effective_sample_size(records, split_cfg),
            check_ground_truth_consistency(records, split_cfg),
            check_length_shortcut(records, split_cfg),
        ):
            result["split"] = split
            all_results.append(result)

    if "validation_dir" in cfg:
        all_results.extend(
            check_annotation_timing(REPO / cfg["validation_dir"], cfg["thresholds"])
        )

    failed = [r for r in all_results if not r["passed"]]
    return {"config": str(config_path), "results": all_results,
            "n_checks": len(all_results), "n_failed": len(failed),
            "passed": not failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="JudgeSense data-audit gate")
    parser.add_argument("--config", default="data/audit_config.json")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write the full report as JSON")
    args = parser.parse_args()

    report = audit(Path(args.config))

    width = 100
    print("=" * width)
    print("JudgeSense data-audit gate")
    print("=" * width)
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        split = r.get("split") or "-"
        print(f"[{status}] {r['check']:<22} split={split:<18} "
              f"observed={r['observed']!r} threshold={r['threshold']!r}")
        if not r["passed"]:
            print(f"       {r['detail']}")
    print("-" * width)
    print(f"{report['n_failed']}/{report['n_checks']} checks failed")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.json_out}")
    if not report["passed"]:
        print("DATA AUDIT FAILED — this dataset must not be released or "
              "used for reported results until every check passes.")
        sys.exit(1)
    print("Data audit passed.")


if __name__ == "__main__":
    main()
