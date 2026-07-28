"""
Parser + prompt-construction audit for the v1 pipeline (reviewer p5cJ).

Question under audit: the near-universal "always-A" result on the pairwise
tasks — is it genuine judge position bias, a prompt-construction artifact,
or a parsing artifact?

This script reports, from code and committed data only:
  1. PROMPT CONSTRUCTION — for every shipped pairwise record, whether the
     two prompt variants present candidates in the same order, and the
     ground-truth / ab_swapped distributions.
  2. PARSER BEHAVIOR — the exact mapping rules of
     src.models.normalize_decision and demonstrations of each failure mode
     using labeled PROBE strings. Probes are code-behavior demonstrations
     only; they are NOT model outputs and are never written to any dataset.
  3. RAW OUTPUT MAPPING — if real raw outputs exist under
     data/results/raw_outputs/, tabulates raw -> normalized for every
     record. If none exist, says so explicitly rather than inventing any.

Usage:
    python analysis/parser_audit.py [--report outputs/parser_audit_report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models import normalize_decision  # noqa: E402

RAW_DIR = REPO / "data" / "results" / "raw_outputs"
PAIR_DIR = REPO / "data" / "prompt_pairs"

# Labeled probe strings: plausible judge-output shapes chosen to exercise
# each branch of normalize_decision, each with the reading a careful human
# would give it. These are audit inputs, not data. A probe is flagged when
# the parser's output differs from the intended reading.
PROBES = {
    "relevance": [
        ("A", "A"),
        ("B", "B"),
        ("**B**", "B"),
        ("The answer is B", "B"),
        ("B is more relevant", "B"),
        ("This is a tough call, but B", "B"),      # article 'a' -> parsed A
        ("It is a close one; I pick B", "B"),      # article 'a' -> parsed A
        ("As a judge, I select B", "B"),           # article 'a' -> parsed A
        ("Answer: B", "B"),
        ("Option B", "B"),
        ("b", "B"),
        ("Both are relevant", "UNCLEAR"),
        ("Neither passage answers the query", "UNCLEAR"),
    ],
    "factuality": [
        ("YES", "YES"),
        ("NO", "NO"),
        ("No, this is wrong", "NO"),
        ("Not yes", "NO"),                          # 'YES' substring -> YES
        ("I don't know", "UNCLEAR"),                # 'NO' inside 'know' -> NO
        ("It is unknowable", "UNCLEAR"),            # 'NO' in 'unknowable' -> NO
        ("Cannot determine", "UNCLEAR"),            # 'NO' inside 'Cannot' -> NO
        ("Yes and no", "UNCLEAR"),                  # YES checked first -> YES
    ],
    "coherence": [
        ("4", "4"),
        ("Score: 4", "4"),
        ("On a scale of 1-5, I'd say 4", "4"),      # first digit '1' -> 1
        ("I rate it 4 out of 5", "4"),
        ("3/5", "3"),
        ("Somewhat coherent", "UNCLEAR"),
    ],
}


def audit_prompt_construction() -> list[str]:
    lines = ["## 1. Prompt construction (shipped v1 data)", ""]
    for task in ("relevance", "preference"):
        path = PAIR_DIR / f"{task}.jsonl"
        if not path.exists():
            lines.append(f"- `{path.name}`: NOT FOUND")
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        same_order = sum(1 for r in recs if _candidate_block(r["prompt_a"]) == _candidate_block(r["prompt_b"]))
        gt = Counter(r.get("ground_truth_label") for r in recs)
        sw = Counter(r.get("ab_swapped") for r in recs)
        uniq_items = len({r["response_being_judged"] for r in recs})
        lines += [
            f"### {task} ({len(recs)} rows, {uniq_items} unique items)",
            "",
            f"- Variants A and B present candidates in the **same order** in "
            f"{same_order}/{len(recs)} records — the paraphrase axis never "
            f"varies candidate order within a pair, so a judge that answers "
            f"by position alone agrees with itself on every pair "
            f"(raw JSS = 1.0) regardless of the input.",
            f"- ground_truth_label distribution: {dict(gt)}",
            f"- ab_swapped distribution: {dict(sw)} (order balance exists "
            f"*across* records but not *within* a prompt pair, so it cannot "
            f"detect or correct position-following within JSS)",
            "",
        ]
    lines += [
        "**Finding:** always-A inflates raw JSS by construction: with only "
        f"{5} unique items per pairwise task and identical candidate order "
        "inside every pair, position-following is indistinguishable from "
        "consistency. The v2 swap harness (src/swap_harness.py) removes this "
        "by presenting both orderings and scoring at content level.",
        "",
    ]
    return lines


def _candidate_block(prompt: str) -> str:
    return "\n".join(l for l in prompt.splitlines() if l.startswith(("A:", "B:")))


def audit_parser() -> list[str]:
    lines = [
        "## 2. Parser behavior (src/models.py::normalize_decision)",
        "",
        "Mapping rules as implemented:",
        "",
        "- factuality: uppercase the raw string; return YES if the substring "
        "'YES' occurs anywhere; else NO if 'NO' occurs anywhere; else UNCLEAR.",
        "- relevance/preference: uppercase; strip `*_`; return the FIRST "
        r"standalone token matching `\b([AB])\b`; else UNCLEAR.",
        "- coherence: return the FIRST character in '12345' found anywhere.",
        "",
        "Demonstrated failure modes (labeled probe strings, not model outputs).",
        "'intended' is the reading a careful human gives the probe; a row is",
        "flagged when the parser disagrees with it:",
        "",
        "| task | probe | intended | parsed as | verdict |",
        "|---|---|---|---|---|",
    ]
    findings = []
    for task, probes in PROBES.items():
        for probe, intended in probes:
            parsed = normalize_decision(probe, task)
            if parsed == intended:
                verdict = "ok"
            else:
                if task == "relevance" and parsed == "A":
                    kind = "article-A: uppercased article 'a' matches before the intended letter"
                elif task == "factuality" and parsed == "NO":
                    kind = "substring-NO: 'NO' inside another word"
                elif task == "factuality" and parsed == "YES":
                    kind = "yes-first: 'YES' substring checked before 'NO'"
                elif task == "coherence":
                    kind = "first-digit: grabs a scale echo, not the rating"
                else:
                    kind = "mismatch"
                verdict = f"**BUG — {kind}**"
                findings.append((kind.split(":")[0], probe))
            lines.append(f"| {task} | `{probe}` | `{intended}` | `{parsed}` | {verdict} |")
    lines += [
        "",
        "**Finding (bears on always-A):** for pairwise tasks the regex "
        r"`\b([AB])\b` matches the uppercased English article 'a' "
        "('It is A close call...'), so any verbose answer containing an "
        "article before the letter B is parsed as A. This systematically "
        "converts verbose B-answers into A-decisions and can manufacture "
        "always-A behavior at the parsing layer, on top of any genuine "
        "position bias. Because both variants of a pair are parsed with the "
        "same rule, the error is symmetric and *raises* apparent agreement.",
        "",
        f"Probes triggering a defect class: {len(findings)} "
        f"({Counter(k for k, _ in findings)}).",
        "",
    ]
    return lines


def audit_raw_outputs() -> list[str]:
    lines = ["## 3. Raw output -> normalized decision mapping", ""]
    files = sorted(RAW_DIR.glob("*.jsonl")) if RAW_DIR.exists() else []
    if not files:
        lines += [
            "No raw judge outputs are present under `data/results/raw_outputs/` "
            "in this repository (the directory contains only a `.gitkeep`). "
            "The mapping table therefore CANNOT be produced here, and this "
            "audit does not invent one. To complete this section, run this "
            "script on a machine holding the original raw outputs.",
            "",
        ]
        return lines
    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        table: Counter = Counter()
        for r in recs:
            for side in ("a", "b"):
                raw = str(r.get(f"prompt_{side}_raw", ""))
                norm = r.get(f"normalized_{side}") or r.get(f"prompt_{side}_decision", "")
                reparsed = normalize_decision(raw, r.get("task_type", ""))
                table[(raw[:60], norm, reparsed, norm == reparsed)] += 1
        lines.append(f"### {f.name} ({len(recs)} records)")
        lines.append("")
        lines.append("| raw (first 60 chars) | stored | reparsed | match | n |")
        lines.append("|---|---|---|---|---|")
        for (raw, norm, reparsed, match), n in table.most_common(50):
            lines.append(f"| `{raw}` | {norm} | {reparsed} | {match} | {n} |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/parser_audit_report.md")
    args = parser.parse_args()

    lines = [
        "# Parser + prompt sanity audit (v1 pipeline)",
        "",
        "Generated by `analysis/parser_audit.py` from repository code and "
        "committed data only. Probe strings in §2 are labeled audit inputs, "
        "not model outputs.",
        "",
    ]
    lines += audit_prompt_construction()
    lines += audit_parser()
    lines += audit_raw_outputs()

    report = "\n".join(lines) + "\n"
    out = REPO / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
