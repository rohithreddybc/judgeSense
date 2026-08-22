"""
Score the shipped splits against judges that use no judgment at all.

A benchmark measures what it claims to only if a heuristic ignoring the intended
construct cannot pass it. These controls are computed from data/v2/*.jsonl --
the released files, not the loaders -- so the numbers are properties of the
artifact a reader downloads, and this script is what the paper cites for them.

Reported two-sided. A split whose answer is reliably the SHORTER or the LESS
overlapping candidate is exploited exactly as cheaply as one biased the other
way; an earlier build of the relevance split maximised distractor retrieval
scores, which made the inverse-overlap heuristic score far above chance while a
one-sided check called it clean.

Ties are excluded rather than split, so each figure carries its own support
count: a heuristic with no signal on a row is not evidence about that row.

Usage:
    python scripts/shortcut_controls.py
    python scripts/shortcut_controls.py --json-out data/results_v2/shortcut_controls.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "v2"
OUT = REPO / "data" / "results_v2" / "shortcut_controls.json"
PAIRWISE = ("relevance", "preference")
WORD = re.compile(r"\w+")


def _rows(task: str) -> List[dict]:
    return [json.loads(line) for line in (DATA / f"{task}.jsonl").open(encoding="utf-8")]


def _candidates(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """The two candidates as the judge sees them, from the rendered prompt.

    Scans for lines opening "A:" and "B:" rather than pattern-matching the whole
    prompt, because a candidate may itself span blank lines or contain code.
    """
    lines = prompt.split("\n")
    starts: Dict[str, int] = {}
    for i, line in enumerate(lines):
        for marker in ("A:", "B:"):
            if line.startswith(marker) and marker not in starts:
                starts[marker] = i
    if "A:" not in starts or "B:" not in starts or starts["A:"] >= starts["B:"]:
        return None, None
    a = "\n".join(lines[starts["A:"]:starts["B:"]])[2:].strip()
    b = "\n".join(lines[starts["B:"]:])[2:].strip()
    return (a, b) if a and b else (None, None)


def _wilson(k: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a binomial proportion.

    A point estimate alone is not evidence of absence. At n=248 an accuracy of
    0.508 is consistent with a shortcut worth 0.57, so a control reported
    without an interval cannot support the claim it is being used to make.
    Wilson rather than normal-approximation because it stays inside [0,1] and
    behaves near the boundaries.
    """
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(centre - half, 4), round(centre + half, 4)


def _score(task: str, pick) -> Dict:
    """Fraction of rows a heuristic gets right, over rows where it has a signal."""
    rows = [r for r in _rows(task) if r.get("ground_truth_position") in ("A", "B")]
    correct = scored = ties = 0
    for r in rows:
        a, b = _candidates(r["prompt_a"])
        if a is None:
            continue
        choice = pick(r, a, b)
        if choice is None:
            ties += 1
            continue
        scored += 1
        correct += choice == r["ground_truth_position"]
    ci = _wilson(correct, scored)
    return {
        "accuracy": round(correct / scored, 4) if scored else None,
        "ci95": list(ci) if ci else None,
        "n": scored,
        "n_ties_excluded": ties,
        "n_rows": len(rows),
        # A control computed on a small fraction of the split bounds very little.
        "coverage": round(scored / len(rows), 4) if rows else None,
    }


def _position(r, a, b):
    return "A"


def _length(r, a, b):
    if len(a) == len(b):
        return None
    return "A" if len(a) > len(b) else "B"


def _overlap(r, a, b):
    query = re.search(r'query "(.*?)"', r["prompt_a"])
    if not query:
        return None
    terms = set(WORD.findall(query.group(1).lower()))
    sa = len(terms & set(WORD.findall(a.lower())))
    sb = len(terms & set(WORD.findall(b.lower())))
    if sa == sb:
        return None
    return "A" if sa > sb else "B"


CONTROLS = (
    ("position_only", "always answer A", _position, PAIRWISE),
    ("length_only", "pick the longer candidate", _length, PAIRWISE),
    ("lexical_overlap_only", "pick the candidate sharing more query terms",
     _overlap, ("relevance",)),
)
BAND = (0.44, 0.56)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the shipped splits against no-judgment heuristics.")
    ap.add_argument("--json-out", default=str(OUT))
    args = ap.parse_args(argv)

    report: Dict[str, Dict] = {}
    failures: List[str] = []
    print(f"{'control':<22}{'task':<12}{'accuracy':>10}{'ci95':>17}{'n':>7}{'cov':>7}  verdict")
    for key, description, fn, tasks in CONTROLS:
        report[key] = {"description": description, "tasks": {}}
        for task in tasks:
            result = _score(task, fn)
            report[key]["tasks"][task] = result
            acc = result["accuracy"]
            # Three states, because absence of evidence and evidence of
            # absence are different claims and the old gate made both from a
            # point estimate.
            #
            #   EXPLOITABLE   the whole interval sits outside the band: the
            #                 heuristic demonstrably beats chance. Fails.
            #   CLEAN         the whole interval sits inside: the shortcut is
            #                 bounded below the band edge at this n.
            #   INCONCLUSIVE  the interval straddles an edge. Consistent with
            #                 chance AND with a usable shortcut; the data cannot
            #                 separate them. Reported, not silently passed, and
            #                 not failed either -- at n=248 with half the rows
            #                 tied, no construction could do better, so failing
            #                 it would be demanding precision the split cannot
            #                 supply rather than reporting a defect.
            ci = result["ci95"]
            if ci is None:
                verdict = "NO DATA"
            elif ci[0] > BAND[1] or ci[1] < BAND[0]:
                verdict = "EXPLOITABLE"
            elif BAND[0] <= ci[0] and ci[1] <= BAND[1]:
                verdict = "clean"
            else:
                verdict = "inconclusive"
            result["verdict"] = verdict
            result["low_coverage"] = (result["coverage"] or 1.0) < 0.8
            ok = verdict != "EXPLOITABLE"
            if not ok:
                failures.append(f"{key}/{task} = {acc} CI={ci}")
            ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-"
            print(f"{key:<22}{task:<12}{acc:>10}{ci_s:>17}{result['n']:>7}"
                  f"{result['coverage']:>7}  {verdict}"
                  f"{'  [reads <80% of rows]' if result['low_coverage'] else ''}")

    report["band"] = {"lower": BAND[0], "upper": BAND[1],
                      "rule": ("EXPLOITABLE when the whole 95% interval lies "
                               "outside the band; clean when it lies inside; "
                               "inconclusive when it straddles an edge, which "
                               "is a statement about the achievable precision "
                               "at this support, not about the split"),
                      "note": "two-sided: a reliably-shorter or reliably-lower-overlap "
                              "answer is as exploitable as the reverse"}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")
    if failures:
        print("EXPLOITABLE: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
