"""
JudgeSense v2 dataset builder — real items, explicit clustering keys, and a
built-in position-bias swap design for pairwise tasks.

Usage:
    python src/dataset_builder_v2.py --output data/v2/ [--items-per-task 250]

Every record's judged text and ground-truth label come from a real source
record loaded by src/data_sources.py; this module only supplies the judge
prompt templates and the pairing/ordering structure. If any source is
unavailable the build fails loudly (DataSourceUnavailableError) — there is
no fallback data path in this file, by design.

Schema notes:
- `item_id` and `prompt_pair_id` are the clustering keys consumed by
  src/metrics_v2.py; `prompt_pair_id` identifies (item, template combination).
- Pairwise tasks (relevance, preference) emit TWO records per prompt pair —
  `ab_order` in {"original", "swapped"} — with `candidate_map` recording
  which underlying candidate sits at each displayed position. See
  src/swap_harness.py for how these are scored.
- `semantic_equivalence_score` is intentionally absent: no verification pass
  has run on the v2 templates yet, and the field is never hardcoded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    from .data_sources import TASK_LOADERS, SourceItem, DEFAULT_ITEMS_PER_TASK
except ImportError:  # `python src/dataset_builder_v2.py` invocation style
    from data_sources import TASK_LOADERS, SourceItem, DEFAULT_ITEMS_PER_TASK  # type: ignore

BUILDER_VERSION = "2.0.0"

# Five semantically equivalent phrasings per task. None inverts polarity or
# changes the label space (the v1 factuality T4 artifact class); a unit test
# asserts the label-direction words match across templates.
TEMPLATES: Dict[str, List[str]] = {
    "factuality": [
        "Is this statement factually correct? Answer YES or NO only.\n\n{text}",
        "Fact-check the statement below. Reply YES (correct) or NO (incorrect).\n\n{text}",
        "Is the following statement accurate? Answer with one word: YES or NO.\n\n{text}",
        "Evaluate the factual accuracy of this statement. Respond YES if it is correct, NO if it is not.\n\n{text}",
        "Judge whether the statement is factually correct. Your entire answer must be YES or NO.\n\n{text}",
    ],
    "coherence": [
        "Rate the coherence of this text from 1 (incoherent) to 5 (very coherent). Reply with one number only.\n\n{text}",
        "How coherent is the following text? Score it 1-5 (1 = incoherent, 5 = very coherent). Number only.\n\n{text}",
        "Assign a coherence rating from 1 to 5, where 1 means incoherent and 5 means very coherent. Answer with a single digit.\n\n{text}",
        "On a 1-5 coherence scale (1 = incoherent, 5 = very coherent), what score does this text deserve? Respond with the number only.\n\n{text}",
        "Score this text's coherence: 1 (incoherent) through 5 (very coherent). Give only the number.\n\n{text}",
    ],
    "relevance": [
        "Which passage is more relevant to the query \"{query}\"? Answer A or B only.\nA: {first}\nB: {second}",
        "For the query \"{query}\", which passage is more relevant? Reply with A or B.\nA: {first}\nB: {second}",
        "Query: \"{query}\". Choose the more relevant passage. Answer A or B.\nA: {first}\nB: {second}",
        "Considering the query \"{query}\", decide which passage is more relevant. Respond A or B only.\nA: {first}\nB: {second}",
        "Given the query \"{query}\", pick the passage with higher relevance. Your answer must be A or B.\nA: {first}\nB: {second}",
    ],
    "preference": [
        "Which response to \"{query}\" is better? Answer A or B only.\nA: {first}\nB: {second}",
        "For the question \"{query}\", which response is of higher quality? Reply with A or B.\nA: {first}\nB: {second}",
        "Question: \"{query}\". Choose the better response. Answer A or B.\nA: {first}\nB: {second}",
        "Considering the question \"{query}\", decide which response is better. Respond A or B only.\nA: {first}\nB: {second}",
        "Given the question \"{query}\", pick the stronger response. Your answer must be A or B.\nA: {first}\nB: {second}",
    ],
}

PAIRWISE_TASKS = {"relevance", "preference"}

# Content-level candidate keys per pairwise task, in "original" display order.
_CANDIDATE_KEYS = {
    "relevance": ("candidate_relevant", "candidate_nonrelevant"),
    "preference": ("candidate_1", "candidate_2"),
}


def _template_combinations(n_templates: int = 5) -> List[tuple]:
    return list(itertools.combinations(range(n_templates), 2))


def _assign_template_combos(items: List[SourceItem], combos: List[tuple],
                            seed: int) -> List[tuple]:
    """One template pair per item, assigned by a seeded permutation that is
    STRATIFIED on the ground-truth label.

    The previous rule was `combos[idx % len(combos)]`. A deterministic
    10-cycle over the template pairs is only safe if the item order carries no
    label periodicity, and the factuality loader emitted a strict
    accurate/inaccurate 2-cycle. The two cycles ran in lockstep, so on the
    shipped v2 build all 10 template pairs mapped to exactly one label
    (250/250) and the pooled per-template balance was T1 50/50, T2 75/25,
    T3 50/50, T4 25/75, T5 50/50 — a constant-YES judge scored 0.75 on T2 and
    0.25 on T4 without reading a single item, which confounds every
    per-template claim.

    Shuffling the loader's emission order alone would leave the balance correct
    only in expectation (±1 sd is roughly 5 points on 100 pooled items per
    template). Dealing each label group round-robin across a seeded rotation of
    the combos instead makes it exact up to integer division: with 125 items of
    each label and 10 combos, every combo receives 12 or 13 of each, so each
    template's pooled balance lands within one item of 50/50.

    For tasks whose content label is constant by construction (relevance, and
    preference where it is candidate_1/candidate_2) there is one stratum or two
    balanced ones, and this degenerates to a seeded round-robin.
    """
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_label[item.ground_truth_label].append(idx)

    assignment: List[tuple] = [None] * len(items)  # type: ignore[list-item]
    # Each stratum walks the same rotation of combos but starts where the
    # previous stratum stopped, so the residue of an uneven division is spread
    # over different combos instead of piling onto the first few.
    rotation = list(combos)
    rng.shuffle(rotation)
    cursor = 0
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        for position, item_idx in enumerate(group):
            assignment[item_idx] = rotation[(cursor + position) % len(rotation)]
        cursor = (cursor + len(group)) % len(rotation)
    return assignment


def _positional_label(content_label: str, candidate_map: Dict[str, str]) -> str:
    """Displayed position (A/B) of the ground-truth candidate."""
    for position, key in candidate_map.items():
        if key == content_label:
            return position
    raise ValueError(f"ground truth '{content_label}' not in candidate_map {candidate_map}")


def build_task_records(task: str, items: List[SourceItem],
                       seed: int = 42) -> List[dict]:
    """Build dataset records for one task from real source items."""
    templates = TEMPLATES[task]
    combos = _template_combinations(len(templates))
    # Label-stratified, seeded assignment. See _assign_template_combos: the
    # previous `idx % len(combos)` rotation was a perfect answer key on
    # factuality because the loader emitted labels in a 2-cycle.
    assignment = _assign_template_combos(items, combos, seed)
    records: List[dict] = []

    for idx, item in enumerate(items):
        ti, tj = assignment[idx]
        prompt_pair_id = f"{item.item_id}#T{ti + 1}-T{tj + 1}"
        base = {
            "task_type": task,
            "item_id": item.item_id,
            "prompt_pair_id": prompt_pair_id,
            "template_a": f"T{ti + 1}",
            "template_b": f"T{tj + 1}",
            "source_benchmark": item.source.source_dataset,
            "source": item.source.to_dict(),
            "builder_version": BUILDER_VERSION,
        }

        if task not in PAIRWISE_TASKS:
            records.append({
                "pair_id": f"{task[:4]}_v2_{idx + 1:04d}",
                **base,
                "prompt_a": templates[ti].format(text=item.text),
                "prompt_b": templates[tj].format(text=item.text),
                "response_being_judged": item.text,
                "ground_truth_label": item.ground_truth_label,
                **({"ground_truth_raw": item.extra["coherence_raw"]}
                   if "coherence_raw" in item.extra else {}),
            })
            continue

        key1, key2 = _CANDIDATE_KEYS[task]
        cand1, cand2 = item.extra[key1], item.extra[key2]
        orderings = {
            "original": {"A": key1, "B": key2},
            "swapped": {"A": key2, "B": key1},
        }
        texts = {key1: cand1, key2: cand2}
        for order_name, candidate_map in orderings.items():
            first, second = texts[candidate_map["A"]], texts[candidate_map["B"]]
            fmt = dict(query=item.text, first=first, second=second)
            records.append({
                "pair_id": f"{task[:4]}_v2_{idx + 1:04d}_{order_name}",
                **base,
                "ab_order": order_name,
                "candidate_map": candidate_map,
                "prompt_a": templates[ti].format(**fmt),
                "prompt_b": templates[tj].format(**fmt),
                "response_being_judged": f"A: {first} | B: {second}",
                "ground_truth_label": item.ground_truth_label,
                "ground_truth_position": _positional_label(item.ground_truth_label, candidate_map),
            })

    return records


def build_all(output_dir: Path, items_per_task: int = DEFAULT_ITEMS_PER_TASK,
              seed: int = 42, tasks: List[str] | None = None) -> dict:
    """
    Build the full v2 dataset. Raises DataSourceUnavailableError if any
    source cannot be loaded — partial or placeholder builds are not written.
    """
    tasks = tasks or list(TASK_LOADERS.keys())
    output_dir = Path(output_dir)

    # Load every source first so a late failure cannot leave a partial build.
    loaded: Dict[str, List[SourceItem]] = {}
    for task in tasks:
        print(f"Loading real source items for task={task} ...")
        loaded[task] = TASK_LOADERS[task](n_items=items_per_task, seed=seed)
        print(f"  {len(loaded[task])} items loaded from "
              f"{loaded[task][0].source.source_dataset}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "builder_version": BUILDER_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "items_per_task": items_per_task,
        "tasks": {},
    }
    for task in tasks:
        records = build_task_records(task, loaded[task], seed=seed)
        path = output_dir / f"{task}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Distinct UPSTREAM source records behind the items: an item_id is unique
        # per (source record, answer field / doc pair), so several items can share
        # one upstream question, article, or topic. Disclosing this makes the
        # effective source diversity explicit rather than implying 250 independent
        # sources per task (reviewer qkzU W3/Q2).
        n_source = len({
            r["source"]["source_record_id"].split("#")[0].split(".")[0]
            for r in records
        })
        manifest["tasks"][task] = {
            "file": path.name,
            "rows": len(records),
            "unique_items": len({r["item_id"] for r in records}),
            "unique_prompt_pairs": len({r["prompt_pair_id"] for r in records}),
            "distinct_source_records": n_source,
            "source_dataset": loaded[task][0].source.source_dataset,
        }
        print(f"  wrote {len(records)} records -> {path}")

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the JudgeSense v2 dataset from real sources")
    parser.add_argument("--output", default="data/v2/", help="Output directory")
    parser.add_argument("--items-per-task", type=int, default=DEFAULT_ITEMS_PER_TASK)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_all(Path(args.output), items_per_task=args.items_per_task, seed=args.seed)


if __name__ == "__main__":
    main()
