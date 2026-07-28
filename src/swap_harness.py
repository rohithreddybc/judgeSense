"""
JudgeSense v2 position-bias swap harness.

Design fix for the v1 always-A degeneracy (12 of 13 judges picked position A
almost universally on pairwise tasks, making raw JSS trivially perfect):
every pairwise (item, template pair) is presented in BOTH candidate
orderings, and scoring happens at the CONTENT level, not the position level.

Given the four presentations of one prompt pair —
    template A x {original, swapped}, template B x {original, swapped} —
a judge's positional answers are mapped through each presentation's
`candidate_map` to the underlying candidate. A content-level decision for a
template exists only if the judge picks the SAME underlying candidate in
both orderings; otherwise the decision is POSITION_INCONSISTENT and scores
as disagreement. Position-bias-corrected JSS is then the agreement between
the two templates' content-level decisions.

Consequence (unit-tested): a judge that always answers "A" is position-
inconsistent on every pair, so its corrected JSS is 0 — always-A behavior
cannot inflate the corrected score. A faithful judge is unaffected.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from collections import Counter

POSITION_INCONSISTENT = "POSITION_INCONSISTENT"
_VALID_POSITIONS = ("A", "B")


def content_decision(
    decision_original: Optional[str],
    decision_swapped: Optional[str],
    candidate_map_original: Dict[str, str],
    candidate_map_swapped: Dict[str, str],
) -> str:
    """
    Map two positional decisions (one per ordering) to a content-level
    decision: the underlying candidate id if both orderings agree on it,
    else POSITION_INCONSISTENT (which callers must score as disagreement).
    Unparseable positional answers are POSITION_INCONSISTENT as well.
    """
    if decision_original not in _VALID_POSITIONS or decision_swapped not in _VALID_POSITIONS:
        return POSITION_INCONSISTENT
    chosen_original = candidate_map_original[decision_original]
    chosen_swapped = candidate_map_swapped[decision_swapped]
    if chosen_original == chosen_swapped:
        return chosen_original
    return POSITION_INCONSISTENT


def collapse_presentations(judged: Sequence[dict]) -> List[dict]:
    """
    Collapse per-presentation judged records into one content-level record
    per prompt pair, suitable for src/metrics_v2.py.

    Input records need: prompt_pair_id, item_id, ab_order, candidate_map,
    decision_a, decision_b (positional decisions for templates A and B of
    that presentation). Exactly the two orderings per prompt pair are
    required; anything else is an error, not a silent skip.
    """
    by_pair: Dict[str, Dict[str, dict]] = {}
    for rec in judged:
        by_pair.setdefault(rec["prompt_pair_id"], {})[rec["ab_order"]] = rec

    collapsed: List[dict] = []
    for pair_id, orders in sorted(by_pair.items()):
        missing = {"original", "swapped"} - set(orders)
        if missing:
            raise ValueError(
                f"prompt_pair {pair_id} is missing ordering(s) {sorted(missing)}; "
                "the swap design requires both presentations of every pair."
            )
        orig, swap = orders["original"], orders["swapped"]
        collapsed.append({
            "prompt_pair_id": pair_id,
            "item_id": orig["item_id"],
            "decision_a": content_decision(
                orig["decision_a"], swap["decision_a"],
                orig["candidate_map"], swap["candidate_map"],
            ),
            "decision_b": content_decision(
                orig["decision_b"], swap["decision_b"],
                orig["candidate_map"], swap["candidate_map"],
            ),
        })
    return collapsed


def position_bias_corrected_jss(judged: Sequence[dict]) -> float:
    """
    Agreement between the two templates' content-level decisions, with
    POSITION_INCONSISTENT scored as disagreement. This is the pairwise-task
    headline metric in v2.
    """
    collapsed = collapse_presentations(judged)
    if not collapsed:
        raise ValueError("No prompt pairs to score.")
    matches = sum(
        1 for rec in collapsed
        if rec["decision_a"] == rec["decision_b"]
        and rec["decision_a"] != POSITION_INCONSISTENT
    )
    return matches / len(collapsed)


def position_bias_rate(judged: Sequence[dict]) -> dict:
    """
    Per-judge position diagnostics:
    - first_position_rate: fraction of all positional decisions that are "A".
    - inconsistency_rate: fraction of (prompt pair, template) slots whose two
      orderings select different underlying candidates (or were unparseable).
    An always-A judge shows first_position_rate 1.0 and inconsistency 1.0.
    """
    positional: Counter = Counter()
    for rec in judged:
        positional[rec["decision_a"]] += 1
        positional[rec["decision_b"]] += 1
    total = sum(positional.values())

    collapsed = collapse_presentations(judged)
    slots = [rec[k] for rec in collapsed for k in ("decision_a", "decision_b")]
    inconsistent = sum(1 for s in slots if s == POSITION_INCONSISTENT)

    return {
        "first_position_rate": positional.get("A", 0) / total if total else 0.0,
        "positional_histogram": dict(positional),
        "inconsistency_rate": inconsistent / len(slots) if slots else 0.0,
        "n_prompt_pairs": len(collapsed),
    }
