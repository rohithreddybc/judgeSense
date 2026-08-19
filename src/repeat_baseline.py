"""
Same-prompt repeat baseline — protocol contract.

The problem (docs/V2_1_STRUCTURAL_AXIS.md §7, carried into the main
instruction axis): JudgeSense reports JSS as if disagreement between two
DIFFERENT phrasings were entirely attributable to prompt wording. No published
number checks that against the ceiling of how often a judge disagrees with
itself on the BYTE-IDENTICAL prompt. Judges run at temperature 0 (0.01 on some
HF endpoints), so self-disagreement should be small — several judges in this
suite are reasoning-tuned with long sampled chains, where "should be" is not
"is". Without the ceiling, every JSS number conflates paraphrase sensitivity
with ordinary decoding variance and the two cannot be separated after the
fact.

This module defines the call-level contract a runner emits and the pairing
step that turns it into the decision-record shape the rest of
`src/metrics_v2.py` already understands. It performs no API calls itself.

── Call-record contract (what a runner emits) ───────────────────────────────

For every (judge, item), the protocol issues the S0/baseline prompt TWICE.
Each call is logged as one dict:

    {
        "judge": "gpt-4o",             # judge registry name
        "item_id": "fact_0001",        # matches the item's canonical id
        "task": "factuality",          # optional, for bookkeeping
        "arm": "S0",                   # REPEAT_ARM; the baseline prompt only
        "repeat_index": 1,             # 1 or 2 — which of the two S0 calls
        "decision": "YES",             # parsed decision, or None/UNCLEAR
    }

`repeat_index` is the field that marks a call as part of the repeat baseline.
Ordinary single-shot arms (instruction paraphrases P1/P2, structural variants
S1-S5) do not set it (absent or None) and `build_repeat_pairs` ignores them —
this module only ever builds the S0-vs-S0 pair.

── Decision-record contract (what the metrics consume) ──────────────────────

`build_repeat_pairs` collapses the two calls for an item into one repeat-pair
record:

    {
        "decision_a": <repeat_index 1 decision>,
        "decision_b": <repeat_index 2 decision>,
        "item_id": "fact_0001",
        "repeat_pair_id": "fact_0001",   # == item_id: exactly one repeat
                                          # pair exists per item
        "arm_a": "S0",
        "arm_b": "S0",
    }

This is consumable directly by `jss`, `chance_corrected_jss`,
`repeat_baseline_jss`, and `cluster_bootstrap_ci(..., cluster_unit="item")`
in `src/metrics_v2.py` — no new cluster key was needed since `item_id` is
already the key `cluster_bootstrap_ci` looks for under `cluster_unit="item"`.
For the paired JSS-vs-JSS_rep delta with a correlation-aware CI, see
`jss_repeat_delta` in `src/metrics_v2.py`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .metrics_v2 import (  # noqa: F401  (re-exported for convenience)
    CLUSTER_UNITS,
    chance_corrected_jss,
    cluster_bootstrap_ci,
    jss,
    jss_repeat_delta,
    repeat_baseline_jss,
)

# The repeat baseline is defined for the S0/baseline arm only — never for a
# paraphrase (P1/P2) or structural (S1-S5) variant. Repeating a variant would
# answer a different question ("is THIS wording noisy") rather than the one
# the ceiling exists to answer ("how noisy is the judge before wording is
# varied at all").
REPEAT_ARM = "S0"


class RepeatBaselineError(ValueError):
    """Raised when call-level repeat records violate the S0-repeat contract."""


def build_repeat_pairs(call_records: Sequence[dict]) -> List[dict]:
    """
    Pair up S0 repeat calls into repeat-pair decision records.

    `call_records` is the full per-call log a runner emits (all arms, all
    repeat_index values, including None for non-repeat calls). Only records
    with `repeat_index` set are considered; everything else is ignored so
    this function can be handed a runner's raw call log directly rather than
    a pre-filtered subset.

    Raises `RepeatBaselineError` if:
      - a `repeat_index` is set on a call whose `arm` is not `REPEAT_ARM`
        ("S0") — the repeat baseline is defined only for the baseline arm;
      - a `repeat_index` is outside {1, 2};
      - the same `repeat_index` appears twice for one item (duplicate call);
      - an item has only one of the two repeat calls (incomplete pair) — a
        JSS_rep number computed from a partial pair would silently understate
        the sample it claims.

    Returns one decision record per item with `decision_a`/`decision_b` (call
    1 / call 2), `item_id`, `repeat_pair_id` (== item_id), and `arm_a`/`arm_b`
    (both `REPEAT_ARM`). See the module docstring for the full shape.
    """
    by_item: Dict[str, Dict[int, dict]] = {}
    for rec in call_records:
        idx = rec.get("repeat_index")
        if idx is None:
            continue
        arm = rec.get("arm", REPEAT_ARM)
        if arm != REPEAT_ARM:
            raise RepeatBaselineError(
                f"repeat_index={idx!r} set on a non-{REPEAT_ARM} call "
                f"(arm={arm!r}, item_id={rec.get('item_id')!r}); the repeat "
                f"baseline is defined only for the {REPEAT_ARM}/baseline arm."
            )
        if idx not in (1, 2):
            raise RepeatBaselineError(
                f"repeat_index must be 1 or 2, got {idx!r} "
                f"(item_id={rec.get('item_id')!r})"
            )
        if "item_id" not in rec:
            raise RepeatBaselineError(f"repeat call missing 'item_id': {rec!r}")
        item = rec["item_id"]
        slot = by_item.setdefault(item, {})
        if idx in slot:
            raise RepeatBaselineError(
                f"duplicate repeat_index={idx} for item_id={item!r}"
            )
        slot[idx] = rec

    pairs: List[dict] = []
    for item, calls in sorted(by_item.items()):
        missing = {1, 2} - calls.keys()
        if missing:
            raise RepeatBaselineError(
                f"item_id={item!r} has repeat_index {sorted(calls.keys())} but "
                f"is missing {sorted(missing)}; both {REPEAT_ARM} calls are "
                "required to form a repeat pair."
            )
        c1, c2 = calls[1], calls[2]
        pairs.append({
            "decision_a": c1.get("decision"),
            "decision_b": c2.get("decision"),
            "item_id": item,
            "repeat_pair_id": item,
            "arm_a": REPEAT_ARM,
            "arm_b": REPEAT_ARM,
        })
    return pairs
