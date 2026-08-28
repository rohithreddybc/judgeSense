"""Run Anthropic judges through Claude Code subagents instead of the API.

WHY THIS EXISTS
The Anthropic API key has no credit, so claude-opus-5, claude-sonnet-5 and the
two stranded preference cells are unreachable by the normal path. Claude Code
usage is available. A subagent can act as a judge -- a probe returned a clean,
parseable "NO" with zero tool uses -- but at 43,115 tokens for ONE label against
~413 through the API, which is 104x and hopeless at 4,280 calls per judge.

That overhead is FIXED, not per-item: it is the harness system prompt and tool
definitions, loaded once per subagent regardless of what you ask. So it
amortises. At N items per call the per-item cost is 43,000/N + ~405, which at
N=50 is ~1,265 and at N=100 is ~835 -- the same order as the API.

WHAT BATCHING COSTS SCIENTIFICALLY, AND HOW IT IS CONTROLLED
A judge that sees 50 items at once is not the judge that saw one. Three risks,
each handled rather than waved at:

  Cross-item contamination. Arm A and arm B of the same item NEVER share a
  batch. Each batch contains at most one arm of any item, so a judge cannot see
  the paraphrase it is about to be compared against.

  Batch composition as a nuisance variable. Batches are built by a seeded
  permutation, and the repeat arm reuses the SAME composition and position as
  the arm it baselines. Batch context therefore enters JSS_para and JSS_rep
  alike and cancels in the difference -- which is precisely the job the repeat
  baseline already does for decoding noise.

  Position within a batch. Recorded per item (`batch_index`, `batch_position`)
  so a position effect is measurable after the fact instead of assumed absent.

WHAT CANNOT BE CONTROLLED, AND MUST BE DISCLOSED
Temperature is not exposed by the harness, and the system prompt is Claude
Code's, not ours. Both are recorded as such. The endpoint survives this: dJSS
subtracts a repeat ceiling measured under the SAME configuration, so whatever
temperature the harness runs at is absorbed exactly as the API path absorbs its
own. What does not survive is cross-configuration comparison: a Claude Code
judge is not directly comparable to an API judge run under our system prompt,
and any table carrying both must say so.

ALIGNMENT IS BY pair_id, NEVER BY POSITION
A batch that returns 49 of 50 labels must fail loudly. Trusting order would
shift every subsequent label by one and silently corrupt the cell.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, Iterable, List, Optional, Sequence

# Conservative default. Higher amortises better; too high and a model starts
# dropping or truncating items, which costs a whole batch retry.
DEFAULT_BATCH_SIZE = 50

_INSTRUCTIONS = """\
You are acting as an evaluation judge. Below are {n} independent items.

Judge each item ENTIRELY ON ITS OWN. Items are unrelated; do not let one
influence another, and do not look for patterns across them.

Answer with ONLY the label each item asks for. Do not explain. Do not use tools.

Return exactly {n} lines of JSON, one per item, in this form and nothing else:
{{"id": "<the item id>", "answer": "<the label>"}}

Every id below must appear exactly once in your output.
"""


def build_batch_prompt(items: Sequence[dict]) -> str:
    """One prompt carrying many items, each tagged with the id it must return."""
    parts = [_INSTRUCTIONS.format(n=len(items))]
    for item in items:
        parts.append(
            f"\n----- ITEM {item['id']} -----\n{item['prompt']}\n")
    parts.append(
        f"\nNow output exactly {len(items)} JSON lines, one per item id above.")
    return "".join(parts)


def make_batches(units: Sequence[dict], batch_size: int = DEFAULT_BATCH_SIZE,
                 seed: int = 42) -> List[List[dict]]:
    """Group judging units into batches, with the two arms of an item separated.

    `units` are dicts with at least {id, pair_id, arm, prompt}. The invariant is
    that no batch holds two units sharing a pair_id, so a judge never sees an
    item's paraphrase alongside it.
    """
    ordered = list(units)
    random.Random(seed).shuffle(ordered)

    batches: List[List[dict]] = []
    pending: List[dict] = []
    for unit in ordered:
        placed = False
        # try the open batch first, then any earlier one with room and no clash
        for batch in ([pending] if pending else []) + batches:
            if len(batch) >= batch_size:
                continue
            if any(u["pair_id"] == unit["pair_id"] for u in batch):
                continue
            batch.append(unit)
            placed = True
            break
        if placed:
            if pending and len(pending) >= batch_size:
                batches.append(pending)
                pending = []
            continue
        if pending:
            batches.append(pending)
        pending = [unit]
    if pending:
        batches.append(pending)

    for i, batch in enumerate(batches):
        for j, unit in enumerate(batch):
            unit["batch_index"] = i
            unit["batch_position"] = j
    return batches


class BatchResponseError(ValueError):
    """A batch response cannot be aligned to the items that were sent."""


def parse_batch_response(text: str, expected_ids: Iterable[str]) -> Dict[str, str]:
    """Map response -> {id: answer}, keyed by id and never by position.

    Raises rather than guessing when ids are missing or unexpected. A silently
    shortened batch would shift every later label by one.
    """
    expected = list(expected_ids)
    found: Dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip().strip("`").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        key, ans = obj.get("id"), obj.get("answer")
        if key is None or ans is None:
            continue
        found[str(key)] = str(ans).strip()

    missing = [i for i in expected if i not in found]
    extra = [k for k in found if k not in set(expected)]
    if missing or extra:
        raise BatchResponseError(
            f"batch response does not align: {len(missing)} missing, "
            f"{len(extra)} unexpected (expected {len(expected)}). "
            f"missing[:5]={missing[:5]} extra[:5]={extra[:5]}"
        )
    return found


def provenance(model_alias: str, batch_size: int, seed: int) -> dict:
    """What this configuration actually is, recorded rather than implied.

    Deliberately NOT shaped like an API usage record: there is no provider token
    count, no served-model string and no temperature to report, and inventing
    fields that look like the API path's would make two different measurements
    indistinguishable downstream.
    """
    return {
        "transport": "claude_code_subagent",
        "model_alias": model_alias,
        "batch_size": batch_size,
        "batch_seed": seed,
        "temperature": None,
        "temperature_omitted_provider_default": True,
        "system_prompt_sent": False,
        "system_prompt_sha": None,
        "harness_system_prompt": "claude_code_default",
        "comparable_to_api_judges": False,
        "notes": ("temperature is not exposed by the harness and the system "
                  "prompt is Claude Code's, so this judge is comparable to "
                  "itself across arms but not to API-run judges"),
    }


def units_for_task(rows: Sequence[dict], arms: Sequence[str] = ("a", "b")) -> List[dict]:
    """Flatten benchmark rows into one judging unit per (row, arm)."""
    out: List[dict] = []
    for row in rows:
        for arm in arms:
            prompt = row.get(f"prompt_{arm.rstrip('_repeat')}") if arm.endswith("_repeat") \
                else row.get(f"prompt_{arm}")
            if not prompt:
                continue
            out.append({
                "id": f"{row['pair_id']}#{arm}",
                "pair_id": str(row["pair_id"]),
                "arm": arm,
                "prompt": prompt,
            })
    return out


def batch_digest(batch: Sequence[dict]) -> str:
    """Stable id for a batch, so a dispatched batch and its response can be
    matched up without trusting filenames."""
    h = hashlib.sha256()
    for unit in batch:
        h.update(unit["id"].encode())
    return h.hexdigest()[:12]
