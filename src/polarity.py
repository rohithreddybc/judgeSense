"""
Polarity-inverted judge templates and label remapping.

Reviewer xmQT, weakness W3 on the v1 submission:

    "Instead of figuring out how to handle the polarity-inverting templates in
    the Factuality task by flipping the expected labels, the authors just
    dropped them. I am not sure whether this artificially increases the
    consistency scores."

The criticism is of the *response*, not the templates. v1 shipped a factuality
template asking the opposite question ("does this response contain factual
errors?", where YES means inaccurate) alongside direct templates sharing one
label space, so YES meant opposite things in different arms. That is a genuine
defect. But excluding those pairings, as v1 did and as the v2 builder does by
construction, removes the hardest cases and can only push measured consistency
up.

This module supplies the alternative the reviewer asked for: keep the inverted
templates, and map every template's raw answer space onto a single canonical
decision space before comparing. A judge that answers both arms correctly then
agrees after remapping, so JSS measures instability rather than label-convention
mismatch — and the inverted arms become the more demanding test they should have
been, instead of being dropped.

Reporting contract: results must be given BOTH ways — direct-only pairings and
direct+inverted — because remapping changes what the number means. Section 3 of
docs/V2_ARCHITECTURE.md keeps the v2 default dataset direct-only; inverted
pairings are an explicitly requested extension, never a silent default.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

UNCLEAR = "UNCLEAR"

# Canonical decision space per task. Every template's answers are mapped into
# this space before any comparison happens.
CANONICAL_LABELS: Dict[str, Tuple[str, ...]] = {
    "factuality": ("accurate", "inaccurate"),
    "coherence": ("1", "2", "3", "4", "5"),
}

DIRECT = "direct"
INVERTED = "inverted"


class PolarityError(ValueError):
    """Raised when a template's polarity contract is violated."""


# Optional polarity-inverted templates, kept OUT of the default template set.
# Each is paired with the remap that carries its raw answers into the canonical
# space. The prompt text and the remap live together deliberately: v1's defect
# was that the inverted template existed while the remap did not.
INVERTED_TEMPLATES: Dict[str, List[dict]] = {
    "factuality": [
        {
            "template_id": "T_INV_1",
            "polarity": INVERTED,
            "text": (
                "Does this statement contain factual errors? "
                "Answer YES (it has errors) or NO (it is accurate).\n\n{text}"
            ),
            # YES means "has errors" here, i.e. the canonical "inaccurate".
            "remap": {"YES": "inaccurate", "NO": "accurate"},
        },
        {
            "template_id": "T_INV_2",
            "polarity": INVERTED,
            "text": (
                "Is this statement factually incorrect? Reply YES if it is "
                "incorrect, NO if it is correct.\n\n{text}"
            ),
            "remap": {"YES": "inaccurate", "NO": "accurate"},
        },
    ],
    "coherence": [
        {
            "template_id": "T_INV_1",
            "polarity": INVERTED,
            # A reversed anchor direction is the Likert analogue of polarity
            # inversion: the same judgment, the opposite end of the scale.
            "text": (
                "Rate this text on a 1-5 incoherence scale, where 1 means very "
                "coherent and 5 means incoherent. Reply with one number only.\n\n{text}"
            ),
            "remap": {"1": "5", "2": "4", "3": "3", "4": "2", "5": "1"},
        },
    ],
}

# Remap for the direct templates already in the v2 builder: their raw answers
# differ from the canonical labels only for factuality (YES/NO vs
# accurate/inaccurate).
DIRECT_REMAP: Dict[str, Dict[str, str]] = {
    "factuality": {"YES": "accurate", "NO": "inaccurate"},
    "coherence": {str(i): str(i) for i in range(1, 6)},
}


def template_remap(task: str, polarity: str, template_id: Optional[str] = None) -> Dict[str, str]:
    """Return the raw-answer to canonical-label map for one template."""
    if task not in CANONICAL_LABELS:
        raise PolarityError(f"no canonical label space defined for task {task!r}")
    if polarity == DIRECT:
        return dict(DIRECT_REMAP[task])
    if polarity == INVERTED:
        for spec in INVERTED_TEMPLATES.get(task, []):
            if template_id is None or spec["template_id"] == template_id:
                return dict(spec["remap"])
        raise PolarityError(f"{task}: no inverted template {template_id!r}")
    raise PolarityError(f"unknown polarity {polarity!r}")


def canonicalize(task: str, raw_label: str, polarity: str,
                 template_id: Optional[str] = None) -> str:
    """
    Map one raw decision into the canonical space for its task.

    UNCLEAR passes through: an unparseable answer has no polarity to correct,
    and silently converting it to a label would manufacture a decision.
    """
    if raw_label is None:
        return UNCLEAR
    label = str(raw_label).strip()
    if label == UNCLEAR:
        return UNCLEAR
    remap = template_remap(task, polarity, template_id)
    if label not in remap:
        return UNCLEAR
    return remap[label]


def canonicalize_record(record: dict) -> dict:
    """
    Return a copy of a decision record with both arms canonicalized.

    Expects `task_type`, `decision_a`/`decision_b`, and per-arm polarity in
    `polarity_a`/`polarity_b` (defaulting to direct) plus optional
    `template_a_id`/`template_b_id`. The original raw decisions are preserved
    under `raw_decision_a`/`raw_decision_b` so a remap can always be audited
    rather than taken on trust.
    """
    task = record["task_type"]
    out = dict(record)
    out["raw_decision_a"] = record["decision_a"]
    out["raw_decision_b"] = record["decision_b"]
    out["decision_a"] = canonicalize(
        task, record["decision_a"],
        record.get("polarity_a", DIRECT), record.get("template_a_id"),
    )
    out["decision_b"] = canonicalize(
        task, record["decision_b"],
        record.get("polarity_b", DIRECT), record.get("template_b_id"),
    )
    out["canonicalized"] = True
    return out


def canonicalize_records(records: Sequence[dict]) -> List[dict]:
    """Canonicalize a sequence of decision records."""
    return [canonicalize_record(r) for r in records]


def assert_remap_is_bijective(task: str) -> None:
    """
    Every template's remap must be a bijection onto the canonical labels.

    A non-bijective remap would merge or drop decisions — silently changing the
    label space, which is the defect class this module exists to handle rather
    than reintroduce.
    """
    canonical = set(CANONICAL_LABELS[task])
    specs = [{"template_id": None, "remap": DIRECT_REMAP[task]}]
    specs += INVERTED_TEMPLATES.get(task, [])
    for spec in specs:
        remap = spec["remap"]
        if len(set(remap.values())) != len(remap):
            raise PolarityError(
                f"{task}/{spec['template_id']}: remap is not injective "
                f"({remap}) — two raw answers collapse to one decision"
            )
        if set(remap.values()) != canonical:
            raise PolarityError(
                f"{task}/{spec['template_id']}: remap does not cover the "
                f"canonical space {sorted(canonical)} (got {sorted(set(remap.values()))})"
            )


def has_inverted_arm(record: dict) -> bool:
    """True when either arm of a pairing uses an inverted template."""
    return INVERTED in (record.get("polarity_a", DIRECT), record.get("polarity_b", DIRECT))


def split_by_polarity(records: Sequence[dict]) -> Dict[str, List[dict]]:
    """
    Partition records into direct-only and inverted-involving pairings.

    The reporting contract requires both slices: excluding the inverted arms is
    what v1 did, and reporting only the pooled number would hide whether the
    harder pairings move the result. Callers report `direct_only` and `all`, and
    the difference between them is itself a finding.
    """
    direct_only = [r for r in records if not has_inverted_arm(r)]
    return {
        "direct_only": direct_only,
        "inverted_involving": [r for r in records if has_inverted_arm(r)],
        "all": list(records),
    }
