"""
JudgeSense v2.1 — structural paraphrase variants (docs/V2_1_STRUCTURAL_AXIS.md).

A structural variant changes the *scaffolding* of a judge prompt — where the
instruction lives, what persona frames it, whether reasoning is elicited, how
the answer must be formatted — while holding the evaluative question and the
label space byte-identical.

Variants carry a pre-registered class:

  Class E (equivalence-preserving)  a competent human given either arm answers
                                    the same question by the same criteria, so
                                    disagreement is judge instability and JSS
                                    keeps its meaning
  Class N (intervention)            the arm plausibly changes the judgment
                                    process itself, so disagreement is NOT
                                    instability; these feed Structural Shift
                                    Rate (see metrics_v2.structural_shift_rate)
                                    and must never be reported as JSS

`assert_no_polarity_drift` enforces that no variant alters the label space or
flips label-direction wording — the v1 factuality Template-4 defect class.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

VARIANT_IDS = ("S0", "S1", "S2", "S3", "S4", "S5")

# Pre-registered class per variant. S3 is borderline by design and may be
# demoted to "N" by the pre-run equivalence audit before any judging; that
# demotion is a config change here, not a code change elsewhere.
VARIANT_CLASS: Dict[str, str] = {
    "S0": "baseline",
    "S1": "E",   # structured output
    "S2": "E",   # system-prompt relocation
    "S3": "E",   # neutral role prime (borderline; demotable)
    "S4": "N",   # chain-of-thought
    "S5": "N",   # expert persona
}

VARIANT_NAME: Dict[str, str] = {
    "S0": "baseline",
    "S1": "structured_output",
    "S2": "system_prompt_relocation",
    "S3": "neutral_role_prime",
    "S4": "chain_of_thought",
    "S5": "expert_persona",
}

# The label space is frozen across every variant of a task.
LABEL_SETS: Dict[str, Sequence[str]] = {
    "factuality": ("YES", "NO"),
    "coherence": ("1", "2", "3", "4", "5"),
    "relevance": ("A", "B"),
    "preference": ("A", "B"),
}

_PERSONA: Dict[str, str] = {
    "factuality": "You are a senior fact-checker at a major newsroom.",
    "coherence": "You are a senior editor who assesses writing quality professionally.",
    "relevance": "You are a senior search-relevance rater.",
    "preference": "You are a senior evaluator of assistant response quality.",
}

_NEUTRAL_PRIME = "You are an impartial evaluator."
_FINAL_MARKER = "FINAL:"

_POINTWISE_TASKS = {"factuality", "coherence"}
_PAIRWISE_TASKS = {"relevance", "preference"}


class VariantError(ValueError):
    """Raised when a variant cannot be rendered or its contract is violated."""


def split_instruction(task: str, rendered_prompt: str) -> tuple[str, str]:
    """
    Split a rendered S0 prompt into (instruction, body).

    Needed by S2, which relocates the instruction into the system message and
    leaves only the judged content in the user message. The split is
    task-shape-aware rather than a generic heuristic:

      pointwise  "INSTRUCTION\\n\\n{text}"            -> split on first blank line
      pairwise   "INSTRUCTION...\\nA: {first}\\nB: .." -> split before the "A:" line

    For pairwise the query stays with the instruction: it is part of what is
    being asked, not part of the candidates being compared.
    """
    if task in _POINTWISE_TASKS:
        parts = rendered_prompt.split("\n\n", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise VariantError(
                f"{task}: cannot split instruction from body; expected a blank "
                f"line separating them in {rendered_prompt[:80]!r}"
            )
        return parts[0].strip(), parts[1].strip()

    if task in _PAIRWISE_TASKS:
        match = re.search(r"\nA:\s", rendered_prompt)
        if match is None:
            raise VariantError(
                f"{task}: cannot split instruction from candidates; no 'A:' line "
                f"in {rendered_prompt[:80]!r}"
            )
        return rendered_prompt[: match.start()].strip(), rendered_prompt[match.start() + 1 :].strip()

    raise VariantError(f"unknown task {task!r}")


def json_label_map(task: str) -> Dict[str, str]:
    """
    Bijective map from the task's label to its JSON `verdict` value.

    Identity by construction: S1 must not rename, merge, or reorder labels.
    Returned explicitly so it can be recorded per row and checked by the audit
    gate rather than assumed.
    """
    labels = LABEL_SETS[task]
    return {label: label for label in labels}


def render_variant(task: str, rendered_prompt: str, variant: str) -> dict:
    """
    Build the message payload for one structural variant of an S0 prompt.

    Returns {"variant", "variant_name", "variant_class", "system", "user",
    "parse_mode", "label_map"}. `system` is None where the variant uses no
    system message. `parse_mode` tells the normalizer which rule applies.

    The evaluative sentence is never rewritten: every variant either wraps or
    relocates the S0 text verbatim.
    """
    if task not in LABEL_SETS:
        raise VariantError(f"unknown task {task!r}")
    if variant not in VARIANT_IDS:
        raise VariantError(f"unknown variant {variant!r}; expected one of {VARIANT_IDS}")

    labels = list(LABEL_SETS[task])
    out = {
        "variant": variant,
        "variant_name": VARIANT_NAME[variant],
        "variant_class": VARIANT_CLASS[variant],
        "system": None,
        "user": rendered_prompt,
        "parse_mode": "plain",
        "label_map": None,
    }

    if variant == "S0":
        return out

    if variant == "S1":
        allowed = " | ".join(json.dumps(label) for label in labels)
        out["system"] = (
            "Respond with a single JSON object and nothing else. It must have "
            f'exactly one key, "verdict", whose value is one of: {allowed}.'
        )
        out["parse_mode"] = "json"
        out["label_map"] = json_label_map(task)
        return out

    if variant == "S2":
        instruction, body = split_instruction(task, rendered_prompt)
        out["system"] = instruction
        out["user"] = body
        return out

    if variant == "S3":
        out["system"] = _NEUTRAL_PRIME
        return out

    if variant == "S4":
        out["system"] = (
            "Think step by step. When you are done reasoning, give your final "
            f"answer on the last line in the form '{_FINAL_MARKER} <answer>'."
        )
        out["parse_mode"] = "final_marker"
        return out

    if variant == "S5":
        out["system"] = _PERSONA[task]
        return out

    raise VariantError(f"unhandled variant {variant!r}")  # pragma: no cover


UNCLEAR = "UNCLEAR"


def parse_variant_output(task: str, raw: str, parse_mode: str) -> str:
    """
    Normalize a judge's raw output to a label, using the variant's parse rule.

    Strict throughout: anything not unambiguously a label is UNCLEAR. The v1
    parser was substring-based and mapped the English article "a" to decision
    "A", "Cannot determine" to "NO", and a "1-5" scale echo to score 1. None of
    those readings survive here.
    """
    labels = list(LABEL_SETS[task])
    if raw is None:
        return UNCLEAR
    text = str(raw).strip()
    if not text:
        return UNCLEAR

    if parse_mode == "json":
        candidate = _extract_json_object(text)
        if candidate is None:
            return UNCLEAR
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return UNCLEAR
        if not isinstance(obj, dict) or set(obj.keys()) != {"verdict"}:
            return UNCLEAR
        verdict = str(obj["verdict"]).strip()
        return verdict if verdict in labels else UNCLEAR

    if parse_mode == "final_marker":
        matches = re.findall(
            rf"{re.escape(_FINAL_MARKER)}\s*(.+)", text, flags=re.IGNORECASE
        )
        if not matches:
            return UNCLEAR
        return _match_label(matches[-1], labels)

    if parse_mode == "plain":
        return _match_label(text, labels)

    raise VariantError(f"unknown parse_mode {parse_mode!r}")


def _extract_json_object(text: str) -> Optional[str]:
    """Return the outermost {...} span, tolerating markdown fencing only."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return stripped[start : end + 1]


def _match_label(text: str, labels: Sequence[str]) -> str:
    """
    Resolve a free-text answer to exactly one label, or UNCLEAR.

    Accepts a bare label, a label with trivial punctuation/markdown, or a
    trailing label. Any text mentioning two different labels is UNCLEAR: an
    answer that names both is not evidence of either.
    """
    cleaned = re.sub(r"[*_`]", "", str(text)).strip()
    bare = cleaned.rstrip(".,!;:").strip()

    # A bare answer is case-insensitive: "b" alone unambiguously means B.
    for label in labels:
        if bare.upper() == label.upper():
            return label

    # Scanning for a label mentioned inside prose is CASE-SENSITIVE when the
    # labels are single characters. Case-insensitively, the English article "a"
    # is a standalone word matching label "A", so "As a judge, I select B" reads
    # as mentioning both and resolves to UNCLEAR — losing a clear B answer. The
    # v1 parser made the worse version of this mistake and returned "A".
    # Multi-character labels (YES/NO) have no such collision, so they stay
    # case-insensitive.
    single_char = all(len(label) == 1 for label in labels)
    flags = 0 if single_char else re.IGNORECASE
    mentioned = {
        label
        for label in labels
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", cleaned, flags)
    }
    if len(mentioned) != 1:
        return UNCLEAR
    return next(iter(mentioned))


def assert_no_polarity_drift(task: str, variants: Sequence[str] = VARIANT_IDS) -> None:
    """
    Assert no variant alters the label space or its direction wording.

    The v1 factuality Template-4 defect shipped a template asking the opposite
    question ("does this contain errors?") while sharing a label space, so YES
    meant "inaccurate" in one arm and "accurate" in another. Variants here are
    wrappers around a frozen evaluative sentence, and this makes that structural
    guarantee checkable rather than assumed.
    """
    labels = set(LABEL_SETS[task])
    probe = _probe_prompt(task)
    for variant in variants:
        payload = render_variant(task, probe, variant)
        if payload["label_map"] is not None:
            mapping = payload["label_map"]
            if set(mapping.keys()) != labels or set(mapping.values()) != labels:
                raise VariantError(
                    f"{task}/{variant}: label_map is not a bijection of {sorted(labels)}"
                )
        combined = f"{payload['system'] or ''}\n{payload['user']}"
        for banned in ("error", "incorrect?", "not coherent", "less relevant", "worse"):
            if banned in combined.lower() and banned not in probe.lower():
                raise VariantError(
                    f"{task}/{variant}: variant introduced polarity-inverting wording "
                    f"{banned!r}"
                )


def _probe_prompt(task: str) -> str:
    """Minimal well-formed S0 prompt used by the polarity check."""
    if task in _POINTWISE_TASKS:
        return "Judge the text below. Answer with one label only.\n\nprobe text"
    return 'Judge for the query "probe query". Answer A or B only.\nA: first\nB: second'


def structural_pair_id(item_id: str, variant: str) -> str:
    """Clustering key for one (item, structural variant) comparison against S0."""
    return f"{item_id}#S0-{variant}"


def enumerate_pairs(item_id: str) -> List[dict]:
    """
    The five structural pairs for one item: (S0,S1) ... (S0,S5).

    Star design anchored at S0 — not a full pairwise cross among arms. All five
    pairs share the item's single S0 call, which is exactly why the reporting
    cluster unit on this axis must be "item" and not "structural_pair".
    """
    return [
        {
            "item_id": item_id,
            "structural_pair_id": structural_pair_id(item_id, variant),
            "variant_a": "S0",
            "variant_b": variant,
            "variant_class": VARIANT_CLASS[variant],
        }
        for variant in VARIANT_IDS
        if variant != "S0"
    ]
