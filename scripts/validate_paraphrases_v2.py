"""
Offline paraphrase-equivalence audit for the SHIPPED v2 templates.

    python scripts/validate_paraphrases_v2.py            # audit all four tasks
    python scripts/validate_paraphrases_v2.py --task relevance
    python scripts/validate_paraphrases_v2.py --json out.json
    python scripts/validate_paraphrases_v2.py --strict   # nonzero exit on WARN too

Makes NO network calls. Everything is derived from data/v2/*.jsonl, which is
read but never written. The v1 counterpart, scripts/validate_paraphrases.py,
asks GPT-4o-mini whether two prompts are equivalent and only ever saw
data/prompt_pairs/; it is left untouched and is not a dependency of this file.

WHAT THIS CAN AND CANNOT DO
---------------------------
No offline check establishes that two instruction templates are semantically
equivalent. Equivalence is a claim about how a competent reader maps each
wording onto the same question, and nothing computable from the character
strings settles that. What the checks below do is BOUND SPECIFIC FAILURE MODES:
each one names a concrete way a "paraphrase" pair could differ in what it asks,
and reports whether the shipped templates exhibit it. Passing every check leaves
the equivalence claim unproven; failing any one of them refutes it.

The checks, and the exact scope of each:

  1. LABEL SPACE (blocking). The set of answers each template requests, parsed
     out of the instruction text with the payload removed. Rules out: two arms
     admitting different answer sets, so a "disagreement" is a parsing or
     format artifact rather than a judgment change. Does not rule out: the same
     token being reached by different reasoning.

  2. POLARITY (blocking). Two parts. (a) Token binding: every explicit gloss
     that ties an answer token to a conclusion ("NO (incorrect)", "1 =
     incoherent", "YES if it is correct") is scored and must not flip sign
     across templates of a task. (b) Predicate direction: the question itself,
     with glosses stripped, must not be inverted ("does this contain errors?",
     "which passage is LESS relevant?"). Rules out: the v1 Template-4 defect,
     where a shared label space hid opposite conclusions, in all four tasks
     rather than two. Does not rule out: threshold shifts inside a shared
     direction ("better" vs "clearly better").

  3. CONSTRUCT / THE ASK (blocking). The quantity the judge is asked to report,
     matched against a hand-declared per-task vocabulary. Rules out: a template
     that asks a different question than its partner (quality where the pair
     asks relevance). Does NOT establish that the terms grouped inside one
     task's vocabulary are synonyms -- that grouping is an authored assumption,
     printed in the report so it can be argued with, not a measurement.

  4. PAYLOAD POSITION (reported, non-blocking). Where the query and candidates
     sit inside the instruction. Rules nothing out; it makes a known confound
     -- serial position of the instruction relative to the material -- visible
     and quantified instead of implicit.

  5. READABILITY / COMPLEXITY (reported, non-blocking). Length, sentence count
     and Flesch scores of the instruction alone. Bounds "one arm is
     intrinsically harder to read"; it cannot show the divergence is harmless.

  6. NON-TRIVIALITY (blocking). The templates in a pair must actually differ.
     A pair of near-identical templates would make the benchmark measure
     nothing while passing every other check.

  7. ARTIFACT/CODE AGREEMENT (blocking). Skeletons recovered from the shipped
     JSONL must equal src/dataset_builder_v2.TEMPLATES, so the audit describes
     what shipped rather than what the builder currently says.

Separately, --design reports the assignment structure: how many templates,
pairs and items per pair, and how many distinct templates each item is seen
under. That measurement decides whether per-template claims are identifiable
at all.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_V2 = REPO_ROOT / "data" / "v2"

TASKS: Tuple[str, ...] = ("factuality", "coherence", "relevance", "preference")
PAIRWISE_TASKS: Set[str] = {"relevance", "preference"}

# Placeholder names used in the recovered skeletons.
PH_TEXT = "{text}"
PH_QUERY = "{query}"
PH_FIRST = "{first}"
PH_SECOND = "{second}"
PLACEHOLDERS = (PH_TEXT, PH_QUERY, PH_FIRST, PH_SECOND)


# ══════════════════════════════════════════════════════════════════════════════
# Declared vocabularies. These are AUTHORED ASSUMPTIONS, not measurements, and
# are printed in the report for exactly that reason.
# ══════════════════════════════════════════════════════════════════════════════

# The construct each task's templates are permitted to ask about. `terms` are
# the surface realisations we accept as asking for that one construct; treating
# them as interchangeable is a judgement call.
CONSTRUCT_VOCAB: Dict[str, Dict[str, object]] = {
    "factuality": {
        "canonical": "factual_accuracy",
        "terms": (
            "factually correct",
            "factually accurate",
            "factual accuracy",
            "fact-check",
            "accurate",
        ),
    },
    "coherence": {
        "canonical": "coherence",
        "terms": ("coherence", "coherent"),
    },
    "relevance": {
        "canonical": "relevance",
        "terms": ("relevance", "relevant"),
    },
    "preference": {
        "canonical": "pairwise_preference",
        "terms": ("better", "higher quality", "stronger", "preferred", "prefer"),
    },
}

# Words that make a gloss or a predicate point at the GOOD end of the construct.
POSITIVE_LEXICON = frozenset(
    """correct accurate true truthful factual coherent relevant better best
    stronger strongest higher highest greater good quality""".split()
)
# ... and at the BAD end.
NEGATIVE_LEXICON = frozenset(
    """incorrect inaccurate untrue false wrong error errors erroneous
    incoherent irrelevant worse worst weaker weakest lower lowest less least
    fewer poorer bad""".split()
)
# Explicit negation markers; they flip whatever polarity the rest of the span
# carries (and stand alone for "NO if it is not").
NEGATION_MARKERS = frozenset({"not", "no longer", "fails", "fail", "isn't", "does not"})

# Comparative direction words for the pairwise tasks and the scale tasks.
DIRECTION_UP = frozenset({"more", "higher", "better", "stronger", "greater", "most", "best"})
DIRECTION_DOWN = frozenset({"less", "lower", "worse", "weaker", "fewer", "least", "worst"})

# Cues that the template demands a bare answer token with no prose around it.
FORMAT_CUES = (
    "only",
    "one word",
    "single digit",
    "single number",
    "one number",
    "number only",
    "entire answer must be",
    "answer must be",
    "with the number",
    "give only",
    "respond with",
)


# ══════════════════════════════════════════════════════════════════════════════
# Findings
# ══════════════════════════════════════════════════════════════════════════════

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"


class Finding:
    __slots__ = ("level", "check", "task", "message")

    def __init__(self, level: str, check: str, task: str, message: str):
        self.level, self.check, self.task, self.message = level, check, task, message

    def to_dict(self) -> dict:
        return {"level": self.level, "check": self.check, "task": self.task,
                "message": self.message}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"[{self.level}] {self.task}/{self.check}: {self.message}"


# ══════════════════════════════════════════════════════════════════════════════
# Loading and skeleton recovery
# ══════════════════════════════════════════════════════════════════════════════

def load_rows(task: str, data_dir: Path = DATA_V2) -> List[dict]:
    """Every record of a shipped task file. Read-only."""
    path = data_dir / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"shipped split not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_pairwise_payload(rbj: str, prompt: str) -> Optional[Tuple[str, str]]:
    """Recover (first, second) from `response_being_judged` == "A: X | B: Y".

    " | B: " can occur inside a candidate, so every occurrence is tried and the
    one that reconstructs the prompt's own "\\nA: X\\nB: Y" tail wins. Returns
    None if no split reconstructs the prompt, which is itself a finding.
    """
    if not rbj.startswith("A: "):
        return None
    body = rbj[3:]
    for match in re.finditer(re.escape(" | B: "), body):
        first, second = body[: match.start()], body[match.end():]
        if prompt.endswith(f"\nA: {first}\nB: {second}"):
            return first, second
    return None


def recover_skeleton(row: dict, side: str) -> Optional[str]:
    """The instruction template behind one shipped prompt, payload removed.

    Derived from the artifact alone: the record carries the exact material that
    was interpolated, so the skeleton is obtained by deleting it rather than by
    diffing against the builder's source. Returns None when the payload cannot
    be located in the prompt, which the caller reports as a FAIL.
    """
    prompt = row[f"prompt_{side}"]
    rbj = row["response_being_judged"]
    task = row["task_type"]

    if task not in PAIRWISE_TASKS:
        if not prompt.endswith(rbj):
            return None
        return prompt[: len(prompt) - len(rbj)] + PH_TEXT

    parts = _split_pairwise_payload(rbj, prompt)
    if parts is None:
        return None
    first, second = parts
    tail = f"\nA: {first}\nB: {second}"
    instruction = prompt[: len(prompt) - len(tail)]

    # Every pairwise template wraps the query in double quotes and quotes
    # nothing else, so the outermost pair of quotes delimits it. A query that
    # itself contains quotes is still captured, since the outer marks remain
    # first and last. Correctness is confirmed downstream: if this were wrong
    # the recovered skeleton would differ between records of the same template.
    open_q, close_q = instruction.find('"'), instruction.rfind('"')
    if open_q < 0 or close_q <= open_q:
        return None
    return (instruction[: open_q + 1] + PH_QUERY + instruction[close_q:]
            + f"\nA: {PH_FIRST}\nB: {PH_SECOND}")


def recover_templates(rows: Sequence[dict]) -> Tuple[Dict[str, str], List[Finding]]:
    """template id -> skeleton, verified identical across every record using it."""
    task = rows[0]["task_type"] if rows else "?"
    seen: Dict[str, Set[str]] = defaultdict(set)
    findings: List[Finding] = []
    unrecoverable = 0

    for row in rows:
        for side in ("a", "b"):
            skeleton = recover_skeleton(row, side)
            if skeleton is None:
                unrecoverable += 1
                continue
            seen[row[f"template_{side}"]].add(skeleton)

    if unrecoverable:
        findings.append(Finding(
            FAIL, "skeleton_recovery", task,
            f"{unrecoverable} prompt(s) do not contain their recorded payload; "
            "the shipped prompt was not produced by interpolating the recorded "
            "material and cannot be audited"))

    templates: Dict[str, str] = {}
    for tid, variants in sorted(seen.items()):
        if len(variants) != 1:
            findings.append(Finding(
                FAIL, "skeleton_recovery", task,
                f"{tid} renders {len(variants)} distinct skeletons across the "
                "shipped rows; a template id does not identify one instruction"))
        templates[tid] = sorted(variants, key=len)[0]
    return templates, findings


def instruction_text(skeleton: str) -> str:
    """Skeleton with every placeholder deleted and whitespace normalised.

    All lexical analysis runs on this string so that no word of any item's
    payload can ever be mistaken for part of the instruction.
    """
    text = skeleton
    for ph in PLACEHOLDERS:
        text = text.replace(ph, " ")
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Check 1 -- label space
# ══════════════════════════════════════════════════════════════════════════════

_RANGE_RE = re.compile(
    r"\b(\d)\b\s*(?:\([^)]*\)\s*)?(?:to|through|or|-|–|—|\.\.)\s*(?:\([^)]*\)\s*)?\b(\d)\b"
)
# The "\nA: {first}\nB: {second}" frame labels the CANDIDATES; it is not part of
# the answer the template requests. Left in, a template that said "Answer 1 or 2"
# while still framing its candidates as A/B would report an unchanged label space.
_CANDIDATE_FRAME_RE = re.compile(r"\n[A-Z]:\s*\{(?:first|second)\}")
_CAPS_RE = re.compile(r"\b[A-Z]{2,6}\b")
_SINGLE_LETTER_RE = re.compile(r"\b[A-Z]\b")


def extract_label_space(skeleton: str) -> Tuple[Set[str], Dict[str, object]]:
    """The set of answers the template REQUESTS, parsed from instruction text.

    Not substring presence: a range is expanded to its members, and a token only
    counts if it appears in the instruction rather than anywhere in the prompt.
    """
    instr = instruction_text(_CANDIDATE_FRAME_RE.sub("", skeleton))
    evidence: Dict[str, object] = {}

    tokens: Set[str] = set(_CAPS_RE.findall(instr))
    if tokens:
        evidence["caps_tokens"] = sorted(tokens)

    letters = set(_SINGLE_LETTER_RE.findall(instr))
    if letters:
        tokens |= letters
        evidence["letter_tokens"] = sorted(letters)

    ranges = _RANGE_RE.findall(instr)
    if ranges:
        lo, hi = min(int(a) for a, _ in ranges), max(int(b) for _, b in ranges)
        if lo <= hi:
            span = {str(n) for n in range(lo, hi + 1)}
            tokens |= span
            evidence["range"] = [lo, hi]

    return tokens, evidence


def extract_format_constraint(skeleton: str) -> List[str]:
    instr = instruction_text(skeleton).lower()
    return [cue for cue in FORMAT_CUES if cue in instr]


# ══════════════════════════════════════════════════════════════════════════════
# Check 2 -- polarity
# ══════════════════════════════════════════════════════════════════════════════

# "NO (incorrect)", "1 (incoherent)"
_GLOSS_PAREN_RE = re.compile(r"\b([A-Z]{1,6}|\d)\b\s*\(([^)]*)\)")
# "1 = incoherent", "5 = very coherent"
_GLOSS_EQ_RE = re.compile(r"\b([A-Z]{1,6}|\d)\b\s*=\s*([^,;.)]+)")
# "1 means incoherent", "5 means very coherent"
_GLOSS_MEANS_RE = re.compile(r"\b([A-Z]{1,6}|\d)\b\s+means\s+([^,;.]+?)(?:\s+and\b|$|[,;.])")
# "YES if it is correct", "NO if it is not"
_GLOSS_IF_RE = re.compile(r"\b([A-Z]{1,6}|\d)\b\s+if\s+([^,;.]+)")


def score_polarity(span: str) -> Optional[int]:
    """+1 / -1 / None for a gloss or predicate fragment.

    A negation marker flips the sign of whatever else the span carries, and by
    itself yields -1 -- which is what makes "NO if it is not" resolve.
    """
    words = re.findall(r"[a-z'-]+", span.lower())
    if not words:
        return None
    negated = any(w in NEGATION_MARKERS for w in words)
    pos = sum(1 for w in words if w in POSITIVE_LEXICON)
    neg = sum(1 for w in words if w in NEGATIVE_LEXICON)
    if pos == 0 and neg == 0:
        return -1 if negated else None
    sign = 1 if pos > neg else -1 if neg > pos else None
    if sign is None:
        return None
    return -sign if negated else sign


def extract_token_bindings(skeleton: str) -> Dict[str, int]:
    """answer token -> +1/-1, from the glosses the template states explicitly.

    Templates that state no gloss return {}; their polarity is carried entirely
    by the predicate check below.
    """
    instr = instruction_text(skeleton)
    bindings: Dict[str, int] = {}
    for regex in (_GLOSS_PAREN_RE, _GLOSS_EQ_RE, _GLOSS_MEANS_RE, _GLOSS_IF_RE):
        for token, gloss in regex.findall(instr):
            sign = score_polarity(gloss)
            if sign is not None:
                bindings.setdefault(token, sign)
    return bindings


def _strip_glosses(instr: str) -> str:
    """The instruction minus every gloss span, i.e. the question being asked.

    Glosses legitimately contain negative words ("NO (incorrect)"), so they must
    be removed before the predicate is scored or every template looks inverted.
    """
    text = re.sub(r"\([^)]*\)", " ", instr)
    text = _GLOSS_EQ_RE.sub(" ", text)
    text = _GLOSS_MEANS_RE.sub(" ", text)
    text = _GLOSS_IF_RE.sub(" ", text)
    text = re.sub(r"\bwhere\b.*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_predicate(skeleton: str) -> Dict[str, object]:
    """Direction of the question itself, gloss spans removed.

    `direction` is +1 unless the predicate is inverted; `inverting_terms` names
    what inverted it. This is the check the v1 Template-4 artifact fails:
    "does this response contain factual errors?" scores -1 while every sibling
    scores +1.
    """
    predicate = _strip_glosses(instruction_text(skeleton))
    # Answer tokens are what the judge REPLIES, not what it is asked to weigh.
    # Scoring them would read a "TRUE or FALSE" label space as an inverted
    # question, because "false" sits in the negative lexicon.
    for token in extract_label_space(skeleton)[0]:
        predicate = re.sub(rf"\b{re.escape(token)}\b", " ", predicate)
    predicate = re.sub(r"\s+", " ", predicate).strip()
    words = re.findall(r"[a-z'-]+", predicate.lower())
    inverting = sorted({w for w in words if w in NEGATIVE_LEXICON or w in DIRECTION_DOWN})
    up = sorted({w for w in words if w in DIRECTION_UP})
    return {
        "predicate": predicate,
        "direction": -1 if inverting else 1,
        "inverting_terms": inverting,
        "direction_terms": up,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Check 3 -- the ask
# ══════════════════════════════════════════════════════════════════════════════

def _exclusive_terms(task: str) -> Set[str]:
    """Construct terms that belong to exactly one task's vocabulary."""
    owners: Dict[str, Set[str]] = defaultdict(set)
    for other, spec in CONSTRUCT_VOCAB.items():
        for term in spec["terms"]:  # type: ignore[index]
            owners[term].add(other)
    return {t for t, who in owners.items() if who == {task}}


def extract_construct(skeleton: str, task: str) -> Dict[str, object]:
    instr = instruction_text(skeleton).lower()
    own = sorted(t for t in CONSTRUCT_VOCAB[task]["terms"] if t in instr)  # type: ignore[index]
    foreign: List[str] = []
    for other in CONSTRUCT_VOCAB:
        if other == task:
            continue
        for term in _exclusive_terms(other):
            if re.search(rf"\b{re.escape(term)}\b", instr):
                foreign.append(f"{other}:{term}")
    return {
        "canonical": CONSTRUCT_VOCAB[task]["canonical"],
        "matched_terms": own,
        "foreign_terms": sorted(foreign),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Check 4 -- payload position
# ══════════════════════════════════════════════════════════════════════════════

def payload_positions(skeleton: str) -> Dict[str, Dict[str, float]]:
    """Where each placeholder sits, under two explicit normalisations.

    `frac_of_template` divides by the length of the template string with the
    placeholder names left in it. `frac_of_instruction` divides by the
    instruction length with placeholders deleted, i.e. the share of instruction
    the judge reads before reaching the material. Both are reported because
    neither is canonical: absolute values depend on the denominator chosen, so
    only the ORDERING and the within-pair delta should be read into.
    """
    instr_len = len(instruction_text(skeleton)) or 1
    out: Dict[str, Dict[str, float]] = {}
    for ph in PLACEHOLDERS:
        idx = skeleton.find(ph)
        if idx < 0:
            continue
        before_instr = len(instruction_text(skeleton[:idx]))
        out[ph.strip("{}")] = {
            "char_offset": idx,
            "frac_of_template": round(idx / len(skeleton), 4),
            "frac_of_instruction": round(before_instr / instr_len, 4),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Check 5 -- readability
# ══════════════════════════════════════════════════════════════════════════════

def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def readability(skeleton: str) -> Dict[str, float]:
    """Flesch scores over the instruction alone. Payload is excluded, so this
    describes how hard the INSTRUCTION is, not the item."""
    instr = instruction_text(skeleton)
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", instr)
    sentences = [s for s in re.split(r"[.!?]+", instr) if s.strip()]
    n_w, n_s = len(words), max(len(sentences), 1)
    n_syl = sum(_syllables(w) for w in words)
    if n_w == 0:
        return {"chars": 0, "words": 0, "sentences": n_s, "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0, "words_per_sentence": 0.0}
    wps, spw = n_w / n_s, n_syl / n_w
    return {
        "chars": len(instr),
        "words": n_w,
        "sentences": n_s,
        "words_per_sentence": round(wps, 2),
        "flesch_reading_ease": round(206.835 - 1.015 * wps - 84.6 * spw, 1),
        "flesch_kincaid_grade": round(0.39 * wps + 11.8 * spw - 15.59, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Check 6 -- non-triviality
# ══════════════════════════════════════════════════════════════════════════════

def wording_divergence(skel_a: str, skel_b: str) -> Dict[str, float]:
    import difflib
    a, b = instruction_text(skel_a), instruction_text(skel_b)
    wa = set(re.findall(r"[a-z']+", a.lower()))
    wb = set(re.findall(r"[a-z']+", b.lower()))
    jaccard = len(wa & wb) / len(wa | wb) if (wa | wb) else 1.0
    return {
        "token_jaccard": round(jaccard, 4),
        "char_similarity": round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4),
        "identical": a == b,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Check 7 -- artifact vs code
# ══════════════════════════════════════════════════════════════════════════════

def code_templates(task: str) -> Optional[List[str]]:
    """src/dataset_builder_v2.TEMPLATES[task], or None if unimportable."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.dataset_builder_v2 import TEMPLATES  # type: ignore
        return list(TEMPLATES[task])
    except Exception:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_dsb_v2", REPO_ROOT / "src" / "dataset_builder_v2.py")
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return list(mod.TEMPLATES[task])
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Design structure (task 3 of the brief)
# ══════════════════════════════════════════════════════════════════════════════

def design_summary(rows: Sequence[dict]) -> Dict[str, object]:
    """Counts that decide whether per-template claims are identifiable.

    If every item is seen under exactly one template PAIR, template identity is
    perfectly nested inside item identity: any difference between templates is
    also a difference between the disjoint sets of items they were shown, and no
    contrast can separate the two. That is a property of the assignment, not of
    the sample size, and no amount of data fixes it.
    """
    templates: Set[str] = set()
    pair_items: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    item_templates: Dict[str, Set[str]] = defaultdict(set)
    item_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    for row in rows:
        ta, tb = row["template_a"], row["template_b"]
        combo = tuple(sorted((ta, tb)))
        templates.update((ta, tb))
        pair_items[combo].add(row["item_id"])
        item_templates[row["item_id"]].update((ta, tb))
        item_pairs[row["item_id"]].add(combo)

    per_pair = {f"{a}-{b}": len(items) for (a, b), items in sorted(pair_items.items())}
    tmpl_counts = Counter(len(v) for v in item_templates.values())
    over_two = sorted(i for i, v in item_templates.items() if len(v) > 2)
    multi_pair = sorted(i for i, v in item_pairs.items() if len(v) > 1)

    n_items = len(item_templates)
    return {
        "rows": len(rows),
        "distinct_items": n_items,
        "distinct_templates": len(templates),
        "template_ids": sorted(templates),
        "distinct_template_pairs": len(pair_items),
        "possible_template_pairs": len(templates) * (len(templates) - 1) // 2,
        "items_per_pair": per_pair,
        "items_per_pair_min": min(per_pair.values()) if per_pair else 0,
        "items_per_pair_max": max(per_pair.values()) if per_pair else 0,
        "rows_per_item": round(len(rows) / n_items, 2) if n_items else 0.0,
        "templates_per_item_distribution": dict(sorted(tmpl_counts.items())),
        "items_under_more_than_two_templates": len(over_two),
        "items_under_more_than_one_pair": len(multi_pair),
        "template_effect_confounded_with_item": not multi_pair,
    }


# ══════════════════════════════════════════════════════════════════════════════
# The audit
# ══════════════════════════════════════════════════════════════════════════════

def check_templates(
    task: str,
    templates: Dict[str, str],
    shipped_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]], List[Finding]]:
    """Checks 1-6 over a task's template skeletons.

    Split out from `audit_task` so the same logic runs over three sources: the
    skeletons recovered from the shipped JSONL, the templates as written in
    src/dataset_builder_v2, and deliberately broken templates in the test suite.
    A check that has never been shown to fire is not evidence of anything, so
    the tests feed this the defects it is supposed to catch.

    `shipped_pairs` defaults to every unordered pair, which is what a template
    edit should be judged against; the audit passes the pairs actually shipped.
    """
    findings: List[Finding] = []
    if shipped_pairs is None:
        ids = sorted(templates)
        shipped_pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]

    per_template: Dict[str, Dict[str, object]] = {}
    for tid, skeleton in templates.items():
        label_space, ls_evidence = extract_label_space(skeleton)
        per_template[tid] = {
            "skeleton": skeleton,
            "instruction": instruction_text(skeleton),
            "label_space": sorted(label_space),
            "label_space_evidence": ls_evidence,
            "format_cues": extract_format_constraint(skeleton),
            "token_bindings": extract_token_bindings(skeleton),
            "predicate": extract_predicate(skeleton),
            "construct": extract_construct(skeleton, task),
            "positions": payload_positions(skeleton),
            "readability": readability(skeleton),
        }

    # ---- 1. label space -----------------------------------------------------
    spaces = {tid: tuple(info["label_space"]) for tid, info in per_template.items()}  # type: ignore[index]
    for tid, space in spaces.items():
        if not space:
            findings.append(Finding(
                FAIL, "label_space", task,
                f"{tid} requests no recognisable answer set; the admissible "
                "answers cannot be read off the instruction"))
    distinct_spaces = set(spaces.values())
    if len(distinct_spaces) > 1:
        detail = "; ".join(f"{tid}={list(sp)}" for tid, sp in sorted(spaces.items()))
        findings.append(Finding(
            FAIL, "label_space", task,
            f"templates request {len(distinct_spaces)} different answer sets: {detail}"))

    fmt = {tid: tuple(info["format_cues"]) for tid, info in per_template.items()}  # type: ignore[index]
    unconstrained = sorted(t for t, cues in fmt.items() if not cues)
    if unconstrained:
        findings.append(Finding(
            WARN, "answer_format", task,
            f"{', '.join(unconstrained)} state no single-token answer constraint, "
            "so their parse surface differs from their partners'"))

    # ---- 2. polarity --------------------------------------------------------
    binding_signs: Dict[str, Dict[str, int]] = defaultdict(dict)
    for tid, info in per_template.items():
        for token, sign in info["token_bindings"].items():  # type: ignore[index]
            binding_signs[token][tid] = sign
    for token, by_tid in sorted(binding_signs.items()):
        if len(set(by_tid.values())) > 1:
            detail = ", ".join(f"{t}={'+' if s > 0 else '-'}" for t, s in sorted(by_tid.items()))
            findings.append(Finding(
                FAIL, "polarity", task,
                f"answer '{token}' is bound to opposite conclusions across "
                f"templates ({detail}); a shared label space is hiding a "
                "polarity inversion"))

    directions = {tid: info["predicate"]["direction"] for tid, info in per_template.items()}  # type: ignore[index]
    if len(set(directions.values())) > 1:
        inverted = sorted(t for t, d in directions.items() if d < 0)
        terms = sorted({w for t in inverted
                        for w in per_template[t]["predicate"]["inverting_terms"]})  # type: ignore[index]
        findings.append(Finding(
            FAIL, "polarity", task,
            f"{', '.join(inverted)} ask an inverted question relative to their "
            f"siblings (inverting terms: {terms})"))
    elif all(d < 0 for d in directions.values()) and directions:
        findings.append(Finding(
            WARN, "polarity", task,
            "every template's predicate scores negative; the polarity lexicon "
            "may not cover this task's wording"))

    # Scale endpoints: the low anchor must be the bad end in every template.
    for tid, info in per_template.items():
        digits = {t: s for t, s in info["token_bindings"].items() if t.isdigit()}  # type: ignore[index]
        if len(digits) >= 2:
            lo, hi = min(digits, key=int), max(digits, key=int)
            if digits[lo] > 0 or digits[hi] < 0:
                findings.append(Finding(
                    FAIL, "polarity", task,
                    f"{tid} anchors its scale the wrong way round: {lo} is glossed "
                    f"{'positively' if digits[lo] > 0 else 'negatively'} and {hi} "
                    f"{'positively' if digits[hi] > 0 else 'negatively'}"))

    # ---- 3. the ask ---------------------------------------------------------
    for tid, info in per_template.items():
        con = info["construct"]  # type: ignore[index]
        if not con["matched_terms"]:
            findings.append(Finding(
                FAIL, "construct", task,
                f"{tid} names no term from the declared {con['canonical']} "
                f"vocabulary {list(CONSTRUCT_VOCAB[task]['terms'])}; what it asks "
                "the judge to report cannot be confirmed"))
        if con["foreign_terms"]:
            findings.append(Finding(
                FAIL, "construct", task,
                f"{tid} names another task's construct term "
                f"({', '.join(con['foreign_terms'])}); it may be asking a "
                "different question than its partner"))
    surface = sorted({t for info in per_template.values()
                      for t in info["construct"]["matched_terms"]})  # type: ignore[index]
    if len(surface) > 1:
        findings.append(Finding(
            INFO, "construct", task,
            f"{len(surface)} distinct surface realisations of the construct in "
            f"use: {surface}. Treating them as one construct is an authored "
            "assumption, not something this audit measures"))

    # ---- 4/5/6. pairwise comparisons ---------------------------------------
    pair_report: Dict[str, Dict[str, object]] = {}
    for a, b in shipped_pairs:
        if a not in per_template or b not in per_template:
            continue
        ia, ib = per_template[a], per_template[b]
        div = wording_divergence(ia["skeleton"], ib["skeleton"])  # type: ignore[arg-type]
        pos_delta = {}
        for ph in set(ia["positions"]) | set(ib["positions"]):  # type: ignore[arg-type]
            pa = ia["positions"].get(ph)  # type: ignore[union-attr]
            pb = ib["positions"].get(ph)  # type: ignore[union-attr]
            if pa and pb:
                pos_delta[ph] = {
                    "frac_of_template": [pa["frac_of_template"], pb["frac_of_template"]],
                    "abs_delta_template": round(
                        abs(pa["frac_of_template"] - pb["frac_of_template"]), 4),
                    "abs_delta_instruction": round(
                        abs(pa["frac_of_instruction"] - pb["frac_of_instruction"]), 4),
                }
        ra, rb = ia["readability"], ib["readability"]  # type: ignore[index]
        pair_report[f"{a}-{b}"] = {
            "wording": div,
            "position_delta": pos_delta,
            "readability_delta": {
                "chars": rb["chars"] - ra["chars"],  # type: ignore[index]
                "words": rb["words"] - ra["words"],  # type: ignore[index]
                "flesch_reading_ease": round(
                    rb["flesch_reading_ease"] - ra["flesch_reading_ease"], 1),  # type: ignore[index]
                "flesch_kincaid_grade": round(
                    rb["flesch_kincaid_grade"] - ra["flesch_kincaid_grade"], 2),  # type: ignore[index]
            },
            "label_space_identical": ia["label_space"] == ib["label_space"],
        }
        if div["identical"]:
            findings.append(Finding(
                FAIL, "non_triviality", task,
                f"{a}-{b} are the same instruction; this pair manipulates nothing"))
        elif div["char_similarity"] > 0.95:
            findings.append(Finding(
                WARN, "non_triviality", task,
                f"{a}-{b} differ by under 5% of characters "
                f"(similarity {div['char_similarity']}); a near-null manipulation"))

    if pair_report:
        worst_pos = max(
            ((name, ph, d["abs_delta_template"])
             for name, rep in pair_report.items()
             for ph, d in rep["position_delta"].items()),  # type: ignore[union-attr]
            key=lambda t: t[2], default=None)
        if worst_pos and worst_pos[2] >= 0.15:
            findings.append(Finding(
                WARN, "payload_position", task,
                f"largest payload-position divergence is {worst_pos[2]:.1%} of the "
                f"template ({worst_pos[0]}, placeholder '{worst_pos[1]}'); the two "
                "arms present the material at materially different points"))
        worst_read = max(
            ((name, abs(rep["readability_delta"]["flesch_kincaid_grade"]))  # type: ignore[index]
             for name, rep in pair_report.items()), key=lambda t: t[1], default=None)
        if worst_read and worst_read[1] >= 3.0:
            findings.append(Finding(
                WARN, "readability", task,
                f"largest instruction grade-level gap is {worst_read[1]:.1f} grades "
                f"({worst_read[0]}); one arm may be intrinsically harder to parse"))

    return per_template, pair_report, findings


def audit_task(task: str, data_dir: Path = DATA_V2) -> Dict[str, object]:
    """Full audit of one shipped split: recovery, checks 1-6, check 7, design."""
    rows = load_rows(task, data_dir)
    templates, findings = recover_templates(rows)
    shipped_pairs = sorted({tuple(sorted((r["template_a"], r["template_b"]))) for r in rows})
    per_template, pair_report, check_findings = check_templates(
        task, templates, shipped_pairs)
    findings.extend(check_findings)

    # ---- 7. artifact vs code ------------------------------------------------
    code = code_templates(task)
    if code is None:
        findings.append(Finding(
            WARN, "artifact_vs_code", task,
            "src/dataset_builder_v2.TEMPLATES could not be imported; the audit "
            "describes the shipped file but cannot confirm the builder agrees"))
    else:
        code_by_id = {f"T{i + 1}": t for i, t in enumerate(code)}
        for tid, skeleton in sorted(templates.items()):
            expected = code_by_id.get(tid)
            if expected is None:
                findings.append(Finding(
                    FAIL, "artifact_vs_code", task,
                    f"shipped data uses {tid} but the builder defines no such template"))
            elif expected != skeleton:
                findings.append(Finding(
                    FAIL, "artifact_vs_code", task,
                    f"{tid} as shipped differs from the builder's definition; the "
                    "released artifact was not produced by the current code"))
        for tid in sorted(set(code_by_id) - set(templates)):
            findings.append(Finding(
                INFO, "artifact_vs_code", task,
                f"builder defines {tid} but no shipped row uses it"))

    return {
        "task": task,
        "templates": per_template,
        "pairs": pair_report,
        "design": design_summary(rows),
        "findings": [f.to_dict() for f in findings],
        "findings_obj": findings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

_BAR = "=" * 78


def _print_task(rep: Dict[str, object]) -> None:
    task = rep["task"]
    tmpl: Dict[str, dict] = rep["templates"]  # type: ignore[assignment]
    design: dict = rep["design"]  # type: ignore[assignment]

    print(f"\n{_BAR}\n{str(task).upper()}\n{_BAR}")

    print(f"\n-- design ---------------------------------------------------------")
    print(f"rows {design['rows']} | distinct items {design['distinct_items']} | "
          f"rows/item {design['rows_per_item']}")
    print(f"distinct templates {design['distinct_templates']} "
          f"{design['template_ids']}")
    print(f"distinct template pairs {design['distinct_template_pairs']} of "
          f"{design['possible_template_pairs']} possible")
    print(f"items per pair: min {design['items_per_pair_min']}, "
          f"max {design['items_per_pair_max']}  {design['items_per_pair']}")
    print(f"templates per item: {design['templates_per_item_distribution']} "
          f"(>2 templates: {design['items_under_more_than_two_templates']} items; "
          f">1 pair: {design['items_under_more_than_one_pair']} items)")
    if design["template_effect_confounded_with_item"]:
        print("IDENTIFIABILITY: every item is seen under exactly one template pair, "
              "so template identity is perfectly nested inside item identity. "
              "Per-template contrasts are NOT identifiable from this design.")

    print(f"\n-- per template ---------------------------------------------------")
    for tid, info in sorted(tmpl.items()):
        pred = info["predicate"]
        con = info["construct"]
        read = info["readability"]
        print(f"\n{tid}: {info['instruction']}")
        print(f"    label space      : {info['label_space']}  {info['label_space_evidence']}")
        print(f"    answer bindings  : "
              f"{ {k: ('+' if v > 0 else '-') for k, v in info['token_bindings'].items()} or '(none stated)'}")
        print(f"    predicate        : direction "
              f"{'+' if pred['direction'] > 0 else '-'} "
              f"| up-terms {pred['direction_terms']} "
              f"| inverting {pred['inverting_terms']}")
        print(f"    construct        : {con['canonical']} via {con['matched_terms']}"
              + (f"  FOREIGN {con['foreign_terms']}" if con["foreign_terms"] else ""))
        print(f"    format cues      : {info['format_cues'] or '(none)'}")
        print(f"    instruction      : {read['chars']} chars, {read['words']} words, "
              f"FRE {read['flesch_reading_ease']}, FK grade {read['flesch_kincaid_grade']}")
        if info["positions"]:
            pos = ", ".join(f"{k} at {v['frac_of_template']:.1%} of template "
                            f"({v['frac_of_instruction']:.1%} of instruction)"
                            for k, v in info["positions"].items())
            print(f"    payload position : {pos}")

    print(f"\n-- shipped pairs --------------------------------------------------")
    hdr = f"{'pair':<9}{'label=':>7}{'jaccard':>9}{'charsim':>9}{'dFK':>7}{'dchars':>8}  max position delta"
    print(hdr)
    print("-" * len(hdr))
    for name, rep_p in sorted(rep["pairs"].items()):  # type: ignore[union-attr]
        pd = rep_p["position_delta"]
        worst = max((d["abs_delta_template"] for d in pd.values()), default=0.0)
        print(f"{name:<9}"
              f"{'yes' if rep_p['label_space_identical'] else 'NO':>7}"
              f"{rep_p['wording']['token_jaccard']:>9.3f}"
              f"{rep_p['wording']['char_similarity']:>9.3f}"
              f"{rep_p['readability_delta']['flesch_kincaid_grade']:>7.2f}"
              f"{rep_p['readability_delta']['chars']:>8}"
              f"  {worst:.1%}")

    findings: List[Finding] = rep["findings_obj"]  # type: ignore[assignment]
    print(f"\n-- findings -------------------------------------------------------")
    if not findings:
        print("none")
    for f in findings:
        print(f"  [{f.level:<4}] {f.check}: {f.message}")


def _print_epilogue(reports: List[Dict[str, object]]) -> None:
    print(f"\n{_BAR}\nWHAT THIS DOES AND DOES NOT ESTABLISH\n{_BAR}")
    print(
        "Passing every check above does NOT show the templates in a pair mean the\n"
        "same thing. It shows that four specific ways of differing are absent:\n"
        "  - they do not request different answer sets (label space)\n"
        "  - no answer token denotes opposite conclusions across arms (polarity)\n"
        "  - none of them names a construct outside its task's declared vocabulary\n"
        "  - no pair is a near-duplicate, so each pair is a real manipulation\n"
        "and that two further differences, payload position and instruction\n"
        "readability, are measured and reported rather than assumed to be zero.\n"
        "\n"
        "Not addressable offline: whether a competent reader maps both wordings to\n"
        "the same question; whether a judge's implicit threshold shifts between\n"
        "'better' and 'of higher quality'; whether tokenisation or pretraining\n"
        "frequency makes one phrasing easier for a particular model. Those need\n"
        "human annotation or a model in the loop, and semantic drift therefore\n"
        "remains a live alternative explanation for part of any measured\n"
        "sensitivity."
    )
    confounded = [r["task"] for r in reports
                  if r["design"]["template_effect_confounded_with_item"]]  # type: ignore[index]
    if confounded:
        print(
            f"\nIDENTIFIABILITY ({', '.join(str(t) for t in confounded)}): each item is "
            "seen under exactly one\ntemplate pair, so the set of items behind template "
            "X is disjoint from the set\nbehind template Y. Any per-template difference "
            "is equally an item-set\ndifference, and the two cannot be separated by any "
            "estimator applied to this\ndesign. Pair-level sensitivity, which compares "
            "two templates WITHIN an item,\nis unaffected -- it is only the "
            "between-template comparison that is lost."
        )


def summarise(reports: List[Dict[str, object]]) -> Tuple[int, int, int]:
    fails = warns = infos = 0
    for rep in reports:
        for f in rep["findings_obj"]:  # type: ignore[union-attr]
            if f.level == FAIL:
                fails += 1
            elif f.level == WARN:
                warns += 1
            else:
                infos += 1
    return fails, warns, infos


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="validate_paraphrases_v2",
        description="Offline paraphrase-equivalence audit of the shipped v2 templates.",
    )
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--data-dir", type=Path, default=DATA_V2,
                    help="directory holding the shipped <task>.jsonl files")
    ap.add_argument("--json", type=Path, metavar="PATH",
                    help="also write the full machine-readable report here")
    ap.add_argument("--quiet", action="store_true", help="findings only")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero on WARN as well as FAIL")
    args = ap.parse_args(argv)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    reports = [audit_task(t, args.data_dir) for t in tasks]

    for rep in reports:
        if args.quiet:
            print(f"\n{rep['task']}:")
            for f in rep["findings_obj"]:  # type: ignore[union-attr]
                print(f"  [{f.level:<4}] {f.check}: {f.message}")
        else:
            _print_task(rep)

    if not args.quiet:
        _print_epilogue(reports)

    fails, warns, infos = summarise(reports)
    print(f"\n{_BAR}")
    print(f"RESULT: {fails} FAIL, {warns} WARN, {infos} INFO across "
          f"{len(reports)} task(s). No API calls were made.")
    print(_BAR)

    if args.json:
        payload = [{k: v for k, v in rep.items() if k != "findings_obj"} for rep in reports]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if fails or (args.strict and warns) else 0


if __name__ == "__main__":
    sys.exit(main())
