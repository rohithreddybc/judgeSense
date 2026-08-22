"""
Paraphrase-equivalence invariants for ALL FOUR v2 tasks.

Replaces the coverage gap in test_dataset_builder_v2::test_templates_do_not_
invert_polarity, which iterated only TEMPLATES["factuality"] and
TEMPLATES["coherence"] -- leaving the two pairwise tasks unchecked -- and
asserted substring presence ("yes" in the template) rather than anything about
the answer set requested or the direction of the question.

Two surfaces are checked, because they can drift apart:

  * src/dataset_builder_v2.TEMPLATES, so an edit to a template fails here
    before any dataset is rebuilt; and
  * data/v2/*.jsonl, so the released artifact is checked as shipped rather
    than as the builder currently describes it.

The negative controls at the bottom are load-bearing. A checker that has never
been shown to fire proves nothing, so each invariant is paired with a template
that breaks it and must be caught -- including the exact v1 Template-4 polarity
artifact the paper cites.

None of this shows the templates in a pair MEAN the same thing; see the module
docstring of scripts/validate_paraphrases_v2.py for what each check does and
does not bound.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "validate_paraphrases_v2", REPO / "scripts" / "validate_paraphrases_v2.py")
vp2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp2)

from src.dataset_builder_v2 import TEMPLATES  # noqa: E402

TASKS = ("factuality", "coherence", "relevance", "preference")
DATA_V2 = REPO / "data" / "v2"


def code_templates(task):
    """The builder's templates keyed the way the shipped rows key them."""
    return {f"T{i + 1}": t for i, t in enumerate(TEMPLATES[task])}


def fails_of(findings, check=None):
    return [f for f in findings
            if f.level == vp2.FAIL and (check is None or f.check == check)]


def check_code(task):
    return vp2.check_templates(task, code_templates(task))[2]


@pytest.fixture(scope="module")
def shipped():
    """Full offline audit of every shipped split, computed once."""
    if not DATA_V2.exists():
        pytest.skip("data/v2 not present")
    return {task: vp2.audit_task(task) for task in TASKS}


# ══════════════════════════════════════════════════════════════════════════════
# The builder's templates
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("task", TASKS)
def test_all_five_templates_request_one_label_space(task):
    """Every template in a task must request the SAME admissible answer set.

    The set is parsed out of the instruction with the payload and the candidate
    frame removed, so this is about what the template asks for, not about which
    characters happen to appear somewhere in the prompt.
    """
    templates = code_templates(task)
    spaces = {tid: tuple(sorted(vp2.extract_label_space(t)[0]))
              for tid, t in templates.items()}
    assert all(spaces.values()), f"{task}: template with no parseable answer set: {spaces}"
    assert len(set(spaces.values())) == 1, f"{task}: label spaces diverge: {spaces}"


@pytest.mark.parametrize("task,expected", [
    ("factuality", ("NO", "YES")),
    ("coherence", ("1", "2", "3", "4", "5")),
    ("relevance", ("A", "B")),
    ("preference", ("A", "B")),
])
def test_label_space_is_the_documented_one(task, expected):
    """Pinned so a silent change of answer set (1-5 to 1-7, YES/NO to TRUE/FALSE)
    fails here rather than quietly changing what the metric is computed over."""
    for tid, template in code_templates(task).items():
        got = tuple(sorted(vp2.extract_label_space(template)[0]))
        assert got == expected, f"{task}/{tid} requests {got}, expected {expected}"


@pytest.mark.parametrize("task", TASKS)
def test_no_polarity_inversion(task):
    """No answer token may denote opposite conclusions across a task's templates,
    and no template may ask the inverted question.

    This is the v1 Template-4 defect generalised to all four tasks: there, a
    template asked whether the response CONTAINED ERRORS while sharing the
    YES/NO space with templates asking whether it was CORRECT, so a judge that
    answered both arms correctly was scored as self-inconsistent.
    """
    failures = fails_of(check_code(task), "polarity")
    assert not failures, f"{task}: " + "; ".join(f.message for f in failures)


@pytest.mark.parametrize("task", TASKS)
def test_scale_anchors_point_the_same_way(task):
    """Where a template glosses its endpoints, the low one must be the bad end.

    A template reading "1 (very coherent) to 5 (incoherent)" has an identical
    label space and passes every substring test, while reversing the meaning of
    every score.
    """
    for tid, template in code_templates(task).items():
        digits = {t: s for t, s in vp2.extract_token_bindings(template).items()
                  if t.isdigit()}
        if len(digits) < 2:
            continue
        lo, hi = min(digits, key=int), max(digits, key=int)
        assert digits[lo] < 0 < digits[hi], (
            f"{task}/{tid}: scale anchors inverted ({lo}->{digits[lo]}, {hi}->{digits[hi]})")


@pytest.mark.parametrize("task", TASKS)
def test_every_template_asks_this_task_construct(task):
    """Each template must name a term from its own task's declared construct
    vocabulary and none from another task's exclusive vocabulary. A relevance
    template that asked which passage is of higher QUALITY is not a paraphrase
    of one asking which is more RELEVANT."""
    failures = fails_of(check_code(task), "construct")
    assert not failures, f"{task}: " + "; ".join(f.message for f in failures)


@pytest.mark.parametrize("task", TASKS)
def test_templates_are_a_real_manipulation(task):
    """No two templates may be identical or near-identical; a null manipulation
    would score perfect stability while measuring nothing."""
    failures = fails_of(check_code(task), "non_triviality")
    assert not failures, f"{task}: " + "; ".join(f.message for f in failures)


# ══════════════════════════════════════════════════════════════════════════════
# The shipped artifact
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("task", TASKS)
def test_shipped_prompts_are_template_plus_recorded_payload(task, shipped):
    """Every shipped prompt must decompose into a recorded template id and the
    material the row says was interpolated. If it does not, the released file
    cannot be audited at all."""
    failures = fails_of(shipped[task]["findings_obj"], "skeleton_recovery")
    assert not failures, f"{task}: " + "; ".join(f.message for f in failures)


@pytest.mark.parametrize("task", TASKS)
def test_shipped_templates_match_the_builder(task, shipped):
    """The released file must have been produced by the templates in the current
    builder, or the checks above describe code that never shipped."""
    failures = fails_of(shipped[task]["findings_obj"], "artifact_vs_code")
    assert not failures, f"{task}: " + "; ".join(f.message for f in failures)


@pytest.mark.parametrize("task", TASKS)
def test_shipped_audit_reports_no_blocking_failure(task, shipped):
    failures = fails_of(shipped[task]["findings_obj"])
    assert not failures, f"{task}: " + "; ".join(
        f"{f.check}: {f.message}" for f in failures)


@pytest.mark.parametrize("task", TASKS)
def test_every_shipped_pair_shares_a_label_space(task, shipped):
    bad = [name for name, rep in shipped[task]["pairs"].items()
           if not rep["label_space_identical"]]
    assert not bad, f"{task}: pairs with divergent label spaces: {bad}"


# ══════════════════════════════════════════════════════════════════════════════
# Design structure -- a tripwire on a stated limitation, not a quality bar
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("task", TASKS)
def test_item_template_confound_is_as_the_paper_states(task, shipped):
    """Each item is issued under exactly one template pair, so template identity
    is perfectly nested inside item identity and no per-template contrast is
    identifiable.

    paper/03_benchmark.tex states this as a limitation. The assertion is here so
    that CHANGING the design -- crossing items with more than one pair, which
    would make per-template effects estimable -- fails this test and forces the
    paper's limitation paragraph to be revisited rather than left stale.
    """
    design = shipped[task]["design"]
    assert design["items_under_more_than_two_templates"] == 0
    assert design["items_under_more_than_one_pair"] == 0
    assert design["template_effect_confounded_with_item"] is True
    assert set(design["templates_per_item_distribution"]) == {2}


@pytest.mark.parametrize("task", TASKS)
def test_all_ten_template_pairs_are_used_and_balanced(task, shipped):
    design = shipped[task]["design"]
    assert design["distinct_templates"] == 5
    assert design["distinct_template_pairs"] == design["possible_template_pairs"] == 10
    assert design["items_per_pair_min"] == design["items_per_pair_max"], (
        f"{task}: unbalanced items per template pair: {design['items_per_pair']}")


# ══════════════════════════════════════════════════════════════════════════════
# Negative controls -- each invariant paired with a defect that must be caught
# ══════════════════════════════════════════════════════════════════════════════

BROKEN = [
    # The v1 Template-4 artifact, verbatim in shape: shared YES/NO space, opposite
    # conclusion. This is the defect the paper cites as a benchmark-design lesson.
    ("factuality", "T4", "polarity",
     "Does this response contain factual errors? Answer NO if accurate, "
     "YES if not.\n\n{text}"),
    ("factuality", "T3", "label_space",
     "Is the following statement accurate? Answer TRUE or FALSE.\n\n{text}"),
    ("coherence", "T1", "polarity",
     "Rate the coherence of this text from 1 (very coherent) to 5 (incoherent). "
     "Reply with one number only.\n\n{text}"),
    ("coherence", "T2", "label_space",
     "How coherent is the following text? Score it 1-7 (1 = incoherent, "
     "7 = very coherent). Number only.\n\n{text}"),
    # The pairwise cases the old test could not reach at all.
    ("relevance", "T3", "polarity",
     'Query: "{query}". Choose the less relevant passage. Answer A or B.'
     "\nA: {first}\nB: {second}"),
    ("relevance", "T3", "construct",
     'Query: "{query}". Choose the higher quality passage. Answer A or B.'
     "\nA: {first}\nB: {second}"),
    ("relevance", "T3", "label_space",
     'Query: "{query}". Choose the more relevant passage. Answer 1 or 2.'
     "\nA: {first}\nB: {second}"),
    ("preference", "T5", "polarity",
     'Given the question "{query}", pick the worse response. Your answer must '
     "be A or B.\nA: {first}\nB: {second}"),
    ("preference", "T3", "construct",
     'Question: "{query}". Choose the more relevant response. Answer A or B.'
     "\nA: {first}\nB: {second}"),
]


@pytest.mark.parametrize("task,tid,check,broken", BROKEN,
                         ids=[f"{t}-{i}-{c}" for t, i, c, _ in BROKEN])
def test_checker_catches_injected_defect(task, tid, check, broken):
    templates = code_templates(task)
    templates[tid] = broken
    findings = vp2.check_templates(task, templates)[2]
    assert fails_of(findings, check), (
        f"{task}/{tid}: injected {check} defect was NOT caught; the "
        f"corresponding invariant test is vacuous. Findings: "
        f"{[(f.level, f.check) for f in findings]}")


def test_duplicate_templates_are_caught():
    templates = code_templates("preference")
    templates["T2"] = templates["T1"]
    findings = vp2.check_templates("preference", templates)[2]
    assert fails_of(findings, "non_triviality")


# ══════════════════════════════════════════════════════════════════════════════
# The audit must stay offline
# ══════════════════════════════════════════════════════════════════════════════

def test_validator_makes_no_network_calls():
    """The v2 validator must be runnable while judge sweeps are in flight, so it
    may not import any client that could reach the network."""
    source = (REPO / "scripts" / "validate_paraphrases_v2.py").read_text(encoding="utf-8")
    for banned in ("import openai", "from openai", "import requests", "import httpx",
                   "import urllib.request", "import anthropic"):
        assert banned not in source, f"validator imports {banned!r}"
