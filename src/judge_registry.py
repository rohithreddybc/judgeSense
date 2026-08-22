"""
JudgeSense v2 judge registry — families, sizes, and matched token budgets.

`src/models.py` (v1) is left untouched so published runs stay reproducible.
This module adds the metadata three reviewer points require, none of which the
v1 registry carried:

  xmQT W1, qkzU Q3   token budgets differed by judge class (20 for
                     instruction-tuned, 1024 for reasoning-tuned), so any
                     difference between those classes confounds architecture
                     with inference configuration. Fixed by making the budget an
                     explicit run-level policy, so a matched-budget control is a
                     configuration rather than a code change.

  WjHn W5, Q4        scale effects were compared across unrelated architectures
                     and training regimes. Fixed by declaring `family` and
                     `size_b`, and exposing the within-family ladders that make
                     a scale comparison meaningful.

  xmQT Limitations   no purpose-built judge models were evaluated, though those
                     are built precisely to resist the failures under study.

`pinned` records whether the model id is a DATED SNAPSHOT or a floating alias.
Most provider aliases resolve to whatever the vendor currently serves, so a
replication cannot be guaranteed to run the same weights. Where a dated snapshot
exists it is used; where it does not, `pinned=False` marks the entry and the
runner records the provider-echoed model string on every call so drift is
detectable after the fact. A scale or family claim must not rest on an unpinned
entry without saying so.

`verified` records whether a checkpoint identifier has been confirmed against
the provider. Entries added from documentation but not yet exercised are marked
False and are excluded from run selection unless explicitly requested — an
unverified model id must fail loudly at selection time rather than at call time
in the middle of a paid run.
"""

from __future__ import annotations

from typing import Dict, List, Optional

REASONING = "reasoning"
INSTRUCT = "instruct"
PURPOSE_BUILT = "purpose_built"

# Budget policies. "native" reproduces the v1 asymmetry; "matched" is the
# control the reviewers asked for.
BUDGET_POLICIES = ("native", "matched")
MATCHED_BUDGET_TOKENS = 1024


JUDGES: Dict[str, dict] = {
    # ── instruction-tuned ───────────────────────────────────────────────────
    "gpt-4o-mini": dict(provider="openai", model_id="gpt-4o-mini-2024-07-18",
                        key="OPENAI_API_KEY", kind=INSTRUCT, family="gpt-4o",
                        size_b=None, native_max_tokens=20, verified=True, pinned=True),
    "gpt-4o": dict(provider="openai", model_id="gpt-4o-2024-08-06",
                   key="OPENAI_API_KEY", kind=INSTRUCT, family="gpt-4o",
                   size_b=None, native_max_tokens=20, verified=True, pinned=True),
    "claude-haiku": dict(provider="anthropic", model_id="claude-haiku-4-5-20251001",
                         key="ANTHROPIC_API_KEY", kind=INSTRUCT, family="claude-4-5",
                         size_b=None, native_max_tokens=20, verified=True, pinned=True),
    "claude-sonnet": dict(provider="anthropic", model_id="claude-sonnet-4-5",
                          key="ANTHROPIC_API_KEY", kind=INSTRUCT, family="claude-4-5",
                          size_b=None, native_max_tokens=20, verified=True, pinned=False),
    "gemini-flash": dict(provider="google", model_id="gemini-2.5-flash",
                         key="GOOGLE_API_KEY", kind=INSTRUCT, family="gemini-2.5",
                         size_b=None, native_max_tokens=20, verified=True, pinned=False),
    "llama3-8b": dict(provider="huggingface", model_id="meta-llama/Llama-3.1-8B-Instruct",
                      key="HF_TOKEN", kind=INSTRUCT, family="llama-3.1",
                      size_b=8, native_max_tokens=20, verified=True, pinned=False),
    "llama3-70b": dict(provider="groq", model_id="llama-3.1-70b-versatile",
                       key="GROQ_API_KEY", kind=INSTRUCT, family="llama-3.1",
                       size_b=70, native_max_tokens=20, verified=True, pinned=False),
    # NOT a 7B model. "mistral-small-latest" is a floating alias that does not
    # resolve to a 7B checkpoint, and size_b feeds family_ladders, so a scale
    # claim would have been built on a parameter count the name asserted and the
    # checkpoint did not have. Renamed, and size_b is None until a versioned
    # checkpoint with a published parameter count is pinned.
    "mistral-small": dict(provider="mistral", model_id="mistral-small-latest",
                          key="MISTRAL_API_KEY", kind=INSTRUCT, family="mistral",
                          size_b=None, native_max_tokens=20, verified=True,
                          pinned=False),
    "qwen": dict(provider="novita", model_id="qwen/qwen-2.5-72b-instruct",
                 key="NOVITA_API_KEY", kind=INSTRUCT, family="qwen-2.5",
                 size_b=72, native_max_tokens=20, verified=True, pinned=False),

    # ── reasoning-tuned ─────────────────────────────────────────────────────
    "deepseek": dict(provider="novita", model_id="deepseek/deepseek-r1",
                     key="NOVITA_API_KEY", kind=REASONING, family="deepseek-r1",
                     size_b=None, native_max_tokens=1024, verified=True, pinned=False),
    "deepseek-v4-flash": dict(provider="novita", model_id="deepseek/deepseek-v4-flash",
                              key="NOVITA_API_KEY", kind=REASONING, family="deepseek-v4",
                              size_b=None, native_max_tokens=1024, verified=True, pinned=False),
    "gpt-5.5": dict(provider="openai", model_id="gpt-5.5",
                    key="OPENAI_API_KEY", kind=REASONING, family="gpt-5",
                    size_b=None, native_max_tokens=1024, verified=True, pinned=False),
    "claude-opus-4-7": dict(provider="anthropic", model_id="claude-opus-4-7",
                            key="ANTHROPIC_API_KEY", kind=REASONING, family="claude-4-7",
                            size_b=None, native_max_tokens=1024, verified=True, pinned=False),
    "qwen-3.6-flash": dict(provider="dashscope", model_id="qwen3.6-35b-a3b",
                           key="DASHSCOPE_API_KEY", kind=REASONING, family="qwen-3.6",
                           size_b=35, native_max_tokens=1024, verified=True, pinned=False),

    # ── purpose-built judges (xmQT Limitations) ─────────────────────────────
    # Model identifiers taken from published model cards and NOT yet exercised
    # against the provider, so they are marked unverified and are excluded from
    # selection until confirmed. Verify before spending a run on them.
    "prometheus-2-7b": dict(provider="huggingface", model_id="prometheus-eval/prometheus-7b-v2.0",
                            key="HF_TOKEN", kind=PURPOSE_BUILT, family="prometheus-2",
                            size_b=7, native_max_tokens=1024, verified=False, pinned=False),
    "prometheus-2-8x7b": dict(provider="huggingface", model_id="prometheus-eval/prometheus-8x7b-v2.0",
                              key="HF_TOKEN", kind=PURPOSE_BUILT, family="prometheus-2",
                              size_b=47, native_max_tokens=1024, verified=False, pinned=False),
    "nemotron-70b": dict(provider="huggingface", model_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                         key="HF_TOKEN", kind=PURPOSE_BUILT, family="nemotron",
                         size_b=70, native_max_tokens=1024, verified=False, pinned=False),
}


class RegistryError(ValueError):
    """Raised on an unknown judge or an unusable selection."""


def max_tokens_for(judge: str, budget_policy: str = "native") -> int:
    """
    Token budget for one judge under a run-level policy.

    "native"  reproduces the v1 setting (20 for instruction-tuned, 1024 for
              reasoning-tuned) and therefore reproduces its confound.
    "matched" gives every judge MATCHED_BUDGET_TOKENS, so a difference between
              judge classes can no longer be explained by the budget. This is
              the control xmQT W1 and qkzU Q3 asked for.
    """
    if judge not in JUDGES:
        raise RegistryError(f"unknown judge {judge!r}")
    if budget_policy not in BUDGET_POLICIES:
        raise RegistryError(
            f"unknown budget_policy {budget_policy!r}; expected one of {BUDGET_POLICIES}"
        )
    if budget_policy == "matched":
        return MATCHED_BUDGET_TOKENS
    return JUDGES[judge]["native_max_tokens"]


def family_ladders(min_rungs: int = 2, verified_only: bool = True) -> Dict[str, List[str]]:
    """
    Within-family judge groups differing only in parameter count.

    A scale claim needs these: comparing a 7B of one family against a 70B of
    another confounds size with architecture and training data, which is what
    WjHn W5 objected to. Only families with a declared `size_b` on at least
    `min_rungs` members qualify.
    """
    grouped: Dict[str, List[str]] = {}
    for name, spec in JUDGES.items():
        if verified_only and not spec["verified"]:
            continue
        if spec["size_b"] is None:
            continue
        grouped.setdefault(spec["family"], []).append(name)
    return {
        family: sorted(members, key=lambda n: JUDGES[n]["size_b"])
        for family, members in grouped.items()
        if len(members) >= min_rungs
    }


def reasoning_judges(verified_only: bool = True) -> List[str]:
    """
    Reasoning-tuned judges available.

    WjHn W3 objected that the "reasoning traces reduce consistency" claim rested
    on DeepSeek-R1 alone. That was a claim scoped to one model, not a shortage of
    models — the registry carries several, so the fix is to test across them or
    drop the claim.
    """
    return sorted(
        name for name, spec in JUDGES.items()
        if spec["kind"] == REASONING and (spec["verified"] or not verified_only)
    )


def purpose_built_judges(verified_only: bool = True) -> List[str]:
    """Judges trained specifically for evaluation (xmQT Limitations)."""
    return sorted(
        name for name, spec in JUDGES.items()
        if spec["kind"] == PURPOSE_BUILT and (spec["verified"] or not verified_only)
    )


# Pre-registered subset for the structural axis: 2 frontier, 2 mid, 2 small,
# families disjoint. Fixed here rather than chosen after seeing results.
STRUCTURAL_AXIS_JUDGES = (
    "gpt-5.5", "claude-opus-4-7",   # frontier
    "gpt-4o", "qwen",               # mid
    "mistral-small", "llama3-8b",   # small
)


def select_judges(names: Optional[List[str]] = None, allow_unverified: bool = False) -> List[str]:
    """
    Resolve a judge selection, rejecting unknown or unverified entries.

    Unverified checkpoints fail here rather than mid-run: discovering a bad model
    id after paying for half a sweep is the expensive way to learn it.
    """
    selection = list(names) if names is not None else [
        n for n, s in JUDGES.items() if s["verified"]
    ]
    unknown = [n for n in selection if n not in JUDGES]
    if unknown:
        raise RegistryError(f"unknown judge(s): {sorted(unknown)}")
    if not allow_unverified:
        unverified = [n for n in selection if not JUDGES[n]["verified"]]
        if unverified:
            raise RegistryError(
                f"unverified model id(s): {sorted(unverified)}. Confirm the "
                "checkpoint against the provider and set verified=True, or pass "
                "allow_unverified=True to run them deliberately."
            )
    if not selection:
        raise RegistryError("empty judge selection")
    return selection


def run_plan(n_calls_per_judge: int, judges: Optional[List[str]] = None,
             budget_policy: str = "native", repeat_calls_per_judge: int = 0) -> dict:
    """
    Call-count plan for a sweep, so budget is stated before it is spent rather
    than discovered afterwards.

    `repeat_calls_per_judge` states the cost of the same-prompt repeat
    baseline in advance: one extra S0 call per (judge, item)
    (docs/V2_1_STRUCTURAL_AXIS.md §7, `src/repeat_baseline.py`). It defaults
    to 0, so existing callers that pre-compute `n_calls_per_judge` themselves
    (e.g. the structural axis, which already bakes its "+1 repeat" into the
    700/1,400 arithmetic in §5) are unaffected — `calls_per_judge` and
    `total_calls` keep their exact prior meaning and value. When set, the
    plan additionally reports the repeat-inclusive totals so both numbers are
    visible side by side rather than one silently replacing the other.
    """
    selection = select_judges(judges)
    n_judges = len(selection)
    calls_with_repeat = n_calls_per_judge + repeat_calls_per_judge
    return {
        "judges": selection,
        "n_judges": n_judges,
        "calls_per_judge": n_calls_per_judge,
        "total_calls": n_calls_per_judge * n_judges,
        "budget_policy": budget_policy,
        "max_tokens": {j: max_tokens_for(j, budget_policy) for j in selection},
        "repeat_calls_per_judge": repeat_calls_per_judge,
        "calls_per_judge_with_repeat": calls_with_repeat,
        "total_calls_with_repeat": calls_with_repeat * n_judges,
    }


# ── Main instruction-axis dataset shape ──────────────────────────────────────
# 1,452 rows = 250 factuality + 250 coherence + 500 relevance + 452 preference.
# Pairwise tasks (relevance, preference) carry both candidate orderings, so
# their row count is 2x their item count; pointwise tasks (factuality,
# coherence) are 1 row per item. Two prompt arms (P1/P2 paraphrases) are
# issued per row for the existing JSS computation.
#
# Preference ships 226 items rather than 250 because the split is held at an
# exact 50/50 winner-longer balance and the human-labelled pool's smaller length
# bucket holds only 113 pairs. See load_preference_items in src/data_sources.py.
#
# These counts drive the printed call budget, so they must never drift from the
# files actually on disk; tests/test_registry_matches_dataset.py asserts they
# agree with data/v2/*.jsonl. A stale constant here understates the spend.
MAIN_AXIS_ROWS_PER_TASK: Dict[str, int] = {
    "factuality": 250,
    "coherence": 250,
    "relevance": 500,
    "preference": 452,
}
MAIN_AXIS_PAIRWISE_TASKS = ("relevance", "preference")
MAIN_AXIS_PROMPT_ARMS_PER_ROW = 2
# The repeat baseline re-issues BOTH prompt arms once per item, not one arm.
# A ceiling measured under a single template cannot absorb noise generated under
# the other, so that noise would be charged to paraphrasing. This constant is
# what the printed budget is built from; leaving it at 1 after the runner began
# issuing two understated the spend by one call per item per judge.
MAIN_AXIS_REPEAT_ARMS_PER_ITEM = 2

MAIN_AXIS_TOTAL_ROWS = sum(MAIN_AXIS_ROWS_PER_TASK.values())
# unique items: pairwise rows fold 2 orderings into 1 item; pointwise rows are
# already 1 row per item.
MAIN_AXIS_TOTAL_ITEMS = sum(
    n // 2 if task in MAIN_AXIS_PAIRWISE_TASKS else n
    for task, n in MAIN_AXIS_ROWS_PER_TASK.items()
)


def main_axis_run_plan(judges: Optional[List[str]] = None, budget_policy: str = "native",
                        include_repeat_baseline: bool = True) -> dict:
    """
    Call-count plan for the main instruction axis (not the structural axis,
    which has its own budget in docs/V2_1_STRUCTURAL_AXIS.md §5).

    Base cost is `MAIN_AXIS_TOTAL_ROWS * MAIN_AXIS_PROMPT_ARMS_PER_ROW` calls
    per judge (1,452 rows x 2 arms = 2,904). The repeat baseline adds one S0
    call per ITEM, not per row: pairwise rows for the same item share a
    single canonical S0 context, so the repeat call is item-scoped exactly
    like the structural axis's shared S0 arm (docs/V2_1_STRUCTURAL_AXIS.md
    §3-4). With `include_repeat_baseline=True` (default) that is
    `MAIN_AXIS_TOTAL_ITEMS` (976) extra calls per judge.
    """
    base_calls = MAIN_AXIS_TOTAL_ROWS * MAIN_AXIS_PROMPT_ARMS_PER_ROW
    repeat_calls = (MAIN_AXIS_TOTAL_ITEMS * MAIN_AXIS_REPEAT_ARMS_PER_ITEM
                    if include_repeat_baseline else 0)
    plan = run_plan(base_calls, judges, budget_policy, repeat_calls_per_judge=repeat_calls)
    plan["dataset"] = {
        "rows_per_task": dict(MAIN_AXIS_ROWS_PER_TASK),
        "total_rows": MAIN_AXIS_TOTAL_ROWS,
        "total_items": MAIN_AXIS_TOTAL_ITEMS,
        "prompt_arms_per_row": MAIN_AXIS_PROMPT_ARMS_PER_ROW,
        "include_repeat_baseline": include_repeat_baseline,
    }
    return plan
