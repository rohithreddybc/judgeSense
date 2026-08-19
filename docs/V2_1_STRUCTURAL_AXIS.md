# JudgeSense v2.1 — Structural Paraphrase Axis (Design)

Design-stage document. Extends `docs/V2_ARCHITECTURE.md`; nothing here is
implemented yet. Addresses reviewer WjHn's W2, their stated core concern and
the largest unaddressed review item:

> "The benchmark excludes chain-of-thought, task-specific system prompts,
> role-priming, and structured outputs, testing only instruction-only
> rewordings... Real-world evaluation instructions vary far more
> substantially, with significant variation hidden in system prompts,
> formatting guidelines, and structured output constraints."

## 1. Semantic contract

A **structural paraphrase** changes the *scaffolding* of a judge prompt —
where the instruction lives, what persona frames it, whether reasoning is
elicited, how the answer must be formatted — while the evaluative question
and the label space are held byte-identical.

Instruction-level paraphrases are plausibly meaning-preserving, so
disagreement between them is defensibly judge instability. Structural
variants are not: chain-of-thought changes the computation, and an expert
persona changes the implied criteria. Treating those disagreements as
"instability" would measure that CoT is a different task.

This is resolved by refusing to put all structural variants under one
metric. Variants are pre-registered into two classes:

- **Class E (equivalence-preserving).** A competent human evaluator given
  either arm would understand themselves to be answering the same question
  by the same criteria. JSS and kappa retain their meaning: *the verdict
  should not depend on this.*
- **Class N (interventions).** The arm plausibly changes the judgment
  process. Disagreement is **not** instability and is never called JSS.
  These get **Structural Shift Rate (SSR)** — the fraction of items whose
  verdict moves relative to the S0 baseline, with directional
  decomposition. The claim is *this intervention moves verdicts this much,
  in this direction*: a practical-sensitivity finding assuming no
  equivalence.

Nothing is excluded from the dataset; the split governs only which metric a
variant may feed. Pooling Class N into JSS is the one thing this design
forbids — a reviewer's rebuttal ("CoT is a different task") would be
correct. All four dimensions WjHn requested are implemented; two cannot
support an equivalence claim and are reported honestly under a different
name.

**Scope refusal.** Full task-specific rubric system prompts
(production-style multi-paragraph rubrics) are *not* implemented. A rubric
defines new evaluation criteria — it changes the construct rather than the
phrasing, and its variant space is unbounded. System-prompt *placement* and
*persona* are the tractable structural components; the rubric case is stated
as future work.

## 2. Variant taxonomy

Baseline **S0** = the item's canonical instruction template, direct answer,
user message only (the existing v2 format). Each variant wraps S0's
evaluative sentence verbatim.

| ID | Variant | Class | Form |
|---|---|---|---|
| S1 | Structured output | E | Same label space emitted as JSON; a system instruction requiring a single JSON object with one `verdict` field drawn from the unchanged label set, plus the unchanged S0 question. Bijective `label_map` recorded per row; deterministic parse. |
| S2 | System-prompt relocation | E | S0 instruction moved word-for-word into the system message; user message carries only the item text. |
| S3 | Neutral role prime | E (borderline) | `You are an impartial evaluator.` in the system message. No expertise claim, no added criteria. |
| S4 | Chain-of-thought | N | Step-by-step reasoning requested, final answer required on the last line under a fixed `FINAL:` marker. Missing marker = UNCLEAR under strict mode. |
| S5 | Expert persona | N | Task-specific, e.g. factuality: `You are a senior fact-checker at a major newsroom.` |

**S3 demotion rule.** If the pre-run equivalence audit (the same gate that
vets instruction templates) finds S3 introduces criteria, it demotes to
Class N *before* any judging. Results are reported with and without S3 in
pooled-E either way.

**Polarity guard.** Variants are wrappers around a frozen evaluative
sentence. A unit test asserts label sets and label-direction words are
identical across all six arms, so the v1 Template-4 defect class cannot
recur; CI checks the `label_map` bijection on every row.

## 3. Experimental design

A **star design anchored at S0**, not a factorial.

Per task: a deterministic stratified subsample of **100 of the 250 items**
(label balance preserved; same `item_id`s, provenance untouched). Each
sampled item contributes **five structural pairs** — (S0,S1) … (S0,S5) — all
rendered from that item's existing canonical template.

The instruction × structure cross is deliberately *not* run within-item.
Because the canonical template rotates across items under the existing
assignment, the interaction is covered marginally across items at zero extra
cost.

**One row** = (item, structural_pair, ab_order), carrying `decision_s0` and
`decision_sk`. Pairwise tasks keep both orderings: 100 items × 5 pairs × 2
orders = 1,000 rows. Pointwise: 500 rows. No content duplication — the five
pairs per item are distinct comparisons sharing one S0 call, exactly as swap
rows share an item. The clustering contract, not row count, carries the
statistics.

**Repeat baseline.** S0 is issued twice per item. `JSS_rep` (same prompt,
two calls) is the noise ceiling every structural comparison is read against.
v1 had no such control and so could never separate prompt sensitivity from
decoding variance.

## 4. Clustering and metrics

`CLUSTER_UNITS` extends to `("row", "structural_pair", "prompt_pair", "item")`.

Nesting: `item` contains either `prompt_pair` (instruction axis) or
`structural_pair` (structural axis), each of which contains rows
(orderings).

**Default reporting unit: `item`** — and on this axis it is mandatory, not
merely conservative. All five pairs for an item share the S0 arm, so errors
correlate within item by construction; resampling `structural_pair` would
repeat the v1 independence mistake one level up.

Metrics:

- **Class E**, per variant (S1, S2, S3 vs S0) and pooled-E: JSS,
  chance-corrected kappa, strict mode, QWK for coherence — each reported
  alongside `JSS_rep`.
- **Class N**, per variant: SSR with directional decomposition (net shift
  per label, sign test; for coherence, mean Likert shift with item-clustered
  CI).
- **Format fragility as first-class numbers**: S4 parse-failure rate, S1
  JSON-malformation rate.

## 5. Budget

Naive full factorial — 5 templates × 5 variants × 250 items × 13 judges ×
(2 pointwise + 2 pairwise × 2 orders) ≈ **487,500 calls**. Infeasible.

This design, per judge:

| Task type | Arithmetic | Calls |
|---|---|---|
| Pointwise (×2 tasks) | 100 items × (1 S0 + 5 variants + 1 repeat) | 700 each |
| Pairwise (×2 tasks) | 700 × 2 orderings | 1,400 each |
| **Per judge** | 2×700 + 2×1,400 | **4,200** |

Judges: **6 of 13**, pre-registered for coverage (2 frontier, 2 mid, 2
small; families disjoint where possible).

**Total = 25,200 calls, roughly 0.65× the v1 run.**

Reductions by leverage: freeze instruction template within item (25x to 1x);
star rather than full pairwise among arms (15 to 5 pairs); S0 shared across
pairs (10 to 6 calls); 100 of 250 items; 6 of 13 judges. S4 outputs cost
roughly 3–5× the tokens, so dollar budget should assume ~1.3× the per-call
average.

**Power.** N_eff = 100 items per (task, variant, judge); 95% CI half-width
≈ ±0.10 on JSS — adequate for the 0.15–0.3 gaps v1 observed. The audit
gate's effective-sample-size check is declared at `cluster_unit="item"`.

## 6. Risks and objections

- **"S3 isn't neutral."** Strongest objection; partially lands. Mitigated by
  the pre-run equivalence audit and demotion rule, and by reporting pooled-E
  with and without S3. If S3 flips class, Class E still has two clean
  members.
- **"You dodged W2 by excluding CoT from JSS."** Does not land. CoT is
  implemented and measured; the argument is explicitly that calling
  CoT-vs-direct disagreement "instability" is indefensible and SSR is the
  honest quantity. The paper must state this position, not hide it.
- **"Disagreement is just decoding noise."** Answered by design: `JSS_rep`
  is the ceiling and structural effects are reported as deltas from it.
  Residual risk only if providers change models mid-run — pin model
  versions, log timestamps.
- **"No within-item template × structure interaction."** Partially lands.
  Only marginal coverage via template rotation. Acknowledged limitation; a
  full cross is a 25× budget item for a second-order effect.
- **"Shared S0 arm biases pair-level estimates."** True, and why item-level
  clustering is mandatory. Point estimates are unbiased and the CI machinery
  already handles it.
- **Gate compatibility.** Audit checks extend mechanically: uniqueness on
  (`item_id`, `structural_pair_id`, `ab_order`), provenance inherited from
  the parent item, per-arm label non-degeneracy, `label_map` bijection, no
  polarity-word drift, declared `cluster_unit`. Nothing in the gate needs
  weakening.

## 7. Repeat baseline: implemented vs outstanding

`JSS_rep` is backfilled onto the main instruction axis, not only the
structural axis, per the open item this section previously carried. Without
a same-prompt repeat baseline, any JSS number conflates prompt sensitivity
with decoding variance — which applies to the v2 main results exactly as
much as it does here.

**Implemented** (protocol + metrics + tests, no judge calls made):

- **Call-record and repeat-pair contract** (`src/repeat_baseline.py`). A
  runner logs one dict per API call, tagged `repeat_index` (1 or 2) when the
  call is a repeat of the S0/baseline arm; single-shot arms (P1/P2, S1-S5)
  leave it unset. `build_repeat_pairs()` collapses the two S0 calls for an
  item into one decision record (`decision_a`/`decision_b`, `item_id`,
  `repeat_pair_id`, `arm_a`/`arm_b`), rejecting incomplete pairs, duplicate
  `repeat_index`, and `repeat_index` set on a non-S0 arm. The result is
  consumable directly by the existing `jss`/`chance_corrected_jss`/
  `cluster_bootstrap_ci` machinery in `src/metrics_v2.py` — no new cluster
  key was needed, since `item_id` already drives `cluster_unit="item"`.
- **Delta reporting** (`jss_repeat_delta` in `src/metrics_v2.py`). Reports
  JSS, JSS_rep, and `delta = JSS - JSS_rep` together, with a percentile
  bootstrap CI computed by resampling ITEMS jointly — one draw per replicate
  feeds both the paraphrase-arm and repeat-arm point estimates — because a
  paraphrase comparison and its item's repeat comparison share the same S0
  call context and are not independent. `cluster_unit` must be passed
  explicitly (no default) and only `"item"` is accepted for this paired
  computation: the repeat arm has no finer-than-item granularity to pair
  against, so `"row"`/`"prompt_pair"`/`"structural_pair"` are rejected with
  an explanatory error rather than silently faked.
- **Budget stated in advance** (`src/judge_registry.py`). `run_plan()`
  gained an opt-in `repeat_calls_per_judge` parameter (default 0, so every
  prior caller and its numbers are unchanged) that reports
  `calls_per_judge_with_repeat` / `total_calls_with_repeat` alongside the
  existing fields. `main_axis_run_plan()` states the main-axis cost
  concretely: base cost is 1,500 rows × 2 prompt arms = 3,000 calls/judge;
  the repeat baseline adds one S0 call per ITEM (pairwise orderings of the
  same item share one S0 context, exactly as on the structural axis), i.e.
  1,000 extra calls/judge (1,000 unique items: 250 factuality + 250
  coherence + 250 relevance + 250 preference, pairwise item counts halved
  from their row counts). Total across the 14 verified judges: 42,000 calls
  without the repeat arm, 56,000 with it (+14,000, +33%).
- Tests: `tests/test_repeat_baseline.py` (contract) and additions to
  `tests/test_metrics_v2.py` / `tests/test_judge_registry.py` (delta
  computation, paired-vs-independent bootstrap behavior, budget totals), all
  on hand-built fixtures — no dataset rows, no judge calls.

**Outstanding — none of this has been run:**

- No runner change exists yet to actually issue the second S0 call; a
  runner must be extended to emit `repeat_index`-tagged call records for
  `build_repeat_pairs()` to consume, on both axes.
- No repeat-baseline data has been collected for either axis, so there are
  no `JSS_rep` numbers, no deltas, and no filled-in confidence intervals for
  any real judge yet — only the fixture-verified computation exists.
- The structural axis's own repeat arm (§3, "S0 is issued twice per item")
  was already budgeted (§5) but is equally unrun; `jss_repeat_delta` applies
  to it identically once that data exists, keyed the same way by `item_id`.
- Whether providers hold model versions stable across the two calls of a
  repeat pair (§6, "residual risk only if providers change models mid-run")
  is a run-time operational concern this implementation does not address;
  pinning and timestamp logging remain future runner work.
