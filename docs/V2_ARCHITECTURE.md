# JudgeSense v2 Rebuild — Architecture

This document is the reviewable design for the v2 rebuild. It is written
*before* the implementation and is the contract the implementation must
satisfy. It covers: (1) real data sourcing with a per-item provenance chain,
(2) the corrected metrics module, (3) the clustering / unit-of-analysis
contract, (4) the position-bias swap harness, and (5) the CI data-audit gate.

Nothing in this document modifies the paper or any v1 result artifact. All v1
code and data remain in place, untouched, for provenance.

---

## 0. Why a rebuild

An audit of the v1 pipeline (see `analysis/` and the PR description) verified
the following defects, which this design addresses head-on:

| # | Defect (verified in v1) | v2 fix |
|---|---|---|
| D1 | `src/dataset_builder.py` contains **no data-loading code**; every item is a hardcoded Python literal, yet each record carries `source_benchmark` labels ("TruthfulQA", "SummEval", "BEIR", "MT-Bench") | Real loaders (`src/data_sources.py`) with a mandatory per-item provenance chain; loud failure when a source is unreachable |
| D2 | "500 pairs" is 202 unique prompt pairs over 75 unique items; coherence is 5 texts × 25 template pairs duplicated to 125 rows | ≥200 *unique* source items per task; no row duplication; uniqueness enforced by the audit gate |
| D3 | `metrics.py` bootstraps all rows independently although rows are repeated measures nested in items | Cluster bootstrap with an explicit, declared resampling unit (`src/metrics_v2.py`) |
| D4 | Coherence `ground_truth_label` is `f"score_{i+1}"` — a loop index | Real SummEval expert coherence annotations carried per item |
| D5 | Pairwise tasks admit an always-A strategy; JSS can be perfect for a judge that ignores the input | Swap harness: every pairwise item is presented in both orderings; position-bias-corrected JSS is 0 for an always-A judge by construction |
| D6 | corr(JSS, entropy) = −0.484: raw JSS partly rewards output-distribution compression | Chance-corrected JSS + per-judge entropy diagnostics reported alongside raw JSS |
| D7 | Substring-based parsing (`normalize_decision`) can manufacture agreement (e.g. the English article "A" matches `\b([AB])\b`) | Parser audit (`analysis/parser_audit.py`); v2 scoring counts UNCLEAR as disagreement in strict mode |
| D8 | Reported counts (9 vs 13 judges, 494 vs 500 pairs) drift between documents | Single machine-generated source of truth: `data/counts.json` |

## 1. Data sourcing (`src/data_sources.py`, `src/dataset_builder_v2.py`)

### 1.1 Sources

All four tasks load from public datasets on the Hugging Face Hub via the
`datasets` library. No item text and no label may originate in this
repository's source code.

| Task | HF dataset | Config / split | Item construction | Ground truth |
|---|---|---|---|---|
| factuality | `truthful_qa` | `generation` / `validation` | one *accurate* item per question (`best_answer`) and one *inaccurate* item (`incorrect_answers[0]`), phrased as "Q → A" statements | accurate / inaccurate, from TruthfulQA's own labeling |
| coherence | `mteb/summeval` | `test` | (source text, machine summary) pairs; each of the 100 documents has 16 machine summaries with expert coherence ratings | mean expert coherence rating (1–5 scale), carried both raw (float) and rounded to the nearest integer |
| relevance | `BeIR/scifact` + `BeIR/scifact-qrels` | `corpus`, `queries` / `train` | query + (relevant doc from qrels, non-relevant doc sampled deterministically from the corpus); both document ids recorded | which candidate is qrels-relevant |
| preference | `lmsys/mt_bench_human_judgments` | `human` | turn-1 response pairs (model_a vs model_b) for a question; tie votes excluded | the human judgment (`winner`) recorded in the source |

Notes:
- The relevance pairing (relevant vs sampled non-relevant) is a *constructed
  pairing of real documents* — both texts and the relevance label are real;
  the pairing procedure is deterministic (seeded) and recorded per item.
- For preference, items are keyed by (question_id, model_a, model_b) and the
  label is a real human vote from the source dataset. Where multiple human
  votes exist for a pair, the majority vote is used and the vote count is
  recorded; pairs with no majority are excluded (and counted).

### 1.2 Provenance chain (mandatory per item)

Every emitted record carries a `source` object:

```json
{
  "source_dataset": "truthful_qa",
  "source_config": "generation",
  "source_split": "validation",
  "source_record_id": "validation[314]",
  "source_fields": {"question": "...", "answer_field": "best_answer"},
  "retrieved_at": "<ISO-8601 UTC>",
  "loader_version": "2.0.0"
}
```

`source_record_id` must identify a specific record in the specific split of
the specific dataset — a constant string naming a benchmark (v1 behavior)
does not satisfy this and is rejected by the audit gate.

### 1.3 Failure policy — no silent fallback, ever

If a dataset cannot be loaded (network policy, missing credentials, schema
drift, API failure), the loader raises `DataSourceUnavailableError` with the
underlying cause. There is **no** fallback path to synthetic, cached-in-code,
or placeholder items. This is deliberate and load-bearing: the v1 crisis was
caused by exactly such a fallback-shaped shortcut. A build with no data is a
correct outcome; a build with invented data is not.

### 1.4 Pair construction

- 5 judge-prompt templates per task (semantically equivalent phrasings; no
  polarity inversion — the v1 T4 factuality artifact class is excluded by
  construction and checked by a unit test).
- Each unique item is assigned template pairs by deterministic rotation over
  the 10 unordered template combinations, one prompt pair per item by
  default (maximizing unique items per row, the opposite of v1's
  duplication).
- `semantic_equivalence_score` is only emitted if a verifier actually ran;
  otherwise the field is absent. It is never hardcoded to 1.0.
- Target: ≥200 unique items per task (audit-gated).

### 1.5 v1 builder

`src/dataset_builder.py` is renamed (with history) to
`src/dataset_builder_v1_DEPRECATED.py` and kept intact for provenance, with a
deprecation header explaining why it must not be used. v1 data files under
`data/prompt_pairs/` are left byte-identical.

## 2. Metrics (`src/metrics_v2.py`)

`metrics.py` (v1) is left untouched so published numbers remain reproducible.
The v2 module is a superset with corrected statistics.

### 2.1 Unit-of-analysis contract

Every record used for scoring carries cluster keys: `item_id`,
`prompt_pair_id` (template combination applied to an item), and row identity.
Every CI-producing function requires an explicit `cluster_unit` argument —
one of `"row"`, `"prompt_pair"`, `"item"` — and reports it in its output.
There is no default that silently assumes independence; callers must declare
the unit, and the declared unit is part of the result payload so downstream
tables cannot drop it.

Rationale: v1 resampled 375 rows independently, but rows are repeated
measures (same item × several template pairs × 3 runs). The audit found
cluster-correct CIs are 2.8× (prompt-pair) to 3.9× (item) wider.

### 2.2 Cluster bootstrap

`cluster_bootstrap_ci(records, metric_fn, cluster_unit, n_bootstrap, seed)`:
resample *clusters* with replacement (as many clusters as observed), take all
rows of each sampled cluster (with multiplicity), apply `metric_fn`,
percentile CI. With `cluster_unit="row"` it reproduces the naive bootstrap
(unit-tested equivalence), making the v1↔v2 difference attributable to
clustering alone.

### 2.3 Metric suite

- `jss` — raw agreement, kept for continuity.
- `chance_corrected_jss` — Cohen's-kappa-style correction using the judge's
  own marginal decision distributions under variants A and B. A judge that
  outputs one label always gets 0, not 1.
- Ordinal agreement for Likert (coherence): quadratic-weighted kappa and
  mean absolute difference, so a 3↔4 flip is not scored like a 1↔5 flip.
- `unclear_policy` on all agreement metrics: `"drop"` (v1 behavior,
  reproducibility) and `"disagree"` (strict mode — malformed/UNCLEAR output
  counts as disagreement). Both are reported; strict is the headline in v2.
  (Motivation: Mistral-7B emits 15.7% UNCLEAR; dropping it inflates JSS.)
- Diagnostics: per-judge label histograms, Shannon entropy of the decision
  distribution, and corr(JSS, entropy) across judges (Pearson + Spearman),
  so the compression-reward failure mode (D6) is visible in every report.

## 3. Position-bias swap harness (`src/swap_harness.py`)

Design fix, not post-hoc correction. For pairwise tasks each (item, template
pair) yields **four** presentations: {template A, template B} × {original
order, swapped order}. Records carry `presentation_id`, `ab_order`
(`original` | `swapped`), and the mapping from displayed position → underlying
candidate id.

Scoring: a judge's positional decision is mapped to a *content-level*
decision (which underlying candidate it chose). For a given template, the
content decision is defined only if the judge picks the same underlying
candidate in both orderings; otherwise it is `POSITION_INCONSISTENT` and is
scored as disagreement. Position-bias-corrected JSS is agreement between the
content-level decisions of template A and template B.

Consequences (unit-tested): an always-A judge has corrected JSS = 0; a
faithful judge is unaffected; position-bias rate (fraction of items where the
judge follows position rather than content) is reported per judge.

## 4. Parser + prompt sanity audit (`analysis/parser_audit.py`)

Static + empirical audit of `normalize_decision` and prompt assembly:

- Enumerates the exact mapping rules and their failure modes with labeled
  probe strings (probes are code-behavior demonstrations, clearly marked;
  they are not model outputs and are never written into any dataset file).
- Runs over any real raw outputs present in `data/results/raw_outputs/`
  and tabulates raw → normalized mappings; states explicitly when no raw
  outputs are available.
- Audits the shipped prompt-pair files: candidate ordering between variants,
  ground-truth distribution, `ab_swapped` coverage — to locate where
  always-A can originate (prompt construction vs parsing vs judge behavior).

## 5. CI data-audit gate (`scripts/data_audit.py` + workflow)

A first-class deliverable: the gate makes the v1 failure mode structurally
non-repeatable. `scripts/data_audit.py` reads a config
(`data/audit_config.json`) declaring the dataset directory, the clustering
unit, and thresholds, and **exits non-zero** if any check fails:

| Check | Fails when |
|---|---|
| `unique_items` | unique items per split < threshold |
| `duplicate_rows` | duplicate-row ratio > threshold |
| `provenance` | any record's `source_benchmark`/`source_dataset` claim is not backed by a per-item `source_record_id` (+ split/config) |
| `label_degeneracy` | a single label exceeds a share threshold, or fewer than 2 labels observed in a classification split |
| `effective_sample_size` | number of clusters at the declared unit < floor |
| `annotation_timing` | median seconds-per-decision in human-validation files < floor |

Every check has a dedicated unit test (fixtures live in `tests/`, are
clearly labeled as test fixtures, and are never written under `data/`).
The GitHub Actions workflow runs (a) the unit-test suite and (b) the audit
gate against the dataset the repository currently ships. **The gate is
expected to fail on the v1 data** — that is the gate demonstrating the
defects it was built to catch; it must turn green only when a real v2
dataset is built and committed.

## 6. Counts source of truth (`scripts/generate_counts.py` → `data/counts.json`)

All reported counts are computed from the data actually on disk plus the code
registries (`SUPPORTED_MODELS`, `EXCLUDED_PAIRS`), never typed by hand:
rows / unique items / unique prompt pairs per task, judge count, excluded
pairs, post-exclusion pair count, expected API-call counts per pass.
Documents must consume this file rather than restating numbers. (The
9-vs-13 judges and 494-vs-500 pairs drift is exactly what this eliminates:
500 rows − 6 excluded pairs = 494; 9 = pass-1 judges, 13 = pass-2 registry.)

## 7. Croissant metadata (`scripts/generate_croissant_v2.py`)

v2 Croissant metadata is *generated from the built data*, never hand-edited,
and includes an accurate `rai:hasSyntheticData`. For the v2 design this is
`true` with an explanation: the responses being judged include
model-generated text from the source benchmarks (SummEval machine summaries,
MT-Bench model responses), and judge prompts are author-written templates;
item texts and labels are drawn verbatim from the sources. The generator
refuses to run when the built data is absent (no metadata without data).
The v1 croissant file is left untouched; note that its
`rai:hasSyntheticData: false` is inaccurate for v1 (whose items were
hand-authored in code) — flagged in the PR, not silently rewritten.

## 8. Out of scope for this rebuild

- Re-running the ~39k judge API calls (requires provider keys and budget).
- Human semantic-equivalence annotation of v2 prompt pairs (requires real
  annotators; v2 therefore emits no `semantic_equivalence_score` until a
  verification pass actually runs).

Both are tracked as follow-ups in the PR description.
