# Errata — JudgeSense v1 dataset and results

**Status: the v1 dataset should not be used.** The accompanying paper was
withdrawn from NeurIPS 2026 (Evaluations and Datasets track) in July 2026 after
the defects below were verified against the released artifact.

This document records what is wrong, with the numbers, so that anyone who has
downloaded the dataset or cited the results can assess the impact. Every figure
here is reproducible from the released files using `scripts/data_audit.py`.

---

## 1. The dataset is smaller than described

It is described as 500 hand-validated prompt pairs. It contains **500 rows, but
202 unique prompt pairs over 75 unique underlying items**:

| Task | Rows | Unique items | Unique prompt pairs |
|---|---:|---:|---:|
| Factuality | 125 | 60 | 80 |
| **Coherence** | 125 | **5** | **25** |
| Relevance | 125 | 5 | 49 |
| Preference | 125 | 5 | 48 |
| **Total** | **500** | **75** | **202** |

Coherence carries the headline finding (the JSS range 0.39–0.99 that separates
judges) and rests on **five texts**. Rows are duplicated within each task, and
each was then evaluated in 3 runs and reported as N=375 per judge/task cell.

## 2. Source provenance is not accurate

Every record carries a `source_benchmark` field naming TruthfulQA, SummEval,
BEIR, or MT-Bench. Those strings are **hardcoded constants**. The dataset
builder contains no data-loading code of any kind — no `load_dataset`, no
download, no read from any source file. Every item is a Python literal written
in the builder.

The items are therefore **not drawn from those benchmarks**, and the fields
should not be relied on as provenance.

## 3. Coherence ground-truth labels are not coherence judgments

The `ground_truth_label` for coherence is generated as the enumeration index of
the item (`score_1` through `score_5`), not a rating of the text. Read
literally the labels run close to inverted: the least coherent text in the set
is assigned the highest score.

These labels do not enter the JSS computation — JSS measures agreement between
two prompt phrasings and never consults the ground truth — so the published JSS
numbers are not contaminated by this. But the field is unusable for any accuracy
evaluation, which is a plausible reason to reuse the dataset.

## 4. The second-annotator claim is not supported

The README, the dataset card, and the paper appendix state that all pairs were
validated by a primary annotator and independently re-reviewed by a second
annotator with full agreement.

The repository contains **500 annotation records, all carrying a single
annotator** (`reviewer: "rohit"`). No second annotator's records exist in the
repository. The claim of independent re-review is not supported by the artifact
and is withdrawn.

Annotation timing is also uneven. Per-record UTC timestamps show the factuality
session took 101 minutes and produced 50 non-equivalent labels, consistent with
genuine review. The other three sessions were all-YES at a median of 0.7–1.8
seconds per decision (relevance: 125 decisions in 3.2 minutes), which is not
consistent with reading and judging two prompts per decision.

## 5. Ten items have contradictory ground truth

Identical judged content appears with **opposite correct answers**:

| Split | Contradictions | Duplicated texts |
|---|---:|---:|
| Relevance | 5 | 5 |
| Preference | 5 | 5 |
| Factuality | 0 | 10 |
| Coherence | 0 | 5 |

A judge answering the same displayed text the same way is scored correct on one
and incorrect on the other, so those items measure nothing.

## 6. Confidence intervals understate uncertainty

The bootstrap resampled all 375 rows per judge/task cell independently. Those
rows are repeated measures: 3 runs nested within duplicated prompt pairs, nested
within a handful of unique items. Resampling them as independent observations
understates uncertainty.

Recomputed with a cluster bootstrap over the correct unit, intervals widen by
**2.8x** (clustering on unique prompt pair) to **3.9x** (clustering on unique
item).

**This falsifies a specific published claim.** The paper reports that the
GPT-5.5 and GPT-4o coherence intervals do not overlap, indicating a
statistically distinct reliability tier:

| Clustering | GPT-5.5 | GPT-4o | Verdict |
|---|---|---|---|
| As published (row) | [0.789, 0.864] | [0.885, 0.941] | separated |
| Unique prompt pair | [0.715, 0.923] | [0.795, 1.000] | **overlap** |
| Unique item | [0.597, 1.000] | [0.755, 1.000] | **overlap** |

The claim of a distinct tier is not supported. The qualitative finding that
judge robustness varies widely is not overturned by clustering alone — the
0.39–0.99 spread is far too large for that — but it rests on five items and
should be treated as provisional.

## 7. Reported counts are inconsistent

The paper and checklist variously report 9 and 13 judges, and 500 and 494 pairs.
The 494 figure is 500 minus a 6-pair exclusion list, but **none of those six
pair identifiers appear in the released data** (the factuality split ships 125
rows spanning `fact_001`–`fact_174`, non-contiguous). The exclusion filter never
fires on the released dataset. The released dataset is 500 rows.

## 8. A decision parser defect

The v1 output parser matched substrings, so the uppercased English article "a"
was read as decision "A", "Cannot determine" as "NO", and a "1–5" scale echo as
score 1.

Quantified against all 38,828 raw decisions, **305 (0.79%)** parse differently
under a strict parser. Mean JSS impact: coherence −0.0000, factuality +0.0006,
relevance −0.0099, preference −0.0191. The headline coherence result is
unaffected. The near-universal always-A behaviour on pairwise tasks is **not**
explained by this defect — its real cause is structural: both prompt variants
present the candidates in the same order in 125/125 pairs, so a judge that
answers by position alone agrees with itself on every pair.

---

## What is being done

A rebuilt v2 dataset is in progress on the `v2-rebuild` branch: 250 unique items
per task loaded from the real upstream benchmarks with per-item provenance
records, an A/B–B/A swap design so position-following cannot score as
consistency, cluster-aware and chance-corrected statistics, a strict parser, and
a CI data-audit gate that fails the build on every defect class listed above.
The v1 data fails that gate on 23 of 32 checks. (The v1 config runs 32 checks: 28 dataset checks plus 4 annotation-timing checks. 28 is the v2 total, where the gate reports 0 of 28 checks failed.)

The paper reports a completed three-judge sweep (claude-haiku, claude-sonnet, claude-opus-4-7) over the v2 dataset, held in data/results_v2/raw and regenerated into every reported figure by scripts/regenerate_results.py. Factuality, coherence and relevance are complete for all three judges; the preference split is complete for claude-haiku only, the other two having stopped mid-run when the API balance was exhausted. An earlier partial sweep made under a decoding configuration since corrected was discarded and does not enter the results. No human validation has been performed on v2

## Reproducing these findings

```bash
python scripts/data_audit.py --config data/audit_config.json
```

Last verified: August 2026.
