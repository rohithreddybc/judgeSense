---
license: cc-by-4.0
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- text-classification
tags:
- llm-as-a-judge
- evaluation
- prompt-sensitivity
- benchmark
- meta-evaluation
pretty_name: JudgeSense
configs:
- config_name: factuality
  data_files:
  - split: test
    path: data/factuality.jsonl
- config_name: coherence
  data_files:
  - split: test
    path: data/coherence.jsonl
- config_name: relevance
  data_files:
  - split: test
    path: data/relevance.jsonl
- config_name: preference
  data_files:
  - split: test
    path: data/preference.jsonl
---

# JudgeSense

A benchmark for measuring whether an LLM judge returns the **same verdict when
the same evaluation request is worded differently**.

The question is deliberately narrow. This dataset does not measure whether a
judge is *correct*; it measures whether it is *reproducible*. A judge that is
wrong on every item but wrong identically under both phrasings scores perfectly
here, and that is intended — a measuring instrument whose reading depends on how
you phrase the question is not usable, however accurate it looks on average.

## Composition

| Task | Source corpus | Items | Rows | Label |
|---|---|---|---|---|
| factuality | `truthful_qa` (generation) | 250 | 250 | accurate / inaccurate |
| coherence | `mteb/summeval` | 250 | 250 | 1–5 expert rating |
| relevance | `BeIR/trec-covid` + graded qrels | 250 | 500 | which passage is relevant |
| preference | `lmsys/mt_bench_human_judgments` | 130 | 260 | which response humans preferred |
| **total** | | **880** | **1,260** | |

Pairwise tasks carry two rows per item, one for each candidate ordering.

Every label is the one the source corpus already carried. No item is authored by
us and no label is assigned or inferred by us. Every row records its source
dataset, split, per-record identifier, and the source configuration where the
source defines one, so any item traces back to the row it came from.

## How to use it

Each row carries `prompt_a` and `prompt_b`: the same item under two instruction
templates that differ in wording and not in what they ask. Send both to your
judge, parse both answers, and measure how often they agree.

```python
from datasets import load_dataset

ds = load_dataset("Rohithreddybc/judgesense-benchmark", "factuality", split="test")
row = ds[0]
a = my_judge(row["prompt_a"])
b = my_judge(row["prompt_b"])
agree = (a == b)          # this is the measurement; ground truth is not consulted
```

**Measure against a repeat baseline, not against zero.** A judge sampling at
nonzero temperature disagrees with itself on the identical prompt, so part of any
disagreement you observe is ordinary decoding variance rather than sensitivity to
wording. Issue `prompt_a` a second time unchanged, and report the *difference*
between paraphrase agreement and repeat agreement. Without that subtraction you
will attribute a judge's sampling noise to paraphrasing. In our own work a
misconfigured temperature produced a repeat ceiling low enough to absorb the
entire effect, and nothing in the output distinguished it from a genuine one.

Run judges at a **matched decoding budget and a fixed temperature across
families**, or differences between judge classes are confounded with inference
configuration.

## What we checked so you do not have to assume it

A benchmark measures what it claims only if it cannot be passed by a heuristic
that ignores the intended construct. These are computed on the released files:

| Heuristic | Task | Accuracy | 95% CI | n | coverage |
|---|---|---|---|---|---|
| always answer A | relevance | 0.500 | [0.456, 0.544] | 500 | 1.00 |
| always answer A | preference | 0.500 | [0.454, 0.546] | 260 | 1.00 |
| pick the longer candidate | relevance | 0.482 | [0.438, 0.526] | 494 | 0.99 |
| pick the longer candidate | preference | 0.508 | [0.446, 0.570] | 256 | 0.98 |
| pick higher query overlap | relevance | 0.540 | [0.478, 0.600] | 252 | 0.50 |

Reproduce with `scripts/shortcut_controls.py` in the code repository. The overlap
control is the weakest: its interval is consistent with a residual lexical signal
worth about 0.57, and it reads only half the relevance rows because the rest tie.

The complement also holds — a benchmark nothing can pass is useless. The best
judge we measured scores 0.944 on factuality, 0.882 on relevance, 0.873 on
preference and 0.452 on coherence.

Read the last of those against the right baseline. Coherence gold is skewed
(`{4: 87, 3: 71, 2: 55, 5: 32, 1: 5}`), so a judge that answers "4" every time
scores **0.348** — not the 0.20 a five-point scale suggests. The best judge beats
that constant by ten points, which makes coherence only weakly discriminating,
and we would rather you knew that before building on it. The other three tasks
clear their majority-class baseline of 0.500 by 37 points or more.

These thresholds were fixed after the first results were in hand, so they are
descriptive checks rather than pre-registered tests, and we describe them that
way.

## Version history, stated plainly

**This dataset has been rebuilt twice, and the earlier versions were defective.**
If you downloaded before 2026-08-21, replace it.

*v1 (withdrawn).* Items were hardcoded in source while described as
benchmark-sourced. 75 unique items. The correct candidate appeared first in every
pairwise item, so a judge that always answered "A" was indistinguishable from a
perfect one. One factuality template inverted the answer polarity, inflating
measured disagreement uniformly across every judge.

*v2.0 (superseded).* Rebuilt from real corpora with provenance, but three defects
survived into the released files and were found afterwards by adversarial audit:

- The template-pair assignment ran in lockstep with the alternating
  correct/incorrect emission, so **the template pair predicted the label on all
  250 factuality items**. File line parity was also a perfect oracle.
- Two templates carried a 75/25 label imbalance, so a constant-answer judge would
  show a large fake template preference.
- **108 of 226 preference items shared candidate text under contradictory gold
  labels** — the same response winning one item and losing another — so a judge
  reasoning correctly and consistently was scored wrong.
- 73 items violated the label rule printed inside them, having been decided by a
  single decisive vote against a plurality of "neither".

*v2.1 (current).* Template pairs are assigned by a label-stratified seeded
permutation and rows are shuffled, so no positional or template rule predicts the
label above chance. The label rule is enforced on decisive votes. Items with
contradictory gold on shared candidate text are removed. Length balance is held
at exactly 50/50 with the buckets matched on vote shape, so removing the verbosity
shortcut does not install a difficulty confound in its place. The preference split
is smaller as a result — 130 items rather than 226 — and we regard that as the
correct trade.

Content hashes (excluding retrieval timestamps):

```
factuality  8875a1f210c7b07f
coherence   37e5caceca0cff3c
relevance   379a228fe28fe422
preference  ceff37f1bf1091ef
```

## Limitations

The two templates in each pair are intended to be meaning-equivalent. We enforce
that their requested answer sets are identical, that neither inverts polarity,
that each names its own task's construct, and that no pair is a near-duplicate —
but no offline check establishes semantic equivalence, and whether a competent
reader maps two wordings onto the same question is not computable from the
strings.

**No item appears under more than one template pair.** Template identity is
therefore nested inside item identity, and no estimator on this data separates a
template effect from an item-set effect. The within-item comparison the benchmark
is built for is unaffected; between-template claims are not identifiable.

Relevance draws on 50 TREC-COVID topics and preference on 62 MT-Bench questions,
so items within a task are not independent. Cluster your intervals at the source
record, not the item, if you need conservative uncertainty.

## Citation

```bibtex
@misc{bellibatlu2026judgesense,
  title         = {JudgeSense: Measuring the Prompt Sensitivity of LLM-as-a-Judge},
  author        = {Bellibatlu, Rohith Reddy and Raff, Edward and Zhang, Wenbin},
  year          = {2026},
  eprint        = {2604.23478},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.23478}
}
```

**Version note.** The preprint at
[arXiv:2604.23478](https://arxiv.org/abs/2604.23478) describes dataset v1, which
was withdrawn; its reported numbers were computed on data this release replaces
and do not apply here. Cite it for the benchmark and the method; take the counts,
splits and results from this card and from
[the code repository](https://github.com/rohithreddybc/judgeSense), which are
regenerated from the current build. A revised paper covering v2.1 is in
preparation.

Licensed CC-BY-4.0. Source corpora retain their own licences.
