---
dataset_info:
  name: judgesense
  version: "2.0"
  license: cc-by-4.0
  task_categories:
    - text-classification
    - question-answering
  language:
    - en
---

# Dataset Card — JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems

> **⚠️ This dataset should not be used.** The accompanying paper was withdrawn
> from NeurIPS 2026 in July 2026 after defects were verified against this
> artifact. See
> [ERRATA.md](https://github.com/rohithreddybc/judgeSense/blob/main/ERRATA.md).

## Summary

JudgeSense is a benchmark for evaluating prompt sensitivity in LLM-as-a-Judge
systems. Each row presents two differently phrased judge prompts applied to the
same item, enabling measurement of how much a judge's decision changes due to
prompt wording alone. The dataset spans four evaluation task types: factuality,
coherence, preference, and relevance.

**Corrected composition.** Previously described as 500 hand-validated prompt
pairs, it contains **500 rows spanning 202 unique prompt pairs over 75 unique
underlying items**. Human validation for paraphrase equivalence was performed by
a **single annotator**, confirming 450 of 500 rows; the remaining 50 involve
Template 4 polarity inversion.

## Tasks Covered

**The `Source` column below does not describe actual provenance.** Those names
are hardcoded constants in the dataset builder, which contains no data-loading
code of any kind. The items are Python literals written in the builder and are
**not drawn from TruthfulQA, SummEval, MT-Bench, or BEIR**. The column is
retained only to document what the released files claim.

| Task | Type | `source_benchmark` (inaccurate) | Rows | Unique items | Label Space |
|------|------|--------|-----:|-------------:|-------------|
| Factuality | Pointwise binary | TruthfulQA | 125 | 60 | `accurate`, `inaccurate` |
| Coherence | Pointwise Likert scale | SummEval | 125 | **5** | `score_1` … `score_5` |
| Preference | Pairwise | MT-Bench | 125 | 5 | `A`, `B` |
| Relevance | Pairwise | BEIR | 125 | 5 | `A`, `B` |

Coherence `ground_truth_label` values are **enumeration indices, not coherence
ratings**. Ten items (5 relevance, 5 preference) carry contradictory ground
truth — identical judged content with opposite correct answers.

Human annotation confirmed 450 pairs as semantically equivalent (`semantic_equivalence_score` = 1.0). The 50 factuality pairs involving Template 4 carry inverted polarity and were labeled NO (non-equivalent label convention) in the human review; they remain in the dataset with their original `semantic_equivalence_score` = 1.0 for backward compatibility, but the evaluation code applies label remapping before computing JSS.

## Intended Use

This dataset is intended for:

- **Prompt sensitivity research** — measuring how LLM judge decisions vary under semantically equivalent prompts
- **Judge robustness benchmarking** — comparing LLM judge models on decision consistency (JSS metric)
- **Prompt engineering research** — understanding which structural prompt features drive decision flips
- **Meta-evaluation** — auditing evaluation pipelines for prompt-induced artifacts

## Out-of-Scope Use

This dataset is **not** intended for:

- Training or fine-tuning LLMs
- Evaluating factual knowledge of LLMs (it tests judge behavior, not knowledge)
- Benchmark leaderboard competition (no held-out test split)

## Metric

The primary metric is the **Judge Sensitivity Score (JSS)**:

```
JSS = (1/N) * Σ [ decisions_a[i] == decisions_b[i] ]
```

Higher JSS means more consistent judge behavior across prompt variants. Flip Rate = 1 − JSS.

## Limitations

- **T4 polarity inversion artifact**: Template variant T4 ("Does this response contain factual errors?") uses an inverted polarity relative to other templates (YES = inaccurate, NO = accurate). This structural difference can masquerade as model inconsistency and inflates flip rates for naive analyses. The paper explicitly identifies and accounts for this.

- **Degenerate pairwise tasks**: In preference and relevance tasks, some prompt pairs may yield degenerate results if the judge always selects the same option (A or B) regardless of content. These cases are annotated with `ground_truth_label` for downstream filtering.

- **Closed label spaces**: All prompts are designed to elicit categorical responses. Judges that return free-text or multi-sentence answers may require normalization before computing JSS.

- **English only**: All prompts and responses are in English.

- **Constructed items, not benchmark-sourced** *(corrected)*: an earlier version of this card stated the judged responses were drawn from TruthfulQA, SummEval, MT-Bench, and BEIR. They are not. Both the judged items and the judge prompts are constructed for this benchmark as literals in the dataset builder. Real-world judge prompts may differ, and the items carry none of the difficulty calibration of the named benchmarks.

- **Very few unique items** *(corrected)*: 75 unique items across the whole dataset, and 5 for coherence — the task carrying the widest reported judge spread. Any statistic computed here has an effective sample size far below the row count, and confidence intervals in the accompanying paper were computed as if rows were independent.

## Citation

```bibtex
@misc{bellibatlu2026judgesense,
      title={JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems},
      author={Rohith Reddy Bellibatlu},
      year={2026},
      eprint={2604.23478},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.23478}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## Contact

Rohith Reddy Bellibatlu — ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)
