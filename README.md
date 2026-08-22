# JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems

A framework for quantifying prompt sensitivity in LLM-as-a-Judge evaluation systems.

[![arXiv](https://img.shields.io/badge/arXiv-2604.23478-red.svg)](https://arxiv.org/abs/2604.23478)
[![Dataset](https://img.shields.io/badge/dataset-HuggingFace-orange.svg)](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Large language models are increasingly deployed as automated judges to evaluate the outputs of other models, yet the reliability of these systems remains poorly understood. **JudgeSense** quantifies prompt sensitivity in LLM-as-a-Judge systems via the **Judge Sensitivity Score (JSS)**: how often a judge's decision changes when prompt phrasing varies while evaluation intent stays constant.

**Dataset**: [Rohithreddybc/judgesense-benchmark](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark) — v2.0, 976 unique items across 4 tasks.

## What the benchmark provides

- **976 unique items** (250 factuality, 250 coherence, 250 relevance, 226 preference), loaded at build time from `truthful_qa`, `mteb/summeval`, `BeIR/trec-covid`, and `lmsys/mt_bench_human_judgments`. Every record carries a provenance chain resolving to a specific row in a specific split, with retrieval timestamp and loader version.
- **Ground truth from the source**: TruthfulQA's accuracy labels, SummEval's expert coherence ratings, TREC-COVID graded relevance judgements (both candidates human-graded: the positive at grade 2, the distractor at grade 0), and real human preference votes from MT-Bench.
- **Position-bias swap design**: pairwise tasks present candidates in both A/B and B/A orderings, so a judge answering by position alone scores ~50%, not 100%.
- **One prompt pair per item** — rows are never duplicated to inflate the count.
- **Cluster-aware statistics**: confidence intervals require an explicitly declared unit of analysis (`row`, `structural_pair`, `prompt_pair`, `item`) and resample clusters, never rows.
- **Chance-corrected and ordinal-aware metrics**: Cohen's kappa over the two arms, quadratic-weighted kappa for the Likert task, and a strict mode counting unparseable output as disagreement rather than dropping it.
- **Polarity remapping** so polarity-inverted templates can be scored rather than excluded (`src/polarity.py`).
- **Data-audit gate** (`scripts/data_audit.py`), run in CI: fails the build on too few unique items, duplicate rows, unbacked provenance, degenerate labels, contradictory ground truth, insufficient effective sample size, or implausible annotation timing.

## The JSS metric

JSS is the fraction of prompt pairs where both phrasings elicit the same decision. It measures agreement between two phrasings and never consults ground truth.

Raw JSS partly rewards judges that compress their output distribution, so chance correction matters — see `src/metrics_v2.py` for the kappa, ordinal, and strict-mode variants alongside it.

## Installation

```bash
git clone https://github.com/rohithreddybc/judgesense.git
cd judgesense
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or install directly via pip (metrics only, minimal dependencies):

```bash
pip install judgesense
# For full evaluation capabilities (API clients, datasets):
pip install "judgesense[full]"
```

## Quickstart

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

Run the full evaluation:

```bash
python src/evaluate.py --model gpt-4o-mini --task factuality --runs 3
```

Compute JSS from results:

```bash
python src/metrics.py --results data/results/raw_outputs/
```

## Dataset

- **HuggingFace**: [Rohithreddybc/judgesense-benchmark](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark) (v2.0)
- **License**: CC-BY-4.0 (upstream datasets retain their own terms)

| Task | Source dataset | Unique items | Rows |
|------|----------------|-------------:|-----:|
| Factuality | `truthful_qa` | 250 | 250 |
| Coherence | `mteb/summeval` | 250 | 250 |
| Relevance | `BeIR/trec-covid` + qrels | 250 | 500 |
| Preference | `lmsys/mt_bench_human_judgments` | 250 | 500 |

Pairwise tasks contribute two rows per item — one per candidate ordering.

### Quick usage

```python
from datasets import load_dataset

# one config per task; pairwise tasks carry extra ordering fields
ds = load_dataset("Rohithreddybc/judgesense-benchmark", "coherence", split="test")
print(len(ds), "items")
print(ds[0]["prompt_a"], ds[0]["prompt_b"])
print(ds[0]["ground_truth_label"], ds[0]["source"]["source_record_id"])
```

### Rebuilding from source

```bash
python src/dataset_builder_v2.py --output data/v2 --items-per-task 250
python scripts/data_audit.py --config data/audit_config_v2.json   # must pass
```

The loaders fetch from the upstream datasets and fail loudly if a source is
unreachable — there is no fallback to cached or synthetic items.

### Schema

```json
{
  "pair_id": "fact_v2_0001",
  "item_id": "fact_truthfulqa_740",
  "prompt_pair_id": "fact_truthfulqa_740#T1-T2",
  "task_type": "factuality",
  "template_a": "T1",
  "template_b": "T2",
  "prompt_a": "Is this statement factually correct? Answer YES or NO only.\n\n...",
  "prompt_b": "Fact-check the statement below. Reply YES (correct) or NO (incorrect).\n\n...",
  "response_being_judged": "...",
  "ground_truth_label": "accurate",
  "source_benchmark": "truthful_qa",
  "source": {
    "source_dataset": "truthful_qa",
    "source_config": "generation",
    "source_split": "validation",
    "source_record_id": "validation[740]",
    "source_fields": {"answer_field": "best_answer"},
    "retrieved_at": "2026-08-07T04:50:42Z",
    "loader_version": "2.0.0"
  },
  "builder_version": "2.0.0"
}
```

`item_id` and `prompt_pair_id` are the clustering keys — use them when computing
confidence intervals. Coherence adds `ground_truth_raw` (the unrounded expert
mean); relevance and preference add `ab_order`, `candidate_map`, and
`ground_truth_position`.

## Results

A judge sweep against the v2 dataset has not yet been run, so no results are
reported here. When it is, results will be published with cluster-aware
confidence intervals at a declared unit of analysis, chance-corrected scores
alongside raw JSS, and malformed-output rates reported rather than dropped.

Results from the earlier v1 dataset are retained in repository history and
summarised in [ERRATA.md](ERRATA.md), together with the reasons they should not
be carried forward.

## Running an evaluation

```bash
# 1. Build the dataset from source (fails loudly if a source is unreachable)
python src/dataset_builder_v2.py --output data/v2 --items-per-task 250

# 2. Gate it — this must pass before any results are computed
python scripts/data_audit.py --config data/audit_config_v2.json

# 3. Run judges (requires API keys; see .env.example)
python src/evaluate.py --model gpt-4o --task coherence

# 4. Score with cluster-aware, chance-corrected metrics
python -c "from src.metrics_v2 import compute_all_metrics_v2"
```

Judge configuration — families, parameter sizes, matched vs native token
budgets, and which checkpoints are verified — lives in `src/judge_registry.py`.
Use `run_plan()` to state a sweep's call count before spending it.

## Repository structure

```
judgesense/
├── data/
│   ├── prompt_pairs/          # 4 JSONL files, one per task type
│   ├── results/               # Raw judge outputs + computed metrics
│   └── validation/manual/     # Human annotation results (500 pairs, 4 tasks)
├── src/
│   ├── dataset_builder.py     # Generates the prompt pair dataset
│   ├── models.py              # API wrappers (OpenAI, Anthropic, Google, Alibaba Cloud, Novita AI, HuggingFace)
│   ├── evaluate.py            # Main evaluation runner
│   ├── metrics.py             # JSS + decision flip rate + Cohen's kappa
│   └── utils.py               # Shared helpers
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_results_analysis.ipynb
│   └── 03_figures.ipynb
├── analysis/
│   ├── factuality_jss_fixed.py    # Recompute JSS with T4 polarity correction
│   ├── per_template_factuality.py # Per-template JSS breakdown
│   ├── factuality_pair_overlap.py # Pair-level flip overlap analysis
│   └── generate_figures.py        # Publication-ready PDF figures
├── outputs/               # CSV results + publication-ready PDF figures
├── figures/               # Paper-ready PDF/PNG figures
├── tests/                 # Unit tests for metrics and dataset (350 tests)
├── requirements.txt
├── .env.example
└── README.md
```

## Citation

If you use JudgeSense in your research, please cite:

```bibtex
@misc{bellibatlu2026judgesense,
      title={JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems}, 
      author={Rohith Reddy Bellibatlu},
      year={2026},
      eprint={2604.23478},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.23478}, 
}
```

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Dataset**: CC-BY-4.0

## Contact

Rohith Reddy Bellibatlu — ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)

---

*This work is part of an independent research portfolio. All evaluations were conducted on public benchmarks and APIs.*
