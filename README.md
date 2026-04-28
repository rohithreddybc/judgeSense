# JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems

A framework for quantifying prompt sensitivity in LLM-as-a-Judge evaluation systems.

[![arXiv](https://img.shields.io/badge/arXiv-2604.23478-red.svg)](https://arxiv.org/abs/2604.23478)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19798166.svg)](https://doi.org/10.5281/zenodo.19798166)
[![Dataset](https://img.shields.io/badge/dataset-HuggingFace-orange.svg)](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Large language models are increasingly deployed as automated judges to evaluate the outputs of other models, yet the reliability of these systems remains poorly understood. **JudgeSense** is a reproducible benchmark that quantifies prompt sensitivity in LLM-as-a-Judge systems via the **Judge Sensitivity Score (JSS)** — a metric measuring how often a judge's evaluation decision changes when prompt phrasing varies while evaluation intent stays constant. We evaluate **9 LLM judges** across **4 evaluation tasks** (factuality, coherence, preference, relevance) with **500 semantically equivalent prompt pairs** and **3 independent runs each**, and uncover systematic sensitivity driven by prompt polarity inversion. Our analysis reveals that polarity-inverted templates can reduce apparent agreement by up to **37 percentage points**, and that sensitivity varies substantially across model families.

This repository contains the full reproducible codebase, datasets, and evaluation artifacts accompanying the paper.

## Key contributions

- **JSS metric**: A novel, formally defined score for judge decision consistency across semantically equivalent prompts.
- **Public dataset**: 500 semantically equivalent prompt pairs across 4 evaluation task types.
- **Empirical evaluation**: Nine LLM judges (GPT-4o, GPT-4o-mini, Claude Haiku, Claude Sonnet, Gemini Flash, LLaMA3-70B, Mistral-7B, DeepSeek, Qwen) tested across 4 task types with JSS ranging from 0.63 to 1.0 before polarity correction.
- **Full reproducibility**: All code, data, and results released under open licenses.

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

This project includes the **JudgeSense benchmark dataset** — 500 validated paraphrase pairs across 4 evaluation task types, released for prompt sensitivity research.

- **HuggingFace**: [Rohithreddybc/judgesense-benchmark](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark)
- **License**: CC-BY-4.0
- **Size**: 500 prompt pairs, 4 task types, 125 pairs per task

> **Key Insight**: Prompt formulation often dominates model architecture in determining apparent judge consistency.

### Quick usage

```python
from datasets import load_dataset

ds = load_dataset("Rohithreddybc/judgesense-benchmark")
pairs = ds["factuality"]
print(f"{len(pairs)} factuality pairs loaded")

# Compute JSS from your judge's decisions
from judgesense import compute_jss
jss = compute_jss(decisions_a, decisions_b)
print(f"JSS: {jss:.3f}")
```

### Schema

```json
{
  "pair_id": "fact_001",
  "task_type": "factuality",
  "source_benchmark": "TruthfulQA",
  "prompt_a": "Is this factually correct? Answer YES or NO only.\n\nResponse: ...",
  "prompt_b": "Fact-check this response. Reply YES (correct) or NO (incorrect).\n\nResponse: ...",
  "response_being_judged": "The Earth orbits around the Sun.",
  "ground_truth_label": "accurate",
  "semantic_equivalence_score": 1.0
}
```

## Key findings

| Model | JSS (raw) | JSS (T4-corrected) | Delta |
|---|---|---|---|
| GPT-4o | 0.63 | 1.00 | +0.37 |
| GPT-4o-mini | 0.63 | 1.00 | +0.37 |
| Claude Haiku | 0.63 | 1.00 | +0.37 |
| Claude Sonnet | 0.63 | 1.00 | +0.37 |
| DeepSeek | 0.63 | 1.00 | +0.37 |
| LLaMA3-70B | 0.63 | 1.00 | +0.37 |
| Gemini Flash | 0.63 | 1.00 | +0.37 |
| Qwen | 0.63 | 1.00 | +0.37 |
| Mistral 7B | 0.71 | 0.88 | +0.17 |

**Finding**: Polarity-inverted prompt templates (T4) reduce raw JSS by 17–37 pp across all models. After T4 correction, 8 of 9 models achieve JSS = 1.0 on factuality, demonstrating that prompt sensitivity in this task is entirely attributable to template polarity rather than semantic ambiguity. Mistral 7B exhibits the highest residual sensitivity (JSS = 0.88 post-correction).

## Reproducing paper results

Exact commands to replicate every number in the paper:

```bash
# 1. Build the prompt pair dataset
python src/dataset_builder.py --output data/prompt_pairs/

# 2. Run evaluations on all three models
bash scripts/run_all_evals.sh

# 3. Compute metrics
python src/metrics.py --results data/results/raw_outputs/ --output data/results/metrics.json

# 4. Run factuality JSS analysis (T4 polarity-corrected)
python analysis/factuality_jss_fixed.py

# 5. Per-template JSS breakdown
python analysis/per_template_factuality.py

# 6. Pair-level flip overlap
python analysis/factuality_pair_overlap.py

# 7. Generate publication figures (outputs/fig1, fig2, fig4)
python analysis/generate_figures.py
```

## Repository structure

```
judgesense/
├── data/
│   ├── prompt_pairs/          # 4 JSONL files, one per task type
│   └── results/               # Raw judge outputs + computed metrics
├── src/
│   ├── dataset_builder.py     # Generates the prompt pair dataset
│   ├── models.py              # API wrappers (OpenAI, HuggingFace, Mistral)
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
├── figures/                   # Paper-ready PDF/PNG figures
├── tests/                     # Unit tests for metrics and dataset (29 tests)
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

Rohith Reddy Bellibatlu - ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)

---

*This work is part of an independent research portfolio. All evaluations were conducted on public benchmarks and APIs.*
