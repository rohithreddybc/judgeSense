# JudgeSense

**How Fragile Are LLM Judges?** A framework for quantifying prompt sensitivity in LLM-as-a-Judge evaluation systems.

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-red.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-HuggingFace-orange.svg)]()

---

## Overview

Large language models are increasingly deployed as automated judges to evaluate the outputs of other models, yet the reliability of these LLM-as-a-Judge systems remains poorly understood. **JudgeSense** introduces a rigorous framework and metric — the **Judge Sensitivity Score (JSS)** — for measuring how much an LLM judge's evaluation decisions change when the prompt phrasing varies while the evaluation intent stays constant.

This repository contains the full reproducible codebase, datasets, and evaluation artifacts accompanying the paper.

## Key contributions

- **JSS metric**: A novel, formally defined score for judge decision consistency across semantically equivalent prompts.
- **Public dataset**: 500 semantically equivalent prompt pairs across 4 evaluation task types.
- **Empirical evaluation**: Three LLM judges (GPT-4o-mini, Llama 3, Mistral-7B) tested on public benchmarks.
- **Full reproducibility**: All code, data, and results released under open licenses.

## Installation

```bash
git clone https://github.com/rohithreddybc/judgesense.git
cd judgesense
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
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

The JudgeSense prompt pair dataset is available on HuggingFace:

- **Link**: `https://huggingface.co/datasets/rohithreddybc/judgesense` *(coming soon)*
- **License**: CC-BY-4.0
- **Size**: 500 prompt pairs, 4 task types, 125 pairs per task

### Schema

```json
{
  "pair_id": "fact_001",
  "task_type": "factuality",
  "source_benchmark": "TruthfulQA",
  "prompt_a": "Evaluate whether the following response is factually accurate...",
  "prompt_b": "As an expert evaluator, determine if this answer contains correct facts...",
  "response_being_judged": "...",
  "ground_truth_label": "accurate",
  "semantic_equivalence_score": 0.94
}
```

## Reproducing paper results

Exact commands to replicate every number in the paper:

```bash
# 1. Build the prompt pair dataset
python src/dataset_builder.py --output data/prompt_pairs/

# 2. Run evaluations on all three models
bash scripts/run_all_evals.sh

# 3. Compute metrics
python src/metrics.py --results data/results/raw_outputs/ --output data/results/metrics.json

# 4. Regenerate paper figures
jupyter nbconvert --execute notebooks/03_figures.ipynb
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
├── figures/                   # Paper-ready PDF/PNG figures
├── tests/                     # Unit tests for metrics and dataset
├── requirements.txt
├── .env.example
└── README.md
```

## Citation

If you use JudgeSense in your research, please cite:

```bibtex
@article{bellibatlu2026judgesense,
  title={How Fragile Are {LLM} Judges? {JudgeSense}: A Framework for Quantifying Prompt Sensitivity in {LLM} Evaluation},
  author={Bellibatlu, Rohith Reddy},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Dataset**: CC-BY-4.0

## Contact

Rohith Reddy Bellibatlu — ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)

---

*This work is part of an independent research portfolio. All evaluations were conducted on public benchmarks and APIs.*
