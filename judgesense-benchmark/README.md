# JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems

[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23478-red.svg)](https://arxiv.org/abs/2604.23478)
[![HuggingFace](https://img.shields.io/badge/dataset-HuggingFace-orange.svg)](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark)

---

## Overview

**JudgeSense** is a benchmark dataset of **500 hand-validated prompt pairs** for measuring prompt sensitivity in LLM-as-a-Judge evaluation systems. Each pair contains two differently phrased but semantically equivalent judge prompts applied to the same response, enabling rigorous measurement of how much a judge's decision changes due to prompt wording alone.

All 500 pairs were validated by a human annotator: 450 confirmed semantically equivalent; 50 pairs involving Template 4 (polarity-inverted) are flagged and handled via label remapping in the evaluation code.

The dataset covers four evaluation task types:

| Task | Source | Pairs | Labels |
|------|--------|-------|--------|
| **Factuality** | TruthfulQA | 125 | accurate / inaccurate |
| **Coherence** | SummEval | 125 | score_1 ... score_5 |
| **Preference** | MT-Bench | 125 | A / B |
| **Relevance** | BEIR | 125 | A / B |

---

## What This Enables

- **Prompt sensitivity evaluation** — measure how fragile a judge is to phrasing variation
- **LLM judge robustness benchmarking** — compare models on decision consistency
- **Detection of prompt-induced artifacts** — identify polarity inversions (T4) and other systematic biases

---

## Quick Start

```python
from utils.load_judgesense import load_task, load_all
from utils.compute_jss import compute_jss

# Load one task
pairs = load_task("factuality")
print(f"{len(pairs)} pairs loaded")

# Load all tasks
all_data = load_all()

# Compute JSS from your judge's decisions
jss = compute_jss(decisions_a, decisions_b)
print(f"JSS: {jss:.3f}")
```

Run the full example:

```bash
cd judgesense-benchmark
python examples/run_jss_example.py
```

---

## Dataset Schema

Each JSONL record has eight fields:

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

---

## Metric: Judge Sensitivity Score (JSS)

JSS is the fraction of pairs where both prompt variants elicit the same decision from the judge:

```
JSS = (1/N) * sum( decisions_a[i] == decisions_b[i] )
```

- **JSS = 1.0** — perfectly consistent; the judge never changes its decision due to prompt phrasing
- **JSS = 0.0** — maximally sensitive; every decision flips between prompts

A high flip rate (= 1 - JSS) indicates the judge's apparent decisions are largely driven by prompt design rather than the content being evaluated.

---

## Benchmark Results (13 judges, pass-2)

### Coherence (most discriminating task)

| Model | JSS | Cohen's kappa |
|---|---|---|
| Claude Sonnet 4.5 | 0.99 | 0.986 |
| Qwen-2.5-72B | 0.92 | 0.846 |
| GPT-4o | 0.92 | 0.828 |
| GPT-5.5 | 0.83 | 0.694 |
| GPT-4o-mini | 0.78 | 0.627 |
| Claude Haiku 4.5 | 0.73 | 0.583 |
| Claude Opus 4.7 | 0.70 | 0.576 |
| LLaMA-3.1-70B | 0.55 | 0.338 |
| DeepSeek-R1 | 0.53 | 0.326 |
| Qwen 3.6 Flash | 0.51 | 0.372 |
| DeepSeek-V4 Flash | 0.50 | 0.350 |
| Mistral-7B | 0.48 | -0.082 |
| Gemini 2.5 Flash | 0.39 | -0.053 |

### Factuality (after T4 polarity correction)

| Model | JSS (raw) | JSS (corrected) | Delta |
|---|---|---|---|
| GPT-4o | 0.63 | 1.00 | +0.37 |
| GPT-4o-mini | 0.63 | 1.00 | +0.37 |
| Claude Haiku 4.5 | 0.63 | 1.00 | +0.37 |
| Claude Sonnet 4.5 | 0.63 | 1.00 | +0.37 |
| DeepSeek-R1 | 0.63 | 1.00 | +0.37 |
| LLaMA-3.1-70B | 0.63 | 1.00 | +0.37 |
| Gemini 2.5 Flash | 0.63 | 1.00 | +0.37 |
| Qwen-2.5-72B | 0.63 | 1.00 | +0.37 |
| Mistral-7B | 0.71 | 0.88 | +0.17 |
| GPT-5.5 | 0.63 | 1.00 | +0.37 |
| Claude Opus 4.7 | 0.63 | 1.00 | +0.37 |
| Qwen 3.6 Flash | 0.63 | 1.00 | +0.37 |
| DeepSeek-V4 Flash | 0.62 | 0.99 | +0.37 |

---

## Key Insights

> **Coherence JSS varies by more than 0.6 units across 13 judges and does not track model scale or recency.**

- Claude Opus 4.7 (0.70) scores lower than Claude Haiku 4.5 (0.73); GPT-5.5 (0.83) scores lower than GPT-4o (0.92)
- Factuality sensitivity is entirely driven by Template 4 polarity inversion, not by model-level inconsistency
- Preference and relevance JSS are degenerate (12 of 13 judges always select option A)
- Total API cost for the 13-model sweep: Novita AI $3.67, Alibaba Cloud $1.00, Anthropic $2.07, OpenAI $3.36

---

## Links

- **GitHub**: [github.com/rohithreddybc/judgesense](https://github.com/rohithreddybc/judgesense)
- **arXiv**: [2604.23478](https://arxiv.org/abs/2604.23478)
- **HuggingFace**: [Rohithreddybc/judgesense-benchmark](https://huggingface.co/datasets/Rohithreddybc/judgesense-benchmark)

---

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
      url={https://arxiv.org/abs/2604.23478}
}
```

---

## License

- **Dataset**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: MIT License

---

*JudgeSense — Independent research. All evaluations conducted on public benchmarks and APIs.*
