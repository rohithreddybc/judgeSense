# JudgeSense Full Experiment Plan

## Objective
Test LLM judge sensitivity to prompt wording variations using Judge Sensitivity Score (JSS) and related metrics.

## Models (9 total)

| Model | Provider | Cost Est. | Status |
|-------|----------|-----------|--------|
| gpt-4o-mini | OpenAI | ~$0.60 | ✓ PASS |
| gpt-4o | OpenAI | ~$6.50 | ✓ PASS |
| claude-haiku | Anthropic | ~$2.40 | ○ SKIP* |
| claude-sonnet | Anthropic | ~$9.00 | ○ SKIP* |
| gemini-flash | Google | free | ✗ QUOTA |
| llama3-8b | HuggingFace | free | ✓ PASS |
| llama3-70b | HuggingFace | free | ✓ PASS |
| mistral-7b | Mistral | free | ✓ PASS |
| qwen | HuggingFace | free | ✓ PASS |

**Status Legend:**
- ✓ PASS: Ready to run
- ○ SKIP: Will skip at runtime (missing key)
- ✗ QUOTA: Free tier quota exhausted (will gracefully fail)

*Anthropic key available; check .env loading at runtime

## Dataset
- **Total pairs:** 500 (125 per task)
- **Task types:**
  - Factuality (125 pairs) — TruthfulQA
  - Coherence (125 pairs) — SummEval
  - Relevance (125 pairs) — BEIR
  - Preference (125 pairs) — MT-Bench
- **Prompt engineering:** All templates <30 words, single-token answers
- **Location:** `data/prompt_pairs/*.jsonl`

## Experiment Design

### Core Metrics
1. **Judge Sensitivity Score (JSS)** — fraction of pairs with same decision across variants
2. **Decision Flip Rate** — 1 - JSS
3. **Cohen's Kappa** — inter-rater agreement corrected for chance
4. **Bootstrap 95% CI** — confidence bounds for JSS

### Evaluation Loop
```
For each model (9 total):
  For each task (4 types):
    For each run (3 iterations):
      For each pair (125):
        1. Call API with prompt_a
        2. Call API with prompt_b (after rate limit delay)
        3. Record: pair_id, raw_decision_a, raw_decision_b, normalized decisions
        4. On error: log and continue (graceful failure)
        5. Append to: data/results/raw_outputs/{model}_{task}.jsonl
```

### Expected Metrics
- **Total API calls:** 9 models × 4 tasks × 3 runs × 125 pairs × 2 calls/pair = ~27,000 calls
- **Estimated cost:** ~$18.50 (mostly OpenAI models)
- **Expected runtime:** 2-4 hours (with rate limiting)
- **Output files:** 36 JSONL files (9 models × 4 tasks)

### Error Handling
- **API failures:** Logged, recorded with "ERROR: {message}", loop continues
- **Quota exceeded:** Gracefully handled, partial results saved
- **Timeouts:** Retry once after 5s, then record as error
- **No crashes:** All exceptions caught at evaluation loop level

## Run Instructions

### Option 1: All models, all tasks, 3 runs (full experiment)
```bash
cd judgeSense
for model in gpt-4o-mini gpt-4o claude-haiku claude-sonnet llama3-8b llama3-70b mistral-7b qwen deepseek; do
  python src/evaluate.py --model $model --task all --runs 3
done
```

### Option 2: Single model
```bash
python src/evaluate.py --model gpt-4o-mini --task all --runs 3
```

### Option 3: Specific task
```bash
python src/evaluate.py --model gpt-4o-mini --task factuality --runs 3
```

### Option 4: Quick test (5 pairs only)
```bash
python src/evaluate.py --model gpt-4o-mini --task factuality --runs 1 --dry-run
```

## Output Format

Each line in `data/results/raw_outputs/{model}_{task}.jsonl`:
```json
{
  "pair_id": "fact_001",
  "task_type": "factuality",
  "model": "gpt-4o-mini",
  "prompt_a_decision_raw": "YES",
  "prompt_a_decision": "YES",
  "prompt_b_decision_raw": "NO, that's incorrect",
  "prompt_b_decision": "NO",
  "run_number": 1,
  "timestamp": "2026-04-21T20:00:00+00:00"
}
```

## Analysis Pipeline (after experiment)

1. **Load results:** `src/metrics.py`
2. **Compute metrics:** JSS, flip_rate, kappa, CI per (model, task)
3. **Generate tables:** Summary statistics
4. **Visualize:** Heatmaps of JSS by model and task type
5. **Statistical tests:** Compare models with bootstrap resampling

## Checkpoints
- [ ] Run starts successfully for first model
- [ ] Results appear in `data/results/raw_outputs/`
- [ ] Cost tracking is accurate
- [ ] All 36 output files created (9 models × 4 tasks)
- [ ] No crashes on API failures
- [ ] Metrics computed successfully

## Notes
- **Costs:** Mainly from OpenAI ($16.50) + Anthropic ($11.40 if keys work) = ~$28
- **Free tier limits:** Gemini will fail after 10-20 calls/day; this is expected
- **Rate limiting:** 0.5-1.0s delays built in to avoid throttling
- **Caching:** No caching; each variant gets fresh API call (intentional)

---

**Experiment Status:** Ready to run
**Last Updated:** 2026-04-21
**API Verification:** 7/10 models passing (see `scripts/verify_results.json`)
