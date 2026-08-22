# Pass-2 revisions — what to run, in order

This file lists everything to execute on **your** machine to land Jeff's pass-2 comments. Code changes (model wrappers, manual-review CLI, max_tokens plumbing) are already applied. The steps below are the runtime work that needs your API keys and your time.

---

## 0. Before anything: back up the existing results

```bash
cd judgeSense
cp -r data/results/raw_outputs data/results/raw_outputs_v1_max20_backup
```

Reason: the rerun overwrites only the cells we re-run, but if anything goes wrong with the new sweep we want a one-command restore.

---

## 1. Add the new API key to `.env`

Append this line (replace the placeholder):

```env
DASHSCOPE_API_KEY=...
# Optional: override the endpoint (defaults to international)
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `NOVITA_API_KEY` are already in `.env` from pass 1 — no change needed for those, but **double-check that the OpenAI and Anthropic keys are on tiers that have access to GPT-5.5 and Claude Opus 4.7** before launching the sweep.

---

## 2. Verify the 4 new judges respond correctly

```bash
python scripts/verify_apis.py
```

This will hit every model in the registry (13 entries) with a one-token "OK" prompt. Expected: 13/13 PASS. The 4 new judges and their resolved SKUs:

| Logical name      | `model_id`                              | Provider |
|---|---|---|
| `gpt-5.5`            | `gpt-5.5`                              | OpenAI |
| `claude-opus-4-7`    | `claude-opus-4-7`                      | Anthropic |
| `qwen-3.6-flash`     | `qwen3.6-35b-a3b`                      | DashScope (Alibaba Cloud Model Studio, intl endpoint) |
| `deepseek-v4-flash`  | `deepseek/deepseek-v4-flash`           | Novita |

---

## 3. Run the manual paraphrase review (Jeff's #1 priority — must do before sweep)

This is the most important change for credibility. Single annotator (you), criteria documented in the script docstring.

```bash
# Plan ~4 hours total. Safe to Ctrl+C and resume.
python scripts/manual_review.py --task factuality
python scripts/manual_review.py --task coherence
python scripts/manual_review.py --task relevance
python scripts/manual_review.py --task preference

# When done:
python scripts/manual_review.py --summarize
```

The summarize command prints per-task YES/NO/UNSURE counts and the agreement % between your manual labels and the gpt-4o-mini classifier. **Those numbers go into the paper** (revised §3.3).

Outputs land in `data/validation/manual/{task}_manual.jsonl`.

---

## 4. Smoke-test the per-class token-cap claim (defensible-mixed-cap argument)

Before launching the full sweep, prove that bumping `max_tokens` from 20 → 1024 doesn't change decisions for the existing non-reasoning models. Pick two of the existing 8 (suggest `gpt-4o` and `llama3-70b`) and re-run **coherence only** at the new cap on a 50-pair subset.

There is no built-in subset flag, so the cleanest way is to temporarily set their `max_tokens` to `1024` in `src/models.py` and add a `--dry-run` (existing flag, limits to 5 pairs). For 50 pairs, take the first 50 from `data/prompt_pairs/coherence.jsonl` into a temp file:

```bash
head -n 50 data/prompt_pairs/coherence.jsonl > /tmp/coherence_smoketest.jsonl

# Run the smoke test (saves to a separate dir so it doesn't pollute the real results)
python src/evaluate.py --model gpt-4o    --task coherence --runs 1 --input /tmp --output /tmp/smoketest_out
python src/evaluate.py --model llama3-70b --task coherence --runs 1 --input /tmp --output /tmp/smoketest_out
```

Then diff the decisions against the existing 50-pair slice in `data/results/raw_outputs/gpt-4o_coherence.jsonl`. If decisions match, the per-class cap framing is empirically defensible (paper §3.5). If they diverge meaningfully (say >5% of pairs flip), we have to re-run all 14 at `max_tokens=1024` — flag this and stop.

---

## 5. Run the new judges (the actual rerun)

Five models to call: deepseek (re-run at `max_tokens=1024`) + 4 new. Per model: 4 tasks × 3 runs × 2 prompts × ~125 pairs = ~3,000 calls.

Easiest: launch the master batch script which spawns all 5 × 4 = 20 windows in parallel.

```cmd
cd judgeSense
bat\run_pass2.bat
```

Or run any subset manually:

```bash
python src/evaluate.py --model deepseek          --task all --runs 3
python src/evaluate.py --model gpt-5.5           --task all --runs 3
python src/evaluate.py --model claude-opus-4-7   --task all --runs 3
python src/evaluate.py --model qwen-3.6-flash    --task all --runs 3
python src/evaluate.py --model deepseek-v4-flash --task all --runs 3
```

**Resumability**: each call checks `data/results/raw_outputs/{model}_{task}.jsonl` for already-completed (pair_id, run) tuples and skips them. Safe to Ctrl+C and re-run.

Expected wall-clock: 8–24 hours depending on provider rate limits and reasoning-model output speed. Run overnight.

Expected cost: $80–$300 total depending on how output-heavy the reasoning models are.

---

## 6. After the sweep: regenerate metrics and figures

```bash
# Update metrics for all 14 models
python src/metrics.py --results data/results/raw_outputs/

# Polarity-corrected factuality (reuses existing analysis script — verify it auto-discovers new models)
python analysis/factuality_jss_fixed.py
python analysis/per_template_factuality.py
python analysis/factuality_pair_overlap.py

# Re-render fig1, fig2, fig4 (now includes the 5 new judges)
python analysis/generate_figures.py
```

If `factuality_jss_fixed.py` or `per_template_factuality.py` hardcoded the 9-model list (likely — check), you'll need a 1-line edit to add the new models to their model lists too. Same pattern as `analysis/generate_figures.py` line 40.

---

## 7. Read the new numbers, then update the paper

Once `outputs/` has the new figures and `data/results/metrics_summary.json` has the new JSS values, ping me with the numbers. I'll update Tables 1–3 and the §4 Findings narrative (some claims like "scale does not predict consistency" may strengthen or weaken depending on where the new judges land).

---

## What's in the paper (already updated for pass 1, will be updated again for pass 2)

The paper revision (Step 3 in our workflow) hasn't started yet — it's blocked on (a) your manual-validation results and (b) the new sweep numbers. I'll do it as soon as you have those.

---

## Summary of code changes already made

| File | Change |
|---|---|
| `src/models.py` | Added `max_tokens` field to every entry of `SUPPORTED_MODELS`; added 4 new entries (gpt-5.5, claude-opus-4-7, qwen-3.6-flash, deepseek-v4-flash) at `max_tokens=1024`; bumped `deepseek` to `max_tokens=1024` |
| `src/evaluate.py` | `_MAX_TOKENS=20` constant replaced with per-model lookup; added `dashscope` provider (OpenAI-compatible, defaults to international endpoint, override via `DASHSCOPE_BASE_URL`); added GPT-5.x parameter routing (`max_completion_tokens` vs `max_tokens`, no temperature); raised `_TIMEOUT` 30→60s; added cost entries for new models; all `_call_*` functions now take `max_tokens` as a parameter |
| `scripts/verify_apis.py` | Added `test_novita` and `test_dashscope` smoke-test functions; switched gemini test to the current `google.genai` SDK; added GPT-5.x parameter routing; added 4 new entries to the `MODELS` registry |
| `scripts/manual_review.py` | **New file.** CLI to hand-label paraphrase pairs; checkpoints to `data/validation/manual/`; `--summarize` reports classifier-vs-human agreement |
| `analysis/generate_figures.py` | Added 4 new model identifiers to `MODELS` and their display names to `DISPLAY` |
| `bat/` | **New folder.** 20 per-(model, task) `.bat` files plus `run_pass2.bat` master launcher and `_single_run.bat` helper |

No existing files were deleted; the old 8-model results stay in `data/results/raw_outputs/` untouched.
