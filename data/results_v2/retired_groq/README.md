# Retired Groq cells — partial, excluded from every metric

Three judges were served by Groq's free tier: `gpt-oss-20b`, `gpt-oss-120b` and
`qwen3.8-27b`. They are registered with `verified=False` and their rows live
here rather than in `raw/`, so nothing downstream can read them by accident.

## Why they were retired

Not for quality. All three answered correctly, parsed cleanly and honoured the
matched 1024-token budget. What stopped them is Groq's free-tier cap of
**200,000 tokens per day, per model**:

```
Rate limit reached for model `openai/gpt-oss-20b` in organization ...
service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199299
```

Measured against the prompt sizes this protocol actually sends — 326 tokens per
call for the gpt-oss pair, 80 for the 27B — finishing the remaining rows needs:

| judge | rows collected | days of quota resets to finish |
|---|---|---|
| gpt-oss-20b | 295 / 1260 | 5.3 |
| gpt-oss-120b | 0 / 1260 | 7.0 |
| qwen3.8-27b | 148 / 1260 | 1.5 |

The cap is per model, so the three would have run in parallel and Groq would
have cleared in about seven days. That is outside the horizon for this
submission, and the judges were dropped rather than the schedule extended.

## What this costs the paper

The `gpt-oss` 20B/120B pair was one of two within-family size ladders. It is
gone. The `qwen-3` 8B/14B/32B ladder on HuggingFace is unaffected and carries
the scale contrast on its own, so the claim survives with one ladder instead of
two.

`qwen3.8-27b` was also the Groq half of a provider contrast against
`qwen3.8-27b-hf` — same 27B weights, different host. That contrast is gone too;
`qwen3.8-27b-hf` remains in the sweep as an ordinary judge.

## Restoring them

A paid Groq tier, not a code change. Flip `verified` back to `True` in
`src/judge_registry.py`, move these files back into `raw/`, and resume — the
runner is append-only and skips completed rows, so the 443 rows here are still
good and would not be paid for twice.
