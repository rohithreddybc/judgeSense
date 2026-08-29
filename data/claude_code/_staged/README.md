# Claude Code transport control — evidence, NOT judges for the main table

These two cells were produced through Claude Code subagents rather than the
Anthropic API, because the API key has no credit. They are deliberately staged
rather than published into `data/results_v2/raw/`.

**They must not enter the main results table.** They are the evidence behind a
negative result about the transport itself.

## What was measured

The same model, `claude-haiku-4-5`, run two ways. Everything about the model is
held fixed; only the transport differs.

| | API (matched budget) | Claude Code (batch 50) |
|---|---|---|
| factuality JSS_para | 0.9560 | 0.9000 |
| factuality **ceiling** | **1.0000** | **0.9100** |
| factuality dJSS | −0.0440 | −0.0100 |
| coherence JSS_para | 0.7920 | 0.5600 |
| coherence **ceiling** | **0.9980** | **0.6840** |
| coherence dJSS | −0.2060 | −0.1240 |

Transport shift: **+0.034** on factuality (1.7× SESOI), **+0.082** on coherence
(4.1× SESOI). Both outside the declared smallest effect of interest.

## Why this disqualifies the transport

The repeat ceiling is the judge's agreement with itself on byte-identical
prompts. Through the harness it falls to 0.910 and then to **0.684** — the judge
disagreeing with itself on nearly a third of identical inputs.

The paper's own standard rules this out. A pilot at ceiling 0.864 was discarded
because "a ceiling that low can absorb the effect being measured", and the
Sonnet relevance cell at 0.789 is reported as uninterpretable. 0.684 is worse
than both. Nine of the ten reported cells sit at 0.916 or above.

The consequence is visible in the numbers: dJSS shrinks toward zero in both
tasks, because harness noise is eating the effect the endpoint exists to isolate.

## Ruled out as causes

- **Not a batching-alignment bug.** All 40 batches across both tasks returned
  50/50 ids, none missing, none extra.
- **Not a parsing failure.** Zero malformed answers; the 1–5 coherence scale
  parsed cleanly.
- **Not a mirroring bug.** The repeat prompts were verified byte-identical to
  the arm prompts they baseline (50/50 on the checked batch).

The transport is what moved.

## Status

`cc-opus-5`, `cc-sonnet-5` and `cc-fable-5` were NOT run. Doing so would have
cost roughly 12M further tokens to produce judges whose ceilings cannot support
the endpoint.
