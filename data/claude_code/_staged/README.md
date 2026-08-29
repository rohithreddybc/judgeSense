# Claude Code harness cells — evidence, NOT judges for the main table

These cells were produced through Claude Code subagents rather than the
Anthropic API, because the API key has no credit. They are deliberately staged
rather than published into `data/results_v2/raw/`.

**They must not enter the main results table.** They carry no declared decoding
configuration, so they are comparable to themselves across arms and not to the
API judges. They are the evidence behind a negative result about the transport
itself.

## Part 1 — the transport control

The same model, `claude-haiku-4-5`, run two ways on all four tasks. Everything
about the model, the items, the templates and the repeat design is held fixed;
only the transport differs.

| task | | API (matched budget) | Claude Code (batch 50) |
|---|---|---|---|
| factuality | JSS_para | 0.9560 | 0.9000 |
| | **ceiling** | **1.0000** | **0.9100** |
| | dJSS | −0.0440 | −0.0100 |
| coherence | JSS_para | 0.7920 | 0.5600 |
| | **ceiling** | **0.9980** | **0.6840** |
| | dJSS | −0.2060 | −0.1240 |
| relevance | JSS_para | 0.9300 | 0.8000 |
| | **ceiling** | **0.9900** | **0.8290** |
| | dJSS | −0.0580 | −0.0290 |
| preference | JSS_para | 0.8960 | 0.8690 |
| | **ceiling** | **0.9650** | **0.8810** |
| | dJSS | −0.0810 | −0.0120 |

Transport shift in dJSS: **+0.034**, **+0.082**, **+0.029**, **+0.069** — all
four outside the declared smallest effect of interest (0.02). The ceiling falls
faster than the paraphrase agreement on every task, so dJSS *shrinks* rather
than grows.

## Part 2 — is the collapse the model or the transport?

Two further judges, coherence only (the task where the collapse is largest).
Neither has an API counterpart, so neither supports a paired contrast. They
answer a narrower question.

| judge (harness, coherence) | JSS_para | **ceiling** | dJSS |
|---|---|---|---|
| cc-haiku-4-5 | 0.560 | **0.684** | −0.124 |
| cc-sonnet-5 | 0.520 | **0.616** | −0.096 |
| cc-opus-5 | 0.696 | **0.848** | −0.152 |

A more capable model narrows the collapse without arresting it. All three
ceilings sit below the level at which any cell in this paper is read, and the
strongest of the three is still below the 0.864 pilot that was discarded
outright.

## Why this disqualifies the transport

The repeat ceiling is the judge's agreement with itself on byte-identical
prompts. Through the harness it falls as low as **0.616** — the judge
disagreeing with itself on well over a third of identical inputs.

The paper's own standard rules this out. A pilot at ceiling 0.864 was discarded
because "a ceiling that low can absorb the effect being measured", and the
Sonnet relevance cell at 0.789 is reported as uninterpretable. Nine of the ten
reported cells sit at 0.916 or above.

## Ruled out as causes

- **Not a batching-alignment bug.** Every batch across all cells returned 50/50
  ids, none missing, none extra. The 40 cross-model batches aligned on first
  dispatch with no re-issues.
- **Not a parsing failure.** Zero malformed answers; the 1–5 coherence scale
  parsed cleanly.
- **Not a mirroring bug.** The repeat prompts were verified byte-identical to
  the arm prompts they baseline.
- **Not a truncated batch accepted in silence.** One Haiku preference batch came
  back eight items short; it was rejected and re-issued rather than accepted,
  because a silently shortened batch would shift every later label within it.

The transport is what moved.

## Status

`cc-fable-5` was not run, and `cc-opus-5` / `cc-sonnet-5` were not run beyond
coherence. Those runs would have cost roughly 13M further tokens to produce
cells whose ceilings cannot support the endpoint in any case.
