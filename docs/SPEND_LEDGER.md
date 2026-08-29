# Spend ledger

Two budgets are in play and they are not interchangeable. **API tokens** are paid
per token to six vendors. **Claude Code tokens** come out of the interactive
subscription. Everything below is measured from committed artifacts
(`data/results_v2/usage.json`, subagent usage reports), not estimated.

## 1. Anthropic API — the original Claude run (2026-08-22)

The three Claude judges in Table 1. Paid before this work began; the balance is
now zero, which is why the two preference cells are short.

| judge | arm calls | input | output |
|---|---|---|---|
| claude-haiku | 4,280 | 1.76M | 26.7k |
| claude-opus-4-7 | 4,010 | 0.63M | 10.9k |
| claude-sonnet | 3,632 | 0.58M | 23.4k |
| **total** | **11,922** | **2.97M** | **61.0k** |

## 2. Multi-vendor API sweep — IN PROGRESS

Six providers, 26 judges, 31,500 planned rows. Split into twelve resumable
processes, one lock per (judge, task).

Figures below are from the last regenerated `usage.json`, written at the
2026-08-27 pause (19,920 rows). They are a floor, not a total, and this section
is rewritten from a fresh `usage.json` when the sweep finishes.

| | |
|---|---|
| calls | 54,428 |
| input | 21.16M |
| output | 1.97M |
| errors | 100 (all recorded, retried on resume) |

Rough cost at list prices for that portion: **$7–9**. Groq's share was free.

Remaining to finish: **~$3**. Projected total for all 26 judges: **~$8.50**
(excluding three flagship models deliberately dropped, which would have added
$10.73 for 11% more judges).

## 3. Claude Code — the agent-harness condition

No API credit. Charged to the interactive subscription.

### 3a. Transport control — Haiku 4.5, four tasks

The paired condition against the API run of the same model.

| task | batches | tokens/batch | total |
|---|---|---|---|
| factuality | 20 | ~43k | ~0.86M |
| coherence | 20 | ~46k | ~0.92M |
| preference | 22 | ~78k | ~1.72M |
| relevance | 40 | ~81k | ~3.24M |
| probes and dress rehearsal | ~6 | ~45k | ~0.27M |
| **total** | **108 dispatches, 102 manifest batches** | | **~7.0M** |

The dispatch count exceeds the manifest count because one preference batch came
back eight items short and was re-issued rather than accepted.

### 3b. Cross-model harness check — coherence only

Whether the collapsed repeat ceiling is a Haiku property or a transport
property. Neither judge has an API counterpart, so neither supports a paired
contrast; both were run on coherence alone, the task where the collapse is
largest.

| judge | batches | mean tokens/batch | total |
|---|---|---|---|
| cc-opus-5 | 20 | ~58.2k | ~1.16M |
| cc-sonnet-5 | 20 | ~66.4k | ~1.33M |
| **total** | **40** | | **~2.49M** |

All 40 batches returned 50 aligned labels on the first dispatch. No re-issues,
no malformed answers, no missing ids.

Batching is what made any of this affordable. One item per subagent costs
43,115 tokens; fifty items cost ~45,000, because the overhead is the harness
system prompt loaded once per subagent and not per item. Unbatched, the four
Haiku tasks alone would have cost **~173M tokens**.

## What was NOT spent, and why

| | |
|---|---|
| `cc-opus-5`, `cc-sonnet-5` on the other three tasks | ~7M tokens saved. Coherence answers the question the check was run to answer; the remaining tasks would have added cost without adding a contrast, since neither judge has an API counterpart to be paired against. |
| `cc-fable-5` | ~6M tokens saved. A third model does not change a conclusion two already support. |
| `qwen3.8-max`, `mistral-large-2512`, `magistral-medium` | $10.73 saved. Three flagship models at $2/1M input: 56% of the sweep budget for 11% of the judges, adding no contrast the cheaper tiers do not already provide. |
| Unbatched harness run | ~166M tokens saved on the Haiku control alone. |

## The cheapest thing in the project

The transport control cost **~1.8M Claude Code tokens** (two tasks) to establish
that the harness route was unusable as a judge condition. Running all four
Claude judges through it as judges first would have cost ~14M and produced
nothing publishable. Running the control first is why that did not happen.
