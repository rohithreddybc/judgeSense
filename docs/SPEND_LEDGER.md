# Spend ledger

Two budgets are in play and they are not interchangeable. **API tokens** are paid
per token to eight vendors. **Claude Code tokens** come out of the interactive
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

## 2. Multi-vendor API sweep — partial, paused at 68.7%

Six providers, 25 judges. Stopped deliberately mid-run and resumable.

| | |
|---|---|
| calls | 54,428 |
| input | 21.16M |
| output | 1.97M |
| errors | 100 (all recorded, retried on resume) |

Rough cost at list prices: **$7–9**. Groq's share was free.

Remaining to finish the sweep: **~$3**, roughly 31% of ~119,840 planned calls.
Projected total for all 25 judges: **$8.50** (excluding the three flagship
models deliberately dropped, which would have added $10.73 for 11% more judges).

## 3. Claude Code — the agent-harness condition

No API credit. Charged to the interactive subscription.

| task | batches | tokens/batch | total |
|---|---|---|---|
| factuality | 20 | ~43k | ~0.86M |
| coherence | 20 | ~46k | ~0.92M |
| preference | 23 (one re-issued) | ~78k | ~1.79M |
| relevance | 40 | ~81k | ~3.24M |
| probes and dress rehearsal | ~6 | ~45k | ~0.27M |
| **total** | **109** | | **~7.1M** |

Batching is what made this affordable. One item per subagent costs 43,115
tokens; fifty items cost ~45,000, because the overhead is the harness system
prompt loaded once per subagent and not per item. Unbatched, the same four
tasks would have cost **~173M tokens per judge**.

## What was NOT spent, and why

| | |
|---|---|
| `cc-opus-5`, `cc-sonnet-5`, `cc-fable-5` | ~12M tokens saved. The transport control showed the harness cannot support the endpoint, so these were never dispatched. |
| `qwen3.8-max`, `mistral-large-2512`, `magistral-medium` | $10.73 saved. Three flagship models at $2/1M input: 56% of the sweep budget for 11% of the judges, adding no contrast the cheaper tiers do not already provide. |
| Unbatched harness run | ~166M tokens saved per judge. |

## The cheapest thing in the project

The transport control cost **~1.8M Claude Code tokens** (two tasks) to establish
that the harness route was unusable. Running all four Claude judges through it
first would have cost ~14M and produced nothing publishable. Running the control
first is why that did not happen.
