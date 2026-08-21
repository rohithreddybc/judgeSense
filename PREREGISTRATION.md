# Pre-registered analysis plan

Written before the full sweep was run and committed ahead of it, so the primary
endpoint cannot be chosen after seeing which comparison happens to be
significant. A one-shot run is credible only if the analysis was fixed in
advance; this file is that fixture, and the paper cites its commit hash.

## Primary endpoint

**ΔJSS = JSS_paraphrase − JSS_repeat**, per judge–task cell.

A NEGATIVE ΔJSS means the judge is less stable under rewording than under
resampling, which is the effect the paper is about.

(Sign-convention correction, made after the first wave landed and recorded here
rather than silently: this file originally wrote the difference the other way
round while `metrics_v2.jss_repeat_delta` — written and tested well before the
run — computes paraphrase minus repeat. The quantity, the estimator, the
clustering unit and the decision rule below are unchanged; only the direction of
subtraction in this prose was wrong. Both orderings were always visible in the
committed output, which reports `jss`, `jss_rep` and `delta` side by side.)

JSS_paraphrase is agreement between the judge's decisions on two
meaning-equivalent phrasings of the same item. JSS_repeat is agreement between
two calls on the *identical* phrasing, which establishes the decoding-noise
ceiling for that judge and task.

ΔJSS is the endpoint rather than raw JSS because raw JSS confounds paraphrase
sensitivity with a judge's ordinary sampling variance. A judge that is merely
noisy scores low on both; only a judge that is specifically destabilised by
rewording shows a negative Δ. The claim the paper makes is about the second
thing.

Uncertainty: 95% cluster bootstrap CI, 2,000 resamples, `cluster_unit="item"`,
via `metrics_v2.cluster_bootstrap_ci`. Item-level clustering is mandatory and
enforced in code: the two `ab_order` rows of a pairwise item share an `item_id`
and the repeat arm nests inside it, so any looser unit understates uncertainty.

### Multiplicity and the decision rule

Fourteen judges across four tasks is 56 intervals. A per-cell "CI excludes zero"
rule applied to all of them yields roughly three spurious findings under a global
null, and whichever cells clear will inevitably be the ones the narrative
foregrounds. So the rule is stated in three parts, before the data exist:

1. **One primary contrast.** The confirmatory claim is the ΔJSS pooled across
   judges within each task, not any individual judge–task cell. Four intervals,
   one per task, Holm-corrected across the four.
2. **Everything per-cell is exploratory.** The 56 cell-level intervals are
   reported in full with Benjamini–Hochberg adjusted values at a 10% false
   discovery rate, and are described as exploratory in the text. No individual
   judge is named as unstable on the basis of an unadjusted cell.
3. **A smallest effect of interest.** |ΔJSS| < 0.02 is declared not practically
   meaningful in advance, whatever its interval does. A judge that loses two
   points of agreement under rewording is not thereby unusable, and an interval
   that excludes zero at that magnitude is a statement about sample size rather
   than about judges.

The minimum detectable effect at the shipped cluster counts is reported with the
results, so a null is distinguishable from an underpowered test.

The direction of an effect is read from the sign, never chosen after the fact.

### Discrimination floor: the control that stops "clean" meaning "useless"

The shortcut controls above establish that no tested heuristic beats chance.
That is a floor, and a floor alone is satisfiable by a benchmark on which
*nothing* works, including a competent judge. A construction that suppresses
exploitable signal can suppress genuine signal by the same mechanism, and the
prior version of this work was withdrawn partly for tasks that did not
discriminate.

We therefore pre-commit a ceiling as well as a floor. On each task, the
best-performing judge must exceed the following position-corrected accuracy for
the task to be reported as discriminating:

| Task | Required | Chance |
|---|---|---|
| factuality | 0.75 | 0.50 |
| coherence (exact-match on 1–5) | 0.40 | 0.20 |
| relevance | 0.70 | 0.50 |
| preference | 0.65 | 0.50 |

A task whose best judge falls below its threshold is reported as
non-discriminating, and no judge ranking is drawn from it. This is declared here
so that it constrains the write-up rather than being chosen once the numbers are
visible; the thresholds are deliberately modest, since the claim they support is
only that the task carries signal a judge can find.

### Support floors

ΔJSS is not emitted for a cell with fewer than 100 clusters, or where more than
50% of items lack a usable pair on either side. Both are enforced in
`scripts/regenerate_results.py` rather than left to inspection.

## Support: which pairs enter JSS

Each arm call ends in exactly one outcome: a **verdict** (parsed to a label), a
**refusal** (the provider flagged the response as declined — Anthropic
`stop_reason="refusal"`, OpenAI-compatible `finish_reason="content_filter"`), or
**malformed output** (a completed response the strict parser cannot map).

JSS and its chance-corrected form are computed over pairs where **both arms
returned verdicts**. A refusal is upstream of any judgement: scoring it as
disagreement asserts the judge rendered two conflicting judgements, which it did
not, and scoring it as a third label would award JSS 1.0 to a judge that refuses
everything.

Refusal is therefore reported as its own construct, per cell:

- `refusal_rate` — refused arm calls / all arm calls
- `refusal_discordance_rate` (RDR) — pairs where exactly one arm was refused
- `consistent_refusal_rate` — pairs where both arms were refused
- `n_verdict_pairs` — the support underlying JSS

RDR is itself a sensitivity statistic and is analysed as one: a nonzero RDR
means a meaning-preserving rewording changed whether the judge was willing to
judge at all.

`jss_strict_refusal_inclusive`, in which every refused arm counts as
disagreement, is reported alongside as a sensitivity analysis, so the
conditioning above can be checked rather than trusted.

Any cross-judge comparison in a cell with nonzero refusals is recomputed on the
**common support** — items every compared judge answered on both arms — so a
reviewer's natural objection, that the judges are being compared on different
subsets, is answered with a reported result rather than an argument.

## Secondary endpoints

1. Judge ranking stability: Kendall's τ between judge rankings under the two
   phrasings, per task.
2. Chance-corrected JSS (Cohen's κ; quadratic-weighted for the Likert coherence
   task), guarding against agreement inflated by a compressed output
   distribution.
3. Position-corrected accuracy on the pairwise tasks, which makes a ceiling
   effect or a position-anchored judge visible directly.
4. Malformed-output rate over both arms.

## Decoding budget

All main results use the **matched** policy: `max_tokens=1024` for every judge
regardless of class. The class-asymmetric alternative confounds judge class with
inference configuration, and measurably truncated real responses mid-sentence.

The budget effect is quantified rather than asserted: an appendix ablation runs a
small subsample under the native policy and reports the ΔJSS difference and the
cap-termination rate under each policy.

## Stopping and staging

The sweep runs cheapest-family-first, with metrics regenerated between waves. If
wave one shows no separation between judges beyond the repeat baseline, that is
recorded as the result rather than treated as a reason to keep spending until
something separates.

## Declared in advance: both outcomes are results

If modern judges turn out to be as stable under paraphrase as under resampling,
that is the paper's finding and it is reported as such. The benchmark's value
does not depend on the sign of the effect: the shortcut controls establish that
the tasks cannot be passed by position, length, or lexical overlap
(0.482–0.514 for each heuristic judge), so a null result is informative about
judges rather than about the instrument.
