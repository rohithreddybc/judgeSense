# Recovery worklist

Everything still outstanding after comparing the current paper against the
version actually submitted to NeurIPS, which carried the coauthors' edits and was
not consulted during the rewrite.

Ordering principle: **a paper that contradicts itself is worse than a paper that
omits something.** So self-contradictions come first, then claims that cite
nobody, then content a reviewer will notice is missing. Within each tier, cheaper
first — a sentence beats a section when both close the same gap.

Rule for every item: if the artifact does not support reinstating it, say so and
drop it. v1's results are void; its arguments mostly are not. The distinction is
whether the claim depends on data that has since been rebuilt.

---

## Tier A — The paper contradicts itself. Fix or delete; do not leave both.

**A1. The polarity remap analysis is promised and never delivered.**
`03_benchmark.tex` states: *"We report the remapped analysis alongside the
excluded-pairing analysis so the exclusion's effect on the reported numbers is
visible rather than assumed."* No such analysis appears. `src/polarity.py`
exists, so this is runnable, not hypothetical. Either run it on the v2 data and
report the number, or strike the sentence. The v1 version had the magnitude (a
uniform ~37-point flip rate) and the inference that made it valuable: *uniform
across every judge implies an artifact of the instrument, not behaviour of the
models.* That reasoning survives the rebuild even though the number does not.

**A2. Malformed rate promised per cell, delivered for two.**
`04_metrics.tex`: *"Malformed-output rate is reported per cell alongside the
refusal statistics below."* `05_results.tex` gives it for two cells and Table 1
has no malformed column. Add the column, or narrow the sentence.

**A3. `wei2024systematic` is characterised incompatibly across the two versions.**
The submitted version describes a study of prompt-template effects on LLM-judge
reliability restricted to TL;DR and HH-RLHF. The rewrite calls it *"systematic
surveys of evaluation practice."* Those are different papers. Check the source.
If the submitted version is right, the citation is both mischaracterised and
filed in the wrong paragraph — it belongs in "Judging judges" as near-neighbour
work, not in background.

**A4. "Discriminating" means two different things across the two papers.**
v1: coherence is the only task that separates judges. This paper: the pairwise
tasks discriminate comfortably and coherence only weakly. Not a factual
contradiction — v1 measured spread in JSS across judges, this measures best-judge
accuracy over majority class — but two papers with the same name and overlapping
authors use one word for opposite conclusions. State that the term was
redefined, and why v1's reading was an artifact of fixed-order layout plus five
coherence items.

---

## Tier B — Claims resting on nobody. Each is a design choice the literature justifies.

**B1. `shi2024judging`.** The submitted version's clause — position bias is
*modulated by the quality gap between candidates* — is the prior-work
justification for our own vote-margin stratification in §3.4. We make the design
choice and cite no one. Restore the clause.

**B2. `thakur2024judging`.** Submitted: 13 judges, sensitivity to prompt
complexity and length, general leniency, and the gap that motivates this work —
no reportable single-number metric for a new judge. The rewrite says "alignment
with human labels." We have under-cited the closest competing claim to our own,
which invites the question of whether prompt-complexity sensitivity was already
shown. v1 pre-empted it.

**B3. `sclar2024quantifying`.** The 76-accuracy-point swing was doing
motivational work; "a wide spread" is not.

**B4. `razavi2025benchmarking`.** That current methods struggle to *predict*
which prompts trip a model is an argument for measuring sensitivity instead —
i.e. for this paper's approach.

**B5. `chiang2023can`.** "Broadly positive but task-dependent agreement"
prefigures our own headline. Currently unused.

**B6. `arabzadeh2025human`.** Keep the rewrite's conceptual contrast, which is
better than v1's. Add back the scope (90 prompts, three judges, Cohen's κ against
TREC labels) and note the overlap neither version mentions: they use TREC labels
and our relevance split is BEIR TREC-COVID.

**B7. Length as a "well-known confound"** with no citation for it being well
known. Keep our own 70% measurement — that was the right call — and restore the
citation beside it.

**B8. "Why not just do multi-prompt evaluation?"** The submitted version answered
this against three named alternatives: multi-prompt evaluation needs re-running
per judge on a study-specific distribution, while a fixed validated set yields a
judge-level property *computed once and portable*; variance approaches need gold
labels, ours needs none; ensemble judging manages sensitivity without revealing
whether any single judge is usable alone. This is the obvious reviewer objection
and the paper currently has no answer on the page.

---

## Tier C — Content a reviewer will notice is missing.

**C1. The templates are never printed.** A paper whose entire construct is that
phrasing changes verdicts asks the reader to accept conclusions about wording
without seeing any wording. They are in `src/dataset_builder_v2.py`. Print all
twenty in an appendix. This also makes the claim about relevance T2/T3 lacking a
single-token constraint checkable rather than assertable.

**C2. No figures.** Four small tables the reader must assemble into a story. A
forest plot of the ten ΔJSS estimates with their intervals and the ±0.02 band
would replace two tables and make "six clear it, four do not" visible at a
glance.

**C3. No malformed output is ever shown.** We argue at length that malformed is
not refusal, and claim 22.4% is a genuine instruction-following failure rather
than a parser artifact. Quoting three of the 168 settles it. They are in the
committed raw outputs.

**C4. Compute and cost.** The submitted version itemised calls, dollars and
days. This paper's stated limitation is that the API balance was exhausted
mid-run, with no accounting anywhere. Reporting it makes the two missing cells
legible as a budget constraint rather than an unexplained gap.

**C5. The system prompt and the inference stack are never disclosed.** The exact
system prompt is one line. The submitted version also noted that on some
providers a system prompt is delivered as a user-turn prefix because the client
transmits no system role — which is itself an unintended paraphrase manipulation,
and directly on-thesis.

**C6. Practitioner recommendations.** Randomise A/B order on every call; report
position-bias rate beside any stability figure; freeze phrasing where sensitivity
is high. The absolute-JSS tiers are correctly dead — the endpoint changed — but
these survive the rebuild and their absence narrows the audience from
practitioners to methodologists.

**C7. The JSS × accuracy 2×2.** Both axes are already on the page and never
crossed. High stability with low accuracy is consistently wrong and rephrasing
will not help; low stability with high accuracy is usable only under frozen
prompts and not comparable across teams. Cheapest high-value addition here.

**C8. What the benchmark is for after publication.** Auditing a candidate judge,
leaderboard reporting, and regression testing when a pipeline changes its judge
or prompt. The regression-testing case is what justifies a fixed set over ad-hoc
prompts.

**C9. Broader impacts and licensing.** The gaming threat model — someone who
knows a judge's sensitivity profile can craft prompts that obtain a desired
verdict — is genuinely dual-use and conspicuous by its absence in a paper this
careful. Licensing: all four corpora are public and permissively licensed, with
no new human data collected.

**C10. Acknowledgements.** Free-tier credits from providers whose models are
benchmarked is a disclosable competing interest, and it is the funding constraint
behind our own exhausted-balance limitation.

**C11. Judge-selection rationale.** We name three judges and say eleven are
unrun, without saying why these three.

**C12. Minimum detectable effect.** We report four cells as indistinguishable
from the 0.02 threshold. An MDE at the shipped cluster counts makes that
quantitative rather than merely candid.

---

## Standing constraints

- Every number added must come from a committed artifact or a command that can be
  re-run. No figure enters the paper by hand.
- Re-run `scripts/regenerate_results.py` before every build.
- Rebuild and check: zero TeX errors, zero undefined references, zero control
  characters in source. A `\ref` that lost its backslash renders as literal text
  and raises no error.
- Verify names in the rendered PDF by inspecting the stream, not by substring
  search — kerning and ligatures defeat it.
- Run the full suite before committing; a test has silently overwritten a shipped
  artifact before.
- When an item cannot be honestly reinstated, delete the claim rather than
  soften it into something unfalsifiable.
