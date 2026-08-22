# Paper scrutiny protocol

Standing instructions for auditing this manuscript before submission. Written
after a blind five-seat review returned reject-in-present-form on a draft that
had already passed a methodology review, a code review, 428 tests and a clean
LaTeX build.

The governing lesson from that round: **not one of the serious defects was
findable by reading the paper.** Every one required comparing a sentence against
the thing it described.

- Table 1 rendered a smoke-test row for a judge named `goodjudge`, because a unit
  test wrote to the tracked artifact. The prose around it was fine.
- The Manski bounds were `null` in all twelve cells because a renamed string
  constant orphaned one comparison. Three sections claimed to report them and one
  described their width.
- The malformed-rate sentence stated its own number backwards, arguing against
  the figure the pipeline actually produces.
- Three citations supported claims their papers do not make.
- The pre-registration was described as fixing thresholds that `git log` shows
  were committed seventy minutes after the first results.

A reader who only reads prose finds none of this. A reader who opens the repo
finds all of it in an afternoon, and a reviewer with a withdrawal in the paper's
history will look.

So: **the unit of verification is the claim–artifact pair, never the sentence.**

---

## Pass 1 — Every number traces to a command

Extract every numeral in the manuscript that is not a section reference, an
equation constant, or a year. For each, name the file and the command that
produces it, and run that command. A number is acceptable only if it appears in
committed output.

Fail conditions:

- The number appears nowhere in `data/`, `tables/`, or `outputs/`.
- The number is computed by a shell one-liner rather than by the pipeline. If the
  paper says no number is transcribed by hand, that must be literally true;
  otherwise change the sentence.
- The number is right but stale — the artifact has since been regenerated.
- The number is a *different quantity* than the sentence claims. Check the
  denominator explicitly: a rate over rows is not a rate over items, a rate over
  attempted calls is not a rate over all calls, and support for a delta is not
  `n_rows`.

Then check the reverse direction: for every metric the pipeline emits, is it
either reported or deliberately unreported? A field that is computed and never
mentioned is usually a finding somebody chose not to look at.

## Pass 2 — Every process claim against `git log`

Search the manuscript for: pre-registered, committed, in advance, before the run,
declared, fixed, frozen, we verified, we measured, we ran, unchanged.

For each, find the commit and check the timestamp against the thing it claims to
precede. `git log --diff-filter=A`, `git show <commit> -- <file>`, and
`git merge-base --is-ancestor` settle these in minutes.

A claim that a rule preceded data is false if the commit that introduced the rule
post-dates the commit that introduced the data. Weakening the claim is always
available and always cheaper than defending it.

## Pass 3 — Every citation against what the cited paper argues

For each `\citep` and `\citet`, read the bibliography entry's *title* and confirm
the cited work is about the claim it is attached to. Titles alone catch most
errors: a paper titled "…Position Bias…" does not support a verbosity claim.

Where a claim needs support the bibliography cannot give, either find a real
source or ground the claim in this paper's own measurement. Never leave a
plausible-looking citation attached to a claim it does not make.

Also: is the closest prior work cited *as* the closest prior work, or buried in a
list? A domain reviewer who knows that paper will notice.

## Pass 4 — The rendered PDF, not the source

Compile, then inspect the output. Source correctness does not imply rendered
correctness.

- `grep -c '^!' main.log` must be zero. A document that produces pages while
  emitting errors is not a document that builds.
- Undefined citations and references must be zero.
- Extract the text of every table and read it. Confirm each contains the rows the
  prose describes and no row it does not.
- Beware naive substring searches on extracted PDF text: LaTeX kerning splits
  words (`Edw)27(ard`) and ligatures replace characters (`Ra\033` for "Raff").
  A negative result from a substring search is not evidence of absence — strip
  kerning or inspect the raw stream before concluding anything is missing.

## Pass 5 — Internal consistency across every file

The manuscript is not the only thing a reviewer reads. Check that these agree,
pairwise, on every shared fact: `paper/*.tex`, `README.md`, `ERRATA.md`,
`PREREGISTRATION.md`, `ONE_SHOT_RUN_CONTRACT.md`, the HuggingFace card, and the
Croissant metadata.

Facts that have drifted before and must be re-checked every time: item and row
counts per split; the total; the audit-gate check count; the number of judges,
run and unrun; source corpus names; content hashes; which analyses exist.

A supporting document that contradicts the paper is worse than one that does not
exist, because the reviewer trusts it as the authors' own account.

## Pass 6 — Claims scoped to the evidence

For every claim, ask what design would be needed to support it, and whether this
design is that design.

Specific traps in this paper:

- Three judges from **one vendor family**. Any cross-judge regularity is a
  within-family observation. Consistency across three same-family models is also
  a small-sample coincidence risk — with three tasks, a shared ordering is a
  1-in-6 event under the null.
- Ten of twelve cells. "Every measurable cell" is doing quiet work; say ten of
  twelve.
- Effects that clear a threshold on the point estimate but not on the interval.
  Apply the declared smallest-effect-of-interest to intervals, and report the
  cells that fail it as nulls.
- A ten-for-ten rejection of a point null that is false a priori is a statement
  about power, not about judges. Say what the estimator can and cannot show.
- Any cell whose arms differ in parse-failure rate is not evidence about wording.

## Pass 7 — Adversarial reading

Read as a reviewer who wants to reject, then ask of each finding: does the
artifact rule this out, or does it merely not mention it?

- What would make this result appear without the claimed mechanism?
- Which analysis choice, if reversed, weakens the conclusion? Was it pre-declared
  or chosen after seeing data?
- Is any cell counted for one claim and excluded from another? Pick one.
- Is any comparison reported in its most favourable form? Report all of them.
- What does the released artifact contain that contradicts the paper?

## Pass 8 — What is claimed and never delivered

Grep the manuscript for forward promises: "we report", "we show", "Appendix",
"below", "alongside". Confirm each names something that exists in the document.

A `\ref` that resolves is not sufficient — the target must contain the analysis
the sentence promises.

---

## Standing rules

**Weaken rather than defend.** Every finding this round was cheaper to fix by
narrowing a claim than by arguing for it. A narrower true claim survives review;
a broader one invites the reviewer to test it.

**Report the unflattering number.** The refusal bounds are wide, coherence
discriminates only weakly, factuality is a null, one cell is uninterpretable.
Each is in the paper because a reviewer would otherwise find it and wonder what
else was omitted.

**A null is a result.** The pre-registration commits to it. Honour that.

**Never assert an absence from a failed search.** Two names in this paper were
reported missing from a PDF that rendered them correctly.

**Re-run the pipeline before every build.** The paper renders whatever the
artifacts last contained, and a test run has silently overwritten them before.

## Exit criterion

The audit is complete when every claim has a named artifact, every process claim
matches `git`, every citation matches its source, the PDF renders zero errors and
the tables the prose describes, all supporting documents agree, and every claim
is scoped to three judges from one vendor family.

Acceptance cannot be guaranteed and should not be promised. What can be
guaranteed is that nothing in the paper is false, and that anything a reviewer
finds is a disagreement about judgement rather than a discovery of error.
