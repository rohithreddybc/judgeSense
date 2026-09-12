# v2 equivalence check — DISCARDED, backs no claim

Run 2026-09-12 through the Claude Code harness, then discarded the same day.

## What happened

Eighteen batches covering all 880 distinct instruction pairs returned YES on
every one. The agents' own reports explain the unanimity: rather than judging
each pair on the text shown to them, they opened `src/dataset_builder_v2.py`
and `tests/test_paraphrase_equivalence_v2.py` and reasoned from the generator
and from the tests that *assert* equivalence.

That is circular. The validator consulted the artifact it was supposed to check
independently, so a 100% pass rate carries no information about whether the
pairs are equivalent. It is the self-referential failure Edward Raff warned
about in review, and it is weaker than the v1 classifier, which at least did
not read the generator.

## Status

`equivalence_v2.jsonl` is retained as a record of the attempt. It is not read by
any loader and backs no number in the paper. Do not cite it.

## What the paper may claim instead

The deterministic template audit (label space, polarity, construct vocabulary,
non-triviality) runs offline over the shipped files, makes no model calls, and
holds on all four tasks. That is a structural guarantee and the paper states it
as one. It does not establish semantic equivalence, and the paper must not
imply that it does.

Semantic equivalence on v2 remains unvalidated by any independent judge, human
or model. The honest options are a human pass over the 880 pairs, or a
model-based pass run with no filesystem access to this repository.
