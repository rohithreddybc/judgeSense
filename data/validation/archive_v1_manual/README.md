# Archived v1 manual-validation records — do not use

These four files were moved here from `data/validation/manual/` on 2026-08-25.
They are retained for the record and are **not** part of the v2 release, are not
read by any loader, and back no claim in the paper.

## What they are

500 records (125 per task) carrying `manual_label`, `reviewer: "rohit"`, and
timestamps from 2026-05-01/02. Their `pair_id` values (`fact_001`, `cohe_001`,
`pref_001`, `relv_001`) are the **v1** identifier scheme. No v2 item uses those
ids, so nothing in the shipped dataset joins to them.

## Why they were pulled

The repository's own audit gate rejects three of the four on
`annotation_timing`:

| file | median gap between decisions | floor |
|---|---|---|
| `relevance_manual.jsonl` | 0.664 s | 2.0 s |
| `preference_manual.jsonl` | 0.704 s | 2.0 s |
| `coherence_manual.jsonl` | 1.818 s | 2.0 s |
| `factuality_manual.jsonl` | 5.297 s | 2.0 s (passes) |

A person reading two full prompts and choosing a label cannot sustain a
sub-second decision rate across 125 consecutive items. The files are named
`*_manual.jsonl` and attributed to a named reviewer, so on their face they
assert human annotation that the timing contradicts. That is the same shape as
the defect that caused v1 to be withdrawn — an artifact whose name claims more
than its contents support — and it is the reason they are quarantined rather
than left in the validation path.

## What the paper says

The manuscript does **not** claim human validation of paraphrase equivalence. It
says the opposite, in Section 3:

> establishing equivalence would require human annotation or a model-in-the-loop
> study we have not run.

That statement is accurate and is unaffected by this archive.

## If you want them back

They are tracked in git and were moved with `git mv`, so the change is
reversible. Before restoring any of them to an evidentiary role, establish how
they were produced; the timing is not consistent with the attribution they
carry.
