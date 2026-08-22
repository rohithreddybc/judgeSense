# One-Shot Run Contract

The full sweep is affordable once. This document is the standing instruction for
any session that resumes this work: what must be true before spending, how to
prove it, and what to do when a stage disagrees with the plan.

The governing rule is **detect corruption while it is still cheap**. Every stage
below is ordered so that the cheapest test that can falsify the run runs first.
A stage that fails stops the sequence. No stage is skipped because an earlier
run of it passed — the point is to re-establish the property against the code
and data that exist right now.

## 0. Run from the right clone

The v2 work lives on `origin/main`. A local clone that is behind does not merely
lack fixes: before commit `2e89bfc` this repository had no `data/v2/` at all, so
a run started from a stale clone silently executes the **v1** dataset — the one
whose defects caused the NeurIPS withdrawal. This failure is invisible at the
command line, because `run_v2.py` does not exist there to complain.

    git fetch origin && git status -sb        # must report up to date with origin/main
    python scripts/freeze_dataset.py          # must print "matches the frozen record"

Both must pass in the directory the run will actually be launched from. Not a
sibling directory, not a scratch copy.

## 1. Prove the dataset is the frozen one

`scripts/freeze_dataset.py` hashes item content while excluding `retrieved_at`,
so it detects a changed corpus and ignores a re-download. A mismatch means the
data moved after the audit and every downstream check is void.

    python scripts/data_audit.py --config data/audit_config_v2.json   # 28/28
    python scripts/freeze_dataset.py                                   # 4/4 match

## 2. Prove the harness before the judges

    python -m pytest tests/ -q

These run without keys or network. They cover the properties that decide whether
a crash costs one row or the whole run: resumability, last-write-wins between
runner and reader, per-item repeat gating, strict parsing, usage capture.

## 3. Prove the metrics respond to known inputs

Unit tests show functions behave. They do not show the pipeline computes the
right number end to end. Feed it simulated judges whose scores are known by
construction and check the output equals the arithmetic requirement:

- oracle judge — accuracy 1.000, JSS 1.000
- anti-oracle — accuracy 0.000, **JSS still 1.000** (JSS must never consult ground truth)
- coin flip — chance-corrected kappa CI covering 0.000
- position follower — accuracy near 0.500 on pairwise tasks; near 1.000 means the
  A/B swap design is broken and the position-bias analysis is invalid
- length follower and lexical-overlap follower — accuracy inside [0.45, 0.55];
  outside that band a shortcut is still exploitable and the task measures the
  shortcut rather than judgement

Write simulated outputs to a temporary directory. Never into `data/results_v2/raw`,
which must contain only real paid calls.

## 4. Pre-flight before the loop

    python src/run_v2.py --judges <judge> --tasks factuality --limit 5 --preflight-only

One live call per judge. A wrong model id, an expired key, or a provider that
rejects the token parameter surfaces here for cents instead of three thousand
calls in. Pre-flight passing is necessary and not sufficient: it proves the call
path, not the prompts.

## 5. Smoke, then read the records

    python src/run_v2.py --judges <judge> --tasks factuality --limit 5 --yes

Then open the raw JSONL and confirm, per record: `decision_a` and `decision_b`
are real labels rather than `UNCLEAR`; `error` is null; `usage_a`/`usage_b` carry
non-null token counts; `ts` is present. A high `UNCLEAR` rate here is the single
most valuable signal available before the sweep — it means the prompt asks for a
format the parser does not accept, and the full run would return noise at full
price.

Re-run the identical command afterwards. It must report `resumed=5 new=0` and
issue no calls. That is the resumability guarantee tested against reality rather
than against a stub.

## 6. Stage the sweep

Run judges in waves, cheapest family first, and regenerate metrics between waves.
A defect that survives every check above will show up as an implausible metric in
wave one, when most of the budget is still unspent.

    python scripts/regenerate_results.py
    python scripts/summarize_usage.py            # cost null without --prices
    python scripts/summarize_usage.py --prices <your-price-table>.json

Costs are reported as null unless a price table is supplied; they are never
estimated. A `lower_bound: true` flag means some call returned no usage, so the
figure understates spend.

## Concurrency

Completed rows are read once at cell start and the output file is append-only
with no lock. Two processes over the same `(judge, task)` will each work the
whole remaining backlog and both bill for it. Parallelise by splitting `--judges`
or `--tasks` into disjoint sets, one process per set. This is documented in the
runner's output and is not enforced in code.

## When a stage disagrees with the plan

Stop. A stage failing is the system working. The expensive failure mode is not a
failed check — it is a passed run whose numbers are wrong, discovered after the
budget is gone.
