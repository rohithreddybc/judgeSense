# Run-readiness spec — prove the pipeline before spending API credit

Goal: a single paid v2 run must produce usable results. The failure mode to
eliminate is a run that COMPLETES but whose outputs are unusable. Every check
below runs with no API calls; each must pass before the real run.

1. **Judge-name parity.** Every verified judge in `judge_registry.JUDGES` must
   resolve through `evaluate.SUPPORTED_MODELS` (same name, provider, model_id,
   key env var). A name only in the registry KeyErrors mid-run.

2. **Dataset↔runner key contract.** Every field `run_v2.run_cell` reads from a
   row (`pair_id`, `item_id`, `prompt_pair_id`, `prompt_a`, `prompt_b`, and the
   `.get()` optionals) must be present in all four `data/v2/*.jsonl` files, or
   safely optional.

3. **Parser correctness per task.** `parse_variant_output(task, raw, "plain")`
   must map realistic judge outputs to the right label for factuality (YES/NO),
   coherence (1-5), relevance/preference (A/B), and return UNCLEAR (not a wrong
   label) on ambiguous text.

4. **Runner→regenerate schema contract.** The record `run_v2` writes must carry
   exactly the fields `regenerate_results` reads (`decision_a/b`, `item_id`,
   `prompt_pair_id`, `ground_truth_label`, `ground_truth_position`,
   `decision_a_repeat`).

5. **Full end-to-end mock over the REAL dataset.** Run all four tasks through
   `run_cell` with a deterministic mock judge (reads the real 1,452 rows),
   then `regenerate_results` over the output. Confirm: every cell produces a
   metrics record; item-level clustering treats the 2 pairwise rows/item as one
   cluster; position-corrected accuracy reads correctly; the repeat-baseline
   delta computes; a LaTeX table is emitted.

6. **Metrics sanity.** On the mock, JSS/kappa/CI are in range, n_items matches
   the dataset, malformed rate is 0 for clean mock output.

7. **Pre-flight catches failure.** A bad model id / missing key must fail
   pre-flight, not the loop.

Any failure is fixed and the check re-run until green.
