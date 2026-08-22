# Invalid run outputs — retained, not used

These 12 files hold 7,854 paid judge calls that cannot enter any result. They are
kept because they cost real money and they document two defects, not because they
are usable.

Two independent reasons they are void:

1. **Temperature.** The Anthropic branch of `usage_meter._request` passed no
   `temperature`, so every Claude judge here sampled at the provider default of
   1.0 while the other providers were pinned to 0.0. The repeat baseline shows
   it directly: `claude-haiku` coherence agrees with itself on only 86.4% of
   byte-identical prompts. That is not a decoding-noise ceiling, it is a
   temperature setting, and it makes any cross-judge comparison uninterpretable.

2. **The dataset changed underneath them.** Every split has since been rebuilt:
   factuality's template assignment was stratified on the label (it had been a
   perfect answer key, 250/250), preference's label rule is now enforced on
   decisive votes and its contradictory-gold items removed. All four content
   hashes differ from the ones these records were produced against.

The `pair_id` values still collide with the rebuilt dataset — all 1,260 of them —
so leaving these in `raw/` would have caused the resume logic to mark every row
complete and the next sweep to issue zero calls while reporting metrics computed
from this data. They are archived here for exactly that reason.

Useful only as an observation about Claude judges at temperature 1.0.
