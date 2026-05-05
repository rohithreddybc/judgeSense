"""
run_jss_example.py — minimal end-to-end JSS demo.

Loads the factuality task, simulates two sets of judge decisions,
and computes the Judge Sensitivity Score.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.load_judgesense import load_task
from utils.compute_jss import compute_jss, flip_rate

random.seed(42)

pairs = load_task("factuality", data_dir=Path(__file__).parent.parent / "data")
print(f"Loaded {len(pairs)} factuality pairs")
print(f"Sample pair_id: {pairs[0]['pair_id']}")
print(f"Sample prompt_a: {pairs[0]['prompt_a'][:60]}...")

decisions_a = [random.choice(["accurate", "inaccurate"]) for _ in pairs]
decisions_b = [random.choice(["accurate", "inaccurate"]) for _ in pairs]

jss = compute_jss(decisions_a, decisions_b)
fr  = flip_rate(decisions_a, decisions_b)

print(f"\nJSS  (simulated): {jss:.3f}")
print(f"Flip rate:        {fr:.3f}")
print("(Simulated random decisions — replace with real judge outputs)")
