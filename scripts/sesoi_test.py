"""Test the pooled contrasts against the SESOI, not against zero.

WHY

The paper already concedes that under the strict policy the endpoint estimates
a non-positive quantity that is zero only when two prompts induce identical
response distributions. A point null of exactly zero is therefore false before
any data are collected, and rejecting it is a statement about power rather than
about the judge. A reviewer who notices this will ask for the test that
actually bears on the claim: is the effect larger in magnitude than the
smallest difference we declared of interest?

This runs that test, one-sided, H0: mean dJSS >= -SESOI against H1: < -SESOI,
using the same per-task pooling and the same Holm correction over four
contrasts.

Reads data/results_v2/multiplicity.json, writes sesoi_test.json beside it.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from multiplicity import holm  # noqa: E402

SRC = REPO / "data" / "results_v2" / "multiplicity.json"
OUT = REPO / "data" / "results_v2" / "sesoi_test.json"
TASKS = ("coherence", "factuality", "preference", "relevance")


def main() -> int:
    d = json.load(io.open(SRC, encoding="utf-8"))
    sesoi = d["sesoi"]
    pooled = d["pooled_by_task_holm"]

    rows = {}
    for t in TASKS:
        v = pooled.get(t)
        if not v:
            continue
        mean, se, n = v["mean_delta"], v["se"], v["n_judges"]
        # one-sided: how far below -SESOI is the mean, in standard errors
        z = (mean + sesoi) / se if se > 0 else float("-inf")
        p = 0.5 * math.erfc(-z / math.sqrt(2)) if z < 0 else 1 - 0.5 * math.erfc(z / math.sqrt(2))
        rows[t] = {"n_judges": n, "mean_delta": mean, "se": se,
                   "margin_beyond_sesoi": round(mean + sesoi, 4),
                   "z": round(z, 3), "p_one_sided": p}

    corrected = holm({t: r["p_one_sided"] for t, r in rows.items()}, alpha=0.05)
    for t, c in corrected.items():
        rows[t].update(c)

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps({"sesoi": sesoi, "null": "mean dJSS >= -SESOI",
                    "alternative": "mean dJSS < -SESOI", "by_task": rows}, indent=1))

    print(f"  H0: mean dJSS >= -{sesoi}   (one-sided, Holm over four contrasts)")
    print(f"  {'task':12}{'n':>4}{'mean':>9}{'margin':>9}{'z':>8}{'p':>11}  Holm")
    for t in TASKS:
        r = rows.get(t)
        if not r:
            continue
        print(f"  {t:12}{r['n_judges']:>4}{r['mean_delta']:>9.4f}"
              f"{r['margin_beyond_sesoi']:>9.4f}{r['z']:>8.2f}{r['p_one_sided']:>11.2e}"
              f"  {'reject' if r.get('reject') else 'RETAIN'}")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
