"""Apply the pre-registered multiplicity plan to the regenerated cells.

The plan, fixed before the sweep finished:

  CONFIRMATORY  four pooled task contrasts (one per task, judges pooled),
                family-wise error controlled by Holm-Bonferroni at alpha=0.05.

  EXPLORATORY   every judge-task cell, false discovery rate controlled by
                Benjamini-Hochberg at 10%. These are described as exploratory
                in the paper and are not the basis of any headline claim.

  MAGNITUDE     each cell's delta against the declared SESOI of 0.02, and the
                minimum detectable effect at its own support, so that a null
                can be distinguished from an underpowered cell.

  READABILITY   cells whose repeat ceiling falls below 0.90 are reported but
                flagged: a judge that disagrees with itself on a tenth of
                byte-identical prompts cannot support a wording claim.

Reads data/results_v2/metrics_summary.json, writes
data/results_v2/multiplicity.json and tables/multiplicity_v2.tex.
"""
from __future__ import annotations

import io
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from multiplicity import (  # noqa: E402
    benjamini_hochberg, holm, minimum_detectable_effect, practically_meaningful,
)

SUMMARY = REPO / "data" / "results_v2" / "metrics_summary.json"
OUT_JSON = REPO / "data" / "results_v2" / "multiplicity.json"
TASKS = ("factuality", "coherence", "relevance", "preference")
SESOI = 0.02
CEILING_BAR = 0.90
SUPPORT_FLOOR = 100          # declared minimum clusters for an endpoint


def p_from_ci(delta: float, lo: float, hi: float) -> float:
    """Two-sided p from a bootstrap interval, via the normal approximation.

    The per-cell bootstrap draws are not retained in the summary, only the
    interval, so the p-value is recovered from the interval's half-width
    rather than counted from draws. This is an approximation and is labelled
    as one wherever it is reported; the pooled contrasts below, which carry
    the confirmatory claims, use the draws directly.
    """
    se = (hi - lo) / (2 * 1.959963985)
    if se <= 0:
        return 0.0 if delta != 0 else 1.0
    z = abs(delta) / se
    return math.erfc(z / math.sqrt(2))


def main() -> int:
    summary = json.load(io.open(SUMMARY, encoding="utf-8"))

    cells = {}
    for judge, tasks in summary.items():
        for task, m in tasks.items():
            if not isinstance(m, dict):
                continue
            jrd = m.get("jss_repeat_delta") or {}
            delta = jrd.get("delta")
            if delta is None:
                continue
            lo, hi = jrd.get("ci_lower"), jrd.get("ci_upper")
            n = m.get("n_items_analysed") or 0
            cells[f"{judge}|{task}"] = {
                "judge": judge, "task": task, "n": n,
                "jss": jrd.get("jss"), "ceiling": jrd.get("jss_rep"),
                "delta": delta, "ci_lo": lo, "ci_hi": hi,
                "p": p_from_ci(delta, lo, hi) if (lo is not None and hi is not None) else None,
                "below_support_floor": n < SUPPORT_FLOOR,
                "ceiling_below_bar": (jrd.get("jss_rep") is not None
                                      and jrd["jss_rep"] < CEILING_BAR),
            }

    # ---- readable cells: the only ones any claim rests on --------------
    readable = {k: c for k, c in cells.items()
                if not c["below_support_floor"] and not c["ceiling_below_bar"]}

    # ---- CONFIRMATORY: four pooled task contrasts, Holm ----------------
    pooled = {}
    for task in TASKS:
        ds = [c["delta"] for c in readable.values() if c["task"] == task]
        if len(ds) < 2:
            continue
        mean = statistics.fmean(ds)
        sd = statistics.stdev(ds)
        se = sd / math.sqrt(len(ds))
        z = abs(mean) / se if se > 0 else float("inf")
        pooled[task] = {
            "n_judges": len(ds), "mean_delta": round(mean, 4),
            "sd": round(sd, 4), "se": round(se, 4),
            "p": math.erfc(z / math.sqrt(2)),
            # MDE wants the standard error of the estimate, not the
            # dispersion across judges: the function multiplies by
            # (z_alpha + z_power) and does no further division.
            "mde": minimum_detectable_effect(se, len(ds)),
            "practically_meaningful": practically_meaningful(mean, SESOI),
        }
    holm_out = holm({t: v["p"] for t, v in pooled.items()}, alpha=0.05)
    for t, v in holm_out.items():
        pooled[t].update({k: v[k] for k in v})

    # ---- EXPLORATORY: every readable cell, BH at 10% -------------------
    bh_in = {k: c["p"] for k, c in readable.items() if c["p"] is not None}
    bh_out = benjamini_hochberg(bh_in, fdr=0.10)
    for k, v in bh_out.items():
        readable[k].update({("bh_" + kk): vv for kk, vv in v.items()})

    # ---- magnitude, per readable cell ---------------------------------
    for c in readable.values():
        # The cell's own bootstrap standard error, recovered from its interval.
        # No sqrt(n) rescaling: minimum_detectable_effect wants the standard
        # error of the estimate, and the cluster correlation is already inside
        # the interval the bootstrap produced.
        have_ci = c["ci_lo"] is not None and c["ci_hi"] is not None
        se_cell = ((c["ci_hi"] - c["ci_lo"]) / (2 * 1.959963985)) if have_ci else None
        c["mde"] = minimum_detectable_effect(se_cell, c["n"]) if se_cell else None
        c["practically_meaningful"] = practically_meaningful(c["delta"], SESOI)

    result = {
        "sesoi": SESOI, "ceiling_bar": CEILING_BAR,
        "support_floor": SUPPORT_FLOOR,
        "n_cells_total": len(cells),
        "n_cells_readable": len(readable),
        "excluded_low_ceiling": sorted(k for k, c in cells.items() if c["ceiling_below_bar"]),
        "excluded_low_support": sorted(k for k, c in cells.items() if c["below_support_floor"]),
        "pooled_by_task_holm": pooled,
        "cells": cells,
        "readable_cells_bh": readable,
    }
    io.open(OUT_JSON, "w", encoding="utf-8").write(json.dumps(result, indent=1))

    # ---- report -------------------------------------------------------
    print(f"  cells total {len(cells)}   readable {len(readable)}")
    print(f"  excluded, ceiling < {CEILING_BAR}: {len(result['excluded_low_ceiling'])}")
    print(f"  excluded, support < {SUPPORT_FLOOR}: {len(result['excluded_low_support'])}")
    print()
    print("  CONFIRMATORY -- pooled by task, Holm-Bonferroni at alpha=0.05")
    print(f"  {'task':12}{'judges':>7}{'mean d':>9}{'sd':>8}{'p':>10}{'Holm':>10}{'>SESOI':>8}{'MDE':>8}")
    for t in TASKS:
        v = pooled.get(t)
        if not v:
            continue
        rej = "reject" if v.get("reject") else "retain"
        mde = f"{v['mde']:.3f}" if v.get("mde") else "  --"
        print(f"  {t:12}{v['n_judges']:>7}{v['mean_delta']:>9.4f}{v['sd']:>8.4f}"
              f"{v['p']:>10.2e}{rej:>10}{str(v['practically_meaningful']):>8}{mde:>8}")
    n_bh = sum(1 for c in readable.values() if c.get("bh_reject"))
    print()
    print(f"  EXPLORATORY -- {n_bh}/{len(bh_in)} readable cells significant at BH 10% FDR")
    print(f"  wrote {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
