"""Recompute the pooled contrasts with judges clustered by vendor.

WHY

The pooled contrast treats each judge's cell as an independent draw. The roster
is not independent: 25 judges come from 8 vendors, and one vendor supplies
seven of them. Sibling checkpoints share pretraining corpora, instruction-tuning
recipes and decoding defaults, so a phrasing sensitivity learned once reappears
across the siblings and is counted as many times as there are siblings.

This is the exact fault the paper diagnoses in its own predecessor, whose
"task dominates judge" claim rested on three same-vendor judges. Committing it
at the level of the confirmatory test would be worse.

ESTIMATOR

Unweighted mean of judge-level dJSS, with a cluster-robust (CR1) standard error
at the vendor level:

    e_g = sum_{i in g} (d_i - dbar)
    V   = G/(G-1) * (n-1)/(n-1) * sum_g e_g^2 / n^2
    SE  = sqrt(V),  t = dbar/SE,  df = G - 1

Inference uses t with G-1 degrees of freedom, not z: with eight clusters the
normal approximation is optimistic. Both the point-null and the SESOI test are
reported, each Holm-corrected over its own family of four.

    python scripts/cluster_by_vendor.py
"""
from __future__ import annotations

import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from judge_registry import JUDGES  # noqa: E402
from multiplicity import holm  # noqa: E402

MULT = REPO / "data" / "results_v2" / "multiplicity.json"
OUT = REPO / "data" / "results_v2" / "vendor_clustered.json"
TASKS = ("coherence", "factuality", "preference", "relevance")
SESOI = 0.02

# provider is the vendor that trained the checkpoint, not the host that serves
# it: llama-4-scout on huggingface is still Meta's model.
VENDOR = {
    "claude": "anthropic", "deepseek": "deepseek", "gemini": "google",
    "gemma": "google", "glm": "zhipu", "kimi": "moonshot",
    "llama": "meta", "mistral": "mistral", "magistral": "mistral",
    "qwen": "alibaba", "gpt-oss": "openai",
}


def vendor(judge: str) -> str:
    fam = ((JUDGES.get(judge) or {}).get("family") or judge).lower()
    for key, v in VENDOR.items():
        if fam.startswith(key) or judge.lower().startswith(key):
            return v
    return "other:" + judge


def t_sf(t: float, df: int) -> float:
    """Upper-tail P(T > t) for Student's t, via the regularised incomplete beta."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)

    def betacf(a, b, x, itmax=200, eps=3e-12):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1e-30 if abs(d) < 1e-30 else d
            c = 1.0 + aa / c
            c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1e-30 if abs(d) < 1e-30 else d
            c = 1.0 + aa / c
            c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < eps:
                break
        return h

    a, b = df / 2.0, 0.5
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a * betacf(a, b, x)
    ib = min(max(ib, 0.0), 1.0)
    p_two = ib if x < (a + 1) / (a + b + 2) else 1.0 - (
        math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) / b
        * betacf(b, a, 1 - x))
    return max(min(p_two / 2.0, 1.0), 0.0) if t > 0 else 1.0 - max(min(p_two / 2.0, 1.0), 0.0)


def cluster_se(vals, groups):
    n = len(vals)
    mean = statistics.fmean(vals)
    by = defaultdict(float)
    for v, g in zip(vals, groups):
        by[g] += (v - mean)
    G = len(by)
    if G < 2:
        return mean, float("nan"), G
    ss = sum(e * e for e in by.values())
    V = (G / (G - 1.0)) * ss / (n * n)
    return mean, math.sqrt(V), G


def main() -> int:
    d = json.load(io.open(MULT, encoding="utf-8"))
    readable = d["readable_cells_bh"]

    out, p_null, p_sesoi = {}, {}, {}
    for task in TASKS:
        cs = [c for c in readable.values() if c["task"] == task]
        vals = [c["delta"] for c in cs]
        grps = [vendor(c["judge"]) for c in cs]
        mean, se, G = cluster_se(vals, grps)
        df = G - 1
        t0 = mean / se
        pn = 2 * t_sf(abs(t0), df)
        # one-sided H0: mean >= -SESOI, i.e. effect no worse than the threshold
        ts = (mean + SESOI) / se
        ps = t_sf(abs(ts), df) if ts < 0 else 1.0 - t_sf(abs(ts), df)
        naive_se = statistics.stdev(vals) / math.sqrt(len(vals))
        out[task] = {
            "n_judges": len(vals), "n_vendors": G, "mean": round(mean, 4),
            "se_cluster": round(se, 5), "se_naive": round(naive_se, 5),
            "inflation": round(se / naive_se, 2),
            "t": round(t0, 2), "df": df, "p_null": pn,
            "t_sesoi": round(ts, 2), "p_sesoi": ps,
            "vendors": sorted({g for g in grps}),
        }
        p_null[task], p_sesoi[task] = pn, ps

    hn, hs = holm(p_null, alpha=0.05), holm(p_sesoi, alpha=0.05)
    for t in TASKS:
        out[t]["holm_null"] = bool(hn[t]["reject"])
        out[t]["holm_sesoi"] = bool(hs[t]["reject"])

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1))

    print(f"  {'task':11}{'judges':>7}{'vendors':>8}{'mean':>9}{'SE_cl':>8}"
          f"{'x naive':>8}{'p null':>10}{'p SESOI':>10}  Holm(SESOI)")
    for t in TASKS:
        v = out[t]
        print(f"  {t:11}{v['n_judges']:>7}{v['n_vendors']:>8}{v['mean']:>9.4f}"
              f"{v['se_cluster']:>8.4f}{v['inflation']:>8.2f}"
              f"{v['p_null']:>10.2e}{v['p_sesoi']:>10.4f}"
              f"  {'reject' if v['holm_sesoi'] else 'RETAIN'}")
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
