"""Re-cluster the pooled contrasts at the SOURCE record, not the item.

WHY

Intervals in the paper cluster at the item, which is correct for the arms and
repeats nested inside one item. But items are themselves nested in source
records: the 500 relevance rows come from 250 TREC-COVID query/passage items
drawn from 50 topics, and the coherence items come from 92 SummEval documents.
Two items from one document are not independent draws, so item-clustered
intervals are too narrow wherever that nesting is strong.

Reviewer p5cJ raised the unit-of-analysis question and the paper answers it at
the item level. This answers the harder version. It needs no API calls: every
row is already committed.

Resamples source records with replacement, recomputes the per-task pooled mean
over the judges, and reports the interval beside the item-clustered one.

    python scripts/source_clustered_bootstrap.py
"""
from __future__ import annotations

import collections
import io
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SNAP = REPO / "data" / "results_v2" / "_snapshot"
V2 = REPO / "data" / "v2"
MULT = REPO / "data" / "results_v2" / "multiplicity.json"
OUT = REPO / "data" / "results_v2" / "source_clustered.json"

TASKS = ("coherence", "factuality", "preference", "relevance")
N_BOOT = 2000
SEED = 42


def source_of() -> dict:
    """pair_id -> the source RECORD it came from, not the item.

    The record is encoded in item_id, whose last underscore-separated field is
    the index of the item within it:

        cohe_summeval_77_5   -> SummEval document 77, summary 5
        relv_treccovid_32_0  -> TREC-COVID topic 32, passage 0

    but only two of the four tasks are nested that way. TruthfulQA items are
    standalone questions (`fact_tqa_0001`) and MT-Bench items carry a hash
    (`pref_mtbench_141_ee4474fa1d9f`), so for those the item IS the record.

    The rule that separates the two cases: strip the trailing index only when
    the field before it is also numeric, which is what a record-plus-index
    scheme looks like. Stripping unconditionally collapsed all 250 factuality
    items onto the single cluster `fact_tqa` and produced a zero-width
    interval, which is how the bug announced itself.

    `source` holds a provenance dict and `source_benchmark` its dataset name,
    so neither identifies the record.
    """
    m = {}
    for p in sorted(V2.glob("*.jsonl")):
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            item = str(r["item_id"])
            parts = item.split("_")
            nested = (len(parts) >= 3 and parts[-1].isdigit()
                      and parts[-2].isdigit())
            m[r["pair_id"]] = "_".join(parts[:-1]) if nested else item
    return m


def agreement(row, arm_a, arm_b):
    a, b = row.get(arm_a), row.get(arm_b)
    if a is None or b is None:
        return None
    return 1.0 if a == b else 0.0


def load_cells(src_map):
    """(judge, task) -> source -> list of (paraphrase_agree, repeat_agree)."""
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(SNAP.glob("*_*.jsonl")):
        judge, task = p.stem.rsplit("_", 1)
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            para = agreement(r, "decision_a", "decision_b")
            rep = agreement(r, "decision_a", "decision_a_repeat")
            if para is None or rep is None:
                continue
            src = src_map.get(r["pair_id"])
            if src is None:
                continue
            cells[(judge, task)][src].append((para, rep))
    return cells


def main() -> int:
    mult = json.load(io.open(MULT, encoding="utf-8"))
    readable = {(c["judge"], c["task"]) for c in mult["readable_cells_bh"].values()}
    cells = load_cells(source_of())
    rng = random.Random(SEED)
    out = {}

    for task in TASKS:
        keys = [k for k in cells if k[1] == task and k in readable]
        if not keys:
            continue
        sources = sorted({s for k in keys for s in cells[k]})
        draws = []
        for _ in range(N_BOOT):
            pick = [sources[rng.randrange(len(sources))] for _ in sources]
            per_judge = []
            for k in keys:
                num_p = num_r = n = 0
                for s in pick:
                    for para, rep in cells[k].get(s, ()):
                        num_p += para
                        num_r += rep
                        n += 1
                if n:
                    per_judge.append(num_p / n - num_r / n)
            if per_judge:
                draws.append(sum(per_judge) / len(per_judge))
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[int(0.975 * len(draws)) - 1]
        item_ci = mult["pooled_by_task_holm"][task]
        out[task] = {
            "n_judges": len(keys), "n_sources": len(sources),
            "mean_delta": round(sum(draws) / len(draws), 4),
            "source_ci95": [round(lo, 4), round(hi, 4)],
            "item_clustered_se": item_ci["se"],
            "item_clustered_mean": item_ci["mean_delta"],
            "excludes_zero": hi < 0,
            "excludes_sesoi": hi < -mult["sesoi"],
        }

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1))

    print(f"  source-clustered bootstrap, {N_BOOT} resamples, seed {SEED}")
    print(f"  {'task':12}{'judges':>7}{'sources':>8}{'mean':>9}{'95% CI':>22}"
          f"{'<0':>5}{'<SESOI':>8}")
    for t in TASKS:
        r = out.get(t)
        if not r:
            continue
        ci = f"[{r['source_ci95'][0]:+.3f}, {r['source_ci95'][1]:+.3f}]"
        print(f"  {t:12}{r['n_judges']:>7}{r['n_sources']:>8}{r['mean_delta']:>9.4f}"
              f"{ci:>22}{'yes' if r['excludes_zero'] else 'NO':>5}"
              f"{'yes' if r['excludes_sesoi'] else 'NO':>8}")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
