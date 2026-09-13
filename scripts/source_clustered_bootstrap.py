"""Re-cluster the pooled contrasts at the SOURCE record, not the item.

WHY

Intervals in the paper cluster at the item, which is correct for the arms and
repeats nested inside one item. But items are themselves nested in source
records, and two items from one record are not independent draws, so
item-clustered intervals are too narrow wherever that nesting is strong.

Resamples source records with replacement, recomputes the per-task pooled mean
over the judges, and reports the interval beside the item-clustered one.

    python scripts/source_clustered_bootstrap.py

CORRECTION, and how the old version was wrong

The first version of this script derived the cluster from the shape of
`item_id`: strip a trailing index when the field before it is also numeric.
That rule reads `cohe_summeval_77_5` and `relv_treccovid_32_0` correctly, and
it silently mis-reads the other two tasks:

    fact_tqa_0001                  -> no numeric field before the index,
                                      so the item was treated as its own record
    pref_mtbench_141_ee4474fa1d9f  -> trailing field is a hash,
                                      so the item was treated as its own record

Both are nested, and the dataset says so in a field the script never read. Every
row carries `source.source_record_id`, and counting distinct values of it gives
125 TruthfulQA records behind 250 factuality items (each question supplies one
accurate and one inaccurate statement) and 68 MT-Bench questions behind 130
preference items. The old rule reported 250 and 130 clusters for those tasks,
which is the item count: for half the study the "source-clustered" interval was
the item-clustered interval under another name.

This version reads `source.source_record_id` and reduces it to the unit the
nesting actually occurs at:

    factuality   validation[548]                         the TruthfulQA record
    coherence    test[77].machine_summaries[5]           the SummEval document
    relevance    query[32]#pair0                         the TREC-COVID topic
    preference   question_id=141;model_a=...;turn=1      the MT-Bench question

Deriving the key from provenance rather than from a naming convention also means
a future rebuild that renames items cannot silently change the clustering.
"""
from __future__ import annotations

import collections
import io
import json
import random
import re
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

# source_record_id -> the record the nesting happens at. Anchored, so a string
# that does not match the expected shape raises instead of silently falling
# through to the item, which is the failure this replaces.
RECORD_RE = {
    "coherence": re.compile(r"^(test\[\d+\])\."),
    "relevance": re.compile(r"^(query\[\d+\])#"),
    "preference": re.compile(r"^(question_id=\d+);"),
    "factuality": re.compile(r"^(validation\[\d+\])$"),
}


def source_of() -> dict:
    """pair_id -> the source RECORD it came from, read from its provenance."""
    m = {}
    for p in sorted(V2.glob("*.jsonl")):
        task = p.stem
        pat = RECORD_RE[task]
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r["source"]["source_record_id"]
            hit = pat.match(rid)
            if not hit:
                raise SystemExit(
                    f"{task}: source_record_id {rid!r} does not match the "
                    f"expected record shape. Clustering would silently fall "
                    f"back to the item, which is the bug this guard replaces."
                )
            m[r["pair_id"]] = f"{task}:{hit.group(1)}"
    return m


def agreement(row, arm_a, arm_b):
    a, b = row.get(arm_a), row.get(arm_b)
    if a is None or b is None:
        return None
    return int(a == b)


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
    raise SystemExit(main())
