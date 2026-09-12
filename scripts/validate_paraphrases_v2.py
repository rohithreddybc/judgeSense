"""Semantic equivalence check for the v2 prompt-paraphrase pairs.

WHY THIS EXISTS

`data/validation/*_paraphrase.jsonl` carries 500 records keyed by v1 ids
(`cohe_001`). No v2 item uses that scheme (`cohe_v2_0001`), so the join is
empty and the shipped dataset has no semantic validation attached to it. The
manual records under `archive_v1_manual/` are v1 too, and their own README says
they back no claim in the paper.

The deterministic template audit in the paper already enforces label space,
polarity, construct vocabulary and non-triviality on every pair. That is a
structural check. It cannot say whether two differently worded instructions ask
for the same judgement, which is the construct assumption the benchmark rests
on. This adds that check, keyed to v2 ids so it actually joins.

It is a MODEL-BASED check, not a human one, and the paper must say so. It is
run through the Claude Code harness rather than a paid API.

    python scripts/validate_paraphrases_v2.py prepare
    python scripts/validate_paraphrases_v2.py ingest
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V2 = REPO / "data" / "v2"
OUT = REPO / "data" / "validation" / "v2"
BATCH = 50

HEADER = """You are checking whether pairs of evaluation INSTRUCTIONS ask for the
same judgement.

Each item below shows two instructions, A and B. They are meant to differ only
in surface wording: the same decision, on the same content, in different words.

For each item answer YES or NO:
  YES  the two instructions ask for the same judgement, and a judge answering
       both correctly would give the same answer
  NO   the two instructions ask for materially different judgements, or one
       inverts the meaning of the answer, or one asks for something the other
       does not

Judge each item entirely on its own. Ignore the content being evaluated; you
are comparing the instructions only.

"""


def load_items():
    items = []
    seen = set()
    for path in sorted(V2.glob("*.jsonl")):
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # pairwise tasks ship each item twice, once per A/B ordering; the
            # instruction pair is identical across the two, so check it once.
            key = r.get("prompt_pair_id") or r["pair_id"]
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "id": r["pair_id"],
                "task": r["task_type"],
                "a": r["template_a"],
                "b": r["template_b"],
            })
    return items


def prepare() -> int:
    items = load_items()
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    (OUT / "answers").mkdir(parents=True, exist_ok=True)
    batches = []
    for n in range(0, len(items), BATCH):
        chunk = items[n:n + BATCH]
        name = f"val_{n // BATCH:04d}"
        body = [HEADER]
        for it in chunk:
            body.append(f"----- ITEM {it['id']} -----")
            body.append(f"task: {it['task']}")
            body.append(f"A: {it['a']}")
            body.append(f"B: {it['b']}")
            body.append("")
        p = OUT / "prompts" / f"{name}.txt"
        io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(body))
        batches.append({"name": name, "prompt_file": str(p),
                        "answer_file": str(OUT / "answers" / f"{name}.jsonl"),
                        "ids": [it["id"] for it in chunk]})
    io.open(OUT / "manifest.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"n_items": len(items), "batch_size": BATCH,
                    "batches": batches}, indent=1))
    print(f"  {len(items)} distinct instruction pairs -> {len(batches)} batches")
    print(f"  prompts: {OUT / 'prompts'}")
    return 0


def ingest() -> int:
    m = json.load(io.open(OUT / "manifest.json", encoding="utf-8"))
    rows, missing, bad = [], [], []
    for e in m["batches"]:
        p = Path(e["answer_file"])
        if not p.exists():
            missing.append(e["name"])
            continue
        got = {}
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                got[str(r["id"])] = str(r["answer"]).strip().upper()
            except (ValueError, KeyError):
                bad.append(e["name"])
        for i in e["ids"]:
            if i in got:
                rows.append({"pair_id": i, "equivalent": got[i],
                             "validator": "claude-code-harness",
                             "method": "model-based, not human"})
            else:
                missing.append(i)
    out = OUT / "equivalence_v2.jsonl"
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    yes = sum(1 for r in rows if r["equivalent"] == "YES")
    print(f"  {len(rows)} judged, {yes} equivalent ({100 * yes / max(len(rows), 1):.1f}%)")
    print(f"  missing: {len(missing)}   malformed batches: {sorted(set(bad))}")
    print(f"  wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "ingest"])
    return prepare() if ap.parse_args().mode == "prepare" else ingest()


if __name__ == "__main__":
    sys.exit(main())
