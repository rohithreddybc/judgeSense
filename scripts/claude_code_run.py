"""Prepare and ingest Claude Code judging batches.

The Anthropic API key has no credit, so Claude judges are reachable only through
Claude Code subagents. A subagent cannot be spawned from inside a Python
process, so the loop is split in three and the middle step is driven by the
assistant:

    prepare   this script writes one prompt file per batch, plus a manifest
    dispatch  the assistant spawns a subagent per batch; each READS its prompt
              file and WRITES its answers, so results never pass through
              anyone's context
    ingest    this script validates every batch and emits raw records in the
              runner's own schema

Why the split matters: ingest refuses to write a partial or misaligned batch.
Judging is the expensive step, and a silently dropped item would shift every
later label in that batch by one.

BUDGET POLICY. These records are written under `claude_code_batched`, NOT
`matched`. The harness exposes no max_tokens and no temperature, so calling them
matched would assert a control that does not exist, and regenerate_results
filters on this field -- mislabelling them would pool an uncontrolled judge with
the matched-budget API judges and nothing downstream would notice.

    python scripts/claude_code_run.py prepare --judge cc-opus-5 --task factuality
    python scripts/claude_code_run.py ingest  --judge cc-opus-5 --task factuality
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from claude_code_judge import (  # noqa: E402
    DEFAULT_BATCH_SIZE, BatchResponseError, batch_digest, build_batch_prompt,
    make_batches, mirror_repeat_batches, parse_batch_response, provenance,
    units_for_task,
)
from structural_variants import parse_variant_output, UNCLEAR  # noqa: E402

DATA = REPO / "data" / "v2"
WORK = REPO / "data" / "claude_code"
RAW = REPO / "data" / "results_v2" / "raw"
STAGING = REPO / "data" / "claude_code" / "_staged"
BUDGET_POLICY = "claude_code_batched"
TASKS = ("factuality", "coherence", "relevance", "preference")


def _rows(task: str) -> List[dict]:
    return [json.loads(l) for l in (DATA / f"{task}.jsonl").open(encoding="utf-8")
            if l.strip()]


def _dirs(judge: str, task: str):
    base = WORK / judge / task
    return base / "prompts", base / "answers", base / "manifest.json"


def prepare(judge: str, task: str, batch_size: int, seed: int) -> int:
    rows = _rows(task)
    arm_batches = make_batches(units_for_task(rows, ("a", "b")), batch_size, seed)
    rep_batches = mirror_repeat_batches(arm_batches)
    all_batches = [("arm", b) for b in arm_batches] + [("rep", b) for b in rep_batches]

    prompts, answers, manifest_path = _dirs(judge, task)
    prompts.mkdir(parents=True, exist_ok=True)
    answers.mkdir(parents=True, exist_ok=True)

    manifest = {"judge": judge, "task": task, "batch_size": batch_size,
                "seed": seed, "budget_policy": BUDGET_POLICY,
                "provenance": provenance(judge, batch_size, seed),
                "n_rows": len(rows), "batches": []}
    for i, (kind, batch) in enumerate(all_batches):
        name = f"{kind}_{i:04d}"
        (prompts / f"{name}.txt").write_text(build_batch_prompt(batch),
                                             encoding="utf-8")
        manifest["batches"].append({
            "name": name, "kind": kind, "digest": batch_digest(batch),
            "ids": [u["id"] for u in batch],
            "prompt_file": str((prompts / f"{name}.txt").resolve()),
            "answer_file": str((answers / f"{name}.jsonl").resolve()),
        })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_units = sum(len(b) for _, b in all_batches)
    print(f"  {judge}/{task}: {len(rows)} rows -> {n_units} units -> "
          f"{len(all_batches)} batches of <= {batch_size}")
    print(f"  prompts: {prompts}")
    print(f"  manifest: {manifest_path}")
    return 0


def ingest(judge: str, task: str, allow_partial: bool, publish: bool = False) -> int:
    _, answers, manifest_path = _dirs(judge, task)
    if not manifest_path.exists():
        raise SystemExit(f"no manifest; run prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    verdicts: Dict[str, str] = {}
    missing_batches, bad = [], []
    for entry in manifest["batches"]:
        path = Path(entry["answer_file"])
        if not path.exists():
            missing_batches.append(entry["name"])
            continue
        try:
            got = parse_batch_response(path.read_text(encoding="utf-8"), entry["ids"])
        except BatchResponseError as exc:
            bad.append(f"{entry['name']}: {exc}")
            continue
        verdicts.update(got)

    if missing_batches or bad:
        print(f"  {len(missing_batches)} batch(es) not answered yet; "
              f"{len(bad)} misaligned")
        for b in bad[:5]:
            print(f"    {b}")
        if not allow_partial:
            print("\n  refusing to write. Judging is the expensive step and a\n"
                  "  misaligned batch would shift every later label in it by one.\n"
                  "  Re-dispatch the listed batches, or pass --allow-partial to\n"
                  "  write only the cells that are complete.")
            return 2

    rows = _rows(task)
    # Ingest writes to a STAGING directory, never straight into the results the
    # paper reads. This script was one careless invocation away from repeating
    # the defect that put "goodjudge & factuality & 10 & 1.000" into Table 1: a
    # dry run wrote a synthetic cell directly into data/results_v2/raw/, where
    # nothing distinguishes it from a paid run. Promotion is a separate,
    # deliberate act (--publish) that refuses to overwrite an existing cell.
    out_path = (RAW if publish else STAGING) / f"{judge}_{task}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if publish and out_path.exists():
        raise SystemExit(
            f"refusing to overwrite {out_path}. A published cell is the record "
            f"of a run that was paid for. Move or delete it deliberately if you "
            f"really mean to replace it.")
    written = skipped = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            pid = str(row["pair_id"])
            d = {arm: verdicts.get(f"{pid}#{arm}")
                 for arm in ("a", "b", "a_repeat", "b_repeat")}
            if d["a"] is None or d["b"] is None:
                skipped += 1
                continue
            rec = {
                "pair_id": pid,
                "item_id": row.get("item_id", pid),
                "prompt_pair_id": row.get("prompt_pair_id", pid),
                "task_type": task,
                "model": judge,
                "ab_order": row.get("ab_order"),
                "ground_truth_label": row.get("ground_truth_label"),
                "ground_truth_position": row.get("ground_truth_position"),
                "budget_policy": BUDGET_POLICY,
                "max_tokens": None,
                "error": None,
                "ts": datetime.now(timezone.utc).isoformat(),
                "transport": "claude_code_subagent",
                "decoding": manifest["provenance"],
            }
            for arm in ("a", "b", "a_repeat", "b_repeat"):
                raw = d[arm]
                if raw is None:
                    continue
                rec[f"prompt_{arm}_raw"] = raw
                rec[f"decision_{arm}"] = parse_variant_output(task, raw, "plain")
            written += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    unclear = sum(1 for r in (json.loads(l) for l in
                              out_path.open(encoding="utf-8") if l.strip())
                  for k in ("decision_a", "decision_b") if r.get(k) == UNCLEAR)
    where = "results" if publish else "STAGING"
    print(f"  wrote {written} rows to {where}: {out_path} "
          f"({skipped} skipped for missing arms, {unclear} malformed arm answers)")
    if not publish:
        print("  staged only. Review, then re-run with --publish to place it "
              "in data/results_v2/raw/ where the metrics pipeline reads it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "ingest"):
        p = sub.add_parser(name)
        p.add_argument("--judge", required=True)
        p.add_argument("--task", required=True, choices=TASKS)
        if name == "prepare":
            p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
            p.add_argument("--seed", type=int, default=42)
        else:
            p.add_argument("--allow-partial", action="store_true")
            p.add_argument("--publish", action="store_true",
                           help="write into data/results_v2/raw/ instead of staging")
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.judge, a.task, a.batch_size, a.seed)
    return ingest(a.judge, a.task, a.allow_partial, a.publish)


if __name__ == "__main__":
    raise SystemExit(main())
