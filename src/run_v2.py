"""
JudgeSense v2 judge runner — crash-safe, resumable, pre-flighted.

This is the "run once" driver. It reads the v2 dataset (data/v2/*.jsonl),
queries each selected judge on both paraphrase arms of every row (and, with
--repeat-baseline, a second call on arm A to establish the decoding-noise
ceiling), and writes raw outputs incrementally so an interrupted run resumes
where it stopped instead of re-spending.

Design guarantees, in order of why they matter for a paid run:

1. RESUMABLE. Output is appended one record per completed row, keyed by
   (pair_id). On restart, rows already present with no error are skipped. A
   crash therefore wastes at most the row in flight (2-3 calls), never the run.
2. PRE-FLIGHT. `--preflight` (implied before any full run unless --skip-preflight)
   checks every selected judge's API key is present and makes ONE real call per
   judge. A bad model id, missing key, or auth failure is surfaced before the
   expensive loop, not 2,904 calls in.
3. PLAN BEFORE SPEND. The call count and per-judge token budget are printed and
   must be confirmed with --yes before a full run starts.
4. NO SILENT LABELS. Parsing is the strict v2 parser; anything unparseable is
   recorded as UNCLEAR with the raw text retained, never coerced to a label.

It reuses the provider clients and single-call-with-retry from src/evaluate.py
(the v1 runner) rather than reimplementing SDK plumbing; it drives model
selection and token budgets from src/judge_registry.py (the v2 source of truth).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

try:
    from .evaluate import _build_client, _append_jsonl, _load_jsonl, _load_env
    from .usage_meter import metered_call as _call, clear_last_meta, take_last_meta
    from .judge_registry import JUDGES, max_tokens_for, select_judges, main_axis_run_plan
    from .structural_variants import parse_variant_output, UNCLEAR
except ImportError:  # `python src/run_v2.py`
    from evaluate import _build_client, _append_jsonl, _load_jsonl, _load_env  # type: ignore
    from usage_meter import metered_call as _call, clear_last_meta, take_last_meta  # type: ignore
    from judge_registry import JUDGES, max_tokens_for, select_judges, main_axis_run_plan  # type: ignore
    from structural_variants import parse_variant_output, UNCLEAR  # type: ignore

# A Windows console defaults to cp1252, which cannot encode the box-drawing and
# arrow characters this script prints. Left alone, the very first status line
# raises UnicodeEncodeError and kills the process -- during a paid run that
# would abort at an arbitrary point mid-sweep, after the calls were billed.
# Printing must never be able to fail, so the streams are forced to UTF-8 with
# replacement rather than the console's ANSI codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # already detached, or a pipe that refuses
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_DATA_DIR = _REPO / "data" / "v2"
_OUT_DIR = _REPO / "data" / "results_v2" / "raw"
TASKS = ("factuality", "coherence", "relevance", "preference")

# The runner queries the model name exactly as registered in judge_registry.
# evaluate._build_client resolves it against SUPPORTED_MODELS; the two registries
# share model names, and this asserts that invariant loudly at startup rather
# than mid-run.
def _resolve_client(judge: str):
    return _build_client(judge)  # raises RuntimeError on missing key / unknown name


def _out_path(judge: str, task: str) -> Path:
    return _OUT_DIR / f"{judge}_{task}.jsonl"


def _completed_pair_ids(path: Path, budget_policy: Optional[str] = None) -> Set[str]:
    """pair_ids whose LAST written record succeeded UNDER THIS POLICY.

    Last-write-wins, not any-write-wins: the file is append-only, so a row that
    errored and was retried has both records present. Judging completion by the
    final record means a successful retry marks the row done (no re-spend) while
    a row whose latest attempt failed is retried. `regenerate_results` applies
    the same rule when reading, so the paid calls and the scored records always
    agree on which attempt counts.
    """
    if not path.exists():
        return set()
    last: Dict[str, Optional[str]] = {}
    for rec in _load_jsonl(path):
        pid = rec.get("pair_id")
        if pid is None:
            continue
        # A row already run under a DIFFERENT decoding budget is not done: it is
        # a different measurement. Keying completion on pair_id alone made a
        # cheap --limit smoke test at one policy permanently shadow the real
        # rows at another, because the runner would never re-issue them and the
        # reader had no idea the cell was a mixture.
        if budget_policy is not None and rec.get("budget_policy") != budget_policy:
            continue
        last[str(pid)] = rec.get("error")
    return {pid for pid, err in last.items() if err is None}


def _decide(provider, client, model_id, prompt: str, task: str,
            max_tokens: int) -> Tuple[str, str, Optional[str], Optional[dict]]:
    """One judge call → (raw, normalized_decision, error, usage_meta).

    error is a string on a hard failure, else None; UNCLEAR is a decision, not
    an error. usage_meta carries this call's token counts and timing, or None
    when the call seam is stubbed (as in the offline tests) or the provider
    returned no usage.

    The slot is cleared BEFORE the call so a stub that writes nothing yields
    None rather than silently re-reporting the previous call's tokens.
    """
    clear_last_meta()
    raw = _call(provider, client, model_id, prompt, max_tokens)
    meta = take_last_meta()
    if isinstance(raw, str) and raw.startswith("ERROR:"):
        return raw, UNCLEAR, raw[len("ERROR:"):], meta
    decision = parse_variant_output(task, raw, "plain")
    return raw, decision, None, meta


def preflight(judges: List[str]) -> bool:
    """One real call per judge on a trivial prompt. Returns True iff all pass.
    This is the cheap insurance that the expensive run will not die on call 1."""
    print("\n── Pre-flight: one live call per judge ──")
    probe = "Is this statement factually correct? Answer YES or NO only.\n\nThe sky is blue."
    ok = True
    for judge in judges:
        try:
            client, model_id, provider = _resolve_client(judge)
        except Exception as exc:
            print(f"  [FAIL] {judge:<20} client/key: {exc}")
            ok = False
            continue
        raw, decision, err, _use = _decide(provider, client, model_id, probe, "factuality",
                                           max_tokens_for(judge, "native"))
        if err:
            print(f"  [FAIL] {judge:<20} call errored: {err[:80]}")
            ok = False
        else:
            print(f"  [ ok ] {judge:<20} -> {decision!r} (raw {raw[:40]!r})")
    print("──" if ok else "── PRE-FLIGHT FAILED; fix the above before a full run ──")
    return ok


def run_cell(judge: str, task: str, budget_policy: str, repeat_baseline: bool,
             limit: Optional[int]) -> Dict[str, int]:
    """Run one (judge, task). Resumable; writes one record per row."""
    rows = _load_jsonl(_DATA_DIR / f"{task}.jsonl")
    if limit:
        rows = rows[:limit]
    out = _out_path(judge, task)
    done = _completed_pair_ids(out, budget_policy)
    client, model_id, provider = _resolve_client(judge)
    max_tokens = max_tokens_for(judge, budget_policy)

    n_done = n_new = n_err = 0
    for row in rows:
        pid = str(row["pair_id"])
        if pid in done:
            n_done += 1
            continue
        raw_a, dec_a, err_a, use_a = _decide(provider, client, model_id, row["prompt_a"], task, max_tokens)
        raw_b, dec_b, err_b, use_b = _decide(provider, client, model_id, row["prompt_b"], task, max_tokens)
        rec = {
            "pair_id": pid,
            "item_id": row.get("item_id"),
            "prompt_pair_id": row.get("prompt_pair_id"),
            "task_type": task,
            "model": judge,
            "ab_order": row.get("ab_order"),
            "ground_truth_label": row.get("ground_truth_label"),
            "ground_truth_position": row.get("ground_truth_position"),
            "prompt_a_raw": raw_a, "decision_a": dec_a,
            "prompt_b_raw": raw_b, "decision_b": dec_b,
            "budget_policy": budget_policy,
            "max_tokens": max_tokens,
            "error": err_a or err_b,
            # Usage is inline on the record rather than in a side log: one append
            # keeps the crash-consistency story unchanged, and usage stays joined
            # to its decision without a second file to desynchronise.
            "ts": _now_iso(),
            "usage_a": use_a,
            "usage_b": use_b,
        }
        # The repeat baseline is ONE extra call per ITEM, not per row. Pairwise
        # tasks emit two rows per item (original + swapped orderings), so firing
        # it on every row would issue 1,452 repeat calls per judge against the
        # 1,000 the run plan budgets — a 12.5% overspend on the total sweep.
        # The noise ceiling only needs one prompt repeated, so it is taken on the
        # canonical ordering.
        # BOTH arms are repeated, not just arm A. The ceiling has to be
        # measured on the same template whose disagreement it is meant to
        # explain: if template B is intrinsically higher-entropy -- longer, more
        # likely to draw a preamble the strict parser rejects -- then noise
        # under B is charged to paraphrasing, because a ceiling measured only
        # under A cannot absorb it. With one arm repeated, the endpoint was
        # partly a property of which template happened to be designated A.
        if repeat_baseline and row.get("ab_order") in (None, "original"):
            raw_r, dec_r, err_r, use_r = _decide(provider, client, model_id, row["prompt_a"], task, max_tokens)
            rec["prompt_a_repeat_raw"] = raw_r
            rec["decision_a_repeat"] = dec_r
            rec["usage_a_repeat"] = use_r
            raw_rb, dec_rb, err_rb, use_rb = _decide(provider, client, model_id, row["prompt_b"], task, max_tokens)
            rec["prompt_b_repeat_raw"] = raw_rb
            rec["decision_b_repeat"] = dec_rb
            rec["usage_b_repeat"] = use_rb
            rec["error"] = rec["error"] or err_r or err_rb
        _append_jsonl(rec, out)
        n_new += 1
        if rec["error"]:
            n_err += 1
    return {"resumed": n_done, "new": n_new, "errors": n_err, "total": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JudgeSense v2 judge runner (resumable, pre-flighted).")
    p.add_argument("--judges", nargs="*", default=None,
                   help="judge names (default: all verified judges in the registry).")
    p.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    # Matched is the DEFAULT, not an option to remember. Two reviewers (xmQT W1,
    # qkzU Q3) named the class-asymmetric budget -- 20 tokens for
    # instruction-tuned judges against 1024 for reasoning-tuned ones -- as
    # confounding architecture with inference configuration. Under the native
    # policy 27 of 120 measured relevance calls terminated at the 20-token cap
    # mid-sentence, so part of that cell's disagreement was truncation rather
    # than sensitivity. The sweep runs once; defaulting to the configuration a
    # reviewer already rejected would waste it. Native remains available for the
    # budget ablation, where it is the object of study rather than an accident.
    p.add_argument("--budget-policy", default="matched", choices=("native", "matched"),
                   help="matched (default): every judge gets the same max_tokens, so "
                        "judge-class differences cannot be explained by the budget. "
                        "native: reproduces the v1 asymmetry, for the ablation only.")
    p.add_argument("--repeat-baseline", action="store_true",
                   help="issue arm A twice to measure the decoding-noise ceiling.")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N rows per (judge,task) — for a cheap smoke test.")
    p.add_argument("--preflight-only", action="store_true", help="run pre-flight and exit.")
    p.add_argument("--skip-preflight", action="store_true", help="skip pre-flight (not advised).")
    p.add_argument("--yes", action="store_true", help="proceed with the full run without prompting.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    judges = select_judges(args.judges)  # rejects unknown/unverified judges loudly

    plan = main_axis_run_plan(judges=judges, include_repeat_baseline=args.repeat_baseline)
    print(f"Judges ({len(judges)}): {', '.join(judges)}")
    print(f"Tasks: {', '.join(args.tasks)} | budget policy: {args.budget_policy}")
    print(f"Planned calls (all tasks): {plan['total_calls']}"
          + (f" + {plan['total_calls_with_repeat'] - plan['total_calls']} repeat"
             if args.repeat_baseline else ""))
    if args.limit:
        print(f"LIMIT={args.limit} rows/cell (smoke test)")

    if not args.skip_preflight:
        if not preflight(judges):
            return 2
        if args.preflight_only:
            return 0
    elif args.preflight_only:
        print("nothing to do: --preflight-only with --skip-preflight")
        return 0

    print(
        "\nNOTE: do not run two processes over the SAME (judge, task) at once.\n"
        "  Completed rows are read once at cell start and the output file is\n"
        "  append-only with no lock, so two processes would each work the whole\n"
        "  remaining backlog and both pay for it. To parallelise, split by\n"
        "  --judges or --tasks into disjoint sets, one process per set."
    )

    if not args.yes:
        print("\nRe-run with --yes to start the full run (this spends API credit).")
        return 0

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for judge in judges:
        for task in args.tasks:
            t0 = time.time()
            stats = run_cell(judge, task, args.budget_policy, args.repeat_baseline, args.limit)
            print(f"[{judge}/{task}] new={stats['new']} resumed={stats['resumed']} "
                  f"errors={stats['errors']} / {stats['total']} rows "
                  f"({time.time()-t0:.0f}s)")
    print("\nDone. Raw outputs in", _OUT_DIR)
    print("Next: regenerate metrics/tables from these with scripts/regenerate_results.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
