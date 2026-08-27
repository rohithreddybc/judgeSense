#!/usr/bin/env bash
# Launch the v2 multi-vendor sweep as one detached process per PROVIDER.
#
# run_v2 is sequential within a process and warns that two processes over the
# SAME (judge, task) would each work the whole backlog and both pay for it.
# Splitting by provider keeps the sets disjoint AND avoids rate-limit
# contention, since each process talks to a different vendor.
#
# Every process is resumable: raw output is append-only and completed rows are
# skipped on restart, so an interrupted run costs nothing but time.
#
# Usage:  bash scripts/run_sweep_parallel.sh [extra run_v2 args...]
#   e.g.  bash scripts/run_sweep_parallel.sh --limit 5      (smoke)
#         bash scripts/run_sweep_parallel.sh                (full)

set -u
cd "$(dirname "$0")/.."
LOG_DIR="logs/sweep"
mkdir -p "$LOG_DIR"

COMMON="--budget-policy matched --repeat-baseline --skip-preflight --yes"

launch () {                       # launch <provider-tag> <judges...>
  local tag="$1"; shift
  local log="$LOG_DIR/${tag}.log"
  echo "  [$tag] $* -> $log"
  nohup python -m src.run_v2 --judges "$@" $COMMON "${EXTRA[@]}" \
        > "$log" 2>&1 &
  echo "$!" > "$LOG_DIR/${tag}.pid"
}

EXTRA=("$@")

echo "launching one process per provider:"
launch google      gemini-flash gemini-3.7-flash
launch huggingface llama3-8b llama-3.3-70b llama-4-scout llama-4-maverick \
                   gemma-4-31b qwen3-8b qwen3-14b qwen3-32b
launch mistral     mistral-small magistral-small
launch novita      qwen deepseek-v4-flash
launch dashscope   qwen-3.6-flash qwen3.7-flash deepseek-v4-flash-ds glm-5.2
launch groq        gpt-oss-20b gpt-oss-120b qwen3.8-27b

echo
echo "launched. monitor with:"
echo "  tail -f $LOG_DIR/*.log"
echo "  bash scripts/sweep_status.sh"
