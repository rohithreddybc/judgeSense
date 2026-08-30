#!/usr/bin/env bash
# Launch the v2 multi-vendor sweep as detached processes, split so that no two
# processes ever share a (judge, task) and each talks to one provider.
#
# Splitting by PROVIDER avoids rate-limit contention. The larger providers are
# split further for throughput; run_cell takes a lock per cell, so an accidental
# overlap is refused rather than silently duplicating paid work.
#
# Every process is resumable: raw output is append-only and completed rows are
# skipped on restart, so an interruption costs time and nothing else.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/sweep
COMMON="--budget-policy matched --repeat-baseline --skip-preflight --yes"
EXTRA=("$@")

launch () {
  local tag="$1"; shift
  echo "  [$tag] $*"
  nohup python -m src.run_v2 --judges "$@" $COMMON "${EXTRA[@]}" \
        > "logs/sweep/${tag}.log" 2>&1 &
  echo "$!" > "logs/sweep/${tag}.pid"
}

echo "launching 23 judges across 5 providers:"
launch google     gemini-flash gemini-3.7-flash gemini-3.1-pro
launch hf1        llama3-8b llama-3.3-70b
launch hf2        llama-4-scout llama-4-maverick
launch hf3        gemma-4-31b qwen3-8b
launch hf4        qwen3-14b qwen3-32b
launch hf5        qwen3.8-27b-hf
launch mistral    mistral-small magistral-small mistral-medium
launch novita     qwen deepseek-v4-flash
launch ds1        qwen-3.6-flash qwen3.7-flash
launch ds2        deepseek-v4-flash-ds glm-5.2
launch ds3        kimi-k3 deepseek-v4-pro
echo
echo "monitor: bash scripts/sweep_status.sh"
