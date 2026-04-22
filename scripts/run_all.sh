#!/bin/bash

# JudgeSense Full Experiment Runner
# Runs 8 models (gpt-4o-mini already done) across all 4 task types with 3 runs each
# Skips gpt-4o-mini — results already exist in data/results/raw_outputs/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================================"
echo "JudgeSense Full Experiment"
echo "========================================================================"
echo "Models: 8 (gpt-4o, claude-haiku, claude-sonnet,"
echo "           llama3-8b, llama3-70b, mistral-7b, qwen, deepseek)"
echo "Tasks: 4 (factuality, coherence, relevance, preference)"
echo "Runs per model: 3"
echo "Note: gpt-4o-mini skipped (already completed)"
echo "========================================================================"
echo ""

MODELS=("gpt-4o" "claude-haiku" "claude-sonnet" "llama3-8b" "llama3-70b" "mistral-7b" "qwen" "deepseek")

TOTAL_MODELS=${#MODELS[@]}
COMPLETED=0

for model in "${MODELS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    echo ""
    echo "[$COMPLETED/$TOTAL_MODELS] Starting $model at $START_TIME"
    echo "Command: python src/evaluate.py --model $model --task all --runs 3"
    echo "========================================================================"

    python src/evaluate.py --model "$model" --task all --runs 3

    END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    if [ $? -eq 0 ]; then
        echo "[OK] $model completed successfully at $END_TIME"
    else
        echo "[!!] $model encountered errors (gracefully handled) at $END_TIME"
    fi
done

echo ""
echo "========================================================================"
echo "Experiment Complete! Computing metrics summary..."
echo "========================================================================"
python src/metrics.py --summarize

echo ""
echo "Results saved to: data/results/raw_outputs/"
echo "Metrics saved to: data/results/metrics_summary.json"
echo ""
