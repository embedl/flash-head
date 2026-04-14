#!/bin/bash
# Copyright (C) 2026 Embedl AB
# Run FlashHead speedup tests on a remote GPU device via SSH.
#
# Syncs the repo, runs baseline (FLASHHEAD_ENABLED=0) and FlashHead
# benchmarks in separate processes (for clean GPU memory between runs),
# then compares results with statistical tests.
#
# The plugin is always installed so vLLM recognizes FlashHead architectures.
# FLASHHEAD_ENABLED=0 disables the patches, giving a clean dense-head baseline.
#
# BENCHMARK_MODE controls what runs:
#   python  - Python script benchmark + speedup test (default)
#
# Usage — only the remote hostname is required, everything else has defaults:
#
#   ./tests/test_on_remote.sh picea.sshfs
#
# Override any variable as needed:
#
#   MODEL=embedl/gemma-3-1B-it-FlashHead \
#   HF_TOKEN=hf_xxx \
#   BENCHMARK_MODE=python \
#   ./tests/test_on_remote.sh picea.sshfs
set -euo pipefail

# ---- Configuration (all have defaults) ----
REMOTE_HOST="${1:?Usage: $0 <remote-host>}"
MODEL="${MODEL:-embedl/gemma-3-1B-it-FlashHead}"
HF_TOKEN="${HF_TOKEN:-}"
BENCHMARK_MODE="${BENCHMARK_MODE:-python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Sync repo ----
echo "[host] Syncing ${REPO_DIR} -> ${REMOTE_HOST}:/home/jonna/tmp/flash-head/"
rsync -az --delete \
    --exclude='.git' --exclude='.idea' --exclude='*.egg-info' \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='build' --exclude='dist' \
    --exclude='*.log' \
    "$REPO_DIR/" "${REMOTE_HOST}:/home/jonna/tmp/flash-head/"

echo "[host] Running tests on ${REMOTE_HOST} (mode=${BENCHMARK_MODE})"
echo "  MODEL=${MODEL}"

ssh "${REMOTE_HOST}" \
    MODEL="${MODEL}" \
    HF_TOKEN="${HF_TOKEN}" \
    BENCHMARK_MODE="${BENCHMARK_MODE}" \
    bash -s <<'REMOTE'
set -euo pipefail

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BENCH_DIR="$HOME/tmp/flashhead_bench"
VENV_DIR="$HOME/tmp/flash-head-venv"
mkdir -p "$BENCH_DIR"

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating venv ==="
    uv venv "$VENV_DIR" --python 3.12 --seed
    source "$VENV_DIR/bin/activate"
    uv pip install vllm==0.13 --torch-backend=auto
else
    source "$VENV_DIR/bin/activate"
fi

pip install "$HOME/tmp/flash-head" 2>&1 | tail -1

if [ "$BENCHMARK_MODE" = "python" ]; then

    echo "=== [Python] Baseline benchmark (FlashHead disabled) ==="
    export FLASHHEAD_ENABLED=0

    python3 "$HOME/tmp/flash-head/tests/run_benchmark.py" \
        --model "$MODEL" \
        --max-tokens 128 --num-warmup 10 --num-runs 10 \
        --gpu-mem 0.7 --max-model-len 4096 \
        --output "$BENCH_DIR/python_baseline.json" \
        --label python-baseline

    echo "=== [Python] FlashHead benchmark ==="
    export FLASHHEAD_ENABLED=1

    python3 "$HOME/tmp/flash-head/tests/run_benchmark.py" \
        --model "$MODEL" \
        --max-tokens 128 --num-warmup 10 --num-runs 10 \
        --gpu-mem 0.7 --max-model-len 4096 \
        --output "$BENCH_DIR/python_flashhead.json" \
        --label python-flashhead

    echo "=== Running comparison tests ==="
    pip install pytest 2>&1 | tail -1
    FLASHHEAD_BENCH_DIR="$BENCH_DIR" \
    python3 -m pytest "$HOME/tmp/flash-head/tests/test_compare.py" -v -s
fi

REMOTE

echo "[host] Done (mode=${BENCHMARK_MODE})."
