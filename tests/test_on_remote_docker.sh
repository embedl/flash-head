#!/bin/bash
# Copyright (C) 2026 Embedl AB
# Run FlashHead speedup tests on a remote GPU device via SSH + Docker.
#
# Syncs the repo, runs baseline (FLASHHEAD_ENABLED=0) and FlashHead
# benchmarks in separate Docker containers (for clean GPU memory
# between runs), then compares results with statistical tests.
#
# The plugin is always installed so vLLM recognizes FlashHead architectures.
# FLASHHEAD_ENABLED=0 disables the patches, giving a clean dense-head baseline.
#
# BENCHMARK_MODE controls what runs:
#   python  - Python script benchmark + speedup test
#   cli     - vllm bench throughput only
#   both    - both, plus cross-validation that they agree (default)
#
# Usage — only the remote hostname is required, everything else has defaults:
#
#   ./tests/test_on_remote.sh agx-thor-san-jose
#
# Override any variable as needed:
#
#   MODEL=embedl/Cosmos-Reason2-2B-W4A16-Edge2-FlashHead \
#   HF_TOKEN=hf_xxx \
#   IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
#   REMOTE_DIR=/tmp \
#   HF_HOME=/home/embedl/.cache/huggingface \
#   BENCHMARK_MODE=python \
#   ./tests/test_on_remote.sh agx-thor-san-jose
set -euo pipefail

# ---- Configuration (all have defaults) ----
REMOTE_HOST="${1:?Usage: $0 <remote-host>}"
MODEL="${MODEL:-embedl/Cosmos-Reason2-2B-W4A16-Edge2-FlashHead-clean}"
HF_TOKEN="${HF_TOKEN}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor}"
REMOTE_DIR="${REMOTE_DIR:-/tmp}"
HF_HOME="${HF_HOME:-/home/embedl/.cache/huggingface}"
BENCHMARK_MODE="${BENCHMARK_MODE:-both}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Sync repo ----
echo "[host] Syncing ${REPO_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR}/flash-head/"
rsync -az --delete \
    --exclude='.git' --exclude='.idea' --exclude='*.egg-info' \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='build' --exclude='dist' \
    "$REPO_DIR/" "${REMOTE_HOST}:${REMOTE_DIR}/flash-head/"

DOCKER_COMMON="--rm --runtime nvidia --network host \
    --ipc=host \
    --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${HF_HOME}:/root/.cache/huggingface \
    -v ${REMOTE_DIR}/flash-head:/opt/flash-head \
    -v /tmp/flashhead_bench:/tmp/flashhead_bench \
    -e HF_TOKEN=${HF_TOKEN} \
    -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn"

echo "[host] Running tests on ${REMOTE_HOST} (mode=${BENCHMARK_MODE})"
echo "  MODEL=${MODEL}"
echo "  IMAGE=${IMAGE}"

DROP_CACHES='sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true'

# Each benchmark runs in its own Docker container so that GPU memory
# is fully released between runs (Jetson unified memory).
ssh "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail

mkdir -p /tmp/flashhead_bench

INSTALL_CMD='pip install /opt/flash-head 2>&1 | tail -1'

if [ "${BENCHMARK_MODE}" = "python" ] || [ "${BENCHMARK_MODE}" = "both" ]; then
    ${DROP_CACHES}

    echo "=== [Python] Baseline benchmark (FlashHead disabled) ==="
    docker run ${DOCKER_COMMON} \
        -e FLASHHEAD_ENABLED=0 \
        ${IMAGE} \
        bash -c "\${INSTALL_CMD} && python3 /opt/flash-head/tests/run_benchmark.py \
            --model ${MODEL} \
            --max-tokens 128 --num-warmup 10 --num-runs 10 \
            --gpu-mem 0.75 --max-model-len 4096 \
            --output /tmp/flashhead_bench/python_baseline.json \
            --label python-baseline"

    ${DROP_CACHES}

    echo "=== [Python] FlashHead benchmark ==="
    docker run ${DOCKER_COMMON} \
        -e FLASHHEAD_ENABLED=1 \
        ${IMAGE} \
        bash -c "\${INSTALL_CMD} && python3 /opt/flash-head/tests/run_benchmark.py \
            --model ${MODEL} \
            --max-tokens 128 --num-warmup 10 --num-runs 10 \
            --gpu-mem 0.75 --max-model-len 4096 \
            --output /tmp/flashhead_bench/python_flashhead.json \
            --label python-flashhead"
fi

if [ "${BENCHMARK_MODE}" = "cli" ] || [ "${BENCHMARK_MODE}" = "both" ]; then
    ${DROP_CACHES}

    echo "=== [CLI] Baseline benchmark (FlashHead disabled) ==="
    docker run ${DOCKER_COMMON} \
        -e FLASHHEAD_ENABLED=0 \
        ${IMAGE} \
        bash -c "\${INSTALL_CMD} && vllm bench latency \
            --model ${MODEL} \
            --num-iters-warmup 10 --num-iters 10 \
            --batch-size 1 \
            --max-model-len 4096 \
            --gpu-mem 0.75 \
            --output-json /tmp/flashhead_bench/cli_baseline.json"

    ${DROP_CACHES}

    echo "=== [CLI] FlashHead benchmark ==="
    docker run ${DOCKER_COMMON} \
        -e FLASHHEAD_ENABLED=1 \
        ${IMAGE} \
        bash -c "\${INSTALL_CMD} && vllm bench latency \
            --model ${MODEL} \
            --num-iters-warmup 10 --num-iters 10 \
            --batch-size 1 \
            --max-model-len 4096 \
            --gpu-mem 0.75 \
            --output-json /tmp/flashhead_bench/cli_flashhead.json"
fi

echo "=== Running comparison tests ==="
docker run ${DOCKER_COMMON} \
    -e MODEL=${MODEL} \
    -e BENCHMARK_MODE=${BENCHMARK_MODE} \
    ${IMAGE} \
    bash -c "\${INSTALL_CMD} && pip install pytest 2>&1 | tail -1 && \
        python3 -m pytest /opt/flash-head/tests/test_compare.py -v -s"
REMOTE

echo "[host] Done (mode=${BENCHMARK_MODE})."
