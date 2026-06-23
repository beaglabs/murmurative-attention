#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Azure Container Job Benchmark + Build + Publish Pipeline
#
# Run on an Azure NCasT4_v3 (T4) or NC_A100_v4 (A100) instance.
#
# Set these env vars:
#   HF_TOKEN           - Hugging Face API token
#   HF_REPO_ID         - e.g. "jdbohrman/murmurative-attention" (optional, default from build.toml)
#   PUBLISH            - "true" to upload to HF Hub (default: false)
#   BENCH_N            - space-separated sequence lengths (default: "1024 2048 4096 8192")
#   BENCH_MODE         - "prefill" | "decode" | "all" (default: "all")
#   BENCH_DTYPE        - "fp16" | "bf16" | "fp32" (default: "fp16")
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

BENCH_N="${BENCH_N:-1024 2048 4096 8192}"
BENCH_MODE="${BENCH_MODE:-all}"
BENCH_DTYPE="${BENCH_DTYPE:-fp16}"
PUBLISH="${PUBLISH:-false}"

echo "============================================"
echo "Murmurative Attention — Benchmark Pipeline"
echo "============================================"
echo "Device:       $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "CUDA:         $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch:      $(python3 -c 'import torch; print(torch.__version__)')"
echo "Flash-Attn:   $(python3 -c 'from flash_attn import __version__; print(__version__)' 2>/dev/null || echo 'not installed')"
echo "CUDA:         $(python3 -c 'import torch; print("available" if torch.cuda.is_available() else "not available")')"
echo ""
echo "BENCH_N:      ${BENCH_N}"
echo "BENCH_MODE:   ${BENCH_MODE}"
echo "BENCH_DTYPE:  ${BENCH_DTYPE}"
echo "PUBLISH:      ${PUBLISH}"
echo "============================================"

# Step 1: Quick correctness check (reference implementation)
echo ""
echo "--- Step 1: Correctness ---"
python3 -m pytest tests/ -v --tb=short -x 2>&1 | tail -20

# Step 2: Benchmark prefill vs flash-attn
# (CUDA kernels are auto-compiled via torch.utils.cpp_extension JIT at first use)
echo ""
echo "--- Step 2: Prefill Benchmark ---"
python3 benchmarks/bench_comparison.py \
    --N ${BENCH_N} \
    --mode ${BENCH_MODE} \
    --dtype ${BENCH_DTYPE} \
    --warmup 10 \
    --iters 30 \
    --output benchmarks/results.json

echo ""
echo "Results saved to benchmarks/results.json"

# Step 3: Publish to Hugging Face Hub (optional)
if [ "${PUBLISH}" = "true" ]; then
    echo ""
    echo "--- Step 3: Publish to Hugging Face Hub ---"

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: HF_TOKEN not set. Cannot publish."
        echo "  Run: export HF_TOKEN=hf_your_token"
        exit 1
    fi

    huggingface-cli login --token "${HF_TOKEN}"

    REPO_ID="${HF_REPO_ID:-}"
    if [ -z "${REPO_ID}" ]; then
        REPO_ID=$(python3 -c "
import tomli
with open('build.toml', 'rb') as f:
    data = tomli.load(f)
print(data.get('general', {}).get('hub', {}).get('repo-id', ''))
" 2>/dev/null || echo "")
    fi

    if [ -z "${REPO_ID}" ]; then
        echo "ERROR: Could not determine HF_REPO_ID. Set HF_REPO_ID or fix build.toml."
        exit 1
    fi

    echo "Publishing to: ${REPO_ID}"
    kernel-builder build-and-upload --repo-id "${REPO_ID}"

    echo ""
    echo "Published! View at: https://huggingface.co/${REPO_ID}"
else
    echo ""
    echo "--- Step 3: Publish (skipped) ---"
    echo "  Set PUBLISH=true and HF_TOKEN to publish to Hugging Face Hub."
fi

echo ""
echo "============================================"
echo "Pipeline complete"
echo "============================================"