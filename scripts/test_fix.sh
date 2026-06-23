#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Script to clone, build, and test the murmurative-kernels CUDA extension.
# Run inside an environment with CUDA + PyTorch (e.g. the Azure container).
# ---------------------------------------------------------------------------

REPO_URL="${REPO_URL:-https://github.com/jdbohrman/murmurative-attention.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace/murmurative-kernels}"

echo "=== Cloning repo (branch: ${BRANCH}) ==="
if [ -d "${WORKDIR}" ]; then
    echo "Directory ${WORKDIR} already exists — pulling latest..."
    git -C "${WORKDIR}" checkout "${BRANCH}"
    git -C "${WORKDIR}" pull origin "${BRANCH}"
else
    git clone --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi

echo ""
echo "=== Installing package (builds CUDA extension) ==="
pip install -e "${WORKDIR}" --no-build-isolation

echo ""
echo "=== Verifying CUDA extension loads ==="
python -c "
import murmurative_attention_ops
import torch
_ = torch.ops.murmurative_attention
print('CUDA extension namespace OK')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Running training benchmark (next-token prediction) ==="
python "${WORKDIR}/benchmarks/bench_training.py" --steps 100 --seq-len 256 --vocab 4096 --d-model 256 --num-heads 4 --num-slots 128 --rounds 4 --batch-size 4

echo ""
echo "=== Running training loop tests ==="
python -m pytest "${WORKDIR}/tests/test_training_loop.py" -v --tb=short

echo ""
echo "=== All tests complete ==="