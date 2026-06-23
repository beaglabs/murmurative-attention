FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 1. System dependencies -------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget ninja-build build-essential cmake \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3 -m pip install --upgrade pip setuptools wheel

# 2. Heavy Python dependencies (cached independently of source changes) ------------
# Align PyTorch with the host CUDA version (12.4).
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Use pre-built flash-attn wheel when available; do not force a source build.
RUN pip install packaging psutil ninja && \
    pip install flash-attn --no-build-isolation

# Test / benchmark helpers.
RUN pip install pytest numpy

# 3. Copy build metadata and C++ sources first ---------------------------------------
COPY pyproject.toml setup.py build.toml /workspace/murmurative-kernels/
COPY torch-ext/ /workspace/murmurative-kernels/torch-ext/
COPY src/ /workspace/murmurative-kernels/src/

WORKDIR /workspace/murmurative-kernels

# 4. Build the CUDA extension and install the package in editable mode ---------------
# Set target architectures explicitly (no GPU available at Docker build time).
# MAX_JOBS=2 keeps memory low enough to compile 5 arches in parallel.
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"
# Limit parallel ninja jobs to avoid OOM in constrained build environments.
ENV MAX_JOBS=2
# --no-build-isolation is required because setup.py imports torch at the top level.
RUN pip install -e . --no-build-isolation

# Verify the compiled extension is importable and registers the torch.ops namespace.
RUN python -c "import murmurative_attention_ops; import torch; _ = torch.ops.murmurative_attention; print('CUDA extension OK')"

# 5. Copy remaining project files (invalidates only this layer on change) ------------
COPY benchmarks/ /workspace/murmurative-kernels/benchmarks/
COPY tests/ /workspace/murmurative-kernels/tests/
COPY scripts/ /workspace/murmurative-kernels/scripts/

# 6. Runtime -------------------------------------------------------------------------
CMD ["bash", "/workspace/murmurative-kernels/scripts/run_bench.sh"]
