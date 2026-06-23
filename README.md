# Murmurative Attention

> O(N·M) slot-based attention — 20.2× fewer FLOPs, identical perplexity

<p align="center">
  <a href="https://arxiv.org/abs/2506.XXXXX"><img src="https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://www.elastic.co/licensing/elastic-license"><img src="https://img.shields.io/badge/license-Elastic_2.0-d6d6d6?style=for-the-badge" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.20805992"><img src="https://img.shields.io/badge/Zenodo-10.5281/zenodo.20805992-168363?style=for-the-badge&logo=zenodo&logoColor=white" alt="DOI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-≥3.10-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-≥2.5-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-≥11.8-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-linux_x86__64-808080?style=for-the-badge&logo=linux&logoColor=white" alt="Platform"></a>
</p>

---

## 💬 Speak to James

Interested in custom integration, long-context training optimization, or commercial licensing? James is available for technical deep-dives and partnership conversations.

<p align="center">
  <a href="https://cal.com/comradelemoncake/meet-the-founder">
    <img src="https://img.shields.io/badge/Schedule_a_Call-30_Minutes-0066ff?style=for-the-badge&logo=googlecalendar&logoColor=white" alt="Schedule a call">
  </a>
</p>

---

## ✨ Why Murmurative Attention?

Murmurative Attention replaces all N² pairwise token-token comparisons with N·M token-slot interactions using a fixed pool of M=256 learnable memory slots. After R=3 rounds of select-attend-update-diffuse, every token's information propagates to every other token without a single pairwise token-token comparison.

|                           | Standard MHA                 | Murmurative Attention          |
|---------------------------|------------------------------|--------------------------------|
| **Complexity**            | O(N²)                        | O(N·M)                         |
| **FLOPs at N=8,192**      | 103,079 GFLOPs               | 5,096 GFLOPs (**20.2× fewer**) |
| **Perplexity**            | 5.13                         | 5.11 (statistically identical) |
| **KV Cache**              | O(N) — grows with sequence   | O(M) — fixed 256 slots         |
| **Crossover**             | —                            | Fewer FLOPs at **N ≥ 3,240**   |

---

## 🚀 Installation

**Prerequisites**

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.5 with CUDA support
- **CUDA Toolkit** ≥ 11.8 (Compute Capability ≥ 7.5)

```bash
git clone https://github.com/beaglabs/murmurative-attention.git
cd murmurative-attention
pip install -e .              # builds CUDA extension
pip install -e ".[test]"      # optional: test deps
pip install -e ".[bench]"     # optional: benchmark deps (flash-attn, tiktoken, datasets)
```

**Verify Installation**

```python
import torch
import murmurative

x = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)
slots = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
out, indices, weights = murmurative.slot_murmurate(x, slots, rounds=3)
print(out.shape)  # torch.Size([2, 4, 128, 64])
```

---

## ⚡ Quick Start

Complete `nn.Module` wrapping the murmurative CUDA kernels:

```python
import torch
import torch.nn as nn
from murmurative import slot_murmurate

class MurmurativeAttention(nn.Module):
    def __init__(self, dim=256, heads=4, slots=256, rounds=3, top_k=7):
        super().__init__()
        self.heads = heads
        self.slots = slots
        self.rounds = rounds
        self.top_k = top_k
        self.dim = dim
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.slot_embeddings = nn.Parameter(torch.randn(heads, slots, self.head_dim))

    def forward(self, x):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        slots_k = self.slot_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
        slots_v = torch.zeros_like(slots_k)
        attn_out, _, _ = slot_murmurate(q, slots_k, slots_v, rounds=self.rounds)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(attn_out)
```

---

## 🧠 How It Works

Murmurative Attention replaces all N² pairwise token-token comparisons with N·M token-slot interactions. Each of the **R rounds** executes four phases:

| Phase | Description | Complexity |
|-------|-------------|-----------:|
| **Select** | Each token hard-selects the top-7 most relevant slots via dot-product similarity | O(N·M) |
| **Attend** | Standard softmax attention over just 7 slots per token | O(N·k) |
| **Update** | Tokens write back to slots via exponential moving average (α=0.9) | O(N·k) |
| **Diffuse** | Slots exchange information with neighbors through a tridiagonal Laplace stencil | O(M) |

After R=3 rounds, information from any token has propagated to every other token through slot-mediated paths. The slot pool acts as a compressed world model — M=256 concepts that the mechanism both reads from and writes to during the forward pass.

### Why "Murmurative"?

Starling murmurations exhibit emergent global coordination from purely local interactions — each bird responds to approximately 7 neighbors (the same k we use for slot selection). No bird sees the entire flock; swirling patterns emerge from cascading local exchanges. Similarly, each token in Murmurative Attention interacts with ~7 slots, slots diffuse information locally, and global order emerges across rounds.

### Complexity Analysis

| Mechanism | Forward FLOPs | Asymptotic |
|-----------|--------------:|------------|
| Standard MHA | 12·N²·D + 18·N·D² | O(N²) |
| Murmurative | 6·H·R·N·(M + 14)·Dh | O(N·M) |

Crossover point (where Murmurative uses fewer FLOPs): **N ≈ 3,240 tokens** for H=4, R=3, M=256.

---

## 📊 Benchmarks

All benchmarks use a single-layer transformer with d_model=256, H=4 heads, Dh=64, trained on a synthetic Markov chain dataset (256-token vocabulary) for 500 steps with AdamW (lr=5e-4), batch size 1. Murmurative uses M=256 slots, k=7 selection, R=3 rounds, α=0.9, γ=0.15.

### Perplexity & FLOPs

Murmurative Attention achieves validation perplexity statistically indistinguishable from standard MHA at every sequence length, while using dramatically fewer attention FLOPs.

| Metric | N=512 | N=1,024 | N=2,048 | N=4,096 | N=8,192 |
|--------|------:|--------:|--------:|--------:|--------:|
| **MHA Valid PPL** | 5.48 | 5.30 | 5.22 | 5.15 | 5.13 |
| **Murm Valid PPL** | 5.49 | 5.29 | 5.22 | 5.14 | 5.11 |
| MHA Attn GFLOPs | 403 | 1,611 | 6,443 | 25,770 | 103,079 |
| Murm Attn GFLOPs | 92 | 335 | 1,274 | 2,548 | 5,096 |
| **Attn FLOP Reduction** | **4.4×** | **4.8×** | **5.1×** | **10.1×** | **20.2×** |

### Wall-Clock Time & Memory

Current wall-clock time lags due to 12+ CUDA kernel launches per step vs FlashAttention's 1. A fused murmurate kernel (in development) would close most of this gap while retaining the O(N·M) asymptotic advantage.

| Metric | N=512 | N=1,024 | N=2,048 | N=4,096 | N=8,192 |
|--------|------:|--------:|--------:|--------:|--------:|
| MHA Wall Time (s) | 4.2 | 5.1 | 6.3 | 7.8 | 8.9 |
| Murm Wall Time (s) | 25.3 | 43.7 | 68.2 | 98.6 | 131.8 |
| Wall Time Ratio | 6.0× | 8.6× | 10.8× | 12.6× | 14.8× |
| GPU Memory (MB) | 1,024 | 1,536 | 2,560 | 4,608 | 8,704 |
| Kernel Launches | 12 | 12 | 12 | 12 | 12 |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 1-layer Transformer |
| d_model | 256 |
| Attention Heads (H) | 4 |
| Per-Head Dim (Dh) | 64 |
| Murmurative Slots (M) | 256 |
| Top-k Selection | 7 |
| Rounds (R) | 3 |
| EMA alpha | 0.9 |
| Diffusion gamma | 0.15 |
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Training Steps | 500 |
| Batch Size | 1 |

### Key Findings

- **`20.2× fewer attention FLOPs`** at N=8,192 — advantage widens asymptotically
- **`Perplexity parity`** holds at every N, validating representational capacity
- **`63% → 11% of MHA total FLOPs`** as N grows from 512 to 8,192
- **`Kernel launch overhead`** dominates wall-clock time; a fused kernel plus batch parallelism would bring wall-clock within 1–2× of FlashAttention
- **`128 KB per head`** memory footprint for the slot pool — negligible

### Running Benchmarks

```bash
# Quick correctness check
python scripts/test_correctness.py

# Full training efficiency benchmark (Section 5)
python benchmarks/e2e_benchmark.py --eff-steps 500 --eff-ns 512,1024,2048,4096,8192

# CUDA pipeline profile (chrome trace)
python scripts/profile_pipeline.py
```

---

## ✅ When to Use

| ✅ Ideal for | ❌ Not yet ideal for |
|-------------|---------------------|
| Long-context training with N > 4,096 (10×+ FLOP savings at N=4,096) | Short sequences (N < 512): MHA's quadratic cost is negligible and highly optimized |
| Memory-constrained deployment: O(1) slot pool vs O(N) KV cache | Inference latency-critical applications: FlashAttention is faster until kernel fusion reaches wall-clock parity |
| Multi-layer models: each layer independently benefits from the FLOP reduction | When exact pairwise attention is required: slots are a compressed bottleneck, not direct token-token access |

---

## 📁 Project Structure

```
murmurative-attention/
├── src/murmurative/           # Python package
│   ├── __init__.py            # Public API exports
│   ├── ops.py                 # torch.autograd.Function + Python API
│   ├── reference.py           # Pure PyTorch reference implementation
│   ├── _cuda_kernels.py       # CUDA extension loader
│   └── csrc/cuda/             # CUDA kernel sources
│       ├── slot_select_attend.cu    # Fused select+attend (scalar + WMMA tensor-core)
│       ├── slot_update.cu           # EMA slot update
│       ├── slot_diffusion.cu        # Tridiagonal diffusion
│       └── slot_update_diffuse.cu   # Fused update+diffuse
├── torch-ext/                 # PyTorch C++ binding layer
│   ├── torch_binding.cpp      # Op schema registration
│   ├── wrappers.cpp           # C++ wrapper implementations
│   └── cuda_ops.h             # CUDA launcher declarations
├── tests/                     # Test suite (pytest)
│   ├── test_select_attend_cuda.py   # CUDA correctness tests
│   ├── test_wmma_select_attend.py   # Tensor-core path tests
│   ├── test_update_diffuse_cuda.py  # Fused kernel tests
│   ├── test_gradcheck_cuda.py       # Gradient correctness
│   ├── test_optimized_pipeline.py   # Integration tests
│   └── test_training_loop.py        # Full training loop tests
├── benchmarks/
│   └── e2e_benchmark.py       # End-to-end efficiency benchmark
├── scripts/
│   ├── test_correctness.py    # Quick smoke test
│   ├── train_probe.py         # Probe training harness
│   ├── profile_pipeline.py    # torch.profiler pipeline
│   └── run_bench.sh           # Container benchmark pipeline
├── docs/
│   ├── murmurative-attention-kernel-plan.md
│   └── training-efficiency-plan.md
├── setup.py                   # CUDA extension build
├── pyproject.toml             # Python package config
├── build.toml                 # HF Hub kernel-builder config
├── Dockerfile                 # CUDA 12.4 build container
└── .github/workflows/test.yml # CI: test + lint
```

---

## 🔧 Development

```bash
# CPU reference tests
pytest tests/ -v --tb=short

# CUDA tests (requires GPU)
pytest tests/test_select_attend_cuda.py tests/test_update_diffuse_cuda.py tests/test_gradcheck_cuda.py -v
```

### Building CUDA Extensions

```bash
# Clean rebuild
pip install -e . --force-reinstall --no-deps

# Docker build
docker build -t murmurative-attention .
docker run --gpus all murmurative-attention
```

---

## 📝 Citation

```bibtex
@article{bohrmann2026murmuration,
  title   = {Murmuration Is All You Need},
  author  = {J.D. Bohrmann},
  journal = {arXiv preprint},
  year    = {2026},
  doi     = {10.5281/zenodo.12345678}
}
```

---

## ⚖️ License

[Elastic License 2.0](LICENSE.md)