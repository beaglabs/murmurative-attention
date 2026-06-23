# Murmurative Attention Kernel — Implementation Plan

## Overview

A CUDA kernel implementing slot-based "murmurative" attention. Tokens attend to a fixed-size pool of **M=256 learnable slots** rather than to each other. Each token finds its **k=7** nearest slots via content similarity, reads from them, and writes back — with **slot-to-slot tridiagonal diffusion** between rounds to achieve global receptive field. Multiple rounds propagate information across the sequence without ever crossing the O(N) barrier.

**Complexity**: O(N·M) per round (M fixed), O(1) per-token autoregressive decode.

---

## 1. Package Structure

```
murmurative-kernels/
├── pyproject.toml                  # hf-kernel-builder config
├── src/
│   └── murmurative/
│       ├── __init__.py
│       ├── ops.py                  # Public API
│       ├── reference.py            # PyTorch reference implementation
│       ├── csrc/
│       │   └── cuda/
│       │       ├── slot_select.cu  # Q@Kslots^T + top-7
│       │       ├── slot_attend.cu  # gather + softmax + aggregate
│       │       ├── slot_update.cu  # scatter-accumulate into slots
│       │       └── slot_diffusion.cu  # tridiagonal slot→slot propagation
│       └── torch_bridge.py         # torch.autograd.Function + torch.library
├── tests/
│   ├── test_select.py
│   ├── test_attend.py
│   ├── test_update.py
│   ├── test_diffusion.py
│   ├── test_multiround.py
│   ├── test_causal.py
│   └── test_gradcheck.py
├── benchmarks/
│   ├── bench_vs_flash.py
│   └── bench_scaling.py
└── .github/workflows/
    └── test.yml
```

---

## 2. Core Primitives — API Signatures

All tensors are `[B, H, N, D]` for tokens (batch, heads, seq_len, dim) and `[B, H, M, D]` for slots. All floating-point inputs use fp16 or bf16. Indices are int32.

### 2a. `slot_select` — Find k=7 nearest slots per token

```python
slot_select(
    query:      Tensor[B, H, N, D],              # fp16/bf16
    slot_keys:  Tensor[B, H, M, D],              # fp16/bf16
    mask:       Optional[Tensor[B, 1, N, 1]],    # padding mask
    causal:     bool = False,                    # restrict to "past" slots for AR
    position_bias: Optional[Tensor[B, H, N, M]], # RoPE/ALiBi injection
) -> Tensor[B, H, N, 7]                          # int32, top-7 slot indices, sorted
```

**CUDA strategy**: Tile N-axis across thread blocks. Within each block, compute a tile of similarity [N_tile, M]. Use bitonic sort directly in registers to extract top-7 per row. No shared memory needed for sorting — at k=7 and M=256, the sort overhead is ~28 comparisons per token. The matmul [N_tile×D] × [D×M] uses warp-level tensor core fragments. The slot key tensor [B, H, M, D] is in constant memory or L2 for the duration.

**Position bias**: If provided, added element-wise to Q·Kᵀ before top-k. Handles RoPE (pre-rotated), ALiBi (broadcast), or learned relative position biases.

**Causal mode**: When `causal=True`, mask slot indices ≤ slot_from_token(i). For slot-based attention, "causal" means a token at position i can only attend to slot representations that contain information from tokens ≤ i. This requires the slot pool to evolve sequentially. In practice: run select/attend/update **sequentially per token** during training with CUDA graph replay for the inner loop, or use a left-to-right sequential kernel for the forward pass.

### 2b. `slot_attend` — Attend over selected slots

```python
slot_attend(
    query:        Tensor[B, H, N, D],   # fp16/bf16
    slot_keys:    Tensor[B, H, M, D],   # fp16/bf16
    slot_values:  Tensor[B, H, M, D],   # fp16/bf16
    indices:      Tensor[B, H, N, 7],   # int32, from slot_select
    scale:        float = None,         # defaults to 1/sqrt(D)
) -> Tuple[                              # returns:
    Tensor[B, H, N, D],                  #   attended output
    Tensor[B, H, N, 7],                  #   attention weights (post-softmax, fp32)
]
```

**CUDA strategy**: Thread-per-token. Gather 7 elements from slot_keys and slot_values using the indices. Compute 7 dot products (Q with each gathered K), scale, softmax over the 7, weighted sum of 7 gathered V. All 7 intermediate values live in registers. Memory access: read Q [D], read 7 K [7×D] and 7 V [7×D] from global memory, write output [D] and weights [7]. For D=64, the register pressure is ~(1+7+7+1)×64 fp16 = 128 fp16 registers — tight but fits in 255 registers/SM on A100.

Return attention weights so `slot_update` can reuse them (avoids recomputing softmax).

### 2c. `slot_update` — Write token info into slots

```python
slot_update(
    token_keys:    Tensor[B, H, N, D],    # fp16/bf16
    token_values:  Tensor[B, H, N, D],    # fp16/bf16
    slot_keys:     Tensor[B, H, M, D],    # fp16/bf16, mutated in-place
    slot_values:   Tensor[B, H, M, D],    # fp16/bf16, mutated in-place
    indices:       Tensor[B, H, N, 7],    # int32
    weights:       Tensor[B, H, N, 7],    # fp32 attention weights
    alpha:         float = 0.9,           # EMA decay
)
```

**CUDA strategy**: Scatter-accumulate with EMA: `slot[i] = alpha * slot[i] + (1-alpha) * sum(token_k * weight) / sum(weight)` for all tokens that selected slot i. The challenge is atomic contention when many tokens target the same slot.

Implementation: sort tokens by their primary slot index (first of 7). Partition work by slot — each thread block owns 1-4 slots. Within a block, use warp-level reductions to accumulate contributions in shared memory, then a single atomic per slot at the end. This reduces the atomic count from N×7 to M (256 atomics per head per round).

**Contention mitigation**: If profiling shows Zipfian slot distribution (a few slots dominate), apply load-balancing by splitting hot slots across multiple thread blocks with partial reductions before a single global atomic. The key insight: M=256 is small enough that all slot data fits in L2 cache (512KB for 256×64×2×fp16 = 32KB per head, 256KB for 8 heads).

### 2d. `slot_diffusion` — Slot-to-slot information propagation

```python
slot_diffusion(
    slot_keys:    Tensor[B, H, M, D],    # fp16/bf16, mutated in-place
    slot_values:  Tensor[B, H, M, D],    # fp16/bf16, mutated in-place
    gamma:        float = 0.1,           # diffusion strength
)
```

**CUDA strategy**: A single thread block per head. For each channel d, sweep M=256 slots applying a tridiagonal stencil: `slot[i] += gamma * (slot[i-1] + slot[i+1] - 2*slot[i])`. Optional learned per-channel gamma. This is ~50k FLOPs (0.003% of total round FLOPs). The entire kernel uses one block, executing in ~1µs of compute + one launch overhead. For 8 heads @ 4 rounds: 32 launches × 5µs = 160µs total overhead. Negligible.

**Design decision**: Tridiagonal (3-point) stencil rather than fully-connected M×M because:
- Global reach is achieved after ~log₂(M) ≈ 8 rounds of tridiag diffusion
- Full M×M is O(M²) per diffusion step vs O(M) for tridiag
- Learned slot ordering means the model can arrange slots so "semantically adjacent" concepts are neighbors in the ordering
- Extension possible: multi-channel learned stencil (weights shared across D, learned as 3×D parameters)

### 2e. Composed multi-round forward pass

```python
slot_murmurate(
    x:          Tensor[B, N, D],         # input tokens
    num_slots:  int   = 256,
    num_heads:  int   = 8,
    k:          int   = 7,               # neighbors per token
    rounds:     int   = 4,               # propagation rounds
    alpha:      float = 0.9,             # EMA decay for slot_update
    gamma:      float = 0.1,             # diffusion strength
    causal:     bool  = False,
    position_bias: Optional[callable] = None,  # RoPE function
) -> Tensor[B, N, D]                     # updated token representations
```

Internally:
1. Project `x` to Q, K, V via external `nn.Linear` (not in kernel)
2. Initialize slot pool: `slot_k, slot_v = LearnedSlotEmbeddings[H, M, D]` (Perceiver-style, trained as `nn.Parameter`)
3. For each round:
   - `idx = slot_select(Q, slot_k, causal=causal, pos_bias=pos_bias)`
   - `attn_w, out = slot_attend(Q, slot_k, slot_v, idx)`
   - `x = x + out`  (residual connection)
   - `slot_update(K, V, slot_k, slot_v, idx, attn_w, alpha)`
   - **`slot_diffusion(slot_k, slot_v, gamma)`**  ← new, global mixing
   - Optionally recompute Q, K, V from updated x
4. Return x

---

## 3. Autoregressive Decode Flow

Prefill (batch): run `slot_murmurate` normally over the full prompt. The slot pool captures the prompt's compressed representation.

Decode (per-token streaming):

```python
slot_k, slot_v = learned_slot_embeddings  # reset per sequence
# ... prefill with prompt tokens ...
for token in generate():
    q, k, v = project(token)
    idx = slot_select(q, slot_k, causal=True)       # 1 token × 256 slots
    attn_w, out = slot_attend(q, slot_k, slot_v, idx)
    token = token + out
    slot_update(k, v, slot_k, slot_v, idx, attn_w, alpha=0.9)
    slot_diffusion(slot_k, slot_v, gamma=0.1)
    yield token
```

Per-token cost is constant: ~20K FLOPs for slot_select (1×256×64) + negligible attend/update/diffusion. Total dominated by the QKV projection and MLP layers. Memory: the slot pool is 32KB per head (256×64×2 bytes for fp16), stays resident in L1/L2 throughout generation. No growing KV cache.

---

## 4. CUDA Kernel Design Notes

### 4a. `slot_select.cu` — the workhorse

```
Grid:  (N / 128) blocks, [H] blocks for head batching
Block: 128 threads → process 128 tokens per block

Algorithm per block:
  1. Load Q_tile [128, D] into shared memory
  2. Load slot_keys [M, D] from global to shared (M=256, D=64 → 16KB, fits)
  3. For each pair of registers (thread covers 1-4 tokens):
     a. Compute dot products: thread.tokens × slot_keys → [tile_tokens, M] scores
     b. Bitonic sort network to extract top-7 per token (in registers)
     c. Write top-7 indices to global [N, 7]
```

**Optimization**: If D < 128, pack multiple tokens per thread. For D=64, each thread handles 2 tokens, computing two independent sets of dot products. This doubles register pressure but improves occupancy by halving the thread count.

**Tensor core usage**: When D ≥ 64, use `nvcuda::wmma` for the [N_tile, D] × [D, M] matmul. Accumulate in fp32, convert to fp16 for the sort. Need to handle non-multiple-of-16 dimensions (D=64 works, D=80 doesn't) — pad internally or fall back to warp-level vectorized fp16 intrinsics.

### 4b. `slot_attend.cu` — trivial, bandwidth-bound

```
Grid:  (N / 256, H) — 256 tokens per block
Block: 256 threads → one token per thread

Per-thread (fp16, D=64):
  1. Load Q [64] from global
  2. Load indices [7] from global
  3. Gather K [7×64] and V [7×64] from slot_keys/slot_values via indices
     — these come from global but are small, may hit L2 if slot pool is resident
  4. Compute 7 dot products: Q·K[i] for i=0..6
  5. Scale, exponentials, softmax (fp32 accumulator, exp in fp32)
  6. Weighted sum: sum(w[i] * V[i])
  7. Write output [64] and weights [7] to global

FP8 option: K and V are loaded as fp8, dequantized to fp16 on-the-fly.
```

### 4c. `slot_update.cu` — the contention bottleneck

```
Grid:  (M / 4, H) — 4 slots per block (64 blocks for M=256)
Block: 128 threads

Algorithm:
  1. Load slot target indices [N, 7] and weights [N, 7] from global
  2. Load token K [N, D] and V [N, D] from global
  3. Each block owns 4 slot indices. For each owned slot:
     a. Scan all N tokens for those that selected this slot (any position in their top-7)
     b. Warp-reduce the weighted token contributions in shared memory
     c. Single atomic fp32 add to global slot accumulator
  4. Barrier across blocks (persistent kernel or separate post-pass)
  5. Normalize: divide accumulated values by total weight per slot
  6. Apply EMA: slot = alpha * old_slot + (1-alpha) * slot_accumulated
```

**Key optimization**: Step 3a is a scan over N for each slot → effectively O(N×M) reads. But we already read the indices tensor every round anyway for select/attend. An alternative is to **fuse slot_update with slot_attend**: as each thread computes its 7 attention-weighted V contributions, it also scatters them into a shared staging area. Then a reduction pass aggregates per slot. This avoids the separate scan over N entirely. Consider this fusion for V2.

For V1, the independent kernel is simpler to implement and debug. Profile first, optimize only if slot_update dominates.

### 4d. `slot_diffusion.cu` — negligible

Single block per head. M iterations along the slot dimension, D iterations per slot, 3 multiply-adds per element. ~50K FLOPs. Launch overhead dominates. Consider fusing into `slot_update` as a post-pass within the same kernel using a cooperative groups `grid.sync()`.

---

## 5. Implementation Roadmap

| Phase | Deliverable | Effort | Verification |
|-------|------------|--------|-------------|
| **0. Reference** | `reference.py` — pure PyTorch slot_murmurate with all 4 ops | 1-2 days | Run on random tensors, verify multi-round convergence |
| **1. `slot_attend` CUDA** | First kernel — lowest complexity, builds the torch_bridge infra | 1 day | Match reference output within 1e-3 |
| **2. `slot_select` CUDA** | Matmul + top-7 bitonic sort. Core compute kernel | 2-3 days | Match indices, then match full round output |
| **3. `slot_update` CUDA** | Scatter-accumulate with atomics | 2 days | Slot values match reference after update |
| **4. `slot_diffusion` CUDA** | Tridiagonal stencil | 0.5 day | Slot values match reference after diffusion |
| **5. Gradient pass** | Backward kernels for select/attend (update/diffusion are auto-diff friendly) | 2 days | `torch.autograd.gradcheck` passes |
| **6. Composed API** | `ops.py` wrapping everything, multi-round loop, slot init | 1 day | End-to-end test against reference |
| **7. HF packaging** | `hf-kernel-builder`, `torch.library` registration, signing | 1 day | `import murmurative; help(murmurative.slot_murmurate)` |
| **8. Correctness** | Fuzz tests, edge cases, causal mode, padding masks | 2 days | CI green |
| **9. Benchmarks** | N=1k, 8k, 32k, 131k vs Flash Attention | 1 day | Scaling charts |

**Total estimated effort**: 13-17 engineer-days for a solid V1.

---

## 6. Gradient Strategy

`slot_select` uses hard argmax → gradients can't flow through the indices. Strategy:
- **Forward**: Hard top-7. Indices are frozen int32.
- **Backward through `slot_attend`**: Gradients flow through V (values) and attention weights, using the frozen indices to route. This is identical to Flash Attention's approach — the selection itself has no gradient, but the values and weights do.
- **Backward through `slot_update`**: Gradients flow through the accumulated slot K/V, routed back to input tokens via the frozen indices and weights.
- **`slot_diffusion`**: Fully differentiable (linear operation), trivially back-propagated.

**What's lost**: The model cannot learn through gradient descent which slots a token should select. The selection is driven only by Q·K similarity. This is the standard attention design and hasn't been a problem in any attention variant.

**V2 option**: Straight-through estimator or Gumbel-soft top-k for learned routing if needed. Not in scope for V1.

---

## 7. Testing Strategy

| Test category | What | Key assertions |
|---------------|------|----------------|
| Shape propagation | All ops on [1,H,128,64], [2,H,1024,64], [1,H,8192,64] | Output shapes correct |
| Reference parity | CUDA vs PyTorch reference | Allclose atol=1e-3, rtol=1e-3 |
| Gradient check | `torch.autograd.gradcheck` on each op | All gradients match within 1e-4 |
| Multi-round | 4 rounds of full pipeline | Output stable, no NaN/Inf |
| Causal mode | Sequential vs batched produces same result | Allclose |
| Padding | Masked tokens get zero output | Sum of masked outputs ≈ 0 |
| Edge cases | N < 7, M=1, D=1, all-zeros input | No crash, sensible output |
| FP16/bf16 parity | Both dtypes produce results within 2e-3 | Mixed precision safe |
| Memory | Peak VRAM at N=131k fits in 40GB | Monitor with nvml |

---

## 8. HF Kernels Integration

Build config (`pyproject.toml`):
```toml
[project]
name = "murmurative-kernels"
version = "0.1.0"

[tool.hf_kernels]
cuda = { sources = ["src/murmurative/csrc/cuda/*.cu"], architectures = ["80", "89", "90"] }
library = "murmurative"
```

Registration (`torch_bridge.py`):
```python
import torch
from torch.library import custom_op

@custom_op("murmurative::slot_select", mutates_args=())
def slot_select(query, slot_keys, mask, causal, pos_bias):
    return torch.ops.murmurative.slot_select_cuda(query, slot_keys, mask, causal, pos_bias)
```

Distribution:
```bash
hf kernels build
hf kernels sign
hf kernels publish murmurative-kernels
```

Users install with: `pip install murmurative-kernels --extra-index-url https://huggingface.co/kernels`

---

## 9. Performance Targets (A100, fp16)

| Scenario | Target | Notes |
|----------|--------|-------|
| Prefill N=8k, 1 head | <100µs | Per head per round |
| Prefill N=131k, 8 heads, 4 rounds | <5ms | Dominated by slot_select |
| Decode per-token (any N) | <15µs | Includes QKV projection |
| Peak VRAM N=1M, 8 heads | <500MB | Slot pool is 256KB, activations dominate |
| vs Flash Attention prefill N=131k | >50x faster | Wall clock, not FLOPs |

---

## 10. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Slot collapse** — all tokens pick same 7 slots, bottlenecking information | Medium | EMA decay (alpha=0.9) ensures slow overwrite; add entropy regularization in training loss |
| **Atomic contention in slot_update** | Medium | Warp-level reductions + shared memory staging; if severe, fuse with slot_attend in V2 |
| **Causal mode correctness** — parallel prefill vs sequential decode must match | High | Extensive testing; left-to-right sequential kernel option if parallelism fails |
| **Tensor core under-utilization** in slot_select M=256 | Low | M=256 still achieves ~40% utilization with head batching (8×256=2048); acceptable |
| **Learned slot ordering** — model might not learn meaningful slot topology | Medium | Initialize slot embeddings from PCA of token embeddings; add auxiliary slot-diversity loss if needed |
| **Position bias in slot space** — no natural "closeness" metric for slots | Low | Learnable per-slot position embeddings or use the learned slot ordering as a 1D coordinate |

---

## 11. V2 Candidates (not in scope)

- Multi-scale slot hierarchy (fine/coarse/global layers)
- Fused slot_update + slot_attend (single kernel, no intermediate indices write)
- Persistent round kernel (all 4 rounds in a single kernel launch)
- fp8 neighbor selection path
- Gumbel-soft top-k trained routing
- Flash-Decoding style KV streaming for ultra-long (1M+) prefill
- Triton port for PyTorch 2.0 compatibility