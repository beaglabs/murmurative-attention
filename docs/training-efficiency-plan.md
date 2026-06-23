# Murmurative Attention — Training Efficiency Plan

## Goal

Prove the core thesis: **O(N·M) FLOPs → same intelligence for lower training cost vs standard attention.**

Current benchmarks show murmurative is 29x slower than SDPA at N=8192 in raw op comparison, and ~2.2x slower at the full model level. The gap is entirely a kernel efficiency problem: the `Q·Kᵀ` matmul in `slot_select` runs on scalar FP32 cores at ~1-3 TFLOPS effective, while the hardware can do ~200 TFLOPS on the same shape via tensor cores.

This plan covers:

1. **Three kernel changes** to close the efficiency gap (target: 5x+ speedup)
2. **Training Efficiency Benchmark** (Section 5 of `e2e_benchmark.py`) to measure the thesis

---

## Part A: Three Kernel Changes

### Change 1: Tensor-Core `slot_select` (est. 4-6 days)

**Impact: ~3x on `slot_select_attend`, ~2x on overall model throughput**

#### Problem

The `slot_select_attend_fwd_kernel` in `src/murmurative/csrc/cuda/slot_select_attend.cu` (lines 76-96) computes dot products with a scalar loop:

```cuda
for (int m = K_NEIGHBORS; m < effective_M; ++m) {
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += q_local[d] * __half2float(smem_keys[m * D + d]);
    }
    ...
```

This is a `[N, D] × [D, M]` matmul running on scalar FP32 cores. At N=8192, D=128, M=256, this is 537M FLOPs per round. At ~1 TFLOPS effective, it takes ~35ms per round — 70% of the kernel's time.

#### Solution

Replace with `nvcuda::wmma` warp-level matrix multiply-accumulate. Requires `#include <mma.h>` and `#include <cuda_fp16.h>`.

**New kernel structure**:

```
Grid:  (ceil(N/64), ceil(effective_M/64), B*H)
Block: 128 threads (4 warps)
Shared memory: Q_tile [64, D] + K_tile [D, 64] = 16KB for D=128

Per block:
  1. Cooperative load Q_tile [64, D] and K_tile [D, 64] into SMEM
  2. Each warp computes one 16×16 wmma tile of scores
     - 4 warps × (N_tile=64, M_tile=64) covers the output tile
  3. Iterate over K-dimension tiles (D/16 steps) accumulating in fp32
  4. After matmul: each thread owns scores[its_tokens, its_slots]
  5. Thread-level top-7 sort + softmax + weighted sum (reuse existing code from lines 118-154)
```

**wmma tile configuration**:

```cuda
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;  // Q [16, 16]
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;  // K^T [16, 16]
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;                   // scores [16, 16]
```

Key: slot keys are stored `[M, D]` row-major. We want `[D, M]` for the matmul. Using `col_major` for the B fragment reads from the same layout without a transpose — it treats the M dimension as columns.

**Backward pass**: The backward `bwd_1_kernel` also does a scalar dot product for `d_query`. This should also use `wmma` for the `[N, D] × [D, M]` portion. The `bwd_2_kernel` (scatter accumulation) is O(N·M) but not matmul-shaped — leave as-is.

**#ifdef fallback**: Keep the scalar path for non-multiple-of-16 dimensions (e.g., D=80):

```cuda
#if __CUDA_ARCH__ >= 700
    if (D % 16 == 0 && M_tile % 16 == 0) {
        wmma_path(...);
    } else
#endif
    {
        scalar_path(...);
    }
```

**Verification**:

- `tests/test_gradcheck.py` must pass (atol relaxed to 5e-3 for fp16 wmma vs reference)
- `tests/test_select.py` and `tests/test_attend.py` must still pass
- Numerical: indices must match scalar path exactly (same top-7). Output and weights within 5e-3.

**Files changed**:

- `src/murmurative/csrc/cuda/slot_select_attend.cu` (forward kernel + backward part 1)
- Possibly a new helper header `src/murmurative/csrc/cuda/wmma_helpers.cuh`

---

### Change 2: Persistent Round Kernel (est. 5-8 days)

**Impact: ~1.3x (eliminates kernel launch overhead + global memory round-trips for slots)**

#### Problem

`ops.py` lines 374-415 launch 2-3 CUDA kernels per round in a Python `for` loop:

```
for r in range(rounds):
    slot_select_attend(...)   # kernel launch 1
    x = x + out                # Python add
    slot_update(...)           # kernel launch 2 (or fused update+diffuse)
    slot_diffusion(...)        # kernel launch 3 (if not fused)
```

With `rounds=4` and the fused path: 8 kernel launches per forward pass, plus 8 `.clone()` allocations. Each launch costs ~5-10µs on modern GPUs. Slot K/V are written to global memory, then immediately read back next round.

#### Solution

Create a single CUDA kernel that runs all rounds internally: `slot_murmurate_fused_kernel`. New file: `src/murmurative/csrc/cuda/slot_murmurate_fused.cu`.

**Kernel design**:

```cuda
__global__ void slot_murmurate_fused_kernel(
    at::Half* x, at::Half* q, at::Half* k, at::Half* v,
    at::Half* slot_k, at::Half* slot_v,     // mutated in-place
    int64_t* indices, float* weights, at::Half* attn_out,
    int B, int H, int N, int D, int M, int effective_M,
    int rounds, float alpha, float gamma, float scale) {

    extern __shared__ char smem[];
    half* smem_slot_k = reinterpret_cast<half*>(smem);
    half* smem_slot_v = smem_slot_k + M * D;
    half* smem_x_tile = smem_slot_v + M * D;

    cg::grid_group grid = cg::this_grid();

    // 1. Init: load slot embeddings into SMEM once
    cooperative_load_slots(smem_slot_k, smem_slot_v, slot_k, slot_v, M, D);

    for (int r = 0; r < rounds; r++) {
        // 2. Tensor-core select+attend (inline, from Change 1)
        wmma_select_attend(q, smem_slot_k, smem_slot_v,
                           attn_out, weights, indices, ...);

        // 3. Residual: x_h += attn_out (each thread adds its portion)
        // 4. Update: scatter token K/V into SMEM slots
        //    (thread-level: find slots this token selected, atomicAdd into SMEM)
        // 5. Normalize accumulated slot contributions
        // 6. Diffusion: tridiagonal stencil on SMEM slots (same as existing diffusion kernel)
        // 7. EMA: smem_slot = alpha * old + (1-alpha) * new

        grid.sync();  // all blocks must finish round before next begins
    }

    // Write final x back to global memory
    // Write final slot_k, slot_v back to global memory
}
```

**Grid**: `(ceil(N/64), ceil(M/64), B*H)` — reuses the tensor-core select tiling.

**Key technical challenge**: `grid.sync()` requires cooperative groups and the kernel must be launched with `cudaLaunchCooperativeKernel`. This means all blocks must fit on the GPU simultaneously. For M=256, D=128, the SMEM per block is:

- slot_k: 256 × 128 × 2 = 64KB
- slot_v: 256 × 128 × 2 = 64KB  
- Q_tile + scratch: ~32KB
- Total: ~160KB per block

A100 has 164KB SMEM per SM. With `ceil(M/64) × B*H = 4 × B*H` blocks and 108 SMs, the grid fits if B*H ≤ 27. For B=4, H=8 → 32 blocks → fits. For larger batches, fall back to the non-persistent path in Python.

**Python side** (`ops.py`): New autograd Function:

```python
class _SlotMurmurateFused(Function):
    @staticmethod
    def forward(ctx, x_h, q_h, k_h, v_h, slot_k, slot_v, ...):
        # Single kernel call for all rounds
        out_x, out_slot_k, out_slot_v, indices, weights = torch.ops.murmurative_attention.slot_murmurate_fused(...)
        ctx.save_for_backward(q_h, k_h, v_h, slot_k, slot_v, indices, weights, ...)
        return out_x, out_slot_k, out_slot_v

    @staticmethod
    def backward(ctx, grad_x, grad_sk, grad_sv):
        # Chain existing backward kernels in correct order
        # (reverse of forward: diffuse → update → select_attend)
        ...
```

**Fallback**: When the persistent kernel can't launch (too many blocks), fall back to the existing per-round launch path with a warning. This preserves correctness for all batch/head configurations.

**Verification**:

- Output matches `slot_murmurate(use_reference=True)` within 1e-3
- `test_gradcheck.py` with the fused function
- `test_multiround.py` still passes
- Profile: confirm kernel launch count drops from 8 to 1

**Files changed**:

- `src/murmurative/csrc/cuda/slot_murmurate_fused.cu` (new)
- `torch-ext/torch_binding.cpp` (register new op)
- `torch-ext/wrappers.cpp` (add wrapper)
- `src/murmurative/ops.py` (add `_SlotMurmurateFused`, use in `slot_murmurate`)
- `setup.py` (add new .cu source)

---

### Change 3: Reduce Default Rounds to 3 (est. 1 day)

**Impact: 1.33x (25% fewer FLOPs per forward pass)**

#### Rationale

The plan's Section 2d states global reach is achieved after ~log₂(M) ≈ 8 diffusion steps. With 3 rounds × 1 diffusion step per round + 1 initial select = enough for ~6 effective diffusion steps. Bumping `gamma` from 0.1 to 0.15 compensates for the missing round.

#### Changes

1. `src/murmurative/ops.py` line 338: `rounds: int = 4` → `rounds: int = 3`
2. `src/murmurative/ops.py` line 340: `gamma: float = 0.1` → `gamma: float = 0.15`
3. `benchmarks/e2e_benchmark.py`: Update model constructors and arg defaults
4. All model constructors that set `rounds=` defaults

#### Validation

Run perplexity benchmark at `--rounds 3` vs `--rounds 4` at N=512, steps=500. If 3-round perplexity is within 5% of 4-round, the change is safe. If quality degrades, keep 4 as default but add a `--fast-rounds` flag.

---

### Combined Impact

| Change | Speedup | Cumulative |
|--------|---------|------------|
| Tensor-core slot_select | ~3x | 3.0x |
| Persistent round kernel | ~1.3x | 3.9x |
| 3 rounds (default) | 1.33x | **5.2x** |

At 5.2x attention speedup, the model-level throughput (Section 3) should reach ~45-50K tok/s at N=8192 — close to MHA's ~57K tok/s. At N=16K+, murmurative should overtake MHA.

---

## Part B: Training Efficiency Benchmark (Section 5)

**File**: `benchmarks/e2e_benchmark.py` — new section after Section 4.

### Purpose

Directly test the plan's core thesis: **"O(N·M) FLOPs → same intelligence for lower training cost."** The benchmark trains matched-size murmurative and MHA models on WikiText-2 at multiple sequence lengths and reports quality-per-compute metrics.

### What It Measures

For each N in `[512, 1024, 2048, 4096]`:

```
Train MurmurativeModel and MHAModel for --eff-steps steps
Measure every --eff-eval-every steps:
  ├── train_ppl, valid_ppl (proxy for "intelligence")
  ├── elapsed wall-clock time (practical cost)
  ├── cumulative GFLOPs consumed (theoretical cost floor)
  └── peak GPU memory
```

**FLOP counting** (cumulative per step):

Murmurative forward FLOPs:
- Embedding: N × D MACs
- QKV projections: 3 × N × D² MACs
- Attention: `count_murmurative_flops(N, Dh, M, rounds, H, effective_M)` × 2 (MACs→FLOPs)
- FFN: 8 × N × D² MACs (2 linear layers with 4× expansion)
- Output head: N × D × V MACs
- Backward: 2× forward (standard approximation)
- Total per step: (fwd + bwd) FLOPs, accumulated over steps

MHA forward FLOPs:
- Same embedding, QKV, FFN, output head
- Attention: `count_torch_sdpa_flops(N, D)` (returns GFLOPs fwd+bwd)
- Same backward ratio

### Output Format

```
=== Section 5: Training Efficiency ===

  N=512  (500 steps, d_model=256, heads=4)
  ─────────────────────────────────────────────────────
    model         train_ppl  valid_ppl   time(s)   GFLOPs   mem(MB)  ppl/GFLOP  ppl/s
  ─────────────────────────────────────────────────────
    murmurative      28.3      30.1       42.1      12.4      1490      2.28     0.67
    mha              26.1      27.8       21.5      18.7      1574      1.40     1.21
    flash            26.0      27.6       18.9      18.7      1490      1.39     1.38
  ─────────────────────────────────────────────────────
    ratio (m/mha)    1.08      1.08       1.96x     0.66x     0.95x    1.63x    0.55x
    ratio (m/flash)  1.09      1.09       2.23x     0.66x     1.00x    1.64x    0.49x

  N=1024 ...
  N=2048 ...
  N=4096 ...

  Crossover Projection
  ──────────────────────────────────────────────────────────────
   Metric                    Current kernels    With tensor cores (3x attn)
  ──────────────────────────────────────────────────────────────
   ppl / GFLOP advantage     N≈512 (already)    N≈512  (wider margin)
   ppl / wall-second parity  N≈12K (est)        N≈4K   (est)
   Memory advantage          N≈2K  (already)     N≈2K   (already)
  ──────────────────────────────────────────────────────────────
```

The "With tensor cores" column takes measured perplexity and GFLOPs (unchanged) but divides wall-clock time by 3x for the attention portion (~50% of total time → 1.5x total speedup → wall time / 1.5).

### CLI Integration

New flags added to `e2e_benchmark.py`:

```
--skip-efficiency        Skip Section 5
--eff-steps 500          Training steps per N (default 500)
--eff-eval-every 100     Evaluate perplexity every N steps
--eff-ns 512 1024 2048   Sequence lengths to sweep (default: 512 1024 2048 4096)
```

### Implementation

Add the following functions to `e2e_benchmark.py`:

1. **`count_model_flops(model_type, N, D, H, Dh, M, rounds, effective_M, vocab_size)`** — returns GFLOPs for one fwd+bwd step
2. **`run_efficiency_benchmark(args)`** — main driver: builds models, runs training loop with FLOP tracking, returns structured results
3. **`print_efficiency_table(results)`** — formats the per-N comparison table
4. **`print_crossover_projection(results)`** — projects crossover N with/without tensor cores

**FLOP counter** (cumulative across training steps):

```python
def count_model_flops_one_step(model_type, B, N, D, H, Dh, M, rounds, em, vocab_size):
    """FLOPs for one fwd+bwd step. Returns GFLOPs."""
    # Common FLOPs (same for both models)
    embed = N * D * 2          # lookup + scale
    qkv = 3 * N * D * D * 2    # 3 projections, MACs→FLOPs
    ffn = 8 * N * D * D * 2    # 2 layers with 4x expansion
    head = N * D * vocab_size * 2

    if model_type == "murmurative":
        attn = count_murmurative_flops(N, Dh, M, rounds, H, em)
    else:
        attn = count_torch_sdpa_flops(N, D)

    total = embed + qkv + ffn + head + attn * 1e9  # attn already in GFLOPs
    return total / 1e9  # return GFLOPs
```

**Training loop** (per N, per model):

```python
def train_with_flop_tracking(model, train_tokens, args, model_type):
    """Train for --eff-steps, tracking FLOPs and perplexity."""
    steps = args.eff_steps
    seq_len = ...  # current N
    batch_size = args.batch_size

    train_losses = []
    eval_log = []
    cumulative_gflops = 0.0
    step_gflops = count_model_flops_one_step(model_type, batch_size, seq_len, ...)

    t0 = time.perf_counter()
    for step in range(steps):
        # Standard training step (same as existing train_perplexity_model loop)
        ...
        cumulative_gflops += step_gflops

        if (step + 1) % args.eff_eval_every == 0:
            valid_ppl = eval_perplexity(model, valid_tokens, seq_len)
            train_ppl = math.exp(avg_train_loss)
            eval_log.append({
                "step": step + 1,
                "train_ppl": train_ppl,
                "valid_ppl": valid_ppl,
                "time_s": time.perf_counter() - t0,
                "gflops": cumulative_gflops,
            })

    total_time = time.perf_counter() - t0
    return eval_log, total_time, cumulative_gflops
```

**Crossover estimation**:

```python
def estimate_crossover(efficiency_rows):
    """
    For each metric (ppl/GFLOP, ppl/s), estimate the N where murmurative
    overtakes MHA. Uses linear interpolation on log-log space.
    """
    # Current kernels: use measured wall-clock times
    # With tensor cores: murmurative attention time *= 1/3, recalculate wall time
    ...
```

---

## Implementation Order

| Step | What | Depends on | Effort | Cumulative impact |
|------|------|-----------|--------|-------------------|
| **1** | Section 5 benchmark code | Nothing | 2-3 days | Proves thesis today (shows FLOP efficiency advantage even if wall-clock trails) |
| **2** | Change 3 (3 rounds default) | Step 1 (validate quality) | 1 day | 1.33x |
| **3** | Change 1 (tensor-core select) | Nothing | 4-6 days | ~3x (cumulative ~4x) |
| **4** | Change 2 (persistent kernel) | Change 1 (uses same wmma) | 5-8 days | ~1.3x (cumulative ~5.2x) |
| **5** | Re-run Section 5 with optimized kernels | Steps 2-4 | 1 day | Final numbers with the full 5x speedup |

**Total**: 13-19 days to prove the thesis end-to-end with optimized kernels.

**Step 1 is independently valuable** — it quantifies the FLOP efficiency advantage regardless of kernel quality, and gives measurable target numbers for Steps 2-4 to hit.

---

## Risk: What If Quality Doesn't Match?

The entire thesis rests on murmurative achieving comparable perplexity to MHA at the same parameter count and training budget. If perplexity lags by >10%, no amount of kernel optimization helps.

**Mitigation**: Step 1 runs first. If murmurative perplexity is >10% worse than MHA at all N, the architecture needs attention before kernel work. Possible fixes:

- Increase slots (M=256→512) to boost capacity
- Increase rounds back to 4
- Add position encoding in slot space (the plan's Section 10 mentions learnable per-slot position embeddings)
- Add auxiliary slot diversity loss to prevent collapse

The Section 5 benchmark surfaces this risk immediately.

---

## Risk: Persistent Kernel Complexity

Cooperative grid launch and grid-level sync add significant complexity and portability concerns. If Change 2 proves too difficult:

**Fallback**: Use CUDA graphs instead. Capture the entire round loop as a graph, eliminating launch overhead without requiring a single fused kernel. This is simpler and more portable (no cooperative launch restrictions). Trade-off: slightly less memory bandwidth savings (slot K/V still go through global memory between rounds, but L2 cache captures them).

---

## Files Summary

### New files
- `src/murmurative/csrc/cuda/wmma_helpers.cuh` — wmma tile helpers
- `src/murmurative/csrc/cuda/slot_murmurate_fused.cu` — persistent round kernel

### Modified files
- `src/murmurative/csrc/cuda/slot_select_attend.cu` — tensor-core fwd + bwd_1
- `torch-ext/torch_binding.cpp` — register new fused op
- `torch-ext/wrappers.cpp` — add wrapper
- `src/murmurative/ops.py` — `rounds=3` default, `_SlotMurmurateFused`
- `benchmarks/e2e_benchmark.py` — Section 5
- `setup.py` — add new .cu source

### Docs
- `docs/training-efficiency-plan.md` — this document