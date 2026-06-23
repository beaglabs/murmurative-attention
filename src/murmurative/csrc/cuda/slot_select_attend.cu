#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>

constexpr int K_NEIGHBORS = 7;
constexpr int BLOCK_DIM = 256;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_N_TILE = 64;
constexpr int WMMA_M_TILE = 64;
constexpr int WMMA_BLOCK_DIM = 128;

// ---------------------------------------------------------------------------
// Forward: fused select + attend
// ---------------------------------------------------------------------------
template <int BLOCK_DIM_X>
__global__ void slot_select_attend_fwd_kernel(
    const at::Half* __restrict__ query,
    const at::Half* __restrict__ slot_keys,
    const at::Half* __restrict__ slot_values,
    const at::Half* __restrict__ position_bias,
    const bool* __restrict__ token_mask,
    at::Half* __restrict__ output,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M,
    int64_t effective_M,
    float scale, bool causal, bool has_position_bias, bool has_mask) {

    extern __shared__ char smem[];
    at::Half* smem_keys = reinterpret_cast<at::Half*>(smem);

    const int64_t bh = blockIdx.y;
    const int64_t b = bh / H;
    const int64_t h = bh % H;

    // Cooperative load of all slot keys for this (b, h) into shared memory
    const int64_t sk_base = bh * M * D;
    for (int i = threadIdx.x; i < M * D; i += BLOCK_DIM_X) {
        smem_keys[i] = slot_keys[sk_base + i];
    }
    __syncthreads();

    const int64_t n = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    if (n >= N) return;

    // Handle fully-masked tokens: return uniform weights, zero output
    if (has_mask) {
        // token_mask is pre-flattened to [B, N] contiguous bool
        if (!token_mask[b * N + n]) {
            const int64_t out_base = bh * N * D + n * D;
            const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
            const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
            for (int d = 0; d < D; ++d) {
                output[out_base + d] = __float2half(0.0f);
            }
            for (int k = 0; k < K_NEIGHBORS; ++k) {
                weights[w_base + k] = 1.0f / K_NEIGHBORS;
                indices[idx_base + k] = k;
            }
            return;
        }
    }

    // Load query into registers
    const int64_t q_base = bh * N * D + n * D;
    float q_local[128];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q_local[d] = __half2float(query[q_base + d]);
    }

    // Maintain running top-K in registers
    float topk_scores[K_NEIGHBORS];
    int   topk_idx[K_NEIGHBORS];

    // Seed with first K slots
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dot += q_local[d] * __half2float(smem_keys[k * D + d]);
        }
        topk_scores[k] = dot * scale;
        topk_idx[k] = k;
    }

    // Scan remaining slots; inline causal / position-bias
    for (int m = K_NEIGHBORS; m < effective_M; ++m) {
        if (causal && m > n) break;

        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dot += q_local[d] * __half2float(smem_keys[m * D + d]);
        }
        float score = dot * scale;

        if (has_position_bias) {
            score += __half2float(position_bias[bh * N * M + n * M + m]);
        }

        // Find minimum in current top-K and replace if better
        int min_k = 0;
        float min_score = topk_scores[0];
        #pragma unroll
        for (int k = 1; k < K_NEIGHBORS; ++k) {
            if (topk_scores[k] < min_score) {
                min_score = topk_scores[k];
                min_k = k;
            }
        }
        if (score > min_score) {
            topk_scores[min_k] = score;
            topk_idx[min_k] = m;
        }
    }

    // Softmax over top-K
    float max_score = topk_scores[0];
    #pragma unroll
    for (int k = 1; k < K_NEIGHBORS; ++k) {
        if (topk_scores[k] > max_score) max_score = topk_scores[k];
    }
    float sum_exp = 0.0f;
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        topk_scores[k] = expf(topk_scores[k] - max_score);
        sum_exp += topk_scores[k];
    }
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        topk_scores[k] /= sum_exp;
    }

    // Gather values and write output
    const int64_t sv_base = bh * M * D;
    const int64_t out_base = bh * N * D + n * D;
    const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
    const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            acc += topk_scores[k] * __half2float(slot_values[sv_base + topk_idx[k] * D + d]);
        }
        output[out_base + d] = __float2half(acc);
    }
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        weights[w_base + k] = topk_scores[k];
        indices[idx_base + k] = topk_idx[k];
    }
}

// ---------------------------------------------------------------------------
// Forward WMMA (tensor-core) path — used when D % 16 == 0 && effective_M % 16 == 0
// Grid: (ceil(effective_M / 64),  ceil(N / 64),  B*H)
// Block: 128 threads (4 warps), SMEM: Q_tile [64,D] + K_tile [D,64] + scores = 48KB
// Backward kernels are reused unchanged (operate on stored indices/weights).
// ---------------------------------------------------------------------------
__global__ void slot_select_attend_fwd_kernel_wmma(
    const at::Half* __restrict__ query,
    const at::Half* __restrict__ slot_keys,
    const at::Half* __restrict__ slot_values,
    const at::Half* __restrict__ position_bias,
    at::Half* __restrict__ output,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M,
    int64_t effective_M,
    float scale, bool causal, bool has_position_bias) {

    using namespace nvcuda::wmma;

    extern __shared__ char smem[];
    half* smem_q       = reinterpret_cast<half*>(smem);
    half* smem_k       = smem_q + WMMA_N_TILE * D;
    float* smem_scores = reinterpret_cast<float*>(smem_k + D * WMMA_M_TILE);

    const int64_t bh       = blockIdx.z;
    const int64_t b        = bh / H;
    const int64_t h        = bh % H;
    const int64_t m_block  = blockIdx.x;
    const int64_t n_block  = blockIdx.y;

    const int64_t n_start  = n_block * WMMA_N_TILE;
    const int64_t m_start  = m_block * WMMA_M_TILE;
    const int64_t n_end    = min(n_start + WMMA_N_TILE, N);
    const int64_t m_end    = min(m_start + WMMA_M_TILE, effective_M);
    const int64_t n_local  = n_end - n_start;
    const int64_t m_local  = m_end - m_start;

    // Cooperative load Q_tile [n_start:n_end, :] into SMEM
    const int64_t q_base = bh * N * D;
    for (int i = threadIdx.x; i < n_local * D; i += WMMA_BLOCK_DIM) {
        int nt = i / D;
        int d  = i % D;
        smem_q[nt * D + d] = query[q_base + (n_start + nt) * D + d];
    }
    // Cooperative load K_tile in col-major [D, m_local]
    const int64_t sk_base = bh * M * D;
    for (int i = threadIdx.x; i < D * m_local; i += WMMA_BLOCK_DIM) {
        int d  = i / m_local;
        int mt = i % m_local;
        smem_k[d * WMMA_M_TILE + mt] = slot_keys[sk_base + (m_start + mt) * D + d];
    }
    __syncthreads();

    // WMMA matmul: each warp owns 16 rows of N
    int warp_id  = threadIdx.x / 32;
    int wn_start = warp_id * WMMA_M;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    for (int mt = 0; mt < m_local; mt += WMMA_N) {
        fill_fragment(acc_frag, 0.0f);

        for (int kt = 0; kt < D; kt += WMMA_K) {
            load_matrix_sync(a_frag, smem_q + wn_start * D + kt, D);
            load_matrix_sync(b_frag, smem_k + kt * WMMA_M_TILE + mt, WMMA_M_TILE);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        for (int t = 0; t < acc_frag.num_elements; ++t) {
            acc_frag.x[t] *= scale;
        }
        store_matrix_sync(smem_scores + wn_start * WMMA_M_TILE + mt,
                          acc_frag, WMMA_M_TILE, mem_row_major);
    }
    __syncthreads();

    // Per-token top-7 + softmax + weighted sum (threads 0..63)
    for (int tn = threadIdx.x; tn < n_local; tn += (WMMA_BLOCK_DIM / 2)) {
        const int64_t n = n_start + tn;

        float top_scores[K_NEIGHBORS];
        int   top_idx[K_NEIGHBORS];

        for (int k = 0; k < K_NEIGHBORS && k < m_local; ++k) {
            top_scores[k] = smem_scores[tn * WMMA_M_TILE + k];
            top_idx[k]    = m_start + k;
        }
        for (int k = m_local; k < K_NEIGHBORS; ++k) {
            top_scores[k] = -FLT_MAX;
            top_idx[k]    = 0;
        }

        for (int ml = K_NEIGHBORS; ml < m_local; ++ml) {
            const int64_t m = m_start + ml;
            if (causal && m > n) break;

            float score = smem_scores[tn * WMMA_M_TILE + ml];
            if (has_position_bias) {
                score += __half2float(position_bias[bh * N * M + n * M + m]);
            }

            int min_k = 0;
            float min_score = top_scores[0];
            for (int k = 1; k < K_NEIGHBORS; ++k) {
                if (top_scores[k] < min_score) {
                    min_score = top_scores[k];
                    min_k = k;
                }
            }
            if (score > min_score) {
                top_scores[min_k] = score;
                top_idx[min_k]    = m;
            }
        }

        if (has_position_bias) {
            for (int k = 0; k < K_NEIGHBORS && k < m_local; ++k) {
                top_scores[k] += __half2float(
                    position_bias[bh * N * M + n * M + top_idx[k]]);
            }
        }

        float max_score = top_scores[0];
        for (int k = 1; k < K_NEIGHBORS; ++k) {
            if (top_scores[k] > max_score) max_score = top_scores[k];
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            top_scores[k] = expf(top_scores[k] - max_score);
            sum_exp += top_scores[k];
        }
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            top_scores[k] /= sum_exp;
        }

        const int64_t sv_base  = bh * M * D;
        const int64_t out_base = bh * N * D + n * D;
        const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
        const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;

        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int k = 0; k < K_NEIGHBORS; ++k) {
                acc += top_scores[k] * __half2float(
                    slot_values[sv_base + top_idx[k] * D + d]);
            }
            output[out_base + d] = __float2half(acc);
        }
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            weights[w_base + k] = top_scores[k];
            indices[idx_base + k] = top_idx[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Backward – part 1 (per-token): d_query  +  temporary d_scores
// ---------------------------------------------------------------------------
template <int BLOCK_DIM_X>
__global__ void slot_select_attend_bwd_1_kernel(
    const at::Half* __restrict__ query,
    const at::Half* __restrict__ slot_keys,
    const at::Half* __restrict__ slot_values,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const at::Half* __restrict__ grad_output,
    at::Half* __restrict__ grad_query,
    float* __restrict__ grad_scores,   // [B, H, N, K] temp buffer
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M,
    float scale) {

    extern __shared__ char smem[];
    at::Half* smem_values = reinterpret_cast<at::Half*>(smem);

    const int64_t bh = blockIdx.y;
    const int64_t sv_base = bh * M * D;
    for (int i = threadIdx.x; i < M * D; i += BLOCK_DIM_X) {
        smem_values[i] = slot_values[sv_base + i];
    }
    __syncthreads();

    const int64_t n = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    if (n >= N) return;

    const int64_t q_base   = bh * N * D + n * D;
    const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
    const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
    const int64_t go_base  = bh * N * D + n * D;

    float go_local[128];
    for (int d = 0; d < D; ++d) {
        go_local[d] = __half2float(grad_output[go_base + d]);
    }

    // d_weights[k] = dot(grad_output[n], slot_values[idx[n,k]])
    float d_weights[K_NEIGHBORS];
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        int64_t slot = indices[idx_base + k];
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dot += go_local[d] * __half2float(smem_values[slot * D + d]);
        }
        d_weights[k] = dot;
    }

    // Back-prop through softmax: d_scores[k] = w_k * (d_w_k - sum_j w_j d_w_j)
    float sum_wdw = 0.0f;
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        sum_wdw += weights[w_base + k] * d_weights[k];
    }
    float d_scores_local[K_NEIGHBORS];
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        d_scores_local[k] = weights[w_base + k] * (d_weights[k] - sum_wdw);
        grad_scores[w_base + k] = d_scores_local[k];
    }

    // d_query[n,d] = sum_k d_scores[k] * slot_keys[idx[n,k], d] * scale
    float q_local[128];
    const int64_t sk_base = bh * M * D;
    for (int d = 0; d < D; ++d) {
        q_local[d] = 0.0f;
    }
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        int64_t slot = indices[idx_base + k];
        float ds = d_scores_local[k] * scale;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            q_local[d] += ds * __half2float(slot_keys[sk_base + slot * D + d]);
        }
    }
    for (int d = 0; d < D; ++d) {
        grad_query[q_base + d] = __float2half(q_local[d]);
    }
}

// ---------------------------------------------------------------------------
// Backward – part 2 (per-slot): d_slot_keys  +  d_slot_values
// ---------------------------------------------------------------------------
template <int BLOCK_DIM_X>
__global__ void slot_select_attend_bwd_2_kernel(
    const at::Half* __restrict__ query,
    const at::Half* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    const float* __restrict__ grad_scores,
    const float* __restrict__ weights,
    float* __restrict__ grad_slot_keys_f32,
    float* __restrict__ grad_slot_values_f32,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M,
    float scale) {

    const int64_t bh = blockIdx.y;
    const int64_t m  = blockIdx.x;
    const int tid = threadIdx.x;

    if (m >= M) return;

    const int64_t sk_base = bh * M * D;

    // Each thread handles a subset of dimensions
    for (int d = tid; d < D; d += BLOCK_DIM_X) {
        float dsk = 0.0f;
        float dsv = 0.0f;

        // Scan all tokens, looking for those that selected slot m
        for (int n = 0; n < N; ++n) {
            const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
            const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
            const int64_t q_base   = bh * N * D + n * D;
            const int64_t go_base  = bh * N * D + n * D;

            for (int k = 0; k < K_NEIGHBORS; ++k) {
                if (indices[idx_base + k] == m) {
                    float ds = grad_scores[w_base + k];
                    float q_val  = __half2float(query[q_base + d]);
                    float go_val = __half2float(grad_output[go_base + d]);
                    float w_val  = weights[w_base + k];

                    dsk += ds * q_val * scale;
                    dsv += w_val * go_val;
                }
            }
        }

        grad_slot_keys_f32[sk_base + m * D + d] = dsk;
        grad_slot_values_f32[sk_base + m * D + d] = dsv;
    }
}

// ---------------------------------------------------------------------------
// C++ launchers
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slot_select_attend_forward(
    const torch::Tensor& query,
    const torch::Tensor& slot_keys,
    const torch::Tensor& slot_values,
    const c10::optional<torch::Tensor>& mask,
    bool causal,
    const c10::optional<torch::Tensor>& position_bias,
    c10::optional<double> scale,
    int64_t effective_M) {

    TORCH_CHECK(query.device().is_cuda(), "query must be CUDA");
    TORCH_CHECK(slot_keys.device().is_cuda(), "slot_keys must be CUDA");
    TORCH_CHECK(slot_values.device().is_cuda(), "slot_values must be CUDA");
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B,H,N,D]");
    TORCH_CHECK(slot_keys.dim() == 4, "slot_keys must be 4D [B,H,M,D]");

    const auto B = query.size(0);
    const auto H = query.size(1);
    const auto N = query.size(2);
    const auto D = query.size(3);
    const auto M = slot_keys.size(2);

    TORCH_CHECK(D <= 128, "slot_select_attend supports D <= 128");
    const int64_t smem_bytes = M * D * sizeof(at::Half);
    TORCH_CHECK(smem_bytes <= 48 * 1024,
                "Dynamic SMEM request (", smem_bytes,
                " bytes) exceeds 48KB. Reduce M*D or D.");

    const float s = scale.has_value()
        ? static_cast<float>(scale.value())
        : 1.0f / std::sqrt(static_cast<float>(D));

    // Determine whether to use the WMMA path
    // Flatten mask to [B, N] bool if provided (needed for has_mask check below)
    torch::Tensor token_mask;
    bool has_mask = false;
    if (mask.has_value()) {
        auto m = mask.value();
        auto mask_4d = m.expand({B, H, N, M}).contiguous();
        token_mask = mask_4d.any(-1).any(1);
        token_mask = token_mask.to(torch::kBool);
        has_mask = true;
    }

    bool use_wmma = false;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    if (props.major >= 7
        && D % WMMA_K == 0
        && effective_M % WMMA_N == 0
        && !has_mask
        && D <= 128) {
        int64_t wmma_smem = WMMA_N_TILE * D * sizeof(at::Half)
                          + D * WMMA_M_TILE * sizeof(at::Half)
                          + WMMA_N_TILE * WMMA_M_TILE * sizeof(float);
        if (wmma_smem <= 48 * 1024) {
            use_wmma = true;
        }
    }

    const int n_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    const int bh_blocks = B * H;

    auto indices = torch::empty(
        {B, H, N, K_NEIGHBORS},
        torch::TensorOptions().dtype(torch::kInt64).device(query.device()));
    auto weights = torch::empty(
        {B, H, N, K_NEIGHBORS},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
    auto output = torch::empty_like(query);

    const at::cuda::OptionalCUDAGuard device_guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (use_wmma) {
        // WMMA path: tile N and M into 64-element blocks
        int n_blocks_wmma = (N + WMMA_N_TILE - 1) / WMMA_N_TILE;
        int m_blocks_wmma = (effective_M + WMMA_M_TILE - 1) / WMMA_M_TILE;
        dim3 grid_wmma(m_blocks_wmma, n_blocks_wmma, bh_blocks);
        dim3 block_wmma(WMMA_BLOCK_DIM);
        int64_t wmma_smem = WMMA_N_TILE * D * sizeof(at::Half)
                          + D * WMMA_M_TILE * sizeof(at::Half)
                          + WMMA_N_TILE * WMMA_M_TILE * sizeof(float);

        slot_select_attend_fwd_kernel_wmma<<<grid_wmma, block_wmma, wmma_smem, stream>>>(
            reinterpret_cast<const at::Half*>(query.data_ptr()),
            reinterpret_cast<const at::Half*>(slot_keys.data_ptr()),
            reinterpret_cast<const at::Half*>(slot_values.data_ptr()),
            position_bias.has_value()
                ? reinterpret_cast<const at::Half*>(position_bias.value().data_ptr())
                : nullptr,
            reinterpret_cast<at::Half*>(output.data_ptr()),
            weights.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            B, H, N, D, M, effective_M, s, causal,
            position_bias.has_value());
    } else {
        // Scalar path (original)
        TORCH_CHECK(D <= 128, "slot_select_attend supports D <= 128");
        const int64_t smem_bytes = M * D * sizeof(at::Half);
        TORCH_CHECK(smem_bytes <= 48 * 1024,
                    "Dynamic SMEM request (", smem_bytes,
                    " bytes) exceeds 48KB. Reduce M*D or D.");

        dim3 grid(n_blocks, bh_blocks);
        dim3 block(BLOCK_DIM);

        slot_select_attend_fwd_kernel<BLOCK_DIM><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const at::Half*>(query.data_ptr()),
            reinterpret_cast<const at::Half*>(slot_keys.data_ptr()),
            reinterpret_cast<const at::Half*>(slot_values.data_ptr()),
            position_bias.has_value()
                ? reinterpret_cast<const at::Half*>(position_bias.value().data_ptr())
                : nullptr,
            has_mask ? token_mask.data_ptr<bool>() : nullptr,
            reinterpret_cast<at::Half*>(output.data_ptr()),
            weights.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            B, H, N, D, M, effective_M, s, causal,
            position_bias.has_value(), has_mask);
    }

    return std::make_tuple(output, weights, indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slot_select_attend_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_weights,   // from downstream
    const torch::Tensor& query,
    const torch::Tensor& slot_keys,
    const torch::Tensor& slot_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    c10::optional<double> scale) {

    TORCH_CHECK(query.device().is_cuda(), "query must be CUDA");
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be CUDA");

    const auto B = query.size(0);
    const auto H = query.size(1);
    const auto N = query.size(2);
    const auto D = query.size(3);
    const auto M = slot_keys.size(2);

    const float s = scale.has_value()
        ? static_cast<float>(scale.value())
        : 1.0f / std::sqrt(static_cast<float>(D));

    const int n_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    const int bh_blocks = B * H;

    auto grad_query = torch::empty_like(query);
    auto grad_slot_keys  = torch::zeros_like(slot_keys);
    auto grad_slot_values = torch::zeros_like(slot_values);

    // Temporary float32 buffers for scatter accumulation
    auto grad_slot_keys_f32  = torch::zeros({B, H, M, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
    auto grad_slot_values_f32 = torch::zeros({B, H, M, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    // Temporary buffer for d_scores
    auto grad_scores = torch::empty(
        {B, H, N, K_NEIGHBORS},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    const int64_t smem_bwd1 = M * D * sizeof(at::Half);
    const int64_t smem_bwd2 = 0; // no dynamic SMEM needed

    dim3 grid1(n_blocks, bh_blocks);
    dim3 block1(BLOCK_DIM);
    dim3 grid2(M, bh_blocks);
    dim3 block2(BLOCK_DIM);

    const at::cuda::OptionalCUDAGuard device_guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Part 1: per-token backward (d_query + d_scores)
    slot_select_attend_bwd_1_kernel<BLOCK_DIM><<<grid1, block1, smem_bwd1, stream>>>(
        reinterpret_cast<const at::Half*>(query.data_ptr()),
        reinterpret_cast<const at::Half*>(slot_keys.data_ptr()),
        reinterpret_cast<const at::Half*>(slot_values.data_ptr()),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<const at::Half*>(grad_output.data_ptr()),
        reinterpret_cast<at::Half*>(grad_query.data_ptr()),
        grad_scores.data_ptr<float>(),
        B, H, N, D, M, s);

    // Part 2: per-slot backward (d_slot_keys + d_slot_values)
    slot_select_attend_bwd_2_kernel<BLOCK_DIM><<<grid2, block2, smem_bwd2, stream>>>(
        reinterpret_cast<const at::Half*>(query.data_ptr()),
        reinterpret_cast<const at::Half*>(grad_output.data_ptr()),
        indices.data_ptr<int64_t>(),
        grad_scores.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_slot_keys_f32.data_ptr<float>(),
        grad_slot_values_f32.data_ptr<float>(),
        B, H, N, D, M, s);

    // Cast float32 scatter accumulators back to half
    grad_slot_keys.copy_(grad_slot_keys_f32.to(grad_slot_keys.dtype()));
    grad_slot_values.copy_(grad_slot_values_f32.to(grad_slot_values.dtype()));

    return std::make_tuple(grad_query, grad_slot_keys, grad_slot_values);
}
