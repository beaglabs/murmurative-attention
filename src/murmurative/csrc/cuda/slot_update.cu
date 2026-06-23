#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cfloat>
#include <cmath>

constexpr int K_NEIGHBORS = 7;

// ---------------------------------------------------------------------------
// Forward: block-per-slot, no atomics
// ---------------------------------------------------------------------------
__global__ void slot_update_fwd_kernel(
    const at::Half* __restrict__ token_keys,
    const at::Half* __restrict__ token_values,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    at::Half* __restrict__ slot_keys,
    at::Half* __restrict__ slot_values,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M, float alpha) {

    const int64_t bh = blockIdx.y;
    const int64_t m  = blockIdx.x;
    const int tid = threadIdx.x;

    if (m >= M) return;

    const int64_t sk_base = bh * M * D;

    float k_acc[128] = {0};
    float v_acc[128] = {0};
    float w_sum = 0.0f;

    // Scan all tokens
    for (int n = 0; n < N; ++n) {
        const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
        const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;

        float w = 0.0f;
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            if (indices[idx_base + k] == m) {
                w += weights[w_base + k];
            }
        }
        if (w == 0.0f) continue;

        if (tid == 0) w_sum += w;

        const int64_t tk_base = bh * N * D + n * D;
        for (int d = tid; d < D; d += blockDim.x) {
            k_acc[d] += w * __half2float(token_keys[tk_base + d]);
            v_acc[d] += w * __half2float(token_values[tk_base + d]);
        }
    }

    __shared__ float sh_w_sum;
    if (tid == 0) sh_w_sum = w_sum;
    __syncthreads();
    float w_sum_val = sh_w_sum;

    if (w_sum_val > 0.0f) {
        float one_minus_alpha = 1.0f - alpha;
        for (int d = tid; d < D; d += blockDim.x) {
            float old_k = __half2float(slot_keys[sk_base + m * D + d]);
            float old_v = __half2float(slot_values[sk_base + m * D + d]);
            float new_k = alpha * old_k + one_minus_alpha * k_acc[d] / w_sum_val;
            float new_v = alpha * old_v + one_minus_alpha * v_acc[d] / w_sum_val;
            slot_keys[sk_base + m * D + d] = __float2half(new_k);
            slot_values[sk_base + m * D + d] = __float2half(new_v);
        }
    }
}

// ---------------------------------------------------------------------------
// Backward – prep (per-slot): recompute accumulators, output temporaries
// ---------------------------------------------------------------------------
__global__ void slot_update_bwd_prep_kernel(
    const at::Half* __restrict__ token_keys,
    const at::Half* __restrict__ token_values,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const at::Half* __restrict__ grad_new_sk,
    const at::Half* __restrict__ grad_new_sv,
    at::Half* __restrict__ grad_old_sk,
    at::Half* __restrict__ grad_old_sv,
    float* __restrict__ temp_dac_k,    // [B,H,M,D]
    float* __restrict__ temp_dac_v,    // [B,H,M,D]
    float* __restrict__ temp_dw_sum,   // [B,H,M]
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M, float alpha) {

    const int64_t bh = blockIdx.y;
    const int64_t m  = blockIdx.x;
    const int tid = threadIdx.x;

    if (m >= M) return;

    const int64_t sk_base = bh * M * D;

    float k_acc[128] = {0};
    float v_acc[128] = {0};
    float w_sum = 0.0f;

    for (int n = 0; n < N; ++n) {
        const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
        const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;

        float w = 0.0f;
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            if (indices[idx_base + k] == m) {
                w += weights[w_base + k];
            }
        }
        if (w == 0.0f) continue;

        if (tid == 0) w_sum += w;

        const int64_t tk_base = bh * N * D + n * D;
        for (int d = tid; d < D; d += blockDim.x) {
            k_acc[d] += w * __half2float(token_keys[tk_base + d]);
            v_acc[d] += w * __half2float(token_values[tk_base + d]);
        }
    }

    __shared__ float sh_w_sum;
    if (tid == 0) sh_w_sum = w_sum;
    __syncthreads();
    float w_sum_val = sh_w_sum;

    if (w_sum_val > 0.0f) {
        // d_old_slots = alpha * d_new_slots  (slot was updated)
        for (int d = tid; d < D; d += blockDim.x) {
            float gsk = __half2float(grad_new_sk[sk_base + m * D + d]);
            float gsv = __half2float(grad_new_sv[sk_base + m * D + d]);
            grad_old_sk[sk_base + m * D + d] = __float2half(alpha * gsk);
            grad_old_sv[sk_base + m * D + d] = __float2half(alpha * gsv);
        }
    } else {
        // d_old_slots = d_new_slots  (slot was unchanged)
        for (int d = tid; d < D; d += blockDim.x) {
            grad_old_sk[sk_base + m * D + d] = grad_new_sk[sk_base + m * D + d];
            grad_old_sv[sk_base + m * D + d] = grad_new_sv[sk_base + m * D + d];
        }
    }

    if (w_sum_val <= 0.0f) {
        for (int d = tid; d < D; d += blockDim.x) {
            temp_dac_k[sk_base + m * D + d] = 0.0f;
            temp_dac_v[sk_base + m * D + d] = 0.0f;
        }
        temp_dw_sum[bh * M + m] = 0.0f;
        return;
    }

    float one_minus_alpha = 1.0f - alpha;
    float inv_w = 1.0f / w_sum_val;

    // Write dac_k and dac_v temporaries
    for (int d = tid; d < D; d += blockDim.x) {
        float gsk = __half2float(grad_new_sk[sk_base + m * D + d]);
        float gsv = __half2float(grad_new_sv[sk_base + m * D + d]);
        temp_dac_k[sk_base + m * D + d] = one_minus_alpha * inv_w * gsk;
        temp_dac_v[sk_base + m * D + d] = one_minus_alpha * inv_w * gsv;
    }

    // Compute dot(accum_k, grad_new_sk) + dot(accum_v, grad_new_sv)
    float dot_k = 0.0f, dot_v = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float gsk = __half2float(grad_new_sk[sk_base + m * D + d]);
        float gsv = __half2float(grad_new_sv[sk_base + m * D + d]);
        dot_k += k_acc[d] * gsk;
        dot_v += v_acc[d] * gsv;
    }

    // Block-level reduction of dot_k and dot_v
    __shared__ float sh_dot_k;
    __shared__ float sh_dot_v;
    if (tid == 0) {
        sh_dot_k = 0.0f;
        sh_dot_v = 0.0f;
    }
    __syncthreads();
    atomicAdd(&sh_dot_k, dot_k);
    atomicAdd(&sh_dot_v, dot_v);
    __syncthreads();

    if (tid == 0) {
        temp_dw_sum[bh * M + m] = -one_minus_alpha * (sh_dot_k + sh_dot_v) * inv_w * inv_w;
    }
}

// ---------------------------------------------------------------------------
// Backward – tokens (per-token): d_token_keys, d_token_values, d_weights
// ---------------------------------------------------------------------------
__global__ void slot_update_bwd_tokens_kernel(
    const at::Half* __restrict__ token_keys,
    const at::Half* __restrict__ token_values,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ temp_dac_k,
    const float* __restrict__ temp_dac_v,
    const float* __restrict__ temp_dw_sum,
    at::Half* __restrict__ grad_token_keys,
    at::Half* __restrict__ grad_token_values,
    float* __restrict__ grad_weights,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M) {

    const int64_t bh = blockIdx.y;
    const int64_t n  = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (n >= N) return;

    const int64_t tk_base  = bh * N * D + n * D;
    const int64_t idx_base = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
    const int64_t w_base   = bh * N * K_NEIGHBORS + n * K_NEIGHBORS;
    const int64_t gtk_base = bh * N * D + n * D;

    float d_tk[128] = {0};
    float d_tv[128] = {0};

    for (int k = 0; k < K_NEIGHBORS; ++k) {
        int64_t m = indices[idx_base + k];
        float w_val = weights[w_base + k];
        float dw_sum_part = temp_dw_sum[bh * M + m];

        float dot_tk = 0.0f;
        float dot_tv = 0.0f;
        for (int d = 0; d < D; ++d) {
            float tk_val = __half2float(token_keys[tk_base + d]);
            float tv_val = __half2float(token_values[tk_base + d]);
            float dac_k = temp_dac_k[bh * M * D + m * D + d];
            float dac_v = temp_dac_v[bh * M * D + m * D + d];

            d_tk[d] += w_val * dac_k;
            d_tv[d] += w_val * dac_v;

            dot_tk += tk_val * dac_k;
            dot_tv += tv_val * dac_v;
        }
        grad_weights[w_base + k] = dot_tk + dot_tv + dw_sum_part;
    }

    for (int d = 0; d < D; ++d) {
        grad_token_keys[gtk_base + d] = __float2half(d_tk[d]);
        grad_token_values[gtk_base + d] = __float2half(d_tv[d]);
    }
}

// ---------------------------------------------------------------------------
// C++ launchers
// ---------------------------------------------------------------------------

void slot_update_forward(
    torch::Tensor& slot_keys,
    torch::Tensor& slot_values,
    const torch::Tensor& token_keys,
    const torch::Tensor& token_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    double alpha) {

    TORCH_CHECK(token_keys.device().is_cuda(), "token_keys must be CUDA");
    TORCH_CHECK(slot_keys.device().is_cuda(), "slot_keys must be CUDA");

    const auto B = token_keys.size(0);
    const auto H = token_keys.size(1);
    const auto N = token_keys.size(2);
    const auto D = token_keys.size(3);
    const auto M = slot_keys.size(2);

    const float a = static_cast<float>(alpha);
    constexpr int THREADS = 128;

    dim3 grid(M, B * H);
    dim3 block(THREADS);

    const at::cuda::OptionalCUDAGuard device_guard(token_keys.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    slot_update_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const at::Half*>(token_keys.data_ptr()),
        reinterpret_cast<const at::Half*>(token_values.data_ptr()),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<at::Half*>(slot_keys.data_ptr()),
        reinterpret_cast<at::Half*>(slot_values.data_ptr()),
        B, H, N, D, M, a);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
slot_update_backward(
    const torch::Tensor& grad_new_sk,
    const torch::Tensor& grad_new_sv,
    const torch::Tensor& token_keys,
    const torch::Tensor& token_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    double alpha) {

    TORCH_CHECK(token_keys.device().is_cuda(), "token_keys must be CUDA");

    const auto B = token_keys.size(0);
    const auto H = token_keys.size(1);
    const auto N = token_keys.size(2);
    const auto D = token_keys.size(3);
    const auto M = grad_new_sk.size(2);

    const float a = static_cast<float>(alpha);
    constexpr int THREADS_SLOT = 128;
    constexpr int THREADS_TOK  = 256;

    auto grad_old_sk = torch::empty_like(grad_new_sk);
    auto grad_old_sv = torch::empty_like(grad_new_sv);
    auto grad_tk = torch::empty_like(token_keys);
    auto grad_tv = torch::empty_like(token_values);
    auto grad_w  = torch::empty_like(weights);

    auto temp_dac_k  = torch::empty({B, H, M, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(token_keys.device()));
    auto temp_dac_v  = torch::empty({B, H, M, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(token_keys.device()));
    auto temp_dw_sum = torch::empty({B, H, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(token_keys.device()));

    dim3 grid1(M, B * H);
    dim3 block1(THREADS_SLOT);
    dim3 grid2((N + THREADS_TOK - 1) / THREADS_TOK, B * H);
    dim3 block2(THREADS_TOK);

    const at::cuda::OptionalCUDAGuard device_guard(token_keys.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    slot_update_bwd_prep_kernel<<<grid1, block1, 0, stream>>>(
        reinterpret_cast<const at::Half*>(token_keys.data_ptr()),
        reinterpret_cast<const at::Half*>(token_values.data_ptr()),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<const at::Half*>(grad_new_sk.data_ptr()),
        reinterpret_cast<const at::Half*>(grad_new_sv.data_ptr()),
        reinterpret_cast<at::Half*>(grad_old_sk.data_ptr()),
        reinterpret_cast<at::Half*>(grad_old_sv.data_ptr()),
        temp_dac_k.data_ptr<float>(),
        temp_dac_v.data_ptr<float>(),
        temp_dw_sum.data_ptr<float>(),
        B, H, N, D, M, a);

    slot_update_bwd_tokens_kernel<<<grid2, block2, 0, stream>>>(
        reinterpret_cast<const at::Half*>(token_keys.data_ptr()),
        reinterpret_cast<const at::Half*>(token_values.data_ptr()),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        temp_dac_k.data_ptr<float>(),
        temp_dac_v.data_ptr<float>(),
        temp_dw_sum.data_ptr<float>(),
        reinterpret_cast<at::Half*>(grad_tk.data_ptr()),
        reinterpret_cast<at::Half*>(grad_tv.data_ptr()),
        grad_w.data_ptr<float>(),
        B, H, N, D, M);

    return std::make_tuple(grad_tk, grad_tv, grad_old_sk, grad_old_sv, grad_w);
}
