#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cmath>

constexpr int K_NEIGHBORS = 7;

// ---------------------------------------------------------------------------
// Fused forward: slot update + tridiagonal diffusion in one launch
//
// Each block owns one slot. It loads neighbor slots at the start (pre-update
// snapshot), scans tokens to compute the EMA accumulator, applies the EMA,
// then applies the tridiagonal diffusion stencil in-place before writing
// the final result back to global memory. This eliminates one intermediate
// global memory round-trip (write EMA-updated slots, read back for diffuse).
// ---------------------------------------------------------------------------
__global__ void slot_update_diffuse_fwd_kernel(
    const at::Half* __restrict__ token_keys,
    const at::Half* __restrict__ token_values,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    at::Half* __restrict__ slot_keys,
    at::Half* __restrict__ slot_values,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t M,
    float alpha, float gamma) {

    const int64_t bh = blockIdx.y;
    const int64_t m  = blockIdx.x;
    const int tid = threadIdx.x;

    if (m >= M) return;

    const int64_t sk_base = bh * M * D;

    // Snapshot neighbor slot values for diffusion (pre-update).
    // Jacobi-style: use original slot values, not concurrently updated ones.
    float left_k[128], left_v[128];
    float right_k[128], right_v[128];
    float my_old_k[128], my_old_v[128];

    for (int d = tid; d < D; d += blockDim.x) {
        my_old_k[d] = __half2float(slot_keys[sk_base + m * D + d]);
        my_old_v[d] = __half2float(slot_values[sk_base + m * D + d]);
    }

    if (m > 0) {
        for (int d = tid; d < D; d += blockDim.x) {
            left_k[d] = __half2float(slot_keys[sk_base + (m - 1) * D + d]);
            left_v[d] = __half2float(slot_values[sk_base + (m - 1) * D + d]);
        }
    }
    if (m < M - 1) {
        for (int d = tid; d < D; d += blockDim.x) {
            right_k[d] = __half2float(slot_keys[sk_base + (m + 1) * D + d]);
            right_v[d] = __half2float(slot_values[sk_base + (m + 1) * D + d]);
        }
    }

    // --- EMA accumulator scan ---
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

    // --- EMA update + diffusion ---
    __shared__ float sh_w_sum;
    if (tid == 0) sh_w_sum = w_sum;
    __syncthreads();
    float w_sum_val = sh_w_sum;

    const float one_minus_alpha = 1.0f - alpha;

    for (int d = tid; d < D; d += blockDim.x) {
        float new_k, new_v;

        if (w_sum_val > 0.0f) {
            new_k = alpha * my_old_k[d] + one_minus_alpha * k_acc[d] / w_sum_val;
            new_v = alpha * my_old_v[d] + one_minus_alpha * v_acc[d] / w_sum_val;
        } else {
            new_k = my_old_k[d];
            new_v = my_old_v[d];
        }

        // Apply tridiagonal diffusion (skip boundaries)
        if (m > 0 && m < M - 1) {
            new_k += gamma * (left_k[d] + right_k[d] - 2.0f * new_k);
            new_v += gamma * (left_v[d] + right_v[d] - 2.0f * new_v);
        }

        slot_keys[sk_base + m * D + d] = __float2half(new_k);
        slot_values[sk_base + m * D + d] = __float2half(new_v);
    }
}

// ---------------------------------------------------------------------------
// C++ launcher
// ---------------------------------------------------------------------------

void slot_update_diffuse_forward(
    torch::Tensor& slot_keys,
    torch::Tensor& slot_values,
    const torch::Tensor& token_keys,
    const torch::Tensor& token_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    double alpha,
    double gamma) {

    TORCH_CHECK(token_keys.device().is_cuda(), "token_keys must be CUDA");
    TORCH_CHECK(slot_keys.device().is_cuda(), "slot_keys must be CUDA");

    const auto B = token_keys.size(0);
    const auto H = token_keys.size(1);
    const auto N = token_keys.size(2);
    const auto D = token_keys.size(3);
    const auto M = slot_keys.size(2);

    const float a = static_cast<float>(alpha);
    const float g = static_cast<float>(gamma);
    constexpr int THREADS = 128;

    dim3 grid(M, B * H);
    dim3 block(THREADS);

    const at::cuda::OptionalCUDAGuard device_guard(token_keys.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    slot_update_diffuse_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const at::Half*>(token_keys.data_ptr()),
        reinterpret_cast<const at::Half*>(token_values.data_ptr()),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<at::Half*>(slot_keys.data_ptr()),
        reinterpret_cast<at::Half*>(slot_values.data_ptr()),
        B, H, N, D, M, a, g);
}