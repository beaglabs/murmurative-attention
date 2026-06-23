#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cmath>

// ---------------------------------------------------------------------------
// Forward diffusion (in-place)
// ---------------------------------------------------------------------------
__global__ void slot_diffusion_fwd_kernel(
    at::Half* __restrict__ slot_keys,
    at::Half* __restrict__ slot_values,
    int64_t B, int64_t H, int64_t M, int64_t D, float gamma) {

    const int64_t bh = blockIdx.x;
    const int64_t m  = threadIdx.x;

    if (m >= M) return;

    const int64_t base = bh * M * D;

    // Boundaries are not diffused
    if (m == 0 || m == M - 1) return;

    // Jacobi-style: read all old neighbour values into per-thread
    // buffers before writing, so results are deterministic regardless
    // of warp scheduling order.
    float new_k[128];
    float new_v[128];

    for (int d = 0; d < D; ++d) {
        float left_k  = __half2float(slot_keys[base + (m - 1) * D + d]);
        float cent_k  = __half2float(slot_keys[base + m * D + d]);
        float right_k = __half2float(slot_keys[base + (m + 1) * D + d]);
        float d2_k = left_k + right_k - 2.0f * cent_k;
        new_k[d] = cent_k + gamma * d2_k;

        float left_v  = __half2float(slot_values[base + (m - 1) * D + d]);
        float cent_v  = __half2float(slot_values[base + m * D + d]);
        float right_v = __half2float(slot_values[base + (m + 1) * D + d]);
        float d2_v = left_v + right_v - 2.0f * cent_v;
        new_v[d] = cent_v + gamma * d2_v;
    }

    __syncthreads();

    for (int d = 0; d < D; ++d) {
        slot_keys[base + m * D + d] = __float2half(new_k[d]);
        slot_values[base + m * D + d] = __float2half(new_v[d]);
    }
}

// ---------------------------------------------------------------------------
// Backward diffusion
//
// Forward:  y[0] = x[0]
//           y[m] = (1-2γ)*x[m] + γ*x[m-1] + γ*x[m+1]   for 1<=m<=M-2
//           y[M-1] = x[M-1]
//
// Backward: dx[0] = dy[0] + γ*dy[1]
//           dx[m] = γ*dy[m-1] + (1-2γ)*dy[m] + γ*dy[m+1]   for 1<=m<=M-2
//           dx[M-1] = dy[M-1] + γ*dy[M-2]
// ---------------------------------------------------------------------------
__global__ void slot_diffusion_bwd_kernel(
    const at::Half* __restrict__ grad_new_keys,
    const at::Half* __restrict__ grad_new_values,
    at::Half* __restrict__ grad_old_keys,
    at::Half* __restrict__ grad_old_values,
    int64_t B, int64_t H, int64_t M, int64_t D, float gamma) {

    const int64_t bh = blockIdx.x;
    const int64_t m  = threadIdx.x;

    if (m >= M) return;

    const int64_t base = bh * M * D;

    for (int d = 0; d < D; ++d) {
        float dyk = __half2float(grad_new_keys[base + m * D + d]);
        float dyv = __half2float(grad_new_values[base + m * D + d]);
        float dxk, dxv;

        if (m == 0) {
            float dyk1 = __half2float(grad_new_keys[base + 1 * D + d]);
            float dyv1 = __half2float(grad_new_values[base + 1 * D + d]);
            dxk = dyk + gamma * dyk1;
            dxv = dyv + gamma * dyv1;
        } else if (m == M - 1) {
            float dykM2 = __half2float(grad_new_keys[base + (M - 2) * D + d]);
            float dyvM2 = __half2float(grad_new_values[base + (M - 2) * D + d]);
            dxk = dyk + gamma * dykM2;
            dxv = dyv + gamma * dyvM2;
        } else if (m == 1) {
            float dyk_right = __half2float(grad_new_keys[base + (m + 1) * D + d]);
            float dyv_right = __half2float(grad_new_values[base + (m + 1) * D + d]);
            dxk = (1.0f - 2.0f * gamma) * dyk + gamma * dyk_right;
            dxv = (1.0f - 2.0f * gamma) * dyv + gamma * dyv_right;
        } else if (m == M - 2) {
            float dyk_left  = __half2float(grad_new_keys[base + (m - 1) * D + d]);
            float dyv_left  = __half2float(grad_new_values[base + (m - 1) * D + d]);
            dxk = gamma * dyk_left + (1.0f - 2.0f * gamma) * dyk;
            dxv = gamma * dyv_left + (1.0f - 2.0f * gamma) * dyv;
        } else {
            float dyk_left  = __half2float(grad_new_keys[base + (m - 1) * D + d]);
            float dyk_right = __half2float(grad_new_keys[base + (m + 1) * D + d]);
            float dyv_left  = __half2float(grad_new_values[base + (m - 1) * D + d]);
            float dyv_right = __half2float(grad_new_values[base + (m + 1) * D + d]);
            dxk = gamma * dyk_left + (1.0f - 2.0f * gamma) * dyk + gamma * dyk_right;
            dxv = gamma * dyv_left + (1.0f - 2.0f * gamma) * dyv + gamma * dyv_right;
        }

        grad_old_keys[base + m * D + d] = __float2half(dxk);
        grad_old_values[base + m * D + d] = __float2half(dxv);
    }
}

// ---------------------------------------------------------------------------
// C++ launchers
// ---------------------------------------------------------------------------

void slot_diffusion_forward(
    torch::Tensor& slot_keys,
    torch::Tensor& slot_values,
    double gamma) {

    TORCH_CHECK(slot_keys.device().is_cuda(), "slot_keys must be CUDA");

    const auto B = slot_keys.size(0);
    const auto H = slot_keys.size(1);
    const auto M = slot_keys.size(2);
    const auto D = slot_keys.size(3);
    const float g = static_cast<float>(gamma);

    dim3 grid(B * H);
    dim3 block(M);

    const at::cuda::OptionalCUDAGuard device_guard(slot_keys.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    slot_diffusion_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<at::Half*>(slot_keys.data_ptr()),
        reinterpret_cast<at::Half*>(slot_values.data_ptr()),
        B, H, M, D, g);
}

std::tuple<torch::Tensor, torch::Tensor>
slot_diffusion_backward(
    const torch::Tensor& grad_new_keys,
    const torch::Tensor& grad_new_values,
    double gamma) {

    TORCH_CHECK(grad_new_keys.device().is_cuda(), "grad_new_keys must be CUDA");

    const auto B = grad_new_keys.size(0);
    const auto H = grad_new_keys.size(1);
    const auto M = grad_new_keys.size(2);
    const auto D = grad_new_keys.size(3);
    const float g = static_cast<float>(gamma);

    auto grad_old_keys = torch::empty_like(grad_new_keys);
    auto grad_old_values = torch::empty_like(grad_new_values);

    dim3 grid(B * H);
    dim3 block(M);

    const at::cuda::OptionalCUDAGuard device_guard(grad_new_keys.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    slot_diffusion_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const at::Half*>(grad_new_keys.data_ptr()),
        reinterpret_cast<const at::Half*>(grad_new_values.data_ptr()),
        reinterpret_cast<at::Half*>(grad_old_keys.data_ptr()),
        reinterpret_cast<at::Half*>(grad_old_values.data_ptr()),
        B, H, M, D, g);

    return std::make_tuple(grad_old_keys, grad_old_values);
}
