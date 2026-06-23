#include <torch/all.h>
#include <optional>
#include <tuple>

#include "cuda_ops.h"
#include "torch_binding.h"

// ---- Fused select + attend wrappers ----

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slot_select_attend(
    const torch::Tensor &query,
    const torch::Tensor &slot_keys,
    const torch::Tensor &slot_values,
    const std::optional<torch::Tensor> &mask,
    bool causal,
    const std::optional<torch::Tensor> &position_bias,
    std::optional<double> scale,
    int64_t effective_M) {

    return slot_select_attend_forward(
        query, slot_keys, slot_values, mask, causal, position_bias, scale, effective_M);
}

// ---- Update wrappers ----

void slot_update(
    torch::Tensor &slot_keys,
    torch::Tensor &slot_values,
    const torch::Tensor &token_keys,
    const torch::Tensor &token_values,
    const torch::Tensor &indices,
    const torch::Tensor &weights,
    double alpha) {

    slot_update_forward(slot_keys, slot_values, token_keys, token_values, indices, weights, alpha);
}

// ---- Diffusion wrappers ----

void slot_diffusion(
    torch::Tensor &slot_keys,
    torch::Tensor &slot_values,
    double gamma) {

    slot_diffusion_forward(slot_keys, slot_values, gamma);
}

// ---- Fused update + diffusion wrapper ----

void slot_update_diffuse(
    torch::Tensor &slot_keys,
    torch::Tensor &slot_values,
    const torch::Tensor &token_keys,
    const torch::Tensor &token_values,
    const torch::Tensor &indices,
    const torch::Tensor &weights,
    double alpha,
    double gamma) {

    slot_update_diffuse_forward(
        slot_keys, slot_values, token_keys, token_values, indices, weights, alpha, gamma);
}
