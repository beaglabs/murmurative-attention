#pragma once

#include <torch/torch.h>
#include <optional>

#include "cuda_ops.h"

// ---- Fused select + attend ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slot_select_attend(
    const torch::Tensor &query,
    const torch::Tensor &slot_keys,
    const torch::Tensor &slot_values,
    const std::optional<torch::Tensor> &mask,
    bool causal,
    const std::optional<torch::Tensor> &position_bias,
    std::optional<double> scale,
    int64_t effective_M);

// ---- Update ----
void slot_update(
    torch::Tensor &slot_keys,
    torch::Tensor &slot_values,
    const torch::Tensor &token_keys,
    const torch::Tensor &token_values,
    const torch::Tensor &indices,
    const torch::Tensor &weights,
    double alpha);

// ---- Diffusion ----
void slot_diffusion(
    torch::Tensor &slot_keys,
    torch::Tensor &slot_values,
    double gamma);

// ---- Fused update + diffusion ----
void slot_update_diffuse(
    torch::Tensor &slot_keys,
    torch::Tensor &slot_values,
    const torch::Tensor &token_keys,
    const torch::Tensor &token_values,
    const torch::Tensor &indices,
    const torch::Tensor &weights,
    double alpha,
    double gamma);
