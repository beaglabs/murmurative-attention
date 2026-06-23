#pragma once

#include <torch/torch.h>
#include <optional>

// ---- Fused select + attend ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slot_select_attend_forward(
    const torch::Tensor& query,
    const torch::Tensor& slot_keys,
    const torch::Tensor& slot_values,
    const c10::optional<torch::Tensor>& mask,
    bool causal,
    const c10::optional<torch::Tensor>& position_bias,
    c10::optional<double> scale,
    int64_t effective_M);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slot_select_attend_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_weights,
    const torch::Tensor& query,
    const torch::Tensor& slot_keys,
    const torch::Tensor& slot_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    c10::optional<double> scale);

// ---- Update ----
void slot_update_forward(
    torch::Tensor& slot_keys,
    torch::Tensor& slot_values,
    const torch::Tensor& token_keys,
    const torch::Tensor& token_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    double alpha);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
slot_update_backward(
    const torch::Tensor& grad_new_sk,
    const torch::Tensor& grad_new_sv,
    const torch::Tensor& token_keys,
    const torch::Tensor& token_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    double alpha);

// ---- Diffusion ----
void slot_diffusion_forward(
    torch::Tensor& slot_keys,
    torch::Tensor& slot_values,
    double gamma);

std::tuple<torch::Tensor, torch::Tensor>
slot_diffusion_backward(
    const torch::Tensor& grad_new_keys,
    const torch::Tensor& grad_new_values,
    double gamma);

// ---- Fused update + diffusion ----
void slot_update_diffuse_forward(
    torch::Tensor& slot_keys,
    torch::Tensor& slot_values,
    const torch::Tensor& token_keys,
    const torch::Tensor& token_values,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    double alpha,
    double gamma);
