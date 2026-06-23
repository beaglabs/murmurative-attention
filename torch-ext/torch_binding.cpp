#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY(murmurative_attention, ops) {
  ops.def(
      "slot_select_attend(Tensor query, Tensor slot_keys, Tensor slot_values, "
      "Tensor? mask, bool causal, Tensor? position_bias, float? scale, int effective_M) -> "
      "(Tensor, Tensor, Tensor)");
  ops.def(
      "slot_select_attend_backward(Tensor grad_output, Tensor grad_weights, "
      "Tensor query, Tensor slot_keys, Tensor slot_values, Tensor indices, "
      "Tensor weights, float? scale) -> (Tensor, Tensor, Tensor)");
  ops.def(
      "slot_update(Tensor(a!) slot_keys, Tensor(b!) slot_values, "
      "Tensor token_keys, Tensor token_values, Tensor indices, "
      "Tensor weights, float alpha) -> ()");
  ops.def(
      "slot_update_backward(Tensor grad_new_sk, Tensor grad_new_sv, "
      "Tensor token_keys, Tensor token_values, Tensor indices, "
      "Tensor weights, float alpha) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  ops.def(
      "slot_diffusion(Tensor(a!) slot_keys, Tensor(b!) slot_values, "
      "float gamma) -> ()");
  ops.def(
      "slot_diffusion_backward(Tensor grad_new_keys, Tensor grad_new_values, "
      "float gamma) -> (Tensor, Tensor)");
  ops.def(
      "slot_update_diffuse(Tensor(a!) slot_keys, Tensor(b!) slot_values, "
      "Tensor token_keys, Tensor token_values, Tensor indices, "
      "Tensor weights, float alpha, float gamma) -> ()");
}

TORCH_LIBRARY_IMPL(murmurative_attention, CUDA, m) {
  m.impl("slot_select_attend", &slot_select_attend);
  m.impl("slot_select_attend_backward", &slot_select_attend_backward);
  m.impl("slot_update", &slot_update);
  m.impl("slot_update_backward", &slot_update_backward);
  m.impl("slot_diffusion", &slot_diffusion);
  m.impl("slot_diffusion_backward", &slot_diffusion_backward);
  m.impl("slot_update_diffuse", &slot_update_diffuse);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
