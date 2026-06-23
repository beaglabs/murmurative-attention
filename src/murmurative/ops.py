import torch
from typing import Callable, Optional, Tuple

from .reference import (
    slot_select_reference,
    slot_attend_reference,
    slot_update_reference,
    slot_diffusion_reference,
    slot_murmurate_reference,
)

_has_cuda_kernels = None


def _check_cuda_kernels() -> bool:
    """Ensure the prebuilt CUDA extension is loaded and available.

    Raises RuntimeError if the extension is not built or CUDA is unavailable.
    """
    global _has_cuda_kernels
    if _has_cuda_kernels is not None:
        return _has_cuda_kernels
    from ._cuda_kernels import ensure_compiled
    ensure_compiled()
    _has_cuda_kernels = True
    return _has_cuda_kernels


def _get_cuda_lib():
    _check_cuda_kernels()
    try:
        return torch.ops.murmurative_attention
    except (AttributeError, RuntimeError) as e:
        raise ImportError(
            f"CUDA kernel namespace 'murmurative_attention' not available. "
            f"Ensure CUDA kernels are compiled. "
            f"Use the reference implementation by importing from murmurative.reference."
        ) from e


def _ensure_cuda_dtype(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.float16:
        raise TypeError(
            f"CUDA kernels require float16 (torch.half) tensors, but {name} "
            f"has dtype {tensor.dtype}. Pass use_reference=True for other dtypes, "
            f"or convert the tensor to float16."
        )
    return tensor


# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------

class _SlotSelectAttend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, slot_keys, slot_values, mask, causal, position_bias, scale,
                effective_M):
        lib = _get_cuda_lib()
        output, weights, indices = lib.slot_select_attend(
            query.contiguous(),
            slot_keys.contiguous(),
            slot_values.contiguous(),
            mask.contiguous() if mask is not None else None,
            causal,
            position_bias.contiguous() if position_bias is not None else None,
            scale,
            effective_M,
        )
        ctx.save_for_backward(query, slot_keys, slot_values, indices, weights)
        ctx.scale = scale
        return output, weights, indices

    @staticmethod
    def backward(ctx, grad_output, grad_weights, grad_indices):
        query, slot_keys, slot_values, indices, weights = ctx.saved_tensors
        lib = _get_cuda_lib()
        if grad_output is None:
            grad_output = torch.zeros_like(query)
        if grad_weights is None:
            grad_weights = torch.zeros_like(weights)
        grad_query, grad_slot_keys, grad_slot_values = lib.slot_select_attend_backward(
            grad_output.contiguous(),
            grad_weights.contiguous(),
            query,
            slot_keys,
            slot_values,
            indices,
            weights,
            ctx.scale,
        )
        return grad_query, grad_slot_keys, grad_slot_values, None, None, None, None, None


class _SlotUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_keys, token_values, slot_keys, slot_values, indices, weights, alpha):
        lib = _get_cuda_lib()
        if not slot_keys.is_contiguous():
            slot_keys = slot_keys.contiguous()
        if not slot_values.is_contiguous():
            slot_values = slot_values.contiguous()
        ctx.save_for_backward(token_keys, token_values, indices, weights)
        ctx.alpha = alpha
        ctx.mark_dirty(slot_keys, slot_values)
        lib.slot_update(
            slot_keys,
            slot_values,
            token_keys.contiguous(),
            token_values.contiguous(),
            indices.contiguous(),
            weights.contiguous(),
            alpha,
        )
        return slot_keys, slot_values

    @staticmethod
    def backward(ctx, grad_new_sk, grad_new_sv):
        token_keys, token_values, indices, weights = ctx.saved_tensors
        lib = _get_cuda_lib()
        grad_tk, grad_tv, grad_old_sk, grad_old_sv, grad_w = lib.slot_update_backward(
            grad_new_sk.contiguous(),
            grad_new_sv.contiguous(),
            token_keys,
            token_values,
            indices,
            weights,
            ctx.alpha,
        )
        return grad_tk, grad_tv, grad_old_sk, grad_old_sv, None, grad_w, None


class _SlotDiffusion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slot_keys, slot_values, gamma):
        lib = _get_cuda_lib()
        if not slot_keys.is_contiguous():
            slot_keys = slot_keys.contiguous()
        if not slot_values.is_contiguous():
            slot_values = slot_values.contiguous()
        ctx.gamma = gamma
        ctx.mark_dirty(slot_keys, slot_values)
        lib.slot_diffusion(slot_keys, slot_values, gamma)
        return slot_keys, slot_values

    @staticmethod
    def backward(ctx, grad_new_sk, grad_new_sv):
        lib = _get_cuda_lib()
        grad_old_sk, grad_old_sv = lib.slot_diffusion_backward(
            grad_new_sk.contiguous(),
            grad_new_sv.contiguous(),
            ctx.gamma,
        )
        return grad_old_sk, grad_old_sv, None


class _SlotUpdateDiffuse(torch.autograd.Function):
    """Fused slot update + diffusion in a single CUDA kernel launch.

    Forward:  EMA update then tridiagonal diffusion, writing to global memory once.
    Backward: chains diffusion_backward then update_backward (two CUDA launches).
    """
    @staticmethod
    def forward(ctx, token_keys, token_values, slot_keys, slot_values,
                indices, weights, alpha, gamma):
        lib = _get_cuda_lib()
        if not slot_keys.is_contiguous():
            slot_keys = slot_keys.contiguous()
        if not slot_values.is_contiguous():
            slot_values = slot_values.contiguous()
        ctx.save_for_backward(token_keys, token_values, indices, weights)
        ctx.alpha = alpha
        ctx.gamma = gamma
        ctx.mark_dirty(slot_keys, slot_values)
        lib.slot_update_diffuse(
            slot_keys,
            slot_values,
            token_keys.contiguous(),
            token_values.contiguous(),
            indices.contiguous(),
            weights.contiguous(),
            alpha,
            gamma,
        )
        return slot_keys, slot_values

    @staticmethod
    def backward(ctx, grad_new_sk, grad_new_sv):
        token_keys, token_values, indices, weights = ctx.saved_tensors
        lib = _get_cuda_lib()

        grad_pre_diffuse_sk, grad_pre_diffuse_sv = lib.slot_diffusion_backward(
            grad_new_sk.contiguous(),
            grad_new_sv.contiguous(),
            ctx.gamma,
        )

        grad_tk, grad_tv, grad_old_sk, grad_old_sv, grad_w = lib.slot_update_backward(
            grad_pre_diffuse_sk.contiguous(),
            grad_pre_diffuse_sv.contiguous(),
            token_keys,
            token_values,
            indices,
            weights,
            ctx.alpha,
        )

        return grad_tk, grad_tv, grad_old_sk, grad_old_sv, None, grad_w, None, None


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def slot_select(
    query: torch.Tensor,
    slot_keys: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    use_reference: bool = False,
) -> torch.Tensor:
    """Select top-7 slots for each query token.

    This always delegates to the reference implementation because the fused
    CUDA kernel (slot_select_attend) covers both selection and attention.
    """
    return slot_select_reference(query, slot_keys, mask, causal, position_bias)


def slot_attend(
    query: torch.Tensor,
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    indices: torch.Tensor,
    scale: Optional[float] = None,
    use_reference: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attend to selected slots.

    This always delegates to the reference implementation because the fused
    CUDA kernel (slot_select_attend) covers both selection and attention.
    """
    return slot_attend_reference(query, slot_keys, slot_values, indices, scale)


def slot_select_attend(
    query: torch.Tensor,
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    effective_M: int = 0,
    use_reference: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused slot selection + attention (CUDA) or reference implementation."""
    M = slot_keys.shape[2]
    em = effective_M if effective_M > 0 else M
    em = min(em, M)

    if use_reference:
        indices = slot_select_reference(
            query, slot_keys, mask=mask, causal=causal,
            position_bias=position_bias, effective_M=em if effective_M > 0 else None,
        )
        output, weights = slot_attend_reference(query, slot_keys, slot_values, indices, scale)
        return output, weights, indices
    _check_cuda_kernels()
    _ensure_cuda_dtype(query, "query")
    _ensure_cuda_dtype(slot_keys, "slot_keys")
    _ensure_cuda_dtype(slot_values, "slot_values")
    if position_bias is not None:
        _ensure_cuda_dtype(position_bias, "position_bias")
    return _SlotSelectAttend.apply(
        query, slot_keys, slot_values, mask, causal, position_bias, scale, em
    )


def slot_update(
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    alpha: float = 0.9,
    use_reference: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_reference:
        return slot_update_reference(
            token_keys, token_values, slot_keys, slot_values, indices, weights, alpha
        )
    _check_cuda_kernels()
    _ensure_cuda_dtype(token_keys, "token_keys")
    _ensure_cuda_dtype(token_values, "token_values")
    _ensure_cuda_dtype(slot_keys, "slot_keys")
    _ensure_cuda_dtype(slot_values, "slot_values")
    # _SlotUpdate mutates slot_keys/slot_values in-place via mark_dirty.
    # PyTorch forbids in-place ops on leaf Variables that require grad,
    # so we silently clone them when necessary.
    if slot_keys.requires_grad and slot_keys.is_leaf:
        slot_keys = slot_keys.clone()
    if slot_values.requires_grad and slot_values.is_leaf:
        slot_values = slot_values.clone()
    return _SlotUpdate.apply(
        token_keys, token_values, slot_keys, slot_values, indices, weights, alpha
    )


def slot_diffusion(
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    gamma: float = 0.1,
    use_reference: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_reference:
        return slot_diffusion_reference(slot_keys, slot_values, gamma)
    _check_cuda_kernels()
    _ensure_cuda_dtype(slot_keys, "slot_keys")
    _ensure_cuda_dtype(slot_values, "slot_values")
    if slot_keys.requires_grad and slot_keys.is_leaf:
        slot_keys = slot_keys.clone()
    if slot_values.requires_grad and slot_values.is_leaf:
        slot_values = slot_values.clone()
    return _SlotDiffusion.apply(slot_keys, slot_values, gamma)


def slot_murmurate(
    x: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    slot_k_emb: torch.Tensor,
    slot_v_emb: torch.Tensor,
    num_heads: int = 8,
    rounds: int = 3,
    alpha: float = 0.9,
    gamma: float = 0.15,
    causal: bool = False,
    mask: Optional[torch.Tensor] = None,
    position_bias_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_reference: bool = False,
    use_dynamic_m: bool = True,
    slot_ratio: int = 8,
    use_fused_update_diffuse: bool = True,
) -> torch.Tensor:
    if not use_reference:
        _check_cuda_kernels()

    B, N, D = x.shape
    H = num_heads
    if D % H != 0:
        raise ValueError(
            f"Embedding dimension D ({D}) must be divisible by num_heads ({H})"
        )
    Dh = D // H
    M = slot_k_emb.shape[1]

    effective_M = M
    if use_dynamic_m:
        M_min = 32
        effective_M = min(M, max(M_min, (N + slot_ratio - 1) // slot_ratio))

    x_h = x.view(B, N, H, Dh).permute(0, 2, 1, 3)
    q_h = q_proj.view(B, N, H, Dh).permute(0, 2, 1, 3)
    k_h = k_proj.view(B, N, H, Dh).permute(0, 2, 1, 3)
    v_h = v_proj.view(B, N, H, Dh).permute(0, 2, 1, 3)

    slot_k = slot_k_emb.unsqueeze(0).expand(B, H, M, Dh).clone()
    slot_v = slot_v_emb.unsqueeze(0).expand(B, H, M, Dh).clone()

    mask_cont = mask.contiguous() if mask is not None else None

    for r in range(rounds):
        pos_bias = None
        if position_bias_fn is not None:
            pos_bias = position_bias_fn(x_h)

        if use_reference:
            indices = slot_select_reference(
                q_h, slot_k, mask=mask, causal=causal,
                position_bias=pos_bias,
                effective_M=effective_M if use_dynamic_m else None,
            )
            attn_out, attn_w = slot_attend_reference(q_h, slot_k, slot_v, indices)
        else:
            attn_out, attn_w, indices = _SlotSelectAttend.apply(
                q_h,
                slot_k,
                slot_v,
                mask_cont,
                causal,
                pos_bias.contiguous() if pos_bias is not None else None,
                None,
                effective_M,
            )
        x_h = x_h + attn_out
        if use_reference:
            slot_k, slot_v = slot_update_reference(
                k_h, v_h, slot_k, slot_v, indices, attn_w, alpha
            )
            slot_k, slot_v = slot_diffusion_reference(slot_k, slot_v, gamma)
        elif use_fused_update_diffuse:
            slot_k, slot_v = _SlotUpdateDiffuse.apply(
                k_h, v_h,
                slot_k.clone(),
                slot_v.clone(),
                indices, attn_w, alpha, gamma
            )
        else:
            slot_k, slot_v = _SlotUpdate.apply(
                k_h, v_h,
                slot_k.clone(),
                slot_v.clone(),
                indices, attn_w, alpha
            )
            slot_k, slot_v = _SlotDiffusion.apply(slot_k, slot_v, gamma)

    output = x_h.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    return output
