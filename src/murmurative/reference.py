import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple


def slot_select_reference(
    query: torch.Tensor,
    slot_keys: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    effective_M: Optional[int] = None,
) -> torch.Tensor:
    B, H, N, D = query.shape
    M = slot_keys.shape[2]
    em = effective_M if effective_M is not None else M
    em = min(em, M)

    scores = torch.einsum("bhnd,bhmd->bhnm", query, slot_keys)

    if position_bias is not None:
        scores = scores + position_bias

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    if causal:
        idx = torch.arange(M, device=query.device).view(1, 1, 1, M)
        seq_idx = torch.arange(N, device=query.device).view(1, 1, N, 1)
        causal_mask = idx > seq_idx
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Mask inactive slots: gradient flows through active slots only
    if em < M:
        tail_mask = torch.arange(M, device=query.device).view(1, 1, 1, M) >= em
        scores = scores.masked_fill(tail_mask, float("-inf"))

    _, indices = torch.topk(scores, k=7, dim=-1, sorted=True)
    return indices.to(torch.int64)


def slot_attend_reference(
    query: torch.Tensor,
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    indices: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = query.shape
    M = slot_keys.shape[2]
    K = indices.shape[-1]

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    indices_safe = indices.clamp(0, M - 1)  # [B, H, N, K]

    idx_flat = indices_safe.view(B, H, N * K, 1).expand(-1, -1, -1, D)
    gathered_k = torch.gather(slot_keys, 2, idx_flat).view(B, H, N, K, D)

    attn_scores = torch.einsum("bhnd,bhnkd->bhnk", query, gathered_k) * scale
    del gathered_k  # free memory before gathering values
    attn_weights = F.softmax(attn_scores, dim=-1)

    gathered_v = torch.gather(slot_values, 2, idx_flat).view(B, H, N, K, D)
    output = torch.einsum("bhnk,bhnkd->bhnd", attn_weights, gathered_v)

    return output, attn_weights.to(torch.float32)


def slot_update_reference(
    token_keys: torch.Tensor,
    token_values: torch.Tensor,
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    alpha: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = token_keys.shape
    M = slot_keys.shape[2]
    K = indices.shape[-1]

    indices_safe = indices.clamp(0, M - 1)  # [B, H, N, K]

    # Accumulate updates in float32 for numerical stability (slots are tiny).
    k_sum = torch.zeros(B, H, M, D, device=token_keys.device, dtype=torch.float32)
    v_sum = torch.zeros(B, H, M, D, device=token_keys.device, dtype=torch.float32)
    w_sum = torch.zeros(B, H, M, 1, device=token_keys.device, dtype=torch.float32)

    # Process each of the K selected slots separately to avoid allocating the
    # massive [B,H,N,K,D] temporary that the batched multiply would create.
    for k in range(K):
        idx_k = indices_safe[..., k]  # [B, H, N]
        w_k = weights[..., k].unsqueeze(-1).float()  # [B, H, N, 1]

        tk_w = token_keys.float() * w_k  # [B, H, N, D]
        tv_w = token_values.float() * w_k

        idx_exp = idx_k.unsqueeze(-1)  # [B, H, N, 1]
        k_sum.scatter_add_(2, idx_exp.expand(-1, -1, -1, D), tk_w)
        v_sum.scatter_add_(2, idx_exp.expand(-1, -1, -1, D), tv_w)
        w_sum.scatter_add_(2, idx_exp, w_k)

    valid = w_sum.squeeze(-1) > 0  # [B, H, M]
    k_update = k_sum / (w_sum + 1e-8)
    v_update = v_sum / (w_sum + 1e-8)

    mask = valid.unsqueeze(-1)  # [B, H, M, 1]
    sk_f = slot_keys.float()
    sv_f = slot_values.float()
    new_sk = torch.where(mask, alpha * sk_f + (1.0 - alpha) * k_update, sk_f)
    new_sv = torch.where(mask, alpha * sv_f + (1.0 - alpha) * v_update, sv_f)

    return new_sk.type_as(slot_keys), new_sv.type_as(slot_values)


def slot_diffusion_reference(
    slot_keys: torch.Tensor,
    slot_values: torch.Tensor,
    gamma: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M = slot_keys.shape[2]

    sk = slot_keys.clone()
    sv = slot_values.clone()

    sk[:, :, 1:M - 1] += gamma * (
        sk[:, :, 0:M - 2] + sk[:, :, 2:M] - 2 * sk[:, :, 1:M - 1]
    )
    sv[:, :, 1:M - 1] += gamma * (
        sv[:, :, 0:M - 2] + sv[:, :, 2:M] - 2 * sv[:, :, 1:M - 1]
    )

    return sk, sv


def slot_murmurate_reference(
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
    use_dynamic_m: bool = False,
    slot_ratio: int = 8,
) -> torch.Tensor:
    B, N, D = x.shape
    H = num_heads
    if D % H != 0:
        raise ValueError(
            f"Embedding dimension D ({D}) must be divisible by num_heads ({H})"
        )
    Dh = D // H

    x_h = x.view(B, N, H, Dh).permute(0, 2, 1, 3)
    q_h = q_proj.view(B, N, H, Dh).permute(0, 2, 1, 3)
    k_h = k_proj.view(B, N, H, Dh).permute(0, 2, 1, 3)
    v_h = v_proj.view(B, N, H, Dh).permute(0, 2, 1, 3)

    M = slot_k_emb.shape[1]
    slot_k = slot_k_emb.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    slot_v = slot_v_emb.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    effective_M = None
    if use_dynamic_m:
        M_min = 32
        effective_M = min(M, max(M_min, (N + slot_ratio - 1) // slot_ratio))

    for r in range(rounds):
        pos_bias = None
        if position_bias_fn is not None:
            pos_bias = position_bias_fn(x_h)

        indices = slot_select_reference(
            q_h, slot_k, mask=mask, causal=causal,
            position_bias=pos_bias, effective_M=effective_M,
        )
        attn_out, attn_w = slot_attend_reference(q_h, slot_k, slot_v, indices)
        x_h = x_h + attn_out
        slot_k, slot_v = slot_update_reference(k_h, v_h, slot_k, slot_v, indices, attn_w, alpha)
        slot_k, slot_v = slot_diffusion_reference(slot_k, slot_v, gamma)

    output = x_h.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    return output