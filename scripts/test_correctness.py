#!/usr/bin/env python3
"""
Quick correctness test: CUDA kernels vs PyTorch reference implementations.
Run after building / JIT-compiling the CUDA extension.
"""
import torch
from murmurative.ops import (
    slot_select, slot_attend, slot_update, slot_diffusion, slot_murmurate
)
from murmurative.reference import (
    slot_select_reference,
    slot_attend_reference,
    slot_update_reference,
    slot_diffusion_reference,
    slot_murmurate_reference,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def assert_close(a, b, msg, atol=1e-3, rtol=1e-3):
    if isinstance(a, tuple):
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{msg}[{i}]", atol, rtol)
        return
    if not torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
        diff = (a.float() - b.float()).abs().max().item()
        print(f"FAIL {msg}: max diff={diff:.6f}")
        raise AssertionError(f"{msg}: max diff={diff:.6f}")
    print(f"OK  {msg}")


def test_slot_select():
    B, H, N, D, M = 2, 4, 128, 64, 256
    q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    sk = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)

    idx_ref = slot_select_reference(q, sk)
    idx_cuda = slot_select(q, sk)
    assert_close(idx_cuda, idx_ref, "slot_select", atol=0, rtol=0)  # exact match

    # causal
    idx_ref_c = slot_select_reference(q, sk, causal=True)
    idx_cuda_c = slot_select(q, sk, causal=True)
    assert_close(idx_cuda_c, idx_ref_c, "slot_select(causal)", atol=0, rtol=0)

    # with position bias
    pb = torch.randn(B, H, N, M, dtype=DTYPE, device=DEVICE)
    idx_ref_pb = slot_select_reference(q, sk, position_bias=pb)
    idx_cuda_pb = slot_select(q, sk, position_bias=pb)
    assert_close(idx_cuda_pb, idx_ref_pb, "slot_select(pos_bias)", atol=0, rtol=0)


def test_slot_attend():
    B, H, N, D, M = 2, 4, 128, 64, 256
    q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    sk = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    sv = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    idx = torch.randint(0, M, (B, H, N, 7), device=DEVICE).long()

    out_ref, w_ref = slot_attend_reference(q, sk, sv, idx)
    out_cuda, w_cuda = slot_attend(q, sk, sv, idx)
    assert_close(out_cuda, out_ref, "slot_attend(output)", atol=2e-3, rtol=2e-3)
    assert_close(w_cuda, w_ref, "slot_attend(weights)", atol=2e-3, rtol=2e-3)


def test_slot_update():
    B, H, N, D, M = 2, 4, 128, 64, 256
    tk = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    tv = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    sk = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    sv = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    idx = torch.randint(0, M, (B, H, N, 7), device=DEVICE).long()
    w = torch.rand(B, H, N, 7, device=DEVICE).float()

    sk_ref = sk.clone()
    sv_ref = sv.clone()
    slot_update_reference(tk, tv, sk_ref, sv_ref, idx, w, alpha=0.9)

    sk_cuda = sk.clone()
    sv_cuda = sv.clone()
    slot_update(tk, tv, sk_cuda, sv_cuda, idx, w, alpha=0.9)

    assert_close(sk_cuda, sk_ref, "slot_update(keys)", atol=2e-3, rtol=2e-3)
    assert_close(sv_cuda, sv_ref, "slot_update(values)", atol=2e-3, rtol=2e-3)


def test_slot_diffusion():
    B, H, M, D = 2, 4, 256, 64
    sk = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    sv = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)

    sk_ref, sv_ref = slot_diffusion_reference(sk.clone(), sv.clone(), gamma=0.1)
    sk_cuda, sv_cuda = slot_diffusion(sk.clone(), sv.clone(), gamma=0.1)

    assert_close(sk_cuda, sk_ref, "slot_diffusion(keys)", atol=1e-3, rtol=1e-3)
    assert_close(sv_cuda, sv_ref, "slot_diffusion(values)", atol=1e-3, rtol=1e-3)


def test_full_pipeline():
    B, N, D = 1, 512, 512
    H = 8
    Dh = D // H
    M = 256
    x = torch.randn(B, N, D, dtype=DTYPE, device=DEVICE)
    q_p = torch.randn(B, N, D, dtype=DTYPE, device=DEVICE)
    k_p = torch.randn(B, N, D, dtype=DTYPE, device=DEVICE)
    v_p = torch.randn(B, N, D, dtype=DTYPE, device=DEVICE)
    sk_emb = torch.randn(H, M, Dh, dtype=DTYPE, device=DEVICE)
    sv_emb = torch.randn(H, M, Dh, dtype=DTYPE, device=DEVICE)

    out_ref = slot_murmurate_reference(
        x, q_p, k_p, v_p, sk_emb, sv_emb,
        num_heads=H, rounds=4, alpha=0.9, gamma=0.1, causal=False,
    )
    out_cuda = slot_murmurate(
        x, q_p, k_p, v_p, sk_emb, sv_emb,
        num_heads=H, rounds=4, alpha=0.9, gamma=0.1, causal=False,
    )
    assert_close(out_cuda, out_ref, "slot_murmurate(full)", atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    test_slot_select()
    test_slot_attend()
    test_slot_update()
    test_slot_diffusion()
    test_full_pipeline()
    print("\nAll correctness tests passed!")
