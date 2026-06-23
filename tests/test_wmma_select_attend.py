"""Tests for the WMMA (tensor-core) slot_select_attend path.

The WMMA path is automatically selected when:
  - D % 16 == 0
  - effective_M % 16 == 0
  - No mask
  - Compute capability >= 7.0
"""
import pytest
import torch
from murmurative.ops import slot_select_attend
from murmurative.reference import slot_select_reference, slot_attend_reference

WMMA_CAPABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7


class TestWMMASelectAttend:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not WMMA_CAPABLE:
            pytest.skip("WMMA requires CUDA compute capability >= 7.0")

    def test_wmma_shape(self):
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert out.shape == (B, H, N, D)
        assert w.shape == (B, H, N, 7)
        assert idx.shape == (B, H, N, 7)

    def test_wmma_weights_sum_to_one(self):
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-3)

    def test_wmma_weights_non_negative(self):
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert (w >= 0).all()

    def test_wmma_indices_valid(self):
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert (idx >= 0).all()
        assert (idx < M).all()

    def test_wmma_matches_reference(self):
        B, H, N, M, D = 1, 2, 16, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        sv = torch.randn(B, H, M, D, dtype=torch.float32)

        q_h = q.half().cuda()
        sk_h = sk.half().cuda()
        sv_h = sv.half().cuda()
        out_cuda, w_cuda, idx_cuda = slot_select_attend(q_h, sk_h, sv_h)

        out_ref, w_ref = slot_attend_reference(q, sk, sv, idx_cuda.cpu())
        assert torch.allclose(out_cuda.cpu().float(), out_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(w_cuda.cpu().float(), w_ref, atol=1e-1, rtol=1e-1)

    def test_wmma_backward(self):
        B, H, N, M, D = 1, 2, 16, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out, w, idx = slot_select_attend(q, sk, sv)
        out.sum().backward()

        assert q.grad is not None
        assert sk.grad is not None
        assert sv.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(sk.grad).any()
        assert not torch.isnan(sv.grad).any()

    def test_wmma_backward_matches_reference(self):
        B, H, N, M, D = 1, 2, 16, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out, w, idx = slot_select_attend(q, sk, sv)
        out.sum().backward()

        q_ref = q.detach().float().cpu().requires_grad_(True)
        sk_ref = sk.detach().float().cpu().requires_grad_(True)
        sv_ref = sv.detach().float().cpu().requires_grad_(True)
        out_ref, _ = slot_attend_reference(q_ref, sk_ref, sv_ref, idx.cpu())
        out_ref.sum().backward()

        assert torch.allclose(q.grad.cpu().float(), q_ref.grad, atol=1e-1, rtol=1e-1)
        assert torch.allclose(sk.grad.cpu().float(), sk_ref.grad, atol=1e-1, rtol=1e-1)
        assert torch.allclose(sv.grad.cpu().float(), sv_ref.grad, atol=1e-1, rtol=1e-1)

    def test_wmma_large_n(self):
        """Test with N > WMMA_N_TILE to cover multi-block path."""
        B, H, N, M, D = 1, 2, 128, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert out.shape == (B, H, N, D)
        assert (idx >= 0).all()
        assert (idx < M).all()

    def test_wmma_large_m(self):
        """Test with effective_M > WMMA_M_TILE to cover multi-block path."""
        B, H, N, M, D = 1, 2, 64, 256, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv, effective_M=256)
        assert out.shape == (B, H, N, D)
        assert (idx >= 0).all()
        assert (idx < 256).all()

    def test_wmma_dynamic_m(self):
        """Test with effective_M < M (dynamic slot count)."""
        B, H, N, M, D = 1, 2, 64, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv, effective_M=64)
        assert (idx < 64).all()

    def test_wmma_causal(self):
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv, causal=True)
        assert out.shape == (B, H, N, D)
        assert (idx >= 0).all()

    def test_wmma_position_bias(self):
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        pb = torch.randn(B, H, N, M, dtype=torch.float16, device="cuda") * 0.1
        out, w, idx = slot_select_attend(q, sk, sv, position_bias=pb)
        assert out.shape == (B, H, N, D)

    def test_wmma_vs_scalar_dispatch(self):
        """Verify that the WMMA path produces same outputs as scalar path
        (using D not divisible by 16 to force scalar path, then compare shapes).
        This test doesn't compare values since D differs."""
        B, H, N, M = 1, 2, 64, 64
        D_wmma = 64
        D_scalar = 50  # not divisible by 16, forces scalar path

        q_w = torch.randn(B, H, N, D_wmma, dtype=torch.float16, device="cuda")
        sk_w = torch.randn(B, H, M, D_wmma, dtype=torch.float16, device="cuda")
        sv_w = torch.randn(B, H, M, D_wmma, dtype=torch.float16, device="cuda")
        out_w, _, _ = slot_select_attend(q_w, sk_w, sv_w)

        q_s = torch.randn(B, H, N, D_scalar, dtype=torch.float16, device="cuda")
        sk_s = torch.randn(B, H, M, D_scalar, dtype=torch.float16, device="cuda")
        sv_s = torch.randn(B, H, M, D_scalar, dtype=torch.float16, device="cuda")
        out_s, _, _ = slot_select_attend(q_s, sk_s, sv_s)

        assert out_w.shape == (B, H, N, D_wmma)
        assert out_s.shape == (B, H, N, D_scalar)

    def test_wmma_fallback_with_mask(self):
        """Mask forces scalar path fallback — WMMA path skips masked inputs."""
        B, H, N, M, D = 1, 2, 64, 64, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        mask = torch.ones(B, 1, N, 1, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv, mask=mask)
        assert out.shape == (B, H, N, D)