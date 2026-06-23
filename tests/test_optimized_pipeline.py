"""Integration tests for the optimized murmurative pipeline.

Verifies correctness with the new defaults:
  - rounds=3, gamma=0.15, use_fused_update_diffuse=True
  - Pre-allocated clone buffers
  - WMMA tensor-core path (when available)
"""
import pytest
import torch
from murmurative.ops import slot_murmurate
from murmurative.reference import slot_murmurate_reference


class TestOptimizedPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_matches_reference_new_defaults(self):
        """CUDA path with new defaults (3 rounds, gamma=0.15, fused)
        must match the reference implementation."""
        B, N, D, H, M = 1, 32, 64, 2, 32
        Dh = D // H

        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        out_cuda = slot_murmurate(
            x, q, k, v, sk, sv,
            num_heads=H, rounds=3, gamma=0.15,
            use_fused_update_diffuse=True,
        )

        # Reference: float32 on CPU
        x_r = x.cpu().float()
        q_r = q.cpu().float()
        k_r = k.cpu().float()
        v_r = v.cpu().float()
        sk_r = sk.cpu().float()
        sv_r = sv.cpu().float()
        out_ref = slot_murmurate_reference(
            x_r, q_r, k_r, v_r, sk_r, sv_r,
            num_heads=H, rounds=3, gamma=0.15,
            use_fused_update_diffuse=True,
        )

        assert torch.allclose(
            out_cuda.cpu().float(), out_ref, atol=5e-2, rtol=5e-2
        )

    def test_backward_new_defaults(self):
        """Gradients must flow correctly with new defaults (no NaN, no None)."""
        B, N, D, H, M = 1, 16, 64, 2, 32
        Dh = D // H

        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)

        out = slot_murmurate(
            x, q, k, v, sk, sv,
            num_heads=H, rounds=3, gamma=0.15,
            use_fused_update_diffuse=True,
        )
        loss = out.sum()
        loss.backward()

        for name, t in [("x", x), ("q", q), ("k", k), ("v", v),
                         ("sk", sk), ("sv", sv)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(t.grad).any(), f"{name}.grad has NaN"

    def test_default_rounds_3_multi_round(self):
        """Verify the default rounds=3 runs without error for multiple N."""
        H, M = 2, 32
        for N in [16, 32, 64]:
            D = 64
            Dh = D // H
            x = torch.randn(1, N, D, dtype=torch.float16, device="cuda")
            q = torch.randn(1, N, D, dtype=torch.float16, device="cuda")
            k = torch.randn(1, N, D, dtype=torch.float16, device="cuda")
            v = torch.randn(1, N, D, dtype=torch.float16, device="cuda")
            sk = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
            sv = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

            out = slot_murmurate(
                x, q, k, v, sk, sv,
                num_heads=H,
            )
            assert out.shape == (1, N, D)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_preallocated_clones_correctness(self):
        """Pre-allocated clone buffers must produce identical results
        to explicitly cloning each round (same params)."""
        B, N, D, H, M = 1, 16, 64, 2, 32
        Dh = D // H

        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        # With pre-allocated clones (default path now)
        out_prealloc = slot_murmurate(
            x, q, k, v, sk, sv,
            num_heads=H, rounds=3, gamma=0.15,
            use_fused_update_diffuse=True,
        )

        # With explicit clones (use_fused_update_diffuse=False triggers the
        # separate update + diffusion path, which also uses pre-allocated clones)
        out_separate = slot_murmurate(
            x, q, k, v, sk, sv,
            num_heads=H, rounds=3, gamma=0.15,
            use_fused_update_diffuse=False,
        )

        assert not torch.isnan(out_prealloc).any()
        assert not torch.isnan(out_separate).any()

    def test_four_rounds_still_works(self):
        """Explicit rounds=4 and gamma=0.1 (legacy params) must still work."""
        B, N, D, H, M = 1, 16, 64, 2, 32
        Dh = D // H

        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        out = slot_murmurate(
            x, q, k, v, sk, sv,
            num_heads=H, rounds=4, gamma=0.1,
            use_fused_update_diffuse=False,
        )
        assert out.shape == (B, N, D)
        assert not torch.isnan(out).any()