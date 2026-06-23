import pytest
import torch
from murmurative.ops import slot_murmurate


class TestTrainingLoop:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_forward_shape(self):
        B, N, D = 2, 64, 64
        H, M = 4, 32
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb, num_heads=H, rounds=2)
        assert out.shape == (B, N, D)

    def test_values_finite(self):
        B, N, D = 1, 32, 64
        H, M = 2, 32
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb, num_heads=H, rounds=2)
        assert torch.isfinite(out).all()

    def test_backward(self):
        B, N, D = 1, 32, 64
        H, M = 2, 32
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb, num_heads=H, rounds=2)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert sk_emb.grad is not None
        assert sv_emb.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()
        assert not torch.isnan(sk_emb.grad).any()
        assert not torch.isnan(sv_emb.grad).any()

    def test_multi_round(self):
        B, N, D = 1, 16, 32
        H, M = 2, 16
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)

        for rounds in [1, 2, 4]:
            out = slot_murmurate(x, q, k, v, sk_emb, sv_emb, num_heads=H, rounds=rounds)
            loss = out.sum()
            loss.backward()
            assert out.shape == (B, N, D)
            assert not torch.isnan(out).any()

    def test_dynamic_m_forward(self):
        B, N, D = 1, 128, 64
        H, M = 4, 256
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb,
                             num_heads=H, rounds=2, use_dynamic_m=True, slot_ratio=8)
        assert out.shape == (B, N, D)
        assert torch.isfinite(out).all()

    def test_dynamic_m_backward(self):
        B, N, D = 1, 64, 64
        H, M = 4, 256
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb,
                             num_heads=H, rounds=2, use_dynamic_m=True, slot_ratio=8)
        loss = out.sum()
        loss.backward()

        for param, name in [(x, "x"), (q, "q"), (k, "k"), (v, "v"),
                            (sk_emb, "sk_emb"), (sv_emb, "sv_emb")]:
            assert param.grad is not None, f"{name} grad is None"
            assert not torch.isnan(param.grad).any(), f"{name} grad has NaN"

    def test_fused_forward(self):
        B, N, D = 1, 64, 64
        H, M = 4, 64
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda")
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda")

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb,
                             num_heads=H, rounds=2, use_fused_update_diffuse=True)
        assert out.shape == (B, N, D)
        assert torch.isfinite(out).all()

    def test_fused_backward(self):
        B, N, D = 1, 32, 64
        H, M = 4, 64
        Dh = D // H
        x = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        q = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)
        sv_emb = torch.randn(H, M, Dh, dtype=torch.float16, device="cuda", requires_grad=True)

        out = slot_murmurate(x, q, k, v, sk_emb, sv_emb,
                             num_heads=H, rounds=2, use_fused_update_diffuse=True)
        loss = out.sum()
        loss.backward()

        for name in ["x", "q", "k", "v", "sk_emb", "sv_emb"]:
            param = {"x": x, "q": q, "k": k, "v": v, "sk_emb": sk_emb, "sv_emb": sv_emb}[name]
            assert param.grad is not None, f"{name} grad is None"
            assert not torch.isnan(param.grad).any(), f"{name} grad has NaN"
