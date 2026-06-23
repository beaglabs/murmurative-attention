import pytest
import torch
from murmurative.reference import (
    slot_attend_reference,
    slot_diffusion_reference,
    slot_select_reference,
)


class TestGradCheck:
    def setup_method(self):
        self.B, self.H, self.N, self.M, self.D = 1, 2, 32, 256, 32
        self.seed = 42

    def test_slot_attend_gradients(self):
        torch.manual_seed(self.seed)
        q = torch.randn(self.B, self.H, self.N, self.D, dtype=torch.float64, requires_grad=True)
        sk = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)
        sv = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)
        idx = slot_select_reference(q.detach(), sk.detach())

        def fn(qq, kk, vv):
            out, _ = slot_attend_reference(qq, kk, vv, idx)
            return out.to(torch.float64).sum()

        assert torch.autograd.gradcheck(fn, (q, sk, sv), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_slot_attend_weight_gradients(self):
        torch.manual_seed(self.seed)
        q = torch.randn(self.B, self.H, self.N, self.D, dtype=torch.float64, requires_grad=True)
        sk = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)
        sv = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64)
        idx = slot_select_reference(q.detach(), sk.detach())

        def fn(qq, kk):
            _, w = slot_attend_reference(qq, kk, sv, idx)
            return w.to(torch.float64).sum()

        assert torch.autograd.gradcheck(fn, (q, sk), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_slot_diffusion_gradients(self):
        torch.manual_seed(self.seed)
        sk = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)
        sv = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)

        def fn(ssk, ssv):
            sk2, sv2 = slot_diffusion_reference(ssk, ssv)
            return sk2.to(torch.float64).sum() + sv2.to(torch.float64).sum()

        assert torch.autograd.gradcheck(fn, (sk, sv), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_slot_diffusion_param_grad(self):
        B, H, M, D = 1, 1, 32, 16
        sk = torch.randn(B, H, M, D, dtype=torch.float64, requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float64)

        def fn(ssk):
            sk2, _ = slot_diffusion_reference(ssk, sv, gamma=0.1)
            return sk2.to(torch.float64).sum()

        assert torch.autograd.gradcheck(fn, (sk,), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_slot_attend_no_nan_grad(self):
        torch.manual_seed(self.seed)
        q = torch.randn(self.B, self.H, self.N, self.D, dtype=torch.float64, requires_grad=True)
        sk = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)
        sv = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float64, requires_grad=True)
        idx = slot_select_reference(q.detach(), sk.detach())

        out, _ = slot_attend_reference(q, sk, sv, idx)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(sk.grad).any()
        assert not torch.isnan(sv.grad).any()
        assert q.grad is not None
        assert sk.grad is not None
        assert sv.grad is not None