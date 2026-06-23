import pytest
import torch
from murmurative.ops import slot_select_attend, slot_update, slot_diffusion
from murmurative.reference import (
    slot_select_reference,
    slot_attend_reference,
    slot_update_reference,
    slot_diffusion_reference,
)


class TestGradCheckCUDA:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_slot_select_attend_backward(self):
        B, H, N, M, D = 1, 2, 16, 64, 16
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out, w, idx = slot_select_attend(q, sk, sv)
        loss = out.sum() + w.sum()
        loss.backward()

        q_ref = q.detach().float().cpu().requires_grad_(True)
        sk_ref = sk.detach().float().cpu().requires_grad_(True)
        sv_ref = sv.detach().float().cpu().requires_grad_(True)
        idx_ref = slot_select_reference(q_ref, sk_ref)
        out_ref, w_ref = slot_attend_reference(q_ref, sk_ref, sv_ref, idx_ref)
        (out_ref.sum() + w_ref.sum()).backward()

        assert torch.allclose(q.grad.cpu().float(), q_ref.grad, atol=5e-2)
        assert torch.allclose(sk.grad.cpu().float(), sk_ref.grad, atol=5e-2)
        assert torch.allclose(sv.grad.cpu().float(), sv_ref.grad, atol=5e-2)

    def test_slot_update_backward(self):
        B, H, N, M, D = 1, 2, 16, 64, 16
        alpha = 0.9
        tk = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        tv = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        idx = torch.randint(0, M, (B, H, N, 7), device="cuda")
        w = torch.rand(B, H, N, 7, device="cuda", dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True)
        w.requires_grad_(True)

        sk_new, sv_new = slot_update(tk, tv, sk, sv, idx, w, alpha)
        (sk_new.sum() + sv_new.sum()).backward()

        tk_ref = tk.detach().float().cpu().requires_grad_(True)
        tv_ref = tv.detach().float().cpu().requires_grad_(True)
        sk_ref = sk.detach().float().cpu().requires_grad_(True)
        sv_ref = sv.detach().float().cpu().requires_grad_(True)
        w_ref = w.detach().float().cpu().requires_grad_(True)
        idx_ref = idx.cpu()
        sk_new_ref, sv_new_ref = slot_update_reference(
            tk_ref, tv_ref, sk_ref, sv_ref, idx_ref, w_ref, alpha
        )
        (sk_new_ref.sum() + sv_new_ref.sum()).backward()

        assert torch.allclose(tk.grad.cpu().float(), tk_ref.grad, atol=5e-2)
        assert torch.allclose(tv.grad.cpu().float(), tv_ref.grad, atol=5e-2)
        assert torch.allclose(sk.grad.cpu().float(), sk_ref.grad, atol=5e-2)
        assert torch.allclose(sv.grad.cpu().float(), sv_ref.grad, atol=5e-2)
        assert torch.allclose(w.grad.cpu().float(), w_ref.grad, atol=5e-2)

    def test_slot_diffusion_backward(self):
        B, H, M, D = 1, 2, 64, 16
        gamma = 0.1
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)

        sk_new, sv_new = slot_diffusion(sk, sv, gamma)
        (sk_new.sum() + sv_new.sum()).backward()

        sk_ref = sk.detach().float().cpu().requires_grad_(True)
        sv_ref = sv.detach().float().cpu().requires_grad_(True)
        sk_new_ref, sv_new_ref = slot_diffusion_reference(sk_ref, sv_ref, gamma)
        (sk_new_ref.sum() + sv_new_ref.sum()).backward()

        assert torch.allclose(sk.grad.cpu().float(), sk_ref.grad, atol=5e-2)
        assert torch.allclose(sv.grad.cpu().float(), sv_ref.grad, atol=5e-2)
