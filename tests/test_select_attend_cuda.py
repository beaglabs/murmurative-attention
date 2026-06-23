import pytest
import torch
from murmurative.ops import slot_select_attend
from murmurative.reference import slot_select_reference, slot_attend_reference


class TestSlotSelectAttendCUDA:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_shape(self):
        B, H, N, M, D = 1, 2, 32, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert out.shape == (B, H, N, D)
        assert w.shape == (B, H, N, 7)
        assert idx.shape == (B, H, N, 7)

    def test_weights_sum_to_one(self):
        B, H, N, M, D = 1, 2, 32, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-3)

    def test_weights_non_negative(self):
        B, H, N, M, D = 1, 2, 32, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert (w >= 0).all()

    def test_indices_valid(self):
        B, H, N, M, D = 1, 2, 32, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv)
        assert (idx >= 0).all()
        assert (idx < M).all()

    def test_matches_reference(self):
        B, H, N, M, D = 1, 2, 16, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        sv = torch.randn(B, H, M, D, dtype=torch.float32)

        q_h = q.half().cuda()
        sk_h = sk.half().cuda()
        sv_h = sv.half().cuda()
        out_cuda, w_cuda, idx_cuda = slot_select_attend(q_h, sk_h, sv_h)

        # Use CUDA indices to compute reference output to avoid fp16/fp32 topk diffs
        out_ref, w_ref = slot_attend_reference(q, sk, sv, idx_cuda.cpu())
        assert torch.allclose(out_cuda.cpu().float(), out_ref, atol=5e-2)
        assert torch.allclose(w_cuda.cpu().float(), w_ref, atol=5e-2)

    def test_causal(self):
        B, H, N, M, D = 1, 2, 16, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out, w, idx = slot_select_attend(q, sk, sv, causal=True)
        assert out.shape == (B, H, N, D)
        assert (idx >= 0).all()

    def test_with_mask(self):
        B, H, N, M, D = 1, 2, 16, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        # Fully mask the last half of tokens
        mask = torch.ones(B, 1, N, 1, dtype=torch.float16, device="cuda")
        mask[:, :, N // 2 :, :] = 0.0
        out, w, idx = slot_select_attend(q, sk, sv, mask=mask)
        assert out.shape == (B, H, N, D)

    def test_position_bias(self):
        B, H, N, M, D = 1, 2, 16, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        pb = torch.randn(B, H, N, M, dtype=torch.float16, device="cuda") * 0.1
        out, w, idx = slot_select_attend(q, sk, sv, position_bias=pb)
        assert out.shape == (B, H, N, D)

    def test_backward_matches_reference(self):
        B, H, N, M, D = 1, 2, 16, 64, 16
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out, w, idx = slot_select_attend(q, sk, sv)
        loss = out.sum()
        loss.backward()

        q_ref = q.detach().float().cpu().requires_grad_(True)
        sk_ref = sk.detach().float().cpu().requires_grad_(True)
        sv_ref = sv.detach().float().cpu().requires_grad_(True)
        idx_ref = slot_select_reference(q_ref, sk_ref)
        out_ref, _ = slot_attend_reference(q_ref, sk_ref, sv_ref, idx_ref)
        out_ref.sum().backward()

        assert torch.allclose(q.grad.cpu().float(), q_ref.grad, atol=5e-2)
        assert torch.allclose(sk.grad.cpu().float(), sk_ref.grad, atol=5e-2)
        assert torch.allclose(sv.grad.cpu().float(), sv_ref.grad, atol=5e-2)

    def test_backward_no_nan(self):
        B, H, N, M, D = 1, 2, 16, 64, 16
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

    def test_dynamic_m_all_slots(self):
        B, H, N, M, D = 1, 2, 16, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        out_full, w_full, idx_full = slot_select_attend(q, sk, sv, effective_M=M)
        out_part, w_part, idx_part = slot_select_attend(q, sk, sv, effective_M=32)
        assert out_full.shape == out_part.shape
        assert (idx_part < 32).all(), "Indices should be within effective_M range"

    def test_dynamic_m_indices_valid(self):
        B, H, N, M, D = 1, 2, 32, 128, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        for em in [16, 32, 64, 128]:
            out, w, idx = slot_select_attend(q, sk, sv, effective_M=em)
            assert (idx >= 0).all()
            assert (idx < em).all(), f"Indices exceed effective_M={em}"

    def test_dynamic_m_backward(self):
        B, H, N, M, D = 1, 2, 16, 64, 16
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out, w, idx = slot_select_attend(q, sk, sv, effective_M=32)
        out.sum().backward()

        assert q.grad is not None
        assert sk.grad is not None
        assert sv.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(sk.grad).any()
        assert not torch.isnan(sv.grad).any()
