import pytest
import torch
from murmurative.reference import slot_select_reference, slot_attend_reference


class TestSlotAttend:
    def setup_method(self):
        self.B, self.H, self.N, self.M, self.D = 1, 4, 128, 256, 64
        self.q = torch.randn(self.B, self.H, self.N, self.D, dtype=torch.float32)
        self.sk = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float32)
        self.sv = torch.randn(self.B, self.H, self.M, self.D, dtype=torch.float32)
        self.idx = slot_select_reference(self.q, self.sk)

    def test_shape_small(self):
        out, w = slot_attend_reference(self.q, self.sk, self.sv, self.idx)
        assert out.shape == (self.B, self.H, self.N, self.D)
        assert w.shape == (self.B, self.H, self.N, 7)

    def test_shape_medium(self):
        B, H, N, M, D = 2, 8, 1024, 256, 64
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        sv = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        out, w = slot_attend_reference(q, sk, sv, idx)
        assert out.shape == (B, H, N, D)
        assert w.shape == (B, H, N, 7)

    def test_weights_sum_to_one(self):
        out, w = slot_attend_reference(self.q, self.sk, self.sv, self.idx)
        assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)

    def test_weights_non_negative(self):
        out, w = slot_attend_reference(self.q, self.sk, self.sv, self.idx)
        assert (w >= 0).all()

    def test_custom_scale(self):
        scale = 0.5
        out, w = slot_attend_reference(self.q, self.sk, self.sv, self.idx, scale=scale)
        assert w.sum(dim=-1).allclose(torch.ones_like(w.sum(dim=-1)), atol=1e-5)

    def test_output_finite(self):
        out, w = slot_attend_reference(self.q, self.sk, self.sv, self.idx)
        assert torch.isfinite(out).all()

    def test_fp16(self):
        q = self.q.half()
        sk = self.sk.half()
        sv = self.sv.half()
        idx = self.idx
        out, w = slot_attend_reference(q.float(), sk.float(), sv.float(), idx)
        assert torch.isfinite(out).all()

    def test_bf16(self):
        q = self.q.bfloat16()
        sk = self.sk.bfloat16()
        sv = self.sv.bfloat16()
        idx = self.idx
        out, w = slot_attend_reference(q.float(), sk.float(), sv.float(), idx)
        assert torch.isfinite(out).all()

    def test_edge_single_head(self):
        B, H, N, M, D = 1, 1, 16, 256, 32
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        sv = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        out, w = slot_attend_reference(q, sk, sv, idx)
        assert out.shape == (B, H, N, D)