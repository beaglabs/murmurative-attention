import pytest
import torch
from murmurative.reference import (
    slot_select_reference,
    slot_attend_reference,
    slot_update_reference,
)


class TestSlotUpdate:
    def setup_method(self):
        self.B, self.H, self.N, self.M, self.D = 1, 4, 128, 256, 64
        self.tk = torch.randn(self.B, self.H, self.N, self.D)
        self.tv = torch.randn(self.B, self.H, self.N, self.D)
        self.sk = torch.randn(self.B, self.H, self.M, self.D)
        self.sv = torch.randn(self.B, self.H, self.M, self.D)
        q = torch.randn(self.B, self.H, self.N, self.D)
        self.idx = slot_select_reference(q, self.sk)
        _, self.w = slot_attend_reference(q, self.sk, self.sv, self.idx)

    def test_output_differs_from_input(self):
        sk_before = self.sk.clone()
        sv_before = self.sv.clone()
        sk_new, sv_new = slot_update_reference(
            self.tk, self.tv, self.sk, self.sv, self.idx, self.w
        )
        assert not torch.allclose(sk_new, sk_before)
        assert not torch.allclose(sv_new, sv_before)

    def test_shape_preserved(self):
        original_shape = self.sk.shape
        sk_new, sv_new = slot_update_reference(
            self.tk, self.tv, self.sk, self.sv, self.idx, self.w
        )
        assert sk_new.shape == original_shape
        assert sv_new.shape == original_shape

    def test_values_finite(self):
        sk_new, sv_new = slot_update_reference(
            self.tk, self.tv, self.sk, self.sv, self.idx, self.w
        )
        assert torch.isfinite(sk_new).all()
        assert torch.isfinite(sv_new).all()

    def test_alpha_zero(self):
        sk_before = self.sk.clone()
        sk_new, sv_new = slot_update_reference(
            self.tk, self.tv, self.sk, self.sv, self.idx, self.w, alpha=0.0
        )
        assert not torch.allclose(sk_new, sk_before)

    def test_alpha_one(self):
        sk_before = self.sk.clone()
        sv_before = self.sv.clone()
        sk_new, sv_new = slot_update_reference(
            self.tk, self.tv, self.sk, self.sv, self.idx, self.w, alpha=1.0
        )
        assert torch.allclose(sk_new, sk_before)
        assert torch.allclose(sv_new, sv_before)

    def test_no_nan(self):
        sk_new, sv_new = slot_update_reference(
            self.tk, self.tv, self.sk, self.sv, self.idx, self.w
        )
        assert not torch.isnan(sk_new).any()
        assert not torch.isnan(sv_new).any()

    def test_fp16(self):
        tk = self.tk.half()
        tv = self.tv.half()
        sk = self.sk.half()
        sv = self.sv.half()
        sk_new, _ = slot_update_reference(
            tk.float(), tv.float(), sk.float(),
            sv.float(), self.idx, self.w.float()
        )
        assert torch.isfinite(sk_new.float()).all()
