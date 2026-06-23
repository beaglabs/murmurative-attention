import pytest
import torch
from murmurative.reference import slot_diffusion_reference


class TestSlotDiffusion:
    def setup_method(self):
        self.B, self.H, self.M, self.D = 1, 4, 256, 64
        self.sk = torch.randn(self.B, self.H, self.M, self.D)
        self.sv = torch.randn(self.B, self.H, self.M, self.D)

    def test_shape_preserved(self):
        sk2, sv2 = slot_diffusion_reference(self.sk, self.sv)
        assert sk2.shape == self.sk.shape
        assert sv2.shape == self.sv.shape

    def test_values_finite(self):
        sk2, sv2 = slot_diffusion_reference(self.sk, self.sv)
        assert torch.isfinite(sk2).all()
        assert torch.isfinite(sv2).all()

    def test_gamma_zero(self):
        sk2, sv2 = slot_diffusion_reference(self.sk, self.sv, gamma=0.0)
        assert torch.allclose(sk2, self.sk)
        assert torch.allclose(sv2, self.sv)

    def test_gamma_changes_values(self):
        sk2, sv2 = slot_diffusion_reference(self.sk, self.sv, gamma=0.5)
        assert not torch.allclose(sk2, self.sk)

    def test_boundaries_preserved(self):
        sk2, sv2 = slot_diffusion_reference(self.sk, self.sv, gamma=0.5)
        assert torch.allclose(sk2[:, :, 0], self.sk[:, :, 0])
        assert torch.allclose(sk2[:, :, -1], self.sk[:, :, -1])

    def test_fp16(self):
        sk = self.sk.half()
        sv = self.sv.half()
        sk2, sv2 = slot_diffusion_reference(sk.float(), sv.float())
        assert torch.isfinite(sk2).all()

    def test_single_head(self):
        sk = torch.randn(1, 1, self.M, self.D)
        sv = torch.randn(1, 1, self.M, self.D)
        sk2, sv2 = slot_diffusion_reference(sk, sv)
        assert sk2.shape == sk.shape

    def test_no_nan(self):
        sk2, sv2 = slot_diffusion_reference(self.sk, self.sv)
        assert not torch.isnan(sk2).any()
        assert not torch.isnan(sv2).any()