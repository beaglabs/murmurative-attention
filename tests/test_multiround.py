import pytest
import torch
from murmurative.reference import slot_murmurate_reference


class TestMultiRound:
    def setup_method(self):
        self.B, self.N, self.D = 1, 128, 512
        self.H = 8
        self.M = 256
        self.Dh = self.D // self.H
        self.x = torch.randn(self.B, self.N, self.D)
        self.q_proj = torch.randn(self.B, self.N, self.D)
        self.k_proj = torch.randn(self.B, self.N, self.D)
        self.v_proj = torch.randn(self.B, self.N, self.D)
        self.sk_emb = torch.randn(self.H, self.M, self.Dh)
        self.sv_emb = torch.randn(self.H, self.M, self.Dh)

    def test_shape_preserved(self):
        out = slot_murmurate_reference(
            self.x, self.q_proj, self.k_proj, self.v_proj,
            self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=4,
        )
        assert out.shape == (self.B, self.N, self.D)

    def test_values_finite(self):
        out = slot_murmurate_reference(
            self.x, self.q_proj, self.k_proj, self.v_proj,
            self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=4,
        )
        assert torch.isfinite(out).all()

    def test_no_nan_multi_round(self):
        for rounds in [1, 2, 4, 8]:
            out = slot_murmurate_reference(
                self.x, self.q_proj, self.k_proj, self.v_proj,
                self.sk_emb, self.sv_emb,
                num_heads=self.H, rounds=rounds,
            )
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_variable_round_count(self):
        outs = []
        for rounds in [1, 2, 4]:
            out = slot_murmurate_reference(
                self.x.clone(), self.q_proj.clone(), self.k_proj.clone(),
                self.v_proj.clone(), self.sk_emb.clone(), self.sv_emb.clone(),
                num_heads=self.H, rounds=rounds,
            )
            outs.append(out)

        assert not torch.allclose(outs[0], outs[1])
        assert not torch.allclose(outs[1], outs[2])

    def test_batch_independence(self):
        B = 2
        x = torch.randn(B, self.N, self.D)
        out = slot_murmurate_reference(
            x, torch.randn(B, self.N, self.D), torch.randn(B, self.N, self.D),
            torch.randn(B, self.N, self.D), self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=2,
        )
        assert out.shape == (B, self.N, self.D)

    def test_large_sequence(self):
        N = 1024
        x = torch.randn(1, N, self.D)
        out = slot_murmurate_reference(
            x, torch.randn(1, N, self.D), torch.randn(1, N, self.D),
            torch.randn(1, N, self.D), self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=1,
        )
        assert out.shape == (1, N, self.D)
        assert torch.isfinite(out).all()

    def test_different_alphas(self):
        for alpha in [0.5, 0.9, 0.99]:
            out = slot_murmurate_reference(
                self.x, self.q_proj, self.k_proj, self.v_proj,
                self.sk_emb, self.sv_emb,
                num_heads=self.H, rounds=2, alpha=alpha,
            )
            assert torch.isfinite(out).all()

    def test_different_gammas(self):
        for gamma in [0.0, 0.1, 0.5]:
            out = slot_murmurate_reference(
                self.x, self.q_proj, self.k_proj, self.v_proj,
                self.sk_emb, self.sv_emb,
                num_heads=self.H, rounds=2, gamma=gamma,
            )
            assert torch.isfinite(out).all()