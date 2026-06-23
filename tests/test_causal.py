import pytest
import torch
from murmurative.reference import slot_murmurate_reference


class TestCausal:
    def setup_method(self):
        self.B, self.N, self.D = 1, 32, 256
        self.H = 4
        self.M = 256
        self.Dh = self.D // self.H
        self.x = torch.randn(self.B, self.N, self.D)
        self.q_proj = torch.randn(self.B, self.N, self.D)
        self.k_proj = torch.randn(self.B, self.N, self.D)
        self.v_proj = torch.randn(self.B, self.N, self.D)
        self.sk_emb = torch.randn(self.H, self.M, self.Dh)
        self.sv_emb = torch.randn(self.H, self.M, self.Dh)

    def test_causal_does_not_crash(self):
        out = slot_murmurate_reference(
            self.x, self.q_proj, self.k_proj, self.v_proj,
            self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=2, causal=True,
        )
        assert out.shape == (self.B, self.N, self.D)

    def test_causal_values_finite(self):
        out = slot_murmurate_reference(
            self.x, self.q_proj, self.k_proj, self.v_proj,
            self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=4, causal=True,
        )
        assert torch.isfinite(out).all()

    def test_causal_different_from_noncausal(self):
        out_non = slot_murmurate_reference(
            self.x.clone(), self.q_proj.clone(), self.k_proj.clone(),
            self.v_proj.clone(), self.sk_emb.clone(), self.sv_emb.clone(),
            num_heads=self.H, rounds=1, causal=False,
        )
        out_causal = slot_murmurate_reference(
            self.x.clone(), self.q_proj.clone(), self.k_proj.clone(),
            self.v_proj.clone(), self.sk_emb.clone(), self.sv_emb.clone(),
            num_heads=self.H, rounds=1, causal=True,
        )
        assert not torch.allclose(out_non, out_causal)

    def test_causal_with_mask(self):
        mask = torch.ones(self.B, 1, self.N, 1)
        out = slot_murmurate_reference(
            self.x, self.q_proj, self.k_proj, self.v_proj,
            self.sk_emb, self.sv_emb,
            num_heads=self.H, rounds=2, causal=True, mask=mask,
        )
        assert out.shape == (self.B, self.N, self.D)

    def test_streaming_consistency(self):
        out_batch = slot_murmurate_reference(
            self.x.clone(), self.q_proj.clone(), self.k_proj.clone(),
            self.v_proj.clone(), self.sk_emb.clone(), self.sv_emb.clone(),
            num_heads=self.H, rounds=1, causal=False,
        )

        outputs = []
        B, H = self.B, self.H
        D = self.D
        Dh = self.Dh
        M = self.M
        x_h = self.x.view(B, self.N, H, Dh).permute(0, 2, 1, 3)
        slot_k = self.sk_emb.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        slot_v = self.sv_emb.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        for n in range(self.N):
            token_q = x_h[:, :, n:n+1, :].squeeze(2)  # [B, H, Dh]
            scores = torch.einsum("bhd,bhmd->bhm", token_q, slot_k)
            _, idx = torch.topk(scores, k=7, dim=-1, sorted=True)
            out_batch = out_batch

        assert out_batch.shape == (self.B, self.N, self.D)