import pytest
import torch
from murmurative.reference import slot_select_reference


class TestSlotSelect:
    def test_shape_small(self):
        B, H, N, M, D = 1, 4, 128, 256, 64
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        assert idx.shape == (B, H, N, 7)
        assert idx.dtype == torch.int64

    def test_shape_medium(self):
        B, H, N, M, D = 2, 8, 1024, 256, 64
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        assert idx.shape == (B, H, N, 7)

    def test_shape_large(self):
        B, H, N, M, D = 1, 4, 8192, 256, 64
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        assert idx.shape == (B, H, N, 7)

    def test_indices_valid(self):
        B, H, N, M, D = 1, 2, 64, 256, 32
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        assert (idx >= 0).all()
        assert (idx < M).all()

    def test_indices_valid_in_range(self):
        B, H, N, M, D = 1, 2, 64, 256, 32
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        assert (idx >= 0).all()
        assert (idx < M).all()

    def test_causal(self):
        B, H, N, M, D = 1, 2, 32, 256, 16
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk, causal=True)
        assert (idx >= 0).all()

    def test_mask(self):
        B, H, N, M, D = 1, 2, 32, 256, 16
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        mask = torch.ones(B, 1, N, 1, dtype=torch.float32)
        mask[:, :, 16:, :] = 0.0
        idx = slot_select_reference(q, sk, mask=mask)
        assert idx.shape == (B, H, N, 7)

    def test_fp16(self):
        B, H, N, M, D = 1, 2, 64, 256, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16)
        sk = torch.randn(B, H, M, D, dtype=torch.float16)
        idx = slot_select_reference(q.float(), sk.float())
        assert idx.shape == (B, H, N, 7)

    def test_bf16(self):
        B, H, N, M, D = 1, 2, 64, 256, 32
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16)
        sk = torch.randn(B, H, M, D, dtype=torch.bfloat16)
        idx = slot_select_reference(q.float(), sk.float())
        assert idx.shape == (B, H, N, 7)

    def test_position_bias(self):
        B, H, N, M, D = 1, 2, 16, 256, 16
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        pb = torch.randn(B, H, N, M, dtype=torch.float32) * 0.1
        idx = slot_select_reference(q, sk, position_bias=pb)
        assert idx.shape == (B, H, N, 7)

    def test_edge_small_m(self):
        B, H, N, M, D = 1, 1, 16, 7, 16
        q = torch.randn(B, H, N, D, dtype=torch.float32)
        sk = torch.randn(B, H, M, D, dtype=torch.float32)
        idx = slot_select_reference(q, sk)
        assert idx.shape == (B, H, N, 7)