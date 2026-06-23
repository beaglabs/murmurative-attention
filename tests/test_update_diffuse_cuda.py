import pytest
import torch
from murmurative.ops import slot_update, slot_diffusion
from murmurative.reference import slot_update_reference, slot_diffusion_reference


class TestSlotUpdateDiffuseCUDA:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def _make_update_inputs(self, B=1, H=2, N=16, M=64, D=16):
        tk = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        tv = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        idx = torch.randint(0, M, (B, H, N, 7), device="cuda")
        w = torch.rand(B, H, N, 7, device="cuda", dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True)
        return tk, tv, sk, sv, idx, w

    def test_update_shape(self):
        tk, tv, sk, sv, idx, w = self._make_update_inputs()
        sk_new, sv_new = slot_update(tk, tv, sk.clone(), sv.clone(), idx, w, alpha=0.9)
        assert sk_new.shape == sk.shape
        assert sv_new.shape == sv.shape

    def test_update_matches_reference(self):
        tk, tv, sk, sv, idx, w = self._make_update_inputs()
        alpha = 0.9

        sk_cuda = sk.clone()
        sv_cuda = sv.clone()
        slot_update(tk, tv, sk_cuda, sv_cuda, idx, w, alpha)

        sk_ref = sk.clone().cpu().float()
        sv_ref = sv.clone().cpu().float()
        slot_update_reference(
            tk.cpu().float(), tv.cpu().float(),
            sk_ref, sv_ref,
            idx.cpu(), w.cpu(), alpha,
        )

        assert torch.allclose(sk_cuda.cpu().float(), sk_ref, atol=1e-2)
        assert torch.allclose(sv_cuda.cpu().float(), sv_ref, atol=1e-2)

    def test_update_values_finite(self):
        tk, tv, sk, sv, idx, w = self._make_update_inputs()
        sk_new, sv_new = slot_update(tk, tv, sk.clone(), sv.clone(), idx, w)
        assert torch.isfinite(sk_new).all()
        assert torch.isfinite(sv_new).all()

    def test_diffusion_shape(self):
        B, H, M, D = 1, 2, 64, 16
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sk_new, sv_new = slot_diffusion(sk.clone(), sv.clone(), gamma=0.1)
        assert sk_new.shape == sk.shape
        assert sv_new.shape == sv.shape

    def test_diffusion_matches_reference(self):
        B, H, M, D = 1, 2, 64, 16
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        gamma = 0.1

        sk_cuda = sk.clone()
        sv_cuda = sv.clone()
        slot_diffusion(sk_cuda, sv_cuda, gamma)

        sk_ref, sv_ref = slot_diffusion_reference(sk.cpu().float(), sv.cpu().float(), gamma)

        assert torch.allclose(sk_cuda.cpu().float(), sk_ref, atol=1e-3)
        assert torch.allclose(sv_cuda.cpu().float(), sv_ref, atol=1e-3)

    def test_diffusion_boundaries_preserved(self):
        B, H, M, D = 1, 2, 64, 16
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sk_before = sk.clone()
        sk_new, sv_new = slot_diffusion(sk, sv, gamma=0.1)
        assert torch.allclose(sk_new[:, :, 0], sk_before[:, :, 0])
        assert torch.allclose(sk_new[:, :, -1], sk_before[:, :, -1])

    def test_diffusion_values_finite(self):
        B, H, M, D = 1, 2, 64, 16
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sk_new, sv_new = slot_diffusion(sk.clone(), sv.clone(), gamma=0.1)
        assert torch.isfinite(sk_new).all()
        assert torch.isfinite(sv_new).all()


class TestSlotUpdateDiffuseCUDA:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def _make_fwd_inputs(self, B=1, H=2, N=16, M=64, D=16):
        tk = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        tv = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
        idx = torch.randint(0, M, (B, H, N, 7), device="cuda")
        w = torch.rand(B, H, N, 7, device="cuda", dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True)
        return tk, tv, sk, sv, idx, w

    def _fused_fwd(self, sk, sv, tk, tv, idx, w, alpha=0.9, gamma=0.1):
        from murmurative.ops import _get_cuda_lib
        lib = _get_cuda_lib()
        sk_out = sk.clone()
        sv_out = sv.clone()
        lib.slot_update_diffuse(
            sk_out, sv_out, tk, tv, idx, w, alpha, gamma
        )
        return sk_out, sv_out

    def _separate_fwd(self, sk, sv, tk, tv, idx, w, alpha=0.9, gamma=0.1):
        sk_out = sk.clone()
        sv_out = sv.clone()
        slot_update(tk, tv, sk_out, sv_out, idx, w, alpha)
        slot_diffusion(sk_out, sv_out, gamma)
        return sk_out, sv_out

    def test_fused_shape(self):
        tk, tv, sk, sv, idx, w = self._make_fwd_inputs()
        sk_out, sv_out = self._fused_fwd(sk, sv, tk, tv, idx, w)
        assert sk_out.shape == sk.shape
        assert sv_out.shape == sv.shape

    def test_fused_values_finite(self):
        tk, tv, sk, sv, idx, w = self._make_fwd_inputs()
        sk_out, sv_out = self._fused_fwd(sk, sv, tk, tv, idx, w)
        assert torch.isfinite(sk_out).all()
        assert torch.isfinite(sv_out).all()

    def _jacobi_reference_fwd(self, sk, sv, tk, tv, idx, w, alpha=0.9, gamma=0.1):
        """Reference matching the fused kernel's Jacobi-style ordering.
        Snapshots original slot values for EMA source + diffusion neighbors,
        then applies EMA + tridiagonal diffusion without cross-slot interference."""
        B, H, N, D = tk.shape
        M = sk.shape[2]
        K = 7

        sk_orig = sk.cpu().float().clone()
        sv_orig = sv.cpu().float().clone()
        sk_out = sk_orig.clone()
        sv_out = sv_orig.clone()
        tk_cpu = tk.cpu().float()
        tv_cpu = tv.cpu().float()
        idx_cpu = idx.cpu()
        w_cpu = w.cpu()

        for b in range(B):
            for h in range(H):
                for m in range(M):
                    k_acc = torch.zeros(D, dtype=torch.float32)
                    v_acc = torch.zeros(D, dtype=torch.float32)
                    w_sum = 0.0
                    for n in range(N):
                        for kk in range(K):
                            if idx_cpu[b, h, n, kk].item() != m:
                                continue
                            cur_w = w_cpu[b, h, n, kk].item()
                            w_sum += cur_w
                            k_acc += cur_w * tk_cpu[b, h, n]
                            v_acc += cur_w * tv_cpu[b, h, n]

                    my_old_k = sk_orig[b, h, m]
                    my_old_v = sv_orig[b, h, m]
                    if w_sum > 0:
                        new_k = alpha * my_old_k + (1.0 - alpha) * k_acc / w_sum
                        new_v = alpha * my_old_v + (1.0 - alpha) * v_acc / w_sum
                    else:
                        new_k = my_old_k
                        new_v = my_old_v

                    if 0 < m < M - 1:
                        left_k = sk_orig[b, h, m - 1]
                        right_k = sk_orig[b, h, m + 1]
                        left_v = sv_orig[b, h, m - 1]
                        right_v = sv_orig[b, h, m + 1]
                        new_k = new_k + gamma * (left_k + right_k - 2.0 * new_k)
                        new_v = new_v + gamma * (left_v + right_v - 2.0 * new_v)

                    sk_out[b, h, m] = new_k
                    sv_out[b, h, m] = new_v

        return sk_out.to(device=sk.device, dtype=sk.dtype), sv_out.to(device=sv.device, dtype=sv.dtype)

    def test_fused_matches_jacobi_reference(self):
        tk, tv, sk, sv, idx, w = self._make_fwd_inputs(N=16, M=32, D=16)
        alpha, gamma = 0.9, 0.1

        sk_fused, sv_fused = self._fused_fwd(sk, sv, tk, tv, idx, w, alpha, gamma)
        sk_ref, sv_ref = self._jacobi_reference_fwd(sk, sv, tk, tv, idx, w, alpha, gamma)

        assert torch.allclose(sk_fused.cpu().float(), sk_ref.cpu().float(), atol=1e-2, rtol=1e-2)
        assert torch.allclose(sv_fused.cpu().float(), sv_ref.cpu().float(), atol=1e-2, rtol=1e-2)

    def test_fused_backward(self):
        B, H, N, M, D = 1, 2, 16, 64, 16
        alpha, gamma = 0.9, 0.1
        tk = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        tv = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sk = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        sv = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda", requires_grad=True)
        idx = torch.randint(0, M, (B, H, N, 7), device="cuda")
        w = torch.rand(B, H, N, 7, device="cuda", dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True)
        w.requires_grad_(True)

        from murmurative.ops import _SlotUpdateDiffuse
        sk_new, sv_new = _SlotUpdateDiffuse.apply(
            tk, tv, sk.clone(), sv.clone(), idx, w, alpha, gamma
        )
        loss = (sk_new.sum() + sv_new.sum())
        loss.backward()

        for param, name in [(tk, "token_keys"), (tv, "token_values"),
                            (sk, "slot_keys"), (sv, "slot_values"), (w, "weights")]:
            assert param.grad is not None, f"{name} grad is None"
            assert not torch.isnan(param.grad).any(), f"{name} grad has NaN"
            assert not torch.isinf(param.grad).any(), f"{name} grad has Inf"
