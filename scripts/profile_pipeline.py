#!/usr/bin/env python3
"""
Profile each kernel in the murmurative pipeline with torch.profiler.
Usage: python scripts/profile_pipeline.py
"""
import torch
from murmurative.ops import slot_select, slot_attend, slot_update, slot_diffusion

DEVICE = "cuda"
DTYPE = torch.float16


def profile_murmurative_round():
    B, H, N, D, M = 1, 8, 8192, 64, 256
    q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    sk = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    sv = torch.randn(B, H, M, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)

    # warmup
    for _ in range(3):
        idx = slot_select(q, sk)
        out, w = slot_attend(q, sk, sv, idx)
        sk2, sv2 = slot_update(k, v, sk, sv, idx, w)
        sk3, sv3 = slot_diffusion(sk2, sv2)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            idx = slot_select(q, sk)
            out, w = slot_attend(q, sk, sv, idx)
            sk2, sv2 = slot_update(k, v, sk, sv, idx, w)
            sk3, sv3 = slot_diffusion(sk2, sv2)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Save chrome trace for inspection
    prof.export_chrome_trace("benchmarks/profile_trace.json")
    print("Trace saved to benchmarks/profile_trace.json")


if __name__ == "__main__":
    profile_murmurative_round()
