#!/usr/bin/env python3
"""
Probe training harness for murmurative attention.
Runs small-scale training loops and logs perplexity to validate
that optimizations do not degrade model quality.

Usage:
    # Baseline (no optimizations)
    python scripts/train_probe.py --steps 500 --use-reference

    # With dynamic M + fusion
    python scripts/train_probe.py --steps 500 --use-dynamic-m --slot-ratio 8 --use-fused

    # CUDA mode
    python scripts/train_probe.py --steps 500 --use-dynamic-m --slot-ratio 8

Results: JSON dict with final perplexity and loss curve printed to stdout.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from murmurative.ops import slot_murmurate
from murmurative.reference import slot_murmurate_reference


def build_synthetic_data(vocab_size, seq_len, num_batches, device, dtype):
    for _ in range(num_batches):
        ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        yield ids[:, :-1], ids[:, 1:]


class MurmurativeProbe(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_slots=256,
        num_rounds=4,
        alpha=0.9,
        gamma=0.1,
        use_dynamic_m=False,
        slot_ratio=8,
        use_fused=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.num_rounds = num_rounds
        self.alpha = alpha
        self.gamma = gamma
        self.use_dynamic_m = use_dynamic_m
        self.slot_ratio = slot_ratio
        self.use_fused = use_fused
        self.dh = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.slot_k_emb = nn.Parameter(torch.randn(num_heads, num_slots, self.dh) * 0.02)
        self.slot_v_emb = nn.Parameter(torch.randn(num_heads, num_slots, self.dh) * 0.02)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, use_reference=False):
        x = self.embed(input_ids)
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        if use_reference:
            attended = slot_murmurate_reference(
                x, q_proj, k_proj, v_proj,
                self.slot_k_emb, self.slot_v_emb,
                num_heads=self.num_heads,
                rounds=self.num_rounds,
                alpha=self.alpha,
                gamma=self.gamma,
            )
        else:
            attended = slot_murmurate(
                x, q_proj, k_proj, v_proj,
                self.slot_k_emb, self.slot_v_emb,
                num_heads=self.num_heads,
                rounds=self.num_rounds,
                alpha=self.alpha,
                gamma=self.gamma,
                use_dynamic_m=self.use_dynamic_m,
                slot_ratio=self.slot_ratio,
                use_fused_update_diffuse=self.use_fused,
            )

        out = self.output_proj(attended)
        logits = self.lm_head(out)
        return logits


def compute_perplexity(loss):
    return math.exp(min(loss, 20))


def main():
    parser = argparse.ArgumentParser(description="Probe training for murmurative attention")
    parser.add_argument("--steps", type=int, default=500, help="training steps")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=256)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use-reference", action="store_true", default=False)
    parser.add_argument("--use-dynamic-m", action="store_true", default=False)
    parser.add_argument("--slot-ratio", type=int, default=8)
    parser.add_argument("--use-fused", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if not args.use_reference and device == "cpu":
        print("Warning: CUDA not available, falling back to reference implementation")
        args.use_reference = True

    model = MurmurativeProbe(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_slots=args.num_slots,
        num_rounds=args.num_rounds,
        use_dynamic_m=args.use_dynamic_m,
        slot_ratio=args.slot_ratio,
        use_fused=args.use_fused,
    )
    model = model.to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-4)

    loss_history = []
    total_loss = 0.0

    data_iter = build_synthetic_data(
        args.vocab_size, args.seq_len, args.steps, device, dtype=torch.long
    )

    for step, (input_ids, target_ids) in enumerate(data_iter):
        optimizer.zero_grad()

        logits = model(input_ids.to(device), use_reference=args.use_reference)
        loss = F.cross_entropy(
            logits.view(-1, args.vocab_size),
            target_ids.view(-1).to(device),
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        loss_history.append(loss_val)

        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            ppl = compute_perplexity(avg_loss)
            print(f"step {step + 1:>4d} | loss={avg_loss:.4f} | ppl={ppl:.2f}")
            total_loss = 0.0

    avg_final_loss = sum(loss_history[-100:]) / min(100, len(loss_history))
    final_ppl = compute_perplexity(avg_final_loss)

    result = {
        "config": {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_slots": args.num_slots,
            "num_rounds": args.num_rounds,
            "use_reference": args.use_reference,
            "use_dynamic_m": args.use_dynamic_m,
            "slot_ratio": args.slot_ratio,
            "use_fused": args.use_fused,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
        },
        "final_loss": avg_final_loss,
        "final_perplexity": final_ppl,
        "loss_history": loss_history,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()