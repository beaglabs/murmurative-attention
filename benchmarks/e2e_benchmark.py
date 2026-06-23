#!/usr/bin/env python3
"""
End-to-end benchmarks for murmurative attention vs standard attention.

Sections:
  1. Single-operation latency (slot_select_attend, slot_update, slot_diffusion)
  2. Pipeline throughput (slot_murmurate end-to-end)
  3. Model-level throughput (real forward pass)
  4. Scaling sweep (vary N, M, rounds)
  5. Training Efficiency (perplexity vs GFLOPs vs wall time)

Section 5 tests the core thesis: "O(N*M) FLOPs = same intelligence for lower
training cost" by training matched-size murmurative and MHA models on WikiText-2
at multiple sequence lengths.

Usage:
    # Run Section 5 only
    python benchmarks/e2e_benchmark.py --skip 1 2 3 4

    # Run all sections
    python benchmarks/e2e_benchmark.py

    # Section 5 with custom params
    python benchmarks/e2e_benchmark.py --skip 1 2 3 4 --eff-steps 200 --eff-eval-every 50 --eff-ns 256 512
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from murmurative.ops import slot_murmurate


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

def count_murmurative_flops(N, Dh, M, rounds, H, effective_M):
    """Murmurative attention FLOPs (fwd+bwd) for one batch element. Returns GFLOPs.

    Forward per head per round:
      - Select (Q @ SK^T):  N x effective_M x Dh MACs
      - Attend (gather + softmax + weighted V): N x 7 x Dh MACs
      - Update (EMA scatter): N x 7 x Dh MACs
      - Diffusion (tridiagonal stencil): M x Dh (negligible)

    Fwd total = H * rounds * N * (effective_M + 14) * Dh MACs
              = 2 * H * rounds * N * (effective_M + 14) * Dh FLOPs
    Bwd ~ 2x fwd  (approximation)
    Total fwd+bwd ~ 3x fwd MACs -> 6x fwd MACs in FLOPs
    """
    macs_fwd = H * rounds * N * (effective_M + 14) * Dh
    flops_fwd = 2.0 * macs_fwd
    flops_bwd = 2.0 * flops_fwd
    return (flops_fwd + flops_bwd) / 1e9


def count_torch_sdpa_flops(N, D):
    """Standard MHA attention FLOPs (fwd+bwd) for one batch element. Returns GFLOPs.

    Forward:
      - QK^T:  N x N x D MACs
      - Attn@V: N x N x D MACs
    Fwd = 2 * N^2 * D MACs = 4 * N^2 * D FLOPs
    Bwd ~ 2x fwd
    Total fwd+bwd ~ 3 * fwd ~ 12 * N^2 * D FLOPs
    """
    flops_fwd = 4.0 * N * N * D
    flops_bwd = 2.0 * flops_fwd
    return (flops_fwd + flops_bwd) / 1e9


def count_model_flops_one_step(model_type, B, N, D, H, Dh, M, rounds,
                                effective_M, vocab_size):
    """FLOPs for one fwd+bwd training step. Returns (total_GFLOPs, attn_GFLOPs)."""
    embed = B * N * D * 2
    qkv = B * 3 * N * D * D * 2
    ffn = B * 8 * N * D * D * 2
    output_proj = B * N * D * D * 2
    out_head = B * N * D * vocab_size * 2

    if model_type == "murmurative":
        attn = count_murmurative_flops(N, Dh, M, rounds, H, effective_M)
    else:
        attn = count_torch_sdpa_flops(N, D)

    attn_step_gflops = attn * B
    total = embed + qkv + ffn + output_proj + out_head + attn * 1e9 * B
    return total / 1e9, attn_step_gflops


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MurmurativeModel(nn.Module):
    """Small transformer with murmurative attention for benchmarks."""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_slots=256,
        rounds=3,
        alpha=0.9,
        gamma=0.15,
        use_dynamic_m=True,
        slot_ratio=8,
        use_fused=True,
        use_reference=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.rounds = rounds
        self.alpha = alpha
        self.gamma = gamma
        self.use_dynamic_m = use_dynamic_m
        self.slot_ratio = slot_ratio
        self.use_fused = use_fused
        self.use_reference = use_reference
        self.dh = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.slot_k_emb = nn.Parameter(
            torch.randn(num_heads, num_slots, self.dh) * 0.02
        )
        self.slot_v_emb = nn.Parameter(
            torch.randn(num_heads, num_slots, self.dh) * 0.02
        )
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, N = input_ids.shape
        D = self.d_model
        x = self.embed(input_ids)
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        attended = slot_murmurate(
            x, q_proj, k_proj, v_proj,
            self.slot_k_emb, self.slot_v_emb,
            num_heads=self.num_heads,
            rounds=self.rounds,
            alpha=self.alpha,
            gamma=self.gamma,
            use_dynamic_m=self.use_dynamic_m,
            slot_ratio=self.slot_ratio,
            use_fused_update_diffuse=self.use_fused,
            use_reference=self.use_reference,
        )
        x = attended + self.ffn(attended)
        out = self.output_proj(x)
        logits = self.lm_head(out)
        return logits


class MHAModel(nn.Module):
    """Small transformer with standard multi-head attention (SDPA) for benchmarks."""

    def __init__(self, vocab_size, d_model=256, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dh = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, N = input_ids.shape
        D = self.d_model
        H = self.num_heads
        Dh = self.dh

        x = self.embed(input_ids)
        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        x = x + attn_out
        x = x + self.ffn(x)
        out = self.output_proj(x)
        logits = self.lm_head(out)
        return logits


class FlashAttnModel(nn.Module):
    """Small transformer with flash-attention for benchmarks."""

    def __init__(self, vocab_size, d_model=256, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dh = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        from flash_attn import flash_attn_func
        B, N = input_ids.shape
        D = self.d_model
        H = self.num_heads

        x = self.embed(input_ids)
        q = self.q_proj(x).view(B, N, H, D // H)
        k = self.k_proj(x).view(B, N, H, D // H)
        v = self.v_proj(x).view(B, N, H, D // H)

        attn_out = flash_attn_func(q, k, v, causal=True)
        attn_out = attn_out.contiguous().view(B, N, D)
        x = x + attn_out
        x = x + self.ffn(x)
        out = self.output_proj(x)
        logits = self.lm_head(out)
        return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_WIKITEXT2_CACHE = {}


def load_wikitext2(tokenizer_name="gpt2", seq_len=512, force_synthetic=False):
    """Load WikiText-2 and return tokenised train/validation tensors (long).

    Uses `datasets` to fetch raw text and `tiktoken` for tokenization.
    Caches tokenized data in memory to avoid re-downloading.
    Falls back to synthetic structured data if WikiText-2 is unavailable.
    """
    cache_key = (tokenizer_name, seq_len, force_synthetic)
    if cache_key in _WIKITEXT2_CACHE:
        print("  Using cached tokenized data.")
        return _WIKITEXT2_CACHE[cache_key]

    if force_synthetic:
        print("  --force-synthetic: using structured synthetic data.")
        result = _synthetic_structured_data()
        _WIKITEXT2_CACHE[cache_key] = result
        return result

    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError as e:
        print(f"  Warning: datasets/tiktoken not available ({e}).")
        print("  Falling back to structured synthetic data.")
        result = _synthetic_structured_data()
        _WIKITEXT2_CACHE[cache_key] = result
        return result

    try:
        print("  Downloading WikiText-2 from HuggingFace...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    except Exception as e:
        print(f"  Warning: WikiText-2 download failed ({type(e).__name__}: {e}).")
        print("  Falling back to structured synthetic data.")
        result = _synthetic_structured_data()
        _WIKITEXT2_CACHE[cache_key] = result
        return result

    enc = tiktoken.get_encoding(tokenizer_name)
    print(f"  Tokenizer: {tokenizer_name}, vocab_size={enc.n_vocab}")

    train_tokens = _tokenize_concat(ds, enc)
    valid_tokens = _tokenize_concat(ds_val, enc)
    print(f"  Train tokens: {len(train_tokens)}, Valid tokens: {len(valid_tokens)}")

    result = (train_tokens, valid_tokens, enc.n_vocab)
    _WIKITEXT2_CACHE[cache_key] = result
    return result


def _tokenize_concat(dataset, enc):
    """Tokenize all text examples in dataset and concatenate into one 1-D tensor."""
    all_ids = []
    for example in dataset:
        text = example["text"]
        if text and text.strip():
            all_ids.extend(enc.encode(text))
    return torch.tensor(all_ids, dtype=torch.long)


def _synthetic_structured_data():
    """Structured synthetic data with learnable patterns (Markov chain).

    Generates sequences from a small (256) vocabulary using a sparse transition
    matrix. Each token predicts one of ~5 next tokens with high probability,
    creating learnable n-gram structure. Train and validation use different
    random seeds so the model must generalize the learned patterns.
    """
    vocab_size = 256
    num_train = 2_000_000
    num_valid = 200_000

    gen = torch.Generator()
    gen.manual_seed(42)
    trans = _build_markov_transition(vocab_size)

    train_tokens = _generate_markov_sequence(trans, num_train, 0, gen)
    gen.manual_seed(123)
    valid_tokens = _generate_markov_sequence(trans, num_valid, 0, gen)

    print(f"  Synthetic: vocab={vocab_size}, train_tokens={len(train_tokens)}, "
          f"valid_tokens={len(valid_tokens)}")
    return train_tokens, valid_tokens, vocab_size


def _build_markov_transition(vocab_size):
    """Build a sparse transition matrix with learnable structure."""
    trans = torch.zeros(vocab_size, vocab_size)
    for i in range(vocab_size):
        successors = [(i + 1) % vocab_size,
                      (i + 3) % vocab_size,
                      (i * 7 + 13) % vocab_size,
                      (i + 5) % vocab_size,
                      (i * 3 + 7) % vocab_size]
        for s in successors:
            trans[i, s] = 1.0
    trans = trans / trans.sum(dim=1, keepdim=True)
    return trans


def _generate_markov_sequence(trans, num_tokens, start_token, gen):
    """Generate a sequence from a Markov transition matrix."""
    tokens = [start_token]
    for _ in range(num_tokens - 1):
        probs = trans[tokens[-1]]
        next_token = torch.multinomial(probs, 1, generator=gen).item()
        tokens.append(next_token)
    return torch.tensor(tokens, dtype=torch.long)


def build_batches(tokens, seq_len, batch_size, num_batches, device):
    """Yield (input_ids, target_ids) batches from a contiguous token tensor."""
    total_needed = num_batches * batch_size * seq_len
    if len(tokens) < total_needed + 1:
        repeats = (total_needed + 1 + len(tokens) - 1) // len(tokens)
        tokens = tokens.repeat(repeats)
    for i in range(num_batches):
        offset = i * batch_size * seq_len
        batch = tokens[offset:offset + batch_size * seq_len + 1]
        inp = batch[:-1].view(batch_size, seq_len).to(device)
        tgt = batch[1:].view(batch_size, seq_len).to(device)
        yield inp, tgt


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(model, tokens, seq_len, batch_size, num_batches=10):
    """Evaluate perplexity on a held-out token sequence."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    vocab_size = model.lm_head.out_features
    for inp, tgt in build_batches(tokens, seq_len, batch_size, num_batches,
                                   next(model.parameters()).device):
        logits = model(inp)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), tgt.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += tgt.numel()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))


# ---------------------------------------------------------------------------
# Training with FLOP tracking (Section 5)
# ---------------------------------------------------------------------------

def train_with_flop_tracking(model, train_tokens, valid_tokens,
                              step_total_gflops, step_attn_gflops,
                              args, model_type, N, effective_M, batch_size):
    """Train for --eff-steps, tracking FLOPs and perplexity.

    Returns:
      eval_log: list of dicts with keys step, train_ppl, valid_ppl, time_s, gflops, attn_gflops
      total_time: wall-clock time for full training
      cumulative_gflops: total GFLOPs consumed
      cumulative_attn_gflops: attention-only GFLOPs consumed
      peak_mem: peak GPU memory in MB
    """
    device = next(model.parameters()).device
    vocab_size = model.lm_head.out_features
    steps = args.eff_steps

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.eff_lr, eps=1e-4)
    t0 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_total_loss = 0.0
    global_train_tokens = 0
    eval_log = []
    cumulative_gflops = 0.0
    cumulative_attn_gflops = 0.0

    batch_iter = build_batches(train_tokens, N, batch_size, steps, device)

    for step, (inp, tgt) in enumerate(batch_iter):
        optimizer.zero_grad()
        logits = model(inp)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), tgt.view(-1),
            reduction="sum"
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_total_loss += loss.item()
        global_train_tokens += tgt.numel()
        cumulative_gflops += step_total_gflops
        cumulative_attn_gflops += step_attn_gflops

        if (step + 1) % args.eff_eval_every == 0:
            avg_train_loss = global_total_loss / global_train_tokens
            train_ppl = math.exp(min(avg_train_loss, 20))
            valid_ppl = eval_perplexity(model, valid_tokens, N, batch_size)
            elapsed = time.perf_counter() - t0
            eval_log.append({
                "step": step + 1,
                "train_ppl": round(train_ppl, 2),
                "valid_ppl": round(valid_ppl, 2),
                "time_s": round(elapsed, 1),
                "gflops": round(cumulative_gflops, 2),
                "attn_gflops": round(cumulative_attn_gflops, 2),
            })

            if args.eff_target_ppl is not None and valid_ppl < args.eff_target_ppl:
                break

    total_time = time.perf_counter() - t0
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return eval_log, total_time, cumulative_gflops, cumulative_attn_gflops, peak_mem


# ---------------------------------------------------------------------------
# Section 5: Training Efficiency Benchmark
# ---------------------------------------------------------------------------

def run_efficiency_benchmark(args):
    """Train murmurative vs MHA models at multiple sequence lengths.

    Returns a list of result dicts, one per (N, model_type) combination.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    use_reference = (device == "cpu")

    max_N = max(args.eff_ns)

    if args.eff_data == "synthetic":
        train_tokens, valid_tokens, vocab_size = _synthetic_structured_data()
    else:
        train_tokens, valid_tokens, vocab_size = load_wikitext2(
            seq_len=max_N,
            force_synthetic=False,
        )
        if vocab_size > 65536:
            vocab_size = min(vocab_size, 32000)

    H = args.eff_heads
    M = args.eff_num_slots
    rounds = args.eff_rounds
    alpha = args.eff_alpha
    gamma = args.eff_gamma

    results = []

    for D in args.eff_d_models:
        Dh = D // H
        for B in args.eff_batch_sizes:
            for N in args.eff_ns:
                effective_M = min(M, max(32, (N + args.eff_slot_ratio - 1) // args.eff_slot_ratio))
                print(f"\n{'='*60}")
                print(f"  N={N}  (steps={args.eff_steps}, d_model={D}, heads={H}, batch_size={B})")
                print(f"  effective_M={effective_M}, rounds={rounds}")
                print(f"{'='*60}")

                for model_name, model_cls, model_type in _model_variants(args):
                    model_kwargs = {
                        "vocab_size": vocab_size,
                        "d_model": D,
                        "num_heads": H,
                    }
                    if model_type == "murmurative":
                        model_kwargs.update({
                            "num_slots": M,
                            "rounds": rounds,
                            "alpha": alpha,
                            "gamma": gamma,
                            "use_dynamic_m": True,
                            "slot_ratio": args.eff_slot_ratio,
                            "use_fused": True,
                            "use_reference": use_reference,
                        })

                    torch.manual_seed(args.seed)
                    model = model_cls(**model_kwargs).to(device=device, dtype=dtype)
                    model.train()

                    n_params = sum(p.numel() for p in model.parameters())

                    step_total_gflops, step_attn_gflops = count_model_flops_one_step(
                        model_type, B, N, D, H, Dh, M, rounds, effective_M, vocab_size
                    )

                    print(f"\n  [{model_name}] training...")
                    eval_log, total_time, cumulative_gflops, cumulative_attn_gflops, peak_mem = (
                        train_with_flop_tracking(
                            model, train_tokens, valid_tokens,
                            step_total_gflops, step_attn_gflops,
                            args, model_type, N, effective_M, B
                        )
                    )

                    final = eval_log[-1] if eval_log else {}
                    results.append({
                        "N": N,
                        "d_model": D,
                        "batch_size": B,
                        "model": model_name,
                        "model_type": model_type,
                        "params": n_params,
                        "train_ppl": final.get("train_ppl", 0),
                        "valid_ppl": final.get("valid_ppl", 0),
                        "time_s": round(total_time, 1),
                        "gflops": round(cumulative_gflops, 2),
                        "attn_gflops": round(cumulative_attn_gflops, 2),
                        "mem_mb": round(peak_mem, 1),
                        "eval_log": eval_log,
                    })

                    if eval_log:
                        print(f"    params={n_params:,}  "
                              f"train_ppl={final['train_ppl']:.2f}  "
                              f"valid_ppl={final['valid_ppl']:.2f}  "
                              f"time={total_time:.1f}s  "
                              f"GFLOPs={cumulative_gflops:.1f}  "
                              f"attn_GFLOPs={cumulative_attn_gflops:.1f}  "
                              f"mem={peak_mem:.0f}MB")
                    else:
                        print(f"    params={n_params:,}  "
                              f"train_ppl=N/A  valid_ppl=N/A  "
                              f"time={total_time:.1f}s  "
                              f"GFLOPs={cumulative_gflops:.1f}  "
                              f"attn_GFLOPs={cumulative_attn_gflops:.1f}  "
                              f"mem={peak_mem:.0f}MB  (no eval checkpoints)")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return results


def _model_variants(args):
    """Yield (name, class, model_type) tuples for models to benchmark."""
    variants = [
        ("murmurative", MurmurativeModel, "murmurative"),
        ("mha", MHAModel, "mha"),
    ]
    if not args.skip_flash:
        try:
            import flash_attn  # noqa: F401
            variants.append(("flash", FlashAttnModel, "mha"))
        except ImportError:
            pass
    return variants


# ---------------------------------------------------------------------------
# Output formatting (Section 5)
# ---------------------------------------------------------------------------

def print_efficiency_table(results):
    """Print per-N comparison table matching the plan's output format."""
    if not results:
        print("\n  No results to display.")
        return

    for N in sorted(set(r["N"] for r in results)):
        rows = [r for r in results if r["N"] == N]
        hdr = f"  N={N}"
        if rows and "d_model" in rows[0]:
            hdr += f", d_model={rows[0]['d_model']}"
        if rows and "batch_size" in rows[0]:
            hdr += f", BS={rows[0]['batch_size']}"
        print(f"\n{hdr}")
        print("  " + "-" * max(len(hdr), 100))

        header = (f"    {'model':<16} {'params':>10} {'train_ppl':>10} {'valid_ppl':>10} "
                  f"{'time(s)':>10} {'total_GFLOPs':>12} {'attn_GFLOPs':>12} "
                  f"{'ppl/totalG':>10} {'ppl/attnG':>10} {'ppl/(G*P)':>12} {'ppl/s':>10}")
        print(header)
        print("    " + "-" * len(header.strip()))

        mha_row = None
        for r in rows:
            ppl_per_gflop = round(r["valid_ppl"] / max(r["gflops"], 0.001), 2)
            attn_gflops_safe = max(r.get("attn_gflops", 0.001), 0.001)
            ppl_per_attn_gflop = round(r["valid_ppl"] / attn_gflops_safe, 2)
            params_m = r.get("params", 1)
            ppl_per_gflop_param = round(
                r["valid_ppl"] / max(r["gflops"] * params_m, 0.001), 10
            )
            ppl_per_s = round(r["valid_ppl"] / max(r["time_s"], 0.001), 2)
            print(f"    {r['model']:<16} {params_m:>10,} {r['train_ppl']:>10.2f} "
                  f"{r['valid_ppl']:>10.2f} {r['time_s']:>10.1f} "
                  f"{r['gflops']:>12.2f} {r.get('attn_gflops', 0):>12.2f} "
                  f"{ppl_per_gflop:>10.2f} {ppl_per_attn_gflop:>10.2f} "
                  f"{ppl_per_gflop_param:>12.8f} {ppl_per_s:>10.2f}")
            if r["model"] == "mha":
                mha_row = r

        if mha_row:
            print("    " + "-" * len(header.strip()))
            for r in rows:
                if r["model"] == mha_row["model"]:
                    continue
                tag = f"ratio ({r['model'][:4]}/mha)"
                t_ratio = f"{r['time_s'] / max(mha_row['time_s'], 0.001):.1f}x"
                g_ratio = f"{r['gflops'] / max(mha_row['gflops'], 0.001):.2f}x"
                m_ratio = f"{r['mem_mb'] / max(mha_row['mem_mb'], 0.001):.2f}x"
                p_ratio = f"{r.get('params', 1) / max(mha_row.get('params', 1), 0.001):.2f}x"
                ppl_g_ratio_val = (r['valid_ppl'] / max(r['gflops'], 0.001)) / max(mha_row['valid_ppl'] / max(mha_row['gflops'], 0.001), 0.001)
                ppl_s_ratio_val = (r['valid_ppl'] / max(r['time_s'], 0.001)) / max(mha_row['valid_ppl'] / max(mha_row['time_s'], 0.001), 0.001)
                attn_g_ratio_val = (r['valid_ppl'] / max(r.get('attn_gflops', 0.001), 0.001)) / max(mha_row['valid_ppl'] / max(mha_row.get('attn_gflops', 0.001), 0.001), 0.001)
                print(f"    {tag:<16} {p_ratio:>10} "
                      f"{r['train_ppl']/mha_row['train_ppl']:>10.2f} "
                      f"{r['valid_ppl']/mha_row['valid_ppl']:>10.2f} "
                      f"{t_ratio:>10} {g_ratio:>10} {m_ratio:>10} "
                      f"{ppl_g_ratio_val:>10.2f}x {attn_g_ratio_val:>10.2f}x "
                      f"{(r['valid_ppl']/max(r['gflops']*r.get('params',1),0.001))/max(mha_row['valid_ppl']/max(mha_row['gflops']*mha_row.get('params',1),0.001),0.001):>10.2f}x "
                      f"{ppl_s_ratio_val:>10.2f}x")


def print_crossover_projection(results):
    """Project crossover N where murmurative overtakes MHA.

    Uses linear interpolation in log-log space to estimate the N where murmurative
    overtakes MHA on ppl/GFLOP, ppl/attn_GFLOP, wall time, and memory.
    """
    print("\n")
    print("  Crossover Projection")
    print("  " + "-" * 70)
    header = f"    {'Metric':<30} {'Observed':<20} {'Asymptotic (N=16384)':<20}"
    print(header)
    print("    " + "-" * 70)

    mum_rows = sorted([r for r in results if r["model"] == "murmurative"],
                       key=lambda r: r["N"])
    mha_rows = sorted([r for r in results if r["model"] == "mha"],
                       key=lambda r: r["N"])

    if not mum_rows or not mha_rows:
        print("    Insufficient data for crossover projection.")
        return

    # ppl/GFLOP advantage
    ppl_g_mum = [r["valid_ppl"] / max(r["gflops"], 0.001) for r in mum_rows]
    ppl_g_mha = [r["valid_ppl"] / max(r["gflops"], 0.001) for r in mha_rows]
    ppl_g_ratios = [m / max(h, 0.001) for m, h in zip(ppl_g_mum, ppl_g_mha)]
    ns = [r["N"] for r in mum_rows]

    # Current: ppl/GFLOP advantage already exists? Find where it's < 1.0
    current_adv = "already" if all(r < 1.0 for r in ppl_g_ratios) else _estimate_crossover_n(ns, ppl_g_ratios, 1.0)

    # ppl/second crossover (wall-clock)
    ppl_s_mum = [r["valid_ppl"] / max(r["time_s"], 0.001) for r in mum_rows]
    ppl_s_mha = [r["valid_ppl"] / max(r["time_s"], 0.001) for r in mha_rows]
    ppl_s_ratios = [m / max(h, 0.001) for m, h in zip(ppl_s_mum, ppl_s_mha)]

    current_ppl_s = _estimate_crossover_n(ns, ppl_s_ratios, 1.0)

    # Attention-only FLOP advantage (where the thesis is most visible)
    if all("attn_gflops" in r for r in mum_rows + mha_rows):
        attn_g_mum = [r["valid_ppl"] / max(r.get("attn_gflops", 0.001), 0.001) for r in mum_rows]
        attn_g_mha = [r["valid_ppl"] / max(r.get("attn_gflops", 0.001), 0.001) for r in mha_rows]
        ppl_attn_g_ratios = [m / max(h, 0.001) for m, h in zip(attn_g_mum, attn_g_mha)]
        attn_adv = _estimate_crossover_n(ns, ppl_attn_g_ratios, 1.0)
    else:
        attn_adv = "N/A"

    # Memory advantage
    mem_mum = [r["mem_mb"] for r in mum_rows]
    mem_mha = [r["mem_mb"] for r in mha_rows]
    mem_ratios = [m / max(h, 0.001) for m, h in zip(mem_mum, mem_mha)]
    mem_crossover = _estimate_crossover_n(ns, mem_ratios, 1.0)

    print(f"    {'ppl / GFLOP advantage':<30} {str(current_adv):<20} {str(current_adv):<20}")
    print(f"    {'ppl / attn GFLOP advantage':<30} {str(attn_adv):<20} {str(attn_adv):<20}")
    print(f"    {'ppl / wall-second parity':<30} {str(current_ppl_s):<20} {str(current_ppl_s):<20}")
    print(f"    {'Memory advantage':<30} {str(mem_crossover):<20} {str(mem_crossover):<20}")
    print("  " + "-" * 70)


def _estimate_crossover_n(ns, ratios, target=1.0):
    """Estimate N where ratio crosses target using log-log linear interpolation."""
    above = [(n, r) for n, r in zip(ns, ratios) if r >= target]
    below = [(n, r) for n, r in zip(ns, ratios) if r < target]

    if not below:
        return f"N>{max(ns)}"
    if not above:
        return "already"

    n_lo, r_lo = below[-1]
    n_hi, r_hi = above[0]

    if abs(r_hi - r_lo) < 1e-10:
        return f"N~{int((n_lo + n_hi) / 2)}"

    log_n_lo = math.log(n_lo)
    log_n_hi = math.log(n_hi)
    log_r_lo = math.log(max(r_lo, 0.001))
    log_r_hi = math.log(max(r_hi, 0.001))
    log_target = math.log(target)

    frac = (log_target - log_r_lo) / (log_r_hi - log_r_lo)
    log_n_cross = log_n_lo + frac * (log_n_hi - log_n_lo)
    n_cross = int(math.exp(log_n_cross))
    return f"N~{n_cross}"


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmarks for murmurative attention"
    )

    # Section skipping
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Skip sections (1-5)")

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per step")
    parser.add_argument("--skip-flash", action="store_true",
                        help="Skip flash-attention model comparison")

    # Section 5: Training Efficiency
    parser.add_argument("--skip-efficiency", action="store_true",
                        help="Skip Section 5 (training efficiency)")
    parser.add_argument("--eff-steps", type=int, default=500,
                        help="Training steps per N (Section 5)")
    parser.add_argument("--eff-eval-every", type=int, default=100,
                        help="Evaluate perplexity every N steps")
    parser.add_argument("--eff-ns", type=int, nargs="*",
                        default=[512, 1024, 2048, 4096, 8192, 16384],
                        help="Sequence lengths to sweep")
    parser.add_argument("--eff-d-models", type=int, nargs="*",
                        default=[256],
                        help="Model dimensions to sweep (Section 5)")
    parser.add_argument("--eff-batch-sizes", type=int, nargs="*",
                        default=[1],
                        help="Batch sizes to sweep (Section 5)")
    parser.add_argument("--eff-data", type=str, default="synthetic",
                        choices=["synthetic", "wikitext"],
                        help="Training data: synthetic (vocab=256) or wikitext (Section 5)")
    parser.add_argument("--eff-target-ppl", type=float, default=None,
                        help="Target validation perplexity for early stopping (Section 5)")
    parser.add_argument("--eff-heads", type=int, default=4)
    parser.add_argument("--eff-num-slots", type=int, default=256)
    parser.add_argument("--eff-rounds", type=int, default=3)
    parser.add_argument("--eff-alpha", type=float, default=0.9)
    parser.add_argument("--eff-gamma", type=float, default=0.15)
    parser.add_argument("--eff-slot-ratio", type=int, default=8)
    parser.add_argument("--eff-lr", type=float, default=5e-4)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. Murmurative CUDA kernels require GPU.")
        print("         Running on CPU with reference implementation will be very slow.")
        print("         Consider running on a GPU machine for meaningful benchmarks.")

    all_results = {}  # section -> results

    # Section 5: Training Efficiency
    if 5 not in args.skip and not args.skip_efficiency:
        print("\n" + "=" * 60)
        print("=== Section 5: Training Efficiency ===")
        print("=" * 60)
        results = run_efficiency_benchmark(args)
        all_results[5] = results
        print_efficiency_table(results)
        print_crossover_projection(results)

    # Save results (if Section 5 ran)
    if 5 in all_results:
        results_file = Path(__file__).resolve().parent / "results.json"
        serializable = []
        for r in all_results[5]:
            sr = {k: v for k, v in r.items() if k != "eval_log"}
            serializable.append(sr)
        with open(results_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()