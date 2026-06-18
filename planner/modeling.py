from __future__ import annotations

import math

VOCAB_SIZE = 50257
HEAD_DIM = 128
ASPECT_RATIO = 64
BLOCK_SIZE = 1024


def depth_config(depth: int, head_dim: int = HEAD_DIM, aspect_ratio: int = ASPECT_RATIO) -> dict:
    d_model = math.ceil(depth * aspect_ratio / head_dim) * head_dim
    return {"depth": depth, "L": depth, "d_model": d_model, "h": d_model // head_dim, "head_dim": head_dim}


def param_plan(depth: int, vocab_size: int = VOCAB_SIZE, block_size: int = BLOCK_SIZE) -> dict:
    cfg = depth_config(depth)
    d = cfg["d_model"]
    L = cfg["L"]
    n_dense = 12 * L * d * d
    n_layernorm = (2 * L + 1) * d
    n_block = n_dense + n_layernorm
    n_vocab = vocab_size * d
    n_total = n_block + n_vocab
    flops_6n = 6 * n_total
    flops_nanogpt = flops_6n + 12 * L * d * block_size
    return {**cfg, "block_size": block_size, "vocab_size": vocab_size, "N_dense": n_dense, "N_layernorm": n_layernorm, "N_block": n_block, "N_vocab": n_vocab, "N_total": n_total, "flops_per_token_6N": flops_6n, "flops_per_token_nanogpt": flops_nanogpt}


def human(n: int | float) -> str:
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n/1e9:.3f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.3f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.3f}K"
    return f"{n:.0f}"
