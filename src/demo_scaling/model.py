"""RoPE decoder-only GPT used by demo_scaling_law.

The model intentionally stays close to nanoGPT but removes learned positional
embedding. This makes the scaling-law parameter count unambiguous: the main N is
all trainable parameters in the actual model, with tied token embedding / lm
head counted once.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 2
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    use_sdpa: bool = True
    depth: int | None = None
    head_dim: int | None = None
    rope_base: float = 10000.0


def depth_to_config(
    depth: int,
    *,
    vocab_size: int = 50257,
    block_size: int = 1024,
    head_dim: int = 128,
    aspect_ratio: int = 64,
    dropout: float = 0.0,
    bias: bool = False,
    use_sdpa: bool = True,
    rope_base: float = 10000.0,
) -> GPTConfig:
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    d_model = math.ceil(depth * aspect_ratio / head_dim) * head_dim
    n_head = d_model // head_dim
    return GPTConfig(
        vocab_size=int(vocab_size),
        block_size=int(block_size),
        n_layer=int(depth),
        n_head=int(n_head),
        n_embd=int(d_model),
        dropout=float(dropout),
        bias=bool(bias),
        use_sdpa=bool(use_sdpa),
        depth=int(depth),
        head_dim=int(head_dim),
        rope_base=float(rope_base),
    )


def precompute_rope_cache(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [T, D/2]
    seq_len = x.size(-2)
    cos = cos[:seq_len].to(device=x.device, dtype=x.dtype)[None, None, :, :]
    sin = sin[:seq_len].to(device=x.device, dtype=x.dtype)[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_sdpa = config.use_sdpa and hasattr(F, "scaled_dot_product_attention")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        cos, sin = precompute_rope_cache(config.block_size, self.head_dim, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            probs = self.attn_dropout(F.softmax(scores, dim=-1))
            y = probs @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def get_param_breakdown(self) -> dict[str, int]:
        named = dict(self.named_parameters())
        params_total = int(sum(p.numel() for p in named.values()))
        params_token_embedding = int(self.tok_emb.weight.numel())
        params_transformer = int(sum(p.numel() for name, p in named.items() if not name.startswith(("tok_emb", "lm_head"))))
        return {
            "params_total": params_total,
            "params_plan": params_total,
            "params_transformer": params_transformer,
            "params_token_embedding": params_token_embedding,
            "params_pos_embedding": 0,
            "params_lm_head_effective": params_token_embedding,
        }

    def estimate_flops_per_token(self) -> int:
        # Main planning rule: C ~= 6 * N_total * D.
        return int(6 * self.get_param_breakdown()["params_total"])

    def export_model_card(self) -> dict[str, object]:
        return {"config": asdict(self.config), "param_breakdown": self.get_param_breakdown(), "flops_per_token": self.estimate_flops_per_token()}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
