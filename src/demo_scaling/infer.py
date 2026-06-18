"""One-shot text generation from a v2 checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
import tiktoken

from .model import GPT, GPTConfig


def load_model(path: str | Path, device: torch.device) -> GPT:
    ckpt = torch.load(path, map_location=device)
    cfg_raw = ckpt["model_config"]
    cfg_keys = {f.name for f in fields(GPTConfig)}
    cfg = GPTConfig(**{k: v for k, v in cfg_raw.items() if k in cfg_keys})
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.checkpoint, device)
    idx = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
