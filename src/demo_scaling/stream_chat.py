"""Interactive streaming generation for manual checkpoint inspection."""

from __future__ import annotations

import argparse
import sys

import torch
import tiktoken

from .infer import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive streaming generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.checkpoint, device)
    print("type prompt; Ctrl-D to exit")
    for line in sys.stdin:
        prompt = line.rstrip("\n")
        if not prompt:
            continue
        ids = enc.encode_ordinary(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        print(prompt, end="", flush=True)
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                idx_cond = idx[:, -model.config.block_size :]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / max(args.temperature, 1e-6)
                if args.top_k > 0:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                idx = torch.cat((idx, next_id), dim=1)
                piece = enc.decode([int(next_id.item())])
                print(piece, end="", flush=True)
        print("\n---")


if __name__ == "__main__":
    main()
