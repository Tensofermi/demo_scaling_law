#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from modeling import human, param_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Print/export depth-family model parameter table.")
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    rows = [param_plan(d) for d in args.depths]
    fields = ["depth", "L", "d_model", "h", "head_dim", "block_size", "N_dense", "N_layernorm", "N_block", "N_vocab", "N_total", "flops_per_token_6N", "flops_per_token_nanogpt"]
    print("depth L d_model h N_dense N_layernorm N_block N_vocab N_total")
    for r in rows:
        print(f"{r['depth']:>5} {r['L']:>2} {r['d_model']:>7} {r['h']:>2} {human(r['N_dense']):>10} {human(r['N_layernorm']):>11} {human(r['N_block']):>10} {human(r['N_vocab']):>10} {human(r['N_total']):>10}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
