#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

from modeling import param_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build experiment matrix CSV from a compact YAML spec.")
    parser.add_argument("--config", default="configs/experiments/smoke.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    rows = []
    for row in cfg.get("rows", []):
        depth = int(row["depth"])
        plan = param_plan(depth)
        out = {
            "run_id": row["run_id"],
            "mode": row.get("mode", "fixed"),
            "source": row.get("source", "mixed"),
            "depth": depth,
            "max_iters": row.get("max_iters", ""),
            "target_flops": row.get("target_flops", ""),
            "target_tokens": "",
            "params_total_planned": plan["N_total"],
            "data_dir": row.get("data_dir", cfg.get("tokenized_root", "train_data/tokenized/gpt2/mixed")),
        }
        if out["target_flops"]:
            out["target_tokens"] = int(float(out["target_flops"]) // max(plan["flops_per_token_6N"], 1))
        rows.append(out)
    if not rows:
        raise SystemExit("matrix config produced zero rows")
    output = Path(args.output or cfg.get("output", "configs/experiments/matrix.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
