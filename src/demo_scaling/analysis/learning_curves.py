from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val curves for v2 runs.")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--include-prefix", action="append", default=[])
    parser.add_argument("--output", default="report/learning_curves")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for log in sorted(Path(args.logs).glob("*/train_log.csv")):
        run_id = log.parent.name
        if args.include_prefix and not any(run_id.startswith(p) for p in args.include_prefix):
            continue
        df = pd.read_csv(log)
        df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
        df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
        part = df.dropna(subset=["iter", "val_loss"])
        if not part.empty:
            plt.plot(part["iter"], part["val_loss"], label=run_id)
    plt.xlabel("step")
    plt.ylabel("quick validation loss")
    plt.title("Validation loss vs step")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out / "val_loss_vs_step.png", dpi=180)
    plt.close()
    print(f"wrote learning curves -> {out}")


if __name__ == "__main__":
    main()
