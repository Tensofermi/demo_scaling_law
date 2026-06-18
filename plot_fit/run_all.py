#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate basic learning curve and run table plots.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--output", default="plot_fit/outputs")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for log in Path("logs").glob("*/train_log.csv"):
        run_id = log.parent.name
        df = pd.read_csv(log)
        if df.empty:
            continue
        df["run_id"] = run_id
        rows.append(df)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        if "val_loss" in df:
            ax.plot(df["iter"], df["val_loss"], label="quick val")
        if "full_val_loss" in df:
            full = df.dropna(subset=["full_val_loss"])
            full = full[full["full_val_loss"].astype(str) != ""]
            if not full.empty:
                ax.plot(full["iter"], full["full_val_loss"].astype(float), "o-", label="full val")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(run_id)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{run_id}_loss.png", dpi=180)
        plt.close(fig)
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        all_df.to_csv(out / "train_logs_combined.csv", index=False)
    print(f"wrote plots -> {out}")


if __name__ == "__main__":
    main()
