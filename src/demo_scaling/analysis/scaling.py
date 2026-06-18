"""基础 scaling law 图表。

第一版先输出 loss-vs-params、loss-vs-tokens、IsoFLOP 散点。
有足够点数后再拟合幂律指数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import ensure_dir


def plot_loglog(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.scatter(df[x], df[y])
    for _, row in df.iterrows():
        plt.annotate(str(row["run_id"])[:18], (row[x], row[y]), fontsize=7)
    plt.xscale("log")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def fit_power_law(df: pd.DataFrame, x: str, y: str) -> dict:
    clean = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[(clean[x] > 0) & (clean[y] > 0)]
    if len(clean) < 3:
        return {"status": "need_at_least_3_points"}
    if clean[x].nunique() < 3:
        return {"status": "need_at_least_3_unique_x_values"}
    coeff = np.polyfit(np.log(clean[x]), np.log(clean[y]), 1)
    return {"status": "ok", "slope": float(coeff[0]), "intercept": float(coeff[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scaling-law plots.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--output", default="report/scaling")
    args = parser.parse_args()
    out = ensure_dir(args.output)
    df = pd.read_csv(args.runs)
    for col in ["params_non_embedding", "tokens_seen", "flops_seen", "final_val_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["final_val_loss"])
    if df.empty:
        print("no completed runs")
        return
    plot_loglog(df, "params_non_embedding", "final_val_loss", out / "loss_vs_params.png", "Loss vs effective params")
    plot_loglog(df, "tokens_seen", "final_val_loss", out / "loss_vs_tokens.png", "Loss vs tokens")
    plot_loglog(df, "flops_seen", "final_val_loss", out / "loss_vs_flops.png", "Loss vs FLOPs")
    summary = {
        "loss_vs_params": fit_power_law(df, "params_non_embedding", "final_val_loss"),
        "loss_vs_tokens": fit_power_law(df, "tokens_seen", "final_val_loss"),
        "loss_vs_flops": fit_power_law(df, "flops_seen", "final_val_loss"),
    }
    (out / "fit_summary.json").write_text(pd.Series(summary).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote scaling plots -> {out}")


if __name__ == "__main__":
    main()
