"""Minimal IsoFLOP analysis using params_total and tail-smoothed loss."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(runs_path: Path, matrix_path: Path | None, loss_column: str) -> pd.DataFrame:
    runs = pd.read_csv(runs_path)
    matrix = pd.read_csv(matrix_path) if matrix_path and matrix_path.exists() else pd.DataFrame()
    if not matrix.empty:
        keep = [c for c in ["run_id", "target_flops", "mode", "source"] if c in matrix.columns]
        runs = runs.merge(matrix[keep], on="run_id", how="inner", suffixes=("", "_matrix"))
        if "target_flops_matrix" in runs.columns:
            runs["flops_budget"] = pd.to_numeric(runs["target_flops_matrix"], errors="coerce")
        else:
            runs["flops_budget"] = pd.to_numeric(runs["target_flops"], errors="coerce")
    else:
        runs["flops_budget"] = pd.to_numeric(runs["target_flops"], errors="coerce")
    runs["params_total"] = pd.to_numeric(runs["params_total"], errors="coerce")
    runs["tokens_seen"] = pd.to_numeric(runs["tokens_seen"], errors="coerce")
    runs[loss_column] = pd.to_numeric(runs[loss_column], errors="coerce")
    return runs.dropna(subset=["flops_budget", "params_total", "tokens_seen", loss_column]).copy()


def fit_frontier(df: pd.DataFrame, loss_column: str) -> pd.DataFrame:
    rows = []
    for c, part in df.groupby("flops_budget"):
        part = part.sort_values("params_total")
        status = "insufficient_points"
        opt_n = opt_d = opt_loss = np.nan
        within = False
        if len(part) >= 3 and part["params_total"].nunique() >= 3:
            x = np.log10(part["params_total"].to_numpy())
            y = part[loss_column].to_numpy()
            a, b, cc = np.polyfit(x, y, 2)
            if a > 0:
                x0 = -b / (2 * a)
                opt_n = 10 ** x0
                opt_loss = a * x0 * x0 + b * x0 + cc
                within = bool(part["params_total"].min() <= opt_n <= part["params_total"].max())
                status = "quadratic_fit" if within else "quadratic_fit_extrapolated"
            else:
                raw = part.loc[part[loss_column].idxmin()]
                opt_n, opt_loss = raw["params_total"], raw[loss_column]
                status = "raw_min_nonconvex_fit"
                within = True
        else:
            raw = part.loc[part[loss_column].idxmin()]
            opt_n, opt_loss = raw["params_total"], raw[loss_column]
            status = "raw_min"
            within = True
        if not np.isnan(opt_n):
            # Report D from observed closest run if formula is not exact in logs.
            closest = part.iloc[(part["params_total"] - opt_n).abs().argmin()]
            opt_d = closest["tokens_seen"]
        rows.append({"flops_budget": c, "num_points": len(part), "status": status, "opt_params": opt_n, "opt_tokens": opt_d, "opt_loss": opt_loss, "opt_within_observed_range": within})
    return pd.DataFrame(rows).sort_values("flops_budget")


def fit_exponents(frontier: pd.DataFrame) -> pd.DataFrame:
    valid = frontier[(frontier["status"] == "quadratic_fit") & frontier["opt_within_observed_range"]].copy()
    if len(valid) < 2:
        return pd.DataFrame([{"num_budgets": len(valid), "status": "insufficient_valid_budgets"}])
    x = np.log10(valid["flops_budget"].to_numpy())
    ncoef = np.polyfit(x, np.log10(valid["opt_params"].to_numpy()), 1)
    dcoef = np.polyfit(x, np.log10(valid["opt_tokens"].to_numpy()), 1)
    return pd.DataFrame([{"num_budgets": len(valid), "status": "ok", "n_exponent_a": ncoef[0], "n_intercept": ncoef[1], "d_exponent_b": dcoef[0], "d_intercept": dcoef[1], "ratio_exponent_b_minus_a": dcoef[0] - ncoef[0]}])


def plot_overall(df: pd.DataFrame, frontier: pd.DataFrame, loss_column: str, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    for c, part in df.groupby("flops_budget"):
        part = part.sort_values("params_total")
        plt.plot(part["params_total"], part[loss_column], "o-", label=f"C={c:.0e}")
    plt.xscale("log")
    plt.xlabel("N = params_total")
    plt.ylabel(loss_column)
    plt.title("IsoFLOP loss vs total parameters")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "isoflop_overall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for c, part in df.groupby("flops_budget"):
        part = part.sort_values("tokens_seen")
        plt.plot(part["tokens_seen"], part[loss_column], "o-", label=f"C={c:.0e}")
    plt.xscale("log")
    plt.xlabel("D = tokens_seen")
    plt.ylabel(loss_column)
    plt.title("IsoFLOP loss vs tokens")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "isoflop_by_tokens.png", dpi=180)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit v2 IsoFLOP curves.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--matrix", default="")
    parser.add_argument("--loss-column", default="tail_smoothed_val_loss")
    parser.add_argument("--output", default="plot_fit/outputs/isoflop")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.runs), Path(args.matrix) if args.matrix else None, args.loss_column)
    df.to_csv(out / "isoflop_loss_table.csv", index=False)
    frontier = fit_frontier(df, args.loss_column)
    frontier.to_csv(out / "isoflop_frontier.csv", index=False)
    exp = fit_exponents(frontier)
    exp.to_csv(out / "scaling_exponents.csv", index=False)
    plot_overall(df, frontier, args.loss_column, out)
    (out / "summary.md").write_text("# IsoFLOP Summary\n\n" + dataframe_to_markdown(frontier) + "\n\n" + dataframe_to_markdown(exp) + "\n", encoding="utf-8")
    print(f"wrote IsoFLOP analysis -> {out}")


if __name__ == "__main__":
    main()
