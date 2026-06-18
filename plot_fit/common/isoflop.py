from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_isoflop, ensure_expected_counts
from .fit import LOSS_COL, build_frontier, scaling_exponents
from .style import COLORS, REFINED_SOURCES, apply_style, savefig

MIXED_MATRICES = [
    "isoflop_mixed_coarse.csv",
    "isoflop_mixed_refineA_low_depth.csv",
    "isoflop_mixed_refineB_c1e18_low_depth.csv",
]
SOURCE_MATRICES = [
    "isoflop_sources_slim_story_code_encyclopedia.csv",
    "isoflop_sources_refine_boundary.csv",
    "isoflop_sources_refine2_c1e17_low_depth.csv",
]


def run_mixed_analysis(project: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_isoflop(project, MIXED_MATRICES, sources=["mixed"])
    ensure_expected_counts(mixed_iso=df)
    df = df.sort_values(["flops_budget", "params_total"])
    df.to_csv(out_dir / "mixed_isoflop_loss_table.csv", index=False)
    frontier, residuals, fit_report = build_frontier(df, ["flops_budget"])
    exponents, power_report = scaling_exponents(frontier, None)
    frontier.to_csv(out_dir / "mixed_fitted_frontier.csv", index=False)
    residuals.to_csv(out_dir / "mixed_fit_residuals.csv", index=False)
    exponents.to_csv(out_dir / "mixed_scaling_exponents.csv", index=False)
    (out_dir / "mixed_lmfit_report.md").write_text(
        "# Mixed IsoFLOP lmfit Report\n\n"
        "主指标为 tail-smoothed full validation loss。每个固定 C 对 L(log10 N) 与 L(log10 D) 分别做二次拟合。\n\n"
        + fit_report + "\n" + power_report,
        encoding="utf-8",
    )
    return df, frontier, exponents


def run_source_analysis(project: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_isoflop(project, SOURCE_MATRICES, sources=REFINED_SOURCES)
    counts = df.groupby("source")["run_id"].count().to_dict()
    for source in REFINED_SOURCES:
        if counts.get(source, 0) == 0:
            raise RuntimeError(f"no refined source IsoFLOP runs for {source}")
    df = df.sort_values(["source", "flops_budget", "params_total"])
    df.to_csv(out_dir / "sources_isoflop_loss_table.csv", index=False)
    frontier, residuals, fit_report = build_frontier(df, ["source", "flops_budget"])
    exponents, power_report = scaling_exponents(frontier, "source")
    frontier.to_csv(out_dir / "sources_fitted_frontier.csv", index=False)
    residuals.to_csv(out_dir / "sources_fit_residuals.csv", index=False)
    exponents.to_csv(out_dir / "sources_scaling_exponents.csv", index=False)
    (out_dir / "sources_lmfit_report.md").write_text(
        "# Source IsoFLOP lmfit Report\n\n"
        "仅纳入 refined 完整度最高的 story/code/encyclopedia。主指标为 tail-smoothed full validation loss。\n\n"
        + fit_report + "\n" + power_report,
        encoding="utf-8",
    )
    return df, frontier, exponents


def _ensure_mixed(project: Path, out_dir: Path):
    return run_mixed_analysis(project, out_dir)


def _ensure_sources(project: Path, out_dir: Path):
    return run_source_analysis(project, out_dir)


def _format_c(c: float) -> str:
    return f"C={c:.0e}"


def plot_mixed_loss_vs_params(project: Path, out_path: Path) -> None:
    out_dir = out_path.parent
    df, frontier, _ = _ensure_mixed(project, out_dir)
    apply_style()
    plt.figure(figsize=(9.2, 6.2))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, frontier["flops_budget"].nunique()))
    for color, (c, part) in zip(cmap, df.groupby("flops_budget")):
        part = part.sort_values("params_total")
        plt.plot(part["params_total"], part[LOSS_COL], marker="o", linewidth=2.0, color=color, label=_format_c(c))
        for _, row in part.iterrows():
            plt.annotate(f"d{int(row['depth'])}", (row["params_total"], row[LOSS_COL]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for _, row in frontier.iterrows():
        observed = row["status"] == "quadratic_fit_observed"
        plt.scatter([row["opt_params"]], [row["opt_loss"]], marker="*", s=220 if observed else 160, c="#ffd23f" if observed else "none", edgecolors="black" if observed else "crimson", linewidths=1.2, zorder=5)
    plt.xscale("log")
    plt.xlabel("N = total parameters")
    plt.ylabel("tail-smoothed full validation loss")
    plt.title("Mixed IsoFLOP U-shaped curves: loss vs model size")
    plt.legend(ncol=2)
    savefig(out_path)


def plot_mixed_loss_vs_tokens(project: Path, out_path: Path) -> None:
    out_dir = out_path.parent
    df, frontier, _ = _ensure_mixed(project, out_dir)
    apply_style()
    plt.figure(figsize=(9.2, 6.2))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.92, frontier["flops_budget"].nunique()))
    for color, (c, part) in zip(cmap, df.groupby("flops_budget")):
        part = part.sort_values("tokens_seen")
        plt.plot(part["tokens_seen"], part[LOSS_COL], marker="o", linewidth=2.0, color=color, label=_format_c(c))
        for _, row in part.iterrows():
            plt.annotate(f"d{int(row['depth'])}", (row["tokens_seen"], row[LOSS_COL]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for _, row in frontier.iterrows():
        observed = row["d_fit_status"] == "quadratic_fit_observed"
        plt.scatter([row["opt_tokens_direct"]], [row["opt_loss_direct_d"]], marker="*", s=220 if observed else 160, c="#ffd23f" if observed else "none", edgecolors="black" if observed else "crimson", linewidths=1.2, zorder=5)
    plt.xscale("log")
    plt.xlabel("D = tokens seen")
    plt.ylabel("tail-smoothed full validation loss")
    plt.title("Mixed IsoFLOP U-shaped curves: loss vs token budget")
    plt.legend(ncol=2)
    savefig(out_path)


def _plot_power(frontier: pd.DataFrame, exponents: pd.DataFrame, y_col: str, out_path: Path, ylabel: str, title: str, source_col: str | None = None) -> None:
    apply_style()
    plt.figure(figsize=(8.4, 5.8))
    if source_col:
        groups = frontier.groupby(source_col)
    else:
        groups = [("mixed", frontier)]
    for source, part in groups:
        color = COLORS.get(str(source), "#111111")
        status_col = "status" if y_col == "opt_params" else "d_fit_status"
        obs = part[part[status_col] == "quadratic_fit_observed"].sort_values("flops_budget")
        diag = part[part[status_col] != "quadratic_fit_observed"].sort_values("flops_budget")
        all_points = part.sort_values("flops_budget")
        if not all_points.empty:
            plt.scatter(
                all_points["flops_budget"],
                all_points[y_col],
                marker="o",
                s=54,
                color=color,
                alpha=0.95,
                label=str(source),
                zorder=4,
            )
            erow = exponents[exponents["source"].astype(str) == str(source)]
            if not erow.empty and len(obs) >= 2:
                if y_col == "opt_params":
                    slope = float(erow.iloc[0]["alpha_for_N_opt"]); intercept = float(erow.iloc[0]["n_intercept"])
                    label = f"{source} fit: alpha={slope:.3f}"
                else:
                    slope = float(erow.iloc[0]["beta_for_D_opt_direct"]); intercept = float(erow.iloc[0]["d_intercept"])
                    label = f"{source} fit: beta={slope:.3f}"
                grid = np.logspace(np.log10(obs["flops_budget"].min()), np.log10(obs["flops_budget"].max()), 100)
                plt.plot(grid, 10 ** (slope * np.log10(grid) + intercept), "--", color=color, alpha=0.85, label=label)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("C = FLOPs budget")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    savefig(out_path)


def plot_mixed_nopt_vs_flops(project: Path, out_path: Path) -> None:
    _, frontier, exponents = _ensure_mixed(project, out_path.parent)
    _plot_power(frontier, exponents, "opt_params", out_path, "N_opt = total parameters", "Mixed compute-optimal model size")


def plot_mixed_dopt_vs_flops(project: Path, out_path: Path) -> None:
    _, frontier, exponents = _ensure_mixed(project, out_path.parent)
    _plot_power(frontier, exponents, "opt_tokens_direct", out_path, "D_opt = directly fitted tokens", "Mixed compute-optimal token budget")


def plot_mixed_powerlaw_combined(project: Path, out_path: Path) -> None:
    _, frontier, exponents = _ensure_mixed(project, out_path.parent)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    erow = exponents.iloc[0]
    for ax, y_col, ylabel, slope_name, intercept_name, slope_label in [
        (axes[0], "opt_params", "N_opt", "alpha_for_N_opt", "n_intercept", "alpha"),
        (axes[1], "opt_tokens_direct", "D_opt", "beta_for_D_opt_direct", "d_intercept", "beta"),
    ]:
        status_col = "status" if y_col == "opt_params" else "d_fit_status"
        obs = frontier[frontier[status_col] == "quadratic_fit_observed"].sort_values("flops_budget")
        all_points = frontier.sort_values("flops_budget")
        ax.scatter(
            all_points["flops_budget"],
            all_points[y_col],
            marker="o",
            s=58,
            color="#111111",
            alpha=0.95,
            label="fitted optima",
            zorder=4,
        )
        slope = float(erow[slope_name]); intercept = float(erow[intercept_name])
        grid = np.logspace(np.log10(obs["flops_budget"].min()), np.log10(obs["flops_budget"].max()), 100)
        ax.plot(grid, 10 ** (slope * np.log10(grid) + intercept), "--", color="#d62728", label=f"{slope_label}={slope:.3f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("C = FLOPs budget")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Mixed compute-optimal scaling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_source_loss_vs_params(project: Path, out_path: Path, source: str) -> None:
    df, frontier, _ = _ensure_sources(project, out_path.parent)
    df = df[df["source"] == source]
    frontier = frontier[frontier["source"] == source]
    apply_style()
    plt.figure(figsize=(9.2, 6.2))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, frontier["flops_budget"].nunique()))
    for color, (c, part) in zip(cmap, df.groupby("flops_budget")):
        part = part.sort_values("params_total")
        plt.plot(part["params_total"], part[LOSS_COL], marker="o", linewidth=2.0, color=color, label=_format_c(c))
        for _, row in part.iterrows():
            plt.annotate(f"d{int(row['depth'])}", (row["params_total"], row[LOSS_COL]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for _, row in frontier.iterrows():
        observed = row["status"] == "quadratic_fit_observed"
        plt.scatter([row["opt_params"]], [row["opt_loss"]], marker="*", s=220 if observed else 160, c="#ffd23f" if observed else "none", edgecolors="black" if observed else "crimson", linewidths=1.2, zorder=5)
    plt.xscale("log")
    plt.xlabel("N = total parameters")
    plt.ylabel("tail-smoothed full validation loss")
    plt.title(f"{source} IsoFLOP U-shaped curves: loss vs model size")
    plt.legend(ncol=2)
    savefig(out_path)


def plot_source_loss_vs_tokens(project: Path, out_path: Path, source: str) -> None:
    df, frontier, _ = _ensure_sources(project, out_path.parent)
    df = df[df["source"] == source]
    frontier = frontier[frontier["source"] == source]
    apply_style()
    plt.figure(figsize=(9.2, 6.2))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.92, frontier["flops_budget"].nunique()))
    for color, (c, part) in zip(cmap, df.groupby("flops_budget")):
        part = part.sort_values("tokens_seen")
        plt.plot(part["tokens_seen"], part[LOSS_COL], marker="o", linewidth=2.0, color=color, label=_format_c(c))
        for _, row in part.iterrows():
            plt.annotate(f"d{int(row['depth'])}", (row["tokens_seen"], row[LOSS_COL]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for _, row in frontier.iterrows():
        observed = row["d_fit_status"] == "quadratic_fit_observed"
        plt.scatter([row["opt_tokens_direct"]], [row["opt_loss_direct_d"]], marker="*", s=220 if observed else 160, c="#ffd23f" if observed else "none", edgecolors="black" if observed else "crimson", linewidths=1.2, zorder=5)
    plt.xscale("log")
    plt.xlabel("D = tokens seen")
    plt.ylabel("tail-smoothed full validation loss")
    plt.title(f"{source} IsoFLOP U-shaped curves: loss vs token budget")
    plt.legend(ncol=2)
    savefig(out_path)


def plot_sources_nopt_vs_flops(project: Path, out_path: Path) -> None:
    _, frontier, exponents = _ensure_sources(project, out_path.parent)
    _plot_power(frontier, exponents, "opt_params", out_path, "N_opt = total parameters", "Per-source compute-optimal model size", source_col="source")


def plot_sources_dopt_vs_flops(project: Path, out_path: Path) -> None:
    _, frontier, exponents = _ensure_sources(project, out_path.parent)
    _plot_power(frontier, exponents, "opt_tokens_direct", out_path, "D_opt = directly fitted tokens", "Per-source compute-optimal token budget", source_col="source")


def plot_sources_powerlaw_combined(project: Path, out_path: Path) -> None:
    _, frontier, exponents = _ensure_sources(project, out_path.parent)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for source in REFINED_SOURCES:
        color = COLORS[source]
        part = frontier[frontier["source"] == source]
        erow = exponents[exponents["source"] == source].iloc[0]
        for ax, y_col, status_col, slope_name, intercept_name, slope_label in [
            (axes[0], "opt_params", "status", "alpha_for_N_opt", "n_intercept", "alpha"),
            (axes[1], "opt_tokens_direct", "d_fit_status", "beta_for_D_opt_direct", "d_intercept", "beta"),
        ]:
            obs = part[part[status_col] == "quadratic_fit_observed"].sort_values("flops_budget")
            if obs.empty:
                continue
            all_points = part.sort_values("flops_budget")
            ax.scatter(
                all_points["flops_budget"],
                all_points[y_col],
                marker="o",
                s=52,
                color=color,
                alpha=0.95,
                label=f"{source}",
                zorder=4,
            )
            slope = float(erow[slope_name]); intercept = float(erow[intercept_name])
            if len(obs) >= 2:
                grid = np.logspace(np.log10(obs["flops_budget"].min()), np.log10(obs["flops_budget"].max()), 100)
                ax.plot(grid, 10 ** (slope * np.log10(grid) + intercept), "--", color=color, alpha=0.85, label=f"{source} {slope_label}={slope:.3f}")
    axes[0].set_title("N_opt vs C")
    axes[1].set_title("D_opt vs C")
    axes[0].set_ylabel("N_opt = total parameters")
    axes[1].set_ylabel("D_opt = directly fitted tokens")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("C = FLOPs budget")
        ax.legend(fontsize=7)
    fig.suptitle("Per-source compute-optimal scaling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
