from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import final_loss, load_fixed_mixed, load_fixed_sources, ensure_expected_counts
from .style import COLORS, SOURCE_ORDER, apply_style, savefig


def _series(df, y_col: str, x_col: str, loglog: bool = True):
    part = df.dropna(subset=[x_col, y_col]).copy()
    if loglog:
        part = part[(part[x_col] > 0) & (part[y_col] > 0)]
    return part[x_col], part[y_col]


def _label_axis(x_col: str) -> str:
    if x_col == "iter":
        return "training step"
    if x_col == "flops_seen":
        return "FLOPs seen = 6 x params_total x tokens_seen"
    return x_col


def write_fixed_mixed_summary(project: Path, out_dir: Path) -> None:
    runs = load_fixed_mixed(project)
    ensure_expected_counts(fixed_mixed=runs)
    rows = []
    for run in runs:
        df = run["df"]
        last = df.dropna(subset=["iter"]).iloc[-1]
        rows.append({
            "run_id": run["run_id"],
            "depth": run["depth"],
            "params_total": run["params_total"],
            "final_quick_val_loss": final_loss(df, "val_loss"),
            "final_full_val_loss": final_loss(df, "full_val_loss"),
            "tokens_seen": int(last["tokens_seen"]),
            "flops_seen": float(last["flops_seen"]),
            "max_memory_gb": float(df["max_memory_gb"].dropna().max()),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fixed_mixed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def write_fixed_sources_summary(project: Path, out_dir: Path) -> None:
    runs = load_fixed_sources(project)
    ensure_expected_counts(fixed_sources=runs)
    rows = []
    for run in runs:
        df = run["df"]
        last = df.dropna(subset=["iter"]).iloc[-1]
        rows.append({
            "run_id": run["run_id"],
            "source": run["source"],
            "depth": run["depth"],
            "params_total": run["params_total"],
            "final_quick_val_loss": final_loss(df, "val_loss"),
            "final_full_val_loss": final_loss(df, "full_val_loss"),
            "tokens_seen": int(last["tokens_seen"]),
            "flops_seen": float(last["flops_seen"]),
            "max_memory_gb": float(df["max_memory_gb"].dropna().max()),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fixed_sources_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def plot_mixed(project: Path, out_path: Path, kind: str, x_col: str, loglog: bool = True) -> None:
    runs = load_fixed_mixed(project)
    ensure_expected_counts(fixed_mixed=runs)
    apply_style()
    plt.figure(figsize=(9.2, 6.0))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, len(runs)))
    for color, run in zip(cmap, runs):
        if kind == "quick":
            x, y = _series(run["df"], "val_loss", x_col, loglog)
            plt.plot(x, y, color=color, linewidth=1.8, label=f"d{run['depth']}")
        elif kind == "full":
            x, y = _series(run["df"], "full_val_loss", x_col, loglog)
            plt.plot(x, y, color=color, marker="o", linewidth=2.1, label=f"d{run['depth']}")
        elif kind == "combined":
            xq, yq = _series(run["df"], "val_loss", x_col, loglog)
            xf, yf = _series(run["df"], "full_val_loss", x_col, loglog)
            plt.plot(xq, yq, color=color, alpha=0.22, linewidth=1.1)
            plt.plot(xf, yf, color=color, marker="o", linewidth=2.0, label=f"d{run['depth']}")
        else:
            raise ValueError(kind)
    if loglog:
        plt.xscale("log"); plt.yscale("log")
    title_kind = {"quick": "quick validation", "full": "full validation", "combined": "quick + full validation"}[kind]
    plt.xlabel(_label_axis(x_col))
    plt.ylabel("validation loss")
    plt.title(f"Fixed-step mixed data: {title_kind}")
    if kind == "combined":
        plt.figtext(0.5, 0.01, "Thin translucent lines: quick val every 10 steps, 2 eval batches. Marked lines: full val every 500 steps/final, 20 eval batches.", ha="center", fontsize=8)
    plt.legend(ncol=2)
    savefig(out_path)


def plot_source(project: Path, out_path: Path, source: str, kind: str, x_col: str, loglog: bool = True) -> None:
    runs = [r for r in load_fixed_sources(project) if r["source"] == source]
    if len(runs) != 4:
        raise RuntimeError(f"expected 4 fixed runs for {source}, got {len(runs)}")
    apply_style()
    plt.figure(figsize=(9.2, 6.0))
    cmap = plt.cm.plasma(np.linspace(0.08, 0.90, len(runs)))
    for color, run in zip(cmap, runs):
        if kind == "quick":
            x, y = _series(run["df"], "val_loss", x_col, loglog)
            plt.plot(x, y, color=color, linewidth=1.8, label=f"d{run['depth']}")
        elif kind == "full":
            x, y = _series(run["df"], "full_val_loss", x_col, loglog)
            plt.plot(x, y, color=color, marker="o", linewidth=2.1, label=f"d{run['depth']}")
        elif kind == "combined":
            xq, yq = _series(run["df"], "val_loss", x_col, loglog)
            xf, yf = _series(run["df"], "full_val_loss", x_col, loglog)
            plt.plot(xq, yq, color=color, alpha=0.22, linewidth=1.1)
            plt.plot(xf, yf, color=color, marker="o", linewidth=2.0, label=f"d{run['depth']}")
        else:
            raise ValueError(kind)
    if loglog:
        plt.xscale("log"); plt.yscale("log")
    title_kind = {"quick": "quick validation", "full": "full validation", "combined": "quick + full validation"}[kind]
    plt.xlabel(_label_axis(x_col))
    plt.ylabel("validation loss")
    plt.title(f"Fixed-step {source}: {title_kind}")
    if kind == "combined":
        plt.figtext(0.5, 0.01, "Thin translucent lines: quick val every 10 steps, 2 eval batches. Marked lines: full val every 500 steps/final, 20 eval batches.", ha="center", fontsize=8)
    plt.legend()
    savefig(out_path)


def plot_source_depth_compare(project: Path, out_path: Path, depth: int, kind: str, x_col: str, loglog: bool = True) -> None:
    runs = [r for r in load_fixed_sources(project) if r["depth"] == depth]
    if len(runs) != 6:
        raise RuntimeError(f"expected 6 fixed source runs for depth {depth}, got {len(runs)}")
    apply_style()
    plt.figure(figsize=(9.4, 6.0))
    for run in sorted(runs, key=lambda r: SOURCE_ORDER.index(r["source"])):
        color = COLORS[run["source"]]
        if kind == "quick":
            x, y = _series(run["df"], "val_loss", x_col, loglog)
            plt.plot(x, y, color=color, linewidth=1.9, label=run["source"])
        elif kind == "full":
            x, y = _series(run["df"], "full_val_loss", x_col, loglog)
            plt.plot(x, y, color=color, marker="o", linewidth=2.1, label=run["source"])
        elif kind == "combined":
            xq, yq = _series(run["df"], "val_loss", x_col, loglog)
            xf, yf = _series(run["df"], "full_val_loss", x_col, loglog)
            plt.plot(xq, yq, color=color, alpha=0.22, linewidth=1.1)
            plt.plot(xf, yf, color=color, marker="o", linewidth=2.0, label=run["source"])
        else:
            raise ValueError(kind)
    if loglog:
        plt.xscale("log"); plt.yscale("log")
    plt.xlabel(_label_axis(x_col))
    plt.ylabel("validation loss")
    plt.title(f"Per-source comparison at depth {depth}: {kind} validation")
    plt.legend(ncol=2)
    savefig(out_path)
