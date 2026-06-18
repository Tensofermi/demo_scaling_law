"""生成 Stage 2 可视化图表。

这个脚本只读取训练后已有的 CSV/JSON，不重新训练模型。输出重点是三类图：

1. validation loss 随 tokens_seen 的变化，用来判断训练是否真的下降。
2. ROI 随 tokens_seen 的变化，用来观察继续训练的边际收益。
3. 每个 bucket 的最终 validation loss，用来比较不同数据源/复杂度桶的学习难度。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils import ensure_dir


def _numeric_frame(path: Path) -> pd.DataFrame:
    """读取训练 CSV，并把常用数值列转成数字。"""

    df = pd.read_csv(path)
    for col in ["iter", "tokens_seen", "flops_seen", "val_loss", "roi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _valid_path(value: object) -> Path | None:
    """把 CSV 里的路径字段转成 Path；空值/NaN/目录都视为不可用。"""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    if not path.exists() or path.is_dir():
        return None
    return path


def plot_val_loss(runs: pd.DataFrame, output: Path) -> None:
    """画所有 run 的 validation loss 曲线。"""

    plt.figure(figsize=(8, 4.8))
    plotted = 0
    for _, run in runs.iterrows():
        log_path = _valid_path(run.get("log_path", ""))
        if log_path is None:
            continue
        df = _numeric_frame(log_path).dropna(subset=["tokens_seen", "val_loss"])
        if df.empty:
            continue
        plt.plot(df["tokens_seen"], df["val_loss"], marker="o", label=str(run["run_id"]))
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Tokens seen")
    plt.ylabel("Validation loss")
    plt.title("Validation loss curves")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "val_loss_curves.png", dpi=180)
    plt.close()


def plot_roi(runs: pd.DataFrame, output: Path) -> None:
    """画所有 run 的 ROI 曲线；只有至少两个 eval 点的 run 才有 ROI。"""

    plt.figure(figsize=(8, 4.8))
    plotted = 0
    for _, run in runs.iterrows():
        log_path = _valid_path(run.get("log_path", ""))
        if log_path is None:
            continue
        df = _numeric_frame(log_path).dropna(subset=["tokens_seen", "roi"])
        if df.empty:
            continue
        plt.plot(df["tokens_seen"], df["roi"], marker="o", label=str(run["run_id"]))
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Tokens seen")
    plt.ylabel("ROI = -delta val_loss / delta FLOPs")
    plt.title("Training ROI curves")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "roi_curves.png", dpi=180)
    plt.close()


def plot_final_bucket_loss(runs: pd.DataFrame, output: Path) -> None:
    """为每个 run 画最终 eval step 的 per-bucket loss 横向柱状图。"""

    for _, run in runs.iterrows():
        run_id = str(run["run_id"])
        bucket_path = _valid_path(run.get("bucket_log_path", ""))
        if bucket_path is None:
            continue
        df = pd.read_csv(bucket_path)
        if df.empty or "iter" not in df.columns:
            continue
        df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
        df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
        final_iter = df["iter"].max()
        final = df[df["iter"] == final_iter].dropna(subset=["val_loss"]).copy()
        if final.empty:
            continue
        final = final.sort_values("val_loss", ascending=True)
        height = max(6.0, min(14.0, 0.28 * len(final)))
        plt.figure(figsize=(9, height))
        plt.barh(final["bucket_id"], final["val_loss"])
        plt.xlabel("Validation loss")
        plt.ylabel("Bucket")
        plt.title(f"Final per-bucket validation loss: {run_id}")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output / f"{run_id}_final_bucket_loss.png", dpi=180)
        plt.close()


def write_stage2_summary(runs: pd.DataFrame, output: Path) -> None:
    """写一份简短 markdown 摘要，方便从报告里引用。"""

    stage_name = output.name or "visualization"
    lines = [
        f"# {stage_name} Visualization Summary",
        "",
        "This summary is generated from completed training runs.",
        "",
        "## Runs",
        "",
        "| run_id | model | tokens_seen | final_val_loss | best_val_loss | device |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in runs.iterrows():
        lines.append(
            "| {run_id} | {model} | {tokens_seen} | {final_val_loss} | {best_val_loss} | {gpu_name} |".format(
                run_id=row.get("run_id", ""),
                model=row.get("model", ""),
                tokens_seen=row.get("tokens_seen", ""),
                final_val_loss=row.get("final_val_loss", ""),
                best_val_loss=row.get("best_val_loss", ""),
                gpu_name=row.get("gpu_name", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `val_loss_curves.png`: validation loss over tokens.",
            "- `roi_curves.png`: marginal validation-loss improvement per FLOP.",
            "- `*_final_bucket_loss.png`: final validation loss by data bucket.",
        ]
    )
    (output / f"{stage_name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage 2 visualizations.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--output", default="report/stage2")
    args = parser.parse_args()

    output = ensure_dir(args.output)
    runs = pd.read_csv(args.runs)
    for col in ["tokens_seen", "final_val_loss", "best_val_loss"]:
        if col in runs.columns:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")
    runs = runs.dropna(subset=["final_val_loss"])
    if runs.empty:
        raise SystemExit("No completed runs with final_val_loss.")

    plot_val_loss(runs, output)
    plot_roi(runs, output)
    plot_final_bucket_loss(runs, output)
    write_stage2_summary(runs, output)
    print(f"wrote visualizations -> {output}")


if __name__ == "__main__":
    main()
