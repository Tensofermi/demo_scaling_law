"""生成 ROI 曲线和 saturation 摘要。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..roi import compute_roi_points
from ..utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ROI curves from train logs.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--output", default="report/roi")
    parser.add_argument("--threshold-ratio", type=float, default=0.2)
    parser.add_argument("--loss-column", default="val_loss", help="train_log 中用于 ROI 的 loss 列；缺失时回退 val_loss")
    parser.add_argument("--smoothing-window", type=int, default=3, help="ROI 前对 eval loss 做 trailing mean 的窗口")
    args = parser.parse_args()
    out = ensure_dir(args.output)
    for old in out.glob("*_roi.png"):
        old.unlink()
    runs = pd.read_csv(args.runs)
    summary_rows = []
    for _, run in runs.iterrows():
        log_path = Path(str(run["log_path"]))
        if not log_path.exists():
            continue
        rows = []
        with log_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("val_loss"):
                    rows.append(row)
        rows = smooth_eval_rows(rows, args.loss_column, args.smoothing_window)
        points = compute_roi_points(rows, threshold_ratio=args.threshold_ratio)
        xs = [p.tokens_seen for p in points if p.roi is not None]
        ys = [p.roi for p in points if p.roi is not None]
        if xs:
            plt.figure(figsize=(7, 4))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Tokens seen")
            plt.ylabel("ROI = -Δval_loss / ΔFLOPs")
            plt.title(str(run["run_id"]))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out / f"{run['run_id']}_roi.png", dpi=160)
            plt.close()
        saturated = [p for p in points if p.saturated]
        summary_rows.append(
            {
                "run_id": run["run_id"],
                "first_saturation_tokens": saturated[0].tokens_seen if saturated else "",
                "last_roi": points[-1].roi if points and points[-1].roi is not None else "",
                "final_val_loss": points[-1].val_loss if points else "",
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / "roi_summary.csv", index=False)
    print(f"wrote ROI report -> {out}")


def smooth_eval_rows(rows: list[dict], loss_column: str, window: int) -> list[dict]:
    """生成用于 ROI 的平滑 eval 序列。

    train_log 通常只有 `val_loss`，没有每一步的 `tail_smoothed_val_loss`。
    这里用 trailing mean 降低单点评估噪声；输出仍写回 `val_loss` 字段，
    复用底层 ROI 计算函数。
    """

    if not rows:
        return []
    vals = []
    for row in rows:
        raw = row.get(loss_column) or row.get("val_loss")
        vals.append(float(raw))
    win = max(1, int(window))
    out = []
    for i, row in enumerate(rows):
        start = max(0, i - win + 1)
        smoothed = sum(vals[start : i + 1]) / (i - start + 1)
        item = dict(row)
        item["raw_val_loss"] = row.get("val_loss", "")
        item["val_loss"] = smoothed
        out.append(item)
    return out


if __name__ == "__main__":
    main()
