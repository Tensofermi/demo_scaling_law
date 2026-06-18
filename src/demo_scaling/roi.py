"""ROI 与数据状态判断。

ROI 定义为每多消耗一单位 FLOPs 带来的 validation loss 下降：
ROI = -Δval_loss / ΔFLOPs。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ROIPoint:
    step: int
    tokens_seen: int
    flops_seen: int
    val_loss: float
    roi: float | None
    saturated: bool


def compute_roi_points(rows: list[dict], threshold_ratio: float = 0.2) -> list[ROIPoint]:
    points: list[ROIPoint] = []
    first_positive: float | None = None
    for prev, cur in zip([None] + rows[:-1], rows):
        roi = None
        saturated = False
        if prev is not None:
            d_loss = float(cur["val_loss"]) - float(prev["val_loss"])
            d_flops = max(float(cur["flops_seen"]) - float(prev["flops_seen"]), 1.0)
            roi = -d_loss / d_flops
            if roi > 0 and first_positive is None:
                first_positive = roi
            if first_positive is not None and roi < threshold_ratio * first_positive:
                saturated = True
        points.append(
            ROIPoint(
                step=int(cur["iter"]),
                tokens_seen=int(cur["tokens_seen"]),
                flops_seen=int(cur["flops_seen"]),
                val_loss=float(cur["val_loss"]),
                roi=roi,
                saturated=saturated,
            )
        )
    return points


def classify_data_state(loss: float, slope: float, loss_median: float, slope_median: float) -> str:
    """基于 bucket loss 与 loss 下降速度给出可解释分类。"""

    high_loss = loss >= loss_median
    fast_drop = slope >= slope_median
    if high_loss and fast_drop:
        return "useful-hard"
    if high_loss and not fast_drop:
        return "noisy-hard"
    if not high_loss and fast_drop:
        return "under-learned"
    return "over-learned"

