"""通用工具：随机种子、设备选择、参数量和 FLOPs 估计。"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 Python/NumPy/PyTorch 随机数，便于复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    """选择训练设备；正式训练建议通过 Slurm 进入 GPU 节点。"""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前节点 torch.cuda.is_available() 为 False")
    return torch.device(requested)


def device_report(device: torch.device) -> dict[str, Any]:
    """返回设备信息，用于写入 metrics.json。"""

    report: dict[str, Any] = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda":
        idx = device.index or torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        report.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 2**30, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return report


def configure_cuda_performance(device: torch.device, allow_tf32: bool = True) -> None:
    """打开 CUDA 上的安全性能开关；bf16 主路径不依赖 TF32，但 fp32 fallback 会受益。"""

    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def peak_bf16_flops(device: torch.device) -> float | None:
    """返回常见单卡 BF16 峰值 FLOPs，用于粗略 MFU 估计。未知设备返回 None。"""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = device.index or torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx).lower()
    if "h100" in name:
        return 989e12
    if "a100" in name:
        return 312e12
    return None


def max_memory_gb(device: torch.device) -> float | None:
    """返回当前进程已分配过的最大 CUDA 显存，单位 GB。"""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / 2**30


def format_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_parameters(model: torch.nn.Module, non_embedding: bool = False) -> int:
    """统计参数量。

    这个函数保留给旧分析兼容。新版本主口径请优先使用
    `GPT.get_param_breakdown()["params_plan"]`。
    """

    total = 0
    for name, p in model.named_parameters():
        if non_embedding and ("tok_emb" in name or "pos_emb" in name):
            continue
        total += p.numel()
    return int(total)


def estimate_train_flops(params: int, tokens: int) -> int:
    """旧版 decoder-only Transformer 训练 FLOPs 近似：C ~= 6ND。"""

    return int(6 * params * tokens)


def estimate_train_flops_from_per_token(flops_per_token: int | float, tokens: int | float) -> int:
    """新版本主口径：训练 FLOPs = nanoGPT-style FLOPs/token * token 数。"""

    return int(float(flops_per_token) * float(tokens))


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def human_int(value: int | float) -> str:
    if value == 0:
        return "0"
    units = ["", "K", "M", "B", "T"]
    idx = min(int(math.log(abs(value), 1000)), len(units) - 1)
    return f"{value / (1000 ** idx):.2f}{units[idx]}"
