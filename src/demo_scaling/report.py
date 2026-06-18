"""生成项目 markdown 报告。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed runs collected yet._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project report markdown.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--output", default="report/project_report.md")
    args = parser.parse_args()
    out = Path(args.output)
    ensure_dir(out.parent)
    runs_path = Path(args.runs)
    if runs_path.exists():
        df = pd.read_csv(runs_path)
        table = dataframe_to_markdown(df)
        n_runs = len(df)
    else:
        table = "_No completed runs collected yet._"
        n_runs = 0
    text = f"""# demo_scaling_law 项目报告

## 目标

本项目面向 LLM 预训练算法实习场景，研究有限算力下数据、模型和训练 token 数之间的关系。重点不是训练大模型，而是构建可复现的小型实验系统，分析数据复杂度、数据效用、训练饱和点和 scaling law。

## 当前完成内容

- 独立实现 decoder-only GPT 训练代码，核心源码位于 `/src/demo_scaling/`。
- 支持多源小样本数据准备、gzip 复杂度指标、tokenization efficiency 统计和 bucket 化。
- 支持 A100/H100 单卡 Slurm 训练，记录 tokens、FLOPs、loss 和 checkpoint。
- 支持 ROI 曲线、loss-vs-params、loss-vs-tokens、loss-vs-FLOPs 图表生成。

## 核心指标

- `gzip_ratio`：文本压缩复杂度 proxy。
- `bits_per_token`：压缩后每个 token 的信息量 proxy。
- `tokens_per_char`：tokenizer 对不同数据源的效率。
- `ROI = -Δval_loss / ΔFLOPs`：继续训练的边际收益。
- `params_non_embedding`：更接近 scaling 分析的有效参数量口径。

## 已收集 run

共 {n_runs} 条。

{table}

## 解释框架

数据 bucket 会被归入四类状态：

- `under-learned`：loss 较低但仍下降较快，继续训练可能有收益。
- `over-learned`：loss 较低且下降较慢，继续训练边际收益小。
- `noisy-hard`：loss 高但下降慢，可能是噪声或模型难以利用的数据。
- `useful-hard`：loss 高且下降快，是值得继续学习的困难数据。

## 下一步

1. 跑完 MVP 实验矩阵，至少覆盖 3 个模型规模和 3 个 token budget。
2. 对多源 bucket 输出 per-bucket validation loss 和 ROI。
3. 用小规模 IsoFLOP 曲线估计 tokens/params ratio，并和 Chinchilla/nanoChat 经验进行比较。
"""
    out.write_text(text, encoding="utf-8")
    print(f"wrote report -> {out}")


if __name__ == "__main__":
    main()
