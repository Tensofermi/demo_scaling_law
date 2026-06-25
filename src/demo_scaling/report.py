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
    text = f"""xxxx"""
    out.write_text(text, encoding="utf-8")
    print(f"wrote report -> {out}")


if __name__ == "__main__":
    main()
