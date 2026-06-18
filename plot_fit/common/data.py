from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .style import REFINED_SOURCES, SOURCE_ORDER

FIXED_MIXED_RE = re.compile(r"fixed_mixed_d(?P<depth>\d+)_10k$")
FIXED_SOURCE_RE = re.compile(r"fixed_(?P<source>story|encyclopedia|news|math|code|dialogue)_d(?P<depth>\d+)_1k$")


def project_root_from_script(script_file: str | Path) -> Path:
    path = Path(script_file).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "logs").exists() and (parent / "configs").exists():
            return parent
    # Scripts live in plot_fit/<section>.
    return path.parents[2]


def read_log(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    numeric_cols = [
        "iter", "tokens_seen", "flops_seen", "train_loss", "val_loss", "full_val_loss",
        "lr", "lr_multiplier", "wall_time_sec", "tokens_per_sec", "tflops_per_sec", "mfu", "max_memory_gb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def load_fixed_mixed(project: Path) -> list[dict]:
    runs = []
    for log_path in sorted((project / "logs").glob("fixed_mixed_d*_10k/train_log.csv")):
        match = FIXED_MIXED_RE.match(log_path.parent.name)
        if not match:
            continue
        metrics = read_metrics(log_path.parent)
        runs.append({
            "run_id": log_path.parent.name,
            "depth": int(match.group("depth")),
            "source": "mixed",
            "params_total": int(metrics["params_total"]),
            "df": read_log(log_path),
            "metrics": metrics,
        })
    return sorted(runs, key=lambda r: r["depth"])


def load_fixed_sources(project: Path) -> list[dict]:
    runs = []
    for log_path in sorted((project / "logs").glob("fixed_*_d*_1k/train_log.csv")):
        match = FIXED_SOURCE_RE.match(log_path.parent.name)
        if not match:
            continue
        metrics = read_metrics(log_path.parent)
        source = match.group("source")
        runs.append({
            "run_id": log_path.parent.name,
            "depth": int(match.group("depth")),
            "source": source,
            "params_total": int(metrics["params_total"]),
            "df": read_log(log_path),
            "metrics": metrics,
        })
    return sorted(runs, key=lambda r: (SOURCE_ORDER.index(r["source"]), r["depth"]))


def final_loss(df: pd.DataFrame, col: str) -> float:
    part = df.dropna(subset=[col])
    if part.empty:
        return float("nan")
    return float(part.iloc[-1][col])


def tail_smoothed_full_val_loss(df: pd.DataFrame, frac: float = 0.15) -> float:
    full = df.dropna(subset=["full_val_loss"])
    if full.empty:
        quick = df.dropna(subset=["val_loss"])
        if quick.empty:
            return float("nan")
        tail_n = max(1, round(len(quick) * frac))
        return float(quick["val_loss"].tail(tail_n).mean())
    tail_n = max(1, round(len(full) * frac))
    return float(full["full_val_loss"].tail(tail_n).mean())


def tail_smoothed_quick_val_loss(df: pd.DataFrame, frac: float = 0.15) -> float:
    quick = df.dropna(subset=["val_loss"])
    if quick.empty:
        return float("nan")
    tail_n = max(1, round(len(quick) * frac))
    return float(quick["val_loss"].tail(tail_n).mean())


def matrix_rows(project: Path, matrix_names: Iterable[str]) -> pd.DataFrame:
    frames = []
    for name in matrix_names:
        path = project / "configs" / "experiments" / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["matrix"] = name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["run_id"], keep="last")


def load_isoflop(project: Path, matrix_names: Iterable[str], sources: Iterable[str] | None = None) -> pd.DataFrame:
    matrix = matrix_rows(project, matrix_names)
    rows = []
    source_filter = set(sources) if sources is not None else None
    for _, mrow in matrix.iterrows():
        run_id = str(mrow["run_id"])
        source = str(mrow.get("source", "mixed"))
        if source_filter is not None and source not in source_filter:
            continue
        run_dir = project / "logs" / run_id
        metrics_path = run_dir / "metrics.json"
        log_path = run_dir / "train_log.csv"
        if not metrics_path.exists() or not log_path.exists():
            continue
        metrics = read_metrics(run_dir)
        log = read_log(log_path)
        full = log.dropna(subset=["full_val_loss"])
        eval_frame = full if not full.empty else log.dropna(subset=["val_loss"])
        if eval_frame.empty:
            continue
        val_col = "full_val_loss" if not full.empty else "val_loss"
        rows.append({
            "run_id": run_id,
            "mode": mrow.get("mode", ""),
            "source": source,
            "depth": int(mrow["depth"]),
            "flops_budget": float(mrow["target_flops"]),
            "target_tokens": int(float(mrow["target_tokens"])),
            "params_total": int(metrics["params_total"]),
            "tokens_seen": int(float(eval_frame["tokens_seen"].iloc[-1])),
            "final_val_loss": float(eval_frame[val_col].iloc[-1]),
            "tail_smoothed_val_loss": tail_smoothed_full_val_loss(log),
            "matrix": mrow.get("matrix", ""),
        })
    return pd.DataFrame(rows)


def ensure_expected_counts(fixed_mixed: list[dict] | None = None, fixed_sources: list[dict] | None = None, mixed_iso: pd.DataFrame | None = None) -> None:
    if fixed_mixed is not None and len(fixed_mixed) != 8:
        raise RuntimeError(f"expected 8 fixed mixed runs, got {len(fixed_mixed)}")
    if fixed_sources is not None and len(fixed_sources) != 24:
        raise RuntimeError(f"expected 24 fixed source runs, got {len(fixed_sources)}")
    if mixed_iso is not None and len(mixed_iso) != 34:
        raise RuntimeError(f"expected 34 mixed IsoFLOP runs, got {len(mixed_iso)}")
