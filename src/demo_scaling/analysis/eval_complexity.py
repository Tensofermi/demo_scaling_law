"""Evaluate checkpoints on gzip-complexity validation buckets.

This diagnostic pass does not change training data or checkpoints. It reads
document-level gzip metrics, samples validation documents from low/mid/high
complexity groups, and evaluates each completed run on those groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..model import GPT, GPTConfig
from ..utils import choose_device, ensure_dir


def resolve_path(path_value: object, project_root: Path) -> Path | None:
    """Resolve checkpoint paths written by older and newer collectors.

    Metrics may contain either absolute paths or project-root relative paths.
    This helper tries the current working directory and the configured project
    root before giving up.
    """

    raw = str(path_value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, project_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metric_col = "raw_bits_per_token" if "raw_bits_per_token" in out.columns else "bits_per_token"
    values = pd.to_numeric(out[metric_col], errors="coerce")
    try:
        out["complexity_group"] = pd.qcut(values, q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
    except ValueError:
        out["complexity_group"] = pd.qcut(values.rank(method="first"), q=3, labels=["low", "mid", "high"]).astype(str)
    return out.dropna(subset=["complexity_group"])


def load_selected_docs(rows: pd.DataFrame) -> list[str]:
    needed: dict[str, set[int]] = defaultdict(set)
    for _, row in rows.iterrows():
        needed[str(row["text_path"])].add(int(row["line_no"]))
    texts: list[str] = []
    for path, line_numbers in needed.items():
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in line_numbers and line.strip():
                    texts.append(json.loads(line).get("text", ""))
    return texts


def encode_group(texts: list[str], encoding_name: str, max_tokens: int) -> np.ndarray:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    ids: list[int] = []
    for text in texts:
        ids.extend(enc.encode_ordinary(text))
        ids.append(enc.eot_token)
        if len(ids) >= max_tokens:
            break
    return np.asarray(ids[:max_tokens], dtype=np.int64)


@torch.no_grad()
def eval_tokens(model: GPT, tokens: np.ndarray, batch_size: int, block_size: int, device: torch.device) -> tuple[float, int]:
    if len(tokens) <= block_size:
        return float("nan"), 0
    starts = list(range(0, len(tokens) - block_size - 1, block_size))
    if not starts:
        starts = [0]
    total_loss = 0.0
    total_items = 0
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    model.eval()
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        x_np = np.stack([tokens[s : s + block_size] for s in batch_starts])
        y_np = np.stack([tokens[s + 1 : s + 1 + block_size] for s in batch_starts])
        x = torch.from_numpy(x_np).long().to(device)
        y = torch.from_numpy(y_np).long().to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        items = int(y.numel())
        total_loss += float(loss.item()) * items
        total_items += items
    return total_loss / max(total_items, 1), total_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on low/mid/high gzip complexity validation buckets.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--doc-metrics", default="train_data/metrics/doc_metrics.csv")
    parser.add_argument("--output", default="results/complexity_losses.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-kind", choices=["best", "final"], default="best")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--encoding", default="gpt2")
    parser.add_argument("--max-docs-per-group", type=int, default=2048)
    parser.add_argument("--max-tokens-per-group", type=int, default=262144)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output)
    ensure_dir(out.parent)
    device = choose_device(args.device)
    project_root = Path(args.project_root).resolve()
    runs = pd.read_csv(args.runs)
    metrics = pd.read_csv(args.doc_metrics)
    if "split" in metrics.columns:
        metrics = metrics[metrics["split"].astype(str) == "val"].copy()
    metrics = assign_groups(metrics)

    group_tokens: dict[str, np.ndarray] = {}
    group_doc_counts: dict[str, int] = {}
    for group, part in metrics.groupby("complexity_group"):
        sample = part.sample(n=min(args.max_docs_per_group, len(part)), random_state=args.seed)
        texts = load_selected_docs(sample)
        group_tokens[str(group)] = encode_group(texts, args.encoding, args.max_tokens_per_group)
        group_doc_counts[str(group)] = len(texts)

    rows = []
    for _, run in runs.iterrows():
        ckpt_col = "best_checkpoint" if args.checkpoint_kind == "best" else "final_checkpoint"
        ckpt_path = resolve_path(run.get(ckpt_col) or run.get("checkpoint"), project_root)
        if ckpt_path is None:
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        cfg = GPTConfig(**checkpoint["model_config"])
        model = GPT(cfg)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        for group, tokens in group_tokens.items():
            loss, eval_tokens_count = eval_tokens(model, tokens, args.batch_size, cfg.block_size, device)
            rows.append(
                {
                    "run_id": run["run_id"],
                    "complexity_group": group,
                    "loss": loss,
                    "eval_tokens": eval_tokens_count,
                    "eval_docs": group_doc_counts[group],
                    "checkpoint_kind": args.checkpoint_kind,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "complexity_group", "loss", "eval_tokens", "eval_docs", "checkpoint_kind"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
