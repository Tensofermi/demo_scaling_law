"""Collect v2 training logs into a scaling-analysis CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "run_id", "depth", "params_total", "params_transformer", "params_token_embedding", "params_pos_embedding",
    "flops_per_token", "target_flops", "target_tokens", "actual_tokens_planned", "actual_flops_planned",
    "tokens_seen", "flops_seen", "best_val_loss", "final_val_loss", "tail_smoothed_val_loss",
    "final_minus_best", "loss_instability_ratio", "loss_stability_status", "tokens_per_iter", "max_iters",
    "gpu_name", "log_path", "bucket_log_path", "best_checkpoint", "final_checkpoint",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        val = row.get("full_val_loss") or row.get("val_loss")
        if val not in (None, ""):
            row["_analysis_val_loss"] = val
            out.append(row)
    return out


def tail_mean(rows: list[dict], frac: float) -> float | None:
    vals = []
    for row in rows:
        try:
            vals.append(float(row["_analysis_val_loss"]))
        except Exception:
            pass
    if not vals:
        return None
    n = max(1, round(len(vals) * frac))
    tail = vals[-n:]
    return sum(tail) / len(tail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect v2 metrics.")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--output", default="results/runs.csv")
    parser.add_argument("--include-prefix", action="append", default=[])
    parser.add_argument("--tail-fraction", type=float, default=0.15)
    parser.add_argument("--instability-threshold", type=float, default=0.05)
    args = parser.parse_args()
    rows = []
    for metrics_path in sorted(Path(args.logs).glob("*/metrics.json")):
        m = read_json(metrics_path)
        run_id = str(m.get("run_id", ""))
        if args.include_prefix and not any(run_id.startswith(p) for p in args.include_prefix):
            continue
        log_path = metrics_path.parent / "train_log.csv"
        erows = eval_rows(log_path)
        if not erows:
            continue
        final = erows[-1]
        final_loss = float(final["_analysis_val_loss"])
        tail = tail_mean(erows, args.tail_fraction) or final_loss
        best = float(m.get("best_val_loss", final_loss))
        diff = final_loss - best
        ratio = diff / max(abs(best), 1e-12)
        rows.append({
            "run_id": run_id,
            "depth": m.get("depth"),
            "params_total": m.get("params_total"),
            "params_transformer": m.get("params_transformer"),
            "params_token_embedding": m.get("params_token_embedding"),
            "params_pos_embedding": m.get("params_pos_embedding", 0),
            "flops_per_token": m.get("flops_per_token"),
            "target_flops": m.get("target_flops"),
            "target_tokens": m.get("target_tokens"),
            "actual_tokens_planned": m.get("actual_tokens_planned"),
            "actual_flops_planned": m.get("actual_flops_planned"),
            "tokens_seen": final.get("tokens_seen", ""),
            "flops_seen": final.get("flops_seen", ""),
            "best_val_loss": best,
            "final_val_loss": final_loss,
            "tail_smoothed_val_loss": tail,
            "final_minus_best": diff,
            "loss_instability_ratio": ratio,
            "loss_stability_status": "unstable_tail" if ratio > args.instability_threshold else "ok",
            "tokens_per_iter": m.get("tokens_per_iter"),
            "max_iters": m.get("max_iters"),
            "gpu_name": m.get("device", {}).get("gpu_name", "cpu"),
            "log_path": str(log_path),
            "bucket_log_path": str(metrics_path.parent / "bucket_val_log.csv"),
            "best_checkpoint": m.get("best_checkpoint"),
            "final_checkpoint": m.get("final_checkpoint"),
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
