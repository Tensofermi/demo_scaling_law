"""Per-bucket data efficiency analysis.

The input is the per-bucket validation log produced during training plus
static document statistics computed before training. The output answers three
practical questions:

1. Which buckets remain high-loss after training?
2. Which buckets are still improving quickly?
3. How do gzip/text-complexity proxies relate to the observed learning state?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..roi import classify_data_state
from ..utils import ensure_dir


METRIC_COLS = [
    "chars",
    "tokens",
    "tokens_per_char",
    "chars_per_token",
    "raw_gzip_ratio",
    "raw_bits_per_token",
    "token_gzip_ratio",
    "token_bits_per_token",
    "duplicate_score",
    "symbol_ratio",
]


def load_bucket_metrics(path: Path) -> pd.DataFrame:
    """Aggregate static document metrics for every bucket."""

    df = pd.read_csv(path)
    if "bucket_id" not in df.columns:
        raise ValueError(f"{path} must contain bucket_id; run demo_scaling.data.bucketize first")
    if "raw_bits_per_token" not in df.columns and "bits_per_token" in df.columns:
        df["raw_bits_per_token"] = df["bits_per_token"]
    if "raw_gzip_ratio" not in df.columns and "gzip_ratio" in df.columns:
        df["raw_gzip_ratio"] = df["gzip_ratio"]
    agg = df.groupby("bucket_id").agg(
        docs=("doc_id", "count"),
        source=("source", "first"),
        chars=("chars", "sum"),
        tokens=("tokens", "sum"),
        tokens_per_char=("tokens_per_char", "mean"),
        chars_per_token=("chars_per_token", "mean"),
        raw_gzip_ratio=("raw_gzip_ratio", "mean"),
        raw_bits_per_token=("raw_bits_per_token", "mean"),
        token_gzip_ratio=("token_gzip_ratio", "mean"),
        token_bits_per_token=("token_bits_per_token", "mean"),
        duplicate_score=("duplicate_score", "mean"),
        symbol_ratio=("symbol_ratio", "mean"),
    )
    return agg.reset_index()


def bucket_training_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize first/final loss and learning speed for each run/bucket."""

    rows = []
    for _, run in runs.iterrows():
        path = Path(str(run["bucket_log_path"]))
        if not path.exists():
            continue
        log = pd.read_csv(path)
        log["iter"] = pd.to_numeric(log["iter"], errors="coerce")
        log["val_loss"] = pd.to_numeric(log["val_loss"], errors="coerce")
        log = log.dropna(subset=["iter", "val_loss"])
        if log.empty:
            continue
        if "eval_type" in log.columns and (log["eval_type"] == "full").any():
            log = log[log["eval_type"] == "full"].copy()
        log = log.groupby(["iter", "bucket_id"], as_index=False)["val_loss"].mean()
        first_iter = log["iter"].min()
        final_iter = log["iter"].max()
        first = log[log["iter"] == first_iter].set_index("bucket_id")
        final = log[log["iter"] == final_iter].set_index("bucket_id")
        for bucket_id in sorted(set(first.index) & set(final.index)):
            initial_loss = float(first.loc[bucket_id, "val_loss"])
            final_loss = float(final.loc[bucket_id, "val_loss"])
            tokens_seen = float(run["tokens_seen"])
            loss_drop = initial_loss - final_loss
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run.get("model", f"d{run.get('depth', '')}"),
                    "params_effective": run.get("params_transformer", run.get("params_total", "")),
                    "tokens_seen": tokens_seen,
                    "bucket_id": bucket_id,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "loss_drop": loss_drop,
                    "loss_drop_per_million_tokens": loss_drop / max(tokens_seen / 1_000_000.0, 1e-12),
                }
            )
    return pd.DataFrame(rows)


def classify_buckets(summary: pd.DataFrame) -> pd.DataFrame:
    """Classify bucket states from final loss and learning speed."""

    loss_median = float(summary["final_loss"].median())
    slope_median = float(summary["loss_drop_per_million_tokens"].median())
    summary = summary.copy()
    summary["state"] = [
        classify_data_state(loss, slope, loss_median, slope_median)
        for loss, slope in zip(summary["final_loss"], summary["loss_drop_per_million_tokens"])
    ]
    return summary


def select_latest_per_bucket(per_run: pd.DataFrame) -> pd.DataFrame:
    """Use the highest-token run available for each bucket as the headline state."""

    ordered = per_run.sort_values(["bucket_id", "tokens_seen", "params_effective"], ascending=[True, False, False])
    return ordered.groupby("bucket_id", as_index=False).head(1).reset_index(drop=True)


def plot_state_counts(df: pd.DataFrame, out: Path) -> None:
    counts = df["state"].value_counts().reindex(["useful-hard", "noisy-hard", "under-learned", "over-learned"]).dropna()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.ylabel("Bucket count")
    plt.title("Bucket learning-state counts")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "bucket_state_counts.png", dpi=180)
    plt.close()


def plot_loss_vs_complexity(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    states = sorted(df["state"].dropna().unique())
    for state in states:
        part = df[df["state"] == state]
        plt.scatter(part["raw_bits_per_token"], part["final_loss"], label=state, alpha=0.8)
    plt.xlabel("Mean raw gzip bits per token")
    plt.ylabel("Final bucket validation loss")
    plt.title("Data complexity vs learned loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "loss_vs_bits_per_token.png", dpi=180)
    plt.close()


def plot_bucket_loss(df: pd.DataFrame, out: Path) -> None:
    show = df.sort_values("final_loss", ascending=False).head(24).sort_values("final_loss")
    plt.figure(figsize=(9, max(5, 0.28 * len(show))))
    plt.barh(show["bucket_id"], show["final_loss"])
    plt.xlabel("Final validation loss")
    plt.ylabel("Bucket")
    plt.title("Highest-loss buckets after training")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "highest_loss_buckets.png", dpi=180)
    plt.close()


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _fmt_int(value: object) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return str(int(value))
    except Exception:
        return "NA"


def write_markdown(df: pd.DataFrame, per_run: pd.DataFrame, out: Path) -> None:
    state_counts = df["state"].value_counts().to_dict()
    hardest = df.sort_values("final_loss", ascending=False).head(8)
    fastest = df.sort_values("loss_drop_per_million_tokens", ascending=False).head(8)
    lines = [
        "# Data Efficiency Report",
        "",
        "This report links static data metrics with training dynamics from completed runs.",
        "",
        "## Summary",
        "",
        f"- Buckets analyzed: {len(df)}",
        f"- Run/bucket observations: {len(per_run)}",
        f"- State counts: {state_counts}",
        "",
        "## Highest-loss buckets",
        "",
        "| bucket_id | source | state | final_loss | loss_drop_per_Mtok | bits_per_token | docs |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in hardest.iterrows():
        lines.append(
            f"| {row.bucket_id} | {row.get('source', 'NA')} | {row.state} | {_fmt_float(row.final_loss)} | "
            f"{_fmt_float(row.loss_drop_per_million_tokens)} | {_fmt_float(row.get('raw_bits_per_token'), 3)} | {_fmt_int(row.get('docs'))} |"
        )
    lines.extend(
        [
            "",
            "## Fastest-improving buckets",
            "",
            "| bucket_id | source | state | final_loss | loss_drop_per_Mtok | bits_per_token | docs |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in fastest.iterrows():
        lines.append(
            f"| {row.bucket_id} | {row.get('source', 'NA')} | {row.state} | {_fmt_float(row.final_loss)} | "
            f"{_fmt_float(row.loss_drop_per_million_tokens)} | {_fmt_float(row.get('raw_bits_per_token'), 3)} | {_fmt_int(row.get('docs'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `useful-hard` buckets have high final loss but also high loss drop per token, so more training or higher sampling weight may still be useful.",
            "- `noisy-hard` buckets have high final loss and low improvement, so they are candidates for quality filtering or separate modeling.",
            "- `under-learned` buckets are low-loss but still improving quickly.",
            "- `over-learned` buckets are low-loss and slow-improving, so their marginal ROI is lower in this setup.",
            "",
            "The classification is relative to the current collected runs. It should be treated as a diagnostic signal, not a universal data-quality label.",
        ]
    )
    (out / "bucket_state.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-bucket data efficiency.")
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--bucket-assignments", default="train_data/buckets/bucket_assignments.csv")
    parser.add_argument("--output", default="results/data_efficiency")
    args = parser.parse_args()

    out = ensure_dir(args.output)
    runs = pd.read_csv(args.runs)
    if "tokens_seen" in runs.columns:
        runs["tokens_seen"] = pd.to_numeric(runs["tokens_seen"], errors="coerce")
    per_run = bucket_training_summary(runs)
    if per_run.empty:
        raise SystemExit("No bucket training logs found.")
    per_run = classify_buckets(per_run)
    metrics = load_bucket_metrics(Path(args.bucket_assignments))
    latest = classify_buckets(select_latest_per_bucket(per_run))
    summary = latest.merge(metrics, on="bucket_id", how="left")

    per_run.to_csv(out / "bucket_run_summary.csv", index=False)
    summary.to_csv(out / "bucket_summary.csv", index=False)
    plot_state_counts(summary, out)
    plot_loss_vs_complexity(summary, out)
    plot_bucket_loss(summary, out)
    write_markdown(summary, per_run, out)
    print(f"wrote data efficiency report -> {out}")


if __name__ == "__main__":
    main()
