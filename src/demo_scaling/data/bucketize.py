"""Bucket documents by static complexity metrics.

The default metric is ``raw_gzip_ratio`` from ``demo_scaling.data.metrics``.
Buckets are built within each source/category so low/mid/high means relative
complexity inside the same data source, not a global quality label.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from ..utils import ensure_dir


def load_docs_by_path(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """批量读取原文。

    旧实现每条 row 都重新打开文件并扫描到 line_no，8 万篇文档会非常慢。
    这里按文件分组，一次顺序读完需要的行。
    """

    needed: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        needed[row["text_path"]].add(row["line_no"])

    docs: dict[tuple[str, int], dict] = {}
    for path, line_numbers in needed.items():
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in line_numbers:
                    docs[(path, i)] = json.loads(line)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket documents by gzip complexity.")
    parser.add_argument("--metrics", default="train_data/metrics/doc_metrics.csv")
    parser.add_argument("--output", default="train_data/buckets")
    parser.add_argument("--metric-column", default="raw_gzip_ratio", help="Metric column used for quantile buckets, e.g. raw_gzip_ratio or token_gzip_ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    out = ensure_dir(args.output)

    metric_col = args.metric_column
    rows = []
    with Path(args.metrics).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if metric_col not in row:
                raise KeyError(f"metric column {metric_col!r} not found in {args.metrics}")
            row[metric_col] = float(row[metric_col])
            row["line_no"] = int(row["line_no"])
            rows.append(row)
    if not rows:
        raise SystemExit(f"no metric rows found in {args.metrics}")
    rows.sort(key=lambda r: (r["source"], r["doc_id"], r["text_path"], r["line_no"]))

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    assigned = []
    for source, items in by_source.items():
        vals = sorted(r[metric_col] for r in items)
        q1 = vals[int(0.33 * (len(vals) - 1))]
        q2 = vals[int(0.66 * (len(vals) - 1))]
        for row in items:
            level = "low" if row[metric_col] <= q1 else "mid" if row[metric_col] <= q2 else "high"
            row["bucket_id"] = f"{source}_{level}"
            assigned.append(row)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in assigned:
        buckets[row["bucket_id"]].append(row)

    docs = load_docs_by_path(assigned)

    assignment_path = out / "bucket_assignments.csv"
    with assignment_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(assigned[0].keys()))
        writer.writeheader()
        writer.writerows(assigned)

    for bucket_id, items in buckets.items():
        random.shuffle(items)
        split_at = max(1, int(len(items) * (1 - args.val_ratio)))
        bucket_dir = ensure_dir(out / bucket_id)
        for split, subset in [("train", items[:split_at]), ("val", items[split_at:] or items[:1])]:
            with (bucket_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for row in subset:
                    doc = docs[(row["text_path"], row["line_no"])]
                    doc["bucket_id"] = bucket_id
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"wrote {len(buckets)} buckets -> {out}")


if __name__ == "__main__":
    main()
