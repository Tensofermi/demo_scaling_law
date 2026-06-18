"""Document-level raw-text and token-id gzip metrics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

WORD_RE = re.compile(r"\w+")
_ENCODER = None


def init_worker(tokenizer: str) -> None:
    global _ENCODER
    _ENCODER = tiktoken.get_encoding(tokenizer)


def gzip_len(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def duplicate_score(text: str) -> float:
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbols / len(text)


def token_bytes(ids: list[int]) -> bytes:
    return np.asarray(ids, dtype=np.uint16).tobytes()


def iter_jsonl(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    yield path, line_no, json.loads(line)


def compute_row(item: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    path_s, line_no, rec = item
    if _ENCODER is None:
        init_worker("gpt2")
    text = str(rec.get("text", ""))
    ids = _ENCODER.encode_ordinary(text)  # type: ignore[union-attr]
    raw = text.encode("utf-8")
    raw_gz = gzip_len(raw)
    tb = token_bytes(ids)
    tok_gz = gzip_len(tb)
    n_tok = max(len(ids), 1)
    return {
        "doc_id": f"{Path(path_s).stem}_{line_no}",
        "category": rec.get("category", "unknown"),
        "source": rec.get("source", "unknown"),
        "split": rec.get("split", Path(path_s).stem),
        "text_path": path_s,
        "line_no": line_no,
        "chars": len(text),
        "tokens": len(ids),
        "tokens_per_char": len(ids) / max(len(text), 1),
        "chars_per_token": len(text) / n_tok,
        "raw_bytes": len(raw),
        "raw_gzip_bytes": raw_gz,
        "raw_gzip_ratio": raw_gz / max(len(raw), 1),
        "raw_bits_per_token": 8 * raw_gz / n_tok,
        "token_bytes": len(tb),
        "token_gzip_bytes": tok_gz,
        "token_gzip_ratio": tok_gz / max(len(tb), 1),
        "token_bits_per_token": 8 * tok_gz / n_tok,
        "duplicate_score": duplicate_score(text),
        "symbol_ratio": symbol_ratio(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute doc-level raw/token gzip metrics.")
    parser.add_argument("--input", default="train_data/processed/splits")
    parser.add_argument("--output", default="train_data/metrics/doc_metrics.csv")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.input)
    paths = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "doc_id", "category", "source", "split", "text_path", "line_no", "chars", "tokens",
        "tokens_per_char", "chars_per_token", "raw_bytes", "raw_gzip_bytes", "raw_gzip_ratio", "raw_bits_per_token",
        "token_bytes", "token_gzip_bytes", "token_gzip_ratio", "token_bits_per_token",
        "duplicate_score", "symbol_ratio",
    ]

    def items():
        for i, (path, line_no, rec) in enumerate(iter_jsonl(paths)):
            if args.limit_docs is not None and i >= args.limit_docs:
                break
            yield (str(path), line_no, rec)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if args.workers <= 1:
            init_worker(args.tokenizer)
            for i, item in enumerate(items(), start=1):
                writer.writerow(compute_row(item))
                if i % 100000 == 0:
                    print(f"[metrics] docs={i:,}", flush=True)
        else:
            with mp.Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
                for i, row in enumerate(pool.imap(compute_row, items(), chunksize=256), start=1):
                    writer.writerow(row)
                    if i % 100000 == 0:
                        print(f"[metrics] docs={i:,}", flush=True)
    print(f"wrote metrics -> {out}")


if __name__ == "__main__":
    main()
