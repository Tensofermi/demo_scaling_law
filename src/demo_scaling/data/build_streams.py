"""Build shuffled GPT-2 token streams for mixed and per-source training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import tiktoken

CATEGORIES = ["story", "encyclopedia", "news", "math", "code", "dialogue"]
SPLITS = ["train", "val", "test"]


def read_records(path: Path, category: str | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if category is None or rec.get("category") == category:
                records.append(rec)
    return records


def flush(buf: list[int], path: Path) -> int:
    if not buf:
        path.touch(exist_ok=True)
        return 0
    arr = np.asarray(buf, dtype=np.uint16)
    with path.open("ab") as f:
        arr.tofile(f)
    n = len(buf)
    buf.clear()
    return n


def write_stream(records: list[dict], out_path: Path, enc, seed: int, chunk_tokens: int, eot_token: int) -> dict:
    rng = random.Random(seed)
    rng.shuffle(records)
    out_path.unlink(missing_ok=True)
    buf: list[int] = []
    docs = 0
    text_tokens = 0
    total_tokens = 0
    by_category = defaultdict(lambda: {"docs": 0, "text_tokens": 0, "tokens_with_eot": 0})
    started = time.time()
    for rec in records:
        ids = enc.encode_ordinary(str(rec.get("text", "")))
        ids.append(eot_token)
        category = rec.get("category", "unknown")
        docs += 1
        text_tokens += len(ids) - 1
        total_tokens += len(ids)
        by_category[category]["docs"] += 1
        by_category[category]["text_tokens"] += len(ids) - 1
        by_category[category]["tokens_with_eot"] += len(ids)
        buf.extend(ids)
        if len(buf) >= chunk_tokens:
            flush(buf, out_path)
    flush(buf, out_path)
    return {"docs": docs, "text_tokens": text_tokens, "tokens_with_eot": total_tokens, "categories": dict(by_category), "seconds": round(time.time() - started, 3)}


def build_one(splits_root: Path, out_root: Path, enc, seed: int, category: str | None, chunk_tokens: int, force: bool) -> None:
    if out_root.exists():
        if force:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"{out_root} exists; pass --force")
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": "gpt2", "encoding": "gpt2", "vocab_size": enc.n_vocab, "eot_token": enc.eot_token, "dtype": "uint16", "seed": seed, "category": category or "mixed", "splits": {}}
    for split in SPLITS:
        records = read_records(splits_root / f"{split}.jsonl", category)
        meta["splits"][split] = write_stream(records, out_root / f"{split}.bin", enc, seed + int(hashlib.sha1(f"{category}:{split}".encode()).hexdigest()[:8], 16) % 1_000_000, chunk_tokens, enc.eot_token)
    meta["bin_lengths"] = {split: (out_root / f"{split}.bin").stat().st_size // 2 for split in SPLITS}
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"built {category or 'mixed'} -> {out_root}")


def discover_categories(splits_root: Path) -> list[str]:
    categories: set[str] = set()
    for split in SPLITS:
        path = splits_root / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                categories.add(str(json.loads(line).get("category", "unknown")))
    return sorted(categories)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 shuffled token streams.")
    parser.add_argument("--splits-root", default="train_data/processed/splits")
    parser.add_argument("--output-root", default="train_data/tokenized/gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-tokens", type=int, default=1_000_000)
    parser.add_argument("--mode", choices=["mixed", "sources", "all"], default="all")
    parser.add_argument("--categories", nargs="*", default=None, help="Source categories to build. Default: discover from split JSONL files.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    enc = tiktoken.get_encoding("gpt2")
    splits_root = Path(args.splits_root)
    out_root = Path(args.output_root)
    if args.mode in ("mixed", "all"):
        build_one(splits_root, out_root / "mixed", enc, args.seed, None, args.chunk_tokens, args.force)
    if args.mode in ("sources", "all"):
        categories = args.categories or discover_categories(splits_root) or CATEGORIES
        for category in categories:
            build_one(splits_root, out_root / "sources" / category, enc, args.seed, category, args.chunk_tokens, args.force)


if __name__ == "__main__":
    main()
