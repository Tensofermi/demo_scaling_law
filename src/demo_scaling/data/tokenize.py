"""把 bucket JSONL 编码为 nanoGPT 风格 memmap。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ..config import load_yaml
from ..tokenizer import build_tokenizer
from ..utils import ensure_dir, write_json


_TOKENIZER = None
_EOT_ID = None


def init_worker(kind: str, encoding: str) -> None:
    global _TOKENIZER, _EOT_ID
    _TOKENIZER = build_tokenizer(kind, encoding_name=encoding)
    _EOT_ID = _TOKENIZER.enc.eot_token


def encode_jsonl(path: Path, tokenizer, eot_id: int) -> np.ndarray:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line).get("text", "")
            ids.extend(tokenizer.encode(text))
            ids.append(eot_id)
    return np.asarray(ids, dtype=np.uint16)


def tokenize_one(task: tuple[str, str, str, str]) -> tuple[str, str, int]:
    bucket_id, split, src_s, out_s = task
    if _TOKENIZER is None or _EOT_ID is None:
        init_worker("gpt2", "gpt2")
    src = Path(src_s)
    out = Path(out_s)
    arr = encode_jsonl(src, _TOKENIZER, _EOT_ID)  # type: ignore[arg-type]
    arr.tofile(out)
    return bucket_id, split, int(len(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize bucket JSONL files.")
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--input", default="data/buckets")
    parser.add_argument("--output", default="data/tokenized")
    parser.add_argument("--workers", type=int, default=1, help="CPU worker 数；例如 30")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    kind = cfg.get("kind", "gpt2")
    encoding = cfg.get("encoding", "gpt2")
    tokenizer = build_tokenizer(kind, encoding_name=encoding)
    eot_id = tokenizer.enc.eot_token
    out_root = ensure_dir(args.output)
    bucket_meta = {}
    tasks: list[tuple[str, str, str, str]] = []
    for bucket_dir in sorted(p for p in Path(args.input).iterdir() if p.is_dir()):
        out_dir = ensure_dir(out_root / bucket_dir.name)
        bucket_meta[bucket_dir.name] = {}
        for split in ["train", "val"]:
            src = bucket_dir / f"{split}.jsonl"
            if not src.exists():
                continue
            tasks.append((bucket_dir.name, split, str(src), str(out_dir / f"{split}.bin")))

    if args.workers <= 1:
        init_worker(kind, encoding)
        results = [tokenize_one(task) for task in tasks]
    else:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(kind, encoding)) as pool:
            results = list(pool.imap_unordered(tokenize_one, tasks, chunksize=1))

    for bucket_id, split, n_tokens in sorted(results):
        bucket_meta[bucket_id][split] = n_tokens
        print(f"{bucket_id}/{split}: {n_tokens:,} tokens")
    write_json(
        out_root / "meta.json",
        {
            "tokenizer": tokenizer.name,
            "encoding": tokenizer.encoding_name,
            "vocab_size": tokenizer.vocab_size,
            "eot_token": eot_id,
            "buckets": bucket_meta,
        },
    )


if __name__ == "__main__":
    main()
