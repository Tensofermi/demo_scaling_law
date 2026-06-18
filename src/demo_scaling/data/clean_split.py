"""Clean, deduplicate, and split raw JSONL docs into train/val/test."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


SPACE_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip().lower())


def clean_text(text: str, min_chars: int) -> str | None:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = SPACE_RE.sub(" ", text).strip()
    if len(text) < min_chars:
        return None
    return text


def read_raw(root: Path, min_chars: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.jsonl")):
        if path.name == "manifest_resolved.json":
            continue
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = clean_text(str(rec.get("text", "")), min_chars)
                if text is None:
                    continue
                h = hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rows.append({"doc_id": rec.get("doc_id", f"{path.stem}_{line_no}"), "source": rec.get("source", path.stem), "category": rec.get("category", path.stem), "text": text})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean/deduplicate raw docs and split 98/1/1 by document.")
    parser.add_argument("--input", default="train_data/raw")
    parser.add_argument("--output", default="train_data/processed/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=20)
    args = parser.parse_args()
    rows = read_raw(Path(args.input), args.min_chars)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, items in splits.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, rec in enumerate(items):
                rec = {**rec, "split": split, "doc_id": rec.get("doc_id", f"{split}_{i}")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary[split] = len(items)
        print(f"{split}: {len(items):,} docs -> {path}")
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
