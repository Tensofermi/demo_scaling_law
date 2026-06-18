"""Sequential token datasets for demo_scaling_law.

Unlike the old random-start memmap sampler, this reader advances a cursor through
one shuffled token stream. It records epoch progress and wraps only when the run
explicitly consumes more tokens than the stream contains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TokenStream:
    bucket_id: str
    train: np.memmap
    val: np.memmap
    weight: float = 1.0
    train_cursor: int = 0
    train_tokens_consumed: int = 0


class SequentialTokenDataset:
    def __init__(self, root: str | Path, block_size: int, bucket_weights: dict[str, float] | None = None) -> None:
        self.root = Path(root)
        self.block_size = int(block_size)
        self.streams: list[TokenStream] = []
        direct_train = self.root / "train.bin"
        direct_val = self.root / "val.bin"
        if direct_train.exists() and direct_val.exists():
            self._append_stream("global", direct_train, direct_val, (bucket_weights or {}).get("global", 1.0))
        for bucket_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            train_path = bucket_dir / "train.bin"
            val_path = bucket_dir / "val.bin"
            if train_path.exists() and val_path.exists():
                self._append_stream(bucket_dir.name, train_path, val_path, (bucket_weights or {}).get(bucket_dir.name, 1.0))
        if not self.streams:
            raise FileNotFoundError(
                f"no usable token stream found under {self.root}; "
                f"train.bin and val.bin must both contain more than block_size+1={self.block_size + 1} tokens. "
                "Use more data or lower model_family.block_size for tiny smoke runs."
            )
        weights = np.array([s.weight for s in self.streams], dtype=np.float64)
        self.probs = weights / weights.sum()

    def _append_stream(self, bucket_id: str, train_path: Path, val_path: Path, weight: float) -> None:
        if train_path.stat().st_size == 0 or val_path.stat().st_size == 0:
            return
        train = np.memmap(train_path, dtype=np.uint16, mode="r")
        val = np.memmap(val_path, dtype=np.uint16, mode="r")
        if len(train) <= self.block_size + 1 or len(val) <= self.block_size + 1:
            return
        self.streams.append(TokenStream(bucket_id=bucket_id, train=train, val=val, weight=float(weight)))

    @property
    def bucket_ids(self) -> list[str]:
        return [s.bucket_id for s in self.streams]

    @property
    def train_tokens_available(self) -> int:
        return int(sum(len(s.train) for s in self.streams))

    def _stream_for(self, bucket_id: str) -> TokenStream:
        for stream in self.streams:
            if stream.bucket_id == bucket_id:
                return stream
        raise KeyError(bucket_id)

    def _batch_from_starts(self, arr: np.ndarray, starts: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = np.arange(self.block_size + 1, dtype=np.int64)
        rows = [np.asarray(arr[(int(start) + offsets) % len(arr)], dtype=np.int64) for start in starts]
        batch = torch.from_numpy(np.stack(rows))
        x = batch[:, :-1]
        y = batch[:, 1:]
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
        if split != "train":
            return self.get_eval_batch(self.streams[0].bucket_id, split, batch_size, 0, device) + (self.streams[0].bucket_id,)
        idx = int(np.random.choice(len(self.streams), p=self.probs)) if len(self.streams) > 1 else 0
        stream = self.streams[idx]
        stride = self.block_size
        starts = (stream.train_cursor + np.arange(batch_size, dtype=np.int64) * stride) % (len(stream.train) - self.block_size - 1)
        stream.train_cursor = int((stream.train_cursor + batch_size * stride) % (len(stream.train) - self.block_size - 1))
        stream.train_tokens_consumed += batch_size * stride
        x, y = self._batch_from_starts(stream.train, starts, device)
        return x, y, stream.bucket_id

    def get_eval_batch(self, bucket_id: str, split: str, batch_size: int, eval_iter: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        stream = self._stream_for(bucket_id)
        arr = stream.val if split == "val" else stream.train
        span = max(1, len(arr) - self.block_size - 1)
        start0 = (eval_iter * batch_size * self.block_size) % span
        starts = (start0 + np.arange(batch_size, dtype=np.int64) * self.block_size) % span
        return self._batch_from_starts(arr, starts, device)

    def epochs_seen(self) -> dict[str, float]:
        return {s.bucket_id: s.train_tokens_consumed / max(len(s.train), 1) for s in self.streams}


def load_meta(root: str | Path) -> dict:
    path = Path(root) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
