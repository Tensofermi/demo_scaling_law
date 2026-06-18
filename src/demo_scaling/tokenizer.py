"""Tokenizer 抽象。

第一版默认使用 GPT-2 byte-level BPE。它不是英文单词表，而是子词/字节级编码：
英文常见词可能是一个 token，中文或生僻文本会被拆成多个 token。
这里把 tokenizer 做成模块，是为了后续接入自训练 BPE 并比较 tokenization efficiency。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tiktoken


@dataclass
class TokenStats:
    chars: int
    tokens: int
    tokens_per_char: float
    chars_per_token: float


class GPT2Tokenizer:
    """GPT-2 tokenizer baseline。"""

    name = "gpt2"

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: Iterable[int]) -> str:
        return self.enc.decode(list(ids))

    def stats(self, text: str) -> TokenStats:
        ids = self.encode(text)
        chars = len(text)
        tokens = len(ids)
        return TokenStats(
            chars=chars,
            tokens=tokens,
            tokens_per_char=tokens / max(chars, 1),
            chars_per_token=chars / max(tokens, 1),
        )

    def save_meta(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"tokenizer={self.name}\nencoding={self.encoding_name}\nvocab_size={self.vocab_size}\n",
            encoding="utf-8",
        )


def build_tokenizer(kind: str = "gpt2", **kwargs) -> GPT2Tokenizer:
    """工厂函数；后续自训练 BPE 可以在这里扩展。"""

    if kind != "gpt2":
        raise ValueError(f"当前版本只实现 gpt2 tokenizer baseline，收到: {kind}")
    return GPT2Tokenizer(kwargs.get("encoding_name", "gpt2"))

