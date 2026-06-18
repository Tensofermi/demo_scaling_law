"""Download or materialize raw JSONL documents from a manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

from demo_scaling.config import load_yaml


def clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def row_to_text(row: dict, fields: list[str] | None) -> str:
    if fields:
        parts = [str(row.get(field, "") or "").strip() for field in fields]
        return "\n\n".join(part for part in parts if part)
    parts = [v.strip() for v in row.values() if isinstance(v, str) and len(v.strip()) >= 20]
    return "\n\n".join(parts)


def synthetic_docs(source: dict, max_docs: int) -> Iterable[dict]:
    category = source.get("category", source.get("id", "synthetic"))
    templates = {
        "story": "Once upon a time, a small model learned from a tiny story about scaling laws.",
        "code": "def train_step(model, batch):\n    loss = model(batch)\n    loss.backward()\n    return loss",
        "dialogue": "User: Can you explain scaling laws?\nAssistant: They relate loss, model size, data, and compute.",
        "math": "Let C = 6ND. If C is fixed, increasing N decreases D proportionally.",
        "news": "Researchers reported a compact experiment on language model scaling and data efficiency.",
        "encyclopedia": "Scaling law is an empirical relationship between model performance and resources.",
    }
    base = templates.get(category, templates["story"])
    for i in range(max_docs):
        yield {"doc_id": f"{source.get('id', category)}_{i}", "source": source.get("source", source.get("id", category)), "category": category, "text": f"{base} Example {i}."}


def hf_docs(source: dict, max_docs: int) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HuggingFace data sources require: pip install -e .[data]") from exc
    ds = load_dataset(
        source["dataset"],
        source.get("dataset_config"),
        split=source.get("split", "train"),
        streaming=bool(source.get("streaming", True)),
        trust_remote_code=bool(source.get("trust_remote_code", False)),
    )
    fields = source.get("text_fields")
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        text = clean_text(row_to_text(dict(row), fields))
        if text:
            yield {"doc_id": f"{source.get('id')}_{i}", "source": source.get("source", source.get("id")), "category": source.get("category", source.get("id")), "text": text}


def local_jsonl_docs(source: dict, max_docs: int) -> Iterable[dict]:
    with Path(source["path"]).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            text = clean_text(str(rec.get("text", "")))
            if text:
                yield {"doc_id": rec.get("doc_id", f"{source.get('id')}_{i}"), "source": source.get("source", rec.get("source", source.get("id"))), "category": source.get("category", rec.get("category", source.get("id"))), "text": text}


def url_jsonl_docs(source: dict, max_docs: int) -> Iterable[dict]:
    tmp = Path(source.get("cache_path", f"/tmp/{source.get('id')}.jsonl"))
    if not tmp.exists():
        urllib.request.urlretrieve(source["url"], tmp)
    yield from local_jsonl_docs({**source, "path": str(tmp)}, max_docs)


def iter_source(source: dict) -> Iterable[dict]:
    max_docs = int(source.get("max_docs", 1000))
    typ = source.get("type", "hf_dataset")
    if typ == "synthetic":
        return synthetic_docs(source, max_docs)
    if typ == "hf_dataset":
        return hf_docs(source, max_docs)
    if typ == "local_jsonl":
        return local_jsonl_docs(source, max_docs)
    if typ == "url_jsonl":
        return url_jsonl_docs(source, max_docs)
    raise ValueError(f"unknown source type: {typ}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/materialize raw source JSONL files.")
    parser.add_argument("--manifest", default="train_data/manifests/default_sources.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--keep-going", action="store_true", help="Continue downloading later sources if one source fails.")
    args = parser.parse_args()
    cfg = load_yaml(args.manifest)
    out_dir = Path(args.output or cfg.get("output_dir", "train_data/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_log = {"manifest": args.manifest, "sources": []}
    for source in cfg.get("sources", []):
        out_path = out_dir / f"{source.get('id')}.jsonl"
        n = 0
        try:
            with out_path.open("w", encoding="utf-8") as f:
                for rec in iter_source(source):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            manifest_log["sources"].append({**source, "output": str(out_path), "docs": n, "status": "ok"})
            print(f"wrote {n:,} docs -> {out_path}")
        except Exception as exc:
            if out_path.exists() and n == 0:
                out_path.unlink()
            manifest_log["sources"].append({**source, "output": str(out_path), "docs": n, "status": "failed", "error": repr(exc)})
            (out_dir / "manifest_resolved.json").write_text(json.dumps(manifest_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            msg = f"source {source.get('id')} ({source.get('dataset', source.get('url', source.get('path', 'unknown')))} failed: {exc}"
            if not args.keep_going:
                raise RuntimeError(msg) from exc
            print(f"WARNING {msg}", flush=True)
    (out_dir / "manifest_resolved.json").write_text(json.dumps(manifest_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(source.get("type", "hf_dataset") == "hf_dataset" for source in cfg.get("sources", [])):
        # datasets streaming can leave background finalizers that crash at interpreter
        # shutdown on some clusters after all files are already written. Exit after
        # flushing so the CLI reflects the successful materialization.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
