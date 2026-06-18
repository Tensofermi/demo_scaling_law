"""Training entrypoint for demo_scaling_law.

The validation loss implementation is intentionally explicit: `estimate_loss`
is the single place that computes quick/full validation loss for train logs and
scaling-law analysis.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import load_yaml
from .dataset import SequentialTokenDataset, load_meta
from .model import GPT, GPTConfig, depth_to_config
from .utils import choose_device, configure_cuda_performance, count_parameters, device_report, ensure_dir, format_optional_float, max_memory_gb, peak_bf16_flops, set_seed, write_json


def build_model_config(cfg: dict[str, Any], vocab_size: int, depth: int | None) -> GPTConfig:
    family = cfg.get("model_family", {})
    chosen_depth = int(depth if depth is not None else family.get("default_depth", 4))
    return depth_to_config(
        chosen_depth,
        vocab_size=int(vocab_size),
        block_size=int(family.get("block_size", 1024)),
        head_dim=int(family.get("head_dim", 128)),
        aspect_ratio=int(family.get("aspect_ratio", 64)),
        dropout=float(family.get("dropout", 0.0)),
        bias=bool(family.get("bias", False)),
        use_sdpa=bool(family.get("use_sdpa", True)),
        rope_base=float(family.get("rope_base", 10000.0)),
    )


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any], device: torch.device) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = bool(train_cfg.get("fused_adamw", device.type == "cuda" and fused_available))
    kwargs = {"fused": True} if use_fused else {}
    lr = float(train_cfg.get("learning_rate", 3e-4))
    betas = tuple(train_cfg.get("betas", [0.9, 0.95]))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
    for group in optimizer.param_groups:
        group["base_lr"] = lr
    return optimizer, {"optimizer": "AdamW", "learning_rate": lr, "betas": list(betas), "weight_decay": weight_decay, "fused_adamw": use_fused}


def lr_multiplier(iter_idx: int, max_iters: int, train_cfg: dict[str, Any]) -> float:
    min_lr_frac = float(train_cfg.get("min_lr_frac", 0.1))
    warmup_cfg = train_cfg.get("warmup_steps", "auto")
    if warmup_cfg == "auto":
        warmup = min(200, max(20, int(round(0.02 * max_iters))))
    else:
        warmup = int(warmup_cfg)
    if warmup > 0 and iter_idx < warmup:
        return (iter_idx + 1) / warmup
    if max_iters <= warmup:
        return 1.0
    progress = (iter_idx - warmup) / max(1, max_iters - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_frac + (1.0 - min_lr_frac) * cosine


def apply_lr(optimizer: torch.optim.Optimizer, multiplier: float) -> float:
    lr = 0.0
    for group in optimizer.param_groups:
        lr = float(group["base_lr"]) * multiplier
        group["lr"] = lr
    return lr


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: SequentialTokenDataset,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    """Compute validation loss over deterministic validation windows.

    Training logs call this for both quick validation and full validation. The
    function never reads train loss rows and never mutates the train cursor.
    """

    model.eval()
    losses: list[float] = []
    per_bucket: dict[str, float] = {}
    for bucket_id in dataset.bucket_ids:
        bucket_losses = []
        for i in range(max(1, eval_iters)):
            x, y = dataset.get_eval_batch(bucket_id, "val", batch_size, i, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            item = float(loss.item())
            bucket_losses.append(item)
            losses.append(item)
        per_bucket[bucket_id] = sum(bucket_losses) / len(bucket_losses)
    model.train()
    return sum(losses) / len(losses), per_bucket


def checkpoint_iters(max_iters: int, n: int) -> set[int]:
    return {max(1, int(round(max_iters * i / n))) for i in range(1, n + 1)}


def save_checkpoint(path: Path, model: GPT, model_config: GPTConfig, iter_idx: int, best_val: float, metadata: dict[str, Any]) -> None:
    torch.save({"model": model.state_dict(), "model_config": asdict(model_config), "iter": iter_idx, "best_val_loss": best_val, **metadata}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train demo_scaling_law RoPE GPT.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--target-flops", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)
    configure_cuda_performance(device, bool(train_cfg.get("allow_tf32", True)))

    data_dir = Path(args.data_dir or cfg.get("data", {}).get("tokenized_dir", "train_data/tokenized/gpt2/mixed"))
    meta = load_meta(data_dir)
    vocab_size = int(meta.get("vocab_size", cfg.get("tokenizer", {}).get("vocab_size", 50257)))
    model_config = build_model_config(cfg, vocab_size, args.depth)
    dataset = SequentialTokenDataset(data_dir, model_config.block_size, cfg.get("data", {}).get("bucket_weights"))

    model = GPT(model_config).to(device)
    raw_model = model
    model_card = model.export_model_card()
    param_breakdown = model_card["param_breakdown"]
    params_total = int(param_breakdown["params_total"])
    flops_per_token = int(model_card["flops_per_token"])

    batch_size = int(train_cfg.get("batch_size", 16))
    grad_accum = int(train_cfg.get("grad_accum_steps", 8))
    tokens_per_iter = batch_size * grad_accum * model_config.block_size
    if args.max_iters is not None:
        max_iters = int(args.max_iters)
        target_tokens = max_iters * tokens_per_iter
        target_flops = 6.0 * params_total * target_tokens
    elif args.target_tokens is not None:
        target_tokens = int(args.target_tokens)
        max_iters = max(1, math.ceil(target_tokens / tokens_per_iter))
        target_flops = 6.0 * params_total * target_tokens
    elif args.target_flops is not None:
        target_flops = float(args.target_flops)
        target_tokens = max(1, int(target_flops // max(6 * params_total, 1)))
        max_iters = max(1, math.ceil(target_tokens / tokens_per_iter))
    else:
        max_iters = int(train_cfg.get("max_iters", 1000))
        target_tokens = max_iters * tokens_per_iter
        target_flops = 6.0 * params_total * target_tokens
    actual_tokens_planned = max_iters * tokens_per_iter
    actual_flops_planned = 6.0 * params_total * actual_tokens_planned

    dtype_name = str(train_cfg.get("dtype", "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"))
    amp_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16 if dtype_name == "float16" else torch.float32
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    compile_enabled = False
    if bool(train_cfg.get("compile", device.type == "cuda")) and device.type == "cuda":
        try:
            model = torch.compile(model, dynamic=False)
            compile_enabled = True
        except Exception as exc:
            print(f"WARNING torch.compile failed, fallback eager: {exc}")

    optimizer, optimizer_report = build_optimizer(raw_model, train_cfg, device)
    quick_eval_interval = int(train_cfg.get("quick_eval_interval", 10))
    quick_eval_iters = int(train_cfg.get("quick_eval_iters", 2))
    full_eval_interval = int(train_cfg.get("full_eval_interval", 500))
    full_eval_iters = int(train_cfg.get("full_eval_iters", 20))
    checkpoint_count = int(train_cfg.get("checkpoint_count", 8))
    ckpt_iters = checkpoint_iters(max_iters, checkpoint_count)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    peak_flops = peak_bf16_flops(device)

    out_dir = ensure_dir(Path(args.output_root or cfg.get("output", {}).get("root", "logs")) / args.run_id)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    log_path = out_dir / "train_log.csv"
    bucket_log_path = out_dir / "bucket_val_log.csv"
    best_path = ckpt_dir / "best.pt"
    final_path = ckpt_dir / "final.pt"
    metadata = {"params_total": params_total, "param_breakdown": param_breakdown, "flops_per_token": flops_per_token}

    print(f"run_id={args.run_id}")
    print(f"device_report={device_report(device)}")
    print(f"model_config={asdict(model_config)}")
    print(f"param_breakdown={param_breakdown}")
    print(f"tokens_per_iter={tokens_per_iter} max_iters={max_iters} target_tokens={target_tokens} target_flops={target_flops:.4e}")
    print(f"optimizer_report={optimizer_report}")

    best_val = float("inf")
    last_perf_time = time.time()
    last_perf_tokens = 0
    started = last_perf_time
    last_train_loss = ""

    with log_path.open("w", newline="", encoding="utf-8") as lf, bucket_log_path.open("w", newline="", encoding="utf-8") as bf:
        fields = ["iter", "tokens_seen", "flops_seen", "train_loss", "val_loss", "full_val_loss", "lr", "lr_multiplier", "wall_time_sec", "tokens_per_sec", "tflops_per_sec", "mfu", "max_memory_gb", "epochs_seen"]
        writer = csv.DictWriter(lf, fieldnames=fields)
        bucket_writer = csv.DictWriter(bf, fieldnames=["iter", "bucket_id", "val_loss", "eval_type"])
        writer.writeheader(); bucket_writer.writeheader()

        for it in range(max_iters + 1):
            lrm = lr_multiplier(it, max_iters, train_cfg)
            lr = apply_lr(optimizer, lrm)
            do_quick = it % quick_eval_interval == 0 or it == max_iters
            do_full = it % full_eval_interval == 0 or it == max_iters
            if do_quick or do_full:
                val_loss = ""
                full_val_loss = ""
                per_bucket: dict[str, float] = {}
                if do_quick:
                    val_loss, per_bucket = estimate_loss(model, dataset, batch_size, quick_eval_iters, device, amp_dtype, use_amp)
                    for bucket_id, loss in per_bucket.items():
                        bucket_writer.writerow({"iter": it, "bucket_id": bucket_id, "val_loss": loss, "eval_type": "quick"})
                if do_full:
                    full_val_loss, per_bucket = estimate_loss(model, dataset, batch_size, full_eval_iters, device, amp_dtype, use_amp)
                    for bucket_id, loss in per_bucket.items():
                        bucket_writer.writerow({"iter": it, "bucket_id": bucket_id, "val_loss": loss, "eval_type": "full"})
                    if float(full_val_loss) < best_val and it > 0:
                        best_val = float(full_val_loss)
                        save_checkpoint(best_path, raw_model, model_config, it, best_val, metadata)
                tokens_seen = it * tokens_per_iter
                flops_seen = 6.0 * params_total * tokens_seen
                writer.writerow({
                    "iter": it,
                    "tokens_seen": tokens_seen,
                    "flops_seen": flops_seen,
                    "train_loss": last_train_loss,
                    "val_loss": val_loss,
                    "full_val_loss": full_val_loss,
                    "lr": lr,
                    "lr_multiplier": lrm,
                    "wall_time_sec": round(time.time() - started, 3),
                    "tokens_per_sec": "",
                    "tflops_per_sec": "",
                    "mfu": "",
                    "max_memory_gb": format_optional_float(max_memory_gb(device), 3),
                    "epochs_seen": dataset.epochs_seen(),
                })
                lf.flush(); bf.flush()
                shown = full_val_loss if full_val_loss != "" else val_loss
                print(f"eval iter={it} tokens={tokens_seen} loss={float(shown):.4f} full={full_val_loss != ''}")
            if it in ckpt_iters and it > 0:
                save_checkpoint(ckpt_dir / f"iter_{it:07d}.pt", raw_model, model_config, it, best_val, metadata)
            if it == max_iters:
                break

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(grad_accum):
                x, y, _ = dataset.get_batch("train", batch_size, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                loss.backward()
                total_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()
            last_train_loss = total_loss
            if (it + 1) % int(train_cfg.get("log_interval", 10)) == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.time()
                tokens_seen = (it + 1) * tokens_per_iter
                delta_tokens = tokens_seen - last_perf_tokens
                dt = max(now - last_perf_time, 1e-9)
                tps = delta_tokens / dt
                flops_per_sec = 6.0 * params_total * delta_tokens / dt
                mfu = flops_per_sec / peak_flops if peak_flops else None
                last_perf_time = now; last_perf_tokens = tokens_seen
                print(f"train iter={it+1} loss={total_loss:.4f} tok/s={tps:,.0f} tflops={flops_per_sec/1e12:.2f} mfu={format_optional_float(mfu, 4)}")

    save_checkpoint(final_path, raw_model, model_config, max_iters, best_val, metadata)
    metrics = {
        "run_id": args.run_id,
        "depth": model_config.depth,
        "model_config": asdict(model_config),
        "params_total": params_total,
        "params_plan": params_total,
        "params_transformer": param_breakdown["params_transformer"],
        "params_token_embedding": param_breakdown["params_token_embedding"],
        "params_pos_embedding": 0,
        "params_non_embedding_old": param_breakdown["params_non_embedding_old"],
        "param_breakdown": param_breakdown,
        "flops_per_token": flops_per_token,
        "target_flops": target_flops,
        "target_tokens": target_tokens,
        "actual_tokens_planned": actual_tokens_planned,
        "actual_flops_planned": actual_flops_planned,
        "tokens_per_iter": tokens_per_iter,
        "max_iters": max_iters,
        "seed": seed,
        "best_val_loss": best_val,
        "device": device_report(device),
        "fast_backend": {"compile": compile_enabled, "allow_tf32": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False, "max_memory_gb": max_memory_gb(device)},
        "optimizer": optimizer_report,
        "log_path": str(log_path),
        "bucket_log_path": str(bucket_log_path),
        "checkpoint_dir": str(ckpt_dir),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "epochs_seen": dataset.epochs_seen(),
    }
    write_json(out_dir / "metrics.json", metrics)
    print(f"done metrics={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
