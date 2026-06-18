from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo_scaling.dataset import SequentialTokenDataset
from demo_scaling.model import GPT, depth_to_config


class ModelTests(unittest.TestCase):
    def test_rope_model_has_no_position_embedding_params(self):
        cfg = depth_to_config(1, vocab_size=128, block_size=16)
        model = GPT(cfg)
        names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any("pos_emb" in name for name in names))
        breakdown = model.get_param_breakdown()
        self.assertEqual(breakdown["params_pos_embedding"], 0)
        self.assertEqual(breakdown["params_total"], sum(p.numel() for p in model.parameters()))

    def test_forward_backward(self):
        cfg = depth_to_config(1, vocab_size=128, block_size=16)
        model = GPT(cfg)
        x = torch.randint(0, 128, (2, 16))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 16, 128))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class DatasetTests(unittest.TestCase):
    def test_sequential_cursor_advances(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arr = np.arange(200, dtype=np.uint16)
            arr.tofile(root / "train.bin")
            arr.tofile(root / "val.bin")
            (root / "meta.json").write_text(json.dumps({"vocab_size": 256}), encoding="utf-8")
            ds = SequentialTokenDataset(root, block_size=8)
            x1, _, _ = ds.get_batch("train", 2, torch.device("cpu"))
            x2, _, _ = ds.get_batch("train", 2, torch.device("cpu"))
            self.assertEqual(x1[0, 0].item(), 0)
            self.assertEqual(x2[0, 0].item(), 16)
            self.assertGreater(ds.epochs_seen()["global"], 0)


class TrainSmokeTests(unittest.TestCase):
    def test_cpu_train_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data = tmp / "data"
            data.mkdir()
            arr = np.arange(4096, dtype=np.uint16) % 128
            arr.tofile(data / "train.bin")
            arr.tofile(data / "val.bin")
            (data / "meta.json").write_text(json.dumps({"vocab_size": 128}), encoding="utf-8")
            cfg = tmp / "train.yaml"
            cfg.write_text(
                """
model_family:
  type: depth
  default_depth: 1
  head_dim: 64
  aspect_ratio: 64
  block_size: 16
  vocab_size: 128
  bias: false
  dropout: 0.0
  use_sdpa: true
  rope_base: 10000.0
data:
  tokenized_dir: DATA_DIR
train:
  seed: 1
  learning_rate: 0.0003
  min_lr_frac: 0.1
  warmup_steps: 1
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  batch_size: 2
  grad_accum_steps: 1
  dtype: float32
  compile: false
  fused_adamw: false
  allow_tf32: false
  log_interval: 1
  quick_eval_interval: 1
  quick_eval_iters: 1
  full_eval_interval: 1
  full_eval_iters: 1
  checkpoint_count: 2
output:
  root: OUT_DIR
""".replace("DATA_DIR", str(data)).replace("OUT_DIR", str(tmp / "logs")),
                encoding="utf-8",
            )
            cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(cfg), "--run-id", "smoke", "--depth", "1", "--max-iters", "2", "--device", "cpu"]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH','')}"
            subprocess.check_call(cmd, env=env)
            self.assertTrue((tmp / "logs" / "smoke" / "metrics.json").exists())
            self.assertTrue((tmp / "logs" / "smoke" / "checkpoints" / "final.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
