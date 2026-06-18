from __future__ import annotations

import csv
import os
import subprocess
import sys


def main() -> None:
    matrix = os.environ["MATRIX"]
    config = os.environ["CONFIG"]
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    with open(matrix, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = row["run_id"]
            metrics = os.path.join("logs", run_id, "metrics.json")
            if os.path.exists(metrics):
                print(f"skip_existing {run_id}", flush=True)
                continue
            cmd = [sys.executable, "-m", "demo_scaling.train", "--config", config, "--run-id", run_id, "--depth", row["depth"], "--data-dir", row["data_dir"], "--device", "cuda"]
            if row.get("max_iters"):
                cmd += ["--max-iters", row["max_iters"]]
            elif row.get("target_flops"):
                cmd += ["--target-flops", row["target_flops"]]
            print("RUN", " ".join(cmd), flush=True)
            if not dry_run:
                subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
