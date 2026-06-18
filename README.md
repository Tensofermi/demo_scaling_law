# demo_scaling_law

A small, reproducible infrastructure demo for LLM pretraining scaling-law experiments.

The repo is designed as an open-source template: it keeps code, configs, data manifests,
Slurm runners, planner utilities, plotting/fitting scripts, and CPU smoke tests. It does
**not** commit raw datasets, tokenized bins, checkpoints, historical logs, or large figures.

## Layout

- `src/demo_scaling/`: RoPE decoder-only GPT, tokenizer, sequential dataset, training, inference, analysis.
- `train_data/`: manifests and docs for download/clean/split/metrics/token stream preparation.
- `planner/`: model-size, FLOPs, token-budget, and matrix planning tools.
- `slurm/`: Slurm matrix runner and CPU smoke script. Job names use `demo-*`.
- `plot_fit/`: lightweight plotting/fitting entrypoints for generated logs.
- `configs/`: training and experiment configs.
- `tests/`: CPU tests.

## Install

```bash
cd /home/tnx/code/demo_scaling_law
pip install -e .
```

Optional dataset download support:

```bash
pip install -e ".[data]"
```

## CPU smoke test

```bash
python run_tests.py
```

## Data pipeline

```bash
python -m demo_scaling.data.download --manifest train_data/manifests/tiny_sources.yaml --output train_data/raw_tiny
python -m demo_scaling.data.clean_split --input train_data/raw_tiny --output train_data/processed/splits --seed 42
python -m demo_scaling.data.metrics --input train_data/processed/splits --output train_data/metrics/doc_metrics.csv --workers 1
python -m demo_scaling.data.build_streams --splits-root train_data/processed/splits --output-root train_data/tokenized/gpt2 --mode all --force
```

For the six-source template, use `train_data/manifests/default_sources.yaml`.
Check each upstream dataset card and license before redistributing data or trained artifacts.

## Planner

```bash
python planner/model_table.py --depths 1 3 5 7 9 11 13 15
python planner/make_matrix.py --config configs/experiments/smoke.yaml --output configs/experiments/smoke.csv
```

## Train

CPU smoke:

```bash
python -m demo_scaling.train --config configs/train_smoke.yaml --run-id smoke_d1 --depth 1 --max-iters 2 --data-dir train_data/tokenized/gpt2/mixed --device cpu
```

Slurm matrix:

```bash
DRY_RUN=1 bash slurm/run_matrix.sbatch configs/experiments/smoke.csv
sbatch slurm/run_matrix.sbatch configs/experiments/smoke.csv
```

## Inference

```bash
python -m demo_scaling.infer --checkpoint logs/smoke_d1/checkpoints/final.pt --prompt "Once upon a time"
python -m demo_scaling.stream_chat --checkpoint logs/smoke_d1/checkpoints/final.pt
```

## Plot

```bash
python -m demo_scaling.analysis.collect --logs logs --output results/runs.csv
python plot_fit/run_all.py --runs results/runs.csv --output plot_fit/outputs
```

## FLOPs convention

The default planning convention is `C ~= 6 * N_total * D`. The planner also reports a
nanoGPT-style attention-corrected FLOPs/token estimate for sensitivity checks.
