#!/usr/bin/env bash
#SBATCH --job-name=demo-matrix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x-%j.out
#SBATCH --error=slurm_log/%x-%j.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MINIFORGE="${MINIFORGE:-/home/tnx/miniforge3}"
ENV_NAME="${ENV_NAME:-scaling}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/train.yaml}"
MATRIX="${1:-${MATRIX:-}}"
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "$MATRIX" ]]; then echo "usage: sbatch slurm/run_matrix.sbatch <matrix.csv>"; exit 2; fi
cd "$PROJECT_DIR"
mkdir -p logs results slurm_log
source "$MINIFORGE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CONFIG MATRIX DRY_RUN
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "job=${SLURM_JOB_ID:-none} matrix=$MATRIX dry_run=$DRY_RUN"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not available in this shell"; fi
python slurm/run_matrix.py
