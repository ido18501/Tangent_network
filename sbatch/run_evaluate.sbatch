#!/bin/bash
#SBATCH --job-name=tangent_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail

cd ~/Tangent_network

# Initialize conda for non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate cs236781-hw

mkdir -p logs
mkdir -p results/eval_warmup

CHECKPOINT="checkpoints_temp_test/best_model.pt"

python -u -m scripts.evaluate_model \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --output-dir results/eval_warmup \
  --family euclidean
