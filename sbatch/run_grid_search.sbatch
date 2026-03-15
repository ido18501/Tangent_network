#!/bin/bash
#SBATCH --job-name=tangent_grid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail

cd ~/Tangent_network

# Initialize conda for non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate cs236781-hw

mkdir -p logs
mkdir -p results

# Optional debug check (very useful in cluster logs)
python -c "import torch; print('Python:', __import__('sys').executable); print('CUDA available:', torch.cuda.is_available())"

python -u -m scripts.grid_search \
  --output-root results/grid_search_01 \
  --family euclidean \
  --patch-sizes 9,11 \
  --embedding-dims 64,128 \
  --temperatures 0.05,0.07,0.1 \
  --negative-min-offsets 6,15 \
  --negative-max-offsets 30,40 \
  --num-epochs 25 \
  --patience 6
