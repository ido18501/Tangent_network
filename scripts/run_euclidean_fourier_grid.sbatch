#!/bin/bash
#SBATCH --job-name=tangent_grid
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate LOS_Net

PROJECT_ROOT="$HOME/Tangent_network"
cd "$PROJECT_ROOT"
mkdir -p logs results

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

GRID_ROOT="results/euclidean_fourier_gt_grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$GRID_ROOT"

LAMBDA_FIRSTS=(0.1 1.0 5.0)
LAMBDA_SECONDS=(0.0 0.1 1.0)

for LF in "${LAMBDA_FIRSTS[@]}"; do
  for LS in "${LAMBDA_SECONDS[@]}"; do
    RUN_NAME="euclid_fourier_lf_${LF}_ls_${LS}"
    RUN_DIR="${GRID_ROOT}/${RUN_NAME}"
    mkdir -p "$RUN_DIR"

    python -u ./scripts.train_operator.py \
      --run-dir "$RUN_DIR" \
      --device cuda \
      --num-workers 4 \
      --train-length 5000 \
      --val-length 1500 \
      --test-length 1500 \
      --transform-family euclidean \
      --mixed-fourier-prob 1.0 \
      --mixed-piecewise-prob 0.0 \
      --point-noise-std 0.001 \
      --orthogonal-noise-std 0.002 \
      --warp-sampling-prob 0.4 \
      --warp-sampling-strength 0.10 \
      --batch-size 64 \
      --num-epochs 50 \
      --early-stopping-patience 8 \
      --lambda-first "$LF" \
      --lambda-second "$LS" \
      2>&1 | tee "$RUN_DIR/train.log"
  done
done
