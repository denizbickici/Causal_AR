#!/bin/bash
#SBATCH --job-name=causal-vjepa
#SBATCH --output=logs/causal-vjepa-%j.out
#SBATCH --error=logs/causal-vjepa-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

# Activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <env_name>

cd /home/dz/Projects/multi-modal_AR/causal_default

# Create log dir if it doesn't exist
mkdir -p logs

python3 main_lavila_causal.py \
  --gpu 0 \
  --num_thread_reader 8 \
  --batch_size 8 \
  --batch_size_val 8 \
  --seq_len 16 \
  --verb_dim 1280 \
  --noun_dim 1280 \
  --action_dim 97 \
  --epochs 800 \
  --lr 1e-4 \
  --checkpoint_dir vjepa_full16 \
  --cudnn_benchmark 1 \
  --pin_memory \
  --dataset ek100 \
  "$@"
