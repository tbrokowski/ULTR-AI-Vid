#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH -A a127
#SBATCH --job-name=ultrai_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
mkdir -p logs

# Run the training inside the EDF environment (container)
srun --environment=/users/lxflk/.edf/ultrai.toml \
     python3 train_clip_drl_mil_Final-2.py \
     --config configs/cscs/tb_drl_mil_fold0.yaml