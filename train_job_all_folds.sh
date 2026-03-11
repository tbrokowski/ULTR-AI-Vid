#!/bin/bash
#SBATCH --time=09:59:59
#SBATCH --partition=normal
#SBATCH -A a127
#SBATCH --job-name=ultrai_train
#SBATCH --output=logs/train_fold%a_%j.out
#SBATCH --error=logs/train_fold%a_%j.err
#SBATCH --array=0-4

set -euo pipefail
mkdir -p logs

FOLD=${SLURM_ARRAY_TASK_ID}
CONFIG="configs/cscs/tb_drl_mil_fold${FOLD}.yaml"

echo "Running fold ${FOLD} with config ${CONFIG}"

srun --environment=/users/lxflk/.edf/ultrai.toml \
     python3 train_clip_drl_mil_Final-2.py \
     --config "${CONFIG}"