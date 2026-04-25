#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <fold_id>" >&2
  exit 1
fi

FOLD="$1"
REPO_ROOT="/users/lxflk/ULTR-AI-Vid"
RUN_ROOT="/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/20260421__train__benin__all-folds__hmv-mil-restore-1gpu-v2"
CFG="${RUN_ROOT}/fold${FOLD}/resolved_config.yaml"

if [[ ! -f "${CFG}" ]]; then
  echo "Resolved config not found: ${CFG}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME="${ULTRAI_NCCL_SOCKET_IFNAME:-lo}"
export NCCL_PLUGIN_P2P=0
unset LD_PRELOAD
ulimit -c 0 || true

echo "Running eval-only for fold ${FOLD}"
echo "Config: ${CFG}"

torchrun \
  --standalone \
  --nproc_per_node 4 \
  --max_restarts 0 \
  -m ultrai.training.train \
  --dataset benin \
  --config "${CFG}" \
  --eval_only
