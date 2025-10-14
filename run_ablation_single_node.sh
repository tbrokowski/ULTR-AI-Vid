#!/usr/bin/env bash
#SBATCH --job-name=tb_ablation
#SBATCH --output=/users/mbarbiere/ULTR-AI/ULTR-AI-Vid/logs/R-%x.%j.out
#SBATCH --error=/users/mbarbiere/ULTR-AI/ULTR-AI-Vid/logs/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=0:59:59
#SBATCH --environment /users/mbarbiere/.edf/run_ai.toml
#SBATCH -A a127

set -euo pipefail
echo "START TIME: $(date)"

CONFIG_FILE="${1:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Usage: sbatch run_ablation_single_node.sh configs/<abl>/foldX.yaml [--epochs 1 ...]"; exit 1
fi

# Allow optional extra args (e.g., --epochs 1) to be forwarded to Python trainer
shift || true
# Centralize dataset overrides so train and eval stay consistent
VIDEO_FOLDER_OVERRIDE="/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos"
EXTRA_ARGS="--video_folder ${VIDEO_FOLDER_OVERRIDE}"

WORKDIR="/users/$USER/ULTR-AI/ULTR-AI-Vid"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# Diagnostics (optional)
echo "python: $(command -v python3)"; python3 -V
python3 - <<'PY' || true
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY

# Runtime env
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800

# --- NCCL networking: kill the plugin, use sockets on single node ---
export NCCL_NET_PLUGIN=none        # <— hard-disable external plugin
export NCCL_NET=Socket             # <— force socket backend
export NCCL_IB_DISABLE=1           # <— already set; keep it
export NCCL_P2P_DISABLE=0
# Prefer loopback; if your site forbids lo, switch to nmn0 instead.
export NCCL_SOCKET_IFNAME=lo       # try 'lo' first
# export NCCL_SOCKET_IFNAME=nmn0   # fallback if 'lo' doesn't fly

# (Optional belt-and-suspenders)
unset LD_PRELOAD                   # in case the plugin is injected via LD_PRELOAD
export NCCL_PLUGIN_P2P=0

# Keep rendezvous on localhost for --standalone
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

CMD="torchrun --standalone --nproc_per_node ${GPUS_PER_NODE} --max_restarts 0 --tee 3 \
  train_ablation_distributed.py --config ${CONFIG_FILE} ${EXTRA_ARGS}"
echo "$CMD"
$CMD

echo "END TIME: $(date)"

# Optional: Run evaluation in the same job after training completes
if [[ "${RUN_EVAL_AFTER_TRAIN:-0}" == "1" ]]; then
  echo "Running evaluation after training (RUN_EVAL_AFTER_TRAIN=1)"
  # Parse values from config (tolerate missing keys)
  MODEL_TYPE=$(grep -E '^model_type:' "$CONFIG_FILE" | sed -E 's/.*model_type:[[:space:]]*"?([^"#]+)"?.*/\1/' | tr -d '[:space:]' || true)
  EXP_DIR=$(grep -E '^experiment_dir:' "$CONFIG_FILE" | sed -E 's/.*experiment_dir:[[:space:]]*"?([^"#]+)"?.*/\1/' || true)
  # Derive ablation base dir by removing trailing /foldX if present
  BASE_DIR=$(echo "$EXP_DIR" | sed -E 's|/fold[0-9]+/?$||')
  # Derive fold from EXP_DIR or split_csv
  FOLD_NUM=$(echo "$EXP_DIR" | sed -nE 's/.*fold([0-9]+).*/\1/p')
  if [[ -z "$FOLD_NUM" ]]; then
    FOLD_NUM=$(grep -E '^split_csv:' "$CONFIG_FILE" | sed -nE 's/.*Fold_([0-9]+)\.csv.*/\1/p' || true)
  fi
  # Resolve checkpoint path (prefer generic best, then directory variants)
  MODEL_CKPT="$EXP_DIR/checkpoint_best.pth"
  if [[ ! -f "$MODEL_CKPT" ]]; then
    if [[ -f "$EXP_DIR/checkpoints/checkpoint_best.pth" ]]; then
      MODEL_CKPT="$EXP_DIR/checkpoints/checkpoint_best.pth"
    else
      CAND=$(ls -1 "$EXP_DIR"/checkpoints/checkpoint_best_metric_*.pth 2>/dev/null | head -n 1 || true)
      if [[ -n "$CAND" ]]; then MODEL_CKPT="$CAND"; fi
    fi
  fi
  OUT_DIR="${BASE_DIR}/eval_results"
  echo "Eval params: model_type=${MODEL_TYPE}, fold=${FOLD_NUM}, ckpt=${MODEL_CKPT}, out=${OUT_DIR}"
  if [[ -z "$MODEL_TYPE" || -z "$BASE_DIR" || -z "$FOLD_NUM" ]]; then
    echo "[WARN] Could not infer evaluation parameters from $CONFIG_FILE; skipping evaluation."
  else
    python3 evaluate_downstream.py --model-type "$MODEL_TYPE" --config "$CONFIG_FILE" \
      --model "$MODEL_CKPT" --fold "$FOLD_NUM" --output-dir "$OUT_DIR" \
      --video_folder "$VIDEO_FOLDER_OVERRIDE" \
      || echo "[WARN] Evaluation failed."
  fi
fi