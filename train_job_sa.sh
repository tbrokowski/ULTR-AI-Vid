#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH -A a127
#SBATCH --gpus=1
#SBATCH --job-name=ultrai_sa_ft
#SBATCH --output=logs/sa_finetune_%j.out
#SBATCH --error=logs/sa_finetune_%j.err

set -euo pipefail
mkdir -p logs

# ============================================================================
# Defaults for finetune_sa.py arguments
# Override them by passing named arguments to sbatch, e.g.:
#   sbatch train_job_sa.sh --config configs/cscs/sa_finetune.yaml --skip-zero-shot
# ============================================================================

CONFIG="configs/cscs/sa_finetune.yaml"
OUTPUT_DIR="/users/lxflk/ULTR-AI-Vid/sa_finetuning_results"
CHECKPOINT_DIR="/users/lxflk/ULTR-AI-Vid/checkpoints/sa_finetune"
BENIN_CHECKPOINT="/users/lxflk/ULTR-AI-Vid/checkpoints/cscs_finalruns/fold2/checkpoint_best.pth"
BENIN_CONFIG="/users/lxflk/ULTR-AI-Vid/configs/cscs/tb_drl_mil_fold2.yaml"

SKIP_ZERO_SHOT=false
SKIP_FINETUNING=false
SKIP_BENIN_EVAL=false
SKIP_PATHOLOGY=true

usage() {
  cat <<EOF
Usage:
  sbatch train_job_sa.sh [OPTIONS]

Options:
  --config PATH               Path to SA config YAML
  --output-dir PATH           Output directory for results/plots/predictions
  --checkpoint-dir PATH       Directory for saving model checkpoints
  --benin-checkpoint PATH     Path to Benin checkpoint
  --benin-config PATH         Path to Benin config YAML

  --skip-zero-shot            Skip zero-shot evaluation
  --skip-finetuning           Skip fine-tuning
  --skip-benin-eval           Skip Benin test set evaluation
  --skip-pathology            Skip pathology plot generation

  --help                      Show this help message

Examples:
  sbatch train_job_sa.sh --config configs/cscs/sa_finetune.yaml
  sbatch train_job_sa.sh --output-dir /users/lxflk/ULTR-AI-Vid/sa_results_v2 --checkpoint-dir /capstor/store/cscs/swissai/a127/ultr-ai/sa_results_v2
  sbatch train_job_sa.sh --skip-zero-shot --skip-benin-eval --skip-pathology
EOF
}

# ============================================================================
# Parse named arguments only
# ============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --benin-checkpoint)
      BENIN_CHECKPOINT="$2"
      shift 2
      ;;
    --benin-config)
      BENIN_CONFIG="$2"
      shift 2
      ;;
    --skip-zero-shot)
      SKIP_ZERO_SHOT=true
      shift
      ;;
    --skip-finetuning)
      SKIP_FINETUNING=true
      shift
      ;;
    --skip-benin-eval)
      SKIP_BENIN_EVAL=true
      shift
      ;;
    --skip-pathology)
      SKIP_PATHOLOGY=true
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      echo "" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ============================================================================
# Build Python argument list
# ============================================================================
PY_ARGS=(
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --benin-checkpoint "${BENIN_CHECKPOINT}"
)

if [[ -n "${BENIN_CONFIG}" ]]; then
  PY_ARGS+=(--benin-config "${BENIN_CONFIG}")
fi

if [[ "${SKIP_ZERO_SHOT}" == true ]]; then
  PY_ARGS+=(--skip-zero-shot)
fi

if [[ "${SKIP_FINETUNING}" == true ]]; then
  PY_ARGS+=(--skip-finetuning)
fi

if [[ "${SKIP_BENIN_EVAL}" == true ]]; then
  PY_ARGS+=(--skip-benin-eval)
fi

if [[ "${SKIP_PATHOLOGY}" == true ]]; then
  PY_ARGS+=(--skip-pathology)
fi

echo "Running SA fine-tuning"
echo "  config:            ${CONFIG}"
echo "  output_dir:        ${OUTPUT_DIR}"
echo "  checkpoint_dir:    ${CHECKPOINT_DIR}"
echo "  benin_checkpoint:  ${BENIN_CHECKPOINT}"
echo "  benin_config:      ${BENIN_CONFIG}"
echo "  skip_zero_shot:    ${SKIP_ZERO_SHOT}"
echo "  skip_finetuning:   ${SKIP_FINETUNING}"
echo "  skip_benin_eval:   ${SKIP_BENIN_EVAL}"
echo "  skip_pathology:    ${SKIP_PATHOLOGY}"

# Prefer the node's NVIDIA driver libraries over the stale CUDA compat libs baked into the EDF image.
srun --environment=/users/lxflk/.edf/ultrai.toml \
     python3 finetune_sa.py \
     "${PY_ARGS[@]}"