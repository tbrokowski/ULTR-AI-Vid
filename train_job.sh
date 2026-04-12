#!/bin/bash
# Example:
#   ./train_job.sh --vision-weights /capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/simclr/benin/<run-name>
#
# Most common use:
#   Train the core model on Benin across all 5 fixed evaluation folds.
#   Stage defaults live in `configs/cscs/train.yaml`, dataset paths live in
#   `configs/cscs/dataset_benin.yaml`, and fold files only contribute the split
#   CSV plus any fold-specific historical overrides. Passing `--config` swaps
#   only the stage config; dataset profiles are still layered from
#   `configs/cscs/dataset_*.yaml`. SA training uses the same stage config with
#   `configs/cscs/dataset_sa.yaml` and requires an explicit `--split-csv`.

set -euo pipefail

REPO_ROOT="/users/lxflk/ULTR-AI-Vid"
SCRATCH_ROOT="/capstor/scratch/cscs/lxflk/ULTR-AI-Vid"
LOG_ROOT="${REPO_ROOT}/logs"
EDF_ENV="/users/lxflk/.edf/ultrai.toml"
RESOLVE_CONFIG="${REPO_ROOT}/ultrai/utils/config_resolver.py"
DEFAULT_BASE_CONFIG="configs/cscs/train.yaml"
BENIN_DATA_PROFILE="configs/cscs/dataset_benin.yaml"
SA_DATA_PROFILE="configs/cscs/dataset_sa.yaml"

WORKER_MODE=false
DATASET="benin"
FOLD="all"
TASK_FOLD=""
CONFIG=""
CUSTOM_CONFIG=false
RUN_NAME=""
RUN_ROOT=""
SPLIT_CSV=""
VISION_WEIGHTS=""
MODEL_WEIGHTS=""
CLIP_UNFREEZE_LAST_N_LAYERS=""
FREEZE_BACKBONE_MODE="default"

usage() {
  cat <<'EOF'
Usage:
  ./train_job.sh [OPTIONS]

Options:
  --dataset {benin,sa}              Dataset to train on. Default: benin.
  --fold INT                        Train one specific Benin fold. Default: all Benin folds. SA is always a single run.
  --config PATH                     Base training config. Default: configs/cscs/train.yaml.
  --split-csv PATH                  Explicit split CSV. Required for SA training.
  --vision-weights PATH             Optional SimCLR vision checkpoint, or a fold-root directory containing fold*/vision_encoder_best.pt.
  --model-weights PATH              Optional full model checkpoint, or a fold-root directory containing fold*/checkpoint_best.pth.
  --clip-unfreeze-last-n-layers N   Override the CLIP unfreeze setting from the resolved config.
  --freeze-backbone                 Force the CLIP backbone to stay frozen.
  --no-freeze-backbone              Force the CLIP backbone to be trainable.
  --run-name NAME                   Optional run name. Default: dynamic timestamped name.
  --run-root PATH                   Optional explicit scratch run directory.
  --help                            Show this help message.
EOF
}

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

infer_init_tag() {
  local tag="baseline"
  local lowered=""

  if [[ -n "${VISION_WEIGHTS}" ]]; then
    lowered="${VISION_WEIGHTS,,}"
    if [[ "${lowered}" == *"simclr"* || "${lowered}" == *"vision_encoder"* ]]; then
      tag="simclr"
    else
      tag="vision-warmstart"
    fi
  elif [[ -n "${MODEL_WEIGHTS}" ]]; then
    lowered="${MODEL_WEIGHTS,,}"
    if [[ "${lowered}" == *"simclr"* && "${lowered}" == *"dann"* ]]; then
      tag="simclr+dann"
    elif [[ "${lowered}" == *"simclr"* ]]; then
      tag="simclr"
    elif [[ "${lowered}" == *"dann"* ]]; then
      tag="dann"
    else
      tag="model-warmstart"
    fi
  fi

  printf '%s\n' "${tag}"
}

default_run_name() {
  local timestamp
  local scope
  local tag

  timestamp="$(date +%Y%m%d_%H%M%S)"
  tag="$(infer_init_tag)"

  if [[ "${DATASET}" == "sa" ]]; then
    scope="single-run"
  elif [[ "${FOLD}" == "all" ]]; then
    scope="all-folds"
  else
    scope="fold${FOLD}"
  fi

  printf '%s__train__%s__%s__%s\n' "${timestamp}" "${DATASET}" "${scope}" "${tag}"
}

resolve_weight_path() {
  local raw_path="$1"
  local task_fold="$2"
  local final_name="$3"

  if [[ -z "${raw_path}" ]]; then
    printf '\n'
    return
  fi

  if [[ -f "${raw_path}" ]]; then
    printf '%s\n' "${raw_path}"
    return
  fi

  if [[ -d "${raw_path}" ]]; then
    if [[ -n "${task_fold}" && -f "${raw_path}/fold${task_fold}/${final_name}" ]]; then
      printf '%s\n' "${raw_path}/fold${task_fold}/${final_name}"
      return
    fi
    if [[ -f "${raw_path}/${final_name}" ]]; then
      printf '%s\n' "${raw_path}/${final_name}"
      return
    fi
  fi

  echo "Could not resolve ${final_name} from path: ${raw_path}" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker)
      WORKER_MODE=true
      shift
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --fold)
      FOLD="$2"
      shift 2
      ;;
    --task-fold)
      TASK_FOLD="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      CUSTOM_CONFIG=true
      shift 2
      ;;
    --custom-config)
      CUSTOM_CONFIG=true
      shift
      ;;
    --split-csv)
      SPLIT_CSV="$2"
      shift 2
      ;;
    --vision-weights)
      VISION_WEIGHTS="$2"
      shift 2
      ;;
    --model-weights)
      MODEL_WEIGHTS="$2"
      shift 2
      ;;
    --clip-unfreeze-last-n-layers)
      CLIP_UNFREEZE_LAST_N_LAYERS="$2"
      shift 2
      ;;
    --freeze-backbone)
      FREEZE_BACKBONE_MODE="freeze"
      shift
      ;;
    --no-freeze-backbone)
      FREEZE_BACKBONE_MODE="unfreeze"
      shift
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_ROOT}"

if [[ -n "${VISION_WEIGHTS}" && ! -e "${VISION_WEIGHTS}" ]]; then
  echo "Vision checkpoint path not found: ${VISION_WEIGHTS}" >&2
  exit 1
fi

if [[ -n "${MODEL_WEIGHTS}" && ! -e "${MODEL_WEIGHTS}" ]]; then
  echo "Model checkpoint path not found: ${MODEL_WEIGHTS}" >&2
  exit 1
fi

if [[ "${DATASET}" != "benin" && "${DATASET}" != "sa" ]]; then
  echo "--dataset must be one of: benin, sa" >&2
  exit 1
fi

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${DEFAULT_BASE_CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Base training config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ "${WORKER_MODE}" != true ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(default_run_name)"
  fi

  if [[ -z "${RUN_ROOT}" ]]; then
    RUN_ROOT="${SCRATCH_ROOT}/runs/train/${DATASET}/${RUN_NAME}"
  fi

  RUN_ROOT="$(abs_path "${RUN_ROOT}")"

  if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to reuse existing run directory: ${RUN_ROOT}" >&2
    exit 1
  fi

  mkdir -p "${RUN_ROOT}"

  if [[ "${DATASET}" == "sa" ]]; then
    if [[ -z "${SPLIT_CSV}" ]]; then
      echo "SA training requires --split-csv so the train/val/test split is explicit." >&2
      exit 1
    fi
    if [[ "${FOLD}" != "all" ]]; then
      echo "SA training does not use predefined fold ids. Remove --fold and pass --split-csv." >&2
      exit 1
    fi
  fi

  COMMON_SUBMIT_ARGS=(
    --parsable
    --account=a127
    --partition=normal
    --time=09:59:59
    --gpus=1
  )

  FORWARDED_ARGS=(
    --worker
    --dataset "${DATASET}"
    --config "${CONFIG}"
    --run-name "${RUN_NAME}"
    --run-root "${RUN_ROOT}"
  )
  if [[ "${CUSTOM_CONFIG}" == true ]]; then
    FORWARDED_ARGS+=(--custom-config)
  fi
  if [[ -n "${SPLIT_CSV}" ]]; then
    FORWARDED_ARGS+=(--split-csv "${SPLIT_CSV}")
  fi
  if [[ -n "${VISION_WEIGHTS}" ]]; then
    FORWARDED_ARGS+=(--vision-weights "${VISION_WEIGHTS}")
  fi
  if [[ -n "${MODEL_WEIGHTS}" ]]; then
    FORWARDED_ARGS+=(--model-weights "${MODEL_WEIGHTS}")
  fi
  if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
    FORWARDED_ARGS+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
  fi
  if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
    FORWARDED_ARGS+=(--freeze-backbone)
  elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
    FORWARDED_ARGS+=(--no-freeze-backbone)
  fi

  if [[ "${DATASET}" == "benin" && "${FOLD}" == "all" ]]; then
    JOB_ID="$(
      sbatch "${COMMON_SUBMIT_ARGS[@]}" \
        --job-name="ultrai_train_benin" \
        --array=0-4 \
        --output="${LOG_ROOT}/train_benin_%A_%a.out" \
        --error="${LOG_ROOT}/train_benin_%A_%a.err" \
        "$0" \
        "${FORWARDED_ARGS[@]}"
    )"
  else
    if [[ "${FOLD}" != "all" ]]; then
      FORWARDED_ARGS+=(--task-fold "${FOLD}")
    fi

    JOB_ID="$(
      sbatch "${COMMON_SUBMIT_ARGS[@]}" \
        --job-name="ultrai_train_${DATASET}" \
        --output="${LOG_ROOT}/train_${DATASET}_%j.out" \
        --error="${LOG_ROOT}/train_${DATASET}_%j.err" \
        "$0" \
        "${FORWARDED_ARGS[@]}"
    )"
  fi

  echo "Submitted training job"
  echo "  job_id:           ${JOB_ID%%;*}"
  echo "  dataset:          ${DATASET}"
  echo "  run_name:         ${RUN_NAME}"
  echo "  run_root:         ${RUN_ROOT}"
  echo "  logs:             ${LOG_ROOT}"
  echo "  base_config:      ${CONFIG}"
  echo "  vision_weights:   ${VISION_WEIGHTS:-<none>}"
  echo "  model_weights:    ${MODEL_WEIGHTS:-<none>}"
  exit 0
fi

if [[ -z "${TASK_FOLD}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_FOLD="${SLURM_ARRAY_TASK_ID}"
fi

if [[ "${DATASET}" == "benin" && -z "${TASK_FOLD}" ]]; then
  echo "Benin training requires a resolved fold id." >&2
  exit 1
fi

OVERRIDE_CONFIGS=()
if [[ "${DATASET}" == "benin" ]]; then
  OVERRIDE_CONFIGS+=("${BENIN_DATA_PROFILE}")
else
  OVERRIDE_CONFIGS+=("${SA_DATA_PROFILE}")
fi

if [[ "${DATASET}" == "benin" ]]; then
  OVERRIDE_CONFIGS+=("configs/cscs/benin_fold${TASK_FOLD}.yaml")
fi

if [[ "${DATASET}" == "benin" ]]; then
  EXPERIMENT_DIR="${RUN_ROOT}/fold${TASK_FOLD}"
  EXPERIMENT_NAME="${RUN_NAME}_fold${TASK_FOLD}"
else
  if [[ -n "${TASK_FOLD}" ]]; then
    EXPERIMENT_DIR="${RUN_ROOT}/fold${TASK_FOLD}"
    EXPERIMENT_NAME="${RUN_NAME}_fold${TASK_FOLD}"
  else
    EXPERIMENT_DIR="${RUN_ROOT}"
    EXPERIMENT_NAME="${RUN_NAME}"
  fi
fi

mkdir -p "${EXPERIMENT_DIR}"

if [[ -n "${TASK_FOLD}" ]]; then
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}/fold${TASK_FOLD}"
else
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
fi
mkdir -p "${LOCAL_RUN_LOG_DIR}"

RESOLVED_CONFIG="${EXPERIMENT_DIR}/resolved_config.yaml"
RESOLVE_ARGS=(
  --base "${CONFIG}"
  --output "${RESOLVED_CONFIG}"
  --set "experiment_name=${EXPERIMENT_NAME}"
  --set "model_name=${EXPERIMENT_NAME}"
  --set "experiment_dir=${EXPERIMENT_DIR}"
  --set "log_dir=${LOCAL_RUN_LOG_DIR}"
  --set "checkpoint_dir=${EXPERIMENT_DIR}"
)

for override_config in "${OVERRIDE_CONFIGS[@]}"; do
  if [[ ! -f "${override_config}" ]]; then
    echo "Override config not found: ${override_config}" >&2
    exit 1
  fi
  RESOLVE_ARGS+=(--override "${override_config}")
done

if [[ -n "${SPLIT_CSV}" ]]; then
  RESOLVE_ARGS+=(--set "split_csv=${SPLIT_CSV}")
fi
if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
  RESOLVE_ARGS+=(--set "clip_unfreeze_last_n_layers=${CLIP_UNFREEZE_LAST_N_LAYERS}")
fi
if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
  RESOLVE_ARGS+=(--set "freeze_backbone=true")
elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
  RESOLVE_ARGS+=(--set "freeze_backbone=false")
fi

python3 "${RESOLVE_CONFIG}" "${RESOLVE_ARGS[@]}"

RESOLVED_VISION_WEIGHTS="$(resolve_weight_path "${VISION_WEIGHTS}" "${TASK_FOLD}" "vision_encoder_best.pt")"
RESOLVED_MODEL_WEIGHTS="$(resolve_weight_path "${MODEL_WEIGHTS}" "${TASK_FOLD}" "checkpoint_best.pth")"

PY_ARGS=(
  --dataset "${DATASET}"
  --config "${RESOLVED_CONFIG}"
)

if [[ -n "${RESOLVED_VISION_WEIGHTS}" ]]; then
  PY_ARGS+=(--vision-pretrained-weights "${RESOLVED_VISION_WEIGHTS}")
fi
if [[ -n "${RESOLVED_MODEL_WEIGHTS}" ]]; then
  PY_ARGS+=(--model-weights "${RESOLVED_MODEL_WEIGHTS}")
fi
if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
  PY_ARGS+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
fi
if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
  PY_ARGS+=(--freeze-backbone)
elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
  PY_ARGS+=(--no-freeze-backbone)
fi

echo "Running supervised training"
echo "  dataset:          ${DATASET}"
echo "  base_config:      ${CONFIG}"
echo "  resolved_config:  ${RESOLVED_CONFIG}"
echo "  experiment_dir:   ${EXPERIMENT_DIR}"
echo "  run_root:         ${RUN_ROOT}"
echo "  local_logs:       ${LOCAL_RUN_LOG_DIR}"
if [[ -n "${TASK_FOLD}" ]]; then
  echo "  fold:             ${TASK_FOLD}"
fi
echo "  vision_weights:   ${RESOLVED_VISION_WEIGHTS:-<none>}"
echo "  model_weights:    ${RESOLVED_MODEL_WEIGHTS:-<none>}"

srun --environment="${EDF_ENV}" \
  python3 -m ultrai.training.train \
  "${PY_ARGS[@]}"

BEST_CHECKPOINT="${EXPERIMENT_DIR}/checkpoint_best.pth"
FINAL_RESULTS_DIR="${EXPERIMENT_DIR}/final_results"
CONFIG_SNAPSHOT="${EXPERIMENT_DIR}/config.yaml"

echo "Training finished"
echo "  best_checkpoint:  ${BEST_CHECKPOINT}"
echo "  final_results:    ${FINAL_RESULTS_DIR}"
echo "  config_snapshot:  ${CONFIG_SNAPSHOT}"
