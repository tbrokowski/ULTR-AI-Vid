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
RESUME_FROM_CHECKPOINT=""
RESUME_FROM_LATEST=false
ALLOW_EXISTING_RUN_ROOT=false
CLIP_UNFREEZE_LAST_N_LAYERS=""
FREEZE_BACKBONE_MODE="default"
GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
JOB_PARTITION="${JOB_PARTITION:-normal}"
JOB_TIME="${JOB_TIME:-11:59:59}"
EPOCHS_OVERRIDE=""

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
  --resume-from-checkpoint PATH     Resume training from a checkpoint, or a fold-root directory containing fold*/checkpoint_latest.pth.
  --resume-from-latest              Resume each fold from checkpoint_latest.pth under the selected run root.
  --allow-existing-run-root         Allow a single clean fold job to reuse an existing run root.
  --clip-unfreeze-last-n-layers N   Override the CLIP unfreeze setting from the resolved config.
  --freeze-backbone                 Force the CLIP backbone to stay frozen.
  --no-freeze-backbone              Force the CLIP backbone to be trainable.
  --gpus-per-job N                  GPUs requested per fold job. Default: 4.
  --partition NAME                  Slurm partition. Default: normal.
  --time LIMIT                      Slurm wall time. Default: 11:59:59 (normal partition max is 12:00:00).
  --epochs N                        Override num_epochs in the resolved config.
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
      if [[ "${WORKER_MODE}" != true ]]; then
        CUSTOM_CONFIG=true
      fi
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
    --resume-from-checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --resume-from-latest)
      RESUME_FROM_LATEST=true
      shift
      ;;
    --allow-existing-run-root)
      ALLOW_EXISTING_RUN_ROOT=true
      shift
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
    --gpus-per-job)
      GPUS_PER_JOB="$2"
      shift 2
      ;;
    --partition)
      JOB_PARTITION="$2"
      shift 2
      ;;
    --time)
      JOB_TIME="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS_OVERRIDE="$2"
      shift 2
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

if [[ -n "${RESUME_FROM_CHECKPOINT}" && ! -e "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint path not found: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${DATASET}" != "benin" && "${DATASET}" != "sa" ]]; then
  echo "--dataset must be one of: benin, sa" >&2
  exit 1
fi

if ! [[ "${GPUS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--gpus-per-job must be a positive integer, got: ${GPUS_PER_JOB}" >&2
  exit 1
fi

if [[ -n "${EPOCHS_OVERRIDE}" ]] && ! [[ "${EPOCHS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--epochs must be a positive integer, got: ${EPOCHS_OVERRIDE}" >&2
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

  if [[ "${ALLOW_EXISTING_RUN_ROOT}" == true && "${DATASET}" == "benin" && "${FOLD}" == "all" ]]; then
    echo "--allow-existing-run-root is only for a single clean fold replacement. Pass --fold INT." >&2
    exit 1
  fi

  if [[ -e "${RUN_ROOT}" && "${RESUME_FROM_LATEST}" != true && "${ALLOW_EXISTING_RUN_ROOT}" != true ]]; then
    echo "Refusing to reuse existing run directory: ${RUN_ROOT}" >&2
    echo "Use --resume-from-latest to continue, or --allow-existing-run-root with --fold INT for a clean fold replacement." >&2
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
    --partition="${JOB_PARTITION}"
    --time="${JOB_TIME}"
    --nodes=1
    --ntasks=1
    --gres="gpu:${GPUS_PER_JOB}"
    --cpus-per-task="$((GPUS_PER_JOB * 8))"
    --environment="${EDF_ENV}"
  )

  FORWARDED_ARGS=(
    --worker
    --dataset "${DATASET}"
    --config "${CONFIG}"
    --run-name "${RUN_NAME}"
    --run-root "${RUN_ROOT}"
    --gpus-per-job "${GPUS_PER_JOB}"
    --partition "${JOB_PARTITION}"
    --time "${JOB_TIME}"
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
  if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    FORWARDED_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi
  if [[ "${RESUME_FROM_LATEST}" == true ]]; then
    FORWARDED_ARGS+=(--resume-from-latest)
  fi
  if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
    FORWARDED_ARGS+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
  fi
  if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
    FORWARDED_ARGS+=(--freeze-backbone)
  elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
    FORWARDED_ARGS+=(--no-freeze-backbone)
  fi
  if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    FORWARDED_ARGS+=(--epochs "${EPOCHS_OVERRIDE}")
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
  echo "  gpus_per_job:     ${GPUS_PER_JOB}"
  echo "  partition:        ${JOB_PARTITION}"
  echo "  time:             ${JOB_TIME}"
  echo "  vision_weights:   ${VISION_WEIGHTS:-<none>}"
  echo "  model_weights:    ${MODEL_WEIGHTS:-<none>}"
  echo "  resume:           ${RESUME_FROM_CHECKPOINT:-$([[ "${RESUME_FROM_LATEST}" == true ]] && echo '<latest>' || echo '<none>')}"
  echo "  allow_existing:   ${ALLOW_EXISTING_RUN_ROOT}"
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
if [[ "${CUSTOM_CONFIG}" != true && "${GPUS_PER_JOB}" == "1" ]]; then
  # Match the original 4-GPU HMV-MIL effective batch:
  # batch_size 2 * accumulation 35 * world_size 4 = 280.
  RESOLVE_ARGS+=(--set "accumulation_steps=140")
fi
if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
  RESOLVE_ARGS+=(--set "num_epochs=${EPOCHS_OVERRIDE}")
fi

python3 "${RESOLVE_CONFIG}" "${RESOLVE_ARGS[@]}"

RESOLVED_VISION_WEIGHTS="$(resolve_weight_path "${VISION_WEIGHTS}" "${TASK_FOLD}" "vision_encoder_best.pt")"
RESOLVED_MODEL_WEIGHTS="$(resolve_weight_path "${MODEL_WEIGHTS}" "${TASK_FOLD}" "checkpoint_best.pth")"
RESOLVED_RESUME_CHECKPOINT=""
if [[ "${RESUME_FROM_LATEST}" == true ]]; then
  RESOLVED_RESUME_CHECKPOINT="${EXPERIMENT_DIR}/checkpoint_latest.pth"
elif [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESOLVED_RESUME_CHECKPOINT="$(resolve_weight_path "${RESUME_FROM_CHECKPOINT}" "${TASK_FOLD}" "checkpoint_latest.pth")"
fi

if [[ -n "${RESOLVED_RESUME_CHECKPOINT}" && ! -f "${RESOLVED_RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESOLVED_RESUME_CHECKPOINT}" >&2
  exit 1
fi

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
if [[ -n "${RESOLVED_RESUME_CHECKPOINT}" ]]; then
  PY_ARGS+=(--resume-from-checkpoint "${RESOLVED_RESUME_CHECKPOINT}")
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
echo "  gpus_per_job:     ${GPUS_PER_JOB}"
echo "  vision_weights:   ${RESOLVED_VISION_WEIGHTS:-<none>}"
echo "  model_weights:    ${RESOLVED_MODEL_WEIGHTS:-<none>}"
echo "  resume:           ${RESOLVED_RESUME_CHECKPOINT:-<none>}"

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

if (( GPUS_PER_JOB > 1 )); then
  TORCHRUN_LOG_DIR="${LOCAL_RUN_LOG_DIR}/torchrun"
  mkdir -p "${TORCHRUN_LOG_DIR}"
  echo "  torchrun_logs:    ${TORCHRUN_LOG_DIR}"

  TRAIN_CMD=(
    torchrun
    --standalone
    --nproc_per_node "${GPUS_PER_JOB}"
    --max_restarts 0
    --log-dir "${TORCHRUN_LOG_DIR}"
    --redirects 3
    --tee 3
    -m ultrai.training.train
  )
else
  TRAIN_CMD=(python3 -u -m ultrai.training.train)
fi

"${TRAIN_CMD[@]}" "${PY_ARGS[@]}"

BEST_CHECKPOINT="${EXPERIMENT_DIR}/checkpoint_best.pth"
FINAL_RESULTS_DIR="${EXPERIMENT_DIR}/final_results"
CONFIG_SNAPSHOT="${EXPERIMENT_DIR}/config.yaml"

echo "Training finished"
echo "  best_checkpoint:  ${BEST_CHECKPOINT}"
echo "  final_results:    ${FINAL_RESULTS_DIR}"
echo "  config_snapshot:  ${CONFIG_SNAPSHOT}"
