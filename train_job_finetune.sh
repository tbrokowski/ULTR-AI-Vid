#!/bin/bash
# Example:
#   ./train_job_finetune.sh --source-checkpoint /capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/<run-name>/fold3/checkpoint_best.pth --source-config configs/cscs/benin_fold3.yaml
#
# Most common use:
#   Fine-tune a Benin checkpoint on SA. Stage defaults live in
#   `configs/cscs/finetune.yaml` and dataset paths come from
#   `configs/cscs/dataset_sa.yaml`. If you target Benin instead, this script
#   keeps the finetuning defaults and only swaps in `configs/cscs/dataset_benin.yaml`
#   plus the requested fold split or explicit `--split-csv`. Passing `--config`
#   swaps only the finetuning stage config; dataset profiles are still layered
#   from `configs/cscs/dataset_*.yaml`.

set -euo pipefail

REPO_ROOT="/users/lxflk/ULTR-AI-Vid"
SCRATCH_ROOT="/capstor/scratch/cscs/lxflk/ULTR-AI-Vid"
LOG_ROOT="${REPO_ROOT}/logs"
EDF_ENV="/users/lxflk/.edf/ultrai.toml"
RESOLVE_CONFIG="${REPO_ROOT}/ultrai/utils/config_resolver.py"
DEFAULT_FINETUNE_CONFIG="configs/cscs/finetune.yaml"
BENIN_DATA_PROFILE="configs/cscs/dataset_benin.yaml"
SA_DATA_PROFILE="configs/cscs/dataset_sa.yaml"

WORKER_MODE=false
SOURCE_DATASET="benin"
TARGET_DATASET="sa"
SOURCE_CHECKPOINT=""
SOURCE_CONFIG=""
CONFIG=""
CUSTOM_CONFIG=false
RUN_NAME=""
RUN_ROOT=""
FOLD=""
SPLIT_CSV=""
CLIP_UNFREEZE_LAST_N_LAYERS=""
FREEZE_BACKBONE_MODE="default"

usage() {
  cat <<'EOF'
Usage:
  ./train_job_finetune.sh [OPTIONS]

Options:
  --source-dataset {benin,sa}       Source checkpoint dataset. Default: benin.
  --target-dataset {sa,benin}       Target dataset to fine-tune on. Default: sa.
  --source-checkpoint PATH          Required source checkpoint from the previous stage.
  --source-config PATH              Source-domain config for optional source re-evaluation after fine-tuning.
  --config PATH                     Base finetuning config. Default: configs/cscs/finetune.yaml.
  --fold INT                        Target Benin fold when target dataset is Benin.
  --split-csv PATH                  Explicit split CSV for target Benin runs. SA fine-tuning builds its own split internally.
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

infer_source_tag() {
  local lowered="${SOURCE_CHECKPOINT,,}"
  if [[ "${lowered}" == *"simclr"* || "${lowered}" == *"vision_encoder"* ]]; then
    printf 'simclr\n'
  else
    printf 'baseline\n'
  fi
}

default_run_name() {
  local timestamp
  local tag
  local scope

  timestamp="$(date +%Y%m%d_%H%M%S)"
  tag="$(infer_source_tag)"

  if [[ -n "${FOLD}" ]]; then
    scope="fold${FOLD}"
  else
    scope="full"
  fi

  printf '%s__finetune__%s_to_%s__%s__%s\n' "${timestamp}" "${SOURCE_DATASET}" "${TARGET_DATASET}" "${scope}" "${tag}"
}

is_nonempty_dir() {
  local path="$1"
  [[ -d "${path}" ]] && find "${path}" -mindepth 1 -print -quit | grep -q .
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker)
      WORKER_MODE=true
      shift
      ;;
    --source-dataset)
      SOURCE_DATASET="$2"
      shift 2
      ;;
    --target-dataset)
      TARGET_DATASET="$2"
      shift 2
      ;;
    --source-checkpoint)
      SOURCE_CHECKPOINT="$2"
      shift 2
      ;;
    --source-config)
      SOURCE_CONFIG="$2"
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
    --fold)
      FOLD="$2"
      shift 2
      ;;
    --split-csv)
      SPLIT_CSV="$2"
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

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
  echo "--source-checkpoint is required." >&2
  exit 1
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi

if [[ -n "${SOURCE_CONFIG}" && ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Source config not found: ${SOURCE_CONFIG}" >&2
  exit 1
fi

if [[ "${SOURCE_DATASET}" != "benin" && "${SOURCE_DATASET}" != "sa" ]]; then
  echo "--source-dataset must be one of: benin, sa" >&2
  exit 1
fi

if [[ "${TARGET_DATASET}" != "sa" && "${TARGET_DATASET}" != "benin" ]]; then
  echo "--target-dataset must be one of: sa, benin" >&2
  exit 1
fi

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${DEFAULT_FINETUNE_CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Base finetune config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ "${WORKER_MODE}" != true ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(default_run_name)"
  fi

  if [[ -z "${RUN_ROOT}" ]]; then
    RUN_ROOT="${SCRATCH_ROOT}/runs/finetune/${SOURCE_DATASET}_to_${TARGET_DATASET}/${RUN_NAME}"
  fi

  RUN_ROOT="$(abs_path "${RUN_ROOT}")"

  if is_nonempty_dir "${RUN_ROOT}"; then
    echo "Refusing to reuse existing non-empty run directory: ${RUN_ROOT}" >&2
    exit 1
  fi

  mkdir -p "${RUN_ROOT}"

  if [[ "${TARGET_DATASET}" == "sa" && -z "${SOURCE_CONFIG}" ]]; then
    echo "No --source-config provided; source-domain re-evaluation will be skipped after fine-tuning." >&2
  fi

  if [[ "${TARGET_DATASET}" == "sa" && -n "${FOLD}" ]]; then
    echo "SA fine-tuning does not use predefined fold ids. Remove --fold." >&2
    exit 1
  fi

  if [[ "${TARGET_DATASET}" == "benin" && "${CUSTOM_CONFIG}" != true && -z "${FOLD}" && -z "${SPLIT_CSV}" ]]; then
    echo "Benin finetuning requires --fold or --split-csv unless you provide a fully custom config." >&2
    exit 1
  fi

  COMMON_SUBMIT_ARGS=(
    --parsable
    --account=a127
    --partition=normal
    --time=05:00:00
    --gpus=1
  )

  FORWARDED_ARGS=(
    --worker
    --source-dataset "${SOURCE_DATASET}"
    --target-dataset "${TARGET_DATASET}"
    --source-checkpoint "${SOURCE_CHECKPOINT}"
    --config "${CONFIG}"
    --run-name "${RUN_NAME}"
    --run-root "${RUN_ROOT}"
  )
  if [[ "${CUSTOM_CONFIG}" == true ]]; then
    FORWARDED_ARGS+=(--custom-config)
  fi

  if [[ -n "${SOURCE_CONFIG}" ]]; then
    FORWARDED_ARGS+=(--source-config "${SOURCE_CONFIG}")
  fi
  if [[ -n "${FOLD}" ]]; then
    FORWARDED_ARGS+=(--fold "${FOLD}")
  fi
  if [[ -n "${SPLIT_CSV}" ]]; then
    FORWARDED_ARGS+=(--split-csv "${SPLIT_CSV}")
  fi
  if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
    FORWARDED_ARGS+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
  fi
  if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
    FORWARDED_ARGS+=(--freeze-backbone)
  elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
    FORWARDED_ARGS+=(--no-freeze-backbone)
  fi

  JOB_ID="$(
    sbatch "${COMMON_SUBMIT_ARGS[@]}" \
      --job-name="ultrai_finetune_${SOURCE_DATASET}_to_${TARGET_DATASET}" \
      --output="${LOG_ROOT}/finetune_${SOURCE_DATASET}_to_${TARGET_DATASET}_%j.out" \
      --error="${LOG_ROOT}/finetune_${SOURCE_DATASET}_to_${TARGET_DATASET}_%j.err" \
      "$0" \
      "${FORWARDED_ARGS[@]}"
  )"

  echo "Submitted finetune job"
  echo "  job_id:             ${JOB_ID%%;*}"
  echo "  source_checkpoint:  ${SOURCE_CHECKPOINT}"
  echo "  source_config:      ${SOURCE_CONFIG:-<none>}"
  echo "  target_dataset:     ${TARGET_DATASET}"
  echo "  run_name:           ${RUN_NAME}"
  echo "  run_root:           ${RUN_ROOT}"
  echo "  logs:               ${LOG_ROOT}"
  echo "  base_config:        ${CONFIG}"
  exit 0
fi

if [[ "${TARGET_DATASET}" == "sa" ]]; then
  if [[ -n "${SPLIT_CSV}" ]]; then
    echo "Ignoring --split-csv for SA fine-tuning: ultrai.training.finetune creates the SA split internally." >&2
  fi

  OUTPUT_DIR="${RUN_ROOT}/results"
  CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
  mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${LOCAL_RUN_LOG_DIR}"

  RESOLVED_CONFIG="${RUN_ROOT}/resolved_finetune_config.yaml"
  RESOLVE_ARGS=(
    --base "${CONFIG}"
    --output "${RESOLVED_CONFIG}"
    --set "experiment_name=${RUN_NAME}"
    --set "model_name=${RUN_NAME}"
    --set "experiment_dir=${RUN_ROOT}"
    --set "checkpoint_dir=${CHECKPOINT_DIR}"
    --set "log_dir=${LOCAL_RUN_LOG_DIR}"
  )
  RESOLVE_ARGS+=(--override "${SA_DATA_PROFILE}")
  python3 "${RESOLVE_CONFIG}" "${RESOLVE_ARGS[@]}"

  PY_ARGS=(
    --config "${RESOLVED_CONFIG}"
    --output-dir "${OUTPUT_DIR}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --log-dir "${LOCAL_RUN_LOG_DIR}"
    --target-dataset sa
    --source-dataset "${SOURCE_DATASET}"
    --source-checkpoint "${SOURCE_CHECKPOINT}"
    --skip-pathology
  )

  if [[ -n "${SOURCE_CONFIG}" ]]; then
    PY_ARGS+=(--source-config "${SOURCE_CONFIG}")
  else
    PY_ARGS+=(--skip-source-eval)
  fi

  if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
    PY_ARGS+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
  fi
  if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
    PY_ARGS+=(--freeze-backbone)
  elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
    PY_ARGS+=(--no-freeze-backbone)
  fi

  echo "Running finetuning"
  echo "  source_dataset:     ${SOURCE_DATASET}"
  echo "  target_dataset:     ${TARGET_DATASET}"
  echo "  base_config:        ${CONFIG}"
  echo "  resolved_config:    ${RESOLVED_CONFIG}"
  echo "  source_checkpoint:  ${SOURCE_CHECKPOINT}"
  echo "  source_config:      ${SOURCE_CONFIG:-<none>}"
  echo "  run_root:           ${RUN_ROOT}"
  echo "  local_logs:         ${LOCAL_RUN_LOG_DIR}"

  srun --environment="${EDF_ENV}" \
    python3 -m ultrai.training.finetune \
    "${PY_ARGS[@]}"

  echo "Finetuning finished"
  echo "  checkpoints:        ${CHECKPOINT_DIR}"
  echo "  results:            ${OUTPUT_DIR}"
  exit 0
fi

if [[ -z "${FOLD}" && -z "${SPLIT_CSV}" && "${CUSTOM_CONFIG}" != true ]]; then
  echo "Benin finetuning needs --fold or --split-csv." >&2
  exit 1
fi

OVERRIDE_CONFIGS=()
OVERRIDE_CONFIGS+=("${BENIN_DATA_PROFILE}")
if [[ -n "${FOLD}" ]]; then
  OVERRIDE_CONFIGS+=("configs/cscs/benin_fold${FOLD}.yaml")
fi

if [[ -n "${FOLD}" ]]; then
  EXPERIMENT_DIR="${RUN_ROOT}/fold${FOLD}"
  EXPERIMENT_NAME="${RUN_NAME}_fold${FOLD}"
else
  EXPERIMENT_DIR="${RUN_ROOT}"
  EXPERIMENT_NAME="${RUN_NAME}"
fi

mkdir -p "${EXPERIMENT_DIR}"

if [[ -n "${FOLD}" ]]; then
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}/fold${FOLD}"
else
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
fi
mkdir -p "${LOCAL_RUN_LOG_DIR}"

RESOLVED_CONFIG="${EXPERIMENT_DIR}/resolved_finetune_config.yaml"
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

python3 "${RESOLVE_CONFIG}" "${RESOLVE_ARGS[@]}"

PY_ARGS=(
  --config "${RESOLVED_CONFIG}"
  --target-dataset benin
  --source-dataset "${SOURCE_DATASET}"
  --source-checkpoint "${SOURCE_CHECKPOINT}"
)

if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
  PY_ARGS+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
fi
if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
  PY_ARGS+=(--freeze-backbone)
elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
  PY_ARGS+=(--no-freeze-backbone)
fi

echo "Running finetuning"
echo "  source_dataset:     ${SOURCE_DATASET}"
echo "  target_dataset:     ${TARGET_DATASET}"
echo "  base_config:        ${CONFIG}"
echo "  resolved_config:    ${RESOLVED_CONFIG}"
echo "  source_checkpoint:  ${SOURCE_CHECKPOINT}"
echo "  experiment_dir:     ${EXPERIMENT_DIR}"
echo "  local_logs:         ${LOCAL_RUN_LOG_DIR}"

srun --environment="${EDF_ENV}" \
  python3 -m ultrai.training.finetune \
  "${PY_ARGS[@]}"

echo "Finetuning finished"
echo "  best_checkpoint:    ${EXPERIMENT_DIR}/checkpoint_best.pth"
echo "  final_results:      ${EXPERIMENT_DIR}/final_results"
