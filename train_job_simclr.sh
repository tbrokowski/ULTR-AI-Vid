#!/bin/bash
# Example:
#   ./train_job_simclr.sh
#
# Most common use:
#   Submit SimCLR CLIP pretraining on Benin across folds 0-4. Stage defaults
#   live in `configs/cscs/simclr.yaml`, dataset paths come from
#   `configs/cscs/dataset_benin.yaml`, and Benin fold files contribute the split
#   CSV. Passing `--config` swaps only the SimCLR stage config; dataset
#   profiles are still layered from `configs/cscs/dataset_*.yaml`. To use SA
#   instead, switch to `--dataset sa` and pass an explicit `--split-csv`.

set -euo pipefail

REPO_ROOT="/users/lxflk/ULTR-AI-Vid"
SCRATCH_ROOT="/capstor/scratch/cscs/lxflk/ULTR-AI-Vid"
LOG_ROOT="${REPO_ROOT}/logs"
EDF_ENV="/users/lxflk/.edf/ultrai.toml"
RESOLVE_CONFIG="${REPO_ROOT}/ultrai/utils/config_resolver.py"
BENIN_DATA_PROFILE="configs/cscs/dataset_benin.yaml"
SA_DATA_PROFILE="configs/cscs/dataset_sa.yaml"

WORKER_MODE=false
DATASET="benin"
FOLD="all"
TASK_FOLD=""
CONFIG="configs/cscs/simclr.yaml"
CUSTOM_CONFIG=false
RUN_NAME=""
RUN_ROOT=""
SPLIT_CSV=""

usage() {
  cat <<'EOF'
Usage:
  ./train_job_simclr.sh [OPTIONS]

Options:
  --dataset {benin,sa}    Dataset to use for SimCLR. Default: benin.
  --fold INT              Train one specific Benin fold. Default: all Benin folds. SA is always a single run.
  --config PATH           SimCLR config YAML. Default: configs/cscs/simclr.yaml.
  --split-csv PATH        Explicit split CSV. Required for SA SimCLR.
  --run-name NAME         Optional run name. Default: dynamic timestamped name.
  --run-root PATH         Optional explicit scratch run directory.
  --help                  Show this help message.
EOF
}

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

yaml_get() {
  python3 - "$1" "$2" <<'PY'
import sys
import yaml

path, key = sys.argv[1], sys.argv[2]
with open(path, "r") as handle:
    data = yaml.safe_load(handle) or {}

value = data
for part in key.split("."):
    if not isinstance(value, dict):
        value = None
        break
    value = value.get(part)

if value is None:
    print("")
else:
    print(value)
PY
}

default_run_name() {
  local timestamp
  local scope

  timestamp="$(date +%Y%m%d_%H%M%S)"
  if [[ "${DATASET}" == "sa" ]]; then
    scope="single-run"
  elif [[ "${FOLD}" == "all" ]]; then
    scope="all-folds"
  else
    scope="fold${FOLD}"
  fi

  printf '%s__simclr__%s__%s\n' "${timestamp}" "${DATASET}" "${scope}"
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

if [[ "${DATASET}" != "benin" && "${DATASET}" != "sa" ]]; then
  echo "--dataset must be one of: benin, sa" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "SimCLR config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ "${WORKER_MODE}" != true ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(default_run_name)"
  fi

  if [[ -z "${RUN_ROOT}" ]]; then
    RUN_ROOT="${SCRATCH_ROOT}/runs/simclr/${DATASET}/${RUN_NAME}"
  fi

  RUN_ROOT="$(abs_path "${RUN_ROOT}")"

  if is_nonempty_dir "${RUN_ROOT}"; then
    echo "Refusing to reuse existing non-empty run directory: ${RUN_ROOT}" >&2
    exit 1
  fi

  mkdir -p "${RUN_ROOT}"

  if [[ "${DATASET}" == "sa" ]]; then
    if [[ -z "${SPLIT_CSV}" ]]; then
      echo "SA SimCLR requires --split-csv so the pretraining split is explicit." >&2
      exit 1
    fi
    if [[ "${FOLD}" != "all" ]]; then
      echo "SA SimCLR does not use predefined fold ids. Remove --fold and pass --split-csv." >&2
      exit 1
    fi
  fi

  COMMON_SUBMIT_ARGS=(
    --parsable
    --account=a127
    --partition=normal
    --time=06:00:00
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

  if [[ "${DATASET}" == "benin" && "${FOLD}" == "all" ]]; then
    JOB_ID="$(
      sbatch "${COMMON_SUBMIT_ARGS[@]}" \
        --job-name="ultrai_simclr_benin" \
        --array=0-4 \
        --output="${LOG_ROOT}/simclr_benin_%A_%a.out" \
        --error="${LOG_ROOT}/simclr_benin_%A_%a.err" \
        "$0" \
        "${FORWARDED_ARGS[@]}"
    )"
  else
    if [[ "${FOLD}" != "all" ]]; then
      FORWARDED_ARGS+=(--task-fold "${FOLD}")
    fi

    JOB_ID="$(
      sbatch "${COMMON_SUBMIT_ARGS[@]}" \
        --job-name="ultrai_simclr_${DATASET}" \
        --output="${LOG_ROOT}/simclr_${DATASET}_%j.out" \
        --error="${LOG_ROOT}/simclr_${DATASET}_%j.err" \
        "$0" \
        "${FORWARDED_ARGS[@]}"
    )"
  fi

  echo "Submitted SimCLR job"
  echo "  job_id:           ${JOB_ID%%;*}"
  echo "  dataset:          ${DATASET}"
  echo "  run_name:         ${RUN_NAME}"
  echo "  run_root:         ${RUN_ROOT}"
  echo "  logs:             ${LOG_ROOT}"
  exit 0
fi

if [[ -z "${TASK_FOLD}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_FOLD="${SLURM_ARRAY_TASK_ID}"
fi

RESOLVED_SPLIT_CSV="${SPLIT_CSV}"
if [[ "${DATASET}" == "benin" ]]; then
  DATA_PROFILE="${BENIN_DATA_PROFILE}"
else
  DATA_PROFILE="${SA_DATA_PROFILE}"
fi

if [[ "${DATASET}" == "benin" ]]; then
  if [[ -z "${TASK_FOLD}" ]]; then
    echo "Benin SimCLR requires a resolved fold id." >&2
    exit 1
  fi
  BENIN_FOLD_CONFIG="configs/cscs/benin_fold${TASK_FOLD}.yaml"
  if [[ -z "${RESOLVED_SPLIT_CSV}" ]]; then
    RESOLVED_SPLIT_CSV="$(yaml_get "${BENIN_FOLD_CONFIG}" "split_csv")"
  fi
fi

if [[ -z "${RESOLVED_SPLIT_CSV}" ]]; then
  echo "Could not resolve a split CSV for SimCLR." >&2
  exit 1
fi

if [[ -n "${TASK_FOLD}" ]]; then
  OUTPUT_DIR="${RUN_ROOT}/fold${TASK_FOLD}"
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}/fold${TASK_FOLD}"
else
  OUTPUT_DIR="${RUN_ROOT}"
  LOCAL_RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
fi

mkdir -p "${OUTPUT_DIR}" "${LOCAL_RUN_LOG_DIR}"

RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_simclr_config.yaml"
RESOLVE_ARGS=(
  --base "${CONFIG}"
  --output "${RESOLVED_CONFIG}"
  --set "experiment_name=${RUN_NAME}"
  --set "output_dir=${OUTPUT_DIR}"
  --set "split_csv=${RESOLVED_SPLIT_CSV}"
)

if [[ -n "${DATA_PROFILE}" ]]; then
  RESOLVE_ARGS+=(--override "${DATA_PROFILE}")
fi

python3 "${RESOLVE_CONFIG}" "${RESOLVE_ARGS[@]}"

PY_ARGS=(
  --config "${RESOLVED_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --log-dir "${LOCAL_RUN_LOG_DIR}"
)

echo "Running SimCLR pretraining"
echo "  dataset:          ${DATASET}"
echo "  base_config:      ${CONFIG}"
echo "  resolved_config:  ${RESOLVED_CONFIG}"
echo "  output_dir:       ${OUTPUT_DIR}"
echo "  split_csv:        ${RESOLVED_SPLIT_CSV}"
echo "  logs:             ${LOCAL_RUN_LOG_DIR}"
if [[ -n "${TASK_FOLD}" ]]; then
  echo "  fold:             ${TASK_FOLD}"
fi

srun --environment="${EDF_ENV}" \
  python3 -m ultrai.training.simclr \
  "${PY_ARGS[@]}"

BEST_VISION_CHECKPOINT="${OUTPUT_DIR}/vision_encoder_best.pt"
BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoint_best.pt"
METRICS_FILE="${OUTPUT_DIR}/metrics.json"

echo "SimCLR finished"
echo "  best_vision_checkpoint: ${BEST_VISION_CHECKPOINT}"
echo "  best_checkpoint:        ${BEST_CHECKPOINT}"
echo "  metrics:                ${METRICS_FILE}"
