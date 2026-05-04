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
#   Use `--dataset benin_sa` to train fold-specific combined SimCLR backbones
#   on each Benin train fold plus the explicit SA split.

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
PRESET=""
RUN_NAME=""
RUN_ROOT=""
SPLIT_CSV=""
TIME_LIMIT=""
EXTRA_CONFIG_SETS=()

usage() {
  cat <<'EOF'
Usage:
  ./train_job_simclr.sh [OPTIONS]

Options:
  --dataset {benin,sa,benin_sa}
                          Dataset to use for SimCLR. Default: benin.
  --fold INT              Train one specific Benin/combined fold. Default: all Benin/combined folds. SA is always a single run.
  --config PATH           SimCLR config YAML. Default: configs/cscs/simclr.yaml.
  --preset NAME           Named config shortcut. Available: legacy, sa_temporal,
                          benin_sa_temporal, benin_sa_strong_patientaware,
                          benin_source_temporal_mild,
                          benin_source_patientaware, benin_source_strong,
                          benin_source_deep_lowlr, benin_source_trainval,
                          benin_source_allunlabeled, benin_source_alldepth.
  --set KEY=VALUE         Extra resolved config override. May be repeated.
  --split-csv PATH        Explicit SA split CSV. Required for SA and benin_sa SimCLR.
  --time-limit HH:MM:SS   Optional Slurm time limit override.
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
  local variant

  timestamp="$(date +%Y%m%d_%H%M%S)"
  if [[ "${DATASET}" == "sa" ]]; then
    scope="single-run"
  elif [[ "${FOLD}" == "all" ]]; then
    scope="all-folds"
  else
    scope="fold${FOLD}"
  fi

  if [[ -n "${PRESET}" ]]; then
    variant="__${PRESET}"
  else
    variant=""
  fi

  printf '%s__simclr__%s__%s%s\n' "${timestamp}" "${DATASET}" "${scope}" "${variant}"
}

config_for_preset() {
  case "$1" in
    legacy|benin_legacy)
      printf '%s\n' "configs/cscs/simclr.yaml"
      ;;
    sa_temporal)
      printf '%s\n' "configs/cscs/simclr_sa_temporal.yaml"
      ;;
    benin_sa_temporal)
      printf '%s\n' "configs/cscs/simclr_benin_sa_temporal.yaml"
      ;;
    benin_sa_strong_patientaware)
      printf '%s\n' "configs/cscs/simclr_benin_sa_strong_patientaware.yaml"
      ;;
    benin_source_temporal_mild)
      printf '%s\n' "configs/cscs/simclr_benin_source_temporal_mild.yaml"
      ;;
    benin_source_patientaware)
      printf '%s\n' "configs/cscs/simclr_benin_source_patientaware.yaml"
      ;;
    benin_source_strong)
      printf '%s\n' "configs/cscs/simclr_benin_source_strong_patientaware.yaml"
      ;;
    benin_source_deep_lowlr)
      printf '%s\n' "configs/cscs/simclr_benin_source_deep_lowlr.yaml"
      ;;
    benin_source_trainval)
      printf '%s\n' "configs/cscs/simclr_benin_source_trainval_patientaware.yaml"
      ;;
    benin_source_allunlabeled)
      printf '%s\n' "configs/cscs/simclr_benin_source_allunlabeled_patientaware.yaml"
      ;;
    benin_source_alldepth)
      printf '%s\n' "configs/cscs/simclr_benin_source_alldepth_patientaware.yaml"
      ;;
    *)
      echo "Unknown SimCLR preset: $1" >&2
      return 1
      ;;
  esac
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
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --custom-config)
      CUSTOM_CONFIG=true
      shift
      ;;
    --set)
      EXTRA_CONFIG_SETS+=("$2")
      shift 2
      ;;
    --split-csv)
      SPLIT_CSV="$2"
      shift 2
      ;;
    --time-limit)
      TIME_LIMIT="$2"
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

if [[ -n "${PRESET}" ]]; then
  if [[ "${CUSTOM_CONFIG}" == true ]]; then
    echo "Use either --preset or --config, not both." >&2
    exit 1
  fi
  CONFIG="$(config_for_preset "${PRESET}")"
fi

if [[ "${DATASET}" != "benin" && "${DATASET}" != "sa" && "${DATASET}" != "benin_sa" ]]; then
  echo "--dataset must be one of: benin, sa, benin_sa" >&2
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

  if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to reuse existing run directory: ${RUN_ROOT}" >&2
    exit 1
  fi

  mkdir -p "${RUN_ROOT}"

  if [[ "${DATASET}" == "sa" || "${DATASET}" == "benin_sa" ]]; then
    if [[ -z "${SPLIT_CSV}" ]]; then
      echo "${DATASET} SimCLR requires --split-csv so the SA pretraining split is explicit." >&2
      exit 1
    fi
  fi

  if [[ "${DATASET}" == "sa" ]]; then
    if [[ "${FOLD}" != "all" ]]; then
      echo "SA SimCLR does not use predefined fold ids. Remove --fold and pass --split-csv." >&2
      exit 1
    fi
  fi

  if [[ -z "${TIME_LIMIT}" ]]; then
    TIME_LIMIT="06:00:00"
    if [[ "${DATASET}" == "benin_sa" ]]; then
      TIME_LIMIT="12:00:00"
    fi
  fi

  COMMON_SUBMIT_ARGS=(
    --parsable
    --account=a127
    --partition=normal
    --time="${TIME_LIMIT}"
    --gpus=1
  )

  FORWARDED_ARGS=(
    --worker
    --dataset "${DATASET}"
    --run-name "${RUN_NAME}"
    --run-root "${RUN_ROOT}"
  )
  if [[ -n "${PRESET}" ]]; then
    FORWARDED_ARGS+=(--preset "${PRESET}")
  else
    FORWARDED_ARGS+=(--config "${CONFIG}")
  fi
  if [[ "${CUSTOM_CONFIG}" == true ]]; then
    FORWARDED_ARGS+=(--custom-config)
  fi
  for config_set in "${EXTRA_CONFIG_SETS[@]}"; do
    FORWARDED_ARGS+=(--set "${config_set}")
  done

  if [[ -n "${SPLIT_CSV}" ]]; then
    FORWARDED_ARGS+=(--split-csv "${SPLIT_CSV}")
  fi

  if [[ ( "${DATASET}" == "benin" || "${DATASET}" == "benin_sa" ) && "${FOLD}" == "all" ]]; then
    JOB_ID="$(
      sbatch "${COMMON_SUBMIT_ARGS[@]}" \
        --job-name="ultrai_simclr_${DATASET}" \
        --array=0-4 \
        --output="${LOG_ROOT}/simclr_${DATASET}_%A_%a.out" \
        --error="${LOG_ROOT}/simclr_${DATASET}_%A_%a.err" \
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
  if [[ -n "${PRESET}" ]]; then
    echo "  preset:           ${PRESET}"
  fi
  if [[ "${#EXTRA_CONFIG_SETS[@]}" -gt 0 ]]; then
    echo "  config_sets:      ${EXTRA_CONFIG_SETS[*]}"
  fi
  echo "  run_name:         ${RUN_NAME}"
  echo "  run_root:         ${RUN_ROOT}"
  echo "  time_limit:       ${TIME_LIMIT}"
  echo "  logs:             ${LOG_ROOT}"
  exit 0
fi

if [[ -z "${TASK_FOLD}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_FOLD="${SLURM_ARRAY_TASK_ID}"
fi

RESOLVED_SPLIT_CSV="${SPLIT_CSV}"
DATA_PROFILE=""
BENIN_SPLIT_CSV=""
SA_SPLIT_CSV=""

if [[ "${DATASET}" == "benin" ]]; then
  DATA_PROFILE="${BENIN_DATA_PROFILE}"
  if [[ -z "${TASK_FOLD}" ]]; then
    echo "Benin SimCLR requires a resolved fold id." >&2
    exit 1
  fi
  BENIN_FOLD_CONFIG="configs/cscs/benin_fold${TASK_FOLD}.yaml"
  if [[ -z "${RESOLVED_SPLIT_CSV}" ]]; then
    RESOLVED_SPLIT_CSV="$(yaml_get "${BENIN_FOLD_CONFIG}" "split_csv")"
  fi
elif [[ "${DATASET}" == "sa" ]]; then
  DATA_PROFILE="${SA_DATA_PROFILE}"
elif [[ "${DATASET}" == "benin_sa" ]]; then
  if [[ -z "${TASK_FOLD}" ]]; then
    echo "Combined Benin+SA SimCLR requires a resolved fold id." >&2
    exit 1
  fi
  BENIN_FOLD_CONFIG="configs/cscs/benin_fold${TASK_FOLD}.yaml"
  BENIN_SPLIT_CSV="$(yaml_get "${BENIN_FOLD_CONFIG}" "split_csv")"
  SA_SPLIT_CSV="${SPLIT_CSV}"
  if [[ -z "${BENIN_SPLIT_CSV}" || -z "${SA_SPLIT_CSV}" ]]; then
    echo "Could not resolve Benin and SA split CSVs for combined SimCLR." >&2
    exit 1
  fi
fi

if [[ "${DATASET}" != "benin_sa" && -z "${RESOLVED_SPLIT_CSV}" ]]; then
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
EXTRA_SET_ARGS=()
for config_set in "${EXTRA_CONFIG_SETS[@]}"; do
  EXTRA_SET_ARGS+=(--set "${config_set}")
done

if [[ "${DATASET}" == "benin_sa" ]]; then
  RESOLVED_BENIN_CONFIG="${OUTPUT_DIR}/resolved_simclr_benin_config.yaml"
  RESOLVED_SA_CONFIG="${OUTPUT_DIR}/resolved_simclr_sa_config.yaml"

  python3 "${RESOLVE_CONFIG}" \
    --base "${CONFIG}" \
    --override "${BENIN_DATA_PROFILE}" \
    --output "${RESOLVED_BENIN_CONFIG}" \
    "${EXTRA_SET_ARGS[@]}" \
    --set "experiment_name=${RUN_NAME}_benin_fold${TASK_FOLD}" \
    --set "output_dir=${OUTPUT_DIR}" \
    --set "split_csv=${BENIN_SPLIT_CSV}" \
    --set "dataset_name=benin" \
    --set "combined_dataset_config_paths=[]"

  python3 "${RESOLVE_CONFIG}" \
    --base "${CONFIG}" \
    --override "${SA_DATA_PROFILE}" \
    --output "${RESOLVED_SA_CONFIG}" \
    "${EXTRA_SET_ARGS[@]}" \
    --set "experiment_name=${RUN_NAME}_sa_fold${TASK_FOLD}" \
    --set "output_dir=${OUTPUT_DIR}" \
    --set "split_csv=${SA_SPLIT_CSV}" \
    --set "dataset_name=sa" \
    --set "combined_dataset_config_paths=[]"

  python3 "${RESOLVE_CONFIG}" \
    --base "${CONFIG}" \
    --output "${RESOLVED_CONFIG}" \
    "${EXTRA_SET_ARGS[@]}" \
    --set "experiment_name=${RUN_NAME}" \
    --set "output_dir=${OUTPUT_DIR}" \
    --set "dataset_name=benin_sa" \
    --set "combined_dataset_config_paths=['${RESOLVED_BENIN_CONFIG}', '${RESOLVED_SA_CONFIG}']"
else
  RESOLVE_ARGS=(
    --base "${CONFIG}"
    --output "${RESOLVED_CONFIG}"
  )

  if [[ -n "${DATA_PROFILE}" ]]; then
    RESOLVE_ARGS+=(--override "${DATA_PROFILE}")
  fi

  RESOLVE_ARGS+=("${EXTRA_SET_ARGS[@]}")
  RESOLVE_ARGS+=(
    --set "experiment_name=${RUN_NAME}"
    --set "output_dir=${OUTPUT_DIR}"
    --set "split_csv=${RESOLVED_SPLIT_CSV}"
  )

  python3 "${RESOLVE_CONFIG}" "${RESOLVE_ARGS[@]}"
fi

PY_ARGS=(
  --config "${RESOLVED_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --log-dir "${LOCAL_RUN_LOG_DIR}"
)

echo "Running SimCLR pretraining"
echo "  dataset:          ${DATASET}"
if [[ -n "${PRESET}" ]]; then
  echo "  preset:           ${PRESET}"
fi
echo "  base_config:      ${CONFIG}"
echo "  resolved_config:  ${RESOLVED_CONFIG}"
echo "  output_dir:       ${OUTPUT_DIR}"
if [[ "${#EXTRA_CONFIG_SETS[@]}" -gt 0 ]]; then
  echo "  config_sets:      ${EXTRA_CONFIG_SETS[*]}"
fi
if [[ "${DATASET}" == "benin_sa" ]]; then
  echo "  benin_split_csv:  ${BENIN_SPLIT_CSV}"
  echo "  sa_split_csv:     ${SA_SPLIT_CSV}"
  echo "  benin_config:     ${RESOLVED_BENIN_CONFIG}"
  echo "  sa_config:        ${RESOLVED_SA_CONFIG}"
else
  echo "  split_csv:        ${RESOLVED_SPLIT_CSV}"
fi
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
