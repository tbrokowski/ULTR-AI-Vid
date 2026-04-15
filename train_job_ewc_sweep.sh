#!/bin/bash
# Submit a principled EWC lambda sweep as a Slurm array.
#
# The sweep keeps Benin test untouched by evaluating post-finetune retention on
# the source validation split only. Final Benin test evaluation can be run later
# for the selected lambda.

set -euo pipefail

REPO_ROOT="/users/lxflk/ULTR-AI-Vid"
SCRATCH_ROOT="/capstor/scratch/cscs/lxflk/ULTR-AI-Vid"
LOG_ROOT="${REPO_ROOT}/logs"
FINETUNE_JOB="${REPO_ROOT}/train_job_finetune.sh"

WORKER_MODE=false
SOURCE_DATASET="benin"
TARGET_DATASET="sa"
SOURCE_CHECKPOINT=""
SOURCE_CONFIG=""
CONFIG="configs/cscs/finetune.yaml"
LAMBDA_CSV="1000,3000,10000,30000,100000,300000"
FISHER_MAX_BATCHES="160"
EWC_PARAMETER_SCOPE="trainable"
EWC_NORMALIZE_PENALTY="false"
MAX_PARALLEL="3"
SWEEP_NAME=""
SWEEP_ROOT=""
CLIP_UNFREEZE_LAST_N_LAYERS=""
FREEZE_BACKBONE_MODE="default"
CONFIG_SETS=()

usage() {
  cat <<'EOF'
Usage:
  ./train_job_ewc_sweep.sh --source-checkpoint PATH --source-config PATH [OPTIONS]

Options:
  --source-checkpoint PATH          Required source checkpoint.
  --source-config PATH              Required resolved source-domain config for Fisher and source-val retention.
  --source-dataset {benin,sa}       Source dataset. Default: benin.
  --target-dataset {sa}             Target dataset. Default: sa.
  --config PATH                     Base fine-tuning config. Default: configs/cscs/finetune.yaml.
  --lambdas CSV                     EWC lambda values. Default: 1000,3000,10000,30000,100000,300000.
  --fisher-max-batches INT          Source Fisher batches per lambda. Default: 160.
  --parameter-scope SCOPE           EWC parameter scope. Default: trainable.
  --normalize-penalty               Average the EWC penalty over consolidated parameters.
  --no-normalize-penalty            Use the raw summed EWC penalty. Default.
  --set KEY=VALUE                   Extra resolved config override for each sweep task. May be repeated.
  --max-parallel INT                Max concurrent array tasks. Default: 3.
  --clip-unfreeze-last-n-layers N   Override CLIP unfreeze setting.
  --freeze-backbone                 Force CLIP backbone frozen.
  --no-freeze-backbone              Force CLIP backbone trainable.
  --sweep-name NAME                 Optional sweep name.
  --sweep-root PATH                 Optional sweep root. Default: scratch finetune/benin_to_sa/<sweep-name>.
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

lambda_label() {
  python3 - "$1" <<'PY'
import decimal
import sys
raw = sys.argv[1]
try:
    value = decimal.Decimal(raw)
    if value == value.to_integral():
        label = str(value.quantize(decimal.Decimal(1)))
    else:
        label = format(value.normalize(), "f")
except Exception:
    label = raw
print(label.replace("-", "m").replace(".", "p").replace("+", ""))
PY
}

default_scope() {
  if [[ "${SOURCE_CHECKPOINT}" =~ /fold([0-9]+)/checkpoint_best\.pth$ ]]; then
    printf 'src-fold%s' "${BASH_REMATCH[1]}"
  else
    printf 'full'
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker)
      WORKER_MODE=true
      shift
      ;;
    --source-checkpoint)
      SOURCE_CHECKPOINT="$2"
      shift 2
      ;;
    --source-config)
      SOURCE_CONFIG="$2"
      shift 2
      ;;
    --source-dataset)
      SOURCE_DATASET="$2"
      shift 2
      ;;
    --target-dataset)
      TARGET_DATASET="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --lambdas)
      LAMBDA_CSV="$2"
      shift 2
      ;;
    --fisher-max-batches)
      FISHER_MAX_BATCHES="$2"
      shift 2
      ;;
    --parameter-scope)
      EWC_PARAMETER_SCOPE="$2"
      shift 2
      ;;
    --normalize-penalty)
      EWC_NORMALIZE_PENALTY="true"
      shift
      ;;
    --no-normalize-penalty)
      EWC_NORMALIZE_PENALTY="false"
      shift
      ;;
    --set)
      CONFIG_SETS+=("$2")
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
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
    --sweep-name)
      SWEEP_NAME="$2"
      shift 2
      ;;
    --sweep-root)
      SWEEP_ROOT="$2"
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

if [[ -z "${SOURCE_CHECKPOINT}" || -z "${SOURCE_CONFIG}" ]]; then
  echo "--source-checkpoint and --source-config are required." >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Source config not found: ${SOURCE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Fine-tuning config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ "${TARGET_DATASET}" != "sa" ]]; then
  echo "EWC sweep is wired for --target-dataset sa." >&2
  exit 1
fi

IFS=',' read -r -a LAMBDAS <<< "${LAMBDA_CSV}"
if [[ "${#LAMBDAS[@]}" -eq 0 ]]; then
  echo "No lambda values provided." >&2
  exit 1
fi

if [[ -z "${SWEEP_NAME}" ]]; then
  SWEEP_NAME="$(date +%Y%m%d_%H%M%S)__ewc_sweep__${SOURCE_DATASET}_to_${TARGET_DATASET}__$(default_scope)"
fi
if [[ -z "${SWEEP_ROOT}" ]]; then
  SWEEP_ROOT="${SCRATCH_ROOT}/runs/finetune/${SOURCE_DATASET}_to_${TARGET_DATASET}/${SWEEP_NAME}"
fi
SWEEP_ROOT="$(abs_path "${SWEEP_ROOT}")"

if [[ "${WORKER_MODE}" != true ]]; then
  mkdir -p "${LOG_ROOT}" "${SWEEP_ROOT}"
  array_max=$((${#LAMBDAS[@]} - 1))
  forwarded_args=(
    --worker
    --source-dataset "${SOURCE_DATASET}"
    --target-dataset "${TARGET_DATASET}"
    --source-checkpoint "${SOURCE_CHECKPOINT}"
    --source-config "${SOURCE_CONFIG}"
    --config "${CONFIG}"
    --lambdas "${LAMBDA_CSV}"
    --fisher-max-batches "${FISHER_MAX_BATCHES}"
    --parameter-scope "${EWC_PARAMETER_SCOPE}"
    --max-parallel "${MAX_PARALLEL}"
    --sweep-name "${SWEEP_NAME}"
    --sweep-root "${SWEEP_ROOT}"
  )
  if [[ "${EWC_NORMALIZE_PENALTY}" == "true" ]]; then
    forwarded_args+=(--normalize-penalty)
  else
    forwarded_args+=(--no-normalize-penalty)
  fi
  for config_set in "${CONFIG_SETS[@]}"; do
    forwarded_args+=(--set "${config_set}")
  done
  if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
    forwarded_args+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
  fi
  if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
    forwarded_args+=(--freeze-backbone)
  elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
    forwarded_args+=(--no-freeze-backbone)
  fi

  job_id="$(
    sbatch \
      --parsable \
      --account=a127 \
      --partition=normal \
      --time=05:00:00 \
      --gpus=1 \
      --array="0-${array_max}%${MAX_PARALLEL}" \
      --job-name="ultrai_ewc_sweep" \
      --output="${LOG_ROOT}/ewc_sweep_%A_%a.out" \
      --error="${LOG_ROOT}/ewc_sweep_%A_%a.err" \
      "$0" \
      "${forwarded_args[@]}"
  )"

  echo "Submitted EWC lambda sweep"
  echo "  job_id:             ${job_id%%;*}"
  echo "  array:              0-${array_max}%${MAX_PARALLEL}"
  echo "  lambdas:            ${LAMBDA_CSV}"
  echo "  fisher_max_batches: ${FISHER_MAX_BATCHES}"
  echo "  parameter_scope:    ${EWC_PARAMETER_SCOPE}"
  echo "  normalize_penalty:  ${EWC_NORMALIZE_PENALTY}"
  echo "  source_checkpoint:  ${SOURCE_CHECKPOINT}"
  echo "  source_config:      ${SOURCE_CONFIG}"
  echo "  sweep_name:         ${SWEEP_NAME}"
  echo "  sweep_root:         ${SWEEP_ROOT}"
  echo "  logs:               ${LOG_ROOT}/ewc_sweep_${job_id%%;*}_<task>.{out,err}"
  exit 0
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#LAMBDAS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${task_id} for ${#LAMBDAS[@]} lambdas" >&2
  exit 1
fi

lambda="${LAMBDAS[${task_id}]}"
label="$(lambda_label "${lambda}")"
run_name="${SWEEP_NAME}__lambda-${label}"
run_root="${SWEEP_ROOT}/lambda-${label}"

finetune_args=(
  --worker
  --source-dataset "${SOURCE_DATASET}"
  --target-dataset "${TARGET_DATASET}"
  --source-checkpoint "${SOURCE_CHECKPOINT}"
  --source-config "${SOURCE_CONFIG}"
  --config "${CONFIG}"
  --ewc
  --run-name "${run_name}"
  --run-root "${run_root}"
  --skip-zero-shot
  --skip-target-test
  --source-eval-splits val
  --set "ewc_lambda=${lambda}"
  --set "ewc_fisher_max_batches=${FISHER_MAX_BATCHES}"
  --set "ewc_parameter_scope=${EWC_PARAMETER_SCOPE}"
  --set "ewc_normalize_penalty=${EWC_NORMALIZE_PENALTY}"
)

for config_set in "${CONFIG_SETS[@]}"; do
  finetune_args+=(--set "${config_set}")
done

if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
  finetune_args+=(--clip-unfreeze-last-n-layers "${CLIP_UNFREEZE_LAST_N_LAYERS}")
fi
if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
  finetune_args+=(--freeze-backbone)
elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
  finetune_args+=(--no-freeze-backbone)
fi

echo "Running EWC sweep task"
echo "  task_id:            ${task_id}"
echo "  ewc_lambda:         ${lambda}"
echo "  fisher_max_batches: ${FISHER_MAX_BATCHES}"
echo "  parameter_scope:    ${EWC_PARAMETER_SCOPE}"
echo "  normalize_penalty:  ${EWC_NORMALIZE_PENALTY}"
echo "  run_name:           ${run_name}"
echo "  run_root:           ${run_root}"

"${FINETUNE_JOB}" "${finetune_args[@]}"
