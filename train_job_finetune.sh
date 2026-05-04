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
DEFAULT_LOCAL_VENV="/users/lxflk/.venvs/ultatron"
RESOLVE_CONFIG="${REPO_ROOT}/ultrai/utils/config_resolver.py"
DEFAULT_FINETUNE_CONFIG="configs/cscs/finetune.yaml"
BALANCE_CLIP2_POS4_CONFIG="configs/cscs/finetune_balance_clip2_pos4_lowlr.yaml"
BALANCE_CLIP0_POS2_CONFIG="configs/cscs/finetune_balance_clip0_pos2_lowlr.yaml"
BENIN_DATA_PROFILE="configs/cscs/dataset_benin.yaml"
SA_DATA_PROFILE="configs/cscs/dataset_sa.yaml"

WORKER_MODE=false
SOURCE_DATASET="benin"
TARGET_DATASET="sa"
SOURCE_CHECKPOINT=""
SOURCE_CONFIG=""
CONFIG=""
CUSTOM_CONFIG=false
FINETUNE_PRESET=""
FINETUNE_PRESET_CONFIG=""
RUN_NAME=""
RUN_ROOT=""
FOLD=""
SPLIT_CSV=""
CLIP_UNFREEZE_LAST_N_LAYERS=""
FREEZE_BACKBONE_MODE="default"
DOMAIN_ADAPTATION="none"
SKIP_ZERO_SHOT=false
SKIP_TARGET_TEST=false
SOURCE_EVAL_SPLITS="test"
ALLOW_MISSING_SOURCE_CHECKPOINT=false
RUNTIME_MODE="edf"
LOCAL_VENV="${DEFAULT_LOCAL_VENV}"
CONFIG_SETS=()

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
  --finetune-preset NAME            Apply a known finetuning preset after the base/dataset config.
                                    Available: balance_clip2_pos4_lowlr, balance_clip0_pos2_lowlr, none.
  --fold INT                        Target Benin fold when target dataset is Benin.
  --split-csv PATH                  Explicit split CSV for target Benin runs. SA fine-tuning builds its own split internally.
  --clip-unfreeze-last-n-layers N   Override the CLIP unfreeze setting from the resolved config.
  --freeze-backbone                 Force the CLIP backbone to stay frozen.
  --no-freeze-backbone              Force the CLIP backbone to be trainable.
  --domain-adaptation {none,dann,fixmatch,ewc,lwf}
                                  Optional adaptation algorithm. DANN and EWC require --source-config and SA target. FixMatch uses the SA training pool as unlabeled target data. LwF uses a frozen source teacher. Default: none.
  --dann                            Shortcut for --domain-adaptation dann.
  --fixmatch                        Shortcut for --domain-adaptation fixmatch.
  --ewc                             Shortcut for --domain-adaptation ewc.
  --lwf                             Shortcut for --domain-adaptation lwf.
  --set KEY=VALUE                    Override a resolved config value after base/dataset profile merging. May be repeated.
  --skip-zero-shot                   Skip target zero-shot evaluation before fine-tuning.
  --skip-target-test                 Skip target-domain test evaluation after fine-tuning.
  --source-eval-splits CSV           Source-domain splits to evaluate after fine-tuning. Default: test.
  --allow-missing-source-checkpoint  Continue even if the submission host cannot stat the source checkpoint path.
  --runtime {edf,local-venv}         Runtime for the Slurm job step. Default: edf.
  --local-venv PATH                  Virtualenv for --runtime local-venv. Default: /users/lxflk/.venvs/ultatron.
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
  local tag="baseline"

  if [[ "${lowered}" == *"simclr"* || "${lowered}" == *"vision_encoder"* ]]; then
    tag="simclr"
  fi
  if [[ "${lowered}" == *"dann"* || "${DOMAIN_ADAPTATION}" == "dann" ]]; then
    if [[ "${tag}" == "baseline" ]]; then
      tag="dann"
    elif [[ "${tag}" != *"dann"* ]]; then
      tag="${tag}+dann"
    fi
  fi
  if [[ "${lowered}" == *"fixmatch"* || "${DOMAIN_ADAPTATION}" == "fixmatch" ]]; then
    if [[ "${tag}" == "baseline" ]]; then
      tag="fixmatch"
    elif [[ "${tag}" != *"fixmatch"* ]]; then
      tag="${tag}+fixmatch"
    fi
  fi
  if [[ "${lowered}" == *"ewc"* || "${DOMAIN_ADAPTATION}" == "ewc" ]]; then
    if [[ "${tag}" == "baseline" ]]; then
      tag="ewc"
    elif [[ "${tag}" != *"ewc"* ]]; then
      tag="${tag}+ewc"
    fi
  fi
  if [[ "${lowered}" == *"lwf"* || "${DOMAIN_ADAPTATION}" == "lwf" ]]; then
    if [[ "${tag}" == "baseline" ]]; then
      tag="lwf"
    elif [[ "${tag}" != *"lwf"* ]]; then
      tag="${tag}+lwf"
    fi
  fi
  if [[ -n "${FINETUNE_PRESET}" && "${FINETUNE_PRESET}" != "none" ]]; then
    if [[ "${tag}" == "baseline" ]]; then
      tag="${FINETUNE_PRESET}"
    elif [[ "${tag}" != *"${FINETUNE_PRESET}"* ]]; then
      tag="${tag}+${FINETUNE_PRESET}"
    fi
  fi

  printf '%s\n' "${tag}"
}

default_run_name() {
  local timestamp
  local tag
  local scope

  timestamp="$(date +%Y%m%d_%H%M%S)"
  tag="$(infer_source_tag)"

  if [[ -n "${FOLD}" ]]; then
    scope="fold${FOLD}"
  elif [[ "${TARGET_DATASET}" == "sa" && "${SOURCE_CHECKPOINT}" =~ /fold([0-9]+)/checkpoint_best\.pth$ ]]; then
    scope="src-fold${BASH_REMATCH[1]}"
  else
    scope="full"
  fi

  printf '%s__finetune__%s_to_%s__%s__%s\n' "${timestamp}" "${SOURCE_DATASET}" "${TARGET_DATASET}" "${scope}" "${tag}"
}

resolve_finetune_preset_config() {
  case "${FINETUNE_PRESET}" in
    ""|"none")
      FINETUNE_PRESET_CONFIG=""
      ;;
    "balance_clip2_pos4_lowlr")
      FINETUNE_PRESET_CONFIG="${BALANCE_CLIP2_POS4_CONFIG}"
      ;;
    "balance_clip0_pos2_lowlr")
      FINETUNE_PRESET_CONFIG="${BALANCE_CLIP0_POS2_CONFIG}"
      ;;
    *)
      echo "--finetune-preset must be one of: balance_clip2_pos4_lowlr, balance_clip0_pos2_lowlr, none" >&2
      exit 1
      ;;
  esac
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
    --finetune-preset)
      FINETUNE_PRESET="$2"
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
    --domain-adaptation)
      DOMAIN_ADAPTATION="$2"
      shift 2
      ;;
    --dann)
      DOMAIN_ADAPTATION="dann"
      shift
      ;;
    --fixmatch)
      DOMAIN_ADAPTATION="fixmatch"
      shift
      ;;
    --ewc)
      DOMAIN_ADAPTATION="ewc"
      shift
      ;;
    --lwf)
      DOMAIN_ADAPTATION="lwf"
      shift
      ;;
    --set)
      CONFIG_SETS+=("$2")
      shift 2
      ;;
    --skip-zero-shot)
      SKIP_ZERO_SHOT=true
      shift
      ;;
    --skip-target-test)
      SKIP_TARGET_TEST=true
      shift
      ;;
    --source-eval-splits)
      SOURCE_EVAL_SPLITS="$2"
      shift 2
      ;;
    --allow-missing-source-checkpoint)
      ALLOW_MISSING_SOURCE_CHECKPOINT=true
      shift
      ;;
    --runtime)
      RUNTIME_MODE="$2"
      shift 2
      ;;
    --local-venv)
      LOCAL_VENV="$2"
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

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
  echo "--source-checkpoint is required." >&2
  exit 1
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  if [[ "${ALLOW_MISSING_SOURCE_CHECKPOINT}" == true ]]; then
    echo "Warning: source checkpoint is not stat-able from this host; proceeding anyway: ${SOURCE_CHECKPOINT}" >&2
  else
    echo "Source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    exit 1
  fi
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

if [[ "${DOMAIN_ADAPTATION}" != "none" && "${DOMAIN_ADAPTATION}" != "dann" && "${DOMAIN_ADAPTATION}" != "fixmatch" && "${DOMAIN_ADAPTATION}" != "ewc" && "${DOMAIN_ADAPTATION}" != "lwf" ]]; then
  echo "--domain-adaptation must be one of: none, dann, fixmatch, ewc, lwf" >&2
  exit 1
fi

if [[ "${DOMAIN_ADAPTATION}" != "none" && "${TARGET_DATASET}" != "sa" ]]; then
  echo "${DOMAIN_ADAPTATION} is currently wired for --target-dataset sa." >&2
  exit 1
fi

if [[ "${RUNTIME_MODE}" != "edf" && "${RUNTIME_MODE}" != "local-venv" ]]; then
  echo "--runtime must be one of: edf, local-venv" >&2
  exit 1
fi

if [[ "${RUNTIME_MODE}" == "local-venv" && ! -x "${LOCAL_VENV}/bin/python" ]]; then
  echo "Local virtualenv does not contain a Python executable: ${LOCAL_VENV}" >&2
  exit 1
fi

if [[ ( "${DOMAIN_ADAPTATION}" == "dann" || "${DOMAIN_ADAPTATION}" == "ewc" ) && -z "${SOURCE_CONFIG}" ]]; then
  echo "${DOMAIN_ADAPTATION} requires --source-config so the source-domain data loader can be built." >&2
  exit 1
fi

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${DEFAULT_FINETUNE_CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Base finetune config not found: ${CONFIG}" >&2
  exit 1
fi

resolve_finetune_preset_config

if [[ -n "${FINETUNE_PRESET_CONFIG}" && ! -f "${FINETUNE_PRESET_CONFIG}" ]]; then
  echo "Finetune preset config not found: ${FINETUNE_PRESET_CONFIG}" >&2
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

  if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to reuse existing run directory: ${RUN_ROOT}" >&2
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
  if [[ -n "${FINETUNE_PRESET}" ]]; then
    FORWARDED_ARGS+=(--finetune-preset "${FINETUNE_PRESET}")
  fi
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
  if [[ "${DOMAIN_ADAPTATION}" != "none" ]]; then
    FORWARDED_ARGS+=(--domain-adaptation "${DOMAIN_ADAPTATION}")
  fi
  if [[ "${SKIP_ZERO_SHOT}" == true ]]; then
    FORWARDED_ARGS+=(--skip-zero-shot)
  fi
  if [[ "${SKIP_TARGET_TEST}" == true ]]; then
    FORWARDED_ARGS+=(--skip-target-test)
  fi
  if [[ -n "${SOURCE_EVAL_SPLITS}" ]]; then
    FORWARDED_ARGS+=(--source-eval-splits "${SOURCE_EVAL_SPLITS}")
  fi
  if [[ "${ALLOW_MISSING_SOURCE_CHECKPOINT}" == true ]]; then
    FORWARDED_ARGS+=(--allow-missing-source-checkpoint)
  fi
  FORWARDED_ARGS+=(--runtime "${RUNTIME_MODE}")
  if [[ -n "${LOCAL_VENV}" ]]; then
    FORWARDED_ARGS+=(--local-venv "${LOCAL_VENV}")
  fi
  for config_set in "${CONFIG_SETS[@]}"; do
    FORWARDED_ARGS+=(--set "${config_set}")
  done

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
  echo "  finetune_preset:    ${FINETUNE_PRESET:-<none>}"
  echo "  domain_adaptation:  ${DOMAIN_ADAPTATION}"
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
    --set "domain_adaptation=${DOMAIN_ADAPTATION}"
  )
  RESOLVE_ARGS+=(--override "${SA_DATA_PROFILE}")
  if [[ -n "${FINETUNE_PRESET_CONFIG}" ]]; then
    RESOLVE_ARGS+=(--override "${FINETUNE_PRESET_CONFIG}")
  fi
  if [[ -n "${CLIP_UNFREEZE_LAST_N_LAYERS}" ]]; then
    RESOLVE_ARGS+=(--set "clip_unfreeze_last_n_layers=${CLIP_UNFREEZE_LAST_N_LAYERS}")
  fi
  if [[ "${FREEZE_BACKBONE_MODE}" == "freeze" ]]; then
    RESOLVE_ARGS+=(--set "freeze_backbone=true")
  elif [[ "${FREEZE_BACKBONE_MODE}" == "unfreeze" ]]; then
    RESOLVE_ARGS+=(--set "freeze_backbone=false")
  fi
  for config_set in "${CONFIG_SETS[@]}"; do
    RESOLVE_ARGS+=(--set "${config_set}")
  done
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
  if [[ "${DOMAIN_ADAPTATION}" != "none" ]]; then
    PY_ARGS+=(--domain-adaptation "${DOMAIN_ADAPTATION}")
  fi
  if [[ "${SKIP_ZERO_SHOT}" == true ]]; then
    PY_ARGS+=(--skip-zero-shot)
  fi
  if [[ "${SKIP_TARGET_TEST}" == true ]]; then
    PY_ARGS+=(--skip-target-test)
  fi
  if [[ -n "${SOURCE_EVAL_SPLITS}" ]]; then
    PY_ARGS+=(--source-eval-splits "${SOURCE_EVAL_SPLITS}")
  fi

  echo "Running finetuning"
  echo "  source_dataset:     ${SOURCE_DATASET}"
  echo "  target_dataset:     ${TARGET_DATASET}"
  echo "  domain_adaptation:  ${DOMAIN_ADAPTATION}"
  echo "  base_config:        ${CONFIG}"
  echo "  finetune_preset:    ${FINETUNE_PRESET:-<none>}"
  echo "  resolved_config:    ${RESOLVED_CONFIG}"
  echo "  source_checkpoint:  ${SOURCE_CHECKPOINT}"
echo "  source_config:      ${SOURCE_CONFIG:-<none>}"
echo "  run_root:           ${RUN_ROOT}"
echo "  local_logs:         ${LOCAL_RUN_LOG_DIR}"

if [[ "${RUNTIME_MODE}" == "local-venv" ]]; then
  PY_ARGS_ESCAPED="$(printf '%q ' "${PY_ARGS[@]}")"
  srun bash -lc "cd ${REPO_ROOT@Q} && source ${LOCAL_VENV@Q}/bin/activate && export PYTHONPATH=${REPO_ROOT@Q}:\${PYTHONPATH:-} && python -m ultrai.training.finetune ${PY_ARGS_ESCAPED}"
else
  srun --environment="${EDF_ENV}" \
    python3 -m ultrai.training.finetune \
    "${PY_ARGS[@]}"
fi

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
  --set "domain_adaptation=${DOMAIN_ADAPTATION}"
)

for override_config in "${OVERRIDE_CONFIGS[@]}"; do
  if [[ ! -f "${override_config}" ]]; then
    echo "Override config not found: ${override_config}" >&2
    exit 1
  fi
  RESOLVE_ARGS+=(--override "${override_config}")
done

if [[ -n "${FINETUNE_PRESET_CONFIG}" ]]; then
  RESOLVE_ARGS+=(--override "${FINETUNE_PRESET_CONFIG}")
fi

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
for config_set in "${CONFIG_SETS[@]}"; do
  RESOLVE_ARGS+=(--set "${config_set}")
done

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
if [[ "${DOMAIN_ADAPTATION}" != "none" ]]; then
  PY_ARGS+=(--domain-adaptation "${DOMAIN_ADAPTATION}")
fi
if [[ -n "${SOURCE_EVAL_SPLITS}" ]]; then
  PY_ARGS+=(--source-eval-splits "${SOURCE_EVAL_SPLITS}")
fi

echo "Running finetuning"
echo "  source_dataset:     ${SOURCE_DATASET}"
echo "  target_dataset:     ${TARGET_DATASET}"
echo "  domain_adaptation:  ${DOMAIN_ADAPTATION}"
echo "  base_config:        ${CONFIG}"
echo "  finetune_preset:    ${FINETUNE_PRESET:-<none>}"
echo "  resolved_config:    ${RESOLVED_CONFIG}"
echo "  source_checkpoint:  ${SOURCE_CHECKPOINT}"
echo "  experiment_dir:     ${EXPERIMENT_DIR}"
echo "  local_logs:         ${LOCAL_RUN_LOG_DIR}"

if [[ "${RUNTIME_MODE}" == "local-venv" ]]; then
  PY_ARGS_ESCAPED="$(printf '%q ' "${PY_ARGS[@]}")"
  srun bash -lc "cd ${REPO_ROOT@Q} && source ${LOCAL_VENV@Q}/bin/activate && export PYTHONPATH=${REPO_ROOT@Q}:\${PYTHONPATH:-} && python -m ultrai.training.finetune ${PY_ARGS_ESCAPED}"
else
  srun --environment="${EDF_ENV}" \
    python3 -m ultrai.training.finetune \
    "${PY_ARGS[@]}"
fi

echo "Finetuning finished"
echo "  best_checkpoint:    ${EXPERIMENT_DIR}/checkpoint_best.pth"
echo "  final_results:      ${EXPERIMENT_DIR}/final_results"
