#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH -A a127
#SBATCH --gpus=1
#SBATCH --job-name=ultrai_domain_probe
#SBATCH --output=logs/domain_probe_%j.out
#SBATCH --error=logs/domain_probe_%j.err

set -euo pipefail
trap 'echo "Domain shift probe failed. Check logs/domain_probe_*.err, logs/domain_probe_*.out, and domain_shift_probe.log in the output directory." >&2' ERR

REPO_ROOT="/users/lxflk/ULTR-AI-Vid"
RUNS_ROOT="/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs"
VENV_PYTHON="/users/lxflk/.venvs/ultrai/bin/python"

cd "${REPO_ROOT}"
mkdir -p logs

CHECKPOINT=""
MODEL_CONFIG=""
BENIN_CONFIG=""
SA_CONFIG=""
OUTPUT_DIR=""
FEATURE_DIR=""

BENIN_SPLIT="test"
SA_SPLIT="test"
BENIN_SPLIT_CSV=""
SA_SPLIT_CSV=""

BATCH_SIZE=""
NUM_WORKERS=""
FRAME_SAMPLING=""
MAX_SITES=""
MAX_PATIENTS_PER_DOMAIN=""
TEST_SIZE="0.25"
NUM_REPEATS="5"
MAX_TRAIN_SAMPLES_PER_DOMAIN="50000"
MAX_TEST_SAMPLES_PER_DOMAIN="20000"
MAX_PLOT_SAMPLES_PER_DOMAIN="2000"
SAVE_FEATURES=true
SAVE_PLOTS=true

usage() {
  cat <<EOF
Usage:
  sbatch train_job_domain_shift_probe.sh [OPTIONS]

Options:
  --checkpoint PATH                    Checkpoint to probe
  --model-config PATH                  Config used to build the checkpointed model
  --benin-config PATH                  Benin dataset config
  --sa-config PATH                     SA dataset config
  --output-dir PATH                    Output directory for probe metrics and summaries
  --feature-dir PATH                   Optional feature cache directory

  --benin-split {train,val,test,all}   Benin split to probe
  --sa-split {train,val,test,all}      SA split to probe
  --benin-split-csv PATH               Optional Benin split CSV override
  --sa-split-csv PATH                  Optional SA split CSV override

  --batch-size INT                     Batch size override
  --num-workers INT                    DataLoader workers override
  --frame-sampling INT                 Frame sampling override
  --max-sites INT                      Max sites override
  --max-patients-per-domain INT        Patient cap per domain
  --test-size FLOAT                    Held-out patient fraction for the classifier
  --num-repeats INT                    Number of grouped classifier repeats
  --max-train-samples-per-domain INT   Train-row cap per domain (<=0 disables cap)
  --max-test-samples-per-domain INT    Test-row cap per domain (<=0 disables cap)
  --max-plot-samples-per-domain INT    PCA point cap per domain (<=0 disables cap)
  --no-save-features                   Skip saving raw feature arrays
  --no-save-plots                      Skip the PNG summaries

Examples:
  sbatch --partition=debug --time=00:30:00 train_job_domain_shift_probe.sh --max-patients-per-domain 4 --frame-sampling 8 --max-sites 3
  sbatch train_job_domain_shift_probe.sh \\
    --checkpoint /capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/<run-name>/checkpoints/finetune_full/checkpoint_best.pth \\
    --model-config /capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/<run-name>/resolved_finetune_config.yaml \\
    --benin-config /capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/<run-name>/fold3/resolved_config.yaml \\
    --sa-config /capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/<run-name>/resolved_finetune_config.yaml \\
    --output-dir ${RUNS_ROOT}/domain_shift_probe_results/sa_finetuned_full_test \\
    --feature-dir ${RUNS_ROOT}/domain_shift_probe_features/sa_finetuned_full_test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --benin-config)
      BENIN_CONFIG="$2"
      shift 2
      ;;
    --sa-config)
      SA_CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --feature-dir)
      FEATURE_DIR="$2"
      shift 2
      ;;
    --benin-split)
      BENIN_SPLIT="$2"
      shift 2
      ;;
    --sa-split)
      SA_SPLIT="$2"
      shift 2
      ;;
    --benin-split-csv)
      BENIN_SPLIT_CSV="$2"
      shift 2
      ;;
    --sa-split-csv)
      SA_SPLIT_CSV="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --frame-sampling)
      FRAME_SAMPLING="$2"
      shift 2
      ;;
    --max-sites)
      MAX_SITES="$2"
      shift 2
      ;;
    --max-patients-per-domain)
      MAX_PATIENTS_PER_DOMAIN="$2"
      shift 2
      ;;
    --test-size)
      TEST_SIZE="$2"
      shift 2
      ;;
    --num-repeats)
      NUM_REPEATS="$2"
      shift 2
      ;;
    --max-train-samples-per-domain)
      MAX_TRAIN_SAMPLES_PER_DOMAIN="$2"
      shift 2
      ;;
    --max-test-samples-per-domain)
      MAX_TEST_SAMPLES_PER_DOMAIN="$2"
      shift 2
      ;;
    --max-plot-samples-per-domain)
      MAX_PLOT_SAMPLES_PER_DOMAIN="$2"
      shift 2
      ;;
    --no-save-features)
      SAVE_FEATURES=false
      shift
      ;;
    --no-save-plots)
      SAVE_PLOTS=false
      shift
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

if [[ -z "${CHECKPOINT}" ]]; then
  echo "--checkpoint is required." >&2
  exit 1
fi

if [[ -z "${BENIN_CONFIG}" ]]; then
  echo "--benin-config is required." >&2
  exit 1
fi

if [[ -z "${SA_CONFIG}" ]]; then
  echo "--sa-config is required." >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ -n "${MODEL_CONFIG}" && ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model config not found: ${MODEL_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${BENIN_CONFIG}" ]]; then
  echo "Benin config not found: ${BENIN_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${SA_CONFIG}" ]]; then
  echo "SA config not found: ${SA_CONFIG}" >&2
  exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Domain-shift venv python not found or not executable: ${VENV_PYTHON}" >&2
  exit 1
fi

PY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --benin-config "${BENIN_CONFIG}"
  --sa-config "${SA_CONFIG}"
  --benin-split "${BENIN_SPLIT}"
  --sa-split "${SA_SPLIT}"
  --test-size "${TEST_SIZE}"
  --num-repeats "${NUM_REPEATS}"
  --max-train-samples-per-domain "${MAX_TRAIN_SAMPLES_PER_DOMAIN}"
  --max-test-samples-per-domain "${MAX_TEST_SAMPLES_PER_DOMAIN}"
  --max-plot-samples-per-domain "${MAX_PLOT_SAMPLES_PER_DOMAIN}"
)

if [[ -n "${MODEL_CONFIG}" ]]; then
  PY_ARGS+=(--model-config "${MODEL_CONFIG}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  PY_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${BENIN_SPLIT_CSV}" ]]; then
  PY_ARGS+=(--benin-split-csv "${BENIN_SPLIT_CSV}")
fi

if [[ -n "${FEATURE_DIR}" ]]; then
  PY_ARGS+=(--feature-dir "${FEATURE_DIR}")
fi

if [[ -n "${SA_SPLIT_CSV}" ]]; then
  PY_ARGS+=(--sa-split-csv "${SA_SPLIT_CSV}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  PY_ARGS+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${NUM_WORKERS}" ]]; then
  PY_ARGS+=(--num-workers "${NUM_WORKERS}")
fi

if [[ -n "${FRAME_SAMPLING}" ]]; then
  PY_ARGS+=(--frame-sampling "${FRAME_SAMPLING}")
fi

if [[ -n "${MAX_SITES}" ]]; then
  PY_ARGS+=(--max-sites "${MAX_SITES}")
fi

if [[ -n "${MAX_PATIENTS_PER_DOMAIN}" ]]; then
  PY_ARGS+=(--max-patients-per-domain "${MAX_PATIENTS_PER_DOMAIN}")
fi

if [[ "${SAVE_FEATURES}" == false ]]; then
  PY_ARGS+=(--no-save-features)
fi

if [[ "${SAVE_PLOTS}" == false ]]; then
  PY_ARGS+=(--no-save-plots)
fi

echo "Running domain shift probe"
echo "  checkpoint:                    ${CHECKPOINT}"
echo "  model_config:                  ${MODEL_CONFIG:-<defaults to benin_config>}"
echo "  benin_config:                  ${BENIN_CONFIG}"
echo "  sa_config:                     ${SA_CONFIG}"
echo "  output_dir:                    ${OUTPUT_DIR:-${RUNS_ROOT}/domain_shift_probe_results/<checkpoint-stem>}"
echo "  feature_dir override:          ${FEATURE_DIR:-${RUNS_ROOT}/domain_shift_probe_features/<run-name>}"
echo "  python:                        ${VENV_PYTHON}"
echo "  benin_split:                   ${BENIN_SPLIT}"
echo "  sa_split:                      ${SA_SPLIT}"
echo "  max_patients_per_domain:       ${MAX_PATIENTS_PER_DOMAIN:-all}"
echo "  frame_sampling override:       ${FRAME_SAMPLING:-config default}"
echo "  max_sites override:            ${MAX_SITES:-config default}"


srun --environment=/users/lxflk/.edf/ultrai.toml \
     env PYTHONUNBUFFERED=1 \
     "${VENV_PYTHON}" -u -m ultrai.analysis.domain_shift.probe \
     "${PY_ARGS[@]}"

echo "Domain shift probe completed successfully."
