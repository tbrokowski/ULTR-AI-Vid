#!/usr/bin/env bash
# TB Ablation: Parallel Job Scheduler (single/multi-batch submitter)

set -euo pipefail

# =========================
# Config
# =========================
CONFIG_BASE_DIR="configs"

declare -A ABLATIONS=(
  ["3d_cnn"]="3dcnn"
  ["cnn_lstm"]="cnnlstm"
  ["video_transformer"]="vivit"
  ["original"]="original"
  ["attention_pool"]="attention_pool"
  ["mean_pool"]="mean_pool"
  ["single_task"]="singletask"
  ["uniform"]="uniform"
  ["no_rl_full_train"]="no_rl_full_train"
  ["rl_inception"]="rl_inception"
  ["r2plus1d"]="r2plus1d"
  ["inception3d"]="inception3d"
)

# Which folds to run
FOLDS=(0 1 2 3 4)

# SLURM runner that launches a *single* experiment on 1 node
SLURM_SCRIPT="run_ablation_single_node.sh"

# Logs
LOG_DIR="/users/$USER/tb_ablation/logs"
mkdir -p "$LOG_DIR"

# Cluster capacity model (edit to your cluster)
TOTAL_NODES=8
GPUS_PER_NODE=4
NODES_PER_EXPERIMENT=1
GPUS_PER_EXPERIMENT=$GPUS_PER_NODE             # typically = GPUS_PER_NODE for 1 node/exp
MAX_PARALLEL_JOBS=$(( TOTAL_NODES / NODES_PER_EXPERIMENT ))

# SLURM defaults
CPUS_PER_TASK=32                               # 288 is overkill; tune if needed
TIME_LIMIT="2:00:00"
ACCOUNT="a127"
RESERVATION=""                                  # e.g. "--reservation=sai-a127"

# Submission pacing
SUBMIT_MODE="parallel"                          # parallel | batch | sequential
WAIT_BETWEEN_JOBS=3

# =========================
# Helpers
# =========================
log_info()    { echo "[INFO]    $*"; }
log_ok()      { echo "[OK]      $*"; }
log_err()     { echo "[ERROR]   $*" >&2; }
hdr()         { echo -e "\n==== $* ====\n"; }

get_running_jobs() { squeue -u "$USER" -h -t RUNNING -r | wc -l; }
get_pending_jobs() { squeue -u "$USER" -h -t PENDING -r | wc -l; }

wait_for_slot() {
  local cap="$1"
  while [ "$(get_running_jobs)" -ge "$cap" ]; do
    log_info "Waiting for slot… (Running: $(get_running_jobs), Pending: $(get_pending_jobs), Cap: $cap)"
    sleep 30
  done
}

have_config() {
  local cfg="$1"
  [[ -f "$cfg" ]]
}

# =========================
# Core submitters
# =========================
submit_parallel_job() {
  local cfg="$1" ablation="$2" fold="$3"
  local job="tb_${ablation}_f${fold}"

  if ! have_config "$cfg"; then
    log_err "Missing config: $cfg"
    return 1
  fi

  # Gate by parallel capacity
  if [[ "$SUBMIT_MODE" == "parallel" ]]; then
    wait_for_slot "$MAX_PARALLEL_JOBS"
  fi

  # Build sbatch
  local cmd="sbatch"
  cmd+=" --job-name=${job}"
  cmd+=" --nodes=${NODES_PER_EXPERIMENT}"
  cmd+=" --ntasks-per-node=1"
  cmd+=" --gres=gpu:${GPUS_PER_EXPERIMENT}"
  cmd+=" --cpus-per-task=${CPUS_PER_TASK}"
  cmd+=" --time=${TIME_LIMIT}"
  cmd+=" -A ${ACCOUNT}"
  [[ -n "$RESERVATION" ]] && cmd+=" ${RESERVATION}"
  cmd+=" --output=${LOG_DIR}/R-%x.%j.out"
  cmd+=" --error=${LOG_DIR}/R-%x.%j.err"
  cmd+=" ${SLURM_SCRIPT} ${cfg}"

  log_info "Submitting: ${job}  (cfg: ${cfg})"
  if out=$(eval "$cmd"); then
    local id; id=$(echo "$out" | grep -oE '[0-9]+$' || true)
    log_ok "Job ${id:-?} submitted: ${job}"
    echo "${id:-?},${job},${ablation},${fold},${cfg}" >> "${LOG_DIR}/submitted_jobs_parallel.csv"
    return 0
  else
    log_err "Submit failed: ${job}"
    return 1
  fi
}

# Optional: batch pack N experiments into one allocation (advanced)
submit_batch_job() {
  local cfgs=("$@")
  local n="${#cfgs[@]}"
  log_info "Submitting batch of $n experiments (packed into one allocation)"
  # For brevity, re-use your earlier batch template if you need multi-node packing.
  # Most users prefer the simpler per-job submitter above.
  log_err "Batch mode template omitted for clarity. Use 'parallel' for robust scheduling."
  return 1
}

# =========================
# Main
# =========================
main() {
  local mode="${1:-$SUBMIT_MODE}"

  hdr "TB Ablation Scheduler"
  echo "Mode: $mode"
  echo "TOTAL_NODES=${TOTAL_NODES}, GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "NODES_PER_EXPERIMENT=${NODES_PER_EXPERIMENT}, GPUS_PER_EXPERIMENT=${GPUS_PER_EXPERIMENT}"
  echo "MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
  echo

  # CSV header
  echo "job_id,job_name,ablation_type,fold,config_file" > "${LOG_DIR}/submitted_jobs_parallel.csv"

  # Build list of (config, ablation, fold)
  declare -a todo=()
  for abl in "${!ABLATIONS[@]}"; do
    dir="${ABLATIONS[$abl]}"
    for fold in "${FOLDS[@]}"; do
      cfg="${CONFIG_BASE_DIR}/${dir}/fold${fold}.yaml"
      # Special case for uniform
      if [[ "$abl" == "uniform" && ! -f "$cfg" ]]; then
        cfg="${CONFIG_BASE_DIR}/${dir}/tb_drl_mil_Final_fold${fold}.yaml"
      fi
      if have_config "$cfg"; then
        todo+=("${cfg}|${abl}|${fold}")
      else
        log_err "Config not found (skipping): $cfg"
      fi
    done
  done

  local total="${#todo[@]}"
  hdr "Found $total experiments"
  [[ "$total" -eq 0 ]] && { log_err "No configs found. Exiting."; exit 1; }

  local ok=0 fail=0

  case "$mode" in
    parallel|sequential)
      # sequential just disables slot waiting throttle; we still sleep a bit
      [[ "$mode" == "sequential" ]] && MAX_PARALLEL_JOBS=1
      for item in "${todo[@]}"; do
        IFS='|' read -r cfg abl fold <<< "$item"
        if submit_parallel_job "$cfg" "$abl" "$fold"; then
          ((ok++))
        else
          ((fail++))
        fi
        sleep "$WAIT_BETWEEN_JOBS"
      done
      ;;
    batch)
      submit_batch_job "${todo[@]}" || fail="$total"
      ;;
    *)
      log_err "Unknown mode: $mode (use: parallel | sequential | batch)"
      exit 1
      ;;
  esac

  hdr "Submission Summary"
  echo "Total: $total"
  echo "Submitted OK: $ok"
  echo "Failed: $fail"
  echo
  echo "CSV: ${LOG_DIR}/submitted_jobs_parallel.csv"
  echo
  echo "Tips:"
  echo "  watch -n 15 'squeue -u $USER'"
  echo "  tail -f ${LOG_DIR}/R-*.err"
}

# Help
if [[ "${1:-}" == "help" || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [parallel|sequential|batch]"
  exit 0
fi

main "$@"