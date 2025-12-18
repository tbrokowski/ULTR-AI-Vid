#!/usr/bin/env bash
# Downstream Evaluation Runner
# Runs evaluate_downstream.py for multiple experiments and folds

set -euo pipefail

# =========================
# Config
# =========================
CONFIG_BASE_DIR="configs/attention_pool_extra3_full_train2"
RESULTS_BASE_DIR="/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results"

# Experiments to evaluate
declare -A EXPERIMENTS=(
#   ["attention_pool_extra4_k1"]="attention_pool_extra4_k1"
#   ["attention_pool_extra4_k8"]="attention_pool_extra4_k8"
    ["attention_pool_extra3_full_train2"]="attention_pool_extra3_full_train2"
)

# Which folds to run
FOLDS=(0 1 2 3 4)

# Model type
MODEL_TYPE="attention_pool"

# Video folder override (adjust this to the correct path)
VIDEO_FOLDER="/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos"

# =========================
# Helpers
# =========================
log_info()    { echo "[INFO]    $*"; }
log_ok()      { echo "[OK]      $*"; }
log_err()     { echo "[ERROR]   $*" >&2; }
hdr()         { echo -e "\n==== $* ====\n"; }

# =========================
# Main
# =========================
main() {
  hdr "Downstream Efficiency Metric Runner"
  
  for exp_name in "${!EXPERIMENTS[@]}"; do
    exp_dir="${EXPERIMENTS[$exp_name]}"
    
    hdr "Processing experiment: $exp_name"
    log_info "Experiment directory: $RESULTS_BASE_DIR/$exp_dir"
    
    for fold in "${FOLDS[@]}"; do
      log_info "Processing fold $fold..."
      
      CONFIG_FILE="${CONFIG_BASE_DIR}/fold${fold}.yaml"
      MODEL_FILE="${RESULTS_BASE_DIR}/${exp_dir}/fold${fold}/checkpoint_best.pth"
      OUTPUT_DIR="${RESULTS_BASE_DIR}/${exp_dir}/eval_results"
      
      # Check if files exist
      if [[ ! -f "$CONFIG_FILE" ]]; then
        log_err "Config file not found: $CONFIG_FILE"
        continue
      fi
      
      if [[ ! -f "$MODEL_FILE" ]]; then
        log_err "Model file not found: $MODEL_FILE"
        continue
      fi
      
      # Run evaluation
      # log_info "Running: python ultr_ai/efficiency/metrics.py --model-type $MODEL_TYPE --config $CONFIG_FILE --model $MODEL_FILE --fold $fold --output-dir $OUTPUT_DIR --video_folder $VIDEO_FOLDER"
      
      if python ultr_ai/efficiency/metrics.py \
        --model-type "$MODEL_TYPE" \
        --config "$CONFIG_FILE" \
        --model "$MODEL_FILE" \
        --fold "$fold" \
        --output-dir "$OUTPUT_DIR" \
        --video_folder "$VIDEO_FOLDER"; then
        log_ok "Fold $fold completed successfully"
      else
        log_err "Fold $fold failed"
      fi
      
      echo ""
    done
    
    log_ok "Experiment $exp_name completed"
    echo ""
  done
  
  hdr "All efficiency metric evaluations complete!"
}

main "$@"
