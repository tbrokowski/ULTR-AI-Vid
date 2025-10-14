#!/usr/bin/env bash
set -euo pipefail

# Train 3dcnn for folds 1..4 with epochs=1
# Usage:
#   ./train_folds_1_to_4.sh                      # uses default experiment 3dcnn
#   ./train_folds_1_to_4.sh <experiment_name>    # override experiment name
#
# Example:
#   ./train_folds_1_to_4.sh 3dcnn
#   ./train_folds_1_to_4.sh attention_pool

EXPERIMENT_NAME="attention_pool"
VIDEO_FOLDER="/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos"
EPOCHS=1

echo "Experiment: $EXPERIMENT_NAME"
echo "Video folder: $VIDEO_FOLDER"
echo "Epochs: $EPOCHS"
echo "Running folds: 1 2 3 4"
echo "========================================"

for FOLD in 1 2 3 4; do
  echo "\n>>> Starting fold $FOLD"
  set -x
  python3 run_training.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --fold "$FOLD" \
    --video_folder "$VIDEO_FOLDER" \
    --epochs "$EPOCHS"
  set +x
  echo "<<< Finished fold $FOLD"
  echo "----------------------------------------"
done

echo "All requested folds completed."