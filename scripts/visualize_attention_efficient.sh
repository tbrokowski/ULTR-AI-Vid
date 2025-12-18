#!/bin/bash
# Efficient script to visualize attention weights for TB positive and negative videos
# Processes each video once across all folds instead of loading it 5 times

# Base model path (will append fold number)
MODEL_BASE_PATH="/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/attention_pool_extra3_full_train2"

# Output directory
OUTPUT_DIR="./visualization_outputs"
VIDEO_DIR="/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos"
CSV_FILE="$OUTPUT_DIR/attention_metrics_all_folds.csv"
mkdir -p "$OUTPUT_DIR"

# Remove old CSV if it exists (we'll create a fresh one)
if [ -f "$CSV_FILE" ]; then
    echo "Removing old CSV file: $CSV_FILE"
    rm "$CSV_FILE"
fi

echo "========================================="
echo "Efficient Attention Visualization"
echo "========================================="
echo ""
echo "Model base path: $MODEL_BASE_PATH"
echo "Output: $OUTPUT_DIR"
echo "CSV file: $CSV_FILE"
echo ""
echo "Processing strategy:"
echo "  - Each video loaded ONCE"
echo "  - Processed with all 5 folds"
echo "  - Much faster than loading video 5 times!"
echo ""

# Define all videos to process
declare -a TB_POSITIVE=(
    "25-5_QPIG_15_1.mp4"
    "25-7_APXG_15_1.mp4"
    "25-24_QLD_15_1.mp4"
    "25-31_UNKNOWN_15_1.mp4"
    "25-34_QASD_15_1.mp4"
)

declare -a TB_NEGATIVE=(
    "25-1_QAIG_15_1.mp4"
    "25-4_QLD_15_1.mp4"
    "25-12_QPID_15_1.mp4"
    "25-14_QLG_15_2.mp4"
    "25-9_APXD_15_1.mp4"
)

# Process TB POSITIVE videos
echo "========================================="
echo "TB POSITIVE PATIENTS"
echo "========================================="
echo ""

video_count=1
for video in "${TB_POSITIVE[@]}"; do
    echo "[$video_count/10] Processing TB+ patient: $video"
    python ultr_ai/plot/visualize_attention_minimal.py \
        --video "$VIDEO_DIR/$video" \
        --batch-mode \
        --model-base-path "$MODEL_BASE_PATH" \
        --folds 0 1 2 3 4 \
        --output "$OUTPUT_DIR" \
        --csv "$CSV_FILE"
    echo ""
    ((video_count++))
done

# Process TB NEGATIVE videos
echo "========================================="
echo "TB NEGATIVE PATIENTS"
echo "========================================="
echo ""

for video in "${TB_NEGATIVE[@]}"; do
    echo "[$video_count/10] Processing TB- patient: $video"
    python ultr_ai/plot/visualize_attention_minimal.py \
        --video "$VIDEO_DIR/$video" \
        --batch-mode \
        --model-base-path "$MODEL_BASE_PATH" \
        --folds 0 1 2 3 4 \
        --output "$OUTPUT_DIR" \
        --csv "$CSV_FILE"
    echo ""
    ((video_count++))
done

echo ""
echo "========================================="
echo "ALL PROCESSING COMPLETE!"
echo "========================================="
echo ""
echo "Visualizations saved to: $OUTPUT_DIR"
echo "CSV metrics saved to: $CSV_FILE"
echo ""
echo "Summary:"
echo "  - Processed 10 videos (5 TB+, 5 TB-)"
echo "  - Each video processed with 5 folds"
echo "  - Total: 50 visualizations + 1 CSV file"
echo ""
echo "Efficiency improvement:"
echo "  - Old method: Load each video 5 times = 50 video loads"
echo "  - New method: Load each video 1 time = 10 video loads"
echo "  - Speed improvement: ~5x faster!"
echo ""
echo "Output files:"
echo "  - Images: <patient>_fold0.png, <patient>_fold1.png, etc."
echo "  - CSV: attention_metrics_all_folds.csv"
echo ""
echo "CSV contains:"
echo "  - Top-3 frame indices and scores for each video/fold"
echo "  - Attention statistics (mean, std, min, max)"
echo "  - Easy comparison across folds and patients"
echo ""
