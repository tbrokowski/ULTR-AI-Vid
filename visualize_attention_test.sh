#!/bin/bash
# Script to visualize attention weights for TB positive and negative videos
# This helps verify that the attention mechanism is learning discriminative patterns

# Model checkpoint path
MODEL_PATH="/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/attention_pool_extra3_full_train/fold0/checkpoint_best.pth"

# Output directory
OUTPUT_DIR="./visualization_outputs/attention_weights_test"
VIDEO_DIR="/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Visualizing Attention Weights"
echo "========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# TB POSITIVE VIDEOS
echo "========================================="
echo "TB POSITIVE PATIENTS"
echo "========================================="
echo ""

echo "[1/10] Patient 25-5 (TB+): 25-5_QPIG_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-5_QPIG_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-5_TB_positive.png"

echo ""
echo "[2/10] Patient 25-7 (TB+): 25-7_APXG_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-7_APXG_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-7_TB_positive.png"

echo ""
echo "[3/10] Patient 25-24 (TB+): 25-24_QLD_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-24_QLD_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-24_TB_positive.png"

echo ""
echo "[4/10] Patient 25-31 (TB+): 25-31_UNKNOWN_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-31_UNKNOWN_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-31_TB_positive.png"

echo ""
echo "[5/10] Patient 25-34 (TB+): 25-34_QASD_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-34_QASD_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-34_TB_positive.png"

echo ""
echo "========================================="
echo "TB NEGATIVE PATIENTS"
echo "========================================="
echo ""

echo "[6/10] Patient 25-1 (TB-): 25-1_QAIG_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-1_QAIG_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-1_TB_negative.png"

echo ""
echo "[7/10] Patient 25-4 (TB-): 25-4_QLD_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-4_QLD_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-4_TB_negative.png"

echo ""
echo "[8/10] Patient 25-12 (TB-): 25-12_QPID_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-12_QPID_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-12_TB_negative.png"

echo ""
echo "[9/10] Patient 25-14 (TB-): 25-14_QLG_15_2.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-14_QLG_15_2.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-14_TB_negative.png"

echo ""
echo "[10/10] Patient 25-9 (TB-): 25-9_APXD_15_1.mp4"
python visualize_attention_minimal.py \
    --video "$VIDEO_DIR/25-9_APXD_15_1.mp4" \
    --checkpoint "$MODEL_PATH" \
    --output "$OUTPUT_DIR/25-9_TB_negative.png"

echo ""
echo "========================================="
echo "DONE!"
echo "========================================="
echo ""
echo "Visualizations saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
echo "  - 5 TB Positive patients"
echo "  - 5 TB Negative patients"
echo ""
echo "Check the output directory for:"
echo "  - Attention weight plots"
echo "  - Frame importance visualizations"
echo "  - Comparison of attention patterns between TB+ and TB-"
echo ""
