#!/usr/bin/env bash
# visualize_all.sh - Integration script for comprehensive results analysis
# Run this after all evaluations complete on the cluster

set -euo pipefail

# Configuration - Update these paths for your cluster setup
RESULTS_BASE="${RESULTS_BASE:-/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results}"
OUTPUT_DIR="${OUTPUT_DIR:-./visualization_outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Model types - Update this list based on your actual experiments
# These should match the directory names in $RESULTS_BASE
MODEL_TYPES=(
    "inception3d"
    "LeViT-RL"
    "LeVit-Attention"
    "3dcnn"
    "cnnlstm" 
    "attention_pool_noInitWeights"
    # "attention_pool"
    # "attention_pool_extra"
    # "attention_pool_extra1"
    # "attention_pool_extra2"
    "attention_pool_extra3"
    "mean_pool_extra3"
    "singletask"
    "uniform_extra3"
    "no_rl_full_train"
    "r2plus1d"
    # "Efficientnet-RL"
    "original_noInitWeights"
    "original"
    "original_test"
    "vivit"
    
)

echo "==============================================="
echo "COMPREHENSIVE RESULTS ANALYSIS"
echo "==============================================="
echo "Results directory: $RESULTS_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Models to analyze: ${MODEL_TYPES[*]}"
echo "==============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run comprehensive analysis
echo "Running comprehensive visualization and analysis..."
"$PYTHON_BIN" visualize_results.py \
    --results_dir "$RESULTS_BASE" \
    --output_dir "$OUTPUT_DIR" \
    --model_types "${MODEL_TYPES[@]}" \
    --num_folds 5 \
    --split test \
    --top_n 6

echo ""
echo "==============================================="
echo "ANALYSIS COMPLETE"
echo "==============================================="
echo "All results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  📊 ROC curves: roc_curves_test_with_ci.pdf"
echo "  📈 PR curves: pr_curves_test_with_ci.pdf"
echo "  📋 Metrics table: metrics_comparison_test.csv"
echo "  📊 Performance vs complexity: performance_vs_complexity_test.pdf"
echo "  📈 Statistical tests: statistical_significance_tests.csv"
echo "  🧠 Attention analysis: attention_analysis_*.pdf"
echo "  🏥 Site analysis: site_analysis_*.pdf"
echo "  📄 Summary: analysis_summary.json"
echo "==============================================="
