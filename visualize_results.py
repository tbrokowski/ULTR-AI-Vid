#!/usr/bin/env python3
"""
Comprehensive Model Results Analysis and Visualization Script
For Nature Digital Medicine Quality Results
"""

import os
import sys
import argparse
import glob
import json
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import h5py

from scipy import stats
from scipy.stats import ttest_rel
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    accuracy_score, balanced_accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

def setup_publication_style(font_size=14, font_family='STIXGeneral'):
    """Set up publication-quality matplotlib styling."""
    plt.style.use('default')  # Reset to default first
    
    # Font and text settings
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size + 2
    plt.rcParams['axes.titlesize'] = font_size + 4
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.titlesize'] = font_size + 6
    
    # Weight settings
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'
    
    # Colors
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    
    # Grid and spines
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.axisbelow'] = True
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.transparent'] = False

# Color schemes for different model types
MODEL_COLORS = {
    'original': '#2E86AB',
    'attention_pool': '#A23B72', 
    'mean_pool': '#F18F01',
    '3d_cnn': '#C73E1D',
    'cnn_lstm': '#7209B7',
    'video_transformer': '#06D6A0',
    'r2plus1d': '#F72585',
    'inception3d': '#4361EE',
    'single_task': '#FB8500',
    'uniform': '#8ECAE6',
    'no_rl_full_train': '#219EBC',
    'Efficientnet_RL': '#023047',
    'LeViT_Attention': '#FFB3BA',
    'LeViT_RL': '#BAFFC9'
}

# Professional color palette for top models
TOP_MODELS_COLORS = ['#36454F', '#00CCFE', '#0247FF', '#0018A7', '#8B0000', '#4B0082']

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_model_results(results_base_dir: str, model_types: List[str], 
                      num_folds: int = 5) -> Dict[str, Dict]:
    """
    Load results for all models and folds from the evaluation pipeline.
    
    Expects cluster structure:
        {results_base_dir}/{model_type}/
            eval_results/
                {split}_full_model_fold{N}_patients.csv
                {split}_full_model_fold{N}_sites.csv
                {split}_full_model_fold{N}_metrics.json
                {split}_full_model_fold{N}_complex_data.h5
            fold0/, fold1/, ... (contain checkpoints)
    
    Returns:
        Dictionary with structure: {model_type: {fold: {split: data}}}
    """
    print("="*70)
    print("LOADING MODEL RESULTS")
    print("="*70)
    
    all_results = {}
    
    for model_type in model_types:
        print(f"\nLoading results for model: {model_type}")
        model_results = {}
        
        # Determine the eval_results directory for this model
        # Try different possible paths
        eval_results_candidates = [
            os.path.join(results_base_dir, model_type, "eval_results"),
            os.path.join(results_base_dir, model_type),  # Fallback if eval_results doesn't exist
        ]
        
        eval_results_dir = None
        for candidate in eval_results_candidates:
            if os.path.exists(candidate):
                eval_results_dir = candidate
                break
        
        if eval_results_dir is None:
            print(f"  ✗ No eval_results directory found for {model_type}")
            print(f"    Tried: {eval_results_candidates}")
            continue
        
        print(f"  Using eval_results directory: {eval_results_dir}")
        
        # Load results for each fold
        for fold in range(num_folds):
            fold_results = {}
            
            # Load data for each split
            for split in ['train', 'val', 'test']:
                try:
                    # Cluster naming convention: {split}_full_model_fold{N}_{type}.{ext}
                    patients_csv = os.path.join(eval_results_dir, f"{split}_full_model_fold{fold}_patients.csv")
                    sites_csv = os.path.join(eval_results_dir, f"{split}_full_model_fold{fold}_sites.csv")
                    metrics_json = os.path.join(eval_results_dir, f"{split}_full_model_fold{fold}_metrics.json")
                    complex_hdf5 = os.path.join(eval_results_dir, f"{split}_full_model_fold{fold}_complex_data.h5")
                    
                    split_data = {}
                    files_found = 0
                    
                    # Load patient data
                    if os.path.exists(patients_csv):
                        split_data['patients'] = pd.read_csv(patients_csv)
                        files_found += 1
                    
                    # Load site data  
                    if os.path.exists(sites_csv):
                        split_data['sites'] = pd.read_csv(sites_csv)
                        files_found += 1
                    
                    # Load metrics
                    if os.path.exists(metrics_json):
                        with open(metrics_json, 'r') as f:
                            split_data['metrics'] = json.load(f)
                        files_found += 1
                    
                    # Load complex data
                    if os.path.exists(complex_hdf5):
                        split_data['complex'] = load_complex_data(complex_hdf5)
                        files_found += 1
                    
                    if split_data:
                        fold_results[split] = split_data
                    else:
                        print(f"    Warning: No files found for {split} fold {fold} (expected in {eval_results_dir})")
                        
                except Exception as e:
                    print(f"    Warning: Error loading {split} data for fold {fold}: {e}")
            
            if fold_results:
                model_results[fold] = fold_results
                splits_found = list(fold_results.keys())
                print(f"  ✓ Loaded fold {fold} ({len(splits_found)} splits: {', '.join(splits_found)})")
        
        if model_results:
            all_results[model_type] = model_results
            print(f"✓ Successfully loaded {model_type} ({len(model_results)} folds)")
        else:
            print(f"✗ No data found for {model_type}")
    
    return all_results

def load_complex_data(hdf5_path: str) -> Dict:
    """Load complex data from HDF5 file."""
    complex_data = {
        'patient_features': {},
        'site_features': {},
        'mil_attention': {},
        'task_logits': {},
        'pathology_scores': {}
    }
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Load patient features
            if 'patient_features' in f:
                for patient_id in f['patient_features'].keys():
                    data = f['patient_features'][patient_id]
                    # Handle both scalar and array data
                    if data.shape == ():
                        complex_data['patient_features'][patient_id] = np.array([data[()]])
                    else:
                        complex_data['patient_features'][patient_id] = data[:]
            
            # Load site features
            if 'site_features' in f:
                for site_key in f['site_features'].keys():
                    data = f['site_features'][site_key]
                    if data.shape == ():
                        complex_data['site_features'][site_key] = np.array([data[()]])
                    else:
                        complex_data['site_features'][site_key] = data[:]
            
            # Load MIL attention
            if 'mil_attention' in f:
                for patient_id in f['mil_attention'].keys():
                    data = f['mil_attention'][patient_id]
                    if data.shape == ():
                        complex_data['mil_attention'][patient_id] = np.array([data[()]])
                    else:
                        complex_data['mil_attention'][patient_id] = data[:]
            
            # Load task logits
            if 'task_logits' in f:
                for task_name in f['task_logits'].keys():
                    complex_data['task_logits'][task_name] = {}
                    for patient_id in f['task_logits'][task_name].keys():
                        data = f['task_logits'][task_name][patient_id]
                        if data.shape == ():
                            complex_data['task_logits'][task_name][patient_id] = np.array([data[()]])
                        else:
                            complex_data['task_logits'][task_name][patient_id] = data[:]
            
            # Load pathology scores
            if 'pathology_scores' in f:
                for patient_id in f['pathology_scores'].keys():
                    data = f['pathology_scores'][patient_id]
                    if data.shape == ():
                        complex_data['pathology_scores'][patient_id] = np.array([data[()]])
                    else:
                        complex_data['pathology_scores'][patient_id] = data[:]
                    
    except Exception as e:
        print(f"Warning: Could not load complex data from {hdf5_path}: {e}")
    
    return complex_data

# =============================================================================
# PERFORMANCE METRICS CALCULATION
# =============================================================================

def calculate_comprehensive_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                                  y_pred: Optional[np.ndarray] = None,
                                  threshold: float = 0.5) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if y_pred is None:
        y_pred = (y_prob > threshold).astype(int)
    
    # Basic metrics
    metrics = {}
    metrics['auc'] = roc_auc_score(y_true, y_prob)
    metrics['auprc'] = average_precision_score(y_true, y_prob)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Likelihood ratios
    metrics['plr'] = metrics['sensitivity'] / (1 - metrics['specificity']) if metrics['specificity'] < 1 else float('inf')
    metrics['nlr'] = (1 - metrics['sensitivity']) / metrics['specificity'] if metrics['specificity'] > 0 else float('inf')
    
    # Performance at specific operating points
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Sensitivity at 90% specificity
    spec_90_idx = np.where(1 - fpr >= 0.9)[0]
    if len(spec_90_idx) > 0:
        metrics['sens_at_90_spec'] = tpr[spec_90_idx[-1]]
    else:
        metrics['sens_at_90_spec'] = 0
    
    # Sensitivity at 70% specificity  
    spec_70_idx = np.where(1 - fpr >= 0.7)[0]
    if len(spec_70_idx) > 0:
        metrics['sens_at_70_spec'] = tpr[spec_70_idx[-1]]
    else:
        metrics['sens_at_70_spec'] = 0
        
    return metrics

def aggregate_cross_fold_metrics(all_results: Dict, model_types: List[str], 
                                split: str = 'test') -> pd.DataFrame:
    """Aggregate metrics across folds for all models."""
    print(f"\n{'='*70}")
    print(f"CALCULATING CROSS-FOLD METRICS ({split.upper()} SPLIT)")
    print(f"{'='*70}")
    
    aggregated_results = []
    
    for model_type in model_types:
        if model_type not in all_results:
            print(f"Warning: No results found for {model_type}")
            continue
            
        print(f"\nProcessing {model_type}...")
        
        # Collect metrics from all folds
        fold_metrics = []
        
        for fold in all_results[model_type].keys():
            if split not in all_results[model_type][fold]:
                continue
                
            fold_data = all_results[model_type][fold][split]
            
            if 'patients' not in fold_data:
                continue
                
            patients_df = fold_data['patients']
            
            # Get TB predictions
            if 'tb_label' in patients_df.columns and 'tb_prob' in patients_df.columns:
                # Filter valid labels
                valid_mask = patients_df['tb_label'] >= 0
                if valid_mask.sum() == 0:
                    continue
                    
                y_true = patients_df.loc[valid_mask, 'tb_label'].values
                y_prob = patients_df.loc[valid_mask, 'tb_prob'].values
                y_pred = patients_df.loc[valid_mask, 'tb_pred'].values if 'tb_pred' in patients_df.columns else None
                
                metrics = calculate_comprehensive_metrics(y_true, y_prob, y_pred)
                metrics['fold'] = fold
                metrics['n_patients'] = len(y_true)
                fold_metrics.append(metrics)
        
        if not fold_metrics:
            print(f"  No valid data found for {model_type}")
            continue
            
        # Calculate statistics across folds
        metrics_df = pd.DataFrame(fold_metrics)
        
        for metric in metrics_df.columns:
            if metric in ['fold', 'n_patients']:
                continue
                
            values = metrics_df[metric].values
            if len(values) == 0:
                continue
                
            # Calculate mean and confidence interval
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            
            # 95% CI using t-distribution
            if len(values) > 1:
                ci_lower, ci_upper = stats.t.interval(
                    0.95, len(values)-1, loc=mean_val, 
                    scale=stats.sem(values)
                )
            else:
                ci_lower = ci_upper = mean_val
            
            aggregated_results.append({
                'model': model_type,
                'metric': metric,
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_folds': len(values),
                'individual_values': values.tolist()
            })
        
        print(f"  ✓ Processed {len(fold_metrics)} folds")
    
    return pd.DataFrame(aggregated_results)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def perform_statistical_tests(all_results: Dict, model_types: List[str], 
                             split: str = 'test') -> Dict:
    """Perform statistical significance testing between models."""
    print(f"\n{'='*70}")
    print(f"STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*70}")
    
    # Collect AUC values for each model
    model_aucs = {}
    
    for model_type in model_types:
        if model_type not in all_results:
            continue
            
        aucs = []
        for fold in all_results[model_type].keys():
            if split not in all_results[model_type][fold]:
                continue
                
            fold_data = all_results[model_type][fold][split]
            if 'patients' not in fold_data:
                continue
                
            patients_df = fold_data['patients']
            if 'tb_label' in patients_df.columns and 'tb_prob' in patients_df.columns:
                valid_mask = patients_df['tb_label'] >= 0
                if valid_mask.sum() > 0:
                    y_true = patients_df.loc[valid_mask, 'tb_label'].values
                    y_prob = patients_df.loc[valid_mask, 'tb_prob'].values
                    auc_score = roc_auc_score(y_true, y_prob)
                    aucs.append(auc_score)
        
        if aucs:
            model_aucs[model_type] = aucs
    
    # Perform pairwise t-tests
    test_results = []
    model_names = list(model_aucs.keys())
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            aucs1, aucs2 = model_aucs[model1], model_aucs[model2]
            
            if len(aucs1) == len(aucs2) and len(aucs1) > 1:
                # Paired t-test
                t_stat, p_value = ttest_rel(aucs1, aucs2)
                mean_diff = np.mean(aucs1) - np.mean(aucs2)
                
                # Cohen's d for paired samples
                diff = np.array(aucs1) - np.array(aucs2)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
                
                test_results.append({
                    'model1': model1,
                    'model2': model2,
                    'mean_diff': mean_diff,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'model1_aucs': aucs1,
                    'model2_aucs': aucs2
                })
    
    # Apply Bonferroni correction
    if test_results:
        p_values = [r['p_value'] for r in test_results]
        bonferroni_alpha = 0.05 / len(p_values)
        
        for result in test_results:
            result['bonferroni_significant'] = result['p_value'] < bonferroni_alpha
            result['bonferroni_alpha'] = bonferroni_alpha
    
    return {
        'model_aucs': model_aucs,
        'pairwise_tests': test_results,
        'bonferroni_alpha': bonferroni_alpha if test_results else None
    }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_roc_curves_with_ci(all_results: Dict, model_types: List[str], 
                             output_dir: str, split: str = 'test',
                             top_n: Optional[int] = None, suffix: str = "") -> None:
    """Create publication-quality ROC curves with confidence intervals."""
    print(f"\n{'='*50}")
    print(f"CREATING ROC CURVES ({split.upper()})")
    print(f"{'='*50}")
    
    # If top_n specified, select top performing models
    if top_n is not None:
        metrics_df = aggregate_cross_fold_metrics(all_results, model_types, split)
        auc_metrics = metrics_df[metrics_df['metric'] == 'auc'].sort_values('mean', ascending=False)
        model_types = auc_metrics.head(top_n)['model'].tolist()
        print(f"Selected top {top_n} models: {model_types}")
    
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = TOP_MODELS_COLORS[:len(model_types)] if len(model_types) <= len(TOP_MODELS_COLORS) else \
             [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(model_types)]
    
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, model_type in enumerate(model_types):
        if model_type not in all_results:
            continue
            
        # Collect ROC curves from all folds
        tpr_list = []
        aucs = []
        
        for fold in all_results[model_type].keys():
            if split not in all_results[model_type][fold]:
                continue
                
            fold_data = all_results[model_type][fold][split]
            if 'patients' not in fold_data:
                continue
                
            patients_df = fold_data['patients']
            
            if 'tb_label' in patients_df.columns and 'tb_prob' in patients_df.columns:
                valid_mask = patients_df['tb_label'] >= 0
                if valid_mask.sum() == 0:
                    continue
                    
                y_true = patients_df.loc[valid_mask, 'tb_label'].values
                y_prob = patients_df.loc[valid_mask, 'tb_prob'].values
                
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                
                tpr_list.append(tpr_interp)
                aucs.append(roc_auc_score(y_true, y_prob))
        
        if not tpr_list:
            continue
            
        # Calculate mean and CI
        tpr_array = np.array(tpr_list)
        mean_tpr = np.mean(tpr_array, axis=0)
        mean_tpr[-1] = 1.0
        
        std_tpr = np.std(tpr_array, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        mean_auc = np.mean(aucs)
        ci_lower_auc = np.percentile(aucs, 2.5)
        ci_upper_auc = np.percentile(aucs, 97.5)
        
        # Plot mean curve
        label = f'{model_type.replace("_", " ").title()} (AUC = {mean_auc:.3f}, 95% CI [{ci_lower_auc:.3f}-{ci_upper_auc:.3f}])'
        ax.plot(mean_fpr, mean_tpr, color=colors[i], lw=3, label=label)
        
        # Fill confidence interval
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i], alpha=0.2)
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)
    
    # Add WHO guidelines highlight
    who_patch = patches.Rectangle((0, 0.9), 0.3, 0.1, linewidth=0, 
                                 facecolor='yellow', alpha=0.3, 
                                 label='WHO Requirements (90% Sens, 70% Spec)')
    ax.add_patch(who_patch)
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Reverse x-axis to show specificity
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    xtick_labels = ['1.0', '0.8', '0.6', '0.4', '0.2', '0.0']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    
    ax.set_xlabel('Specificity', fontsize=18, fontweight='bold')
    ax.set_ylabel('Sensitivity (Recall)', fontsize=18, fontweight='bold')
    title_suffix = suffix.replace("_", " ").title() if suffix else ""
    ax.set_title(f'ROC Curves with 95% Confidence Intervals ({split.title()} Set){title_suffix}', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'roc_curves_{split}_with_ci{suffix}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curves saved to {output_path}")

def create_pr_curves_with_ci(all_results: Dict, model_types: List[str], 
                           output_dir: str, split: str = 'test',
                           top_n: Optional[int] = None, suffix: str = "") -> None:
    """Create precision-recall curves with confidence intervals."""
    print(f"\nCreating PR curves ({split})...")
    
    if top_n is not None:
        metrics_df = aggregate_cross_fold_metrics(all_results, model_types, split)
        auprc_metrics = metrics_df[metrics_df['metric'] == 'auprc'].sort_values('mean', ascending=False)
        model_types = auprc_metrics.head(top_n)['model'].tolist()
    
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = TOP_MODELS_COLORS[:len(model_types)] if len(model_types) <= len(TOP_MODELS_COLORS) else \
             [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(model_types)]
    
    mean_recall = np.linspace(0, 1, 100)
    
    for i, model_type in enumerate(model_types):
        if model_type not in all_results:
            continue
            
        precision_list = []
        auprcs = []
        
        for fold in all_results[model_type].keys():
            if split not in all_results[model_type][fold]:
                continue
                
            fold_data = all_results[model_type][fold][split]
            if 'patients' not in fold_data:
                continue
                
            patients_df = fold_data['patients']
            
            if 'tb_label' in patients_df.columns and 'tb_prob' in patients_df.columns:
                valid_mask = patients_df['tb_label'] >= 0
                if valid_mask.sum() == 0:
                    continue
                    
                y_true = patients_df.loc[valid_mask, 'tb_label'].values
                y_prob = patients_df.loc[valid_mask, 'tb_prob'].values
                
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                precision_interp = np.interp(mean_recall[::-1], recall[::-1], precision[::-1])[::-1]
                
                precision_list.append(precision_interp)
                auprcs.append(average_precision_score(y_true, y_prob))
        
        if not precision_list:
            continue
            
        precision_array = np.array(precision_list)
        mean_precision = np.mean(precision_array, axis=0)
        std_precision = np.std(precision_array, axis=0)
        
        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = np.maximum(mean_precision - std_precision, 0)
        
        mean_auprc = np.mean(auprcs)
        ci_lower_auprc = np.percentile(auprcs, 2.5)
        ci_upper_auprc = np.percentile(auprcs, 97.5)
        
        label = f'{model_type.replace("_", " ").title()} (AUPRC = {mean_auprc:.3f}, 95% CI [{ci_lower_auprc:.3f}-{ci_upper_auprc:.3f}])'
        ax.plot(mean_recall, mean_precision, color=colors[i], lw=3, label=label)
        ax.fill_between(mean_recall, precision_lower, precision_upper, color=colors[i], alpha=0.2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=18, fontweight='bold')
    ax.set_title(f'Precision-Recall Curves with 95% Confidence Intervals ({split.title()} Set)', 
                fontsize=20, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'pr_curves_{split}_with_ci.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ PR curves saved to {output_path}")

def create_metrics_comparison_table(metrics_df: pd.DataFrame, output_dir: str,
                                   split: str = 'test') -> None:
    """Create comprehensive metrics comparison table."""
    print(f"\nCreating metrics comparison table ({split})...")
    
    # Pivot to get metrics as columns
    table_data = []
    
    models = metrics_df['model'].unique()
    key_metrics = ['auc', 'auprc', 'accuracy', 'sensitivity', 'specificity', 
                  'f1', 'ppv', 'npv', 'sens_at_90_spec', 'sens_at_70_spec']
    
    for model in models:
        row = {'Model': model.replace('_', ' ').title()}
        
        for metric in key_metrics:
            metric_data = metrics_df[(metrics_df['model'] == model) & 
                                   (metrics_df['metric'] == metric)]
            
            if len(metric_data) > 0:
                mean_val = metric_data['mean'].iloc[0]
                ci_lower = metric_data['ci_lower'].iloc[0]
                ci_upper = metric_data['ci_upper'].iloc[0]
                
                if metric in ['auc', 'auprc']:
                    row[metric.upper()] = f"{mean_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
                else:
                    row[metric.replace('_', ' ').title()] = f"{mean_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
            else:
                if metric in ['auc', 'auprc']:
                    row[metric.upper()] = "N/A"
                else:
                    row[metric.replace('_', ' ').title()] = "N/A"
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'metrics_comparison_{split}.csv')
    table_df.to_csv(csv_path, index=False)
    
    # Create formatted table plot
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(20, len(models) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    
    plt.title(f'Model Performance Comparison ({split.title()} Set)\nValues shown as Mean (95% CI)', 
             fontsize=16, fontweight='bold', pad=20)
    
    table_path = os.path.join(output_dir, f'metrics_table_{split}.pdf')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.savefig(table_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics table saved to {csv_path} and {table_path}")

def create_performance_vs_complexity_plot(all_results: Dict, model_types: List[str],
                                        output_dir: str, split: str = 'test') -> None:
    """Create performance vs model complexity/parameter count plot."""
    print(f"\nCreating performance vs complexity plot ({split})...")
    
    # Estimate model complexity (you may need to adjust these based on actual architectures)
    model_complexity = {
        'original': 100,  # Million parameters (example)
        'attention_pool': 95,
        'mean_pool': 85,
        '3d_cnn': 150,
        'cnn_lstm': 120,
        'video_transformer': 200,
        'r2plus1d': 180,
        'inception3d': 220,
        'single_task': 50,
        'uniform': 90,
        'no_rl_full_train': 95,
        'Efficientnet_RL': 70,
        'LeViT_Attention': 60,
        'LeViT_RL': 65
    }
    
    # Get AUC values
    metrics_df = aggregate_cross_fold_metrics(all_results, model_types, split)
    auc_data = metrics_df[metrics_df['metric'] == 'auc']
    
    complexity_values = []
    auc_means = []
    auc_errors = []
    labels = []
    
    for _, row in auc_data.iterrows():
        model = row['model']
        if model in model_complexity:
            complexity_values.append(model_complexity[model])
            auc_means.append(row['mean'])
            auc_errors.append((row['ci_upper'] - row['ci_lower']) / 2)
            labels.append(model.replace('_', ' ').title())
    
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with error bars
    colors = [MODEL_COLORS.get(label.lower().replace(' ', '_'), 'blue') for label in labels]
    
    for i, (x, y, yerr, label, color) in enumerate(zip(complexity_values, auc_means, auc_errors, labels, colors)):
        ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=10, color=color, 
                   label=label, capsize=5, capthick=2)
    
    ax.set_xlabel('Model Complexity (Million Parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Performance vs Model Complexity ({split.title()} Set)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'performance_vs_complexity_{split}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Performance vs complexity plot saved to {output_path}")

# =============================================================================
# PATHOLOGY ENSEMBLE MODELING
# =============================================================================

def train_pathology_ml_models(all_results: Dict, top_models: List[str],
                             output_dir: str, split: str = 'test') -> Dict:
    """
    Train classical ML models using neural network pathology predictions as features.
    
    For each pathology type, train ML models where:
    - Input: Neural network pathology predictions for that site
    - Target: Ground truth binary label for that pathology
    
    This demonstrates the clinical value of interpretable pathology outputs.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING PATHOLOGY-LEVEL ML MODELS ({split.upper()})")
    print(f"{'='*70}")
    
    pathology_names = ['a_lines', 'large_consolidation', 'pleural_effusion', 'other_pathology']
    ml_algorithms = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'XGBoost': None  # Will try to import XGBoost if available
    }
    
    # Try to import XGBoost
    try:
        from xgboost import XGBClassifier
        ml_algorithms['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
    except ImportError:
        print("  XGBoost not available, skipping...")
        del ml_algorithms['XGBoost']
    
    ensemble_results = {}
    
    for model_type in top_models:
        if model_type not in all_results:
            continue
            
        print(f"\nProcessing {model_type}...")
        model_ensemble_results = {}
        
        # Collect site-level data across all folds
        all_site_data = []
        
        for fold in all_results[model_type].keys():
            # Get training data
            if 'train' in all_results[model_type][fold] and 'sites' in all_results[model_type][fold]['train']:
                train_sites = all_results[model_type][fold]['train']['sites'].copy()
                train_sites['fold'] = fold
                train_sites['data_split'] = 'train'
                all_site_data.append(train_sites)
            
            # Get test data
            if split in all_results[model_type][fold] and 'sites' in all_results[model_type][fold][split]:
                test_sites = all_results[model_type][fold][split]['sites'].copy()
                test_sites['fold'] = fold
                test_sites['data_split'] = split
                all_site_data.append(test_sites)
        
        if not all_site_data:
            print(f"  No valid data found for {model_type}")
            continue
            
        combined_sites = pd.concat(all_site_data, ignore_index=True)
        print(f"  Combined {len(combined_sites)} sites across all folds")
        
        # Train ML models for each pathology
        for pathology in pathology_names:
            print(f"    Training models for {pathology}...")
            
            # Check if we have the required columns
            finding_col = f'{pathology}_finding'
            prob_cols = [f'{pathology}_prob', f'{pathology}_logit']
            
            # Find available prediction columns
            available_pred_cols = [col for col in prob_cols if col in combined_sites.columns]
            
            if finding_col not in combined_sites.columns or not available_pred_cols:
                print(f"      Missing data for {pathology}, skipping...")
                continue
            
            # Prepare features and targets
            feature_cols = available_pred_cols.copy()
            
            # Add additional features if available
            additional_features = ['site_index', 'mil_attention']
            for feat in additional_features:
                if feat in combined_sites.columns:
                    feature_cols.append(feat)
            
            # Filter valid samples
            valid_mask = (combined_sites[finding_col] >= 0) & \
                        (~combined_sites[available_pred_cols].isna().any(axis=1))
            
            if valid_mask.sum() < 50:  # Need sufficient samples
                print(f"      Insufficient valid samples for {pathology} ({valid_mask.sum()}), skipping...")
                continue
            
            valid_data = combined_sites[valid_mask].copy()
            
            # Prepare features and targets
            X = valid_data[feature_cols].values
            y = valid_data[finding_col].values.astype(int)
            
            print(f"      Training on {len(X)} samples, {feature_cols}")
            print(f"      Class distribution: {np.bincount(y)}")
            
            # Perform cross-validation by fold
            fold_results = {}
            unique_folds = sorted(valid_data['fold'].unique())
            
            if len(unique_folds) < 2:
                print(f"      Insufficient folds for CV, skipping...")
                continue
            
            for ml_name, ml_model in ml_algorithms.items():
                if ml_model is None:
                    continue
                    
                print(f"        Training {ml_name}...")
                
                fold_scores = []
                fold_predictions = []
                
                for test_fold in unique_folds:
                    # Split by fold
                    train_mask = valid_data['fold'] != test_fold
                    test_mask = valid_data['fold'] == test_fold
                    
                    if train_mask.sum() == 0 or test_mask.sum() == 0:
                        continue
                    
                    X_train, X_test = X[train_mask], X[test_mask]
                    y_train, y_test = y[train_mask], y[test_mask]
                    
                    # Skip if no positive samples in training
                    if len(np.unique(y_train)) < 2:
                        continue
                    
                    # Train model
                    model_copy = type(ml_model)(**ml_model.get_params())
                    model_copy.fit(X_train, y_train)
                    
                    # Predict
                    if hasattr(model_copy, 'predict_proba'):
                        y_pred_proba = model_copy.predict_proba(X_test)
                        if y_pred_proba.shape[1] > 1:
                            y_pred_proba = y_pred_proba[:, 1]
                        else:
                            y_pred_proba = y_pred_proba[:, 0]
                    else:
                        y_pred_proba = model_copy.decision_function(X_test)
                    
                    # Calculate metrics
                    try:
                        fold_auc = roc_auc_score(y_test, y_pred_proba)
                        fold_scores.append(fold_auc)
                        
                        fold_predictions.append({
                            'fold': test_fold,
                            'y_true': y_test,
                            'y_pred_proba': y_pred_proba,
                            'auc': fold_auc
                        })
                    except ValueError as e:
                        print(f"          Error calculating AUC for fold {test_fold}: {e}")
                        continue
                
                if fold_scores:
                    mean_auc = np.mean(fold_scores)
                    std_auc = np.std(fold_scores, ddof=1) if len(fold_scores) > 1 else 0
                    
                    fold_results[ml_name] = {
                        'mean_auc': mean_auc,
                        'std_auc': std_auc,
                        'fold_scores': fold_scores,
                        'fold_predictions': fold_predictions,
                        'feature_names': feature_cols,
                        'n_samples': len(X),
                        'class_distribution': np.bincount(y).tolist()
                    }
                    
                    print(f"          {ml_name}: AUC = {mean_auc:.3f} ± {std_auc:.3f}")
            
            if fold_results:
                model_ensemble_results[pathology] = fold_results
        
        if model_ensemble_results:
            ensemble_results[model_type] = model_ensemble_results
            print(f"  ✓ Completed ensemble training for {model_type}")
    
    # Save results
    ensemble_summary_path = os.path.join(output_dir, f'pathology_ensemble_results_{split}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    ensemble_results_serializable = {}
    for model_type, model_results in ensemble_results.items():
        ensemble_results_serializable[model_type] = {}
        for pathology, pathology_results in model_results.items():
            ensemble_results_serializable[model_type][pathology] = {}
            for ml_name, ml_results in pathology_results.items():
                serializable_results = ml_results.copy()
                # Remove predictions to avoid large file size
                if 'fold_predictions' in serializable_results:
                    del serializable_results['fold_predictions']
                ensemble_results_serializable[model_type][pathology][ml_name] = serializable_results
    
    with open(ensemble_summary_path, 'w') as f:
        json.dump(ensemble_results_serializable, f, indent=2)
    
    print(f"\n✓ Ensemble results saved to {ensemble_summary_path}")
    
    return ensemble_results

def compare_neural_vs_ml_pathology_predictions(ensemble_results: Dict, output_dir: str,
                                             split: str = 'test') -> None:
    """
    Create comparison plots between neural network and ML model pathology predictions.
    """
    print(f"\nCreating neural vs ML pathology prediction comparison ({split})...")
    
    pathology_names = ['a_lines', 'large_consolidation', 'pleural_effusion', 'other_pathology']
    
    # Collect all results for comparison
    comparison_data = []
    
    for model_type, model_results in ensemble_results.items():
        for pathology in pathology_names:
            if pathology not in model_results:
                continue
                
            pathology_results = model_results[pathology]
            
            # Neural network baseline (using the fact that ML models were trained on NN predictions)
            # We'll estimate this from the feature importance or use a simple baseline
            
            for ml_name, ml_results in pathology_results.items():
                comparison_data.append({
                    'model_type': model_type,
                    'pathology': pathology,
                    'ml_algorithm': ml_name,
                    'ml_auc': ml_results['mean_auc'],
                    'ml_auc_std': ml_results['std_auc'],
                    'n_samples': ml_results['n_samples']
                })
    
    if not comparison_data:
        print("  No comparison data available")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualization
    setup_publication_style()
    
    # Group by pathology for separate subplots
    unique_pathologies = comparison_df['pathology'].unique()
    n_pathologies = len(unique_pathologies)
    
    if n_pathologies == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, pathology in enumerate(unique_pathologies[:4]):  # Maximum 4 pathologies
        if i >= len(axes):
            break
            
        ax = axes[i]
        pathology_data = comparison_df[comparison_df['pathology'] == pathology]
        
        # Group by ML algorithm
        ml_algorithms = pathology_data['ml_algorithm'].unique()
        
        x_pos = np.arange(len(pathology_data))
        
        # Create bar plot
        bars = ax.bar(x_pos, pathology_data['ml_auc'], 
                     yerr=pathology_data['ml_auc_std'],
                     capsize=5, alpha=0.7, color=colors[:len(x_pos)])
        
        # Customize plot
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('AUC Score')
        ax.set_title(f'{pathology.replace("_", " ").title()} Prediction Performance')
        ax.set_ylim([0, 1])
        
        # Set x-axis labels
        labels = [f"{row['model_type']}\n{row['ml_algorithm']}" 
                 for _, row in pathology_data.iterrows()]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, auc_val, std_val in zip(bars, pathology_data['ml_auc'], pathology_data['ml_auc_std']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                   f'{auc_val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line at 0.5 (random performance)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        ax.legend()
    
    # Hide unused subplots
    for j in range(len(unique_pathologies), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Pathology-Level ML Model Performance Comparison ({split.title()} Set)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'pathology_ml_comparison_{split}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Pathology ML comparison saved to {output_path}")

def create_pathology_ensemble_summary_table(ensemble_results: Dict, output_dir: str,
                                           split: str = 'test') -> None:
    """
    Create comprehensive summary table of pathology ensemble results.
    """
    print(f"\nCreating pathology ensemble summary table ({split})...")
    
    # Collect all results
    table_data = []
    
    for model_type, model_results in ensemble_results.items():
        for pathology, pathology_results in model_results.items():
            for ml_name, ml_results in pathology_results.items():
                mean_auc = ml_results['mean_auc']
                std_auc = ml_results['std_auc']
                n_samples = ml_results['n_samples']
                class_dist = ml_results['class_distribution']
                
                # Calculate 95% CI
                n_folds = len(ml_results['fold_scores'])
                if n_folds > 1:
                    ci_lower, ci_upper = stats.t.interval(
                        0.95, n_folds-1, loc=mean_auc,
                        scale=stats.sem(ml_results['fold_scores'])
                    )
                else:
                    ci_lower = ci_upper = mean_auc
                
                table_data.append({
                    'Neural Network Model': model_type.replace('_', ' ').title(),
                    'Pathology': pathology.replace('_', ' ').title(),
                    'ML Algorithm': ml_name,
                    'AUC Mean': f"{mean_auc:.3f}",
                    'AUC 95% CI': f"[{ci_lower:.3f}-{ci_upper:.3f}]",
                    'Samples': n_samples,
                    'Positive Rate': f"{class_dist[1]/(class_dist[0]+class_dist[1]):.3f}" if len(class_dist) > 1 else "N/A",
                    'Cross-Val Folds': n_folds
                })
    
    if not table_data:
        print("  No ensemble results to tabulate")
        return
    
    ensemble_df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'pathology_ensemble_summary_{split}.csv')
    ensemble_df.to_csv(csv_path, index=False)
    
    # Create formatted table visualization
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(20, len(ensemble_df) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=ensemble_df.values, colLabels=ensemble_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        else:
            # Highlight best performing results
            if j == 3:  # AUC column
                try:
                    auc_val = float(ensemble_df.iloc[i-1]['AUC Mean'])
                    if auc_val > 0.8:
                        cell.set_facecolor('#90EE90')  # Light green for high AUC
                    elif auc_val > 0.7:
                        cell.set_facecolor('#FFE4B5')  # Light orange for medium AUC
                except:
                    pass
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    
    plt.title(f'Pathology-Level ML Model Performance Summary ({split.title()} Set)\n'
             f'Classical ML models trained on neural network pathology predictions', 
             fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(output_dir, f'pathology_ensemble_table_{split}.pdf')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.savefig(table_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Ensemble summary table saved to {csv_path} and {table_path}")

# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def analyze_attention_patterns(all_results: Dict, top_models: List[str], 
                              output_dir: str, split: str = 'test') -> None:
    """Analyze and visualize attention patterns for top models."""
    print(f"\nAnalyzing attention patterns ({split})...")
    
    setup_publication_style()
    
    for model_type in top_models:
        if model_type not in all_results:
            continue
            
        print(f"  Processing {model_type}...")
        
        # Collect attention data across folds
        all_attention_data = []
        
        for fold in all_results[model_type].keys():
            if split not in all_results[model_type][fold]:
                continue
                
            fold_data = all_results[model_type][fold][split]
            
            if 'complex' in fold_data and 'mil_attention' in fold_data['complex']:
                mil_attention = fold_data['complex']['mil_attention']
                
                if 'patients' in fold_data:
                    patients_df = fold_data['patients']
                    
                    for _, row in patients_df.iterrows():
                        patient_id = str(row['patient_id'])
                        if patient_id in mil_attention:
                            attention = mil_attention[patient_id]
                            tb_label = row['tb_label'] if 'tb_label' in row else -1
                            
                            all_attention_data.append({
                                'patient_id': patient_id,
                                'tb_label': tb_label,
                                'attention': attention,
                                'fold': fold
                            })
        
        if not all_attention_data:
            print(f"    No attention data found for {model_type}")
            continue
            
        # Create attention visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attention Pattern Analysis - {model_type.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # Filter by TB label
        tb_positive = [d for d in all_attention_data if d['tb_label'] == 1]
        tb_negative = [d for d in all_attention_data if d['tb_label'] == 0]
        
        # 1. Attention distribution by TB status
        ax = axes[0, 0]
        if tb_positive and tb_negative:
            pos_attention_means = [np.mean(d['attention']) for d in tb_positive]
            neg_attention_means = [np.mean(d['attention']) for d in tb_negative]
            
            ax.hist(pos_attention_means, alpha=0.7, label='TB+', bins=20, color='red')
            ax.hist(neg_attention_means, alpha=0.7, label='TB-', bins=20, color='blue')
            ax.set_xlabel('Mean Attention Weight')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Mean Attention Weights')
            ax.legend()
        
        # 2. Attention heatmap
        ax = axes[0, 1]
        if all_attention_data:
            # Create matrix of attention weights (patients x sites)
            max_sites = max(len(d['attention']) for d in all_attention_data)
            attention_matrix = np.zeros((len(all_attention_data), max_sites))
            
            for i, d in enumerate(all_attention_data):
                attention = d['attention']
                attention_matrix[i, :len(attention)] = attention
            
            # Sort by TB label for visualization
            tb_labels = [d['tb_label'] for d in all_attention_data]
            sort_idx = np.argsort(tb_labels)
            attention_matrix = attention_matrix[sort_idx]
            
            im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
            ax.set_xlabel('Site Position')
            ax.set_ylabel('Patient (sorted by TB status)')
            ax.set_title('Attention Heatmap')
            plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # 3. Site-wise attention comparison
        ax = axes[1, 0]
        if tb_positive and tb_negative:
            max_sites = max(len(d['attention']) for d in all_attention_data)
            
            pos_site_attention = np.zeros((len(tb_positive), max_sites))
            neg_site_attention = np.zeros((len(tb_negative), max_sites))
            
            for i, d in enumerate(tb_positive):
                pos_site_attention[i, :len(d['attention'])] = d['attention']
            
            for i, d in enumerate(tb_negative):
                neg_site_attention[i, :len(d['attention'])] = d['attention']
            
            # Calculate mean attention per site
            pos_mean = np.mean(pos_site_attention, axis=0)
            neg_mean = np.mean(neg_site_attention, axis=0)
            
            sites = range(max_sites)
            ax.bar([s - 0.2 for s in sites], pos_mean, width=0.4, label='TB+', color='red', alpha=0.7)
            ax.bar([s + 0.2 for s in sites], neg_mean, width=0.4, label='TB-', color='blue', alpha=0.7)
            ax.set_xlabel('Site Position')
            ax.set_ylabel('Mean Attention Weight')
            ax.set_title('Site-wise Attention Comparison')
            ax.legend()
        
        # 4. Attention entropy analysis
        ax = axes[1, 1]
        if all_attention_data:
            entropies = []
            labels = []
            
            for d in all_attention_data:
                attention = d['attention']
                # Calculate entropy
                attention_norm = attention / np.sum(attention) if np.sum(attention) > 0 else attention
                entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-8))
                entropies.append(entropy)
                labels.append(d['tb_label'])
            
            # Group by TB status
            tb_pos_entropy = [e for e, l in zip(entropies, labels) if l == 1]
            tb_neg_entropy = [e for e, l in zip(entropies, labels) if l == 0]
            
            if tb_pos_entropy and tb_neg_entropy:
                ax.boxplot([tb_neg_entropy, tb_pos_entropy], labels=['TB-', 'TB+'])
                ax.set_ylabel('Attention Entropy')
                ax.set_title('Attention Entropy by TB Status')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'attention_analysis_{model_type}_{split}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Attention analysis saved to {output_path}")

def analyze_site_level_predictions(all_results: Dict, top_models: List[str],
                                 output_dir: str, split: str = 'test') -> None:
    """Analyze site-level pathology predictions and create ML comparison."""
    print(f"\nAnalyzing site-level predictions ({split})...")
    
    pathology_names = ['a_lines', 'large_consolidation', 'pleural_effusion', 'other_pathology']
    
    for model_type in top_models:
        if model_type not in all_results:
            continue
            
        print(f"  Processing {model_type}...")
        
        # Collect site-level data
        all_site_data = []
        
        for fold in all_results[model_type].keys():
            if split not in all_results[model_type][fold]:
                continue
                
            fold_data = all_results[model_type][fold][split]
            
            if 'sites' in fold_data:
                sites_df = fold_data['sites'].copy()
                sites_df['fold'] = fold
                all_site_data.append(sites_df)
        
        if not all_site_data:
            continue
            
        combined_sites = pd.concat(all_site_data, ignore_index=True)
        
        # Analyze pathology predictions
        setup_publication_style()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Site-Level Pathology Analysis - {model_type.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Pathology prevalence
        ax = axes[0, 0]
        prevalences = []
        pathology_labels = []
        
        for pathology in pathology_names:
            finding_col = f'{pathology}_finding'
            if finding_col in combined_sites.columns:
                valid_mask = combined_sites[finding_col] >= 0
                if valid_mask.sum() > 0:
                    prevalence = combined_sites.loc[valid_mask, finding_col].mean()
                    prevalences.append(prevalence)
                    pathology_labels.append(pathology.replace('_', ' ').title())
        
        if prevalences:
            bars = ax.bar(pathology_labels, prevalences, color='steelblue', alpha=0.7)
            ax.set_ylabel('Prevalence')
            ax.set_title('Pathology Prevalence in Dataset')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, prevalences):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')
        
        # 2. Pathology prediction performance
        ax = axes[0, 1]
        pathology_aucs = []
        
        for pathology in pathology_names:
            finding_col = f'{pathology}_finding'
            prob_col = f'{pathology}_prob'
            
            if finding_col in combined_sites.columns and prob_col in combined_sites.columns:
                valid_mask = (combined_sites[finding_col] >= 0) & (~combined_sites[prob_col].isna())
                if valid_mask.sum() > 10:  # Need sufficient samples
                    y_true = combined_sites.loc[valid_mask, finding_col].values
                    y_prob = combined_sites.loc[valid_mask, prob_col].values
                    
                    try:
                        auc_score = roc_auc_score(y_true, y_prob)
                        pathology_aucs.append(auc_score)
                    except:
                        pathology_aucs.append(0)
                else:
                    pathology_aucs.append(0)
        
        if pathology_aucs:
            bars = ax.bar(pathology_labels, pathology_aucs, color='orange', alpha=0.7)
            ax.set_ylabel('AUC Score')
            ax.set_title('Pathology Prediction Performance')
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
            ax.legend()
            
            for bar, val in zip(bars, pathology_aucs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')
        
        # 3. Site distribution
        ax = axes[1, 0]
        if 'site_index' in combined_sites.columns:
            site_counts = combined_sites['site_index'].value_counts().sort_index()
            ax.bar(site_counts.index, site_counts.values, color='green', alpha=0.7)
            ax.set_xlabel('Site Index')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Sites')
        
        # 4. TB rate by site
        ax = axes[1, 1]
        if 'site_index' in combined_sites.columns and 'tb_label' in combined_sites.columns:
            site_tb_rates = combined_sites.groupby('site_index')['tb_label'].agg(['mean', 'count'])
            # Only show sites with sufficient samples
            site_tb_rates = site_tb_rates[site_tb_rates['count'] >= 10]
            
            bars = ax.bar(site_tb_rates.index, site_tb_rates['mean'], color='red', alpha=0.7)
            ax.set_xlabel('Site Index')
            ax.set_ylabel('TB Positive Rate')
            ax.set_title('TB Rate by Site')
            
            for bar, val in zip(bars, site_tb_rates['mean']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'site_analysis_{model_type}_{split}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Site analysis saved to {output_path}")

def create_statistical_significance_table(statistical_results: Dict, output_dir: str) -> None:
    """Create formatted table of statistical significance results."""
    print("\nCreating statistical significance table...")
    
    pairwise_tests = statistical_results['pairwise_tests']
    bonferroni_alpha = statistical_results.get('bonferroni_alpha', 0.05)
    
    if not pairwise_tests:
        print("  No statistical tests to report")
        return
    
    # Create table data
    table_data = []
    
    for test in pairwise_tests:
        significance = ""
        if test['p_value'] < 0.001:
            significance = "***"
        elif test['p_value'] < 0.01:
            significance = "**"
        elif test['p_value'] < 0.05:
            significance = "*"
        
        bonferroni_sig = "Yes" if test['bonferroni_significant'] else "No"
        
        effect_size = ""
        d = abs(test['cohens_d'])
        if d < 0.2:
            effect_size = "Negligible"
        elif d < 0.5:
            effect_size = "Small"
        elif d < 0.8:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        table_data.append({
            'Comparison': f"{test['model1']} vs {test['model2']}",
            'Mean Difference': f"{test['mean_diff']:.4f}",
            'p-value': f"{test['p_value']:.4f}",
            'Significance': significance,
            'Bonferroni Significant': bonferroni_sig,
            "Cohen's d": f"{test['cohens_d']:.3f}",
            'Effect Size': effect_size
        })
    
    # Create DataFrame and save
    significance_df = pd.DataFrame(table_data)
    csv_path = os.path.join(output_dir, 'statistical_significance_tests.csv')
    significance_df.to_csv(csv_path, index=False)
    
    # Create formatted table plot
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(16, len(table_data) * 0.6 + 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=significance_df.values, colLabels=significance_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        else:
            # Highlight significant results
            if j == 4 and table_data[i-1]['Bonferroni Significant'] == 'Yes':  # Bonferroni significant
                cell.set_facecolor('#90EE90')  # Light green
            elif j == 3 and table_data[i-1]['Significance']:  # Significant
                cell.set_facecolor('#FFE4B5')  # Light orange
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    
    plt.title(f'Statistical Significance Testing Results\n'
             f'Bonferroni corrected α = {bonferroni_alpha:.4f}\n'
             f'*p<0.05, **p<0.01, ***p<0.001', 
             fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(output_dir, 'statistical_significance_table.pdf')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.savefig(table_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Statistical significance table saved to {csv_path} and {table_path}")

# =============================================================================
# Data for report
# =============================================================================

def create_latex_macros(metrics_df: pd.DataFrame, output_dir: str, split: str = 'test') -> None:
    """
    Create LaTeX macros file with all relevant numbers for the report.
    
    Generates a .tex file with \newcommand macros containing model performance metrics
    that can be directly imported into a LaTeX report.
    """
    print(f"\n{'='*70}")
    print(f"CREATING LATEX MACROS FILE")
    print(f"{'='*70}")
    
    # Mapping from internal model names to LaTeX-friendly names
    model_name_mapping = {
        'original': 'CLIPRLOurs',
        'attention_pool': 'CLIPAttention',
        '3dcnn': 'ThreeDResNet',
        '3d_cnn': 'ThreeDResNet',
        'cnnlstm': 'CNNLSTM',
        'cnn_lstm': 'CNNLSTM',
        'vivit': 'VideoTransformer',
        'video_transformer': 'VideoTransformer',
        'r2plus1d': 'RTwoPlusOneD',
        'inception3d': 'InceptionThreeD',
        'mean_pool': 'CLIPMeanPool',
        'uniform': 'Uniform',
        'singletask': 'SingleTask',
        'single_task': 'SingleTask',
        'no_rl_full_train': 'NoRLFullTrain',
        'Efficientnet_RL': 'EfficientNetRL',
        'LeViT_Attention': 'LeViTAttention',
        'LeViT_RL': 'LeViTRL'
    }
    
    # Metrics to extract
    metrics_to_extract = ['auc', 'sensitivity', 'specificity', 'accuracy', 'f1', 
                         'ppv', 'npv', 'balanced_accuracy', 'precision',
                         'sens_at_90_spec', 'sens_at_70_spec', 'auprc']
    
    latex_commands = []
    latex_commands.append("% LaTeX macros for model performance metrics")
    latex_commands.append(f"% Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    latex_commands.append(f"% Split: {split}")
    latex_commands.append("")
    
    # Get unique models
    models = metrics_df['model'].unique()
    
    for model in models:
        # Get LaTeX-friendly name
        latex_model_name = model_name_mapping.get(model, model.replace('_', '').title())
        
        latex_commands.append(f"% Metrics for {model}")
        
        for metric in metrics_to_extract:
            # Filter data for this model and metric
            metric_data = metrics_df[(metrics_df['model'] == model) & 
                                    (metrics_df['metric'] == metric)]
            
            if len(metric_data) > 0:
                mean_val = metric_data['mean'].iloc[0]
                std_val = metric_data['std'].iloc[0]
                ci_lower = metric_data['ci_lower'].iloc[0]
                ci_upper = metric_data['ci_upper'].iloc[0]
                
                # Create macro names
                metric_name = metric.replace('_', '')
                
                # Mean value
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}Mean}}{{{mean_val:.3f}}}"
                )
                
                # Standard deviation
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}Std}}{{{std_val:.3f}}}"
                )
                
                # 95% CI lower bound
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}CILower}}{{{ci_lower:.3f}}}"
                )
                
                # 95% CI upper bound
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}CIUpper}}{{{ci_upper:.3f}}}"
                )
            else:
                # If metric not found, use TBU
                metric_name = metric.replace('_', '')
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}Mean}}{{TBU}}"
                )
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}Std}}{{TBU}}"
                )
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}CILower}}{{TBU}}"
                )
                latex_commands.append(
                    f"\\newcommand{{\\{latex_model_name}{metric_name.capitalize()}CIUpper}}{{TBU}}"
                )
        
        latex_commands.append("")
    
    # Add convenience macros for the specific table in the user's request
    latex_commands.append("% Convenience macros for architecture comparison table")
    latex_commands.append("")
    
    # For each model in the table (use only one variant per model to avoid duplicates)
    table_models = [
        ('original', 'CLIPRLOurs'),
        ('attention_pool', 'CLIPAttention'),
        ('3dcnn', 'ThreeDResNet'),
        ('cnnlstm', 'CNNLSTM'),
        ('vivit', 'VideoTransformer'),
        ('r2plus1d', 'RTwoPlusOneD')
    ]
    
    for internal_name, latex_name in table_models:
        # Check if this model exists in the data
        model_data = metrics_df[metrics_df['model'] == internal_name]
        
        if len(model_data) > 0:
            # Get AUC
            auc_data = model_data[model_data['metric'] == 'auc']
            if len(auc_data) > 0:
                auc_mean = auc_data['mean'].iloc[0]
                auc_std = auc_data['std'].iloc[0]
                latex_commands.append(
                    f"\\newcommand{{\\{latex_name}AUC}}{{{auc_mean:.3f} \\pm {auc_std:.3f}}}"
                )
            else:
                latex_commands.append(f"\\newcommand{{\\{latex_name}AUC}}{{TBU \\pm TBU}}")
            
            # Get Sensitivity
            sens_data = model_data[model_data['metric'] == 'sensitivity']
            if len(sens_data) > 0:
                sens_mean = sens_data['mean'].iloc[0]
                sens_std = sens_data['std'].iloc[0]
                latex_commands.append(
                    f"\\newcommand{{\\{latex_name}Sensitivity}}{{{sens_mean:.3f} \\pm {sens_std:.3f}}}"
                )
            else:
                latex_commands.append(f"\\newcommand{{\\{latex_name}Sensitivity}}{{TBU \\pm TBU}}")
            
            # Get Specificity
            spec_data = model_data[model_data['metric'] == 'specificity']
            if len(spec_data) > 0:
                spec_mean = spec_data['mean'].iloc[0]
                spec_std = spec_data['std'].iloc[0]
                latex_commands.append(
                    f"\\newcommand{{\\{latex_name}Specificity}}{{{spec_mean:.3f} \\pm {spec_std:.3f}}}"
                )
            else:
                latex_commands.append(f"\\newcommand{{\\{latex_name}Specificity}}{{TBU \\pm TBU}}")
        else:
            # Model not found, use TBU
            latex_commands.append(f"\\newcommand{{\\{latex_name}AUC}}{{TBU \\pm TBU}}")
            latex_commands.append(f"\\newcommand{{\\{latex_name}Sensitivity}}{{TBU \\pm TBU}}")
            latex_commands.append(f"\\newcommand{{\\{latex_name}Specificity}}{{TBU \\pm TBU}}")
        
        latex_commands.append("")
    
    # Write to file
    output_path = os.path.join(output_dir, f'model_metrics_macros_{split}.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_commands))
    
    print(f"✓ LaTeX macros saved to {output_path}")
    print(f"  Generated {len([c for c in latex_commands if c.startswith('\\newcommand')])} macro commands")



# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Results Analysis')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base directory containing all model results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis outputs')
    parser.add_argument('--model_types', type=str, nargs='+', 
                       default=['original', 'attention_pool', 'mean_pool', '3d_cnn', 
                               'cnn_lstm', 'video_transformer'],
                       help='Model types to analyze')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to analyze')
    parser.add_argument('--top_n', type=int, default=6,
                       help='Number of top models to include in detailed analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE MODEL RESULTS ANALYSIS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model types: {args.model_types}")
    print(f"Number of folds: {args.num_folds}")
    print(f"Split: {args.split}")
    print("="*80)
    
    # Load all results
    all_results = load_model_results(args.results_dir, args.model_types, args.num_folds)
    
    if not all_results:
        print("❌ No results found. Please check the results directory structure.")
        return
    
    # Calculate comprehensive metrics
    metrics_df = aggregate_cross_fold_metrics(all_results, args.model_types, args.split)
    
    # Perform statistical analysis
    statistical_results = perform_statistical_tests(all_results, args.model_types, args.split)
    
    # Select top performing models
    auc_metrics = metrics_df[metrics_df['metric'] == 'auc'].sort_values('mean', ascending=False)
    top_models = auc_metrics.head(args.top_n)['model'].tolist()
    
    print(f"\n🏆 Top {args.top_n} models by AUC:")
    for i, model in enumerate(top_models, 1):
        auc_data = auc_metrics[auc_metrics['model'] == model].iloc[0]
        print(f"  {i}. {model}: {auc_data['mean']:.4f} (95% CI: {auc_data['ci_lower']:.4f}-{auc_data['ci_upper']:.4f})")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
    print(f"{'='*70}")
    
    # 1. ROC curves with ALL models for comprehensive comparison
    print("Creating ROC curves for ALL models...")
    create_roc_curves_with_ci(all_results, args.model_types, args.output_dir, args.split, 
                             top_n=None, suffix="_all_models")
    
    # 2. ROC curves for TOP models only (cleaner visualization)
    print("Creating ROC curves for top performing models...")
    create_roc_curves_with_ci(all_results, top_models, args.output_dir, args.split, 
                             top_n=None, suffix="_top_models")
    
    # 3. Precision-Recall curves for ALL models
    print("Creating PR curves for ALL models...")
    create_pr_curves_with_ci(all_results, args.model_types, args.output_dir, args.split, 
                           top_n=None, suffix="_all_models")
    
    # 4. Precision-Recall curves for TOP models only
    print("Creating PR curves for top performing models...")
    create_pr_curves_with_ci(all_results, top_models, args.output_dir, args.split, 
                           top_n=None, suffix="_top_models")
    
    # 5. Comprehensive metrics table (ALL models)
    create_metrics_comparison_table(metrics_df, args.output_dir, args.split)
    
    # 6. Performance vs complexity plot (ALL models)
    create_performance_vs_complexity_plot(all_results, args.model_types, args.output_dir, args.split)
    
    # 7. Statistical significance table (ALL models)
    create_statistical_significance_table(statistical_results, args.output_dir)
    
    # Advanced analysis for top models
    print(f"\n{'='*70}")
    print("ADVANCED ANALYSIS FOR TOP MODELS")
    print(f"{'='*70}")
    
    # 6. Attention pattern analysis
    analyze_attention_patterns(all_results, top_models[:3], args.output_dir, args.split)
    
    # 7. Site-level analysis
    analyze_site_level_predictions(all_results, top_models[:3], args.output_dir, args.split)
    
    # 8. Pathology ensemble modeling - Train ML models on neural network pathology predictions
    print(f"\n{'='*70}")
    print("PATHOLOGY ENSEMBLE MODELING")
    print(f"{'='*70}")
    
    ensemble_results = train_pathology_ml_models(all_results, top_models[:3], args.output_dir, args.split)
    
    if ensemble_results:
        # Create pathology ensemble comparison plots
        compare_neural_vs_ml_pathology_predictions(ensemble_results, args.output_dir, args.split)
        
        # Create pathology ensemble summary table
        create_pathology_ensemble_summary_table(ensemble_results, args.output_dir, args.split)
    
    # Generate LaTeX macros for report
    print(f"\n{'='*70}")
    print("GENERATING LATEX MACROS FOR REPORT")
    print(f"{'='*70}")
    create_latex_macros(metrics_df, args.output_dir, args.split)
    
    # Save summary report
    summary_data = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'results_directory': args.results_dir,
        'models_analyzed': args.model_types,
        'number_of_folds': args.num_folds,
        'split_analyzed': args.split,
        'top_models': top_models,
        'statistical_tests_performed': len(statistical_results.get('pairwise_tests', [])),
        'bonferroni_alpha': statistical_results.get('bonferroni_alpha')
    }
    
    with open(os.path.join(args.output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"📁 All outputs saved to: {args.output_dir}")
    print(f"📊 Generated visualizations:")
    print(f"   • ROC curves with confidence intervals")
    print(f"   • Precision-Recall curves")  
    print(f"   • Comprehensive metrics comparison table")
    print(f"   • Performance vs complexity analysis")
    print(f"   • Statistical significance testing")
    print(f"   • Attention pattern analysis (top 3 models)")
    print(f"   • Site-level pathology analysis (top 3 models)")
    print(f"   • Pathology ensemble modeling (ML models trained on NN predictions)")
    print(f"   • Neural vs ML pathology prediction comparisons")
    print(f"📄 LaTeX report file:")
    print(f"   • model_metrics_macros_{args.split}.tex (macro definitions)")
    print(f"🏆 Best performing model: {top_models[0]} (AUC: {auc_metrics.iloc[0]['mean']:.4f})")
    if ensemble_results:
        print(f"🔬 Pathology ensemble models trained for {len(ensemble_results)} neural network models")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()