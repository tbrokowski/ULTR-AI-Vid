#!/usr/bin/env python3
"""
SA Fine-tuning Pipeline

Complete pipeline for fine-tuning Benin model on South Africa data:
1. Load SA data (259 patients with LUS)
2. Create 60/40 split stratified by TB, HIV, Prior TB
3. Evaluate Benin model zero-shot on 40% test set
4. Create training subsets (full, 80%, 60%, 40%) from 60% pool
5. Fine-tune on each subset
6. Evaluate best model on Benin test set
7. Generate pathology AUROC plots
"""

import os
import sys
import json
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CRITICAL: Module patch must happen BEFORE any imports from train_ablation_distributed
# ============================================================================
# This ensures that when train_ablation_distributed imports 'dataset',
# it actually gets dataset_sa instead
def _patch_dataset_module():
    """Patch sys.modules to use dataset_sa for SA data loading."""
    import dataset_sa
    sys.modules['dataset'] = dataset_sa
    logger.info("✓ Patched dataset module to use dataset_sa for SA data")

# ============================================================================
# Data Loading Functions
# ============================================================================

def identify_lus_patients(file_metadata_csv: str) -> Set[str]:
    """
    Identify patients with available LUS video data.
    
    Args:
        file_metadata_csv: Path to file metadata CSV
        
    Returns:
        Set of patient IDs with LUS data (including format variations)
    """
    logger.info(f"Identifying LUS patients from {file_metadata_csv}")
    
    if not os.path.exists(file_metadata_csv):
        raise FileNotFoundError(f"File metadata not found: {file_metadata_csv}")
    
    metadata_df = pd.read_csv(file_metadata_csv)
    logger.info(f"  Total LUS files: {len(metadata_df)}")
    
    # Extract patient IDs
    if 'Patient ID' in metadata_df.columns:
        patient_ids = set(metadata_df['Patient ID'].astype(str).unique())
    elif 'patient_id' in metadata_df.columns:
        patient_ids = set(metadata_df['patient_id'].astype(str).unique())
    else:
        raise ValueError("Could not find Patient ID column in metadata")
    
    logger.info(f"  Unique patients with LUS data: {len(patient_ids)}")
    
    # Handle ID format variations (e.g., IP270 -> 27-270)
    expanded_ids = patient_ids.copy()
    for pid in list(patient_ids):
        if pid.startswith('IP') and len(pid) > 2:
            numeric_part = pid[2:]
            if len(numeric_part) >= 3:
                expanded_ids.add(f"27-{numeric_part}")
                expanded_ids.add(f"27-{numeric_part[1:]}")
                expanded_ids.add(f"27-{numeric_part[2:]}")
    
    logger.info(f"  Including ID format variations: {len(expanded_ids)} total IDs")
    return expanded_ids


def load_sa_data(
    clinical_data_path: str,
    labels_csv: str,
    lus_patient_ids: Set[str]
) -> pd.DataFrame:
    """
    Load and merge SA clinical and pathology data, filtered to LUS patients.
    
    Args:
        clinical_data_path: Path to clinical data CSV
        labels_csv: Path to pathology labels CSV (RSA-specific)
        lus_patient_ids: Set of patient IDs with LUS data
        
    Returns:
        Merged dataframe with 259 patients
    """
    logger.info(f"Loading SA data...")
    logger.info(f"  Clinical data: {clinical_data_path}")
    logger.info(f"  Labels: {labels_csv}")
    
    # Load pathology labels
    labels_df = pd.read_csv(labels_csv)
    if 'record_id' in labels_df.columns:
        labels_df['patient_id'] = labels_df['record_id'].astype(str)
    elif 'patient_id' in labels_df.columns:
        labels_df['patient_id'] = labels_df['patient_id'].astype(str)
    else:
        raise ValueError("Could not find patient_id or record_id in labels CSV")
    
    logger.info(f"  Total patients in pathology labels: {len(labels_df)}")
    
    # Extract numeric IDs for matching (handles "25-XXX" vs "27-XXX" if needed)
    def extract_numeric_id(pid):
        pid_str = str(pid).strip()
        if '-' in pid_str:
            return pid_str.split('-')[-1]
        return pid_str
    
    labels_df['numeric_id'] = labels_df['patient_id'].apply(extract_numeric_id)
    
    # Filter labels by matching numeric part with LUS patient IDs
    lus_numeric_ids = set()
    for pid in lus_patient_ids:
        if '-' in str(pid):
            lus_numeric_ids.add(str(pid).split('-')[-1])
        else:
            lus_numeric_ids.add(str(pid))
    
    filtered_pathology = labels_df[
        labels_df['numeric_id'].isin(lus_numeric_ids)
    ].copy()
    logger.info(f"  Patients with LUS and pathology labels (by numeric ID): {len(filtered_pathology)}")
    
    # Load clinical data
    clinical_df = pd.read_csv(clinical_data_path)
    if 'record_id' in clinical_df.columns:
        clinical_df['patient_id'] = clinical_df['record_id'].astype(str)
    elif 'patient_id' in clinical_df.columns:
        clinical_df['patient_id'] = clinical_df['patient_id'].astype(str)
    else:
        raise ValueError("Could not find patient_id or record_id in clinical CSV")
    
    clinical_df['numeric_id'] = clinical_df['patient_id'].apply(extract_numeric_id)
    
    filtered_clinical = clinical_df[
        clinical_df['numeric_id'].isin(lus_numeric_ids)
    ].copy()
    logger.info(f"  Patients with LUS and clinical data: {len(filtered_clinical)}")
    
    # Merge clinical with pathology labels by numeric_id
    # Keep clinical patient_id (27-XXX format) as it matches file metadata
    merged_df = pd.merge(
        filtered_clinical,
        filtered_pathology[['numeric_id', 'TB Label']],
        on='numeric_id',
        how='inner'
    )
    # Drop the numeric_id column after merge
    merged_df = merged_df.drop(columns=['numeric_id'])
    
    logger.info(f"  Final merged data: {len(merged_df)} patients")
    
    # Prepare stratification columns
    # HIV column
    if 'hiv' in merged_df.columns:
        merged_df['HIV'] = merged_df['hiv'].apply(lambda x: 1 if x == 1 else 0)
        logger.info(f"  Using HIV column 'hiv' (1=Positive, 0=Negative)")
    else:
        raise ValueError("HIV column 'hiv' not found in clinical data")
    
    # Prior TB column
    if 'previous_tb_diagnosis' in merged_df.columns:
        # Map: 1=No, 2=Resolved, 4=Active -> Prior TB+ = 2 or 4
        merged_df['Prior_TB'] = merged_df['previous_tb_diagnosis'].apply(
            lambda x: 1 if x in [2, 4] else 0
        )
        logger.info(f"  Using Prior TB column 'previous_tb_diagnosis' (1=No, 2=Resolved, 4=Active)")
    else:
        raise ValueError("Prior TB column 'previous_tb_diagnosis' not found in clinical data")
    
    # TB Label
    merged_df['TB_Label'] = merged_df['TB Label'].apply(lambda x: 1 if x == 1 else 0)
    
    # Stratification summary
    logger.info(f"\n  Stratification summary:")
    logger.info(f"    Total patients: {len(merged_df)}")
    logger.info(f"    TB+: {merged_df['TB_Label'].sum()} ({merged_df['TB_Label'].mean()*100:.1f}%)")
    logger.info(f"    HIV+: {merged_df['HIV'].sum()} ({merged_df['HIV'].mean()*100:.1f}%)")
    logger.info(f"    Prior TB+: {merged_df['Prior_TB'].sum()} ({merged_df['Prior_TB'].mean()*100:.1f}%)")
    
    return merged_df


# ============================================================================
# Split Creation Functions
# ============================================================================

def create_initial_split(
    merged_df: pd.DataFrame,
    test_size: float = 0.4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create initial 60/40 split stratified by TB, HIV, and Prior TB.
    
    Args:
        merged_df: Merged dataframe with all patients
        test_size: Fraction for test set (default 0.4)
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"\n{'='*60}")
    logger.info("Creating initial 60/40 split (stratified by TB, prior TB, HIV)")
    logger.info(f"{'='*60}")
    
    # Create stratification key (combine TB, HIV, Prior TB)
    stratify_key = (
        merged_df['TB_Label'].astype(str) + '_' +
        merged_df['HIV'].astype(str) + '_' +
        merged_df['Prior_TB'].astype(str)
    )
    
    train_df, test_df = train_test_split(
        merged_df,
        test_size=test_size,
        stratify=stratify_key,
        random_state=random_state
    )
    
    logger.info(f"\n  Initial Split:")
    logger.info(f"    Training pool (60%): {len(train_df)} patients")
    logger.info(f"    Test set (40%): {len(test_df)} patients")
    
    logger.info(f"\n  Training pool stratification:")
    logger.info(f"    TB+: {train_df['TB_Label'].sum()} ({train_df['TB_Label'].mean()*100:.1f}%)")
    logger.info(f"    HIV+: {train_df['HIV'].sum()} ({train_df['HIV'].mean()*100:.1f}%)")
    logger.info(f"    Prior TB+: {train_df['Prior_TB'].sum()} ({train_df['Prior_TB'].mean()*100:.1f}%)")
    
    logger.info(f"\n  Test set stratification:")
    logger.info(f"    TB+: {test_df['TB_Label'].sum()} ({test_df['TB_Label'].mean()*100:.1f}%)")
    logger.info(f"    HIV+: {test_df['HIV'].sum()} ({test_df['HIV'].mean()*100:.1f}%)")
    logger.info(f"    Prior TB+: {test_df['Prior_TB'].sum()} ({test_df['Prior_TB'].mean()*100:.1f}%)")
    
    return train_df, test_df


def create_finetuning_subsets(
    train_pool_df: pd.DataFrame,
    subset_sizes: List[float] = [1.0],
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create stratified training subsets from training pool.
    
    Args:
        train_pool_df: Training pool dataframe (60% of data)
        subset_sizes: List of subset sizes (fractions of training pool). Default [1.0] means full training set only.
        random_state: Random seed
        
    Returns:
        Dictionary mapping subset names to dataframes
    """
    logger.info(f"\n{'='*60}")
    logger.info("Creating stratified fine-tuning subsets")
    logger.info(f"{'='*60}")
    logger.info(f"  Training pool size: {len(train_pool_df)} patients")
    
    subsets = {}
    stratify_key = train_pool_df['TB_Label']  # Stratify by TB only
    
    for subset_size in subset_sizes:
        if subset_size >= 1.0:
            subset_name = "full"
            subset_df = train_pool_df.copy()
        else:
            subset_name = f"{int(subset_size*100)}pct"
            sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=random_state)
            train_indices, _ = next(sss.split(train_pool_df, stratify_key))
            subset_df = train_pool_df.iloc[train_indices].copy().reset_index(drop=True)
        
        logger.info(f"  {subset_name}: {len(subset_df)} patients, "
                   f"TB+: {subset_df['TB_Label'].sum()} ({subset_df['TB_Label'].mean()*100:.1f}%)")
        
        subsets[subset_name] = subset_df
    
    return subsets


def map_patient_ids_to_metadata_format(
    patient_ids: List[str],
    file_metadata_csv: str
) -> List[str]:
    """
    Map patient IDs to the exact format used in file metadata.
    
    Args:
        patient_ids: List of patient IDs to map
        file_metadata_csv: Path to file metadata CSV
        
    Returns:
        List of mapped patient IDs in file metadata format
    """
    if not os.path.exists(file_metadata_csv):
        logger.warning(f"  File metadata not found: {file_metadata_csv}, using original IDs")
        return [str(pid).strip() for pid in patient_ids]
    
    file_metadata_df = pd.read_csv(file_metadata_csv)
    if 'Patient ID' in file_metadata_df.columns:
        metadata_patient_ids = set(file_metadata_df['Patient ID'].astype(str).str.strip().unique())
    else:
        metadata_patient_ids = set(file_metadata_df['patient_id'].astype(str).str.strip().unique())
    
    mapped_ids = []
    for pid in patient_ids:
        pid_str = str(pid).strip()
        if pid_str in metadata_patient_ids:
            mapped_ids.append(pid_str)
        else:
            # Try case-insensitive match
            pid_lower = pid_str.lower()
            matches = [mpid for mpid in metadata_patient_ids if mpid.lower() == pid_lower]
            if matches:
                mapped_ids.append(matches[0])
            else:
                # Try normalized match (remove dashes/underscores)
                pid_normalized = pid_str.replace('-', '').replace('_', '')
                matches = [mpid for mpid in metadata_patient_ids 
                          if mpid.replace('-', '').replace('_', '') == pid_normalized]
                if matches:
                    mapped_ids.append(matches[0])
                else:
                    mapped_ids.append(pid_str)
                    logger.warning(f"    Patient ID '{pid_str}' not found in file metadata")
    
    matched = sum(1 for mid in mapped_ids if mid in metadata_patient_ids)
    logger.info(f"  Mapped {len(patient_ids)} patient IDs: {matched}/{len(patient_ids)} matched to file metadata format")
    
    return mapped_ids


def create_split_csv(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    output_path: str,
    file_metadata_csv: Optional[str] = None
):
    """
    Create a split CSV file in the format expected by the dataset.
    
    Args:
        train_ids: List of training patient IDs
        val_ids: List of validation patient IDs
        test_ids: List of test patient IDs
        output_path: Path to save split CSV
        file_metadata_csv: Optional path to file metadata for ID mapping
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Map IDs to file metadata format if provided
    if file_metadata_csv:
        train_ids = map_patient_ids_to_metadata_format(train_ids, file_metadata_csv)
        val_ids = map_patient_ids_to_metadata_format(val_ids, file_metadata_csv) if val_ids else []
        test_ids = map_patient_ids_to_metadata_format(test_ids, file_metadata_csv)
    else:
        # Normalize IDs (strip whitespace)
        train_ids = [str(pid).strip() for pid in train_ids if pid and str(pid).strip()]
        val_ids = [str(pid).strip() for pid in val_ids if pid and str(pid).strip()]
        test_ids = [str(pid).strip() for pid in test_ids if pid and str(pid).strip()]
    
    # Create DataFrame
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    
    # Pad shorter lists with empty strings
    train_ids_padded = train_ids + [''] * (max_len - len(train_ids))
    val_ids_padded = val_ids + [''] * (max_len - len(val_ids))
    test_ids_padded = test_ids + [''] * (max_len - len(test_ids))
    
    split_df = pd.DataFrame({
        'train_ids': train_ids_padded,
        'valid_ids': val_ids_padded,
        'test_ids': test_ids_padded
    })
    
    split_df.to_csv(output_path, index=False)
    logger.info(f"  Split CSV saved: {output_path} (train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})")


# ============================================================================
# Helper Functions
# ============================================================================

def load_benin_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load Benin checkpoint weights."""
    logger.info(f"Loading Benin checkpoint from {checkpoint_path}")
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    else:
        return checkpoint


def _prepare_inputs_from_batch(batch, device, config):
    """Prepare batch inputs for model forward pass."""
    inputs = {
        'site_videos': batch['site_videos'].to(device, non_blocking=True),
        'site_indices': batch['site_indices'].to(device, non_blocking=True),
        'is_patient_level': True
    }
    
    if 'site_masks' in batch and batch['site_masks'] is not None:
        inputs['site_masks'] = batch['site_masks'].to(device, non_blocking=True)
    else:
        # Create default mask
        B, N = batch['site_videos'].shape[0], batch['site_videos'].shape[1]
        inputs['site_masks'] = torch.ones(B, N, dtype=torch.bool, device=device)
    
    if 'site_findings' in batch and batch['site_findings'] is not None:
        inputs['site_findings'] = batch['site_findings'].to(device, non_blocking=True)
    
    return inputs


def _safe_binary_auroc_auprc(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Calculate AUROC and AUPRC safely handling edge cases."""
    if len(labels) == 0 or len(probs) == 0:
        logger.warning("  No valid predictions; AUROC/AUPRC set to NaN")
        return np.nan, np.nan
    
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        logger.warning(f"  Only one class present ({unique_labels.tolist()}); metrics undefined")
        return np.nan, np.nan
    
    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        return auroc, auprc
    except Exception as e:
        logger.warning(f"  Error calculating metrics: {e}")
        return np.nan, np.nan


def save_comprehensive_predictions(
    trainer,
    data_loader,
    output_dir: str,
    prefix: str,
    config
) -> Dict:
    """
    Save comprehensive predictions in evaluate_downstream.py format.
    
    Saves:
    - {prefix}_patient_predictions.csv: Patient-level TB predictions
    - {prefix}_site_predictions.csv: Site-level pathology predictions
    - {prefix}_complex_data.pkl: Features, attention weights, etc.
    
    Args:
        trainer: AblationTrainer instance
        data_loader: DataLoader to evaluate
        output_dir: Directory to save outputs
        prefix: Prefix for filenames (e.g., 'benin_zero_shot', 'finetuned')
        config: Configuration object
        
    Returns:
        Dictionary with DataFrames and file paths
    """
    import pickle
    
    logger.info(f"  Collecting comprehensive predictions...")
    os.makedirs(output_dir, exist_ok=True)
    
    trainer.model.eval()
    patient_records = []
    site_records = []
    complex_data = {
        'patient_features': {},
        'site_features': {},
        'mil_attention': {},
        'task_logits': {},
        'pathology_scores': {}
    }
    
    pathology_names = getattr(config, 'pathology_names', ['A-line', 'Large Consolidations', 'Other Pathology'])
    active_tasks = config.active_tasks
    use_pathology_loss = config.use_pathology_loss
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Prepare inputs
                inputs = _prepare_inputs_from_batch(batch, trainer.device, config)
                outputs = trainer.model(inputs)
                
                # Get task predictions
                task_logits = outputs.get('task_logits', {})
                task_probs = {}
                task_preds = {}
                for task_name, logits in task_logits.items():
                    task_probs[task_name] = torch.sigmoid(logits.squeeze())
                    task_preds[task_name] = (task_probs[task_name] > 0.5).float()
                
                # Get TB labels and IDs
                tb_labels = batch['tb_labels'].to(trainer.device).squeeze()
                patient_ids = batch['patient_ids']
                batch_size = len(patient_ids)
                
                # Get optional outputs
                patient_features = outputs.get('patient_features')
                site_features = outputs.get('site_features')
                mil_attention = outputs.get('mil_attention')
                pathology_scores = outputs.get('pathology_logits')
                
                # Get site info
                site_masks = batch.get('site_mask', torch.ones(batch_size, 1))
                site_indices = batch.get('site_indices', torch.zeros(batch_size, 1, dtype=torch.long))
                site_findings = batch.get('site_findings')
                
                # Process each patient in batch
                for i in range(batch_size):
                    patient_id = str(patient_ids[i])
                    
                    # Get task predictions for this patient
                    task_labels = {}
                    task_logits_patient = {}
                    task_probs_patient = {}
                    task_preds_patient = {}
                    
                    if 'TB Label' in active_tasks:
                        task_labels['tb'] = tb_labels[i].cpu().item()
                        if 'TB Label' in task_logits:
                            task_logits_patient['tb'] = task_logits['TB Label'][i].cpu().item()
                            task_probs_patient['tb'] = task_probs['TB Label'][i].cpu().item()
                            task_preds_patient['tb'] = task_preds['TB Label'][i].cpu().item()
                    
                    # Number of valid sites
                    num_sites = site_masks[i].sum().item() if site_masks is not None else 0
                    
                    # Store patient-level complex data
                    if patient_features is not None:
                        complex_data['patient_features'][patient_id] = patient_features[i].cpu().numpy()
                    
                    if mil_attention is not None:
                        complex_data['mil_attention'][patient_id] = mil_attention[i].cpu().numpy()
                    
                    # Store task logits
                    for task_name in active_tasks:
                        if task_name not in complex_data['task_logits']:
                            complex_data['task_logits'][task_name] = {}
                        if task_name in task_logits:
                            complex_data['task_logits'][task_name][patient_id] = task_logits[task_name][i].cpu().numpy()
                    
                    if use_pathology_loss and pathology_scores is not None:
                        complex_data['pathology_scores'][patient_id] = pathology_scores[i].cpu().numpy()
                    
                    # Create patient-level record
                    patient_record = {
                        'patient_id': patient_id,
                        'num_valid_sites': int(num_sites),
                        'tb_label': task_labels.get('tb', -1),
                        'tb_logit': task_logits_patient.get('tb', float('nan')),
                        'tb_prob': task_probs_patient.get('tb', float('nan')),
                        'tb_pred': task_preds_patient.get('tb', float('nan')),
                    }
                    patient_records.append(patient_record)
                    
                    # Create site-level records
                    if site_findings is not None and num_sites > 0:
                        for s in range(int(num_sites)):
                            site_idx = site_indices[i, s].cpu().item()
                            site_finding = site_findings[i, s].cpu().numpy()
                            
                            site_record = {
                                'patient_id': patient_id,
                                'site_position': s,
                                'site_index': int(site_idx),
                                'tb_label': task_labels.get('tb', -1),
                                'tb_logit': task_logits_patient.get('tb', float('nan')),
                                'tb_prob': task_probs_patient.get('tb', float('nan')),
                                'tb_pred': task_preds_patient.get('tb', float('nan')),
                            }
                            
                            # Add site findings (ground truth)
                            for p_idx, p_name in enumerate(pathology_names):
                                if p_idx < len(site_finding):
                                    site_record[f'{p_name}_finding'] = float(site_finding[p_idx])
                            
                            # Add site pathology predictions
                            if use_pathology_loss and pathology_scores is not None:
                                site_path_scores = pathology_scores[i, s].cpu().numpy()
                                for p_idx, p_name in enumerate(pathology_names):
                                    if p_idx < len(site_path_scores):
                                        site_record[f'{p_name}_logit'] = float(site_path_scores[p_idx])
                                        site_record[f'{p_name}_prob'] = float(1 / (1 + np.exp(-site_path_scores[p_idx])))
                                        site_record[f'{p_name}_pred'] = int(site_path_scores[p_idx] > 0)
                            
                            # Add MIL attention
                            if mil_attention is not None:
                                site_record['mil_attention'] = float(mil_attention[i, s].cpu().item())
                            
                            # Store site features
                            if site_features is not None:
                                site_key = f"{patient_id}_site_{site_idx}"
                                complex_data['site_features'][site_key] = site_features[i, s].cpu().numpy()
                            
                            site_records.append(site_record)
            
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Create DataFrames
    patient_df = pd.DataFrame(patient_records)
    site_df = pd.DataFrame(site_records) if site_records else pd.DataFrame()
    
    # Save CSVs
    patient_csv_path = os.path.join(output_dir, f'{prefix}_patient_predictions.csv')
    patient_df.to_csv(patient_csv_path, index=False)
    logger.info(f"  ✓ Patient predictions: {patient_csv_path} ({len(patient_df)} patients)")
    
    site_csv_path = None
    if len(site_df) > 0:
        site_csv_path = os.path.join(output_dir, f'{prefix}_site_predictions.csv')
        site_df.to_csv(site_csv_path, index=False)
        logger.info(f"  ✓ Site predictions: {site_csv_path} ({len(site_df)} sites)")
    
    # Save complex data
    complex_path = os.path.join(output_dir, f'{prefix}_complex_data.pkl')
    with open(complex_path, 'wb') as f:
        pickle.dump(complex_data, f)
    logger.info(f"  ✓ Complex data: {complex_path}")
    
    return {
        'patient_df': patient_df,
        'site_df': site_df,
        'complex_data': complex_data,
        'patient_csv_path': patient_csv_path,
        'site_csv_path': site_csv_path,
        'complex_path': complex_path
    }


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_benin_zero_shot(
    benin_checkpoint_path: str,
    test_ids: List[str],
    config,
    output_dir: str
) -> Dict:
    """
    Evaluate Benin model (zero-shot) on SA test set.
    
    Args:
        benin_checkpoint_path: Path to Benin checkpoint
        test_ids: List of test patient IDs
        config: Configuration object
        output_dir: Output directory
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Step 6: Evaluating Benin Model (Zero-shot)")
    logger.info(f"{'='*60}")
    logger.info("Evaluating Benin model (zero-shot) on test set...")
    
    # Map test IDs to file metadata format
    test_ids = map_patient_ids_to_metadata_format(test_ids, config.file_metadata_csv)
    logger.info(f"  Test set: {len(test_ids)} patients")
    logger.info(f"  Sample IDs: {test_ids[:5]}")
    
    # Create split CSV
    test_split_csv = os.path.join(output_dir, 'splits', 'benin_test_split.csv')
    create_split_csv(
        train_ids=[],
        val_ids=[],
        test_ids=test_ids,
        output_path=test_split_csv,
        file_metadata_csv=None  # Already mapped above
    )
    
    # Import after patch (already done in main)
    from train_ablation_distributed import setup_distributed, cleanup_distributed, AblationTrainer, Config
    
    rank, world_size, local_rank = setup_distributed()
    
    try:
        # Create config for evaluation
        eval_config = Config()
        eval_config.load_from_yaml(config.__dict__.get('_yaml_path', 'sa_finetuning.yaml'))
        eval_config.split_csv = test_split_csv
        eval_config.experiment_name = "benin_zero_shot_eval"
        eval_config.experiment_dir = os.path.join(output_dir, "benin_zero_shot")
        eval_config.model_weights = benin_checkpoint_path
        eval_config.reset_optimizers = False
        eval_config.train = False
        
        # Distributed settings
        eval_config.rank = rank
        eval_config.world_size = world_size
        eval_config.local_rank = local_rank
        eval_config.distributed = world_size > 1
        
        # Create directories
        for dir_path in [eval_config.experiment_dir, eval_config.checkpoint_dir, 
                         eval_config.log_dir, eval_config.save_dir, eval_config.pred_save_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize trainer (uses dataset_sa due to patch)
        logger.info("  Creating trainer with SA dataset...")
        trainer = AblationTrainer(eval_config, rank, world_size, local_rank)
        logger.info(f"  ✓ Test dataset size: {len(trainer.test_loader.dataset)} patients")
        
        # Load Benin checkpoint
        benin_state_dict = load_benin_checkpoint(benin_checkpoint_path, trainer.device)
        model_state_dict = trainer.model.state_dict()
        filtered_state_dict = {k: v for k, v in benin_state_dict.items() 
                              if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
        model_state_dict.update(filtered_state_dict)
        trainer.model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"  Loaded {len(filtered_state_dict)}/{len(benin_state_dict)} parameters")
        
        # Evaluate
        logger.info("  Running evaluation...")
        test_loss, test_metrics = trainer.validate(0, trainer.test_loader, 'test')
        
        # Save comprehensive predictions (like evaluate_downstream.py)
        pred_results = save_comprehensive_predictions(
            trainer=trainer,
            data_loader=trainer.test_loader,
            output_dir=eval_config.experiment_dir,
            prefix='benin_zero_shot',
            config=eval_config
        )
        
        # Calculate metrics from patient predictions
        patient_df = pred_results['patient_df']
        valid_mask = patient_df['tb_label'] >= 0
        if valid_mask.sum() > 0:
            auroc, auprc = _safe_binary_auroc_auprc(
                patient_df.loc[valid_mask, 'tb_label'].values,
                patient_df.loc[valid_mask, 'tb_prob'].values
            )
        else:
            auroc, auprc = np.nan, np.nan
        
        results = {
            'model': 'benin_zero_shot',
            'n_test': len(patient_df),
            'auroc': auroc,
            'auprc': auprc,
            'test_metrics': {k.replace('TB Label_', ''): v for k, v in test_metrics.items() 
                           if k.startswith('TB Label_')},
            'prediction_files': {
                'patient_csv': pred_results['patient_csv_path'],
                'site_csv': pred_results['site_csv_path'],
                'complex_data': pred_results['complex_path']
            }
        }
        
        # Save results
        results_path = os.path.join(eval_config.experiment_dir, 'benin_zero_shot_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"  ✅ Zero-shot AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        return results
    
    finally:
        cleanup_distributed()


def create_oversampled_split_csv(
    train_ids: List[str],
    train_df: pd.DataFrame,
    output_path: str,
    positive_class_multiplier: int = 10,
    file_metadata_csv: str = None
) -> List[str]:
    """
    Create oversampled split CSV by repeating TB+ patients.
    
    Args:
        train_ids: List of training patient IDs
        train_df: DataFrame with patient_id and TB_Label columns
        output_path: Path to save split CSV
        positive_class_multiplier: How many times to repeat TB+ patients
        file_metadata_csv: Optional file metadata for ID mapping
        
    Returns:
        List of oversampled train IDs
    """
    # Separate TB+ and TB- patients
    tb_positive = train_df[train_df['TB_Label'] == 1]['patient_id'].tolist()
    tb_negative = train_df[train_df['TB_Label'] == 0]['patient_id'].tolist()
    
    logger.info(f"  Oversampling TB+ patients:")
    logger.info(f"    Original TB+: {len(tb_positive)}")
    logger.info(f"    Original TB-: {len(tb_negative)}")
    logger.info(f"    Multiplier: {positive_class_multiplier}x")
    
    # Oversample TB+ patients
    oversampled_positive = tb_positive * positive_class_multiplier
    oversampled_train_ids = oversampled_positive + tb_negative
    
    # Shuffle to mix TB+ and TB-
    import random
    random.shuffle(oversampled_train_ids)
    
    logger.info(f"    After oversampling:")
    logger.info(f"      TB+: {len(oversampled_positive)} ({len(oversampled_positive)/(len(oversampled_train_ids))*100:.1f}%)")
    logger.info(f"      TB-: {len(tb_negative)} ({len(tb_negative)/(len(oversampled_train_ids))*100:.1f}%)")
    logger.info(f"      Total: {len(oversampled_train_ids)} training samples")
    
    # Map to file metadata format if needed
    if file_metadata_csv:
        oversampled_train_ids = map_patient_ids_to_metadata_format(oversampled_train_ids, file_metadata_csv)
    
    # Create split CSV
    split_df = pd.DataFrame({
        'train_ids': pd.Series(oversampled_train_ids),
        'val_ids': pd.Series(dtype=str),
        'test_ids': pd.Series(dtype=str)
    })
    
    split_df.to_csv(output_path, index=False)
    
    return oversampled_train_ids


def run_finetuning_experiment(
    config,
    train_subsets: Dict[str, pd.DataFrame],
    test_ids: List[str],
    benin_checkpoint_path: str,
    output_dir: str,
    checkpoint_dir: str
) -> Dict:
    """
    Run fine-tuning experiment on all training subsets.
    
    Args:
        config: Configuration object
        train_subsets: Dictionary mapping subset names to dataframes
        test_ids: List of test patient IDs
        benin_checkpoint_path: Path to Benin checkpoint
        output_dir: Output directory for results/plots
        checkpoint_dir: Directory for saving model checkpoints
        
    Returns:
        Dictionary with results and best checkpoint path
    """
    logger.info(f"\n{'='*60}")
    logger.info("Starting fine-tuning experiment")
    logger.info(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    from train_ablation_distributed import setup_distributed, cleanup_distributed, AblationTrainer, Config
    
    rank, world_size, local_rank = setup_distributed()
    
    all_results = {}
    best_auroc = -1
    best_checkpoint = None
    
    try:
        # Map test IDs to file metadata format
        test_ids = map_patient_ids_to_metadata_format(test_ids, config.file_metadata_csv)
        logger.info(f"  Mapped {len(test_ids)} test patient IDs to file metadata format")
        
        for subset_name, train_subset_df in train_subsets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Fine-tuning subset: {subset_name}")
            logger.info(f"{'='*60}")
            
            # Map train IDs to file metadata format
            train_ids = map_patient_ids_to_metadata_format(
                train_subset_df['patient_id'].tolist(),
                config.file_metadata_csv
            )
            
            # Get TB labels for class imbalance handling
            tb_pos_count = train_subset_df['TB_Label'].sum()
            tb_neg_count = (train_subset_df['TB_Label'] == 0).sum()
            logger.info(f"  Train: {len(train_ids)} patients (TB+: {tb_pos_count}, TB-: {tb_neg_count})")
            logger.info(f"  Test: {len(test_ids)} patients")
            
            # Create directories: results go to output_dir, checkpoints go to checkpoint_dir
            subset_results_dir = os.path.join(output_dir, f"finetune_{subset_name}")
            subset_checkpoint_dir = os.path.join(checkpoint_dir, f"finetune_{subset_name}")
            os.makedirs(subset_results_dir, exist_ok=True)
            os.makedirs(subset_checkpoint_dir, exist_ok=True)
            logger.info(f"  Results directory: {subset_results_dir}")
            logger.info(f"  Checkpoint directory: {subset_checkpoint_dir}")
            
            # Create split CSV with optional oversampling
            subset_split_csv = os.path.join(subset_results_dir, "split.csv")
            
            # Check if oversampling is enabled in config
            oversample_enabled = getattr(config, 'oversample_positive_class', False)
            positive_multiplier = getattr(config, 'positive_class_multiplier', 10)
            
            if oversample_enabled and tb_pos_count > 0:
                logger.info(f"  📊 Applying oversampling to address class imbalance...")
                train_ids_for_split = create_oversampled_split_csv(
                    train_ids=train_ids,
                    train_df=train_subset_df,
                    output_path=subset_split_csv,
                    positive_class_multiplier=positive_multiplier,
                    file_metadata_csv=None  # Already mapped
                )
                # Note: Oversampled split CSV already created, but we need to add test_ids
                split_df = pd.read_csv(subset_split_csv)
                split_df['test_ids'] = pd.Series(test_ids)
                split_df.to_csv(subset_split_csv, index=False)
            else:
                if not oversample_enabled:
                    logger.info(f"  Oversampling disabled in config")
                create_split_csv(
                    train_ids=train_ids,
                    val_ids=[],
                    test_ids=test_ids,
                    output_path=subset_split_csv,
                    file_metadata_csv=None  # Already mapped above
                )
            
            # Update config
            subset_config = Config()
            subset_config.load_from_yaml(config.__dict__.get('_yaml_path', 'sa_finetuning.yaml'))
            subset_config.split_csv = subset_split_csv
            subset_config.experiment_name = f"sa_finetune_{subset_name}"
            subset_config.experiment_dir = subset_checkpoint_dir  # Use capstor for experiment dir (where checkpoints save)
            subset_config.checkpoint_dir = subset_checkpoint_dir  # Capstor checkpoint directory
            subset_config.log_dir = os.path.join(subset_results_dir, 'logs')  # Local logs
            subset_config.save_dir = subset_checkpoint_dir  # CRITICAL: Use capstor for saves (trainer uses this for checkpoints)
            subset_config.pred_save_dir = os.path.join(subset_results_dir, 'predictions')  # Local predictions
            subset_config.model_weights = benin_checkpoint_path
            subset_config.reset_optimizers = True
            subset_config.use_train_metric_when_no_val = True
            
            if not hasattr(subset_config, 'active_tasks') or 'TB Label' not in subset_config.active_tasks:
                subset_config.active_tasks = ["TB Label"]
            
            # Log class imbalance handling
            logger.info(f"  Class Imbalance Handling:")
            logger.info(f"    TB+ positive weight: {getattr(subset_config, 'task_pos_weights', {}).get('TB Label', 1.0)}")
            logger.info(f"    Oversampling enabled: {getattr(subset_config, 'oversample_positive_class', False)}")
            logger.info(f"    Augmentation enabled: {getattr(subset_config, 'use_augmentation', False)}")
            
            # Distributed settings
            subset_config.rank = rank
            subset_config.world_size = world_size
            subset_config.local_rank = local_rank
            subset_config.distributed = world_size > 1
            
            # Create directories
            for dir_path in [subset_config.experiment_dir, subset_config.checkpoint_dir,
                           subset_config.log_dir, subset_config.save_dir, subset_config.pred_save_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Initialize trainer
            trainer = AblationTrainer(subset_config, rank, world_size, local_rank)
            
            # Load Benin checkpoint
            benin_state_dict = load_benin_checkpoint(benin_checkpoint_path, trainer.device)
            model_state_dict = trainer.model.state_dict()
            filtered_state_dict = {k: v for k, v in benin_state_dict.items()
                                 if k in model_state_dict and model_state_dict[k].shape == v.shape}
            
            model_state_dict.update(filtered_state_dict)
            trainer.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"  Loaded {len(filtered_state_dict)}/{len(benin_state_dict)} parameters")
            
            # Train
            logger.info(f"  Starting fine-tuning...")
            best_metric, best_epoch = trainer.train()
            logger.info(f"  Best {subset_config.eval_metric} = {best_metric:.4f} at epoch {best_epoch+1}")
            
            # Evaluate on test set
            logger.info(f"  Evaluating on test set...")
            test_loss, test_metrics = trainer.validate(0, trainer.test_loader, 'test')
            
            # Save comprehensive predictions (like evaluate_downstream.py)
            pred_results = save_comprehensive_predictions(
                trainer=trainer,
                data_loader=trainer.test_loader,
                output_dir=subset_results_dir,
                prefix=f'finetuned_{subset_name}',
                config=subset_config
            )
            
            # Calculate metrics from patient predictions
            patient_df = pred_results['patient_df']
            valid_mask = patient_df['tb_label'] >= 0
            if valid_mask.sum() > 0:
                auroc, auprc = _safe_binary_auroc_auprc(
                    patient_df.loc[valid_mask, 'tb_label'].values,
                    patient_df.loc[valid_mask, 'tb_prob'].values
                )
            else:
                auroc, auprc = np.nan, np.nan
            
            subset_result = {
                'subset_name': subset_name,
                'n_train': len(train_subset_df),
                'n_test': len(test_ids),
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'auroc': auroc,
                'auprc': auprc,
                'test_metrics': {k.replace('TB Label_', ''): v for k, v in test_metrics.items() 
                               if k.startswith('TB Label_')},
                'prediction_files': {
                    'patient_csv': pred_results['patient_csv_path'],
                    'site_csv': pred_results['site_csv_path'],
                    'complex_data': pred_results['complex_path']
                }
            }
            
            # Save checkpoint path
            checkpoint_path = os.path.join(subset_config.checkpoint_dir, 'checkpoint_best.pth')
            subset_result['checkpoint_path'] = checkpoint_path
            logger.info(f"  Checkpoint saved to: {checkpoint_path}")
            
            # Save results
            results_path = os.path.join(subset_results_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(subset_result, f, indent=2, default=str)
            
            all_results[subset_name] = subset_result
            
            logger.info(f"  ✅ {subset_name} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
            
            # Track best model
            if not np.isnan(auroc) and auroc > best_auroc:
                best_auroc = auroc
                best_checkpoint = checkpoint_path
        
        return {
            'all_results': all_results,
            'best_checkpoint': best_checkpoint,
            'best_auroc': best_auroc
        }
    
    finally:
        cleanup_distributed()


def evaluate_on_benin_test(
    finetuned_checkpoint_path: str,
    benin_fold2_config_path: str,
    output_dir: str
) -> Dict:
    """
    Evaluate SA-finetuned model on Benin fold2 test set.
    
    Args:
        finetuned_checkpoint_path: Path to best SA-finetuned checkpoint
        benin_fold2_config_path: Path to Benin fold2 config YAML
        output_dir: Output directory
        
    Returns:
        Dictionary with evaluation results and comparison
    """
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating SA-Finetuned Model on Benin Fold2 Test Set")
    logger.info(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    import importlib
    if 'dataset' in sys.modules and hasattr(sys.modules['dataset'], '__file__'):
        # Reload the original dataset module
        import dataset as original_dataset
        sys.modules['dataset'] = original_dataset
    
    from train_ablation_distributed import setup_distributed, cleanup_distributed, AblationTrainer, Config
    
    rank, world_size, local_rank = setup_distributed()
    
    try:
        # Load Benin fold2 config
        benin_config = Config()
        benin_config.load_from_yaml(benin_fold2_config_path)
        
        # Create evaluation config
        eval_config = Config()
        eval_config.load_from_yaml(benin_fold2_config_path)
        eval_config.experiment_name = "sa_finetuned_on_benin_eval"
        eval_config.experiment_dir = os.path.join(output_dir, "benin_test_evaluation")
        eval_config.model_weights = finetuned_checkpoint_path
        eval_config.train = False
        eval_config.evaluate_best_valid_model = True
        
        # Use Benin data paths
        eval_config.root_dir = benin_config.root_dir
        eval_config.labels_csv = benin_config.labels_csv
        eval_config.file_metadata_csv = benin_config.file_metadata_csv
        eval_config.video_folder = benin_config.video_folder
        eval_config.split_csv = benin_config.split_csv
        
        # Distributed settings
        eval_config.rank = rank
        eval_config.world_size = world_size
        eval_config.local_rank = local_rank
        eval_config.distributed = world_size > 1
        
        # Create directories
        os.makedirs(eval_config.experiment_dir, exist_ok=True)
        os.makedirs(eval_config.pred_save_dir, exist_ok=True)
        
        # Initialize trainer (uses regular dataset.py for Benin data)
        trainer = AblationTrainer(eval_config, rank, world_size, local_rank)
        
        # Load fine-tuned checkpoint
        finetuned_state_dict = load_benin_checkpoint(finetuned_checkpoint_path, trainer.device)
        trainer.model.load_state_dict(finetuned_state_dict, strict=False)
        logger.info(f"  Loaded fine-tuned SA model from: {finetuned_checkpoint_path}")
        
        # Evaluate on Benin test set
        logger.info("  Running evaluation on Benin test set...")
        test_loss, test_metrics = trainer.validate(0, trainer.test_loader, 'test')
        
        # Save comprehensive predictions (like evaluate_downstream.py)
        pred_results = save_comprehensive_predictions(
            trainer=trainer,
            data_loader=trainer.test_loader,
            output_dir=eval_config.experiment_dir,
            prefix='sa_finetuned_benin_test',
            config=eval_config
        )
        
        # Calculate metrics from patient predictions
        patient_df = pred_results['patient_df']
        valid_mask = patient_df['tb_label'] >= 0
        if valid_mask.sum() > 0:
            auroc, auprc = _safe_binary_auroc_auprc(
                patient_df.loc[valid_mask, 'tb_label'].values,
                patient_df.loc[valid_mask, 'tb_prob'].values
            )
        else:
            auroc, auprc = np.nan, np.nan
        
        results = {
            'model': 'sa_finetuned_on_benin',
            'n_test': len(patient_df),
            'auroc': auroc,
            'auprc': auprc,
            'test_metrics': {k.replace('TB Label_', ''): v for k, v in test_metrics.items() 
                           if k.startswith('TB Label_')},
            'prediction_files': {
                'patient_csv': pred_results['patient_csv_path'],
                'site_csv': pred_results['site_csv_path'],
                'complex_data': pred_results['complex_path']
            }
        }
        
        # Load HMV-MIL baseline for comparison
        hmv_mil_path = '/users/tbrokowski/ULTR-AI-Vid/ablation_preds/attention_pool_differentiable/test_full_model_fold2_patients.csv'
        if os.path.exists(hmv_mil_path):
            hmv_mil_df = pd.read_csv(hmv_mil_path)
            if 'tb_prob' in hmv_mil_df.columns and 'tb_label' in hmv_mil_df.columns:
                hmv_auroc, hmv_auprc = _safe_binary_auroc_auprc(
                    hmv_mil_df['tb_label'].values,
                    hmv_mil_df['tb_prob'].values
                )
                results['comparison'] = {
                    'hmv_mil_auroc': hmv_auroc,
                    'hmv_mil_auprc': hmv_auprc,
                    'sa_finetuned_auroc': auroc,
                    'sa_finetuned_auprc': auprc,
                    'auroc_delta': auroc - hmv_auroc if not np.isnan(auroc) and not np.isnan(hmv_auroc) else np.nan,
                    'auprc_delta': auprc - hmv_auprc if not np.isnan(auprc) and not np.isnan(hmv_auprc) else np.nan
                }
                logger.info(f"  HMV-MIL baseline - AUROC: {hmv_auroc:.4f}, AUPRC: {hmv_auprc:.4f}")
                logger.info(f"  SA-Finetuned - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
                logger.info(f"  Delta - AUROC: {results['comparison']['auroc_delta']:.4f}, AUPRC: {results['comparison']['auprc_delta']:.4f}")
        
        # Save results
        results_path = os.path.join(eval_config.experiment_dir, 'benin_fold2_test_results_finetuned.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"  ✅ SA-Finetuned on Benin Test - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        return results
        
    finally:
        cleanup_distributed()


def generate_pathology_auroc_plots(
    finetuned_checkpoint_path: str,
    test_ids: List[str],
    config,
    output_dir: str
) -> Dict:
    """
    Generate pathology detection AUROC plots for best fine-tuned model.
    
    Args:
        finetuned_checkpoint_path: Path to best fine-tuned checkpoint
        test_ids: List of test patient IDs
        config: Configuration object
        output_dir: Output directory
        
    Returns:
        Dictionary with pathology metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info("Generating Pathology AUROC Plots")
    logger.info(f"{'='*60}")
    
    if plt is None:
        logger.warning("  matplotlib not available, skipping plots")
        return {}
    
    # Re-patch dataset module for SA data
    _patch_dataset_module()
    
    from train_ablation_distributed import setup_distributed, cleanup_distributed, AblationTrainer, Config
    
    rank, world_size, local_rank = setup_distributed()
    
    try:
        # Map test IDs
        test_ids = map_patient_ids_to_metadata_format(test_ids, config.file_metadata_csv)
        
        # Create split CSV
        test_split_csv = os.path.join(output_dir, 'splits', 'pathology_eval_split.csv')
        create_split_csv(
            train_ids=[],
            val_ids=[],
            test_ids=test_ids,
            output_path=test_split_csv,
            file_metadata_csv=None
        )
        
        # Create config
        eval_config = Config()
        eval_config.load_from_yaml(config.__dict__.get('_yaml_path', 'sa_finetuning.yaml'))
        eval_config.split_csv = test_split_csv
        eval_config.experiment_name = "pathology_eval"
        eval_config.experiment_dir = os.path.join(output_dir, "pathology_plots")
        eval_config.model_weights = finetuned_checkpoint_path
        eval_config.train = False
        
        # Distributed settings
        eval_config.rank = rank
        eval_config.world_size = world_size
        eval_config.local_rank = local_rank
        eval_config.distributed = world_size > 1
        
        os.makedirs(eval_config.experiment_dir, exist_ok=True)
        os.makedirs(eval_config.pred_save_dir, exist_ok=True)
        
        # Initialize trainer
        trainer = AblationTrainer(eval_config, rank, world_size, local_rank)
        
        # Load checkpoint
        checkpoint_state_dict = load_benin_checkpoint(finetuned_checkpoint_path, trainer.device)
        trainer.model.load_state_dict(checkpoint_state_dict, strict=False)
        
        # Evaluate and collect pathology predictions
        trainer.model.eval()
        pathology_predictions = {
            'a_lines': [],
            'large_consolidation': [],
            'pleural_effusion': [],
            'other_pathology': []
        }
        pathology_labels = {
            'a_lines': [],
            'large_consolidation': [],
            'pleural_effusion': [],
            'other_pathology': []
        }
        
        with torch.no_grad():
            for batch in trainer.test_loader:
                inputs = _prepare_inputs_from_batch(batch, trainer.device, eval_config)
                outputs = trainer.model(inputs)
                
                # Get pathology predictions
                if 'pathology_scores' in outputs:
                    pathology_scores = outputs['pathology_scores']  # [B, N, 4]
                    site_findings = batch['site_findings'].to(trainer.device)  # [B, N, 4]
                    
                    # Aggregate across sites (max or mean)
                    pathology_scores_agg = pathology_scores.max(dim=1)[0]  # [B, 4]
                    pathology_labels_agg = site_findings.max(dim=1)[0]  # [B, 4]
                    
                    # Extract per-pathology
                    pathology_predictions['a_lines'].extend(
                        torch.sigmoid(pathology_scores_agg[:, 0]).cpu().numpy()
                    )
                    pathology_predictions['large_consolidation'].extend(
                        torch.sigmoid(pathology_scores_agg[:, 1]).cpu().numpy()
                    )
                    pathology_predictions['pleural_effusion'].extend(
                        torch.sigmoid(pathology_scores_agg[:, 2]).cpu().numpy()
                    )
                    pathology_predictions['other_pathology'].extend(
                        torch.sigmoid(pathology_scores_agg[:, 3]).cpu().numpy()
                    )
                    
                    pathology_labels['a_lines'].extend(pathology_labels_agg[:, 0].cpu().numpy())
                    pathology_labels['large_consolidation'].extend(pathology_labels_agg[:, 1].cpu().numpy())
                    pathology_labels['pleural_effusion'].extend(pathology_labels_agg[:, 2].cpu().numpy())
                    pathology_labels['other_pathology'].extend(pathology_labels_agg[:, 3].cpu().numpy())
        
        # Calculate metrics and create plots
        pathology_metrics = {}
        os.makedirs(os.path.join(eval_config.experiment_dir, 'plots'), exist_ok=True)
        
        pathology_names = {
            'a_lines': 'A-lines',
            'large_consolidation': 'Large Consolidations',
            'pleural_effusion': 'Pleural Effusion',
            'other_pathology': 'Other Pathology'
        }
        
        for path_name, display_name in pathology_names.items():
            labels = np.array(pathology_labels[path_name])
            probs = np.array(pathology_predictions[path_name])
            
            # Filter out invalid labels (-1)
            valid_mask = labels != -1
            if valid_mask.sum() > 0:
                labels_valid = labels[valid_mask]
                probs_valid = probs[valid_mask]
                
                auroc, auprc = _safe_binary_auroc_auprc(labels_valid, probs_valid)
                pathology_metrics[path_name] = {
                    'auroc': auroc,
                    'auprc': auprc,
                    'n_samples': len(labels_valid)
                }
                
                # Create ROC curve plot
                if not np.isnan(auroc):
                    fpr, tpr, _ = roc_curve(labels_valid, probs_valid)
                    
                    plt.figure(figsize=(8, 8))
                    plt.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate', fontsize=12)
                    plt.ylabel('True Positive Rate', fontsize=12)
                    plt.title(f'ROC Curve: {display_name}', fontsize=14)
                    plt.legend(loc='lower right', fontsize=11)
                    plt.grid(True, alpha=0.3)
                    
                    plot_path = os.path.join(eval_config.experiment_dir, 'plots', f'auroc_{path_name}.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"  ✓ Saved plot: {plot_path} (AUROC: {auroc:.3f})")
        
        # Save metrics
        metrics_path = os.path.join(eval_config.experiment_dir, 'pathology_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(pathology_metrics, f, indent=2, default=str)
        
        logger.info(f"  ✅ Pathology evaluation complete")
        return pathology_metrics
        
    finally:
        cleanup_distributed()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point."""
    # CRITICAL: Patch dataset module FIRST, before any other imports
    _patch_dataset_module()
    
    # Now safe to import train_ablation_distributed
    from train_ablation_distributed import Config
    
    parser = argparse.ArgumentParser(description='SA Fine-tuning Pipeline')
    parser.add_argument('--config', type=str, default='sa_finetuning.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, 
                       default='./sa_finetuning_results',
                       help='Output directory for results (plots, predictions, etc.)')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/sa_finetuning',
                       help='Directory for saving model checkpoints (capstor recommended)')
    parser.add_argument('--benin-checkpoint', type=str,
                       default='/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/attention_pool_extra3_full_train2/fold2/checkpoint_best.pth',
                       help='Path to Benin fold2 checkpoint')
    parser.add_argument('--benin-config', type=str,
                       help='Path to Benin fold2 config YAML (optional)')
    parser.add_argument('--skip-zero-shot', action='store_true',
                       help='Skip zero-shot evaluation')
    parser.add_argument('--skip-finetuning', action='store_true',
                       help='Skip fine-tuning experiments')
    parser.add_argument('--skip-benin-eval', action='store_true',
                       help='Skip Benin test set evaluation')
    parser.add_argument('--skip-pathology', action='store_true',
                       help='Skip pathology plots generation')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info("SA Fine-tuning Pipeline")
    logger.info(f"{'='*80}")
    
    # Load configuration
    config = Config()
    config.load_from_yaml(args.config)
    config.__dict__['_yaml_path'] = args.config
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'splits'), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Log output directory information
    logger.info(f"Output directory for plots/results (absolute): {os.path.abspath(args.output_dir)}")
    logger.info(f"Checkpoint directory for model weights (absolute): {os.path.abspath(args.checkpoint_dir)}")
    logger.info(f"Checkpoints will be saved to: {args.checkpoint_dir}/[subset_name]/checkpoint_best.pth")
    
    # Step 1: Identify LUS patients
    logger.info(f"\n{'='*60}")
    logger.info("Step 1: Identifying LUS Patients")
    logger.info(f"{'='*60}")
    lus_patient_ids = identify_lus_patients(config.file_metadata_csv)
    
    # Step 2: Load SA data
    logger.info(f"\n{'='*60}")
    logger.info("Step 2: Loading SA Data")
    logger.info(f"{'='*60}")
    merged_df = load_sa_data(
        clinical_data_path=config.sa_clinical_data,
        labels_csv=config.labels_csv,
        lus_patient_ids=lus_patient_ids
    )
    
    logger.info(f"✅ Successfully loaded {len(merged_df)} patients with LUS data")
    logger.info(f"   TB+: {merged_df['TB_Label'].sum()} patients")
    logger.info(f"   TB-: {(merged_df['TB_Label'] == 0).sum()} patients")
    
    # Step 3: Create 60/40 split
    logger.info(f"\n{'='*60}")
    logger.info("Step 3: Creating 60/40 Split")
    logger.info(f"{'='*60}")
    train_df, test_df = create_initial_split(merged_df)
    
    # Step 4: Create training subsets
    logger.info(f"\n{'='*60}")
    logger.info("Step 4: Creating Training Subsets")
    logger.info(f"{'='*60}")
    train_subsets = create_finetuning_subsets(train_df)
    
    # Step 5: Create split CSVs
    logger.info(f"\n{'='*60}")
    logger.info("Step 5: Creating Split CSVs")
    logger.info(f"{'='*60}")
    for subset_name, subset_df in train_subsets.items():
        train_ids = subset_df['patient_id'].tolist()
        test_ids_list = test_df['patient_id'].tolist()
        
        split_csv_path = os.path.join(args.output_dir, 'splits', f'split_{subset_name}.csv')
        create_split_csv(
            train_ids=train_ids,
            val_ids=[],
            test_ids=test_ids_list,
            output_path=split_csv_path,
            file_metadata_csv=config.file_metadata_csv
        )
        logger.info(f"  ✅ {subset_name}: {len(train_ids)} train, {len(test_ids_list)} test")
    
    # Step 6: Evaluate Benin model zero-shot (optional)
    test_ids_list = test_df['patient_id'].tolist()
    
    if not args.skip_zero_shot:
        logger.info(f"\n{'='*60}")
        logger.info("Step 6: Evaluating Benin Model (Zero-shot)")
        logger.info(f"{'='*60}")
        try:
            zero_shot_results = evaluate_benin_zero_shot(
                benin_checkpoint_path=args.benin_checkpoint,
                test_ids=test_ids_list,
                config=config,
                output_dir=args.output_dir
            )
            logger.info(f"✅ Zero-shot evaluation complete: AUROC={zero_shot_results.get('auroc', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"❌ Zero-shot evaluation failed: {e}")
    else:
        logger.info("⏭️  Skipping zero-shot evaluation")
    
    # Step 7: Fine-tune on training subsets (optional)
    best_checkpoint = None
    if not args.skip_finetuning:
        logger.info(f"\n{'='*60}")
        logger.info("Step 7: Fine-tuning on Full Training Set (60%)")
        logger.info(f"{'='*60}")
        try:
            finetuning_results = run_finetuning_experiment(
                config=config,
                train_subsets=train_subsets,
                test_ids=test_ids_list,
                benin_checkpoint_path=args.benin_checkpoint,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir
            )
            best_checkpoint = finetuning_results.get('best_checkpoint')
            logger.info(f"✅ Fine-tuning complete")
            logger.info(f"   Best checkpoint: {best_checkpoint}")
            logger.info(f"   Best AUROC: {finetuning_results.get('best_auroc', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to find checkpoint anyway
            full_checkpoint = os.path.join(args.checkpoint_dir, 'finetune_full', 'checkpoint_best.pth')
            if os.path.exists(full_checkpoint):
                best_checkpoint = full_checkpoint
                logger.info(f"   Found checkpoint despite error: {best_checkpoint}")
    else:
        logger.info("⏭️  Skipping fine-tuning")
    
    # Step 8: Evaluate best model on Benin test set (optional)
    if not args.skip_benin_eval and best_checkpoint and args.benin_config:
        logger.info(f"\n{'='*60}")
        logger.info("Step 8: Evaluating on Benin Test Set")
        logger.info(f"{'='*60}")
        try:
            benin_eval_results = evaluate_on_benin_test(
                finetuned_checkpoint_path=best_checkpoint,
                benin_fold2_config_path=args.benin_config,
                output_dir=args.output_dir
            )
            logger.info(f"✅ Benin test evaluation complete: AUROC={benin_eval_results.get('auroc', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"❌ Benin test evaluation failed: {e}")
    else:
        if args.skip_benin_eval:
            logger.info("⏭️  Skipping Benin test set evaluation")
        elif not best_checkpoint:
            logger.warning("⚠️  No best checkpoint available for Benin evaluation")
        elif not args.benin_config:
            logger.warning("⚠️  No Benin config provided for evaluation")
    
    # Step 9: Generate pathology plots (optional)
    if not args.skip_pathology and best_checkpoint:
        logger.info(f"\n{'='*60}")
        logger.info("Step 9: Generating Pathology AUROC Plots")
        logger.info(f"{'='*60}")
        try:
            pathology_results = generate_pathology_auroc_plots(
                finetuned_checkpoint_path=best_checkpoint,
                test_ids=test_ids_list,
                config=config,
                output_dir=args.output_dir
            )
            logger.info(f"✅ Pathology plots generated")
            for path_name, metrics in pathology_results.items():
                logger.info(f"   {path_name}: AUROC={metrics.get('auroc', 'N/A'):.3f}")
        except Exception as e:
            logger.error(f"❌ Pathology plot generation failed: {e}")
    else:
        if args.skip_pathology:
            logger.info("⏭️  Skipping pathology plots")
        elif not best_checkpoint:
            logger.warning("⚠️  No best checkpoint available for pathology evaluation")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("SA Fine-tuning Pipeline Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Total patients: {len(merged_df)}")
    logger.info(f"  Training pool: {len(train_df)}")
    logger.info(f"  Test set: {len(test_df)}")
    logger.info("")


if __name__ == '__main__':
    main()
    
    
#python finetune_sa.py --config sa_finetuning.yaml --benin-checkpoint /capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/attention_pool_extra3_full_train2/fold2/checkpoint_best.pth --benin-config /users/tbrokowski/ULTR-AI-Vid/configs/attention_pool_differentiable/fold2.yaml