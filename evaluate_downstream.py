import os
import sys
import logging
import json
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import argparse
from pathlib import Path
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add paths
PROJECT_ROOT = "."
SRC_PATH = "."
NETWORK_PATH = "."

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, NETWORK_PATH)

from dataset import LungUltrasoundDataModule
#from NetworkArchitecture.CLIP_DRL_Aug26 import MultiTaskModel
from config import load_config, MultiTaskConfig


from NetworkArchitecture.CLIP_DRL_Aug11 import MultiTaskModel
from NetworkArchitecture.ablation_models import create_ablation_model



def evaluate_model_for_downstream(model, dataloader, device, active_tasks, 
                                  use_pathology_loss=True):
    """
    Evaluate model and save data in format compatible with downstream multimodal pipeline.
    
    Returns:
        - patient_df: Patient-level aggregated data
        - site_df: Site-level data with all findings
        - complex_data: Dictionary for HDF5 storage
        - metrics: Evaluation metrics
    """
    print("=== Evaluating Model for Downstream Pipeline ===")
    model.eval()
    
    # Storage
    patient_records = []
    site_records = []
    
    # Complex data storage - MUST match dataset_multimodal.py expectations
    complex_data = {
        'patient_features': {},        # patient_id -> [feature_dim]
        'site_features': {},          # "patientid_site_siteidx" -> [feature_dim]
        'mil_attention': {},          # patient_id -> [max_sites]
        'task_logits': {},            # task_name -> patient_id -> logits
        'pathology_scores': {}        # patient_id -> [max_sites, num_pathologies]
    }
    
    stats = {
        'total_patients': 0,
        'total_sites': 0,
        'total_batches': 0,
        'failed_batches': 0,
    }
    
    pathology_names = ['a_lines', 'large_consolidation', 'pleural_effusion', 'other_pathology']
    
    print(f"Processing {len(dataloader)} batches...")
    print(f"Active tasks: {active_tasks}")
    print(f"Use pathology loss: {use_pathology_loss}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Extract batch data
                patient_ids = batch['patient_ids']
                site_videos = batch['site_videos'].to(device)
                site_indices = batch['site_indices'].to(device)
                site_masks = batch['site_masks'].to(device)
                site_findings = batch['site_findings'].to(device)
                
                # Multi-task labels
                tb_labels = batch['tb_labels'].to(device).float()
                pneumonia_labels = batch.get('pneumonia_labels', torch.zeros_like(tb_labels)).to(device).float()
                covid_labels = batch.get('covid_labels', torch.zeros_like(tb_labels)).to(device).float()
                
                batch_size = len(patient_ids)
                stats['total_patients'] += batch_size
                stats['total_batches'] += 1
                
                # Prepare model inputs
                inputs = {
                    'site_videos': site_videos,
                    'site_indices': site_indices,
                    'site_masks': site_masks,
                    'site_findings': site_findings,
                    'is_patient_level': True
                }
                
                # Forward pass
                outputs = model(inputs)
                
                # Extract outputs
                task_logits = outputs.get('task_logits', {})
                mil_attention = outputs.get('mil_attention', None)
                patient_features = outputs.get('patient_features', None)
                site_features = outputs.get('site_features', None)
                pathology_scores = outputs.get('pathology_scores', None) if use_pathology_loss else None
                
                # Get predictions
                task_probs = {}
                task_preds = {}
                for task_name in active_tasks:
                    if task_name in task_logits:
                        task_probs[task_name] = torch.sigmoid(task_logits[task_name])
                        task_preds[task_name] = (task_probs[task_name] > 0.5).float()
                
                # Process each patient
                for i in range(batch_size):
                    patient_id = str(patient_ids[i])  # Ensure string
                    
                    # Get labels
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
                    
                    if 'Pneumonia Label' in active_tasks:
                        task_labels['pneumonia'] = pneumonia_labels[i].cpu().item()
                        if 'Pneumonia Label' in task_logits:
                            task_logits_patient['pneumonia'] = task_logits['Pneumonia Label'][i].cpu().item()
                            task_probs_patient['pneumonia'] = task_probs['Pneumonia Label'][i].cpu().item()
                            task_preds_patient['pneumonia'] = task_preds['Pneumonia Label'][i].cpu().item()
                    
                    # Number of valid sites
                    num_sites = site_masks[i].sum().item()
                    stats['total_sites'] += num_sites
                    
                    # Store patient-level complex data - CRITICAL FORMAT
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
                    
                    # Create patient-level record - COMPATIBLE WITH DOWNSTREAM
                    patient_record = {
                        'patient_id': patient_id,
                        'num_valid_sites': int(num_sites),
                        'tb_label': task_labels.get('tb', -1),
                        'tb_logit': task_logits_patient.get('tb', float('nan')),
                        'tb_prob': task_probs_patient.get('tb', float('nan')),
                        'tb_pred': task_preds_patient.get('tb', float('nan')),
                    }
                    
                    # Add optional tasks
                    if 'pneumonia' in task_labels:
                        patient_record['pneumonia_label'] = task_labels['pneumonia']
                        patient_record['pneumonia_prob'] = task_probs_patient.get('pneumonia', float('nan'))
                    
                    patient_records.append(patient_record)
                    
                    for s in range(int(num_sites)):
                        site_idx = site_indices[i, s].cpu().item()
                        site_finding = site_findings[i, s].cpu().numpy()
                        
                        # Create site-level record - COMPATIBLE FORMAT
                        site_record = {
                            'patient_id': patient_id,
                            'site_position': s,  # Position in patient's site array
                            'site_index': int(site_idx),  # Anatomical site index
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
                        
                        if site_features is not None:
                            # Key format: "patientid_site_siteidx"
                            site_key = f"{patient_id}_site_{site_idx}"
                            complex_data['site_features'][site_key] = site_features[i, s].cpu().numpy()
                        
                        site_records.append(site_record)
                
                # Memory management
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"❌ OOM at batch {batch_idx}, skipping")
                    stats['failed_batches'] += 1
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"❌ Error at batch {batch_idx}: {e}")
                stats['failed_batches'] += 1
                continue
    
    print("\n=== Evaluation Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create DataFrames
    print(f"\nCreating DataFrames from {len(patient_records)} patient records and {len(site_records)} site records...")
    
    patient_df = pd.DataFrame(patient_records)
    site_df = pd.DataFrame(site_records)
    
    print(f"Patient DataFrame shape: {patient_df.shape}")
    print(f"Site DataFrame shape: {site_df.shape}")
    
    # Calculate metrics
    metrics = {}
    if len(patient_df) > 0 and 'tb_label' in patient_df.columns:
        valid_mask = patient_df['tb_label'] >= 0
        if valid_mask.sum() > 0:
            patient_targets = patient_df.loc[valid_mask, 'tb_label'].values
            patient_probs = patient_df.loc[valid_mask, 'tb_prob'].values
            patient_preds = patient_df.loc[valid_mask, 'tb_pred'].values
            
            try:
                from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
                metrics = {
                    'auc': float(roc_auc_score(patient_targets, patient_probs)),
                    'accuracy': float(accuracy_score(patient_targets, patient_preds)),
                    'sensitivity': float(recall_score(patient_targets, patient_preds, zero_division=0)),
                    'precision': float(precision_score(patient_targets, patient_preds, zero_division=0)),
                }
                print(f"\nMetrics: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate metrics: {e}")
    
    return patient_df, site_df, complex_data, metrics


def save_for_downstream_pipeline(patient_df, site_df, complex_data, metrics, 
                                 output_dir, split_name, fold_num):
    """
    Save data in format compatible with dataset_multimodal.py
    
    Critical formats:
    1. HDF5 structure must match dataset_multimodal.py expectations
    2. CSV columns must match what dataset expects
    3. File naming must follow pattern: {split}_full_model_fold{fold}_*
    """
    
    print(f"=== Saving Data for Downstream Pipeline ===")
    print(f"Output directory: {output_dir}")
    print(f"Split: {split_name}, Fold: {fold_num}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # File naming pattern - MUST MATCH dataset_multimodal.py expectations
    base_name = f"{split_name}_full_model_fold{fold_num}"
    
    saved_files = {}
    
    # 1. Save patient-level CSV (aggregated)
    patient_csv_path = os.path.join(output_dir, f'{base_name}_patients.csv')
    patient_df.to_csv(patient_csv_path, index=False)
    saved_files['patient_csv'] = patient_csv_path
    print(f"✓ Patient CSV: {patient_csv_path}")
    
    # 2. Save site-level CSV - THIS IS WHAT dataset_multimodal.py LOADS
    site_csv_path = os.path.join(output_dir, f'{base_name}_sites.csv')
    site_df.to_csv(site_csv_path, index=False)
    saved_files['site_csv'] = site_csv_path
    print(f"✓ Site CSV: {site_csv_path}")
    
    # 3. Save complex data in HDF5 - CRITICAL FORMAT
    hdf5_path = os.path.join(output_dir, f'{base_name}_complex_data.h5')
    
    try:
        with h5py.File(hdf5_path, 'w') as f:
            # Metadata
            f.attrs['split'] = split_name
            f.attrs['fold'] = fold_num
            f.attrs['num_patients'] = len(patient_df)
            f.attrs['num_sites'] = len(site_df)
            
            # patient_features group 
            if complex_data['patient_features']:
                patient_grp = f.create_group('patient_features')
                for patient_id, features in complex_data['patient_features'].items():
                    # Store with patient_id as key
                    patient_grp.create_dataset(str(patient_id), data=features)
                print(f"    ✓ Saved {len(complex_data['patient_features'])} patient embeddings")
            
            # site_features group
            if complex_data['site_features']:
                site_grp = f.create_group('site_features')
                for site_key, features in complex_data['site_features'].items():
                    # Key format: "patientid_site_siteidx"
                    site_grp.create_dataset(site_key, data=features)
                print(f"    ✓ Saved {len(complex_data['site_features'])} site embeddings")
            
            # MIL attention
            if complex_data['mil_attention']:
                mil_grp = f.create_group('mil_attention')
                for patient_id, attention in complex_data['mil_attention'].items():
                    mil_grp.create_dataset(str(patient_id), data=attention)
                print(f"    ✓ Saved MIL attention for {len(complex_data['mil_attention'])} patients")
            
            # Task logits
            if complex_data['task_logits']:
                task_grp = f.create_group('task_logits')
                for task_name, task_data in complex_data['task_logits'].items():
                    task_subgrp = task_grp.create_group(task_name.replace(' ', '_'))
                    for patient_id, logits in task_data.items():
                        task_subgrp.create_dataset(str(patient_id), data=logits)
                print(f"    ✓ Saved task logits for {len(complex_data['task_logits'])} tasks")
            
            # Pathology scores
            if complex_data['pathology_scores']:
                pathology_grp = f.create_group('pathology_scores')
                for patient_id, scores in complex_data['pathology_scores'].items():
                    pathology_grp.create_dataset(str(patient_id), data=scores)
                print(f"    ✓ Saved pathology scores for {len(complex_data['pathology_scores'])} patients")
        
        saved_files['hdf5'] = hdf5_path
        print(f"✓ HDF5: {hdf5_path}")
        
    except Exception as e:
        print(f"❌ Error saving HDF5: {e}")
    
    # 4. Save metrics
    metrics_path = os.path.join(output_dir, f'{base_name}_metrics.json')
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        saved_files['metrics'] = metrics_path
        print(f"✓ Metrics: {metrics_path}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
    
    print(f"\n✅ All files saved for {split_name} fold {fold_num}")
    
    return saved_files


def verify_compatibility(output_dir, split_name, fold_num):
    """Verify that saved files are compatible with dataset_multimodal.py"""
    
    print(f"\n=== Verifying Compatibility ===")
    
    base_name = f"{split_name}_full_model_fold{fold_num}"
    
    # Check files exist
    site_csv = os.path.join(output_dir, f'{base_name}_sites.csv')
    hdf5_file = os.path.join(output_dir, f'{base_name}_complex_data.h5')
    
    checks = {
        'site_csv_exists': os.path.exists(site_csv),
        'hdf5_exists': os.path.exists(hdf5_file),
    }
    
    # Check CSV format
    if checks['site_csv_exists']:
        try:
            df = pd.read_csv(site_csv)
            required_cols = ['patient_id', 'tb_label', 'tb_prob']
            checks['csv_has_required_cols'] = all(col in df.columns for col in required_cols)
            checks['csv_patient_ids_are_strings'] = df['patient_id'].dtype == 'object'
            print(f"  ✓ CSV format valid")
        except Exception as e:
            print(f"  ❌ CSV error: {e}")
            checks['csv_has_required_cols'] = False
    
    # Check HDF5 format
    if checks['hdf5_exists']:
        try:
            with h5py.File(hdf5_file, 'r') as f:
                checks['hdf5_has_patient_features'] = 'patient_features' in f
                checks['hdf5_has_site_features'] = 'site_features' in f
                
                if checks['hdf5_has_patient_features']:
                    num_patients = len(f['patient_features'].keys())
                    print(f"  ✓ HDF5 has {num_patients} patient embeddings")
                
                if checks['hdf5_has_site_features']:
                    num_sites = len(f['site_features'].keys())
                    print(f"  ✓ HDF5 has {num_sites} site embeddings")
                    
                    # Check key format
                    sample_key = list(f['site_features'].keys())[0]
                    checks['site_key_format_correct'] = '_site_' in sample_key
                    print(f"  ✓ Site key format: {sample_key}")
        except Exception as e:
            print(f"  ❌ HDF5 error: {e}")
            checks['hdf5_has_patient_features'] = False
            checks['hdf5_has_site_features'] = False
    
    # Summary
    all_passed = all(checks.values())
    
    print(f"\n{'='*50}")
    print(f"Compatibility Check: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print(f"{'='*50}")
    
    for check, passed in checks.items():
        status = '✅' if passed else '❌'
        print(f"  {status} {check}")
    
    return all_passed


def _infer_eval_output_dir_from_config(config) -> str:
    """
    Derive eval output directory from config.experiment_dir.
    
    Structure: /capstor/.../ablation_results/{experiment_name}/eval_results/
    All folds save to the same eval_results directory within their experiment.
    
    Example:
      experiment_dir: /capstor/.../ablation_results/3dcnn/fold0
      -> eval_results: /capstor/.../ablation_results/3dcnn/eval_results
    """
    try:
        exp_dir = getattr(config, 'experiment_dir', None)
        if not exp_dir:
            # Default to external storage location
            return '/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/default/eval_results'
        
        p = Path(exp_dir)
        
        # Remove trailing foldX if present to get experiment base directory
        name = p.name
        if name.startswith('fold') and name[4:].isdigit():
            # exp_dir is like: .../ablation_results/3dcnn/fold0
            # base should be: .../ablation_results/3dcnn
            experiment_base = p.parent
        else:
            # exp_dir is already the experiment base (no fold suffix)
            experiment_base = p
        
        # Eval results go in eval_results/ within the experiment directory
        # This way all folds for an experiment share the same eval_results folder
        return str(experiment_base / 'eval_results')
    except Exception as e:
        # Fallback to external storage
        print(f"Warning: Could not infer eval output dir: {e}")
        return '/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/default/eval_results'


def process_all_folds(model_type: str, config_paths: list, model_paths: list, output_base_dir: str, video_folder_override: str | None = None):
    """
    Process all folds to generate data for downstream pipeline.
    
    Args:
        config_paths: List of paths to config files for each fold
        model_paths: List of paths to model checkpoints for each fold
        output_base_dir: Base directory to save all outputs
    """
    
    print("\n" + "="*60)
    print("GENERATING DATA FOR ALL FOLDS")
    print("="*60)
    print(f"Number of folds: {len(config_paths)}")
    print(f"Output directory: {output_base_dir}")
    print("="*60 + "\n")
    
    successful_folds = []
    failed_folds = []
    
    for fold_idx, (config_path, model_path) in enumerate(zip(config_paths, model_paths)):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}")
        print(f"{'='*60}")
        print(f"Config: {config_path}")
        print(f"Model: {model_path}")
        
        # Check if files exist
        if not os.path.exists(config_path):
            print(f"❌ Config not found: {config_path}")
            failed_folds.append(fold_idx)
            continue
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            failed_folds.append(fold_idx)
            continue
        
        try:
            # Set device
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            
            # Load config
            config = load_config(config_file=config_path)
            print(f"✓ Config loaded")

            # Optional override for video folder
            if video_folder_override:
                try:
                    old_vf = getattr(config, 'video_folder', None)
                except Exception:
                    old_vf = None
                setattr(config, 'video_folder', video_folder_override)
                print(f"✓ Overriding video_folder: {old_vf} -> {video_folder_override}")
            
            # Extract configuration
            active_tasks = getattr(config, 'active_tasks', ['TB Label'])
            use_pathology_loss = getattr(config, 'use_pathology_loss', True)
            
            # Setup data module
            data_module = LungUltrasoundDataModule(
                root_dir=config.root_dir,
                labels_csv=config.labels_csv,
                file_metadata_csv=config.file_metadata_csv,
                image_folder=config.image_folder,
                video_folder=config.video_folder,
                split_csv=config.split_csv,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                frame_sampling=config.frame_sampling,
                depth_filter=config.depth_filter,
                cache_size=100,
                files_per_site=getattr(config, 'files_per_site', 'all'),
                site_order=getattr(config, 'site_order', None),
                pad_missing_sites=getattr(config, 'pad_missing_sites', True),
                max_sites=getattr(config, 'max_sites', None),
            )
            print(f"✓ Data module created")
            
            # Initialize model
            #model = MultiTaskModel(config)
            model = create_ablation_model(model_type, config)
            print(f"✓ Model initialized: {model_type}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            print(f"✓ Model loaded and ready")
            
            # Setup data
            data_module.setup(stage='patient_level')
            
            # Process each split
            for split_name in ['train', 'val', 'test']:
                print(f"\n--- Processing {split_name.upper()} split ---")
                
                dataloader = data_module.patient_level_dataloader(split_name)
                
                # Evaluate
                patient_df, site_df, complex_data, metrics = evaluate_model_for_downstream(
                    model, dataloader, device, active_tasks, use_pathology_loss
                )
                
                # Determine output dir per fold from config if user provided default
                per_fold_output_dir = output_base_dir
                if per_fold_output_dir == 'ULTR-CLIP/results' or not per_fold_output_dir:
                    per_fold_output_dir = _infer_eval_output_dir_from_config(config)
                    print(f"Inferred per-fold output dir: {per_fold_output_dir}")

                # Save in compatible format
                saved_files = save_for_downstream_pipeline(
                    patient_df, site_df, complex_data, metrics,
                    per_fold_output_dir, split_name, fold_idx
                )
                
                # Verify compatibility
                verify_compatibility(per_fold_output_dir, split_name, fold_idx)
            
            successful_folds.append(fold_idx)
            print(f"\n✅ Fold {fold_idx} completed successfully")
            
        except Exception as e:
            print(f"\n❌ Error processing fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            failed_folds.append(fold_idx)
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Successful folds: {successful_folds} ({len(successful_folds)}/{len(config_paths)})")
    if failed_folds:
        print(f"❌ Failed folds: {failed_folds}")
    print(f"\n📁 All data saved to: {output_base_dir}")
    print("="*60)
    
    # Generate file list for user
    print("\nGenerated files per fold:")
    print("  - {split}_full_model_fold{N}_sites.csv")
    print("  - {split}_full_model_fold{N}_complex_data.h5")
    print("  - {split}_full_model_fold{N}_metrics.json")
    print("\nThese files are now compatible with:")
    print("  - dataset_multimodal.py")
    print("  - run_experiments_multimodal.py")
    print("  - run_experiments_uncertainty.py")


def main():
    parser = argparse.ArgumentParser(description='Generate data for downstream multimodal pipeline')
    
    # Data arguments
    parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='original')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='ULTR-CLIP/results',
                       help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')

    
    # Multi-fold processing
    parser.add_argument('--process-all-folds', action='store_true',
                       help='Process all folds at once')
    parser.add_argument('--config-pattern', type=str,
                       help='Pattern for config files (e.g., "configs/fold_{}.yaml")')
    parser.add_argument('--model-pattern', type=str,
                       help='Pattern for model files (e.g., "checkpoints/fold_{}/best.pth")')
    parser.add_argument('--num-folds', type=int, default=5,
                       help='Number of folds to process')
    
    args = parser.parse_args()
    
    if args.process_all_folds:
        # Process all folds
        if not args.config_pattern or not args.model_pattern:
            print("Error: --config-pattern and --model-pattern required for --process-all-folds")
            return
        
        config_paths = [args.config_pattern.format(i) for i in range(args.num_folds)]
        model_paths = [args.model_pattern.format(i) for i in range(args.num_folds)]

        process_all_folds(args.model_type, config_paths, model_paths, args.output_dir, video_folder_override=args.video_folder)

    else:
        # Process single fold
        if not args.config or not args.model:
            print("Error: --config and --model required for single fold processing")
            return
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config = load_config(config_file=args.config)
        # Optional override for video folder
        if args.video_folder:
            try:
                old_vf = getattr(config, 'video_folder', None)
            except Exception:
                old_vf = None
            setattr(config, 'video_folder', args.video_folder)
            print(f"✓ Overriding video_folder: {old_vf} -> {args.video_folder}")
        
        # Load model
        model = create_ablation_model(args.model_type, config)
        print(f"✓ Model initialized: {args.model_type}")
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        print("Successfully loaded model")
        model.eval()
        
        # Setup data
        data_module = LungUltrasoundDataModule(
            root_dir=config.root_dir,
            labels_csv=config.labels_csv,
            file_metadata_csv=config.file_metadata_csv,
            image_folder=config.image_folder,
            video_folder=config.video_folder,
            split_csv=config.split_csv,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            frame_sampling=config.frame_sampling,
            depth_filter=config.depth_filter,
            cache_size=100,
        )
        
        data_module.setup(stage='patient_level')
        
        # Process each split
        # Determine default output dir if user left default
        output_dir = args.output_dir
        if output_dir == 'ULTR-CLIP/results' or not output_dir:
            output_dir = _infer_eval_output_dir_from_config(config)
            print(f"Inferred output dir: {output_dir}")

        for split_name in ['train', 'val', 'test']:
            print(f"\nProcessing {split_name} split...")
            
            dataloader = data_module.patient_level_dataloader(split_name)
            
            active_tasks = getattr(config, 'active_tasks', ['TB Label'])
            use_pathology_loss = getattr(config, 'use_pathology_loss', True)
            
            patient_df, site_df, complex_data, metrics = evaluate_model_for_downstream(
                model, dataloader, device, active_tasks, use_pathology_loss
            )
            
            saved_files = save_for_downstream_pipeline(
                patient_df, site_df, complex_data, metrics,
                output_dir, split_name, args.fold
            )
            
            verify_compatibility(output_dir, split_name, args.fold)
        
        print(f"\n✅ Fold {args.fold} processing complete!")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Multimodal Pipeline Data Generator")
    print("="*60)
    print("\nUsage examples:")
    print("\n1. Process single fold:")
    print("   python evaluate_downstream.py --model-type cnn_lstm --config configs/cnnlstmfold0.yaml --model ablation_results/cnnlstm/fold0/checkpoint_best.pth --fold 0 --output-dir Tests/results")
    print("\n2. Process all folds:")
    print("   python evaluate_downstream.py --process-all-folds \\")
    print("       --config-pattern 'configs/fold_{}.yaml' \\")
    print("       --model-pattern 'checkpoints/fold_{}/best.pth' \\")
    print("       --num-folds 5 \\")
    print("       --output-dir ULTR-CLIP/results")
    print("="*60 + "\n")
    
    main()
    
    
    #python evaluate_downstream.py --model-type cnn_lstm --config configs/cnnlstmfold0.yaml --model ablation_results/cnnlstm/fold0/checkpoints/checkpoint_best.pth --fold 0 --output-dir Tests/results

    #python evaluate_downstream.py --model-type 3d_cnn --config configs/3dcnn/fold0.yaml --model ablation_results/3dcnn/fold0/checkpoints/checkpoint_best.pth --fold 0 --output-dir Tests/results

    
    #     model_map = {
    #     'original': MultiTaskModel,
    #     'no_rl': NoRLMultiTaskModel,
    #     'no_rl_full_train': NoRLFullTrainMultiTaskModel,
    #     'mean_pool': MeanPoolMultiTaskModel,
    #     'attention_pool': AttentionPoolMultiTaskModel,
    #     'single_task': SingleTaskMultiTaskModel,
    #     '3d_cnn': ResNet3DMultiTaskModel,  # Using ResNet3D
    #     'cnn_lstm': CNNLSTMMultiTaskModel,
    #     'R2+1d': R2Plus1DMultiTaskModel,
    #     'Inception': InceptionMultiTaskModel,
    #     'InceptionRLBackbone': RLInceptionMultiTaskModel,
    #     'video_transformer': VideoTransformerMultiTaskModel,  # Using ViViT
    # }