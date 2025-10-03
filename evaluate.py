import os
import sys
import logging
import json
import pickle
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add both project root and src to path
PROJECT_ROOT = "."
SRC_PATH = "."
NETWORK_PATH = "."

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, NETWORK_PATH)

# Updated imports to match your new training system
from dataset import LungUltrasoundDataModule
#from NetworkArchitecture.CLIP_Multitask_Aug5 import MultiTaskModel
from NetworkArchitecture.CLIP_DRL_Aug11 import MultiTaskModel   # UPDATE TO AUG26
from config import load_config, MultiTaskConfig

try:
    from NetworkArchitecture.monitoring_utils import log_model_component_status
except ImportError as e:
    print(e)
    logger = logging.getLogger(__name__)
    logger.warning("Monitoring utilities not available")
    log_model_component_status = lambda *args: None
        
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration if logging was already configured
)
logger = logging.getLogger(__name__)


def evaluate_model_comprehensive(model, dataloader, device, active_tasks, use_pathology_loss=True, save_complex_data=True):
    """
    Comprehensive evaluation for multi-task model that saves all model outputs and creates structured dataframes.
    
    Args:
        model: MultiTaskModel instance
        dataloader: DataLoader for evaluation
        device: torch.device
        active_tasks: List of active task names (e.g., ['TB Label', 'Pneumonia Label'])
        use_pathology_loss: Whether pathology prediction is enabled
        save_complex_data: Whether to save complex tensor data
        
    Returns:
        - patient_df: Patient-level dataframe
        - site_df: Site-level dataframe  
        - complex_data: Dictionary with complex tensor data
        - metrics: Evaluation metrics
    """
    logger.info("=== Starting Comprehensive Multi-Task Model Evaluation ===")
    model.eval()
    
    # Storage for structured data
    patient_records = []
    site_records = []
    
    # Storage for complex data
    complex_data = {
        'patient_features': {},        # patient_id -> numpy array [feature_dim]
        'mil_attention': {},          # patient_id -> numpy array [max_sites]
        'site_features': {},          # (patient_id, site_idx) -> numpy array [feature_dim]
        'site_rl_data': {},          # (patient_id, site_idx) -> dict
        'task_logits': {},           # task_name -> patient_id -> numpy array
    }
    
    if use_pathology_loss:
        complex_data['pathology_scores'] = {}  # patient_id -> numpy array [max_sites, num_pathologies]
    
    # Statistics tracking
    stats = {
        'total_patients': 0,
        'total_sites': 0,
        'total_batches': 0,
        'failed_batches': 0,
    }
    
    pathology_names = ['a_lines', 'b_lines', 'small_consolidations', 'large_consolidations', 'pleural_effusion']
    
    logger.info(f"Processing {len(dataloader)} batches...")
    logger.info(f"Active tasks: {active_tasks}")
    logger.info(f"Use pathology loss: {use_pathology_loss}")
    
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
                pneumonia_labels = batch['pneumonia_labels'].to(device).float()
                covid_labels = batch['covid_labels'].to(device).float()
                
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
                
                # Extract model outputs
                task_logits = outputs.get('task_logits', {})
                
                # Get multi-task probabilities and predictions
                task_probs = {}
                task_preds = {}
                for task_name in active_tasks:
                    if task_name in task_logits:
                        task_probs[task_name] = torch.sigmoid(task_logits[task_name])
                        task_preds[task_name] = (task_probs[task_name] > 0.5).float()
                
                # Extract other outputs
                mil_attention = outputs.get('mil_attention', None)
                patient_features = outputs.get('patient_features', None)
                site_features = outputs.get('site_features', None)
                site_rl_data = outputs.get('site_rl_data', None)
                pathology_scores = outputs.get('pathology_scores', None) if use_pathology_loss else None
                
                # Process each patient in the batch
                for i in range(batch_size):
                    patient_id = patient_ids[i]
                    
                    # Get task labels and predictions for this patient
                    task_labels = {}
                    task_logits_patient = {}
                    task_probs_patient = {}
                    task_preds_patient = {}
                    
                    if 'TB Label' in active_tasks:
                        task_labels['TB Label'] = tb_labels[i].cpu().item()
                        if 'TB Label' in task_logits:
                            task_logits_patient['TB Label'] = task_logits['TB Label'][i].cpu().item()
                            task_probs_patient['TB Label'] = task_probs['TB Label'][i].cpu().item()
                            task_preds_patient['TB Label'] = task_preds['TB Label'][i].cpu().item()
                    
                    if 'Pneumonia Label' in active_tasks:
                        task_labels['Pneumonia Label'] = pneumonia_labels[i].cpu().item()
                        if 'Pneumonia Label' in task_logits:
                            task_logits_patient['Pneumonia Label'] = task_logits['Pneumonia Label'][i].cpu().item()
                            task_probs_patient['Pneumonia Label'] = task_probs['Pneumonia Label'][i].cpu().item()
                            task_preds_patient['Pneumonia Label'] = task_preds['Pneumonia Label'][i].cpu().item()
                    
                    if 'Covid Label' in active_tasks:
                        task_labels['Covid Label'] = covid_labels[i].cpu().item()
                        if 'Covid Label' in task_logits:
                            task_logits_patient['Covid Label'] = task_logits['Covid Label'][i].cpu().item()
                            task_probs_patient['Covid Label'] = task_probs['Covid Label'][i].cpu().item()
                            task_preds_patient['Covid Label'] = task_preds['Covid Label'][i].cpu().item()
                    
                    # Get number of valid sites for this patient
                    num_sites = site_masks[i].sum().item()
                    stats['total_sites'] += num_sites
                    
                    # Store patient-level complex data
                    if save_complex_data:
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
                        'batch_idx': batch_idx,
                        'patient_id': patient_id,
                        'num_valid_sites': num_sites,
                    }
                    
                    # Add task-specific fields
                    for task_name in active_tasks:
                        prefix = task_name.lower().replace(' ', '_')
                        patient_record[f'{prefix}_label'] = task_labels.get(task_name, -1)
                        patient_record[f'{prefix}_logit'] = task_logits_patient.get(task_name, float('nan'))
                        patient_record[f'{prefix}_prob'] = task_probs_patient.get(task_name, float('nan'))
                        patient_record[f'{prefix}_pred'] = task_preds_patient.get(task_name, float('nan'))
                    
                    patient_records.append(patient_record)
                    
                    # Process each valid site for this patient
                    for s in range(num_sites):
                        site_idx = site_indices[i, s].cpu().item()
                        site_finding = site_findings[i, s].cpu().numpy()
                        site_mask_value = site_masks[i, s].cpu().item()
                        
                        # Create site-level record
                        site_record = {
                            'batch_idx': batch_idx,
                            'patient_id': patient_id,
                            'site_position': s,  # Position within patient's sites (0, 1, 2, ...)
                            'site_index': site_idx,  # Anatomical site index
                            'site_mask': site_mask_value,
                        }
                        
                        # Add task labels and predictions for reference
                        for task_name in active_tasks:
                            prefix = task_name.lower().replace(' ', '_')
                            site_record[f'{prefix}_label'] = task_labels.get(task_name, -1)
                            site_record[f'{prefix}_logit'] = task_logits_patient.get(task_name, float('nan'))
                            site_record[f'{prefix}_prob'] = task_probs_patient.get(task_name, float('nan'))
                            site_record[f'{prefix}_pred'] = task_preds_patient.get(task_name, float('nan'))
                        
                        # Add site findings (ground truth pathology labels)
                        for p_idx, p_name in enumerate(pathology_names):
                            if p_idx < len(site_finding):
                                site_record[f'{p_name}_finding'] = site_finding[p_idx]
                        
                        # Add site pathology predictions if available
                        if use_pathology_loss and pathology_scores is not None:
                            site_path_scores = pathology_scores[i, s].cpu().numpy()
                            
                            for p_idx, p_name in enumerate(pathology_names):
                                if p_idx < len(site_path_scores):
                                    site_record[f'{p_name}_logit'] = site_path_scores[p_idx]
                                    site_record[f'{p_name}_prob'] = 1 / (1 + np.exp(-site_path_scores[p_idx]))
                                    site_record[f'{p_name}_pred'] = int(site_path_scores[p_idx] > 0)
                        
                        # Add MIL attention for this site if available
                        if mil_attention is not None:
                            site_record['mil_attention'] = mil_attention[i, s].cpu().item()
                        
                        # Store site-level complex data
                        if save_complex_data:
                            if site_features is not None:
                                complex_data['site_features'][(patient_id, site_idx)] = site_features[i, s].cpu().numpy()
                            
                            if site_rl_data is not None and i < len(site_rl_data) and s < len(site_rl_data[i]):
                                # Convert RL data to serializable format
                                rl_data_item = site_rl_data[i][s]
                                serializable_rl_data = {}
                                
                                for key, value in rl_data_item.items():
                                    if isinstance(value, torch.Tensor):
                                        serializable_rl_data[key] = value.cpu().numpy()
                                    else:
                                        serializable_rl_data[key] = value
                                
                                complex_data['site_rl_data'][(patient_id, site_idx)] = serializable_rl_data
                        
                        site_records.append(site_record)
                
                # Memory management
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"OOM during evaluation at batch {batch_idx}, skipping")
                    stats['failed_batches'] += 1
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error at batch {batch_idx}: {e}")
                stats['failed_batches'] += 1
                continue
    
    logger.info("\n=== Evaluation Statistics ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Create DataFrames
    logger.info(f"\nCreating DataFrames from {len(patient_records)} patient records and {len(site_records)} site records...")
    
    patient_df = pd.DataFrame(patient_records)
    site_df = pd.DataFrame(site_records)
    
    logger.info(f"Patient DataFrame shape: {patient_df.shape}")
    logger.info(f"Site DataFrame shape: {site_df.shape}")
    logger.info(f"Patient DataFrame columns: {list(patient_df.columns)}")
    logger.info(f"Site DataFrame columns: {list(site_df.columns)}")
    
    # Calculate metrics for each active task
    metrics = {}
    
    for task_name in active_tasks:
        if len(patient_df) > 0:
            prefix = task_name.lower().replace(' ', '_')
            label_col = f'{prefix}_label'
            prob_col = f'{prefix}_prob'
            pred_col = f'{prefix}_pred'
            
            if label_col in patient_df.columns and prob_col in patient_df.columns:
                # Filter out invalid labels
                valid_mask = patient_df[label_col] >= 0
                
                if valid_mask.sum() > 0:
                    patient_targets = patient_df.loc[valid_mask, label_col].values
                    patient_probs = patient_df.loc[valid_mask, prob_col].values
                    patient_preds = patient_df.loc[valid_mask, pred_col].values
                    
                    logger.info(f"\n{task_name} targets distribution: {np.bincount(patient_targets.astype(int))}")
                    logger.info(f"{task_name} predictions distribution: {np.bincount(patient_preds.astype(int))}")
                    
                    try:
                        task_metrics = calculate_metrics(patient_targets, patient_preds, probabilities=patient_probs)
                        # Add task prefix to metrics
                        for key, value in task_metrics.items():
                            metrics[f'{task_name}_{key}'] = value
                        logger.info(f"{task_name} metrics calculated: {list(task_metrics.keys())}")
                    except Exception as e:
                        logger.warning(f"Could not calculate metrics for {task_name}: {e}")
    
    print(f"\nComplex data summary:")
    for key, data_dict in complex_data.items():
        if isinstance(data_dict, dict):
            print(f"  {key}: {len(data_dict)} items")
        else:
            print(f"  {key}: {type(data_dict)}")
    
    return patient_df, site_df, complex_data, metrics


def calculate_metrics(targets, predictions, probabilities=None):
    """Calculate standard classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, zero_division=0)
        
        # Calculate specificity manually
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        if probabilities is not None and len(np.unique(targets)) > 1:
            metrics['auc'] = roc_auc_score(targets, probabilities)
            metrics['auprc'] = average_precision_score(targets, probabilities)
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Provide default values
        metrics['accuracy'] = 0.0
        metrics['auc'] = 0.5
    
    return metrics


def save_comprehensive_results(patient_df, site_df, complex_data, metrics, output_dir, split_name, experiment_name, fold_num):
    """Save comprehensive evaluation results in multiple formats."""
    
    logger.info(f"=== Saving Comprehensive Results to {output_dir} ===")
    
    # Create filename base
    if fold_num is not None:
        filename_base = f'{split_name}_{experiment_name}_fold{fold_num}'
    else:
        filename_base = f'{split_name}_{experiment_name}'
    
    logger.info(f"Filename base: {filename_base}")
    
    saved_files = {}
    
    # 1. Save patient-level dataframe
    patient_csv_path = os.path.join(output_dir, f'{filename_base}_patients.csv')
    patient_df.to_csv(patient_csv_path, index=False)
    saved_files['patient_csv'] = patient_csv_path
    logger.info(f"Patient data saved: {patient_csv_path}")
    
    # 2. Save site-level dataframe
    site_csv_path = os.path.join(output_dir, f'{filename_base}_sites.csv')
    site_df.to_csv(site_csv_path, index=False)
    saved_files['site_csv'] = site_csv_path
    logger.info(f"Site data saved: {site_csv_path}")
    
    # 3. Save complex data using HDF5
    hdf5_path = os.path.join(output_dir, f'{filename_base}_complex_data.h5')
    
    try:
        with h5py.File(hdf5_path, 'w') as f:
            # Add metadata
            f.attrs['split'] = split_name
            f.attrs['experiment_name'] = experiment_name
            if fold_num is not None:
                f.attrs['fold'] = fold_num
            f.attrs['num_patients'] = len(patient_df)
            f.attrs['num_sites'] = len(site_df)
            
            # Patient features
            if complex_data['patient_features']:
                patient_grp = f.create_group('patient_features')
                for patient_id, features in complex_data['patient_features'].items():
                    patient_grp.create_dataset(str(patient_id), data=features)
                logger.info(f"    Patient features: {len(complex_data['patient_features'])} patients")
            
            # MIL attention
            if complex_data['mil_attention']:
                mil_grp = f.create_group('mil_attention')
                for patient_id, attention in complex_data['mil_attention'].items():
                    mil_grp.create_dataset(str(patient_id), data=attention)
                logger.info(f"    MIL attention: {len(complex_data['mil_attention'])} patients")
            
            # Site features
            if complex_data['site_features']:
                site_grp = f.create_group('site_features')
                for (patient_id, site_idx), features in complex_data['site_features'].items():
                    dataset_name = f"{patient_id}_site_{site_idx}"
                    site_grp.create_dataset(dataset_name, data=features)
                logger.info(f"    Site features: {len(complex_data['site_features'])} sites")
            
            # Task logits
            if complex_data['task_logits']:
                task_grp = f.create_group('task_logits')
                for task_name, task_data in complex_data['task_logits'].items():
                    task_subgrp = task_grp.create_group(task_name.replace(' ', '_'))
                    for patient_id, logits in task_data.items():
                        task_subgrp.create_dataset(str(patient_id), data=logits)
                logger.info(f"    Task logits: {len(complex_data['task_logits'])} tasks")
            
            # Pathology scores
            if 'pathology_scores' in complex_data and complex_data['pathology_scores']:
                pathology_grp = f.create_group('pathology_scores')
                for patient_id, scores in complex_data['pathology_scores'].items():
                    pathology_grp.create_dataset(str(patient_id), data=scores)
                logger.info(f"    Pathology scores: {len(complex_data['pathology_scores'])} patients")
        
        saved_files['complex_hdf5'] = hdf5_path
        logger.info(f"Complex data saved: {hdf5_path}")
    except Exception as e:
        logger.error(f"HDF5: Failed to save ({e})")
    
    # 4. Save metrics as JSON
    metrics_path = os.path.join(output_dir, f'{filename_base}_metrics.json')
    try:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = value.item()
            else:
                json_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        saved_files['metrics_json'] = metrics_path
        logger.info(f"Metrics saved: {metrics_path}")
    except Exception as e:
        logger.error(f"Metrics: Failed to save ({e})")
    
    logger.info(f"\nAll results saved with base name: {filename_base}")
    return saved_files


def run_comprehensive_evaluation(test_config):
    """Run the comprehensive evaluation pipeline for multi-task model."""
    
    logger.info("=== Loading Multi-Task Model and Data ===")

    # Set device
    device = torch.device(f'cuda:{test_config["gpu_id"]}' if test_config['gpu_id'] >= 0 and torch.cuda.is_available() else 'cpu')
    if test_config['gpu_id'] >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(test_config['gpu_id'])

    # Create output directory
    os.makedirs(test_config['output_dir'], exist_ok=True)

    # Load configuration using the actual config system
    config = load_config(config_file=test_config['config_path'])
    
    logger.info("Config loaded successfully")
    logger.info(f"  Active tasks: {getattr(config, 'active_tasks', 'Not specified')}")
    logger.info(f"  Selection strategy: {getattr(config, 'selection_strategy', 'Not specified')}")
    logger.info(f"  Use pathology loss: {getattr(config, 'use_pathology_loss', 'Not specified')}")
    logger.info(f"  Experiment dir: {getattr(config, 'experiment_dir', 'Not specified')}")
    
    # Extract multi-task configuration
    active_tasks = getattr(config, 'active_tasks', ['TB Label'])
    use_pathology_loss = getattr(config, 'use_pathology_loss', True)
    
    logger.info(f"Active tasks: {active_tasks}")
    logger.info(f"Use pathology loss: {use_pathology_loss}")
    
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

    # Initialize model
    model = MultiTaskModel(config)

    # Use model_path from config if available, otherwise use test_config
    model_path = getattr(config, 'model_path', None) or test_config['model_path']
    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Print checkpoint info
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model_state_dict from checkpoint")
        
        # Print additional checkpoint info if available
        if 'epoch' in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            logger.info(f"  Best metric: {checkpoint['best_metric']:.4f}")
        if 'active_tasks' in checkpoint:
            logger.info(f"  Checkpoint active tasks: {checkpoint['active_tasks']}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded state dict directly from checkpoint")
        
    model = model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} and set to eval mode")

    data_module.setup(stage='patient_level')

    def run_split_evaluation(split_name, dataloader):
        """Helper function to run evaluation on a specific split."""
        patient_df, site_df, complex_data, metrics = evaluate_model_comprehensive(
            model, dataloader, device, active_tasks, use_pathology_loss, test_config['save_complex_data']
        )
        
        logger.info(f"=== Saving {split_name} Results ===")
        
        # Save results
        saved_files = save_comprehensive_results(
            patient_df, site_df, complex_data, metrics, 
            test_config['output_dir'], split_name, 'multi_task_model', test_config['fold']
        )
        
        return patient_df, site_df, complex_data, metrics, saved_files

    # Run evaluation based on split configuration
    if test_config['split'] == 'train':
        train_dataloader = data_module.patient_level_dataloader('train')
        return run_split_evaluation('train', train_dataloader)

    elif test_config['split'] == 'val':
        val_dataloader = data_module.patient_level_dataloader('val')
        return run_split_evaluation('val', val_dataloader)

    elif test_config['split'] == 'test':
        test_dataloader = data_module.patient_level_dataloader('test')
        return run_split_evaluation('test', test_dataloader)

    else:  # 'all' or any other value
        train_dataloader = data_module.patient_level_dataloader('train')
        val_dataloader = data_module.patient_level_dataloader('val')
        test_dataloader = data_module.patient_level_dataloader('test')

        # Save Train Data

        
        # Save Test Data
        logger.info("Running Test Data")
        test_results = run_split_evaluation('test', test_dataloader)
        
        # Save Val Data
        logger.info("Running Val Data")
        val_results = run_split_evaluation('val', val_dataloader)

        logger.info("Running Train Data")
        train_results = run_split_evaluation('train', train_dataloader)
        
        return test_results  # Return test results as primary


def analyze_saved_results(patient_df, site_df, active_tasks, use_pathology_loss=True):
    """Analyze the saved results and generate insights."""
    
    logger.info("=== Multi-Task Data Analysis ===")
    
    # Patient-level analysis for each task
    logger.info(f"\nPatient-level Analysis:")
    logger.info(f"  Total patients: {len(patient_df)}")
    
    for task_name in active_tasks:
        prefix = task_name.lower().replace(' ', '_')
        label_col = f'{prefix}_label'
        prob_col = f'{prefix}_prob'
        
        if label_col in patient_df.columns:
            valid_mask = patient_df[label_col] >= 0
            if valid_mask.sum() > 0:
                positive_count = patient_df.loc[valid_mask, label_col].sum()
                total_count = valid_mask.sum()
                avg_prob = patient_df.loc[valid_mask, prob_col].mean() if prob_col in patient_df.columns else 0
                logger.info(f"  {task_name}: {positive_count}/{total_count} ({positive_count/total_count*100:.1f}%)")
                logger.info(f"    Average probability: {avg_prob:.3f}")
    
    # Site-level analysis
    logger.info(f"\nSite-level Analysis:")
    logger.info(f"  Total sites: {len(site_df)}")
    logger.info(f"  Sites per patient: {len(site_df) / len(patient_df):.1f}")
    logger.info(f"  Unique anatomical sites: {site_df['site_index'].nunique()}")
    logger.info(f"  Site index range: {site_df['site_index'].min()}-{site_df['site_index'].max()}")
    
    # Pathology analysis
    if use_pathology_loss:
        pathology_cols = [col for col in site_df.columns if col.endswith('_finding')]
        logger.info(f"\nPathology Findings (Ground Truth):")
        for col in pathology_cols:
            pathology_name = col.replace('_finding', '').replace('_', ' ').title()
            positive_sites = site_df[col].sum()
            total_sites = len(site_df)
            logger.info(f"  {pathology_name}: {positive_sites}/{total_sites} ({positive_sites/total_sites*100:.1f}%)")
        
        # Prediction analysis
        pathology_pred_cols = [col for col in site_df.columns if col.endswith('_pred')]
        if pathology_pred_cols:
            logger.info(f"\nPathology Predictions:")
            for col in pathology_pred_cols:
                pathology_name = col.replace('_pred', '').replace('_', ' ').title()
                predicted_positive = site_df[col].sum()
                total_sites = len(site_df)
                logger.info(f"  {pathology_name}: {predicted_positive}/{total_sites} ({predicted_positive/total_sites*100:.1f}%)")
    
    return patient_df, site_df


def create_cross_fold_summary(successful_folds, results_dir):
    """Create a summary across all successful folds."""
    logger.info("Creating cross-fold summary...")
    
    all_metrics = []
    
    for fold in successful_folds:
        # Try to load metrics from each fold
        for split in ['train', 'val', 'test']:
            metrics_file = os.path.join(results_dir, f'{split}_multi_task_model_fold{fold}_metrics.json')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        fold_metrics = json.load(f)
                    
                    # Add fold and split information
                    fold_metrics['fold'] = fold
                    fold_metrics['split'] = split
                    all_metrics.append(fold_metrics)
                    
                except Exception as e:
                    logger.warning(f"Could not load metrics from {metrics_file}: {e}")
    
    if all_metrics:
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(all_metrics)
        summary_file = os.path.join(results_dir, 'cross_fold_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Cross-fold summary saved to: {summary_file}")
        
        # Print average metrics
        logger.info("\nAverage Metrics Across Folds:")
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        avg_metrics = summary_df.groupby('split')[numeric_cols].mean()
        
        for split in ['train', 'val', 'test']:
            if split in avg_metrics.index:
                logger.info(f"\n{split.upper()} (avg across {len(successful_folds)} folds):")
                for col in ['TB Label_auc', 'TB Label_accuracy', 'Pneumonia Label_auc', 'Pneumonia Label_accuracy']:
                    if col in avg_metrics.columns:
                        logger.info(f"  {col}: {avg_metrics.loc[split, col]:.4f}")
    else:
        logger.warning("No metrics files found for cross-fold summary")