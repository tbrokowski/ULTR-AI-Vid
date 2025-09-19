import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')

# Import our custom modules
from dataset import create_dataloaders, analyze_dataset_distribution
from models import create_model, MCDropoutModel, DeepEnsemble

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        return False

def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive evaluation metrics"""
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. Cannot compute AUC.")
        return {
            'auc': 0.5,
            'ap': float(np.mean(y_true)),
            'balanced_accuracy': 0.5,
            'ece': float('nan'),
            'brier_score': float('nan')
        }
    
    # Classification metrics
    try:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        auc = ap = balanced_acc = 0.5
    
    # Calibration metrics
    try:
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ece = np.mean(np.abs(fraction_pos - mean_pred))  # Expected Calibration Error
    except Exception as e:
        logger.warning(f"Error computing calibration: {e}")
        ece = float('nan')
    
    # Brier score
    try:
        brier_score = np.mean((y_prob - y_true) ** 2)
    except:
        brier_score = float('nan')
    
    return {
        'auc': float(auc),
        'ap': float(ap),
        'balanced_accuracy': float(balanced_acc),
        'ece': float(ece) if not np.isnan(ece) else 0.0,
        'brier_score': float(brier_score) if not np.isnan(brier_score) else 0.0
    }

def train_epoch(model, dataloader, criterion, optimizer, device, model_type='standard', epoch=0):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    predictions = []
    probabilities = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        try:
            clinical_features = batch['clinical_features'].to(device)
            patient_embeddings = batch['patient_embedding'].to(device)
            labels = batch['tb_label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass based on model type
            if model_type == 'contrastive':
                outputs = model(clinical_features, patient_embeddings)
                loss = criterion(outputs, labels)
                # Add contrastive loss
                contrastive_loss = model.contrastive_loss(clinical_features, patient_embeddings, labels)
                loss += 0.1 * contrastive_loss  # Weight contrastive loss
            elif model_type == 'variational':
                pred_mu, pred_logvar, kl_loss = model(clinical_features, patient_embeddings)
                reconstruction_loss = criterion(pred_mu, labels)
                loss = reconstruction_loss + 0.001 * kl_loss  # Weight KL divergence
                outputs = pred_mu
            elif model_type == 'evidential':
                outputs, uncertainty_dict = model(clinical_features, patient_embeddings)
                # Use evidential loss
                loss = model.evidential_loss(outputs, labels, uncertainty_dict, epoch=epoch)
            elif model_type == 'mud':
                outputs, uncertainty_dict = model(clinical_features, patient_embeddings)
                loss = criterion(outputs, labels)
                # Add uncertainty regularization
                unc_reg = torch.mean(uncertainty_dict['total_uncertainty'])
                loss += 0.01 * unc_reg
            elif model_type == 'mc_frequency':
                # MC frequency model uses base model forward pass
                outputs = model(clinical_features, patient_embeddings)
                loss = criterion(outputs, labels)
            else:
                outputs = model(clinical_features, patient_embeddings)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            probabilities.extend(probs.flatten())
            predictions.extend(preds.flatten())
            targets.extend(labels.cpu().numpy().flatten())
            
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            continue
    
    if len(targets) == 0:
        logger.error("No valid training batches!")
        return 0.0, {'auc': 0.5, 'ap': 0.5, 'balanced_accuracy': 0.5, 'ece': 0.0, 'brier_score': 0.0}
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(targets, predictions, probabilities)
    
    return avg_loss, metrics

def validate_epoch(model, dataloader, criterion, device, model_type='standard'):
    """Validate for one epoch"""
    
    model.eval()
    total_loss = 0.0
    predictions = []
    probabilities = []
    targets = []
    patient_ids = []
    uncertainty_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            try:
                clinical_features = batch['clinical_features'].to(device)
                patient_embeddings = batch['patient_embedding'].to(device)
                labels = batch['tb_label'].to(device).unsqueeze(1)
                
                # Forward pass based on model type
                if model_type == 'variational':
                    pred_mu, pred_logvar, kl_loss = model(clinical_features, patient_embeddings)
                    loss = criterion(pred_mu, labels) + 0.001 * kl_loss
                    outputs = pred_mu
                    # Uncertainty from variance
                    uncertainty = torch.exp(pred_logvar).cpu().numpy()
                elif model_type == 'evidential':
                    outputs, uncertainty_dict = model(clinical_features, patient_embeddings)
                    loss = model.evidential_loss(outputs, labels, uncertainty_dict)
                    uncertainty = uncertainty_dict['total_uncertainty'].cpu().numpy()
                elif model_type == 'mud':
                    outputs, uncertainty_dict = model(clinical_features, patient_embeddings)
                    loss = criterion(outputs, labels)
                    uncertainty = uncertainty_dict['total_uncertainty'].cpu().numpy()
                elif model_type == 'mc_frequency':
                    outputs = model(clinical_features, patient_embeddings)
                    loss = criterion(outputs, labels)
                    uncertainty = np.zeros(outputs.shape[0])  # MC frequency uses special prediction method
                else:
                    outputs = model(clinical_features, patient_embeddings)
                    loss = criterion(outputs, labels)
                    uncertainty = np.zeros(outputs.shape[0])  # No uncertainty for standard models
                
                total_loss += loss.item()
                
                # Store predictions
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                probabilities.extend(probs.flatten())
                predictions.extend(preds.flatten())
                targets.extend(labels.cpu().numpy().flatten())
                patient_ids.extend(batch['patient_id'])
                uncertainty_scores.extend(uncertainty.flatten())
                
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue
    
    if len(targets) == 0:
        logger.error("No valid validation batches!")
        return 0.0, {'auc': 0.5, 'ap': 0.5, 'balanced_accuracy': 0.5, 'ece': 0.0, 'brier_score': 0.0}, {}
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(targets, predictions, probabilities)
    
    return avg_loss, metrics, {
        'patient_ids': patient_ids,
        'predictions': predictions,
        'probabilities': probabilities,
        'targets': targets,
        'uncertainties': uncertainty_scores
    }

def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, 
                model_type='standard', save_path=None):
    """Train a single model with early stopping"""
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)  # Disable verbose for cleaner output
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_ap': [], 'val_ap': []
    }
    
    best_val_auc = 0.0
    
    # Training progress bar
    epoch_pbar = tqdm(range(epochs), desc=f"Training {model_type}", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    for epoch in epoch_pbar:
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, model_type, epoch)
        
        # Validation
        val_loss, val_metrics, val_predictions = validate_epoch(model, val_loader, criterion, device, model_type)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_ap'].append(train_metrics['ap'])
        history['val_ap'].append(val_metrics['ap'])
        
        # Update progress bar with metrics
        epoch_pbar.set_postfix({
            'Train_Loss': f'{train_loss:.3f}',
            'Val_Loss': f'{val_loss:.3f}', 
            'Val_AUC': f'{val_metrics["auc"]:.3f}',
            'Best_AUC': f'{best_val_auc:.3f}'
        })
        
        # Learning rate scheduling
        scheduler.step(val_metrics['auc'])
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.3f}',
                'Val_Loss': f'{val_loss:.3f}', 
                'Val_AUC': f'{val_metrics["auc"]:.3f}',
                'Best_AUC': f'{best_val_auc:.3f} ⭐'
            })
            if save_path:
                try:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_auc': val_metrics['auc'],
                        'history': history
                    }, save_path)
                except Exception as e:
                    logger.error(f"Error saving model: {e}")
        
        # Early stopping
        if early_stopping(val_metrics['auc'], model):
            epoch_pbar.set_description(f"Training {model_type} (Early Stop)")
            break
    
    epoch_pbar.close()
    logger.info(f"Training completed. Best Val AUC: {best_val_auc:.4f}")
    
    return history, best_val_auc

def evaluate_model_with_uncertainty(model, dataloader, device, model_type='standard', 
                                  num_mc_samples=100):
    """Evaluate model with uncertainty estimation"""
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_uncertainties = []
    all_targets = []
    all_patient_ids = []
    
    if model_type == 'mc_dropout':
        # Monte Carlo Dropout
        for batch in tqdm(dataloader, desc="MC Evaluation", leave=False):
            try:
                clinical_features = batch['clinical_features'].to(device)
                patient_embeddings = batch['patient_embedding'].to(device)
                labels = batch['tb_label'].cpu().numpy()
                
                mean_pred, uncertainty = model.predict_with_uncertainty(
                    clinical_features, patient_embeddings, num_mc_samples
                )
                
                all_probabilities.extend(mean_pred.flatten())
                all_predictions.extend((mean_pred > 0.5).astype(int).flatten())
                all_uncertainties.extend(uncertainty.flatten())
                all_targets.extend(labels.flatten())
                all_patient_ids.extend(batch['patient_id'])
            except Exception as e:
                logger.error(f"Error in MC dropout evaluation: {e}")
                continue
    
    elif model_type == 'mc_frequency':
        # Monte Carlo Frequency Analysis
        for batch in tqdm(dataloader, desc="MC Frequency Evaluation", leave=False, position=2):
            try:
                clinical_features = batch['clinical_features'].to(device)
                patient_embeddings = batch['patient_embedding'].to(device)
                labels = batch['tb_label'].cpu().numpy()
                
                mean_pred, uncertainty_dict = model.predict_with_uncertainty(
                    clinical_features, patient_embeddings, num_mc_samples
                )
                
                # Use composite uncertainty as primary uncertainty measure
                uncertainty = uncertainty_dict['composite_uncertainty']
                
                all_probabilities.extend(mean_pred.flatten())
                all_predictions.extend((mean_pred > 0.5).astype(int).flatten())
                all_uncertainties.extend(uncertainty.flatten())
                all_targets.extend(labels.flatten())
                all_patient_ids.extend(batch['patient_id'])
            except Exception as e:
                logger.error(f"Error in MC frequency evaluation: {e}")
                continue
    
    else:
        # Standard evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Standard Evaluation", leave=False, position=2):
                try:
                    clinical_features = batch['clinical_features'].to(device)
                    patient_embeddings = batch['patient_embedding'].to(device)
                    labels = batch['tb_label'].cpu().numpy()
                    
                    if model_type == 'mud':
                        outputs, uncertainty_dict = model(clinical_features, patient_embeddings)
                        uncertainty = uncertainty_dict['total_uncertainty'].cpu().numpy()
                    elif model_type == 'variational':
                        pred_mu, pred_logvar, _ = model(clinical_features, patient_embeddings)
                        outputs = pred_mu
                        uncertainty = torch.exp(pred_logvar).cpu().numpy()
                    elif model_type == 'evidential':
                        outputs, uncertainty_dict = model(clinical_features, patient_embeddings)
                        uncertainty = uncertainty_dict['total_uncertainty'].cpu().numpy()
                    else:
                        outputs = model(clinical_features, patient_embeddings)
                        uncertainty = np.zeros(outputs.shape[0])
                    
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    all_probabilities.extend(probs.flatten())
                    all_predictions.extend(preds.flatten())
                    all_uncertainties.extend(uncertainty.flatten())
                    all_targets.extend(labels.flatten())
                    all_patient_ids.extend(batch['patient_id'])
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue
    
    if len(all_targets) == 0:
        logger.error("No valid evaluation batches!")
        return {
            'patient_ids': [],
            'predictions': [],
            'probabilities': [],
            'targets': [],
            'uncertainties': [],
            'metrics': {'auc': 0.5, 'ap': 0.5, 'balanced_accuracy': 0.5, 'ece': 0.0, 'brier_score': 0.0}
        }
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_predictions, all_probabilities)
    
    return {
        'patient_ids': all_patient_ids,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets,
        'uncertainties': all_uncertainties,
        'metrics': metrics
    }

def run_cross_validation(config):
    """Run complete cross-validation experiments"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Results storage
    all_results = {}
    
    # Model configurations to test
    model_configs = {
        # Non-multimodal baselines
        'clinical_only': {'model_type': 'clinical_only'},
        'imaging_only': {'model_type': 'imaging_only'},
        
        # Simple fusion methods
        'simple_concat': {'model_type': 'simple_concat'},
        'early_fusion': {'model_type': 'early_fusion'},
        'late_fusion': {'model_type': 'late_fusion'},
        
        # Advanced fusion methods
        'contrastive': {'model_type': 'contrastive'},
        'cross_attention': {'model_type': 'cross_attention'},
        'bilinear': {'model_type': 'bilinear'},
        
        # Novel methods
        'ccg': {'model_type': 'ccg'},  # Clinical Contextual Gating
        
        # Uncertainty estimation methods
        'variational': {'model_type': 'variational'},
        'mud': {'model_type': 'mud'},  # Multimodal Uncertainty Decomposition
        'evidential': {'model_type': 'evidential'},  # Evidential Deep Learning
        'mc_frequency': {'model_type': 'mc_frequency', 'base_model_type': 'simple_concat'},  # MC Frequency Analysis
    }
    
    # Run experiments for each model type
    for model_name, model_config in tqdm(model_configs.items(), desc="Training Models", position=0):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name.upper()} Model")
        logger.info(f"{'='*50}")
        
        fold_results = []
        
        # Cross-validation loop with progress bar
        fold_pbar = tqdm(range(config['num_folds']), desc=f"{model_name} Folds", position=1, leave=False)
        
        for fold in fold_pbar:
            fold_pbar.set_description(f"{model_name} Fold {fold+1}/{config['num_folds']}")
            
            try:
                # Create dataloaders
                train_loader, val_loader, test_loader, scaler = create_dataloaders(
                    clinical_data_path=config['clinical_data_path'],
                    base_h5_path=config['base_h5_path'],
                    base_site_path=config['base_site_path'],
                    fold_num=fold,
                    base_fold_path=config.get('base_fold_path'),
                    batch_size=config['batch_size']
                )
                
                # Analyze dataset
                train_stats = analyze_dataset_distribution(train_loader.dataset)
                val_stats = analyze_dataset_distribution(val_loader.dataset)
                test_stats = analyze_dataset_distribution(test_loader.dataset)
                
                # Get feature dimensions and schema
                sample = train_loader.dataset[0]
                clinical_dim = sample['clinical_features'].shape[0]
                embedding_dim = sample['patient_embedding'].shape[0]
                feature_schema = sample.get('feature_schema', None)
                feature_order = getattr(train_loader.dataset, 'feature_order', None)
                
                # Create model with advanced clinical processing
                model = create_model(
                    model_config['model_type'], 
                    clinical_dim, 
                    embedding_dim,
                    feature_schema=feature_schema,
                    feature_order=feature_order,
                    hidden_dim=config['hidden_dim'],
                    dropout=config['dropout']
                ).to(device)
                
                # Training
                save_path = os.path.join(config['save_dir'], f"{model_name}_fold{fold}_best.pth")
                history, best_val_auc = train_model(
                    model, train_loader, val_loader, device,
                    epochs=config['epochs'],
                    lr=config['learning_rate'],
                    model_type=model_config['model_type'],
                    save_path=save_path
                )
                
                # Load best model for evaluation
                if os.path.exists(save_path):
                    try:
                        checkpoint = torch.load(save_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                    except Exception as e:
                        logger.warning(f"Could not load best model: {e}")
                
                # Test evaluation
                test_results = evaluate_model_with_uncertainty(
                    model, test_loader, device, 
                    model_type=model_config['model_type']
                )
                
                # Update progress bar with results
                fold_pbar.set_postfix({
                    'Val_AUC': f'{best_val_auc:.3f}',
                    'Test_AUC': f'{test_results["metrics"]["auc"]:.3f}'
                })
                
                # Save predictions
                if test_results['patient_ids']:
                    predictions_df = pd.DataFrame({
                        'patient_id': test_results['patient_ids'],
                        'tb_label': test_results['targets'],
                        'tb_prediction': test_results['predictions'],
                        'tb_probability': test_results['probabilities'],
                        'uncertainty': test_results['uncertainties'],
                        'fold': fold,
                        'model': model_name
                    })
                    
                    pred_save_path = os.path.join(config['save_dir'], f"{model_name}_fold{fold}_predictions.csv")
                    predictions_df.to_csv(pred_save_path, index=False)
                else:
                    pred_save_path = None
                
                # Store fold results
                fold_result = {
                    'fold': fold,
                    'model': model_name,
                    'train_stats': train_stats,
                    'val_stats': val_stats,
                    'test_stats': test_stats,
                    'best_val_auc': best_val_auc,
                    'test_metrics': test_results['metrics'],
                    'history': history,
                    'predictions_file': pred_save_path
                }
                
                fold_results.append(fold_result)
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                # Add a failed fold result
                fold_results.append({
                    'fold': fold,
                    'model': model_name,
                    'error': str(e),
                    'test_metrics': {'auc': 0.0, 'ap': 0.0, 'balanced_accuracy': 0.0}
                })
                fold_pbar.set_postfix({'Status': 'FAILED'})
                continue
        
        fold_pbar.close()
        
        # Aggregate results across folds (only successful folds)
        successful_folds = [r for r in fold_results if 'error' not in r]
        
        if successful_folds:
            test_aucs = [r['test_metrics']['auc'] for r in successful_folds]
            test_aps = [r['test_metrics']['ap'] for r in successful_folds]
            test_bal_accs = [r['test_metrics']['balanced_accuracy'] for r in successful_folds]
            
            model_summary = {
                'model': model_name,
                'successful_folds': len(successful_folds),
                'total_folds': len(fold_results),
                'test_auc_mean': np.mean(test_aucs),
                'test_auc_std': np.std(test_aucs),
                'test_ap_mean': np.mean(test_aps),
                'test_ap_std': np.std(test_aps),
                'test_bal_acc_mean': np.mean(test_bal_accs),
                'test_bal_acc_std': np.std(test_bal_accs),
                'fold_results': fold_results
            }
            
            logger.info(f"\n{model_name.upper()} Summary:")
            logger.info(f"  Successful folds: {model_summary['successful_folds']}/{model_summary['total_folds']}")
            logger.info(f"  Test AUC: {model_summary['test_auc_mean']:.4f} ± {model_summary['test_auc_std']:.4f}")
            logger.info(f"  Test AP: {model_summary['test_ap_mean']:.4f} ± {model_summary['test_ap_std']:.4f}")
            logger.info(f"  Test Bal Acc: {model_summary['test_bal_acc_mean']:.4f} ± {model_summary['test_bal_acc_std']:.4f}")
        else:
            model_summary = {
                'model': model_name,
                'successful_folds': 0,
                'total_folds': len(fold_results),
                'test_auc_mean': 0.0,
                'test_auc_std': 0.0,
                'test_ap_mean': 0.0,
                'test_ap_std': 0.0,
                'test_bal_acc_mean': 0.0,
                'test_bal_acc_std': 0.0,
                'fold_results': fold_results
            }
            
            logger.warning(f"\n{model_name.upper()} FAILED - No successful folds")
        
        all_results[model_name] = model_summary
    
    # Save complete results
    results_path = os.path.join(config['save_dir'], 'complete_results.json')
    try:
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(all_results), f, indent=2)
        logger.info(f"Results saved to: {results_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Create summary table
    create_results_summary(all_results, config['save_dir'])
    
    # Print final progress summary
    print("\n" + "="*80)
    print("🎯 EXPERIMENT COMPLETION SUMMARY")
    print("="*80)
    
    successful_models = 0
    total_models = len(all_results)
    
    for model_name, results in all_results.items():
        if results['successful_folds'] > 0:
            successful_models += 1
            status = f"✅ {results['successful_folds']}/{results['total_folds']} folds"
            auc = f"AUC: {results['test_auc_mean']:.3f}±{results['test_auc_std']:.3f}"
        else:
            status = "❌ FAILED"
            auc = "N/A"
        print(f"  {model_name:15s} | {status:15s} | {auc}")
    
    print("="*80)
    print(f"📊 Overall Success: {successful_models}/{total_models} models completed successfully")
    print(f"💾 Results saved to: {config['save_dir']}")
    print("="*80)
    
    return all_results

def create_results_summary(results, save_dir):
    """Create a summary table of all results"""
    
    summary_data = []
    for model_name, model_results in results.items():
        if model_results['successful_folds'] > 0:
            summary_data.append({
                'Model': model_name,
                'Successful_Folds': f"{model_results['successful_folds']}/{model_results['total_folds']}",
                'Test AUC': f"{model_results['test_auc_mean']:.3f} ± {model_results['test_auc_std']:.3f}",
                'Test AP': f"{model_results['test_ap_mean']:.3f} ± {model_results['test_ap_std']:.3f}",
                'Test Balanced Acc': f"{model_results['test_bal_acc_mean']:.3f} ± {model_results['test_bal_acc_std']:.3f}"
            })
        else:
            summary_data.append({
                'Model': model_name,
                'Successful_Folds': f"0/{model_results['total_folds']}",
                'Test AUC': "FAILED",
                'Test AP': "FAILED",
                'Test Balanced Acc': "FAILED"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    try:
        summary_path = os.path.join(save_dir, 'results_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(summary_df.to_string(index=False))
        logger.info(f"\nResults saved to: {summary_path}")
    except Exception as e:
        logger.error(f"Error creating summary: {e}")

def main():
    parser = argparse.ArgumentParser(description='Multimodal TB Classification Experiments')
    
    # Data paths
    parser.add_argument('--clinical_data_path', type=str, required=True,
                      help='Path to clinical data CSV file')
    parser.add_argument('--base_h5_path', type=str, required=True,
                      help='Base path to H5 files directory')
    parser.add_argument('--base_site_path', type=str, required=True,
                      help='Base path to site CSV files directory')
    parser.add_argument('--base_fold_path', type=str, default=None,
                      help='Base path to fold CSV files directory (optional)')
    
    # Training parameters
    parser.add_argument('--num_folds', type=int, default=5,
                      help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension for models')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configuration
    config = {
        'clinical_data_path': args.clinical_data_path,
        'base_h5_path': args.base_h5_path,
        'base_site_path': args.base_site_path,
        'base_fold_path': args.base_fold_path,
        'num_folds': args.num_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'save_dir': args.save_dir
    }
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to: {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
    
    logger.info("Starting Multimodal TB Classification Experiments")
    logger.info(f"Configuration: {config}")
    
    # Run experiments
    try:
        print("\n🚀 Starting Multimodal TB Classification Experiments")
        print(f"📊 Training 13 models with {config['num_folds']}-fold cross-validation")
        print(f"⚙️ Configuration: {config}")
        print("="*80)
        
        results = run_cross_validation(config)
        
        print("\n🎉 Experiments completed successfully!")
        print(f"📁 All results saved to: {args.save_dir}")
        print("\n📋 Key files generated:")
        print(f"  • {args.save_dir}/results_summary.csv - Performance comparison")
        print(f"  • {args.save_dir}/complete_results.json - Detailed results")
        print(f"  • {args.save_dir}/*_predictions.csv - Model predictions")
        
    except Exception as e:
        logger.error(f"Experiments failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧠 Multimodal TB Classification with Uncertainty Estimation")
    print("=" * 60)
    print("📊 13 Models | 🎯 5-Fold CV | 🔬 Advanced Uncertainty Methods")
    print("=" * 60)
    main()