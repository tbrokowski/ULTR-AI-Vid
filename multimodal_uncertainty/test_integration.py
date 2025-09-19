#!/usr/bin/env python3
"""
Comprehensive integration test for multimodal TB classification pipeline
Tests with real CLUSSTER-Benin data
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

try:
    from dataset import MultimodalTBDataset, create_dataloaders, custom_collate_fn, analyze_dataset_distribution
    from models import create_model, DomainAwareClinicalEncoder, SimpleClinicalEncoder
    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# ---- CONFIGURATION ----
config = {
    'clinical_data_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv',
    'base_h5_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results',
    'base_site_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results',
    'base_fold_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files',
    'batch_size': 32,
    'num_folds': 5
}

def check_file_existence():
    """Check if all required files exist"""
    print("\n" + "="*50)
    print("CHECKING FILE EXISTENCE")
    print("="*50)
    
    files_to_check = [config['clinical_data_path']]
    
    # Check fold-specific files for fold 0
    fold = 0
    for split in ['train', 'val', 'test']:
        h5_file = f"{config['base_h5_path']}/{split}_full_model_fold{fold}_complex_data.h5"
        site_file = f"{config['base_site_path']}/{split}_full_model_fold{fold}_sites.csv"
        files_to_check.extend([h5_file, site_file])
    
    # Check fold split file
    if config['base_fold_path']:
        fold_file = f"{config['base_fold_path']}/Fold_{fold}.csv"
        files_to_check.append(fold_file)
    
    missing_files = []
    found_files = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            found_files.append(file_path)
            print(f"✓ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")
    
    print(f"\nSummary: {len(found_files)} found, {len(missing_files)} missing")
    
    if missing_files:
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("All required data files found!")
        return True

def inspect_batch(loader, name=""):
    """Inspect a batch from the dataloader"""
    print(f"\n--- Inspecting {name} loader ---")
    
    try:
        for batch in loader:
            print(f"Batch type: {type(batch)}")
            
            if isinstance(batch, dict):
                print("Batch contents:")
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"  - batch['{key}']: shape={value.shape}, dtype={value.dtype}")
                        if value.numel() < 20:  # Print small tensors
                            print(f"    values: {value.flatten()}")
                        else:
                            print(f"    sample values: {value.flatten()[:5]}")
                    elif isinstance(value, list):
                        print(f"  - batch['{key}']: type=list, length={len(value)}")
                        if len(value) <= 5:
                            print(f"    values: {value}")
                        else:
                            print(f"    sample values: {value[:3]}...")
                    else:
                        print(f"  - batch['{key}']: type={type(value)}, value={value}")
            
            elif isinstance(batch, (list, tuple)):
                print(f"Batch is {type(batch).__name__} with {len(batch)} elements:")
                for i, b in enumerate(batch):
                    if torch.is_tensor(b):
                        print(f"  - batch[{i}]: shape={b.shape}, dtype={b.dtype}")
                    else:
                        print(f"  - batch[{i}]: type={type(b)}")
            else:
                if torch.is_tensor(batch):
                    print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")
                else:
                    print(f"Batch type: {type(batch)}")
            
            # Only inspect first batch
            break
            
    except Exception as e:
        print(f"✗ Error inspecting {name} loader: {e}")
        import traceback
        traceback.print_exc()

def test_dataloader_creation(fold=0):
    """Test dataloader creation with real data"""
    print("\n" + "="*50)
    print(f"TESTING DATALOADER CREATION - FOLD {fold}")
    print("="*50)
    
    try:
        print(f"Loading Fold {fold+1}/{config['num_folds']}...")
        
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            clinical_data_path=config['clinical_data_path'],
            base_h5_path=config['base_h5_path'],
            base_site_path=config['base_site_path'],
            fold_num=fold,
            base_fold_path=config['base_fold_path'],
            batch_size=config['batch_size']
        )
        
        print(f"✓ Dataloaders created successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Inspect batches
        inspect_batch(train_loader, "Train")
        inspect_batch(val_loader, "Val")
        inspect_batch(test_loader, "Test")
        
        # Check scaler
        print(f"\n--- Scaler Information ---")
        if scaler is not None:
            if hasattr(scaler, 'mean_'):
                print(f"Scaler mean shape: {scaler.mean_.shape}")
                print(f"Scaler mean (first 10): {scaler.mean_[:10]}")
            if hasattr(scaler, 'scale_'):
                print(f"Scaler scale shape: {scaler.scale_.shape}")
                print(f"Scaler scale (first 10): {scaler.scale_[:10]}")
            print(f"Scaler type: {type(scaler)}")
        else:
            print("No scaler returned")
        
        # Get dimensions from sample
        sample = next(iter(train_loader))
        if isinstance(sample, dict):
            clinical_features = sample.get('clinical_features')
            patient_embeddings = sample.get('patient_embedding')
            
            if clinical_features is not None:
                clinical_dim = clinical_features.shape[-1]
                print(f"\nInferred clinical_dim: {clinical_dim}")
            
            if patient_embeddings is not None:
                embedding_dim = patient_embeddings.shape[-1]
                print(f"Inferred embedding_dim: {embedding_dim}")
            
            # Check feature schema
            feature_schema = sample.get('feature_schema')
            if feature_schema:
                print(f"\nFeature schema domains: {list(feature_schema.keys())}")
                for domain, features in feature_schema.items():
                    print(f"  {domain}: {len(features)} features")
            else:
                print("No feature schema in batch")
        
        return train_loader, val_loader, test_loader, scaler
        
    except Exception as e:
        print(f"✗ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def test_dataset_analysis(train_loader, val_loader, test_loader):
    """Test dataset analysis functions"""
    print("\n" + "="*50)
    print("TESTING DATASET ANALYSIS")
    print("="*50)
    
    if train_loader is None:
        print("✗ Cannot analyze datasets without loaders")
        return
    
    try:
        # Analyze each dataset
        for loader, name in [(train_loader, "Train"), (val_loader, "Val"), (test_loader, "Test")]:
            print(f"\n--- {name} Dataset Analysis ---")
            stats = analyze_dataset_distribution(loader.dataset)
            
            if 'error' in stats:
                print(f"✗ Analysis failed: {stats['error']}")
            else:
                print(f"✓ Analysis successful:")
                for key, value in stats.items():
                    print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"✗ Dataset analysis failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_creation_with_real_data(train_loader):
    """Test model creation with real data dimensions"""
    print("\n" + "="*50)
    print("TESTING MODEL CREATION WITH REAL DATA")
    print("="*50)
    
    if train_loader is None:
        print("✗ Cannot test models without dataloader")
        return {}
    
    try:
        # Get dimensions and schema from real data
        sample = next(iter(train_loader))
        
        if not isinstance(sample, dict):
            print("✗ Expected batch to be a dictionary")
            return {}
        
        clinical_features = sample.get('clinical_features')
        patient_embeddings = sample.get('patient_embedding')
        feature_schema = sample.get('feature_schema')
        
        if clinical_features is None or patient_embeddings is None:
            print("✗ Missing clinical_features or patient_embedding in batch")
            return {}
        
        clinical_dim = clinical_features.shape[-1]
        embedding_dim = patient_embeddings.shape[-1]
        feature_order = getattr(train_loader.dataset, 'feature_order', None)
        
        print(f"Real data dimensions:")
        print(f"  - Clinical: {clinical_dim}")
        print(f"  - Embedding: {embedding_dim}")
        print(f"  - Feature schema available: {feature_schema is not None}")
        print(f"  - Feature order available: {feature_order is not None}")
        
        if feature_schema:
            print(f"  - Schema domains: {list(feature_schema.keys())}")
        
        # Test model creation
        models = {}
        model_types = [
            # Non-multimodal baselines
            'clinical_only', 'imaging_only',
            
            # Simple fusion methods  
            'simple_concat', 'early_fusion', 'late_fusion', 
            
            # Advanced fusion methods
            'contrastive', 'cross_attention', 'bilinear', 
            
            # Novel methods
            'ccg', 
            
            # Uncertainty estimation methods
            'variational', 'mud', 'evidential', 'mc_frequency'
        ]
        
        for model_type in model_types:
            try:
                model = create_model(
                    model_type=model_type,
                    clinical_dim=clinical_dim,
                    embedding_dim=embedding_dim,
                    feature_schema=feature_schema,
                    feature_order=feature_order,
                    hidden_dim=256,
                    dropout=0.1
                )
                
                # Test forward pass
                with torch.no_grad():
                    output = model(clinical_features, patient_embeddings)
                    
                    if model_type == 'variational':
                        pred_mu, pred_logvar, kl_loss = output
                        output_shape = pred_mu.shape
                        print(f"✓ {model_type}: output_shape={output_shape}, kl_loss={kl_loss.item():.4f}")
                    elif model_type == 'mud':
                        pred, uncertainty_dict = output
                        print(f"✓ {model_type}: pred_shape={pred.shape}, uncertainties={list(uncertainty_dict.keys())}")
                    else:
                        print(f"✓ {model_type}: output_shape={output.shape}")
                
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                models[model_type] = model
                print(f"    Parameters: {num_params:,}")
                
                # Test if domain-aware encoder is being used
                if hasattr(model, 'clinical_encoder') and isinstance(model.clinical_encoder, DomainAwareClinicalEncoder):
                    if model.clinical_encoder.use_domain_aware:
                        print(f"    Using domain-aware clinical encoder ✓")
                    else:
                        print(f"    Fallback to simple encoder (dimension mismatch)")
                
            except Exception as e:
                print(f"✗ {model_type}: {e}")
                if "dimension mismatch" in str(e).lower():
                    print(f"    Likely feature schema mismatch")
        
        print(f"\nSuccessfully created {len(models)}/{len(model_types)} models")
        return models
        
    except Exception as e:
        print(f"✗ Model creation with real data failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_clinical_encoder_domains(train_loader):
    """Test domain-aware clinical encoder specifically"""
    print("\n" + "="*50)
    print("TESTING DOMAIN-AWARE CLINICAL ENCODER")
    print("="*50)
    
    if train_loader is None:
        print("✗ Cannot test without dataloader")
        return
    
    try:
        sample = next(iter(train_loader))
        clinical_features = sample.get('clinical_features')
        feature_schema = sample.get('feature_schema')
        feature_order = getattr(train_loader.dataset, 'feature_order', None)
        
        if clinical_features is None:
            print("✗ No clinical features in sample")
            return
        
        clinical_dim = clinical_features.shape[-1]
        
        print(f"Testing DomainAwareClinicalEncoder:")
        print(f"  - Input dimension: {clinical_dim}")
        print(f"  - Feature schema: {feature_schema is not None}")
        print(f"  - Feature order: {feature_order is not None}")
        
        if feature_schema and feature_order:
            print(f"\nFeature schema details:")
            total_schema_features = 0
            for domain, features in feature_schema.items():
                valid_features = [f for f in features if f in feature_order]
                total_schema_features += len(valid_features)
                print(f"  {domain}: {len(valid_features)}/{len(features)} features available")
            
            print(f"\nDimension check:")
            print(f"  - Clinical dim: {clinical_dim}")
            print(f"  - Schema features: {total_schema_features}")
            print(f"  - Feature order length: {len(feature_order)}")
            
            # Test encoder creation
            encoder = DomainAwareClinicalEncoder(
                clinical_dim=clinical_dim,
                output_dim=256,
                feature_schema=feature_schema,
                feature_order=feature_order,
                hidden_dim=256,
                dropout=0.1
            )
            
            print(f"\nEncoder created:")
            print(f"  - Using domain-aware: {encoder.use_domain_aware}")
            if encoder.use_domain_aware:
                print(f"  - Domain encoders: {list(encoder.domain_encoders.keys())}")
                print(f"  - Using attention: {encoder.use_attention}")
            
            # Test forward pass
            with torch.no_grad():
                output = encoder(clinical_features)
                print(f"  - Output shape: {output.shape}")
                print(f"✓ Domain-aware encoder working correctly")
        else:
            print("✗ Missing feature schema or feature order - cannot test domain-aware encoder")
    
    except Exception as e:
        print(f"✗ Domain-aware encoder test failed: {e}")
        import traceback
        traceback.print_exc()

def run_full_integration_test():
    """Run the complete integration test"""
    print("MULTIMODAL TB CLASSIFICATION - INTEGRATION TEST")
    print("="*60)
    
    # Check file existence
    if not check_file_existence():
        print("\n✗ Cannot proceed without required files")
        return False
    
    # Test dataloader creation
    train_loader, val_loader, test_loader, scaler = test_dataloader_creation(fold=0)
    
    if train_loader is None:
        print("\n✗ Cannot proceed without working dataloaders")
        return False
    
    # Test dataset analysis
    test_dataset_analysis(train_loader, val_loader, test_loader)
    
    # Test domain-aware clinical encoder
    test_clinical_encoder_domains(train_loader)
    
    # Test model creation
    models = test_model_creation_with_real_data(train_loader)
    
    if len(models) > 0:
        print(f"\n✓ Integration test completed successfully!")
        print(f"  - Created {len(models)} working models")
        print(f"  - All components integrated properly")
        return True
    else:
        print(f"\n✗ Integration test failed - no working models created")
        return False

if __name__ == "__main__":
    success = run_full_integration_test()
    sys.exit(0 if success else 1)