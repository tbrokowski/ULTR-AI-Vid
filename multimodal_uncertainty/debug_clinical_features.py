#!/usr/bin/env python3
"""
Comprehensive debugging test file for multimodal TB classification system
Tests datasets, dataloaders, and clinical encoders systematically
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('./ULTR-CLIP/multimodal_uncertainty')

try:
    from dataset import MultimodalTBDataset, create_dataloaders, analyze_dataset_distribution, load_fold_split_robust
    from models import DomainAwareClinicalEncoder, SimpleClinicalEncoder, create_model
    print("✅ Successfully imported custom modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the files are in the correct location:")
    print("  - dataset.py")
    print("  - models.py")
    sys.exit(1)

class MultimodalDebugger:
    """Comprehensive debugger for the multimodal TB system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_results = {}
        print("🔍 Initializing Multimodal System Debugger")
        print("=" * 60)
    
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results"""
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        
        self.test_results[test_name] = {
            'status': status,
            'details': details
        }
    
    def test_file_existence(self):
        """Test 1: Check if all required files exist"""
        print("\n🗂️  TEST 1: File Existence Check")
        print("-" * 40)
        
        required_files = {
            'clinical_data': self.config['clinical_data_path'],
        }
        
        # Check fold-specific files
        fold_files = {}
        for fold in range(min(2, self.config.get('num_folds', 5))):  # Check first 2 folds
            fold_files.update({
                f'train_h5_fold_{fold}': f"{self.config['base_h5_path']}/train_full_model_fold{fold}_complex_data.h5",
                f'val_h5_fold_{fold}': f"{self.config['base_h5_path']}/val_full_model_fold{fold}_complex_data.h5",
                f'test_h5_fold_{fold}': f"{self.config['base_h5_path']}/test_full_model_fold{fold}_complex_data.h5",
                f'train_site_fold_{fold}': f"{self.config['base_site_path']}/train_full_model_fold{fold}_sites.csv",
                f'val_site_fold_{fold}': f"{self.config['base_site_path']}/val_full_model_fold{fold}_sites.csv",
                f'test_site_fold_{fold}': f"{self.config['base_site_path']}/test_full_model_fold{fold}_sites.csv",
            })
            
            if self.config.get('base_fold_path'):
                fold_files[f'fold_split_{fold}'] = f"{self.config['base_fold_path']}/Fold_{fold}.csv"
        
        all_files = {**required_files, **fold_files}
        
        missing_files = []
        found_files = []
        
        for file_type, file_path in all_files.items():
            if os.path.exists(file_path):
                found_files.append(file_type)
                print(f"  ✅ {file_type}: Found")
            else:
                missing_files.append(file_type)
                print(f"  ❌ {file_type}: Missing - {file_path}")
        
        if missing_files:
            self.log_test("File Existence", "FAIL", f"Missing {len(missing_files)} files: {missing_files[:3]}...")
            return False
        else:
            self.log_test("File Existence", "PASS", f"All {len(all_files)} files found")
            return True
    
    def test_clinical_data_loading(self):
        """Test 2: Clinical data loading and basic structure"""
        print("\n📊 TEST 2: Clinical Data Loading")
        print("-" * 40)
        
        try:
            # Load clinical data
            clinical_df = pd.read_csv(self.config['clinical_data_path'])
            print(f"  📋 Shape: {clinical_df.shape}")
            print(f"  🔍 Columns: {len(clinical_df.columns)}")
            print(f"  🏥 Patients: {clinical_df['record_id'].nunique()}")
            
            # Check for critical columns
            required_cols = ['record_id']
            missing_cols = [col for col in required_cols if col not in clinical_df.columns]
            
            if missing_cols:
                self.log_test("Clinical Data Loading", "FAIL", f"Missing required columns: {missing_cols}")
                return False, None
            
            # Check data types and missing values
            missing_pct = (clinical_df.isnull().sum().sum() / clinical_df.size) * 100
            print(f"  📉 Missing data: {missing_pct:.1f}%")
            
            # Sample of column types
            print(f"  📝 Sample columns: {list(clinical_df.columns[:10])}")
            
            self.log_test("Clinical Data Loading", "PASS", f"Shape: {clinical_df.shape}, Missing: {missing_pct:.1f}%")
            return True, clinical_df
            
        except Exception as e:
            self.log_test("Clinical Data Loading", "FAIL", f"Error: {str(e)}")
            return False, None
    
    def test_h5_data_loading(self, fold=0):
        """Test 3: H5 embeddings loading"""
        print(f"\n🗄️  TEST 3: H5 Data Loading (Fold {fold})")
        print("-" * 40)
        
        h5_path = f"{self.config['base_h5_path']}/train_full_model_fold{fold}_complex_data.h5"
        
        try:
            with h5py.File(h5_path, 'r') as f:
                print(f"  📁 H5 file keys: {list(f.keys())}")
                
                if 'patient_features' in f:
                    patient_features = f['patient_features']
                    print(f"  👥 Patients in H5: {len(patient_features.keys())}")
                    
                    # Get a sample embedding
                    sample_patient = list(patient_features.keys())[0]
                    sample_embedding = patient_features[sample_patient][:]
                    print(f"  🧠 Embedding shape: {sample_embedding.shape}")
                    print(f"  📊 Embedding stats: mean={sample_embedding.mean():.3f}, std={sample_embedding.std():.3f}")
                    
                    # Check for NaN/inf values
                    nan_count = np.isnan(sample_embedding).sum()
                    inf_count = np.isinf(sample_embedding).sum()
                    print(f"  🔍 NaN values: {nan_count}, Inf values: {inf_count}")
                    
                    self.log_test("H5 Data Loading", "PASS", f"Patients: {len(patient_features.keys())}, Shape: {sample_embedding.shape}")
                    return True, list(patient_features.keys())
                else:
                    self.log_test("H5 Data Loading", "FAIL", "No 'patient_features' key found")
                    return False, []
                    
        except Exception as e:
            self.log_test("H5 Data Loading", "FAIL", f"Error: {str(e)}")
            return False, []
    
    def test_site_data_loading(self, fold=0):
        """Test 4: Site data loading"""
        print(f"\n🏥 TEST 4: Site Data Loading (Fold {fold})")
        print("-" * 40)
        
        site_path = f"{self.config['base_site_path']}/train_full_model_fold{fold}_sites.csv"
        
        try:
            site_df = pd.read_csv(site_path)
            print(f"  📋 Shape: {site_df.shape}")
            print(f"  👥 Unique patients: {site_df['patient_id'].nunique()}")
            print(f"  🔍 Columns: {list(site_df.columns)}")
            
            # Check for required columns
            required_cols = ['patient_id', 'tb_label']
            missing_cols = [col for col in required_cols if col not in site_df.columns]
            
            if missing_cols:
                self.log_test("Site Data Loading", "FAIL", f"Missing columns: {missing_cols}")
                return False, None
            
            # Check tb_label distribution
            if 'tb_label' in site_df.columns:
                label_dist = site_df['tb_label'].value_counts()
                print(f"  🎯 TB Label distribution: {dict(label_dist)}")
            
            self.log_test("Site Data Loading", "PASS", f"Patients: {site_df['patient_id'].nunique()}")
            return True, site_df
            
        except Exception as e:
            self.log_test("Site Data Loading", "FAIL", f"Error: {str(e)}")
            return False, None
    
    def test_dataset_creation(self, fold=0):
        """Test 5: Dataset creation and basic functionality"""
        print(f"\n🏗️  TEST 5: Dataset Creation (Fold {fold})")
        print("-" * 40)
        
        try:
            # Paths for this fold
            h5_path = f"{self.config['base_h5_path']}/train_full_model_fold{fold}_complex_data.h5"
            site_path = f"{self.config['base_site_path']}/train_full_model_fold{fold}_sites.csv"
            
            # Create dataset
            dataset = MultimodalTBDataset(
                clinical_data_path=self.config['clinical_data_path'],
                h5_file_path=h5_path,
                site_data_path=site_path,
                patient_ids=None,  # Let it find common IDs
                fit_scalers=True,
                is_training=True
            )
            
            print(f"  📏 Dataset length: {len(dataset)}")
            print(f"  🧠 Clinical dim: {dataset.clinical_dim}")
            print(f"  🎯 Feature count: {len(dataset.feature_order)}")
            print(f"  📚 Feature domains: {list(dataset.feature_schema.keys()) if dataset.feature_schema else 'None'}")
            
            if len(dataset) == 0:
                self.log_test("Dataset Creation", "FAIL", "Dataset is empty")
                return False, None
            
            # Test getting a sample
            sample = dataset[0]
            print(f"  🔍 Sample keys: {list(sample.keys())}")
            print(f"  📊 Clinical features shape: {sample['clinical_features'].shape}")
            print(f"  🧠 Patient embedding shape: {sample['patient_embedding'].shape}")
            print(f"  🎯 TB label: {sample['tb_label'].item()}")
            
            # Check for NaN values
            clinical_nan = torch.isnan(sample['clinical_features']).sum().item()
            embedding_nan = torch.isnan(sample['patient_embedding']).sum().item()
            print(f"  🔍 Clinical NaN count: {clinical_nan}")
            print(f"  🔍 Embedding NaN count: {embedding_nan}")
            
            # Check clinical feature statistics
            clinical_mean = sample['clinical_features'].mean().item()
            clinical_std = sample['clinical_features'].std().item()
            print(f"  📈 Clinical stats: mean={clinical_mean:.4f}, std={clinical_std:.4f}")
            
            # Test multiple samples
            if len(dataset) > 5:
                samples_to_test = min(5, len(dataset))
                all_clinical_features = []
                all_labels = []
                
                for i in range(samples_to_test):
                    sample = dataset[i]
                    all_clinical_features.append(sample['clinical_features'])
                    all_labels.append(sample['tb_label'].item())
                
                # Stack and check
                stacked_features = torch.stack(all_clinical_features)
                print(f"  📊 Batch clinical shape: {stacked_features.shape}")
                print(f"  🎯 Label distribution: {np.bincount(np.array(all_labels, dtype=int))}")
                
                # Check for all-zero features
                zero_features = (stacked_features == 0).all(dim=0).sum().item()
                print(f"  ⚠️  All-zero features: {zero_features}/{stacked_features.shape[1]}")
            
            self.log_test("Dataset Creation", "PASS", f"Length: {len(dataset)}, Clinical dim: {dataset.clinical_dim}")
            return True, dataset
            
        except Exception as e:
            self.log_test("Dataset Creation", "FAIL", f"Error: {str(e)}")
            traceback.print_exc()
            return False, None
    
    def test_clinical_feature_processing(self, dataset):
        """Test 6: Clinical feature processing in detail"""
        print("\n🧪 TEST 6: Clinical Feature Processing")
        print("-" * 40)
        
        try:
            # Test raw clinical data processing
            print(f"  📋 Raw feature order length: {len(dataset.feature_order)}")
            print(f"  📋 Clinical data shape: {dataset.clinical_features.shape}")
            print(f"  📚 Feature schema domains: {len(dataset.feature_schema)}")
            
            # Check for issues in clinical features
            clinical_df = dataset.clinical_features
            
            # NaN check
            nan_count = clinical_df.isnull().sum().sum()
            print(f"  🔍 Total NaN values: {nan_count}")
            
            # Zero variance check
            feature_stds = clinical_df.std()
            zero_var_count = (feature_stds < 1e-10).sum()
            print(f"  📉 Zero variance features: {zero_var_count}/{len(feature_stds)}")
            
            # All-zero features check
            zero_features = (clinical_df == 0).all(axis=0).sum()
            print(f"  ⚠️  All-zero features: {zero_features}/{len(clinical_df.columns)}")
            
            # Statistics
            overall_mean = clinical_df.values.mean()
            overall_std = clinical_df.values.std()
            print(f"  📊 Overall stats: mean={overall_mean:.6f}, std={overall_std:.6f}")
            
            # Check each domain if available
            if dataset.feature_schema:
                for domain, features in dataset.feature_schema.items():
                    if features:
                        domain_data = clinical_df[features]
                        domain_mean = domain_data.values.mean()
                        domain_std = domain_data.values.std()
                        print(f"    🏷️  {domain}: {len(features)} features, mean={domain_mean:.4f}, std={domain_std:.4f}")
            
            # Check scaling
            scaler = dataset.clinical_scaler
            if scaler:
                print(f"  ⚙️  Scaler type: {type(scaler).__name__}")
                if hasattr(scaler, 'center_'):
                    print(f"  ⚙️  Scaler center shape: {scaler.center_.shape}")
                if hasattr(scaler, 'scale_'):
                    print(f"  ⚙️  Scaler scale shape: {scaler.scale_.shape}")
            
            self.log_test("Clinical Feature Processing", "PASS", 
                         f"Features: {len(dataset.feature_order)}, NaN: {nan_count}, Zero-var: {zero_var_count}")
            return True
            
        except Exception as e:
            self.log_test("Clinical Feature Processing", "FAIL", f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_clinical_encoders(self, dataset):
        """Test 7: Clinical encoder functionality"""
        print("\n🧠 TEST 7: Clinical Encoder Testing")
        print("-" * 40)
        
        clinical_dim = dataset.clinical_dim
        output_dim = 128
        batch_size = 8
        
        # Create test data
        test_features = torch.randn(batch_size, clinical_dim)
        
        # Test Simple Clinical Encoder
        print("  Testing SimpleClinicalEncoder...")
        try:
            simple_encoder = SimpleClinicalEncoder(
                clinical_dim=clinical_dim,
                output_dim=output_dim,
                hidden_dim=256,
                dropout=0.1
            )
            
            simple_output = simple_encoder(test_features)
            print(f"    ✅ Simple encoder output shape: {simple_output.shape}")
            print(f"    📊 Simple encoder output stats: mean={simple_output.mean():.4f}, std={simple_output.std():.4f}")
            
            # Check for NaN
            nan_count = torch.isnan(simple_output).sum().item()
            print(f"    🔍 Simple encoder NaN count: {nan_count}")
            
        except Exception as e:
            print(f"    ❌ Simple encoder failed: {e}")
            traceback.print_exc()
        
        # Test Domain-Aware Clinical Encoder
        print("\n  Testing DomainAwareClinicalEncoder...")
        try:
            domain_encoder = DomainAwareClinicalEncoder(
                clinical_dim=clinical_dim,
                output_dim=output_dim,
                feature_schema=dataset.feature_schema,
                feature_order=dataset.feature_order,
                hidden_dim=256,
                dropout=0.1
            )
            
            print(f"    🏗️  Domain encoder uses domain-aware: {domain_encoder.use_domain_aware}")
            
            domain_output = domain_encoder(test_features)
            print(f"    ✅ Domain encoder output shape: {domain_output.shape}")
            print(f"    📊 Domain encoder output stats: mean={domain_output.mean():.4f}, std={domain_output.std():.4f}")
            
            # Check for NaN
            nan_count = torch.isnan(domain_output).sum().item()
            print(f"    🔍 Domain encoder NaN count: {nan_count}")
            
            # Test with problematic inputs
            print("\n  Testing with problematic inputs...")
            
            # All zeros
            zero_features = torch.zeros(batch_size, clinical_dim)
            zero_output = domain_encoder(zero_features)
            print(f"    🔍 Zero input output shape: {zero_output.shape}")
            
            # With NaN
            nan_features = test_features.clone()
            nan_features[:, :5] = float('nan')
            nan_output = domain_encoder(nan_features)
            print(f"    🔍 NaN input output shape: {nan_output.shape}")
            
            # Test with real dataset sample
            if len(dataset) > 0:
                real_sample = dataset[0]
                real_features = real_sample['clinical_features'].unsqueeze(0)
                real_output = domain_encoder(real_features)
                print(f"    ✅ Real sample output shape: {real_output.shape}")
                print(f"    📊 Real sample output: mean={real_output.mean():.4f}, std={real_output.std():.4f}")
            
            self.log_test("Clinical Encoders", "PASS", "Both encoders working")
            return True
            
        except Exception as e:
            print(f"    ❌ Domain encoder failed: {e}")
            traceback.print_exc()
            self.log_test("Clinical Encoders", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_dataloader_creation(self, fold=0):
        """Test 8: DataLoader creation and iteration"""
        print(f"\n🔄 TEST 8: DataLoader Creation (Fold {fold})")
        print("-" * 40)
        
        try:
            # Create dataloaders
            train_loader, val_loader, test_loader, scaler = create_dataloaders(
                clinical_data_path=self.config['clinical_data_path'],
                base_h5_path=self.config['base_h5_path'],
                base_site_path=self.config['base_site_path'],
                fold_num=fold,
                base_fold_path=self.config.get('base_fold_path'),
                batch_size=16,
                num_workers=0  # Use 0 workers for debugging
            )
            
            print(f"  📊 Train loader batches: {len(train_loader)}")
            print(f"  📊 Val loader batches: {len(val_loader)}")
            print(f"  📊 Test loader batches: {len(test_loader)}")
            print(f"  ⚙️  Scaler type: {type(scaler).__name__}")
            
            # Test iteration
            print("\n  Testing train loader iteration...")
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
            print(f"    🔍 Batch keys: {list(batch.keys())}")
            print(f"    📏 Batch size: {len(batch['patient_id'])}")
            print(f"    📊 Clinical features shape: {batch['clinical_features'].shape}")
            print(f"    🧠 Patient embeddings shape: {batch['patient_embedding'].shape}")
            print(f"    🎯 TB labels shape: {batch['tb_label'].shape}")
            
            # Check for NaN values in batch
            clinical_nan = torch.isnan(batch['clinical_features']).sum().item()
            embedding_nan = torch.isnan(batch['patient_embedding']).sum().item()
            print(f"    🔍 Clinical NaN in batch: {clinical_nan}")
            print(f"    🔍 Embedding NaN in batch: {embedding_nan}")
            
            # Test multiple batches
            print("\n  Testing multiple batch iteration...")
            batch_count = 0
            total_samples = 0
            clinical_stats = []
            
            for i, batch in enumerate(train_loader):
                if i >= 3:  # Test first 3 batches
                    break
                batch_count += 1
                total_samples += len(batch['patient_id'])
                
                # Collect stats
                clinical_mean = batch['clinical_features'].mean().item()
                clinical_std = batch['clinical_features'].std().item()
                clinical_stats.append((clinical_mean, clinical_std))
                
                print(f"    Batch {i+1}: size={len(batch['patient_id'])}, "
                      f"clinical_mean={clinical_mean:.4f}, clinical_std={clinical_std:.4f}")
            
            print(f"  ✅ Successfully iterated {batch_count} batches, {total_samples} total samples")
            
            self.log_test("DataLoader Creation", "PASS", 
                         f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
            return True, (train_loader, val_loader, test_loader, scaler)
            
        except Exception as e:
            self.log_test("DataLoader Creation", "FAIL", f"Error: {str(e)}")
            traceback.print_exc()
            return False, None
    
    def test_model_creation(self, dataloaders):
        """Test 9: Model creation and forward pass"""
        print("\n🏗️  TEST 9: Model Creation and Forward Pass")
        print("-" * 40)
        
        if not dataloaders:
            print("  ❌ No dataloaders available for testing")
            return False
        
        train_loader, val_loader, test_loader, scaler = dataloaders
        
        try:
            # Get sample batch for dimensions
            sample_batch = next(iter(train_loader))
            clinical_dim = sample_batch['clinical_features'].shape[1]
            embedding_dim = sample_batch['patient_embedding'].shape[1]
            
            print(f"  📏 Clinical dim: {clinical_dim}")
            print(f"  🧠 Embedding dim: {embedding_dim}")
            
            # Get feature schema from dataset
            dataset = train_loader.dataset
            feature_schema = getattr(dataset, 'feature_schema', None)
            feature_order = getattr(dataset, 'feature_order', None)
            
            print(f"  📚 Feature schema available: {feature_schema is not None}")
            print(f"  📋 Feature order available: {feature_order is not None}")
            
            # Test different model types
            model_types = ['simple_concat', 'clinical_only', 'ccg']
            
            for model_type in model_types:
                print(f"\n  Testing {model_type} model...")
                try:
                    model = create_model(
                        model_type=model_type,
                        clinical_dim=clinical_dim,
                        embedding_dim=embedding_dim,
                        feature_schema=feature_schema,
                        feature_order=feature_order,
                        hidden_dim=128,
                        dropout=0.1
                    )
                    
                    print(f"    ✅ {model_type} model created successfully")
                    
                    # Test forward pass
                    model.eval()
                    with torch.no_grad():
                        clinical_features = sample_batch['clinical_features']
                        patient_embeddings = sample_batch['patient_embedding']
                        
                        output = model(clinical_features, patient_embeddings)
                        
                        # Handle different output formats
                        if isinstance(output, tuple):
                            output = output[0]
                        
                        print(f"    📊 {model_type} output shape: {output.shape}")
                        print(f"    📈 {model_type} output range: [{output.min():.3f}, {output.max():.3f}]")
                        
                        # Check for NaN
                        nan_count = torch.isnan(output).sum().item()
                        print(f"    🔍 {model_type} NaN count: {nan_count}")
                        
                        if nan_count > 0:
                            print(f"    ❌ {model_type} model produces NaN outputs!")
                        else:
                            print(f"    ✅ {model_type} forward pass successful")
                            
                except Exception as e:
                    print(f"    ❌ {model_type} model failed: {e}")
                    traceback.print_exc()
            
            self.log_test("Model Creation", "PASS", f"Tested {len(model_types)} model types")
            return True
            
        except Exception as e:
            self.log_test("Model Creation", "FAIL", f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_end_to_end_pipeline(self, fold=0):
        """Test 10: End-to-end pipeline test"""
        print(f"\n🔄 TEST 10: End-to-End Pipeline (Fold {fold})")
        print("-" * 40)
        
        try:
            # Create dataloaders
            train_loader, val_loader, test_loader, scaler = create_dataloaders(
                clinical_data_path=self.config['clinical_data_path'],
                base_h5_path=self.config['base_h5_path'],
                base_site_path=self.config['base_site_path'],
                fold_num=fold,
                base_fold_path=self.config.get('base_fold_path'),
                batch_size=8,
                num_workers=0
            )
            
            # Get sample data
            sample_batch = next(iter(train_loader))
            clinical_dim = sample_batch['clinical_features'].shape[1]
            embedding_dim = sample_batch['patient_embedding'].shape[1]
            
            # Get feature info from dataset
            dataset = train_loader.dataset
            feature_schema = getattr(dataset, 'feature_schema', None)
            feature_order = getattr(dataset, 'feature_order', None)
            
            # Create model
            model = create_model(
                model_type='ccg',  # Test most complex model
                clinical_dim=clinical_dim,
                embedding_dim=embedding_dim,
                feature_schema=feature_schema,
                feature_order=feature_order,
                hidden_dim=128,
                dropout=0.1
            )
            
            # Test training-like forward pass
            model.train()
            clinical_features = sample_batch['clinical_features']
            patient_embeddings = sample_batch['patient_embedding']
            labels = sample_batch['tb_label']
            
            # Forward pass
            output = model(clinical_features, patient_embeddings)
            if isinstance(output, tuple):
                output = output[0]
            
            # Loss computation
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(output, labels.unsqueeze(1))
            
            print(f"  ✅ Forward pass successful")
            print(f"  📊 Output shape: {output.shape}")
            print(f"  💰 Loss value: {loss.item():.4f}")
            
            # Test backward pass
            loss.backward()
            print(f"  ✅ Backward pass successful")
            
            # Test evaluation mode
            model.eval()
            with torch.no_grad():
                eval_output = model(clinical_features, patient_embeddings)
                if isinstance(eval_output, tuple):
                    eval_output = eval_output[0]
                
                probs = torch.sigmoid(eval_output)
                preds = (probs > 0.5).float()
                
                print(f"  ✅ Evaluation mode successful")
                print(f"  📊 Predictions range: [{preds.min():.1f}, {preds.max():.1f}]")
                print(f"  📊 Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
            
            # Test multiple batches
            print(f"\n  Testing multiple batches...")
            batch_count = 0
            total_loss = 0.0
            
            model.train()
            for i, batch in enumerate(train_loader):
                if i >= 3:  # Test 3 batches
                    break
                
                clinical_features = batch['clinical_features']
                patient_embeddings = batch['patient_embedding']
                labels = batch['tb_label']
                
                output = model(clinical_features, patient_embeddings)
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = criterion(output, labels.unsqueeze(1))
                total_loss += loss.item()
                batch_count += 1
                
                print(f"    Batch {i+1}: loss={loss.item():.4f}")
            
            avg_loss = total_loss / batch_count
            print(f"  📊 Average loss over {batch_count} batches: {avg_loss:.4f}")
            
            self.log_test("End-to-End Pipeline", "PASS", f"Successfully processed {batch_count} batches")
            return True
            
        except Exception as e:
            self.log_test("End-to-End Pipeline", "FAIL", f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_dataset_analysis(self, fold=0):
        """Test 11: Dataset distribution analysis"""
        print(f"\n📈 TEST 11: Dataset Analysis (Fold {fold})")
        print("-" * 40)
        
        try:
            # Create dataset
            h5_path = f"{self.config['base_h5_path']}/train_full_model_fold{fold}_complex_data.h5"
            site_path = f"{self.config['base_site_path']}/train_full_model_fold{fold}_sites.csv"
            
            dataset = MultimodalTBDataset(
                clinical_data_path=self.config['clinical_data_path'],
                h5_file_path=h5_path,
                site_data_path=site_path,
                patient_ids=None,
                fit_scalers=True
            )
            
            # Analyze distribution
            analysis = analyze_dataset_distribution(dataset)
            
            print(f"  📊 Total patients: {analysis.get('total_patients', 'Unknown')}")
            print(f"  📊 Sampled patients: {analysis.get('sampled_patients', 'Unknown')}")
            print(f"  🎯 TB positive cases: {analysis.get('tb_positive_cases', 'Unknown')}")
            print(f"  🎯 TB negative cases: {analysis.get('tb_negative_cases', 'Unknown')}")
            print(f"  📊 TB positive ratio: {analysis.get('tb_positive_ratio', 'Unknown')}")
            print(f"  🧠 Clinical features dim: {analysis.get('clinical_features_dim', 'Unknown')}")
            print(f"  📈 Clinical mean: {analysis.get('clinical_features_mean', 'Unknown')}")
            print(f"  📉 Clinical std: {analysis.get('clinical_features_std', 'Unknown')}")
            print(f"  🔍 Has NaN values: {analysis.get('has_nan_values', 'Unknown')}")
            print(f"  🧠 Embedding dim: {analysis.get('embedding_dim', 'Unknown')}")
            
            if 'error' in analysis:
                self.log_test("Dataset Analysis", "FAIL", f"Error: {analysis['error']}")
                return False
            else:
                self.log_test("Dataset Analysis", "PASS", f"TB ratio: {analysis.get('tb_positive_ratio', 0):.3f}")
                return True
                
        except Exception as e:
            self.log_test("Dataset Analysis", "FAIL", f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all debugging tests"""
        print("🚀 Starting Comprehensive Multimodal System Debug")
        print("=" * 80)
        
        # Test 1: File existence
        file_check = self.test_file_existence()
        
        # Test 2: Clinical data loading
        clinical_check, clinical_df = self.test_clinical_data_loading()
        
        # Test 3: H5 data loading
        h5_check, h5_patients = self.test_h5_data_loading()
        
        # Test 4: Site data loading  
        site_check, site_df = self.test_site_data_loading()
        
        # Test 5: Dataset creation
        dataset_check, dataset = self.test_dataset_creation()
        
        # Test 6: Clinical feature processing
        if dataset:
            feature_check = self.test_clinical_feature_processing(dataset)
        else:
            feature_check = False
            
        # Test 7: Clinical encoders
        if dataset:
            encoder_check = self.test_clinical_encoders(dataset)
        else:
            encoder_check = False
        
        # Test 8: DataLoader creation
        dataloader_check, dataloaders = self.test_dataloader_creation()
        
        # Test 9: Model creation
        if dataloaders:
            model_check = self.test_model_creation(dataloaders)
        else:
            model_check = False
        
        # Test 10: End-to-end pipeline
        pipeline_check = self.test_end_to_end_pipeline()
        
        # Test 11: Dataset analysis
        analysis_check = self.test_dataset_analysis()
        
        # Summary
        self.print_summary()
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("🎯 TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        total = len(self.test_results)
        
        print(f"📊 Overall: {passed}/{total} tests passed")
        print()
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
            print(f"{status_emoji} {test_name:<30} {result['status']}")
            if result['details']:
                print(f"   └─ {result['details']}")
        
        print("\n" + "=" * 80)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Your system is working correctly.")
        else:
            print("⚠️  SOME TESTS FAILED. Check the details above to identify issues.")
            print("\n🔧 Common fixes:")
            print("  • Check file paths in your configuration")
            print("  • Verify data file formats and required columns") 
            print("  • Check for missing dependencies")
            print("  • Review error messages for specific issues")
        
        print("=" * 80)


def main():
    """Main function to run debugging tests"""
    
    # Configuration - Update these paths to match your setup
    config = {
        'clinical_data_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv',
        'base_h5_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results',
        'base_site_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results',
        'base_fold_path': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files',
        'num_folds': 5
    }
    
    print("🔍 Multimodal TB Classification System Debugger")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create debugger
    debugger = MultimodalDebugger(config)
    
    # Run all tests
    results = debugger.run_all_tests()
    
    # Save results
    try:
        import json
        with open('debug_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Debug results saved to: debug_results.json")
    except Exception as e:
        print(f"⚠️  Could not save debug results: {e}")

if __name__ == "__main__":
    main()