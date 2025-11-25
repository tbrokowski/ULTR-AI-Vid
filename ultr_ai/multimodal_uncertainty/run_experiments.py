#!/usr/bin/env python3
"""
Example script to run the multimodal TB classification experiments
"""

import subprocess
import os
import sys
from pathlib import Path

def run_experiments():
    """Run the complete experimental pipeline"""
    
    # Define paths (your actual data locations)
    clinical_data_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv'
    base_h5_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results'
    base_site_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results'
    base_fold_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files'
    
    # Create results directory
    results_dir = './multimodal_tb_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Training parameters
    config = {
        'clinical_data_path': clinical_data_path,
        'base_h5_path': base_h5_path,
        'base_site_path': base_site_path,
        'base_fold_path': base_fold_path,
        'num_folds': 5,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'hidden_dim': 256,
        'dropout': 0.1,
        'save_dir': results_dir
    }
    
    # Build command - using your directory structure
    cmd = [
        'python3', 'ULTR-CLIP/multimodal_uncertainty/train.py',  # Assumes train.py is in current directory or ULTR-CLIP/multimodal_uncertainty/
        '--clinical_data_path', config['clinical_data_path'],
        '--base_h5_path', config['base_h5_path'],
        '--base_site_path', config['base_site_path'],
        '--base_fold_path', config['base_fold_path'],
        '--num_folds', str(config['num_folds']),
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--learning_rate', str(config['learning_rate']),
        '--hidden_dim', str(config['hidden_dim']),
        '--dropout', str(config['dropout']),
        '--save_dir', config['save_dir']
    ]
    
    print("Starting multimodal TB classification experiments...")
    print("This will train 13 different models:")
    print("├── Baselines: clinical_only, imaging_only")
    print("├── Simple Fusion: simple_concat, early_fusion, late_fusion")  
    print("├── Advanced Fusion: contrastive, cross_attention, bilinear")
    print("├── Novel Methods: ccg (Clinical Contextual Gating)")
    print("└── Uncertainty: variational, mud, evidential, mc_frequency")
    print()
    print("📊 Progress tracking enabled with tqdm:")
    print("  🔄 Model progress: 13 models")
    print("  📁 Fold progress: 5 folds per model") 
    print("  🏃 Epoch progress: up to 100 epochs per fold")
    print("  🔍 Batch progress: training/validation batches")
    print()
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {results_dir}")
    print("\n⏱️ Expected runtime: 6-12 hours for full experiments")
    print("🚀 Starting now...")
    print("="*60)
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Experiments completed successfully!")
        print(result.stdout)
        
        print(f"\nResults saved to: {results_dir}")
        print("Check these files:")
        print(f"  - {results_dir}/results_summary.csv")
        print(f"  - {results_dir}/complete_results.json")
        print(f"  - {results_dir}/*_predictions.csv")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running experiments: {e}")
        print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Standard output: {e.stdout}")
        sys.exit(1)

def quick_test():
    """Run a quick test with limited data to check if everything works"""
    
    print("Running quick test with limited parameters...")
    
    # Same paths as above
    clinical_data_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv'
    base_h5_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results'
    base_site_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results'
    base_fold_path = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files'
    
    results_dir = './test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Quick test parameters
    cmd = [
        'python3', 'ULTR-CLIP/multimodal_uncertainty/train.py',
        '--clinical_data_path', clinical_data_path,
        '--base_h5_path', base_h5_path,
        '--base_site_path', base_site_path,
        '--base_fold_path', base_fold_path,
        '--num_folds', '2',  # Only 2 folds for quick test
        '--epochs', '5',     # Only 5 epochs for quick test
        '--batch_size', '16',
        '--learning_rate', '0.001',
        '--hidden_dim', '128',  # Smaller hidden dim
        '--dropout', '0.1',
        '--save_dir', results_dir
    ]
    
    print("Testing with 2 folds, 5 epochs...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Quick test completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Quick test failed: {e}")
        print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Standard output: {e.stdout}")
        return False

def check_data_files():
    """Check if data files exist at expected locations"""
    
    base_paths = {
        'clinical': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv',
        'h5_base': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results',
        'site_base': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/ULTR-CLIP/results',
        'fold_base': '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files'
    }
    
    files_to_check = [base_paths['clinical']]
    
    # Check fold-specific files (at least first 2 folds)
    for fold in range(2):
        files_to_check.extend([
            f"{base_paths['h5_base']}/train_full_model_fold{fold}_complex_data.h5",
            f"{base_paths['h5_base']}/val_full_model_fold{fold}_complex_data.h5",
            f"{base_paths['h5_base']}/test_full_model_fold{fold}_complex_data.h5",
            f"{base_paths['site_base']}/train_full_model_fold{fold}_sites.csv",
            f"{base_paths['site_base']}/val_full_model_fold{fold}_sites.csv",
            f"{base_paths['site_base']}/test_full_model_fold{fold}_sites.csv",
        ])
    
    # Check fold split files
    for fold in range(2):
        files_to_check.append(f"{base_paths['fold_base']}/Fold_{fold}.csv")
    
    print("Checking data files...")
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
        for file_path in missing_files[:10]:  # Show first 10
            print(f"  - {file_path}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        
        print("\nNote: Missing files will prevent some folds from running,")
        print("but the experiment may still work with available data.")
        return False
    else:
        print("All required data files found!")
        return True

def setup_files():
    """Help set up the required files in the correct locations"""
    
    print("Setting up multimodal TB classification files...")
    print("\nYou need these files in your ULTR-CLIP/multimodal_uncertainty/ directory:")
    print("  - dataset.py (multimodal dataset loader)")
    print("  - models.py (13 different model architectures)")  
    print("  - train.py (training script)")
    print("  - run_experiments.py (this file)")
    print("  - test_integration.py (optional - for testing)")
    
    current_dir = os.getcwd()
    target_dir = "ULTR-CLIP/multimodal_uncertainty"
    
    if not os.path.exists(target_dir):
        print(f"\nCreating directory: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
    
    required_files = ['dataset.py', 'models.py', 'train.py']
    missing_files = []
    
    for filename in required_files:
        file_path = os.path.join(target_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        print(f"\nMissing files in {target_dir}:")
        for filename in missing_files:
            print(f"  - {filename}")
        print("\nPlease copy the updated files to the correct location.")
    else:
        print(f"\nAll required files found in {target_dir}!")
    
    return len(missing_files) == 0

def main():
    """Main function to choose what to run"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            print("Running quick test...")
            if quick_test():
                print("\nQuick test passed! You can now run full experiments.")
                print("To run full experiments: python3 run_experiments.py full")
            else:
                print("Quick test failed. Please check your setup.")
                
        elif sys.argv[1] == 'check':
            check_data_files()
            
        elif sys.argv[1] == 'setup':
            setup_files()
            
        elif sys.argv[1] == 'full':
            print("Checking prerequisites...")
            data_ok = check_data_files()
            files_ok = setup_files()
            
            if data_ok or input("\nSome data files missing. Continue anyway? (y/N): ").lower() == 'y':
                run_experiments()
            else:
                print("Please fix data file issues and try again.")
                
        else:
            print("Usage: python3 run_experiments.py [test|check|setup|full]")
    else:
        print("Multimodal TB Classification Experiments")
        print("=" * 50)
        print("This system trains 13 different models:")
        print("  Baselines: clinical_only, imaging_only") 
        print("  Simple Fusion: simple_concat, early_fusion, late_fusion")
        print("  Advanced Fusion: contrastive, cross_attention, bilinear")
        print("  Novel Methods: ccg (Clinical Contextual Gating)")
        print("  Uncertainty: variational, mud, evidential, mc_frequency")
        print()
        print("Usage options:")
        print("  python3 run_experiments.py check  - Check if data files exist")
        print("  python3 run_experiments.py setup  - Check if code files are in place")
        print("  python3 run_experiments.py test   - Run quick test (2 folds, 5 epochs)")
        print("  python3 run_experiments.py full   - Run full experiments (5 folds, 100 epochs)")
        print()
        
        # Default: check everything
        print("Checking setup...")
        data_ok = check_data_files()
        print()
        files_ok = setup_files()
        
        if data_ok and files_ok:
            print("\n✓ Everything looks good! You can run:")
            print("  python3 run_experiments.py test   # Quick test")
            print("  python3 run_experiments.py full   # Full experiments")
        else:
            print("\n✗ Please fix the issues above before running experiments.")

if __name__ == "__main__":
    main()


# Run all 13 models with 5-fold CV, 100 epochs each
#python3 ULTR-CLIP/multimodal_uncertainty/run_experiments.py full