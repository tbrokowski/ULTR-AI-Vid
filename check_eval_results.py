#!/usr/bin/env python3
"""
Quick script to check evaluation results structure and availability
For debugging and validating that all expected files exist before visualization
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import json

# Base directory on cluster
CLUSTER_BASE = "/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results"

def check_experiment_results(experiment_name: str, num_folds: int = 5, verbose: bool = True):
    """
    Check if all expected evaluation results exist for an experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'inception3d', 'LeViT-RL')
        num_folds: Number of folds to check (default: 5)
        verbose: Print detailed information
    
    Returns:
        Dictionary with status and missing files
    """
    experiment_path = Path(CLUSTER_BASE) / experiment_name
    eval_results_path = experiment_path / "eval_results"
    
    status = {
        'experiment_exists': experiment_path.exists(),
        'eval_results_exists': eval_results_path.exists(),
        'folds_found': [],
        'missing_folds': [],
        'results_by_fold': defaultdict(dict),
        'missing_files': [],
        'all_complete': False
    }
    
    if not experiment_path.exists():
        if verbose:
            print(f"❌ Experiment directory not found: {experiment_path}")
        return status
    
    if not eval_results_path.exists():
        if verbose:
            print(f"❌ eval_results directory not found: {eval_results_path}")
        return status
    
    # Check for fold directories
    for fold_num in range(num_folds):
        fold_dir = experiment_path / f"fold{fold_num}"
        if fold_dir.exists():
            status['folds_found'].append(fold_num)
        else:
            status['missing_folds'].append(fold_num)
    
    # Check for evaluation result files
    expected_splits = ['train', 'val', 'test']
    expected_types = ['patients.csv', 'sites.csv', 'metrics.json', 'complex_data.h5']
    
    for fold_num in range(num_folds):
        fold_status = {
            'train': {'complete': True, 'files': {}},
            'val': {'complete': True, 'files': {}},
            'test': {'complete': True, 'files': {}}
        }
        
        for split in expected_splits:
            for file_type in expected_types:
                filename = f"{split}_full_model_fold{fold_num}_{file_type}"
                filepath = eval_results_path / filename
                
                exists = filepath.exists()
                fold_status[split]['files'][file_type] = exists
                
                if not exists:
                    fold_status[split]['complete'] = False
                    status['missing_files'].append(str(filepath))
        
        status['results_by_fold'][fold_num] = fold_status
    
    # Check if all folds have complete results
    status['all_complete'] = (
        len(status['folds_found']) == num_folds and
        len(status['missing_files']) == 0
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*70}")
        print(f"Path: {experiment_path}")
        print(f"Eval Results: {eval_results_path}")
        print(f"\nFold Directories: {status['folds_found']} found, {status['missing_folds']} missing")
        
        for fold_num in range(num_folds):
            print(f"\nFold {fold_num}:")
            fold_data = status['results_by_fold'][fold_num]
            for split in expected_splits:
                split_data = fold_data[split]
                status_icon = "✓" if split_data['complete'] else "✗"
                missing_count = sum(1 for v in split_data['files'].values() if not v)
                print(f"  {status_icon} {split:5s}: {4 - missing_count}/4 files")
                
                if not split_data['complete'] and verbose:
                    for file_type, exists in split_data['files'].items():
                        if not exists:
                            print(f"      ❌ Missing: {file_type}")
        
        print(f"\n{'='*70}")
        if status['all_complete']:
            print("✅ All evaluation results complete and ready for visualization!")
        else:
            print(f"⚠️  Missing {len(status['missing_files'])} files")
        print(f"{'='*70}\n")
    
    return status


def check_all_experiments(experiments: list = None, num_folds: int = 5):
    """
    Check multiple experiments at once.
    
    Args:
        experiments: List of experiment names to check. If None, scans the base directory.
        num_folds: Number of folds expected per experiment
    """
    if experiments is None:
        # Auto-discover experiments
        base_path = Path(CLUSTER_BASE)
        if base_path.exists():
            experiments = [d.name for d in base_path.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
        else:
            print(f"❌ Cluster base directory not found: {CLUSTER_BASE}")
            return {}
    
    results = {}
    summary = {
        'total': len(experiments),
        'complete': 0,
        'incomplete': 0,
        'missing': 0
    }
    
    for exp in experiments:
        status = check_experiment_results(exp, num_folds, verbose=False)
        results[exp] = status
        
        if not status['experiment_exists']:
            summary['missing'] += 1
        elif status['all_complete']:
            summary['complete'] += 1
        else:
            summary['incomplete'] += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Total experiments: {summary['total']}")
    print(f"✅ Complete: {summary['complete']}")
    print(f"⚠️  Incomplete: {summary['incomplete']}")
    print(f"❌ Not found: {summary['missing']}")
    print(f"{'='*70}\n")
    
    # Detailed table
    print(f"{'Experiment':<25} {'Folds':<10} {'Status':<15} {'Missing Files'}")
    print("-" * 70)
    
    for exp, status in results.items():
        if not status['experiment_exists']:
            print(f"{exp:<25} {'N/A':<10} {'Not Found':<15} N/A")
        else:
            folds_str = f"{len(status['folds_found'])}/{num_folds}"
            status_str = "✅ Complete" if status['all_complete'] else "⚠️ Incomplete"
            missing_count = len(status['missing_files'])
            print(f"{exp:<25} {folds_str:<10} {status_str:<15} {missing_count}")
    
    return results


def save_status_report(results: dict, output_file: str = "eval_status_report.json"):
    """Save detailed status report to JSON file."""
    # Convert defaultdict to dict for JSON serialization
    serializable_results = {}
    for exp, status in results.items():
        serializable_results[exp] = {
            'experiment_exists': status['experiment_exists'],
            'eval_results_exists': status['eval_results_exists'],
            'folds_found': status['folds_found'],
            'missing_folds': status['missing_folds'],
            'results_by_fold': dict(status['results_by_fold']),
            'missing_files': status['missing_files'],
            'all_complete': status['all_complete']
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check evaluation results completeness")
    parser.add_argument('--experiment', type=str, help='Specific experiment to check')
    parser.add_argument('--experiments', nargs='+', help='List of experiments to check')
    parser.add_argument('--all', action='store_true', help='Check all experiments in base directory')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--save-report', type=str, help='Save detailed report to JSON file')
    
    args = parser.parse_args()
    
    if args.experiment:
        # Check single experiment
        status = check_experiment_results(args.experiment, args.num_folds, verbose=True)
        if args.save_report:
            save_status_report({args.experiment: status}, args.save_report)
    
    elif args.experiments:
        # Check specific list of experiments
        results = check_all_experiments(args.experiments, args.num_folds)
        if args.save_report:
            save_status_report(results, args.save_report)
    
    elif args.all:
        # Check all experiments
        results = check_all_experiments(None, args.num_folds)
        if args.save_report:
            save_status_report(results, args.save_report)
    
    else:
        # Default: check common experiments
        common_experiments = [
            'inception3d', 'LeViT-RL', 'LeViT-Attention',
            '3dcnn', 'cnn_lstm', 'attention_pool', 'mean_pool',
            'r2plus1d', 'uniform', 'singletask', 'no_rl_full_train'
        ]
        print("Checking common experiments...")
        results = check_all_experiments(common_experiments, args.num_folds)
        if args.save_report:
            save_status_report(results, args.save_report)
