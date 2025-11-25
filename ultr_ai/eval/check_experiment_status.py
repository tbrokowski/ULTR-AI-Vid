#!/usr/bin/env python3
"""
Script to check the status of all ablation experiments.
"""

import os
import glob
import re
from pathlib import Path
from collections import defaultdict

def check_log_for_completion(log_file):
    """Check if a log file indicates successful training and evaluation."""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for various success indicators
        has_training = 'Training started' in content or 'Epoch' in content
        has_best_model = 'Best model' in content or 'best_model' in content
        has_evaluation = 'Evaluation' in content or 'eval' in content.lower()
        has_metrics = 'AUC' in content or 'Accuracy' in content or 'F1' in content
        has_error = 'Error' in content or 'Exception' in content or 'Traceback' in content
        has_cuda_oom = 'CUDA out of memory' in content or 'OutOfMemoryError' in content
        has_killed = 'Killed' in content or 'SIGKILL' in content
        has_completed = 'Training completed' in content or 'Finished' in content
        
        # Check for specific completion phrases
        has_final_results = 'Final' in content and ('results' in content.lower() or 'metrics' in content.lower())
        
        return {
            'has_training': has_training,
            'has_best_model': has_best_model,
            'has_evaluation': has_evaluation,
            'has_metrics': has_metrics,
            'has_error': has_error,
            'has_cuda_oom': has_cuda_oom,
            'has_killed': has_killed,
            'has_completed': has_completed,
            'has_final_results': has_final_results,
            'file_size': os.path.getsize(log_file)
        }
    except Exception as e:
        return {'error': str(e), 'file_size': 0}

def check_experiment(exp_dir):
    """Check all folds for an experiment."""
    exp_name = os.path.basename(exp_dir)
    results = {}
    
    # Find all fold directories
    fold_dirs = sorted(glob.glob(os.path.join(exp_dir, 'fold*')))
    
    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        
        # Find log files
        out_files = glob.glob(os.path.join(fold_dir, '**', '*.out'), recursive=True)
        err_files = glob.glob(os.path.join(fold_dir, '**', '*.err'), recursive=True)
        
        fold_status = {
            'out_files': out_files,
            'err_files': err_files,
            'status': 'unknown'
        }
        
        # Check .out file
        if out_files:
            out_status = check_log_for_completion(out_files[0])
            fold_status['out_status'] = out_status
            
            # Determine overall status
            if out_status.get('has_error') or out_status.get('has_cuda_oom'):
                fold_status['status'] = 'FAILED - Error/OOM'
            elif out_status.get('has_killed'):
                fold_status['status'] = 'FAILED - Killed'
            elif out_status.get('has_completed') or (out_status.get('has_evaluation') and out_status.get('has_metrics')):
                fold_status['status'] = 'COMPLETED'
            elif out_status.get('has_training'):
                fold_status['status'] = 'INCOMPLETE - Training started but not finished'
            elif out_status.get('file_size', 0) == 0:
                fold_status['status'] = 'EMPTY - No output'
            else:
                fold_status['status'] = 'UNKNOWN'
        else:
            fold_status['status'] = 'NO LOGS FOUND'
            
        # Check .err file for errors
        if err_files:
            with open(err_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                err_content = f.read()
            fold_status['err_size'] = os.path.getsize(err_files[0])
            if len(err_content.strip()) > 0:
                fold_status['has_stderr'] = True
                # Look for specific errors
                if 'CUDA out of memory' in err_content:
                    fold_status['status'] = 'FAILED - CUDA OOM'
                elif 'Error' in err_content or 'Exception' in err_content:
                    fold_status['status'] = 'FAILED - Error in stderr'
        
        results[fold_name] = fold_status
    
    return results

def main():
    base_dir = '/users/mbarbiere/ULTR-AI/ULTR-AI-Vid/ablation_results'
    
    # Get all experiment directories (exclude 'logs' subdirectory)
    exp_dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) 
                if os.path.isdir(d) and os.path.basename(d) != 'logs']
    
    all_results = {}
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        print(f"\n{'='*80}")
        print(f"Checking experiment: {exp_name}")
        print(f"{'='*80}")
        
        results = check_experiment(exp_dir)
        all_results[exp_name] = results
        
        # Print summary for this experiment
        fold_statuses = [results[fold]['status'] for fold in results]
        completed = sum(1 for s in fold_statuses if 'COMPLETED' in s)
        failed = sum(1 for s in fold_statuses if 'FAILED' in s)
        incomplete = sum(1 for s in fold_statuses if 'INCOMPLETE' in s)
        
        print(f"\nSummary for {exp_name}:")
        print(f"  Total folds: {len(results)}")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")
        print(f"  Incomplete: {incomplete}")
        print(f"  Other: {len(results) - completed - failed - incomplete}")
        
        # Print details for each fold
        for fold_name in sorted(results.keys()):
            fold_data = results[fold_name]
            print(f"\n  {fold_name}: {fold_data['status']}")
            if 'out_status' in fold_data:
                out_status = fold_data['out_status']
                print(f"    - Log size: {out_status.get('file_size', 0)} bytes")
                print(f"    - Has training: {out_status.get('has_training', False)}")
                print(f"    - Has evaluation: {out_status.get('has_evaluation', False)}")
                print(f"    - Has metrics: {out_status.get('has_metrics', False)}")
                print(f"    - Has errors: {out_status.get('has_error', False)}")
                if out_status.get('has_cuda_oom'):
                    print(f"    - CUDA OOM detected!")
                if out_status.get('has_killed'):
                    print(f"    - Process was killed!")
            if fold_data.get('has_stderr'):
                print(f"    - Stderr size: {fold_data.get('err_size', 0)} bytes")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_experiments = len(all_results)
    fully_completed = 0
    partially_completed = 0
    failed_experiments = 0
    
    for exp_name, results in all_results.items():
        fold_statuses = [results[fold]['status'] for fold in results]
        completed = sum(1 for s in fold_statuses if 'COMPLETED' in s)
        failed = sum(1 for s in fold_statuses if 'FAILED' in s)
        
        if completed == len(results):
            fully_completed += 1
        elif completed > 0:
            partially_completed += 1
        if failed > 0:
            failed_experiments += 1
    
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Fully completed (all folds): {fully_completed}")
    print(f"Partially completed (some folds): {partially_completed}")
    print(f"With failures: {failed_experiments}")
    
    print(f"\n{'='*80}")
    print("READY FOR VISUALIZATION?")
    print(f"{'='*80}")
    
    for exp_name in sorted(all_results.keys()):
        results = all_results[exp_name]
        fold_statuses = [results[fold]['status'] for fold in results]
        completed = sum(1 for s in fold_statuses if 'COMPLETED' in s)
        total_folds = len(results)
        
        if completed == total_folds:
            status = "✓ READY"
        elif completed >= 4:  # Most folds completed
            status = "⚠ MOSTLY READY"
        else:
            status = "✗ NOT READY"
        
        print(f"{status:20} {exp_name:30} ({completed}/{total_folds} folds)")

if __name__ == '__main__':
    main()
