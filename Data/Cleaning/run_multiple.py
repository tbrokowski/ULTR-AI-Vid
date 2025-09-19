#!/usr/bin/env python3

import subprocess
import time
import sys
import os
from datetime import datetime

def run_command(cmd, iteration, total):
    """Run the given command and handle its execution"""
    start_time = datetime.now()
    print(f"\n[{start_time}] Starting execution {iteration}/{total}...")
    print(f"Running command: {cmd}")
    
    try:
        # Execute the command and wait for it to complete
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, text=True)
        
        # Real-time output handling
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()
        
        # Get the return code
        return_code = process.wait()
        end_time = datetime.now()
        duration = end_time - start_time
        
        if return_code == 0:
            print(f"[{end_time}] Execution {iteration}/{total} completed successfully.")
            print(f"Duration: {duration}")
            return True
        else:
            error_output = process.stderr.read()
            print(f"[{end_time}] Execution {iteration}/{total} failed with return code {return_code}")
            print(f"Error: {error_output}")
            return False
            
    except Exception as e:
        print(f"Exception occurred during execution {iteration}/{total}: {str(e)}")
        return False

def main():
    # Base command and output path
    base_cmd = "python3 /gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/Cleaning/process_directories.py"
    output_path = "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v2"
    
    # Define the different input paths
    input_paths = [
        "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/Uncleaned_Videos/VideoDownloads/Uncleaned_v1",
        "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/Uncleaned_Videos/VideoDownloads/Uncleaned_v2",
        "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/Uncleaned_Videos/VideoDownloads/Uncleaned_v3",
        "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/Uncleaned_Videos/VideoDownloads/Uncleaned_v4",
    ]
    
    print(f"Starting sequential execution with different inputs at {datetime.now()}")
    print("-" * 80)
    
    # Run commands with different inputs
    successful_runs = 0
    failed_runs = 0
    total_runs = len(input_paths)
    
    for i, input_path in enumerate(input_paths, 1):
        # Construct the full command
        command = f"{base_cmd} --input '{input_path}' --output '{output_path}'"
        
        # Run the command
        success = run_command(command, i, total_runs)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            
        # Optional: Add a short pause between runs
        if i < total_runs:
            print(f"Waiting 5 seconds before starting next execution...")
            time.sleep(5)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"All executions completed at {datetime.now()}")
    print(f"Summary: {successful_runs} successful runs, {failed_runs} failed runs")
    print("=" * 80)

if __name__ == "__main__":
    main()