import os
import sys
import logging
import json
import h5py
import torch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
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

from ultr_ai.dataset import LungUltrasoundDataModule
from ultr_ai.config import load_config

from ultr_ai.network_architecture import create_ablation_model

logger = logging.getLogger(__name__)

def compute_num_params(model):
    """Compute the number of parameters for each component of the model."""
    param_counts = {}
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        component_name = name.split('.')[0]
        if component_name not in param_counts:
            param_counts[component_name] = 0
        param_counts[component_name] += num_params
    param_counts['total'] = total_params
    return param_counts

def dtype_of_weights(model):
    """ Compute the types of the weights in the model. """
    type_counts = {}
    for name, param in model.named_parameters():
        param_type = str(param.dtype)
        if param_type not in type_counts:
            type_counts[param_type] = 0
        type_counts[param_type] += param.numel()
    return type_counts

def compute_memory_usage(num_params, dtype_counts):
    """Estimate the memory size of the model based on parameter counts and data types."""
    type_size_map = {
        'torch.float32': 4,
        'torch.float16': 2,
        'torch.int64': 8,
        'torch.int32': 4,
        'torch.uint8': 1,
    }
    
    total_size_bytes = 0
    for dtype, count in dtype_counts.items():
        size_per_param = type_size_map.get(dtype, 4)  # Default to 4 bytes if unknown
        total_size_bytes += count * size_per_param
    
    return total_size_bytes / (1000 ** 2)  # Convert to MB


def compute_inference_time(model, dataloader, device, num_iterations=100):
    """Compute average inference time over a number of iterations."""
    model.to(device)
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Measuring Inference Time")):
            if i >= num_iterations:
                break
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
           
            outputs = model(batch)
    
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / num_iterations
    return avg_time_per_batch


def main():
    parser = argparse.ArgumentParser(description='Evaluate inference efficiency of models.')
    
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
        print(f"Overriding video_folder: {old_vf} -> {args.video_folder}")

    # Load model
    model = create_ablation_model(args.model_type, config)
    print(f"Model initialized: {args.model_type}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
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
        batch_size=1,
        num_workers=config.num_workers,
        frame_sampling=config.frame_sampling,
        depth_filter=config.depth_filter,
        cache_size=100,
    )
    
    data_module.setup(stage='patient_level')

    videos = next(iter(data_module.patient_level_dataloader('test')))
    # print(videos["site_videos"].shape) # torch.Size([1, 24, 32, 3, 224, 224])

    # Compute efficiency metrics
    num_params = compute_num_params(model)
    dtype_counts = dtype_of_weights(model)

    print(f"Number of parameters: {num_params}")
    print(f"dtype counts: {dtype_counts}")
    print(f"Estimated model size (MB): {compute_memory_usage(num_params, dtype_counts)}")
    inference_time = compute_inference_time(model, data_module.patient_level_dataloader('test'), device)
    print(f"Average inference time per batch (s): {inference_time}")
    
    # if args.process_all_folds:
    #     # Process all folds
    #     if not args.config_pattern or not args.model_pattern:
    #         print("Error: --config-pattern and --model-pattern required for --process-all-folds")
    #         return
        
    #     config_paths = [args.config_pattern.format(i) for i in range(args.num_folds)]
    #     model_paths = [args.model_pattern.format(i) for i in range(args.num_folds)]

    #     process_all_folds(args.model_type, config_paths, model_paths, args.output_dir, video_folder_override=args.video_folder)

    # else:
    #     # Process single fold
    #     if not args.config or not args.model:
    #         print("Error: --config and --model required for single fold processing")
    #         return
        
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    #     # Load config
    #     config = load_config(config_file=args.config)
    #     # Optional override for video folder
    #     if args.video_folder:
    #         try:
    #             old_vf = getattr(config, 'video_folder', None)
    #         except Exception:
    #             old_vf = None
    #         setattr(config, 'video_folder', args.video_folder)
    #         print(f"✓ Overriding video_folder: {old_vf} -> {args.video_folder}")
        
    #     # Load model
    #     model = create_ablation_model(args.model_type, config)
    #     print(f"✓ Model initialized: {args.model_type}")
    #     checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    #     if 'model_state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #     else:
    #         model.load_state_dict(checkpoint)
    #     model = model.to(device)
    #     print("Successfully loaded model")
    #     model.eval()
        
    #     # Setup data
    #     data_module = LungUltrasoundDataModule(
    #         root_dir=config.root_dir,
    #         labels_csv=config.labels_csv,
    #         file_metadata_csv=config.file_metadata_csv,
    #         image_folder=config.image_folder,
    #         video_folder=config.video_folder,
    #         split_csv=config.split_csv,
    #         batch_size=config.batch_size,
    #         num_workers=config.num_workers,
    #         frame_sampling=config.frame_sampling,
    #         depth_filter=config.depth_filter,
    #         cache_size=100,
    #     )
        
    #     data_module.setup(stage='patient_level')
        
    #     # Process each split
    #     # Determine default output dir if user left default
    #     output_dir = args.output_dir
    #     if output_dir == 'ULTR-CLIP/results' or not output_dir:
    #         output_dir = _infer_eval_output_dir_from_config(config)
    #         print(f"Inferred output dir: {output_dir}")

    #     for split_name in ['train', 'val', 'test']:
    #         print(f"\nProcessing {split_name} split...")
            
    #         dataloader = data_module.patient_level_dataloader(split_name)
            
    #         active_tasks = getattr(config, 'active_tasks', ['TB Label'])
    #         use_pathology_loss = getattr(config, 'use_pathology_loss', True)
            
    #         patient_df, site_df, complex_data, metrics = evaluate_model_for_downstream(
    #             model, dataloader, device, active_tasks, use_pathology_loss
    #         )
            
    #         saved_files = save_for_downstream_pipeline(
    #             patient_df, site_df, complex_data, metrics,
    #             output_dir, split_name, args.fold
    #         )
            
    #         verify_compatibility(output_dir, split_name, args.fold)
        
    #     print(f"\n✅ Fold {args.fold} processing complete!")


if __name__ == "__main__":
    # Example usage
    main()