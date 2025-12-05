"""Convert the current pytorch model to tf.js"""
import os
import shutil
import subprocess
import torch
import tensorflow as tf
import argparse

import torch.nn as nn

from ultr_ai.config import load_config
from ultr_ai.network_architecture import create_ablation_model


def convert_to_tfjs(model, output_dir, input_shape=(1, 3, 224, 224)):
    """
    Converts a PyTorch model to TensorFlow.js format.
    
    Args:
        model: The PyTorch model instance (already loaded).
        output_dir: The folder where the TF.js files will be saved.
        input_shape: Tuple for dummy input.
    """
    print(f"Converting model to TF.js format at {output_dir}...")

    # Paths for intermediate files
    onnx_path = "temp_model.onnx"
    tf_saved_model_path = "temp_tf_saved_model"

    try:
        # Try to export to ONNX
        print("Step 1/3: Exporting to ONNX...")
        model.eval()
        
        # Create dummy input based on the shape provided
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # From ONNX to TensorFlow SavedModel
        print("Step 2/3: Converting ONNX to TensorFlow SavedModel...")
        
        # Run the command using onnx2tf
        cmd_onnx2tf = [
            "onnx2tf", 
            "-i", onnx_path, 
            "-o", tf_saved_model_path,
            "-oiqt" 
        ]
        
        # Run command and check for errors
        subprocess.run(cmd_onnx2tf, check=True)

        # Convert to TensorFlow.js format
        print("Step 3/3: Converting SavedModel to web format...")

        # Ensure output directory is clean
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        cmd_tfjs = [
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            "--output_node_names=output",
            "--saved_model_tags=serve",
            tf_saved_model_path,
            output_dir
        ]

        subprocess.run(cmd_tfjs, check=True)
        
        print(f"\nSUCCESS: Model converted! Files located in '{output_dir}/'")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: A conversion command failed. Return code: {e.returncode}")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
    finally:
        # Clean
        print("Cleaning up temporary intermediate files...")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(tf_saved_model_path):
            shutil.rmtree(tf_saved_model_path)

def load_from_tfjs(model_path):
    """
    Loads a TensorFlow.js model and saves it as a TensorFlow SavedModel.
    
    Args:
        model_path: Path to the TF.js model directory.
        output_dir: Directory to save the TensorFlow SavedModel.
    """
    print(f"Loading TF.js model from {model_path}")
    
    try:
        # Load the TF.js model
        model = tf.keras.models.load_model(model_path)
        
        print(f"\nSUCCESS: Model loaded")
        return model
    
    except Exception as e:
        print(f"\nERROR: An error occurred while loading or saving the model: {e}")



def main():
    parser = argparse.ArgumentParser(description='Convert model formats.')
    
    # Data arguments
    # parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='original')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='ULTR-CLIP/results',
                       help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')

    
    # Multi-fold processing
    # parser.add_argument('--process-all-folds', action='store_true',
    #                    help='Process all folds at once')
    # parser.add_argument('--config-pattern', type=str,
    #                    help='Pattern for config files (e.g., "configs/fold_{}.yaml")')
    # parser.add_argument('--model-pattern', type=str,
    #                    help='Pattern for model files (e.g., "checkpoints/fold_{}/best.pth")')
    # parser.add_argument('--num-folds', type=int, default=5,
    #                    help='Number of folds to process')
    
    args = parser.parse_args()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load config
    config = load_config(config_file=args.config)
    # Optional override for video folder
    # if args.video_folder:
    #     try:
    #         old_vf = getattr(config, 'video_folder', None)
    #     except Exception:
    #         old_vf = None
    #     setattr(config, 'video_folder', args.video_folder)
    #     print(f"Overriding video_folder: {old_vf} -> {args.video_folder}")

    # Load model
    model = create_ablation_model(args.model_type, config)
    print(f"Model initialized: {args.model_type}")
    checkpoint = torch.load(args.model, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Successfully loaded model")
    model.eval()

    # Testing model

    model = nn.Sequential(
        nn.linear(10, 20),
        nn.ReLU(),
        nn.linear(20, 5)
    )
    test = torch.ones((1, 10))
    with torch.no_grad():
        output = model(test)
    print(f"Test output: {output}")

    # Convert to TF.js
    convert_to_tfjs(model, args.output_dir)

    # Load from TF.js
    tf_model = load_from_tfjs(args.output_dir)
    test = tf.ones((1, 10))
    output = tf_model(test)
    print(f"TF.js model test output: {output}")
    
    # Setup data
    # data_module = LungUltrasoundDataModule(
    #     root_dir=config.root_dir,
    #     labels_csv=config.labels_csv,
    #     file_metadata_csv=config.file_metadata_csv,
    #     image_folder=config.image_folder,
    #     video_folder=config.video_folder,
    #     split_csv=config.split_csv,
    #     batch_size=1,
    #     num_workers=config.num_workers,
    #     frame_sampling=config.frame_sampling,
    #     depth_filter=config.depth_filter,
    #     cache_size=100,
    # )
    
    # data_module.setup(stage='patient_level')
    # videos = next(iter(data_module.patient_level_dataloader('test')))
    # print(videos["site_videos"].shape) # torch.Size([1, 24, 32, 3, 224, 224])

    # Compute efficiency metrics
    # num_params = compute_num_params(model)
    # dtype_counts = dtype_of_weights(model)

    # print(f"Number of parameters: {num_params}")
    # print(f"dtype counts: {dtype_counts}")
    # print(f"Estimated model size (MB): {compute_memory_usage(num_params, dtype_counts)}")
    # inference_time = compute_inference_time(model, data_module.patient_level_dataloader('test'), device)
    # print(f"Average inference time per batch (s): {inference_time}")



if __name__ == '__main__':
    main()