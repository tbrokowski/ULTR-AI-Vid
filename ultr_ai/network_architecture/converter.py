"""Convert the current pytorch model to tf.js"""
import os
import sys
import shutil
import subprocess
import torch
import argparse
import warnings
import json
import numpy as np

warnings.filterwarnings("ignore", message="Protobuf gencode version")

import torch.nn as nn

# Add paths
PROJECT_ROOT = "."
SRC_PATH = "."
NETWORK_PATH = "."

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, NETWORK_PATH)

from ultr_ai.config import load_config
from ultr_ai.network_architecture import create_ablation_model

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def convert_to_tfjs(model, output_dir, input_shape=(1, 3, 224, 224)):
    """
    Converts PyTorch -> Keras (via Nobuco) -> TF.js
    """
    # Force TensorFlow to use legacy Keras (fixes Keras 3 compatibility issues)
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    print(f"Starting Nobuco conversion for input shape: {input_shape}")
    
    # 1. Imports inside function to avoid global dependency issues
    try:
        import tensorflow as tf
        import keras
        import nobuco
        from nobuco import ChannelOrder
        
        print(f"[DEBUG] TensorFlow Version: {tf.__version__}")
        print(f"[DEBUG] Keras Version: {keras.__version__}")
        
        if keras.__version__.startswith("3"):
            print("\n[CRITICAL WARNING] Keras 3 detected. Nobuco requires Keras 2.")
            print("Please run: pip install \"tensorflow<2.16\" \"keras<3\" --force-reinstall\n")
            
    except ImportError as e:
        print(f"Error: Missing dependencies. Please install: pip install nobuco tensorflow tensorflowjs")
        print(f"Details: {e}")
        return False

    # Ensure output directories exist
    tf_path = os.path.join(output_dir, "tf_saved_model")
    web_path = os.path.join(output_dir, "web_model")
    
    # Clean previous attempts
    if os.path.exists(tf_path): shutil.rmtree(tf_path)
    if os.path.exists(web_path): shutil.rmtree(web_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # --- Step 1: Convert PyTorch to Keras (TensorFlow) directly ---
        print("-> Converting PyTorch to Keras/TensorFlow via Nobuco...")
        model.eval()
        
        # Create dummy input on the correct device
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Nobuco conversion
        # inputs_channel_order=ChannelOrder.TENSORFLOW ensures NHWC format (better for Web/JS)
        keras_model = nobuco.pytorch_to_keras(
            model,
            args=[dummy_input],
            inputs_channel_order=ChannelOrder.TENSORFLOW, 
            outputs_channel_order=ChannelOrder.TENSORFLOW,
            save_trace_html=False
        )

        # Save as TensorFlow SavedModel
        print(f"-> Saving intermediate TF model to {tf_path}...")
        keras_model.save(tf_path, save_format="tf")

        # --- Step 2: Convert SavedModel to TF.js ---
        print("-> Converting SavedModel to TF.js format...")
        
        if shutil.which('tensorflowjs_converter') is None:
            print("Error: 'tensorflowjs_converter' not found in PATH.")
            print("Please run: pip install tensorflowjs")
            return False

        cmd = [
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            # Skip op check allows custom/complex layers to pass through
            "--skip_op_check", 
            tf_path,
            web_path
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"Success! Web model saved to: {web_path}")
        return True

    except Exception as e:
        print(f"\nConversion failed: {e}")
        return False


def load_from_tfjs(model_path):
    """
    Loads a TensorFlow.js model for inference testing.
    
    For TF.js graph models, we need to use TFSMLayer or load the SavedModel directly.
    This function provides a way to verify the converted model.
    
    Args:
        model_path: Path to the output directory containing 'tf_saved_model'.
    
    Returns:
        A callable model or None if loading fails.
    """
    import tensorflow as tf
    
    # Python cannot natively load the 'web_model/model.json' (that is for JS).
    # However, our conversion process generates a 'tf_saved_model' folder 
    # which IS loadable by Python and is mathematically identical.
    tf_saved_model_path = os.path.join(model_path, "tf_saved_model")
    
    print(f"Loading verification model from {tf_saved_model_path}")
    
    if not os.path.exists(tf_saved_model_path):
        print("Error: TF SavedModel not found. Cannot verify in Python.")
        return None

    try:
        loaded_model = tf.saved_model.load(tf_saved_model_path)
        inference_func = loaded_model.signatures['serving_default']
        
        # Wrap the TF function to return a tensor directly (matching PyTorch behavior for the test)
        def model_wrapper(x):
            # ONNX export usually names inputs 'input' or 'input_1'
            # We inspect the signature to be safe
            key = list(inference_func.structured_input_signature[1].keys())[0]
            out = inference_func(**{key: x})
            # Return the first output tensor
            return list(out.values())[0]
            
        return model_wrapper
        
    except Exception as e:
        print(f"Failed to load TF model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert model formats.')
    
    # Data arguments
    # parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='original')
    parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='ULTR-CLIP/results',
                        help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')

    
    # Multi-fold processing
    # parser.add_argument('--process-all-folds', action='store_true',
    #                     help='Process all folds at once')
    # parser.add_argument('--config-pattern', type=str,
    #                     help='Pattern for config files (e.g., "configs/fold_{}.yaml")')
    # parser.add_argument('--model-pattern', type=str,
    #                     help='Pattern for model files (e.g., "checkpoints/fold_{}/best.pth")')
    # parser.add_argument('--num-folds', type=int, default=5,
    #                     help='Number of folds to process')
    
    args = parser.parse_args()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load config
    # config = load_config(config_file=args.config)
    # Optional override for video folder
    # if args.video_folder:
    #     try:
    #         old_vf = getattr(config, 'video_folder', None)
    #     except Exception:
    #         old_vf = None
    #     setattr(config, 'video_folder', args.video_folder)
    #     print(f"Overriding video_folder: {old_vf} -> {args.video_folder}")

    # Load model
    # model = create_ablation_model(args.model_type, config)
    # print(f"Model initialized: {args.model_type}")
    # checkpoint = torch.load(args.model, weights_only=False)
    # if 'model_state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     model.load_state_dict(checkpoint)
    # print("Successfully loaded model")
    # model.eval()

    # Testing model

    model = SimpleModel()
    model.eval()
    test = torch.ones((1, 10))
    with torch.no_grad():
        output = model(test)
    print(f"PyTorch test output: {output}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    dummy_input_shape = (1, 10)
    dummy_input = torch.randn(*dummy_input_shape)

    # Convert to ONNX
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        dynamo=True,
        verbose=True)
     

    # # Convert to TF.js
    # success = convert_to_tfjs(model, args.output_dir, input_shape=(1, 10))

    # if success:
    #     # Load from TF.js and test
    #     tf_model = load_from_tfjs(args.output_dir)
    #     if tf_model is not None:
    #         import tensorflow as tf
    #         test_tf = tf.ones((1, 10), dtype=tf.float32)
    #         output_tf = tf_model(test_tf)
    #         print(f"TF.js model test output: {output_tf}")
            
    #         # Compare outputs
    #         print("\n=== Comparison ===")
    #         print(f"PyTorch output: {output.numpy()}")
    #         print(f"TF.js output:   {output_tf.numpy()}")
    #     else:
    #         print("Failed to load TF.js model for testing")
    # else:
    #     print("Conversion failed, skipping TF.js model test")
    
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