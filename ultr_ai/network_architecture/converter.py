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
from ultr_ai.dataset import LungUltrasoundDataModule

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


class ONNXModelWrapper(nn.Module):
    """
    Wrapper module for ONNX export.
    Converts dictionary inputs to positional tensor arguments and extracts only tensor outputs.
    """
    def __init__(self, model):
        super(ONNXModelWrapper, self).__init__()
        self.model = model
    
    def forward(
        self,
        site_videos,
        site_images,
        site_findings,
        site_indices,
        site_counts,
        site_masks,
        batch_padding_masks,
        real_data_masks,
        tb_labels,
        pneumonia_labels,
        covid_labels
    ):
        """
        Forward pass that reconstructs the dictionary input expected by the model.
        
        Args:
            All tensor inputs from the original batch dictionary
        
        Returns:
            Tuple of tensor outputs only (task logits) - traceable by ONNX
        """
        # Reconstruct the batch dictionary
        batch = {
            "site_videos": site_videos,
            "site_images": site_images,
            "site_findings": site_findings,
            "site_indices": site_indices,
            "site_counts": site_counts,
            "site_masks": site_masks,
            "batch_padding_masks": batch_padding_masks,
            "real_data_masks": real_data_masks,
            "tb_labels": tb_labels,
            "pneumonia_labels": pneumonia_labels,
            "covid_labels": covid_labels,
            "_mask_type": "batch_padding"
        }
        
        # Get model outputs
        outputs = self.model(batch)
        
        # Extract only tensor outputs for ONNX export
        # Return task logits in a consistent order
        task_logits = outputs['task_logits']
        
        # Collect available task logits
        logits_list = []
        for task_name in ['TB Label', 'Pneumonia Label', 'COVID Label']:
            if task_name in task_logits and task_logits[task_name] is not None:
                logits_list.append(task_logits[task_name].unsqueeze(-1))  # [B] -> [B, 1]
        
        # Concatenate all logits into a single tensor [B, num_tasks]
        if logits_list:
            return torch.cat(logits_list, dim=-1)
        else:
            # Fallback: return zeros if no logits available
            return torch.zeros(site_videos.shape[0], 3)


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
    # import tensorflow as tf
    
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

def construct_dummy_input():
    # patient_ids: ['25-103']
    # tb_labels: torch.Size([1])
    # pneumonia_labels: torch.Size([1])
    # covid_labels: torch.Size([1])
    # site_indices: torch.Size([1, 24])
    # site_counts: torch.Size([1])
    # site_videos: torch.Size([1, 24, 32, 3, 224, 224])
    # site_images: torch.Size([1, 24, 3, 224, 224])
    # site_findings: torch.Size([1, 24, 4])
    # site_masks: torch.Size([1, 24])
    # batch_padding_masks: torch.Size([1, 24])
    # real_data_masks: torch.Size([1, 24])
    # _mask_type: batch_padding

    return {
        "patient_ids": ['25-103'],
        "tb_labels": torch.randint(0, 2, (1,)),
        "pneumonia_labels": torch.randint(0, 2, (1,)),
        "covid_labels": torch.randint(0, 2, (1,)),
        "site_indices": torch.randint(0, 10, (1, 24)),
        "site_counts": torch.randint(1, 25, (1,)),
        "site_videos": torch.randn((1, 24, 32, 3, 224, 224)),
        "site_images": torch.randn((1, 24, 3, 224, 224)),
        "site_findings": torch.randn((1, 24, 4)),
        "site_masks": torch.randint(0, 2, (1, 24), dtype=torch.bool),
        "batch_padding_masks": torch.randint(0, 2, (1, 24), dtype=torch.bool),
        "real_data_masks": torch.randint(0, 2, (1, 24), dtype=torch.bool),
        "_mask_type": "batch_padding"
    }


def main():
    parser = argparse.ArgumentParser(description='Convert model formats.')
    
    # Data arguments
    # parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='attention_pool')
    parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory', default="C:\\Users\\mattb\\Documents\\ULTR-AI-Checkpoints")
    parser.add_argument('--config', type=str, help='Path to config file', default="configs/attention_pool_extra3_full_train2/fold0.yaml")
    parser.add_argument('--model', type=str, help='Path to model checkpoint', default="C:\\Users\\mattb\\Documents\\ULTR-AI-Checkpoints\\fold0\\checkpoint_best.pth")
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
    checkpoint = torch.load(args.model, weights_only=False, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Successfully loaded model")
    model.eval()

    # Testing model

    model_simple = SimpleModel()
    model_simple.eval()
    test = torch.ones((1, 10))
    with torch.no_grad():
        output = model_simple(test)
    print(f"PyTorch test output: {output}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    dummy_input_shape_simple = (1, 10)
    dummy_input = torch.randn(*dummy_input_shape_simple)

    # Convert to ONNX
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    torch.onnx.export(
        model_simple,
        dummy_input,
        onnx_path,
        dynamo=True,
        verbose=True)
    
    dummy_input_shape = (1, 24, 32, 3, 224, 224)
    dummy_input = torch.randn(*dummy_input_shape)

    model.eval()
    with torch.no_grad():
        output = model(construct_dummy_input())
        print(output)

    # Create ONNX-compatible wrapper
    print("\n=== Wrapping model for ONNX export ===")
    wrapped_model = ONNXModelWrapper(model)
    wrapped_model.eval()
    
    # Prepare individual tensor inputs (no dictionaries)
    dummy_batch = construct_dummy_input()
    dummy_inputs = (
        dummy_batch["site_videos"],
        dummy_batch["site_images"],
        dummy_batch["site_findings"],
        dummy_batch["site_indices"],
        dummy_batch["site_counts"],
        dummy_batch["site_masks"],
        dummy_batch["batch_padding_masks"],
        dummy_batch["real_data_masks"],
        dummy_batch["tb_labels"],
        dummy_batch["pneumonia_labels"],
        dummy_batch["covid_labels"]
    )
    
    # Verify wrapper works
    with torch.no_grad():
        wrapped_output = wrapped_model(*dummy_inputs)
        print("Wrapped model output:", wrapped_output)

    # Convert to ONNX
    onnx_path = os.path.join(args.output_dir, "model_full.onnx")
    print(f"\n=== Exporting to ONNX: {onnx_path} ===")
    
    # Define dynamic axes for tensors that have variable dimensions
    dynamic_axes = {
        "site_videos": {0: "batch", 1: "num_sites"},
        "site_images": {0: "batch", 1: "num_sites"},
        "site_findings": {0: "batch", 1: "num_sites"},
        "site_indices": {0: "batch", 1: "num_sites"},
        "site_counts": {0: "batch"},
        "site_masks": {0: "batch", 1: "num_sites"},
        "batch_padding_masks": {0: "batch", 1: "num_sites"},
        "real_data_masks": {0: "batch", 1: "num_sites"},
        "tb_labels": {0: "batch"},
        "pneumonia_labels": {0: "batch"},
        "covid_labels": {0: "batch"},
        "logits": {0: "batch"}  # Single output: [batch, num_tasks]
    }
    
    # Use traditional TorchScript-based export (dynamo doesn't support data-dependent control flow)
    print("Using traditional TorchScript-based ONNX export...")
    print("Note: Your model has data-dependent control flow which requires TorchScript tracing.\n")
    
    # Force legacy TorchScript-based exporter by setting dynamo=False explicitly
    torch.onnx.export(
        wrapped_model,
        dummy_inputs,
        onnx_path,
        input_names=[
            "site_videos",
            "site_images", 
            "site_findings",
            "site_indices",
            "site_counts",
            "site_masks",
            "batch_padding_masks",
            "real_data_masks",
            "tb_labels",
            "pneumonia_labels",
            "covid_labels"
        ],
        output_names=["logits"],  # Single concatenated output [batch, num_tasks]
        dynamic_axes=dynamic_axes,
        opset_version=18,  # Use opset 18 as suggested by the warning
        do_constant_folding=True,
        dynamo=False,  # Explicitly disable dynamo to use legacy TorchScript exporter
        verbose=True
    )
    print(f"\n✓ ONNX export completed: {onnx_path}")
    
    # Test ONNX model
    print(f"\n{'='*60}")
    print("Testing ONNX Model")
    print(f"{'='*60}\n")
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        print(f"Loading ONNX model from: {onnx_path}")
        ort_session = ort.InferenceSession(onnx_path)
        
        # Print model info
        print(f"\nONNX Model Info:")
        print(f"  Inputs:")
        for input_meta in ort_session.get_inputs():
            print(f"    - {input_meta.name}: {input_meta.shape} ({input_meta.type})")
        print(f"  Outputs:")
        for output_meta in ort_session.get_outputs():
            print(f"    - {output_meta.name}: {output_meta.shape} ({output_meta.type})")
        
        # Prepare ONNX inputs - only include inputs the model actually expects
        # Get the list of expected input names from the ONNX model
        expected_inputs = {inp.name for inp in ort_session.get_inputs()}
        
        # Map of all possible inputs
        all_inputs = {
            "site_videos": dummy_batch["site_videos"].numpy(),
            "site_images": dummy_batch["site_images"].numpy(),
            "site_findings": dummy_batch["site_findings"].numpy(),
            "site_indices": dummy_batch["site_indices"].numpy(),
            "site_counts": dummy_batch["site_counts"].numpy(),
            "site_masks": dummy_batch["site_masks"].numpy(),
            "batch_padding_masks": dummy_batch["batch_padding_masks"].numpy(),
            "real_data_masks": dummy_batch["real_data_masks"].numpy(),
            "tb_labels": dummy_batch["tb_labels"].numpy(),
            "pneumonia_labels": dummy_batch["pneumonia_labels"].numpy(),
            "covid_labels": dummy_batch["covid_labels"].numpy()
        }
        
        # Only use inputs that the ONNX model expects
        onnx_inputs = {name: all_inputs[name] for name in expected_inputs if name in all_inputs}
        
        print(f"\nUsing {len(onnx_inputs)} inputs: {list(onnx_inputs.keys())}")
        
        # Run ONNX inference
        print("Running ONNX inference...")
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]  # [batch, num_tasks]
        
        print(f"ONNX output shape: {onnx_logits.shape}")
        print(f"ONNX logits:\n{onnx_logits}")
        
        # Compare with wrapped PyTorch output
        pytorch_logits = wrapped_output.numpy()
        
        print(f"\nPyTorch output shape: {pytorch_logits.shape}")
        print(f"PyTorch logits:\n{pytorch_logits}")
        
        # Compute differences
        diff = np.abs(onnx_logits - pytorch_logits)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n{'='*60}")
        print("Comparison Results")
        print(f"{'='*60}\n")
        print(f"Maximum absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # Check if outputs match
        tolerance = 1e-5
        if np.allclose(onnx_logits, pytorch_logits, atol=tolerance):
            print(f"\n✓ PASSED: Outputs match within tolerance ({tolerance})")
        else:
            print(f"\n⚠ WARNING: Outputs differ by more than tolerance ({tolerance})")
            print(f"Difference matrix:\n{diff}")
        
        # Show probabilities
        print(f"\n{'='*60}")
        print("Probability Predictions")
        print(f"{'='*60}\n")
        
        onnx_probs = 1 / (1 + np.exp(-onnx_logits))
        pytorch_probs = 1 / (1 + np.exp(-pytorch_logits))
        
        task_names = ['TB', 'Pneumonia', 'COVID']
        print("ONNX Probabilities:")
        for i, task in enumerate(task_names):
            if i < onnx_probs.shape[1]:
                print(f"  {task}: {onnx_probs[0, i]:.4f}")
        
        print("\nPyTorch Probabilities:")
        for i, task in enumerate(task_names):
            if i < pytorch_probs.shape[1]:
                print(f"  {task}: {pytorch_probs[0, i]:.4f}")
        
        print(f"\n{'='*60}\n")
        
    except ImportError:
        print("\n⚠ WARNING: onnxruntime not installed. Skipping ONNX validation.")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"\n✗ ERROR during ONNX validation: {e}")
        import traceback
        traceback.print_exc()
    
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