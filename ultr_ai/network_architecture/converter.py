"""Convert the current pytorch model to tf.js"""

"""
Structure of the attention_pool model's forward input and output:

Input Dict:
    patient_ids: ['25-103']
    tb_labels: torch.Size([1])
    pneumonia_labels: torch.Size([1])
    covid_labels: torch.Size([1])
    site_indices: torch.Size([1, 24])
    site_counts: torch.Size([1])
    site_videos: torch.Size([1, 24, 32, 3, 224, 224])
    site_images: torch.Size([1, 24, 3, 224, 224])
    site_findings: torch.Size([1, 24, 4])
    site_masks: torch.Size([1, 24])
    batch_padding_masks: torch.Size([1, 24])
    real_data_masks: torch.Size([1, 24])
    _mask_type: batch_padding

Output Dict:
    task_logits: dict, example: {'TB Label': tensor([0.1041], grad_fn=<SqueezeBackward1>)}
    patient_pathology_scores: torch.Size([1, 4]), example: tensor([[-0.1297, -0.1474, -0.2922,  0.2340]], grad_fn=<SqueezeBackward1>)
    patient_features: torch.Size([1, 512]), example value: tensor([[-1.6737e+00,  1.2444e+00, ... , -1.1030e+00,  1.6990e+00]], grad_fn=<SqueezeBackward1>)
    pathology_scores: torch.Size([1, 24, 4]), example: torch.Size([1, 24, 4]), value: tensor([[[-0.6251, -0.7291, -1.4302,  1.1436], .... [ 0.0000,  0.0000,  0.0000,  0.0000]]], grad_fn=<StackBackward0>)
    mil_attention: torch.Size([1, 24]), example: tensor([[0.0971, 0.0000, 0.1073, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ..., 0.0000, grad_fn=<SoftmaxBackward0>)
    site_features: torch.Size([1, 24, 512]), example: tensor([[[ 0.8532,  0.2620,  0.8520,  ..., -0.8633, -0.0293,  0.8508], [ 0.8502,  0.2265,  0.8531,  ..., -0.8650, -0.0161,  0.8479], [ 0.8558,  0.2430,  0.8482,  ..., -0.8604, -0.0273,  0.8546], ..., grad_fn=<StackBackward0>)
    site_metadata: list of lists, where each sublits contains 10 dicts, example of a single dict: [...., [{'batch_idx': 0, 'site_idx': 7, 'selected_indices': None, 'action_logits': tensor([[1.0978, 1.0982, 1.0980, 1.0979, 1.0983, 1.0985, 1.0980, 1.0977, 1.0982,
                                                                                                                                    1.0979, 1.0980, 1.0979, 1.0978, 1.0977, 1.0983, 1.0979, 1.0980, 1.0980,
                                                                                                                                    1.0983, 1.0980, 1.0976, 1.0982, 1.0981, 1.0982, 1.0982, 1.0979, 1.0983,
                                                                                                                                    1.0979, 1.0982, 1.0982, 1.0980, 1.0983]],
                                                                                                                                grad_fn=<MaskedFillBackward0>), 'state_values': tensor([[0.]], requires_grad=True)}, ...], ...]
    site_outputs: list of 10 dicts, example of a single dict: [{'batch_idx': 0, 'site_idx': 9, 'selected_indices': None, 'action_logits': tensor([[1.1021, 1.1024, 1.1012, 1.1022, 1.1021, 1.1024, 1.1019, 1.1016, 1.1028,
                                                                                                                                                1.1019, 1.1026, 1.1024, 1.1022, 1.1030, 1.1020, 1.1021, 1.1025, 1.1020,
                                                                                                                                                1.1024, 1.1014, 1.1016, 1.1017, 1.1018, 1.1023, 1.1024, 1.1018, 1.1022,
                                                                                                                                                1.1024, 1.1025, 1.1022, 1.1017, 1.1023]],
                                                                grad_fn=<MaskedFillBackward0>), 'state_values': tensor([[0.]], requires_grad=True)}, ...]
    tb_logits: torch.Size([1]), example: tensor([0.1041], grad_fn=<SqueezeBackward1>)
"""
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
import onnxruntime as ort

# Set seed
torch.manual_seed(0)

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

# Flags
TEST_SIMPLE_MODEL = True

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

def load_model(args, config, simple=False):
    if simple:
        model = SimpleModel()
    else:
        model = create_ablation_model(args.model_type, config)
        print(f"Model initialized: {args.model_type}")
        checkpoint = torch.load(args.model, weights_only=False, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded model")

    model.eval()
    return model

def convert_to_onnx(model, output_path, simple=False):
    print(f"\n=== Exporting to ONNX: {output_path} ===")
    
    if simple:
        dummy_input_shape_simple = (1, 10)
        dummy_input = torch.randn(*dummy_input_shape_simple)

        # Convert to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            dynamo=True,
            # verbose=True
            )
        
    else:
        dummy_inputs = construct_dummy_input(to_numpy=False, as_dict=False)
        dynamic_axes = construct_dynamic_axes()
        
        # Force legacy TorchScript-based exporter by setting dynamo=False explicitly
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
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
            # verbose=True
        )
    print(f"ONNX model saved to: {output_path}")

def construct_dummy_input_dict():
    # Set seed
    torch.manual_seed(0)

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

def construct_dummy_input(to_numpy=False, as_dict=False):
    dummy_dict = construct_dummy_input_dict()
    dummy_tuple = (
        dummy_dict["site_videos"],
        dummy_dict["site_images"],
        dummy_dict["site_findings"],
        dummy_dict["site_indices"],
        dummy_dict["site_counts"],
        dummy_dict["site_masks"],
        dummy_dict["batch_padding_masks"],
        dummy_dict["real_data_masks"],
        dummy_dict["tb_labels"],
        dummy_dict["pneumonia_labels"],
        dummy_dict["covid_labels"]
    )
    if as_dict and not to_numpy:
        return dummy_dict
    elif as_dict and to_numpy:
        return {
            "site_videos": dummy_dict["site_videos"].numpy(),
            "site_images": dummy_dict["site_images"].numpy(),
            "site_findings": dummy_dict["site_findings"].numpy(),
            "site_indices": dummy_dict["site_indices"].numpy(),
            "site_counts": dummy_dict["site_counts"].numpy(),
            "site_masks": dummy_dict["site_masks"].numpy(),
            "batch_padding_masks": dummy_dict["batch_padding_masks"].numpy(),
            "real_data_masks": dummy_dict["real_data_masks"].numpy(),
            "tb_labels": dummy_dict["tb_labels"].numpy(),
            "pneumonia_labels": dummy_dict["pneumonia_labels"].numpy(),
            "covid_labels": dummy_dict["covid_labels"].numpy()
        }
    else:
        return dummy_tuple

def construct_dynamic_axes():
    return {
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

def run_inference_onnx(output_path):
    # Load ONNX model
        print(f"Loading ONNX model from: {output_path}")
        ort_session = ort.InferenceSession(output_path)
        
        # Print model info
        print(f"\nONNX Model Info:")
        print(f"  Inputs:")
        for input_meta in ort_session.get_inputs():
            print(f"    - {input_meta.name}: {input_meta.shape} ({input_meta.type})")
        print(f"  Outputs:")
        for output_meta in ort_session.get_outputs():
            print(f"    - {output_meta.name}: {output_meta.shape} ({output_meta.type})")
        
        # Prepare ONNX inputs - dynamically based on what the model expects
        expected_inputs = {inp.name for inp in ort_session.get_inputs()}
        
        # Create input dictionary based on expected inputs
        onnx_inputs = {}
        all_inputs = construct_dummy_input(to_numpy=True, as_dict=True)
        
        for input_name in expected_inputs:
            if input_name in all_inputs:
                # Use the prepared dummy inputs for the full model
                onnx_inputs[input_name] = all_inputs[input_name]
            elif input_name == 'x':
                # Simple model input: fixed shape (1, 10)
                np.random.seed(0)
                onnx_inputs[input_name] = np.random.randn(1, 10).astype(np.float32)
            else:
                raise ValueError(f"Unknown input '{input_name}' - please add handling for this input")
        
        print(f"\nUsing {len(onnx_inputs)} inputs: {list(onnx_inputs.keys())}")
        
        # Run ONNX inference
        print("Running ONNX inference...")
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]

        return onnx_logits

def compare_logits(onnx_logits, pytorch_logits):
    print(f"ONNX output shape: {onnx_logits.shape}")
    print(f"ONNX logits:\n{onnx_logits}")
    
    # Compare with wrapped PyTorch output
    pytorch_logits = pytorch_logits.numpy()
    
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
        print(f"\nPASSED: Outputs match within tolerance ({tolerance})")
    else:
        print(f"\nWARNING: Outputs differ by more than tolerance ({tolerance})")
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

def main():
    # ----------------------------------
    # Argument parsing
    # ----------------------------------
    parser = argparse.ArgumentParser(description='Convert model formats.')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='attention_pool')
    parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory', default="C:\\Users\\mattb\\Documents\\ULTR-AI-Checkpoints")
    parser.add_argument('--config', type=str, help='Path to config file', default="configs/attention_pool_extra3_full_train2/fold0.yaml")
    parser.add_argument('--model', type=str, help='Path to model checkpoint', default="C:\\Users\\mattb\\Documents\\ULTR-AI-Checkpoints\\fold0\\checkpoint_best.pth")
    parser.add_argument('--output-dir', type=str, default='JS_models', help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    
    args = parser.parse_args()

    # Update paths with fold number
    args.config = args.config.replace("fold0", f"fold{args.fold}")
    args.model = args.model.replace("fold0", f"fold{args.fold}")
    args.output_dir = os.path.join(args.output_dir, f"fold{args.fold}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

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

    
    # ----------------------------------
    # Model Conversion
    # ----------------------------------

    # Testing model
    if TEST_SIMPLE_MODEL:
        file_name = "model_simple.onnx"
        output_path_simple = os.path.join(args.output_dir, file_name)
        model_simple = load_model(args, config, simple=True)

        # Verify wrapper works
        with torch.no_grad():
            np.random.seed(0)
            x = torch.tensor(np.random.randn(1, 10).astype(np.float32))
            simple_output = model_simple(x)
            print("Wrapped model output:", simple_output)
        convert_to_onnx(model_simple, output_path_simple, simple=True)

        # Test ONNX model
        print(f"\n{'='*60}")
        print("Testing Simple ONNX Model")
        print(f"{'='*60}\n")

        onnx_logits = run_inference_onnx(output_path_simple)
        compare_logits(onnx_logits, simple_output)

    # Real model
    model = load_model(args, config, simple=False)
    for key, value in model(construct_dummy_input_dict()).items():
        if key == "site_metadata":
            print(f"{key}: (list of {len(value)} {type(value[0])}s, there are {len(value[0])} of them)")
            print(value[0][0].keys())

            


    raise NotImplemented("ONNX export for full model is currently disabled for safety - please enable when ready")
    wrapped_model = ONNXModelWrapper(model)
    wrapped_model.eval()

    # Prepare individual tensor inputs (no dictionaries)
    dummy_inputs = construct_dummy_input(to_numpy=False, as_dict=False)
    
    # Verify wrapper works
    with torch.no_grad():
        wrapped_output = wrapped_model(*dummy_inputs)
        print("Wrapped model output:", wrapped_output)

    # Use traditional TorchScript-based export (dynamo doesn't support data-dependent control flow)
    print("Using traditional TorchScript-based ONNX export...")

    file_name = "model_full.onnx"
    output_path = os.path.join(args.output_dir, file_name)
    
    convert_to_onnx(wrapped_model, output_path, simple=False)

    
    # Test ONNX model
    print(f"\n{'='*60}")
    print("Testing ONNX Model")
    print(f"{'='*60}\n")

    onnx_logits = run_inference_onnx(output_path)
    compare_logits(onnx_logits, wrapped_output)


    # Load model
    
    # dummy_input_shape_simple = (1, 10)
    # dummy_input = torch.randn(*dummy_input_shape_simple)

    # # Convert to ONNX
    # onnx_path = os.path.join(args.output_dir, "model.onnx")
    # torch.onnx.export(
    #     model_simple,
    #     dummy_input,
    #     onnx_path,
    #     dynamo=True,
    #     verbose=True)
    
    # dummy_input_shape = (1, 24, 32, 3, 224, 224)
    # dummy_input = torch.randn(*dummy_input_shape)

    # model.eval()
    # with torch.no_grad():
    #     output = model(construct_dummy_input())
    #     print(output)


    # # Convert to ONNX
    # onnx_path = os.path.join(args.output_dir, "model_full.onnx")
    # print(f"\n=== Exporting to ONNX: {onnx_path} ===")
    
    # Define dynamic axes for tensors that have variable dimensions
    # dynamic_axes = {
    #     "site_videos": {0: "batch", 1: "num_sites"},
    #     "site_images": {0: "batch", 1: "num_sites"},
    #     "site_findings": {0: "batch", 1: "num_sites"},
    #     "site_indices": {0: "batch", 1: "num_sites"},
    #     "site_counts": {0: "batch"},
    #     "site_masks": {0: "batch", 1: "num_sites"},
    #     "batch_padding_masks": {0: "batch", 1: "num_sites"},
    #     "real_data_masks": {0: "batch", 1: "num_sites"},
    #     "tb_labels": {0: "batch"},
    #     "pneumonia_labels": {0: "batch"},
    #     "covid_labels": {0: "batch"},
    #     "logits": {0: "batch"}  # Single output: [batch, num_tasks]
    # }
        
    # # Load ONNX model
    # print(f"Loading ONNX model from: {output_path}")
    # ort_session = ort.InferenceSession(output_path)
    
    # # Print model info
    # print(f"\nONNX Model Info:")
    # print(f"  Inputs:")
    # for input_meta in ort_session.get_inputs():
    #     print(f"    - {input_meta.name}: {input_meta.shape} ({input_meta.type})")
    # print(f"  Outputs:")
    # for output_meta in ort_session.get_outputs():
    #     print(f"    - {output_meta.name}: {output_meta.shape} ({output_meta.type})")
    
    # # Prepare ONNX inputs - dynamically based on what the model expects
    # expected_inputs = {inp.name for inp in ort_session.get_inputs()}
    
    # # Create input dictionary based on expected inputs
    # onnx_inputs = {}
    # all_inputs = construct_dummy_input(to_numpy=True, as_dict=True)
    
    # for input_name in expected_inputs:
    #     if input_name in all_inputs:
    #         # Use the prepared dummy inputs for the full model
    #         onnx_inputs[input_name] = all_inputs[input_name]
    #     else:
    #         # Handle other inputs (e.g., 'x' for simple model)
    #         # Infer shape from ONNX model metadata
    #         input_meta = next(inp for inp in ort_session.get_inputs() if inp.name == input_name)
    #         shape = [dim if isinstance(dim, int) else 1 for dim in input_meta.shape]
    #         onnx_inputs[input_name] = np.random.randn(*shape).astype(np.float32)
    
    # print(f"\nUsing {len(onnx_inputs)} inputs: {list(onnx_inputs.keys())}")
    
    # # Run ONNX inference
    # print("Running ONNX inference...")
    # onnx_outputs = ort_session.run(None, onnx_inputs)
    # onnx_logits = onnx_outputs[0]

    # Compare outputs
    
    
    # print(f"ONNX output shape: {onnx_logits.shape}")
    # print(f"ONNX logits:\n{onnx_logits}")
    
    # # Compare with wrapped PyTorch output
    # pytorch_logits = wrapped_output.numpy()
    
    # print(f"\nPyTorch output shape: {pytorch_logits.shape}")
    # print(f"PyTorch logits:\n{pytorch_logits}")
    
    # # Compute differences
    # diff = np.abs(onnx_logits - pytorch_logits)
    # max_diff = np.max(diff)
    # mean_diff = np.mean(diff)
    
    # print(f"\n{'='*60}")
    # print("Comparison Results")
    # print(f"{'='*60}\n")
    # print(f"Maximum absolute difference: {max_diff:.6f}")
    # print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # # Check if outputs match
    # tolerance = 1e-5
    # if np.allclose(onnx_logits, pytorch_logits, atol=tolerance):
    #     print(f"\n✓ PASSED: Outputs match within tolerance ({tolerance})")
    # else:
    #     print(f"\n⚠ WARNING: Outputs differ by more than tolerance ({tolerance})")
    #     print(f"Difference matrix:\n{diff}")
    
    # # Show probabilities
    # print(f"\n{'='*60}")
    # print("Probability Predictions")
    # print(f"{'='*60}\n")
    
    # onnx_probs = 1 / (1 + np.exp(-onnx_logits))
    # pytorch_probs = 1 / (1 + np.exp(-pytorch_logits))
    
    # task_names = ['TB', 'Pneumonia', 'COVID']
    # print("ONNX Probabilities:")
    # for i, task in enumerate(task_names):
    #     if i < onnx_probs.shape[1]:
    #         print(f"  {task}: {onnx_probs[0, i]:.4f}")
    
    # print("\nPyTorch Probabilities:")
    # for i, task in enumerate(task_names):
    #     if i < pytorch_probs.shape[1]:
    #         print(f"  {task}: {pytorch_probs[0, i]:.4f}")
    
    # print(f"\n{'='*60}\n")
        
    # except ImportError:
    #     print("\n⚠ WARNING: onnxruntime not installed. Skipping ONNX validation.")
    #     print("Install with: pip install onnxruntime")
    # except Exception as e:
    #     print(f"\n✗ ERROR during ONNX validation: {e}")
    #     import traceback
    #     traceback.print_exc()
    
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