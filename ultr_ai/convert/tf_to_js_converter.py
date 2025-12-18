"""File that loads a TF model and saves the weights"""

import sys
import os
import argparse

# Add paths
PROJECT_ROOT = "."
SRC_PATH = "."
NETWORK_PATH = "."

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, NETWORK_PATH)

import tensorflow as tf

from ultr_ai.config import load_config
from ultr_ai.tf.network_architecture import create_ablation_model_tf
from ultr_ai.convert.utils import construct_dummy_input_dict


def main():
    # ----------------------------------
    # Argument parsing
    # ----------------------------------
    parser = argparse.ArgumentParser(description='Convert model formats.')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='attention_pool')
    parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory', default="C:\\Users\\mattb\\Documents\\ULTR-AI-Checkpoints")
    parser.add_argument('--config', type=str, help='Path to config file', default="configs/attention_pool_extra3_full_train2/fold0.yaml")
    parser.add_argument('--output-dir', type=str, default='ultr_ai\\convert\\JS_models', help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    
    args = parser.parse_args()

    # Update paths with fold number
    args.config = args.config.replace("fold0", f"fold{args.fold}")
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


    model = create_ablation_model_tf(args.model_type, config)
    print(f"Model initialized: {args.model_type}")

    # Build the model by running a forward pass with dummy input
    print("Building model with dummy input...")
    import torch
    dummy_input = construct_dummy_input_dict()
    tf_input = {
        k: tf.constant(v.numpy()) if isinstance(v, torch.Tensor) else v
        for k, v in dummy_input.items()
    }
    output = model(tf_input, training=False)
    print("Model built successfully.")
    
    # Explicitly build all nested submodels/layers
    print("Ensuring all submodels are built...")
    patient_features = output.get('patient_features')
    if patient_features is not None:
        # Build task classifiers with the correct input shape
        for task_key, classifier in model.task_classifiers.items():
            _ = classifier(patient_features)
            print(f"  Built classifier: {task_key}")
        # Build tb_classifier for backward compatibility
        _ = model.tb_classifier(patient_features)
        print("  Built tb_classifier")
    
    print("All submodels built successfully.")

    # Save model weights as HDF5
    weights_path = os.path.join(args.output_dir, f"{args.model_type}_weights.h5")
    model.save_weights(weights_path)
    print(f"Model weights saved to: {weights_path}")

    # Save weights in a JS-compatible format (no tensorflowjs dependency needed)
    import json
    import numpy as np
    
    tfjs_output_dir = os.path.join(args.output_dir, f"{args.model_type}_tfjs_model")
    os.makedirs(tfjs_output_dir, exist_ok=True)
    
    def collect_weights_recursive(layer, prefix=""):
        """Recursively collect weights from all layers including nested ones."""
        weights_dict = {}
        layer_name = prefix + layer.name if prefix else layer.name
        
        # Get weights from this layer
        layer_weights = layer.get_weights()
        if layer_weights:
            weights_dict[layer_name] = {
                'shapes': [list(w.shape) for w in layer_weights],
                'dtypes': [str(w.dtype) for w in layer_weights]
            }
        
        # Recurse into sublayers
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                weights_dict.update(collect_weights_recursive(sublayer, prefix=layer_name + "/"))
        
        # Handle dict-based sublayers (like task_classifiers)
        if hasattr(layer, 'task_classifiers'):
            for task_key, classifier in layer.task_classifiers.items():
                weights_dict.update(collect_weights_recursive(classifier, prefix=layer_name + "/task_classifiers/"))
        
        return weights_dict
    
    # Collect weight metadata
    weights_metadata = collect_weights_recursive(model)
    
    # Save weights as binary files + manifest (similar to TFJS format)
    manifest = {
        'format': 'weights_manifest',
        'model_type': args.model_type,
        'weights_files': [],
        'weights_spec': {}
    }
    
    weight_data_chunks = []
    current_offset = 0
    
    for var in model.trainable_variables + model.non_trainable_variables:
        var_name = var.name
        var_data = var.numpy()
        
        # Save binary data
        weight_data_chunks.append(var_data.astype(np.float32).tobytes())
        
        manifest['weights_spec'][var_name] = {
            'shape': list(var_data.shape),
            'dtype': 'float32',
            'offset': current_offset,
            'size': var_data.size * 4  # float32 = 4 bytes
        }
        current_offset += var_data.size * 4
    
    # Write binary weights file
    weights_bin_path = os.path.join(tfjs_output_dir, 'weights.bin')
    with open(weights_bin_path, 'wb') as f:
        for chunk in weight_data_chunks:
            f.write(chunk)
    
    manifest['weights_files'].append('weights.bin')
    manifest['total_bytes'] = current_offset
    
    # Write manifest
    manifest_path = os.path.join(tfjs_output_dir, 'model.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"TFJS-compatible weights saved to: {tfjs_output_dir}")
    print(f"  - model.json (manifest)")
    print(f"  - weights.bin ({current_offset / 1024 / 1024:.2f} MB)")

if __name__ == '__main__':
    main()