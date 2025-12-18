import numpy as np
import torch


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

def construct_dummy_videos():
    # Set seed
    torch.manual_seed(0)

    return construct_dummy_input_dict()["site_videos"]


def compare_logits(logits, other_logits):
    """Compare logits from two different models (e.g., ONNX and PyTorch)."""
    print(f"Output shape first logits: {logits.shape}")
    print(f"Output first logits:\n{logits}")
    
    # Bring to numpy
    logits = logits.numpy()
    other_logits = other_logits.numpy()
    
    print(f"\nPyTorch output shape: {other_logits.shape}")
    print(f"PyTorch logits:\n{other_logits}")
    
    # Compute differences
    diff = np.abs(logits - other_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}\n")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Check if outputs match
    tolerance = 1e-5
    if np.allclose(logits, other_logits, atol=tolerance):
        print(f"\nPASSED: Outputs match within tolerance ({tolerance})")
    else:
        print(f"\nWARNING: Outputs differ by more than tolerance ({tolerance})")
        print(f"Difference matrix:\n{diff}")
    
    # Show probabilities
    print(f"\n{'='*60}")
    print("Probability Predictions")
    print(f"{'='*60}\n")
    
    onnx_probs = 1 / (1 + np.exp(-logits))
    pytorch_probs = 1 / (1 + np.exp(-other_logits))
    
    task_names = ['TB', 'Pneumonia', 'COVID']
    print("ONNX Probabilities:")
    for i, task in enumerate(task_names):
        if i < onnx_probs.shape[1]:
            print(f"  {task}: {onnx_probs[0, i]:.4f}")
    
    print("\nPyTorch Probabilities:")
    for i, task in enumerate(task_names):
        if i < pytorch_probs.shape[1]:
            print(f"  {task}: {pytorch_probs[0, i]:.4f}")


# ============================================================================
# CLIP VISION MODEL WEIGHT MAPPING (PyTorch -> TensorFlow)
# ============================================================================

def build_clip_vision_weight_mapping(num_layers=12):
    """
    Build a mapping from PyTorch CLIP Vision Model weight names to TensorFlow weight names.
    
    The mapping also includes information about whether a transpose is needed.
    
    Returns:
        dict: {pytorch_name: (tf_name_pattern, needs_transpose, transpose_axes)}
        
    Weight name patterns:
        PyTorch: vision_model.embeddings.class_embedding
        TensorFlow: tfclip_vision_model/clip/vision_model/embeddings/class_embedding:0
        
    Transpose requirements:
        - Dense/Linear weights: transpose (PyTorch [out, in] -> TF [in, out])
        - Conv2D weights: transpose (PyTorch [out, in, H, W] -> TF [H, W, in, out])
        - LayerNorm, biases, embeddings: no transpose needed
    """
    
    mapping = {}
    
    # Embeddings
    mapping['vision_model.embeddings.class_embedding'] = (
        'clip/vision_model/embeddings/class_embedding:0', 
        False, 
        None
    )
    mapping['vision_model.embeddings.patch_embedding.weight'] = (
        'clip/vision_model/embeddings/patch_embedding/kernel:0',
        True,
        (2, 3, 1, 0)  # PyTorch [out, in, H, W] -> TF [H, W, in, out]
    )
    mapping['vision_model.embeddings.position_embedding.weight'] = (
        'clip/vision_model/embeddings/position_embedding/embeddings:0',
        False,
        None
    )
    
    # Pre-LayerNorm
    mapping['vision_model.pre_layrnorm.weight'] = (
        'clip/vision_model/pre_layrnorm/gamma:0',
        False,
        None
    )
    mapping['vision_model.pre_layrnorm.bias'] = (
        'clip/vision_model/pre_layrnorm/beta:0',
        False,
        None
    )
    
    # Encoder layers
    for i in range(num_layers):
        layer_prefix_pt = f'vision_model.encoder.layers.{i}'
        layer_prefix_tf = f'clip/vision_model/encoder/layers_._{i}'
        
        # Self-attention projections (q, k, v, out)
        for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            mapping[f'{layer_prefix_pt}.self_attn.{proj}.weight'] = (
                f'{layer_prefix_tf}/self_attn/{proj}/kernel:0',
                True,
                (1, 0)  # Transpose for Dense
            )
            mapping[f'{layer_prefix_pt}.self_attn.{proj}.bias'] = (
                f'{layer_prefix_tf}/self_attn/{proj}/bias:0',
                False,
                None
            )
        
        # Layer norms
        mapping[f'{layer_prefix_pt}.layer_norm1.weight'] = (
            f'{layer_prefix_tf}/layer_norm1/gamma:0',
            False,
            None
        )
        mapping[f'{layer_prefix_pt}.layer_norm1.bias'] = (
            f'{layer_prefix_tf}/layer_norm1/beta:0',
            False,
            None
        )
        mapping[f'{layer_prefix_pt}.layer_norm2.weight'] = (
            f'{layer_prefix_tf}/layer_norm2/gamma:0',
            False,
            None
        )
        mapping[f'{layer_prefix_pt}.layer_norm2.bias'] = (
            f'{layer_prefix_tf}/layer_norm2/beta:0',
            False,
            None
        )
        
        # MLP
        mapping[f'{layer_prefix_pt}.mlp.fc1.weight'] = (
            f'{layer_prefix_tf}/mlp/fc1/kernel:0',
            True,
            (1, 0)  # Transpose for Dense
        )
        mapping[f'{layer_prefix_pt}.mlp.fc1.bias'] = (
            f'{layer_prefix_tf}/mlp/fc1/bias:0',
            False,
            None
        )
        mapping[f'{layer_prefix_pt}.mlp.fc2.weight'] = (
            f'{layer_prefix_tf}/mlp/fc2/kernel:0',
            True,
            (1, 0)  # Transpose for Dense
        )
        mapping[f'{layer_prefix_pt}.mlp.fc2.bias'] = (
            f'{layer_prefix_tf}/mlp/fc2/bias:0',
            False,
            None
        )
    
    # Post-LayerNorm
    mapping['vision_model.post_layernorm.weight'] = (
        'clip/vision_model/post_layernorm/gamma:0',
        False,
        None
    )
    mapping['vision_model.post_layernorm.bias'] = (
        'clip/vision_model/post_layernorm/beta:0',
        False,
        None
    )
    
    return mapping


def transfer_clip_vision_weights_pt_to_tf(pytorch_state_dict, tf_model, num_layers=12):
    """
    Transfer weights from a PyTorch CLIP Vision Model state dict to a TensorFlow model.
    
    Args:
        pytorch_state_dict: dict from PyTorch model's state_dict() or safetensors file
        tf_model: TensorFlow TFCLIPVisionModel instance
        num_layers: Number of transformer layers (default 12 for ViT-B/32)
        
    Returns:
        tuple: (num_matched, num_total, unmatched_keys)
    """
    mapping = build_clip_vision_weight_mapping(num_layers)
    
    # Build a lookup dict for TF variables by their name pattern
    tf_var_dict = {}
    for var in tf_model.trainable_variables:
        # Remove the model prefix (e.g., 'tfclip_vision_model/') 
        # and keep everything after
        name = var.name
        # Find the 'clip/' part and use from there
        if 'clip/' in name:
            key = name[name.index('clip/'):]
            tf_var_dict[key] = var
    
    matched = 0
    unmatched = []
    
    for pt_key, (tf_key_pattern, needs_transpose, transpose_axes) in mapping.items():
        # Handle different safetensors format prefixes:
        # - Standard HuggingFace: vision_model.embeddings.class_embedding
        # - Custom saved models may have: vision_model.vision_model.embeddings.class_embedding
        possible_keys = [
            pt_key,  # e.g., vision_model.embeddings.class_embedding
            f"vision_model.{pt_key}",  # e.g., vision_model.vision_model.embeddings.class_embedding
        ]
        # Also try without the vision_model prefix if it starts with it
        if pt_key.startswith('vision_model.'):
            possible_keys.append(pt_key[len('vision_model.'):])  # e.g., embeddings.class_embedding
        
        # Find the weight in the pytorch state dict
        pt_weight = None
        matched_key = None
        for key in possible_keys:
            if key in pytorch_state_dict:
                pt_weight = pytorch_state_dict[key]
                matched_key = key
                break
        
        if pt_weight is None:
            unmatched.append(f"PT key not found: {pt_key} (tried: {possible_keys})")
            continue
            
        # Convert to numpy if it's a tensor
        # Handle BFloat16 by converting to float32 first
        if hasattr(pt_weight, 'dtype') and pt_weight.dtype == torch.bfloat16:
            pt_weight = pt_weight.float()  # Convert to float32
        
        if hasattr(pt_weight, 'cpu'):
            pt_weight = pt_weight.cpu().numpy()
        elif hasattr(pt_weight, 'numpy'):
            pt_weight = pt_weight.numpy()
        
        # Find the TF variable
        if tf_key_pattern not in tf_var_dict:
            unmatched.append(f"TF key not found: {tf_key_pattern}")
            continue
            
        tf_var = tf_var_dict[tf_key_pattern]
        
        # Apply transpose if needed
        if needs_transpose and transpose_axes is not None:
            pt_weight = np.transpose(pt_weight, transpose_axes)
        
        # Verify shapes match
        if pt_weight.shape != tuple(tf_var.shape):
            unmatched.append(
                f"Shape mismatch for {pt_key}: PT {pt_weight.shape} vs TF {tuple(tf_var.shape)}"
            )
            continue
        
        # Assign the weight
        tf_var.assign(pt_weight)
        matched += 1
    
    return matched, len(mapping), unmatched


def load_clip_weights_from_safetensors_to_tf(safetensors_path, tf_model, num_layers=12):
    """
    Load CLIP Vision weights from a safetensors file and transfer to TensorFlow model.
    
    Args:
        safetensors_path: Path to the safetensors file
        tf_model: TensorFlow TFCLIPVisionModel instance
        num_layers: Number of transformer layers
        
    Returns:
        tuple: (num_matched, num_total, unmatched_keys)
    """
    from safetensors import safe_open
    
    # Load weights from safetensors
    pytorch_state_dict = {}
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            pytorch_state_dict[key] = f.get_tensor(key)
    
    return transfer_clip_vision_weights_pt_to_tf(pytorch_state_dict, tf_model, num_layers)