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