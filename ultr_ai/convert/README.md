# ultr_ai/convert Module

This module provides utilities for converting, exporting, and preparing deep learning models (primarily PyTorch) for deployment, including ONNX and TensorFlow.js formats. It is designed to help users deploy trained models for inference in various environments, such as web applications using JS models.

## File Overview

- **onnx_converter.py**: Main script for converting PyTorch models to ONNX and TensorFlow.js formats. Handles model loading, wrapping, dummy input construction, ONNX export, and inference testing. Contains detailed logic for preparing model inputs/outputs for deployment.
- **utils.py**: Helper functions for comparing model outputs, post-processing, and other conversion-related utilities.
- **(Other files)**: May include additional scripts for specific conversion tasks, legacy support, or format-specific utilities. Please refer to file-level docstrings for details.

## Model Input and Output Structure

The primary model supported by this module (e.g., the `attention_pool` model) expects a complex, structured input and produces a rich output dictionary. Understanding this structure is crucial for successful deployment and integration.

### Model Input (as Python Dictionary)

| Key                  | Shape / Type                | Description                                      |
|----------------------|-----------------------------|--------------------------------------------------|
| `patient_ids`        | list of str                 | Patient identifiers                              |
| `tb_labels`          | `[B]` (torch.Tensor)        | Tuberculosis labels (per patient)                |
| `pneumonia_labels`   | `[B]` (torch.Tensor)        | Pneumonia labels (per patient)                   |
| `covid_labels`       | `[B]` (torch.Tensor)        | COVID-19 labels (per patient)                    |
| `site_indices`       | `[B, S]` (torch.Tensor)     | Indices for each site                            |
| `site_counts`        | `[B]` (torch.Tensor)        | Number of sites per patient                      |
| `site_videos`        | `[B, S, F, 3, 224, 224]`    | Video data: batch, sites, frames, channels, H, W  |
| `site_images`        | `[B, S, 3, 224, 224]`       | Image data: batch, sites, channels, H, W         |
| `site_findings`      | `[B, S, 4]` (torch.Tensor)  | Site-level findings/features                     |
| `site_masks`         | `[B, S]` (torch.BoolTensor) | Mask for valid sites                             |
| `batch_padding_masks`| `[B, S]` (torch.BoolTensor) | Mask for batch padding                           |
| `real_data_masks`    | `[B, S]` (torch.BoolTensor) | Mask for real (non-padded) data                  |
| `_mask_type`         | str                         | Masking strategy (e.g., 'batch_padding')         |

- `B` = batch size (usually 1 for inference)
- `S` = number of sites (e.g., 24)
- `F` = number of video frames (e.g., 32)

### Model Output (as Python Dictionary)

| Key                      | Type / Shape                | Description                                      |
|--------------------------|-----------------------------|--------------------------------------------------|
| `task_logits`            | dict                        | Task-specific logits (e.g., TB, Pneumonia, COVID) |
| `patient_pathology_scores`| `[B, 4]` (torch.Tensor)    | Pathology scores per patient                      |
| `patient_features`       | `[B, 512]` (torch.Tensor)   | Patient-level feature embeddings                  |
| `pathology_scores`       | `[B, S, 4]` (torch.Tensor)  | Pathology scores per site                         |
| `mil_attention`          | `[B, S]` (torch.Tensor)     | MIL attention weights per site                    |
| `site_features`          | `[B, S, 512]` (torch.Tensor)| Site-level feature embeddings                     |
| `site_metadata`          | list of lists of dicts      | Metadata for each site                            |
| `site_outputs`           | list of dicts               | Site-level output details                         |
| `tb_logits`              | `[B]` (torch.Tensor)        | TB logits (legacy, for compatibility)             |

#### Example Output (abridged):
```
{
	'task_logits': {'TB Label': tensor([0.1041]), ...},
	'patient_pathology_scores': tensor([[-0.1297, -0.1474, -0.2922, 0.2340]]),
	'mil_attention': tensor([[0.0971, 0.0000, ...]]),
	...
}
```

### ONNX/JS Model Export

- The ONNX export wraps the model to accept only tensor inputs (no dictionaries) and outputs a single tensor of concatenated logits for deployment.
- Input order for ONNX/JS models:
	1. `site_videos`
	2. `site_images`
	3. `site_findings`
	4. `site_indices`
	5. `site_counts`
	6. `site_masks`
	7. `batch_padding_masks`
	8. `real_data_masks`
	9. `tb_labels`
	10. `pneumonia_labels`
	11. `covid_labels`
- Output: Single tensor `[B, num_tasks]` (e.g., `[1, 3]` for TB, Pneumonia, COVID)

### Where to Find JS Models

- Exported ONNX and TensorFlow.js models are saved in the `JS_models/` directory (with subfolders per fold, e.g., `JS_models/fold0/`).
- These models are ready for deployment in web applications or other JS-based inference environments.

