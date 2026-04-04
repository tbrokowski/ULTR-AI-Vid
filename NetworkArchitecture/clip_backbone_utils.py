import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors import safe_open
from transformers import CLIPVisionModel


logger = logging.getLogger(__name__)


def create_clip_vision_encoder(
    clip_model_name: str = "openai/clip-vit-base-patch32",
    local_weights_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> CLIPVisionModel:
    """Build a CLIP vision encoder and optionally refresh it from local weights."""
    vision_encoder = CLIPVisionModel.from_pretrained(
        clip_model_name,
        dtype=dtype,
    )
    load_local_clip_weights(vision_encoder, local_weights_dir)
    return vision_encoder


def load_local_clip_weights(
    vision_encoder: CLIPVisionModel,
    local_weights_dir: Optional[str],
) -> int:
    """Load CLIP vision weights from a local safetensors file when available."""
    if not local_weights_dir:
        return 0

    local_weights_path = Path(local_weights_dir) / "model.safetensors"
    if not local_weights_path.exists():
        return 0

    logger.info("Loading CLIP vision weights from %s", local_weights_path)
    model_state_dict = vision_encoder.state_dict()
    matched_state_dict: Dict[str, torch.Tensor] = {}

    with safe_open(str(local_weights_path), framework="pt", device="cpu") as handle:
        for key, value in model_state_dict.items():
            safetensors_key = f"vision_model.{key}"
            if safetensors_key not in handle.keys():
                continue

            tensor = handle.get_tensor(safetensors_key)
            if tensor.shape == value.shape:
                matched_state_dict[key] = tensor

    if not matched_state_dict:
        logger.warning("No CLIP vision weights matched from %s", local_weights_path)
        return 0

    vision_encoder.load_state_dict(matched_state_dict, strict=False)
    logger.info(
        "Loaded %d/%d CLIP vision tensors from local safetensors",
        len(matched_state_dict),
        len(model_state_dict),
    )
    return len(matched_state_dict)


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    stripped: Dict[str, Any] = {}
    prefix_len = len(prefix)
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[prefix_len:]] = value
    return stripped


def extract_vision_encoder_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a CLIP vision encoder state dict from several checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "vision_encoder_state_dict" in checkpoint:
            state_dict = checkpoint["vision_encoder_state_dict"]
        elif "vision_state_dict" in checkpoint:
            state_dict = checkpoint["vision_state_dict"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a state dict")

    if any(key.startswith("module.vision_encoder.") for key in state_dict):
        return _strip_prefix(state_dict, "module.vision_encoder.")

    if any(key.startswith("vision_encoder.") for key in state_dict):
        return _strip_prefix(state_dict, "vision_encoder.")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = _strip_prefix(state_dict, "module.")

    return state_dict


def load_vision_encoder_weights(
    vision_encoder: CLIPVisionModel,
    checkpoint_path: str,
    map_location: str = "cpu",
) -> Tuple[int, int]:
    """Load a CLIP vision encoder checkpoint, filtering incompatible tensors."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    candidate_state_dict = extract_vision_encoder_state_dict(checkpoint)
    model_state_dict = vision_encoder.state_dict()

    filtered_state_dict = {
        key: value
        for key, value in candidate_state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape == value.shape
    }

    if not filtered_state_dict:
        raise ValueError(
            f"No compatible CLIP vision tensors found in checkpoint: {checkpoint_path}"
        )

    model_state_dict.update(filtered_state_dict)
    vision_encoder.load_state_dict(model_state_dict, strict=False)
    logger.info(
        "Loaded %d/%d compatible CLIP vision tensors from %s",
        len(filtered_state_dict),
        len(model_state_dict),
        checkpoint_path,
    )
    return len(filtered_state_dict), len(model_state_dict)


def configure_clip_trainability(
    vision_encoder: CLIPVisionModel,
    freeze_backbone: bool = False,
    unfreeze_last_n_layers: int = 1,
    train_visual_projection: bool = True,
) -> None:
    """
    Freeze the CLIP backbone by default, then selectively unfreeze the requested tail.

    `unfreeze_last_n_layers >= total_layers` is treated as full encoder fine-tuning.
    """
    for param in vision_encoder.parameters():
        param.requires_grad = False

    if hasattr(vision_encoder, "visual_projection") and train_visual_projection:
        for param in vision_encoder.visual_projection.parameters():
            param.requires_grad = True

    if freeze_backbone:
        return

    layers = getattr(getattr(vision_encoder, "vision_model", None), "encoder", None)
    encoder_layers = getattr(layers, "layers", None)
    if encoder_layers is None:
        return

    total_layers = len(encoder_layers)
    requested_layers = max(int(unfreeze_last_n_layers), 0)

    if requested_layers <= 0:
        return

    if requested_layers >= total_layers:
        for param in vision_encoder.parameters():
            param.requires_grad = True
        return

    for layer in encoder_layers[-requested_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    post_layernorm = getattr(getattr(vision_encoder, "vision_model", None), "post_layernorm", None)
    if post_layernorm is not None:
        for param in post_layernorm.parameters():
            param.requires_grad = True


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
