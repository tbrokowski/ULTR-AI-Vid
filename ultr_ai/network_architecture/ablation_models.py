import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict
import logging

# PyTorch model imports
import torchvision.models as models
import torchvision.models.video as video_models
from transformers import CLIPVisionModel, VideoMAEModel, VideoMAEConfig
from safetensors import safe_open
import os
import gc
# Import base components from original model
from .CLIP_DRL_Aug11 import (
    PathologyModule, SiteIntegrationModule, DeepAttentionMIL, MultiTaskModel
)

logger = logging.getLogger(__name__)




# =============================================================================
# MODEL FACTORY AND REGISTRY
# =============================================================================

def create_ablation_model(model_type, config):
    """Factory function to create ablation models."""
    
    model_map = {
        'original': MultiTaskModel,
        'no_rl': NoRLMultiTaskModel,
        'no_rl_full_train': NoRLFullTrainMultiTaskModel,
        'mean_pool': MeanPoolMultiTaskModel,
        'attention_pool': AttentionPoolMultiTaskModel,
        'single_task': SingleTaskMultiTaskModel,
        '3d_cnn': ResNet3DMultiTaskModel,  # Using ResNet3D
        'cnn_lstm': CNNLSTMMultiTaskModel,
        'R2+1d': R2Plus1DMultiTaskModel,
        'Inception': InceptionMultiTaskModel,
        'InceptionRLBackbone': RLInceptionMultiTaskModel,
        'video_transformer': VideoTransformerMultiTaskModel,  # Using ViViT
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    logger.info(f"Creating {model_type} model using PyTorch backbones (ViViT for video_transformer)")
    return model_map[model_type](config)