import logging

from ultr_ai.network_architecture.components import MultiTaskModel
from ultr_ai.network_architecture.ablation_models import (
    NoRLMultiTaskModel, NoRLFullTrainMultiTaskModel,
    MeanPoolMultiTaskModel, AttentionPoolMultiTaskModel,
    SingleTaskMultiTaskModel
)
from ultr_ai.network_architecture.other_models import (
    ResNet3DMultiTaskModel, CNNLSTMMultiTaskModel,
    R2Plus1DMultiTaskModel, InceptionMultiTaskModel,
    VideoTransformerMultiTaskModel, RLInceptionMultiTaskModel
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