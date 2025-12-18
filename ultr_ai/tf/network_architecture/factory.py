import logging

from ultr_ai.tf.network_architecture.components import MultiTaskModelTF
from ultr_ai.tf.network_architecture.ablation_models import (
    AttentionPoolMultiTaskModelTF
)

logger = logging.getLogger(__name__)




# =============================================================================
# MODEL FACTORY AND REGISTRY
# =============================================================================

def create_ablation_model_tf(model_type, config):
    """Factory function to create ablation models. TF version."""
    
    model_map = {
        'original': MultiTaskModelTF,
        # 'no_rl': NoRLMultiTaskModelTF,
        # 'no_rl_full_train': NoRLFullTrainMultiTaskModelTF,
        # 'mean_pool': MeanPoolMultiTaskModelTF,
        'attention_pool': AttentionPoolMultiTaskModelTF,
        # 'single_task': SingleTaskMultiTaskModelTF,
        # '3d_cnn': ResNet3DMultiTaskModelTF,  # Using ResNet3D
        # 'cnn_lstm': CNNLSTMMultiTaskModelTF,
        # 'R2+1d': R2Plus1DMultiTaskModelTF,
        # 'Inception': InceptionMultiTaskModelTF,
        # 'InceptionRLBackbone': RLInceptionMultiTaskModelTF,
        # 'video_transformer': VideoTransformerMultiTaskModelTF,  # Using ViViT
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    logger.info(f"Creating {model_type} model using PyTorch backbones (ViViT for video_transformer)")
    return model_map[model_type](config)