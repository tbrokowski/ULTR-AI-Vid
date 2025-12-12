"""
TensorFlow implementation of the AttentionPool MultiTask Model.
Ablation variant that uses attention pooling instead of RL-based frame selection.
"""

import logging
from ultr_ai.tf.network_architecture.components.multi_task_model import MultiTaskModelTF
from ultr_ai.tf.network_architecture.components.selectors import AttentionPoolSelectorTF

logger = logging.getLogger(__name__)


class AttentionPoolMultiTaskModelTF(MultiTaskModelTF):
    """Ablation: Attention-pool baseline with learned attention (no RL). TF version."""
    
    def __init__(self, config, **kwargs):
        config.selection_strategy = 'attention_pool_tf'
        super().__init__(config, **kwargs)
        
        # Get temperature from config (default 0.5 if not specified)
        temperature = getattr(config, 'attention_temperature', 0.5)
        
        # Replace the frame selector with attention pool variant
        self.frame_selector = AttentionPoolSelectorTF(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            num_heads=8,
            temperature=temperature
        )
        logger.info(f"Using AttentionPoolSelectorTF for attention-pool-tf ablation (temperature={temperature})")
