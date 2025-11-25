import logging
from ultr_ai.network_architecture.components import MultiTaskModel, UniformFrameSelector

logger = logging.getLogger(__name__)

class NoRLMultiTaskModel(MultiTaskModel):
    """Ablation: No-RL selector with uniform temporal subsampling."""
    
    def __init__(self, config):
        config.selection_strategy = 'uniform'
        super().__init__(config)
        
        self.frame_selector = UniformFrameSelector(
            feature_dim=self.vision_dim,
            output_dim=self.hidden_dim,
            k_frames=3
        )
        logger.info("Using UniformFrameSelector for no-RL ablation")