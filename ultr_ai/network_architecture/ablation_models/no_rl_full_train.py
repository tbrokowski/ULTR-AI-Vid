import logging
from ultr_ai.network_architecture.components import MultiTaskModel, UniformFrameSelector

logger = logging.getLogger(__name__)

class NoRLFullTrainMultiTaskModel(MultiTaskModel):
    """Ablation: No-RL selector with full CLIP training (all parameters unfrozen)."""
    
    def __init__(self, config):
        config.selection_strategy = 'uniform'
        config.freeze_clip = False  # Unfreeze CLIP for full training
        super().__init__(config)
        
        self.frame_selector = UniformFrameSelector(
            feature_dim=self.vision_dim,
            output_dim=self.hidden_dim,
            k_frames=8
        )
        
        # Explicitly unfreeze all CLIP parameters
        if hasattr(self, 'clip_model'):
            for param in self.clip_model.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all CLIP parameters for full training")
        
        logger.info("Using UniformFrameSelector with full CLIP training (no-RL-full-train ablation)")
