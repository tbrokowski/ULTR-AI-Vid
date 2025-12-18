import logging
from ultr_ai.network_architecture.components import MultiTaskModel

logger = logging.getLogger(__name__)

class SingleTaskMultiTaskModel(MultiTaskModel):
    """Ablation: RL Selector but no pathology detection (single-task)."""
    
    def __init__(self, config):
        config.use_pathology_loss = False
        super().__init__(config)
        logger.info("Using RL selector without pathology detection (single-task)")
        