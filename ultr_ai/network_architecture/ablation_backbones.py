# =============================================================================
# ABLATION MODELS USING PYTORCH BACKBONES
# =============================================================================



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


class MeanPoolMultiTaskModel(MultiTaskModel):
    """Ablation: Mean-pool baseline - average all frame features per site."""
    
    def __init__(self, config):
        config.selection_strategy = 'mean_pool'
        super().__init__(config)
        
        self.frame_selector = MeanPoolSelector(
            feature_dim=self.vision_dim,
            output_dim=self.hidden_dim
        )
        logger.info("Using MeanPoolSelector for mean-pool ablation")
    
    def process_site(self, video, site_idx, mask=None, batch_idx=None, site_pos=None):
        """Process site with mean pooling - average all valid frames."""
        clip_features = self.extract_clip_features(video)
        
        action_logits, state_values, enhanced_features = self.frame_selector(
            clip_features, mask, batch_idx, site_pos
        )
        
        # Mean pool over all valid frames
        if mask is not None:
            valid_mask = mask[0]
            if valid_mask.any():
                valid_features = enhanced_features[0, valid_mask]
                pooled_features = valid_features.mean(dim=0, keepdim=True)
            else:
                pooled_features = enhanced_features[0, :1]
        else:
            pooled_features = enhanced_features[0].mean(dim=0, keepdim=True)
        
        selected_features = pooled_features.repeat(3, 1).unsqueeze(0)
        selected_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
        
        # Process pathologies
        pathology_scores = None
        if self.use_pathology_loss and self.pathology_modules is not None:
            pathology_scores = []
            for module in self.pathology_modules:
                score, _, _ = module(selected_features, selected_mask)
                pathology_scores.append(score)
            pathology_scores = torch.cat(pathology_scores, dim=1)
        
        return {
            'selected_features': selected_features,
            'selected_indices': torch.arange(3, device=video.device).unsqueeze(0),
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos
        }


class AttentionPoolMultiTaskModel(MultiTaskModel):
    """Ablation: Attention-pool baseline with learned attention (no RL)."""
    
    def __init__(self, config):
        config.selection_strategy = 'attention_pool'
        super().__init__(config)
        
        # Get temperature from config (default 0.5 if not specified)
        temperature = getattr(config, 'attention_temperature', 0.5)
        
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            num_heads=8,
            temperature=temperature
        )
        logger.info(f"Using AttentionPoolSelector for attention-pool ablation (temperature={temperature})")


class SingleTaskMultiTaskModel(MultiTaskModel):
    """Ablation: RL Selector but no pathology detection (single-task)."""
    
    def __init__(self, config):
        config.use_pathology_loss = False
        super().__init__(config)
        logger.info("Using RL selector without pathology detection (single-task)")
        
        
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


