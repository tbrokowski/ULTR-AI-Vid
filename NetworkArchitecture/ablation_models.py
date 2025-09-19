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
# FRAME SELECTORS
# =============================================================================

class UniformFrameSelector(nn.Module):
    """No-RL baseline: Uniform temporal subsampling of k frames per site."""
    
    def __init__(self, feature_dim=768, output_dim=512, k_frames=3, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k_frames = k_frames
        
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Compatibility attributes
        self.saved_actions = []
        self.temperature = 1.0
    
    def get_temperature(self):
        return self.temperature
    
    def clear_history(self):
        self.saved_actions = []
    
    def reset_rewards(self):
        pass
    
    def reset_temperature(self):
        pass
    
    def update_temperature(self, decay=None):
        return self.temperature
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        encoded_features = self.feature_projection(features)
        action_logits = torch.ones(batch_size, seq_len, device=device)
        
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, -1e9)
        
        state_values = torch.zeros(batch_size, 1, device=device, requires_grad=True)
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        batch_size, seq_len = logits.shape
        device = logits.device
        
        actions = []
        for b in range(batch_size):
            valid_indices = torch.where(logits[b] > -1e8)[0]
            
            if len(valid_indices) == 0:
                action = torch.tensor(0, device=device)
            elif len(valid_indices) <= self.k_frames:
                selected = valid_indices.repeat((self.k_frames + len(valid_indices) - 1) // len(valid_indices))
                action = selected[0]
            else:
                step = len(valid_indices) // self.k_frames
                uniform_indices = torch.arange(0, len(valid_indices), step, device=device)[:self.k_frames]
                selected_indices = valid_indices[uniform_indices]
                action = selected_indices[0]
            
            actions.append(action)
        
        actions = torch.stack(actions)
        log_probs = torch.zeros_like(actions, dtype=torch.float)
        return actions, log_probs


class MeanPoolSelector(nn.Module):
    """Mean-pool baseline: Average all frame features per site."""
    
    def __init__(self, feature_dim=768, output_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Compatibility attributes
        self.saved_actions = []
        self.temperature = 1.0
    
    def get_temperature(self):
        return self.temperature
    
    def clear_history(self):
        self.saved_actions = []
    
    def reset_rewards(self):
        pass
    
    def reset_temperature(self):
        pass
    
    def update_temperature(self, decay=None):
        return self.temperature
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        encoded_features = self.feature_projection(features)
        action_logits = torch.ones(batch_size, seq_len, device=device)
        
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, -1e9)
        
        state_values = torch.zeros(batch_size, 1, device=device, requires_grad=True)
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        batch_size = logits.shape[0]
        device = logits.device
        
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        return actions, log_probs


class AttentionPoolSelector(nn.Module):
    """Attention-pool baseline: Learned attention over frames without RL."""
    
    def __init__(self, feature_dim=768, hidden_dim=512, output_dim=512, num_heads=8, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.attention_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Compatibility attributes
        self.saved_actions = []
        self.temperature = 1.0
    
    def get_temperature(self):
        return self.temperature
    
    def clear_history(self):
        self.saved_actions = []
    
    def reset_rewards(self):
        pass
    
    def reset_temperature(self):
        pass
    
    def update_temperature(self, decay=None):
        return self.temperature
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        encoded = self.feature_encoder(features)
        
        key_padding_mask = ~mask if mask is not None else None
        attended, _ = self.attention(
            encoded, encoded, encoded,
            key_padding_mask=key_padding_mask
        )
        
        attention_scores = self.attention_scorer(attended).squeeze(-1)
        
        if mask is not None:
            if attention_scores.dtype == torch.float16:
                mask_value = torch.finfo(torch.float16).min
            else:
                mask_value = -1e4  # Safe for both fp32 and fp16
            
            attention_scores = attention_scores.masked_fill(~mask, mask_value)
        
        
        output_features = self.output_projection(attended)
        state_values = torch.zeros(batch_size, 1, device=device, requires_grad=True)
        
        return attention_scores, state_values, output_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        batch_size = logits.shape[0]
        device = logits.device
        
        k = 3
        if logits.shape[1] >= k:
            _, top_indices = torch.topk(logits, k=k, dim=1)
            actions = top_indices[:, 0]
        else:
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        return actions, action_log_probs


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
        
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            num_heads=8
        )
        logger.info("Using AttentionPoolSelector for attention-pool ablation")


class SingleTaskMultiTaskModel(MultiTaskModel):
    """Ablation: RL Selector but no pathology detection (single-task)."""
    
    def __init__(self, config):
        config.use_pathology_loss = False
        super().__init__(config)
        logger.info("Using RL selector without pathology detection (single-task)")

class ResNet3DMultiTaskModel(nn.Module):
    """Memory-efficient ResNet3D backbone from torchvision."""
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 15)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = '3d_resnet'
        
        # Memory optimization settings
        self.backbone_frozen = getattr(config, 'backbone_frozen', False)  # NEW: Keep backbone frozen by default
        self.max_sites_per_forward = getattr(config, 'max_sites_per_forward', 4)  # NEW: Process sites in chunks
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)  # NEW
        
        logger.info("Using Memory-Efficient ResNet3D backbone")
        
        # Use smaller ResNet3D model for better memory efficiency
        try:
            # Use R(2+1)D ResNet18 - more memory efficient than 3D ResNet
            self.backbone = video_models.r2plus1d_18(pretrained=True)
            logger.info("Loaded R(2+1)D ResNet18")
        except:
            try:
                # Fallback to MC3 ResNet18
                self.backbone = video_models.mc3_18(pretrained=True) 
                logger.info("Loaded MC3 ResNet18")
            except:
                # Final fallback
                self.backbone = video_models.r3d_18(pretrained=True)
                logger.info("Loaded R3D ResNet18")
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Enable gradient checkpointing if available and requested
        if self.use_gradient_checkpointing and hasattr(self.backbone, 'gradient_checkpointing'):
            self.backbone.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing on backbone")
        
        # Freeze backbone by default for memory efficiency
        if self.backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen for memory efficiency")
        
        # Get feature dimension
        backbone_dim = 512
        
        # Smaller projection layer to reduce memory
        projection_dim = min(self.hidden_dim, 256)  # NEW: Cap projection size
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Additional projection if needed
        if projection_dim != self.hidden_dim:
            self.feature_upscale = nn.Linear(projection_dim, self.hidden_dim)
        else:
            self.feature_upscale = nn.Identity()
        
        # Pathology modules (if enabled) - use smaller hidden dim
        if self.use_pathology_loss:
            pathology_hidden = min(self.hidden_dim // 2, 128)  # NEW: Smaller pathology modules
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=pathology_hidden,
                    dropout=self.dropout_rate,
                    name=f'pathology_{i}'
                ) for i in range(self.num_pathologies)
            ])
        else:
            self.pathology_modules = None
        
        # Site integration - smaller embedding
        if self.use_pathology_loss:
            self.site_integration = SiteIntegrationModule(
                feature_dim=self.hidden_dim,
                site_embed_dim=32,  # NEW: Reduced from 64
                hidden_dim=self.hidden_dim,
                num_sites=self.num_sites,
                num_pathologies=self.num_pathologies,
                dropout=self.dropout_rate
            )
        else:
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        # Patient-level MIL - smaller hidden dim
        mil_hidden = min(self.hidden_dim // 2, 128)  # NEW: Smaller MIL
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=mil_hidden,
            dropout=self.dropout_rate,
            num_heads=4  # NEW: Reduced from 8
        )
        
        # Task classifiers - smaller hidden layers
        classifier_hidden = min(self.hidden_dim // 2, 128)  # NEW: Smaller classifiers
        self.task_classifiers = nn.ModuleDict()
        for task_name in self.active_tasks:
            task_key = task_name.replace(' ', '_').replace('Label', 'label')
            self.task_classifiers[task_key] = nn.Sequential(
                nn.Linear(self.hidden_dim, classifier_hidden),
                nn.LayerNorm(classifier_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(classifier_hidden, self.num_classes)
            )
        
        # TB classifier for backward compatibility
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        
        # Dummy frame selector for compatibility
        self.frame_selector = self._create_dummy_selector()
        
        # Report memory optimizations
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def _create_dummy_selector(self):
        """Create dummy frame selector for compatibility."""
        class DummySelector:
            def __init__(self):
                self.saved_actions = []
                self.temperature = 1.0
            
            def get_temperature(self):
                return self.temperature
            
            def clear_history(self):
                self.saved_actions = []
            
            def reset_rewards(self):
                pass
            
            def reset_temperature(self):
                return self.temperature
            
            def update_temperature(self, **kwargs):
                return self.temperature
        
        return DummySelector()
    
    def _extract_video_features(self, video, use_amp=True):
        """Memory-efficient video feature extraction."""
        # Reshape for ResNet3D: [1, C, T, H, W]
        video_input = video.permute(1, 0, 2, 3).unsqueeze(0)
        
        # Use appropriate context for feature extraction
        context = torch.amp.autocast('cuda') if use_amp else torch.enable_grad()
        
        if self.backbone_frozen:
            # Extract features without gradients to save memory
            with torch.no_grad(), context:
                video_features = self.backbone(video_input)
        else:
            # Extract with gradients but use checkpointing
            with context:
                if self.use_gradient_checkpointing and hasattr(self.backbone, 'gradient_checkpointing'):
                    video_features = torch.utils.checkpoint.checkpoint(self.backbone, video_input)
                else:
                    video_features = self.backbone(video_input)
        
        return video_features
    
    def _process_sites_chunked(self, site_videos, site_masks, batch_idx, use_amp=True):
        """Process sites in memory-efficient chunks."""
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        
        # Process each sample in batch
        for b in range(batch_size):
            valid_sites = site_masks[b].sum().item()
            
            if valid_sites == 0:
                # Handle empty case
                site_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                pathology_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                all_site_features.append(site_features)
                all_pathology_scores.append(pathology_scores)
                continue
            
            sample_features = []
            sample_pathology_scores = []
            
            # Process sites in chunks to avoid OOM
            for start_idx in range(0, valid_sites, self.max_sites_per_forward):
                end_idx = min(start_idx + self.max_sites_per_forward, valid_sites)
                chunk_size = end_idx - start_idx
                
                chunk_features = []
                chunk_pathology_scores = []
                
                # Process each site in the chunk
                for n in range(start_idx, end_idx):
                    video = site_videos[b, n]  # [T, C, H, W]
                    
                    try:
                        # Extract features with memory-efficient method
                        video_features = self._extract_video_features(video, use_amp)
                        
                        # Project features
                        projected_features = self.feature_projection(video_features)
                        final_features = self.feature_upscale(projected_features)
                        
                        chunk_features.append(final_features)
                        
                        # Process pathology if enabled
                        if self.use_pathology_loss and self.pathology_modules is not None:
                            # Create dummy representation for pathology modules
                            dummy_frames = final_features.unsqueeze(1).repeat(1, 3, 1)
                            dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                            
                            pathology_scores = []
                            for module in self.pathology_modules:
                                score, _, _ = module(dummy_frames, dummy_mask)
                                pathology_scores.append(score)
                            
                            chunk_pathology_scores.append(torch.cat(pathology_scores, dim=1))
                        
                        # Clear intermediate variables
                        del video_features, projected_features, final_features
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logger.warning(f"OOM processing site {n} in batch {batch_idx}, using zero features")
                            # Use zero features as fallback
                            zero_features = torch.zeros(1, self.hidden_dim, device=site_videos.device)
                            chunk_features.append(zero_features)
                            
                            if self.use_pathology_loss:
                                zero_pathology = torch.zeros(1, self.num_pathologies, device=site_videos.device)
                                chunk_pathology_scores.append(zero_pathology)
                            
                            # Emergency cleanup
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise e
                
                # Collect chunk results
                if chunk_features:
                    sample_features.extend(chunk_features)
                if chunk_pathology_scores:
                    sample_pathology_scores.extend(chunk_pathology_scores)
            
            # Pad and collect sample results
            if sample_features:
                sample_features_tensor = torch.cat(sample_features, dim=0)
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = sample_features_tensor
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss and sample_pathology_scores:
                    sample_pathology_tensor = torch.cat(sample_pathology_scores, dim=0)
                    padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                    padded_scores[:valid_sites] = sample_pathology_tensor
                    all_pathology_scores.append(padded_scores)
                else:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
            else:
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
        
        return all_site_features, all_pathology_scores
    
    def forward(self, inputs):
        """Memory-efficient forward pass."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        # Check for overly large batches
        total_sites = site_masks.sum().item()
        if total_sites > 50:  # Configurable threshold
            logger.warning(f"Large batch detected ({total_sites} sites), consider reducing batch size")
        
        # Use chunked processing for memory efficiency
        try:
            use_amp = hasattr(self, 'use_amp') and getattr(self, 'use_amp', True)
            all_site_features, all_pathology_scores = self._process_sites_chunked(
                site_videos, site_masks, batch_idx=0, use_amp=use_amp
            )
            
            # Stack features
            site_features = torch.stack(all_site_features)
            pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.error("OOM during chunked processing, falling back to minimal processing")
                # Emergency fallback: use zero features
                site_features = torch.zeros(batch_size, max_sites, self.hidden_dim, device=site_videos.device)
                pathology_scores = torch.zeros(batch_size, max_sites, self.num_pathologies, device=site_videos.device) if self.use_pathology_loss else None
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Site integration
        try:
            if self.use_pathology_loss:
                integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
            else:
                integrated_features = self.site_integration(site_features)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning("OOM in site integration, using passthrough")
                integrated_features = site_features
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Patient-level MIL
        try:
            patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning("OOM in patient MIL, using mean pooling")
                # Fallback to simple mean pooling
                valid_features = integrated_features * site_masks.unsqueeze(-1).float()
                patient_features = valid_features.sum(dim=1) / site_masks.sum(dim=1, keepdim=True).float().clamp(min=1)
                mil_attention = torch.ones_like(site_masks).float() / site_masks.sum(dim=1, keepdim=True).float().clamp(min=1)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Classification
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                try:
                    tb_logits = self.tb_classifier(patient_features)
                    if self.num_classes == 1:
                        tb_logits = tb_logits.squeeze(-1)
                    task_logits['TB Label'] = tb_logits
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning("OOM in TB classifier, using zeros")
                        task_logits['TB Label'] = torch.zeros(batch_size, device=site_videos.device)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
        
        return {
            'task_logits': task_logits,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_rl_data': []
        }
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Reuse loss computation from MultiTaskModel."""
        return MultiTaskModel.compute_losses(self, outputs, targets, pos_weights)
    
    def unfreeze_backbone_gradually(self, epoch, total_epochs):
        """Gradually unfreeze backbone layers as training progresses."""
        if not self.backbone_frozen:
            return
            
        # Unfreeze strategy: start from top layers and work backwards
        unfreeze_schedule = {
            total_epochs // 4: ['layer4', 'avgpool', 'fc'],           # 25% through
            total_epochs // 2: ['layer3', 'layer4', 'avgpool', 'fc'], # 50% through  
            3 * total_epochs // 4: ['layer2', 'layer3', 'layer4', 'avgpool', 'fc'], # 75% through
        }
        
        for target_epoch, layers_to_unfreeze in unfreeze_schedule.items():
            if epoch == target_epoch:
                logger.info(f"Unfreezing backbone layers at epoch {epoch}: {layers_to_unfreeze}")
                for name, param in self.backbone.named_parameters():
                    if any(layer in name for layer in layers_to_unfreeze):
                        param.requires_grad = True
                        logger.debug(f"Unfroze: {name}")
                break
    
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return {
                'current_gb': current_memory,
                'max_gb': max_memory, 
                'total_gb': total_memory,
                'usage_percent': (current_memory / total_memory) * 100
            }
        return {'current_gb': 0, 'max_gb': 0, 'total_gb': 0, 'usage_percent': 0}




        
class CNNLSTMMultiTaskModel(nn.Module):
    """Ablation: CNN-LSTM using PyTorch ResNet + LSTM."""
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 15)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = 'cnn_lstm'
        
        logger.info("Using CNN-LSTM with PyTorch ResNet + LSTM")
        
        # CNN backbone (ResNet18)
        self.cnn_backbone = models.resnet18(pretrained=True)
        # Remove final layers
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])
        
        # Adaptive pooling to get fixed-size features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # CNN feature dimension (ResNet18 conv features = 512)
        cnn_dim = 512
        lstm_hidden = 256
        
        # Project CNN features
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Output projection
        self.feature_projection = nn.Sequential(
            nn.Linear(lstm_hidden * 2, self.hidden_dim),  # *2 for bidirectional
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Frame selector for selecting key frames from LSTM output
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=8
        )
        
        # Same downstream modules as others
        if self.use_pathology_loss:
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim // 2,
                    dropout=self.dropout_rate,
                    name=f'pathology_{i}'
                ) for i in range(self.num_pathologies)
            ])
        
        if self.use_pathology_loss:
            self.site_integration = SiteIntegrationModule(
                feature_dim=self.hidden_dim,
                site_embed_dim=64,
                hidden_dim=self.hidden_dim,
                num_sites=self.num_sites,
                num_pathologies=self.num_pathologies,
                dropout=self.dropout_rate
            )
        else:
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
            num_heads=8
        )
        
        # Task classifiers
        self.task_classifiers = nn.ModuleDict()
        for task_name in self.active_tasks:
            task_key = task_name.replace(' ', '_').replace('Label', 'label')
            self.task_classifiers[task_key] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
        
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
    
    def process_video_cnn_lstm(self, video):
        """Process video through CNN-LSTM pipeline."""
        batch_size, num_frames, channels, height, width = video.shape
        
        # Reshape to process all frames together
        frames = video.view(-1, channels, height, width)  # [B*T, C, H, W]
        
        # Extract CNN features
        cnn_features = self.cnn_backbone(frames)  # [B*T, 512, H', W']
        cnn_features = self.cnn_pool(cnn_features)  # [B*T, 512, 1, 1]
        cnn_features = cnn_features.view(-1, 512)  # [B*T, 512]
        
        # Project CNN features
        projected_features = self.cnn_projection(cnn_features)  # [B*T, lstm_hidden]
        
        # Reshape for LSTM
        lstm_input = projected_features.view(batch_size, num_frames, -1)  # [B, T, lstm_hidden]
        
        # LSTM processing
        lstm_output, _ = self.lstm(lstm_input)  # [B, T, lstm_hidden*2]
        
        # Project to final output dimension
        output_features = self.feature_projection(lstm_output)  # [B, T, hidden_dim]
        
        return output_features
    
    def forward(self, inputs):
        """Forward pass using CNN-LSTM backbone."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        
        # Process each site
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            
            valid_sites = site_masks[b].sum().item()
            for n in range(valid_sites):
                video = site_videos[b, n].unsqueeze(0)  # [1, T, C, H, W]
                
                # Extract frame-level features using CNN-LSTM
                frame_features = self.process_video_cnn_lstm(video)  # [1, T, hidden_dim]
                
                # Select key frames using attention
                _, _, selected_features = self.frame_selector(frame_features)
                
                # Get top-3 frames (or fewer if not enough frames)
                num_frames = min(3, frame_features.shape[1])
                if num_frames > 0:
                    # Use attention scores to select frames
                    attention_scores, _, _ = self.frame_selector(frame_features)
                    _, top_indices = torch.topk(attention_scores[0], k=num_frames)
                    selected = selected_features[0, top_indices].mean(dim=0, keepdim=True)  # [1, hidden_dim]
                else:
                    selected = selected_features[0, :1]  # First frame
                
                site_features.append(selected)
                
                # Process pathology
                if self.use_pathology_loss and self.pathology_modules is not None:
                    dummy_frames = selected.unsqueeze(1).repeat(1, 3, 1)  # [1, 3, hidden_dim]
                    dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                    
                    pathology_scores = []
                    for module in self.pathology_modules:
                        score, _, _ = module(dummy_frames, dummy_mask)
                        pathology_scores.append(score)
                    
                    site_pathology_scores.append(torch.cat(pathology_scores, dim=1))
            
            # Handle padding (same as ResNet3D)
            if site_features:
                site_features = torch.cat(site_features, dim=0)
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = site_features
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss and site_pathology_scores:
                    site_pathology_scores = torch.cat(site_pathology_scores, dim=0)
                    padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                    padded_scores[:valid_sites] = site_pathology_scores
                    all_pathology_scores.append(padded_scores)
                else:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
            else:
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
        
        # Rest same as ResNet3D
        site_features = torch.stack(all_site_features)
        pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
        
        if self.use_pathology_loss:
            integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
        else:
            integrated_features = self.site_integration(site_features)
        
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                tb_logits = self.tb_classifier(patient_features)
                if self.num_classes == 1:
                    tb_logits = tb_logits.squeeze(-1)
                task_logits['TB Label'] = tb_logits
        
        return {
            'task_logits': task_logits,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_rl_data': []
        }
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Reuse loss computation from MultiTaskModel."""
        return MultiTaskModel.compute_losses(self, outputs, targets, pos_weights)


class VideoTransformerMultiTaskModel(nn.Module):
    """Ablation: Video Vision Transformer (ViViT) for video understanding."""
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 15)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = 'vivit'
        
        logger.info("Using Video Vision Transformer (ViViT) architecture")
        
        # ViViT configuration
        self.num_frames = 16  # Standard for ViViT
        self.patch_size = 16
        self.image_size = 224
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 12
        
        # Create ViViT model
        self.video_transformer = self._create_vivit_model()
        logger.info("Created ViViT model (unpretrained)")
        
        # Feature projection from transformer output
        transformer_dim = self.hidden_size
        self.feature_projection = nn.Sequential(
            nn.Linear(transformer_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Frame selector for temporal attention
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=8
        )
        
        # Same downstream modules
        if self.use_pathology_loss:
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim // 2,
                    dropout=self.dropout_rate,
                    name=f'pathology_{i}'
                ) for i in range(self.num_pathologies)
            ])
        
        if self.use_pathology_loss:
            self.site_integration = SiteIntegrationModule(
                feature_dim=self.hidden_dim,
                site_embed_dim=64,
                hidden_dim=self.hidden_dim,
                num_sites=self.num_sites,
                num_pathologies=self.num_pathologies,
                dropout=self.dropout_rate
            )
        else:
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
            num_heads=8
        )
        
        # Task classifiers
        self.task_classifiers = nn.ModuleDict()
        for task_name in self.active_tasks:
            task_key = task_name.replace(' ', '_').replace('Label', 'label')
            self.task_classifiers[task_key] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
        
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
    
    def _create_vivit_model(self):
        """Create ViViT (Video Vision Transformer) model."""
        class ViViTModel(nn.Module):
            def __init__(self, 
                         image_size=224,
                         patch_size=16, 
                         num_frames=16,
                         hidden_size=768,
                         num_heads=12,
                         num_layers=12,
                         dropout=0.1):
                super().__init__()
                
                self.image_size = image_size
                self.patch_size = patch_size
                self.num_frames = num_frames
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                
                # Calculate number of patches
                self.num_patches_per_frame = (image_size // patch_size) ** 2
                self.total_patches = self.num_patches_per_frame * num_frames
                
                # Patch embedding - convert video patches to embeddings
                self.patch_embedding = nn.Conv3d(
                    in_channels=3,
                    out_channels=hidden_size,
                    kernel_size=(1, patch_size, patch_size),
                    stride=(1, patch_size, patch_size)
                )
                
                # Positional embeddings
                # Spatial position embeddings for each frame
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, self.num_patches_per_frame, hidden_size) * 0.02
                )
                
                # Temporal position embeddings for frames
                self.temporal_pos_embedding = nn.Parameter(
                    torch.randn(1, num_frames, hidden_size) * 0.02
                )
                
                # Class token
                self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
                
                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True  # Pre-norm like in ViT
                )
                
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers
                )
                
                # Layer norm
                self.layer_norm = nn.LayerNorm(hidden_size)
                
                # Dropout
                self.dropout = nn.Dropout(dropout)
                
                # Initialize weights
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                """Initialize weights following ViT/ViViT conventions."""
                if isinstance(module, (nn.Linear, nn.Conv3d)):
                    torch.nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.ones_(module.weight)
            
            def forward(self, videos):
                """
                Forward pass for ViViT.
                
                Args:
                    videos: [B, T, C, H, W] or [B, C, T, H, W]
                """
                # Ensure correct format [B, C, T, H, W]
                if videos.dim() == 5 and videos.shape[2] == 3:
                    videos = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
                
                batch_size = videos.shape[0]
                
                # Extract patches: [B, hidden_size, T, H//patch_size, W//patch_size]
                patches = self.patch_embedding(videos)
                
                # Reshape to [B, T, num_patches_per_frame, hidden_size]
                patches = patches.permute(0, 2, 3, 4, 1)  # [B, T, H', W', hidden_size]
                patches = patches.reshape(batch_size, self.num_frames, self.num_patches_per_frame, self.hidden_size)
                
                # Add spatial positional embeddings to each frame
                patches = patches + self.spatial_pos_embedding.unsqueeze(1)  # Broadcast across time
                
                # Reshape to [B, T*num_patches_per_frame, hidden_size]
                patches = patches.reshape(batch_size, self.total_patches, self.hidden_size)
                
                # Add temporal positional embeddings
                # Create temporal embedding for each patch
                temporal_emb = self.temporal_pos_embedding.repeat_interleave(self.num_patches_per_frame, dim=1)
                patches = patches + temporal_emb
                
                # Add class token
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                patches = torch.cat([cls_tokens, patches], dim=1)
                
                # Apply dropout
                patches = self.dropout(patches)
                
                # Transformer encoding
                encoded = self.transformer(patches)
                
                # Apply final layer norm
                encoded = self.layer_norm(encoded)
                
                # Split CLS token and patch tokens
                cls_output = encoded[:, 0]  # [B, hidden_size]
                patch_outputs = encoded[:, 1:]  # [B, total_patches, hidden_size]
                
                # Reshape patch outputs back to spatial-temporal format
                patch_outputs = patch_outputs.reshape(
                    batch_size, self.num_frames, self.num_patches_per_frame, self.hidden_size
                )
                
                # Average patch outputs per frame to get frame-level features
                frame_features = patch_outputs.mean(dim=2)  # [B, T, hidden_size]
                
                # Create output similar to other transformer models
                output = type('ViViTOutput', (), {
                    'last_hidden_state': frame_features,
                    'pooler_output': cls_output,
                    'patch_outputs': patch_outputs
                })()
                
                return output
        
        return ViViTModel(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout_rate
        )
    
    def process_video_vivit(self, video):
        """Process video through ViViT."""
        batch_size, num_frames, channels, height, width = video.shape
        
        # Resize if needed
        if height != self.image_size or width != self.image_size:
            video = F.interpolate(
                video.view(-1, channels, height, width),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).view(batch_size, num_frames, channels, self.image_size, self.image_size)
        
        # Sample frames if too many
        if num_frames > self.num_frames:
            # Uniform sampling
            indices = torch.linspace(0, num_frames - 1, self.num_frames, dtype=torch.long, device=video.device)
            video = video[:, indices]
        elif num_frames < self.num_frames:
            # Pad frames if too few by repeating the last frame
            padding_needed = self.num_frames - num_frames
            last_frame = video[:, -1:].repeat(1, padding_needed, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=1)
        
        # Process through ViViT
        try:
            outputs = self.video_transformer(video)
            
            # Get frame-level features
            if hasattr(outputs, 'last_hidden_state'):
                frame_features = outputs.last_hidden_state  # [B, T, hidden_size]
            else:
                # Fallback to pooler output repeated across frames
                frame_features = outputs.pooler_output.unsqueeze(1).repeat(1, self.num_frames, 1)
            
            # Project features
            projected_features = self.feature_projection(frame_features)
            
            return projected_features
            
        except Exception as e:
            logger.warning(f"Error in ViViT processing: {e}")
            # Fallback to zero features
            return torch.zeros(batch_size, self.num_frames, self.hidden_dim, 
                             device=video.device, requires_grad=True)
    
    def forward(self, inputs):
        """Forward pass using ViViT."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        
        # Process each site
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            
            valid_sites = site_masks[b].sum().item()
            for n in range(valid_sites):
                video = site_videos[b, n].unsqueeze(0)  # [1, T, C, H, W]
                
                try:
                    # Extract features using ViViT
                    frame_features = self.process_video_vivit(video)  # [1, T, hidden_dim]
                    
                    # Select key frames using attention
                    attention_scores, _, selected_features = self.frame_selector(frame_features)
                    
                    # Get top-3 frames
                    num_frames = min(3, frame_features.shape[1])
                    if num_frames > 0:
                        _, top_indices = torch.topk(attention_scores[0], k=num_frames)
                        selected = selected_features[0, top_indices].mean(dim=0, keepdim=True)
                    else:
                        selected = selected_features[0, :1]
                    
                    site_features.append(selected)
                    
                    # Process pathology
                    if self.use_pathology_loss and self.pathology_modules is not None:
                        dummy_frames = selected.unsqueeze(1).repeat(1, 3, 1)
                        dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                        
                        pathology_scores = []
                        for module in self.pathology_modules:
                            score, _, _ = module(dummy_frames, dummy_mask)
                            pathology_scores.append(score)
                        
                        site_pathology_scores.append(torch.cat(pathology_scores, dim=1))
                
                except Exception as e:
                    logger.warning(f"Error processing video with ViViT: {e}")
                    # Fallback to zeros
                    site_features.append(torch.zeros(1, self.hidden_dim, device=video.device))
                    if self.use_pathology_loss:
                        site_pathology_scores.append(torch.zeros(1, self.num_pathologies, device=video.device))
            
            # Handle padding
            if site_features:
                site_features = torch.cat(site_features, dim=0)
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = site_features
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss and site_pathology_scores:
                    site_pathology_scores = torch.cat(site_pathology_scores, dim=0)
                    padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                    padded_scores[:valid_sites] = site_pathology_scores
                    all_pathology_scores.append(padded_scores)
                else:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
            else:
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
        
        # Rest same as other models
        site_features = torch.stack(all_site_features)
        pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
        
        if self.use_pathology_loss:
            integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
        else:
            integrated_features = self.site_integration(site_features)
        
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                tb_logits = self.tb_classifier(patient_features)
                if self.num_classes == 1:
                    tb_logits = tb_logits.squeeze(-1)
                task_logits['TB Label'] = tb_logits
        
        return {
            'task_logits': task_logits,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_rl_data': []
        }
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Reuse loss computation from MultiTaskModel."""
        return MultiTaskModel.compute_losses(self, outputs, targets, pos_weights)


# =============================================================================
# MODEL FACTORY AND REGISTRY
# =============================================================================

def create_ablation_model(model_type, config):
    """Factory function to create ablation models."""
    
    model_map = {
        'original': MultiTaskModel,
        'no_rl': NoRLMultiTaskModel,
        'mean_pool': MeanPoolMultiTaskModel,
        'attention_pool': AttentionPoolMultiTaskModel,
        'single_task': SingleTaskMultiTaskModel,
        '3d_cnn': ResNet3DMultiTaskModel,  # Using ResNet3D
        'cnn_lstm': CNNLSTMMultiTaskModel,
        'video_transformer': VideoTransformerMultiTaskModel,  # Using ViViT
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    logger.info(f"Creating {model_type} model using PyTorch backbones (ViViT for video_transformer)")
    return model_map[model_type](config)


# Aliases for backward compatibility
CNN3DMultiTaskModel = ResNet3DMultiTaskModel  # Use ResNet3D instead