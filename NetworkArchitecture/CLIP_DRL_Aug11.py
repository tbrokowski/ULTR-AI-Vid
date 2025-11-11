

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict
from transformers import CLIPVisionModel
from safetensors import safe_open
import os
import logging


logger = logging.getLogger(__name__)

# Common mobile-friendly backbones (for config documentation)
MOBILE_BACKBONES = [
    "mobilenetv3_large_100",
    "efficientnet_lite0",
    "ghostnetv2_100",
    "regnety_400mf",
    "levit_256",
    "deit_tiny_distilled_patch16_224"
]

# Optional timm support
try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False

class RewardNormalizer:
    """Tracks reward statistics and normalizes rewards."""
    
    def __init__(self, momentum=0.99, epsilon=1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.momentum = momentum
        self.epsilon = epsilon
    
    def update(self, rewards):
        """Update statistics with new rewards."""
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        # Update running statistics
        self.count += batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / max(self.count, 1)
        
        # Update variance
        new_weight = batch_count / max(self.count, 1)
        self.var = (1 - new_weight) * self.var + new_weight * batch_var + \
                   new_weight * (1 - new_weight) * delta ** 2
    
    def normalize(self, reward):
        """Normalize a reward using current statistics."""
        if self.count > 10:  # Only normalize after seeing enough samples
            return (reward - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return reward
    
    def reset(self):
        """Reset normalizer statistics."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


class FrameSelectionAgent(nn.Module):
    """
    Enhanced frame selection agent that selects diagnostically relevant frames
    with built-in uncertainty estimation and exploration mechanisms.
    """
    
    def __init__(
        self,
        feature_dim=768,  # CLIP ViT-B/32 dimension
        hidden_dim=512,
        output_dim=512,
        num_frame_features=16,
        min_temperature=0.1,
        max_temperature=5.0,
        temperature_decay=0.995,
        entropy_weight=0.01,
        use_frame_history=True,
        device=None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature = max_temperature  # Start with high temperature
        self.temperature_decay = temperature_decay
        self.entropy_weight = entropy_weight
        self.use_frame_history = use_frame_history
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature encoder - extracts multiscale features from raw CLIP features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Multi-scale context extractor
        self.context_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]  # Different context window sizes
        ])
        
        # Temporal position embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 100, hidden_dim // 2)  # Max 100 frames
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.pos_expand = nn.Linear(hidden_dim // 2, feature_dim)
        
        # Frame history encoder (if enabled)
        if use_frame_history:
            self.history_encoder = nn.GRU(
                input_size=output_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True
            )
            
            # Combine history with current features
            self.history_projection = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        
        # Policy network (produces action logits)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value network (estimates state value)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output feature projection for downstream tasks
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Initialize tracking variables
        self.rewards = []
        self.pathology_rewards = []
        self.saved_actions = []
        self.frame_history = {}

    def reset_data_after_update(self):
        """Reset data after policy parameters are updated.
        This maintains on-policy RL training consistency."""
        self.rewards = []
        self.pathology_rewards = []
        self.saved_actions = []  # Clear saved actions after a policy update
    
    def extract_multiscale_features(self, features):
        """Extract multi-scale context features."""
        # Encode base features
        encoded = self.feature_encoder(features)  # [B, T, H]
        
        # Prepare for 1D convolution
        x_t = encoded.transpose(1, 2)  # [B, H, T]
        
        # Apply multi-scale context extraction
        context_features = []
        for conv in self.context_layers:
            # Apply convolution and activation
            conv_feats = F.gelu(conv(x_t))
            # Return to original shape
            conv_feats = conv_feats.transpose(1, 2)  # [B, T, H/4]
            context_features.append(conv_feats)
        
        # Concatenate all scales
        multiscale_features = torch.cat(context_features, dim=2)  # [B, T, H]
        
        return multiscale_features, encoded
    
    def update_temperature(self, decay=None):
        """Update temperature parameter to gradually focus exploration."""
        if decay is None:
            decay = self.temperature_decay
        
        self.temperature = max(
            self.min_temperature,
            self.temperature * decay
        )
        return self.temperature
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.temperature
    
    def clear_history(self):
        """Clear frame selection history."""
        self.frame_history = {}
    
    def reset_rewards(self):
        """Reset stored rewards."""
        self.rewards = []
        self.pathology_rewards = []
    
    def get_history_key(self, batch_idx, site_idx=None):
        """Get key for accessing history dictionary."""
        if site_idx is not None:
            return f"{batch_idx}_{site_idx}"
        return str(batch_idx)
    
    def update_frame_history(self, batch_idx, site_idx, features):
        """Update frame history for a batch/site."""
        key = self.get_history_key(batch_idx, site_idx)
        
        if key not in self.frame_history:
            self.frame_history[key] = []
        
        # Add current features to history (store tensors detached)
        self.frame_history[key].append(features.detach())
        
        # Limit history length
        max_history = 5
        if len(self.frame_history[key]) > max_history:
            self.frame_history[key] = self.frame_history[key][-max_history:]
    
    def get_frame_history(self, batch_idx, site_idx=None):
        """Get frame history for a batch/site."""
        key = self.get_history_key(batch_idx, site_idx)
        return self.frame_history.get(key, [])

    def reset_temperature(self):
        """Reset temperature to initial value."""
        self.temperature = self.max_temperature
        return self.temperature
    
    def clear_history(self):
        """Clear all tracking data."""
        self.frame_history = {}
        self.saved_actions = []
        self.rewards = []
        self.pathology_rewards = []
    
    def reset_rewards(self):
        """Reset reward tracking."""
        self.rewards = []
        self.pathology_rewards = []
    
    def get_history_embedding(self, batch_idx, site_idx, device):
        """Get encoded history embeddings."""
        history = self.get_frame_history(batch_idx, site_idx)
        
        if not history or not self.use_frame_history:
            # Return zero embedding if no history or history disabled
            return torch.zeros(1, self.hidden_dim, device=device)
        
        # Stack history items
        history_tensor = torch.cat(history, dim=0)  # [history_len, hidden_dim]
        
        # Add batch dimension
        history_tensor = history_tensor.unsqueeze(0)  # [1, history_len, hidden_dim]
        
        # Process with GRU
        _, hidden = self.history_encoder(history_tensor)
        
        # Project to match feature dimensions
        history_embedding = self.history_projection(hidden.squeeze(0))  # [1, hidden_dim]
        
        return history_embedding
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        """
        Process features to generate action logits and values.
        
        Args:
            features: Frame features [B, T, D]
            mask: Optional mask [B, T]
            batch_idxs: Batch indices for tracking history
            site_idxs: Site indices for tracking history
            
        Returns:
            action_logits: Action logits [B, T]
            state_values: State values [B, 1]
            encoded_features: Encoded features [B, T, output_dim]
        """
        batch_size, seq_len = features.shape[:2]
        
        # Handle missing mask
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=features.device)
        
        # Default batch/site indices
        if batch_idxs is None:
            batch_idxs = torch.arange(batch_size, device=features.device)
        if site_idxs is None:
            site_idxs = torch.zeros(batch_size, dtype=torch.long, device=features.device)
         
        # Add positional embedding for temporal context
        max_pos_len = min(seq_len, self.pos_embedding.shape[1])  # Don't exceed embedding size
        pos_emb = self.pos_embedding[:, :max_pos_len, :]  # [1, T, hidden_dim//2]

        # Expand to batch size
        pos_emb = pos_emb.expand(batch_size, -1, -1)  # [B, T, hidden_dim//2]

        # If sequence is longer than max positional embeddings, pad with zeros
        if seq_len > max_pos_len:
            padding = torch.zeros(
                batch_size, 
                seq_len - max_pos_len, 
                pos_emb.shape[-1], 
                device=features.device
            )
            pos_emb = torch.cat([pos_emb, padding], dim=1)  # [B, seq_len, hidden_dim//2]

        # Expand positional embedding to match feature dimension using pre-declared layer
        pos_emb_expanded = self.pos_expand(pos_emb)  # [B, T, feature_dim]

        # Add positional encoding to input features
        features_with_pos = features + pos_emb_expanded

        # Extract multi-scale features (using enhanced features)
        multiscale_features, base_features = self.extract_multiscale_features(features_with_pos)
    
        
        # Initialize outputs
        action_logits = torch.zeros(batch_size, seq_len, device=features.device)
        ##########Change Here
        #state_values = torch.zeros(batch_size, 1, device=features.device)
        state_values_list = []
        #################
        encoded_features = torch.zeros(batch_size, seq_len, self.output_dim, device=features.device)
        
        # Process each batch item individually (for history tracking)
        for b in range(batch_size):
            # Get indices for this batch item
            b_idx = batch_idxs[b] if isinstance(batch_idxs, (list, torch.Tensor)) else batch_idxs
            s_idx = site_idxs[b] if isinstance(site_idxs, (list, torch.Tensor)) else site_idxs
            
            # Get history embedding
            history_emb = self.get_history_embedding(b_idx, s_idx, features.device)
            
            ############## CHANGE HERE 
            # Apply masking to find valid frames
            # valid_indices = torch.where(mask[b])[0]
            # if len(valid_indices) == 0:
            #     continue

            valid_indices = torch.where(mask[b])[0]
            if len(valid_indices) == 0:
                # FIXED: Create zero state value with gradients for empty case
                dummy_input = torch.zeros(1, self.hidden_dim * 2, device=features.device, requires_grad=True)
                state_value = self.value_net(dummy_input)
                state_values_list.append(state_value)
                continue
            ####################


            # Get features for valid frames
            valid_features = multiscale_features[b, valid_indices]
            
            # Create policy inputs (combine features with history info)
            # Expand history embedding to match feature dimensions
            expanded_history = history_emb.expand(len(valid_indices), -1)
            
            # Concatenate with features
            policy_inputs = torch.cat([valid_features, expanded_history], dim=1)
            
            # Get action logits and state value
            frame_logits = self.policy_net(policy_inputs).squeeze(-1).float() 
            
            # Add exploration bonus based on temperature
            exploration_bonus = torch.randn_like(frame_logits) * self.temperature * 0.1
            frame_logits = frame_logits + exploration_bonus
            
            # Update action logits for valid frames
            action_logits[b, valid_indices] = frame_logits
            
            # Get overall state value from average features
            ###########Change Here 
            # avg_features = valid_features.mean(dim=0, keepdim=True)
            # avg_state_input = torch.cat([avg_features, history_emb], dim=1)
            # state_values[b] = self.value_net(avg_state_input)
            
            # # Generate output features for all frames
            # encoded_features[b] = self.output_projection(base_features[b])

            avg_features = valid_features.mean(dim=0, keepdim=True)
            avg_state_input = torch.cat([avg_features, history_emb], dim=1)
            state_value = self.value_net(avg_state_input)  # This should have gradients
            state_values_list.append(state_value)
            
            # Generate output features for all frames
            encoded_features[b] = self.output_projection(base_features[b])
        
        # FIXED: Stack state values to preserve gradients
        if state_values_list:
            state_values = torch.cat(state_values_list, dim=0).view(batch_size, -1)
        else:
            state_values = torch.zeros(batch_size, 1, device=features.device, requires_grad=True)
    
        
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        """
        Select action with simple training vs evaluation behavior.
        Training: Full temperature sampling with gradients
        Evaluation: Lower temperature sampling without gradients
        """
        batch_size = logits.shape[0]
        
        if self.training:
            #==================================================
            # TRAINING: Full exploration + gradients + action storage
            #==================================================
            temperature = self.temperature  # Full temperature
            
            # Ensure gradients flow
            if not logits.requires_grad:
                dummy_param = next((p for p in self.parameters() if p.requires_grad), None)
                if dummy_param is not None:
                    logits = logits + dummy_param.sum() * 0.0
            
            # Sample from distribution
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Store actions for reward attribution
            if batch_idx is not None:
                for i in range(batch_size):
                    b_idx = batch_idx[i] if isinstance(batch_idx, (list, torch.Tensor)) else batch_idx
                    s_idx = site_idx[i] if isinstance(site_idx, (list, torch.Tensor)) else site_idx
                    
                    if isinstance(b_idx, torch.Tensor):
                        b_idx = b_idx.item()
                    if isinstance(s_idx, torch.Tensor):
                        s_idx = s_idx.item()
                    
                    action_data = {
                        'batch_idx': b_idx,
                        'site_idx': s_idx,
                        'logits': logits[i],
                        'action': action[i],
                        'log_prob': log_prob[i],
                        'entropy': entropy[i]
                    }
                    
                    if state_values is not None:
                        action_data['state_value'] = state_values[i]
                    
                    if encoded_features is not None and action[i] < encoded_features.shape[1]:
                        selected_features = encoded_features[i, action[i]].unsqueeze(0).detach()
                        self.update_frame_history(b_idx, s_idx, selected_features)
                    
                    self.saved_actions.append(action_data)
            
            return action, log_prob
            
        # else:
        #     #==================================================
        #     # EVALUATION: Lower temperature sampling + no gradients + no storage
        #     #==================================================
        #     with torch.no_grad():
        #         # Use lower temperature for more focused but still diverse sampling
        #         eval_temperature = self.temperature * 0.3  # 30% of training temperature
        #         eval_temperature = max(eval_temperature, 0.2)  # Minimum threshold
                
        #         scaled_logits = logits / eval_temperature
        #         probs = F.softmax(scaled_logits, dim=-1)
        #         dist = torch.distributions.Categorical(probs=probs)
                
        #         # Still sample (for diversity) but more focused
        #         action = dist.sample()
                
        #         # Update frame history for consistency but don't store for rewards
        #         if encoded_features is not None and batch_idx is not None:
        #             for i in range(batch_size):
        #                 b_idx = batch_idx[i] if isinstance(batch_idx, (list, torch.Tensor)) else batch_idx
        #                 s_idx = site_idx[i] if isinstance(site_idx, (list, torch.Tensor)) else site_idx
                        
        #                 if isinstance(b_idx, torch.Tensor):
        #                     b_idx = b_idx.item()
        #                 if isinstance(s_idx, torch.Tensor):
        #                     s_idx = s_idx.item()
                        
        #                 if action[i] < encoded_features.shape[1]:
        #                     selected_features = encoded_features[i, action[i]].unsqueeze(0)
        #                     self.update_frame_history(b_idx, s_idx, selected_features)
                
        #         return action, None  # No log_prob in eval mode
        else:
        #==================================================
        # EVALUATION: Deterministic (greedy) argmax over logits
        #==================================================
            with torch.no_grad():
                # Greedy selection (no temperature, no sampling)
                action = logits.argmax(dim=-1)  # [B]
                
                # Optional: compute log_prob for metrics/analytics
                # (not used for gradients or reward attribution)
                log_prob = torch.gather(
                    F.log_softmax(logits, dim=-1),
                    dim=-1,
                    index=action.unsqueeze(-1)
                ).squeeze(-1)  # [B]
                
                # Update frame history for consistency but don't store for rewards
                if encoded_features is not None and batch_idx is not None:
                    for i in range(batch_size):
                        b_idx = batch_idx[i] if isinstance(batch_idx, (list, torch.Tensor)) else batch_idx
                        s_idx = site_idx[i] if isinstance(site_idx, (list, torch.Tensor)) else site_idx
                        
                        if isinstance(b_idx, torch.Tensor):
                            b_idx = b_idx.item()
                        if isinstance(s_idx, torch.Tensor):
                            s_idx = s_idx.item()
                        
                        if action[i] < encoded_features.shape[1]:
                            selected_features = encoded_features[i, action[i]].unsqueeze(0)
                            self.update_frame_history(b_idx, s_idx, selected_features)
                
                return action, log_prob  # log_prob provided for logging only


class PathologyModule(nn.Module):
    """Streamlined pathology classification module without redundant attention."""
    
    def __init__(self, 
                 feature_dim=512,
                 hidden_dim=256,
                 dropout=0.3,
                 name=None):
        super().__init__()
        self.name = name
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Attention-based frame weighting
        self.frame_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Pathology classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, mask=None):
        """Process features to detect pathology."""
        # Apply feature refinement
        refined = self.feature_refine(features)  # [B, k, hidden_dim]
        
        # Calculate attention weights
        attn_logits = self.frame_attention(refined)  # [B, k, 1]
        
        # Apply mask if provided
        if mask is not None:
            attn_logits = attn_logits.float().masked_fill(~mask.unsqueeze(-1), -1e9)
        
        # Get attention weights
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, k, 1]
        
        # Apply weighted pooling
        pooled = torch.bmm(
            attn_weights.transpose(1, 2),  # [B, 1, k]
            refined                        # [B, k, hidden_dim]
        )  # [B, 1, hidden_dim]
        
        # Classify
        score = self.classifier(pooled.squeeze(1))  # [B, 1]
        
        return score, attn_weights.squeeze(-1), pooled.squeeze(1)


class SiteIntegrationModule(nn.Module):
    """Integrates features from multiple anatomical sites."""
    
    def __init__(self, 
                 feature_dim=512,
                 site_embed_dim=256,
                 hidden_dim=512,
                 num_sites=15,
                 num_pathologies=5,
                 dropout=0.3):
        super().__init__()
        
        # Site embedding
        self.site_embedding = nn.Embedding(num_sites + 1, site_embed_dim)  # +1 for padding
        
        # Feature integration
        self.integration = nn.Sequential(
            nn.Linear(feature_dim + site_embed_dim + num_pathologies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, site_features, site_indices, pathology_scores):
        """
        Integrate site features with anatomical context.
        
        Args:
            site_features: Features for each site [B, N, D]
            site_indices: Indices of anatomical sites [B, N]
            pathology_scores: Pathology scores [B, N, num_pathologies]
            
        Returns:
            integrated_features: Integrated site features [B, N, hidden_dim]
        """
        # Get site embeddings
        site_embeddings = self.site_embedding(site_indices)  # [B, N, site_embed_dim]
        
        # Concatenate site features, embeddings, and pathology scores
        combined = torch.cat([site_features, site_embeddings, pathology_scores], dim=2)
        
        # Apply integration layers
        integrated = self.integration(combined)
        
        return integrated


class DeepAttentionMIL(nn.Module):
    """Deep attention-based Multiple Instance Learning for TB classification."""
    
    def __init__(self, 
                 feature_dim=512,
                 hidden_dim=512,
                 dropout=0.3,
                 num_heads=8):
        super().__init__()
        
        # First attention layer (instance-level)
        self.attention1 = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Second attention layer (bag-level)
        self.attention2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Gating mechanism
        self.gating = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, mask=None):
        """
        Apply deep attention MIL.
        
        Args:
            features: Features from all sites [B, N, D]
            mask: Site mask [B, N]
            
        Returns:
            aggregated: Aggregated features [B, D]
            attention_weights: Attention weights [B, N]
        """
        # Apply first attention layer with the corrected mask
        key_padding_mask = ~mask if mask is not None else None
        attended_features, _ = self.attention1(
            features, features, features,
            key_padding_mask=key_padding_mask
        )
        
        # Apply transformation
        transformed = self.transform(attended_features)
        
        # Apply residual connection
        enhanced = transformed + features
        
        # Calculate attention logits
        attn_logits = self.attention2(enhanced).squeeze(-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            try:
                # Try a different approach to masking - no unsqueeze
                attn_logits = attn_logits.float()
                for i in range(attn_logits.size(0)):
                    for j in range(attn_logits.size(1)):
                        if j < mask.size(1) and not mask[i, j]:
                            attn_logits[i, j] = -1e9
            except Exception as e:
                logger.warning(f"Error in manual masking: {e}")
                # If all else fails, don't mask
                attn_logits = attn_logits.float()
        
        # Get attention weights
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, N]
        
        # Apply gating
        gate_weights = self.gating(enhanced)  # [B, N, 1]
        
        # Apply gated attention
        gated_attention = attn_weights.unsqueeze(-1) * gate_weights  # [B, N, 1]
        
        # Normalize gated attention
        normalizer = gated_attention.sum(dim=1, keepdim=True) + 1e-6
        normalized_attention = gated_attention / normalizer
        
        # Apply weighted aggregation
        aggregated = torch.bmm(
            normalized_attention.transpose(1, 2),  # [B, 1, N]
            enhanced                               # [B, N, D]
        ).squeeze(1)  # [B, D]
        
        return aggregated, attn_weights


class MultiTaskModel(nn.Module):
    """
    Updated TB_DRL_MODEL that maintains original architecture but provides
    the interface expected by the new training system.
    
    This model:
    - Keeps the original TB classification + pathology detection architecture
    - Returns task_logits dict instead of tb_logits for compatibility
    - Maintains all original functionality
    - Works with the new training structure
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 21)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # For compatibility with new training system
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = getattr(config, 'selection_strategy', 'RL')
        
        logger.info(f"MultiTaskModel configured for tasks: {self.active_tasks}")
        logger.info(f"Using pathology loss: {self.use_pathology_loss}")
        logger.info(f"Frame selection strategy: {self.selection_strategy}")
        
        # Log attention parameters for debugging
        attention_temp = getattr(config, 'attention_temperature', 0.5)
        entropy_w = getattr(config, 'entropy_weight', 0.001)
        logger.info(f"Attention temperature: {attention_temp} (lower=sharper, higher=softer)")
        logger.info(f"Entropy regularization weight: {entropy_w}")
        
        # Vision Backbone (mobile-friendly options supported)
        self.backbone = getattr(config, 'backbone', 'clip')  # e.g., 'clip' or a timm model name like 'mobilenetv3_large_100'
        self.backbone_model_name = getattr(config, 'backbone_model_name', 'openai/clip-vit-base-patch32')
        self.freeze_backbone = getattr(config, 'freeze_backbone', False)
        self.pretrained = getattr(config, 'pretrained', True)
        self.backbone_image_size = getattr(config, 'backbone_image_size', 224)

        self._init_vision_backbone()
        self.feature_noise_std = 0.05
        
        # Enhanced RL-based frame selection (original architecture)
        self.frame_selector = FrameSelectionAgent(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            num_frame_features=16,
            min_temperature=getattr(config, 'temperature_min', 0.1),
            max_temperature=getattr(config, 'temperature_max', 5.0),
            temperature_decay=getattr(config, 'temperature_decay', 0.995),
            entropy_weight=getattr(config, 'entropy_weight', 0.01),
            use_frame_history=getattr(config, 'use_frame_history', True),
            device=self.device
        )
        # Freeze/unfreeze according to config
        self._freeze_backbone(freeze=self.freeze_backbone)
        logger.info(f"Using original FrameSelectionAgent for {self.selection_strategy} strategy")
        
        # Pathology modules
        self.pathology_names = [
            'a_lines',
            'large_consolidation',
            'pleural_effusion',
            'other_pathology'
        ]
        
        if self.use_pathology_loss:
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim // 2,
                    dropout=self.dropout_rate,
                    name=name
                ) for name in self.pathology_names
            ])
        else:
            self.pathology_modules = None
        
        # Site integration
        if self.use_pathology_loss:
            self.site_integration = SiteIntegrationModule(
                feature_dim=self.hidden_dim,
                site_embed_dim=256,
                hidden_dim=self.hidden_dim,
                num_sites=self.num_sites,
                num_pathologies=self.num_pathologies,
                dropout=self.dropout_rate
            )
        else:
            # Simple site integration without pathology
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        # Patient-level MIL
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
            num_heads=8
        )
        
        # Task classifiers - using ModuleDict for compatibility
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
        
        # Keep the original TB classifier for backward compatibility
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

    def _init_vision_backbone(self):
        """
        Initialize a vision backbone.
        Supported:
          - 'clip' (uses HuggingFace CLIPVisionModel with pooler_output)
          - Any timm model name (e.g., 'mobilenetv3_large_100', 'efficientnet_lite0', 'ghostnetv2', 'levit_256', 'deit_tiny_distilled_patch16_224')
        Sets:
          - self.vision_encoder: callable/module that returns a tensor [B*T, D] or feature map
          - self.vision_dim: output feature dimension D
          - self._vision_kind: 'clip', 'timm_features', or 'timm_pooled'
          - self.vision_pool / self.vision_proj if needed (for timm)
        """
        # Default outputs
        self._vision_kind = 'clip'
        # Use CLIP only when backbone explicitly requests 'clip'
        if str(self.backbone).lower() == 'clip':
            from transformers import CLIPVisionModel  # local import to avoid hard dep if using timm-only
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                self.backbone_model_name,
                dtype=torch.float32
            )
            # CLIP ViT-B/32 has 768-d pooler_output
            self.vision_dim = getattr(self.vision_encoder.config, 'hidden_size', 768)
            self._vision_kind = 'clip'
            # Optional: local weights loading (kept from original)
            local_weights_path = os.path.join(
                getattr(self.config, 'local_weights_dir', 'NetworkArchitecture/CLIP_weights'),
                'model.safetensors'
            )
            if os.path.exists(local_weights_path):
                logger.info(f"Loading CLIP weights from {local_weights_path}")
                try:
                    from safetensors import safe_open as _safe_open
                    with _safe_open(local_weights_path, framework='pt', device='cpu') as f:
                        vision_state_dict = {}
                        model_state_dict = self.vision_encoder.state_dict()
                        matched_keys = 0
                        for key in model_state_dict.keys():
                            safetensors_key = f"vision_model.{key}"
                            if safetensors_key in f.keys():
                                tensor = f.get_tensor(safetensors_key)
                                if tensor.shape == model_state_dict[key].shape:
                                    vision_state_dict[key] = tensor
                                    matched_keys += 1
                        if matched_keys > 0:
                            logger.info(f"Successfully matched {matched_keys}/{len(model_state_dict)} CLIP weights")
                            self.vision_encoder.load_state_dict(vision_state_dict, strict=False)
                        else:
                            logger.info("No weights could be matched from the safetensors file")
                except Exception as e:
                    logger.warning(f"Failed to load local CLIP weights: {e}")
            else:
                logger.info("No local CLIP weights file found, using default pretrained weights")
        else:
            # Use a timm backbone
            if not _HAS_TIMM:
                raise ImportError("timm is not installed but a timm backbone was requested.")
            self._vision_kind = 'timm'
            model_name = str(self.backbone)
            logger.info(f"Initializing timm backbone: {model_name} (pretrained={self.pretrained})")

            # ---- HANDLE LEVIT FIRST (avoid features_only and filter_fn assert) ----
            if 'levit' in model_name.lower():
                try:
                    # Load with classifier intact so pretrained weights can map
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=self.pretrained
                    )
                except Exception as e:
                    logger.warning(f"LeViT pretrained load failed ({e}); retrying with pretrained=False to bypass filter.")
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=False
                    )

                # Determine feature dim
                self.vision_dim = getattr(self.vision_encoder, 'num_features', None)
                if not isinstance(self.vision_dim, int) or self.vision_dim <= 0:
                    dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                    with torch.no_grad():
                        if hasattr(self.vision_encoder, 'forward_features'):
                            feat_vec = self.vision_encoder.forward_features(dummy)
                        else:
                            feat_vec = self.vision_encoder(dummy)
                    self.vision_dim = int(feat_vec.shape[1])

                # Remove classifier; enforce global pooling for features
                if hasattr(self.vision_encoder, 'reset_classifier'):
                    self.vision_encoder.reset_classifier(0, global_pool='avg')

                self.vision_pool = None
                self.vision_proj = None
                self._vision_kind = 'timm_pooled'
                logger.info(f"Vision backbone: kind=timm_pooled(levit), name={model_name}, vision_dim={self.vision_dim}")
                return
            # ---- END LEVIT EARLY RETURN ----

            # Identify non-convolutional transformer-like models
            nonconv_keys = ['vit', 'deit', 'swin', 'maxvit', 'eva', 'beit', 'convnextv2']
            is_nonconv = any(k in model_name.lower() for k in nonconv_keys)

            if not is_nonconv:
                # Prefer features_only path for CNNs
                try:
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=self.pretrained,
                        features_only=True,
                        out_indices=[-1]
                    )
                    try:
                        self.vision_dim = int(self.vision_encoder.feature_info[-1]['num_chs'])
                    except Exception:
                        dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                        with torch.no_grad():
                            feat = self.vision_encoder(dummy)[0]
                        self.vision_dim = feat.shape[1]
                    self.vision_pool = nn.AdaptiveAvgPool2d(1)
                    self.vision_proj = None
                    self._vision_kind = 'timm_features'
                    logger.info(f"Vision backbone: kind=timm_features, name={model_name}, vision_dim={self.vision_dim}")
                except Exception as e:
                    logger.info(f"features_only path unavailable for {model_name} ({e}). Falling back to pooled forward.")
                    try:
                        self.vision_encoder = timm.create_model(
                            model_name,
                            pretrained=self.pretrained,
                            num_classes=0,
                            global_pool='avg'
                        )
                    except Exception as e2:
                        logger.warning(f"Pooled forward with pretrained=True failed for {model_name} ({e2}). Retrying with pretrained=False.")
                        self.vision_encoder = timm.create_model(
                            model_name,
                            pretrained=False,
                            num_classes=0,
                            global_pool='avg'
                        )
                    self.vision_dim = getattr(self.vision_encoder, 'num_features', None)
                    if not isinstance(self.vision_dim, int) or self.vision_dim <= 0:
                        dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                        with torch.no_grad():
                            feat_vec = self.vision_encoder(dummy)
                        self.vision_dim = int(feat_vec.shape[1])
                    self.vision_pool = None
                    self.vision_proj = None
                    self._vision_kind = 'timm_pooled'
                    logger.info(f"Vision backbone: kind=timm_pooled, name={model_name}, vision_dim={self.vision_dim}")
            else:
                # Transformer-like models: pooled forward
                try:
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=self.pretrained,
                        num_classes=0,
                        global_pool='avg'
                    )
                except Exception as e:
                    logger.warning(f"Transformer pooled forward with pretrained=True failed for {model_name} ({e}). Retrying with pretrained=False.")
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=False,
                        num_classes=0,
                        global_pool='avg'
                    )
                self.vision_dim = getattr(self.vision_encoder, 'num_features', None)
                if not isinstance(self.vision_dim, int) or self.vision_dim <= 0:
                    dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                    with torch.no_grad():
                        feat_vec = self.vision_encoder(dummy)
                    self.vision_dim = int(feat_vec.shape[1])
                self.vision_pool = None
                self.vision_proj = None
                self._vision_kind = 'timm_pooled'
                logger.info(f"Vision backbone: kind=timm_pooled, name={model_name}, vision_dim={self.vision_dim}")
                
                
    def _freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze the vision backbone. If using CLIP and freeze=False,
        we still keep most of CLIP frozen by default, except the last block or visual projection.
        """
        if self._vision_kind == 'clip':
            # Start by freezing all
            for p in self.vision_encoder.parameters():
                p.requires_grad = not freeze
            if not freeze:
                # Unfreeze only the last block by default (safer for fine-tuning on device)
                if hasattr(self.vision_encoder, 'visual_projection'):
                    for p in self.vision_encoder.visual_projection.parameters():
                        p.requires_grad = True
                elif hasattr(self.vision_encoder, 'vision_model') and hasattr(self.vision_encoder.vision_model, 'encoder'):
                    layers = self.vision_encoder.vision_model.encoder.layers
                    if len(layers) > 0:
                        for p in layers[-1].parameters():
                            p.requires_grad = True
        else:
            # timm backbone
            for p in self.vision_encoder.parameters():
                p.requires_grad = not freeze

    def _extract_vision_features(self, frames):
        """
        Generic feature extractor → returns [B, T, D] where D=self.vision_dim.
        For CLIP, uses pooler_output.
        For timm, supports both features_only and pooled models.
        """
        batch_size, num_frames, channels, height, width = frames.shape
        x = frames.view(-1, channels, height, width)  # [B*T, C, H, W]

        if self._vision_kind == 'clip':
            with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
                outputs = self.vision_encoder(x)
                feats = outputs.pooler_output  # [B*T, D]
        elif self._vision_kind == 'timm_features':
            with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
                feat_maps = self.vision_encoder(x)[0]  # [B*T, C, h, w]
                pooled = self.vision_pool(feat_maps)   # [B*T, C, 1, 1]
                feats = pooled.flatten(1)              # [B*T, C]
                if self.vision_proj is not None:
                    feats = self.vision_proj(feats)
        elif self._vision_kind == 'timm_pooled':
            with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
                feats = self.vision_encoder(x)         # [B*T, D] already pooled
        else:
            raise RuntimeError(f"Unknown vision kind: {self._vision_kind}")

        return feats.view(batch_size, num_frames, -1)

    def extract_clip_features(self, frames):
        # Backward compatibility: keep method name but delegate to generic extractor
        return self._extract_vision_features(frames)
    
    def process_site(self, video, site_idx, mask=None, batch_idx=None, site_pos=None):
        """
        Process a single site's video.
        Args:
            video: Video frames [1, T, C, H, W]
            site_idx: Site index
            mask: Frame mask [1, T]
            batch_idx: Batch index for tracking
            site_pos: Site position for tracking
        """
        # 1) Extract vision features (CLIP or timm)
        clip_features = self._extract_vision_features(video)  # [1, T, vision_dim]
        # 2) Run the attention-based frame selector
        action_logits, state_values, enhanced_features = self.frame_selector(
            clip_features, mask, batch_idx, site_pos
        )  # action_logits: [1, T], enhanced_features: [1, T, D]
        # 3) Differentiable hard top-k selection
        selected_repr, action_logp, topk_idx, mask_st = self.frame_selector.select_action(
            logits=action_logits,              # [1, T]
            state_values=state_values,
            encoded_features=enhanced_features,  # [1, T, D]
            batch_idx=batch_idx,
            site_idx=site_pos,
            k=3,
            tau=0.1,
        )
        # selected_repr is [1, D] - pooled representation
        # For pathology modules, we need [1, 1, D] (batch, seq_len=1, features)
        selected_features = selected_repr.unsqueeze(1)  # [1, 1, D]
        selected_mask = torch.ones(1, 1, dtype=torch.bool, device=video.device)
        selected_indices = topk_idx  # [1, 3] —
        # 5) Process pathologies (if enabled)
        pathology_scores = None
        if self.use_pathology_loss and self.pathology_modules is not None:
            pathology_scores = []
            pathology_attentions = []
            pathology_features = []
            for module in self.pathology_modules:
                score, attention, features = module(selected_features, selected_mask)
                pathology_scores.append(score)
                pathology_attentions.append(attention)
                pathology_features.append(features)
            # Concatenate all pathology outputs -> [1, num_pathologies]
            pathology_scores = torch.cat(pathology_scores, dim=1)
        # 6) Return results
        return {
            'selected_features': selected_features,      # [1, 1, D]
            'selected_indices': selected_indices,        # [1, 3]
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos,
            'selection_mask': mask_st,                   # [1, T]
            'selection_logp': action_logp,               # [1]
        }

    def process_patient(self, site_videos, site_indices, site_masks):
        """
        Process videos from multiple anatomical sites for a patient.
        
        Args:
            site_videos: Videos from different sites [B, N, T, C, H, W]
            site_indices: Anatomical site indices [B, N]
            site_masks: Site masks [B, N]
        """
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        all_site_metadata = []
        
        # Process each patient
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            site_metadata = []
            
            # Process each valid site
            valid_sites = site_masks[b].sum().item()
            for n in range(valid_sites):
                # Get video and site index
                video = site_videos[b, n].unsqueeze(0)  # [1, T, C, H, W]
                site_idx = site_indices[b, n].item()
                
                # Create frame masks (all valid initially)
                frame_mask = torch.ones(1, video.shape[1], dtype=torch.bool, device=video.device)
                
                # Process site
                site_output = self.process_site(
                    video, site_idx, frame_mask, batch_idx=b, site_pos=n
                )
                
                # Get selected features and squeeze out sequence dimension
                # selected_features is [1, 1, D], we need [1, D]
                selected_features = site_output['selected_features'].squeeze(1)  # [1, D]
                site_features.append(selected_features)
                
                if self.use_pathology_loss and site_output['pathology_scores'] is not None:
                    site_pathology_scores.append(site_output['pathology_scores'])
                
                # Store site metadata (action logits, state values, etc. for all selection strategies)
                site_metadata.append({
                    'batch_idx': b,
                    'site_idx': n,
                    'selected_indices': site_output['selected_indices'],
                    'action_logits': site_output['action_logits'],
                    'state_values': site_output['state_values']
                })
            
            # Stack outputs for this patient
            if site_features:
                site_features = torch.cat(site_features, dim=0)  # [valid_sites, hidden_dim]
                
                # Create padded tensors
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = site_features
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss:
                    if site_pathology_scores:
                        site_pathology_scores = torch.cat(site_pathology_scores, dim=0)  # [valid_sites, num_pathologies]
                        padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                        padded_scores[:valid_sites] = site_pathology_scores
                        all_pathology_scores.append(padded_scores)
                    else:
                        all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
                
                all_site_metadata.append(site_metadata)
            else:
                # No valid sites
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                if self.use_pathology_loss:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
                all_site_metadata.append([])
        
        # Stack across batch
        all_site_features = torch.stack(all_site_features)  # [B, N, hidden_dim]
        
        if self.use_pathology_loss:
            all_pathology_scores = torch.stack(all_pathology_scores)  # [B, N, num_pathologies]
        else:
            all_pathology_scores = None
        
        return all_site_features, all_pathology_scores, all_site_metadata
    
    def forward(self, inputs):
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary containing:
                - site_videos: Videos from different sites [B, N, T, C, H, W]
                - site_indices: Anatomical site indices [B, N]
                - site_masks: Site masks [B, N]
        """
        # Extract inputs
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        # Process all sites for all patients
        site_features, pathology_scores, site_metadata = self.process_patient(
            site_videos, site_indices, site_masks
        )
        
        # Integrate site features with anatomical context
        if self.use_pathology_loss:
            integrated_features = self.site_integration(
                site_features, site_indices, pathology_scores
            )
        else:
            integrated_features = self.site_integration(site_features)
        
        # Apply patient-level MIL
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
        if self.training and hasattr(self, 'feature_noise_std'):
            noise = torch.randn_like(patient_features) * self.feature_noise_std
            patient_features = patient_features + noise
        
        # Multi-task classification (for compatibility with new training system)
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                # Use the original TB classifier
                tb_logits = self.tb_classifier(patient_features)
                if self.num_classes == 1:
                    tb_logits = tb_logits.squeeze(-1)  # [B]
                task_logits['TB Label'] = tb_logits
            else:
                # Use task-specific classifiers for other tasks
                task_key = task_name.replace(' ', '_').replace('Label', 'label')
                if task_key in self.task_classifiers:
                    logits = self.task_classifiers[task_key](patient_features)
                    if self.num_classes == 1:
                        logits = logits.squeeze(-1)
                    task_logits[task_name] = logits

        # Calculate patient-level pathology scores using MIL attention (if enabled)
        patient_pathology_scores = None
        if self.use_pathology_loss and pathology_scores is not None:
            patient_pathology_scores = torch.bmm(
                mil_attention.unsqueeze(1),  # [B, 1, N]
                pathology_scores  # [B, N, num_pathologies]
            ).squeeze(1)  # [B, num_pathologies]
        
        # Flatten site_metadata into site_outputs for entropy loss computation
        # Each element in site_metadata is a list of dicts for one patient
        site_outputs = []
        for patient_sites in site_metadata:
            site_outputs.extend(patient_sites)
        
        # Return comprehensive output (compatible with new training system)
        output = {
            'task_logits': task_logits,  # NEW: Dict of task_name -> logits
            'patient_pathology_scores': patient_pathology_scores,
            'patient_features': patient_features,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_metadata': site_metadata,  # Metadata for all frame selection strategies
            'site_outputs': site_outputs  # Flattened version for entropy loss computation
        }
        
        # Keep backward compatibility - also include tb_logits
        if 'TB Label' in task_logits:
            output['tb_logits'] = task_logits['TB Label']
        
        return output
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Fixed loss computation with proper gradient handling."""
        loss_dict = {}
        total_loss = 0.0
        
        # Default positive weights
        if pos_weights is None:
            pos_weights = {'TB Label': 1.4}
        
        # Task classification losses (existing code is mostly fine)
        for task_name in self.active_tasks:
            if task_name in outputs.get('task_logits', {}):
                # Get target labels
                if task_name == 'TB Label':
                    target_labels = targets['tb_labels']
                elif task_name == 'Pneumonia Label':
                    target_labels = targets['pneumonia_labels']
                elif task_name == 'Covid Label':
                    target_labels = targets['covid_labels']
                else:
                    continue
                
                # Skip if no valid labels
                valid_mask = target_labels >= 0
                if not valid_mask.any():
                    continue
                
                logits = outputs['task_logits'][task_name]
                
                # Get positive weight for this task
                pos_weight = pos_weights.get(task_name, 2.0)
                pos_weight_tensor = torch.tensor(pos_weight, device=logits.device)
                
                # Binary cross-entropy loss
                task_loss = F.binary_cross_entropy_with_logits(
                    logits[valid_mask],
                    target_labels[valid_mask].float(),
                    pos_weight=pos_weight_tensor
                )
                
                # Apply task weight
                task_weight = self.task_weights.get(task_name, 1.0)
                weighted_task_loss = task_loss * task_weight
                
                loss_dict[f'{task_name}_loss'] = task_loss.item()
                total_loss += weighted_task_loss
        
        # Add entropy regularization to encourage discriminative attention
        # This incentivizes the model to NOT have uniform attention weights
        if 'site_outputs' in outputs and self.training:
            entropy_loss = 0.0
            num_sites = 0
            
            # Get temperature from config (same as used in process_site())
            tau = getattr(self.config, 'attention_temperature', 0.5)
            
            for site_output in outputs['site_outputs']:
                if 'action_logits' in site_output and site_output['action_logits'] is not None:
                    action_logits = site_output['action_logits']
                    
                    # Apply temperature scaling (same as in process_site())
                    # This ensures entropy is computed on the ACTUAL distribution used for selection
                    scaled_logits = action_logits / tau
                    
                    # Compute attention weights (probabilities) with temperature
                    attention_probs = F.softmax(scaled_logits, dim=-1)
                    
                    # Compute entropy: H = -sum(p * log(p))
                    # High entropy = uniform distribution (bad - all frames equally weighted)
                    # Low entropy = peaked distribution (good - focuses on few frames)
                    log_probs = F.log_softmax(scaled_logits, dim=-1)
                    entropy = -torch.sum(attention_probs * log_probs, dim=-1).mean()
                    
                    entropy_loss += entropy
                    num_sites += 1
            
            if num_sites > 0:
                # Average entropy across sites
                avg_entropy = entropy_loss / num_sites
                
                # We want to MINIMIZE entropy (encourage peaked distributions)
                # Use entropy_weight from config (default 0.001)
                entropy_weight = getattr(self.config, 'entropy_weight', 0.001)
                entropy_penalty = entropy_weight * avg_entropy
                total_loss += entropy_penalty
                
                # Log both raw entropy and weighted penalty for monitoring
                loss_dict['attention_entropy'] = avg_entropy.item()
                loss_dict['attention_entropy_penalty'] = entropy_penalty.item()
        
        # FIXED: Pathology detection loss
        if self.use_pathology_loss and 'pathology_scores' in outputs and outputs['pathology_scores'] is not None:
            pathology_scores = outputs['pathology_scores']
            pathology_labels = targets['pathology_labels']
            
            # CRITICAL FIX: Don't force gradients, check if they exist naturally
            # if self.training and not pathology_scores.requires_grad:
            #     logger.warning("Pathology scores lack gradients - skipping pathology loss")
            #     # Skip pathology loss this iteration rather than forcing gradients
            # else:
                # Process each pathology normally
            for i in range(self.num_pathologies):
                # Extract scores and labels for this pathology
                if pathology_scores.dim() == 3:  # [B, N, num_pathologies]
                    path_score_i = pathology_scores[:, :, i]
                    path_label_i = pathology_labels[:, :, i]
                else:
                    path_score_i = pathology_scores[:, i]
                    path_label_i = pathology_labels[:, i]
                
                # Valid mask
                valid_mask = path_label_i >= 0
                
                if valid_mask.any():
                    # Positive weights for different pathologies
                    pos_weights_path = [1.0, 4.0, 4.0, 4.0, 15.0]
                    pos_weight = torch.tensor(pos_weights_path[i % len(pos_weights_path)], device=pathology_scores.device)
                    
                    # Binary cross-entropy loss
                    loss_name = f'pathology_{i}_loss'
                    p_loss = F.binary_cross_entropy_with_logits(
                        path_score_i[valid_mask],
                        path_label_i[valid_mask].float(),
                        pos_weight=pos_weight
                    )
                    
                    # Store loss
                    loss_dict[loss_name] = p_loss.item()
                    
                    # Add to total loss
                    pathology_weight = 0.2
                    total_loss += pathology_weight * p_loss
        
        # Ensure total_loss is a proper tensor with gradients
        if isinstance(total_loss, (int, float)):
            if total_loss == 0:
                # Create a dummy loss with gradients if needed
                dummy_param = next(self.parameters())
                total_loss = torch.tensor(0.0, device=dummy_param.device, requires_grad=True)
        
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


# For backward compatibility with original training code
TB_DRL_MODEL = MultiTaskModel


# Factory function for creating actor-critic trainer (for compatibility)
class ActorCriticTrainer:
    """Compatibility wrapper for RL training."""
    
    def __init__(self, frame_selector, config):
        self.frame_selector = frame_selector
        self.config = config
        
        # Only used for RL frame selectors
        if hasattr(frame_selector, 'policy_net'):
            self.actor_optimizer = None  # Will be set externally
            self.critic_optimizer = None  # Will be set externally
        
        self.reward_normalizer = RewardNormalizer()
    
    def setup_external_optimizers(self, actor_optimizer, critic_optimizer):
        """Allow external management of optimizers."""
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer