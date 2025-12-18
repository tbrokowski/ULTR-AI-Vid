

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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