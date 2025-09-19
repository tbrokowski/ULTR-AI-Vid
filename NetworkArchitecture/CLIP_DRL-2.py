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

# Constants
SITE_MAPPING = OrderedDict([
    ("<PAD>", 0),
    ("QAID", 1),
    ("QAIG", 2),
    ("QASD", 3),
    ("QASG", 4),
    ("QLD", 5),
    ("QLG", 6),
    ("QPID", 7),
    ("QPIG", 8),
    ("QPSD", 9),
    ("QPSG", 10),
    ("APXD", 11),
    ("APXG", 12),
    ("QSLD", 13),
    ("QSLG", 14)
])


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
        
        # Extract multi-scale features
        multiscale_features, base_features = self.extract_multiscale_features(features)
        
        # Add positional embedding for temporal context
        pos_idx = torch.arange(seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        pos_idx = torch.clamp(pos_idx, max=99)  # Limit to max positional embedding size
        pos_emb = self.pos_embedding[:, pos_idx].to(features.device)
        
        # Initialize outputs
        action_logits = torch.zeros(batch_size, seq_len, device=features.device)
        state_values = torch.zeros(batch_size, 1, device=features.device)
        encoded_features = torch.zeros(batch_size, seq_len, self.output_dim, device=features.device)
        
        # Process each batch item individually (for history tracking)
        for b in range(batch_size):
            # Get indices for this batch item
            b_idx = batch_idxs[b] if isinstance(batch_idxs, (list, torch.Tensor)) else batch_idxs
            s_idx = site_idxs[b] if isinstance(site_idxs, (list, torch.Tensor)) else site_idxs
            
            # Get history embedding
            history_emb = self.get_history_embedding(b_idx, s_idx, features.device)
            
            # Apply masking to find valid frames
            valid_indices = torch.where(mask[b])[0]
            if len(valid_indices) == 0:
                continue
            
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
            avg_features = valid_features.mean(dim=0, keepdim=True)
            avg_state_input = torch.cat([avg_features, history_emb], dim=1)
            state_values[b] = self.value_net(avg_state_input)
            
            # Generate output features for all frames
            encoded_features[b] = self.output_projection(base_features[b])
        
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        """Select action based on logits and state values."""
        batch_size = logits.shape[0]
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Create categorical distribution
        probs = F.softmax(scaled_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        # Get entropy
        entropy = dist.entropy()
        
        # Store action data for reward attribution
        if batch_idx is not None:
            for i in range(batch_size):
                b_idx = batch_idx[i] if isinstance(batch_idx, (list, torch.Tensor)) else batch_idx
                s_idx = site_idx[i] if isinstance(site_idx, (list, torch.Tensor)) else site_idx
                
                action_data = {
                    'batch_idx': b_idx,
                    'site_idx': s_idx,
                    'logits': logits[i].detach(),
                    'action': action[i].detach(),
                    'log_prob': log_prob[i].detach(),
                    'entropy': entropy[i].detach()
                }
                
                # Add state value if available
                if state_values is not None:
                    action_data['state_value'] = state_values[i].detach()
                
                # Update frame history if encoded features available
                if encoded_features is not None and action[i] < encoded_features.shape[1]:
                    # Get features for selected frame
                    selected_features = encoded_features[i, action[i]].unsqueeze(0)
                    self.update_frame_history(b_idx, s_idx, selected_features)
                
                # Save action data
                self.saved_actions.append(action_data)
        
        return action, log_prob


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
                 site_embed_dim=64,
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
        # Apply first attention layer
        
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
                print(f"Error in manual masking: {e}")
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


class TB_DRL_MODEL(nn.Module):
    """
    Optimized Deep Reinforcement Learning model for TB classification from
    lung ultrasound videos with multi-site integration.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 5)
        self.num_sites = getattr(config, 'num_sites', 15)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # CLIP Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32
        )

        self.feature_noise_std = 0.05
        
        # Load local weights if available
        local_weights_path = os.path.join(
            getattr(config, 'local_weights_dir', '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/NetworkArchitecture/CLIP_weights'),
            'model.safetensors'
        )
        
        if os.path.exists(local_weights_path):
            print(f"Loading weights from {local_weights_path}")
            with safe_open(local_weights_path, framework='pt', device='cpu') as f:
                all_keys = f.keys()
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
                
                # Update model with matched weights
                if matched_keys > 0:
                    print(f"Successfully matched {matched_keys}/{len(model_state_dict)} weights")
                    self.vision_encoder.load_state_dict(vision_state_dict, strict=False)
                else:
                    print("No weights could be matched from the safetensors file")
        
        # Freeze backbone if specified
        # if getattr(config, 'freeze_backbone', False):
        #     for param in self.vision_encoder.parameters():
        #         param.requires_grad = False

        self._freeze_clip_except_last_layer()
        
        # Vision feature dimension
        self.vision_dim = 768  # CLIP ViT-B/32 dimension
        
        # Enhanced RL-based frame selection
        self.frame_selector = FrameSelectionAgent(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            #temperature=getattr(config, 'temperature', 3.0),
            num_frame_features=16,
            device=self.device
        )
        
        # Pathology modules
        self.pathology_names = [
            'a_lines',
            'b_lines',
            'small_consolidation',
            'large_consolidation',
            'pleural_effusion',
        ]
        
        self.pathology_modules = nn.ModuleList([
            PathologyModule(
                feature_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                dropout=self.dropout_rate,
                name=name
            ) for name in self.pathology_names
        ])
        
        # Site integration
        self.site_integration = SiteIntegrationModule(
            feature_dim=self.hidden_dim,
            site_embed_dim=64,
            hidden_dim=self.hidden_dim,
            num_sites=self.num_sites,
            num_pathologies=self.num_pathologies,
            dropout=self.dropout_rate
        )
        
        # Patient-level MIL
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
            num_heads=8
        )
        
        # TB classifier
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

    def _freeze_clip_except_last_layer(self):
        """Freeze all CLIP parameters except the last layer."""
        # First freeze everything
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Then unfreeze the last layer (visual projection)
        if hasattr(self.vision_encoder, 'visual_projection'):
            for param in self.vision_encoder.visual_projection.parameters():
                param.requires_grad = True
        
        # If no visual_projection, unfreeze the final transformer layer
        elif hasattr(self.vision_encoder, 'vision_model') and hasattr(self.vision_encoder.vision_model, 'encoder'):
            layers = self.vision_encoder.vision_model.encoder.layers
            if len(layers) > 0:
                for param in layers[-1].parameters():
                    param.requires_grad = True
    
    def extract_clip_features(self, frames):
        """Extract features using CLIP vision encoder."""
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Reshape for CLIP input
        frames_flat = frames.view(-1, channels, height, width)
        
        # Extract features
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            outputs = self.vision_encoder(frames_flat)
            
            # Get features from the last hidden state
            # Shape: [batch_size * num_frames, vision_dim]
            features = outputs.pooler_output
            
            # Reshape back to [batch_size, num_frames, vision_dim]
            features = features.view(batch_size, num_frames, -1)
        
        return features
    
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
        # Extract CLIP features
        clip_features = self.extract_clip_features(video)  # [1, T, vision_dim]
        
        # Select key frames using enhanced frame selector
        action_logits, state_values, enhanced_features = self.frame_selector(
            clip_features, mask, batch_idx, site_pos
        )
        
        # Sample actions (frame indices)
        actions, _ = self.frame_selector.select_action(
            action_logits, state_values, enhanced_features, batch_idx, site_pos
        )
        
        # Get indices of 3 highest-scoring frames
        if mask is not None:
            # Apply mask to get valid logits
            valid_mask = mask[0]
            valid_logits = action_logits[0, valid_mask]
            valid_indices = torch.where(valid_mask)[0]
            
            if valid_indices.numel() > 0:
                if valid_indices.numel() >= 3:
                    # Get top 3 indices
                    _, top_local_indices = torch.topk(valid_logits, k=3)
                    selected_indices = valid_indices[top_local_indices]
                else:
                    # Not enough valid frames, repeat the valid ones
                    selected_indices = valid_indices.repeat((3 + valid_indices.numel() - 1) // valid_indices.numel())
                    selected_indices = selected_indices[:3]
            else:
                # No valid frames
                selected_indices = torch.zeros(3, dtype=torch.long, device=video.device)
        else:
            # Get top 3 indices directly
            _, selected_indices = torch.topk(action_logits[0], k=3)
        
        # Get features for selected frames
        selected_features = torch.stack([
            enhanced_features[0, idx] for idx in selected_indices
        ])
        
        # Add batch dimension
        selected_features = selected_features.unsqueeze(0)  # [1, 3, hidden_dim]
        
        # Create mask for selected features
        selected_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
        
        # Process pathologies
        pathology_scores = []
        pathology_attentions = []
        pathology_features = []
        
        for module in self.pathology_modules:
            score, attention, features = module(selected_features, selected_mask)
            pathology_scores.append(score)
            pathology_attentions.append(attention)
            pathology_features.append(features)
        
        # Stack pathology outputs
        pathology_scores = torch.cat(pathology_scores, dim=1)  # [1, num_pathologies]
        
        # Return comprehensive output
        return {
            'selected_features': selected_features,
            'selected_indices': selected_indices.unsqueeze(0),  # [1, 3]
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos
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
        all_site_rl_data = []
        
        # Process each patient
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            site_rl_data = []
            
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
                
                # Get selected features and pathology scores
                selected_features = site_output['selected_features'].mean(dim=1)  # [1, hidden_dim]
                site_features.append(selected_features)
                site_pathology_scores.append(site_output['pathology_scores'])
                
                # Store RL data
                site_rl_data.append({
                    'batch_idx': b,
                    'site_idx': n,
                    'selected_indices': site_output['selected_indices'],
                    'action_logits': site_output['action_logits'],
                    'state_values': site_output['state_values']
                })
            
            # Stack outputs for this patient
            if site_features:
                site_features = torch.cat(site_features, dim=0)  # [valid_sites, hidden_dim]
                site_pathology_scores = torch.cat(site_pathology_scores, dim=0)  # [valid_sites, num_pathologies]
                
                # Create padded tensors
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = site_features
                
                padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                padded_scores[:valid_sites] = site_pathology_scores
                
                all_site_features.append(padded_features)
                all_pathology_scores.append(padded_scores)
                all_site_rl_data.append(site_rl_data)
            else:
                # No valid sites
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
                all_site_rl_data.append([])
        
        # Stack across batch
        all_site_features = torch.stack(all_site_features)  # [B, N, hidden_dim]
        all_pathology_scores = torch.stack(all_pathology_scores)  # [B, N, num_pathologies]
        
        return all_site_features, all_pathology_scores, all_site_rl_data
    
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
        site_features, pathology_scores, site_rl_data = self.process_patient(
            site_videos, site_indices, site_masks
        )
        
        # Integrate site features with anatomical context
        integrated_features = self.site_integration(
            site_features, site_indices, pathology_scores
        )
        
        # Apply patient-level MIL
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
        if self.training and hasattr(self, 'feature_noise_std'):
            noise = torch.randn_like(patient_features) * self.feature_noise_std
            patient_features = patient_features + noise
        
        # TB classification
        tb_logits = self.tb_classifier(patient_features)
        if self.num_classes == 1:
            tb_logits = tb_logits.squeeze(-1)  # [B]


        # Calculate patient-level pathology scores using MIL attention
        patient_pathology_scores = torch.bmm(
            mil_attention.unsqueeze(1),  # [B, 1, N]
            pathology_scores  # [B, N, num_pathologies]
        ).squeeze(1)  # [B, num_pathologies]
        
        # Return comprehensive output
        return {
            'tb_logits': tb_logits,
            'patient_pathology_scores': patient_pathology_scores,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_rl_data': site_rl_data
        }
    
    def compute_losses(self, outputs, targets):
        """
        Compute combined losses for TB classification and pathology detection
        with proper gradient handling.
        
        Args:
            outputs: Model outputs
            targets: Target labels
        
        Returns:
            total_loss: Combined loss with gradient information preserved
        """
        import logging
        logger = logging.getLogger(__name__)
        losses = {}
        
        # TB classification loss
        if 'tb_logits' in outputs and 'tb_labels' in targets:
            tb_logits = outputs['tb_logits']
            tb_labels = targets['tb_labels']
            
            # Ensure tb_logits has gradients (only warn during training mode)
            if self.training and not tb_logits.requires_grad:
                logger.warning("TB logits lack gradients - creating connection to computation graph")
                # Find a parameter with gradients to connect to
                param_with_grad = next((p for p in self.parameters() if p.requires_grad), None)
                if param_with_grad is not None:
                    connection_factor = (param_with_grad.sum() * 0.0 + 1.0)
                    tb_logits = tb_logits * connection_factor
            
            
            # Binary cross-entropy loss
            tb_loss = F.binary_cross_entropy_with_logits(
                tb_logits.view(-1),
                tb_labels.float().view(-1),
                pos_weight=torch.tensor(1.4, device=tb_logits.device)
            )
            
            # Store loss
            losses['tb_loss'] = tb_loss
            
            # Compute rewards for RL agent - DETACH ALL TENSORS FOR REWARDS
            with torch.no_grad():
                probs = torch.sigmoid(tb_logits)
                preds = (probs > 0.5).float()
                correct = (preds == tb_labels).float()
                
                # Asymmetric rewards
                rewards = torch.where(
                    correct > 0,
                    1.0 + torch.abs(probs - 0.5),  # Correct: 1.0-1.5
                    -2.0 * torch.abs(probs - 0.5)  # Incorrect: -0.0 to -1.0
                )
                
                # Store rewards with explicit detaching
                if hasattr(self.frame_selector, 'rewards'):
                    batch_size = tb_labels.shape[0]
                    for b in range(batch_size):
                        reward_value = rewards[b].item()  # Use scalar value
                        
                        # Find all actions for this batch
                        batch_actions = [a for a in self.frame_selector.saved_actions 
                                        if a.get('batch_idx') == b]
                        
                        for action_data in batch_actions:
                            # Create reward entry with explicit detaching
                            self.frame_selector.rewards.append({
                                'value': reward_value,  # Already a scalar
                                'type': 'tb',
                                'batch_idx': b,
                                'site_idx': action_data.get('site_idx'),
                                'logits': action_data['logits'].detach().clone(),  # Explicitly detach
                                'action': action_data['action'].detach().clone(),  # Explicitly detach
                                'log_prob': action_data.get('log_prob', None),  # Keep original for RL
                                'state_value': action_data.get('state_value', torch.tensor(0.0))  # Keep original for RL
                            })
        
        # Pathology detection loss
        if 'pathology_scores' in outputs and 'pathology_labels' in targets:
            pathology_scores = outputs['pathology_scores']
            pathology_labels = targets['pathology_labels']
            
            # Ensure pathology_scores has gradients (only in training mode)
            if self.training and not pathology_scores.requires_grad:
                logger.warning("Pathology scores lack gradients - creating connection")
                # Find a parameter with gradients
                param_with_grad = next((p for p in self.parameters() if p.requires_grad), None)
                if param_with_grad is not None:
                    connection_factor = (param_with_grad.sum() * 0.0 + 1.0)
                    pathology_scores = pathology_scores * connection_factor
            
            # Process each pathology
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
                    pos_weights = [1.0, 4.0, 4.0, 4.0, 15.0]
                    pos_weight = torch.tensor(pos_weights[i % len(pos_weights)], device=pathology_scores.device)
                    
                    # Binary cross-entropy loss
                    loss_name = f'pathology_{i}_loss'
                    p_loss = F.binary_cross_entropy_with_logits(
                        path_score_i[valid_mask],
                        path_label_i[valid_mask].float(),
                        pos_weight=pos_weight
                    )
                    
                    # Store loss
                    losses[loss_name] = p_loss
                    
                    # Store pathology rewards - DETACH ALL TENSORS
                    with torch.no_grad():
                        probs_i = torch.sigmoid(path_score_i[valid_mask])
                        preds_i = (probs_i > 0.5).float()
                        correct_i = (preds_i == path_label_i[valid_mask].float()).float()
                        
                        # Calculate reward factors
                        if hasattr(self.frame_selector, 'pathology_rewards'):
                            importance = [1.0, 1.2, 1.1, 1.3, 1.5][i % 5]
                            
                            # Get batch indices
                            batch_indices = torch.nonzero(valid_mask.any(dim=1)).squeeze(-1)
                            
                            for b_idx in batch_indices:
                                batch_valid = valid_mask[b_idx]
                                if not batch_valid.any():
                                    continue
                                
                                # Calculate correctness
                                site_indices = torch.nonzero(batch_valid).squeeze(-1)
                                batch_probs = torch.sigmoid(path_score_i[b_idx, batch_valid])
                                batch_preds = (batch_probs > 0.5).float()
                                batch_labels = path_label_i[b_idx, batch_valid].float()
                                batch_correct = (batch_preds == batch_labels).float()
                                
                                # Average correctness
                                avg_correct = batch_correct.mean().item()  # Use scalar
                                
                                # Reward with importance
                                path_reward = (2 * avg_correct - 1) * importance
                                
                                # Find actions
                                b_idx_item = b_idx.item() if isinstance(b_idx, torch.Tensor) else b_idx
                                batch_actions = [a for a in self.frame_selector.saved_actions 
                                            if a.get('batch_idx') == b_idx_item]
                                
                                for action_data in batch_actions:
                                    # Store with explicit detaching
                                    self.frame_selector.pathology_rewards.append({
                                        'value': path_reward,  # Scalar
                                        'type': f'pathology_{i}',
                                        'batch_idx': b_idx_item,
                                        'site_idx': action_data.get('site_idx'),
                                        'logits': action_data['logits'].detach().clone(),  # Detach
                                        'action': action_data['action'].detach().clone(),  # Detach
                                        'log_prob': action_data.get('log_prob', None),  # Keep original for RL
                                        'state_value': action_data.get('state_value', torch.tensor(0.0))  # Keep original for RL
                                    })
        
        # Calculate total loss
        tb_weight = 1.0
        pathology_weight = 0.9
        
        # Explicitly create a total loss tensor that requires gradients
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        if 'tb_loss' in losses:
            total_loss = total_loss + tb_weight * losses['tb_loss']
        
        # Add pathology losses
        pathology_losses = [v for k, v in losses.items() if k.startswith('pathology_')]
        if pathology_losses:
            pathology_avg = sum(pathology_losses) #/ len(pathology_losses)
            total_loss = total_loss + pathology_weight * pathology_avg
        
        # Final check
        if self.training and not total_loss.requires_grad:
            logger.error("Total loss still doesn't require gradients! Creating final connection.")
            param_with_grad = next((p for p in self.parameters() if p.requires_grad), None)
            if param_with_grad is not None:
                connection_factor = (param_with_grad.sum() * 0.0 + 1.0)
                total_loss = total_loss * connection_factor
        
        return total_loss