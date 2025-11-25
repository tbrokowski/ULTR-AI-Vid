import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict

# Site mapping for anatomical positions
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

class FrameSelectionAgent(nn.Module):
    """Reinforcement learning agent for key frame selection."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 action_type: str = 'discrete',
                 temperature: float = 1.0):
        """
        Initialize the frame selection agent.
        
        Args:
            feature_dim: Dimension of frame features
            hidden_dim: Hidden dimension of policy network
            action_type: 'discrete' or 'continuous' actions
            temperature: Temperature for softmax sampling
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_type = action_type
        self.temperature = temperature

        
        # Policy network for selecting frames
        self.policy_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)  # Importance score for each frame
        )
        
        # Value network for actor-critic
        self.value_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # For RL training
        self.saved_log_probs = []
        self.rewards = []
        
        # Trainable temperature parameter
        self.log_temperature = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(temperature)))
    
    def forward(self, frame_features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            frame_features: Frame features [B, T, D]
            mask: Optional mask for padding [B, T]
            
        Returns:
            action_logits: Action logits [B, T]
            state_values: State values [B]
        """
        batch_size, num_frames, _ = frame_features.shape
        
        # Compute importance scores for each frame
        action_logits = self.policy_network(frame_features).squeeze(-1)  # [B, T]
        
        # Apply mask if provided
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, float('-inf'))
        
        # Compute state values for critic
        avg_features = frame_features.mean(dim=1)  # [B, D]
        state_values = self.value_network(avg_features).squeeze(-1)  # [B]
        
        return action_logits, state_values
    
    def select_action(self, action_logits, batch_idx=None, example_idx=None):
        """
        Select action based on action logits.
        
        Args:
            action_logits: Action logits [B, T]
            
        Returns:
            actions: Selected actions [B]
            log_probs: Log probabilities of selected actions [B]
        """
        # Safety check
        if action_logits.numel() == 0 or torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            return (torch.tensor([], device=action_logits.device, dtype=torch.long), 
                    torch.tensor([], device=action_logits.device))
        
        temperature = torch.exp(self.log_temperature).clamp(0.1, 5.0)
        
        # Apply temperature to logits
        scaled_logits = action_logits / temperature
        
        # Convert to probabilities with stable softmax
        scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sample actions
        if self.training:
            try:
                # Create categorical distribution
                m = torch.distributions.Categorical(probs)
                
                # Sample actions
                actions = m.sample()  # [B]
                log_probs = m.log_prob(actions)  # [B]
                
                # Store log_probs for RL training - we only need the values, not the computational graph
                self.saved_log_probs.append(log_probs.detach())  # Detach is important here
                
                return actions, log_probs
            except Exception as e:
                # Fallback to greedy selection on error
                actions = torch.argmax(probs, dim=-1)
                return actions, None
        else:
            # During inference, just take the most probable action
            actions = torch.argmax(probs, dim=-1)
            return actions, None
    
    def get_reward(self, logits, targets):
        """Get reward based on prediction accuracy."""
        with torch.no_grad():
            if logits.dim() > 1:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                probs = torch.sigmoid(logits.unsqueeze(0))
                preds = (probs > 0.5).float()
                
            # Binary accuracy as reward
            correct = (preds == targets).float()
            
            # Higher reward for confident correct predictions
            confidence = torch.abs(probs - 0.5) * 2  # 0.5->0, 0 or 1->1
            reward = correct * confidence.mean()
            
            # Add probability margin as additional reward component
            prob_margin = torch.abs(probs - 0.5) * 2
            reward += 0.2 * prob_margin.mean()
            
            self.rewards.append(reward.item())
            
            return reward

class PathologyAttentionModule(nn.Module):
    """Specialized attention module for a specific lung pathology."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 name: str = None):
        """
        Initialize pathology-specific attention module.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            name: Name of the pathology
        """
        super().__init__()
        self.name = name
        
        # Feature transformation for query, key, value
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=2, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Pathology classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        Process features to detect specific pathology.
        
        Args:
            features: Frame features [B, T, D]
            mask: Padding mask [B, T]
            
        Returns:
            pathology_score: Pathology detection score [B, 1]
            attention_weights: Attention weights [B, T]
            attended_features: Pathology-specific features [B, hidden_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Transform features
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            query=q, 
            key=k, 
            value=v,
            key_padding_mask=~mask if mask is not None else None,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Add & norm
        attended_features = self.norm1(attended_features + q)
        
        # Feed-forward & norm
        ff_output = self.ffn(attended_features)
        attended_features = self.norm2(attended_features + ff_output)

        if attention_weights.dim() == 1:
            attention_weights = attention_weights.unsqueeze(0)  # Add batch dimension
        if attention_weights.dim() == 2:
            attention_weights = attention_weights.unsqueeze(1)  # Add middle dimension

        #print("Weights", attention_weights.shape)
        #print("Features",attended_features.shape)

        # Global pooling with attention weights
        attended_features = torch.bmm(
            attention_weights,  # [B, 1, T]
            attended_features  # [B, T, D]
        ).squeeze(1)  # [B, D]
        
        # Classify pathology
        pathology_score = self.classifier(attended_features)  # [B, 1]
        
        return pathology_score, attention_weights, attended_features

class SpatialFeatureExtraction(nn.Module):
    """Extracts spatial features from key frames using pathology-specific attention modules."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize spatial feature extraction module.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for attention modules
            dropout: Dropout rate
        """
        super().__init__()
        
        # Pathology attention modules
        self.pathology_names = [
            'a_lines',          # 0: A-lines
            'b_lines',          # 1: B-lines
            'confluent_b',      # 2: Confluent B-lines
            'small_consolidation', # 3: Small consolidations/nodules
            'large_consolidation', # 4: Large consolidations
            'pleural_effusion',  # 5: Pleural effusion
            'pattern a',
            'not_measured',
        ]
        
        self.pathology_modules = nn.ModuleList([
            PathologyAttentionModule(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                name=name
            ) for name in self.pathology_names
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(self.pathology_names), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        Extract spatial features using pathology-specific attention.
        
        Args:
            features: Frame features [B, T, D]
            mask: Padding mask [B, T]
            
        Returns:
            dict: Dictionary containing:
                - pathology_scores: Pathology scores [B, num_pathologies]
                - attention_weights: Attention weights [num_pathologies, B, T]
                - spatial_features: Fused spatial features [B, hidden_dim]
        """
        pathology_scores = []
        attention_weights = []
        pathology_features = []
        
        # Process each pathology
        for module in self.pathology_modules:
            score, attn, feat = module(features, mask)
            pathology_scores.append(score)
            attention_weights.append(attn)
            pathology_features.append(feat)
        
        # Stack scores and weights
        pathology_scores = torch.cat(pathology_scores, dim=1)  # [B, num_pathologies]
        attention_weights = torch.stack(attention_weights)  # [num_pathologies, B, T]
        
        # Concatenate and fuse pathology features
        concat_features = torch.cat(pathology_features, dim=1)  # [B, num_pathologies*hidden_dim]
        spatial_features = self.feature_fusion(concat_features)  # [B, hidden_dim]
        
        return {
            'pathology_scores': pathology_scores,
            'attention_weights': attention_weights,
            'spatial_features': spatial_features
        }

class TemporalFeatureExtraction(nn.Module):
    """Extracts temporal features from sequential frames."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        """
        Initialize temporal feature extraction.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension
            kernel_size: Kernel size for 1D convolution
            dropout: Dropout rate
        """
        super().__init__()
        
        # Temporal convolution
        self.conv1d = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        Extract temporal features from sequence.
        
        Args:
            features: Frame features [B, T, D]
            mask: Padding mask [B, T]
            
        Returns:
            temporal_features: Temporal features [B, hidden_dim]
            attention_weights: Attention weights [B, T]
        """
        batch_size, seq_len, feat_dim = features.shape
        
        # Transpose for convolution
        features_t = features.transpose(1, 2)  # [B, D, T]
        
        # Apply convolution
        conv_features = F.gelu(self.conv1d(features_t))  # [B, hidden_dim, T]
        
        # Transpose back
        conv_features = conv_features.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Apply attention pooling
        attention_logits = self.attention_pool(conv_features).squeeze(-1)  # [B, T]
        
        # Apply mask if provided
        if mask is not None:
            attention_logits = attention_logits.masked_fill(~mask, float('-inf'))
        
        # Attention weights
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, T]
        
        # Weighted sum
        temporal_features = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, T]
            conv_features  # [B, T, hidden_dim]
        ).squeeze(1)  # [B, hidden_dim]
        
        return temporal_features, attention_weights

class SiteLevelIntegration(nn.Module):
    """Integrates spatial and temporal features at the site level."""
    
    def __init__(self, 
                 spatial_dim: int = 256,
                 temporal_dim: int = 256,
                 site_embed_dim: int = 64,
                 hidden_dim: int = 512,
                 num_pathologies: int = 8,
                 dropout: float = 0.3):
        """
        Initialize site-level integration.
        
        Args:
            spatial_dim: Dimension of spatial features
            temporal_dim: Dimension of temporal features
            site_embed_dim: Dimension of site embeddings
            hidden_dim: Hidden dimension
            num_pathologies: Number of pathologies
            dropout: Dropout rate
        """
        super().__init__()
        
        # Site embedding
        self.site_embedding = nn.Embedding(len(SITE_MAPPING), site_embed_dim)
        
        # Feature fusion
        combined_dim = spatial_dim + temporal_dim + site_embed_dim + num_pathologies
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                pathology_scores: torch.Tensor,
                site_indices: torch.Tensor):
        """
        Integrate features at site level.
        
        Args:
            spatial_features: Spatial features [B, spatial_dim]
            temporal_features: Temporal features [B, temporal_dim]
            pathology_scores: Pathology scores [B, num_pathologies]
            site_indices: Site indices [B]
            
        Returns:
            site_features: Integrated site features [B, hidden_dim]
        """
        # Get site embeddings
        site_embeddings = self.site_embedding(site_indices)  # [B, site_embed_dim]


        # Concatenate all features
        combined_features = torch.cat([
            spatial_features,
            temporal_features,
            site_embeddings,
            pathology_scores
        ], dim=1)  # [B, combined_dim]
        
        # Fuse features
        site_features = self.feature_fusion(combined_features)  # [B, hidden_dim]
        
        return site_features

class MultiInstanceLearning(nn.Module):
    """Multiple Instance Learning for patient-level aggregation."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize MIL module.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        Aggregate features with MIL attention.
        
        Args:
            features: Instance features [B, N, D]
            mask: Instance mask [B, N]
            
        Returns:
            aggregated_features: Aggregated features [B, D]
            attention_weights: Attention weights [B, N]
        """
        # Apply attention mechanism
        a_v = self.attention_V(features)  # [B, N, hidden_dim]
        a_u = self.attention_U(features)  # [B, N, hidden_dim]
        a = self.attention_weights(a_v * a_u)  # [B, N, 1]
        a = a.squeeze(-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(~mask, float('-inf'))
        
        # Softmax attention
        attention_weights = F.softmax(a, dim=1)  # [B, N]
        
        # Weighted aggregation
        aggregated_features = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, N]
            features  # [B, N, D]
        ).squeeze(1)  # [B, D]
        
        # Transform features
        aggregated_features = self.feature_transform(aggregated_features)  # [B, D]
        
        return aggregated_features, attention_weights

class DRLMILModel(nn.Module):
    """
    Deep Reinforcement Learning Multiple Instance Learning Model for TB diagnosis
    from lung ultrasound videos.
    """
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.frame_sampling = getattr(config, 'frame_sampling', 16)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 8)
        
        # Feature extraction backbone
        backbone_name = getattr(config, 'backbone', 'resnet18')
        self.backbone_dim = 512 if backbone_name == 'resnet18' else 2048
        self.backbone = self._get_backbone(backbone_name)
        
        # Freeze backbone if specified
        if getattr(config, 'freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # RL-based frame selection
        self.frame_selector = FrameSelectionAgent(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            action_type='discrete',
            temperature=getattr(config, 'temperature', 1.0)
        )
        
        # Spatial feature extraction with pathology attention
        self.spatial_feature_extraction = SpatialFeatureExtraction(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate
        )
        
        # Temporal feature extraction
        self.temporal_feature_extraction = TemporalFeatureExtraction(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            kernel_size=3,
            dropout=self.dropout_rate
        )
        
        # Site-level integration
        self.site_integration = SiteLevelIntegration(
            spatial_dim=self.hidden_dim // 2,
            temporal_dim=self.hidden_dim // 2,
            site_embed_dim=64,
            hidden_dim=self.hidden_dim,
            num_pathologies=self.num_pathologies,
            dropout=self.dropout_rate
        )
        
        # Patient-level MIL aggregation
        self.patient_mil = MultiInstanceLearning(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate
        )
        
        # TB classification
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        self.tb_classifier[-1].bias.data.fill_(1.0)
    
    def _get_backbone(self, backbone_name):
        """Get the feature extraction backbone."""
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def _extract_frame_features(self, frames):
        """
        Extract features from frames using the backbone.
        
        Args:
            frames: Video frames [B, T, C, H, W]
            
        Returns:
            frame_features: Frame features [B, T, hidden_dim]
        """
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        
        # Reshape for backbone
        flat_frames = frames.view(-1, frames.shape[2], frames.shape[3], frames.shape[4])
        
        # Extract features
        with torch.no_grad() if not self.backbone.training else torch.enable_grad():
            features = self.backbone(flat_frames)
            features = features.view(batch_size, num_frames, -1)
        
        # Project features
        frame_features = self.feature_projection(features)
        
        return frame_features
    
    def _select_key_frames(self, frame_features, mask=None, batch_idx=None):
        """
        Select key frames using RL agent with better tracking.
        
        Args:
            frame_features: Frame features [B, T, D]
            mask: Frame mask [B, T]
            batch_idx: Optional batch index for better reward tracking
            
        Returns:
            selected_indices: Indices of selected frames [B]
            action_logits: Action logits [B, T]
            state_values: State values [B]
        """
        # Get action logits and state values
        action_logits, state_values = self.frame_selector(frame_features, mask)
        
        # Select actions (frame indices) with batch tracking
        batch_size = frame_features.shape[0]
        selected_indices_list = []
        log_probs_list = []
        
        for b in range(batch_size):
            # Select for each example in batch separately for better tracking
            example_logits = action_logits[b:b+1]
            selected_idx, log_prob = self.frame_selector.select_action(
                example_logits, 
                batch_idx=batch_idx, 
                example_idx=b
            )
            selected_indices_list.append(selected_idx)
            if log_prob is not None:
                log_probs_list.append(log_prob)
        
        # Stack results
        selected_indices = torch.cat(selected_indices_list)
        
        return selected_indices, action_logits, state_values
    
    def _get_surrounding_frames(self, frame_features, selected_indices, window_size=2):
        """
        Get surrounding frames for temporal context.
        
        Args:
            frame_features: Frame features [B, T, D]
            selected_indices: Selected frame indices [B]
            window_size: Number of frames to include on each side
            
        Returns:
            context_features: Features of selected frame and surrounding frames [B, 2*window_size+1, D]
        """
        batch_size, num_frames, feat_dim = frame_features.shape
        context_size = 2 * window_size + 1
        
        # Initialize context features
        context_features = torch.zeros(batch_size, context_size, feat_dim, device=frame_features.device)
        
        # For each batch item
        for b in range(batch_size):
            center_idx = selected_indices[b]
            
            # Valid range of frames
            start_idx = max(0, center_idx - window_size)
            end_idx = min(num_frames, center_idx + window_size + 1)
            
            # Get context frames
            context = frame_features[b, start_idx:end_idx]
            
            # Handle edge cases with padding
            context_start_idx = max(0, window_size - center_idx)
            context_end_idx = context_start_idx + (end_idx - start_idx)
            
            context_features[b, context_start_idx:context_end_idx] = context
        
        return context_features
    
    def _process_site(self, video, site_idx, mask=None):
        """
        Process a single site's video.
        
        Args:
            video: Video tensor [B, T, C, H, W] where B=1
            site_idx: Site index
            mask: Optional frame mask [B, T]
            
        Returns:
            dict: Dictionary of site-level features and outputs
        """
        # Extract frame features
        frame_features = self._extract_frame_features(video)  # [1, T, D]
        
        # Select key frame with RL
        selected_idx, action_logits, state_value = self._select_key_frames(frame_features, mask)  # [1]
        
        # Get context frames around selected frame
        context_features = self._get_surrounding_frames(frame_features, selected_idx)  # [1, 5, D]
        
        # Extract spatial features from selected frame
        selected_feature = torch.stack([frame_features[b, idx] for b, idx in enumerate(selected_idx)])  # [1, D]
        spatial_output = self.spatial_feature_extraction(selected_feature.unsqueeze(1))  # Dict
        
        # Extract temporal features from context
        temporal_features, temp_attn = self.temporal_feature_extraction(context_features)  # [1, D]
        
        # Integrate spatial and temporal features
        site_features = self.site_integration(
            spatial_output['spatial_features'],
            temporal_features,
            spatial_output['pathology_scores'],
            site_idx.unsqueeze(0) if isinstance(site_idx, torch.Tensor) else torch.tensor([site_idx], device=video.device)
        )  # [1, D]
        
        # Return comprehensive output
        return {
            'site_features': site_features,
            'pathology_scores': spatial_output['pathology_scores'],
            'selected_idx': selected_idx,
            'action_logits': action_logits,
            'state_value': state_value,
            'spatial_attn': spatial_output['attention_weights'],
            'temporal_attn': temp_attn
        }
    
    def _process_patient_videos(self, inputs):
        """
        Process patient-level videos with MIL.
        
        Args:
            inputs: Dictionary containing:
                - videos: Tensor [B, N, T, C, H, W]
                - site_indices: Tensor [B, N]
                - site_masks: Tensor [B, N]
            
        Returns:
            dict: Dictionary of patient-level outputs and losses
        """
        videos = inputs['site_videos']  # [B, N, T, C, H, W]
        site_indices = inputs['site_indices']  # [B, N]
        site_masks = inputs['site_masks']  # [B, N]
        
        batch_size, max_sites = videos.shape[0], videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        all_site_rl_data = []
        
        # Process each patient
        for b in range(batch_size):
            site_features = []
            pathology_scores = []
            site_rl_data = []
            
            # Process each site for this patient
            valid_sites = site_masks[b].sum().item()
            for n in range(valid_sites):
                # Get video and site index
                video = videos[b, n].unsqueeze(0)  # [1, T, C, H, W]
                site_idx = site_indices[b, n].item()
                
                # Process site
                site_output = self._process_site(video, site_idx)
                
                # Collect outputs
                site_features.append(site_output['site_features'])
                pathology_scores.append(site_output['pathology_scores'])
                site_rl_data.append({
                    'selected_idx': site_output['selected_idx'],
                    'action_logits': site_output['action_logits'],
                    'state_value': site_output['state_value']
                })
            
            # Stack site features and scores for this patient
            if site_features:
                site_features = torch.cat(site_features, dim=0)  # [valid_sites, D]
                pathology_scores = torch.cat(pathology_scores, dim=0)  # [valid_sites, num_pathologies]
                
                # Create padded tensors
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=videos.device)
                padded_features[:valid_sites] = site_features
                
                padded_scores = torch.zeros(max_sites, self.num_pathologies, device=videos.device)
                padded_scores[:valid_sites] = pathology_scores
                
                all_site_features.append(padded_features)
                all_pathology_scores.append(padded_scores)
                all_site_rl_data.append(site_rl_data)
            else:
                # No valid sites, create empty tensors
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=videos.device))
                all_site_rl_data.append([])
        
        # Stack site features across batch
        all_site_features = torch.stack(all_site_features)  # [B, N, D]
        all_pathology_scores = torch.stack(all_pathology_scores)  # [B, N, num_pathologies]
        
        # Apply MIL for patient-level aggregation
        patient_features, mil_attention = self.patient_mil(all_site_features, site_masks)  # [B, D]
        
        # TB classification
        tb_logits = self.tb_classifier(patient_features)  # [B, num_classes] or [B, 1]
        if self.num_classes == 1:
            tb_logits = tb_logits.squeeze(-1)  # [B]
        
        # Calculate patient-level pathology scores
        patient_pathology_scores = torch.bmm(
            mil_attention.unsqueeze(1),  # [B, 1, N]
            all_pathology_scores  # [B, N, num_pathologies]
        ).squeeze(1)  # [B, num_pathologies]
        
        # Return comprehensive output
        return {
            'tb_logits': tb_logits,
            'patient_pathology_scores': patient_pathology_scores,
            'mil_attention': mil_attention,
            'site_features': all_site_features,
            'pathology_scores': all_pathology_scores,
            'site_rl_data': all_site_rl_data
        }
    
    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            dict: Dictionary of outputs and losses
        """
        # Check if patient-level data
        is_patient_level = inputs.get('is_patient_level', 
                                     'site_masks' in inputs and 'site_videos' in inputs)
        
        if is_patient_level:
            # Process patient-level videos
            outputs = self._process_patient_videos(inputs)
            return outputs
            
            # Return TB logits as primary output for compatibility with existing training code
            if isinstance(outputs, dict) and 'tb_logits' in outputs:
                primary_output = outputs['tb_logits']
            else:
                primary_output = outputs
                
            return primary_output
        else:
            # For site-level processing or compatibility with different input formats
            if 'videos' in inputs:
                videos = inputs['videos']  # [B, T, C, H, W]
                site_indices = inputs.get('site_indices', torch.zeros(videos.shape[0], dtype=torch.long, device=videos.device))
                
                # Extract frame features
                frame_features = self._extract_frame_features(videos)
                
                # Select key frames
                selected_indices, action_logits, state_values = self._select_key_frames(frame_features)
                
                # Process selected frames
                batch_size = videos.shape[0]
                site_outputs = []
                
                for b in range(batch_size):
                    site_output = self._process_site(
                        videos[b:b+1],
                        site_indices[b].item() if isinstance(site_indices, torch.Tensor) else site_indices
                    )
                    site_outputs.append(site_output)
                
                # Extract pathology scores
                pathology_scores = torch.cat([output['pathology_scores'] for output in site_outputs])
                
                return pathology_scores
            
            else:
                # Fallback for incompatible inputs
                raise ValueError("Unsupported input format")
    
    # def compute_losses(self, outputs, targets):
    #     """
    #     Compute all losses for training.
        
    #     Args:
    #         outputs: Dictionary of model outputs
    #         targets: Dictionary of targets
            
    #     Returns:
    #         dict: Dictionary of losses
    #     """
    #     losses = {}
        
    #     # TB classification loss
    #     if 'tb_logits' in outputs and 'tb_labels' in targets:
    #         tb_logits = outputs['tb_logits']
    #         tb_labels = targets['tb_labels']
            
    #         # Binary classification loss
    #         if self.num_classes == 1 or (isinstance(tb_logits, torch.Tensor) and tb_logits.shape[-1] == 1):
    #             losses['tb_loss'] = F.binary_cross_entropy_with_logits(
    #                 tb_logits.view(-1),
    #                 tb_labels.float().view(-1),
    #                 pos_weight= torch.tensor(1.6)
    #             )
    #         else:
    #             losses['tb_loss'] = F.cross_entropy(tb_logits, tb_labels)
        
    #     # Pathology classification loss if targets available
    #     if 'pathology_scores' in outputs and 'pathology_labels' in targets:

    #         pathology_scores = outputs['pathology_scores']
    #         pathology_labels = targets['pathology_labels']

    #         #print("Pathology scores:", pathology_scores)
    #         #print("Pathology labels", pathology_labels)
            
    #         # Binary classification for each pathology
    #         losses['pathology_loss'] = F.binary_cross_entropy_with_logits(
    #             pathology_scores.view(-1, self.num_pathologies),
    #             pathology_labels.float().view(-1, self.num_pathologies)
    #         )
        
    #     # RL policy gradient loss
    #     if 'site_rl_data' in outputs:
    #         rl_data = outputs['site_rl_data']
            
    #         # Skip if no RL data
    #         if rl_data and isinstance(rl_data, list) and all(isinstance(d, list) for d in rl_data):
    #             # No direct RL loss here - handled separately in training loop
    #             # This avoids detached gradients when mixed with supervised losses
    #             losses['rl_loss'] = torch.tensor(0.0, device=outputs['tb_logits'].device)
        
    #     # Compute total loss (weighted sum)
    #     total_loss = 0.0
    #     for loss_name, loss_value in losses.items():
    #         weight = getattr(self.config, f"{loss_name}_weight", 1.0)
    #         total_loss += weight * loss_value
        
    #     losses['total_loss'] = total_loss

    #     return total_loss
        
    #     #return losses
    def compute_losses(self, outputs, targets):
        """
        Compute all losses for training.
        
        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of targets
            
        Returns:
            total_loss: Total loss for optimization
        """
        losses = {}
        
        # TB classification loss
        if 'tb_logits' in outputs and 'tb_labels' in targets:
            tb_logits = outputs['tb_logits']
            tb_labels = targets['tb_labels']
            
            # Binary classification loss
            if self.num_classes == 1 or (isinstance(tb_logits, torch.Tensor) and tb_logits.shape[-1] == 1):
                losses['tb_loss'] = F.binary_cross_entropy_with_logits(
                    tb_logits.view(-1),
                    tb_labels.float().view(-1),
                    pos_weight=torch.tensor(2.5, device=tb_logits.device)
                )
            else:
                losses['tb_loss'] = F.cross_entropy(tb_logits, tb_labels)
            
            # Compute rewards for frame selection based on TB prediction
            with torch.no_grad():
                probs = torch.sigmoid(tb_logits)
                preds = (probs > 0.5).float()
                correct = (preds == tb_labels).float()
                confidence = torch.abs(probs - 0.5) * 2
                reward = correct * confidence
                
                # Use the reward to calculate an advantage function
                # Store this for use in the RL loss component
                self.advantage = reward.detach()
        
        # Pathology classification loss if targets available
        if 'pathology_scores' in outputs and 'pathology_labels' in targets:
            pathology_scores = outputs['pathology_scores']
            pathology_labels = targets['pathology_labels']
            
            losses['pathology_loss'] = F.binary_cross_entropy_with_logits(
                pathology_scores.view(-1, self.num_pathologies),
                pathology_labels.float().view(-1, self.num_pathologies)
            )
        
        # RL loss component - integrate with main graph
        if 'site_rl_data' in outputs and hasattr(self, 'advantage'):
            rl_data = outputs['site_rl_data']
            if rl_data and isinstance(rl_data, list) and all(isinstance(d, dict) for d in rl_data):
                # Extract action_logits from RL data
                rl_loss = 0
                for i, site_data in enumerate(rl_data):
                    if 'action_logits' in site_data and self.advantage is not None:
                        logits = site_data['action_logits']
                        # Add a small entropy term to encourage exploration
                        probs = F.softmax(logits, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
                        # Use advantage for policy gradient
                        rl_loss -= entropy * 0.01  # Small entropy bonus
                        if self.advantage.numel() > i:
                            rl_loss -= self.advantage[i] * site_data.get('log_prob', 0.0)
                
                if rl_loss != 0:
                    losses['rl_loss'] = rl_loss
        
        # Compute total loss
        total_loss = 0.0
        for loss_name, loss_value in losses.items():
            weight = getattr(self.config, f"{loss_name}_weight", 1.0)
            total_loss += weight * loss_value
        
        return total_loss