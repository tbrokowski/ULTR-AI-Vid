"""
TensorFlow implementation of general components for the ULTR-AI model.
Converted from PyTorch to TensorFlow/Keras.
"""

import tensorflow as tf
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


class RewardNormalizerTF:
    """Tracks reward statistics and normalizes rewards. TF version."""
    
    def __init__(self, momentum=0.99, epsilon=1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.momentum = momentum
        self.epsilon = epsilon
    
    def update(self, rewards):
        """Update statistics with new rewards."""
        if isinstance(rewards, tf.Tensor):
            rewards = rewards.numpy()
        
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


class FrameSelectionAgentTF(tf.keras.Model):
    """
    Enhanced frame selection agent that selects diagnostically relevant frames
    with built-in uncertainty estimation and exploration mechanisms.
    TensorFlow/Keras implementation.
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature = max_temperature  # Start with high temperature
        self.temperature_decay = temperature_decay
        self.entropy_weight = entropy_weight
        self.use_frame_history = use_frame_history
        
        # Feature encoder - extracts multiscale features from raw CLIP features
        self.feature_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh')
        ], name='feature_encoder')
        
        # Multi-scale context extractor (Conv1D layers)
        self.context_layers = [
            tf.keras.layers.Conv1D(hidden_dim // 4, kernel_size=k, padding='same', name=f'context_conv_{k}')
            for k in [1, 3, 5, 7]  # Different context window sizes
        ]
        
        # Temporal position embedding
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, 100, hidden_dim // 2),  # Max 100 frames
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        self.pos_expand = tf.keras.layers.Dense(feature_dim, name='pos_expand')
        
        # Frame history encoder (if enabled)
        if use_frame_history:
            self.history_encoder = tf.keras.layers.GRU(
                units=hidden_dim // 2,
                return_sequences=False,
                return_state=True,
                name='history_encoder'
            )
            
            # Combine history with current features
            self.history_projection = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim),
                tf.keras.layers.LayerNormalization(epsilon=1e-05),
                tf.keras.layers.Activation('gelu')
            ], name='history_projection')
        
        # Policy network (produces action logits)
        self.policy_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ], name='policy_net')
        
        # Value network (estimates state value)
        self.value_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ], name='value_net')
        
        # Output feature projection for downstream tasks
        self.output_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh')
        ], name='output_projection')
        
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
    
    def extract_multiscale_features(self, features, training=False):
        """Extract multi-scale context features."""
        # Encode base features
        encoded = self.feature_encoder(features, training=training)  # [B, T, H]
        
        # Apply multi-scale context extraction
        context_features = []
        for conv in self.context_layers:
            # Apply convolution and activation
            conv_feats = tf.nn.gelu(conv(encoded))  # [B, T, H/4]
            context_features.append(conv_feats)
        
        # Concatenate all scales
        multiscale_features = tf.concat(context_features, axis=2)  # [B, T, H]
        
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
        
        # Add current features to history (store tensors)
        self.frame_history[key].append(tf.stop_gradient(features))
        
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
    
    def get_history_embedding(self, batch_idx, site_idx):
        """Get encoded history embeddings."""
        history = self.get_frame_history(batch_idx, site_idx)
        
        if not history or not self.use_frame_history:
            # Return zero embedding if no history or history disabled
            return tf.zeros((1, self.hidden_dim))
        
        # Stack history items
        history_tensor = tf.concat(history, axis=0)  # [history_len, hidden_dim]
        
        # Add batch dimension
        history_tensor = tf.expand_dims(history_tensor, 0)  # [1, history_len, hidden_dim]
        
        # Process with GRU
        _, hidden = self.history_encoder(history_tensor)
        
        # Project to match feature dimensions
        history_embedding = self.history_projection(hidden)  # [1, hidden_dim]
        
        return history_embedding
    
    def call(self, features, mask=None, batch_idxs=None, site_idxs=None, training=False):
        """
        Process features to generate action logits and values.
        
        Args:
            features: Frame features [B, T, D]
            mask: Optional mask [B, T]
            batch_idxs: Batch indices for tracking history
            site_idxs: Site indices for tracking history
            training: Whether in training mode
            
        Returns:
            action_logits: Action logits [B, T]
            state_values: State values [B, 1]
            encoded_features: Encoded features [B, T, output_dim]
        """
        batch_size = tf.shape(features)[0]
        seq_len = tf.shape(features)[1]
        
        # Handle missing mask
        if mask is None:
            mask = tf.ones((batch_size, seq_len), dtype=tf.bool)
        
        # Default batch/site indices
        if batch_idxs is None:
            batch_idxs = tf.range(batch_size)
        if site_idxs is None:
            site_idxs = tf.zeros((batch_size,), dtype=tf.int32)
         
        # Add positional embedding for temporal context
        max_pos_len = min(seq_len, self.pos_embedding.shape[1])
        pos_emb = self.pos_embedding[:, :max_pos_len, :]  # [1, T, hidden_dim//2]

        # Tile to batch size
        pos_emb = tf.tile(pos_emb, [batch_size, 1, 1])  # [B, T, hidden_dim//2]

        # If sequence is longer than max positional embeddings, pad with zeros
        if seq_len > max_pos_len:
            padding = tf.zeros((batch_size, seq_len - max_pos_len, pos_emb.shape[-1]))
            pos_emb = tf.concat([pos_emb, padding], axis=1)  # [B, seq_len, hidden_dim//2]

        # Expand positional embedding to match feature dimension using pre-declared layer
        pos_emb_expanded = self.pos_expand(pos_emb)  # [B, T, feature_dim]

        # Add positional encoding to input features
        features_with_pos = features + pos_emb_expanded

        # Extract multi-scale features (using enhanced features)
        multiscale_features, base_features = self.extract_multiscale_features(features_with_pos, training=training)
    
        # Initialize outputs as lists (for dynamic building)
        action_logits_list = []
        state_values_list = []
        encoded_features_list = []
        
        # Process each batch item individually (for history tracking)
        # NOTE: This loop is needed for history tracking but is not efficient in TF
        # For production, consider vectorizing this if history tracking is not needed
        for b in range(batch_size):
            # Get indices for this batch item
            b_idx = batch_idxs[b].numpy() if hasattr(batch_idxs[b], 'numpy') else batch_idxs[b]
            s_idx = site_idxs[b].numpy() if hasattr(site_idxs[b], 'numpy') else site_idxs[b]
            
            # Get history embedding
            history_emb = self.get_history_embedding(b_idx, s_idx)
            
            # Apply masking to find valid frames
            valid_mask_b = mask[b]
            valid_indices = tf.where(valid_mask_b)[:, 0]
            
            if tf.size(valid_indices) == 0:
                # Create zero state value for empty case
                dummy_input = tf.zeros((1, self.hidden_dim * 2))
                state_value = self.value_net(dummy_input, training=training)
                state_values_list.append(state_value)
                action_logits_list.append(tf.zeros((1, seq_len)))
                encoded_features_list.append(tf.zeros((1, seq_len, self.output_dim)))
                continue

            # Get features for valid frames
            valid_features = tf.gather(multiscale_features[b], valid_indices)
            
            # Create policy inputs (combine features with history info)
            # Expand history embedding to match feature dimensions
            expanded_history = tf.tile(history_emb, [tf.shape(valid_features)[0], 1])
            
            # Concatenate with features
            policy_inputs = tf.concat([valid_features, expanded_history], axis=1)
            
            # Get action logits
            frame_logits = tf.squeeze(self.policy_net(policy_inputs, training=training), axis=-1)
            frame_logits = tf.cast(frame_logits, tf.float32)
            
            # Add exploration bonus based on temperature
            if training:
                exploration_bonus = tf.random.normal(tf.shape(frame_logits)) * self.temperature * 0.1
                frame_logits = frame_logits + exploration_bonus
            
            # Create full logits tensor and scatter valid values
            full_logits = tf.zeros((seq_len,))
            full_logits = tf.tensor_scatter_nd_update(
                full_logits, 
                tf.expand_dims(valid_indices, 1), 
                frame_logits
            )
            action_logits_list.append(tf.expand_dims(full_logits, 0))
            
            # Get overall state value from average features
            avg_features = tf.reduce_mean(valid_features, axis=0, keepdims=True)
            avg_state_input = tf.concat([avg_features, history_emb], axis=1)
            state_value = self.value_net(avg_state_input, training=training)
            state_values_list.append(state_value)
            
            # Generate output features for all frames
            enc_feats = self.output_projection(base_features[b:b+1], training=training)
            encoded_features_list.append(enc_feats)
        
        # Stack outputs
        action_logits = tf.concat(action_logits_list, axis=0)
        state_values = tf.concat(state_values_list, axis=0)
        encoded_features = tf.concat(encoded_features_list, axis=0)
    
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, 
                      batch_idx=None, site_idx=None, training=False):
        """
        Select action with simple training vs evaluation behavior.
        Training: Full temperature sampling with gradients
        Evaluation: Lower temperature sampling without gradients
        """
        batch_size = tf.shape(logits)[0]
        
        if training:
            # TRAINING: Full exploration + action storage
            temperature = self.temperature
            
            # Sample from distribution using Gumbel-Softmax trick for differentiability
            scaled_logits = logits / temperature
            probs = tf.nn.softmax(scaled_logits, axis=-1)
            
            # Sample actions
            action = tf.random.categorical(tf.math.log(probs + 1e-10), num_samples=1)
            action = tf.squeeze(action, axis=-1)
            
            # Compute log probabilities
            log_probs = tf.nn.log_softmax(scaled_logits, axis=-1)
            action_log_probs = tf.gather(log_probs, action, axis=1, batch_dims=1)
            
            # Store actions for reward attribution (simplified - batch-level)
            if batch_idx is not None:
                for i in range(batch_size):
                    b_idx = batch_idx[i].numpy() if hasattr(batch_idx[i], 'numpy') else batch_idx[i]
                    s_idx = site_idx[i].numpy() if hasattr(site_idx[i], 'numpy') else site_idx[i]
                    
                    action_data = {
                        'batch_idx': b_idx,
                        'site_idx': s_idx,
                        'logits': logits[i],
                        'action': action[i],
                        'log_prob': action_log_probs[i],
                    }
                    
                    if state_values is not None:
                        action_data['state_value'] = state_values[i]
                    
                    if encoded_features is not None:
                        action_val = action[i].numpy() if hasattr(action[i], 'numpy') else action[i]
                        if action_val < encoded_features.shape[1]:
                            selected_features = tf.expand_dims(encoded_features[i, action_val], 0)
                            self.update_frame_history(b_idx, s_idx, tf.stop_gradient(selected_features))
                    
                    self.saved_actions.append(action_data)
            
            return action, action_log_probs
            
        else:
            # EVALUATION: Deterministic (greedy) argmax over logits
            action = tf.argmax(logits, axis=-1, output_type=tf.int32)
            
            # Compute log_prob for metrics/analytics
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            action_log_probs = tf.gather(log_probs, action, axis=1, batch_dims=1)
            
            # Update frame history for consistency
            if encoded_features is not None and batch_idx is not None:
                for i in range(batch_size):
                    b_idx = batch_idx[i].numpy() if hasattr(batch_idx[i], 'numpy') else batch_idx[i]
                    s_idx = site_idx[i].numpy() if hasattr(site_idx[i], 'numpy') else site_idx[i]
                    
                    action_val = action[i].numpy() if hasattr(action[i], 'numpy') else action[i]
                    if action_val < encoded_features.shape[1]:
                        selected_features = tf.expand_dims(encoded_features[i, action_val], 0)
                        self.update_frame_history(b_idx, s_idx, tf.stop_gradient(selected_features))
            
            return action, action_log_probs


class PathologyModuleTF(tf.keras.Model):
    """Streamlined pathology classification module without redundant attention. TF version."""
    
    def __init__(self, 
                 feature_dim=512,
                 hidden_dim=256,
                 dropout=0.3,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.pathology_name = name  # Store the pathology name
        self._hidden_dim = hidden_dim
        
        # Feature refinement
        self.feature_refine = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(dropout)
        ], name='feature_refine')
        
        # Attention-based frame weighting
        self.frame_attention = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim // 2),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dense(1)
        ], name='frame_attention')
        
        # Pathology classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1)
        ], name='classifier')
    
    def call(self, features, mask=None, training=False):
        """Process features to detect pathology."""
        # Apply feature refinement
        refined = self.feature_refine(features, training=training)  # [B, k, hidden_dim]
        
        # Calculate attention weights
        attn_logits = self.frame_attention(refined, training=training)  # [B, k, 1]
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            attn_logits = tf.where(
                tf.cast(mask_expanded, tf.bool),
                attn_logits,
                tf.ones_like(attn_logits) * -1e9
            )
        
        # Get attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=1)  # [B, k, 1]
        
        # Apply weighted pooling
        pooled = tf.matmul(
            tf.transpose(attn_weights, [0, 2, 1]),  # [B, 1, k]
            refined                                  # [B, k, hidden_dim]
        )  # [B, 1, hidden_dim]
        
        # Classify
        score = self.classifier(tf.squeeze(pooled, axis=1), training=training)  # [B, 1]
        
        return score, tf.squeeze(attn_weights, axis=-1), tf.squeeze(pooled, axis=1)


class SiteIntegrationModuleTF(tf.keras.Model):
    """Integrates features from multiple anatomical sites. TF version."""
    
    def __init__(self, 
                 feature_dim=512,
                 site_embed_dim=256,
                 hidden_dim=512,
                 num_sites=15,
                 num_pathologies=5,
                 dropout=0.3,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Site embedding
        self.site_embedding = tf.keras.layers.Embedding(num_sites + 1, site_embed_dim)  # +1 for padding
        
        # Feature integration
        self.integration = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('gelu')
        ], name='integration')
        
        self._feature_dim = feature_dim
        self._site_embed_dim = site_embed_dim
        self._num_pathologies = num_pathologies
    
    def call(self, site_features, site_indices, pathology_scores, training=False):
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
        combined = tf.concat([site_features, site_embeddings, pathology_scores], axis=2)
        
        # Apply integration layers
        integrated = self.integration(combined, training=training)
        
        return integrated


class DeepAttentionMILTF(tf.keras.Model):
    """Deep attention-based Multiple Instance Learning for TB classification. TF version."""
    
    def __init__(self, 
                 feature_dim=512,
                 hidden_dim=512,
                 dropout=0.3,
                 num_heads=8,
                 **kwargs):
        super().__init__(**kwargs)
        
        self._feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        
        # First attention layer (instance-level)
        self.attention1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=feature_dim // num_heads,
            dropout=dropout
        )
        
        # Feature transformation
        self.transform = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(feature_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05)
        ], name='transform')
        
        # Second attention layer (bag-level)
        self.attention2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dense(1)
        ], name='attention2')
        
        # Gating mechanism
        self.gating = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ], name='gating')
    
    def call(self, features, mask=None, training=False):
        """
        Apply deep attention MIL.
        
        Args:
            features: Features from all sites [B, N, D]
            mask: Site mask [B, N]
            
        Returns:
            aggregated: Aggregated features [B, D]
            attention_weights: Attention weights [B, N]
        """
        # Create attention mask for MultiHeadAttention
        # TF MHA uses attention_mask where True means "attend" (opposite of key_padding_mask)
        if mask is not None:
            # Create attention mask: [B, 1, 1, N] for broadcasting
            attention_mask = tf.cast(mask, tf.float32)
            attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 1)
        else:
            attention_mask = None
        
        # Apply first attention layer
        attended_features = self.attention1(
            features, features, features,
            attention_mask=attention_mask,
            training=training
        )
        
        # Apply transformation
        transformed = self.transform(attended_features, training=training)
        
        # Apply residual connection
        enhanced = transformed + features
        
        # Calculate attention logits
        attn_logits = tf.squeeze(self.attention2(enhanced, training=training), axis=-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            attn_logits = tf.where(
                tf.cast(mask, tf.bool),
                attn_logits,
                tf.ones_like(attn_logits) * -1e9
            )
        
        # Get attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=1)  # [B, N]
        
        # Apply gating
        gate_weights = self.gating(enhanced, training=training)  # [B, N, 1]
        
        # Apply gated attention
        gated_attention = tf.expand_dims(attn_weights, -1) * gate_weights  # [B, N, 1]
        
        # Normalize gated attention
        normalizer = tf.reduce_sum(gated_attention, axis=1, keepdims=True) + 1e-6
        normalized_attention = gated_attention / normalizer
        
        # Apply weighted aggregation
        aggregated = tf.squeeze(tf.matmul(
            tf.transpose(normalized_attention, [0, 2, 1]),  # [B, 1, N]
            enhanced                                        # [B, N, D]
        ), axis=1)  # [B, D]
        
        return aggregated, attn_weights


# Factory function for creating actor-critic trainer (for compatibility)
class ActorCriticTrainerTF:
    """Compatibility wrapper for RL training."""
    
    def __init__(self, frame_selector, config):
        self.frame_selector = frame_selector
        self.config = config
        
        # Only used for RL frame selectors
        if hasattr(frame_selector, 'policy_net'):
            self.actor_optimizer = None  # Will be set externally
            self.critic_optimizer = None  # Will be set externally
        
        self.reward_normalizer = RewardNormalizerTF()
    
    def setup_external_optimizers(self, actor_optimizer, critic_optimizer):
        """Allow external management of optimizers."""
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
