# =============================================================================
# FRAME SELECTORS - TensorFlow Version
# =============================================================================

import tensorflow as tf


class AttentionPoolSelectorTF(tf.keras.Model):
    """Attention-pool baseline: Learned attention over frames without RL. TF version."""
    
    def __init__(self, feature_dim=768, hidden_dim=512, output_dim=512, num_heads=8, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.feature_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(0.1)
        ], name='feature_encoder')
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=0.1
        )
        
        self.attention_scorer = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim // 2),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dense(1)
        ], name='attention_scorer')
        
        self.output_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('tanh')
        ], name='output_projection')
        
        # Compatibility attributes
        self.saved_actions = []
        # NOTE: Temperature is stored for compatibility but not used in this selector
        # The actual temperature scaling is applied in the main model's process_site() method
        self.temperature = temperature
    
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
    
    def call(self, features, mask=None, batch_idxs=None, site_idxs=None, training=False):
        batch_size = tf.shape(features)[0]
        seq_len = tf.shape(features)[1]
        
        encoded = self.feature_encoder(features, training=training)
        
        # Create attention mask for MultiHeadAttention
        # TF MHA uses attention_mask where True means "attend"
        if mask is not None:
            attention_mask = tf.cast(mask, tf.float32)
            attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 1)
        else:
            attention_mask = None
        
        attended = self.attention(
            encoded, encoded, encoded,
            attention_mask=attention_mask,
            training=training
        )
        
        attention_scores = tf.squeeze(self.attention_scorer(attended, training=training), axis=-1)
        
        if mask is not None:
            # Use a safe mask value
            mask_value = -1e4
            attention_scores = tf.where(
                tf.cast(mask, tf.bool),
                attention_scores,
                tf.ones_like(attention_scores) * mask_value
            )
        
        # Compute softmax for differentiable selection
        # NOTE: Temperature is applied in the main model's process_site() method, not here
        # This frame selector outputs raw attention logits
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # [B, T]
        
        # Apply soft attention to get weighted features
        # This allows gradient to flow through the attention mechanism
        weighted_attended = tf.expand_dims(attention_weights, -1) * attended  # [B, T, hidden_dim]
        
        output_features = self.output_projection(weighted_attended, training=training)
        state_values = tf.zeros((batch_size, 1))
        
        return attention_scores, state_values, output_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None, training=False):
        """
        Select action using soft attention over all frames (single aggregated vector approach).
        
        NOTE: This method outputs raw logits and dummy actions for API compatibility.
        The actual temperature-scaled attention is computed in the main model's process_site() method.
        The frame selector just provides the raw attention scores.
        """
        batch_size = tf.shape(logits)[0]
        
        # For backward compatibility, return dummy "actions" (not actually used downstream)
        # The actual frame aggregation happens via soft attention in process_site()
        actions = tf.zeros((batch_size,), dtype=tf.int32)
        
        # Compute log probabilities (for potential RL integration)
        log_probs = tf.nn.log_softmax(logits, axis=1)
        action_log_probs = tf.gather(log_probs, actions, axis=1, batch_dims=1)
        
        return actions, action_log_probs
