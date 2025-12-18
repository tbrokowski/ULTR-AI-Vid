"""
TensorFlow implementation of the MultiTask Model.
Converted from PyTorch to TensorFlow/Keras.

NOTE: The vision backbone (_init_vision_backbone, _extract_vision_features) requires 
significant changes because CLIP and timm are PyTorch-specific libraries.
Options for TensorFlow:
1. Use TensorFlow Hub for pre-trained vision models
2. Use HuggingFace TFCLIPVisionModel for CLIP
3. Use tf.keras.applications for standard backbones

The current implementation provides a placeholder that you should customize.
"""

import tensorflow as tf
import os
import logging

from ultr_ai.tf.network_architecture.components.general_components import (
    PathologyModuleTF, SiteIntegrationModuleTF, DeepAttentionMILTF, FrameSelectionAgentTF
)

logger = logging.getLogger(__name__)

# Optional TensorFlow Hub support
try:
    import tensorflow_hub as hub
    _HAS_TF_HUB = True
except Exception:
    hub = None
    _HAS_TF_HUB = False


class MultiTaskModelTF(tf.keras.Model):
    """
    TensorFlow implementation of the MultiTask model for TB classification.
    
    This model:
    - Keeps the original TB classification + pathology detection architecture
    - Returns task_logits dict instead of tb_logits for compatibility
    - Maintains all original functionality
    - Works with the new training structure
    
    NOTE: Vision backbone initialization requires customization for your specific
    use case. See _init_vision_backbone() for details.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 21)
        
        # For compatibility with new training system
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = getattr(config, 'selection_strategy', 'RL')
        
        logger.info(f"MultiTaskModelTF configured for tasks: {self.active_tasks}")
        logger.info(f"Using pathology loss: {self.use_pathology_loss}")
        logger.info(f"Frame selection strategy: {self.selection_strategy}")
        
        # Log attention parameters for debugging
        attention_temp = getattr(config, 'attention_temperature', 0.5)
        entropy_w = getattr(config, 'entropy_weight', 0.001)
        logger.info(f"Attention temperature: {attention_temp} (lower=sharper, higher=softer)")
        logger.info(f"Entropy regularization weight: {entropy_w}")
        
        # Vision Backbone configuration
        self.backbone = getattr(config, 'backbone', 'clip')
        self.backbone_model_name = getattr(config, 'backbone_model_name', 'openai/clip-vit-base-patch32')
        self.freeze_backbone = getattr(config, 'freeze_backbone', False)
        self.pretrained = getattr(config, 'pretrained', True)
        self.backbone_image_size = getattr(config, 'backbone_image_size', 224)

        self._init_vision_backbone()
        self.feature_noise_std = 0.05
        
        # Enhanced RL-based frame selection
        self.frame_selector = FrameSelectionAgentTF(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            num_frame_features=16,
            min_temperature=getattr(config, 'temperature_min', 0.1),
            max_temperature=getattr(config, 'temperature_max', 5.0),
            temperature_decay=getattr(config, 'temperature_decay', 0.995),
            entropy_weight=getattr(config, 'entropy_weight', 0.01),
            use_frame_history=getattr(config, 'use_frame_history', True),
        )
        logger.info(f"Using FrameSelectionAgentTF for {self.selection_strategy} strategy")
        
        # Pathology modules
        self.pathology_names = [
            'a_lines',
            'large_consolidation',
            'pleural_effusion',
            'other_pathology'
        ]
        
        if self.use_pathology_loss:
            self.pathology_modules = [
                PathologyModuleTF(
                    feature_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim // 2,
                    dropout=self.dropout_rate,
                    name=name
                ) for name in self.pathology_names
            ]
        else:
            self.pathology_modules = None
        
        # Site integration
        if self.use_pathology_loss:
            self.site_integration = SiteIntegrationModuleTF(
                feature_dim=self.hidden_dim,
                site_embed_dim=256,
                hidden_dim=self.hidden_dim,
                num_sites=self.num_sites,
                num_pathologies=self.num_pathologies,
                dropout=self.dropout_rate
            )
        else:
            # Simple site integration without pathology
            self.site_integration_simple = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden_dim),
                tf.keras.layers.LayerNormalization(epsilon=1e-05),
                tf.keras.layers.Activation('gelu'),
                tf.keras.layers.Dropout(self.dropout_rate)
            ], name='site_integration_simple')
        
        # Patient-level MIL
        self.patient_mil = DeepAttentionMILTF(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
            num_heads=8
        )
        
        # Task classifiers - using dict
        self.task_classifiers = {}
        for task_name in self.active_tasks:
            task_key = task_name.replace(' ', '_').replace('Label', 'label')
            self.task_classifiers[task_key] = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden_dim // 2),
                tf.keras.layers.LayerNormalization(epsilon=1e-05),
                tf.keras.layers.Activation('gelu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_classes)
            ], name=f'classifier_{task_key}')
        
        # Keep the original TB classifier for backward compatibility
        self.tb_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim // 2),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_classes)
        ], name='tb_classifier')

    def build(self, input_shape):
        """Build all sublayers. Called automatically on first call."""
        # Build is handled by Keras when sublayers are called
        # This method exists to satisfy Keras build requirements
        super().build(input_shape)

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
            from ultr_ai.convert.utils import load_clip_weights_from_safetensors_to_tf
            from transformers import TFCLIPVisionModel

            self.vision_encoder = TFCLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                from_pt=True,
                # dtype=tf.float32
            )
            # CLIP ViT-B/32 has 768-d pooler_output
            self.vision_dim = getattr(self.vision_encoder.config, 'hidden_size', 768)
            self._vision_kind = 'clip'
            # Optional: local weights loading using the new mapping
            local_weights_path = os.path.join(
                'ultr_ai/network_architecture/CLIP_weights',
                'model.safetensors'
            )
            # TEMPORARILY DISABLED: Skip local weights loading to debug shape issue
            if os.path.exists(local_weights_path):
                logger.info(f"Loading CLIP weights from {local_weights_path}")
                try:
                    matched, total, unmatched = load_clip_weights_from_safetensors_to_tf(
                        local_weights_path, 
                        self.vision_encoder, 
                        num_layers=12
                    )
                    logger.info(f"Successfully matched {matched}/{total} CLIP weights")
                    if unmatched:
                        logger.info(f"Unmatched weights ({len(unmatched)}):")
                        for msg in unmatched[:10]:  # Print first 10
                            logger.info(f"  - {msg}")
                        if len(unmatched) > 10:
                            logger.info(f"  ... and {len(unmatched) - 10} more")
                except Exception as e:
                    logger.warning(f"Failed to load local CLIP weights: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.info("No local CLIP weights file found, using default pretrained weights")

        else:
            raise NotImplementedError(f"Please implement vision {self.backbone} backbone initialization for TensorFlow/Keras.")
                      
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
            raise NotImplementedError(f"Backbone {self._vision_kind} freezing not implemented for this vision kind.")

    def _extract_vision_features(self, frames, training=False):
        """
        Generic feature extractor → returns [B, T, D] where D=self.vision_dim.
        
        Args:
            frames: Video frames [B, T, C, H, W] or [B, T, H, W, C]
        """
        # Get shape
        shape = tf.shape(frames)
        batch_size, num_frames = shape[0], shape[1]
        
        # Determine input format
        # Check static shape if available
        static_shape = frames.shape.as_list()
        
        # Detect if input is channels-first [B, T, C, H, W] or channels-last [B, T, H, W, C]
        is_channels_first = (static_shape[2] == 3) or (static_shape[-1] != 3 and static_shape[2] is not None)
        
        if self._vision_kind == 'clip':
            # HuggingFace TF CLIP expects NCHW format (channels first): [B, C, H, W]
            if is_channels_first:
                # Input is [B, T, C, H, W] - already channels first
                # Reshape to [B*T, C, H, W]
                x = tf.reshape(frames, [batch_size * num_frames, static_shape[2], static_shape[3], static_shape[4]])
            else:
                # Input is [B, T, H, W, C] - transpose to [B, T, C, H, W] first
                frames = tf.transpose(frames, [0, 1, 4, 2, 3])
                frame_shape = tf.shape(frames)
                x = tf.reshape(frames, [batch_size * num_frames, frame_shape[2], frame_shape[3], frame_shape[4]])
            
            outputs = self.vision_encoder(pixel_values=x, training=training)
            feats = outputs.pooler_output  # [B*T, D]
        else:
            # Other backbones expect NHWC format (channels last)
            if is_channels_first:
                # Input is [B, T, C, H, W] - transpose to [B, T, H, W, C]
                frames = tf.transpose(frames, [0, 1, 3, 4, 2])
            # Reshape to [B*T, H, W, C]
            frame_shape = tf.shape(frames)
            x = tf.reshape(frames, [batch_size * num_frames, frame_shape[2], frame_shape[3], frame_shape[4]])
            feats = self.vision_encoder(x, training=training)  # [B*T, D]

        return tf.reshape(feats, [batch_size, num_frames, -1])

    def extract_clip_features(self, frames, training=False):
        """Backward compatibility: keep method name but delegate to generic extractor."""
        return self._extract_vision_features(frames, training=training)
    
    def process_site(self, video, site_idx, mask=None, batch_idx=None, site_pos=None, training=False):
        """
        Process a single site's video using soft attention aggregation.
        
        Args:
            video: Video frames [1, T, H, W, C] or [1, T, C, H, W]
            site_idx: Site index
            mask: Frame mask [1, T]
            batch_idx: Batch index for tracking
            site_pos: Site position for tracking
            training: Whether in training mode
            
        Returns:
            Dictionary with:
                - selected_features: Aggregated site features [1, hidden_dim]
                - pathology_scores: Pathology predictions [1, num_pathologies] if enabled
                - action_logits: Frame attention logits [1, T]
                - state_values: Value estimates for RL
        """
        # Extract vision features
        clip_features = self._extract_vision_features(video, training=training)  # [1, T, vision_dim]
        
        # Select key frames using enhanced frame selector
        action_logits, state_values, enhanced_features = self.frame_selector(
            clip_features, mask=mask, batch_idxs=batch_idx, site_idxs=site_pos, training=training
        )
        
        # Sample actions (frame indices) - kept for RL compatibility but not used for selection
        actions, _ = self.frame_selector.select_action(
            action_logits, state_values, enhanced_features, batch_idx, site_pos, training=training
        )
        
        # Use SOFT attention-based selection with temperature-controlled sharpness
        tau = getattr(self.config, 'attention_temperature', 0.5)
        
        if mask is not None:
            # Apply mask to logits
            valid_mask = mask[0]
            masked_logits = tf.identity(action_logits[0])
            mask_value = -1e9
            masked_logits = tf.where(
                tf.cast(valid_mask, tf.bool),
                masked_logits,
                tf.ones_like(masked_logits) * mask_value
            )
            
            # Compute attention weights with temperature
            attention_weights = tf.nn.softmax(masked_logits / tau, axis=0)
            
            # Renormalize after masking
            attention_weights = attention_weights * tf.cast(valid_mask, tf.float32)
            attention_weights = attention_weights / (tf.reduce_sum(attention_weights) + 1e-9)
        else:
            attention_weights = tf.nn.softmax(action_logits[0] / tau, axis=0)
        
        # Weighted sum of features (differentiable)
        selected_features = tf.reduce_sum(
            tf.expand_dims(attention_weights, -1) * enhanced_features[0],  # [T, hidden_dim]
            axis=0
        )  # [hidden_dim]
        
        # Add batch dimension: [1, hidden_dim]
        selected_features = tf.expand_dims(selected_features, 0)
        
        # Add sequence dimension for pathology modules: [1, 1, hidden_dim]
        selected_features_with_seq = tf.expand_dims(selected_features, 1)
        
        # Mask for single aggregated feature
        selected_mask = tf.ones((1, 1), dtype=tf.bool)
        
        # Process pathologies (if enabled)
        pathology_scores = None
        if self.use_pathology_loss and self.pathology_modules is not None:
            pathology_scores_list = []
            
            for module in self.pathology_modules:
                score, attention, features = module(selected_features_with_seq, selected_mask, training=training)
                pathology_scores_list.append(score)
            
            # Stack pathology outputs
            pathology_scores = tf.concat(pathology_scores_list, axis=1)  # [1, num_pathologies]
        
        return {
            'selected_features': selected_features,
            'selected_indices': None,  # No longer using hard indices
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos
        }

    def process_patient(self, site_videos, site_indices, site_masks, training=False):
        """
        Process videos from multiple anatomical sites for a patient.
        
        Args:
            site_videos: Videos from different sites [B, N, T, C, H, W] (channels-first, same as PyTorch)
                        or [B, N, T, H, W, C] (channels-last, TF native)
            site_indices: Anatomical site indices [B, N]
            site_masks: Site masks [B, N]
        """
        # Use static shape for batch_size and max_sites to allow Python loops
        static_shape = site_videos.shape.as_list()
        batch_size = static_shape[0] if static_shape[0] is not None else tf.shape(site_videos)[0]
        max_sites = static_shape[1] if static_shape[1] is not None else tf.shape(site_videos)[1]
        
        all_site_features = []
        all_pathology_scores = []
        all_site_metadata = []
        
        # Process each patient
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            site_metadata = []
            
            # Process each valid site - use max_sites and check validity inside loop
            for n in range(max_sites):
                # Check if this site is valid
                is_valid = site_masks[b, n]
                
                # Get video and site index
                video = tf.expand_dims(site_videos[b, n], 0)  # [1, T, H, W, C]
                site_idx = site_indices[b, n]  # Keep as tensor, don't convert to int
                
                # Create frame masks (all valid initially)
                num_frames = tf.shape(video)[1]
                frame_mask = tf.ones((1, num_frames), dtype=tf.bool)
                
                # Process site
                site_output = self.process_site(
                    video, site_idx, frame_mask, batch_idx=b, site_pos=n, training=training
                )
                
                # Get selected features
                selected_features = site_output['selected_features']  # [1, hidden_dim]
                
                # Zero out features for invalid sites
                selected_features = tf.where(
                    tf.cast(is_valid, tf.bool),
                    selected_features,
                    tf.zeros_like(selected_features)
                )
                site_features.append(selected_features)
                
                if self.use_pathology_loss and site_output['pathology_scores'] is not None:
                    pathology_scores = tf.where(
                        tf.cast(is_valid, tf.bool),
                        site_output['pathology_scores'],
                        tf.zeros_like(site_output['pathology_scores'])
                    )
                    site_pathology_scores.append(pathology_scores)
                
                # Store site metadata
                site_metadata.append({
                    'batch_idx': b,
                    'site_idx': n,
                    'selected_indices': site_output['selected_indices'],
                    'action_logits': site_output['action_logits'],
                    'state_values': site_output['state_values']
                })
            
            # Stack outputs for this patient - all sites already processed with masking
            if site_features:
                site_features_stacked = tf.concat(site_features, axis=0)  # [max_sites, hidden_dim]
                all_site_features.append(site_features_stacked)
                
                if self.use_pathology_loss:
                    if site_pathology_scores:
                        site_pathology_stacked = tf.concat(site_pathology_scores, axis=0)
                        all_pathology_scores.append(site_pathology_stacked)
                    else:
                        all_pathology_scores.append(tf.zeros((max_sites, self.num_pathologies)))
                
                all_site_metadata.append(site_metadata)
            else:
                # No valid sites
                all_site_features.append(tf.zeros((max_sites, self.hidden_dim)))
                if self.use_pathology_loss:
                    all_pathology_scores.append(tf.zeros((max_sites, self.num_pathologies)))
                all_site_metadata.append([])
        
        # Stack across batch
        all_site_features = tf.stack(all_site_features)  # [B, N, hidden_dim]
        
        if self.use_pathology_loss:
            all_pathology_scores = tf.stack(all_pathology_scores)  # [B, N, num_pathologies]
        else:
            all_pathology_scores = None
        
        return all_site_features, all_pathology_scores, all_site_metadata
    
    def call(self, inputs, training=False):
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary containing:
                - patient_ids: List of patient IDs (optional, not used in forward pass)
                - tb_labels: TB labels [B] (optional, used for loss computation)
                - pneumonia_labels: Pneumonia labels [B] (optional, used for loss computation)
                - covid_labels: COVID labels [B] (optional, used for loss computation)
                - site_indices: Anatomical site indices [B, N]
                - site_counts: Number of valid sites per patient [B] (optional)
                - site_videos: Videos from different sites [B, N, T, C, H, W] (channels-first, same as PyTorch)
                - site_images: Representative images [B, N, C, H, W] (optional)
                - site_findings: Site findings [B, N, num_pathologies] (optional)
                - site_masks: Site masks [B, N]
                - batch_padding_masks: Batch padding masks [B, N] (optional)
                - real_data_masks: Real data masks [B, N] (optional)
                - _mask_type: Type of masking used (optional)
        """
        # Extract required inputs
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        # Process all sites for all patients
        site_features, pathology_scores, site_metadata = self.process_patient(
            site_videos, site_indices, site_masks, training=training
        )
        
        # Integrate site features with anatomical context
        if self.use_pathology_loss:
            integrated_features = self.site_integration(
                site_features, site_indices, pathology_scores, training=training
            )
        else:
            integrated_features = self.site_integration_simple(site_features, training=training)
        
        # Apply patient-level MIL
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks, training=training)
        
        if training and hasattr(self, 'feature_noise_std'):
            noise = tf.random.normal(tf.shape(patient_features)) * self.feature_noise_std
            patient_features = patient_features + noise
        
        # Multi-task classification
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                tb_logits = self.tb_classifier(patient_features, training=training)
                if self.num_classes == 1:
                    tb_logits = tf.squeeze(tb_logits, axis=-1)
                task_logits['TB Label'] = tb_logits
            else:
                task_key = task_name.replace(' ', '_').replace('Label', 'label')
                if task_key in self.task_classifiers:
                    logits = self.task_classifiers[task_key](patient_features, training=training)
                    if self.num_classes == 1:
                        logits = tf.squeeze(logits, axis=-1)
                    task_logits[task_name] = logits

        # Calculate patient-level pathology scores using MIL attention (if enabled)
        patient_pathology_scores = None
        if self.use_pathology_loss and pathology_scores is not None:
            patient_pathology_scores = tf.squeeze(tf.matmul(
                tf.expand_dims(mil_attention, 1),  # [B, 1, N]
                pathology_scores  # [B, N, num_pathologies]
            ), axis=1)  # [B, num_pathologies]
        
        # Flatten site_metadata into site_outputs for entropy loss computation
        site_outputs = []
        for patient_sites in site_metadata:
            site_outputs.extend(patient_sites)
        
        # Return comprehensive output
        output = {
            'task_logits': task_logits,
            'patient_pathology_scores': patient_pathology_scores,
            'patient_features': patient_features,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_metadata': site_metadata,
            'site_outputs': site_outputs
        }
        
        # Keep backward compatibility
        if 'TB Label' in task_logits:
            output['tb_logits'] = task_logits['TB Label']
        
        return output
    
    def compute_losses(self, outputs, targets, pos_weights=None, training=True):
        """Compute losses for training."""
        loss_dict = {}
        total_loss = 0.0
        
        # Default positive weights
        if pos_weights is None:
            pos_weights = {'TB Label': 1.4}
        
        # Task classification losses
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
                if not tf.reduce_any(valid_mask):
                    continue
                
                logits = outputs['task_logits'][task_name]
                
                # Get positive weight for this task
                pos_weight = pos_weights.get(task_name, 2.0)
                
                # Binary cross-entropy loss with logits
                valid_logits = tf.boolean_mask(logits, valid_mask)
                valid_labels = tf.boolean_mask(target_labels, valid_mask)
                
                # Manual BCE with pos_weight
                bce = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(valid_labels, tf.float32),
                    logits=valid_logits
                )
                # Apply pos_weight: weight positive examples more
                weight = tf.where(
                    tf.cast(valid_labels, tf.bool),
                    tf.ones_like(bce) * pos_weight,
                    tf.ones_like(bce)
                )
                task_loss = tf.reduce_mean(bce * weight)
                
                # Apply task weight
                task_weight = self.task_weights.get(task_name, 1.0)
                weighted_task_loss = task_loss * task_weight
                
                loss_dict[f'{task_name}_loss'] = float(task_loss)
                total_loss = total_loss + weighted_task_loss
        
        # Add entropy regularization
        if 'site_outputs' in outputs and training:
            entropy_loss = 0.0
            num_sites = 0
            
            tau = getattr(self.config, 'attention_temperature', 0.5)
            
            for site_output in outputs['site_outputs']:
                if 'action_logits' in site_output and site_output['action_logits'] is not None:
                    action_logits = site_output['action_logits']
                    
                    # Apply temperature scaling
                    scaled_logits = action_logits / tau
                    
                    # Compute attention weights
                    attention_probs = tf.nn.softmax(scaled_logits, axis=-1)
                    
                    # Compute entropy
                    log_probs = tf.nn.log_softmax(scaled_logits, axis=-1)
                    entropy = -tf.reduce_sum(attention_probs * log_probs, axis=-1)
                    entropy = tf.reduce_mean(entropy)
                    
                    entropy_loss = entropy_loss + entropy
                    num_sites += 1
            
            if num_sites > 0:
                avg_entropy = entropy_loss / num_sites
                entropy_weight = getattr(self.config, 'entropy_weight', 0.001)
                entropy_penalty = entropy_weight * avg_entropy
                total_loss = total_loss + entropy_penalty
                
                loss_dict['attention_entropy'] = float(avg_entropy)
                loss_dict['attention_entropy_penalty'] = float(entropy_penalty)
        
        # Pathology detection loss
        if self.use_pathology_loss and 'pathology_scores' in outputs and outputs['pathology_scores'] is not None:
            pathology_scores = outputs['pathology_scores']
            pathology_labels = targets['pathology_labels']
            
            for i in range(self.num_pathologies):
                if len(pathology_scores.shape) == 3:  # [B, N, num_pathologies]
                    path_score_i = pathology_scores[:, :, i]
                    path_label_i = pathology_labels[:, :, i]
                else:
                    path_score_i = pathology_scores[:, i]
                    path_label_i = pathology_labels[:, i]
                
                # Valid mask
                valid_mask = path_label_i >= 0
                
                if tf.reduce_any(valid_mask):
                    pos_weights_path = [1.0, 4.0, 4.0, 4.0, 15.0]
                    pos_weight = pos_weights_path[i % len(pos_weights_path)]
                    
                    valid_scores = tf.boolean_mask(path_score_i, valid_mask)
                    valid_labels = tf.boolean_mask(path_label_i, valid_mask)
                    
                    bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(valid_labels, tf.float32),
                        logits=valid_scores
                    )
                    weight = tf.where(
                        tf.cast(valid_labels, tf.bool),
                        tf.ones_like(bce) * pos_weight,
                        tf.ones_like(bce)
                    )
                    p_loss = tf.reduce_mean(bce * weight)
                    
                    loss_dict[f'pathology_{i}_loss'] = float(p_loss)
                    
                    pathology_weight = 0.2
                    total_loss = total_loss + pathology_weight * p_loss
        
        loss_dict['total_loss'] = float(total_loss) if isinstance(total_loss, tf.Tensor) else total_loss
        
        return total_loss, loss_dict


# For backward compatibility
TB_DRL_MODEL_TF = MultiTaskModelTF
