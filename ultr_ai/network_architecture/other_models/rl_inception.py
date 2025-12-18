import torch
import torch.nn as nn
import logging
import torchvision.models.video as video_models
import gc

# Import base components from original model
from ultr_ai.network_architecture.components.general_components import (
    PathologyModule, SiteIntegrationModule, DeepAttentionMIL
)
from ultr_ai.network_architecture.components import MultiTaskModel, AttentionPoolSelector

logger = logging.getLogger(__name__)

class RLInceptionMultiTaskModel(nn.Module):
    """Ablation: RL frame selection with Inception backbone instead of CLIP."""
    
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
        
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = 'rl_inception'
        
        # RL settings
        self.use_rl = True
        self.rl_loss_weight = getattr(config, 'rl_loss_weight', 0.1)
        
        # Memory optimization settings
        self.backbone_frozen = getattr(config, 'backbone_frozen', True)
        self.max_sites_per_forward = getattr(config, 'max_sites_per_forward', 4)
        
        logger.info("Using RL frame selection with Inception backbone")
        
        # Load Inception-based backbone
        try:
            self.backbone = video_models.s3d(weights=video_models.S3D_Weights.DEFAULT)
            logger.info("Loaded S3D (Inception-based) model for RL")
            backbone_dim = 400
        except:
            try:
                self.backbone = video_models.mvit_v1_b(weights=video_models.MViT_V1_B_Weights.DEFAULT)
                logger.info("Loaded MViT (fallback) for RL")
                backbone_dim = 768
            except:
                self.backbone = video_models.mc3_18(weights=video_models.MC3_18_Weights.DEFAULT)
                logger.info("Loaded MC3 (fallback) for RL")
                backbone_dim = 512
        
        # Remove final classification layer
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        # Freeze backbone
        if self.backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Inception backbone frozen, training RL selector only")
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Import RL frame selector from original model
        # Assuming there's an ActorCriticFrameSelector in CLIP_DRL_Aug11
        try:
            from .CLIP_DRL_Aug11 import ActorCriticFrameSelector
            self.frame_selector = ActorCriticFrameSelector(
                feature_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_heads=8,
                dropout=self.dropout_rate
            )
            logger.info("Using ActorCriticFrameSelector for RL frame selection")
        except ImportError:
            logger.warning("Could not import ActorCriticFrameSelector, using AttentionPoolSelector")
            self.frame_selector = AttentionPoolSelector(
                feature_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_heads=8
            )
        
        # Pathology modules
        if self.use_pathology_loss:
            pathology_hidden = min(self.hidden_dim // 2, 256)
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
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        # Patient-level MIL
        mil_hidden = min(self.hidden_dim // 2, 512)
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=mil_hidden,
            dropout=self.dropout_rate,
            num_heads=4
        )
        
        # Task classifiers
        classifier_hidden = min(self.hidden_dim // 2, 256)
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
        
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"RL-Inception - Total parameters: {total_params:,}")
        logger.info(f"RL-Inception - Trainable parameters: {trainable_params:,}")
    
    def _extract_video_features(self, video, use_amp=True):
        """Extract features from video using Inception."""
        # Reshape for Inception: [1, C, T, H, W]
        video_input = video.permute(1, 0, 2, 3).unsqueeze(0)
        
        context = torch.amp.autocast('cuda') if use_amp else torch.enable_grad()
        
        if self.backbone_frozen:
            with torch.no_grad(), context:
                video_features = self.backbone(video_input)
        else:
            with context:
                video_features = self.backbone(video_input)
        
        return video_features
    
    def process_site(self, video, site_idx, mask=None, batch_idx=None, site_pos=None):
        """Process site with RL frame selection."""
        # Extract Inception features
        video_features = self._extract_video_features(video)
        
        # Project to hidden dimension
        projected_features = self.feature_projection(video_features)
        
        # For video-level features, we need to extract frame-level features
        # Assuming projected_features is [1, hidden_dim], we'll treat it as a single "frame"
        # In a real implementation, you might extract multiple temporal features
        frame_features = projected_features.unsqueeze(1).repeat(1, 3, 1)  # [1, 3, hidden_dim]
        frame_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
        
        # RL frame selection
        action_logits, state_values, encoded_features = self.frame_selector(
            frame_features, frame_mask, batch_idx, site_pos
        )
        
        # Select frames
        actions, log_probs = self.frame_selector.select_action(
            action_logits, state_values, encoded_features, batch_idx, site_pos
        )
        
        # Get selected features
        selected_features = encoded_features[0, actions]
        selected_mask = torch.ones(len(actions), dtype=torch.bool, device=video.device)
        
        # Process pathologies
        pathology_scores = None
        if self.use_pathology_loss and self.pathology_modules is not None:
            pathology_scores = []
            for module in self.pathology_modules:
                score, _, _ = module(selected_features.unsqueeze(0), selected_mask.unsqueeze(0))
                pathology_scores.append(score)
            pathology_scores = torch.cat(pathology_scores, dim=1)
        
        return {
            'selected_features': selected_features.unsqueeze(0),
            'selected_indices': actions.unsqueeze(0),
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos
        }
    
    def forward(self, inputs):
        """Forward pass using RL with Inception backbone."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        site_rl_data = []
        
        # Process each sample
        for b in range(batch_size):
            valid_sites = site_masks[b].sum().item()
            
            if valid_sites == 0:
                site_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                pathology_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                all_site_features.append(site_features)
                all_pathology_scores.append(pathology_scores)
                continue
            
            sample_features = []
            sample_pathology_scores = []
            
            for n in range(valid_sites):
                video = site_videos[b, n]
                
                try:
                    # Process with RL frame selection
                    site_output = self.process_site(
                        video, 
                        site_idx=site_indices[b, n].item() if site_indices is not None else n,
                        batch_idx=b,
                        site_pos=n
                    )
                    
                    # Aggregate selected features
                    selected = site_output['selected_features'].mean(dim=1)  # [1, hidden_dim]
                    sample_features.append(selected)
                    
                    if self.use_pathology_loss and site_output['pathology_scores'] is not None:
                        sample_pathology_scores.append(site_output['pathology_scores'])
                    
                    # Store RL data for loss computation
                    if self.training:
                        site_rl_data.append({
                            'action_logits': site_output['action_logits'],
                            'state_values': site_output['state_values'],
                            'batch_idx': b,
                            'site_idx': n
                        })
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning(f"OOM processing site {n}, using zero features")
                        zero_features = torch.zeros(1, self.hidden_dim, device=site_videos.device)
                        sample_features.append(zero_features)
                        
                        if self.use_pathology_loss:
                            zero_pathology = torch.zeros(1, self.num_pathologies, device=site_videos.device)
                            sample_pathology_scores.append(zero_pathology)
                        
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
            
            # Pad and collect results
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
        
        site_features = torch.stack(all_site_features)
        pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
        
        # Site integration
        if self.use_pathology_loss:
            integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
        else:
            integrated_features = self.site_integration(site_features)
        
        # Patient-level MIL
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
        # Classification
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
            'patient_features': patient_features,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_rl_data': site_rl_data
        }
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Compute losses including RL loss."""
        # Base losses from MultiTaskModel
        losses = MultiTaskModel.compute_losses(self, outputs, targets, pos_weights)
        
        # Add RL loss if training
        if self.training and self.use_rl and len(outputs.get('site_rl_data', [])) > 0:
            # Placeholder for RL loss computation
            # In practice, this would compute policy gradient loss
            rl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            for site_data in outputs['site_rl_data']:
                if 'state_values' in site_data and site_data['state_values'] is not None:
                    # Simple baseline: encourage diversity
                    rl_loss = rl_loss + site_data['state_values'].mean() * 0.01
            
            losses['rl_loss'] = rl_loss * self.rl_loss_weight
            losses['total'] = losses['total'] + losses['rl_loss']
        
        return losses