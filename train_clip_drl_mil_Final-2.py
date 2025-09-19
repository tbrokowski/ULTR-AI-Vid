import os
import sys
import time
import json
import yaml
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import logging
import random
from typing import Dict, List, Optional, Tuple, Union
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, auc
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataloaders.dataset_multitask import LungUltrasoundDataModule
from NetworkArchitecture.OOMHandler import OOMHandler
# Import the updated model from the new file location
from NetworkArchitecture.CLIP_DRL_Aug11 import MultiTaskModel

# Optional monitoring utilities
try:
    from NetworkArchitecture.monitoring_utils import RLTrainingMonitor, GradientMonitor, log_model_component_status
except ImportError:
    logger.warning("Monitoring utilities not available")
    RLTrainingMonitor = None
    GradientMonitor = None
    log_model_component_status = lambda *args: None
        
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import gc
import torch.cuda

def optimize_memory():
    """Free up memory cache."""
    gc.collect()
    torch.cuda.empty_cache()


class TBTrainer:
    """TB-focused trainer using the modern multi-task architecture with single optimizer RL."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        self.best_metric = None
        self.best_epoch = 0
        self.epoch = 0
        self.epochs_without_improvement = 0

        # TB-focused configuration (using multi-task architecture but only TB task)
        self.active_tasks = ['TB Label']  # Only TB classification
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = {'TB Label': 1.0}
        self.task_pos_weights = {'TB Label': getattr(config, 'pos_weight', 2.0)}
        
        # RL configuration
        self.selection_strategy = getattr(config, 'selection_strategy', 'RL')
        if self.selection_strategy == 'RL':
            self.accumulated_rewards = []  
            self.accumulated_pathology_rewards = []
            self.rl_accumulation_step = 0  
            self.rl_accumulation_steps = getattr(self.config, 'rl_accumulation_steps', 8)
        
            self.reward_params = {
                'patient_weight': getattr(config, 'patient_weight', 1.0),
                'pathology_weight': getattr(config, 'pathology_weight', 1.0),
                'correct_factor': getattr(config, 'correct_factor', 1.5),
                'incorrect_factor': getattr(config, 'incorrect_factor', 3.0),
                'reward_scale': getattr(config, 'reward_scale', 3.0),
                'entropy_weight': getattr(config, 'entropy_weight', 0.01)
            }

        self._set_seed(config.seed)
        
        # Set up tensorboard writer
        #self.writer = SummaryWriter(os.path.join(config.log_dir, config.model_name))
        
        # Set up data and model
        self._setup_data()
        self._setup_model()
        self._setup_training()

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_data(self):
        """Set up the data module with updated structure."""
        self.data_module = LungUltrasoundDataModule(
            root_dir=self.config.root_dir,
            labels_csv=self.config.labels_csv,
            file_metadata_csv=self.config.file_metadata_csv,
            image_folder=self.config.image_folder,
            video_folder=self.config.video_folder,
            split_csv=self.config.split_csv,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            frame_sampling=self.config.frame_sampling,
            depth_filter=self.config.depth_filter,
            cache_size=100,  # Cache size for videos
            files_per_site=getattr(self.config, 'files_per_site', 1), 
            site_order=getattr(self.config, 'site_order', None),
            pad_missing_sites=getattr(self.config, 'pad_missing_sites', True),    
            max_sites=getattr(self.config, 'max_sites', 15),
        )
        
        # Setup data module for different stages
        self.data_module.setup(stage='patient_level')
        
        # Get dataloaders
        self.train_loader = self.data_module.patient_level_dataloader('train')
        self.val_loader = self.data_module.patient_level_dataloader('val')
        self.test_loader = self.data_module.patient_level_dataloader('test')
        
        logger.info(f"Training dataset size: {len(self.data_module.patient_train)}")
        logger.info(f"Validation dataset size: {len(self.data_module.patient_val)}")
        logger.info(f"Test dataset size: {len(self.data_module.patient_test)}")
    
    def _setup_model(self):
        """Set up the model configured for TB classification."""
        
        # The updated MultiTaskModel handles TB classification + pathology with the new interface
        self.model = MultiTaskModel(self.config)
        
        # Load pretrained weights if specified
        if hasattr(self.config, 'model_weights') and self.config.model_weights:
            try:
                checkpoint = torch.load(self.config.model_weights, map_location=self.config.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model_state' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {self.config.model_weights}")
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                logger.error(f"Continuing with randomly initialized weights")
        
        self.model = self.model.to(self.config.device)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {trainable_params:,}")

        # Optional monitoring setup
        if RLTrainingMonitor is not None:
            try:
                if hasattr(self, 'rl_monitor'):
                    self.rl_monitor = RLTrainingMonitor(
                        save_dir=os.path.join(self.config.experiment_dir, 'rl_monitoring'),
                        window_size=200
                    )
                if hasattr(self, 'gradient_monitor'):
                    self.gradient_monitor = GradientMonitor()
                
                log_model_component_status(self.model, logger)
            except:
                pass  

    def _setup_training(self):
        """Set up optimizers with component-specific optimization using SINGLE RL optimizer."""

        backbone_params = []      
        pathology_params = []    
        patient_pipeline_params = [] 
        task_classifier_params = []
        
        # Separate pathology parameters by module
        num_pathology_modules = len(self.model.pathology_modules) if self.model.pathology_modules else 0
        pathology_module_params = [[] for _ in range(num_pathology_modules)]
        
        for name, param in self.model.named_parameters():
            if 'vision_encoder' in name and param.requires_grad:
                backbone_params.append(param)
            elif 'multi_feature_extraction' in name or 'multi_scale_extraction' in name:
                backbone_params.append(param)
            elif 'frame_selector' in name:
                # NOTE: frame_selector params will be handled separately for RL
                continue
            elif 'pathology_modules' in name and self.use_pathology_loss:
                for i in range(num_pathology_modules):
                    if f'pathology_modules.{i}' in name or f'pathology_modules[{i}]' in name:
                        pathology_module_params[i].append(param)
                        break
            elif 'task_classifiers' in name:
                task_classifier_params.append(param)
            elif any(component in name for component in ['site_integration', 'patient_mil', 'cross_site_attention']):
                patient_pipeline_params.append(param)
        
        # Backbone optimizer
        self.backbone_optimizer = optim.AdamW(
            backbone_params,
            lr=getattr(self.config, 'backbone_lr', 0.00001),  
            weight_decay=getattr(self.config, 'backbone_weight_decay', 0.00001)
        )
        
        # Pathology optimizers (if using pathology loss)
        self.pathology_optimizers = []
        if self.use_pathology_loss:
            for i, module_params in enumerate(pathology_module_params):
                if module_params: 
                    optimizer = optim.AdamW(
                        module_params,
                        lr=getattr(self.config, 'pathology_lr', 0.0001),
                        weight_decay=getattr(self.config, 'pathology_weight_decay', 0.00001),
                    )
                    self.pathology_optimizers.append(optimizer)
            logger.info(f"Created {len(self.pathology_optimizers)} pathology optimizers")
        
        # Patient pipeline optimizer (includes task classifiers)
        all_patient_params = patient_pipeline_params + task_classifier_params
        self.patient_pipeline_optimizer = optim.AdamW(
            all_patient_params,
            lr=getattr(self.config, 'patient_pipeline_lr', 0.001),  
            weight_decay=getattr(self.config, 'patient_pipeline_weight_decay', 0.00001),
        )

        # SINGLE RL optimizer (if using RL) - FIXED APPROACH
        if self.selection_strategy == 'RL':
            frame_selector_params = []
            for name, param in self.model.named_parameters():
                if 'frame_selector' in name:
                    frame_selector_params.append(param)
            
            self.frame_selector_optimizer = optim.Adam(
                frame_selector_params,
                lr=getattr(self.config, 'rl_learning_rate', 0.0001),
                weight_decay=1e-6,
                eps=1e-8,
                betas=(0.9, 0.999)  # Standard Adam betas
            )
            
            logger.info(f"Created single frame_selector optimizer with {len(frame_selector_params)} parameters")
        
        # Set up schedulers
        self.schedulers = []
        batches_per_epoch = len(self.train_loader) if hasattr(self, 'train_loader') else 100  # fallback
        total_steps = self.config.num_epochs * batches_per_epoch
        
        # Backbone scheduler
        self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
            self.backbone_optimizer,
            T_max=total_steps,
            eta_min=getattr(self.config, 'backbone_eta_min', 1e-6),
        ))
        
        # Pathology schedulers - smooth cosine decay
        for optimizer in self.pathology_optimizers:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'pathology_eta_min', 1e-6),
            ))
    
        # Patient pipeline scheduler - smooth cosine decay
        self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
            self.patient_pipeline_optimizer,
            T_max=total_steps,
            eta_min=getattr(self.config, 'patient_pipeline_eta_min', 1e-6),
        ))
        
        # Frame selector scheduler (if using RL) - smooth cosine decay
        if self.selection_strategy == 'RL':
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.frame_selector_optimizer,
                T_max=total_steps,
                eta_min=5e-6,
            ))
        
        # Mixed precision training
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.backbone_scaler = torch.amp.GradScaler()
            self.pathology_scalers = [torch.amp.GradScaler() for p in self.pathology_optimizers]
            self.patient_pipeline_scaler = torch.amp.GradScaler()
            
            # Add frame_selector scaler for RL
            if self.selection_strategy == 'RL':
                self.frame_selector_scaler = torch.amp.GradScaler()

        # Load optimizer states if resuming
        if hasattr(self.config, 'model_weights') and self.config.model_weights and not getattr(self.config, 'reset_optimizers', True):
            try:
                checkpoint = torch.load(self.config.model_weights, map_location=self.device, weights_only=False)
                
                # Load optimizer states if present
                if 'backbone_optimizer_state_dict' in checkpoint:
                    self.backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer_state_dict'])
                    logger.info("Loaded backbone optimizer state")

                if self.selection_strategy == 'RL':
                    if 'frame_selector_optimizer_state_dict' in checkpoint:
                        self.frame_selector_optimizer.load_state_dict(checkpoint['frame_selector_optimizer_state_dict'])
                        logger.info("Loaded frame_selector optimizer state")
                     
                if 'patient_pipeline_optimizer_state_dict' in checkpoint:
                    self.patient_pipeline_optimizer.load_state_dict(checkpoint['patient_pipeline_optimizer_state_dict'])
                    logger.info("Loaded patient pipeline optimizer state")
                
                if 'pathology_optimizers_state_dicts' in checkpoint:
                    for i, (opt, state_dict) in enumerate(zip(self.pathology_optimizers, checkpoint['pathology_optimizers_state_dicts'])):
                        opt.load_state_dict(state_dict)
                        logger.info(f"Loaded pathology optimizer {i} state")
                
                if 'schedulers_state_dicts' in checkpoint:
                    for i, state_dict in enumerate(checkpoint['schedulers_state_dicts']):
                        if i < len(self.schedulers):
                            self.schedulers[i].load_state_dict(state_dict)
                    logger.info("Loaded scheduler states")
                
                # Load AMP scaler states if available
                if self.use_amp:
                    if 'backbone_scaler_state_dict' in checkpoint:
                        self.backbone_scaler.load_state_dict(checkpoint['backbone_scaler_state_dict'])
                    
                    if 'patient_pipeline_scaler_state_dict' in checkpoint:
                        self.patient_pipeline_scaler.load_state_dict(checkpoint['patient_pipeline_scaler_state_dict'])
                    
                    if 'pathology_scalers_state_dicts' in checkpoint:
                        for scaler, state_dict in zip(self.pathology_scalers, checkpoint['pathology_scalers_state_dicts']):
                            scaler.load_state_dict(state_dict)

                    if self.selection_strategy == 'RL':
                        if 'frame_selector_scaler_state_dict' in checkpoint:
                            self.frame_selector_scaler.load_state_dict(checkpoint['frame_selector_scaler_state_dict'])
                            logger.info("Loaded frame_selector scaler state")
                    
                    logger.info("Loaded AMP scaler states")
                
                # Load training state
                if 'best_metric' in checkpoint:
                    self.best_metric = checkpoint['best_metric']
                
                if 'best_epoch' in checkpoint:
                    self.best_epoch = checkpoint['best_epoch']
                
                if 'epochs_without_improvement' in checkpoint:
                    self.epochs_without_improvement = checkpoint['epochs_without_improvement']
                
                logger.info("Successfully loaded optimizer and scheduler states from checkpoint")
                
            except Exception as e:
                logger.error(f"Failed to load optimizer states: {e}")
                logger.info("Continuing with fresh optimizer states")

    def _reset_for_epoch(self):
        """Reset state for a new training epoch."""
        
        if self.selection_strategy == 'RL':
            if hasattr(self.model, 'frame_selector') and hasattr(self.model.frame_selector, 'clear_history'):
                self.model.frame_selector.clear_history()
            
            if hasattr(self.model, 'frame_selector') and hasattr(self.model.frame_selector, 'reset_rewards'):
                self.model.frame_selector.reset_rewards()

            if hasattr(self.model, 'frame_selector') and hasattr(self.model.frame_selector, 'reset_temperature'):
                self.model.frame_selector.reset_temperature()
            
            if hasattr(self.model, 'frame_selector'):
                self.model.frame_selector.saved_actions = []
        
        self.backbone_optimizer.zero_grad()
        
        if self.selection_strategy == 'RL':
            if hasattr(self, 'frame_selector_optimizer'):
                self.frame_selector_optimizer.zero_grad()
        
        self.patient_pipeline_optimizer.zero_grad()
        
        for opt in self.pathology_optimizers:
            opt.zero_grad()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.debug("Reset epoch state: cleared history, rewards, gradients, and CUDA cache")

    def _compute_site_specific_rewards(self, outputs, targets, batch_idx):
        """
        FIXED: Compute site-specific rewards with proper CUDA tensor handling.
        """
        if self.selection_strategy != 'RL':
            return []
            
        rewards = []
        
        site_rl_data = outputs.get('site_rl_data', [])
        pathology_scores = outputs.get('pathology_scores')  
        site_masks = targets.get('site_masks')
        pathology_labels = targets.get('pathology_labels') 
        
        # Get TB task logits and labels
        task_logits = outputs.get('task_logits', {})
        
        batch_size = len(site_rl_data)
        
        # Track reward components for analysis
        reward_components = {
            'patient_rewards': [],
            'pathology_rewards': [],
            'combined_rewards': []
        }
        
        for b in range(batch_size):
            patient_site_data = site_rl_data[b]
            
            with torch.no_grad():
                # FIXED: More stable patient reward computation
                patient_reward = 0.0
                
                if 'TB Label' in task_logits:
                    tb_logit = task_logits['TB Label'][b].detach()
                    tb_label = targets['tb_labels'][b].detach()
                    
                    # Skip if invalid label
                    if tb_label >= 0:
                        tb_prob = torch.sigmoid(tb_logit)
                        tb_pred = (tb_prob > 0.5).float()
                        tb_correct = (tb_pred == tb_label).float()

                        # FIXED: Simpler, more stable reward calculation
                        # Base reward: ±1.0 for correct/incorrect
                        base_reward = 2.0 * tb_correct - 1.0  # Maps {0,1} -> {-1,1}
                        
                        # Confidence bonus: 0 to 0.5 based on how far from decision boundary
                        confidence = torch.abs(tb_prob - 0.5)  # 0 to 0.5
                        confidence_multiplier = 1.0 + confidence  # 1.0 to 1.5
                        
                        # Final reward: ranges from -1.5 to +1.5
                        patient_reward = base_reward * confidence_multiplier
                        patient_reward = patient_reward.item()  # Convert to Python float
            
            for site_data in patient_site_data:
                site_idx = site_data['site_idx']
                
                site_pathology_reward = 0.0
                
                # FIXED: Simplified pathology reward (if enabled)
                if self.use_pathology_loss and pathology_scores is not None and site_idx < pathology_scores.shape[1]:
                    site_pathology_scores = pathology_scores[b, site_idx] 
                    site_pathology_labels = pathology_labels[b, site_idx]  
                    
                    valid_pathologies = 0
                    pathology_rewards_sum = 0.0
                    
                    for p in range(self.config.num_pathologies):
                        if site_pathology_labels[p] >= 0:  
                            with torch.no_grad():
                                path_prob = torch.sigmoid(site_pathology_scores[p])
                                path_pred = (path_prob > 0.5).float()
                                path_correct = (path_pred == site_pathology_labels[p]).float()
                                
                                # Simple pathology reward: ±0.5 for correct/incorrect
                                path_reward = 2.0 * path_correct - 1.0  # {-1, 1}
                                path_reward *= 0.5  # Scale to ±0.5
                                path_reward = path_reward.item()  # FIXED: Convert to Python float
                                
                                pathology_rewards_sum += path_reward
                                valid_pathologies += 1
                    
                    if valid_pathologies > 0:
                        site_pathology_reward = pathology_rewards_sum / valid_pathologies
                
                # FIXED: Combine rewards WITHOUT excessive scaling
                patient_weight = self.reward_params.get('patient_weight', 1.0)
                pathology_weight = self.reward_params.get('pathology_weight', 0.8) if self.use_pathology_loss else 0.0
                
                # NO MORE reward_scale multiplication!
                combined_reward = (
                    patient_weight * patient_reward +
                    pathology_weight * site_pathology_reward
                )
                
                # FIXED: Reasonable clamps that won't saturate (or remove clamps entirely)
                # combined_reward = np.clip(combined_reward, -3.0, 3.0)
                combined_reward = float(combined_reward)  # No clamping for now
                
                # FIXED: Track components for debugging - ensure all are Python floats
                reward_components['patient_rewards'].append(float(patient_reward))
                reward_components['pathology_rewards'].append(float(site_pathology_reward))
                reward_components['combined_rewards'].append(float(combined_reward))
                
                rewards.append({
                    'batch_idx': b,
                    'site_idx': site_idx,
                    'reward': combined_reward,
                    'patient_reward': patient_reward,
                    'pathology_reward': site_pathology_reward,
                    'valid_pathologies': valid_pathologies if self.use_pathology_loss else 0
                })
        
        # FIXED: Detailed logging for debugging - now all values are Python floats
        if batch_idx % 20 == 0 and reward_components['combined_rewards']:
            patient_rewards = reward_components['patient_rewards']
            pathology_rewards = reward_components['pathology_rewards']
            combined_rewards = reward_components['combined_rewards']
            
            logger.info(f"Reward Components - Batch {batch_idx}:")
            logger.info(f"  Patient: {np.mean(patient_rewards):.3f}±{np.std(patient_rewards):.3f} "
                    f"[{np.min(patient_rewards):.3f}, {np.max(patient_rewards):.3f}]")
            logger.info(f"  Pathology: {np.mean(pathology_rewards):.3f}±{np.std(pathology_rewards):.3f}")
            logger.info(f"  Combined: {np.mean(combined_rewards):.3f}±{np.std(combined_rewards):.3f} "
                    f"[{np.min(combined_rewards):.3f}, {np.max(combined_rewards):.3f}]")
        
        return rewards

    def _monitor_rl_training(self, rl_loss, avg_reward, epoch, batch_idx):
        """
        Monitor RL training progress and detect potential issues.
        """
        # Initialize tracking if not exists
        if not hasattr(self, 'rl_training_history'):
            self.rl_training_history = {
                'losses': [],
                'rewards': [],
                'learning_rates': [],
                'temperatures': [],
                'gradient_norms': []
            }
        
        # Record current values
        self.rl_training_history['losses'].append(rl_loss)
        self.rl_training_history['rewards'].append(avg_reward)
        
        current_lr = self.frame_selector_optimizer.param_groups[0]['lr']
        self.rl_training_history['learning_rates'].append(current_lr)
        
        if hasattr(self.model.frame_selector, 'temperature'):
            current_temp = self.model.frame_selector.temperature
            self.rl_training_history['temperatures'].append(current_temp)
        
        # Analysis every 100 batches
        if batch_idx % 100 == 0 and len(self.rl_training_history['losses']) > 20:
            recent_losses = self.rl_training_history['losses'][-20:]
            recent_rewards = self.rl_training_history['rewards'][-20:]
            
            loss_trend = np.mean(recent_losses[-10:]) - np.mean(recent_losses[:10])
            reward_trend = np.mean(recent_rewards[-10:]) - np.mean(recent_rewards[:10])
            
            logger.info(f"=== RL Training Analysis (Batch {batch_idx}) ===")
            logger.info(f"Recent Loss Trend: {loss_trend:+.4f} (negative is good)")
            logger.info(f"Recent Reward Trend: {reward_trend:+.4f} (positive is good)")
            logger.info(f"Current LR: {current_lr:.6f}")
            
            if hasattr(self.model.frame_selector, 'temperature'):
                logger.info(f"Current Temperature: {self.model.frame_selector.temperature:.4f}")
            
            # Detect potential issues
            avg_recent_loss = np.mean(recent_losses)
            if avg_recent_loss > 20.0:
                logger.warning(f"RL loss is high ({avg_recent_loss:.2f}). Consider reducing learning rate.")
            
            if loss_trend > 5.0:
                logger.warning(f"RL loss is increasing rapidly ({loss_trend:+.2f}). Check for instability.")
            
            if abs(reward_trend) < 0.01 and len(self.rl_training_history['rewards']) > 100:
                logger.warning("RL rewards are not changing. Check reward function or learning rate.")
            
            # Adaptive adjustments (optional - can be disabled)
            if getattr(self.config, 'adaptive_rl_adjustments', True):
                # Reduce learning rate if loss is exploding
                if avg_recent_loss > 30.0 and loss_trend > 10.0:
                    new_lr = current_lr * 0.5
                    for param_group in self.frame_selector_optimizer.param_groups:
                        param_group['lr'] = new_lr
                    logger.warning(f"Reduced RL learning rate to {new_lr:.6f} due to instability")
                
                # Increase temperature if rewards are stuck
                if abs(reward_trend) < 0.005 and len(self.rl_training_history['rewards']) > 50:
                    if hasattr(self.model.frame_selector, 'temperature'):
                        self.model.frame_selector.temperature = min(
                            self.model.frame_selector.temperature * 1.1, 
                            2.0
                        )
                        logger.info(f"Increased temperature to {self.model.frame_selector.temperature:.4f} for exploration")


    # FIX 1: Update _update_rl_policy to handle mixed precision properly

    def _update_rl_policy(self, inputs, targets, batch_idx):
        """Fixed RL policy update with proper mixed precision handling."""
        
        if self.selection_strategy != 'RL':
            return 0.0
        
        if not hasattr(self, 'frame_selector_optimizer'):
            return 0.0
        
        accumulation_steps = getattr(self.config, 'rl_accumulation_steps', 8)
        is_accumulation_start = batch_idx % accumulation_steps == 0
        is_accumulation_end = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader))
        
        if is_accumulation_start:
            self.frame_selector_optimizer.zero_grad()
        
        try:
            # Clear previous actions
            self.model.frame_selector.saved_actions = []
            
            # Store gradient states
            original_requires_grad = {}
            for name, param in self.model.named_parameters():
                original_requires_grad[name] = param.requires_grad
                if 'frame_selector' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            
            # Forward pass - KEEP mixed precision for forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
            
            # Compute rewards
            site_rewards = self._compute_site_specific_rewards(outputs, targets, batch_idx)
            
            if not site_rewards:
                return 0.0
            
            # Get saved actions
            saved_actions = self.model.frame_selector.saved_actions
            if not saved_actions:
                return 0.0
            
            # Match actions with rewards and collect all reward values
            action_reward_pairs = []
            all_reward_values = []
            
            reward_dict = {}
            for reward_data in site_rewards:
                key = (reward_data['batch_idx'], reward_data['site_idx'])
                reward_dict[key] = float(reward_data['reward'])
            
            for action_data in saved_actions:
                key = (action_data['batch_idx'], action_data['site_idx'])
                if key not in reward_dict:
                    continue
                
                reward_value = reward_dict[key]
                all_reward_values.append(float(reward_value))
                action_reward_pairs.append((action_data, reward_value))
            
            if not action_reward_pairs:
                return 0.0
            
            # Reward normalization
            all_rewards = np.array(all_reward_values, dtype=np.float32)
            
            if len(all_rewards) > 1:
                reward_mean = all_rewards.mean()
                reward_std = all_rewards.std() + 1e-8
                normalized_rewards = (all_rewards - reward_mean) / reward_std
                normalized_rewards = np.clip(normalized_rewards, -3.0, 3.0)
            else:
                normalized_rewards = all_rewards
                reward_mean = float(all_rewards[0])
                reward_std = 1.0
            
            # FIXED: Process rewards WITHOUT autocast to avoid dtype issues
            policy_losses = []
            value_losses = []
            advantages = []
            log_probs = []
            
            for i, (action_data, original_reward) in enumerate(action_reward_pairs):
                normalized_reward = normalized_rewards[i]
                
                log_prob = action_data.get('log_prob')
                state_value = action_data.get('state_value')
                
                if log_prob is None:
                    continue
                
                # FIXED: Ensure consistent dtypes for RL computation
                reward_tensor = torch.tensor(normalized_reward, device=log_prob.device, dtype=torch.float32)
                
                # FIXED: Proper tensor shape handling for value loss
                if state_value is not None and state_value.requires_grad:
                    # Ensure both tensors are scalars and same dtype
                    state_value_scalar = state_value.squeeze().float()  # Convert to float32
                    reward_scalar = reward_tensor.squeeze().float()     # Ensure float32
                    
                    # Both should now be scalars
                    if state_value_scalar.dim() > 0:
                        state_value_scalar = state_value_scalar.mean()
                    if reward_scalar.dim() > 0:
                        reward_scalar = reward_scalar.mean()
                    
                    advantage = reward_scalar - state_value_scalar.detach()
                    
                    # FIXED: Value loss with proper shapes
                    value_loss = F.mse_loss(state_value_scalar, reward_scalar)
                    value_losses.append(value_loss)
                else:
                    advantage = reward_tensor
                
                advantages.append(advantage.item())
                log_probs.append(log_prob.item())
                
                # Policy loss
                policy_loss = -log_prob.float() * advantage.float()  # Ensure float32
                policy_losses.append(policy_loss)
            
            if not policy_losses:
                return 0.0
            
            # FIXED: Combine losses with consistent dtypes
            total_policy_loss = torch.stack(policy_losses).mean().float()
            total_value_loss = torch.stack(value_losses).mean().float() if value_losses else torch.tensor(0.0, dtype=torch.float32, device=total_policy_loss.device)
            
            # Combined loss
            value_loss_weight = 0.5
            total_loss = total_policy_loss + value_loss_weight * total_value_loss
            total_loss = total_loss / accumulation_steps
            
            # FIXED: Logging
            if batch_idx % 20 == 0:
                advantage_mean = np.mean(advantages) if advantages else 0.0
                advantage_std = np.std(advantages) if advantages else 0.0
                log_prob_mean = np.mean(log_probs) if log_probs else 0.0
                
                logger.info(f"RL Training Debug - Batch {batch_idx}:")
                logger.info(f"  Raw Rewards: {np.mean(all_reward_values):.3f}±{np.std(all_reward_values):.3f}")
                logger.info(f"  Norm Rewards: {np.mean(normalized_rewards):.3f}±{np.std(normalized_rewards):.3f}")
                logger.info(f"  Advantages: {advantage_mean:.3f}±{advantage_std:.3f}")
                logger.info(f"  Log Probs: {log_prob_mean:.3f}")
                logger.info(f"  Policy Loss: {total_policy_loss.item():.4f}")
                logger.info(f"  Value Loss: {total_value_loss.item():.4f}")
                logger.info(f"  Total Loss: {total_loss.item() * accumulation_steps:.4f}")
            
            # FIXED: Backward pass with dtype handling
            if self.use_amp:
                # IMPORTANT: Don't use autocast for RL backward pass due to dtype issues
                # The forward pass used autocast, but RL loss computation uses float32
                self.frame_selector_scaler.scale(total_loss).backward()
                
                if is_accumulation_end:
                    self.frame_selector_scaler.unscale_(self.frame_selector_optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.frame_selector.parameters(), 
                        max_norm=getattr(self.config, 'frame_selector_max_norm', 0.5)
                    )
                    
                    if batch_idx % 20 == 0:
                        logger.info(f"  Gradient Norm: {grad_norm:.4f}")
                    
                    if grad_norm > 2.0:
                        logger.warning(f"Large RL gradients detected: {grad_norm:.3f}")
                    
                    self.frame_selector_scaler.step(self.frame_selector_optimizer)
                    self.frame_selector_scaler.update()
            else:
                total_loss.backward()
                
                if is_accumulation_end:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.frame_selector.parameters(), 
                        max_norm=getattr(self.config, 'frame_selector_max_norm', 0.5)
                    )
                    
                    if batch_idx % 20 == 0:
                        logger.info(f"  Gradient Norm: {grad_norm:.4f}")
                    
                    if grad_norm > 2.0:
                        logger.warning(f"Large RL gradients detected: {grad_norm:.3f}")
                    
                    self.frame_selector_optimizer.step()
            
            if is_accumulation_end:
                self.model.frame_selector.saved_actions = []
            
            rl_loss_value = total_loss.item() * accumulation_steps
            
            # Main logging
            if batch_idx % 20 == 0:
                original_avg_reward = np.mean(all_reward_values)
                normalized_avg_reward = np.mean(normalized_rewards)
                
                log_msg = (f'RL Update - Loss: {rl_loss_value:.4f}, '
                        f'Raw Reward: {original_avg_reward:.3f}, '
                        f'Norm Reward: {normalized_avg_reward:.3f}, '
                        f'Temp: {self.model.frame_selector.get_temperature():.3f}, '
                        f'Actions: {len(saved_actions)}, '
                        f'Sites: {len(site_rewards)}')
                
                if is_accumulation_end:
                    log_msg += " [STEP]"
                else:
                    log_msg += " [ACCUM]"
                
                logger.info(log_msg)
            
            return rl_loss_value
            
        except Exception as e:
            logger.error(f"RL update error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.frame_selector_optimizer.zero_grad()
            return 0.0
        
        finally:
            # Restore gradient states
            for name, param in self.model.named_parameters():
                if name in original_requires_grad:
                    param.requires_grad = original_requires_grad[name]
            
            if torch.cuda.is_available() and batch_idx % 4 == 0:
                torch.cuda.empty_cache()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint with TB-focused metrics."""
        save_dir = pathlib.Path(self.config.experiment_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'backbone_optimizer_state_dict': self.backbone_optimizer.state_dict(),
            'patient_pipeline_optimizer_state_dict': self.patient_pipeline_optimizer.state_dict(),
            'pathology_optimizers_state_dicts': [opt.state_dict() for opt in self.pathology_optimizers],
            'schedulers_state_dicts': [sched.state_dict() for sched in self.schedulers],
            'config': self.config.to_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'active_tasks': self.active_tasks,
            'use_pathology_loss': self.use_pathology_loss
        }
        
        # Add RL-specific state if using RL
        if self.selection_strategy == 'RL':
            checkpoint.update({
                'frame_selector_optimizer_state_dict': self.frame_selector_optimizer.state_dict(),
            })
        
        if self.use_amp:
            checkpoint.update({
                'backbone_scaler_state_dict': self.backbone_scaler.state_dict(),
                'patient_pipeline_scaler_state_dict': self.patient_pipeline_scaler.state_dict(),
                'pathology_scalers_state_dicts': [scaler.state_dict() for scaler in self.pathology_scalers],
            })
            
            if self.selection_strategy == 'RL':
                checkpoint['frame_selector_scaler_state_dict'] = self.frame_selector_scaler.state_dict()
        
        latest_path = save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Use TB Label metrics for checkpoint naming
        eval_metric_key = f"TB Label_{self.config.eval_metric}"
        
        if epoch % 5 == 0 and eval_metric_key in metrics:
            metric_value = metrics[eval_metric_key]
            epoch_path = save_dir / f"checkpoint_epoch_{epoch:03d}_metric_{metric_value:.4f}.pth"
            torch.save(checkpoint, epoch_path)
        
        if is_best:
            metric_value = metrics[eval_metric_key]
            best_path = save_dir / f"checkpoint_best_metric_{metric_value:.4f}.pth"
            torch.save(checkpoint, best_path)
            
            best_generic_path = save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_generic_path)
            
            logger.info(f"New best model saved to {best_path}")
        
        return latest_path

    def save_rl_training_history(self, epoch):
        """Save RL training history for analysis."""
        if hasattr(self, 'rl_training_history'):
            import json
            import os
            
            history_file = os.path.join(self.config.experiment_dir, f'rl_history_epoch_{epoch}.json')
            
            # Convert numpy arrays to lists for JSON serialization
            history_to_save = {}
            for key, values in self.rl_training_history.items():
                if isinstance(values, list):
                    history_to_save[key] = [float(v) if not isinstance(v, (list, dict)) else v for v in values]
                else:
                    history_to_save[key] = values
            
            with open(history_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
            
            logger.info(f"Saved RL training history to {history_file}")


    def train_epoch(self, epoch):
        """
        TB-focused training epoch using the modern training structure.
        """
        self.model.train()
        self.epoch = epoch
        
        if self.selection_strategy == 'RL' and hasattr(self.model.frame_selector, 'update_temperature'):
            decay_rate = getattr(self.config, 'temperature_decay', 0.995)
            new_temp = self.model.frame_selector.update_temperature(decay=decay_rate)
            logger.info(f"Frame selection temperature: {new_temp:.4f} (epoch {epoch+1}) [decay: {decay_rate}]")
            
            # ADDED: Prevent temperature from getting too low too fast
            if new_temp < 0.3 and epoch < 5:  # Keep higher temp for first 5 epochs
                self.model.frame_selector.temperature = max(new_temp, 0.5)
                logger.info(f"Temperature adjusted to: {self.model.frame_selector.temperature:.4f} (early training)")


        # Reset model state
        self._reset_for_epoch()
        
        # Initialize tracking metrics
        running_losses = {
            'total': 0.0,
            'tb_loss': 0.0,
            'rl': 0.0
        }
        
        if self.use_pathology_loss:
            running_losses['pathology'] = 0.0
        
        # Metrics tracking for TB only
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []
        
        # Optimization parameters
        accumulation_steps = self.config.accumulation_steps
        
        data_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        batch_iter = iter(self.train_loader)
        try:
            first_batch = next(batch_iter)
            if data_stream:
                with torch.cuda.stream(data_stream):
                    # Prefetch data to GPU
                    site_videos = first_batch['site_videos'].to(self.device, non_blocking=True)
                    site_indices = first_batch['site_indices'].to(self.device, non_blocking=True)
                    site_masks = first_batch['site_masks'].to(self.device, non_blocking=True)
                    site_findings = first_batch['site_findings'].to(self.device, non_blocking=True)
                    
                    # TB labels (only TB task)
                    tb_labels = first_batch['tb_labels'].to(self.device, non_blocking=True).float()
                    # We need to create dummy labels for the multi-task model
                    pneumonia_labels = torch.full_like(tb_labels, -1)  # Invalid labels
                    covid_labels = torch.full_like(tb_labels, -1)  # Invalid labels
            else:
                # Standard data transfer
                site_videos = first_batch['site_videos'].to(self.device)
                site_indices = first_batch['site_indices'].to(self.device)
                site_masks = first_batch['site_masks'].to(self.device)
                site_findings = first_batch['site_findings'].to(self.device)
                
                tb_labels = first_batch['tb_labels'].to(self.device).float()
                pneumonia_labels = torch.full_like(tb_labels, -1)
                covid_labels = torch.full_like(tb_labels, -1)
                
        except StopIteration:
            progress_bar.close()
            return 0.0, {}
        
        # Process batches
        for batch_idx in range(len(self.train_loader)):
            try:
                next_batch = None
                if batch_idx + 1 < len(self.train_loader):
                    try:
                        next_batch = next(batch_iter)
                        if data_stream:
                            with torch.cuda.stream(data_stream):
                                # Prefetch next batch
                                next_site_videos = next_batch['site_videos'].to(self.device, non_blocking=True)
                                next_site_indices = next_batch['site_indices'].to(self.device, non_blocking=True)
                                next_site_masks = next_batch['site_masks'].to(self.device, non_blocking=True)
                                next_site_findings = next_batch['site_findings'].to(self.device, non_blocking=True)
                                
                                next_tb_labels = next_batch['tb_labels'].to(self.device, non_blocking=True).float()
                                next_pneumonia_labels = torch.full_like(next_tb_labels, -1)
                                next_covid_labels = torch.full_like(next_tb_labels, -1)
                    except StopIteration:
                        next_batch = None
                
                if data_stream:
                    torch.cuda.current_stream().wait_stream(data_stream)
                
                # Skip very large batches
                if site_videos.shape[1] > 55:
                    if next_batch is not None:
                        site_videos = next_site_videos
                        site_indices = next_site_indices
                        site_masks = next_site_masks
                        site_findings = next_site_findings
                        tb_labels = next_tb_labels
                        pneumonia_labels = next_pneumonia_labels
                        covid_labels = next_covid_labels
                    continue
                
                # Prepare inputs and targets
                inputs = {
                    'site_videos': site_videos,
                    'site_indices': site_indices,
                    'site_masks': site_masks,
                    'site_findings': site_findings,
                    'is_patient_level': True
                }
                
                targets = {
                    'tb_labels': tb_labels,
                    'pneumonia_labels': pneumonia_labels,  # Dummy invalid labels
                    'covid_labels': covid_labels,  # Dummy invalid labels
                    'pathology_labels': site_findings,
                    'site_masks': site_masks,
                }

                #==================================================
                # 1. RL Frame Selector Update (if using RL)
                #==================================================
                rl_loss_value = self._update_rl_policy(inputs, targets, batch_idx)
                if rl_loss_value > 0:
                    running_losses['rl'] += rl_loss_value
                
                if torch.cuda.is_available() and batch_idx % 4 == 0:
                    torch.cuda.empty_cache()

                #==================================================
                # 2. Pathology Modules Update (if enabled)
                #==================================================
                if self.use_pathology_loss:
                    for path_idx in range(self.config.num_pathologies):
                        if path_idx >= len(self.pathology_optimizers):
                            continue

                        for param in self.model.parameters():
                            param.requires_grad = False
                        
                        for name, param in self.model.named_parameters():
                            if f'pathology_modules.{path_idx}' in name or f'pathology_modules[{path_idx}]' in name:
                                param.requires_grad = True

                        if batch_idx % accumulation_steps == 0:
                            self.pathology_optimizers[path_idx].zero_grad()
                        
                        try:
                            if self.use_amp:
                                with torch.amp.autocast('cuda'):
                                    path_outputs = self.model(inputs)
                                    path_scores = path_outputs['pathology_scores']
                                    
                                    if path_scores.dim() == 3:
                                        path_score_i = path_scores[:, :, path_idx]
                                        path_label_i = site_findings[:, :, path_idx] if site_findings.dim() == 3 else site_findings[:, path_idx]
                                    else:
                                        path_score_i = path_scores[:, path_idx]
                                        path_label_i = site_findings[:, path_idx]
                                    
                                    valid_mask = path_label_i >= 0
                                    if valid_mask.any():
                                        pos_weight = getattr(self.config, 'pathology_pos_weights', [2.0, 4.0, 3.0])[path_idx] if hasattr(self.config, 'pathology_pos_weights') else 2.0
                                        pos_weight_tensor = torch.tensor(pos_weight, device=self.device)
                                        
                                        path_loss = F.binary_cross_entropy_with_logits(
                                            path_score_i[valid_mask],
                                            path_label_i[valid_mask].float(),
                                            pos_weight=pos_weight_tensor
                                        )
                                        
                                        path_loss = path_loss / accumulation_steps
                                        running_losses['pathology'] += path_loss.item() * accumulation_steps
                                        
                                        self.pathology_scalers[path_idx].scale(path_loss).backward()
                                        
                                        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                                            self.pathology_scalers[path_idx].unscale_(self.pathology_optimizers[path_idx])
                                            torch.nn.utils.clip_grad_norm_(
                                                [p for name, p in self.model.named_parameters() 
                                                 if f'pathology_modules.{path_idx}' in name and p.requires_grad], 
                                                max_norm=1.0
                                            )
                                            self.pathology_scalers[path_idx].step(self.pathology_optimizers[path_idx])
                                            self.pathology_scalers[path_idx].update()
                                    
                                    del path_outputs, path_scores, path_score_i, path_label_i
                            else:
                                # Non-AMP version
                                path_outputs = self.model(inputs)
                                path_scores = path_outputs['pathology_scores']
                                
                                if path_scores.dim() == 3:
                                    path_score_i = path_scores[:, :, path_idx]
                                    path_label_i = site_findings[:, :, path_idx] if site_findings.dim() == 3 else site_findings[:, path_idx]
                                else:
                                    path_score_i = path_scores[:, path_idx]
                                    path_label_i = site_findings[:, path_idx]
                                
                                valid_mask = path_label_i >= 0
                                if valid_mask.any():
                                    pos_weight = getattr(self.config, 'pathology_pos_weights', [2.0, 4.0, 3.0])[path_idx] if hasattr(self.config, 'pathology_pos_weights') else 2.0
                                    pos_weight_tensor = torch.tensor(pos_weight, device=self.device)
                                    
                                    path_loss = F.binary_cross_entropy_with_logits(
                                        path_score_i[valid_mask],
                                        path_label_i[valid_mask].float(),
                                        pos_weight=pos_weight_tensor
                                    )
                                    
                                    path_loss = path_loss / accumulation_steps
                                    running_losses['pathology'] += path_loss.item() * accumulation_steps
                                    path_loss.backward()
                                    
                                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                                        torch.nn.utils.clip_grad_norm_(
                                            [p for name, p in self.model.named_parameters() 
                                             if f'pathology_modules.{path_idx}' in name and p.requires_grad], 
                                            max_norm=1.0
                                        )
                                        self.pathology_optimizers[path_idx].step()
                                
                                del path_outputs, path_scores, path_score_i, path_label_i
                        
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                logger.warning(f"OOM in pathology {path_idx} update batch {batch_idx}, skipping")
                                gc.collect()
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                
                if torch.cuda.is_available() and batch_idx % 4 == 0:
                    torch.cuda.empty_cache()

                #==================================================
                # 3. TB Patient Classifier Update
                #==================================================

                for param in self.model.parameters():
                    param.requires_grad = False

                # Enable gradients for patient pipeline and task classifiers
                for name, param in self.model.named_parameters():
                    if any(component in name for component in ['site_integration', 'patient_mil', 'task_classifiers', 'cross_site_attention']):
                        param.requires_grad = True
                
                if batch_idx % accumulation_steps == 0:
                    self.patient_pipeline_optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                        
                        # Compute TB-focused loss
                        total_loss, loss_dict = self.model.compute_losses(outputs, targets, self.task_pos_weights)
                        total_loss = total_loss / accumulation_steps
                        
                        # Record TB loss
                        if 'TB Label_loss' in loss_dict:
                            running_losses['tb_loss'] += loss_dict['TB Label_loss']
                        
                        self.patient_pipeline_scaler.scale(total_loss).backward()
                        
                        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                            self.patient_pipeline_scaler.unscale_(self.patient_pipeline_optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                [p for name, p in self.model.named_parameters() 
                                 if any(component in name for component in ['site_integration', 'patient_mil', 'task_classifiers', 'cross_site_attention']) and p.requires_grad], 
                                max_norm=1.0
                            )
                            self.patient_pipeline_scaler.step(self.patient_pipeline_optimizer)
                            self.patient_pipeline_scaler.update() 
                       
                else:
                    # Non-AMP version
                    outputs = self.model(inputs)
                    
                    # Compute TB-focused loss
                    total_loss, loss_dict = self.model.compute_losses(outputs, targets, self.task_pos_weights)
                    total_loss = total_loss / accumulation_steps
                    
                    # Record TB loss
                    if 'TB Label_loss' in loss_dict:
                        running_losses['tb_loss'] += loss_dict['TB Label_loss']
                    
                    total_loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.model.named_parameters() 
                             if any(component in name for component in ['site_integration', 'patient_mil', 'task_classifiers', 'cross_site_attention']) and p.requires_grad], 
                            max_norm=1.0
                        )
                        self.patient_pipeline_optimizer.step()
                
                # Collect metrics for TB
                with torch.no_grad():
                    task_logits = outputs.get('task_logits', {})
                    
                    if 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_predictions.append(preds.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                    
                    # Collect pathology metrics if enabled
                    if self.use_pathology_loss and 'pathology_scores' in outputs:
                        path_scores = outputs['pathology_scores']
                        path_labels = targets['pathology_labels']
                        
                        if path_scores.dim() == 3:
                            B, N, P = path_scores.shape
                            path_scores = path_scores.reshape(-1, P)
                            path_labels = path_labels.reshape(-1, P)
                        
                        valid_mask = path_labels >= 0
                        
                        pathology_scores_list.append(path_scores.detach().cpu())
                        pathology_labels_list.append(path_labels.detach().cpu())
                        pathology_masks_list.append(valid_mask.detach().cpu())

                #==================================================
                # 4. VISION ENCODER UPDATE
                #==================================================
            
                initial_memory = torch.cuda.memory_allocated() / 1024**2

                for param in self.model.parameters():
                    param.requires_grad = False

                unfrozen_count = 0
                layers_to_train = [
                    'vision_encoder.vision_model.encoder.layers.11',
                    'vision_encoder.vision_model.encoder.layers.10',
                    'vision_encoder.visual_projection',
                    'output_projection'
                ]

                for name, param in self.model.named_parameters():
                    if any(layer in name for layer in layers_to_train):
                        param.requires_grad = True
                        unfrozen_count += 1

                gc.collect()
                torch.cuda.empty_cache()

                if batch_idx % accumulation_steps == 0:
                    self.backbone_optimizer.zero_grad()

                try:
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            backbone_outputs = self.model(inputs)
                            backbone_loss, _ = self.model.compute_losses(backbone_outputs, targets, self.task_pos_weights)
                            backbone_loss = backbone_loss / accumulation_steps
                            running_losses['total'] += backbone_loss.item() * accumulation_steps
                            
                            self.backbone_scaler.scale(backbone_loss).backward()
                            
                            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                                self.backbone_scaler.unscale_(self.backbone_optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in self.model.parameters() if p.requires_grad], 
                                    max_norm=0.5
                                )
                                self.backbone_scaler.step(self.backbone_optimizer)
                                self.backbone_scaler.update()
                    else:
                        backbone_outputs = self.model(inputs)
                        backbone_loss, _ = self.model.compute_losses(backbone_outputs, targets, self.task_pos_weights)
                        backbone_loss = backbone_loss / accumulation_steps
                        running_losses['total'] += backbone_loss.item() * accumulation_steps
                        
                        backbone_loss.backward()
                        
                        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad], 
                                max_norm=0.5
                            )
                            self.backbone_optimizer.step()
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning(f"OOM in backbone update batch {batch_idx}, skipping")
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        raise e

                #==================================================
                # Memory cleanup and batch swapping
                #==================================================
                
                if batch_idx % 2 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Swap batches
                if next_batch is not None:
                    site_videos = next_site_videos
                    site_indices = next_site_indices
                    site_masks = next_site_masks
                    site_findings = next_site_findings
                    tb_labels = next_tb_labels
                    pneumonia_labels = next_pneumonia_labels
                    covid_labels = next_covid_labels
                
                # Update progress bar
                if batch_idx % 2 == 0:
                    progress_bar.update(2 if batch_idx > 0 else 1)

                    def to_scalar(value):
                        if isinstance(value, torch.Tensor):
                            return value.item()
                        return value 
                
                    # Build progress display
                    progress_dict = {
                        'total_loss': to_scalar(running_losses['total'] / (batch_idx + 1)),
                        'tb_loss': to_scalar(running_losses['tb_loss'] / (batch_idx + 1)),
                    }
                    
                    if self.use_pathology_loss and 'pathology' in running_losses:
                        progress_dict['path_loss'] = to_scalar(running_losses['pathology'] / (batch_idx + 1))
                    
                    if self.selection_strategy == 'RL':
                        progress_dict['rl_loss'] = to_scalar(running_losses['rl'] / (batch_idx + 1))
                        progress_dict['temp'] = f"{self.model.frame_selector.get_temperature():.2f}"
                    
                    progress_bar.set_postfix(progress_dict)
                    
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"OOM in batch {batch_idx}, skipping")
                    
                    # Reset optimizers
                    self.backbone_optimizer.zero_grad()
                    if self.selection_strategy == 'RL':
                        self.frame_selector_optimizer.zero_grad()
                    self.patient_pipeline_optimizer.zero_grad()
                    for opt in self.pathology_optimizers:
                        opt.zero_grad()
                    
                    # Reset scalers
                    if self.use_amp:
                        self.backbone_scaler = torch.amp.GradScaler()
                        self.pathology_scalers = [torch.amp.GradScaler() for p in self.pathology_optimizers]
                        self.patient_pipeline_scaler = torch.amp.GradScaler()
                        if self.selection_strategy == 'RL':
                            self.frame_selector_scaler = torch.amp.GradScaler()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Runtime error: {e}")
                    raise e
        
        progress_bar.close()
        
        # Calculate metrics for TB
        all_metrics = {}
        
        if all_tb_targets and all_tb_predictions and all_tb_logits:
            tb_targets = torch.cat(all_tb_targets).numpy()
            tb_predictions = torch.cat(all_tb_predictions).numpy()
            tb_logits = torch.cat(all_tb_logits)
            
            tb_metrics = self._calculate_metrics(tb_targets, tb_predictions, tb_logits.numpy(), None, "TB Label")
            
            # Add TB prefix to metrics
            for key, value in tb_metrics.items():
                all_metrics[f'TB Label_{key}'] = value
        
        # Add pathology metrics if enabled
        if self.use_pathology_loss and pathology_labels_list and pathology_scores_list:
            all_path_scores = torch.cat(pathology_scores_list, dim=0)
            all_path_labels = torch.cat(pathology_labels_list, dim=0)
            all_path_masks = torch.cat(pathology_masks_list, dim=0)
            
            path_metrics = self._calculate_pathology_metrics(
                all_path_scores, all_path_labels, all_path_masks)
            
            all_metrics.update(path_metrics)
        
        # Log metrics
        logger.info(f"Train TB metrics:")
        tb_metrics = {k.replace('TB Label_', ''): v for k, v in all_metrics.items() 
                      if k.startswith('TB Label_')}
        if tb_metrics:
            logger.info(f"  TB Label: " + " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items()]))
        
        # Log pathology metrics if enabled
        if self.use_pathology_loss:
            path_metrics = {k: v for k, v in all_metrics.items() if '/' in k}
            if path_metrics:
                logger.info(f"Train Pathology metrics:")
                pathology_names = getattr(self.config, 'pathology_classes', 
                                        ['A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'])
                for name in pathology_names:
                    name_metrics = {k.split('/')[-1]: v for k, v in path_metrics.items() 
                                  if k.startswith(f'{name}/')}
                    if name_metrics:
                        logger.info(f"  {name}: " + " | ".join([f"{k}: {v:.4f}" for k, v in name_metrics.items()]))
        
        # Step all schedulers
        for scheduler in self.schedulers:
            scheduler.step()

        self.save_rl_training_history(epoch)
        
        return running_losses['total'] / max(1, len(self.train_loader)), all_metrics

    def validate(self, epoch, loader=None, split_name="val"):
        """
        TB-focused validation.
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        running_loss = 0.0

        # TB tracking
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        all_tb_probs = []
        
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []  
        
        progress_bar = tqdm(loader, desc=f"{split_name.capitalize()} Evaluation")
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Move data to device
                    site_videos = batch['site_videos'].to(self.device)
                    site_indices = batch['site_indices'].to(self.device)
                    site_masks = batch['site_masks'].to(self.device)
                    site_findings = batch['site_findings'].to(self.device)
                    
                    # TB labels (only TB task)
                    tb_labels = batch['tb_labels'].to(self.device).float()
                    # Dummy labels for multi-task model
                    pneumonia_labels = torch.full_like(tb_labels, -1)
                    covid_labels = torch.full_like(tb_labels, -1)
                    
                    # Prepare inputs
                    inputs = {
                        'site_videos': site_videos,
                        'site_indices': site_indices,
                        'site_masks': site_masks,
                        'site_findings': site_findings,
                        'is_patient_level': True
                    }

                    targets = {
                        'tb_labels': tb_labels,
                        'pneumonia_labels': pneumonia_labels,
                        'covid_labels': covid_labels,
                        'pathology_labels': site_findings
                    }
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    # Use the new compute_losses interface
                    loss, _ = self.model.compute_losses(outputs, targets, self.task_pos_weights)
                       
                    running_loss += loss.item()
                    
                    # Collect predictions for TB
                    task_logits = outputs.get('task_logits', {})
                    
                    if 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_predictions.append(preds.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                        all_tb_probs.append(probs.detach().cpu())
                    elif 'tb_logits' in outputs:  # Backward compatibility
                        logits = outputs['tb_logits']
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_predictions.append(preds.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                        all_tb_probs.append(probs.detach().cpu())

                    # Collect pathology metrics if enabled
                    if self.use_pathology_loss and 'pathology_scores' in outputs:
                        path_scores = outputs['pathology_scores']
                        path_labels = targets['pathology_labels']
                        
                        if path_scores.dim() == 3:
                            B, N, P = path_scores.shape
                            path_scores = path_scores.reshape(-1, P)
                            path_labels = path_labels.reshape(-1, P)
                        
                        valid_mask = path_labels >= 0
                        
                        pathology_scores_list.append(path_scores.detach().cpu())
                        pathology_labels_list.append(path_labels.detach().cpu())
                        pathology_masks_list.append(valid_mask.detach().cpu())
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': running_loss / (progress_bar.n + 1)
                    })
                    
                    # Clean up memory
                    del site_videos, site_indices, site_masks, site_findings, inputs
                    del tb_labels, pneumonia_labels, covid_labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.warning(f"WARNING: Out of memory during validation, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Calculate validation metrics
        val_loss = running_loss / len(loader)
        
        # Calculate metrics for TB
        all_metrics = {'loss': val_loss}
        
        if all_tb_targets and all_tb_predictions:
            tb_targets = torch.cat(all_tb_targets)
            tb_predictions = torch.cat(all_tb_predictions)
            tb_logits = torch.cat(all_tb_logits)
            tb_probs = torch.cat(all_tb_probs)
            
            tb_metrics = self._calculate_metrics(
                tb_targets.numpy(), 
                tb_predictions.numpy(), 
                tb_logits.numpy(),
                tb_probs.numpy(),
                "TB Label"
            )
            
            # Add TB prefix to metrics
            for key, value in tb_metrics.items():
                all_metrics[f'TB Label_{key}'] = value

        # Add pathology metrics if enabled
        if self.use_pathology_loss and pathology_labels_list and pathology_scores_list:
            all_path_scores = torch.cat(pathology_scores_list, dim=0)
            all_path_labels = torch.cat(pathology_labels_list, dim=0)
            all_path_masks = torch.cat(pathology_masks_list, dim=0)
            
            path_metrics = self._calculate_pathology_metrics(
                all_path_scores, all_path_labels, all_path_masks)
            
            all_metrics.update(path_metrics)

        # Log metrics
        logger.info(f"{split_name} TB metrics:")
        tb_metrics = {k.replace('TB Label_', ''): v for k, v in all_metrics.items() 
                      if k.startswith('TB Label_') and not '/' in k}
        if tb_metrics:
            logger.info(f"  TB Label: " + " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items()]))

        # Log pathology metrics if enabled
        if self.use_pathology_loss:
            path_metrics = {k: v for k, v in all_metrics.items() if '/' in k}
            if path_metrics:
                logger.info(f"{split_name} Pathology metrics:")
                pathology_names = getattr(self.config, 'pathology_classes', 
                                        ['A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'])
                for name in pathology_names:
                    name_metrics = {k.split('/')[-1]: v for k, v in path_metrics.items() 
                                  if k.startswith(f'{name}/')}
                    if name_metrics:
                        logger.info(f"  {name}: " + " | ".join([f"{k}: {v:.4f}" for k, v in name_metrics.items()]))
        
        # Print detailed metrics for TB
        if all_tb_targets:
            tb_targets = torch.cat(all_tb_targets).numpy().flatten()
            tb_predictions = torch.cat(all_tb_predictions).numpy().flatten()
            
            try:
                cm = confusion_matrix(tb_targets, tb_predictions)
                logger.info(f"\nConfusion Matrix for TB ({split_name}):\n{cm}")
                
                report = classification_report(tb_targets, tb_predictions)
                logger.info(f"\nClassification Report for TB ({split_name}):\n{report}")
            except Exception as e:
                logger.warning(f"Could not compute confusion matrix or classification report: {e}")
        
        return val_loss, all_metrics
    
    def _calculate_metrics(self, targets, predictions, logits, probs=None, task_name=""):
        """Calculate performance metrics for TB classification."""
        metrics = {}
        
        try:
            # Ensure correct shapes
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.flatten()
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            # Print diagnostic information
            diagnostic_data = {
                'target': targets.flatten(),
                'prediction': predictions.flatten()
            }
            
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                diagnostic_data['logit'] = logits.flatten()
            else:
                diagnostic_data['logit'] = logits[:, 0].flatten()
            
            if probs is not None:
                if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
                    diagnostic_data['probability'] = probs.flatten()
                else:
                    diagnostic_data['probability'] = probs[:, 0].flatten()
            
            df = pd.DataFrame(diagnostic_data)
            
            # Log sample rows for diagnostic purposes
            sample_rows = df.sample(min(5, len(df)))
            logger.info(f"\nDiagnostic sample for {task_name} (5 random rows):")
            logger.info(f"\n{sample_rows}")
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['precision'] = precision_score(targets, predictions, zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, zero_division=0)
            metrics['specificity'] = recall_score(1-targets, 1-predictions, zero_division=0)
            metrics['f1'] = f1_score(targets, predictions, zero_division=0)
            
            # Calculate AUC if probabilities are provided
            if probs is not None:
                try:
                    metrics['auc'] = roc_auc_score(targets, probs)
                    metrics['auprc'] = average_precision_score(targets, probs)
                except ValueError:
                    # Handle case with only one class
                    metrics['auc'] = 0.5
                    metrics['auprc'] = 0.5
        
        except Exception as e:
            logger.error(f"Error calculating metrics for {task_name}: {e}")
            # Provide default values
            if 'accuracy' not in metrics:
                metrics['accuracy'] = 0.0
            if self.config.eval_metric not in metrics:
                metrics[self.config.eval_metric] = 0.0
        
        return metrics

    def _calculate_pathology_metrics(self, scores, labels, masks):
        """Calculate metrics for each pathology class."""
        metrics = {}
        
        pathology_names = getattr(self.config, 'pathology_classes', [
            'A-line',
            'Large consolidations', 
            'Pleural Effusion',
            'Other Pathology'
        ])
        
        scores_np = scores.numpy()
        labels_np = labels.numpy()
        masks_np = masks.numpy()
        
        auroc_values = []
        auprc_values = []
        f1_values = []
        
        for i, name in enumerate(pathology_names):
            if i < scores_np.shape[1]:
                valid_indices = masks_np[:, i]
                
                if valid_indices.sum() > 0:
                    class_scores = scores_np[valid_indices, i]
                    class_labels = labels_np[valid_indices, i]
                    
                    if len(np.unique(class_labels)) < 2:
                        logger.info(f"Skipping {name} metrics - only one class present")
                        continue
                    
                    try:
                        class_probs = 1 / (1 + np.exp(-class_scores))
                        class_preds = (class_probs > 0.5).astype(np.float32)
                        
                        auroc = roc_auc_score(class_labels, class_probs)
                        auprc = average_precision_score(class_labels, class_probs)
                        f1 = f1_score(class_labels, class_preds, zero_division=0)
                        
                        metrics[f'{name}/auroc'] = auroc
                        metrics[f'{name}/auprc'] = auprc
                        metrics[f'{name}/f1'] = f1
                        
                        auroc_values.append(auroc)
                        auprc_values.append(auprc)
                        f1_values.append(f1)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating metrics for {name}: {e}")
        
        # Calculate macro-average metrics
        if auroc_values:
            metrics['pathology/macro_auroc'] = np.mean(auroc_values)
        if auprc_values:
            metrics['pathology/macro_auprc'] = np.mean(auprc_values)
        if f1_values:
            metrics['pathology/macro_f1'] = np.mean(f1_values)
        
        return metrics
    
    def train(self, resume_from_checkpoint=None):
        """
        Train the TB classification model.
        """
        logger.info(f"Starting TB training with task: {self.active_tasks}")
        logger.info(f"Using {self.selection_strategy} frame selection strategy")
        logger.info(f"Pathology loss enabled: {self.use_pathology_loss}")

        start_epoch = 0

        if resume_from_checkpoint:
            start_epoch = self.resume_training_from_checkpoint(resume_from_checkpoint)
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.info(f"Starting training for {self.config.num_epochs} epochs...")
            self.best_metric = float('-inf') if self.config.eval_metric_goal == 'max' else float('inf')
            self.best_epoch = 0
            self.epochs_without_improvement = 0

        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            train_loss, train_metrics = self.train_epoch(epoch)
        
            val_loss, val_metrics = self.validate(epoch)
          
            # Use TB Label metric as primary
            eval_metric_key = f"TB Label_{self.config.eval_metric}"
            
            current_metric = val_metrics.get(eval_metric_key, val_loss)
            is_best = False
            
            if self.config.eval_metric_goal == 'max':
                if current_metric > self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    logger.info(f"New best model with {eval_metric_key}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"No improvement. Best {eval_metric_key} is still: {self.best_metric:.4f} from epoch {self.best_epoch+1}")
            else:  # 'min'
                if current_metric < self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    logger.info(f"New best model with {eval_metric_key}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"No improvement. Best {eval_metric_key} is still: {self.best_metric:.4f} from epoch {self.best_epoch+1}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training completed. Best model from epoch {self.best_epoch+1} with {eval_metric_key}: {self.best_metric:.4f}")
        
        # Evaluate the best model if requested
        if self.config.evaluate_best_valid_model:
            self._evaluate_best_model()
        
        return self.best_metric, self.best_epoch
    
    def _evaluate_best_model(self):
        """
        Evaluate the best model on training, validation, and test sets with TB focus.
        """
        logger.info("Evaluating best TB model on all splits...")
        
        # Create final results directory
        final_results_dir = os.path.join(self.config.experiment_dir, "final_results")
        os.makedirs(final_results_dir, exist_ok=True)
        
        # Create subdirectories
        tb_results_dir = os.path.join(final_results_dir, "tb_results")
        pathology_results_dir = os.path.join(final_results_dir, "pathology_results")
        os.makedirs(tb_results_dir, exist_ok=True)
        if self.use_pathology_loss:
            os.makedirs(pathology_results_dir, exist_ok=True)

        # Load best model
        if self.config.train == False:
            # Evaluation-only mode - create model with same config
            self.model = MultiTaskModel(self.config)
            
            if hasattr(self.config, 'best_model_path') and self.config.best_model_path:
                try:
                    checkpoint = torch.load(self.config.best_model_path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'model_state' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded pretrained weights from {self.config.best_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load pretrained weights: {e}")
        else:
            # Training mode - load best checkpoint
            best_model_path = os.path.join(self.config.experiment_dir, "checkpoint_best.pth")
            if not os.path.exists(best_model_path):
                logger.warning(f"Best model checkpoint not found at {best_model_path}. Skipping evaluation.")
                return
        
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model_state' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info(f"Loaded best model from {best_model_path}")
            except Exception as e:
                logger.error(f"Failed to load best model: {e}")
                return

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Evaluate on each split
        splits = [
            ('test', self.test_loader, 'test'),
            ('validation', self.val_loader, 'val'),
            ('train', self.train_loader, 'train'),
        ]
        
        # Dictionary to store all metrics for final summary
        all_tb_metrics = {}
        all_pathology_metrics = {}
        
        # Get pathology class names
        pathology_class_names = getattr(self.config, 'pathology_classes', [
            f"pathology_{i}" for i in range(self.config.num_pathologies)]) if self.use_pathology_loss else []
        
        # Evaluate on each split
        for split_name, loader, file_prefix in splits:
            logger.info(f"Evaluating on {split_name} set...")
            
            # Run detailed evaluation
            detailed_results, tb_metrics, pathology_metrics = self._run_detailed_tb_evaluation(loader, split_name)
            
            # Store metrics for this split
            all_tb_metrics[split_name] = tb_metrics
            
            if self.use_pathology_loss:
                all_pathology_metrics[split_name] = pathology_metrics
            
            # Log TB metrics
            tb_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items() 
                                         if k != 'loss' and isinstance(v, (int, float))])
            logger.info(f"{split_name.capitalize()} TB Metrics: {tb_metrics_str}")
            
            # Save detailed predictions to CSV
            predictions_file = os.path.join(final_results_dir, f"{file_prefix}_predictions.csv")
            detailed_results.to_csv(predictions_file, index=False)
            logger.info(f"Detailed {split_name} predictions saved to {predictions_file}")
            
            # Save ROC curves for TB
            if 'auc' in tb_metrics and not np.isnan(tb_metrics['auc']):
                self._plot_roc_curve(
                    detailed_results['tb_target'], 
                    detailed_results['tb_probability'], 
                    os.path.join(tb_results_dir, f"{file_prefix}_tb_roc_curve.png"),
                    title=f"TB ROC Curve - {split_name} (AUC: {tb_metrics['auc']:.4f})"
                )
            
            # Save pathology metrics and ROC curves
            if self.use_pathology_loss:
                for pathology_name, metrics in pathology_metrics.items():
                    pathology_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                                                       if k != 'loss' and isinstance(v, (int, float)) and not np.isnan(v)])
                    logger.info(f"{split_name.capitalize()} {pathology_name} Metrics: {pathology_metrics_str}")
                    
                    pathology_dir = os.path.join(pathology_results_dir, pathology_name.lower().replace(" ", "_"))
                    os.makedirs(pathology_dir, exist_ok=True)
                    
                    if 'auc' in metrics and not np.isnan(metrics['auc']):
                        self._plot_roc_curve(
                            detailed_results[f'{pathology_name}_target'], 
                            detailed_results[f'{pathology_name}_probability'], 
                            os.path.join(pathology_dir, f"{file_prefix}_{pathology_name}_roc_curve.png"),
                            title=f"{pathology_name} ROC Curve - {split_name} (AUC: {metrics['auc']:.4f})"
                        )
        
        # Save TB metrics summary to CSV
        try:
            # TB summary
            tb_summary_data = []
            for split_name, metrics in all_tb_metrics.items():
                row = {'Split': split_name}
                row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                tb_summary_data.append(row)
            
            if tb_summary_data:
                tb_metrics_summary = pd.DataFrame(tb_summary_data)
                tb_summary_file = os.path.join(tb_results_dir, "tb_metrics_summary.csv")
                tb_metrics_summary.to_csv(tb_summary_file, index=False)
                logger.info(f"Summary of TB metrics saved to {tb_summary_file}")
                
                # Plot comparative TB metrics
                self._plot_comparative_metrics(
                    tb_metrics_summary, 
                    os.path.join(tb_results_dir, "tb_comparative_metrics.png"), 
                    title="TB Classification Performance"
                )
            
            # Pathology summaries
            if self.use_pathology_loss and pathology_class_names:
                for pathology_name in pathology_class_names:
                    pathology_summary_data = []
                    for split_name in all_pathology_metrics:
                        if pathology_name in all_pathology_metrics[split_name]:
                            row = {'Split': split_name}
                            metrics = all_pathology_metrics[split_name][pathology_name]
                            row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                            pathology_summary_data.append(row)
                    
                    if pathology_summary_data:
                        pathology_dir = os.path.join(pathology_results_dir, pathology_name.lower().replace(" ", "_"))
                        os.makedirs(pathology_dir, exist_ok=True)
                        
                        pathology_metrics_summary = pd.DataFrame(pathology_summary_data)
                        pathology_summary_file = os.path.join(pathology_dir, f"{pathology_name}_metrics_summary.csv")
                        pathology_metrics_summary.to_csv(pathology_summary_file, index=False)
                        logger.info(f"Summary of {pathology_name} metrics saved to {pathology_summary_file}")
                        
                        # Plot comparative pathology metrics
                        self._plot_comparative_metrics(
                            pathology_metrics_summary, 
                            os.path.join(pathology_dir, f"{pathology_name}_comparative_metrics.png"),
                            title=f"{pathology_name} Classification Performance"
                        )
                
                # Create consolidated pathology summary
                if pathology_class_names:
                    self._create_consolidated_pathology_summary(
                        all_pathology_metrics, pathology_class_names, 
                        os.path.join(pathology_results_dir, "all_pathologies_summary.csv")
                    )
            
        except Exception as e:
            logger.error(f"Error saving metrics summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _create_consolidated_pathology_summary(self, all_pathology_metrics, pathology_class_names, output_file):
        """Creates a consolidated summary of all pathology metrics across splits."""
        try:
            metrics_to_include = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc', 'auprc']
            
            consolidated_data = []
            
            for split_name in all_pathology_metrics:
                for pathology_name in pathology_class_names:
                    if pathology_name in all_pathology_metrics[split_name]:
                        metrics = all_pathology_metrics[split_name][pathology_name]
                        
                        row = {
                            'Split': split_name,
                            'Pathology': pathology_name
                        }
                        
                        for metric in metrics_to_include:
                            if metric in metrics and isinstance(metrics[metric], (int, float)):
                                row[metric] = metrics[metric]
                            else:
                                row[metric] = float('nan')
                        
                        consolidated_data.append(row)
            
            consolidated_df = pd.DataFrame(consolidated_data)
            consolidated_df.to_csv(output_file, index=False)
            logger.info(f"Consolidated pathology metrics summary saved to {output_file}")
            
            # Also create a pivot table view for easier comparison
            pivot_file = output_file.replace('.csv', '_pivot.csv')
            for metric in metrics_to_include:
                metric_pivot = consolidated_df.pivot_table(
                    index='Pathology', 
                    columns='Split', 
                    values=metric
                )
                metric_pivot.to_csv(pivot_file.replace('.csv', f'_{metric}.csv'))
                logger.info(f"Pivot table for {metric} saved to {pivot_file.replace('.csv', f'_{metric}.csv')}")
                
        except Exception as e:
            logger.error(f"Error creating consolidated pathology summary: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_roc_curve(self, y_true, y_score, output_path, title="ROC Curve"):
        """Plot ROC curve and save to file."""
        try:
            plt.figure(figsize=(10, 8))
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"ROC curve saved to {output_path}")
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")

    def _plot_comparative_metrics(self, metrics_df, output_path, title="Model Performance Comparison"):
        """Plot comparative metrics across different splits."""
        try:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc', 'auprc']
            metrics_to_plot = [m for m in metrics_to_plot if m in metrics_df.columns]
            
            if not metrics_to_plot:
                logger.warning("No metrics available for comparison plot")
                return
            
            plt.figure(figsize=(12, 8))
            
            splits = metrics_df['Split'].tolist()
            n_splits = len(splits)
            n_metrics = len(metrics_to_plot)
            bar_width = 0.8 / n_metrics
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_df.columns:
                    positions = np.arange(n_splits) + (i - n_metrics/2 + 0.5) * bar_width
                    values = metrics_df[metric].values
                    
                    if not np.isnan(values).all():
                        plt.bar(positions, values, width=bar_width, label=metric.capitalize())
            
            plt.xlabel('Data Split')
            plt.ylabel('Score')
            plt.title(title)
            plt.xticks(np.arange(n_splits), splits)
            plt.ylim(0, 1.0)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=n_metrics)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Comparative metrics plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Error plotting comparative metrics: {e}")
    
    def _run_detailed_tb_evaluation(self, loader, split_name):
        """
        Run detailed evaluation on a data loader with TB focus.
        
        Returns:
            tuple: (detailed_results_df, tb_metrics_dict, pathology_metrics_dict)
        """
        self.model.eval()
        all_patient_ids = []
        
        # TB tracking
        all_tb_targets = []
        all_tb_logits = []
        all_tb_probs = []
        all_tb_preds = []
        
        all_pathology_targets_np = []
        all_pathology_logits_np = []
        all_pathology_probs_np = []
        all_pathology_preds_np = []
        
        running_loss = 0.0
        
        pathology_dim = None
        first_batch = True
        
        progress_bar = tqdm(loader, desc=f"{split_name.capitalize()} Detailed Evaluation")
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Get patient IDs
                    patient_ids = batch['patient_ids']
                    
                    # Move data to device
                    site_videos = batch['site_videos'].to(self.device)
                    site_indices = batch['site_indices'].to(self.device)
                    site_masks = batch['site_masks'].to(self.device)
                    site_findings = batch['site_findings'].to(self.device)
                    
                    # TB labels (only TB task)
                    tb_labels = batch['tb_labels'].to(self.device).float()
                    # Dummy labels for multi-task model
                    pneumonia_labels = torch.full_like(tb_labels, -1)
                    covid_labels = torch.full_like(tb_labels, -1)
                    
                    if first_batch:
                        logger.info(f"Site findings shape: {site_findings.shape}")
                        if len(site_findings.shape) > 1:
                            pathology_dim = site_findings.shape[1]
                        else:
                            pathology_dim = 1
                        logger.info(f"Detected {pathology_dim} pathology dimensions")
                        first_batch = False
                    
                    # Prepare inputs
                    inputs = {
                        'site_videos': site_videos,
                        'site_indices': site_indices,
                        'site_masks': site_masks,
                        'site_findings': site_findings,
                        'is_patient_level': True
                    }
                    
                    targets = {
                        'tb_labels': tb_labels,
                        'pneumonia_labels': pneumonia_labels,
                        'covid_labels': covid_labels,
                        'pathology_labels': site_findings
                    }
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Compute loss using new interface
                    try:
                        loss, _ = self.model.compute_losses(outputs, targets, self.task_pos_weights)
                        running_loss += loss.item()
                    except Exception as e:
                        logger.warning(f"Error computing loss: {e}")
                        running_loss += 0.0
                    
                    # Process TB task
                    task_logits = outputs.get('task_logits', {})
                    
                    if 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                        all_tb_probs.append(probs.detach().cpu())
                        all_tb_preds.append(preds.detach().cpu())
                    elif 'tb_logits' in outputs:  # Backward compatibility
                        logits = outputs['tb_logits']
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                        all_tb_probs.append(probs.detach().cpu())
                        all_tb_preds.append(preds.detach().cpu())
                    
                    # Collect pathology data if enabled
                    if self.use_pathology_loss and 'pathology_scores' in outputs:
                        pathology_logits = outputs['pathology_scores']
                        pathology_probs = torch.sigmoid(pathology_logits)
                        pathology_preds = (pathology_probs > 0.5).float()
                        
                        all_pathology_targets_np.append(site_findings.detach().cpu().numpy())
                        all_pathology_logits_np.append(pathology_logits.detach().cpu().numpy())
                        all_pathology_probs_np.append(pathology_probs.detach().cpu().numpy())
                        all_pathology_preds_np.append(pathology_preds.detach().cpu().numpy())
                    
                    # Store patient IDs
                    all_patient_ids.extend(patient_ids)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': running_loss / (progress_bar.n + 1)
                    })
                    
                    # Clean up memory
                    del site_videos, site_indices, site_masks, site_findings, inputs
                    del tb_labels, pneumonia_labels, covid_labels, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.warning(f"WARNING: Out of memory during evaluation, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        logger.error(f"Runtime error during evaluation: {e}")
                        raise e
                except Exception as e:
                    logger.error(f"Unexpected error during evaluation: {e}")
                    continue
        
        # Calculate average loss
        val_loss = running_loss / len(loader)
        
        # Process results for TB
        tb_metrics = {}
        detailed_results = pd.DataFrame({'patient_id': all_patient_ids})
        
        if all_tb_targets:
            tb_targets = torch.cat(all_tb_targets).numpy()
            tb_logits = torch.cat(all_tb_logits).numpy()
            tb_probs = torch.cat(all_tb_probs).numpy()
            tb_preds = torch.cat(all_tb_preds).numpy()
            
            # Flatten if needed
            if tb_targets.ndim > 1 and tb_targets.shape[1] == 1:
                tb_targets = tb_targets.flatten()
            if tb_preds.ndim > 1 and tb_preds.shape[1] == 1:
                tb_preds = tb_preds.flatten()
            if tb_probs.ndim > 1 and tb_probs.shape[1] == 1:
                tb_probs = tb_probs.flatten()
            if tb_logits.ndim > 1 and tb_logits.shape[1] == 1:
                tb_logits = tb_logits.flatten()
            
            # Calculate metrics
            tb_metrics = self._calculate_binary_metrics(
                tb_targets, tb_preds, tb_probs, "TB Label"
            )
            tb_metrics['loss'] = val_loss
            
            # Add to detailed results
            detailed_results['tb_target'] = tb_targets
            detailed_results['tb_prediction'] = tb_preds
            detailed_results['tb_probability'] = tb_probs
            detailed_results['tb_logit'] = tb_logits
        
        # Handle pathology results
        pathology_metrics = {}
        
        if self.use_pathology_loss and all_pathology_targets_np and pathology_dim is not None:
            try:
                pathology_targets = np.vstack(all_pathology_targets_np)
                pathology_logits = np.vstack(all_pathology_logits_np)
                pathology_probs = np.vstack(all_pathology_probs_np)
                pathology_preds = np.vstack(all_pathology_preds_np)
                
                pathology_class_names = getattr(self.config, 'pathology_classes', [
                    f"pathology_{i}" for i in range(pathology_dim)])
                
                for i, class_name in enumerate(pathology_class_names):
                    if i < pathology_targets.shape[1]:
                        class_targets = pathology_targets[:, i]
                        class_preds = pathology_preds[:, i]
                        class_probs = pathology_probs[:, i]
                        
                        pathology_metrics[class_name] = self._calculate_binary_metrics(
                            class_targets, class_preds, class_probs, f"Pathology: {class_name}"
                        )
                        
                        detailed_results[f'{class_name}_target'] = class_targets
                        detailed_results[f'{class_name}_prediction'] = class_preds
                        detailed_results[f'{class_name}_probability'] = class_probs
                        detailed_results[f'{class_name}_logit'] = pathology_logits[:, i]
            
            except Exception as e:
                logger.error(f"Error processing pathology results: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return detailed_results, tb_metrics, pathology_metrics

    def _calculate_binary_metrics(self, targets, predictions, probabilities, name=""):
        """Calculate binary classification metrics for a single target."""
        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['precision'] = precision_score(targets, predictions, average='binary', zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, average='binary', zero_division=0)
            metrics['specificity'] = recall_score(1-targets, 1-predictions, zero_division=0)
            metrics['f1'] = f1_score(targets, predictions, average='binary', zero_division=0)
            
            if len(np.unique(targets)) > 1:
                metrics['auc'] = roc_auc_score(targets, probabilities)
                metrics['auprc'] = average_precision_score(targets, probabilities)
            else:
                logger.warning(f"{name}: Only one class present, skipping AUC and AUPRC calculation")
                metrics['auc'] = float('nan')
                metrics['auprc'] = float('nan')
            
            # Confusion matrix values
            tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn
            
            # Print diagnostic information
            diagnostic_data = {
                'target': targets.flatten(),
                'prediction': predictions.flatten(),
                'probability': probabilities.flatten()
            }
            
            df = pd.DataFrame(diagnostic_data)
            
            sample_rows = df.sample(min(5, len(df)))
            logger.info(f"\nDiagnostic sample for {name} (5 random rows):")
            logger.info(f"\n{sample_rows}")
            
            logger.info(f"\n{name} - Probability stats - min: {df['probability'].min():.4f}, max: {df['probability'].max():.4f}, mean: {df['probability'].mean():.4f}")
            
            cm = confusion_matrix(targets, predictions)
            logger.info(f"\nConfusion Matrix ({name}):\n{cm}")
            
            report = classification_report(targets, predictions)
            logger.info(f"\nClassification Report ({name}):\n{report}")
            
            class_counts = pd.Series(targets).value_counts()
            logger.info(f"\nClass distribution for {name}:\n{class_counts}")
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for {name}: {e}")
        
        return metrics

    def load_checkpoint(self, checkpoint_path, load_optimizers=True, load_schedulers=True):
        """Load checkpoint and restore training state."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state loaded successfully")
            
            if load_optimizers:
                if 'backbone_optimizer_state_dict' in checkpoint:
                    self.backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer_state_dict'])
                
                if self.selection_strategy == 'RL':
                    if 'frame_selector_optimizer_state_dict' in checkpoint:
                        self.frame_selector_optimizer.load_state_dict(checkpoint['frame_selector_optimizer_state_dict'])
                
                if 'patient_pipeline_optimizer_state_dict' in checkpoint:
                    self.patient_pipeline_optimizer.load_state_dict(checkpoint['patient_pipeline_optimizer_state_dict'])
                
                if 'pathology_optimizers_state_dicts' in checkpoint:
                    for opt, state_dict in zip(self.pathology_optimizers, checkpoint['pathology_optimizers_state_dicts']):
                        opt.load_state_dict(state_dict)
                
                logger.info("Optimizer states loaded successfully")
            
            if load_schedulers and 'schedulers_state_dicts' in checkpoint:
                for scheduler, state_dict in zip(self.schedulers, checkpoint['schedulers_state_dicts']):
                    scheduler.load_state_dict(state_dict)
                logger.info("Scheduler states loaded successfully")
            
            if self.use_amp:
                if 'backbone_scaler_state_dict' in checkpoint:
                    self.backbone_scaler.load_state_dict(checkpoint['backbone_scaler_state_dict'])
                
                if 'patient_pipeline_scaler_state_dict' in checkpoint:
                    self.patient_pipeline_scaler.load_state_dict(checkpoint['patient_pipeline_scaler_state_dict'])
                
                if 'pathology_scalers_state_dicts' in checkpoint:
                    for scaler, state_dict in zip(self.pathology_scalers, checkpoint['pathology_scalers_state_dicts']):
                        scaler.load_state_dict(state_dict)
                
                if self.selection_strategy == 'RL' and 'frame_selector_scaler_state_dict' in checkpoint:
                    self.frame_selector_scaler.load_state_dict(checkpoint['frame_selector_scaler_state_dict'])
                
                logger.info("AMP scaler states loaded successfully")
            
            if 'best_metric' in checkpoint:
                self.best_metric = checkpoint['best_metric']
            
            if 'best_epoch' in checkpoint:
                self.best_epoch = checkpoint['best_epoch']
            
            if 'epochs_without_improvement' in checkpoint:
                self.epochs_without_improvement = checkpoint['epochs_without_improvement']
            
            return checkpoint.get('epoch', 0)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise e

    def resume_training_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        start_epoch = self.load_checkpoint(checkpoint_path, load_optimizers=True, load_schedulers=True)
        
        logger.info(f"Training will resume from epoch {start_epoch + 1}")
        logger.info(f"Best metric so far: {self.best_metric:.4f} (epoch {self.best_epoch + 1})")
        logger.info(f"Epochs without improvement: {self.epochs_without_improvement}")
        
        return start_epoch + 1


def print_gpu_memory():
    """Print GPU memory usage for debugging."""
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available")


# Import config module - try both locations for compatibility
try:
    from config import parse_args_and_load_config
except ImportError:
    logger.warning("Advanced config module not found, using basic config")
    
    # Basic config functionality for backward compatibility
    # class Config:
    #     def __init__(self, params=None):
    #         # Data paths
    #         self.train = True
    #         self.root_dir="/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v3"
    #         self.labels_csv="/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v3/labels/labels.csv"
    #         self.file_metadata_csv="/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v3/processed_files_2.csv"
    #         self.split_csv = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files/Fold_1.csv'

    #         self.video_folder = 'videos'
    #         self.image_folder = 'images'
            
    #         # Model config
    #         self.model_type = 'tb_rl_mil'
    #         self.model_name = 'drl_mil_tb_classifier_fold1'
    #         self.backbone = 'resnet18'
    #         self.freeze_backbone = False
    #         self.hidden_dim = 512
    #         self.dropout_rate = 0.3
    #         self.num_pathologies = 5
    #         self.pretrained = True
    #         self.num_classes = 1
    #         self.in_channels = 3
    #         self.temperature = 0.9
    #         self.reset_optimizers = False
            
    #         # Data preprocessing
    #         self.target_height = 224
    #         self.target_width = 224
    #         self.depth_filter = '15'
    #         self.frame_sampling = 32
    #         self.num_sites = 15
    #         self.mode = 'video'
    #         self.pooling = 'attention'
            
    #         # Training settings
    #         self.task = "TB Label"
    #         self.batch_size = 2
    #         self.num_workers = 6
    #         self.learning_rate = 0.00001
    #         self.rl_learning_rate = 0.0001  # Single RL learning rate
    #         self.weight_decay = 0.00001
    #         self.num_epochs = 20
    #         self.early_stopping_patience = 8
    #         self.accumulation_steps = 8
    #         self.use_amp = True
    #         self.seed = 42
            
    #         # Multi-task config for compatibility
    #         self.active_tasks = ['TB Label']
    #         self.use_pathology_loss = True
    #         self.task_weights = {'TB Label': 1.0}
    #         self.selection_strategy = 'RL'
            
    #         # New dataset parameters
    #         self.files_per_site = 1
    #         self.site_order = None
    #         self.pad_missing_sites = True
    #         self.max_sites = 15
            
    #         self.classification_type = "binary"
    #         self.pos_weight = 1.4
    #         self.gamma = 0.98
            
    #         # RL parameters
    #         self.gamma_rl = 0.99
    #         self.temperature_min = 0.1
    #         self.temperature_max = 5.0
    #         self.temperature_decay = 0.995
    #         self.entropy_weight = 0.05
    #         self.use_frame_history = True
    #         self.tb_weight = 1.0
    #         self.pathology_weight = 0.9
    #         self.rl_accumulation_steps = 8
    #         self.frame_selector_max_norm = 1.0
            
    #         # Evaluation settings
    #         self.eval_metric = "auc"
    #         self.eval_metric_goal = "max"
    #         self.evaluate_best_valid_model = True

    #         self.local_weights_dir = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/NetworkArchitecture/CLIP_weights'
            
    #         # Optimizer settings for new training structure
    #         self.backbone_lr = 0.00001
    #         self.backbone_weight_decay = 0.00001
    #         self.backbone_T_0 = 4
    #         self.backbone_T_mult = 2
    #         self.backbone_eta_min = 1e-6
            
    #         self.pathology_lr = 0.0001
    #         self.pathology_weight_decay = 0.00001
    #         self.pathology_T_0 = 4
    #         self.pathology_T_mult = 2
    #         self.pathology_eta_min = 1e-6
            
    #         self.patient_pipeline_lr = 0.001
    #         self.patient_pipeline_weight_decay = 0.00001
    #         self.patient_pipeline_T_0 = 4
    #         self.patient_pipeline_T_mult = 2
    #         self.patient_pipeline_eta_min = 1e-6
            
    #         # Directories
    #         self.log_dir = "logs"
    #         self.save_dir = "models"
    #         self.checkpoint_dir = "checkpoints"
    #         self.pred_save_dir = "predictions"
            
    #         # Device
    #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #         # Pathology-specific positive weights
    #         self.pathology_pos_weights = [1.0, 4.0, 4.0, 4.0, 15.0]
            
    #         # Create required directories
    #         os.makedirs(self.log_dir, exist_ok=True)
    #         os.makedirs(self.save_dir, exist_ok=True)
    #         os.makedirs(self.checkpoint_dir, exist_ok=True)
    #         os.makedirs(self.pred_save_dir, exist_ok=True)
            
    #         # Create experiment-specific directories
    #         self.experiment_dir = os.path.join(self.checkpoint_dir, self.model_name)
    #         os.makedirs(self.experiment_dir, exist_ok=True)
            
    #         # Update config with provided parameters
    #         if params:
    #             for key, value in params.items():
    #                 setattr(self, key, value)
        
    #     def to_dict(self):
    #         """Convert configuration to dictionary."""
    #         return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
        
    #     def save(self, path):
    #         """Save configuration to YAML file."""
    #         import yaml
    #         with open(path, 'w') as f:
    #             yaml.dump(self.to_dict(), f)
        
    #     def load(self, path):
    #         """Load configuration from YAML file."""
    #         import yaml
    #         with open(path, 'r') as f:
    #             params = yaml.safe_load(f)
    #             for key, value in params.items():
    #                 setattr(self, key, value)
    #         logger.info(f"Loaded configuration from {path}")

    # def parse_args_and_load_config():
    #     """Simple argument parser and config loader."""
    #     parser = argparse.ArgumentParser(description='Training for TB detection using RL-MIL model.')
        
    #     parser.add_argument('--config', type=str, help='Path to config YAML file')
    #     parser.add_argument('--lr', type=float, help='Learning rate')
    #     parser.add_argument('--batch_size', type=int, help='Batch size')
    #     parser.add_argument('--epochs', type=int, help='Number of epochs')
    #     parser.add_argument('--seed', type=int, help='Random seed')
    #     parser.add_argument('--model_weights', type=str, help='Path to model weights')
    #     parser.add_argument('--best_model_path', type=str, help='Path to best model for evaluation')
    #     parser.add_argument('--train', action='store_true', default=True, help='Train mode')
    #     parser.add_argument('--eval_only', action='store_true', help='Evaluation only mode')
        
    #     args = parser.parse_args()
        
    #     # Create config
    #     config = Config()
        
    #     # Load YAML config if file exists
    #     if args.config and os.path.exists(args.config):
    #         config.load(args.config)
        
    #     # Override with command-line arguments
    #     if args.lr:
    #         config.learning_rate = args.lr
    #     if args.batch_size:
    #         config.batch_size = args.batch_size
    #     if args.epochs:
    #         config.num_epochs = args.epochs
    #     if args.seed:
    #         config.seed = args.seed
    #     if args.model_weights:
    #         config.model_weights = args.model_weights
    #     if args.best_model_path:
    #         config.best_model_path = args.best_model_path
    #     if args.eval_only:
    #         config.train = False
        
    #     return config

def main():
    """Main training function with command line argument support."""
    
    # Parse command line arguments and load configuration
    config = parse_args_and_load_config()
    
    logger.info("TB Classification Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Create experiment directory and save config
    os.makedirs(config.experiment_dir, exist_ok=True)
    config_path = os.path.join(config.experiment_dir, "config.yaml")
    config.save(config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    # GPU/Device information
    if torch.cuda.is_available():
        logger.info(f"Using device: {config.device}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        print_gpu_memory()
    else:
        logger.info("Using CPU")

    # Initialize trainer
    trainer = TBTrainer(config)
    
    # Check for resume checkpoint
    resume_checkpoint = None
    if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint is not None:
        if not os.path.exists(config.resume_from_checkpoint):
            logger.error(f"Resume checkpoint not found: {config.resume_from_checkpoint}")
            return
        resume_checkpoint = config.resume_from_checkpoint
        logger.info(f"Will resume training from: {resume_checkpoint}")
    
    # Check if we're in evaluation-only mode
    if not config.train:
        logger.info("Running in evaluation-only mode")
        trainer._evaluate_best_model()
        return
    
    # Start training
    logger.info("Starting TB training...")
    best_metric, best_epoch = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training complete! Best metric: {best_metric:.4f} at epoch {best_epoch+1}")
    return best_metric, best_epoch


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in training: {e}")
        raise



# python3 train_clip_drl_mil_Final.py --config configs/Finalruns/tb_drl_mil_Final_fold1.yaml
# python3 train_clip_drl_mil_Final.py --config configs/Finalruns/tb_drl_mil_Final_fold2.yaml
# python3 train_clip_drl_mil_Final.py --config configs/Finalruns/tb_drl_mil_Final_fold3.yaml
# python3 train_clip_drl_mil_Final.py --config configs/Finalruns/tb_drl_mil_Final_fold4.yaml


#python3 train_clip_drl_mil_Final.py --config configs/random/tb_drl_mil_Final_fold0.yaml
