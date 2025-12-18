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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataloaders.dataset_multitask import LungUltrasoundDataModule
from NetworkArchitecture.OOMHandler import OOMHandler

from NetworkArchitecture.ablation_models import create_ablation_model

try:
    from NetworkArchitecture.monitoring_utils import log_model_component_status
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Monitoring utilities not available")
    log_model_component_status = lambda *args: None
        
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


class AblationTrainer:
    """Ablation model trainer - following the exact memory pattern of the original."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        self.best_metric = None
        self.best_epoch = 0
        self.epoch = 0
        self.epochs_without_improvement = 0

        self.model_type = getattr(config, 'model_type', 'no_rl')
        self.active_tasks = ['TB Label'] 
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = {'TB Label': 1.0}
        self.task_pos_weights = {'TB Label': getattr(config, 'pos_weight', 2.0)}
        
        logger.info(f"Using ablation model type: {self.model_type}")
        logger.info(f"Pathology loss enabled: {self.use_pathology_loss}")

        self._set_seed(config.seed)
        
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
        """Set up the data module - exact same as original."""
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
            cache_size=100,  
            files_per_site=getattr(self.config, 'files_per_site', 1), 
            site_order=getattr(self.config, 'site_order', None),
            pad_missing_sites=getattr(self.config, 'pad_missing_sites', True),    
            max_sites=getattr(self.config, 'max_sites', 15),
        )
        
        self.data_module.setup(stage='patient_level')
        
        self.train_loader = self.data_module.patient_level_dataloader('train')
        self.val_loader = self.data_module.patient_level_dataloader('val')
        self.test_loader = self.data_module.patient_level_dataloader('test')
        
        logger.info(f"Training dataset size: {len(self.data_module.patient_train)}")
        logger.info(f"Validation dataset size: {len(self.data_module.patient_val)}")
        logger.info(f"Test dataset size: {len(self.data_module.patient_test)}")
    
    def _setup_model(self):
        """Set up the ablation model."""
        
        self.model = create_ablation_model(self.model_type, self.config)
        
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

        try:
            log_model_component_status(self.model, logger)
        except:
            pass  

    def _setup_training(self):
        """Set up optimizers - EXACT same pattern as original."""

        backbone_params = []      
        pathology_params = []    
        patient_pipeline_params = [] 
        task_classifier_params = []
        
        num_pathology_modules = 0
        if hasattr(self.model, 'pathology_modules') and self.model.pathology_modules:
            num_pathology_modules = len(self.model.pathology_modules)
        pathology_module_params = [[] for _ in range(num_pathology_modules)]
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if any(component in name for component in ['vision_encoder', 'cnn_backbone', 'video_transformer', 'backbone']):
                backbone_params.append(param)
            elif 'multi_feature_extraction' in name or 'multi_scale_extraction' in name:
                backbone_params.append(param)
            elif 'pathology_modules' in name and self.use_pathology_loss:
                for i in range(num_pathology_modules):
                    if f'pathology_modules.{i}' in name or f'pathology_modules[{i}]' in name:
                        pathology_module_params[i].append(param)
                        break
            elif 'task_classifiers' in name or 'tb_classifier' in name:
                task_classifier_params.append(param)
            elif any(component in name for component in ['site_integration', 'patient_mil', 'cross_site_attention', 'frame_selector']):
                patient_pipeline_params.append(param)
            else:
                patient_pipeline_params.append(param)
        
        if backbone_params:
            self.backbone_optimizer = optim.AdamW(
                backbone_params,
                lr=getattr(self.config, 'backbone_lr', 0.00001),  
                weight_decay=getattr(self.config, 'backbone_weight_decay', 0.00001)
            )
            logger.info(f"Created backbone optimizer with {len(backbone_params)} parameters")
        else:
            self.backbone_optimizer = None
            logger.info("No backbone parameters found")
        
        self.pathology_optimizers = []
        if self.use_pathology_loss and num_pathology_modules > 0:
            for i, module_params in enumerate(pathology_module_params):
                if module_params: 
                    optimizer = optim.AdamW(
                        module_params,
                        lr=getattr(self.config, 'pathology_lr', 0.0001),
                        weight_decay=getattr(self.config, 'pathology_weight_decay', 0.00001),
                    )
                    self.pathology_optimizers.append(optimizer)
            logger.info(f"Created {len(self.pathology_optimizers)} pathology optimizers")
        
        # Patient pipeline optimizer 
        all_patient_params = patient_pipeline_params + task_classifier_params
        if all_patient_params:
            self.patient_pipeline_optimizer = optim.AdamW(
                all_patient_params,
                lr=getattr(self.config, 'patient_pipeline_lr', 0.001),  
                weight_decay=getattr(self.config, 'patient_pipeline_weight_decay', 0.00001),
            )
            logger.info(f"Created patient pipeline optimizer with {len(all_patient_params)} parameters")
        else:
            self.patient_pipeline_optimizer = None
            logger.info("No patient pipeline parameters found")

        # Set up schedulers
        self.schedulers = []
        batches_per_epoch = len(self.train_loader) if hasattr(self, 'train_loader') else 100
        total_steps = self.config.num_epochs * batches_per_epoch
        
        # Backbone scheduler
        if self.backbone_optimizer:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.backbone_optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'backbone_eta_min', 1e-6),
            ))
        
        # Pathology schedulers
        for optimizer in self.pathology_optimizers:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'pathology_eta_min', 1e-6),
            ))
    
        # Patient pipeline scheduler
        if self.patient_pipeline_optimizer:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.patient_pipeline_optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'patient_pipeline_eta_min', 1e-6),
            ))
        
        # Mixed precision training 
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            if self.backbone_optimizer:
                self.backbone_scaler = torch.amp.GradScaler()
            self.pathology_scalers = [torch.amp.GradScaler() for _ in self.pathology_optimizers]
            if self.patient_pipeline_optimizer:
                self.patient_pipeline_scaler = torch.amp.GradScaler()

        # Load optimizer states if resuming
        if hasattr(self.config, 'model_weights') and self.config.model_weights and not getattr(self.config, 'reset_optimizers', True):
            try:
                checkpoint = torch.load(self.config.model_weights, map_location=self.device, weights_only=False)
                
                # Load optimizer states if present
                if 'backbone_optimizer_state_dict' in checkpoint and self.backbone_optimizer:
                    self.backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer_state_dict'])
                    logger.info("Loaded backbone optimizer state")
                     
                if 'patient_pipeline_optimizer_state_dict' in checkpoint and self.patient_pipeline_optimizer:
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
                    if 'backbone_scaler_state_dict' in checkpoint and hasattr(self, 'backbone_scaler'):
                        self.backbone_scaler.load_state_dict(checkpoint['backbone_scaler_state_dict'])
                    
                    if 'patient_pipeline_scaler_state_dict' in checkpoint and hasattr(self, 'patient_pipeline_scaler'):
                        self.patient_pipeline_scaler.load_state_dict(checkpoint['patient_pipeline_scaler_state_dict'])
                    
                    if 'pathology_scalers_state_dicts' in checkpoint:
                        for scaler, state_dict in zip(self.pathology_scalers, checkpoint['pathology_scalers_state_dicts']):
                            scaler.load_state_dict(state_dict)
                    
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
        """Reset state for a new training epoch - same as original."""
        
        # Reset optimizers
        if self.backbone_optimizer:
            self.backbone_optimizer.zero_grad()
        
        if self.patient_pipeline_optimizer:
            self.patient_pipeline_optimizer.zero_grad()
        
        for opt in self.pathology_optimizers:
            opt.zero_grad()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.debug("Reset epoch state: cleared gradients and CUDA cache")

    def train_epoch(self, epoch):
        """Training epoch - EXACT same pattern as original but for ablation models."""
        self.model.train()
        self.epoch = epoch
        
        # Reset model state
        self._reset_for_epoch()
        
        # Initialize tracking metrics
        running_losses = {
            'total': 0.0,
            'tb_loss': 0.0,
        }
        
        if self.use_pathology_loss:
            running_losses['pathology'] = 0.0
        
        # Metrics tracking for TB 
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []
        
        accumulation_steps = self.config.accumulation_steps
        
        self.prefetch_gpu = getattr(self.config, "prefetch_gpu", False)
        data_stream = torch.cuda.Stream() if (torch.cuda.is_available() and self.prefetch_gpu) else None

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        batch_iter = iter(self.train_loader)
        try:
            first_batch = next(batch_iter)

            site_videos   = first_batch['site_videos'].to(self.device, non_blocking=True)
            site_indices  = first_batch['site_indices'].to(self.device, non_blocking=True)
            site_masks    = first_batch['site_masks'].to(self.device, non_blocking=True)
            site_findings = first_batch['site_findings'].to(self.device, non_blocking=True)
            tb_labels     = first_batch['tb_labels'].to(self.device, non_blocking=True).float()
            pneumonia_labels = torch.full_like(tb_labels, -1)
            covid_labels     = torch.full_like(tb_labels, -1)

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

                        next_site_videos   = next_batch['site_videos'].pin_memory()
                        next_site_indices  = next_batch['site_indices'].pin_memory()
                        next_site_masks    = next_batch['site_masks'].pin_memory()
                        next_site_findings = next_batch['site_findings'].pin_memory()
                        next_tb_labels     = next_batch['tb_labels'].pin_memory().float()
                        next_pneumonia_labels = torch.full_like(next_tb_labels, -1)
                        next_covid_labels     = torch.full_like(next_tb_labels, -1)

                        if data_stream:
                            with torch.cuda.stream(data_stream):
                                next_site_videos   = next_site_videos.to(self.device, non_blocking=True)
                                next_site_indices  = next_site_indices.to(self.device, non_blocking=True)
                                next_site_masks    = next_site_masks.to(self.device, non_blocking=True)
                                next_site_findings = next_site_findings.to(self.device, non_blocking=True)
                                next_tb_labels     = next_tb_labels.to(self.device, non_blocking=True)
                                next_pneumonia_labels = next_pneumonia_labels.to(self.device, non_blocking=True)
                                next_covid_labels     = next_covid_labels.to(self.device, non_blocking=True)

                    except StopIteration:
                        next_batch = None

                if data_stream:
                    torch.cuda.current_stream().wait_stream(data_stream)

                if site_videos.shape[1] > 55:
                    if next_batch is not None:
                        if not self.prefetch_gpu:
                            site_videos   = next_site_videos.to(self.device, non_blocking=True)
                            site_indices  = next_site_indices.to(self.device, non_blocking=True)
                            site_masks    = next_site_masks.to(self.device, non_blocking=True)
                            site_findings = next_site_findings.to(self.device, non_blocking=True)
                            tb_labels     = next_tb_labels.to(self.device, non_blocking=True)
                            pneumonia_labels = next_pneumonia_labels.to(self.device, non_blocking=True)
                            covid_labels     = next_covid_labels.to(self.device, non_blocking=True)
                        else:
                            site_videos, site_indices, site_masks, site_findings = \
                                next_site_videos, next_site_indices, next_site_masks, next_site_findings
                            tb_labels, pneumonia_labels, covid_labels = \
                                next_tb_labels, next_pneumonia_labels, next_covid_labels

                        del next_site_videos, next_site_indices, next_site_masks, next_site_findings
                        del next_tb_labels, next_pneumonia_labels, next_covid_labels
                    continue

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
                    'pathology_labels': site_findings,
                    'site_masks': site_masks,
                }

                #==================================================
                # 1. Pathology Modules Update
                #==================================================
                if self.use_pathology_loss and self.pathology_optimizers:
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
                # 2. TB Patient Classifier Update 
                #==================================================

                for param in self.model.parameters():
                    param.requires_grad = False

                # Enable gradients for patient pipeline and task classifiers
                for name, param in self.model.named_parameters():
                    if any(component in name for component in ['site_integration', 'patient_mil', 'task_classifiers', 'cross_site_attention', 'tb_classifier']):
                        param.requires_grad = True
                
                if batch_idx % accumulation_steps == 0 and self.patient_pipeline_optimizer:
                    self.patient_pipeline_optimizer.zero_grad()
                
                if self.use_amp and self.patient_pipeline_optimizer:
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
                                 if any(component in name for component in ['site_integration', 'patient_mil', 'task_classifiers', 'cross_site_attention', 'tb_classifier']) and p.requires_grad], 
                                max_norm=1.0
                            )
                            self.patient_pipeline_scaler.step(self.patient_pipeline_optimizer)
                            self.patient_pipeline_scaler.update() 
                elif self.patient_pipeline_optimizer:
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
                             if any(component in name for component in ['site_integration', 'patient_mil', 'task_classifiers', 'cross_site_attention', 'tb_classifier']) and p.requires_grad], 
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
                # 3. BACKBONE UPDATE 
                #==================================================

                # Cleanup from previous steps
                try:
                    if 'outputs' in locals(): del outputs
                    if 'total_loss' in locals(): del total_loss
                    if 'loss_dict' in locals(): del loss_dict
                    if 'logits' in locals(): del logits
                    if 'probs' in locals(): del probs
                    if 'preds' in locals(): del preds
                    if 'task_logits' in locals(): del task_logits
                except: pass

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Check if backbone optimizer exists
                if self.backbone_optimizer is None:
                    logger.debug("Backbone optimizer disabled, skipping backbone update")
                elif self.backbone_optimizer:
                    
                    # Detect model type
                    param_names = [name for name, _ in self.model.named_parameters()]
                    is_clip_model = any('vision_encoder' in name for name in param_names)
                    is_3d_cnn_model = any('backbone' in name or 'cnn_backbone' in name for name in param_names)
                    is_video_transformer_model = any('video_transformer' in name for name in param_names)
                    
                    # Get current memory status
                    if torch.cuda.is_available():
                        current_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        available_memory_gb = total_memory_gb - current_memory_gb
                    else:
                        current_memory_gb = 0
                        total_memory_gb = 0
                        available_memory_gb = float('inf')
                    
                    logger.info(f"=== BACKBONE UPDATE - {current_memory_gb:.1f}GB used, {available_memory_gb:.1f}GB available ===")
                    
                    # Initialize OOM tracking
                    if not hasattr(self, '_backbone_oom_count'):
                        self._backbone_oom_count = 0
                    
                    # Skip if too many previous OOM errors
                    if self._backbone_oom_count >= 3:
                        logger.warning(f"Skipping backbone update after {self._backbone_oom_count} OOM errors")
                        return
                    
                    # Disable all gradients first
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
                    # Smart unfreezing based on model type and available memory
                    unfrozen_count = 0
                    
                    if is_clip_model:
                    # Conservative for CLIP + include frame_selector for attention models
                        clip_layers = [
                            'vision_encoder.vision_model.encoder.layers.11',
                            'vision_encoder.vision_model.encoder.layers.10',
                            'vision_encoder.visual_projection',
                            'output_projection'
                        ]
                        
                        # Add frame_selector parameters to backbone training
                        frame_selector_components = [
                            'frame_selector'  # Include all frame_selector parameters
                        ]
                        
                        for name, param in self.model.named_parameters():
                            # CLIP backbone layers
                            if any(layer in name for layer in clip_layers):
                                param.requires_grad = True
                                unfrozen_count += 1
                                logger.debug(f"Unfroze CLIP layer: {name}")
                            
                            # Frame selector (for attention-based models)
                            elif any(component in name for component in frame_selector_components):
                                param.requires_grad = True
                                unfrozen_count += 1
                                logger.debug(f"Unfroze frame selector: {name}")
                        
                        logger.info(f"CLIP + Frame Selector: training {unfrozen_count} parameter groups")
                    
                        
                    elif is_3d_cnn_model or is_video_transformer_model:
                        
                        if available_memory_gb > 15: 
                            #logger.info("High memory available - training full backbone")
                            # Train everything except task-specific layers
                            skip_components = ['pathology_modules', 'task_classifiers', 'tb_classifier', 'patient_mil', 'site_integration']
                            for name, param in self.model.named_parameters():
                                if not any(skip in name for skip in skip_components):
                                    param.requires_grad = True
                                    unfrozen_count += 1
                                    
                        elif available_memory_gb > 8:  
                            # Train only deeper layers + projections
                            moderate_components = [
                                'backbone.layer3', 'backbone.layer4', 'backbone.avgpool', 'backbone.fc',
                                'cnn_backbone.layer3', 'cnn_backbone.layer4', 
                                'video_transformer.encoder.layers.10', 'video_transformer.encoder.layers.11',
                                'feature_projection', 'cnn_projection'
                            ]
                            for name, param in self.model.named_parameters():
                                if any(comp in name for comp in moderate_components):
                                    param.requires_grad = True
                                    unfrozen_count += 1
                                    
                        else:  # Low memory
                            # Train only final layers
                            minimal_components = [
                                'backbone.layer4', 'backbone.avgpool', 'backbone.fc',
                                'cnn_backbone.layer4',
                                'video_transformer.encoder.layers.11', 
                                'feature_projection', 'cnn_projection'
                            ]
                            for name, param in self.model.named_parameters():
                                if any(comp in name for comp in minimal_components):
                                    param.requires_grad = True
                                    unfrozen_count += 1
                    
                    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    
                    # Skip if no parameters to train
                    if unfrozen_count == 0:
                        logger.warning("No backbone parameters to train")
                        for param in self.model.parameters():
                            param.requires_grad = True
                        return
                    
                    # Zero gradients
                    if batch_idx % accumulation_steps == 0:
                        self.backbone_optimizer.zero_grad()
                    
                    # Memory-optimized forward pass
                    try:
                        # Enable gradient checkpointing if available 
                        if hasattr(self.model, 'gradient_checkpointing_enable'):
                            self.model.gradient_checkpointing_enable()
                        
                        if self.use_amp and hasattr(self, 'backbone_scaler'):
                            with torch.amp.autocast('cuda'):
                                backbone_outputs = self.model(inputs)
                                backbone_loss, _ = self.model.compute_losses(backbone_outputs, targets, self.task_pos_weights)
                                backbone_loss = backbone_loss / accumulation_steps
                                running_losses['total'] += backbone_loss.item() * accumulation_steps
                                
                                del backbone_outputs
                                
                                self.backbone_scaler.scale(backbone_loss).backward()
                                del backbone_loss
                                
                                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                                    self.backbone_scaler.unscale_(self.backbone_optimizer)
                                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=0.5)
                                    self.backbone_scaler.step(self.backbone_optimizer)
                                    self.backbone_scaler.update()
                        else:
                            backbone_outputs = self.model(inputs)
                            backbone_loss, _ = self.model.compute_losses(backbone_outputs, targets, self.task_pos_weights)
                            backbone_loss = backbone_loss / accumulation_steps
                            running_losses['total'] += backbone_loss.item() * accumulation_steps
                            
                            del backbone_outputs
                            backbone_loss.backward()
                            del backbone_loss
                            
                            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=0.5)
                                self.backbone_optimizer.step()
                        
                        if self._backbone_oom_count > 0:
                            #logger.info(f"✓ Backbone update successful after {self._backbone_oom_count} previous OOM errors")
                            self._backbone_oom_count = 0
                        else:
                            logger.debug("✓ Backbone update completed successfully")
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            self._backbone_oom_count += 1
                            
                            current_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                            logger.error(f"✗ OOM in backbone update (batch {batch_idx}) - {current_mem:.1f}GB used")
                            logger.error(f"   OOM count: {self._backbone_oom_count}, trainable params: {trainable_params:,}")
                            
                            # Emergency cleanup
                            if self.backbone_optimizer:
                                self.backbone_optimizer.zero_grad()
                            if self.use_amp and hasattr(self, 'backbone_scaler'):
                                self.backbone_scaler = torch.amp.GradScaler()
                            
                            # Aggressive cleanup
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            # Adaptive response based on OOM count
                            if self._backbone_oom_count == 1:
                                logger.warning("Will try smaller backbone scope next batch")
                            elif self._backbone_oom_count == 2:
                                logger.warning("Will try minimal backbone scope next batch") 
                            else:
                                logger.error("Disabling backbone updates due to persistent OOM")
                                self.backbone_optimizer = None
                                
                        else:
                            raise e
                    
                    # Disable gradient checkpointing after use
                    if hasattr(self.model, 'gradient_checkpointing_disable'):
                        self.model.gradient_checkpointing_disable()
                    
                    # Final cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Re-enable all gradients
                for param in self.model.parameters():
                    param.requires_grad = True
                #==================================================
                # Memory cleanup and batch swapping 
                #==================================================
                
                if batch_idx % 2 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Swap batches
                if next_batch is not None:
                    if not self.prefetch_gpu:
                        site_videos   = next_site_videos.to(self.device, non_blocking=True)
                        site_indices  = next_site_indices.to(self.device, non_blocking=True)
                        site_masks    = next_site_masks.to(self.device, non_blocking=True)
                        site_findings = next_site_findings.to(self.device, non_blocking=True)
                        tb_labels     = next_tb_labels.to(self.device, non_blocking=True)
                        pneumonia_labels = next_pneumonia_labels.to(self.device, non_blocking=True)
                        covid_labels     = next_covid_labels.to(self.device, non_blocking=True)
                    else:
                        site_videos, site_indices, site_masks, site_findings = \
                            next_site_videos, next_site_indices, next_site_masks, next_site_findings
                        tb_labels, pneumonia_labels, covid_labels = \
                            next_tb_labels, next_pneumonia_labels, next_covid_labels

                    del next_site_videos, next_site_indices, next_site_masks, next_site_findings
                    del next_tb_labels, next_pneumonia_labels, next_covid_labels
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
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
                    
                    progress_bar.set_postfix(progress_dict)
                    
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"OOM in batch {batch_idx}, skipping")
                    
                    # Reset optimizers 
                    if self.backbone_optimizer:
                        self.backbone_optimizer.zero_grad()
                    if self.patient_pipeline_optimizer:
                        self.patient_pipeline_optimizer.zero_grad()
                    for opt in self.pathology_optimizers:
                        opt.zero_grad()
                    
                    # Reset scalers 
                    if self.use_amp:
                        if hasattr(self, 'backbone_scaler'):
                            self.backbone_scaler = torch.amp.GradScaler()
                        self.pathology_scalers = [torch.amp.GradScaler() for _ in self.pathology_optimizers]
                        if hasattr(self, 'patient_pipeline_scaler'):
                            self.patient_pipeline_scaler = torch.amp.GradScaler()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Runtime error: {e}")
                    raise e
        
        progress_bar.close()
        
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
        
        return running_losses['total'] / max(1, len(self.train_loader)), all_metrics


    def validate(self, epoch, loader=None, split_name="val"):
        """TB-focused validation."""
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
                    elif 'tb_logits' in outputs:  
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
        
        if self.use_amp:
            checkpoint.update({
                'backbone_scaler_state_dict': self.backbone_scaler.state_dict(),
                'patient_pipeline_scaler_state_dict': self.patient_pipeline_scaler.state_dict(),
                'pathology_scalers_state_dicts': [scaler.state_dict() for scaler in self.pathology_scalers],
            })
            
            
        
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
    
    def train(self, resume_from_checkpoint=None):
        """Train the TB classification model using ablation architecture."""
        logger.info(f"Starting TB training with ablation model: {self.model_type}")
        logger.info(f"Active tasks: {self.active_tasks}")
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
        """Evaluate the best model on training, validation, and test sets with TB focus."""
        logger.info("Evaluating best TB ablation model on all splits...")
        
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
            # Evaluation-only mode
            self.model = create_ablation_model(self.model_type, self.config)
            
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
                    title=f"TB Classification Performance - {self.model_type}"
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
                            title=f"{pathology_name} Classification Performance - {self.model_type}"
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
        """Run detailed evaluation on a data loader with TB focus."""
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
                    elif 'tb_logits' in outputs:  
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
                if 'backbone_optimizer_state_dict' in checkpoint and self.backbone_optimizer:
                    self.backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer_state_dict'])
                
                if 'patient_pipeline_optimizer_state_dict' in checkpoint and self.patient_pipeline_optimizer:
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
                if 'backbone_scaler_state_dict' in checkpoint and hasattr(self, 'backbone_scaler'):
                    self.backbone_scaler.load_state_dict(checkpoint['backbone_scaler_state_dict'])
                
                if 'patient_pipeline_scaler_state_dict' in checkpoint and hasattr(self, 'patient_pipeline_scaler'):
                    self.patient_pipeline_scaler.load_state_dict(checkpoint['patient_pipeline_scaler_state_dict'])
                
                if 'pathology_scalers_state_dicts' in checkpoint:
                    for scaler, state_dict in zip(self.pathology_scalers, checkpoint['pathology_scalers_state_dicts']):
                        scaler.load_state_dict(state_dict)
                
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

import yaml
import argparse
import os

class Config:
    def __init__(self, params=None):
        # Initialize with default values first
        self._set_defaults()
        
        # Update config with provided parameters
        if params:
            for key, value in params.items():
                setattr(self, key, value)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Data paths - these will be overridden by YAML
        self.train = True
        self.root_dir = "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v3"
        self.labels_csv = "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v3/labels/labels.csv"
        self.file_metadata_csv = "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v3/processed_files_2.csv"
        self.split_csv = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/test_files/Fold_0.csv'

        self.video_folder = 'videos'
        self.image_folder = 'images'
        
        # Model config
        self.model_type = 'no_rl'  # Default ablation model
        self.model_name = 'ablation_tb_classifier_fold0'
        self.backbone = 'resnet18'
        self.freeze_backbone = False
        self.hidden_dim = 512
        self.dropout_rate = 0.3
        self.num_pathologies = 4
        self.pretrained = True
        self.num_classes = 1
        self.in_channels = 3
        self.reset_optimizers = False
        
        # Data preprocessing
        self.target_height = 224
        self.target_width = 224
        self.depth_filter = '15'
        self.frame_sampling = 32
        self.num_sites = 15
        self.mode = 'video'
        self.pooling = 'attention'
        
        # Training settings
        self.task = "TB Label"
        self.batch_size = 2
        self.num_workers = 6
        self.learning_rate = 0.00001
        self.weight_decay = 0.00001
        self.num_epochs = 20
        self.early_stopping_patience = 8
        self.accumulation_steps = 8
        self.use_amp = True
        self.seed = 42
        
        # Multi-task config for compatibility
        self.active_tasks = ['TB Label']
        self.use_pathology_loss = True
        self.task_weights = {'TB Label': 1.0}
        
        # New dataset parameters
        self.files_per_site = 1
        self.site_order = None
        self.pad_missing_sites = True
        self.max_sites = 15
        
        self.classification_type = "binary"
        self.pos_weight = 1.4
        
        # Evaluation settings
        self.eval_metric = "auc"
        self.eval_metric_goal = "max"
        self.evaluate_best_valid_model = True

        self.local_weights_dir = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/NetworkArchitecture/CLIP_weights'
        
        # Optimizer settings for new training structure
        self.backbone_lr = 0.00001
        self.backbone_weight_decay = 0.00001
        self.backbone_T_0 = 4
        self.backbone_T_mult = 2
        self.backbone_eta_min = 1e-6
        
        self.pathology_lr = 0.0001
        self.pathology_weight_decay = 0.00001
        self.pathology_T_0 = 4
        self.pathology_T_mult = 2
        self.pathology_eta_min = 1e-6
        
        self.patient_pipeline_lr = 0.001
        self.patient_pipeline_weight_decay = 0.00001
        self.patient_pipeline_T_0 = 4
        self.patient_pipeline_T_mult = 2
        self.patient_pipeline_eta_min = 1e-6
        
        # Directories
        self.log_dir = "logs"
        self.save_dir = "models"
        self.checkpoint_dir = "checkpoints"
        self.pred_save_dir = "predictions"
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathology-specific positive weights
        self.pathology_pos_weights = [1.0, 4.0, 4.0, 4.0]
        
        # Pathology class names
        self.pathology_classes = [
            'A-line',
            'Large consolidations', 
            'Pleural Effusion',
            'Other Pathology'
        ]
        
        # Create required directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.pred_save_dir, exist_ok=True)
    
    def load_from_yaml(self, yaml_path):
        """Load configuration from YAML file and override defaults."""
        if not os.path.exists(yaml_path):
            logger.warning(f"Config file not found: {yaml_path}")
            return
            
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            logger.info(f"Loading configuration from {yaml_path}")
            
            # Update config with YAML values
            for key, value in yaml_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"  Updated {key}: {value}")
                else:
                    logger.warning(f"  Unknown config key: {key}")
            
            # Create experiment directory from loaded config
            if hasattr(self, 'experiment_dir'):
                os.makedirs(self.experiment_dir, exist_ok=True)
            else:
                # Create experiment-specific directories from model_name
                self.experiment_dir = os.path.join(self.checkpoint_dir, self.model_name)
                os.makedirs(self.experiment_dir, exist_ok=True)
            
            logger.info(f"Configuration successfully loaded from {yaml_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {yaml_path}: {e}")
            raise e
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
    
    def save(self, path):
        """Save configuration to YAML file."""
        try:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
            raise e


def parse_args_and_load_config():
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(description='Training for TB detection using ablation models.')
    
    # Config file argument
    parser.add_argument('--config', type=str, required=False, 
                       help='Path to config YAML file')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, help='Ablation model type', 
                      choices=['no_rl', 'mean_pool', 'attention_pool', 'single_task', 
                              '3d_cnn', 'cnn_lstm', 'video_transformer'])
    
    # Training arguments
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Model loading arguments
    parser.add_argument('--model_weights', type=str, help='Path to model weights')
    parser.add_argument('--best_model_path', type=str, help='Path to best model for evaluation')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    
    # Mode arguments
    parser.add_argument('--train', action='store_true', default=True, help='Train mode')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only mode')
    
    args = parser.parse_args()
    
    # Create config with defaults
    config = Config()
    
    # Load YAML config if provided
    if args.config:
        if os.path.exists(args.config):
            config.load_from_yaml(args.config)
        else:
            logger.error(f"Config file not found: {args.config}")
            raise FileNotFoundError(f"Config file not found: {args.config}")
    else:
        logger.info("No config file provided, using defaults")
    
    # Override with command-line arguments (these take precedence)
    if args.model_type is not None:
        config.model_type = args.model_type
        logger.info(f"Override model_type: {args.model_type}")
        
    if args.lr is not None:
        config.learning_rate = args.lr
        logger.info(f"Override learning_rate: {args.lr}")
        
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        logger.info(f"Override batch_size: {args.batch_size}")
        
    if args.epochs is not None:
        config.num_epochs = args.epochs
        logger.info(f"Override num_epochs: {args.epochs}")
        
    if args.seed is not None:
        config.seed = args.seed
        logger.info(f"Override seed: {args.seed}")
        
    if args.model_weights is not None:
        config.model_weights = args.model_weights
        logger.info(f"Override model_weights: {args.model_weights}")
        
    if args.best_model_path is not None:
        config.best_model_path = args.best_model_path
        logger.info(f"Override best_model_path: {args.best_model_path}")
        
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
        logger.info(f"Override resume_from_checkpoint: {args.resume_from_checkpoint}")
        
    if args.eval_only:
        config.train = False
        logger.info("Override train: False (evaluation only)")
    
    return config

def main():
    """Main training function with command line argument support."""
    
    # Parse command line arguments and load configuration
    config = parse_args_and_load_config()
    
    logger.info("TB Ablation Classification Configuration:")
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
    trainer = AblationTrainer(config)
    
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
    logger.info(f"Starting TB training with ablation model: {config.model_type}...")
    best_metric, best_epoch = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training complete! Best metric: {best_metric:.4f} at epoch {best_epoch+1}")
    return best_metric, best_epoch


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in training: {e}")
        raise



# # 3D CNN ablation
# python3 train_ablation.py --config configs/3dcnn/fold0.yaml

# # CNN-LSTM ablation
# python3 train_ablation.py --config configs/cnnlstm/fold0.yaml

# # Video Transformer (ViViT) ablation
# python3 train_ablation.py --config configs/vivit/fold0.yaml


# # Attention pooling ablation
# python3 train_ablation.py --config configs/attention_pool/fold0.yaml

# # Mean pooling ablation
# python3 train_ablation.py --config configs/mean_pool/fold4.yaml

# # Single task ablation
# python3 train_ablation.py --config configs/singletask/fold0.yaml

# # Uniform/No-RL ablation
# python3 train_ablation.py --config configs/uniform/tb_drl_mil_Final_fold0.yaml

