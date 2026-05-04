import os
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_NET'] = 'Socket'
os.environ['NCCL_NET_PLUGIN'] = 'none'
os.environ['NCCL_SOCKET_IFNAME'] = os.environ.get('ULTRAI_NCCL_SOCKET_IFNAME', 'lo')
os.environ['NCCL_PLUGIN_P2P'] = '0'
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
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:  # type: ignore
        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            logging.getLogger(__name__).warning(
                "tensorboard not available; SummaryWriter calls will be no-ops"
            )

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def close(self):
            pass
from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    class _PlotStub:
        def __init__(self, name):
            logging.getLogger(__name__).warning(
                "%s not available; plotting calls will be no-ops", name
            )

        def __getattr__(self, attr):
            def _noop(*args, **kwargs):
                return None
            return _noop

    plt = _PlotStub("matplotlib")
    sns = _PlotStub("seaborn")
from datetime import timedelta
try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None

from ultrai.data.registry import load_dataset_adapter
try:
    from NetworkArchitecture.dinov3_prototype_finetune import (
        DINOv3PrototypeClusterer,
        PrototypeLossWeights,
        SiteVideoDataset,
        collate_site_video_batch,
    )
except Exception:
    DINOv3PrototypeClusterer = None  # type: ignore
    PrototypeLossWeights = None  # type: ignore
    SiteVideoDataset = None  # type: ignore
    collate_site_video_batch = None  # type: ignore

try:
    from NetworkArchitecture.dinov3_prompt_policy import DINOv3PromptPolicy, PromptPolicyConfig
    from NetworkArchitecture.dinov3_bert_model import DINOv3VisionEncoder
    from sam_video.wrapper import build_sam_video_wrapper, Prompts
    from supervision.medgemma_supervisor import MedGemmaSupervisor, MedGemmaSupervisorConfig
    from supervision.reward_model import RewardCNN, RewardModelConfig, reward_loss, MedGemmaJSONLDataset
except Exception:
    DINOv3PromptPolicy = None  # type: ignore
    PromptPolicyConfig = None  # type: ignore
    DINOv3VisionEncoder = None  # type: ignore
    build_sam_video_wrapper = None  # type: ignore
    Prompts = None  # type: ignore
    MedGemmaSupervisor = None  # type: ignore
    MedGemmaSupervisorConfig = None  # type: ignore
    RewardCNN = None  # type: ignore
    RewardModelConfig = None  # type: ignore
    reward_loss = None  # type: ignore
    MedGemmaJSONLDataset = None  # type: ignore
try:
    from NetworkArchitecture.ablation_models import create_ablation_model
except ImportError:
    def create_ablation_model(*args, **kwargs):  # type: ignore
        raise ImportError(
            "NetworkArchitecture.ablation_models could not be imported. "
            "Ensure Transformers and related vision backbones are installed."
        )

try:
    from NetworkArchitecture.multi_backbone_ensemble import ensemble_distillation_loss
except ImportError:
    ensemble_distillation_loss = None  # type: ignore

try:
    from NetworkArchitecture.monitoring_utils import log_model_component_status
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Monitoring utilities not available")
    log_model_component_status = lambda *args: None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_DATASET_ADAPTER = load_dataset_adapter("benin")


def set_dataset_adapter(dataset_name_or_module):
    global _DATASET_ADAPTER
    if isinstance(dataset_name_or_module, str):
        _DATASET_ADAPTER = load_dataset_adapter(dataset_name_or_module)
    else:
        _DATASET_ADAPTER = dataset_name_or_module


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # Set device BEFORE initializing process group
        torch.cuda.set_device(local_rank)
        
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "world_size": world_size,
            "rank": rank,
            "timeout": timedelta(minutes=30),
        }
        dist.init_process_group(**init_kwargs)
        
        if rank == 0:
            logger.info(f"Distributed training initialized: {world_size} GPUs")
            logger.info(f"NCCL version: {torch.cuda.nccl.version()}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def distributed_barrier():
    """Synchronize ranks without triggering NCCL's unknown-device warning."""
    if not dist.is_initialized():
        return

    barrier_kwargs = {}
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        barrier_kwargs["device_ids"] = [torch.cuda.current_device()]

    dist.barrier(**barrier_kwargs)


def reduce_dict(input_dict, average=True):
    """
    Reduce values in a dictionary across all processes.
    
    Args:
        input_dict: Dictionary with values to reduce
        average: Whether to average or sum the values
    """
    if not dist.is_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    
    return reduced_dict


# ============================================================================
# DDP Helper: Gather and Concatenate Numpy Arrays
# ============================================================================
def _ddp_concat_numpy(local_arr):
    """
    Gather numpy arrays from all ranks and concatenate along axis 0.
    If not in distributed mode, returns local_arr.
    """
    if not dist.is_initialized():
        return local_arr
    world_size = dist.get_world_size()
    parts = [None] * world_size
    # all_gather_object works with NCCL/Gloo and arbitrary Python objects
    dist.all_gather_object(parts, local_arr)
    parts = [p for p in parts if p is not None and len(p) > 0]
    if len(parts) == 0:
        return local_arr
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def _ddp_concat_list(local_items):
    """
    Gather Python lists from all ranks and concatenate them.
    If not in distributed mode, returns a local list copy.
    """
    local_items = list(local_items)
    if not dist.is_initialized():
        return local_items
    world_size = dist.get_world_size()
    parts = [None] * world_size
    dist.all_gather_object(parts, local_items)
    merged = []
    for part in parts:
        if part:
            merged.extend(part)
    return merged


# ============================================================================
# Configuration Class
# ============================================================================

class Config:
    def __init__(self, params=None):
        """Initialize configuration with defaults and optional overrides."""
        self._set_defaults()
        
        if params:
            for key, value in params.items():
                setattr(self, key, value)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Training mode
        self.train = True
        # Execution mode (do NOT use self.mode; reserved for dataset mode: video/image/both)
        # - 'train': standard ablation training (default)
        # - 'finetune_prototypes': finetune DINOv3 patch prototypes for shared clusters
        self.run_mode = "train"
        
        # Data paths
        self.root_dir = ""
        self.labels_csv = ""
        self.file_metadata_csv = ""
        self.split_csv = ''
        self.video_folder = 'videos'
        self.image_folder = 'images'
        
        # Model config
        self.model_type = 'no_rl'
        self.model_name = 'ablation_tb_classifier_fold0'
        self.backbone = 'resnet18'
        self.freeze_backbone = False
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.vision_pretrained_weights = None
        self.clip_unfreeze_last_n_layers = 1
        self.train_visual_projection = True
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
        self.legacy_hmv_mil_loss = False
        self.legacy_hmv_mil_phase_updates = False
        self.pathology_aux_weight = 0.2
        self.pathology_aux_pos_weights = [1.0, 4.0, 4.0, 4.0, 15.0]
        
        self.active_tasks = ['TB Label']
        self.use_pathology_loss = True
        self.task_weights = {'TB Label': 1.0}
        
        # Dataset parameters
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
        
        self.local_weights_dir = '/NetworkArchitecture/CLIP_weights'
        
        # Optimizer settings
        self.backbone_lr = 0.00001
        self.backbone_weight_decay = 0.00001
        self.backbone_eta_min = 1e-6
        # Backbone update schedule / OOM handling
        self.backbone_update_every = 2
        self.backbone_max_oom = 3
        self.disable_backbone_on_oom = True
        
        self.pathology_lr = 0.0001
        self.pathology_weight_decay = 0.00001
        self.pathology_eta_min = 1e-6
        
        self.patient_pipeline_lr = 0.001
        self.patient_pipeline_weight_decay = 0.00001
        self.patient_pipeline_eta_min = 1e-6
        
        # Directories
        self.log_dir = "logs"
        self.save_dir = "models"
        self.checkpoint_dir = "checkpoints"
        self.pred_save_dir = "predictions"
        self.checkpoint_base_dir = "/capstor/store/cscs/swissai/a127/ultr-ai"
        self.experiment_dir = None  # Will be set based on experiment_name
        
        # Pathology settings
        self.pathology_pos_weights = [1.0, 4.0, 4.0, 4.0]
        self.pathology_classes = [
            'A-line',
            'Large consolidations', 
            'Pleural Effusion',
            'Other Pathology'
        ]
        
        # Distributed training settings
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        
        # Device (will be set based on local_rank)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prototype finetuning defaults
        self.num_prototypes = 16
        self.prototype_temperature = 0.1
        self.prototype_proj_dim = 256
        self.prototype_loss_conf = 1.0
        self.prototype_loss_balance = 1.0
        self.prototype_loss_path = 1.0
        self.prototype_loss_temporal = 0.0
        self.finetune_epochs = 5
        self.finetune_batch_size = 8
        # Unfreeze last N DINOv3 blocks during finetuning (None => follow existing freeze policy)
        self.dinov3_unfreeze_last_blocks = 0
        # Holdout fraction for prototype finetune evaluation (0 => no split)
        self.prototype_val_fraction = 0.2

        # MedGemma + SAM-video pipeline defaults
        self.medgemma_model_id = "google/medgemma-1.5-4b-it"
        self.medgemma_jsonl_out = "./medgemma_supervision.jsonl"
        self.medgemma_alpha = 0.5
        self.reward_jsonl_in = "./medgemma_supervision.jsonl"
        self.reward_ckpt_out = "./reward_model.pt"
        self.reward_ckpt_in = "./reward_model.pt"
        self.prompt_policy_ckpt_out = "./prompt_policy.pt"
        self.prompt_policy_ckpt_in = "./prompt_policy.pt"
        self.prompt_policy_lr = 1e-4
        self.reward_lr = 1e-4
        self.prompt_train_steps = 200
        self.reward_train_epochs = 1
        self.score_num_frames = 5

    def load_from_yaml(self, yaml_path):
        """Load configuration from YAML file (upstream-friendly)."""
        if not os.path.exists(yaml_path):
            if is_main_process():
                logger.warning(f"Config file not found: {yaml_path}")
            return
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
            if is_main_process():
                logger.info(f"Loading configuration from {yaml_path}")

            # Accept ALL keys (no unknown-key warnings)
            for key, value in yaml_config.items():
                setattr(self, key, value)

            # Ensure experiment_dir is set after potential overrides
            if not getattr(self, 'experiment_dir', None):
                self.experiment_dir = os.path.join(self.checkpoint_dir, self.model_name)

            # Normalize output directories to external /capstor location
            CAPSTOR_ROOT = os.environ.get(
                "CAPSTOR_ROOT",
                "/capstor/store/cscs/swissai/a127/ultr-ai"
            )

            def _to_capstor_path(p):
                if not isinstance(p, str) or not p:
                    return p
                if p.startswith('/'):
                    return p
                if p.startswith('capstor/'):
                    return os.path.join(CAPSTOR_ROOT, p[len('capstor/'):])
                if p.startswith('./capstor/'):
                    return os.path.join(CAPSTOR_ROOT, p[len('./capstor/'):])
                return p

            for key in ['experiment_dir', 'checkpoint_dir', 'log_dir', 'save_dir', 'pred_save_dir']:
                if hasattr(self, key):
                    setattr(self, key, _to_capstor_path(getattr(self, key)))

            if is_main_process():
                logger.info(f"Configuration successfully loaded from {yaml_path}")
        except Exception as e:
            if is_main_process():
                logger.error(f"Error loading config from {yaml_path}: {e}")
            raise e

    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and not callable(v)}

    def save(self, path):
        """Save configuration to YAML file."""
        if not is_main_process():
            return
        try:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
            raise e


def _compute_4way_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> Dict[str, Any]:
    """Compute confusion matrix + per-class precision/recall/F1 + accuracy (numpy-only)."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    per_class: Dict[int, Dict[str, float]] = {}
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec / (prec + rec))) if (prec + rec) > 0 else 0.0
        support = int(cm[c, :].sum())
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": float(support)}

    bal_acc = float(np.mean([per_class[c]["recall"] for c in range(num_classes)]))
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "confusion_matrix": cm, "per_class": per_class}


@torch.no_grad()
def _eval_prototype_model(model, loader, device, world_size: int) -> Dict[str, Any]:
    """Evaluate 4-way pathology head and gather across ranks if DDP."""
    model.eval()
    all_true = []
    all_pred = []
    for videos, labels, _metas in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outs = model(videos) if not isinstance(model, DDP) else model.module(videos)
        logits = outs["path_logits"]
        pred = torch.argmax(logits, dim=-1)
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0) if len(all_true) else np.zeros((0,), dtype=int)
    y_pred = np.concatenate(all_pred, axis=0) if len(all_pred) else np.zeros((0,), dtype=int)

    y_true = _ddp_concat_numpy(y_true)
    y_pred = _ddp_concat_numpy(y_pred)
    return _compute_4way_metrics_numpy(y_true, y_pred, num_classes=4)



def train_prototype_finetune(config, rank=0, world_size=1, local_rank=0):
        """
        Finetune DINOv3 patch prototypes for stable, shared clusters across videos.
        Runs under DDP when world_size > 1.
        """
        if DINOv3PrototypeClusterer is None or SiteVideoDataset is None:
            raise ImportError(
                "Prototype finetuning dependencies missing. Ensure "
                "NetworkArchitecture/dinov3_prototype_finetune.py imports correctly."
            )

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        config.device = device

        # Ensure DINOv3 unfreeze policy is applied via existing backbone freeze mechanism
        if getattr(config, "dinov3_unfreeze_last_blocks", 0) and int(config.dinov3_unfreeze_last_blocks) > 0:
            setattr(config, "dinov3_trainable_blocks", int(config.dinov3_unfreeze_last_blocks))

        video_folder = getattr(config, "video_folder", "videos")
        if not os.path.isabs(video_folder):
            video_folder = os.path.join(getattr(config, "root_dir", "."), video_folder)

        full_dataset = SiteVideoDataset(
            labels_csv=getattr(config, "labels_csv"),
            file_metadata_csv=getattr(config, "file_metadata_csv"),
            video_folder=video_folder,
            frame_sampling=int(getattr(config, "frame_sampling", 32)),
            resize_size=(int(getattr(config, "target_height", 224)), int(getattr(config, "target_width", 224))),
            depth_filter=str(getattr(config, "depth_filter", "15")),
        )

        # Optional train/val split to report classification performance
        val_fraction = float(getattr(config, "prototype_val_fraction", 0.0) or 0.0)
        if val_fraction > 0:
            n = len(full_dataset)
            n_val = max(1, int(n * val_fraction))
            n_train = max(1, n - n_val)
            gen = torch.Generator().manual_seed(int(getattr(config, "seed", 42)))
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=gen)
        else:
            train_dataset, val_dataset = full_dataset, None

        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(getattr(config, "finetune_batch_size", 8)),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(getattr(config, "num_workers", 4)),
            pin_memory=True,
            collate_fn=collate_site_video_batch,
        )
        val_loader = None
        if val_dataset is not None:
            val_sampler = None
            if world_size > 1:
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(getattr(config, "finetune_batch_size", 8)),
                shuffle=False,
                sampler=val_sampler,
                num_workers=int(getattr(config, "num_workers", 4)),
                pin_memory=True,
                collate_fn=collate_site_video_batch,
            )

        loss_weights = PrototypeLossWeights(
            conf=float(getattr(config, "prototype_loss_conf", 1.0)),
            balance=float(getattr(config, "prototype_loss_balance", 1.0)),
            path=float(getattr(config, "prototype_loss_path", 1.0)),
            temporal=float(getattr(config, "prototype_loss_temporal", 0.0)),
        )

        model = DINOv3PrototypeClusterer(
            config=config,
            num_prototypes=int(getattr(config, "num_prototypes", 16)),
            proj_dim=int(getattr(config, "prototype_proj_dim", 256)),
            temperature=float(getattr(config, "prototype_temperature", 0.1)),
            loss_weights=loss_weights,
            num_pathology_classes=int(getattr(config, "num_pathologies", 4)),
        ).to(device)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

        opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(config, "learning_rate", 1e-5)), weight_decay=float(getattr(config, "weight_decay", 1e-5)))

        use_amp = bool(getattr(config, "use_amp", True)) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        epochs = int(getattr(config, "finetune_epochs", 5))
        if is_main_process():
            logger.info("Starting prototype finetune: epochs=%d, K=%d, proj_dim=%d, temp=%.4f",
                        epochs, getattr(config, "num_prototypes", 16), getattr(config, "prototype_proj_dim", 256), getattr(config, "prototype_temperature", 0.1))

        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            model.train()
            running = {}
            n_batches = 0

            for videos, labels, _metas in train_loader:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outs = model(videos) if not isinstance(model, DDP) else model.module(videos)
                    loss, metrics = (model.module.compute_losses(outs, labels) if isinstance(model, DDP) else model.compute_losses(outs, labels))

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + float(v)
                n_batches += 1

            if is_main_process():
                avg = {k: v / max(n_batches, 1) for k, v in running.items()}
                logger.info("Proto finetune epoch %d/%d: %s", epoch + 1, epochs, avg)

            # Classification evaluation (val if available, else train) to assess performance
            eval_loader = val_loader if val_loader is not None else train_loader
            eval_split = "val" if val_loader is not None else "train"
            metrics = _eval_prototype_model(model, eval_loader, device, world_size)
            if is_main_process():
                logger.info(
                    "Proto pathology eval (%s): acc=%.4f bal_acc=%.4f",
                    eval_split,
                    metrics["accuracy"],
                    metrics["balanced_accuracy"],
                )
                logger.info("Proto pathology eval confusion_matrix:\n%s", metrics["confusion_matrix"])
                logger.info("Proto pathology eval per_class: %s", metrics["per_class"])

            if world_size > 1:
                distributed_barrier()

        # Save checkpoint (rank 0)
        if is_main_process():
            out_dir = os.path.join(getattr(config, "experiment_dir", "."), "finetune_prototypes")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "prototype_ckpt.pt")
            to_save = model.module if isinstance(model, DDP) else model
            to_save.save_checkpoint(ckpt_path, extra={"config": config.to_dict() if hasattr(config, "to_dict") else {}})
            logger.info("Saved prototype checkpoint to %s", ckpt_path)

        if world_size > 1:
            distributed_barrier()

        return True
# ============================================================================
# Ablation Trainer with Full Distributed Support
# ============================================================================

class AblationTrainer:
    """
    Ablation model trainer with full distributed training support.
    Supports multi-GPU and multi-node training with proper gradient accumulation.
    """
    
    def __init__(self, config, rank=0, world_size=1, local_rank=0):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        
        # Set device based on local rank
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        config.device = self.device
        
        # Only log from main process
        if is_main_process():
            # Main process: ensure INFO level
            logging.getLogger().setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
            print("✓ Main process: Logger set to INFO level")
        else:
            # Other processes: reduce to WARNING
            logging.getLogger().setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)
            print(f"  Worker process {rank}: Logger set to WARNING level")
            
        print(f"\n{'='*80}")
        print(f"PROCESS INITIALIZATION:")
        print(f"  Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        print(f"  is_distributed: {self.is_distributed}")
        print(f"  is_main_process(): {is_main_process()}")
        print(f"{'='*80}\n")
    
        
        # Training state
        self.best_metric = None
        self.best_epoch = 0
        self.epoch = 0
        self.epochs_without_improvement = 0
        self.backbone_oom_count = 0
        self.disable_backbone_updates = False
        
        # Model configuration
        self.model_type = getattr(config, 'model_type', 'no_rl')
        self.active_tasks = list(getattr(config, 'active_tasks', ['TB Label']))
        self.use_pathology_loss = bool(getattr(config, 'use_pathology_loss', True))

        # Task weights (dict) and positive-class weights
        self.task_weights = dict(getattr(config, 'task_weights', {'TB Label': 1.0}))

        if hasattr(config, 'task_pos_weights'):
            # YAML dict, e.g., {"TB Label": 1.4}
            self.task_pos_weights = dict(getattr(config, 'task_pos_weights'))
        else:
            # Fallback: scalar pos_weight applied to all active tasks
            self.task_pos_weights = {task: float(getattr(config, 'pos_weight', 1.0))
                                    for task in self.active_tasks}

        # Optional global pathology loss weight from YAML
        self.pathology_weight = float(getattr(config, 'pathology_weight', 1.0))
        self.legacy_hmv_mil_phase_updates = bool(
            getattr(config, 'legacy_hmv_mil_phase_updates', False)
        )
        
        if is_main_process():
            logger.info(f"Using ablation model type: {self.model_type}")
            logger.info(f"Pathology loss enabled: {self.use_pathology_loss}")
            logger.info(f"Legacy HMV-MIL phase updates: {self.legacy_hmv_mil_phase_updates}")
            logger.info(f"Distributed training: {self.is_distributed} (world_size={world_size})")
        
        self._set_seed(config.seed)
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        # Add rank to seed for data loading diversity
        seed = seed + self.rank
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_weighted_train_sampler(self, dataset):
        """Oversample TB-positive patients at the sampler level."""
        patients = getattr(dataset, 'patients', None)
        if not patients:
            return None

        tb_labels = []
        for patient in patients:
            patient_labels = patient.get('patient_labels', {}) if isinstance(patient, dict) else {}
            try:
                tb_labels.append(int(patient_labels.get('TB Label', -1)))
            except (TypeError, ValueError):
                tb_labels.append(-1)

        tb_labels = np.asarray(tb_labels, dtype=np.int64)
        positive_mask = tb_labels == 1
        negative_mask = tb_labels == 0

        num_positive = int(positive_mask.sum())
        num_negative = int(negative_mask.sum())

        if num_positive == 0 or num_negative == 0:
            if is_main_process():
                logger.warning(
                    "Skipping weighted oversampling because class counts are pos=%d, neg=%d",
                    num_positive,
                    num_negative,
                )
            return None

        positive_multiplier = max(1, int(getattr(self.config, 'positive_class_multiplier', 1)))
        sample_weights = np.ones(len(tb_labels), dtype=np.float64)
        sample_weights[positive_mask] = float(positive_multiplier)

        num_samples = num_negative + (num_positive * positive_multiplier)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )

        if is_main_process():
            logger.info(
                "Using WeightedRandomSampler for train set: pos=%d, neg=%d, multiplier=%d, samples/epoch=%d",
                num_positive,
                num_negative,
                positive_multiplier,
                num_samples,
            )

        return sampler
    
    def _setup_data(self):
        """Set up the data module with distributed samplers."""
        data_module_cls = _DATASET_ADAPTER.LungUltrasoundDataModule
        collate_patient_batch = _DATASET_ADAPTER.collate_patient_batch

        self.data_module = data_module_cls(
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
        
        # Create distributed samplers if using multiple GPUs
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.data_module.patient_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.config.seed,
                drop_last=True  # Important for consistent batch sizes
            )
            self.val_sampler = DistributedSampler(
                self.data_module.patient_val,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            self.test_sampler = DistributedSampler(
                self.data_module.patient_test,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
            if getattr(self.config, 'oversample_positive_class', False):
                self.train_sampler = self._build_weighted_train_sampler(self.data_module.patient_train)
        
        # Create data loaders
        train_shuffle = (self.train_sampler is None)
        if len(self.data_module.patient_train) == 0:
            train_shuffle = False
        
        self.train_loader = DataLoader(
            self.data_module.patient_train,
            batch_size=self.config.batch_size,
            sampler=self.train_sampler,                 # DDP sampler
            shuffle=train_shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_patient_batch,
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        self.val_loader = DataLoader(
            self.data_module.patient_val,
            batch_size=self.config.batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_patient_batch,          
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        self.test_loader = DataLoader(
            self.data_module.patient_test,
            batch_size=self.config.batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_patient_batch,           
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
        
        if is_main_process():
            logger.info(f"Training dataset size: {len(self.data_module.patient_train)}")
            logger.info(f"Validation dataset size: {len(self.data_module.patient_val)}")
            logger.info(f"Test dataset size: {len(self.data_module.patient_test)}")
            logger.info(f"Training batches per epoch: {len(self.train_loader)}")
            if isinstance(self.train_sampler, WeightedRandomSampler):
                logger.info(f"Weighted sampler samples per epoch: {self.train_sampler.num_samples}")
    
    def _setup_model(self):
        """Set up the ablation model with DDP support."""
        self.model = create_ablation_model(self.model_type, self.config)
        
        # Load pretrained weights if provided
        if hasattr(self.config, 'model_weights') and self.config.model_weights:
            try:
                checkpoint = torch.load(
                    self.config.model_weights,
                    map_location=self.device,
                    weights_only=False,
                )
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from previous DDP training)
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict)
                
                if is_main_process():
                    logger.info(f"Loaded model weights from {self.config.model_weights}")
            except Exception as e:
                if is_main_process():
                    logger.error(f"Failed to load pretrained weights: {e}")
                    logger.error("Continuing with randomly initialized weights")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if using distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True  # Set to False if all params are always used
            )
            if is_main_process():
                logger.info("Model wrapped with DistributedDataParallel")
        
        # Get the underlying model (unwrapped)
        self.model_without_ddp = self.model.module if self.is_distributed else self.model
        
        # Optional: teacher model for distillation (distilled_edge)
        self.teacher_model = None
        if (
            self.model_type == "distilled_edge"
            and hasattr(self.config, "teacher_checkpoint")
            and self.config.teacher_checkpoint
        ):
            try:
                import copy as _copy

                teacher_config = _copy.deepcopy(self.config)
                teacher_config.model_type = "multi_backbone_ensemble"
                self.teacher_model = create_ablation_model(
                    "multi_backbone_ensemble",
                    teacher_config,
                )

                # Load teacher checkpoint
                checkpoint = torch.load(
                    self.config.teacher_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint

                if list(state_dict.keys())[0].startswith("module."):
                    state_dict = {
                        k.replace("module.", ""): v for k, v in state_dict.items()
                    }

                self.teacher_model.load_state_dict(state_dict, strict=False)
                self.teacher_model = self.teacher_model.to(self.device)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

                if is_main_process():
                    logger.info(
                        f"Loaded teacher model for distillation from {self.config.teacher_checkpoint}"
                    )
            except Exception as e:
                if is_main_process():
                    logger.error(
                        f"Failed to initialize teacher model for distillation: {e}"
                    )
                self.teacher_model = None

        trainable_params = sum(
            p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad
        )
        
        if is_main_process():
            logger.info(f"Number of trainable parameters: {trainable_params:,}")
            try:
                log_model_component_status(self.model_without_ddp, logger)
            except:
                pass
    
    def _setup_training(self):
        """Set up optimizers and schedulers."""
        # Organize parameters by component
        backbone_params = []
        pathology_params = []
        patient_pipeline_params = []
        task_classifier_params = []
        
        num_pathology_modules = 0
        if hasattr(self.model_without_ddp, 'pathology_modules') and \
           self.model_without_ddp.pathology_modules:
            num_pathology_modules = len(self.model_without_ddp.pathology_modules)
        
        pathology_module_params = [[] for _ in range(num_pathology_modules)]
        
        # Categorize parameters
        for name, param in self.model_without_ddp.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(component in name for component in [
                'vision_encoder', 'cnn_backbone', 'video_transformer', 'backbone',
                'multi_feature_extraction', 'multi_scale_extraction'
            ]):
                backbone_params.append(param)
            elif 'pathology_modules' in name and self.use_pathology_loss:
                for i in range(num_pathology_modules):
                    if f'pathology_modules.{i}' in name or f'pathology_modules[{i}]' in name:
                        pathology_module_params[i].append(param)
                        break
            elif 'task_classifiers' in name or 'tb_classifier' in name:
                task_classifier_params.append(param)
            elif any(component in name for component in [
                'site_integration', 'patient_mil', 'cross_site_attention', 'frame_selector',
                'lstm_temporal_pool', 'site_embedding', 'site_embedding_proj'
            ]):
                patient_pipeline_params.append(param)
            else:
                patient_pipeline_params.append(param)
        
        # Create optimizers
        if backbone_params:
            self.backbone_optimizer = optim.AdamW(
                backbone_params,
                lr=getattr(self.config, 'backbone_lr', 0.00001),
                weight_decay=getattr(self.config, 'backbone_weight_decay', 0.00001)
            )
            if is_main_process():
                logger.info(f"Created backbone optimizer with {len(backbone_params)} parameters")
        else:
            self.backbone_optimizer = None
            if is_main_process():
                logger.info("No backbone parameters found")
        
        # Pathology optimizers
        self.pathology_optimizers = []
        if self.use_pathology_loss and num_pathology_modules > 0:
            for i, module_params in enumerate(pathology_module_params):
                if module_params:
                    optimizer = optim.AdamW(
                        module_params,
                        lr=getattr(self.config, 'pathology_lr', 0.0001),
                        weight_decay=getattr(self.config, 'pathology_weight_decay', 0.00001)
                    )
                    self.pathology_optimizers.append(optimizer)
            if is_main_process():
                logger.info(f"Created {len(self.pathology_optimizers)} pathology optimizers")
        
        # Patient pipeline optimizer
        all_patient_params = patient_pipeline_params + task_classifier_params
        if all_patient_params:
            self.patient_pipeline_optimizer = optim.AdamW(
                all_patient_params,
                lr=getattr(self.config, 'patient_pipeline_lr', 0.001),
                weight_decay=getattr(self.config, 'patient_pipeline_weight_decay', 0.00001)
            )
            if is_main_process():
                logger.info(f"Created patient pipeline optimizer with {len(all_patient_params)} parameters")
        else:
            self.patient_pipeline_optimizer = None
            if is_main_process():
                logger.info("No patient pipeline parameters found")
        
        # Set up schedulers
        self.schedulers = []
        batches_per_epoch = len(self.train_loader)
        total_steps = self.config.num_epochs * batches_per_epoch
        if batches_per_epoch == 0:
            if is_main_process():
                logger.warning(
                    "Training loader has 0 batches; skipping LR scheduler setup. "
                    "Check split CSV and dataset filtering."
                )
            total_steps = 0
        
        if self.backbone_optimizer and total_steps > 0:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.backbone_optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'backbone_eta_min', 1e-6)
            ))
        
        if total_steps > 0:
            for optimizer in self.pathology_optimizers:
                self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=getattr(self.config, 'pathology_eta_min', 1e-6)
                ))
        
        if self.patient_pipeline_optimizer and total_steps > 0:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.patient_pipeline_optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'patient_pipeline_eta_min', 1e-6)
            ))
        
        # Mixed precision training
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            if self.backbone_optimizer:
                self.backbone_scaler = torch.amp.GradScaler('cuda')
            self.pathology_scalers = [torch.amp.GradScaler('cuda') for _ in self.pathology_optimizers]
            if self.patient_pipeline_optimizer:
                self.patient_pipeline_scaler = torch.amp.GradScaler('cuda')
        
        if is_main_process():
            logger.info(f"Mixed precision training: {self.use_amp}")

    def _reset_frame_selector_state(self, clear_history=False, reset_temperature=False):
        """Clear cached RL frame-selector state that can retain GPU tensors."""
        selector = getattr(self.model_without_ddp, 'frame_selector', None)
        if selector is None:
            return

        if hasattr(selector, 'saved_actions'):
            selector.saved_actions = []

        if hasattr(selector, 'reset_rewards'):
            try:
                selector.reset_rewards()
            except Exception:
                pass

        if clear_history:
            if hasattr(selector, 'clear_history'):
                try:
                    selector.clear_history()
                except Exception:
                    pass
            elif hasattr(selector, 'frame_history'):
                selector.frame_history = {}

        if reset_temperature and hasattr(selector, 'reset_temperature'):
            try:
                selector.reset_temperature()
            except Exception:
                pass
    
    def train_epoch(self, epoch):
        """
        Training epoch with proper distributed gradient accumulation.
        """
        if len(self.train_loader) == 0:
            if is_main_process():
                logger.warning("Train loader is empty; skipping epoch.")
            return 0.0, {}
        # Set epoch for distributed sampler
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        self.model.train()
        self.epoch = epoch
        
        # Initialize tracking metrics
        running_losses = {
            'total': 0.0,
            'tb_loss': 0.0,
            'backbone': 0.0,
        }
        
        if self.use_pathology_loss:
            running_losses['pathology'] = 0.0
        
        # Metrics tracking
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []
        
        accumulation_steps = self.config.accumulation_steps
        
        # Calculate effective batch size
        effective_batch_size = self.config.batch_size * accumulation_steps
        if self.is_distributed:
            effective_batch_size *= self.world_size
        
        if is_main_process():
            logger.info(f"Effective batch size: {effective_batch_size} "
                       f"(batch_size={self.config.batch_size} × "
                       f"accumulation_steps={accumulation_steps} × "
                       f"num_gpus={self.world_size})")
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            disable=not is_main_process()
        )

        def _sync_oom_flag(local_oom):
            if not self.is_distributed:
                return local_oom
            # Clear any pending CUDA operations before attempting collective
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                flag = torch.tensor(1 if local_oom else 0, device=self.device)
                dist.all_reduce(flag, op=dist.ReduceOp.SUM, async_op=False)
                return flag.item() > 0
            except Exception as e:
                # If all_reduce fails, assume OOM and try to recover
                if is_main_process():
                    logger.warning(f"Failed to sync OOM flag: {e}, assuming OOM on all ranks")
                # Try a barrier to synchronize
                try:
                    distributed_barrier()
                except:
                    pass
                return True  # Assume OOM to be safe

        def _cleanup_after_oom():
            self._reset_frame_selector_state(clear_history=True)
            if self.backbone_optimizer:
                self.backbone_optimizer.zero_grad()
            if self.patient_pipeline_optimizer:
                self.patient_pipeline_optimizer.zero_grad()
            for opt in self.pathology_optimizers:
                opt.zero_grad()
            if self.use_amp:
                if hasattr(self, 'backbone_scaler'):
                    self.backbone_scaler = torch.amp.GradScaler('cuda')
                self.pathology_scalers = [torch.amp.GradScaler('cuda') for _ in self.pathology_optimizers]
                if hasattr(self, 'patient_pipeline_scaler'):
                    self.patient_pipeline_scaler = torch.amp.GradScaler('cuda')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _set_all_requires_grad(enabled):
            for param in self.model_without_ddp.parameters():
                param.requires_grad = enabled

        def _set_requires_grad_for_components(components):
            _set_all_requires_grad(False)
            for name, param in self.model_without_ddp.named_parameters():
                if any(component in name for component in components):
                    param.requires_grad = True

        def _legacy_phase_updates_enabled():
            return bool(getattr(self.config, 'legacy_hmv_mil_phase_updates', False))
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                oom_in_batch = False
                # Determine if this is a sync step
                is_accumulation_step = (batch_idx + 1) % accumulation_steps != 0
                is_last_batch = (batch_idx + 1) == len(self.train_loader)
                should_sync = not is_accumulation_step or is_last_batch
                
                # Move data to device
                site_videos = batch['site_videos'].to(self.device, non_blocking=True)
                site_indices = batch['site_indices'].to(self.device, non_blocking=True)
                # site_masks may be missing in some DataLoader outputs; guard against KeyError
                if 'site_masks' in batch and batch['site_masks'] is not None:
                    site_masks = batch['site_masks'].to(self.device, non_blocking=True)
                else:
                    print("site_masks not found in batch; creating default mask")
                    # Create a default mask (all valid) matching site_findings or site_indices
                    if 'site_findings' in batch and batch['site_findings'] is not None:
                        sf = batch['site_findings']
                        # site_findings shape might be (B, N, P) or (B, N)
                        N = sf.shape[1] if sf.ndim >= 2 else (batch['site_indices'].shape[1] if 'site_indices' in batch else getattr(self.config, 'max_sites', 15))
                    elif 'site_indices' in batch and batch['site_indices'] is not None:
                        N = batch['site_indices'].shape[1]
                    else:
                        N = getattr(self.config, 'max_sites', 15)

                    site_masks = torch.ones((batch['site_videos'].shape[0], N), dtype=torch.bool, device=self.device)
                site_findings = batch['site_findings'].to(self.device, non_blocking=True)
                tb_labels = batch['tb_labels'].to(self.device, non_blocking=True).float()
                pneumonia_labels = torch.full_like(tb_labels, -1)
                covid_labels = torch.full_like(tb_labels, -1)
                
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
                
                # ============================================
                # 1. Pathology Modules Update
                # ============================================
                if self.use_pathology_loss and self.pathology_optimizers:
                    try:
                        if _legacy_phase_updates_enabled():
                            for path_idx in range(self.config.num_pathologies):
                                if path_idx >= len(self.pathology_optimizers):
                                    continue

                                _set_all_requires_grad(False)
                                for name, param in self.model_without_ddp.named_parameters():
                                    if (f'pathology_modules.{path_idx}' in name or
                                            f'pathology_modules[{path_idx}]' in name):
                                        param.requires_grad = True

                                if batch_idx % accumulation_steps == 0:
                                    self.pathology_optimizers[path_idx].zero_grad()

                                sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                                with sync_context():
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
                                                running_losses['pathology'] += path_loss.item()
                                            else:
                                                path_loss = path_score_i.sum() * 0.0

                                            path_loss = path_loss / accumulation_steps
                                            self.pathology_scalers[path_idx].scale(path_loss).backward()

                                        del path_outputs, path_scores, path_score_i, path_label_i, path_loss
                                    else:
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
                                            running_losses['pathology'] += path_loss.item()
                                        else:
                                            path_loss = path_score_i.sum() * 0.0

                                        path_loss = path_loss / accumulation_steps
                                        path_loss.backward()

                                        del path_outputs, path_scores, path_score_i, path_label_i, path_loss

                                if should_sync:
                                    if self.use_amp:
                                        self.pathology_scalers[path_idx].unscale_(self.pathology_optimizers[path_idx])
                                        torch.nn.utils.clip_grad_norm_(
                                            [p for name, p in self.model_without_ddp.named_parameters()
                                             if f'pathology_modules.{path_idx}' in name and p.requires_grad],
                                            max_norm=1.0
                                        )
                                        self.pathology_scalers[path_idx].step(self.pathology_optimizers[path_idx])
                                        self.pathology_scalers[path_idx].update()
                                    else:
                                        torch.nn.utils.clip_grad_norm_(
                                            [p for name, p in self.model_without_ddp.named_parameters()
                                             if f'pathology_modules.{path_idx}' in name and p.requires_grad],
                                            max_norm=1.0
                                        )
                                        self.pathology_optimizers[path_idx].step()
                        else:
                            # Zero gradients at start of accumulation
                            if batch_idx % accumulation_steps == 0:
                                for opt in self.pathology_optimizers:
                                    opt.zero_grad()

                            sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                            with sync_context():
                                if self.use_amp:
                                    with torch.amp.autocast('cuda'):
                                        path_outputs = self.model(inputs)
                                        path_scores = path_outputs['pathology_scores']

                                        path_loss_total = 0.0
                                        for path_idx in range(self.config.num_pathologies):
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
                                                path_loss_i = F.binary_cross_entropy_with_logits(
                                                    path_score_i[valid_mask],
                                                    path_label_i[valid_mask].float(),
                                                    pos_weight=pos_weight_tensor
                                                )
                                                running_losses['pathology'] += path_loss_i.item()
                                            else:
                                                # Keep DDP backward calls aligned across ranks
                                                path_loss_i = path_score_i.sum() * 0.0

                                            path_loss_total = path_loss_total + path_loss_i

                                        path_loss_total = path_loss_total / accumulation_steps
                                        self.pathology_scalers[0].scale(path_loss_total).backward()

                                    del path_outputs, path_scores, path_loss_total
                                else:
                                    path_outputs = self.model(inputs)
                                    path_scores = path_outputs['pathology_scores']

                                    path_loss_total = 0.0
                                    for path_idx in range(self.config.num_pathologies):
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
                                            path_loss_i = F.binary_cross_entropy_with_logits(
                                                path_score_i[valid_mask],
                                                path_label_i[valid_mask].float(),
                                                pos_weight=pos_weight_tensor
                                            )
                                            running_losses['pathology'] += path_loss_i.item()
                                        else:
                                            # Keep DDP backward calls aligned across ranks
                                            path_loss_i = path_score_i.sum() * 0.0
                                        
                                        path_loss_total = path_loss_total + path_loss_i

                                    path_loss_total = path_loss_total / accumulation_steps
                                    path_loss_total.backward()

                                    del path_outputs, path_scores, path_loss_total

                            # Update optimizers at end of accumulation
                            if should_sync:
                                if self.use_amp:
                                    for opt in self.pathology_optimizers:
                                        self.pathology_scalers[0].unscale_(opt)
                                    torch.nn.utils.clip_grad_norm_(
                                        [p for name, p in self.model_without_ddp.named_parameters()
                                         if 'pathology_modules' in name and p.requires_grad],
                                        max_norm=1.0
                                    )
                                    for opt in self.pathology_optimizers:
                                        self.pathology_scalers[0].step(opt)
                                    self.pathology_scalers[0].update()
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        [p for name, p in self.model_without_ddp.named_parameters()
                                         if 'pathology_modules' in name and p.requires_grad],
                                        max_norm=1.0
                                    )
                                    for opt in self.pathology_optimizers:
                                        opt.step()
                    
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            if is_main_process():
                                logger.warning(f"OOM in pathology update batch {batch_idx}, skipping")
                            # Clear pending operations before syncing
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            oom_in_batch = True
                        else:
                            raise e

                    if _sync_oom_flag(oom_in_batch):
                        if _legacy_phase_updates_enabled():
                            _set_all_requires_grad(True)
                        _cleanup_after_oom()
                        # Barrier to ensure all ranks skip together
                        if self.is_distributed:
                            try:
                                distributed_barrier()
                            except:
                                pass
                        continue
                
                # ============================================
                # 2. TB Patient Classifier Update
                # ============================================
                patient_components = [
                    'site_integration', 'patient_mil', 'task_classifiers',
                    'cross_site_attention', 'tb_classifier'
                ]
                if _legacy_phase_updates_enabled():
                    _set_requires_grad_for_components(patient_components)

                if batch_idx % accumulation_steps == 0 and self.patient_pipeline_optimizer:
                    self.patient_pipeline_optimizer.zero_grad()
                
                try:
                    # Use context manager for gradient synchronization
                    sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                    
                    with sync_context():
                        if self.use_amp and self.patient_pipeline_optimizer:
                            with torch.amp.autocast('cuda'):
                                outputs = self.model(inputs)

                                # Distillation branch for distilled_edge student
                                if (
                                    self.model_type == "distilled_edge"
                                    and ensemble_distillation_loss is not None
                                    and self.teacher_model is not None
                                ):
                                    with torch.no_grad():
                                        teacher_outputs = self.teacher_model(inputs)
                                    total_loss, loss_dict = ensemble_distillation_loss(
                                        self.model_without_ddp,
                                        outputs,
                                        teacher_outputs,
                                        targets,
                                        self.task_pos_weights,
                                        self.config,
                                    )
                                else:
                                    total_loss, loss_dict = self.model_without_ddp.compute_losses(
                                        outputs, targets, self.task_pos_weights
                                    )

                                total_loss = total_loss / accumulation_steps

                                if "TB Label_loss" in loss_dict:
                                    running_losses["tb_loss"] += loss_dict["TB Label_loss"]
                                running_losses["total"] += total_loss.item() * accumulation_steps

                                self.patient_pipeline_scaler.scale(total_loss).backward()
                        elif self.patient_pipeline_optimizer:
                            outputs = self.model(inputs)

                            if (
                                self.model_type == "distilled_edge"
                                and ensemble_distillation_loss is not None
                                and self.teacher_model is not None
                            ):
                                with torch.no_grad():
                                    teacher_outputs = self.teacher_model(inputs)
                                total_loss, loss_dict = ensemble_distillation_loss(
                                    self.model_without_ddp,
                                    outputs,
                                    teacher_outputs,
                                    targets,
                                    self.task_pos_weights,
                                    self.config,
                                )
                            else:
                                total_loss, loss_dict = self.model_without_ddp.compute_losses(
                                    outputs, targets, self.task_pos_weights
                                )

                            total_loss = total_loss / accumulation_steps

                            if "TB Label_loss" in loss_dict:
                                running_losses["tb_loss"] += loss_dict["TB Label_loss"]
                            running_losses["total"] += total_loss.item() * accumulation_steps

                            total_loss.backward()
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        if is_main_process():
                            logger.warning(f"OOM in patient update batch {batch_idx}, skipping")
                        # Clear pending operations before syncing
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        oom_in_batch = True
                    else:
                        raise e
                
                if _sync_oom_flag(oom_in_batch):
                    if _legacy_phase_updates_enabled():
                        _set_all_requires_grad(True)
                    _cleanup_after_oom()
                    # Barrier to ensure all ranks skip together
                    if self.is_distributed:
                        try:
                            distributed_barrier()
                        except:
                            pass
                    continue
                
                # Update optimizer at end of accumulation
                if should_sync and self.patient_pipeline_optimizer:
                    if self.use_amp:
                        self.patient_pipeline_scaler.unscale_(self.patient_pipeline_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.model_without_ddp.named_parameters()
                             if any(component in name for component in [
                                 'site_integration', 'patient_mil', 'task_classifiers',
                                 'cross_site_attention', 'tb_classifier'
                             ]) and p.requires_grad],
                            max_norm=1.0
                        )
                        self.patient_pipeline_scaler.step(self.patient_pipeline_optimizer)
                        self.patient_pipeline_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.model_without_ddp.named_parameters()
                             if any(component in name for component in [
                                 'site_integration', 'patient_mil', 'task_classifiers',
                                 'cross_site_attention', 'tb_classifier'
                             ]) and p.requires_grad],
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
                
                # ============================================
                # 3. Backbone Update
                # ============================================
                backbone_update_every = max(1, int(getattr(self.config, 'backbone_update_every', 2)))
                if self.backbone_optimizer and not self.disable_backbone_updates and batch_idx % backbone_update_every == 0:
                    # Cleanup from previous steps
                    if 'outputs' in locals(): del outputs
                    if 'total_loss' in locals(): del total_loss
                    if 'loss_dict' in locals(): del loss_dict
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if _legacy_phase_updates_enabled():
                        _set_requires_grad_for_components([
                            'vision_encoder', 'cnn_backbone', 'video_transformer', 'backbone',
                            'multi_feature_extraction', 'multi_scale_extraction',
                            'feature_projection', 'cnn_projection', 'output_projection',
                            'frame_selector'
                        ])
                    
                    if batch_idx % accumulation_steps == 0:
                        self.backbone_optimizer.zero_grad()
                    
                    try:
                        sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                        
                        with sync_context():
                            if self.use_amp:
                                with torch.amp.autocast('cuda'):
                                    backbone_outputs = self.model(inputs)
                                    backbone_loss, _ = self.model_without_ddp.compute_losses(
                                        backbone_outputs, targets, self.task_pos_weights
                                    )
                                    backbone_loss = backbone_loss / accumulation_steps
                                    running_losses['backbone'] += backbone_loss.item() * accumulation_steps
                                    
                                    del backbone_outputs
                                    self.backbone_scaler.scale(backbone_loss).backward()
                                    del backbone_loss
                            else:
                                backbone_outputs = self.model(inputs)
                                backbone_loss, _ = self.model_without_ddp.compute_losses(
                                    backbone_outputs, targets, self.task_pos_weights
                                )
                                backbone_loss = backbone_loss / accumulation_steps
                                running_losses['backbone'] += backbone_loss.item() * accumulation_steps
                                
                                del backbone_outputs
                                backbone_loss.backward()
                                del backbone_loss
                        
                        # Update optimizer at end of accumulation
                        if should_sync:
                            if self.use_amp:
                                self.backbone_scaler.unscale_(self.backbone_optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in self.model_without_ddp.parameters() if p.requires_grad],
                                    max_norm=0.5
                                )
                                self.backbone_scaler.step(self.backbone_optimizer)
                                self.backbone_scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in self.model_without_ddp.parameters() if p.requires_grad],
                                    max_norm=0.5
                                )
                                self.backbone_optimizer.step()
                    
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            if is_main_process():
                                logger.warning(f"OOM in backbone update batch {batch_idx}, skipping")
                            # Clear pending operations before syncing
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            if self.backbone_optimizer:
                                self.backbone_optimizer.zero_grad()
                            oom_in_batch = True
                            self.backbone_oom_count += 1
                            if (getattr(self.config, 'disable_backbone_on_oom', True) and
                                    self.backbone_oom_count >= int(getattr(self.config, 'backbone_max_oom', 3))):
                                self.disable_backbone_updates = True
                                if is_main_process():
                                    logger.warning(
                                        f"Disabling backbone updates after {self.backbone_oom_count} OOMs. "
                                        "Consider increasing backbone_update_every or freezing the backbone."
                                    )
                        else:
                            raise e

                if _sync_oom_flag(oom_in_batch):
                    if _legacy_phase_updates_enabled():
                        _set_all_requires_grad(True)
                    _cleanup_after_oom()
                    # Barrier to ensure all ranks skip together
                    if self.is_distributed:
                        try:
                            distributed_barrier()
                        except:
                            pass
                    continue

                if _legacy_phase_updates_enabled():
                    _set_all_requires_grad(True)
                
                # Memory cleanup
                self._reset_frame_selector_state()
                if batch_idx % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update progress bar (only on main process)
                if is_main_process() and batch_idx % 2 == 0:
                    def to_scalar(value):
                        if isinstance(value, torch.Tensor):
                            return value.item()
                        return value
                    
                    progress_dict = {
                        'total_loss': to_scalar(running_losses['total'] / max(1, batch_idx + 1)),
                        'tb_loss': to_scalar(running_losses['tb_loss'] / max(1, batch_idx + 1)),
                    }
                    
                    if self.use_pathology_loss and 'pathology' in running_losses:
                        progress_dict['path_loss'] = to_scalar(running_losses['pathology'] / max(1, batch_idx + 1))
                    if self.backbone_optimizer:
                        progress_dict['backbone_loss'] = to_scalar(running_losses['backbone'] / max(1, batch_idx + 1))
                    
                    progress_bar.set_postfix(progress_dict)
            
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    if is_main_process():
                        logger.warning(f"OOM in batch {batch_idx}, skipping")
                    # Clear pending operations before syncing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    oom_in_batch = True
                    if _sync_oom_flag(oom_in_batch):
                        if _legacy_phase_updates_enabled():
                            _set_all_requires_grad(True)
                        _cleanup_after_oom()
                        # Barrier to ensure all ranks skip together
                        if self.is_distributed:
                            try:
                                distributed_barrier()
                            except:
                                pass
                        continue
                else:
                    if is_main_process():
                        logger.error(f"Runtime error: {e}")
                    raise e
        
        progress_bar.close()
        self._reset_frame_selector_state(clear_history=True)
        
        # Gather metrics from all processes
        if self.is_distributed:
            # Convert losses to tensors for all_reduce
            loss_tensor = torch.tensor([
                running_losses['total'],
                running_losses['tb_loss'],
                running_losses.get('pathology', 0.0),
                len(self.train_loader)
            ], device=self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            
            running_losses['total'] = loss_tensor[0].item()
            running_losses['tb_loss'] = loss_tensor[1].item()
            if self.use_pathology_loss:
                running_losses['pathology'] = loss_tensor[2].item()
            
            num_batches = int(loss_tensor[3].item())
        else:
            num_batches = len(self.train_loader)
        
        # Calculate metrics
        all_metrics = {}

        tb_targets_np = None
        tb_preds_np = None
        tb_logits_np = None

        if all_tb_targets and all_tb_predictions and all_tb_logits:
            local_targets = torch.cat(all_tb_targets).cpu().numpy()
            local_preds   = torch.cat(all_tb_predictions).cpu().numpy()
            local_logits  = torch.cat(all_tb_logits).cpu().numpy()

            tb_targets_np = _ddp_concat_numpy(local_targets)
            tb_preds_np   = _ddp_concat_numpy(local_preds)
            tb_logits_np  = _ddp_concat_numpy(local_logits)

            tb_metrics = self._calculate_metrics(
                tb_targets_np, tb_preds_np, tb_logits_np, None, "TB Label"
            )
            for key, value in tb_metrics.items():
                all_metrics[f'TB Label_{key}'] = value

        # Pathology gather for train metrics
        if self.use_pathology_loss and pathology_labels_list and pathology_scores_list:
            local_scores = torch.cat(pathology_scores_list, dim=0).cpu().numpy()
            local_labels = torch.cat(pathology_labels_list, dim=0).cpu().numpy()
            local_masks  = torch.cat(pathology_masks_list,  dim=0).cpu().numpy()

            all_scores_np = _ddp_concat_numpy(local_scores)
            all_labels_np = _ddp_concat_numpy(local_labels)
            all_masks_np  = _ddp_concat_numpy(local_masks)

            path_metrics = self._calculate_pathology_metrics(
                torch.from_numpy(all_scores_np),
                torch.from_numpy(all_labels_np),
                torch.from_numpy(all_masks_np)
            )
            all_metrics.update(path_metrics)
        
        # Log metrics (only on main process)
        if is_main_process():
            logger.info(f"Train TB metrics:")
            tb_metrics = {k.replace('TB Label_', ''): v for k, v in all_metrics.items()
                         if k.startswith('TB Label_')}
            if tb_metrics:
                logger.info(f"  TB Label: " + " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items()]))
            
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
        
        return running_losses['total'] / max(1, num_batches), all_metrics
    
    def validate(self, epoch, loader=None, split_name="val"):
        """Validation with distributed support."""
        if loader is None:
            loader = self.val_loader
        self._reset_frame_selector_state(clear_history=True)
        
        # Check if loader is empty
        if loader is None or len(loader) == 0:
            if is_main_process():
                logger.warning(f"No {split_name} data available, skipping {split_name} evaluation")
            # Return default metrics
            all_metrics = {'loss': 0.0}
            if self.use_pathology_loss:
                all_metrics.update({
                    'A-line_auroc': 0.0,
                    'A-line_auprc': 0.0,
                    'A-line_f1': 0.0,
                    'Large Consolidations_auroc': 0.0,
                    'Large Consolidations_auprc': 0.0,
                    'Large Consolidations_f1': 0.0,
                    'Pleural Effusion_auroc': 0.0,
                    'Pleural Effusion_auprc': 0.0,
                    'Pleural Effusion_f1': 0.0,
                    'Other Pathology_auroc': 0.0,
                    'Other Pathology_auprc': 0.0,
                    'Other Pathology_f1': 0.0,
                })
            # Add TB metrics with default values
            for task in self.active_tasks:
                all_metrics.update({
                    f'{task}_accuracy': 0.0,
                    f'{task}_precision': 0.0,
                    f'{task}_recall': 0.0,
                    f'{task}_specificity': 0.0,
                    f'{task}_f1': 0.0,
                    f'{task}_auc': 0.0,
                    f'{task}_auprc': 0.0,
                })
            return 0.0, all_metrics
        
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
        
        progress_bar = tqdm(
            loader,
            desc=f"{split_name.capitalize()} Evaluation",
            disable=not is_main_process()
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Move data to device
                    site_videos = batch['site_videos'].to(self.device)
                    site_indices = batch['site_indices'].to(self.device)
                    # Guard for missing site_masks in validation/test
                    if 'site_masks' in batch and batch['site_masks'] is not None:
                        site_masks = batch['site_masks'].to(self.device)
                    else:
                        # Create a default all-valid mask if not provided
                        if 'site_findings' in batch and batch['site_findings'] is not None:
                            sf = batch['site_findings']
                            N = sf.shape[1] if sf.ndim >= 2 else (
                                batch['site_indices'].shape[1] if 'site_indices' in batch else getattr(self.config, 'max_sites', 15)
                            )
                        elif 'site_indices' in batch and batch['site_indices'] is not None:
                            N = batch['site_indices'].shape[1]
                        else:
                            N = getattr(self.config, 'max_sites', 15)
                        site_masks = torch.ones((batch['site_videos'].shape[0], N), dtype=torch.bool, device=self.device)

                    site_findings = batch['site_findings'].to(self.device)
                    tb_labels = batch['tb_labels'].to(self.device).float()
                    pneumonia_labels = torch.full_like(tb_labels, -1)
                    covid_labels = torch.full_like(tb_labels, -1)
                    
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
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss, _ = self.model_without_ddp.compute_losses(
                        outputs, targets, self.task_pos_weights
                    )
                    
                    running_loss += loss.item()
                    
                    # Collect predictions for TB (robust extraction)
                    task_logits = outputs.get('task_logits', {})

                    logits = None
                    if isinstance(task_logits, dict) and 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                    elif 'tb_logits' in outputs:
                        logits = outputs['tb_logits']
                    elif 'logits' in outputs:
                        # Fallback for models that return a generic 'logits'
                        logits = outputs['logits']

                    if logits is not None:
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
                    if is_main_process():
                        progress_bar.set_postfix({
                            'loss': running_loss / (progress_bar.n + 1)
                        })
                    
                    # Clean up memory
                    del site_videos, site_indices, site_masks, site_findings, inputs
                    del tb_labels, pneumonia_labels, covid_labels, outputs, loss
                    self._reset_frame_selector_state()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if is_main_process():
                            logger.warning("OOM during validation, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Gather metrics from all processes
        if self.is_distributed:
            loss_tensor = torch.tensor([running_loss, len(loader)], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            running_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
        else:
            num_batches = len(loader)
        
        # Handle empty validation set
        if num_batches == 0:
            if is_main_process():
                logger.warning(f"No {split_name} batches available, skipping {split_name} evaluation")
            # Return default metrics
            all_metrics = {'loss': 0.0}
            if self.use_pathology_loss:
                all_metrics.update({
                    'A-line_auroc': 0.0,
                    'A-line_auprc': 0.0,
                    'A-line_f1': 0.0,
                    'Large Consolidations_auroc': 0.0,
                    'Large Consolidations_auprc': 0.0,
                    'Large Consolidations_f1': 0.0,
                    'Pleural Effusion_auroc': 0.0,
                    'Pleural Effusion_auprc': 0.0,
                    'Pleural Effusion_f1': 0.0,
                    'Other Pathology_auroc': 0.0,
                    'Other Pathology_auprc': 0.0,
                    'Other Pathology_f1': 0.0,
                })
            # Add TB metrics with default values
            for task in self.active_tasks:
                all_metrics.update({
                    f'{task}_accuracy': 0.0,
                    f'{task}_precision': 0.0,
                    f'{task}_recall': 0.0,
                    f'{task}_specificity': 0.0,
                    f'{task}_f1': 0.0,
                    f'{task}_auc': 0.0,
                    f'{task}_auprc': 0.0,
                })
            return 0.0, all_metrics
        
        # Calculate validation metrics
        val_loss = running_loss / num_batches
        all_metrics = {'loss': val_loss}

        # ---- Gather TB tensors across all ranks before computing metrics ----
        tb_targets_np = None
        tb_preds_np = None
        tb_logits_np = None
        tb_probs_np = None

        if all_tb_targets and all_tb_predictions:
            local_targets = torch.cat(all_tb_targets).cpu().numpy()
            local_preds   = torch.cat(all_tb_predictions).cpu().numpy()
            local_logits  = torch.cat(all_tb_logits).cpu().numpy()
            local_probs   = torch.cat(all_tb_probs).cpu().numpy()

            tb_targets_np = _ddp_concat_numpy(local_targets)
            tb_preds_np   = _ddp_concat_numpy(local_preds)
            tb_logits_np  = _ddp_concat_numpy(local_logits)
            tb_probs_np   = _ddp_concat_numpy(local_probs)

            tb_metrics = self._calculate_metrics(
                tb_targets_np,
                tb_preds_np,
                tb_logits_np,
                tb_probs_np,
                "TB Label"
            )
            for key, value in tb_metrics.items():
                all_metrics[f'TB Label_{key}'] = value

        # ---- Gather pathology tensors across all ranks before computing metrics ----
        if self.use_pathology_loss and pathology_labels_list and pathology_scores_list:
            local_scores = torch.cat(pathology_scores_list, dim=0).cpu().numpy()
            local_labels = torch.cat(pathology_labels_list, dim=0).cpu().numpy()
            local_masks  = torch.cat(pathology_masks_list,  dim=0).cpu().numpy()

            all_scores_np = _ddp_concat_numpy(local_scores)
            all_labels_np = _ddp_concat_numpy(local_labels)
            all_masks_np  = _ddp_concat_numpy(local_masks)

            # Convert back to tensors for the existing helper
            path_metrics = self._calculate_pathology_metrics(
                torch.from_numpy(all_scores_np),
                torch.from_numpy(all_labels_np),
                torch.from_numpy(all_masks_np)
            )
            all_metrics.update(path_metrics)
        
        # Log metrics (only on main process)
        if is_main_process():
            logger.info(f"{split_name} TB metrics:")
            tb_metrics = {k.replace('TB Label_', ''): v for k, v in all_metrics.items()
                         if k.startswith('TB Label_') and '/' not in k}
            if tb_metrics:
                logger.info(f"  TB Label: " + " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items()]))

            if self.use_pathology_loss:
                path_metrics = {k: v for k, v in all_metrics.items() if '/' in k}
                # Macro pathology metrics block (inserted before per-class metrics)
                macro_auroc = path_metrics.get('pathology/macro_auroc')
                macro_auprc = path_metrics.get('pathology/macro_auprc')
                macro_f1 = path_metrics.get('pathology/macro_f1')
                if macro_auroc is not None or macro_auprc is not None or macro_f1 is not None:
                    logger.info(f"{split_name} Pathology metrics:")
                    parts = []
                    if macro_auroc is not None:
                        parts.append(f"auroc: {macro_auroc:.4f}")
                    if macro_auprc is not None:
                        parts.append(f"auprc: {macro_auprc:.4f}")
                    if macro_f1 is not None:
                        parts.append(f"f1: {macro_f1:.4f}")
                    logger.info("  Macro Average: " + " | ".join(parts))
                # Per-class metrics (unchanged)
                pathology_names = getattr(self.config, 'pathology_classes',
                                          ['A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'])
                for name in pathology_names:
                    name_metrics = {k.split('/')[-1]: v for k, v in path_metrics.items()
                                    if k.startswith(f'{name}/')}
                    if name_metrics:
                        logger.info(f"  {name}: " + " | ".join([f"{k}: {v:.4f}" for k, v in name_metrics.items()]))

            # Print confusion matrix on the full (gathered) split
            if tb_targets_np is not None and tb_preds_np is not None:
                try:
                    cm = confusion_matrix(tb_targets_np.flatten(), tb_preds_np.flatten())
                    logger.info(f"\nConfusion Matrix for TB ({split_name}):\n{cm}")
                    report = classification_report(tb_targets_np.flatten(), tb_preds_np.flatten())
                    logger.info(f"\nClassification Report for TB ({split_name}):\n{report}")
                except Exception as e:
                    logger.info(f"Could not compute confusion matrix: {e}")
        
        return val_loss, all_metrics
    
    def _calculate_metrics(self, targets, predictions, logits, probs=None, task_name=""):
        """Calculate performance metrics."""
        metrics = {}
        
        try:
            # Ensure correct shapes
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.flatten()
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['precision'] = precision_score(targets, predictions, zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, zero_division=0)
            metrics['specificity'] = recall_score(1-targets, 1-predictions, zero_division=0)
            metrics['f1'] = f1_score(targets, predictions, zero_division=0)
            
            # Calculate AUC; if probs not provided, derive from logits via sigmoid
            try:
                if probs is None and logits is not None:
                    # logits is a numpy array here; apply sigmoid safely
                    probs = 1.0 / (1.0 + np.exp(-logits))
                if probs is not None:
                    if probs.ndim == 2 and probs.shape[1] == 1:
                        probs = probs.flatten()
                    metrics['auc'] = roc_auc_score(targets, probs)
                    metrics['auprc'] = average_precision_score(targets, probs)
            except ValueError:
                metrics['auc'] = 0.5
                metrics['auprc'] = 0.5
        
        except Exception as e:
            if is_main_process():
                logger.error(f"Error calculating metrics for {task_name}: {e}")
            if 'accuracy' not in metrics:
                metrics['accuracy'] = 0.0
            if self.config.eval_metric not in metrics:
                metrics[self.config.eval_metric] = 0.0
        
        return metrics
    
    def _calculate_pathology_metrics(self, scores, labels, masks):
        """Calculate metrics for each pathology class."""
        metrics = {}
        
        pathology_names = getattr(self.config, 'pathology_classes', [
            'A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'
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
                        if is_main_process():
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
        """Save checkpoint (only on main process)."""
        metrics = dict(metrics or {})
        if not is_main_process():
            return None
        
        save_dir = pathlib.Path(self.config.experiment_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (unwrap DDP if needed)
        model_state_dict = self.model_without_ddp.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'backbone_optimizer_state_dict': self.backbone_optimizer.state_dict() if self.backbone_optimizer else None,
            'patient_pipeline_optimizer_state_dict': self.patient_pipeline_optimizer.state_dict() if self.patient_pipeline_optimizer else None,
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
                'backbone_scaler_state_dict': self.backbone_scaler.state_dict() if hasattr(self, 'backbone_scaler') else None,
                'patient_pipeline_scaler_state_dict': self.patient_pipeline_scaler.state_dict() if hasattr(self, 'patient_pipeline_scaler') else None,
                'pathology_scalers_state_dicts': [scaler.state_dict() for scaler in self.pathology_scalers],
            })
        
        # Save latest checkpoint
        latest_path = save_dir / "checkpoint_latest.pth"
        print(f"Saving latest checkpoint to {latest_path}...")
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
        
        # Save periodic checkpoints
        eval_metric_key = f"TB Label_{self.config.eval_metric}"
        
        if epoch % 5 == 0 and eval_metric_key in metrics:
            metric_value = metrics.get(eval_metric_key, metrics.get('loss', float('nan')))
            epoch_path = save_dir / f"checkpoint_epoch_{epoch:03d}_metric_{metric_value:.4f}.pth"
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            metric_value = metrics.get(eval_metric_key, metrics.get('loss', float('nan')))
            best_path = save_dir / f"checkpoint_best_metric_{metric_value:.4f}.pth"
            torch.save(checkpoint, best_path)
            
            best_generic_path = save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_generic_path)
            
            logger.info(f"New best model saved to {best_path}")
        
        return latest_path
    
    def train(self, resume_from_checkpoint=None):
        """Train the model."""
        if is_main_process():
            logger.info(f"Starting TB training with ablation model: {self.model_type}")
            logger.info(f"Active tasks: {self.active_tasks}")
            logger.info(f"Pathology loss enabled: {self.use_pathology_loss}")
        
        start_epoch = 0
        
        if resume_from_checkpoint:
            start_epoch = self.resume_training_from_checkpoint(resume_from_checkpoint)
            if is_main_process():
                logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            if is_main_process():
                logger.info(f"Starting training for {self.config.num_epochs} epochs...")
            self.best_metric = float('-inf') if self.config.eval_metric_goal == 'max' else float('inf')
            self.best_epoch = 0
            self.epochs_without_improvement = 0
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self._reset_frame_selector_state(clear_history=True, reset_temperature=True)
            if is_main_process():
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            train_loss, train_metrics = self.train_epoch(epoch)
            
            val_loss, val_metrics = self.validate(epoch)
            use_train_metrics = (
                getattr(self.config, 'use_train_metric_when_no_val', False)
                and len(self.data_module.patient_val) == 0
            )

            # Log concise epoch summary with key metrics
            if is_main_process():
                eval_metric_key = f"TB Label_{self.config.eval_metric}"
                summary_source = train_metrics if use_train_metrics else val_metrics
                summary_auc = summary_source.get(eval_metric_key)
                summary_f1 = summary_source.get('TB Label_f1')
                summary_acc = summary_source.get('TB Label_accuracy')
                summary_loss = train_loss if use_train_metrics else val_loss
                summary_prefix = "train_loss" if use_train_metrics else "val_loss"
                logger.info(
                    "Epoch %d Summary — %s: %.4f%s%s%s" % (
                        epoch + 1,
                        summary_prefix,
                        summary_loss,
                        f", {eval_metric_key}: {summary_auc:.4f}" if summary_auc is not None else "",
                        f", TB Label f1: {summary_f1:.4f}" if summary_f1 is not None else "",
                        f", TB Label acc: {summary_acc:.4f}" if summary_acc is not None else ""
                    )
                )
            
            # Use TB Label metric as primary
            eval_metric_key = f"TB Label_{self.config.eval_metric}"
            metric_source = train_metrics if use_train_metrics else val_metrics
            fallback_loss = train_loss if use_train_metrics else val_loss
            current_metric = metric_source.get(eval_metric_key, fallback_loss)
            is_best = False
            
            if self.config.eval_metric_goal == 'max':
                if current_metric > self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    if is_main_process():
                        logger.info(f"New best model with {eval_metric_key}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    if is_main_process():
                        logger.info(f"No improvement. Best {eval_metric_key}: {self.best_metric:.4f} from epoch {self.best_epoch+1}")
            else:
                if current_metric < self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    if is_main_process():
                        logger.info(f"New best model with {eval_metric_key}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    if is_main_process():
                        logger.info(f"No improvement. Best {eval_metric_key}: {self.best_metric:.4f} from epoch {self.best_epoch+1}")
            
            # Save checkpoint (only on main process)
            checkpoint_metrics = dict(metric_source or {})
            checkpoint_metrics.setdefault('loss', fallback_loss)
            self.save_checkpoint(epoch, checkpoint_metrics, is_best)
            
            # Synchronize all processes
            if self.is_distributed:
                distributed_barrier()
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                if is_main_process():
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if is_main_process():
            logger.info(f"Training completed. Best model from epoch {self.best_epoch+1} with {eval_metric_key}: {self.best_metric:.4f}")
        
        # Evaluate the best model if requested
        # IMPORTANT: Run evaluation on ALL ranks to avoid hanging collectives
        if self.config.evaluate_best_valid_model:
            if self.is_distributed:
                # Ensure all ranks finished training and checkpoints are visible
                distributed_barrier()
            self._evaluate_best_model()
            if self.is_distributed:
                # Ensure all ranks complete evaluation before teardown
                distributed_barrier()
        
        return self.best_metric, self.best_epoch
    
    def _evaluate_split_for_artifacts(self, loader, split_name):
        """Run evaluation and return metrics plus patient-level TB predictions."""
        if loader is None or len(loader) == 0:
            return 0.0, {'loss': 0.0}, pd.DataFrame()

        self.model.eval()
        self._reset_frame_selector_state(clear_history=True)

        running_loss = 0.0
        all_patient_ids = []
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        all_tb_probs = []
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []

        progress_bar = tqdm(
            loader,
            desc=f"{split_name.capitalize()} Artifact Evaluation",
            disable=not is_main_process(),
        )

        with torch.no_grad():
            for batch in progress_bar:
                try:
                    site_videos = batch['site_videos'].to(self.device)
                    site_indices = batch['site_indices'].to(self.device)
                    if 'site_masks' in batch and batch['site_masks'] is not None:
                        site_masks = batch['site_masks'].to(self.device)
                    else:
                        if 'site_findings' in batch and batch['site_findings'] is not None:
                            site_findings_for_shape = batch['site_findings']
                            num_sites = site_findings_for_shape.shape[1] if site_findings_for_shape.ndim >= 2 else getattr(self.config, 'max_sites', 15)
                        elif 'site_indices' in batch and batch['site_indices'] is not None:
                            num_sites = batch['site_indices'].shape[1]
                        else:
                            num_sites = getattr(self.config, 'max_sites', 15)
                        site_masks = torch.ones((batch['site_videos'].shape[0], num_sites), dtype=torch.bool, device=self.device)

                    site_findings = batch['site_findings'].to(self.device)
                    tb_labels = batch['tb_labels'].to(self.device).float()
                    pneumonia_labels = torch.full_like(tb_labels, -1)
                    covid_labels = torch.full_like(tb_labels, -1)

                    inputs = {
                        'site_videos': site_videos,
                        'site_indices': site_indices,
                        'site_masks': site_masks,
                        'site_findings': site_findings,
                        'is_patient_level': True,
                    }
                    targets = {
                        'tb_labels': tb_labels,
                        'pneumonia_labels': pneumonia_labels,
                        'covid_labels': covid_labels,
                        'pathology_labels': site_findings,
                        'site_masks': site_masks,
                    }

                    outputs = self.model(inputs)
                    loss, _ = self.model_without_ddp.compute_losses(
                        outputs,
                        targets,
                        self.task_pos_weights,
                    )
                    running_loss += loss.item()

                    task_logits = outputs.get('task_logits', {})
                    logits = None
                    if isinstance(task_logits, dict) and 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                    elif 'tb_logits' in outputs:
                        logits = outputs['tb_logits']
                    elif 'logits' in outputs:
                        logits = outputs['logits']

                    if logits is not None:
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_predictions.append(preds.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                        all_tb_probs.append(probs.detach().cpu())

                        patient_ids = batch.get('patient_ids')
                        if patient_ids is None:
                            patient_ids = [f"{split_name}_{len(all_patient_ids) + i}" for i in range(tb_labels.shape[0])]
                        all_patient_ids.extend([str(patient_id) for patient_id in patient_ids])

                    if self.use_pathology_loss and 'pathology_scores' in outputs:
                        path_scores = outputs['pathology_scores']
                        path_labels = targets['pathology_labels']
                        if path_scores.dim() == 3:
                            _, _, num_pathologies = path_scores.shape
                            path_scores = path_scores.reshape(-1, num_pathologies)
                            path_labels = path_labels.reshape(-1, num_pathologies)

                        valid_mask = path_labels >= 0
                        pathology_scores_list.append(path_scores.detach().cpu())
                        pathology_labels_list.append(path_labels.detach().cpu())
                        pathology_masks_list.append(valid_mask.detach().cpu())

                    if is_main_process():
                        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

                    del site_videos, site_indices, site_masks, site_findings, inputs
                    del tb_labels, pneumonia_labels, covid_labels, outputs, loss
                    self._reset_frame_selector_state()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if is_main_process():
                            logger.warning("OOM during artifact evaluation, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

        if self.is_distributed:
            loss_tensor = torch.tensor([running_loss, len(loader)], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            running_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
        else:
            num_batches = len(loader)

        split_loss = running_loss / max(1, num_batches)
        metrics = {'loss': split_loss}
        predictions_df = pd.DataFrame()

        if all_tb_targets and all_tb_predictions and all_tb_logits:
            local_tb_targets = torch.cat(all_tb_targets).cpu().numpy()
            local_tb_preds = torch.cat(all_tb_predictions).cpu().numpy()
            local_tb_logits = torch.cat(all_tb_logits).cpu().numpy()
            local_tb_probs = torch.cat(all_tb_probs).cpu().numpy()
        else:
            local_tb_targets = np.empty((0,), dtype=np.float32)
            local_tb_preds = np.empty((0,), dtype=np.float32)
            local_tb_logits = np.empty((0,), dtype=np.float32)
            local_tb_probs = np.empty((0,), dtype=np.float32)

        tb_targets_np = _ddp_concat_numpy(local_tb_targets)
        tb_preds_np = _ddp_concat_numpy(local_tb_preds)
        tb_logits_np = _ddp_concat_numpy(local_tb_logits)
        tb_probs_np = _ddp_concat_numpy(local_tb_probs)
        patient_ids = _ddp_concat_list(all_patient_ids)

        if len(tb_targets_np) > 0:
            tb_metrics = self._calculate_metrics(
                tb_targets_np,
                tb_preds_np,
                tb_logits_np,
                tb_probs_np,
                "TB Label",
            )
            metrics.update({f'TB Label_{key}': value for key, value in tb_metrics.items()})

            tb_targets_np = np.asarray(tb_targets_np).reshape(-1)
            tb_preds_np = np.asarray(tb_preds_np).reshape(-1)
            tb_probs_np = np.asarray(tb_probs_np).reshape(-1)
            tb_logits_np = np.asarray(tb_logits_np).reshape(-1)
            if len(patient_ids) != len(tb_targets_np):
                patient_ids = [f"{split_name}_{i}" for i in range(len(tb_targets_np))]

            predictions_df = pd.DataFrame({
                'patient_id': patient_ids,
                'tb_target': tb_targets_np,
                'tb_prediction': tb_preds_np,
                'tb_probability': tb_probs_np,
                'tb_logit': tb_logits_np,
            })

        if self.use_pathology_loss:
            if pathology_labels_list and pathology_scores_list:
                local_scores = torch.cat(pathology_scores_list, dim=0).cpu().numpy()
                local_labels = torch.cat(pathology_labels_list, dim=0).cpu().numpy()
                local_masks = torch.cat(pathology_masks_list, dim=0).cpu().numpy()
            else:
                num_pathologies = getattr(self.config, 'num_pathologies', 4)
                local_scores = np.empty((0, num_pathologies), dtype=np.float32)
                local_labels = np.empty((0, num_pathologies), dtype=np.float32)
                local_masks = np.empty((0, num_pathologies), dtype=bool)

            all_scores_np = _ddp_concat_numpy(local_scores)
            all_labels_np = _ddp_concat_numpy(local_labels)
            all_masks_np = _ddp_concat_numpy(local_masks)
            if len(all_scores_np) > 0:
                path_metrics = self._calculate_pathology_metrics(
                    torch.from_numpy(all_scores_np),
                    torch.from_numpy(all_labels_np),
                    torch.from_numpy(all_masks_np),
                )
                metrics.update(path_metrics)

        return split_loss, metrics, predictions_df

    def _evaluate_best_model(self):
        """Evaluate the best model and persist final metrics/prediction artifacts."""
        logger.info("Evaluating best TB ablation model on all splits...")
        self._reset_frame_selector_state(clear_history=True)

        final_results_dir = os.path.join(self.config.experiment_dir, "final_results")
        tb_results_dir = os.path.join(final_results_dir, "tb_results")
        pathology_results_dir = os.path.join(final_results_dir, "pathology_results")
        if is_main_process():
            os.makedirs(tb_results_dir, exist_ok=True)
            if self.use_pathology_loss:
                os.makedirs(pathology_results_dir, exist_ok=True)
        
        # Clear GPU cache before evaluation to avoid OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load best model
        best_model_path = os.path.join(self.config.experiment_dir, "checkpoint_best.pth")
        if not os.path.exists(best_model_path):
            logger.warning(f"Best model checkpoint not found at {best_model_path}. Skipping evaluation.")
            return
        
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to load best model: {e}")
            return
        
        # Evaluate on each split
        self.model.eval()
        tb_summary_rows = []
        pathology_summary_rows = []
        
        for split_name, loader, file_prefix in [
            ('test', self.test_loader, 'test'),
            ('validation', self.val_loader, 'val'),
            ('train', self.train_loader, 'train'),
        ]:
            logger.info(f"Evaluating on {split_name} set...")
            val_loss, metrics, predictions_df = self._evaluate_split_for_artifacts(
                loader,
                split_name,
            )
            if is_main_process():
                eval_metric_key = f"TB Label_{self.config.eval_metric}"
                summary_auc = metrics.get(eval_metric_key)
                summary_f1 = metrics.get('TB Label_f1')
                summary_acc = metrics.get('TB Label_accuracy')
                logger.info(
                    f"{split_name.capitalize()} Summary — loss: {val_loss:.4f}"
                    + (f", {eval_metric_key}: {summary_auc:.4f}" if summary_auc is not None else "")
                    + (f", TB Label f1: {summary_f1:.4f}" if summary_f1 is not None else "")
                    + (f", TB Label acc: {summary_acc:.4f}" if summary_acc is not None else "")
                )

                predictions_file = os.path.join(final_results_dir, f"{file_prefix}_predictions.csv")
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(f"Detailed {split_name} predictions saved to {predictions_file}")

                tb_row = {'Split': split_name, 'loss': val_loss}
                for key, value in metrics.items():
                    if key.startswith('TB Label_') and isinstance(value, (int, float, np.floating)):
                        tb_row[key.replace('TB Label_', '')] = float(value)
                tb_summary_rows.append(tb_row)

                if self.use_pathology_loss:
                    pathology_names = getattr(self.config, 'pathology_classes', [
                        'A-line', 'Large Consolidations', 'Pleural Effusion', 'Other Pathology'
                    ])
                    for pathology_name in pathology_names:
                        row = {'Split': split_name, 'Pathology': pathology_name}
                        found = False
                        for metric_name in ['auroc', 'auprc', 'f1']:
                            value = metrics.get(f'{pathology_name}/{metric_name}')
                            if isinstance(value, (int, float, np.floating)):
                                row[metric_name] = float(value)
                                found = True
                        if found:
                            pathology_summary_rows.append(row)
            
            # Clear GPU cache between splits to avoid OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if is_main_process():
            if tb_summary_rows:
                tb_summary_file = os.path.join(tb_results_dir, "tb_metrics_summary.csv")
                tb_metrics_summary = pd.DataFrame(tb_summary_rows)
                tb_metrics_summary.to_csv(tb_summary_file, index=False)
                logger.info(f"Summary of TB metrics saved to {tb_summary_file}")

            if self.use_pathology_loss and pathology_summary_rows:
                pathology_summary_file = os.path.join(pathology_results_dir, "all_pathologies_summary.csv")
                pathology_summary = pd.DataFrame(pathology_summary_rows)
                pathology_summary.to_csv(pathology_summary_file, index=False)
                logger.info(f"Consolidated pathology metrics saved to {pathology_summary_file}")

                for pathology_name, pathology_df in pathology_summary.groupby('Pathology'):
                    pathology_dir = os.path.join(pathology_results_dir, pathology_name.lower().replace(" ", "_"))
                    os.makedirs(pathology_dir, exist_ok=True)
                    pathology_file = os.path.join(pathology_dir, f"{pathology_name}_metrics_summary.csv")
                    pathology_df.drop(columns=['Pathology']).to_csv(pathology_file, index=False)
                    logger.info(f"Summary of {pathology_name} metrics saved to {pathology_file}")
    
    def resume_training_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint."""
        if is_main_process():
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer states
        if checkpoint.get('backbone_optimizer_state_dict') and self.backbone_optimizer:
            self.backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer_state_dict'])
        
        if checkpoint.get('patient_pipeline_optimizer_state_dict') and self.patient_pipeline_optimizer:
            self.patient_pipeline_optimizer.load_state_dict(checkpoint['patient_pipeline_optimizer_state_dict'])
        
        if checkpoint.get('pathology_optimizers_state_dicts'):
            for opt, state_dict in zip(self.pathology_optimizers, checkpoint['pathology_optimizers_state_dicts']):
                opt.load_state_dict(state_dict)
        
        # Load scheduler states
        if checkpoint.get('schedulers_state_dicts'):
            for scheduler, state_dict in zip(self.schedulers, checkpoint['schedulers_state_dicts']):
                scheduler.load_state_dict(state_dict)
        
        # Load AMP scaler states
        if self.use_amp:
            if checkpoint.get('backbone_scaler_state_dict') and hasattr(self, 'backbone_scaler'):
                self.backbone_scaler.load_state_dict(checkpoint['backbone_scaler_state_dict'])
            
            if checkpoint.get('patient_pipeline_scaler_state_dict') and hasattr(self, 'patient_pipeline_scaler'):
                self.patient_pipeline_scaler.load_state_dict(checkpoint['patient_pipeline_scaler_state_dict'])
            
            if checkpoint.get('pathology_scalers_state_dicts'):
                for scaler, state_dict in zip(self.pathology_scalers, checkpoint['pathology_scalers_state_dicts']):
                    scaler.load_state_dict(state_dict)
        
        # Load training state
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        if is_main_process():
            logger.info(f"Training will resume from epoch {start_epoch}")
            logger.info(f"Best metric so far: {self.best_metric:.4f} (epoch {self.best_epoch + 1})")
        
        return start_epoch


# ============================================================================
# Main Function
# ============================================================================

def parse_args_and_load_config(argv=None):
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(description='Distributed training for TB detection using ablation models.')
    
    # Config file argument
    parser.add_argument('--config', type=str, required=False,
                       help='Path to config YAML file')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, help='Ablation model type',
                      choices=['no_rl', 'mean_pool', 'attention_pool', 'single_task',
                              '3d_cnn', 'cnn_lstm', 'video_transformer'])
    
    # Training arguments
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Data arguments
    parser.add_argument('--root_dir', type=str, help='Path to dataset root directory')
    parser.add_argument('--labels_csv', type=str, help='Path to labels CSV')
    parser.add_argument('--file_metadata_csv', type=str, help='Path to file metadata CSV')
    parser.add_argument('--split_csv', type=str, help='Path to split CSV')
    parser.add_argument('--video_folder', type=str, help='Path to video folder')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')

    # Output arguments
    parser.add_argument('--experiment_dir', type=str, help='Directory for experiment artifacts')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory for checkpoints')
    parser.add_argument('--log_dir', type=str, help='Directory for trainer logs')
    parser.add_argument('--save_dir', type=str, help='Directory for trainer saves')
    parser.add_argument('--pred_save_dir', type=str, help='Directory for saved predictions')
    
    # Model loading arguments
    parser.add_argument('--model_weights', '--model-weights', dest='model_weights', type=str, help='Path to model weights')
    parser.add_argument('--best_model_path', '--best-model-path', dest='best_model_path', type=str, help='Path to best model for evaluation')
    parser.add_argument('--resume_from_checkpoint', '--resume-from-checkpoint', dest='resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--vision_pretrained_weights', '--vision-pretrained-weights', dest='vision_pretrained_weights', type=str, help='Path to a CLIP vision warm-start checkpoint')
    parser.add_argument('--clip_unfreeze_last_n_layers', '--clip-unfreeze-last-n-layers', dest='clip_unfreeze_last_n_layers', type=int, help='How many CLIP encoder blocks to fine-tune')
    parser.add_argument('--freeze_backbone', '--freeze-backbone', dest='freeze_backbone', action='store_true', help='Keep the CLIP encoder frozen')
    parser.add_argument('--no_freeze_backbone', '--no-freeze-backbone', dest='freeze_backbone', action='store_false', help='Allow CLIP encoder fine-tuning')
    
    # Mode arguments
    parser.add_argument('--train', action='store_true', default=True, help='Train mode')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only mode')
    parser.add_argument(
        '--mode',
        type=str,
        help='Execution mode',
        choices=[
            'train',
            'finetune_prototypes',
            'generate_medgemma_labels',
            'train_reward_model',
            'train_prompt_policy',
        ],
    )
    parser.add_argument('--medgemma_jsonl_out', type=str, help='Output JSONL for MedGemma supervision')
    parser.add_argument('--reward_jsonl_in', type=str, help='Input JSONL for reward model training')
    parser.add_argument('--reward_ckpt_out', type=str, help='Output checkpoint for reward model')
    parser.add_argument('--reward_ckpt_in', type=str, help='Input checkpoint for reward model')
    parser.add_argument('--prompt_policy_ckpt_out', type=str, help='Output checkpoint for prompt policy')
    parser.add_argument('--prompt_policy_ckpt_in', type=str, help='Input checkpoint for prompt policy')
    parser.set_defaults(freeze_backbone=None)
    
    args = parser.parse_args(argv)
    
    # Create config with defaults
    config = Config()
    
    # Load YAML config if provided
    if args.config:
        if os.path.exists(args.config):
            config.load_from_yaml(args.config)
        else:
            if is_main_process():
                logger.error(f"Config file not found: {args.config}")
            raise FileNotFoundError(f"Config file not found: {args.config}")
    else:
        if is_main_process():
            logger.info("No config file provided, using defaults")
    
    # Override with command-line arguments
    if args.model_type is not None:
        config.model_type = args.model_type

    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    
    if args.lr is not None:
        config.learning_rate = args.lr
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.video_folder is not None:
        config.video_folder = args.video_folder

    if args.root_dir is not None:
        config.root_dir = args.root_dir

    if args.labels_csv is not None:
        config.labels_csv = args.labels_csv

    if args.file_metadata_csv is not None:
        config.file_metadata_csv = args.file_metadata_csv

    if args.split_csv is not None:
        config.split_csv = args.split_csv

    if args.num_workers is not None:
        config.num_workers = args.num_workers

    if args.experiment_dir is not None:
        config.experiment_dir = args.experiment_dir

    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    if args.log_dir is not None:
        config.log_dir = args.log_dir

    if args.save_dir is not None:
        config.save_dir = args.save_dir

    if args.pred_save_dir is not None:
        config.pred_save_dir = args.pred_save_dir
    
    if args.model_weights is not None:
        config.model_weights = args.model_weights
    
    if args.best_model_path is not None:
        config.best_model_path = args.best_model_path
    
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint

    if args.vision_pretrained_weights is not None:
        config.vision_pretrained_weights = args.vision_pretrained_weights

    if args.clip_unfreeze_last_n_layers is not None:
        config.clip_unfreeze_last_n_layers = args.clip_unfreeze_last_n_layers

    if args.freeze_backbone is not None:
        config.freeze_backbone = args.freeze_backbone
    
    if args.eval_only:
        config.train = False
    if args.mode:
        config.run_mode = args.mode
    for k in [
        "medgemma_jsonl_out",
        "reward_jsonl_in",
        "reward_ckpt_out",
        "reward_ckpt_in",
        "prompt_policy_ckpt_out",
        "prompt_policy_ckpt_in",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            setattr(config, k, v)
    
    return config


def main(argv=None):
    """Main training function with distributed support."""
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    if is_main_process():
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Script location: {os.path.abspath(__file__)}")
    
    # Parse command line arguments and load configuration
    config = parse_args_and_load_config(argv)
    
    # DEBUG: Print where files will be saved
    if is_main_process():
        logger.info(f"Experiment directory (absolute): {os.path.abspath(config.experiment_dir)}")
        logger.info(f"Checkpoint directory (absolute): {os.path.abspath(config.checkpoint_dir)}")

    # Optional: authenticate to Hugging Face Hub (for private or gated models)
    if hf_login is not None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            hf_login(token=hf_token, add_to_git_credential=True)

    # Update config with distributed info
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.distributed = world_size > 1

    # Prototype finetuning mode (shared patch clusters)
    if getattr(config, "run_mode", "train") == "finetune_prototypes":
        if is_main_process():
            logger.info("Running mode=finetune_prototypes (shared cluster codebook training)")
        train_prototype_finetune(config, rank=rank, world_size=world_size, local_rank=local_rank)
        cleanup_distributed()
        return
    
    if is_main_process():
        logger.info("TB Ablation Classification Configuration:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")

        # Create finalized artifact directories after all overrides are applied.
        for path in [
            config.experiment_dir,
            config.checkpoint_dir,
            config.log_dir,
            config.save_dir,
            config.pred_save_dir,
        ]:
            if path:
                os.makedirs(path, exist_ok=True)

        config_path = os.path.join(config.experiment_dir, "config.yaml")
        config.save(config_path)
        logger.info(f"Configuration saved to {config_path}")
    
    # Wait for main process to create directories
    if config.distributed:
        distributed_barrier()
    
    # GPU/Device information
    if torch.cuda.is_available() and is_main_process():
        logger.info(f"Using {world_size} GPU(s)")
        logger.info(f"Rank {rank}, Local Rank {local_rank}")
        logger.info(f"Device name: {torch.cuda.get_device_name(local_rank)}")
    
    try:
        # Initialize trainer with distributed parameters
        trainer = AblationTrainer(config, rank, world_size, local_rank)
        
        # Check for resume checkpoint
        resume_checkpoint = None
        if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint is not None:
            if not os.path.exists(config.resume_from_checkpoint):
                if is_main_process():
                    logger.error(f"Resume checkpoint not found: {config.resume_from_checkpoint}")
                cleanup_distributed()
                return
            resume_checkpoint = config.resume_from_checkpoint
        
        # Check if we're in evaluation-only mode
        if not config.train:
            if is_main_process():
                logger.info("Running in evaluation-only mode")
            # Run evaluation on ALL ranks to keep collectives symmetric
            if trainer.is_distributed:
                distributed_barrier()
            trainer._evaluate_best_model()
            if trainer.is_distributed:
                distributed_barrier()
            cleanup_distributed()
            return
        
        # Start training
        if is_main_process():
            logger.info(f"Starting TB training with ablation model: {config.model_type}...")
        
        best_metric, best_epoch = trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        if is_main_process():
            logger.info(f"Training complete! Best metric: {best_metric:.4f} at epoch {best_epoch+1}")
        
        cleanup_distributed()
        return best_metric, best_epoch
    
    except Exception as e:
        logger.exception(f"Error in training: {e}")
        cleanup_distributed()
        raise

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            cleanup_distributed()
        except Exception:
            pass
    
# # 3D CNN ablation
# python3 train_ablation_distributed.py --config configs/3dcnn/fold0.yaml
# sbatch run_ablation_single_node.sh configs/3dcnn/fold1.yaml

# # CNN-LSTM ablation
# python3 train_ablation_distributed.py --config configs/cnnlstm/fold0.yaml
# sbatch run_ablation_single_node.sh configs/cnnlstm/fold1.yaml

# # Video Transformer (ViViT) ablation
# python3 train_ablation_distributed.py --config configs/vivit/fold0.yaml
# sbatch run_ablation_single_node.sh configs/vivit/fold0.yaml

# # R2Plus1D ablation
# python3 train_ablation_distributed.py --config configs/r2plus1d/fold0.yaml
# sbatch run_ablation_single_node.sh configs/r2plus1d/fold0.yaml

# # Inception3D ablation
# python3 train_ablation_distributed.py --config configs/inception3d/fold0.yaml
# sbatch run_ablation_single_node.sh configs/inception3d/fold0.yaml

# # Attention pooling ablation
# python3 train_ablation_distributed.py --config configs/attention_pool/fold0.yaml
# sbatch run_ablation_single_node.sh configs/attention_pool/fold0.yaml

# # LeVIT Attention pooling ablation
# python3 train_ablation_distributed.py --config configs/LeVit-Attention/fold0.yaml
# sbatch run_ablation_single_node.sh configs/LeVit-Attention/fold0.yaml

# # Mean pooling ablation
# python3 train_ablation_distributed.py --config configs/mean_pool/fold4.yaml
# sbatch run_ablation_single_node.sh configs/mean_pool/fold0.yaml

# # Single task ablation
# python3 train_ablation_distributed.py --config configs/singletask/fold0.yaml
# sbatch run_ablation_single_node.sh configs/singletask/fold0.yaml

# # Uniform/No-RL ablation
# python3 train_ablation_distributed.py --config configs/uniform/fold0.yaml
# sbatch run_ablation_single_node.sh configs/uniform/fold0.yaml

# # No-RL full training ablation
# python3 train_ablation_distributed.py --config configs/no_rl_full_train/fold0.yaml
# sbatch run_ablation_single_node.sh configs/no_rl_full_train/fold0.yaml

# # RL with Inception ablation
# python3 train_ablation_distributed.py --config configs/rl_inception/fold0.yaml
# sbatch run_ablation_single_node.sh configs/rl_inception/fold0.yaml

# Original 
# python3 train_ablation_distributed.py --config configs/original/fold0.yaml
# sbatch run_ablation_single_node.sh configs/original/fold0.yaml

# Efficientnet-RL 
# python3 train_ablation_distributed.py --config configs/Efficientnet-RL/fold0.yaml
# sbatch run_ablation_single_node.sh configs/Efficientnet-RL/fold0.yaml

#Dinov3
# python3 train_ablation_distributed.py --config configs/dinov3_bert/fold0.yaml
# sbatch run_ablation_single_node.sh configs/dinov3_bert/fold0.yaml


# bash submit_parallel_ablations.sh parallel
# watch -n 30 'squeue -u $USER'
