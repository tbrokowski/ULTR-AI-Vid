#!/usr/bin/env python3
"""
DANN fine-tuning utilities for cross-domain ultrasound adaptation.

The trainer keeps the existing supervised model and dataset adapters intact,
then adds a small adversarial domain classifier on top of the model's
patient-level representation. Source and target batches are optimized together:
supervised task loss keeps the TB/pathology heads useful, while the gradient
reversal domain loss encourages domain-invariant patient features.
"""

import json
import logging
import math
import random
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from NetworkArchitecture.ablation_models import create_ablation_model
from ultrai.data.registry import load_dataset_adapter


logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        return GradientReversalFunction.apply(inputs, lambda_value)


class DomainClassifier(nn.Module):
    """Small MLP domain head used by DANN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_domains: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _state_dict_from_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint did not contain a state dict: {checkpoint_path}")

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    return state_dict


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "to_dict"):
        return _yaml_safe(config.to_dict())
    return _yaml_safe({
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and not callable(value)
    })


def _task_pos_weights(config: Any) -> Dict[str, float]:
    if hasattr(config, "task_pos_weights") and getattr(config, "task_pos_weights") is not None:
        return {key: float(value) for key, value in getattr(config, "task_pos_weights").items()}
    return {
        task: float(getattr(config, "pos_weight", 1.0))
        for task in getattr(config, "active_tasks", ["TB Label"])
    }


def _make_data_module(config: Any, dataset_name: str, batch_size: Optional[int] = None):
    adapter = load_dataset_adapter(dataset_name)
    data_module = adapter.LungUltrasoundDataModule(
        root_dir=config.root_dir,
        labels_csv=config.labels_csv,
        file_metadata_csv=config.file_metadata_csv,
        image_folder=getattr(config, "image_folder", "images"),
        video_folder=config.video_folder,
        split_csv=getattr(config, "split_csv", None),
        batch_size=int(batch_size or getattr(config, "batch_size", 2)),
        num_workers=int(getattr(config, "num_workers", 4)),
        frame_sampling=int(getattr(config, "frame_sampling", 32)),
        depth_filter=getattr(config, "depth_filter", "all"),
        cache_size=int(getattr(config, "cache_size", 100)),
        files_per_site=getattr(config, "files_per_site", 1),
        site_order=getattr(config, "site_order", None),
        pad_missing_sites=getattr(config, "pad_missing_sites", True),
        max_sites=getattr(config, "max_sites", 15),
    )
    data_module.setup(stage="patient_level")
    return adapter, data_module


def _make_loader(
    dataset,
    collate_fn,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": (shuffle if len(dataset) > 0 else False) and sampler is None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last and len(dataset) >= batch_size,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def _build_weighted_train_sampler(dataset, config: Any, label: str) -> Optional[WeightedRandomSampler]:
    """Oversample TB-positive patients for small target-domain DANN splits."""
    patients = getattr(dataset, "patients", None)
    if not patients:
        return None

    tb_labels = []
    for patient in patients:
        patient_labels = patient.get("patient_labels", {}) if isinstance(patient, dict) else {}
        try:
            tb_labels.append(int(patient_labels.get("TB Label", -1)))
        except (TypeError, ValueError):
            tb_labels.append(-1)

    tb_labels = np.asarray(tb_labels, dtype=np.int64)
    positive_mask = tb_labels == 1
    negative_mask = tb_labels == 0
    num_positive = int(positive_mask.sum())
    num_negative = int(negative_mask.sum())

    if num_positive == 0 or num_negative == 0:
        logger.warning(
            "Skipping DANN %s weighted sampler because class counts are pos=%d, neg=%d",
            label,
            num_positive,
            num_negative,
        )
        return None

    positive_multiplier = max(1, int(getattr(config, "positive_class_multiplier", 1)))
    sample_weights = np.ones(len(tb_labels), dtype=np.float64)
    sample_weights[positive_mask] = float(positive_multiplier)
    num_samples = num_negative + (num_positive * positive_multiplier)

    logger.info(
        "Using DANN %s WeightedRandomSampler: pos=%d, neg=%d, multiplier=%d, samples/epoch=%d",
        label,
        num_positive,
        num_negative,
        positive_multiplier,
        num_samples,
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


class DANNTrainer:
    """Single-process DANN trainer used by the public fine-tuning pipeline."""

    def __init__(
        self,
        target_config: Any,
        source_config: Any,
        source_dataset: str,
        target_dataset: str,
        source_checkpoint_path: str,
        output_dir: str,
        checkpoint_dir: str,
    ) -> None:
        self.target_config = target_config
        self.source_config = source_config
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_checkpoint_path = source_checkpoint_path
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            getattr(target_config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for DANN but unavailable; switching to CPU")
            self.device = torch.device("cpu")
        target_config.device = self.device
        source_config.device = self.device

        self._set_seed(int(getattr(target_config, "seed", 42)))
        self._setup_data()
        self._setup_model()
        self._setup_optimizers()

        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.history = []

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_data(self) -> None:
        target_batch_size = int(getattr(self.target_config, "batch_size", 2))
        source_batch_size_raw = getattr(self.target_config, "dann_source_batch_size", None)
        source_batch_size = (
            target_batch_size
            if source_batch_size_raw in (None, "")
            else int(source_batch_size_raw)
        )
        target_workers = int(getattr(self.target_config, "num_workers", 4))
        source_workers = int(getattr(self.source_config, "num_workers", target_workers))

        self.target_adapter, self.target_data_module = _make_data_module(
            self.target_config,
            self.target_dataset,
            batch_size=target_batch_size,
        )
        self.source_adapter, self.source_data_module = _make_data_module(
            self.source_config,
            self.source_dataset,
            batch_size=source_batch_size,
        )

        target_sampler = None
        oversample_target = bool(
            getattr(
                self.target_config,
                "dann_oversample_target_positive_class",
                getattr(self.target_config, "oversample_positive_class", False),
            )
        )
        if oversample_target:
            target_sampler = _build_weighted_train_sampler(
                self.target_data_module.patient_train,
                self.target_config,
                "target-train",
            )

        self.target_train_loader = _make_loader(
            self.target_data_module.patient_train,
            self.target_adapter.collate_patient_batch,
            target_batch_size,
            target_workers,
            shuffle=True,
            drop_last=True,
            sampler=target_sampler,
        )
        self.target_val_loader = _make_loader(
            self.target_data_module.patient_val,
            self.target_adapter.collate_patient_batch,
            target_batch_size,
            target_workers,
            shuffle=False,
            drop_last=False,
        )
        self.target_test_loader = _make_loader(
            self.target_data_module.patient_test,
            self.target_adapter.collate_patient_batch,
            target_batch_size,
            target_workers,
            shuffle=False,
            drop_last=False,
        )
        self.source_train_loader = _make_loader(
            self.source_data_module.patient_train,
            self.source_adapter.collate_patient_batch,
            source_batch_size,
            source_workers,
            shuffle=True,
            drop_last=True,
        )

        if len(self.target_train_loader) == 0:
            raise ValueError("DANN target train loader is empty")
        if len(self.source_train_loader) == 0:
            raise ValueError("DANN source train loader is empty")

        logger.info(
            "DANN data: source train=%d patients (%d batches), target train=%d val=%d test=%d patients (%d train batches)",
            len(self.source_data_module.patient_train),
            len(self.source_train_loader),
            len(self.target_data_module.patient_train),
            len(self.target_data_module.patient_val),
            len(self.target_data_module.patient_test),
            len(self.target_train_loader),
        )

    def _setup_model(self) -> None:
        self.model = create_ablation_model(
            getattr(self.target_config, "model_type", "tb_rl_mil"),
            self.target_config,
        )

        source_state_dict = _state_dict_from_checkpoint(self.source_checkpoint_path, self.device)
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in source_state_dict.items()
            if key in model_state_dict and model_state_dict[key].shape == value.shape
        }
        if not filtered_state_dict:
            raise ValueError(f"No compatible model tensors found in {self.source_checkpoint_path}")

        model_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_state_dict, strict=False)
        logger.info(
            "Loaded %d/%d compatible source checkpoint tensors for DANN from %s",
            len(filtered_state_dict),
            len(source_state_dict),
            self.source_checkpoint_path,
        )

        feature_dim = int(getattr(self.target_config, "hidden_dim", 512))
        domain_hidden_dim = int(getattr(self.target_config, "dann_domain_hidden_dim", max(128, feature_dim // 2)))
        domain_dropout = float(getattr(self.target_config, "dann_domain_dropout", 0.2))
        self.domain_classifier = DomainClassifier(
            input_dim=feature_dim,
            hidden_dim=domain_hidden_dim,
            dropout=domain_dropout,
            num_domains=2,
        )
        self.grl = GradientReversal()

        self.model = self.model.to(self.device)
        self.domain_classifier = self.domain_classifier.to(self.device)

        logger.info(
            "DANN trainable parameters: model=%s, domain_classifier=%s",
            f"{sum(param.numel() for param in self.model.parameters() if param.requires_grad):,}",
            f"{sum(param.numel() for param in self.domain_classifier.parameters() if param.requires_grad):,}",
        )

    def _setup_optimizers(self) -> None:
        model_params = [param for param in self.model.parameters() if param.requires_grad]
        domain_params = [param for param in self.domain_classifier.parameters() if param.requires_grad]
        if not model_params:
            raise ValueError("DANN model has no trainable parameters")

        model_lr = float(getattr(self.target_config, "dann_learning_rate", getattr(self.target_config, "patient_pipeline_lr", 1e-4)))
        domain_lr = float(getattr(self.target_config, "dann_domain_lr", model_lr))
        weight_decay = float(getattr(self.target_config, "dann_weight_decay", getattr(self.target_config, "weight_decay", 1e-5)))
        domain_weight_decay = float(getattr(self.target_config, "dann_domain_weight_decay", weight_decay))

        self.optimizer = torch.optim.AdamW(
            [
                {"params": model_params, "lr": model_lr, "weight_decay": weight_decay},
                {"params": domain_params, "lr": domain_lr, "weight_decay": domain_weight_decay},
            ]
        )

        total_steps = max(1, len(self.target_train_loader) * int(getattr(self.target_config, "num_epochs", 1)))
        eta_min = float(getattr(self.target_config, "dann_eta_min", 1e-6))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=eta_min,
        )
        self.use_amp = bool(getattr(self.target_config, "use_amp", True)) and self.device.type == "cuda"
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=self.use_amp)

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        site_videos = batch["site_videos"].to(self.device, non_blocking=True)
        site_indices = batch["site_indices"].to(self.device, non_blocking=True)
        if "site_masks" in batch and batch["site_masks"] is not None:
            site_masks = batch["site_masks"].to(self.device, non_blocking=True)
        else:
            site_masks = torch.ones(
                site_videos.shape[0],
                site_videos.shape[1],
                dtype=torch.bool,
                device=self.device,
            )
        site_findings = batch["site_findings"].to(self.device, non_blocking=True)
        tb_labels = batch["tb_labels"].to(self.device, non_blocking=True).float()
        pneumonia_labels = batch.get("pneumonia_labels")
        covid_labels = batch.get("covid_labels")
        if pneumonia_labels is None:
            pneumonia_labels = torch.full_like(tb_labels, -1)
        else:
            pneumonia_labels = pneumonia_labels.to(self.device, non_blocking=True).float()
        if covid_labels is None:
            covid_labels = torch.full_like(tb_labels, -1)
        else:
            covid_labels = covid_labels.to(self.device, non_blocking=True).float()

        inputs = {
            "site_videos": site_videos,
            "site_indices": site_indices,
            "site_masks": site_masks,
            "site_findings": site_findings,
            "is_patient_level": True,
        }
        targets = {
            "tb_labels": tb_labels,
            "pneumonia_labels": pneumonia_labels,
            "covid_labels": covid_labels,
            "pathology_labels": site_findings,
            "site_masks": site_masks,
        }
        return inputs, targets

    def _grl_lambda(self, global_step: int, total_steps: int) -> float:
        max_lambda = float(getattr(self.target_config, "dann_lambda", 1.0))
        if not bool(getattr(self.target_config, "dann_lambda_schedule", True)):
            return max_lambda

        warmup_steps = int(getattr(self.target_config, "dann_domain_warmup_steps", 0) or 0)
        warmup_epochs = float(getattr(self.target_config, "dann_domain_warmup_epochs", 0.0) or 0.0)
        if warmup_epochs > 0:
            warmup_steps = max(warmup_steps, int(round(warmup_epochs * len(self.target_train_loader))))
        if global_step < warmup_steps:
            return 0.0

        gamma = float(getattr(self.target_config, "dann_lambda_gamma", 10.0))
        scheduled_steps = max(float(total_steps - warmup_steps), 1.0)
        progress = min(max(float(global_step - warmup_steps) / scheduled_steps, 0.0), 1.0)
        return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)

    def _domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        lambda_value: float,
    ) -> Tuple[torch.Tensor, float]:
        features = torch.cat([source_features, target_features], dim=0)
        reversed_features = self.grl(features, lambda_value)
        logits = self.domain_classifier(reversed_features)
        labels = torch.cat(
            [
                torch.zeros(source_features.shape[0], dtype=torch.long, device=features.device),
                torch.ones(target_features.shape[0], dtype=torch.long, device=features.device),
            ],
            dim=0,
        )
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        return loss, accuracy

    def _reset_model_state(self, clear_history: bool = False) -> None:
        selector = getattr(self.model, "frame_selector", None)
        if selector is None:
            return
        if hasattr(selector, "saved_actions"):
            selector.saved_actions = []
        if hasattr(selector, "reset_rewards"):
            selector.reset_rewards()
        if clear_history and hasattr(selector, "clear_history"):
            selector.clear_history()

    def _reset_frame_selector_state(self, clear_history: bool = False) -> None:
        self._reset_model_state(clear_history=clear_history)

    def train_epoch(self, epoch: int, total_steps: int, start_step: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.domain_classifier.train()
        self._reset_model_state(clear_history=True)

        source_iter = cycle(self.source_train_loader)
        source_pos_weights = _task_pos_weights(self.source_config)
        target_pos_weights = _task_pos_weights(self.target_config)
        use_source_task = bool(getattr(self.target_config, "dann_use_source_task_loss", True))
        use_target_task = bool(getattr(self.target_config, "dann_use_target_task_loss", True))
        source_task_weight = float(getattr(self.target_config, "dann_source_task_weight", 1.0))
        target_task_weight = float(getattr(self.target_config, "dann_target_task_weight", 1.0))
        domain_loss_weight = float(getattr(self.target_config, "dann_domain_loss_weight", 1.0))
        grad_clip_norm = float(getattr(self.target_config, "dann_gradient_clip_norm", 1.0))
        accumulation_steps = max(1, int(getattr(self.target_config, "dann_accumulation_steps", 1)))
        log_every_steps = max(1, int(getattr(self.target_config, "dann_log_every_steps", 10)))

        running = {
            "loss": 0.0,
            "source_task_loss": 0.0,
            "target_task_loss": 0.0,
            "domain_loss": 0.0,
            "domain_accuracy": 0.0,
        }
        steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.target_train_loader,
            desc=f"DANN Epoch {epoch + 1}/{int(getattr(self.target_config, 'num_epochs', 1))}",
        )
        for batch_idx, target_batch in enumerate(progress):
            source_batch = next(source_iter)
            source_inputs, source_targets = self._prepare_batch(source_batch)
            target_inputs, target_targets = self._prepare_batch(target_batch)
            global_step = start_step + batch_idx
            lambda_value = self._grl_lambda(global_step, total_steps)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                source_outputs = self.model(source_inputs)
                target_outputs = self.model(target_inputs)

                source_task_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                target_task_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if use_source_task:
                    source_task_loss, _ = self.model.compute_losses(
                        source_outputs,
                        source_targets,
                        source_pos_weights,
                    )
                if use_target_task:
                    target_task_loss, _ = self.model.compute_losses(
                        target_outputs,
                        target_targets,
                        target_pos_weights,
                    )

                domain_loss, domain_accuracy = self._domain_loss(
                    source_outputs["patient_features"],
                    target_outputs["patient_features"],
                    lambda_value,
                )
                total_loss = (
                    source_task_weight * source_task_loss
                    + target_task_weight * target_task_loss
                    + domain_loss_weight * domain_loss
                )
                scaled_loss = total_loss / accumulation_steps

            self.scaler.scale(scaled_loss).backward()
            should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(self.target_train_loader))
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.domain_classifier.parameters()),
                        grad_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            running["loss"] += float(total_loss.detach().item())
            running["source_task_loss"] += float(source_task_loss.detach().item())
            running["target_task_loss"] += float(target_task_loss.detach().item())
            running["domain_loss"] += float(domain_loss.detach().item())
            running["domain_accuracy"] += float(domain_accuracy)
            steps += 1

            if (batch_idx + 1) % log_every_steps == 0:
                logger.info(
                    "DANN epoch=%d step=%d/%d loss=%.4f src_task=%.4f tgt_task=%.4f domain=%.4f domain_acc=%.4f grl_lambda=%.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(self.target_train_loader),
                    running["loss"] / steps,
                    running["source_task_loss"] / steps,
                    running["target_task_loss"] / steps,
                    running["domain_loss"] / steps,
                    running["domain_accuracy"] / steps,
                    lambda_value,
                )

            progress.set_postfix(
                {
                    "loss": running["loss"] / max(steps, 1),
                    "domain_acc": running["domain_accuracy"] / max(steps, 1),
                    "lambda": f"{lambda_value:.2f}",
                }
            )
            self._reset_model_state()

        return running["loss"] / max(steps, 1), {key: value / max(steps, 1) for key, value in running.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, split_name: str = "val") -> Tuple[float, Dict[str, float]]:
        if loader is None or len(loader) == 0:
            logger.warning("No %s data available for DANN validation", split_name)
            return 0.0, {"loss": 0.0, "TB Label_auc": 0.0, "TB Label_auprc": 0.0}

        self.model.eval()
        self.domain_classifier.eval()
        self._reset_model_state(clear_history=True)
        target_pos_weights = _task_pos_weights(self.target_config)

        losses = []
        labels = []
        probs = []
        preds = []

        for batch in tqdm(loader, desc=f"DANN {split_name.capitalize()} Evaluation"):
            inputs, targets = self._prepare_batch(batch)
            outputs = self.model(inputs)
            loss, _ = self.model.compute_losses(outputs, targets, target_pos_weights)
            losses.append(float(loss.item()))

            task_logits = outputs.get("task_logits", {})
            if "TB Label" in task_logits:
                logits = task_logits["TB Label"]
            elif "tb_logits" in outputs:
                logits = outputs["tb_logits"]
            else:
                continue
            batch_probs = torch.sigmoid(logits)
            labels.append(targets["tb_labels"].detach().cpu())
            probs.append(batch_probs.detach().cpu())
            preds.append((batch_probs > 0.5).float().detach().cpu())
            self._reset_model_state()

        metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
        if labels and probs and preds:
            label_np = torch.cat(labels).numpy().reshape(-1)
            prob_np = torch.cat(probs).numpy().reshape(-1)
            pred_np = torch.cat(preds).numpy().reshape(-1)
            metrics.update(_binary_metrics(label_np, prob_np, pred_np))

        logger.info(
            "DANN %s metrics: loss=%.4f auc=%.4f auprc=%.4f",
            split_name,
            metrics.get("loss", 0.0),
            metrics.get("TB Label_auc", 0.0),
            metrics.get("TB Label_auprc", 0.0),
        )
        return metrics["loss"], metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool) -> Path:
        checkpoint = {
            "epoch": epoch,
            "algorithm": "dann",
            "model_state_dict": self.model.state_dict(),
            "domain_classifier_state_dict": self.domain_classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": _config_to_dict(self.target_config),
            "source_config": _config_to_dict(self.source_config),
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "metrics": metrics,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "checkpoint_best.pth")
            metric = metrics.get(f"TB Label_{getattr(self.target_config, 'eval_metric', 'auc')}", metrics.get("loss", 0.0))
            torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_best_metric_{metric:.4f}.pth")
        return latest_path

    def train(self) -> Tuple[float, int]:
        import yaml

        with open(self.checkpoint_dir / "config.yaml", "w") as handle:
            yaml.safe_dump(_config_to_dict(self.target_config), handle, sort_keys=False)
        with open(self.checkpoint_dir / "source_config.yaml", "w") as handle:
            yaml.safe_dump(_config_to_dict(self.source_config), handle, sort_keys=False)

        num_epochs = int(getattr(self.target_config, "num_epochs", 1))
        total_steps = max(1, num_epochs * len(self.target_train_loader))
        patience = int(getattr(self.target_config, "early_stopping_patience", 8))
        metric_name = f"TB Label_{getattr(self.target_config, 'eval_metric', 'auc')}"
        min_best_epoch = max(1, int(getattr(self.target_config, "dann_min_best_epoch", 1) or 1))

        for epoch in range(num_epochs):
            epoch_number = epoch + 1
            train_loss, train_metrics = self.train_epoch(
                epoch,
                total_steps=total_steps,
                start_step=epoch * len(self.target_train_loader),
            )
            val_loader = self.target_val_loader if len(self.target_val_loader) > 0 else self.target_train_loader
            use_train_metric = len(self.target_val_loader) == 0
            val_loss, val_metrics = self.validate(
                val_loader,
                split_name="train" if use_train_metric else "val",
            )

            metric = val_metrics.get(metric_name, -val_loss)
            eligible_for_best = epoch_number >= min_best_epoch
            metric_improved = metric > self.best_metric
            is_best = eligible_for_best and metric_improved
            if is_best:
                self.best_metric = metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            elif eligible_for_best:
                self.epochs_without_improvement += 1
            else:
                logger.info(
                    "DANN epoch %d/%d metric %.4f is before min best epoch %d; not checkpointing as best",
                    epoch_number,
                    num_epochs,
                    metric,
                    min_best_epoch,
                )

            epoch_record = {
                "epoch": epoch_number,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_metric": metric,
                "eligible_for_best": eligible_for_best,
                "metric_improved": metric_improved,
                "is_best": is_best,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            self.history.append(epoch_record)
            self.save_checkpoint(epoch, {**val_metrics, "loss": val_loss}, is_best=is_best)
            self._write_metrics()

            logger.info(
                "DANN epoch %d/%d complete: train_loss=%.4f val_loss=%.4f %s=%.4f best=%.4f",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                metric_name,
                metric,
                self.best_metric,
            )

            if self.epochs_without_improvement >= patience:
                logger.info("DANN early stopping after %d epochs without improvement", self.epochs_without_improvement)
                break

        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "domain_classifier_state_dict" in checkpoint:
                self.domain_classifier.load_state_dict(checkpoint["domain_classifier_state_dict"], strict=False)

        return self.best_metric, self.best_epoch

    def _write_metrics(self) -> None:
        payload = {
            "algorithm": "dann",
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else None,
            "history": self.history,
        }
        with open(self.output_dir / "dann_metrics.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)


def _binary_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    labels = labels.reshape(-1)
    probs = probs.reshape(-1)
    preds = preds.reshape(-1)
    metrics = {
        "TB Label_accuracy": accuracy_score(labels, preds),
        "TB Label_precision": precision_score(labels, preds, zero_division=0),
        "TB Label_recall": recall_score(labels, preds, zero_division=0),
        "TB Label_specificity": recall_score(1 - labels, 1 - preds, zero_division=0),
        "TB Label_f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["TB Label_auc"] = roc_auc_score(labels, probs)
        metrics["TB Label_auprc"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["TB Label_auc"] = 0.5
        metrics["TB Label_auprc"] = 0.5

    positive_mask = labels == 1
    negative_mask = labels == 0
    metrics.update(
        {
            "TB Label_n_positive": int(positive_mask.sum()),
            "TB Label_n_negative": int(negative_mask.sum()),
            "TB Label_pred_positive_count": int(preds.sum()),
            "TB Label_pred_positive_rate": float(preds.mean()) if preds.size else 0.0,
            "TB Label_prob_mean": float(probs.mean()) if probs.size else 0.0,
            "TB Label_prob_std": float(probs.std()) if probs.size else 0.0,
            "TB Label_prob_min": float(probs.min()) if probs.size else 0.0,
            "TB Label_prob_max": float(probs.max()) if probs.size else 0.0,
            "TB Label_positive_prob_mean": float(probs[positive_mask].mean()) if positive_mask.any() else 0.0,
            "TB Label_negative_prob_mean": float(probs[negative_mask].mean()) if negative_mask.any() else 0.0,
        }
    )
    return metrics
