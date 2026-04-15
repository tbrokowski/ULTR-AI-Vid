#!/usr/bin/env python3
"""
FixMatch fine-tuning utilities for target-domain ultrasound adaptation.

The trainer keeps the existing patient-level dataset adapters and model factory
intact. It optimizes supervised loss on the labeled target subset, then adds a
FixMatch consistency term on target-domain training-pool patients whose weak
view produces high-confidence pseudo-labels.
"""

import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from NetworkArchitecture.ablation_models import create_ablation_model
from ultrai.data.adapters.common import SimpleVideoTransforms
from ultrai.data.registry import load_dataset_adapter


logger = logging.getLogger(__name__)


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


def _make_data_module(
    config: Any,
    dataset_name: str,
    batch_size: Optional[int] = None,
    train_transform: str = "default",
):
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
    if train_transform == "simple":
        data_module.train_video_transforms = SimpleVideoTransforms()
    data_module.setup(stage="patient_level")
    return adapter, data_module


def _make_loader(
    dataset,
    collate_fn,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if len(dataset) > 0 else False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last and len(dataset) >= batch_size,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


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
    return metrics


class FixMatchTrainer:
    """Single-process FixMatch trainer used by the public fine-tuning pipeline."""

    def __init__(
        self,
        target_config: Any,
        target_dataset: str,
        source_checkpoint_path: str,
        output_dir: str,
        checkpoint_dir: str,
        unlabeled_split_csv: str,
    ) -> None:
        self.target_config = target_config
        self.target_dataset = target_dataset
        self.source_checkpoint_path = source_checkpoint_path
        self.unlabeled_split_csv = unlabeled_split_csv
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            getattr(target_config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for FixMatch but unavailable; switching to CPU")
            self.device = torch.device("cpu")
        target_config.device = self.device

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
        unlabeled_batch_size_raw = getattr(self.target_config, "fixmatch_unlabeled_batch_size", None)
        if unlabeled_batch_size_raw in (None, ""):
            batch_ratio = float(getattr(self.target_config, "fixmatch_unlabeled_batch_ratio", 1.0))
            unlabeled_batch_size = max(1, int(round(target_batch_size * batch_ratio)))
        else:
            unlabeled_batch_size = int(unlabeled_batch_size_raw)

        target_workers = int(getattr(self.target_config, "num_workers", 4))

        self.target_adapter, self.target_data_module = _make_data_module(
            self.target_config,
            self.target_dataset,
            batch_size=target_batch_size,
            train_transform="default",
        )

        unlabeled_config = SimpleNamespace(**_config_to_dict(self.target_config))
        unlabeled_config.split_csv = self.unlabeled_split_csv
        self.unlabeled_adapter, self.unlabeled_data_module = _make_data_module(
            unlabeled_config,
            self.target_dataset,
            batch_size=unlabeled_batch_size,
            train_transform="simple",
        )

        self.target_train_loader = _make_loader(
            self.target_data_module.patient_train,
            self.target_adapter.collate_patient_batch,
            target_batch_size,
            target_workers,
            shuffle=True,
            drop_last=True,
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
        self.unlabeled_train_loader = _make_loader(
            self.unlabeled_data_module.patient_train,
            self.unlabeled_adapter.collate_patient_batch,
            unlabeled_batch_size,
            target_workers,
            shuffle=True,
            drop_last=True,
        )

        if len(self.target_train_loader) == 0:
            raise ValueError("FixMatch labeled target train loader is empty")
        if len(self.unlabeled_train_loader) == 0:
            raise ValueError("FixMatch unlabeled target train loader is empty")

        logger.info(
            "FixMatch data: labeled target train=%d patients (%d batches), unlabeled target train=%d patients (%d batches), val=%d, test=%d",
            len(self.target_data_module.patient_train),
            len(self.target_train_loader),
            len(self.unlabeled_data_module.patient_train),
            len(self.unlabeled_train_loader),
            len(self.target_data_module.patient_val),
            len(self.target_data_module.patient_test),
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
        self.model = self.model.to(self.device)

        logger.info(
            "Loaded %d/%d compatible source checkpoint tensors for FixMatch from %s",
            len(filtered_state_dict),
            len(source_state_dict),
            self.source_checkpoint_path,
        )
        logger.info(
            "FixMatch trainable parameters: model=%s",
            f"{sum(param.numel() for param in self.model.parameters() if param.requires_grad):,}",
        )

    def _setup_optimizers(self) -> None:
        model_params = [param for param in self.model.parameters() if param.requires_grad]
        if not model_params:
            raise ValueError("FixMatch model has no trainable parameters")

        model_lr = float(
            getattr(
                self.target_config,
                "fixmatch_learning_rate",
                getattr(self.target_config, "patient_pipeline_lr", 1e-4),
            )
        )
        weight_decay = float(
            getattr(
                self.target_config,
                "fixmatch_weight_decay",
                getattr(self.target_config, "weight_decay", 1e-5),
            )
        )
        self.optimizer = torch.optim.AdamW(model_params, lr=model_lr, weight_decay=weight_decay)

        total_steps = max(1, len(self.target_train_loader) * int(getattr(self.target_config, "num_epochs", 1)))
        eta_min = float(getattr(self.target_config, "fixmatch_eta_min", 1e-6))
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

    def _clone_inputs_with_videos(
        self,
        inputs: Dict[str, torch.Tensor],
        site_videos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        view_inputs = dict(inputs)
        view_inputs["site_videos"] = site_videos
        return view_inputs

    def _weak_view(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        videos = inputs["site_videos"].clone()
        weak_noise_std = float(getattr(self.target_config, "fixmatch_weak_noise_std", 0.0))
        if weak_noise_std > 0:
            valid = inputs["site_masks"].view(*inputs["site_masks"].shape, 1, 1, 1, 1).to(videos.dtype)
            videos = videos + torch.randn_like(videos) * weak_noise_std * valid
        return self._clone_inputs_with_videos(inputs, videos)

    def _strong_view(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        videos = inputs["site_videos"].clone()
        site_masks = inputs["site_masks"].bool()
        valid = site_masks.view(*site_masks.shape, 1, 1, 1, 1).to(videos.dtype)

        flip_prob = float(getattr(self.target_config, "fixmatch_strong_flip_prob", 0.5))
        if flip_prob > 0:
            flip_mask = (torch.rand(site_masks.shape, device=videos.device) < flip_prob) & site_masks
            if flip_mask.any():
                videos[flip_mask] = torch.flip(videos[flip_mask], dims=[-1])

        intensity_jitter = float(getattr(self.target_config, "fixmatch_strong_intensity_jitter", 0.15))
        if intensity_jitter > 0:
            factor = 1.0 + (torch.rand((*site_masks.shape, 1, 1, 1, 1), device=videos.device) * 2.0 - 1.0) * intensity_jitter
            shift = (torch.rand((*site_masks.shape, 1, 1, 1, 1), device=videos.device) * 2.0 - 1.0) * intensity_jitter
            videos = (videos * factor + shift) * valid + videos * (1.0 - valid)

        noise_std = float(getattr(self.target_config, "fixmatch_strong_noise_std", 0.05))
        if noise_std > 0:
            videos = videos + torch.randn_like(videos) * noise_std * valid

        frame_dropout_prob = float(getattr(self.target_config, "fixmatch_strong_temporal_dropout_prob", 0.15))
        if frame_dropout_prob > 0 and videos.shape[2] > 1:
            frame_mask = (
                torch.rand((*site_masks.shape, videos.shape[2], 1, 1, 1), device=videos.device)
                < frame_dropout_prob
            )
            frame_mask = frame_mask & site_masks.view(*site_masks.shape, 1, 1, 1, 1)
            videos = torch.where(frame_mask, torch.zeros_like(videos), videos)

        cutout_prob = float(getattr(self.target_config, "fixmatch_strong_cutout_prob", 0.5))
        cutout_ratio = float(getattr(self.target_config, "fixmatch_strong_cutout_ratio", 0.25))
        if cutout_prob > 0 and cutout_ratio > 0:
            _, _, _, _, height, width = videos.shape
            cut_h = max(1, int(round(height * cutout_ratio)))
            cut_w = max(1, int(round(width * cutout_ratio)))
            cutout_mask = (torch.rand(site_masks.shape, device=videos.device) < cutout_prob) & site_masks
            cutout_positions = torch.nonzero(cutout_mask, as_tuple=False)
            for batch_idx, site_idx in cutout_positions.tolist():
                top = random.randint(0, max(height - cut_h, 0))
                left = random.randint(0, max(width - cut_w, 0))
                videos[batch_idx, site_idx, :, :, top:top + cut_h, left:left + cut_w] = 0.0

        clip_min = float(getattr(self.target_config, "fixmatch_aug_clip_min", -4.0))
        clip_max = float(getattr(self.target_config, "fixmatch_aug_clip_max", 4.0))
        if clip_min < clip_max:
            videos = videos.clamp(min=clip_min, max=clip_max)

        return self._clone_inputs_with_videos(inputs, videos)

    def _get_tb_logits(self, outputs: Dict[str, Any]) -> torch.Tensor:
        task_logits = outputs.get("task_logits", {})
        if "TB Label" in task_logits:
            logits = task_logits["TB Label"]
        elif "tb_logits" in outputs:
            logits = outputs["tb_logits"]
        else:
            raise KeyError("Model output did not contain TB logits")
        return logits.reshape(-1)

    def _pseudo_labels(self, weak_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = max(float(getattr(self.target_config, "fixmatch_temperature", 1.0)), 1e-6)
        threshold = float(getattr(self.target_config, "fixmatch_confidence_threshold", 0.95))
        probs = torch.sigmoid(weak_logits / temperature)
        confidence = torch.maximum(probs, 1.0 - probs)
        pseudo = (probs >= 0.5).float()
        mask = confidence >= threshold
        return pseudo.detach(), mask.detach(), confidence.detach()

    def _fixmatch_loss(
        self,
        strong_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = F.binary_cross_entropy_with_logits(strong_logits, pseudo_labels, reduction="none")
        mask = confidence_mask.to(losses.dtype)
        if losses.numel() == 0:
            return strong_logits.sum() * 0.0
        return (losses * mask).mean()

    def _unsup_weight(self, epoch: int) -> float:
        base_weight = float(getattr(self.target_config, "fixmatch_lambda_u", 1.0))
        warmup_epochs = int(getattr(self.target_config, "fixmatch_unsup_warmup_epochs", 1))
        if warmup_epochs <= 0:
            return base_weight
        return base_weight * min(1.0, float(epoch + 1) / float(warmup_epochs))

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self._reset_model_state(clear_history=True)

        unlabeled_iter = iter(self.unlabeled_train_loader)
        pos_weights = _task_pos_weights(self.target_config)
        unsup_weight = self._unsup_weight(epoch)
        grad_clip_norm = float(getattr(self.target_config, "fixmatch_gradient_clip_norm", 1.0))
        accumulation_steps = max(1, int(getattr(self.target_config, "fixmatch_accumulation_steps", 1)))
        log_every_steps = max(1, int(getattr(self.target_config, "fixmatch_log_every_steps", 10)))

        running = {
            "loss": 0.0,
            "supervised_loss": 0.0,
            "unsupervised_loss": 0.0,
            "confidence_rate": 0.0,
            "mean_confidence": 0.0,
            "pseudo_positive_rate": 0.0,
        }
        steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.target_train_loader,
            desc=f"FixMatch Epoch {epoch + 1}/{int(getattr(self.target_config, 'num_epochs', 1))}",
        )
        for batch_idx, labeled_batch in enumerate(progress):
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_train_loader)
                unlabeled_batch = next(unlabeled_iter)

            labeled_inputs, labeled_targets = self._prepare_batch(labeled_batch)
            unlabeled_inputs, _ = self._prepare_batch(unlabeled_batch)
            weak_inputs = self._weak_view(unlabeled_inputs)
            strong_inputs = self._strong_view(unlabeled_inputs)

            self.model.eval()
            self._reset_model_state(clear_history=True)
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                weak_outputs = self.model(weak_inputs)
                weak_logits = self._get_tb_logits(weak_outputs)
                pseudo_labels, confidence_mask, confidence = self._pseudo_labels(weak_logits)
            self._reset_model_state(clear_history=True)
            self.model.train()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                labeled_outputs = self.model(labeled_inputs)
                supervised_loss, _ = self.model.compute_losses(
                    labeled_outputs,
                    labeled_targets,
                    pos_weights,
                )

                strong_outputs = self.model(strong_inputs)
                strong_logits = self._get_tb_logits(strong_outputs)
                unsupervised_loss = self._fixmatch_loss(
                    strong_logits,
                    pseudo_labels,
                    confidence_mask,
                )
                total_loss = supervised_loss + unsup_weight * unsupervised_loss
                scaled_loss = total_loss / accumulation_steps

            self.scaler.scale(scaled_loss).backward()
            should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(self.target_train_loader))
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            self._reset_model_state()

            confident = confidence_mask.float()
            if confidence_mask.any():
                pseudo_positive_rate = pseudo_labels[confidence_mask].float().mean().item()
            else:
                pseudo_positive_rate = 0.0

            running["loss"] += float(total_loss.detach().item())
            running["supervised_loss"] += float(supervised_loss.detach().item())
            running["unsupervised_loss"] += float(unsupervised_loss.detach().item())
            running["confidence_rate"] += float(confident.mean().item()) if confident.numel() else 0.0
            running["mean_confidence"] += float(confidence.mean().item()) if confidence.numel() else 0.0
            running["pseudo_positive_rate"] += float(pseudo_positive_rate)
            steps += 1

            if (batch_idx + 1) % log_every_steps == 0:
                logger.info(
                    "FixMatch epoch=%d step=%d/%d loss=%.4f sup=%.4f unsup=%.4f conf_rate=%.4f mean_conf=%.4f lambda_u=%.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(self.target_train_loader),
                    running["loss"] / steps,
                    running["supervised_loss"] / steps,
                    running["unsupervised_loss"] / steps,
                    running["confidence_rate"] / steps,
                    running["mean_confidence"] / steps,
                    unsup_weight,
                )

            progress.set_postfix(
                {
                    "loss": running["loss"] / max(steps, 1),
                    "sup": running["supervised_loss"] / max(steps, 1),
                    "conf": running["confidence_rate"] / max(steps, 1),
                }
            )

        return running["loss"] / max(steps, 1), {key: value / max(steps, 1) for key, value in running.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, split_name: str = "val") -> Tuple[float, Dict[str, float]]:
        if loader is None or len(loader) == 0:
            logger.warning("No %s data available for FixMatch validation", split_name)
            return 0.0, {"loss": 0.0, "TB Label_auc": 0.0, "TB Label_auprc": 0.0}

        self.model.eval()
        self._reset_model_state(clear_history=True)
        target_pos_weights = _task_pos_weights(self.target_config)

        losses = []
        labels = []
        probs = []
        preds = []

        for batch in tqdm(loader, desc=f"FixMatch {split_name.capitalize()} Evaluation"):
            inputs, targets = self._prepare_batch(batch)
            outputs = self.model(inputs)
            loss, _ = self.model.compute_losses(outputs, targets, target_pos_weights)
            losses.append(float(loss.item()))

            logits = self._get_tb_logits(outputs)
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
            valid_mask = label_np >= 0
            if valid_mask.any():
                metrics.update(_binary_metrics(label_np[valid_mask], prob_np[valid_mask], pred_np[valid_mask]))

        logger.info(
            "FixMatch %s metrics: loss=%.4f auc=%.4f auprc=%.4f",
            split_name,
            metrics.get("loss", 0.0),
            metrics.get("TB Label_auc", 0.0),
            metrics.get("TB Label_auprc", 0.0),
        )
        return metrics["loss"], metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool) -> Path:
        checkpoint = {
            "epoch": epoch,
            "algorithm": "fixmatch",
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": _config_to_dict(self.target_config),
            "target_dataset": self.target_dataset,
            "source_checkpoint_path": self.source_checkpoint_path,
            "unlabeled_split_csv": self.unlabeled_split_csv,
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

        num_epochs = int(getattr(self.target_config, "num_epochs", 1))
        patience = int(getattr(self.target_config, "early_stopping_patience", 8))
        metric_name = f"TB Label_{getattr(self.target_config, 'eval_metric', 'auc')}"

        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loader = self.target_val_loader if len(self.target_val_loader) > 0 else self.target_train_loader
            use_train_metric = len(self.target_val_loader) == 0
            val_loss, val_metrics = self.validate(
                val_loader,
                split_name="train" if use_train_metric else "val",
            )

            metric = val_metrics.get(metric_name, -val_loss)
            is_best = metric > self.best_metric
            if is_best:
                self.best_metric = metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_metric": metric,
                "is_best": is_best,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            self.history.append(epoch_record)
            self.save_checkpoint(epoch, {**val_metrics, "loss": val_loss}, is_best=is_best)
            self._write_metrics()

            logger.info(
                "FixMatch epoch %d/%d complete: train_loss=%.4f val_loss=%.4f %s=%.4f best=%.4f",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                metric_name,
                metric,
                self.best_metric,
            )

            if self.epochs_without_improvement >= patience:
                logger.info("FixMatch early stopping after %d epochs without improvement", self.epochs_without_improvement)
                break

        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return self.best_metric, self.best_epoch

    def _write_metrics(self) -> None:
        payload = {
            "algorithm": "fixmatch",
            "target_dataset": self.target_dataset,
            "source_checkpoint_path": self.source_checkpoint_path,
            "unlabeled_split_csv": self.unlabeled_split_csv,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else None,
            "history": self.history,
        }
        with open(self.output_dir / "fixmatch_metrics.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
