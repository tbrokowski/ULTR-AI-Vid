#!/usr/bin/env python3
"""
Learning without Forgetting fine-tuning utilities.

LwF keeps the source-domain checkpoint as a frozen teacher. The target-domain
student is initialized from the same checkpoint, then optimized with the usual
supervised target loss plus temperature-scaled distillation losses that keep the
student close to the teacher on old outputs.
"""

import json
import logging
import random
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
    if config is None:
        return {}
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


def _as_name_list(value: Any) -> Tuple[str, ...]:
    if value in (None, "", []):
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


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


class LWFTrainer:
    """Single-process LwF trainer used by the public fine-tuning pipeline."""

    def __init__(
        self,
        target_config: Any,
        target_dataset: str,
        source_checkpoint_path: str,
        output_dir: str,
        checkpoint_dir: str,
        source_config: Optional[Any] = None,
        source_dataset: Optional[str] = None,
    ) -> None:
        self.target_config = target_config
        self.target_dataset = target_dataset
        self.source_checkpoint_path = source_checkpoint_path
        self.source_config = source_config
        self.source_dataset = source_dataset
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            getattr(target_config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for LwF but unavailable; switching to CPU")
            self.device = torch.device("cpu")
        target_config.device = self.device
        if source_config is not None:
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
        target_workers = int(getattr(self.target_config, "num_workers", 4))

        self.target_adapter, self.target_data_module = _make_data_module(
            self.target_config,
            self.target_dataset,
            batch_size=target_batch_size,
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

        self.source_train_loader = None
        self.source_adapter = None
        self.source_data_module = None
        self.source_distillation_weight = float(
            getattr(self.target_config, "lwf_source_distillation_weight", 0.0) or 0.0
        )
        if self.source_distillation_weight > 0:
            if self.source_config is None or not self.source_dataset:
                raise ValueError(
                    "lwf_source_distillation_weight > 0 requires --source-config and --source-dataset"
                )
            source_batch_size_raw = getattr(self.target_config, "lwf_source_batch_size", None)
            source_batch_size = (
                target_batch_size
                if source_batch_size_raw in (None, "")
                else int(source_batch_size_raw)
            )
            source_workers = int(getattr(self.source_config, "num_workers", target_workers))
            self.source_adapter, self.source_data_module = _make_data_module(
                self.source_config,
                self.source_dataset,
                batch_size=source_batch_size,
            )
            self.source_train_loader = _make_loader(
                self.source_data_module.patient_train,
                self.source_adapter.collate_patient_batch,
                source_batch_size,
                source_workers,
                shuffle=True,
                drop_last=False,
            )
            if len(self.source_train_loader) == 0:
                raise ValueError("LwF source distillation loader is empty")

        if len(self.target_train_loader) == 0:
            raise ValueError("LwF target train loader is empty")

        logger.info(
            "LwF data: target train=%d val=%d test=%d patients (%d train batches), source_distillation=%s",
            len(self.target_data_module.patient_train),
            len(self.target_data_module.patient_val),
            len(self.target_data_module.patient_test),
            len(self.target_train_loader),
            (
                f"{len(self.source_data_module.patient_train)} patients ({len(self.source_train_loader)} batches)"
                if self.source_train_loader is not None
                else "disabled"
            ),
        )

    def _setup_model(self) -> None:
        model_type = getattr(self.target_config, "model_type", "tb_rl_mil")
        self.model = create_ablation_model(model_type, self.target_config)
        self.teacher_model = create_ablation_model(model_type, self.target_config)

        source_state_dict = _state_dict_from_checkpoint(self.source_checkpoint_path, self.device)
        student_loaded = self._load_compatible_state(self.model, source_state_dict)
        teacher_loaded = self._load_compatible_state(self.teacher_model, source_state_dict)
        if student_loaded == 0 or teacher_loaded == 0:
            raise ValueError(f"No compatible model tensors found in {self.source_checkpoint_path}")

        self.model = self.model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

        logger.info(
            "Loaded source checkpoint for LwF: student=%d/%d tensors, teacher=%d/%d tensors from %s",
            student_loaded,
            len(source_state_dict),
            teacher_loaded,
            len(source_state_dict),
            self.source_checkpoint_path,
        )
        logger.info(
            "LwF trainable parameters: student=%s, teacher=%s frozen",
            f"{sum(param.numel() for param in self.model.parameters() if param.requires_grad):,}",
            f"{sum(param.numel() for param in self.teacher_model.parameters()):,}",
        )

    def _load_compatible_state(self, model: torch.nn.Module, source_state_dict: Dict[str, torch.Tensor]) -> int:
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in source_state_dict.items()
            if key in model_state_dict and model_state_dict[key].shape == value.shape
        }
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
        return len(filtered_state_dict)

    def _setup_optimizers(self) -> None:
        model_params = [param for param in self.model.parameters() if param.requires_grad]
        if not model_params:
            raise ValueError("LwF model has no trainable parameters")

        model_lr = float(
            getattr(
                self.target_config,
                "lwf_learning_rate",
                getattr(self.target_config, "patient_pipeline_lr", 1e-4),
            )
        )
        weight_decay = float(
            getattr(
                self.target_config,
                "lwf_weight_decay",
                getattr(self.target_config, "weight_decay", 1e-5),
            )
        )
        self.optimizer = torch.optim.AdamW(model_params, lr=model_lr, weight_decay=weight_decay)

        total_steps = max(1, len(self.target_train_loader) * int(getattr(self.target_config, "num_epochs", 1)))
        eta_min = float(getattr(self.target_config, "lwf_eta_min", 1e-6))
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

    def _reset_model_state(self, model: Optional[torch.nn.Module] = None, clear_history: bool = False) -> None:
        active_model = model if model is not None else self.model
        selector = getattr(active_model, "frame_selector", None)
        if selector is None:
            return
        if hasattr(selector, "saved_actions"):
            selector.saved_actions = []
        if hasattr(selector, "reset_rewards"):
            selector.reset_rewards()
        if clear_history and hasattr(selector, "clear_history"):
            selector.clear_history()

    def _reset_frame_selector_state(self, clear_history: bool = False) -> None:
        self._reset_model_state(self.model, clear_history=clear_history)

    def _teacher_forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.teacher_model.eval()
        self._reset_model_state(self.teacher_model, clear_history=True)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            teacher_outputs = self.teacher_model(inputs)
        self._reset_model_state(self.teacher_model, clear_history=True)
        return teacher_outputs

    def _get_tb_logits(self, outputs: Dict[str, Any]) -> torch.Tensor:
        task_logits = outputs.get("task_logits", {})
        if "TB Label" in task_logits:
            logits = task_logits["TB Label"]
        elif "tb_logits" in outputs:
            logits = outputs["tb_logits"]
        else:
            raise KeyError("Model output did not contain TB logits")
        return logits.reshape(-1)

    def _distillation_weight(self, epoch: int, base_weight_name: str, default: float) -> float:
        base_weight = float(getattr(self.target_config, base_weight_name, default))
        warmup_epochs = int(getattr(self.target_config, "lwf_warmup_epochs", 0))
        if warmup_epochs <= 0:
            return base_weight
        return base_weight * min(1.0, float(epoch + 1) / float(warmup_epochs))

    def _binary_logit_distillation(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"LwF distillation shape mismatch: student={tuple(student_logits.shape)}, "
                f"teacher={tuple(teacher_logits.shape)}"
            )
        temperature = max(float(getattr(self.target_config, "lwf_temperature", 2.0)), 1e-6)
        student_logits = student_logits.float().reshape(-1)
        teacher_logits = teacher_logits.detach().float().reshape(-1)
        if student_logits.numel() == 0:
            return student_logits.sum() * 0.0
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        return (
            F.binary_cross_entropy_with_logits(
                student_logits / temperature,
                teacher_probs,
                reduction="mean",
            )
            * temperature
            * temperature
        )

    def _task_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, int]:
        student_task_logits = student_outputs.get("task_logits", {})
        teacher_task_logits = teacher_outputs.get("task_logits", {})
        task_names = _as_name_list(getattr(self.target_config, "lwf_distill_tasks", ()))
        if not task_names:
            task_names = tuple(getattr(self.target_config, "active_tasks", ["TB Label"]))

        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        count = 0
        for task_name in task_names:
            if task_name not in student_task_logits or task_name not in teacher_task_logits:
                continue
            student_logits = student_task_logits[task_name]
            teacher_logits = teacher_task_logits[task_name]
            if student_logits.shape != teacher_logits.shape:
                logger.warning(
                    "Skipping LwF task distillation for %s due shape mismatch: student=%s teacher=%s",
                    task_name,
                    tuple(student_logits.shape),
                    tuple(teacher_logits.shape),
                )
                continue
            loss = loss + self._binary_logit_distillation(student_logits, teacher_logits)
            count += 1

        return loss, count

    def _pathology_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        student_scores = student_outputs.get("pathology_scores")
        teacher_scores = teacher_outputs.get("pathology_scores")
        if student_scores is None or teacher_scores is None:
            return torch.zeros((), dtype=torch.float32, device=self.device), 0
        if student_scores.shape != teacher_scores.shape:
            logger.warning(
                "Skipping LwF pathology distillation due shape mismatch: student=%s teacher=%s",
                tuple(student_scores.shape),
                tuple(teacher_scores.shape),
            )
            return torch.zeros((), dtype=torch.float32, device=self.device), 0

        if student_scores.dim() == 3 and "site_masks" in targets:
            valid_mask = targets["site_masks"].bool().unsqueeze(-1).expand_as(student_scores)
            if not valid_mask.any():
                return student_scores.sum() * 0.0, 0
            student_scores = student_scores[valid_mask]
            teacher_scores = teacher_scores[valid_mask]

        return self._binary_logit_distillation(student_scores, teacher_scores), 1

    def _feature_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, int]:
        feature_keys = _as_name_list(getattr(self.target_config, "lwf_feature_keys", ("patient_features",)))
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        count = 0
        for key in feature_keys:
            student_features = student_outputs.get(key)
            teacher_features = teacher_outputs.get(key)
            if student_features is None or teacher_features is None:
                continue
            if student_features.shape != teacher_features.shape:
                logger.warning(
                    "Skipping LwF feature distillation for %s due shape mismatch: student=%s teacher=%s",
                    key,
                    tuple(student_features.shape),
                    tuple(teacher_features.shape),
                )
                continue
            loss = loss + F.mse_loss(student_features.float(), teacher_features.detach().float())
            count += 1
        return loss, count

    def _distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        targets: Dict[str, torch.Tensor],
        epoch: int,
        prefix: str = "",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        task_weight = self._distillation_weight(epoch, "lwf_lambda", 1.0)
        pathology_weight = self._distillation_weight(epoch, "lwf_pathology_lambda", 0.25)
        feature_weight = self._distillation_weight(epoch, "lwf_feature_lambda", 0.0)

        task_loss, task_count = self._task_distillation_loss(student_outputs, teacher_outputs)
        pathology_loss, pathology_count = (
            self._pathology_distillation_loss(student_outputs, teacher_outputs, targets)
            if pathology_weight > 0
            else (torch.zeros((), dtype=torch.float32, device=self.device), 0)
        )
        feature_loss, feature_count = (
            self._feature_distillation_loss(student_outputs, teacher_outputs)
            if feature_weight > 0
            else (torch.zeros((), dtype=torch.float32, device=self.device), 0)
        )

        weighted_task_loss = task_weight * task_loss
        weighted_pathology_loss = pathology_weight * pathology_loss
        weighted_feature_loss = feature_weight * feature_loss
        total = weighted_task_loss + weighted_pathology_loss + weighted_feature_loss

        metrics = {
            f"{prefix}distillation_loss": float(total.detach().item()),
            f"{prefix}task_distillation_loss": float(task_loss.detach().item()),
            f"{prefix}pathology_distillation_loss": float(pathology_loss.detach().item()),
            f"{prefix}feature_distillation_loss": float(feature_loss.detach().item()),
            f"{prefix}weighted_task_distillation_loss": float(weighted_task_loss.detach().item()),
            f"{prefix}weighted_pathology_distillation_loss": float(weighted_pathology_loss.detach().item()),
            f"{prefix}weighted_feature_distillation_loss": float(weighted_feature_loss.detach().item()),
            f"{prefix}distilled_tasks": float(task_count),
            f"{prefix}distilled_pathology_heads": float(pathology_count),
            f"{prefix}distilled_feature_sets": float(feature_count),
        }
        return total, metrics

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.teacher_model.eval()
        self._reset_model_state(self.model, clear_history=True)
        self._reset_model_state(self.teacher_model, clear_history=True)

        source_iter = cycle(self.source_train_loader) if self.source_train_loader is not None else None
        pos_weights = _task_pos_weights(self.target_config)
        supervised_weight = float(getattr(self.target_config, "lwf_supervised_weight", 1.0))
        grad_clip_norm = float(getattr(self.target_config, "lwf_gradient_clip_norm", 1.0))
        accumulation_steps = max(1, int(getattr(self.target_config, "lwf_accumulation_steps", 1)))
        log_every_steps = max(1, int(getattr(self.target_config, "lwf_log_every_steps", 10)))

        running: Dict[str, float] = {
            "loss": 0.0,
            "supervised_loss": 0.0,
            "weighted_supervised_loss": 0.0,
        }
        steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.target_train_loader,
            desc=f"LwF Epoch {epoch + 1}/{int(getattr(self.target_config, 'num_epochs', 1))}",
        )
        for batch_idx, target_batch in enumerate(progress):
            inputs, targets = self._prepare_batch(target_batch)
            teacher_outputs = self._teacher_forward(inputs)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                student_outputs = self.model(inputs)
                supervised_loss, _ = self.model.compute_losses(student_outputs, targets, pos_weights)
                distillation_loss, distillation_metrics = self._distillation_loss(
                    student_outputs,
                    teacher_outputs,
                    targets,
                    epoch,
                )
                weighted_supervised_loss = supervised_weight * supervised_loss
                total_loss = weighted_supervised_loss + distillation_loss

                source_distillation_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                source_distillation_metrics: Dict[str, float] = {}
                if source_iter is not None:
                    source_inputs, source_targets = self._prepare_batch(next(source_iter))
                    source_teacher_outputs = self._teacher_forward(source_inputs)
                    source_student_outputs = self.model(source_inputs)
                    source_distillation_loss, source_distillation_metrics = self._distillation_loss(
                        source_student_outputs,
                        source_teacher_outputs,
                        source_targets,
                        epoch,
                        prefix="source_",
                    )
                    source_distillation_loss = self.source_distillation_weight * source_distillation_loss
                    total_loss = total_loss + source_distillation_loss

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
            self._reset_model_state(self.model)

            step_metrics = {
                "loss": float(total_loss.detach().item()),
                "supervised_loss": float(supervised_loss.detach().item()),
                "weighted_supervised_loss": float(weighted_supervised_loss.detach().item()),
                **distillation_metrics,
            }
            if source_distillation_metrics:
                step_metrics.update(source_distillation_metrics)
                step_metrics["weighted_source_distillation_loss"] = float(source_distillation_loss.detach().item())

            for key, value in step_metrics.items():
                running[key] = running.get(key, 0.0) + value
            steps += 1

            if (batch_idx + 1) % log_every_steps == 0:
                logger.info(
                    "LwF epoch=%d step=%d/%d loss=%.4f sup=%.4f kd=%.4f path_kd=%.4f lambda=%.4f temp=%.2f",
                    epoch + 1,
                    batch_idx + 1,
                    len(self.target_train_loader),
                    running["loss"] / steps,
                    running["supervised_loss"] / steps,
                    running.get("distillation_loss", 0.0) / steps,
                    running.get("weighted_pathology_distillation_loss", 0.0) / steps,
                    self._distillation_weight(epoch, "lwf_lambda", 1.0),
                    float(getattr(self.target_config, "lwf_temperature", 2.0)),
                )

            progress.set_postfix(
                {
                    "loss": running["loss"] / max(steps, 1),
                    "sup": running["supervised_loss"] / max(steps, 1),
                    "kd": running.get("distillation_loss", 0.0) / max(steps, 1),
                }
            )

        return running["loss"] / max(steps, 1), {key: value / max(steps, 1) for key, value in running.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, split_name: str = "val") -> Tuple[float, Dict[str, float]]:
        if loader is None or len(loader) == 0:
            logger.warning("No %s data available for LwF validation", split_name)
            return 0.0, {"loss": 0.0, "TB Label_auc": 0.0, "TB Label_auprc": 0.0}

        self.model.eval()
        self.teacher_model.eval()
        self._reset_model_state(self.model, clear_history=True)
        target_pos_weights = _task_pos_weights(self.target_config)

        losses = []
        labels = []
        probs = []
        preds = []

        for batch in tqdm(loader, desc=f"LwF {split_name.capitalize()} Evaluation"):
            inputs, targets = self._prepare_batch(batch)
            outputs = self.model(inputs)
            loss, _ = self.model.compute_losses(outputs, targets, target_pos_weights)
            losses.append(float(loss.item()))

            logits = self._get_tb_logits(outputs)
            batch_probs = torch.sigmoid(logits)
            labels.append(targets["tb_labels"].detach().cpu())
            probs.append(batch_probs.detach().cpu())
            preds.append((batch_probs > 0.5).float().detach().cpu())
            self._reset_model_state(self.model)

        metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
        if labels and probs and preds:
            label_np = torch.cat(labels).numpy().reshape(-1)
            prob_np = torch.cat(probs).numpy().reshape(-1)
            pred_np = torch.cat(preds).numpy().reshape(-1)
            valid_mask = label_np >= 0
            if valid_mask.any():
                metrics.update(_binary_metrics(label_np[valid_mask], prob_np[valid_mask], pred_np[valid_mask]))

        logger.info(
            "LwF %s metrics: loss=%.4f auc=%.4f auprc=%.4f",
            split_name,
            metrics.get("loss", 0.0),
            metrics.get("TB Label_auc", 0.0),
            metrics.get("TB Label_auprc", 0.0),
        )
        return metrics["loss"], metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool) -> Path:
        checkpoint = {
            "epoch": epoch,
            "algorithm": "lwf",
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": _config_to_dict(self.target_config),
            "source_config": _config_to_dict(self.source_config),
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "source_checkpoint_path": self.source_checkpoint_path,
            "lwf_summary": self.summary(),
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
        if self.source_config is not None:
            with open(self.checkpoint_dir / "source_config.yaml", "w") as handle:
                yaml.safe_dump(_config_to_dict(self.source_config), handle, sort_keys=False)

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
                "LwF epoch %d/%d complete: train_loss=%.4f val_loss=%.4f %s=%.4f best=%.4f",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                metric_name,
                metric,
                self.best_metric,
            )

            if self.epochs_without_improvement >= patience:
                logger.info("LwF early stopping after %d epochs without improvement", self.epochs_without_improvement)
                break

        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return self.best_metric, self.best_epoch

    def summary(self) -> Dict[str, Any]:
        return {
            "lwf_lambda": float(getattr(self.target_config, "lwf_lambda", 1.0)),
            "lwf_temperature": float(getattr(self.target_config, "lwf_temperature", 2.0)),
            "lwf_pathology_lambda": float(getattr(self.target_config, "lwf_pathology_lambda", 0.25)),
            "lwf_feature_lambda": float(getattr(self.target_config, "lwf_feature_lambda", 0.0)),
            "lwf_supervised_weight": float(getattr(self.target_config, "lwf_supervised_weight", 1.0)),
            "lwf_warmup_epochs": int(getattr(self.target_config, "lwf_warmup_epochs", 0)),
            "lwf_distill_tasks": list(_as_name_list(getattr(self.target_config, "lwf_distill_tasks", ()))),
            "lwf_feature_keys": list(_as_name_list(getattr(self.target_config, "lwf_feature_keys", ("patient_features",)))),
            "lwf_source_distillation_weight": self.source_distillation_weight,
            "source_distillation_enabled": self.source_train_loader is not None,
        }

    def _write_metrics(self) -> None:
        payload = {
            "algorithm": "lwf",
            "target_dataset": self.target_dataset,
            "source_dataset": self.source_dataset,
            "source_checkpoint_path": self.source_checkpoint_path,
            "lwf_summary": self.summary(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else None,
            "history": self.history,
        }
        with open(self.output_dir / "lwf_metrics.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
