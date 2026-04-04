#!/usr/bin/env python3
"""
SimCLR pretraining for CLIP vision features on ultrasound videos.

This stage learns a domain-adapted CLIP vision encoder on a chosen ultrasound
dataset, then exports a vision-only checkpoint that downstream training jobs
can use as a warm start.
"""

import argparse
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from NetworkArchitecture.clip_backbone_utils import (
    configure_clip_trainability,
    count_trainable_parameters,
    create_clip_vision_encoder,
)


logger = logging.getLogger("simclr")


@dataclass
class SimCLRConfig:
    experiment_name: str = "simclr"
    output_dir: str = "./runs/domain_shift/simclr/default"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir: str = "/users/lxflk/ULTR-AI-Vid/Data"
    file_metadata_csv: str = "/users/lxflk/ULTR-AI-Vid/Data/processed_files_2.csv"
    video_folder: str = "/capstor/scratch/cscs/tbrokowski/ultr-ai/BeninVideos"
    split_csv: Optional[str] = None
    pretrain_splits: List[str] = field(default_factory=lambda: ["train"])
    max_videos: Optional[int] = None

    frame_sampling: int = 32
    batch_size: int = 8
    num_workers: int = 6
    cache_size: int = 0

    target_height: int = 224
    target_width: int = 224
    mean: List[float] = field(default_factory=lambda: [0.45, 0.45, 0.45])
    std: List[float] = field(default_factory=lambda: [0.225, 0.225, 0.225])

    crop_scale_min: float = 0.7
    crop_scale_max: float = 1.0
    degrees: float = 20.0
    translate: List[float] = field(default_factory=lambda: [0.1, 0.1])
    brightness: float = 0.25
    contrast: float = 0.25
    blur_prob: float = 0.3
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.0
    noise_std: float = 0.12

    clip_model_name: str = "openai/clip-vit-base-patch32"
    local_weights_dir: str = "/users/lxflk/CLIP_weights/"
    freeze_backbone: bool = False
    clip_unfreeze_last_n_layers: int = 4
    train_visual_projection: bool = True

    projection_dim: int = 128
    projection_hidden_dim: int = 1024
    temperature: float = 0.1

    num_epochs: int = 30
    warmup_epochs: int = 3
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-6
    accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    log_every_steps: int = 20
    save_every_epochs: int = 1

    def __post_init__(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, switching SimCLR to CPU")
            self.device = "cpu"

        if not self.pretrain_splits:
            self.pretrain_splits = ["train"]

        self.pretrain_splits = [split.lower() for split in self.pretrain_splits]

    @classmethod
    def load(cls, path: str) -> "SimCLRConfig":
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in raw.items() if key in valid_keys}
        return cls(**filtered)

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "simclr_train.log"),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _normalize_split_name(split_name: str) -> str:
    split_name = split_name.strip().lower()
    if split_name == "val":
        return "valid"
    return split_name


def read_split_patient_ids(split_csv: str, selected_splits: Sequence[str]) -> Set[str]:
    split_df = pd.read_csv(split_csv, dtype=str).fillna("")
    normalized_splits = {_normalize_split_name(split) for split in selected_splits}
    patient_ids: Set[str] = set()

    if {"train_ids", "valid_ids", "test_ids"}.intersection(split_df.columns):
        column_map = {
            "train": "train_ids",
            "valid": "valid_ids",
            "test": "test_ids",
        }
        for split_name in normalized_splits:
            column_name = column_map.get(split_name)
            if column_name and column_name in split_df.columns:
                series = split_df[column_name].astype(str).str.strip()
                patient_ids.update(pid for pid in series if pid)
    elif {"patient_id", "split"}.issubset(split_df.columns):
        split_df["split"] = split_df["split"].astype(str).str.lower().map(_normalize_split_name)
        subset_df = split_df[split_df["split"].isin(normalized_splits)]
        patient_ids.update(pid for pid in subset_df["patient_id"].astype(str).str.strip() if pid)
    else:
        raise ValueError(f"Unsupported split CSV format: {split_csv}")

    return patient_ids


class SimCLRVideoViewTransform:
    """Two-view augmentation tuned for temporally coherent ultrasound clips."""

    def __init__(
        self,
        resize_size: Tuple[int, int],
        crop_scale: Tuple[float, float],
        degrees: float,
        translate: Tuple[float, float],
        brightness: float,
        contrast: float,
        blur_prob: float,
        blur_sigma: Tuple[float, float],
        noise_std: float,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.resize_size = resize_size
        self.crop_scale = crop_scale
        self.degrees = degrees
        self.translate = translate
        self.brightness = brightness
        self.contrast = contrast
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        if not frames:
            return torch.empty(0, 3, *self.resize_size)

        crop_params = transforms.RandomResizedCrop.get_params(
            frames[0],
            scale=self.crop_scale,
            ratio=(0.9, 1.1),
        )
        apply_affine = random.random() < 0.7
        angle = random.uniform(-self.degrees, self.degrees) if apply_affine else 0.0
        tx = random.uniform(-self.translate[0], self.translate[0])
        ty = random.uniform(-self.translate[1], self.translate[1])
        brightness_factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
        contrast_factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
        apply_blur = random.random() < self.blur_prob
        blur_radius = random.uniform(*self.blur_sigma) if apply_blur else 0.0

        augmented_frames: List[torch.Tensor] = []
        for frame in frames:
            frame = TF.resized_crop(frame, *crop_params, self.resize_size)
            frame = ImageEnhance.Contrast(frame).enhance(1.2)
            if apply_affine:
                width, height = frame.size
                frame = TF.affine(
                    frame,
                    angle=angle,
                    translate=(int(tx * width), int(ty * height)),
                    scale=1.0,
                    shear=0.0,
                    fill=0,
                )
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            if apply_blur:
                frame = frame.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            augmented_frames.append(TF.to_tensor(frame))

        video_tensor = torch.stack(augmented_frames)
        speckle = torch.randn_like(video_tensor) * self.noise_std
        video_tensor = torch.clamp(video_tensor * (1 + speckle), 0.0, 1.0)
        return (video_tensor - self.mean) / self.std


class SimCLRVideoDataset(Dataset):
    def __init__(self, config: SimCLRConfig) -> None:
        self.config = config
        self.metadata_df = pd.read_csv(config.file_metadata_csv)
        self.video_root = self._resolve_video_root(config.root_dir, config.video_folder)
        self.view_transform = SimCLRVideoViewTransform(
            resize_size=(config.target_height, config.target_width),
            crop_scale=(config.crop_scale_min, config.crop_scale_max),
            degrees=config.degrees,
            translate=(config.translate[0], config.translate[1]),
            brightness=config.brightness,
            contrast=config.contrast,
            blur_prob=config.blur_prob,
            blur_sigma=(config.blur_sigma_min, config.blur_sigma_max),
            noise_std=config.noise_std,
            mean=config.mean,
            std=config.std,
        )

        self.records = self._build_records()
        if config.cache_size > 0:
            self.video_cache = lru_cache(maxsize=config.cache_size)(self._load_video_uncached)
            logger.info("Enabled SimCLR in-memory video cache with maxsize=%d", config.cache_size)
        else:
            self.video_cache = self._load_video_uncached
            logger.info("Disabled SimCLR in-memory video cache")

    def _resolve_video_root(self, root_dir: str, video_folder: str) -> Path:
        video_path = Path(video_folder)
        if video_path.is_absolute():
            return video_path
        return Path(root_dir) / video_folder

    def _build_records(self) -> List[Dict[str, object]]:
        metadata_df = self.metadata_df.copy()
        patient_col = "Patient ID"
        if patient_col not in metadata_df.columns:
            raise ValueError(f"'{patient_col}' column missing from {self.config.file_metadata_csv}")

        if self.config.split_csv:
            selected_patient_ids = read_split_patient_ids(
                self.config.split_csv,
                self.config.pretrain_splits,
            )
            metadata_df[patient_col] = metadata_df[patient_col].astype(str).str.strip()
            metadata_df = metadata_df[metadata_df[patient_col].isin(selected_patient_ids)]

        type_col = "type" if "type" in metadata_df.columns else "Type" if "Type" in metadata_df.columns else None
        if type_col is not None:
            metadata_df[type_col] = metadata_df[type_col].astype(str).str.lower()
            metadata_df = metadata_df[metadata_df[type_col] == "video"]

        records: List[Dict[str, object]] = []
        for _, row in metadata_df.iterrows():
            patient_id = str(row["Patient ID"]).strip()
            site = str(row["Site"]).strip()
            depth = str(row["Depth"]).strip()
            count = str(row["Count"]).strip()
            file_name = None
            if "New File Name" in row and pd.notna(row["New File Name"]):
                file_name = str(row["New File Name"]).strip()
            if not file_name:
                file_name = f"{patient_id}_{site}_{depth}_{count}.mp4"

            video_path = self.video_root / file_name
            if not video_path.exists():
                continue

            records.append(
                {
                    "patient_id": patient_id,
                    "site": site,
                    "depth": depth,
                    "count": count,
                    "video_path": str(video_path),
                    "file_name": file_name,
                }
            )

        if self.config.max_videos:
            records = records[: int(self.config.max_videos)]

        if not records:
            raise ValueError("No videos found for SimCLR pretraining")

        logger.info(
            "Prepared SimCLR dataset with %d videos from %d patients",
            len(records),
            len({record["patient_id"] for record in records}),
        )
        return records

    def export_index(self, output_path: Path) -> None:
        pd.DataFrame(self.records).to_csv(output_path, index=False)

    def _sample_frame_indices(self, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            return np.zeros(self.config.frame_sampling, dtype=int)
        return np.linspace(0, frame_count - 1, self.config.frame_sampling, dtype=int)

    def _load_video_uncached(self, video_path: str) -> Tuple[Image.Image, ...]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        frames: List[Image.Image] = []
        for frame_idx in self._sample_frame_indices(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        cap.release()
        if not frames:
            raise RuntimeError(f"Failed to sample frames from {video_path}")

        return tuple(frame.copy() for frame in frames)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        for _ in range(3):
            record = self.records[index]
            try:
                frames = [frame.copy() for frame in self.video_cache(str(record["video_path"]))]
                return {
                    "view1": self.view_transform(frames),
                    "view2": self.view_transform(frames),
                    "patient_id": record["patient_id"],
                    "site": record["site"],
                    "video_path": record["video_path"],
                }
            except Exception as exc:
                logger.warning("Retrying after SimCLR video load failure on %s: %s", record["video_path"], exc)
                index = (index + 1) % len(self.records)

        raise RuntimeError(f"Repeated video load failures near dataset index {index}")


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class SimCLRVideoEncoder(nn.Module):
    def __init__(self, config: SimCLRConfig) -> None:
        super().__init__()
        self.vision_encoder = create_clip_vision_encoder(
            clip_model_name=config.clip_model_name,
            local_weights_dir=config.local_weights_dir,
            dtype=torch.float32,
        )
        configure_clip_trainability(
            self.vision_encoder,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_n_layers=config.clip_unfreeze_last_n_layers,
            train_visual_projection=config.train_visual_projection,
        )
        self.projector = SimCLRProjectionHead(
            input_dim=768,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_dim,
        )

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = videos.shape
        flat_videos = videos.view(-1, channels, height, width)
        outputs = self.vision_encoder(flat_videos)
        frame_features = outputs.pooler_output.view(batch_size, num_frames, -1)
        return frame_features.mean(dim=1)

    def forward(self, videos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode_video(videos)
        projections = F.normalize(self.projector(embeddings), dim=1)
        return embeddings, projections


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    batch_size = z1.shape[0]
    # Keep the contrastive logits in fp32 even when the encoder runs under AMP.
    representations = torch.cat([z1, z2], dim=0).float()
    similarity = torch.matmul(representations, representations.T) / temperature

    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    mask_value = torch.finfo(similarity.dtype).min
    similarity = similarity.masked_fill(mask, mask_value)

    positive_indices = torch.arange(batch_size, device=similarity.device)
    targets = torch.cat([positive_indices + batch_size, positive_indices], dim=0)

    loss = F.cross_entropy(similarity, targets)
    top1 = (similarity.argmax(dim=1) == targets).float().mean().item()
    return loss, top1


def build_scheduler(
    optimizer: AdamW,
    steps_per_epoch: int,
    config: SimCLRConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * config.num_epochs)
    warmup_steps = max(0, steps_per_epoch * config.warmup_epochs)
    min_lr_ratio = config.min_learning_rate / config.learning_rate

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = 0.0
        if total_steps > warmup_steps:
            progress = float(current_step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class SimCLRTrainer:
    def __init__(self, config: SimCLRConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = SimCLRVideoDataset(config)
        if config.batch_size < 2:
            raise ValueError("SimCLR requires batch_size >= 2")
        if len(self.dataset) < config.batch_size:
            raise ValueError(
                f"SimCLR dataset has {len(self.dataset)} videos, smaller than batch_size={config.batch_size}"
            )
        self.dataset.export_index(self.output_dir / "video_index.csv")

        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.num_workers > 0,
        )

        self.model = SimCLRVideoEncoder(config).to(self.device)
        logger.info(
            "SimCLR trainable parameters: %s",
            f"{count_trainable_parameters(self.model):,}",
        )

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, len(self.loader), config)
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            scaler_device,
            enabled=config.use_amp and self.device.type == "cuda",
        )

        self.best_loss = float("inf")
        self.best_epoch = -1
        self.history: List[Dict[str, float]] = []

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool) -> None:
        checkpoint = {
            "epoch": epoch,
            "config": asdict(self.config),
            "metrics": metrics,
            "model_state_dict": self.model.state_dict(),
            "vision_encoder_state_dict": self.model.vision_encoder.state_dict(),
            "projector_state_dict": self.model.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
        }
        vision_checkpoint = {
            "epoch": epoch,
            "config": asdict(self.config),
            "metrics": metrics,
            "vision_encoder_state_dict": self.model.vision_encoder.state_dict(),
        }

        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")
        torch.save(vision_checkpoint, self.output_dir / "vision_encoder_latest.pt")

        if (epoch + 1) % self.config.save_every_epochs == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")
            torch.save(vision_checkpoint, self.output_dir / f"vision_encoder_epoch_{epoch + 1:03d}.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            torch.save(vision_checkpoint, self.output_dir / "vision_encoder_best.pt")

    def train(self) -> Dict[str, object]:
        self.config.save(str(self.output_dir / "config.yaml"))

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            sample_batches = 0

            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(self.loader):
                view1 = batch["view1"].to(self.device, non_blocking=True)
                view2 = batch["view2"].to(self.device, non_blocking=True)

                autocast_enabled = self.config.use_amp and self.device.type == "cuda"
                with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    _, proj1 = self.model(view1)
                    _, proj2 = self.model(view2)

                loss, batch_top1 = nt_xent_loss(proj1, proj2, self.config.temperature)
                loss = loss / self.config.accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                epoch_loss += loss.item() * self.config.accumulation_steps
                epoch_acc += batch_top1
                sample_batches += 1

                if (step + 1) % self.config.log_every_steps == 0:
                    logger.info(
                        "Epoch %d/%d step %d/%d loss=%.4f contrastive_top1=%.4f lr=%.6f",
                        epoch + 1,
                        self.config.num_epochs,
                        step + 1,
                        len(self.loader),
                        epoch_loss / max(sample_batches, 1),
                        epoch_acc / max(sample_batches, 1),
                        self.optimizer.param_groups[0]["lr"],
                    )

            if sample_batches % self.config.accumulation_steps != 0:
                self.scaler.unscale_(self.optimizer)
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            if sample_batches == 0:
                raise RuntimeError("SimCLR dataloader produced zero batches")

            average_loss = epoch_loss / sample_batches
            average_acc = epoch_acc / sample_batches
            metrics = {
                "epoch": epoch + 1,
                "loss": average_loss,
                "contrastive_top1": average_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(metrics)

            is_best = average_loss < self.best_loss
            if is_best:
                self.best_loss = average_loss
                self.best_epoch = epoch + 1

            self._save_checkpoint(epoch, metrics, is_best=is_best)
            with open(self.output_dir / "metrics.json", "w") as handle:
                json.dump(
                    {
                        "best_loss": self.best_loss,
                        "best_epoch": self.best_epoch,
                        "num_videos": len(self.dataset),
                        "num_patients": len({record["patient_id"] for record in self.dataset.records}),
                        "history": self.history,
                    },
                    handle,
                    indent=2,
                )

            logger.info(
                "Finished epoch %d/%d loss=%.4f contrastive_top1=%.4f best_loss=%.4f",
                epoch + 1,
                self.config.num_epochs,
                average_loss,
                average_acc,
                self.best_loss,
            )

        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "output_dir": str(self.output_dir),
            "best_checkpoint": str(self.output_dir / "checkpoint_best.pt"),
            "best_vision_checkpoint": str(self.output_dir / "vision_encoder_best.pt"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR pretraining for ultrasound videos")
    parser.add_argument("--config", type=str, required=True, help="Path to SimCLR YAML config")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--root-dir", type=str, help="Override dataset root directory")
    parser.add_argument("--file-metadata-csv", type=str, help="Override file metadata CSV")
    parser.add_argument("--video-folder", type=str, help="Override video folder")
    parser.add_argument("--split-csv", type=str, help="Override split CSV")
    parser.add_argument("--log-dir", type=str, help="Directory for SimCLR training logs")
    parser.add_argument("--pretrain-splits", nargs="+", help="Splits to use for unlabeled pretraining")
    parser.add_argument("--max-videos", type=int, help="Optional cap for smoke tests")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker override")
    parser.add_argument("--cache-size", type=int, help="Decoded-video cache size; 0 disables caching")
    parser.add_argument("--num-epochs", type=int, help="Epoch count override")
    parser.add_argument("--device", type=str, help="Device override")
    parser.add_argument("--clip-unfreeze-last-n-layers", type=int, help="How many CLIP blocks to fine-tune")
    return parser.parse_args()


def main() -> Dict[str, object]:
    args = parse_args()
    config = SimCLRConfig.load(args.config)

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.root_dir is not None:
        config.root_dir = args.root_dir
    if args.file_metadata_csv is not None:
        config.file_metadata_csv = args.file_metadata_csv
    if args.video_folder is not None:
        config.video_folder = args.video_folder
    if args.split_csv is not None:
        config.split_csv = args.split_csv
    if args.pretrain_splits is not None:
        config.pretrain_splits = args.pretrain_splits
    if args.max_videos is not None:
        config.max_videos = args.max_videos
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.cache_size is not None:
        config.cache_size = args.cache_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.device is not None:
        config.device = args.device
    if args.clip_unfreeze_last_n_layers is not None:
        config.clip_unfreeze_last_n_layers = args.clip_unfreeze_last_n_layers

    output_dir = Path(config.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir is not None else output_dir
    setup_logging(log_dir)
    set_seed(config.seed)

    logger.info("SimCLR configuration:")
    for key, value in asdict(config).items():
        logger.info("  %s: %s", key, value)

    trainer = SimCLRTrainer(config)
    results = trainer.train()
    logger.info("SimCLR pretraining complete: %s", results)
    return results


if __name__ == "__main__":
    main()
