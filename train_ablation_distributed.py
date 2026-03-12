#!/usr/bin/env python3
"""Compatibility layer for the legacy distributed ablation entrypoint."""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import yaml

from config import MultiTaskConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_TRAINER_MODULE = None
_DATASET_MODULES: Dict[str, Any] = {}


def _load_module(module_name: str, filename: str):
    module_path = _REPO_ROOT / filename
    if not module_path.exists():
        raise FileNotFoundError(f"Required module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_trainer_module():
    global _TRAINER_MODULE
    if _TRAINER_MODULE is None:
        _TRAINER_MODULE = _load_module(
            "_ultr_ai_train_clip_drl_mil_final_2",
            "train_clip_drl_mil_Final-2.py",
        )
        _patch_legacy_trainer(_TRAINER_MODULE)
    return _TRAINER_MODULE


def _load_dataset_module(kind: str):
    if kind not in _DATASET_MODULES:
        filename = "dataset_sa.py" if kind == "sa" else "dataset.py"
        module_name = f"_ultr_ai_{kind}_dataset"
        _DATASET_MODULES[kind] = _load_module(module_name, filename)
    return _DATASET_MODULES[kind]


def _prepare_frame_selector_for_eval(model):
    frame_selector = getattr(model, "frame_selector", None)
    if frame_selector is None:
        return None, {}

    if hasattr(frame_selector, "clear_history"):
        frame_selector.clear_history()
    else:
        if hasattr(frame_selector, "frame_history"):
            frame_selector.frame_history = {}
        if hasattr(frame_selector, "saved_actions"):
            frame_selector.saved_actions = []

    restore_state: Dict[str, Any] = {}
    if hasattr(frame_selector, "temperature"):
        restore_state["temperature"] = frame_selector.temperature
        frame_selector.temperature = 0.0

    return frame_selector, restore_state


def _restore_frame_selector_after_eval(frame_selector, restore_state: Dict[str, Any]):
    if frame_selector is None:
        return

    if "temperature" in restore_state:
        frame_selector.temperature = restore_state["temperature"]


def _patch_legacy_trainer(trainer_module):
    trainer_cls = trainer_module.TBTrainer
    if getattr(trainer_cls, "_compat_runtime_patch", False):
        return

    original_validate = trainer_cls.validate
    original_detailed_eval = trainer_cls._run_detailed_tb_evaluation

    def validate(self, epoch, loader=None, split_name="val"):
        if loader is None:
            loader = self.val_loader

        if len(loader) == 0:
            if split_name == "val" and getattr(self.config, "use_train_metric_when_no_val", False):
                trainer_module.logger.warning(
                    "Validation loader is empty; using training loader metrics for model selection"
                )
                loader = self.train_loader
                split_name = "train_fallback"
            else:
                trainer_module.logger.warning(
                    "%s loader is empty; skipping evaluation",
                    split_name.capitalize(),
                )
                return float("nan"), {}

        frame_selector, restore_state = _prepare_frame_selector_for_eval(self.model)
        try:
            return original_validate(self, epoch, loader=loader, split_name=split_name)
        finally:
            _restore_frame_selector_after_eval(frame_selector, restore_state)

    def _run_detailed_tb_evaluation(self, loader, split_name):
        if len(loader) == 0:
            trainer_module.logger.warning(
                "%s loader is empty; skipping detailed evaluation",
                split_name.capitalize(),
            )
            empty_df = trainer_module.pd.DataFrame(columns=["patient_id"])
            return empty_df, {"loss": float("nan")}, {}, {}

        frame_selector, restore_state = _prepare_frame_selector_for_eval(self.model)
        try:
            return original_detailed_eval(self, loader, split_name)
        finally:
            _restore_frame_selector_after_eval(frame_selector, restore_state)

    trainer_cls.validate = validate
    trainer_cls._run_detailed_tb_evaluation = _run_detailed_tb_evaluation
    trainer_cls._compat_runtime_patch = True


def _should_use_sa_dataset(config: "Config") -> bool:
    labels_csv = str(getattr(config, "labels_csv", "")).lower()
    file_metadata_csv = str(getattr(config, "file_metadata_csv", "")).lower()
    root_dir = str(getattr(config, "root_dir", "")).lower()

    return any(
        marker in value
        for marker in ("rsa_pathology_labels.csv", "rsa_videos", "verosdata")
        for value in (labels_csv, file_metadata_csv, root_dir)
    )


def _configure_dataset_binding(config: "Config"):
    trainer_module = _load_trainer_module()
    dataset_kind = "sa" if _should_use_sa_dataset(config) else "benin"
    dataset_module = _load_dataset_module(dataset_kind)
    _patch_empty_dataset_dataloader(dataset_module)
    trainer_module.LungUltrasoundDataModule = dataset_module.LungUltrasoundDataModule
    logger.info("Using %s data module for this run", dataset_kind.upper())
    return trainer_module


def _patch_empty_dataset_dataloader(dataset_module):
    data_module_cls = dataset_module.LungUltrasoundDataModule
    if getattr(data_module_cls, "_compat_empty_dataset_patch", False):
        return

    original_method = data_module_cls.patient_level_dataloader

    def patient_level_dataloader(self, split="train"):
        try:
            return original_method(self, split)
        except ValueError as exc:
            if "num_samples should be a positive integer value" not in str(exc):
                raise

            if split == "train":
                dataset = self.patient_train
            elif split == "val":
                dataset = self.patient_val
            elif split == "test":
                dataset = self.patient_test
            else:
                raise

            logger.warning(
                "Dataset split '%s' is empty; falling back to a non-shuffled loader for evaluation compatibility",
                split,
            )
            return dataset_module.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=dataset_module.collate_patient_batch,
                prefetch_factor=2,
            )

    data_module_cls.patient_level_dataloader = patient_level_dataloader
    data_module_cls._compat_empty_dataset_patch = True


def _derive_slurm_master_addr() -> Optional[str]:
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return None

    try:
        import subprocess

        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            check=True,
            capture_output=True,
            text=True,
        )
        first_host = result.stdout.splitlines()[0].strip()
        return first_host or None
    except Exception:
        return None


def setup_distributed():
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", _derive_slurm_master_addr() or "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "Initialized distributed process group: rank=%s world_size=%s local_rank=%s",
            rank,
            world_size,
            local_rank,
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class Config:
    """Config shim that keeps typed defaults and preserves extra YAML fields."""

    def __init__(self, **overrides: Any):
        self._known_fields = set(MultiTaskConfig.__dataclass_fields__.keys())
        self._extra_fields = set()

        base = MultiTaskConfig()
        for key, value in base.to_dict().items():
            setattr(self, key, value)

        if overrides:
            self._apply(overrides)

    def _apply(self, values: Dict[str, Any]):
        typed_updates = {k: v for k, v in values.items() if k in self._known_fields}
        if typed_updates:
            base_values = {
                field_name: getattr(self, field_name)
                for field_name in self._known_fields
                if hasattr(self, field_name)
            }
            base_values.update(typed_updates)
            typed_config = MultiTaskConfig.from_dict(base_values)
            for key, value in typed_config.to_dict().items():
                setattr(self, key, value)

        for key, value in values.items():
            if key not in self._known_fields:
                setattr(self, key, value)
                self._extra_fields.add(key)

        if hasattr(self, "pathology_classes") and not hasattr(self, "pathology_names"):
            self.pathology_names = list(self.pathology_classes)
            self._extra_fields.add("pathology_names")

    def load_from_yaml(self, filepath: str):
        with open(filepath, "r") as handle:
            config_dict = yaml.safe_load(handle) or {}

        self._apply(config_dict)
        self._yaml_path = filepath
        return self

    @classmethod
    def load(cls, filepath: str):
        return cls().load_from_yaml(filepath)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]):
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.device):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def save(self, filepath: str):
        with open(filepath, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, default_flow_style=False, sort_keys=False)


class AblationTrainer:
    """Thin wrapper around the existing TB trainer with runtime dataset binding."""

    def __init__(self, config: Config, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        if str(getattr(config, "device", "cpu")).startswith("cuda"):
            if torch.cuda.is_available():
                config.device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
            else:
                config.device = "cpu"

        trainer_module = _configure_dataset_binding(config)
        self._inner = trainer_module.TBTrainer(config)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
