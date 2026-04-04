"""
Dataset adapter registry for the public training entrypoints.

Each adapter module exposes the same interface:
- `PatientLevelDataset`
- `collate_patient_batch`
- `LungUltrasoundDataModule`

This lets `ultrai.training.train`, `ultrai.training.finetune`, and analysis
tools switch datasets by loading a different adapter module, while keeping the
core trainer logic dataset-agnostic.
"""

import importlib
from types import ModuleType


DATASET_ADAPTERS = {
    "benin": "ultrai.data.adapters.benin",
    "sa": "ultrai.data.adapters.sa",
}


def load_dataset_adapter(dataset_name: str) -> ModuleType:
    if dataset_name not in DATASET_ADAPTERS:
        raise ValueError(f"Unsupported dataset adapter: {dataset_name}")
    return importlib.import_module(DATASET_ADAPTERS[dataset_name])


def get_dataset_adapter(dataset_name: str) -> ModuleType:
    return load_dataset_adapter(dataset_name)
