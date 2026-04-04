"""Public data-layer helpers for selecting dataset adapters."""

from .registry import get_dataset_adapter, load_dataset_adapter

__all__ = ["get_dataset_adapter", "load_dataset_adapter"]
