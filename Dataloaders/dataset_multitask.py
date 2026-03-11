"""
Compatibility wrapper so existing training scripts can import
`LungUltrasoundDataModule` from `Dataloaders.dataset_multitask`
while the implementation lives in `dataset.py` at the repo root.
"""

from dataset import LungUltrasoundDataModule

__all__ = ["LungUltrasoundDataModule"]

