#!/usr/bin/env python3

"""Shared path resolution helpers for domain-shift analysis."""

import os
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = Path(
    os.environ.get(
        "ULTRAI_RUNS_ROOT",
        "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs",
    )
).expanduser()

DEFAULT_RESULTS_ROOT = RUNS_ROOT / "domain_shift_probe_results"
DEFAULT_FEATURE_ROOT = RUNS_ROOT / "domain_shift_probe_features"

LEGACY_RESULTS_ROOT = REPO_ROOT / "domain_shift_probe_results"
LEGACY_FEATURE_ROOT = REPO_ROOT / "checkpoints" / "domain_shift_probe_features"

LEGACY_SA_SPLIT_CANDIDATES = [
    REPO_ROOT / "sa_finetuning_results_retry" / "splits" / "split_full.csv",
    REPO_ROOT / "sa_finetuning_results" / "splits" / "split_full.csv",
]


def resolve_project_path(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    return None


def associated_results_dir(feature_dir: Path) -> Optional[Path]:
    for feature_root, results_root in [
        (DEFAULT_FEATURE_ROOT, DEFAULT_RESULTS_ROOT),
        (LEGACY_FEATURE_ROOT, LEGACY_RESULTS_ROOT),
    ]:
        try:
            relative_dir = feature_dir.relative_to(feature_root)
        except ValueError:
            continue
        return results_root / relative_dir
    return None


def default_feature_dir_for_output_dir(output_dir: Path) -> Path:
    for results_root, feature_root in [
        (DEFAULT_RESULTS_ROOT, DEFAULT_FEATURE_ROOT),
        (LEGACY_RESULTS_ROOT, LEGACY_FEATURE_ROOT),
    ]:
        try:
            relative_dir = output_dir.relative_to(results_root)
        except ValueError:
            continue
        return feature_root / relative_dir
    return DEFAULT_FEATURE_ROOT / output_dir.name


def infer_sa_split_csv(
    checkpoint_path: Path,
    sa_config_path: Optional[Path] = None,
    explicit_path: Optional[str] = None,
) -> Optional[Path]:
    explicit_candidate = resolve_project_path(explicit_path)
    if explicit_candidate is not None:
        if not explicit_candidate.exists():
            raise FileNotFoundError(f"SA split CSV does not exist: {explicit_candidate}")
        return explicit_candidate

    checkpoint_candidates: List[Path] = []
    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        checkpoint_candidates.append(parent / "results" / "splits" / "split_full.csv")

    config_candidates: List[Path] = []
    if sa_config_path is not None:
        for parent in [sa_config_path.parent, *sa_config_path.parents]:
            config_candidates.append(parent / "results" / "splits" / "split_full.csv")

    return first_existing_path(
        checkpoint_candidates + config_candidates + LEGACY_SA_SPLIT_CANDIDATES
    )
