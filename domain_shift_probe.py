#!/usr/bin/env python3
"""
Probe Benin vs SA domain shift at three representation levels:
1. CLIP frame embeddings
2. Per-scan pooled representations
3. Post-MIL patient representations

The script loads a checkpoint plus separate Benin and SA dataset configs,
extracts the requested features, trains a simple grouped domain classifier
for each level, and writes repo-local artifacts without touching existing
training outputs.
"""

import argparse
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset

import dataset as benin_dataset
import dataset_sa as sa_dataset
from config import MultiTaskConfig, load_config
from NetworkArchitecture.CLIP_DRL_Aug11 import MultiTaskModel

logger = logging.getLogger("domain_shift_probe")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


LEVEL_SPECS = {
    "clip_frame": {
        "label": "CLIP frame embeddings",
        "feature_file": "clip_frame_features.npy",
        "metadata_file": "clip_frame_metadata.csv",
    },
    "scan": {
        "label": "Scan representations",
        "feature_file": "scan_features.npy",
        "metadata_file": "scan_metadata.csv",
    },
    "patient": {
        "label": "Patient representations",
        "feature_file": "patient_features.npy",
        "metadata_file": "patient_metadata.csv",
    },
}

DEFAULT_SA_SPLIT_CANDIDATES = [
    Path("sa_finetuning_results_retry/splits/split_full.csv"),
    Path("sa_finetuning_results/splits/split_full.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-shift probe for Benin vs SA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to probe")
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Config used to build the checkpointed model; defaults to --benin-config",
    )
    parser.add_argument("--benin-config", type=str, required=True, help="Benin dataset config")
    parser.add_argument("--sa-config", type=str, required=True, help="SA dataset config")
    parser.add_argument("--benin-split-csv", type=str, default=None, help="Optional Benin split CSV override")
    parser.add_argument(
        "--sa-split-csv",
        type=str,
        default=None,
        help="Optional SA split CSV override; if omitted the script resolves a repo-local default",
    )
    parser.add_argument(
        "--benin-split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Benin split to probe",
    )
    parser.add_argument(
        "--sa-split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="SA split to probe",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory inside the repo; a unique directory is created if it already exists",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for raw feature arrays and metadata; "
            "defaults to checkpoints/domain_shift_probe_features/<run-name>"
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker override")
    parser.add_argument("--frame-sampling", type=int, default=None, help="Frame sampling override")
    parser.add_argument("--max-sites", type=int, default=None, help="Max sites override")
    parser.add_argument(
        "--max-patients-per-domain",
        type=int,
        default=None,
        help="Optional patient cap per domain for smoke tests",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out patient fraction for the classifier")
    parser.add_argument("--num-repeats", type=int, default=5, help="Number of grouped classifier repeats")
    parser.add_argument(
        "--max-train-samples-per-domain",
        type=int,
        default=50000,
        help="Sample cap per domain during classifier training; <=0 disables the cap",
    )
    parser.add_argument(
        "--max-test-samples-per-domain",
        type=int,
        default=20000,
        help="Sample cap per domain during classifier evaluation; <=0 disables the cap",
    )
    parser.add_argument(
        "--max-plot-samples-per-domain",
        type=int,
        default=2000,
        help="Point cap per domain for PCA scatter plots; <=0 disables the cap",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.set_defaults(save_features=True, save_plots=True)
    parser.add_argument("--save-features", dest="save_features", action="store_true", help="Save raw feature arrays")
    parser.add_argument("--no-save-features", dest="save_features", action="store_false", help="Skip raw feature arrays")
    parser.add_argument("--save-plots", dest="save_plots", action="store_true", help="Save summary plots")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="Skip summary plots")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "domain_shift_probe.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    logging.captureWarnings(True)


def sanitize_cap(value: Optional[int]) -> Optional[int]:
    if value is None or value <= 0:
        return None
    return int(value)


def resolve_repo_relative(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def resolve_output_dir(requested_output_dir: Optional[str], checkpoint_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parent
    if requested_output_dir is None:
        base = repo_root / "domain_shift_probe_results" / checkpoint_path.stem
    else:
        base = resolve_repo_relative(requested_output_dir)

    if not base.exists():
        return base

    suffix = 1
    while True:
        candidate = base.parent / f"{base.name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def resolve_feature_dir(requested_feature_dir: Optional[str], output_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parent

    if requested_feature_dir is not None:
        return resolve_repo_relative(requested_feature_dir)

    results_root = repo_root / "domain_shift_probe_results"
    try:
        relative_run_dir = output_dir.relative_to(results_root)
    except ValueError:
        relative_run_dir = Path(output_dir.name)

    return repo_root / "checkpoints" / "domain_shift_probe_features" / relative_run_dir


def resolve_default_sa_split_csv(explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        resolved = resolve_repo_relative(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"SA split CSV does not exist: {resolved}")
        return resolved

    for candidate in DEFAULT_SA_SPLIT_CANDIDATES:
        resolved = resolve_repo_relative(str(candidate))
        if resolved.exists():
            return resolved

    return None


def load_runtime_config(
    config_path: str,
    device: Optional[str],
    batch_size: Optional[int],
    num_workers: Optional[int],
    frame_sampling: Optional[int],
    max_sites: Optional[int],
    split_csv_override: Optional[Path],
) -> MultiTaskConfig:
    config = load_config(config_file=config_path)
    config.train = False
    config.evaluate_best_valid_model = False

    if device is not None:
        config.device = device
    if batch_size is not None:
        config.batch_size = batch_size
    if num_workers is not None:
        config.num_workers = num_workers
    if frame_sampling is not None:
        config.frame_sampling = frame_sampling
    if max_sites is not None:
        config.max_sites = max_sites
    if split_csv_override is not None:
        config.split_csv = str(split_csv_override)

    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        config.device = "cpu"

    return config


def load_model(model_config: MultiTaskConfig, checkpoint_path: Path) -> MultiTaskModel:
    logger.info("Loading model from %s", checkpoint_path)
    model = MultiTaskModel(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=model_config.device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(model_config.device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


def create_dataloader(
    dataset_module,
    dataset_name: str,
    config: MultiTaskConfig,
    split: str,
    max_patients: Optional[int],
    seed: int,
) -> Tuple[DataLoader, int]:
    dataset_kwargs = {
        "root_dir": config.root_dir,
        "labels_csv": config.labels_csv,
        "file_metadata_csv": config.file_metadata_csv,
        "image_folder": config.image_folder,
        "video_folder": config.video_folder,
        "split_csv": config.split_csv,
        "video_transforms": dataset_module.SimpleVideoTransforms(),
        "mode": getattr(config, "mode", "video"),
        "frame_sampling": config.frame_sampling,
        "depth_filter": config.depth_filter,
        "cache_size": 100,
        "files_per_site": getattr(config, "files_per_site", "all"),
        "site_order": getattr(config, "site_order", None),
        "pad_missing_sites": getattr(config, "pad_missing_sites", True),
        "max_sites": getattr(config, "max_sites", None),
    }

    requested_splits = [split] if split != "all" else ["train", "val", "test"]
    datasets = []
    for split_name in requested_splits:
        patient_dataset = dataset_module.PatientLevelDataset(
            split=split_name,
            **dataset_kwargs,
        )
        logger.info("%s %s split dataset size: %d patients", dataset_name, split_name, len(patient_dataset))
        datasets.append(patient_dataset)

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    dataset_len = len(dataset)
    if max_patients is not None and max_patients < dataset_len:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(dataset_len, size=max_patients, replace=False))
        dataset = Subset(dataset, indices.tolist())
        dataset_len = len(dataset)

    if dataset_len == 0:
        raise RuntimeError(
            f"{dataset_name} split '{split}' produced 0 patients. "
            f"root_dir={config.root_dir} file_metadata_csv={config.file_metadata_csv} "
            f"labels_csv={config.labels_csv} split_csv={config.split_csv}"
        )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": str(config.device).startswith("cuda"),
        "collate_fn": dataset_module.collate_patient_batch,
    }
    if config.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    return DataLoader(**dataloader_kwargs), dataset_len


def prepare_inputs(batch: Dict, device: str) -> Dict[str, torch.Tensor]:
    inputs = {
        "site_videos": batch["site_videos"].to(device, non_blocking=True),
        "site_indices": batch["site_indices"].to(device, non_blocking=True),
        "site_masks": batch["site_masks"].to(device, non_blocking=True),
    }
    if "site_findings" in batch and batch["site_findings"] is not None:
        inputs["site_findings"] = batch["site_findings"].to(device, non_blocking=True)
    return inputs


def empty_level_store() -> Dict[str, Dict[str, List]]:
    return {
        level_name: {"feature_chunks": [], "metadata_rows": []}
        for level_name in LEVEL_SPECS
    }


def append_patient_features(
    store: Dict[str, Dict[str, List]],
    domain_name: str,
    patient_ids: Sequence[str],
    patient_features: torch.Tensor,
    site_counts: Optional[torch.Tensor],
) -> None:
    patient_array = patient_features.detach().cpu().numpy().astype(np.float32, copy=False)
    store["patient"]["feature_chunks"].append(patient_array)

    for idx, patient_id in enumerate(patient_ids):
        num_sites = int(site_counts[idx].item()) if site_counts is not None else None
        store["patient"]["metadata_rows"].append(
            {"domain": domain_name, "patient_id": str(patient_id), "num_sites": num_sites}
        )


def append_scan_features(
    store: Dict[str, Dict[str, List]],
    domain_name: str,
    patient_ids: Sequence[str],
    scan_probe_rows: List[Dict],
    mil_attention: torch.Tensor,
) -> None:
    for item in scan_probe_rows:
        batch_index = int(item["batch_index"])
        site_position = int(item["site_position"])
        feature_row = item["scan_features"].detach().cpu().numpy().astype(np.float32, copy=False)

        store["scan"]["feature_chunks"].append(feature_row[None, :])
        store["scan"]["metadata_rows"].append(
            {
                "domain": domain_name,
                "patient_id": str(patient_ids[batch_index]),
                "site_position": site_position,
                "site_index": int(item["site_index"]),
                "mil_attention": float(mil_attention[batch_index, site_position].detach().cpu().item()),
                "selected_indices": ";".join(str(int(x)) for x in item["selected_indices"].tolist()),
            }
        )


def append_frame_features(
    store: Dict[str, Dict[str, List]],
    domain_name: str,
    patient_ids: Sequence[str],
    frame_probe_rows: List[Dict],
) -> None:
    for item in frame_probe_rows:
        batch_index = int(item["batch_index"])
        clip_features = item["clip_features"].detach().cpu().numpy().astype(np.float32, copy=False)

        if clip_features.size == 0:
            continue

        store["clip_frame"]["feature_chunks"].append(clip_features)
        for frame_index in range(clip_features.shape[0]):
            store["clip_frame"]["metadata_rows"].append(
                {
                    "domain": domain_name,
                    "patient_id": str(patient_ids[batch_index]),
                    "site_position": int(item["site_position"]),
                    "site_index": int(item["site_index"]),
                    "frame_index": int(frame_index),
                }
            )


def finalize_feature_store(store: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, object]]:
    finalized = {}
    for level_name, level_store in store.items():
        if level_store["feature_chunks"]:
            features = np.concatenate(level_store["feature_chunks"], axis=0)
        else:
            features = np.zeros((0, 0), dtype=np.float32)
        metadata = pd.DataFrame(level_store["metadata_rows"])
        finalized[level_name] = {"features": features, "metadata": metadata}
    return finalized


def extract_domain_features(
    model: MultiTaskModel,
    dataloader: DataLoader,
    domain_name: str,
    device: str,
    use_amp: bool,
) -> Dict[str, Dict[str, object]]:
    logger.info("Extracting %s features", domain_name)
    store = empty_level_store()
    total_batches = len(dataloader)
    autocast_enabled = use_amp and str(device).startswith("cuda")

    for batch_idx, batch in enumerate(dataloader):
        patient_ids = batch.get("patient_ids", [])
        if len(patient_ids) == 0:
            continue

        model.frame_selector.clear_history()
        inputs = prepare_inputs(batch, device)

        with torch.inference_mode():
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_context:
                site_features, pathology_scores, _, probe_data = model.process_patient(
                    inputs["site_videos"],
                    inputs["site_indices"],
                    inputs["site_masks"],
                    collect_probe_data=True,
                )

                if model.use_pathology_loss:
                    integrated_features = model.site_integration(
                        site_features, inputs["site_indices"], pathology_scores
                    )
                else:
                    integrated_features = model.site_integration(site_features)

                patient_features, mil_attention = model.patient_mil(integrated_features, inputs["site_masks"])

        append_patient_features(store, domain_name, patient_ids, patient_features, batch.get("site_counts"))
        append_scan_features(store, domain_name, patient_ids, probe_data["scan_features"], mil_attention)
        append_frame_features(store, domain_name, patient_ids, probe_data["frame_features"])

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            logger.info(
                "%s progress: batch %d/%d | frame_rows=%d scan_rows=%d patient_rows=%d",
                domain_name,
                batch_idx + 1,
                total_batches,
                len(store["clip_frame"]["metadata_rows"]),
                len(store["scan"]["metadata_rows"]),
                len(store["patient"]["metadata_rows"]),
            )

        if str(device).startswith("cuda") and batch_idx % 8 == 0:
            torch.cuda.empty_cache()

    finalized = finalize_feature_store(store)
    for level_name, payload in finalized.items():
        logger.info("%s %s samples: %d", domain_name, level_name, int(payload["features"].shape[0]))
    return finalized


def combine_domain_payloads(
    benin_payload: Dict[str, Dict[str, object]],
    sa_payload: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    combined = {}
    for level_name in LEVEL_SPECS:
        feature_parts = []
        metadata_parts = []
        for payload in [benin_payload[level_name], sa_payload[level_name]]:
            if payload["features"].shape[0] > 0:
                feature_parts.append(payload["features"])
                metadata_parts.append(payload["metadata"])

        if feature_parts:
            features = np.concatenate(feature_parts, axis=0)
            metadata = pd.concat(metadata_parts, ignore_index=True)
        else:
            features = np.zeros((0, 0), dtype=np.float32)
            metadata = pd.DataFrame()

        combined[level_name] = {"features": features, "metadata": metadata}
    return combined


def split_patients_by_domain(metadata: pd.DataFrame, test_size: float, seed: int) -> Tuple[set, set]:
    train_patients = set()
    test_patients = set()

    for domain_name, domain_df in metadata.groupby("domain"):
        patient_keys = sorted(
            f"{domain_name}::{patient_id}" for patient_id in domain_df["patient_id"].astype(str).unique()
        )
        if len(patient_keys) < 2:
            raise ValueError(f"Need at least two patients in domain '{domain_name}' to create train/test splits")

        rng = np.random.default_rng(seed)
        shuffled = np.array(patient_keys, dtype=object)
        rng.shuffle(shuffled)

        n_test = int(np.ceil(len(shuffled) * test_size))
        n_test = min(max(n_test, 1), len(shuffled) - 1)

        test_subset = set(shuffled[:n_test].tolist())
        train_subset = set(shuffled[n_test:].tolist())
        train_patients.update(train_subset)
        test_patients.update(test_subset)

    return train_patients, test_patients


def cap_indices_per_domain(indices: np.ndarray, domains: np.ndarray, cap: Optional[int], seed: int) -> np.ndarray:
    if cap is None:
        return indices

    rng = np.random.default_rng(seed)
    kept = []
    for domain_name in np.unique(domains[indices]):
        domain_indices = indices[domains[indices] == domain_name]
        if len(domain_indices) > cap:
            domain_indices = np.sort(rng.choice(domain_indices, size=cap, replace=False))
        kept.append(domain_indices)

    if not kept:
        return np.array([], dtype=np.int64)
    return np.concatenate(kept)


def compute_classifier_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_domain_classifier(
    features: np.ndarray,
    metadata: pd.DataFrame,
    level_name: str,
    test_size: float,
    num_repeats: int,
    max_train_samples_per_domain: Optional[int],
    max_test_samples_per_domain: Optional[int],
    seed: int,
) -> Dict[str, object]:
    if len(metadata) != features.shape[0]:
        raise ValueError(
            f"Metadata/feature row mismatch for {level_name}: "
            f"{len(metadata)} metadata rows vs {features.shape[0]} feature rows"
        )
    if features.shape[0] == 0:
        raise ValueError(f"No features extracted for {level_name}")

    domains = metadata["domain"].astype(str).to_numpy()
    patient_keys = (metadata["domain"].astype(str) + "::" + metadata["patient_id"].astype(str)).to_numpy()
    labels = (domains == "sa").astype(np.int64)

    repeats = []
    for repeat_idx in range(num_repeats):
        train_patients, test_patients = split_patients_by_domain(metadata, test_size=test_size, seed=seed + repeat_idx)
        train_mask = np.isin(patient_keys, list(train_patients))
        test_mask = np.isin(patient_keys, list(test_patients))

        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)

        train_indices = cap_indices_per_domain(train_indices, domains, max_train_samples_per_domain, seed + repeat_idx)
        test_indices = cap_indices_per_domain(test_indices, domains, max_test_samples_per_domain, seed + 1000 + repeat_idx)

        classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed + repeat_idx,
                    ),
                ),
            ]
        )

        classifier.fit(features[train_indices], labels[train_indices])
        probs = classifier.predict_proba(features[test_indices])[:, 1]
        metrics = compute_classifier_metrics(labels[test_indices], probs)
        metrics.update(
            {
                "repeat_index": repeat_idx,
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "train_patients": int(len(train_patients)),
                "test_patients": int(len(test_patients)),
                "train_benin_rows": int(np.sum(domains[train_indices] == "benin")),
                "train_sa_rows": int(np.sum(domains[train_indices] == "sa")),
                "test_benin_rows": int(np.sum(domains[test_indices] == "benin")),
                "test_sa_rows": int(np.sum(domains[test_indices] == "sa")),
            }
        )
        repeats.append(metrics)

    metric_names = ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc"]
    summary = {
        "level_name": level_name,
        "level_label": LEVEL_SPECS[level_name]["label"],
        "num_rows": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_patients": int(metadata["patient_id"].astype(str).nunique()),
        "num_benin_patients": int(metadata.loc[metadata["domain"] == "benin", "patient_id"].astype(str).nunique()),
        "num_sa_patients": int(metadata.loc[metadata["domain"] == "sa", "patient_id"].astype(str).nunique()),
        "metrics_mean": {},
        "metrics_std": {},
        "repeats": repeats,
    }

    for metric_name in metric_names:
        values = np.array([repeat[metric_name] for repeat in repeats], dtype=np.float64)
        summary["metrics_mean"][metric_name] = float(values.mean())
        summary["metrics_std"][metric_name] = float(values.std(ddof=0))

    return summary


def save_feature_payloads(features_dir: Path, combined_payload: Dict[str, Dict[str, object]]) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    for level_name, payload in combined_payload.items():
        spec = LEVEL_SPECS[level_name]
        np.save(features_dir / spec["feature_file"], payload["features"])
        payload["metadata"].to_csv(features_dir / spec["metadata_file"], index=False)


def subsample_metadata_indices(metadata: pd.DataFrame, cap_per_domain: Optional[int], seed: int) -> np.ndarray:
    if metadata.empty:
        return np.array([], dtype=np.int64)

    all_indices = np.arange(len(metadata))
    if cap_per_domain is None:
        return all_indices

    rng = np.random.default_rng(seed)
    selected = []
    for domain_name, group in metadata.groupby("domain"):
        indices = group.index.to_numpy()
        if len(indices) > cap_per_domain:
            indices = np.sort(rng.choice(indices, size=cap_per_domain, replace=False))
        selected.append(indices)

    return np.concatenate(selected) if selected else np.array([], dtype=np.int64)


def save_pca_overview(
    output_dir: Path,
    combined_payload: Dict[str, Dict[str, object]],
    max_plot_samples_per_domain: Optional[int],
    seed: int,
) -> Optional[Path]:
    if plt is None:
        logger.info("matplotlib not available; skipping PCA overview")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"benin": "#2a6f97", "sa": "#f4a261"}

    for ax, level_name in zip(axes, ["clip_frame", "scan", "patient"]):
        payload = combined_payload[level_name]
        metadata = payload["metadata"]
        features = payload["features"]

        if len(metadata) == 0 or features.shape[0] < 2:
            ax.set_title(f"{LEVEL_SPECS[level_name]['label']}\n(no data)")
            ax.axis("off")
            continue

        selected_indices = subsample_metadata_indices(metadata, max_plot_samples_per_domain, seed)
        subset_metadata = metadata.iloc[selected_indices].reset_index(drop=True)
        subset_features = features[selected_indices]

        standardized = StandardScaler().fit_transform(subset_features)
        coords = PCA(n_components=2, random_state=seed).fit_transform(standardized)

        for domain_name in ["benin", "sa"]:
            mask = subset_metadata["domain"].to_numpy() == domain_name
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=10,
                    alpha=0.45,
                    c=colors[domain_name],
                    label=domain_name,
                )

        ax.set_title(LEVEL_SPECS[level_name]["label"])
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Benin vs SA PCA view of extracted features", y=1.02)
    fig.tight_layout()

    plot_path = output_dir / "embedding_pca_overview.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_summary_report(output_dir: Path, summaries: Dict[str, Dict[str, object]]) -> Path:
    frame_auroc = summaries["clip_frame"]["metrics_mean"]["auroc"]
    scan_auroc = summaries["scan"]["metrics_mean"]["auroc"]
    patient_auroc = summaries["patient"]["metrics_mean"]["auroc"]

    report_lines = [
        "Domain Shift Probe Summary",
        "",
        "Representation note",
        "  Scan representation here is the current code path's pooled selected-frame site vector before site integration.",
        "",
    ]

    for level_name in ["clip_frame", "scan", "patient"]:
        summary = summaries[level_name]
        report_lines.append(summary["level_label"])
        report_lines.append(
            f"  AUROC: {summary['metrics_mean']['auroc']:.4f} +/- {summary['metrics_std']['auroc']:.4f}"
        )
        report_lines.append(
            f"  Accuracy: {summary['metrics_mean']['accuracy']:.4f} +/- {summary['metrics_std']['accuracy']:.4f}"
        )
        report_lines.append(f"  Rows / Patients: {summary['num_rows']} / {summary['num_patients']}")
        report_lines.append("")

    report_lines.append("Interpretation")
    if frame_auroc >= 0.80:
        report_lines.append(
            "- High separability already at raw CLIP frame embeddings: appearance/encoder shift is strong and SimCLR-style SSL is plausible."
        )
    elif frame_auroc <= 0.65 and (scan_auroc >= 0.80 or patient_auroc >= 0.80):
        report_lines.append(
            "- Raw CLIP embeddings are not strongly separable, but later representations are: the shift is more likely being introduced by scan aggregation, site coverage, or protocol/missingness."
        )
    else:
        report_lines.append(
            "- Frame-level and later-level separability are mixed: both low-level appearance and higher-level aggregation effects may matter."
        )

    if patient_auroc > scan_auroc + 0.10 and patient_auroc >= 0.80:
        report_lines.append(
            "- Separability increases again at the patient level: the model may be exploiting domain-specific site presence/importance patterns, which SimCLR alone is unlikely to fix."
        )

    report_lines.append(
        "- This probe cannot rule out population-specific signal shifts; if separation appears mainly at later levels, inspect site distribution and missingness before assuming SSL will solve it."
    )

    report_path = output_dir / "summary.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    return report_path


def save_auroc_bar_plot(output_dir: Path, summaries: Dict[str, Dict[str, object]]) -> Optional[Path]:
    if plt is None:
        logger.info("matplotlib not available; skipping AUROC bar plot")
        return None

    levels = ["clip_frame", "scan", "patient"]
    labels = [summaries[level]["level_label"] for level in levels]
    means = [summaries[level]["metrics_mean"]["auroc"] for level in levels]
    stds = [summaries[level]["metrics_std"]["auroc"] for level in levels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=stds, color=["#2a6f97", "#95afc0", "#f4a261"], capsize=6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Domain-classifier AUROC")
    ax.set_title("Benin vs SA separability by representation level")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    plot_path = output_dir / "domain_probe_auroc.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_manifest(
    output_dir: Path,
    feature_dir: Optional[Path],
    args: argparse.Namespace,
    benin_config: MultiTaskConfig,
    sa_config: MultiTaskConfig,
    model_config: MultiTaskConfig,
) -> Path:
    manifest = {
        "checkpoint": str(resolve_repo_relative(args.checkpoint)),
        "model_config": str(resolve_repo_relative(args.model_config or args.benin_config)),
        "benin_config": str(resolve_repo_relative(args.benin_config)),
        "sa_config": str(resolve_repo_relative(args.sa_config)),
        "benin_split": args.benin_split,
        "sa_split": args.sa_split,
        "resolved_benin_split_csv": benin_config.split_csv,
        "resolved_sa_split_csv": sa_config.split_csv,
        "device": model_config.device,
        "batch_size": benin_config.batch_size,
        "num_workers": benin_config.num_workers,
        "frame_sampling": benin_config.frame_sampling,
        "max_sites": benin_config.max_sites,
        "seed": args.seed,
        "num_repeats": args.num_repeats,
        "test_size": args.test_size,
        "max_patients_per_domain": args.max_patients_per_domain,
        "max_train_samples_per_domain": sanitize_cap(args.max_train_samples_per_domain),
        "max_test_samples_per_domain": sanitize_cap(args.max_test_samples_per_domain),
        "max_plot_samples_per_domain": sanitize_cap(args.max_plot_samples_per_domain),
        "save_features": args.save_features,
        "save_plots": args.save_plots,
        "feature_dir": str(feature_dir) if feature_dir is not None else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_repo_relative(args.checkpoint)
    output_dir = resolve_output_dir(args.output_dir, checkpoint_path)
    feature_dir = resolve_feature_dir(args.feature_dir, output_dir) if args.save_features else None
    setup_logging(output_dir)
    start_time = time.time()

    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        benin_split_csv = resolve_repo_relative(args.benin_split_csv) if args.benin_split_csv else None
        if benin_split_csv is not None and not benin_split_csv.exists():
            raise FileNotFoundError(f"Benin split CSV does not exist: {benin_split_csv}")

        sa_split_csv = resolve_default_sa_split_csv(args.sa_split_csv)

        logger.info("Output directory: %s", output_dir)
        logger.info("Feature directory: %s", feature_dir if feature_dir is not None else "(not saving features)")
        logger.info("Checkpoint: %s", checkpoint_path)
        logger.info("Model config: %s", resolve_repo_relative(args.model_config or args.benin_config))
        logger.info("Benin config: %s", resolve_repo_relative(args.benin_config))
        logger.info("SA config: %s", resolve_repo_relative(args.sa_config))
        logger.info("Resolved Benin split CSV: %s", benin_split_csv)
        logger.info("Resolved SA split CSV: %s", sa_split_csv)

        model_config = load_runtime_config(
            config_path=args.model_config or args.benin_config,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            frame_sampling=args.frame_sampling,
            max_sites=args.max_sites,
            split_csv_override=None,
        )
        benin_config = load_runtime_config(
            config_path=args.benin_config,
            device=model_config.device,
            batch_size=args.batch_size or model_config.batch_size,
            num_workers=args.num_workers if args.num_workers is not None else model_config.num_workers,
            frame_sampling=args.frame_sampling or model_config.frame_sampling,
            max_sites=args.max_sites,
            split_csv_override=benin_split_csv,
        )
        sa_config = load_runtime_config(
            config_path=args.sa_config,
            device=model_config.device,
            batch_size=args.batch_size or model_config.batch_size,
            num_workers=args.num_workers if args.num_workers is not None else model_config.num_workers,
            frame_sampling=args.frame_sampling or model_config.frame_sampling,
            max_sites=args.max_sites,
            split_csv_override=sa_split_csv,
        )

        save_manifest(output_dir, feature_dir, args, benin_config, sa_config, model_config)
        logger.info("Manifest saved to %s", output_dir / "manifest.json")

        model = load_model(model_config, checkpoint_path)

        benin_loader, benin_num_patients = create_dataloader(
            benin_dataset,
            "Benin",
            benin_config,
            split=args.benin_split,
            max_patients=args.max_patients_per_domain,
            seed=args.seed,
        )
        sa_loader, sa_num_patients = create_dataloader(
            sa_dataset,
            "SA",
            sa_config,
            split=args.sa_split,
            max_patients=args.max_patients_per_domain,
            seed=args.seed,
        )

        logger.info("Benin dataset size: %d patients across %d batches", benin_num_patients, len(benin_loader))
        logger.info("SA dataset size: %d patients across %d batches", sa_num_patients, len(sa_loader))

        benin_payload = extract_domain_features(
            model=model,
            dataloader=benin_loader,
            domain_name="benin",
            device=model_config.device,
            use_amp=getattr(model_config, "use_amp", True),
        )
        sa_payload = extract_domain_features(
            model=model,
            dataloader=sa_loader,
            domain_name="sa",
            device=model_config.device,
            use_amp=getattr(model_config, "use_amp", True),
        )

        combined_payload = combine_domain_payloads(benin_payload, sa_payload)
        for level_name, payload in combined_payload.items():
            logger.info(
                "Combined %s rows=%d features=%d",
                level_name,
                payload["features"].shape[0],
                payload["features"].shape[1] if payload["features"].ndim == 2 else 0,
            )

        if args.save_features:
            logger.info("Saving extracted feature payloads")
            save_feature_payloads(feature_dir, combined_payload)
            logger.info("Feature payloads saved under %s", feature_dir)

        summaries = {}
        repeats_payload = {}
        max_train_cap = sanitize_cap(args.max_train_samples_per_domain)
        max_test_cap = sanitize_cap(args.max_test_samples_per_domain)

        for level_name, payload in combined_payload.items():
            logger.info("Running domain classifier for %s", level_name)
            summary = run_domain_classifier(
                features=payload["features"],
                metadata=payload["metadata"],
                level_name=level_name,
                test_size=args.test_size,
                num_repeats=args.num_repeats,
                max_train_samples_per_domain=max_train_cap,
                max_test_samples_per_domain=max_test_cap,
                seed=args.seed,
            )
            summaries[level_name] = summary
            repeats_payload[level_name] = summary["repeats"]
            logger.info(
                "%s AUROC: %.4f +/- %.4f",
                LEVEL_SPECS[level_name]["label"],
                summary["metrics_mean"]["auroc"],
                summary["metrics_std"]["auroc"],
            )

        metrics_path = output_dir / "domain_classifier_metrics.json"
        metrics_path.write_text(json.dumps(summaries, indent=2) + "\n")
        repeats_path = output_dir / "domain_classifier_repeats.json"
        repeats_path.write_text(json.dumps(repeats_payload, indent=2) + "\n")

        report_path = write_summary_report(output_dir, summaries)
        auroc_plot_path = save_auroc_bar_plot(output_dir, summaries) if args.save_plots else None
        pca_plot_path = (
            save_pca_overview(
                output_dir,
                combined_payload,
                sanitize_cap(args.max_plot_samples_per_domain),
                args.seed,
            )
            if args.save_plots
            else None
        )

        logger.info("Saved metrics to %s", metrics_path)
        logger.info("Saved repeats to %s", repeats_path)
        logger.info("Saved summary to %s", report_path)
        if auroc_plot_path is not None:
            logger.info("Saved AUROC plot to %s", auroc_plot_path)
        if pca_plot_path is not None:
            logger.info("Saved PCA overview to %s", pca_plot_path)

        elapsed_minutes = (time.time() - start_time) / 60.0
        logger.info("Domain shift probe completed successfully in %.2f minutes", elapsed_minutes)
    except Exception:
        logger.exception("Domain shift probe failed")
        raise


if __name__ == "__main__":
    main()
