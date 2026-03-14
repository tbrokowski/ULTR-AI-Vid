#!/usr/bin/env python3

"""Create t-SNE plots for cached domain-shift CLIP frame embeddings.

This script is the inspection companion to `domain_shift_probe.py`.
It reads cached CLIP frame embeddings from:

- `clip_frame_features.npy`
- `clip_frame_metadata.csv`

and creates several t-SNE visualizations.

Plot families
-------------
There are four plot families in this script:

1. Overall domain/TB plot
   - Output: `clip_frame_tsne_overall.png`
   - Uses all anatomical sites together.
   - Marker shape encodes domain:
     - Benin -> circle
     - SA -> triangle
   - Point color encodes TB label:
     - TB negative -> green
     - TB positive -> red

2. Site-filtered domain/TB plots
   - Outputs like:
     - `clip_frame_tsne_laterality_right.png`
     - `clip_frame_tsne_region_anterior.png`
     - `clip_frame_tsne_site_code_QASD.png`
   - These still use domain marker + TB color.
   - The difference is that rows are filtered to one anatomical subset first.

3. Overall site-legend plot
   - Output like:
     - `clip_frame_tsne_site_legend_site_code.png`
   - Uses all rows together, but now color/legend show anatomical site or site
     group rather than domain/TB.

4. Overall site+domain overlay plot
   - Output like:
     - `clip_frame_tsne_site_domain_overlay_region.png`
   - Uses all rows together.
   - Point color shows anatomical site or site group.
   - Marker shape still shows domain.
   - This is the plot to use if you want to see anatomical structure while still
     visually separating Benin and SA.

Important conceptual distinction
--------------------------------
- `--site-plot-mode`
  Creates extra FILTERED plots that still show domain/TB.
  Example:
  `--site-plot-mode laterality`
  means one domain/TB plot for `right` and one for `left`.

- `--site-legend-mode`
  Creates ONE additional overall plot where anatomy is the legend.

- `--site-domain-overlay-mode`
  Creates ONE additional overall plot where anatomy is shown by color and domain
  is still shown by marker shape.

Available anatomy grouping modes
--------------------------------
- `site_code`
  Exact site codes, e.g. `QAID`, `QASG`, `QPSD`, `APXD`.

- `laterality`
  `right`, `left`, optionally `unknown`.

- `region`
  `anterior`, `lateral`, `posterior`, `apical`, optionally `unknown`.

- `family`
  `standard`, `sweep`, optionally `unknown`.

Sampling
--------
The cached frame-level arrays are usually large, so the script samples rows
before running t-SNE:

- `--max-samples-per-combination`
  Used by domain/TB plots.
  Sampling is per `(domain, tb_label)` combination.

- `--max-samples-per-site-group`
  Used by the site-legend plot.
  Sampling is per site or site-group value.

- `--max-samples-per-site-domain-combination`
  Used by the site+domain overlay plot.
  Sampling is per `(site-group, domain)` combination so both domains remain
  visible inside a site group when possible.

Most important arguments
------------------------
- `--run-name` / `--feature-dir`
  Choose which cached feature directory to read.

- `--site-plot-mode`
  Controls the filtered domain/TB plots.
  Use `none` to disable them.

- `--site-groups`
  Optional manual subset for `--site-plot-mode`.
  Example:
  `--site-plot-mode site_code --site-groups QAID QASD APXD`

- `--site-legend-mode`
  Controls the one overall anatomy-colored plot with an anatomy legend.
  Use `none` to disable it.

- `--site-legend-groups`
  Optional manual subset for `--site-legend-mode`.

- `--site-domain-overlay-mode`
  Controls the one overall plot where anatomy is color and domain is marker.
  Use `region` if you want a compact, interpretable overview.
  Use `none` to disable it.

- `--site-domain-overlay-groups`
  Optional manual subset for `--site-domain-overlay-mode`.

- `--include-unknown-site-groups`
  Include `unknown` groups where relevant.
  This can be informative, but it may dominate if many rows are unmapped.

- `--normalize`
  Feature preprocessing before PCA/t-SNE:
  - `l2`
  - `standardize`
  - `none`

- `--pca-dim`
  PCA dimension before t-SNE. Default `50`.
  Set `<= 0` to disable PCA.

Typical commands
----------------
Overall domain/TB plot only:
`python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode none`

Overall domain/TB + per-laterality domain/TB plots:
`python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode laterality --site-legend-mode none --site-domain-overlay-mode none`

Overall plot with anatomical sites in the legend:
`python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode none --site-legend-mode site_code --site-domain-overlay-mode none`

Overall plot with regions by color and Benin/SA by marker:
`python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode region`
"""

import argparse
import csv
import inspect
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from domain_shift_plot_utils import (
    default_site_groups,
    load_tb_label_lookup,
    lookup_tb_label,
    site_group_value,
    site_metadata_from_code,
    site_metadata_from_index,
)


LOGGER = logging.getLogger("domain_shift_clip_tsne")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_ROOT = REPO_ROOT / "checkpoints" / "domain_shift_probe_features"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "domain_shift_probe_results"
FEATURE_FILE = "clip_frame_features.npy"
METADATA_FILE = "clip_frame_metadata.csv"

DOMAIN_LABELS = {"benin": "Benin", "sa": "SA"}
DOMAIN_MARKERS = {"benin": "o", "sa": "^"}
TB_LABELS = {0: "TB negative", 1: "TB positive"}
TB_COLORS = {0: "#2E8B57", 1: "#D73027"}
GROUP_LABELS = {
    "right": "Right hemithorax",
    "left": "Left hemithorax",
    "unknown": "Unknown site",
    "standard": "Standard anatomical views",
    "sweep": "Sweep views",
    "anterior": "Anterior sites",
    "lateral": "Lateral sites",
    "posterior": "Posterior sites",
    "apical": "Apical sites",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create t-SNE plots from cached CLIP frame embeddings. "
            "Supports overall domain/TB plots, filtered domain/TB plots, "
            "overall anatomy-legend plots, and overall anatomy-color + domain-marker plots."
        ),
        epilog=(
            "Examples:\n"
            "  Overall domain/TB only:\n"
            "    python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode none\n\n"
            "  Overall + laterality-filtered domain/TB:\n"
            "    python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode laterality --site-legend-mode none --site-domain-overlay-mode none\n\n"
            "  Overall anatomy legend plot:\n"
            "    python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode none --site-legend-mode site_code --site-domain-overlay-mode none\n\n"
            "  Overall region colors + domain markers:\n"
            "    python domain_shift_clip_tsne.py --run-name benin_pretrained_fold3_test --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode region"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Feature directory containing clip_frame_features.npy and clip_frame_metadata.csv",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name under checkpoints/domain_shift_probe_features/ to use instead of --feature-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where plots and sampled coordinates will be written",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest.json from a domain_shift_probe_results run",
    )
    parser.add_argument(
        "--benin-config",
        type=str,
        default=None,
        help="Benin config override; used to recover TB labels when metadata lacks them",
    )
    parser.add_argument(
        "--sa-config",
        type=str,
        default=None,
        help="SA config override; used to recover TB labels when metadata lacks them",
    )
    parser.add_argument(
        "--max-samples-per-combination",
        type=int,
        default=600,
        help="Maximum sampled rows per (domain, TB) combination for domain/TB plots",
    )
    parser.add_argument(
        "--max-samples-per-site-group",
        type=int,
        default=250,
        help="Maximum sampled rows per anatomical site or site-group in the site-legend plot",
    )
    parser.add_argument(
        "--max-samples-per-site-domain-combination",
        type=int,
        default=250,
        help="Maximum sampled rows per (site-group, domain) combination in the overlay plot",
    )
    parser.add_argument(
        "--site-plot-mode",
        type=str,
        default="laterality",
        choices=["none", "laterality", "family", "region", "site_code"],
        help=(
            "Generate extra FILTERED plots that still use domain marker + TB color. "
            "Example: laterality -> one plot for right, one for left."
        ),
    )
    parser.add_argument(
        "--site-groups",
        nargs="*",
        default=None,
        help=(
            "Optional explicit subset for --site-plot-mode. "
            "Example: --site-plot-mode site_code --site-groups QAID QASD APXD"
        ),
    )
    parser.add_argument(
        "--site-legend-mode",
        type=str,
        default="site_code",
        choices=["none", "laterality", "family", "region", "site_code"],
        help=(
            "Generate ONE additional overall plot where the legend is anatomy itself "
            "(site_code, region, laterality, or family), not domain/TB."
        ),
    )
    parser.add_argument(
        "--site-legend-groups",
        nargs="*",
        default=None,
        help=(
            "Optional explicit subset for --site-legend-mode. "
            "Example: --site-legend-mode region --site-legend-groups anterior posterior"
        ),
    )
    parser.add_argument(
        "--site-domain-overlay-mode",
        type=str,
        default="none",
        choices=["none", "laterality", "family", "region", "site_code"],
        help=(
            "Generate ONE additional overall plot where anatomy is color and domain "
            "is marker shape."
        ),
    )
    parser.add_argument(
        "--site-domain-overlay-groups",
        nargs="*",
        default=None,
        help=(
            "Optional explicit subset for --site-domain-overlay-mode. "
            "Example: --site-domain-overlay-mode region --site-domain-overlay-groups anterior posterior"
        ),
    )
    parser.add_argument(
        "--include-unknown-site-groups",
        action="store_true",
        help="Include unknown site groups in site-related plots",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="l2",
        choices=["none", "l2", "standardize"],
        help="Feature preprocessing before PCA/t-SNE",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA dimension before t-SNE; <=0 disables PCA",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity; automatically reduced when needed",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=20.0,
        help="Scatter point size",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.78,
        help="Scatter alpha",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved figure DPI",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling, PCA, and t-SNE",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def resolve_repo_relative(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def discover_feature_dir(feature_root: Path) -> Path:
    candidates = []
    if feature_root.exists():
        for candidate in feature_root.iterdir():
            if not candidate.is_dir():
                continue
            if (candidate / FEATURE_FILE).exists() and (candidate / METADATA_FILE).exists():
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            "No feature directories found under {0}. Pass --feature-dir explicitly.".format(feature_root)
        )

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    LOGGER.info("Auto-selected most recent feature directory: %s", candidates[0])
    return candidates[0]


def resolve_feature_dir(args: argparse.Namespace) -> Path:
    if args.feature_dir is not None:
        feature_dir = resolve_repo_relative(args.feature_dir)
    elif args.run_name is not None:
        feature_dir = DEFAULT_FEATURE_ROOT / args.run_name
    else:
        feature_dir = discover_feature_dir(DEFAULT_FEATURE_ROOT)

    if feature_dir is None or not feature_dir.exists():
        raise FileNotFoundError("Feature directory does not exist: {0}".format(feature_dir))
    if not (feature_dir / FEATURE_FILE).exists():
        raise FileNotFoundError("Missing feature file: {0}".format(feature_dir / FEATURE_FILE))
    if not (feature_dir / METADATA_FILE).exists():
        raise FileNotFoundError("Missing metadata file: {0}".format(feature_dir / METADATA_FILE))
    return feature_dir


def resolve_associated_results_dir(feature_dir: Path) -> Optional[Path]:
    candidate = DEFAULT_RESULTS_ROOT / feature_dir.name
    return candidate if candidate.exists() else None


def resolve_manifest_path(args: argparse.Namespace, feature_dir: Path) -> Optional[Path]:
    if args.manifest is not None:
        manifest_path = resolve_repo_relative(args.manifest)
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError("Manifest does not exist: {0}".format(manifest_path))
        return manifest_path

    results_dir = resolve_associated_results_dir(feature_dir)
    if results_dir is None:
        return None

    candidate = results_dir / "manifest.json"
    return candidate if candidate.exists() else None


def resolve_output_dir(args: argparse.Namespace, feature_dir: Path) -> Path:
    if args.output_dir is not None:
        output_dir = resolve_repo_relative(args.output_dir)
    else:
        results_dir = resolve_associated_results_dir(feature_dir)
        if results_dir is not None:
            output_dir = results_dir / "tsne_plots"
        else:
            output_dir = feature_dir / "tsne_plots"

    if output_dir is None:
        raise ValueError("Unable to resolve output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def load_yaml_config(config_path: Path) -> Dict[str, object]:
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def read_metadata_rows(metadata_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    value_str = str(value).strip()
    if value_str == "":
        return None
    try:
        return int(float(value_str))
    except ValueError:
        return None


def load_label_lookups(
    args: argparse.Namespace,
    manifest: Optional[Dict[str, object]],
) -> Dict[str, Dict[str, int]]:
    config_paths: Dict[str, Optional[Path]] = {
        "benin": resolve_repo_relative(args.benin_config),
        "sa": resolve_repo_relative(args.sa_config),
    }
    if manifest is not None:
        if config_paths["benin"] is None and manifest.get("benin_config") is not None:
            config_paths["benin"] = resolve_repo_relative(str(manifest["benin_config"]))
        if config_paths["sa"] is None and manifest.get("sa_config") is not None:
            config_paths["sa"] = resolve_repo_relative(str(manifest["sa_config"]))

    lookups: Dict[str, Dict[str, int]] = {}
    for domain_name, config_path in config_paths.items():
        if config_path is None:
            continue
        config = load_yaml_config(config_path)
        labels_csv = resolve_repo_relative(str(config["labels_csv"]))
        if labels_csv is None or not labels_csv.exists():
            raise FileNotFoundError("Labels CSV does not exist: {0}".format(labels_csv))
        lookups[domain_name] = load_tb_label_lookup(labels_csv)
        LOGGER.info(
            "Loaded %d TB labels for %s from %s",
            len(lookups[domain_name]),
            domain_name,
            labels_csv,
        )
    return lookups


def enrich_metadata_rows(
    rows: Sequence[Dict[str, object]],
    label_lookups: Dict[str, Dict[str, int]],
) -> Tuple[List[Dict[str, object]], Counter]:
    enriched: List[Dict[str, object]] = []
    stats: Counter = Counter()

    for row_index, row in enumerate(rows):
        enriched_row = dict(row)
        enriched_row["row_index"] = row_index
        enriched_row["domain"] = str(row.get("domain", "")).strip().lower()
        enriched_row["patient_id"] = str(row.get("patient_id", "")).strip()
        enriched_row["site_index"] = parse_optional_int(row.get("site_index"))
        enriched_row["site_position"] = parse_optional_int(row.get("site_position"))
        enriched_row["frame_index"] = parse_optional_int(row.get("frame_index"))

        raw_site_code = str(row.get("site_code", "")).strip()
        if raw_site_code:
            site_meta = site_metadata_from_code(raw_site_code)
        else:
            site_meta = site_metadata_from_index(enriched_row.get("site_index"))
        enriched_row.update(site_meta)

        tb_label = parse_optional_int(row.get("tb_label"))
        if tb_label is None:
            tb_label = lookup_tb_label(
                enriched_row["patient_id"],
                label_lookups.get(enriched_row["domain"], {}),
            )

        if tb_label in (0, 1):
            enriched_row["tb_label"] = tb_label
            enriched_row["tb_status"] = TB_LABELS[tb_label]
            stats["tb_labels_found"] += 1
        else:
            enriched_row["tb_label"] = None
            enriched_row["tb_status"] = "unknown"
            stats["tb_labels_missing"] += 1

        if enriched_row["site_code"] == "UNKNOWN":
            stats["unknown_site_rows"] += 1
        enriched.append(enriched_row)

    return enriched, stats


def drop_invalid_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    valid_rows = [row for row in rows if row.get("tb_label") in (0, 1)]
    dropped = len(rows) - len(valid_rows)
    if dropped > 0:
        LOGGER.warning("Dropped %d rows without a valid TB label", dropped)
    return valid_rows


def sample_indices_by_group(
    rows: Sequence[Dict[str, object]],
    group_keys: Sequence[object],
    max_samples_per_group: int,
    seed: int,
) -> np.ndarray:
    grouped_indices: Dict[object, List[int]] = defaultdict(list)
    for position, group_key in enumerate(group_keys):
        grouped_indices[group_key].append(position)

    rng = np.random.default_rng(seed)
    selected_positions: List[int] = []
    for group_key in sorted(grouped_indices, key=lambda value: str(value)):
        positions = np.asarray(grouped_indices[group_key], dtype=np.int64)
        if max_samples_per_group > 0 and len(positions) > max_samples_per_group:
            positions = np.sort(rng.choice(positions, size=max_samples_per_group, replace=False))
        selected_positions.extend(positions.tolist())

    return np.asarray(selected_positions, dtype=np.int64)


def sample_indices_domain_tb(
    rows: Sequence[Dict[str, object]],
    max_samples_per_combination: int,
    seed: int,
) -> np.ndarray:
    group_keys = [(str(row["domain"]), int(row["tb_label"])) for row in rows]
    return sample_indices_by_group(rows, group_keys, max_samples_per_combination, seed)


def sample_indices_site_group(
    rows: Sequence[Dict[str, object]],
    group_by: str,
    max_samples_per_group: int,
    seed: int,
) -> np.ndarray:
    group_keys = [site_group_value(row, group_by) for row in rows]
    return sample_indices_by_group(rows, group_keys, max_samples_per_group, seed)


def sample_indices_site_domain_group(
    rows: Sequence[Dict[str, object]],
    group_by: str,
    max_samples_per_group: int,
    seed: int,
) -> np.ndarray:
    group_keys = [(site_group_value(row, group_by), str(row["domain"])) for row in rows]
    return sample_indices_by_group(rows, group_keys, max_samples_per_group, seed)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return features / norms


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-8, None)
    return (features - mean) / std


def preprocess_features(
    features: np.ndarray,
    normalize: str,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    processed = np.asarray(features, dtype=np.float32)
    if normalize == "l2":
        processed = l2_normalize(processed)
    elif normalize == "standardize":
        processed = standardize(processed)

    if pca_dim > 0 and processed.shape[1] > pca_dim and processed.shape[0] > 2:
        n_components = min(pca_dim, processed.shape[0] - 1, processed.shape[1])
        processed = PCA(n_components=n_components, random_state=seed).fit_transform(processed)
    return processed


def effective_perplexity(requested: float, n_samples: int) -> float:
    if n_samples <= 3:
        raise ValueError("Need at least 4 samples for t-SNE")
    return max(2.0, min(requested, float(n_samples - 1) / 3.0))


def compute_tsne(
    features: np.ndarray,
    perplexity: float,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    tsne_kwargs = {
        "n_components": 2,
        "init": "pca",
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE.__init__).parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(features)


def plot_counts_text(rows: Sequence[Dict[str, object]]) -> str:
    counts = Counter((str(row["domain"]), int(row["tb_label"])) for row in rows)
    lines = []
    for domain_name in ["benin", "sa"]:
        for tb_label in [0, 1]:
            value = counts.get((domain_name, tb_label), 0)
            lines.append("{0}, {1}: {2}".format(DOMAIN_LABELS[domain_name], TB_LABELS[tb_label], value))
    return "\n".join(lines)


def group_title(group_by: str, group_name: str) -> str:
    if group_by == "site_code":
        return "Site {0}".format(group_name)
    return GROUP_LABELS.get(group_name, group_name.replace("_", " ").title())


def group_display_label(group_by: str, group_name: str) -> str:
    if group_by == "site_code":
        return str(group_name)
    return GROUP_LABELS.get(group_name, group_name.replace("_", " ").title())


def group_names_for_mode(
    all_rows: Sequence[Dict[str, object]],
    group_by: str,
    explicit_groups: Optional[Sequence[str]],
    include_unknown_groups: bool,
) -> List[str]:
    if group_by == "none":
        return []

    if explicit_groups:
        if group_by == "site_code":
            groups = [str(group).strip().upper() for group in explicit_groups]
        else:
            groups = [str(group).strip().lower() for group in explicit_groups]
    else:
        groups = default_site_groups(group_by, include_unknown_groups)

    available_groups = {site_group_value(row, group_by) for row in all_rows}
    return [group_name for group_name in groups if group_name in available_groups]


def save_points_csv(path: Path, rows: Sequence[Dict[str, object]], coords: np.ndarray) -> None:
    fieldnames = [
        "domain",
        "patient_id",
        "tb_label",
        "tb_status",
        "site_index",
        "site_code",
        "site_laterality",
        "site_region",
        "site_family",
        "site_position",
        "frame_index",
        "tsne_1",
        "tsne_2",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, coord in zip(rows, coords):
            writer.writerow(
                {
                    "domain": row["domain"],
                    "patient_id": row["patient_id"],
                    "tb_label": row["tb_label"],
                    "tb_status": row["tb_status"],
                    "site_index": row["site_index"],
                    "site_code": row["site_code"],
                    "site_laterality": row["site_laterality"],
                    "site_region": row["site_region"],
                    "site_family": row["site_family"],
                    "site_position": row["site_position"],
                    "frame_index": row["frame_index"],
                    "tsne_1": float(coord[0]),
                    "tsne_2": float(coord[1]),
                }
            )


def build_site_group_colors(group_names: Sequence[str]) -> Dict[str, object]:
    non_unknown = [group_name for group_name in group_names if str(group_name).lower() != "unknown"]
    colors: Dict[str, object] = {}
    if non_unknown:
        cmap = plt.get_cmap("tab20", len(non_unknown))
        for index, group_name in enumerate(non_unknown):
            colors[group_name] = cmap(index)
    for group_name in group_names:
        if str(group_name).lower() == "unknown":
            colors[group_name] = "#9E9E9E"
    return colors


def render_domain_tb_plot(
    coords: np.ndarray,
    rows: Sequence[Dict[str, object]],
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.4))

    for domain_name in ["benin", "sa"]:
        for tb_label in [0, 1]:
            mask = np.array(
                [
                    (str(row["domain"]) == domain_name) and (int(row["tb_label"]) == tb_label)
                    for row in rows
                ],
                dtype=bool,
            )
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                alpha=alpha,
                c=TB_COLORS[tb_label],
                marker=DOMAIN_MARKERS[domain_name],
                edgecolors="white",
                linewidths=0.35,
            )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)

    domain_handles = [
        Line2D(
            [0],
            [0],
            marker=DOMAIN_MARKERS[domain_name],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
            label=DOMAIN_LABELS[domain_name],
        )
        for domain_name in ["benin", "sa"]
    ]
    tb_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=TB_COLORS[tb_label],
            markerfacecolor=TB_COLORS[tb_label],
            markeredgecolor=TB_COLORS[tb_label],
            markersize=8,
            linestyle="None",
            label=TB_LABELS[tb_label],
        )
        for tb_label in [0, 1]
    ]

    legend_domain = ax.legend(handles=domain_handles, title="Domain", loc="upper right", frameon=True)
    ax.add_artist(legend_domain)
    ax.legend(handles=tb_handles, title="TB label", loc="lower right", frameon=True)

    ax.text(
        0.015,
        0.015,
        plot_counts_text(rows),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_site_legend_plot(
    coords: np.ndarray,
    rows: Sequence[Dict[str, object]],
    group_by: str,
    group_names: Sequence[str],
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 7.4))
    colors = build_site_group_colors(group_names)

    handles = []
    for group_name in group_names:
        mask = np.array([site_group_value(row, group_by) == group_name for row in rows], dtype=bool)
        if not np.any(mask):
            continue
        color = colors[group_name]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color],
            marker="o",
            edgecolors="white",
            linewidths=0.25,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                linestyle="None",
                label=group_display_label(group_by, group_name),
            )
        )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(
        handles=handles,
        title="Anatomical site" if group_by == "site_code" else "Site group",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_site_domain_overlay_plot(
    coords: np.ndarray,
    rows: Sequence[Dict[str, object]],
    group_by: str,
    group_names: Sequence[str],
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 7.5))
    colors = build_site_group_colors(group_names)

    for group_name in group_names:
        color = colors[group_name]
        for domain_name in ["benin", "sa"]:
            mask = np.array(
                [
                    (site_group_value(row, group_by) == group_name) and (str(row["domain"]) == domain_name)
                    for row in rows
                ],
                dtype=bool,
            )
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                alpha=alpha,
                c=[color],
                marker=DOMAIN_MARKERS[domain_name],
                edgecolors="white",
                linewidths=0.30,
            )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)

    domain_handles = [
        Line2D(
            [0],
            [0],
            marker=DOMAIN_MARKERS[domain_name],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
            label=DOMAIN_LABELS[domain_name],
        )
        for domain_name in ["benin", "sa"]
    ]
    site_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[group_name],
            markerfacecolor=colors[group_name],
            markeredgecolor=colors[group_name],
            markersize=7,
            linestyle="None",
            label=group_display_label(group_by, group_name),
        )
        for group_name in group_names
    ]

    legend_domain = ax.legend(handles=domain_handles, title="Domain", loc="upper right", frameon=True)
    ax.add_artist(legend_domain)
    ax.legend(
        handles=site_handles,
        title="Anatomical site" if group_by == "site_code" else "Site group",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_plot_specs(
    all_rows: Sequence[Dict[str, object]],
    site_plot_mode: str,
    explicit_groups: Optional[Sequence[str]],
    include_unknown_groups: bool,
) -> List[Tuple[str, str, Optional[str]]]:
    specs: List[Tuple[str, str, Optional[str]]] = [("overall", "All sites", None)]
    if site_plot_mode == "none":
        return specs

    groups = group_names_for_mode(all_rows, site_plot_mode, explicit_groups, include_unknown_groups)
    for group_name in groups:
        specs.append((site_plot_mode, group_title(site_plot_mode, group_name), group_name))
    return specs


def filter_rows_for_plot(
    all_rows: Sequence[Dict[str, object]],
    plot_mode: str,
    group_name: Optional[str],
) -> List[Dict[str, object]]:
    if plot_mode == "overall" or group_name is None:
        return list(all_rows)
    return [row for row in all_rows if site_group_value(row, plot_mode) == group_name]


def sanitize_slug(value: str) -> str:
    safe = []
    for char in value.lower():
        safe.append(char if char.isalnum() else "_")
    slug = "".join(safe).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "plot"


def main() -> None:
    args = parse_args()
    setup_logging()

    feature_dir = resolve_feature_dir(args)
    output_dir = resolve_output_dir(args, feature_dir)
    manifest_path = resolve_manifest_path(args, feature_dir)
    manifest = load_json(manifest_path) if manifest_path is not None else None

    LOGGER.info("Feature directory: %s", feature_dir)
    LOGGER.info("Output directory: %s", output_dir)
    if manifest_path is not None:
        LOGGER.info("Manifest: %s", manifest_path)

    features = np.load(feature_dir / FEATURE_FILE, mmap_mode="r")
    rows = read_metadata_rows(feature_dir / METADATA_FILE)
    if features.shape[0] != len(rows):
        raise ValueError(
            "Feature/metadata mismatch: {0} rows in features vs {1} rows in metadata".format(
                features.shape[0],
                len(rows),
            )
        )

    need_label_lookup = any(parse_optional_int(row.get("tb_label")) is None for row in rows)
    label_lookups = load_label_lookups(args, manifest) if need_label_lookup else {}
    if need_label_lookup:
        domains_needing_lookup = {
            str(row.get("domain", "")).strip().lower()
            for row in rows
            if parse_optional_int(row.get("tb_label")) is None
        }
        missing_domains = sorted(domain for domain in domains_needing_lookup if domain and domain not in label_lookups)
        if missing_domains:
            raise ValueError(
                "Need config paths for TB label recovery in domains: {0}. "
                "Pass --manifest or explicit --benin-config/--sa-config.".format(", ".join(missing_domains))
            )

    enriched_rows, enrich_stats = enrich_metadata_rows(rows, label_lookups)
    enriched_rows = drop_invalid_rows(enriched_rows)

    LOGGER.info(
        "Loaded %d CLIP frame rows (%d unknown-site rows, %d missing TB labels before filtering)",
        len(rows),
        enrich_stats.get("unknown_site_rows", 0),
        enrich_stats.get("tb_labels_missing", 0),
    )
    LOGGER.info("Using %d valid rows after metadata enrichment", len(enriched_rows))

    plot_specs = build_plot_specs(
        enriched_rows,
        args.site_plot_mode,
        args.site_groups,
        args.include_unknown_site_groups,
    )

    summary = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "max_samples_per_combination": args.max_samples_per_combination,
        "max_samples_per_site_group": args.max_samples_per_site_group,
        "max_samples_per_site_domain_combination": args.max_samples_per_site_domain_combination,
        "normalize": args.normalize,
        "pca_dim": args.pca_dim,
        "perplexity": args.perplexity,
        "learning_rate": args.learning_rate,
        "random_state": args.random_state,
        "plots": [],
    }

    # Domain/TB plots: one overall plus optional filtered subsets.
    for plot_index, (plot_mode, plot_title, group_name) in enumerate(plot_specs):
        plot_rows = filter_rows_for_plot(enriched_rows, plot_mode, group_name)
        if len(plot_rows) < 12:
            LOGGER.warning("Skipping %s: only %d eligible rows", plot_title, len(plot_rows))
            continue

        domains_present = sorted({str(row["domain"]) for row in plot_rows})
        tb_present = sorted({int(row["tb_label"]) for row in plot_rows})
        if len(domains_present) < 2:
            LOGGER.warning(
                "%s includes only one domain after filtering: %s",
                plot_title,
                ", ".join(domains_present) if domains_present else "(none)",
            )
        if len(tb_present) < 2:
            LOGGER.warning(
                "%s includes only one TB class after filtering: %s",
                plot_title,
                ", ".join(TB_LABELS[tb_label] for tb_label in tb_present) if tb_present else "(none)",
            )

        sampled_positions = sample_indices_domain_tb(
            plot_rows,
            max_samples_per_combination=args.max_samples_per_combination,
            seed=args.random_state + plot_index,
        )
        if sampled_positions.size < 12:
            LOGGER.warning("Skipping %s: only %d sampled rows", plot_title, sampled_positions.size)
            continue

        sampled_rows = [plot_rows[position] for position in sampled_positions.tolist()]
        feature_indices = np.asarray([int(row["row_index"]) for row in sampled_rows], dtype=np.int64)
        sampled_features = np.asarray(features[feature_indices], dtype=np.float32)
        processed_features = preprocess_features(
            sampled_features,
            normalize=args.normalize,
            pca_dim=args.pca_dim,
            seed=args.random_state,
        )
        perplexity = effective_perplexity(args.perplexity, processed_features.shape[0])
        coords = compute_tsne(
            processed_features,
            perplexity=perplexity,
            learning_rate=args.learning_rate,
            seed=args.random_state + plot_index,
        )

        suffix = "overall" if group_name is None else sanitize_slug("{0}_{1}".format(plot_mode, group_name))
        image_path = output_dir / "clip_frame_tsne_{0}.png".format(suffix)
        csv_path = output_dir / "clip_frame_tsne_{0}_points.csv".format(suffix)

        render_domain_tb_plot(
            coords,
            sampled_rows,
            title="{0}\nCLIP frame embeddings".format(plot_title),
            output_path=image_path,
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
        )
        save_points_csv(csv_path, sampled_rows, coords)

        counts = Counter((str(row["domain"]), int(row["tb_label"])) for row in sampled_rows)
        LOGGER.info(
            "Saved %s (%d points, perplexity=%.2f) to %s",
            plot_title,
            len(sampled_rows),
            perplexity,
            image_path,
        )

        summary["plots"].append(
            {
                "plot_kind": "domain_tb",
                "plot_mode": plot_mode,
                "group_name": group_name,
                "title": plot_title,
                "n_points": len(sampled_rows),
                "perplexity": perplexity,
                "image_path": str(image_path),
                "points_csv": str(csv_path),
                "counts": {
                    "{0}|tb_{1}".format(domain_name, tb_label): int(counts.get((domain_name, tb_label), 0))
                    for domain_name in ["benin", "sa"]
                    for tb_label in [0, 1]
                },
            }
        )

    # Overall anatomy legend plot.
    if args.site_legend_mode != "none":
        site_legend_groups = group_names_for_mode(
            enriched_rows,
            args.site_legend_mode,
            args.site_legend_groups,
            args.include_unknown_site_groups,
        )
        site_legend_group_set = set(site_legend_groups)
        site_legend_rows = [
            row for row in enriched_rows if site_group_value(row, args.site_legend_mode) in site_legend_group_set
        ]

        if len(site_legend_rows) < 12:
            LOGGER.warning(
                "Skipping site-legend plot for %s: only %d eligible rows",
                args.site_legend_mode,
                len(site_legend_rows),
            )
        else:
            sampled_positions = sample_indices_site_group(
                site_legend_rows,
                group_by=args.site_legend_mode,
                max_samples_per_group=args.max_samples_per_site_group,
                seed=args.random_state + 100,
            )
            if sampled_positions.size < 12:
                LOGGER.warning(
                    "Skipping site-legend plot for %s: only %d sampled rows",
                    args.site_legend_mode,
                    sampled_positions.size,
                )
            else:
                sampled_rows = [site_legend_rows[position] for position in sampled_positions.tolist()]
                feature_indices = np.asarray([int(row["row_index"]) for row in sampled_rows], dtype=np.int64)
                sampled_features = np.asarray(features[feature_indices], dtype=np.float32)
                processed_features = preprocess_features(
                    sampled_features,
                    normalize=args.normalize,
                    pca_dim=args.pca_dim,
                    seed=args.random_state,
                )
                perplexity = effective_perplexity(args.perplexity, processed_features.shape[0])
                coords = compute_tsne(
                    processed_features,
                    perplexity=perplexity,
                    learning_rate=args.learning_rate,
                    seed=args.random_state + 100,
                )

                image_path = output_dir / "clip_frame_tsne_site_legend_{0}.png".format(
                    sanitize_slug(args.site_legend_mode)
                )
                csv_path = output_dir / "clip_frame_tsne_site_legend_{0}_points.csv".format(
                    sanitize_slug(args.site_legend_mode)
                )
                render_site_legend_plot(
                    coords,
                    sampled_rows,
                    group_by=args.site_legend_mode,
                    group_names=site_legend_groups,
                    title="All sites by {0}\nCLIP frame embeddings".format(
                        "anatomical site" if args.site_legend_mode == "site_code" else args.site_legend_mode
                    ),
                    output_path=image_path,
                    point_size=args.point_size,
                    alpha=args.alpha,
                    dpi=args.dpi,
                )
                save_points_csv(csv_path, sampled_rows, coords)

                group_counts = Counter(site_group_value(row, args.site_legend_mode) for row in sampled_rows)
                LOGGER.info(
                    "Saved site-legend plot for %s (%d points, perplexity=%.2f) to %s",
                    args.site_legend_mode,
                    len(sampled_rows),
                    perplexity,
                    image_path,
                )

                summary["plots"].append(
                    {
                        "plot_kind": "site_legend",
                        "plot_mode": args.site_legend_mode,
                        "group_name": None,
                        "title": "All sites by {0}".format(args.site_legend_mode),
                        "n_points": len(sampled_rows),
                        "perplexity": perplexity,
                        "image_path": str(image_path),
                        "points_csv": str(csv_path),
                        "counts": {
                            group_name: int(group_counts.get(group_name, 0))
                            for group_name in site_legend_groups
                        },
                    }
                )

    # Overall anatomy-color + domain-marker overlay plot.
    if args.site_domain_overlay_mode != "none":
        overlay_groups = group_names_for_mode(
            enriched_rows,
            args.site_domain_overlay_mode,
            args.site_domain_overlay_groups,
            args.include_unknown_site_groups,
        )
        overlay_group_set = set(overlay_groups)
        overlay_rows = [
            row for row in enriched_rows if site_group_value(row, args.site_domain_overlay_mode) in overlay_group_set
        ]

        if len(overlay_rows) < 12:
            LOGGER.warning(
                "Skipping site-domain overlay plot for %s: only %d eligible rows",
                args.site_domain_overlay_mode,
                len(overlay_rows),
            )
        else:
            overlay_domains_present = sorted({str(row["domain"]) for row in overlay_rows})
            if len(overlay_domains_present) < 2:
                LOGGER.warning(
                    "Site-domain overlay for %s includes only one domain after filtering: %s",
                    args.site_domain_overlay_mode,
                    ", ".join(overlay_domains_present) if overlay_domains_present else "(none)",
                )
            sampled_positions = sample_indices_site_domain_group(
                overlay_rows,
                group_by=args.site_domain_overlay_mode,
                max_samples_per_group=args.max_samples_per_site_domain_combination,
                seed=args.random_state + 200,
            )
            if sampled_positions.size < 12:
                LOGGER.warning(
                    "Skipping site-domain overlay plot for %s: only %d sampled rows",
                    args.site_domain_overlay_mode,
                    sampled_positions.size,
                )
            else:
                sampled_rows = [overlay_rows[position] for position in sampled_positions.tolist()]
                feature_indices = np.asarray([int(row["row_index"]) for row in sampled_rows], dtype=np.int64)
                sampled_features = np.asarray(features[feature_indices], dtype=np.float32)
                processed_features = preprocess_features(
                    sampled_features,
                    normalize=args.normalize,
                    pca_dim=args.pca_dim,
                    seed=args.random_state,
                )
                perplexity = effective_perplexity(args.perplexity, processed_features.shape[0])
                coords = compute_tsne(
                    processed_features,
                    perplexity=perplexity,
                    learning_rate=args.learning_rate,
                    seed=args.random_state + 200,
                )

                image_path = output_dir / "clip_frame_tsne_site_domain_overlay_{0}.png".format(
                    sanitize_slug(args.site_domain_overlay_mode)
                )
                csv_path = output_dir / "clip_frame_tsne_site_domain_overlay_{0}_points.csv".format(
                    sanitize_slug(args.site_domain_overlay_mode)
                )
                render_site_domain_overlay_plot(
                    coords,
                    sampled_rows,
                    group_by=args.site_domain_overlay_mode,
                    group_names=overlay_groups,
                    title="All sites by {0} with domain markers\nCLIP frame embeddings".format(
                        "anatomical site"
                        if args.site_domain_overlay_mode == "site_code"
                        else args.site_domain_overlay_mode
                    ),
                    output_path=image_path,
                    point_size=args.point_size,
                    alpha=args.alpha,
                    dpi=args.dpi,
                )
                save_points_csv(csv_path, sampled_rows, coords)

                overlay_counts = Counter(
                    (site_group_value(row, args.site_domain_overlay_mode), str(row["domain"])) for row in sampled_rows
                )
                LOGGER.info(
                    "Saved site-domain overlay plot for %s (%d points, perplexity=%.2f) to %s",
                    args.site_domain_overlay_mode,
                    len(sampled_rows),
                    perplexity,
                    image_path,
                )

                summary["plots"].append(
                    {
                        "plot_kind": "site_domain_overlay",
                        "plot_mode": args.site_domain_overlay_mode,
                        "group_name": None,
                        "title": "All sites by {0} with domain markers".format(args.site_domain_overlay_mode),
                        "n_points": len(sampled_rows),
                        "perplexity": perplexity,
                        "image_path": str(image_path),
                        "points_csv": str(csv_path),
                        "counts": {
                            "{0}|{1}".format(group_name, domain_name): int(
                                overlay_counts.get((group_name, domain_name), 0)
                            )
                            for group_name in overlay_groups
                            for domain_name in ["benin", "sa"]
                        },
                    }
                )

    summary_path = output_dir / "clip_frame_tsne_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
