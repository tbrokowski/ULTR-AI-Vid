#!/usr/bin/env python3

"""Create embedding plots for cached domain-shift CLIP frame embeddings.

This script is the inspection companion to `domain_shift_probe.py`.
It reads cached CLIP frame embeddings from:

- `clip_frame_features.npy`
- `clip_frame_metadata.csv`

and creates several embedding visualizations.

Outputs are organized by projection and domain scope, for example:
- `embedding_plots/tsne_2d/all_domains/...`
- `embedding_plots/tsne_2d/sa_only/...`
- `embedding_plots/umap_3d/benin_only/...`

Plot families
-------------
There are four plot families in this script:

1. Overall domain/TB plot
   - Output stem example: `clip_frame_tsne_2d_all_domains_overall`
   - Uses all anatomical sites together.
   - Marker shape encodes domain:
     - Benin -> circle
     - SA -> triangle
   - Point color encodes TB label:
     - TB negative -> green
     - TB positive -> red

2. Site-filtered domain/TB plots
   - Output stems like:
     - `clip_frame_tsne_2d_sa_only_laterality_right`
     - `clip_frame_umap_3d_all_domains_region_anterior`
     - `clip_frame_tsne_2d_benin_only_site_code_qasd`
   - These still use domain marker + TB color.
   - The difference is that rows are filtered to one anatomical subset first.

3. Overall site-legend plot
   - Output stem like:
     - `clip_frame_tsne_2d_all_domains_site_legend_site_code`
   - Uses all rows together, but now color/legend show anatomical site or site
     group rather than domain/TB.

4. Overall site+domain overlay plot
   - Output stem like:
     - `clip_frame_tsne_2d_all_domains_site_domain_overlay_region`
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

- `--domain-filter`
  Restrict outputs to `all`, `benin`, or `sa`.
  This scope is reflected in both folder names and filenames.

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

- `--embedding-methods`
  Which projections to create. Default keeps the original behavior: `tsne`.
  Add `umap` to also write UMAP outputs.

- `--cluster-method`
  Optional clustering on the final 2D/3D coordinates.
  Use `kmeans` or `dbscan`.

Typical commands
----------------
python domain_shift_clip_tsne.py \
  --feature-dir /users/lxflk/ULTR-AI-Vid/checkpoints/domain_shift_probe_features/sa_finetuned_full_test \
  --domain-filter sa \
  --site-plot-mode none \
  --site-legend-mode none \
  --site-domain-overlay-mode none
  
python domain_shift_clip_tsne.py \
  --feature-dir /users/lxflk/ULTR-AI-Vid/checkpoints/domain_shift_probe_features/sa_finetuned_full_test \
  --domain-filter sa \
  --site-plot-mode none \
  --site-legend-mode none \
  --site-domain-overlay-mode site_code
  
python domain_shift_clip_tsne.py \
  --feature-dir /users/lxflk/ULTR-AI-Vid/checkpoints/domain_shift_probe_features/sa_finetuned_full_test \
  --domain-filter sa \
  --site-plot-mode none \
  --site-legend-mode none \
  --site-domain-overlay-mode region
  
python domain_shift_clip_tsne.py \
  --feature-dir /users/lxflk/ULTR-AI-Vid/checkpoints/domain_shift_probe_features/sa_finetuned_full_test \
  --site-plot-mode none \
  --site-legend-mode none \
  --site-domain-overlay-mode region \
  --embedding-methods tsne umap \
  --umap-dims 2 3 \
  --cluster-method kmeans \
  --num-clusters 4
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import umap
except ImportError:
    umap = None

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for domain_shift_clip_tsne.py. "
        "Install scikit-learn in the environment used to run this script."
    ) from exc

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
PLOT_MIN_SAMPLES = 12

DOMAIN_LABELS = {"benin": "Benin", "sa": "SA"}
DOMAIN_MARKERS = {"benin": "o", "sa": "^"}
DOMAIN_ORDER = ["benin", "sa"]
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
            "Create t-SNE and optional UMAP plots from cached CLIP frame embeddings. "
            "Supports overall domain/TB plots, filtered domain/TB plots, "
            "overall anatomy-legend plots, overall anatomy-color + domain-marker plots, "
            "single-domain filtering, and optional clustering on the final coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  SA-only overall domain/TB plot:\n"
            "    python domain_shift_clip_tsne.py --run-name sa_finetuned_full_test --domain-filter sa --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode none\n\n"
            "  SA-only region overlay:\n"
            "    python domain_shift_clip_tsne.py --run-name sa_finetuned_full_test --domain-filter sa --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode region\n\n"
            "  2D t-SNE + 2D/3D UMAP with KMeans clustering:\n"
            "    python domain_shift_clip_tsne.py --run-name sa_finetuned_full_test --site-plot-mode none --site-legend-mode none --site-domain-overlay-mode region --embedding-methods tsne umap --umap-dims 2 3 --cluster-method kmeans --num-clusters 4"
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
        help=(
            "Root directory where plots, sampled coordinates, and clustering results will be written. "
            "If omitted, defaults to results/<run>/embedding_plots/ when available."
        ),
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
        "--domain-filter",
        type=str,
        default="all",
        choices=["all", "benin", "sa"],
        help="Restrict all plots to a single domain or keep all domains together",
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
        "--embedding-methods",
        nargs="+",
        default=["tsne"],
        choices=["tsne", "umap"],
        help="Embedding methods to run. Default keeps the existing behavior: t-SNE only.",
    )
    parser.add_argument(
        "--tsne-dims",
        nargs="+",
        type=int,
        default=[2],
        choices=[2, 3],
        help="Output dimensionalities for t-SNE runs",
    )
    parser.add_argument(
        "--umap-dims",
        nargs="+",
        type=int,
        default=[2, 3],
        choices=[2, 3],
        help="Output dimensionalities for UMAP runs",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="Requested UMAP number of neighbors; automatically reduced when needed",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter",
    )
    parser.add_argument(
        "--umap-metric",
        type=str,
        default="euclidean",
        help="UMAP distance metric",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="none",
        choices=["none", "kmeans", "dbscan"],
        help="Optional clustering to run on the final low-dimensional coordinates",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=4,
        help="Cluster count for KMeans",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.75,
        help="DBSCAN eps value when --cluster-method dbscan",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples value when --cluster-method dbscan",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=22.0,
        help="Elevation angle for saved 3D plots",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=38.0,
        help="Azimuth angle for saved 3D plots",
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
            output_dir = results_dir / "embedding_plots"
        else:
            output_dir = feature_dir / "embedding_plots"

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


def effective_umap_neighbors(requested: int, n_samples: int) -> int:
    if n_samples <= 2:
        raise ValueError("Need at least 3 samples for UMAP")
    return max(2, min(int(requested), n_samples - 1))


def projection_display_name(method: str) -> str:
    return "t-SNE" if method == "tsne" else "UMAP"


def projection_axis_label(method: str, axis_index: int) -> str:
    return "{0} {1}".format(projection_display_name(method), axis_index)


def domain_scope_slug(domain_filter: str) -> str:
    if domain_filter == "all":
        return "all_domains"
    return "{0}_only".format(domain_filter)


def domain_scope_title_suffix(domain_filter: str) -> str:
    if domain_filter == "all":
        return ""
    return " ({0} only)".format(DOMAIN_LABELS[domain_filter])


def domains_in_rows(rows: Sequence[Dict[str, object]]) -> List[str]:
    available = {str(row.get("domain", "")).strip().lower() for row in rows}
    ordered = [domain_name for domain_name in DOMAIN_ORDER if domain_name in available]
    extras = sorted(available.difference(DOMAIN_ORDER))
    return ordered + extras


def filter_rows_by_domain(
    rows: Sequence[Dict[str, object]],
    domain_filter: str,
) -> List[Dict[str, object]]:
    if domain_filter == "all":
        return list(rows)
    return [row for row in rows if str(row.get("domain", "")).strip().lower() == domain_filter]


def unique_ints(values: Sequence[int]) -> List[int]:
    ordered: List[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in ordered:
            ordered.append(int_value)
    return ordered


def projection_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, int]]:
    specs: List[Tuple[str, int]] = []
    if "tsne" in args.embedding_methods:
        specs.extend(("tsne", n_components) for n_components in unique_ints(args.tsne_dims))
    if "umap" in args.embedding_methods:
        specs.extend(("umap", n_components) for n_components in unique_ints(args.umap_dims))
    return specs


def projection_seed_offset(method: str, n_components: int) -> int:
    if method == "tsne":
        return n_components * 1000
    return 5000 + n_components * 1000


def resolve_projection_output_dirs(
    output_root: Path,
    method: str,
    n_components: int,
    domain_filter: str,
) -> Dict[str, Path]:
    run_dir = output_root / "{0}_{1}d".format(method, n_components) / domain_scope_slug(domain_filter)
    figures_dir = run_dir / "figures"
    points_dir = run_dir / "points"
    clusters_dir = run_dir / "clusters"
    for directory in [run_dir, figures_dir, points_dir, clusters_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "figures_dir": figures_dir,
        "points_dir": points_dir,
        "clusters_dir": clusters_dir,
    }


def build_plot_stem(
    method: str,
    n_components: int,
    domain_filter: str,
    suffix: str,
) -> str:
    return "clip_frame_{0}_{1}d_{2}_{3}".format(
        sanitize_slug(method),
        int(n_components),
        domain_scope_slug(domain_filter),
        sanitize_slug(suffix),
    )


def compute_embedding(
    features: np.ndarray,
    method: str,
    n_components: int,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if method == "tsne":
        perplexity = effective_perplexity(args.perplexity, features.shape[0])
        tsne_kwargs = {
            "n_components": n_components,
            "init": "pca",
            "perplexity": perplexity,
            "learning_rate": args.learning_rate,
            "random_state": seed,
        }
        if "max_iter" in inspect.signature(TSNE.__init__).parameters:
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000
        coords = TSNE(**tsne_kwargs).fit_transform(features)
        return coords, {
            "embedding_method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "learning_rate": args.learning_rate,
        }

    if method == "umap":
        if umap is None:
            raise ImportError(
                "UMAP support requires the 'umap-learn' package. "
                "Install umap-learn in the environment used to run this script."
            )
        n_neighbors = effective_umap_neighbors(args.umap_n_neighbors, features.shape[0])
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=seed,
        )
        coords = reducer.fit_transform(features)
        return coords, {
            "embedding_method": method,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
        }

    raise ValueError("Unsupported embedding method: {0}".format(method))


def compute_clusters(
    coords: np.ndarray,
    method: str,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if method == "kmeans":
        if coords.shape[0] < args.num_clusters:
            raise ValueError(
                "KMeans requested {0} clusters but only {1} points are available".format(
                    args.num_clusters,
                    coords.shape[0],
                )
            )
        clusterer = KMeans(n_clusters=args.num_clusters, random_state=seed, n_init=10)
        labels = clusterer.fit_predict(coords)
        return labels, {
            "cluster_method": method,
            "requested_num_clusters": args.num_clusters,
            "n_clusters_found": int(len(set(int(label) for label in labels.tolist()))),
        }

    if method == "dbscan":
        clusterer = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
        labels = clusterer.fit_predict(coords)
        non_noise_clusters = {int(label) for label in labels.tolist() if int(label) >= 0}
        return labels, {
            "cluster_method": method,
            "eps": args.dbscan_eps,
            "min_samples": args.dbscan_min_samples,
            "n_clusters_found": int(len(non_noise_clusters)),
            "noise_points": int(np.sum(labels == -1)),
        }

    raise ValueError("Unsupported cluster method: {0}".format(method))


def cluster_display_label(cluster_id: int) -> str:
    if int(cluster_id) < 0:
        return "Noise"
    return "Cluster {0}".format(int(cluster_id))


def build_cluster_colors(cluster_ids: Sequence[int]) -> Dict[int, object]:
    ordered_clusters = [int(cluster_id) for cluster_id in cluster_ids if int(cluster_id) >= 0]
    colors: Dict[int, object] = {}
    if ordered_clusters:
        cmap = plt.get_cmap("tab20", len(ordered_clusters))
        for index, cluster_id in enumerate(ordered_clusters):
            colors[int(cluster_id)] = cmap(index)
    for cluster_id in cluster_ids:
        if int(cluster_id) < 0:
            colors[int(cluster_id)] = "#9E9E9E"
    return colors


def cluster_counts_text(cluster_labels: Sequence[int]) -> str:
    counts = Counter(int(label) for label in cluster_labels)
    cluster_ids = sorted(counts, key=lambda label: (label < 0, label))
    return "\n".join(
        "{0}: {1}".format(cluster_display_label(cluster_id), int(counts[cluster_id]))
        for cluster_id in cluster_ids
    )


def plot_counts_text(rows: Sequence[Dict[str, object]]) -> str:
    counts = Counter((str(row["domain"]), int(row["tb_label"])) for row in rows)
    lines = []
    for domain_name in domains_in_rows(rows):
        if domain_name not in DOMAIN_LABELS:
            continue
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


def save_points_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    coords: np.ndarray,
    coord_prefix: str,
    extra_columns: Optional[Dict[str, Sequence[object]]] = None,
) -> None:
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
    ]
    coord_fieldnames = ["{0}_{1}".format(coord_prefix, axis_index + 1) for axis_index in range(coords.shape[1])]
    fieldnames.extend(coord_fieldnames)
    extra_columns = extra_columns or {}
    fieldnames.extend(list(extra_columns.keys()))

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (row, coord) in enumerate(zip(rows, coords)):
            record = {
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
            }
            for axis_index, fieldname in enumerate(coord_fieldnames):
                record[fieldname] = float(coord[axis_index])
            for column_name, values in extra_columns.items():
                record[column_name] = values[row_index]
            writer.writerow(record)


def save_cluster_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    coords: np.ndarray,
    coord_prefix: str,
    cluster_labels: Sequence[int],
) -> None:
    save_points_csv(
        path,
        rows,
        coords,
        coord_prefix=coord_prefix,
        extra_columns={"cluster_id": [int(label) for label in cluster_labels]},
    )


def create_projection_axes(n_components: int, figsize_2d: Tuple[float, float], figsize_3d: Tuple[float, float]):
    if n_components == 3:
        fig = plt.figure(figsize=figsize_3d)
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax
    fig, ax = plt.subplots(figsize=figsize_2d)
    return fig, ax


def configure_projection_axis(
    ax,
    method: str,
    n_components: int,
    title: str,
    view_elev: float,
    view_azim: float,
) -> None:
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(projection_axis_label(method, 1))
    ax.set_ylabel(projection_axis_label(method, 2))
    if n_components == 3:
        ax.set_zlabel(projection_axis_label(method, 3))
        ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(alpha=0.18, linewidth=0.5)


def scatter_projected_points(
    ax,
    coords: np.ndarray,
    mask: np.ndarray,
    n_components: int,
    point_size: float,
    alpha: float,
    color,
    marker: str,
    linewidths: float,
) -> None:
    scatter_kwargs = {
        "s": point_size,
        "alpha": alpha,
        "color": color,
        "marker": marker,
        "edgecolors": "white",
        "linewidths": linewidths,
    }
    if n_components == 3:
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], **scatter_kwargs)
        return
    ax.scatter(coords[mask, 0], coords[mask, 1], **scatter_kwargs)


def annotate_projection_text(ax, n_components: int, text: str) -> None:
    text_kwargs = {
        "fontsize": 9,
        "verticalalignment": "bottom",
        "bbox": {"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    }
    if n_components == 3:
        ax.text2D(0.015, 0.015, text, transform=ax.transAxes, **text_kwargs)
        return
    ax.text(0.015, 0.015, text, transform=ax.transAxes, **text_kwargs)


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
    embedding_method: str,
    n_components: int,
    point_size: float,
    alpha: float,
    dpi: int,
    view_elev: float,
    view_azim: float,
) -> None:
    fig, ax = create_projection_axes(
        n_components,
        figsize_2d=(8.8, 7.4),
        figsize_3d=(9.8, 8.2),
    )

    for domain_name in domains_in_rows(rows):
        if domain_name not in DOMAIN_LABELS:
            continue
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
            scatter_projected_points(
                ax,
                coords,
                mask,
                n_components=n_components,
                point_size=point_size,
                alpha=alpha,
                color=TB_COLORS[tb_label],
                marker=DOMAIN_MARKERS[domain_name],
                linewidths=0.35,
            )

    configure_projection_axis(
        ax,
        method=embedding_method,
        n_components=n_components,
        title=title,
        view_elev=view_elev,
        view_azim=view_azim,
    )

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
        for domain_name in domains_in_rows(rows)
        if domain_name in DOMAIN_LABELS
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

    annotate_projection_text(ax, n_components, plot_counts_text(rows))

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
    embedding_method: str,
    n_components: int,
    point_size: float,
    alpha: float,
    dpi: int,
    view_elev: float,
    view_azim: float,
) -> None:
    fig, ax = create_projection_axes(
        n_components,
        figsize_2d=(10.6, 7.4),
        figsize_3d=(11.4, 8.5),
    )
    colors = build_site_group_colors(group_names)

    handles = []
    for group_name in group_names:
        mask = np.array([site_group_value(row, group_by) == group_name for row in rows], dtype=bool)
        if not np.any(mask):
            continue
        color = colors[group_name]
        scatter_projected_points(
            ax,
            coords,
            mask,
            n_components=n_components,
            point_size=point_size,
            alpha=alpha,
            color=color,
            marker="o",
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

    configure_projection_axis(
        ax,
        method=embedding_method,
        n_components=n_components,
        title=title,
        view_elev=view_elev,
        view_azim=view_azim,
    )
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
    embedding_method: str,
    n_components: int,
    point_size: float,
    alpha: float,
    dpi: int,
    view_elev: float,
    view_azim: float,
) -> None:
    fig, ax = create_projection_axes(
        n_components,
        figsize_2d=(10.8, 7.5),
        figsize_3d=(11.6, 8.6),
    )
    colors = build_site_group_colors(group_names)

    for group_name in group_names:
        color = colors[group_name]
        for domain_name in domains_in_rows(rows):
            if domain_name not in DOMAIN_LABELS:
                continue
            mask = np.array(
                [
                    (site_group_value(row, group_by) == group_name) and (str(row["domain"]) == domain_name)
                    for row in rows
                ],
                dtype=bool,
            )
            if not np.any(mask):
                continue
            scatter_projected_points(
                ax,
                coords,
                mask,
                n_components=n_components,
                point_size=point_size,
                alpha=alpha,
                color=color,
                marker=DOMAIN_MARKERS[domain_name],
                linewidths=0.30,
            )

    configure_projection_axis(
        ax,
        method=embedding_method,
        n_components=n_components,
        title=title,
        view_elev=view_elev,
        view_azim=view_azim,
    )

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
        for domain_name in domains_in_rows(rows)
        if domain_name in DOMAIN_LABELS
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


def render_cluster_plot(
    coords: np.ndarray,
    rows: Sequence[Dict[str, object]],
    cluster_labels: Sequence[int],
    title: str,
    output_path: Path,
    embedding_method: str,
    n_components: int,
    point_size: float,
    alpha: float,
    dpi: int,
    view_elev: float,
    view_azim: float,
) -> None:
    fig, ax = create_projection_axes(
        n_components,
        figsize_2d=(10.4, 7.4),
        figsize_3d=(11.3, 8.4),
    )
    cluster_ids = sorted({int(label) for label in cluster_labels}, key=lambda label: (label < 0, label))
    colors = build_cluster_colors(cluster_ids)

    for cluster_id in cluster_ids:
        for domain_name in domains_in_rows(rows):
            if domain_name not in DOMAIN_LABELS:
                continue
            mask = np.array(
                [
                    (int(cluster_label) == int(cluster_id)) and (str(row["domain"]) == domain_name)
                    for row, cluster_label in zip(rows, cluster_labels)
                ],
                dtype=bool,
            )
            if not np.any(mask):
                continue
            scatter_projected_points(
                ax,
                coords,
                mask,
                n_components=n_components,
                point_size=point_size,
                alpha=alpha,
                color=colors[int(cluster_id)],
                marker=DOMAIN_MARKERS[domain_name],
                linewidths=0.30,
            )

    configure_projection_axis(
        ax,
        method=embedding_method,
        n_components=n_components,
        title=title,
        view_elev=view_elev,
        view_azim=view_azim,
    )

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
        for domain_name in domains_in_rows(rows)
        if domain_name in DOMAIN_LABELS
    ]
    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[int(cluster_id)],
            markerfacecolor=colors[int(cluster_id)],
            markeredgecolor=colors[int(cluster_id)],
            markersize=7,
            linestyle="None",
            label=cluster_display_label(int(cluster_id)),
        )
        for cluster_id in cluster_ids
    ]

    legend_domain = ax.legend(handles=domain_handles, title="Domain", loc="upper right", frameon=True)
    ax.add_artist(legend_domain)
    ax.legend(
        handles=cluster_handles,
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
    )
    annotate_projection_text(ax, n_components, cluster_counts_text(cluster_labels))

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
    output_root = resolve_output_dir(args, feature_dir)
    manifest_path = resolve_manifest_path(args, feature_dir)
    manifest = load_json(manifest_path) if manifest_path is not None else None

    LOGGER.info("Feature directory: %s", feature_dir)
    LOGGER.info("Output root: %s", output_root)
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
    scoped_rows = filter_rows_by_domain(enriched_rows, args.domain_filter)
    if not scoped_rows:
        raise ValueError("No rows remain after applying --domain-filter {0}".format(args.domain_filter))

    LOGGER.info(
        "Loaded %d CLIP frame rows (%d unknown-site rows, %d missing TB labels before filtering)",
        len(rows),
        enrich_stats.get("unknown_site_rows", 0),
        enrich_stats.get("tb_labels_missing", 0),
    )
    LOGGER.info("Using %d valid rows after metadata enrichment", len(enriched_rows))
    LOGGER.info(
        "Using %d rows after domain filtering (%s)",
        len(scoped_rows),
        domain_scope_slug(args.domain_filter),
    )

    plot_specs = build_plot_specs(
        scoped_rows,
        args.site_plot_mode,
        args.site_groups,
        args.include_unknown_site_groups,
    )
    projection_specs = projection_specs_from_args(args)
    if not projection_specs:
        raise ValueError("No projections selected. Adjust --embedding-methods/--tsne-dims/--umap-dims.")

    summary = {
        "feature_dir": str(feature_dir),
        "output_root": str(output_root),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "domain_filter": args.domain_filter,
        "domain_scope": domain_scope_slug(args.domain_filter),
        "projection_specs": [
            {"embedding_method": method, "n_components": n_components}
            for method, n_components in projection_specs
        ],
        "cluster_method": args.cluster_method,
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

    for projection_index, (embedding_method, n_components) in enumerate(projection_specs):
        coord_prefix = sanitize_slug(embedding_method)
        method_display = projection_display_name(embedding_method)
        projection_dirs = resolve_projection_output_dirs(
            output_root,
            embedding_method,
            n_components,
            args.domain_filter,
        )
        LOGGER.info(
            "Generating %s %dD plots in %s",
            method_display,
            n_components,
            projection_dirs["run_dir"],
        )

        # Domain/TB plots: one overall plus optional filtered subsets.
        for plot_index, (plot_mode, plot_title, group_name) in enumerate(plot_specs):
            plot_rows = filter_rows_for_plot(scoped_rows, plot_mode, group_name)
            if len(plot_rows) < PLOT_MIN_SAMPLES:
                LOGGER.warning("Skipping %s: only %d eligible rows", plot_title, len(plot_rows))
                continue

            domains_present = sorted({str(row["domain"]) for row in plot_rows})
            tb_present = sorted({int(row["tb_label"]) for row in plot_rows})
            if args.domain_filter == "all" and len(domains_present) < 2:
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
            if sampled_positions.size < PLOT_MIN_SAMPLES:
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
            embedding_seed = args.random_state + projection_seed_offset(embedding_method, n_components) + plot_index
            coords, embedding_meta = compute_embedding(
                processed_features,
                method=embedding_method,
                n_components=n_components,
                args=args,
                seed=embedding_seed,
            )

            suffix = "overall" if group_name is None else "{0}_{1}".format(plot_mode, group_name)
            plot_stem = build_plot_stem(embedding_method, n_components, args.domain_filter, suffix)
            image_path = projection_dirs["figures_dir"] / "{0}.png".format(plot_stem)
            csv_path = projection_dirs["points_dir"] / "{0}_points.csv".format(plot_stem)
            full_title = "{0}{1}\nCLIP frame embeddings ({2}, {3}D)".format(
                plot_title,
                domain_scope_title_suffix(args.domain_filter),
                method_display,
                n_components,
            )

            render_domain_tb_plot(
                coords,
                sampled_rows,
                title=full_title,
                output_path=image_path,
                embedding_method=embedding_method,
                n_components=n_components,
                point_size=args.point_size,
                alpha=args.alpha,
                dpi=args.dpi,
                view_elev=args.view_elev,
                view_azim=args.view_azim,
            )
            save_points_csv(csv_path, sampled_rows, coords, coord_prefix=coord_prefix)

            counts = Counter((str(row["domain"]), int(row["tb_label"])) for row in sampled_rows)
            LOGGER.info(
                "Saved %s (%d points, %s %dD) to %s",
                plot_title,
                len(sampled_rows),
                method_display,
                n_components,
                image_path,
            )

            plot_record = {
                "plot_kind": "domain_tb",
                "plot_mode": plot_mode,
                "group_name": group_name,
                "title": full_title,
                "domain_filter": args.domain_filter,
                "domain_scope": domain_scope_slug(args.domain_filter),
                "embedding_method": embedding_method,
                "n_components": n_components,
                "n_points": len(sampled_rows),
                "image_path": str(image_path),
                "points_csv": str(csv_path),
                "output_dir": str(projection_dirs["run_dir"]),
                "counts": {
                    "{0}|tb_{1}".format(domain_name, tb_label): int(counts.get((domain_name, tb_label), 0))
                    for domain_name in domains_in_rows(sampled_rows)
                    if domain_name in DOMAIN_LABELS
                    for tb_label in [0, 1]
                },
            }
            plot_record.update(embedding_meta)

            if args.cluster_method != "none":
                try:
                    cluster_labels, cluster_meta = compute_clusters(
                        coords,
                        method=args.cluster_method,
                        args=args,
                        seed=embedding_seed + 100000,
                    )
                except ValueError as exc:
                    LOGGER.warning("Skipping clustering for %s: %s", plot_title, exc)
                else:
                    cluster_path = projection_dirs["clusters_dir"] / "{0}_{1}_clusters.png".format(
                        plot_stem,
                        sanitize_slug(args.cluster_method),
                    )
                    cluster_csv_path = projection_dirs["clusters_dir"] / "{0}_{1}_clusters.csv".format(
                        plot_stem,
                        sanitize_slug(args.cluster_method),
                    )
                    render_cluster_plot(
                        coords,
                        sampled_rows,
                        cluster_labels=cluster_labels,
                        title="{0}\nClusters ({1})".format(full_title, args.cluster_method.upper()),
                        output_path=cluster_path,
                        embedding_method=embedding_method,
                        n_components=n_components,
                        point_size=args.point_size,
                        alpha=args.alpha,
                        dpi=args.dpi,
                        view_elev=args.view_elev,
                        view_azim=args.view_azim,
                    )
                    save_cluster_csv(
                        cluster_csv_path,
                        sampled_rows,
                        coords,
                        coord_prefix=coord_prefix,
                        cluster_labels=cluster_labels,
                    )
                    cluster_counts = Counter(int(label) for label in cluster_labels)
                    plot_record["cluster"] = {
                        **cluster_meta,
                        "image_path": str(cluster_path),
                        "cluster_csv": str(cluster_csv_path),
                        "counts": {
                            ("noise" if cluster_id < 0 else "cluster_{0}".format(cluster_id)): int(
                                cluster_counts[cluster_id]
                            )
                            for cluster_id in sorted(cluster_counts)
                        },
                    }

            summary["plots"].append(plot_record)

        # Overall anatomy legend plot.
        if args.site_legend_mode != "none":
            site_legend_groups = group_names_for_mode(
                scoped_rows,
                args.site_legend_mode,
                args.site_legend_groups,
                args.include_unknown_site_groups,
            )
            site_legend_group_set = set(site_legend_groups)
            site_legend_rows = [
                row for row in scoped_rows if site_group_value(row, args.site_legend_mode) in site_legend_group_set
            ]

            if len(site_legend_rows) < PLOT_MIN_SAMPLES:
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
                if sampled_positions.size < PLOT_MIN_SAMPLES:
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
                    embedding_seed = args.random_state + projection_seed_offset(embedding_method, n_components) + 100
                    coords, embedding_meta = compute_embedding(
                        processed_features,
                        method=embedding_method,
                        n_components=n_components,
                        args=args,
                        seed=embedding_seed,
                    )

                    plot_stem = build_plot_stem(
                        embedding_method,
                        n_components,
                        args.domain_filter,
                        "site_legend_{0}".format(args.site_legend_mode),
                    )
                    image_path = projection_dirs["figures_dir"] / "{0}.png".format(plot_stem)
                    csv_path = projection_dirs["points_dir"] / "{0}_points.csv".format(plot_stem)
                    full_title = "All sites by {0}{1}\nCLIP frame embeddings ({2}, {3}D)".format(
                        "anatomical site" if args.site_legend_mode == "site_code" else args.site_legend_mode,
                        domain_scope_title_suffix(args.domain_filter),
                        method_display,
                        n_components,
                    )

                    render_site_legend_plot(
                        coords,
                        sampled_rows,
                        group_by=args.site_legend_mode,
                        group_names=site_legend_groups,
                        title=full_title,
                        output_path=image_path,
                        embedding_method=embedding_method,
                        n_components=n_components,
                        point_size=args.point_size,
                        alpha=args.alpha,
                        dpi=args.dpi,
                        view_elev=args.view_elev,
                        view_azim=args.view_azim,
                    )
                    save_points_csv(csv_path, sampled_rows, coords, coord_prefix=coord_prefix)

                    group_counts = Counter(site_group_value(row, args.site_legend_mode) for row in sampled_rows)
                    LOGGER.info(
                        "Saved site-legend plot for %s (%d points, %s %dD) to %s",
                        args.site_legend_mode,
                        len(sampled_rows),
                        method_display,
                        n_components,
                        image_path,
                    )

                    plot_record = {
                        "plot_kind": "site_legend",
                        "plot_mode": args.site_legend_mode,
                        "group_name": None,
                        "title": full_title,
                        "domain_filter": args.domain_filter,
                        "domain_scope": domain_scope_slug(args.domain_filter),
                        "embedding_method": embedding_method,
                        "n_components": n_components,
                        "n_points": len(sampled_rows),
                        "image_path": str(image_path),
                        "points_csv": str(csv_path),
                        "output_dir": str(projection_dirs["run_dir"]),
                        "counts": {
                            group_name: int(group_counts.get(group_name, 0))
                            for group_name in site_legend_groups
                        },
                    }
                    plot_record.update(embedding_meta)

                    if args.cluster_method != "none":
                        try:
                            cluster_labels, cluster_meta = compute_clusters(
                                coords,
                                method=args.cluster_method,
                                args=args,
                                seed=embedding_seed + 100000,
                            )
                        except ValueError as exc:
                            LOGGER.warning("Skipping clustering for site-legend plot %s: %s", args.site_legend_mode, exc)
                        else:
                            cluster_path = projection_dirs["clusters_dir"] / "{0}_{1}_clusters.png".format(
                                plot_stem,
                                sanitize_slug(args.cluster_method),
                            )
                            cluster_csv_path = projection_dirs["clusters_dir"] / "{0}_{1}_clusters.csv".format(
                                plot_stem,
                                sanitize_slug(args.cluster_method),
                            )
                            render_cluster_plot(
                                coords,
                                sampled_rows,
                                cluster_labels=cluster_labels,
                                title="{0}\nClusters ({1})".format(full_title, args.cluster_method.upper()),
                                output_path=cluster_path,
                                embedding_method=embedding_method,
                                n_components=n_components,
                                point_size=args.point_size,
                                alpha=args.alpha,
                                dpi=args.dpi,
                                view_elev=args.view_elev,
                                view_azim=args.view_azim,
                            )
                            save_cluster_csv(
                                cluster_csv_path,
                                sampled_rows,
                                coords,
                                coord_prefix=coord_prefix,
                                cluster_labels=cluster_labels,
                            )
                            cluster_counts = Counter(int(label) for label in cluster_labels)
                            plot_record["cluster"] = {
                                **cluster_meta,
                                "image_path": str(cluster_path),
                                "cluster_csv": str(cluster_csv_path),
                                "counts": {
                                    ("noise" if cluster_id < 0 else "cluster_{0}".format(cluster_id)): int(
                                        cluster_counts[cluster_id]
                                    )
                                    for cluster_id in sorted(cluster_counts)
                                },
                            }

                    summary["plots"].append(plot_record)

        # Overall anatomy-color + domain-marker overlay plot.
        if args.site_domain_overlay_mode != "none":
            overlay_groups = group_names_for_mode(
                scoped_rows,
                args.site_domain_overlay_mode,
                args.site_domain_overlay_groups,
                args.include_unknown_site_groups,
            )
            overlay_group_set = set(overlay_groups)
            overlay_rows = [
                row for row in scoped_rows if site_group_value(row, args.site_domain_overlay_mode) in overlay_group_set
            ]

            if len(overlay_rows) < PLOT_MIN_SAMPLES:
                LOGGER.warning(
                    "Skipping site-domain overlay plot for %s: only %d eligible rows",
                    args.site_domain_overlay_mode,
                    len(overlay_rows),
                )
            else:
                overlay_domains_present = sorted({str(row["domain"]) for row in overlay_rows})
                if args.domain_filter == "all" and len(overlay_domains_present) < 2:
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
                if sampled_positions.size < PLOT_MIN_SAMPLES:
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
                    embedding_seed = args.random_state + projection_seed_offset(embedding_method, n_components) + 200
                    coords, embedding_meta = compute_embedding(
                        processed_features,
                        method=embedding_method,
                        n_components=n_components,
                        args=args,
                        seed=embedding_seed,
                    )

                    plot_stem = build_plot_stem(
                        embedding_method,
                        n_components,
                        args.domain_filter,
                        "site_domain_overlay_{0}".format(args.site_domain_overlay_mode),
                    )
                    image_path = projection_dirs["figures_dir"] / "{0}.png".format(plot_stem)
                    csv_path = projection_dirs["points_dir"] / "{0}_points.csv".format(plot_stem)
                    full_title = "All sites by {0} with domain markers{1}\nCLIP frame embeddings ({2}, {3}D)".format(
                        "anatomical site"
                        if args.site_domain_overlay_mode == "site_code"
                        else args.site_domain_overlay_mode,
                        domain_scope_title_suffix(args.domain_filter),
                        method_display,
                        n_components,
                    )

                    render_site_domain_overlay_plot(
                        coords,
                        sampled_rows,
                        group_by=args.site_domain_overlay_mode,
                        group_names=overlay_groups,
                        title=full_title,
                        output_path=image_path,
                        embedding_method=embedding_method,
                        n_components=n_components,
                        point_size=args.point_size,
                        alpha=args.alpha,
                        dpi=args.dpi,
                        view_elev=args.view_elev,
                        view_azim=args.view_azim,
                    )
                    save_points_csv(csv_path, sampled_rows, coords, coord_prefix=coord_prefix)

                    overlay_counts = Counter(
                        (site_group_value(row, args.site_domain_overlay_mode), str(row["domain"]))
                        for row in sampled_rows
                    )
                    LOGGER.info(
                        "Saved site-domain overlay plot for %s (%d points, %s %dD) to %s",
                        args.site_domain_overlay_mode,
                        len(sampled_rows),
                        method_display,
                        n_components,
                        image_path,
                    )

                    plot_record = {
                        "plot_kind": "site_domain_overlay",
                        "plot_mode": args.site_domain_overlay_mode,
                        "group_name": None,
                        "title": full_title,
                        "domain_filter": args.domain_filter,
                        "domain_scope": domain_scope_slug(args.domain_filter),
                        "embedding_method": embedding_method,
                        "n_components": n_components,
                        "n_points": len(sampled_rows),
                        "image_path": str(image_path),
                        "points_csv": str(csv_path),
                        "output_dir": str(projection_dirs["run_dir"]),
                        "counts": {
                            "{0}|{1}".format(group_name, domain_name): int(
                                overlay_counts.get((group_name, domain_name), 0)
                            )
                            for group_name in overlay_groups
                            for domain_name in domains_in_rows(sampled_rows)
                            if domain_name in DOMAIN_LABELS
                        },
                    }
                    plot_record.update(embedding_meta)

                    if args.cluster_method != "none":
                        try:
                            cluster_labels, cluster_meta = compute_clusters(
                                coords,
                                method=args.cluster_method,
                                args=args,
                                seed=embedding_seed + 100000,
                            )
                        except ValueError as exc:
                            LOGGER.warning(
                                "Skipping clustering for site-domain overlay plot %s: %s",
                                args.site_domain_overlay_mode,
                                exc,
                            )
                        else:
                            cluster_path = projection_dirs["clusters_dir"] / "{0}_{1}_clusters.png".format(
                                plot_stem,
                                sanitize_slug(args.cluster_method),
                            )
                            cluster_csv_path = projection_dirs["clusters_dir"] / "{0}_{1}_clusters.csv".format(
                                plot_stem,
                                sanitize_slug(args.cluster_method),
                            )
                            render_cluster_plot(
                                coords,
                                sampled_rows,
                                cluster_labels=cluster_labels,
                                title="{0}\nClusters ({1})".format(full_title, args.cluster_method.upper()),
                                output_path=cluster_path,
                                embedding_method=embedding_method,
                                n_components=n_components,
                                point_size=args.point_size,
                                alpha=args.alpha,
                                dpi=args.dpi,
                                view_elev=args.view_elev,
                                view_azim=args.view_azim,
                            )
                            save_cluster_csv(
                                cluster_csv_path,
                                sampled_rows,
                                coords,
                                coord_prefix=coord_prefix,
                                cluster_labels=cluster_labels,
                            )
                            cluster_counts = Counter(int(label) for label in cluster_labels)
                            plot_record["cluster"] = {
                                **cluster_meta,
                                "image_path": str(cluster_path),
                                "cluster_csv": str(cluster_csv_path),
                                "counts": {
                                    ("noise" if cluster_id < 0 else "cluster_{0}".format(cluster_id)): int(
                                        cluster_counts[cluster_id]
                                    )
                                    for cluster_id in sorted(cluster_counts)
                                },
                            }

                    summary["plots"].append(plot_record)

    summary_suffix_parts = [
        domain_scope_slug(args.domain_filter),
        sanitize_slug(
            "_".join("{0}_{1}d".format(method, n_components) for method, n_components in projection_specs)
        ),
    ]
    if args.cluster_method != "none":
        summary_suffix_parts.append(sanitize_slug(args.cluster_method))
    summary_path = output_root / "clip_frame_embedding_summary_{0}.json".format(
        "_".join(summary_suffix_parts)
    )
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
