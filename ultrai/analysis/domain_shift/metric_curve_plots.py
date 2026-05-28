#!/usr/bin/env python3
"""Fold-averaged ROC and precision-recall plots for domain-shift reports."""

import html
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        compute_roc,
        load_labels_scores,
        svg_polyline,
    )
except ImportError:  # pragma: no cover - convenience for direct script use
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        compute_roc,
        load_labels_scores,
        svg_polyline,
    )


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


class FoldMetricCurves:
    def __init__(
        self,
        fold: int,
        prediction_csv: Path,
        roc_points: List[Tuple[float, float]],
        auroc: float,
        pr_points: List[Tuple[float, float]],
        auprc: float,
        n_samples: int,
        n_positive: int,
        n_negative: int,
    ) -> None:
        self.fold = fold
        self.prediction_csv = prediction_csv
        self.roc_points = roc_points
        self.auroc = auroc
        self.pr_points = pr_points
        self.auprc = auprc
        self.n_samples = n_samples
        self.n_positive = n_positive
        self.n_negative = n_negative


class MeanMetricCurves:
    def __init__(
        self,
        label: str,
        color: str,
        dash: Optional[str],
        folds: Sequence[FoldMetricCurves],
        roc_mean: List[Tuple[float, float]],
        roc_lower: List[Tuple[float, float]],
        roc_upper: List[Tuple[float, float]],
        pr_mean: List[Tuple[float, float]],
        pr_lower: List[Tuple[float, float]],
        pr_upper: List[Tuple[float, float]],
        auroc: Dict[str, float],
        auprc: Dict[str, float],
        pr_prevalence: float,
    ) -> None:
        self.label = label
        self.color = color
        self.dash = dash
        self.folds = list(folds)
        self.roc_mean = roc_mean
        self.roc_lower = roc_lower
        self.roc_upper = roc_upper
        self.pr_mean = pr_mean
        self.pr_lower = pr_lower
        self.pr_upper = pr_upper
        self.auroc = auroc
        self.auprc = auprc
        self.pr_prevalence = pr_prevalence


def compute_pr(labels: Sequence[int], scores: Sequence[float]) -> Tuple[List[Tuple[float, float]], float]:
    """Return precision-recall points and sklearn-style average precision."""
    positives = float(sum(labels))
    if positives <= 0.0:
        raise ValueError("Precision-recall requires at least one positive label")

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    points: List[Tuple[float, float]] = [(0.0, 1.0)]
    tp = 0.0
    fp = 0.0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                tp += 1.0
            else:
                fp += 1.0
            index += 1
        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 1.0
        points.append((recall, precision))
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall

    if points[-1][0] < 1.0:
        points.append((1.0, positives / float(len(labels))))
    return points, average_precision


def source_test_prediction_csvs(source_run_root: Path, folds: Sequence[int] = tuple(range(5))) -> Dict[int, Path]:
    csvs = {
        int(fold): Path(source_run_root) / "fold{0}".format(fold) / "final_results" / "test_predictions.csv"
        for fold in folds
    }
    missing = [path for path in csvs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source prediction CSVs:\n" + "\n".join(str(path) for path in missing))
    return csvs


def prediction_csvs_from_run_roots(run_roots_by_fold: Dict[int, Path], *relative_parts: str) -> Dict[int, Path]:
    if not relative_parts:
        raise ValueError("Provide the prediction CSV path relative to each fold run root")
    csvs = {
        int(fold): Path(run_root).joinpath(*relative_parts)
        for fold, run_root in sorted(run_roots_by_fold.items())
    }
    missing = [path for path in csvs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction CSVs:\n" + "\n".join(str(path) for path in missing))
    return csvs


def default_curve_output_svg(run_roots_by_fold: Dict[int, Path], output_filename: str) -> Path:
    if not run_roots_by_fold:
        raise ValueError("No run roots were provided")
    first_root = Path(next(iter(dict(sorted(run_roots_by_fold.items())).values())))
    return first_root.parent / "curve_reports" / output_filename


def load_fold_metric_curves(fold: int, prediction_csv: Path) -> FoldMetricCurves:
    labels, scores = load_labels_scores(prediction_csv)
    roc_points, auroc = compute_roc(labels, scores)
    pr_points, auprc = compute_pr(labels, scores)
    return FoldMetricCurves(
        fold=fold,
        prediction_csv=prediction_csv,
        roc_points=roc_points,
        auroc=auroc,
        pr_points=pr_points,
        auprc=auprc,
        n_samples=len(labels),
        n_positive=int(sum(labels)),
        n_negative=int(len(labels) - sum(labels)),
    )


def compress_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    by_x: Dict[float, float] = {}
    for x_value, y_value in points:
        by_x[x_value] = max(y_value, by_x.get(x_value, 0.0))
    compressed = sorted(by_x.items())
    if not compressed:
        return []
    if compressed[0][0] > 0.0:
        compressed.insert(0, (0.0, compressed[0][1]))
    if compressed[-1][0] < 1.0:
        compressed.append((1.0, compressed[-1][1]))
    return compressed


def interpolate_y(points: Sequence[Tuple[float, float]], x_value: float) -> float:
    compressed = compress_points(points)
    if not compressed:
        raise ValueError("Cannot interpolate an empty curve")
    if x_value <= compressed[0][0]:
        return compressed[0][1]
    for index in range(1, len(compressed)):
        x0, y0 = compressed[index - 1]
        x1, y1 = compressed[index]
        if x_value <= x1:
            if abs(x1 - x0) < 1e-12:
                return max(y0, y1)
            fraction = (x_value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return compressed[-1][1]


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot compute a confidence interval for no values")
    mean_value = statistics.mean(values)
    if len(values) == 1:
        return mean_value, mean_value, mean_value
    sd_value = statistics.stdev(values)
    t_value = T_CRITICAL_95.get(len(values) - 1, 1.96)
    half_width = t_value * sd_value / math.sqrt(float(len(values)))
    return mean_value, mean_value - half_width, mean_value + half_width


def mean_curve_ci(
    curves: Sequence[Sequence[Tuple[float, float]]],
    grid: Sequence[float],
    force_roc_endpoints: bool = False,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    mean_points: List[Tuple[float, float]] = []
    lower_points: List[Tuple[float, float]] = []
    upper_points: List[Tuple[float, float]] = []
    for x_value in grid:
        values = [interpolate_y(curve, x_value) for curve in curves]
        mean_value, lower_value, upper_value = mean_ci(values)
        mean_points.append((x_value, min(1.0, max(0.0, mean_value))))
        lower_points.append((x_value, min(1.0, max(0.0, lower_value))))
        upper_points.append((x_value, min(1.0, max(0.0, upper_value))))

    if force_roc_endpoints:
        mean_points[0] = (0.0, 0.0)
        lower_points[0] = (0.0, 0.0)
        upper_points[0] = (0.0, 0.0)
        mean_points[-1] = (1.0, 1.0)
        lower_points[-1] = (1.0, 1.0)
        upper_points[-1] = (1.0, 1.0)
    return mean_points, lower_points, upper_points


def polygon_points(
    lower_points: Sequence[Tuple[float, float]],
    upper_points: Sequence[Tuple[float, float]],
    map_x,
    map_y,
) -> str:
    points = list(upper_points) + list(reversed(lower_points))
    return " ".join("{:.2f},{:.2f}".format(map_x(x), map_y(y)) for x, y in points)


def metric_summary(values: Sequence[float]) -> Dict[str, float]:
    mean_value, lower_value, upper_value = mean_ci(values)
    return {
        "mean": mean_value,
        "ci_lower": max(0.0, lower_value),
        "ci_upper": min(1.0, upper_value),
        "ci_half_width": max(mean_value - lower_value, upper_value - mean_value),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def prevalence(folds: Sequence[FoldMetricCurves]) -> float:
    n_positive = sum(fold.n_positive for fold in folds)
    n_total = sum(fold.n_samples for fold in folds)
    if n_total == 0:
        raise ValueError("Cannot compute prevalence with no samples")
    return n_positive / float(n_total)


CURVE_LINE_OPTIONS = {
    "source": ("Source", "#1f77b4", None),
    "zero_shot_sa": ("Zero-Shot SA", "#ff7f0e", "9 7"),
    "finetuned_sa": ("Finetuned SA", "#2ca02c", None),
    "source_retention": ("Source Retention", "#9467bd", "12 6"),
    "prevalence": ("Prevalence", "#7f7f7f", "9 7"),
}

CURVE_LINE_ALIASES = {
    "source": "source",
    "zero-shot sa": "zero_shot_sa",
    "zero_shot_sa": "zero_shot_sa",
    "zero shot sa": "zero_shot_sa",
    "zeroshot sa": "zero_shot_sa",
    "finetuned sa": "finetuned_sa",
    "fine-tuned sa": "finetuned_sa",
    "fine_tuned_sa": "finetuned_sa",
    "finetuned_sa": "finetuned_sa",
    "target finetuned": "finetuned_sa",
    "target_finetuned": "finetuned_sa",
    "source retention": "source_retention",
    "source_retention": "source_retention",
    "prevalence": "prevalence",
    "prevalence baseline": "prevalence",
    "prevalence_baseline": "prevalence",
}

CURVE_LINE_DOMAINS = {
    "source": "Benin",
    "source_retention": "Benin",
    "zero_shot_sa": "SA",
    "finetuned_sa": "SA",
}


def _normalize_curve_line(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    normalized_with_underscores = normalized.replace(" ", "_")
    for candidate in (normalized, normalized_with_underscores):
        canonical = CURVE_LINE_ALIASES.get(candidate)
        if canonical is not None:
            return canonical
    valid = ", ".join(option[0] for option in CURVE_LINE_OPTIONS.values())
    raise ValueError(f"lines_to_show entries must be one of: {valid}")


def _normalize_lines_to_show(lines_to_show: Sequence[str]) -> List[str]:
    raw_lines = [lines_to_show] if isinstance(lines_to_show, str) else list(lines_to_show)
    normalized: List[str] = []
    for value in raw_lines:
        key = _normalize_curve_line(value)
        if key not in normalized:
            normalized.append(key)
    return normalized


def _existing_source_prediction_csvs(source_run_root: Optional[Path]) -> Optional[Dict[int, Path]]:
    if source_run_root is None:
        return None
    csvs = {
        int(fold): Path(source_run_root) / "fold{0}".format(fold) / "final_results" / "test_predictions.csv"
        for fold in range(5)
    }
    if any(not path.exists() for path in csvs.values()):
        return None
    return csvs


def _existing_prediction_csvs_from_roots(
    run_roots_by_fold: Dict[int, Path],
    relative_parts: Sequence[str],
    fallback_roots_by_fold: Optional[Dict[int, Path]] = None,
) -> Optional[Dict[int, Path]]:
    if not run_roots_by_fold or not relative_parts:
        return None
    csvs: Dict[int, Path] = {}
    for fold, run_root in sorted(run_roots_by_fold.items()):
        path = Path(run_root).joinpath(*relative_parts)
        if not path.exists() and fallback_roots_by_fold:
            fallback_root = fallback_roots_by_fold.get(fold)
            if fallback_root is not None:
                fallback_path = Path(fallback_root).joinpath(*relative_parts)
                if fallback_path.exists():
                    path = fallback_path
        if not path.exists():
            return None
        csvs[int(fold)] = path
    return csvs


def _summarize_prediction_csvs(
    label: str,
    color: str,
    dash: Optional[str],
    prediction_csvs_by_fold: Dict[int, Path],
    grid: Sequence[float],
) -> MeanMetricCurves:
    folds = [
        load_fold_metric_curves(int(fold), Path(path))
        for fold, path in sorted(prediction_csvs_by_fold.items())
    ]
    roc_mean, roc_lower, roc_upper = mean_curve_ci(
        [fold.roc_points for fold in folds],
        grid,
        force_roc_endpoints=True,
    )
    pr_mean, pr_lower, pr_upper = mean_curve_ci(
        [fold.pr_points for fold in folds],
        grid,
        force_roc_endpoints=False,
    )
    return MeanMetricCurves(
        label=label,
        color=color,
        dash=dash,
        folds=folds,
        roc_mean=roc_mean,
        roc_lower=roc_lower,
        roc_upper=roc_upper,
        pr_mean=pr_mean,
        pr_lower=pr_lower,
        pr_upper=pr_upper,
        auroc=metric_summary([fold.auroc for fold in folds]),
        auprc=metric_summary([fold.auprc for fold in folds]),
        pr_prevalence=prevalence(folds),
    )


def _legend_metric_label(metric: str, values: Dict[str, float], label: Optional[str] = None) -> str:
    prefix = "{label}: ".format(label=label) if label else ""
    return "{prefix}{metric} = {mean:.2f} (95% CI {low:.2f}-{high:.2f})".format(
        prefix=prefix,
        metric=metric,
        mean=values["mean"],
        low=values["ci_lower"],
        high=values["ci_upper"],
    )


def _estimate_legend_text_width(text: str, font_size: int = 20) -> float:
    """Approximate Arial/Helvetica SVG text width for sizing legend boxes."""
    width = 0.0
    for char in text:
        if char == " ":
            width += 0.28
        elif char in ".,:;":
            width += 0.24
        elif char in "()[]{}":
            width += 0.32
        elif char in "-=/+":
            width += 0.50
        elif char in "ilI":
            width += 0.26
        elif char in "mwMW":
            width += 0.82
        elif char.isupper():
            width += 0.68
        elif char.isdigit():
            width += 0.56
        else:
            width += 0.52
    return width * float(font_size)


def _normalize_legend_position(position: str) -> str:
    normalized = str(position).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "upper_right": "upper_right",
        "top_right": "upper_right",
        "lower_right": "lower_right",
        "bottom_right": "lower_right",
        "below": "below",
        "below_graphs": "below",
        "outside_bottom": "below",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"legend_position must be one of: {valid}") from exc


def _write_panel(
    lines: List[str],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Dict[str, object]],
    chance_kind: str,
    legend_position: str = "lower_right",
    chance_value: Optional[float] = None,
    reference_lines: Optional[Sequence[Dict[str, object]]] = None,
) -> List[Tuple[str, Optional[str], str]]:
    axis_color = "#2b2f33"
    grid_color = "#e3e5e8"
    chance_color = "#7f7f7f"

    def map_x(value: float) -> float:
        return panel_x + value * panel_w

    def map_y(value: float) -> float:
        return panel_y + (1.0 - value) * panel_h

    title_style = "font-family: Arial, Helvetica, sans-serif; font-size: 30px; font-weight: 800; fill: #282828; letter-spacing: 0"
    tick_style = "font-family: Arial, Helvetica, sans-serif; font-size: 24px; fill: #333333; letter-spacing: 0"
    label_style = "font-family: Arial, Helvetica, sans-serif; font-size: 28px; font-weight: 800; fill: #282828; letter-spacing: 0"
    legend_style = "font-family: Arial, Helvetica, sans-serif; font-size: 20px; fill: #2d2d2d; letter-spacing: 0"

    lines.append(
        '<text x="{x}" y="{y}" text-anchor="middle" style="{style}">{title}</text>'.format(
            x=panel_x + panel_w / 2,
            y=panel_y - 35,
            style=title_style,
            title=html.escape(title),
        )
    )
    lines.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#b8b8b8" stroke-width="1.5"/>'.format(
            x=panel_x,
            y=panel_y,
            w=panel_w,
            h=panel_h,
        )
    )

    for index in range(6):
        value = index / 5.0
        x_value = map_x(value)
        y_value = map_y(value)
        lines.append(
            '<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{color}" stroke-width="1" stroke-dasharray="3 4"/>'.format(
                x=x_value,
                top=panel_y,
                bottom=panel_y + panel_h,
                color=grid_color,
            )
        )
        lines.append(
            '<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{color}" stroke-width="1" stroke-dasharray="3 4"/>'.format(
                left=panel_x,
                right=panel_x + panel_w,
                y=y_value,
                color=grid_color,
            )
        )

    reference_rows: List[Tuple[str, Optional[str], str]] = []
    if chance_kind == "roc":
        lines.append(
            '<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
                x0=map_x(0.0),
                y0=map_y(0.0),
                x1=map_x(1.0),
                y1=map_y(1.0),
                color=chance_color,
            )
        )
        reference_rows.append((chance_color, "9 7", "Chance (AUROC=0.50)"))
    elif chance_kind == "pr":
        if chance_value is not None:
            baseline = float(chance_value)
            lines.append(
                '<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
                    x0=map_x(0.0),
                    x1=map_x(1.0),
                    y=map_y(baseline),
                    color=chance_color,
                )
            )
            reference_rows.append((chance_color, "9 7", "Prevalence (AUPRC={:.2f})".format(baseline)))

    for item in reference_lines or ():
        value = float(item["value"])
        color = str(item.get("color", chance_color))
        dash = item.get("dash", "9 7")
        dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
        lines.append(
            '<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2.5"{dash}/>'.format(
                x0=map_x(0.0),
                x1=map_x(1.0),
                y=map_y(value),
                color=color,
                dash=dash_attr,
            )
        )
        reference_rows.append((color, str(dash) if dash else None, str(item["label"])))

    for item in series:
        lines.append(
            '<polygon points="{points}" fill="{color}" opacity="0.10"/>'.format(
                points=polygon_points(
                    item["lower_points"],  # type: ignore[arg-type]
                    item["upper_points"],  # type: ignore[arg-type]
                    map_x,
                    map_y,
                ),
                color=item["color"],
            )
        )
    for item in series:
        dash = item.get("dash")
        dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
        lines.append(
            '<polyline points="{points}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"{dash}/>'.format(
                points=svg_polyline(item["mean_points"], map_x, map_y),  # type: ignore[arg-type]
                color=item["color"],
                dash=dash_attr,
            )
        )
    lines.append(
        '<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2"/>'.format(
            x=panel_x,
            y1=panel_y,
            y2=panel_y + panel_h,
            color=axis_color,
        )
    )
    lines.append(
        '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2"/>'.format(
            x1=panel_x,
            x2=panel_x + panel_w,
            y=panel_y + panel_h,
            color=axis_color,
        )
    )

    for index in range(6):
        value = index / 5.0
        x_value = map_x(value)
        y_value = map_y(value)
        lines.append(
            '<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"/>'.format(
                x=x_value,
                y1=panel_y + panel_h,
                y2=panel_y + panel_h + 8,
                color=axis_color,
            )
        )
        lines.append(
            '<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" style="{style}">{value:.1f}</text>'.format(
                x=x_value,
                y=panel_y + panel_h + 42,
                style=tick_style,
                value=value,
            )
        )
        lines.append(
            '<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/>'.format(
                x1=panel_x - 8,
                x2=panel_x,
                y=y_value,
                color=axis_color,
            )
        )
        lines.append(
            '<text x="{x:.2f}" y="{y:.2f}" text-anchor="end" dominant-baseline="middle" style="{style}">{value:.1f}</text>'.format(
                x=panel_x - 18,
                y=y_value,
                style=tick_style,
                value=value,
            )
        )

    lines.append(
        '<text x="{x}" y="{y}" text-anchor="middle" style="{style}">{label}</text>'.format(
            x=panel_x + panel_w / 2,
            y=panel_y + panel_h + 88,
            style=label_style,
            label=html.escape(x_label),
        )
    )
    y_axis_x = panel_x - 92
    y_axis_y = panel_y + panel_h / 2
    lines.append(
        '<text x="{x}" y="{y}" text-anchor="middle" transform="rotate(-90 {x} {y})" style="{style}">{label}</text>'.format(
            x=y_axis_x,
            y=y_axis_y,
            style=label_style,
            label=html.escape(y_label),
        )
    )

    rows = [
        (item["color"], item.get("dash"), item["legend_text"])
        for item in series
    ]
    rows.extend(reference_rows)
    if series:
        band_label = "Shaded band: fold mean 95% CI" if len(series) == 1 else "Shaded bands: fold mean 95% CI"
        band_color = series[0]["color"] if len(series) == 1 else "#9ca3af"
        rows.append((band_color, "shade", band_label))

    legend_position = _normalize_legend_position(legend_position)
    if legend_position == "below":
        return [(str(row_color), str(dash) if dash else None, str(text)) for row_color, dash, text in rows]

    text_x_offset = 112
    right_padding = 24
    legend_w = int(
        math.ceil(text_x_offset + max(_estimate_legend_text_width(text) for _, _, text in rows) + right_padding)
    )
    legend_w = min(panel_w - 40, max(360, legend_w))
    legend_h = 52 + max(0, len(rows) - 1) * 38
    legend_x = panel_x + panel_w - legend_w - 20
    if legend_position == "upper_right":
        legend_y = panel_y + 20
    else:
        legend_y = panel_y + panel_h - legend_h - 20
    lines.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#000000" opacity="0.18"/>'.format(
            x=legend_x + 8,
            y=legend_y + 8,
            w=legend_w,
            h=legend_h,
        )
    )
    lines.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#c7c7c7" stroke-width="2"/>'.format(
            x=legend_x,
            y=legend_y,
            w=legend_w,
            h=legend_h,
        )
    )
    for offset, (row_color, dash, text) in enumerate(rows):
        y_value = legend_y + 30 + offset * 38
        if dash == "shade":
            lines.append(
                '<rect x="{x}" y="{y}" width="70" height="16" fill="{color}" opacity="0.18"/>'.format(
                    x=legend_x + 22,
                    y=y_value - 8,
                    color=row_color,
                )
            )
        else:
            dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
            lines.append(
                '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="5"{dash}/>'.format(
                    x1=legend_x + 22,
                    x2=legend_x + 92,
                    y=y_value,
                    color=row_color,
                    dash=dash_attr,
                )
            )
        lines.append(
            '<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{text}</text>'.format(
                x=legend_x + text_x_offset,
                y=y_value,
                style=legend_style,
                text=html.escape(text),
            )
        )
    return [(str(row_color), str(dash) if dash else None, str(text)) for row_color, dash, text in rows]


def _measure_horizontal_legend_group(
    rows: Sequence[Tuple[str, Optional[str], str]],
    width: int,
    font_size: int = 18,
) -> List[List[Tuple[str, Optional[str], str, int]]]:
    marker_w = 64
    text_gap = 16
    item_gap = 34
    max_row_w = width - 48
    wrapped: List[List[Tuple[str, Optional[str], str, int]]] = []
    current_row: List[Tuple[str, Optional[str], str, int]] = []
    current_w = 0
    for color, dash, text in rows:
        item_w = int(math.ceil(marker_w + text_gap + _estimate_legend_text_width(text, font_size) + item_gap))
        if current_row and current_w + item_w > max_row_w:
            wrapped.append(current_row)
            current_row = []
            current_w = 0
        current_row.append((color, dash, text, item_w))
        current_w += item_w
    if current_row:
        wrapped.append(current_row)
    return wrapped


def _draw_horizontal_legend_group(
    lines: List[str],
    x: int,
    y: int,
    width: int,
    title: str,
    rows: Sequence[Tuple[str, Optional[str], str]],
) -> int:
    if not rows:
        return y

    font_size = 18
    wrapped_rows = _measure_horizontal_legend_group(rows, width, font_size=font_size)
    row_gap = 32
    top_padding = 22
    title_h = 22
    bottom_padding = 18
    box_h = top_padding + title_h + len(wrapped_rows) * row_gap + bottom_padding
    title_style = "font-family: Arial, Helvetica, sans-serif; font-size: 18px; font-weight: 800; fill: #282828; letter-spacing: 0"
    item_style = "font-family: Arial, Helvetica, sans-serif; font-size: 18px; fill: #2d2d2d; letter-spacing: 0"

    lines.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#000000" opacity="0.12"/>'.format(
            x=x + 7,
            y=y + 7,
            w=width,
            h=box_h,
        )
    )
    lines.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" fill-opacity="0.98" stroke="#c7c7c7" stroke-width="2"/>'.format(
            x=x,
            y=y,
            w=width,
            h=box_h,
        )
    )
    lines.append(
        '<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{title}</text>'.format(
            x=x + 24,
            y=y + top_padding,
            style=title_style,
            title=html.escape(title),
        )
    )

    row_y = y + top_padding + title_h + 9
    for wrapped_row in wrapped_rows:
        item_x = x + 24
        for color, dash, text, item_w in wrapped_row:
            if dash == "shade":
                lines.append(
                    '<rect x="{x}" y="{y}" width="64" height="15" fill="{color}" opacity="0.18"/>'.format(
                        x=item_x,
                        y=row_y - 8,
                        color=color,
                    )
                )
            else:
                dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
                lines.append(
                    '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="4.5"{dash}/>'.format(
                        x1=item_x,
                        x2=item_x + 64,
                        y=row_y,
                        color=color,
                        dash=dash_attr,
                    )
                )
            lines.append(
                '<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{text}</text>'.format(
                    x=item_x + 80,
                    y=row_y,
                    style=item_style,
                    text=html.escape(text),
                )
            )
            item_x += item_w
        row_y += row_gap

    return y + box_h


def write_roc_pr_svg(
    path: Path,
    title: str,
    subtitle: str,
    evaluation_label: str,
    roc_mean: Sequence[Tuple[float, float]],
    roc_lower: Sequence[Tuple[float, float]],
    roc_upper: Sequence[Tuple[float, float]],
    pr_mean: Sequence[Tuple[float, float]],
    pr_lower: Sequence[Tuple[float, float]],
    pr_upper: Sequence[Tuple[float, float]],
    auroc_summary: Dict[str, float],
    auprc_summary: Dict[str, float],
    pr_prevalence: float,
    color: str = "#2ca02c",
    roc_legend_position: str = "lower_right",
    pr_legend_position: str = "lower_right",
) -> None:
    evaluations = [
        {
            "label": evaluation_label,
            "color": color,
            "dash": None,
            "roc_mean": roc_mean,
            "roc_lower": roc_lower,
            "roc_upper": roc_upper,
            "pr_mean": pr_mean,
            "pr_lower": pr_lower,
            "pr_upper": pr_upper,
            "auroc": auroc_summary,
            "auprc": auprc_summary,
        }
    ]
    write_multi_roc_pr_svg(
        path=path,
        title=title,
        subtitle=subtitle,
        evaluations=evaluations,
        pr_prevalence=pr_prevalence,
        roc_legend_position=roc_legend_position,
        pr_legend_position=pr_legend_position,
        include_evaluation_labels=False,
    )


def write_multi_roc_pr_svg(
    path: Path,
    title: str,
    subtitle: str,
    evaluations: Sequence[Dict[str, object]],
    pr_prevalence: Optional[float],
    pr_reference_lines: Optional[Sequence[Dict[str, object]]] = None,
    roc_legend_position: str = "lower_right",
    pr_legend_position: str = "lower_right",
    include_evaluation_labels: bool = True,
) -> None:
    if not evaluations:
        raise ValueError("Cannot write ROC/PR SVG without at least one evaluation")
    width = 2200
    roc_legend_position = _normalize_legend_position(roc_legend_position)
    pr_legend_position = _normalize_legend_position(pr_legend_position)
    height = 1480 if "below" in {roc_legend_position, pr_legend_position} else 1180
    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=width,
            h=height,
        )
    )
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    title_style = "font-family: Arial, Helvetica, sans-serif; font-size: 42px; font-weight: 800; fill: #282828; letter-spacing: 0"
    subtitle_style = "font-family: Arial, Helvetica, sans-serif; font-size: 34px; font-weight: 800; fill: #282828; letter-spacing: 0"
    lines.append(
        '<text x="{x}" y="58" text-anchor="middle" style="{style}">{title}</text>'.format(
            x=width / 2,
            style=title_style,
            title=html.escape(title),
        )
    )
    lines.append(
        '<text x="{x}" y="104" text-anchor="middle" style="{style}">{subtitle}</text>'.format(
            x=width / 2,
            style=subtitle_style,
            subtitle=html.escape(subtitle),
        )
    )

    roc_series = [
        {
            "color": item["color"],
            "dash": item.get("dash"),
            "mean_points": item["roc_mean"],
            "lower_points": item["roc_lower"],
            "upper_points": item["roc_upper"],
            "legend_text": _legend_metric_label(
                "AUROC",
                item["auroc"],  # type: ignore[arg-type]
                str(item["label"]) if include_evaluation_labels else None,
            ),
        }
        for item in evaluations
    ]
    pr_series = [
        {
            "color": item["color"],
            "dash": item.get("dash"),
            "mean_points": item["pr_mean"],
            "lower_points": item["pr_lower"],
            "upper_points": item["pr_upper"],
            "legend_text": _legend_metric_label(
                "AUPRC",
                item["auprc"],  # type: ignore[arg-type]
                str(item["label"]) if include_evaluation_labels else None,
            ),
        }
        for item in evaluations
    ]

    roc_legend_rows = _write_panel(
        lines,
        panel_x=150,
        panel_y=195,
        panel_w=820,
        panel_h=760,
        title="ROC Curve",
        x_label="False Positive Rate (1 - Specificity)",
        y_label="True Positive Rate (Sensitivity)",
        series=roc_series,
        chance_kind="roc",
        legend_position=roc_legend_position,
    )
    pr_legend_rows = _write_panel(
        lines,
        panel_x=1230,
        panel_y=195,
        panel_w=820,
        panel_h=760,
        title="Precision-Recall Curve",
        x_label="Recall (Sensitivity)",
        y_label="Precision (PPV)",
        series=pr_series,
        chance_kind="pr",
        legend_position=pr_legend_position,
        chance_value=pr_prevalence,
        reference_lines=pr_reference_lines,
    )

    legend_y = 1098
    if roc_legend_position == "below":
        _draw_horizontal_legend_group(
            lines,
            x=150,
            y=legend_y,
            width=820,
            title="ROC legend",
            rows=roc_legend_rows,
        )
    if pr_legend_position == "below":
        _draw_horizontal_legend_group(
            lines,
            x=1230,
            y=legend_y,
            width=820,
            title="Precision-Recall legend",
            rows=pr_legend_rows,
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_svg(path: Path) -> None:
    try:
        from IPython.display import SVG, display
    except ImportError:
        return
    display(SVG(filename=str(path)))


def plot_single_evaluation_roc_pr(
    prediction_csvs_by_fold: Dict[int, Path],
    evaluation_label: str,
    title: str,
    subtitle: str,
    output_svg: Path,
    grid_size: int = 201,
    color: str = "#2ca02c",
    roc_legend_position: str = "lower_right",
    pr_legend_position: str = "lower_right",
    show: bool = True,
) -> Dict[str, object]:
    """Write and optionally display one ROC/PR figure for one evaluation run.

    ``prediction_csvs_by_fold`` should contain only the run/evaluation to plot.
    This keeps each figure source-specific, fine-tune-specific, or
    domain-adaptation-specific depending on the snippet that calls it.
    """
    if not prediction_csvs_by_fold:
        raise ValueError("No prediction CSVs were provided")

    missing = [Path(path) for path in prediction_csvs_by_fold.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing prediction CSVs:\n" + "\n".join(str(path) for path in missing))

    folds = [
        load_fold_metric_curves(int(fold), Path(path))
        for fold, path in sorted(prediction_csvs_by_fold.items())
    ]
    grid_size = max(int(grid_size), 2)
    grid = [index / float(grid_size - 1) for index in range(grid_size)]

    roc_mean, roc_lower, roc_upper = mean_curve_ci(
        [fold.roc_points for fold in folds],
        grid,
        force_roc_endpoints=True,
    )
    pr_mean, pr_lower, pr_upper = mean_curve_ci(
        [fold.pr_points for fold in folds],
        grid,
        force_roc_endpoints=False,
    )
    auroc_values = [fold.auroc for fold in folds]
    auprc_values = [fold.auprc for fold in folds]
    auroc = metric_summary(auroc_values)
    auprc = metric_summary(auprc_values)
    pr_prevalence = prevalence(folds)

    output_svg = Path(output_svg)
    write_roc_pr_svg(
        output_svg,
        title=title,
        subtitle=subtitle,
        evaluation_label=evaluation_label,
        roc_mean=roc_mean,
        roc_lower=roc_lower,
        roc_upper=roc_upper,
        pr_mean=pr_mean,
        pr_lower=pr_lower,
        pr_upper=pr_upper,
        auroc_summary=auroc,
        auprc_summary=auprc,
        pr_prevalence=pr_prevalence,
        color=color,
        roc_legend_position=roc_legend_position,
        pr_legend_position=pr_legend_position,
    )
    if show:
        display_svg(output_svg)

    return {
        "output_svg": str(output_svg),
        "evaluation_label": evaluation_label,
        "n_folds": len(folds),
        "folds": [fold.fold for fold in folds],
        "prediction_csvs_by_fold": {str(fold): str(path) for fold, path in sorted(prediction_csvs_by_fold.items())},
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": pr_prevalence,
        "n_samples_total": sum(fold.n_samples for fold in folds),
        "n_positive_total": sum(fold.n_positive for fold in folds),
        "n_negative_total": sum(fold.n_negative for fold in folds),
    }


def _mean_curve_result(curve: MeanMetricCurves) -> Dict[str, object]:
    return {
        "label": curve.label,
        "folds": [fold.fold for fold in curve.folds],
        "prediction_csvs_by_fold": {str(fold.fold): str(fold.prediction_csv) for fold in curve.folds},
        "auroc": curve.auroc,
        "auprc": curve.auprc,
        "prevalence": curve.pr_prevalence,
        "n_samples_total": sum(fold.n_samples for fold in curve.folds),
        "n_positive_total": sum(fold.n_positive for fold in curve.folds),
        "n_negative_total": sum(fold.n_negative for fold in curve.folds),
    }


def plot_selected_run_roots_roc_pr(
    run_roots_by_fold: Dict[int, Path],
    *relative_prediction_csv: str,
    evaluation_label: str,
    title: str,
    subtitle: str,
    lines_to_show: Sequence[str],
    source_run_root: Optional[Path] = None,
    zero_shot_reference_roots_by_fold: Optional[Dict[int, Path]] = None,
    output_svg: Optional[Path] = None,
    output_filename: Optional[str] = None,
    grid_size: int = 201,
    roc_legend_position: str = "lower_right",
    pr_legend_position: str = "lower_right",
    show: bool = True,
) -> Dict[str, object]:
    if output_svg is None:
        filename = output_filename or "{0}_roc_pr_mean_ci.svg".format(filename_slug(evaluation_label))
        output_svg = default_curve_output_svg(run_roots_by_fold, filename)

    grid_size = max(int(grid_size), 2)
    grid = [index / float(grid_size - 1) for index in range(grid_size)]
    selected = _normalize_lines_to_show(lines_to_show)
    show_prevalence = "prevalence" in selected
    curves: List[MeanMetricCurves] = []
    plotted_keys: List[str] = []
    skipped_lines: List[str] = []

    for key in selected:
        if key == "prevalence":
            continue
        label, color, dash = CURVE_LINE_OPTIONS[key]
        if key == "source":
            csvs = _existing_source_prediction_csvs(source_run_root)
        elif key == "zero_shot_sa":
            csvs = _existing_prediction_csvs_from_roots(
                run_roots_by_fold,
                ("results", "source_zero_shot", "source_zero_shot_patient_predictions.csv"),
                fallback_roots_by_fold=zero_shot_reference_roots_by_fold,
            )
        elif key == "finetuned_sa":
            csvs = _existing_prediction_csvs_from_roots(run_roots_by_fold, relative_prediction_csv)
        elif key == "source_retention":
            csvs = _existing_prediction_csvs_from_roots(
                run_roots_by_fold,
                ("results", "source_test_evaluation", "finetuned_source_test_patient_predictions.csv"),
            )
        else:  # pragma: no cover - _normalize_lines_to_show validates this.
            csvs = None

        if csvs is None:
            skipped_lines.append(label)
            continue
        curves.append(_summarize_prediction_csvs(label, color, dash, csvs, grid))
        plotted_keys.append(key)

    if not curves:
        requested = ", ".join(CURVE_LINE_OPTIONS[key][0] for key in selected)
        raise ValueError(f"None of the requested curve lines had complete prediction CSVs: {requested}")

    pr_reference_lines: List[Dict[str, object]] = []
    if show_prevalence:
        prevalence_by_domain: Dict[str, float] = {}
        for key, curve in zip(plotted_keys, curves):
            domain = CURVE_LINE_DOMAINS.get(key)
            if domain is not None and domain not in prevalence_by_domain:
                prevalence_by_domain[domain] = curve.pr_prevalence
        for domain, color in (("SA", "#6b7280"), ("Benin", "#374151")):
            value = prevalence_by_domain.get(domain)
            if value is None:
                continue
            pr_reference_lines.append(
                {
                    "label": "{domain} prevalence (AUPRC={value:.2f})".format(domain=domain, value=value),
                    "value": value,
                    "color": color,
                    "dash": "9 7",
                }
            )
        if not pr_reference_lines:
            skipped_lines.append("Prevalence")
    svg_evaluations = [
        {
            "label": curve.label,
            "color": curve.color,
            "dash": curve.dash,
            "roc_mean": curve.roc_mean,
            "roc_lower": curve.roc_lower,
            "roc_upper": curve.roc_upper,
            "pr_mean": curve.pr_mean,
            "pr_lower": curve.pr_lower,
            "pr_upper": curve.pr_upper,
            "auroc": curve.auroc,
            "auprc": curve.auprc,
        }
        for curve in curves
    ]

    output_svg = Path(output_svg)
    write_multi_roc_pr_svg(
        output_svg,
        title=title,
        subtitle=subtitle,
        evaluations=svg_evaluations,
        pr_prevalence=None,
        pr_reference_lines=pr_reference_lines,
        roc_legend_position=roc_legend_position,
        pr_legend_position=pr_legend_position,
        include_evaluation_labels=True,
    )
    if show:
        display_svg(output_svg)

    results_by_label = {curve.label: _mean_curve_result(curve) for curve in curves}
    primary_curve = next((curve for curve in curves if curve.label == "Finetuned SA"), curves[0])
    return {
        "output_svg": str(output_svg),
        "evaluation_label": evaluation_label,
        "lines_to_show": [CURVE_LINE_OPTIONS[key][0] for key in selected],
        "plotted_lines": [curve.label for curve in curves],
        "prevalence_lines": [str(item["label"]) for item in pr_reference_lines],
        "skipped_lines": skipped_lines,
        "evaluations": results_by_label,
        "auroc": primary_curve.auroc,
        "auprc": primary_curve.auprc,
        "prevalence": primary_curve.pr_prevalence,
    }


def filename_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "metric_curve"


def plot_run_roots_roc_pr(
    run_roots_by_fold: Dict[int, Path],
    *relative_prediction_csv: str,
    evaluation_label: str,
    title: str,
    subtitle: str,
    lines_to_show: Optional[Sequence[str]] = None,
    source_run_root: Optional[Path] = None,
    zero_shot_reference_roots_by_fold: Optional[Dict[int, Path]] = None,
    output_svg: Optional[Path] = None,
    output_filename: Optional[str] = None,
    grid_size: int = 201,
    color: str = "#2ca02c",
    roc_legend_position: str = "lower_right",
    pr_legend_position: str = "lower_right",
    show: bool = True,
) -> Dict[str, object]:
    if lines_to_show is not None:
        return plot_selected_run_roots_roc_pr(
            run_roots_by_fold,
            *relative_prediction_csv,
            evaluation_label=evaluation_label,
            title=title,
            subtitle=subtitle,
            lines_to_show=lines_to_show,
            source_run_root=source_run_root,
            zero_shot_reference_roots_by_fold=zero_shot_reference_roots_by_fold,
            output_svg=output_svg,
            output_filename=output_filename,
            grid_size=grid_size,
            roc_legend_position=roc_legend_position,
            pr_legend_position=pr_legend_position,
            show=show,
        )

    if output_svg is None:
        filename = output_filename or "{0}_roc_pr_mean_ci.svg".format(filename_slug(evaluation_label))
        output_svg = default_curve_output_svg(run_roots_by_fold, filename)
    return plot_single_evaluation_roc_pr(
        prediction_csvs_from_run_roots(run_roots_by_fold, *relative_prediction_csv),
        evaluation_label=evaluation_label,
        title=title,
        subtitle=subtitle,
        output_svg=output_svg,
        grid_size=grid_size,
        color=color,
        roc_legend_position=roc_legend_position,
        pr_legend_position=pr_legend_position,
        show=show,
    )


def plot_source_run_roc_pr(
    source_run_root: Path,
    evaluation_label: str = "Benin Source Baseline",
    title: str = "Benin Source Baseline Test Set Performance",
    subtitle: str = "Mean ROC and Precision-Recall Curves Across Five Folds",
    output_svg: Optional[Path] = None,
    grid_size: int = 201,
    color: str = "#2ca02c",
    roc_legend_position: str = "lower_right",
    pr_legend_position: str = "lower_right",
    show: bool = True,
) -> Dict[str, object]:
    source_run_root = Path(source_run_root)
    if output_svg is None:
        output_svg = source_run_root / "final_results" / "tb_results" / "source_baseline_roc_pr_mean_ci.svg"
    return plot_single_evaluation_roc_pr(
        source_test_prediction_csvs(source_run_root),
        evaluation_label=evaluation_label,
        title=title,
        subtitle=subtitle,
        output_svg=output_svg,
        grid_size=grid_size,
        color=color,
        roc_legend_position=roc_legend_position,
        pr_legend_position=pr_legend_position,
        show=show,
    )
