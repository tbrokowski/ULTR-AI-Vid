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


def _legend_metric_label(label: str, metric: str, values: Dict[str, float]) -> str:
    return "{label}: {metric}={mean:.2f} (95% CI {low:.2f}-{high:.2f})".format(
        label=label,
        metric=metric,
        mean=values["mean"],
        low=values["ci_lower"],
        high=values["ci_upper"],
    )


def _write_panel(
    lines: List[str],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    title: str,
    x_label: str,
    y_label: str,
    mean_points: Sequence[Tuple[float, float]],
    lower_points: Sequence[Tuple[float, float]],
    upper_points: Sequence[Tuple[float, float]],
    color: str,
    metric_label_text: str,
    chance_kind: str,
    chance_value: Optional[float] = None,
) -> None:
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
        chance_label = "Chance (AUROC=0.50)"
    elif chance_kind == "pr":
        baseline = float(chance_value or 0.0)
        lines.append(
            '<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
                x0=map_x(0.0),
                x1=map_x(1.0),
                y=map_y(baseline),
                color=chance_color,
            )
        )
        chance_label = "Prevalence (AUPRC={:.2f})".format(baseline)
    else:
        chance_label = ""

    lines.append(
        '<polygon points="{points}" fill="{color}" opacity="0.14"/>'.format(
            points=polygon_points(lower_points, upper_points, map_x, map_y),
            color=color,
        )
    )
    lines.append(
        '<polyline points="{points}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>'.format(
            points=svg_polyline(mean_points, map_x, map_y),
            color=color,
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

    legend_w = 680
    legend_h = 128
    legend_x = panel_x + panel_w - legend_w - 20
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

    rows = [
        (color, None, metric_label_text),
        (chance_color, "9 7", chance_label),
        (color, "shade", "Shaded band: fold mean 95% CI"),
    ]
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
                x=legend_x + 112,
                y=y_value,
                style=legend_style,
                text=html.escape(text),
            )
        )


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
) -> None:
    width = 2200
    height = 1180
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

    _write_panel(
        lines,
        panel_x=150,
        panel_y=195,
        panel_w=820,
        panel_h=760,
        title="ROC Curve",
        x_label="False Positive Rate (1 - Specificity)",
        y_label="True Positive Rate (Sensitivity)",
        mean_points=roc_mean,
        lower_points=roc_lower,
        upper_points=roc_upper,
        color=color,
        metric_label_text=_legend_metric_label(evaluation_label, "AUROC", auroc_summary),
        chance_kind="roc",
    )
    _write_panel(
        lines,
        panel_x=1230,
        panel_y=195,
        panel_w=820,
        panel_h=760,
        title="Precision-Recall Curve",
        x_label="Recall (Sensitivity)",
        y_label="Precision (PPV)",
        mean_points=pr_mean,
        lower_points=pr_lower,
        upper_points=pr_upper,
        color=color,
        metric_label_text=_legend_metric_label(evaluation_label, "AUPRC", auprc_summary),
        chance_kind="pr",
        chance_value=pr_prevalence,
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


def filename_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "metric_curve"


def plot_run_roots_roc_pr(
    run_roots_by_fold: Dict[int, Path],
    *relative_prediction_csv: str,
    evaluation_label: str,
    title: str,
    subtitle: str,
    output_svg: Optional[Path] = None,
    output_filename: Optional[str] = None,
    grid_size: int = 201,
    color: str = "#2ca02c",
    show: bool = True,
) -> Dict[str, object]:
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
        show=show,
    )
