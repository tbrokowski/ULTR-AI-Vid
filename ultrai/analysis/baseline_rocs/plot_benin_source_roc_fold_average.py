#!/usr/bin/env python3
"""Plot the original Benin source-model ROC curves across CV folds.

The default input is the original Benin baseline run. Each fold contributes its
held-out Benin test predictions from ``final_results/test_predictions.csv``.
The mean ROC is computed by interpolating each fold curve onto a shared FPR
grid; the thin curves show the individual folds and the bold curve shows the
fold-average trend.
"""

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


try:
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        SCRATCH_ROOT,
        compute_roc,
        convert_svg_to_png,
        load_labels_scores,
        simple_yaml_values,
        step_points,
        svg_polyline,
        write_csv,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        SCRATCH_ROOT,
        compute_roc,
        convert_svg_to_png,
        load_labels_scores,
        simple_yaml_values,
        step_points,
        svg_polyline,
        write_csv,
    )


DEFAULT_TRAIN_ROOT = SCRATCH_ROOT / "runs" / "train" / "benin"
DEFAULT_RUN_DIR = DEFAULT_TRAIN_ROOT / "20260404_150537__train__benin__all-folds__baseline"
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / "runs" / "baseline_roc_comparison" / "benin"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_DIR.name


class FoldCurve:
    def __init__(
        self,
        fold: int,
        fold_dir: Path,
        prediction_csv: Path,
        points: List[Tuple[float, float]],
        auroc: float,
        n_test: int,
        n_positive: int,
        n_negative: int,
        config: Dict[str, object],
    ) -> None:
        self.fold = fold
        self.fold_dir = fold_dir
        self.prediction_csv = prediction_csv
        self.points = points
        self.auroc = auroc
        self.n_test = n_test
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.config = config


def fold_number(path: Path) -> Optional[int]:
    match = re.search(r"fold(\d+)$", path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_folds(run_dir: Path) -> List[Path]:
    folds = [
        path
        for path in run_dir.glob("fold*")
        if path.is_dir()
        and fold_number(path) is not None
        and (path / "final_results" / "test_predictions.csv").exists()
    ]
    return sorted(folds, key=lambda path: fold_number(path))


def load_fold_curve(fold_dir: Path) -> FoldCurve:
    prediction_csv = fold_dir / "final_results" / "test_predictions.csv"
    labels, scores = load_labels_scores(prediction_csv)
    points, auroc = compute_roc(labels, scores)
    config = simple_yaml_values(
        fold_dir / "resolved_config.yaml",
        [
            "model_name",
            "backbone",
            "freeze_backbone",
            "clip_unfreeze_last_n_layers",
            "use_pathology_loss",
            "pathology_weight",
        ],
    )
    fold = fold_number(fold_dir)
    if fold is None:
        raise ValueError("Could not parse fold number from {path}".format(path=fold_dir))
    return FoldCurve(
        fold=fold,
        fold_dir=fold_dir,
        prediction_csv=prediction_csv,
        points=points,
        auroc=auroc,
        n_test=len(labels),
        n_positive=int(sum(labels)),
        n_negative=int(len(labels) - sum(labels)),
        config=config,
    )


def compress_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    by_fpr: Dict[float, float] = {}
    for fpr, tpr in points:
        by_fpr[fpr] = max(tpr, by_fpr.get(fpr, 0.0))
    compressed = sorted(by_fpr.items())
    if compressed[0][0] > 0.0:
        compressed.insert(0, (0.0, 0.0))
    if compressed[-1][0] < 1.0:
        compressed.append((1.0, 1.0))
    return compressed


def interpolate_tpr(points: Sequence[Tuple[float, float]], fpr: float) -> float:
    compressed = compress_points(points)
    if fpr <= compressed[0][0]:
        return compressed[0][1]
    for index in range(1, len(compressed)):
        x0, y0 = compressed[index - 1]
        x1, y1 = compressed[index]
        if fpr <= x1:
            if abs(x1 - x0) < 1e-12:
                return max(y0, y1)
            fraction = (fpr - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return compressed[-1][1]


def mean_curve(
    curves: Sequence[Sequence[Tuple[float, float]]],
    grid: Sequence[float],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    mean_points: List[Tuple[float, float]] = []
    lower_points: List[Tuple[float, float]] = []
    upper_points: List[Tuple[float, float]] = []
    for fpr in grid:
        values = [interpolate_tpr(curve, fpr) for curve in curves]
        mean_value = statistics.mean(values)
        sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_points.append((fpr, mean_value))
        lower_points.append((fpr, max(0.0, mean_value - sd_value)))
        upper_points.append((fpr, min(1.0, mean_value + sd_value)))
    mean_points[0] = (0.0, 0.0)
    lower_points[0] = (0.0, 0.0)
    upper_points[0] = (0.0, 0.0)
    mean_points[-1] = (1.0, 1.0)
    lower_points[-1] = (1.0, 1.0)
    upper_points[-1] = (1.0, 1.0)
    return mean_points, lower_points, upper_points


def trapezoid_auc(points: Sequence[Tuple[float, float]]) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def fmt_mean_sd(values: Sequence[float]) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return "{:.2f} +/- {:.2f}".format(mean, sd)


def polygon_points(
    lower_points: Sequence[Tuple[float, float]],
    upper_points: Sequence[Tuple[float, float]],
    map_x,
    map_y,
) -> str:
    points = list(upper_points) + list(reversed(lower_points))
    return " ".join("{:.2f},{:.2f}".format(map_x(x), map_y(y)) for x, y in points)


def pooled_auroc(folds: Sequence[FoldCurve]) -> float:
    labels: List[int] = []
    scores: List[float] = []
    for fold in folds:
        fold_labels, fold_scores = load_labels_scores(fold.prediction_csv)
        labels.extend(fold_labels)
        scores.extend(fold_scores)
    _, auroc = compute_roc(labels, scores)
    return auroc


def write_average_svg(
    path: Path,
    run_name: str,
    folds: Sequence[FoldCurve],
    mean_points: Sequence[Tuple[float, float]],
    lower_points: Sequence[Tuple[float, float]],
    upper_points: Sequence[Tuple[float, float]],
) -> None:
    width = 1500
    height = 1500
    margin_left = 130
    margin_right = 70
    margin_top = 150
    margin_bottom = 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    axis_color = "#2b2f33"
    grid_color = "#e3e5e8"
    curve_color = "#00897b"

    def map_x(value: float) -> float:
        return margin_left + value * plot_w

    def map_y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_h

    aurocs = [fold.auroc for fold in folds]
    legend_x = width - 630
    legend_y = height - 255
    legend_w = 600
    legend_h = 145

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(w=width, h=height))
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    title_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 44px; font-weight: 800; fill: #282828; letter-spacing: 2px'
    lines.append('<text x="{x}" y="48" text-anchor="middle" style="{style}">ROC Curve: Benin Test Set Performance</text>'.format(x=width / 2, style=title_style))
    lines.append('<text x="{x}" y="94" text-anchor="middle" style="{style}">Original Benin Model, Mean Across Folds</text>'.format(x=width / 2, style=title_style))

    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#b8b8b8" stroke-width="1.5"/>'.format(
        x=margin_left,
        y=margin_top,
        w=plot_w,
        h=plot_h,
    ))

    for i in range(6):
        value = i / 5.0
        x = map_x(value)
        y = map_y(value)
        lines.append('<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{color}" stroke-width="1" stroke-dasharray="3 4"/>'.format(
            x=x,
            top=margin_top,
            bottom=margin_top + plot_h,
            color=grid_color,
        ))
        lines.append('<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{color}" stroke-width="1" stroke-dasharray="3 4"/>'.format(
            left=margin_left,
            right=margin_left + plot_w,
            y=y,
            color=grid_color,
        ))

    lines.append('<polygon points="{points}" fill="{color}" opacity="0.12"/>'.format(
        points=polygon_points(lower_points, upper_points, map_x, map_y),
        color=curve_color,
    ))

    for fold in folds:
        lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8" stroke-opacity="0.25"/>'.format(
            points=svg_polyline(step_points(fold.points), map_x, map_y),
            color=curve_color,
        ))

    lines.append('<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#7f7f7f" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
        x0=map_x(0.0),
        y0=map_y(0.0),
        x1=map_x(1.0),
        y1=map_y(1.0),
    ))
    lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'.format(
        points=svg_polyline(mean_points, map_x, map_y),
        color=curve_color,
    ))

    lines.append('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2"/>'.format(x=margin_left, y1=margin_top, y2=margin_top + plot_h, color=axis_color))
    lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2"/>'.format(x1=margin_left, x2=margin_left + plot_w, y=margin_top + plot_h, color=axis_color))

    tick_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 30px; fill: #333333'
    for i in range(6):
        value = i / 5.0
        x = map_x(value)
        y = map_y(value)
        lines.append('<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"/>'.format(
            x=x,
            y1=margin_top + plot_h,
            y2=margin_top + plot_h + 8,
            color=axis_color,
        ))
        lines.append('<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" style="{style}">{value:.1f}</text>'.format(
            x=x,
            y=margin_top + plot_h + 45,
            style=tick_style,
            value=value,
        ))
        lines.append('<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/>'.format(
            x1=margin_left - 8,
            x2=margin_left,
            y=y,
            color=axis_color,
        ))
        lines.append('<text x="{x:.2f}" y="{y:.2f}" text-anchor="end" dominant-baseline="middle" style="{style}">{value:.1f}</text>'.format(
            x=margin_left - 20,
            y=y,
            style=tick_style,
            value=value,
        ))

    label_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 38px; font-weight: 800; fill: #282828; letter-spacing: 2px'
    lines.append('<text x="{x}" y="{y}" text-anchor="middle" style="{style}">False Positive Rate (1 - Specificity)</text>'.format(
        x=margin_left + plot_w / 2,
        y=height - 30,
        style=label_style,
    ))
    lines.append('<text x="42" y="{y}" text-anchor="middle" transform="rotate(-90 42 {y})" style="{style}">True Positive Rate (Sensitivity)</text>'.format(
        y=margin_top + plot_h / 2,
        style=label_style,
    ))

    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#000000" opacity="0.20"/>'.format(x=legend_x + 8, y=legend_y + 8, w=legend_w, h=legend_h))
    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="#c7c7c7" stroke-width="2"/>'.format(x=legend_x, y=legend_y, w=legend_w, h=legend_h))
    legend_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 27px; fill: #2d2d2d'
    legend_rows = [
        (curve_color, None, "Mean ROC: AUROC={}".format(fmt_mean_sd(aurocs))),
        (curve_color, "1 11", "Thin lines: folds 0-4"),
        ("#7f7f7f", "9 7", "Chance (AUROC=0.50)"),
        ("#666666", None, "Shading: +/- 1 SD across folds"),
    ]
    for offset, (color, dash, text) in enumerate(legend_rows):
        y = legend_y + 33 + offset * 34
        if offset < 3:
            dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
            opacity_attr = ' stroke-opacity="0.35"' if offset == 1 else ""
            lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="5"{dash}{opacity}/>'.format(
                x1=legend_x + 20,
                x2=legend_x + 85,
                y=y,
                color=color,
                dash=dash_attr,
                opacity=opacity_attr,
            ))
        else:
            lines.append('<rect x="{x}" y="{y}" width="65" height="12" fill="{color}" opacity="0.15"/>'.format(
                x=legend_x + 20,
                y=y - 6,
                color=curve_color,
            ))
        lines.append('<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{text}</text>'.format(
            x=legend_x + 110,
            y=y,
            style=legend_style,
            text=html.escape(text),
        ))

    lines.append('<!-- Run: {run_name} -->'.format(run_name=html.escape(run_name)))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=201, help="Number of FPR grid points for mean ROC interpolation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    fold_dirs = discover_folds(run_dir)
    if not fold_dirs:
        raise FileNotFoundError("No fold*/final_results/test_predictions.csv files found under {path}".format(path=run_dir))

    folds = [load_fold_curve(fold_dir) for fold_dir in fold_dirs]
    grid_size = max(int(args.grid_size), 2)
    grid = [index / float(grid_size - 1) for index in range(grid_size)]
    mean_points, lower_points, upper_points = mean_curve([fold.points for fold in folds], grid)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "benin_test_roc_mean_across_folds.svg"
    png_path = output_dir / "benin_test_roc_mean_across_folds.png"
    metrics_csv = output_dir / "benin_test_roc_fold_metrics.csv"
    mean_points_csv = output_dir / "benin_test_roc_mean_points.csv"
    summary_json = output_dir / "benin_test_roc_fold_average_summary.json"
    manifest_json = output_dir / "manifest.json"

    write_average_svg(svg_path, run_dir.name, folds, mean_points, lower_points, upper_points)
    png_written = convert_svg_to_png(svg_path, png_path)

    metric_rows: List[Dict[str, object]] = []
    for fold in folds:
        metric_rows.append(
            {
                "fold": fold.fold,
                "fold_dir": str(fold.fold_dir),
                "prediction_csv": str(fold.prediction_csv),
                "benin_test_auroc": fold.auroc,
                "n_test": fold.n_test,
                "n_positive": fold.n_positive,
                "n_negative": fold.n_negative,
                "freeze_backbone": fold.config.get("freeze_backbone"),
                "clip_unfreeze_last_n_layers": fold.config.get("clip_unfreeze_last_n_layers"),
                "use_pathology_loss": fold.config.get("use_pathology_loss"),
                "pathology_weight": fold.config.get("pathology_weight"),
            }
        )
    write_csv(metrics_csv, metric_rows)

    point_rows: List[Dict[str, object]] = []
    for index, fpr in enumerate(grid):
        point_rows.append(
            {
                "point_index": index,
                "fpr": fpr,
                "benin_test_mean_tpr": mean_points[index][1],
                "benin_test_lower_1sd_tpr": lower_points[index][1],
                "benin_test_upper_1sd_tpr": upper_points[index][1],
            }
        )
    write_csv(mean_points_csv, point_rows)

    aurocs = [fold.auroc for fold in folds]
    summary = {
        "analysis": "benin_source_roc_fold_average",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "n_folds": len(folds),
        "folds": [fold.fold for fold in folds],
        "fold_prediction_csvs": [str(fold.prediction_csv) for fold in folds],
        "benin_test_auroc_mean": statistics.mean(aurocs),
        "benin_test_auroc_sd": statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0,
        "benin_test_mean_curve_auroc": trapezoid_auc(mean_points),
        "benin_test_pooled_out_of_fold_auroc": pooled_auroc(folds),
        "n_test_total": sum(fold.n_test for fold in folds),
        "n_positive_total": sum(fold.n_positive for fold in folds),
        "n_negative_total": sum(fold.n_negative for fold in folds),
        "outputs": {
            "svg": str(svg_path),
            "png": str(png_path) if png_written else None,
            "fold_metrics_csv": str(metrics_csv),
            "mean_points_csv": str(mean_points_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
        },
        "interpretation_note": "The curve average is computed by interpolating each fold ROC to a shared FPR grid. The pooled AUROC concatenates the held-out test predictions across folds.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Run: {run}".format(run=run_dir))
    print("Folds: {folds}".format(folds=", ".join(str(fold.fold) for fold in folds)))
    print("Benin test AUROC: {value}".format(value=fmt_mean_sd(aurocs)))
    print("Pooled out-of-fold AUROC: {value:.4f}".format(value=summary["benin_test_pooled_out_of_fold_auroc"]))
    print("SVG: {path}".format(path=svg_path))
    if png_written:
        print("PNG: {path}".format(path=png_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
