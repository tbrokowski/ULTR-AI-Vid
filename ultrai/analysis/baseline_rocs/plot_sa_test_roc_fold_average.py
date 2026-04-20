#!/usr/bin/env python3
"""Plot mean transfer and source-retention ROC curves across source folds.

The five runs use the same SA train/val/test split, so this is a model/checkpoint
average rather than an independent SA cross-validation estimate. The mean ROC is
computed by interpolating each fold curve onto a shared FPR grid. The Benin
source-retention curve uses each fold's original Benin test split after SA
fine-tuning.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        DEFAULT_RUN_ROOT,
        DEFAULT_OUTPUT_ROOT,
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
        DEFAULT_RUN_ROOT,
        DEFAULT_OUTPUT_ROOT,
        compute_roc,
        convert_svg_to_png,
        load_labels_scores,
        simple_yaml_values,
        step_points,
        svg_polyline,
        write_csv,
    )


DEFAULT_FAMILY_GLOB = "20260418__finetune__benin_to_sa__src-fold*__baseline-fixednorm-retrained"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "20260418__baseline-fixednorm-retrained__source-fold-average"


class FoldCurve:
    def __init__(
        self,
        fold: int,
        run_name: str,
        zero_points: List[Tuple[float, float]],
        zero_auroc: float,
        fine_points: List[Tuple[float, float]],
        fine_auroc: float,
        source_points: List[Tuple[float, float]],
        source_auroc: float,
        config: Dict[str, object],
    ) -> None:
        self.fold = fold
        self.run_name = run_name
        self.zero_points = zero_points
        self.zero_auroc = zero_auroc
        self.fine_points = fine_points
        self.fine_auroc = fine_auroc
        self.source_points = source_points
        self.source_auroc = source_auroc
        self.config = config


def fold_number(run_name: str) -> Optional[int]:
    match = re.search(r"src-fold(\d+)", run_name)
    if not match:
        return None
    return int(match.group(1))


def discover_runs(run_root: Path, family_glob: str) -> List[Path]:
    runs = [path for path in run_root.glob(family_glob) if path.is_dir()]
    runs = [
        path
        for path in runs
        if (path / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv").exists()
        and (path / "results" / "finetune_full" / "finetuned_full_patient_predictions.csv").exists()
        and (path / "results" / "source_test_evaluation" / "finetuned_source_test_patient_predictions.csv").exists()
        and fold_number(path.name) is not None
    ]
    return sorted(runs, key=lambda path: fold_number(path.name))


def load_fold_curve(run_dir: Path) -> FoldCurve:
    zero_csv = run_dir / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv"
    fine_csv = run_dir / "results" / "finetune_full" / "finetuned_full_patient_predictions.csv"
    source_csv = run_dir / "results" / "source_test_evaluation" / "finetuned_source_test_patient_predictions.csv"
    zero_labels, zero_scores = load_labels_scores(zero_csv)
    fine_labels, fine_scores = load_labels_scores(fine_csv)
    source_labels, source_scores = load_labels_scores(source_csv)
    zero_points, zero_auroc = compute_roc(zero_labels, zero_scores)
    fine_points, fine_auroc = compute_roc(fine_labels, fine_scores)
    source_points, source_auroc = compute_roc(source_labels, source_scores)
    config = simple_yaml_values(
        run_dir / "resolved_finetune_config.yaml",
        ["freeze_backbone", "clip_unfreeze_last_n_layers", "val_size", "oversample_positive_class"],
    )
    fold = fold_number(run_dir.name)
    if fold is None:
        raise ValueError("Could not parse source fold from {name}".format(name=run_dir.name))
    return FoldCurve(fold, run_dir.name, zero_points, zero_auroc, fine_points, fine_auroc, source_points, source_auroc, config)


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


def mean_curve(curves: Sequence[Sequence[Tuple[float, float]]], grid: Sequence[float]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
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


def write_average_svg(
    path: Path,
    folds: Sequence[FoldCurve],
    zero_mean: Sequence[Tuple[float, float]],
    zero_lower: Sequence[Tuple[float, float]],
    zero_upper: Sequence[Tuple[float, float]],
    fine_mean: Sequence[Tuple[float, float]],
    fine_lower: Sequence[Tuple[float, float]],
    fine_upper: Sequence[Tuple[float, float]],
    source_mean: Sequence[Tuple[float, float]],
    source_lower: Sequence[Tuple[float, float]],
    source_upper: Sequence[Tuple[float, float]],
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
    red = "#d62728"
    green = "#2ca02c"
    blue = "#1f77b4"

    def map_x(value: float) -> float:
        return margin_left + value * plot_w

    def map_y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_h

    zero_aurocs = [fold.zero_auroc for fold in folds]
    fine_aurocs = [fold.fine_auroc for fold in folds]
    source_aurocs = [fold.source_auroc for fold in folds]
    legend_x = width - 705
    legend_y = height - 305
    legend_w = 675
    legend_h = 195

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(w=width, h=height))
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    title_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 44px; font-weight: 800; fill: #282828; letter-spacing: 2px'
    lines.append('<text x="{x}" y="48" text-anchor="middle" style="{style}">ROC Curves: Transfer and Source Retention</text>'.format(x=width / 2, style=title_style))
    lines.append('<text x="{x}" y="94" text-anchor="middle" style="{style}">Mean Across Benin Source Folds</text>'.format(x=width / 2, style=title_style))

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

    lines.append('<polygon points="{points}" fill="{color}" opacity="0.10"/>'.format(
        points=polygon_points(zero_lower, zero_upper, map_x, map_y),
        color=red,
    ))
    lines.append('<polygon points="{points}" fill="{color}" opacity="0.10"/>'.format(
        points=polygon_points(fine_lower, fine_upper, map_x, map_y),
        color=green,
    ))
    lines.append('<polygon points="{points}" fill="{color}" opacity="0.09"/>'.format(
        points=polygon_points(source_lower, source_upper, map_x, map_y),
        color=blue,
    ))

    for fold in folds:
        lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-opacity="0.22" stroke-dasharray="16 10"/>'.format(
            points=svg_polyline(step_points(fold.zero_points), map_x, map_y),
            color=red,
        ))
        lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-opacity="0.22"/>'.format(
            points=svg_polyline(step_points(fold.fine_points), map_x, map_y),
            color=green,
        ))
        lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-opacity="0.20" stroke-dasharray="4 8"/>'.format(
            points=svg_polyline(step_points(fold.source_points), map_x, map_y),
            color=blue,
        ))

    lines.append('<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#7f7f7f" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
        x0=map_x(0.0),
        y0=map_y(0.0),
        x1=map_x(1.0),
        y1=map_y(1.0),
    ))
    lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="25 14"/>'.format(
        points=svg_polyline(zero_mean, map_x, map_y),
        color=red,
    ))
    lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'.format(
        points=svg_polyline(fine_mean, map_x, map_y),
        color=green,
    ))
    lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="4 12"/>'.format(
        points=svg_polyline(source_mean, map_x, map_y),
        color=blue,
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
        (red, "25 14", "Benin zero-shot on SA: AUROC={}".format(fmt_mean_sd(zero_aurocs))),
        (green, None, "SA fine-tuned on SA: AUROC={}".format(fmt_mean_sd(fine_aurocs))),
        (blue, "4 12", "SA fine-tuned on Benin: AUROC={}".format(fmt_mean_sd(source_aurocs))),
        ("#7f7f7f", "9 7", "Chance (AUROC=0.50)"),
        ("#666666", None, "Shading: +/- 1 SD across source folds"),
    ]
    for offset, (color, dash, text) in enumerate(legend_rows):
        y = legend_y + 29 + offset * 34
        if offset < 4:
            dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
            lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="5"{dash}/>'.format(
                x1=legend_x + 20,
                x2=legend_x + 85,
                y=y,
                color=color,
                dash=dash_attr,
            ))
        else:
            lines.append('<rect x="{x}" y="{y}" width="65" height="12" fill="#999999" opacity="0.18"/>'.format(
                x=legend_x + 20,
                y=y - 6,
            ))
        lines.append('<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{text}</text>'.format(
            x=legend_x + 110,
            y=y,
            style=legend_style,
            text=html.escape(text),
        ))

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--family-glob", default=DEFAULT_FAMILY_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=201, help="Number of FPR grid points for mean ROC interpolation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = discover_runs(args.run_root, args.family_glob)
    if not runs:
        raise FileNotFoundError("No completed runs matched {pattern} under {root}".format(pattern=args.family_glob, root=args.run_root))
    folds = [load_fold_curve(run) for run in runs]

    grid_size = max(int(args.grid_size), 2)
    grid = [index / float(grid_size - 1) for index in range(grid_size)]
    zero_mean, zero_lower, zero_upper = mean_curve([fold.zero_points for fold in folds], grid)
    fine_mean, fine_lower, fine_upper = mean_curve([fold.fine_points for fold in folds], grid)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "transfer_retention_roc_mean_across_source_folds.svg"
    png_path = output_dir / "transfer_retention_roc_mean_across_source_folds.png"
    metrics_csv = output_dir / "transfer_retention_roc_fold_metrics.csv"
    mean_points_csv = output_dir / "transfer_retention_roc_mean_points.csv"
    summary_json = output_dir / "transfer_retention_roc_fold_average_summary.json"
    manifest_json = output_dir / "manifest.json"

    source_mean, source_lower, source_upper = mean_curve([fold.source_points for fold in folds], grid)

    write_average_svg(
        svg_path,
        folds,
        zero_mean,
        zero_lower,
        zero_upper,
        fine_mean,
        fine_lower,
        fine_upper,
        source_mean,
        source_lower,
        source_upper,
    )
    png_written = convert_svg_to_png(svg_path, png_path)

    metric_rows: List[Dict[str, object]] = []
    for fold in folds:
        metric_rows.append(
            {
                "fold": fold.fold,
                "run_name": fold.run_name,
                "zero_shot_auroc": fold.zero_auroc,
                "sa_fine_tuned_auroc": fold.fine_auroc,
                "source_benin_after_sa_finetune_auroc": fold.source_auroc,
                "delta_auroc": fold.fine_auroc - fold.zero_auroc,
                "source_retention_delta_vs_zero_shot_sa": fold.source_auroc - fold.zero_auroc,
                "freeze_backbone": fold.config.get("freeze_backbone"),
                "clip_unfreeze_last_n_layers": fold.config.get("clip_unfreeze_last_n_layers"),
            }
        )
    write_csv(metrics_csv, metric_rows)

    point_rows: List[Dict[str, object]] = []
    for index, fpr in enumerate(grid):
        point_rows.append(
            {
                "point_index": index,
                "fpr": fpr,
                "zero_shot_mean_tpr": zero_mean[index][1],
                "zero_shot_lower_1sd_tpr": zero_lower[index][1],
                "zero_shot_upper_1sd_tpr": zero_upper[index][1],
                "sa_fine_tuned_mean_tpr": fine_mean[index][1],
                "sa_fine_tuned_lower_1sd_tpr": fine_lower[index][1],
                "sa_fine_tuned_upper_1sd_tpr": fine_upper[index][1],
                "source_benin_after_sa_finetune_mean_tpr": source_mean[index][1],
                "source_benin_after_sa_finetune_lower_1sd_tpr": source_lower[index][1],
                "source_benin_after_sa_finetune_upper_1sd_tpr": source_upper[index][1],
            }
        )
    write_csv(mean_points_csv, point_rows)

    zero_aurocs = [fold.zero_auroc for fold in folds]
    fine_aurocs = [fold.fine_auroc for fold in folds]
    source_aurocs = [fold.source_auroc for fold in folds]
    deltas = [fold.fine_auroc - fold.zero_auroc for fold in folds]
    summary = {
        "analysis": "baseline_roc_fold_average",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_root": str(args.run_root),
        "family_glob": args.family_glob,
        "n_folds": len(folds),
        "folds": [fold.fold for fold in folds],
        "runs": [str(args.run_root / fold.run_name) for fold in folds],
        "zero_shot_auroc_mean": statistics.mean(zero_aurocs),
        "zero_shot_auroc_sd": statistics.stdev(zero_aurocs) if len(zero_aurocs) > 1 else 0.0,
        "zero_shot_mean_curve_auroc": trapezoid_auc(zero_mean),
        "sa_fine_tuned_auroc_mean": statistics.mean(fine_aurocs),
        "sa_fine_tuned_auroc_sd": statistics.stdev(fine_aurocs) if len(fine_aurocs) > 1 else 0.0,
        "sa_fine_tuned_mean_curve_auroc": trapezoid_auc(fine_mean),
        "source_benin_after_sa_finetune_auroc_mean": statistics.mean(source_aurocs),
        "source_benin_after_sa_finetune_auroc_sd": statistics.stdev(source_aurocs) if len(source_aurocs) > 1 else 0.0,
        "source_benin_after_sa_finetune_mean_curve_auroc": trapezoid_auc(source_mean),
        "delta_auroc_mean": statistics.mean(deltas),
        "delta_auroc_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "outputs": {
            "svg": str(svg_path),
            "png": str(png_path) if png_written else None,
            "fold_metrics_csv": str(metrics_csv),
            "mean_points_csv": str(mean_points_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
        },
        "interpretation_note": "SA curves are means across five Benin source-fold checkpoints evaluated on the same SA split. The Benin source-retention curve evaluates each SA-fine-tuned checkpoint on its original Benin source test fold.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Folds: {folds}".format(folds=", ".join(str(fold.fold) for fold in folds)))
    print("Benin zero-shot AUROC: {value}".format(value=fmt_mean_sd(zero_aurocs)))
    print("SA fine-tuned AUROC: {value}".format(value=fmt_mean_sd(fine_aurocs)))
    print("SA fine-tuned on Benin source AUROC: {value}".format(value=fmt_mean_sd(source_aurocs)))
    print("SVG: {path}".format(path=svg_path))
    if png_written:
        print("PNG: {path}".format(path=png_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
