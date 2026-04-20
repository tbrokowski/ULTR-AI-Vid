#!/usr/bin/env python3
"""Plot ROC and precision-recall curves for the SA-SimCLR transfer run.

Defaults target the SA-SimCLR backbone trained on South African videos and then
used for Benin source training plus Benin-to-SA fine-tuning. The figures average
curves across the five Benin source folds by interpolating each fold onto a
shared grid. Thin curves show individual folds; bold curves show the fold mean.
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


DEFAULT_SOURCE_RUN_DIR = SCRATCH_ROOT / "runs" / "train" / "benin" / "20260418__train__benin__all-folds__sa-simclr-temporal-v1"
DEFAULT_FINETUNE_RUN_ROOT = SCRATCH_ROOT / "runs" / "finetune" / "benin_to_sa"
DEFAULT_FINETUNE_FAMILY_GLOB = "20260418__finetune__benin_to_sa__src-fold*__sa-simclr-temporal-v1-freezeclip"
DEFAULT_OUTPUT_DIR = (
    SCRATCH_ROOT
    / "runs"
    / "baseline_roc_comparison"
    / "benin_to_sa"
    / "20260418__sa-simclr-temporal-v1-freezeclip__source-fold-average"
)


class Evaluation:
    def __init__(
        self,
        key: str,
        label: str,
        cohort: str,
        color: str,
        dash: Optional[str],
    ) -> None:
        self.key = key
        self.label = label
        self.cohort = cohort
        self.color = color
        self.dash = dash


class FoldEvaluation:
    def __init__(
        self,
        fold: int,
        evaluation: Evaluation,
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
        self.evaluation = evaluation
        self.prediction_csv = prediction_csv
        self.roc_points = roc_points
        self.auroc = auroc
        self.pr_points = pr_points
        self.auprc = auprc
        self.n_samples = n_samples
        self.n_positive = n_positive
        self.n_negative = n_negative


class CurveSummary:
    def __init__(
        self,
        evaluation: Evaluation,
        fold_evaluations: List[FoldEvaluation],
        roc_mean: List[Tuple[float, float]],
        roc_lower: List[Tuple[float, float]],
        roc_upper: List[Tuple[float, float]],
        pr_mean: List[Tuple[float, float]],
        pr_lower: List[Tuple[float, float]],
        pr_upper: List[Tuple[float, float]],
    ) -> None:
        self.evaluation = evaluation
        self.fold_evaluations = fold_evaluations
        self.roc_mean = roc_mean
        self.roc_lower = roc_lower
        self.roc_upper = roc_upper
        self.pr_mean = pr_mean
        self.pr_lower = pr_lower
        self.pr_upper = pr_upper


EVALUATIONS = [
    Evaluation("benin_source", "Benin source", "Benin", "#00897b", None),
    Evaluation("zero_shot_sa", "Zero-shot SA", "South Africa", "#d62728", "25 14"),
    Evaluation("sa_fine_tuned", "SA fine-tuned", "South Africa", "#2ca02c", None),
    Evaluation("source_after_sa_ft", "Source after SA FT", "Benin", "#1f77b4", "4 12"),
]


def fold_number_from_name(name: str) -> Optional[int]:
    src_match = re.search(r"src-fold(\d+)", name)
    if src_match:
        return int(src_match.group(1))
    fold_match = re.search(r"fold(\d+)$", name)
    if fold_match:
        return int(fold_match.group(1))
    return None


def discover_finetune_runs(run_root: Path, family_glob: str) -> Dict[int, Path]:
    runs: Dict[int, Path] = {}
    for run_dir in run_root.glob(family_glob):
        fold = fold_number_from_name(run_dir.name)
        if fold is None:
            continue
        required = [
            run_dir / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv",
            run_dir / "results" / "finetune_full" / "finetuned_full_patient_predictions.csv",
            run_dir / "results" / "source_test_evaluation" / "finetuned_source_test_patient_predictions.csv",
        ]
        if all(path.exists() for path in required):
            runs[fold] = run_dir
    return dict(sorted(runs.items()))


def source_fold_dirs(source_run_dir: Path) -> Dict[int, Path]:
    folds: Dict[int, Path] = {}
    for fold_dir in source_run_dir.glob("fold*"):
        fold = fold_number_from_name(fold_dir.name)
        prediction_csv = fold_dir / "final_results" / "test_predictions.csv"
        if fold is not None and prediction_csv.exists():
            folds[fold] = fold_dir
    return dict(sorted(folds.items()))


def compute_pr(labels: Sequence[int], scores: Sequence[float]) -> Tuple[List[Tuple[float, float]], float]:
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


def load_evaluation(fold: int, evaluation: Evaluation, prediction_csv: Path) -> FoldEvaluation:
    labels, scores = load_labels_scores(prediction_csv)
    roc_points, auroc = compute_roc(labels, scores)
    pr_points, auprc = compute_pr(labels, scores)
    return FoldEvaluation(
        fold=fold,
        evaluation=evaluation,
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
    for x, y in points:
        by_x[x] = max(y, by_x.get(x, 0.0))
    compressed = sorted(by_x.items())
    if compressed[0][0] > 0.0:
        compressed.insert(0, (0.0, compressed[0][1]))
    if compressed[-1][0] < 1.0:
        compressed.append((1.0, compressed[-1][1]))
    return compressed


def interpolate_y(points: Sequence[Tuple[float, float]], x_value: float) -> float:
    compressed = compress_points(points)
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


def mean_curve(
    curves: Sequence[Sequence[Tuple[float, float]]],
    grid: Sequence[float],
    force_endpoints: bool,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    mean_points: List[Tuple[float, float]] = []
    lower_points: List[Tuple[float, float]] = []
    upper_points: List[Tuple[float, float]] = []
    for x_value in grid:
        values = [interpolate_y(curve, x_value) for curve in curves]
        mean_value = statistics.mean(values)
        sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_points.append((x_value, mean_value))
        lower_points.append((x_value, max(0.0, mean_value - sd_value)))
        upper_points.append((x_value, min(1.0, mean_value + sd_value)))
    if force_endpoints:
        mean_points[0] = (0.0, 0.0)
        lower_points[0] = (0.0, 0.0)
        upper_points[0] = (0.0, 0.0)
        mean_points[-1] = (1.0, 1.0)
        lower_points[-1] = (1.0, 1.0)
        upper_points[-1] = (1.0, 1.0)
    return mean_points, lower_points, upper_points


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


def prevalence(fold_evaluations: Sequence[FoldEvaluation]) -> float:
    n_positive = sum(item.n_positive for item in fold_evaluations)
    n_total = sum(item.n_samples for item in fold_evaluations)
    return n_positive / float(n_total)


def write_multicurve_svg(
    path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    summaries: Sequence[CurveSummary],
    metric_attr: str,
    metric_label: str,
    mean_attr: str,
    lower_attr: str,
    upper_attr: str,
    fold_points_attr: str,
    chance: str,
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

    def map_x(value: float) -> float:
        return margin_left + value * plot_w

    def map_y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_h

    if chance == "pr":
        legend_x = margin_left + plot_w - 640
        legend_y = margin_top - 24
    else:
        legend_x = width - 690
        legend_y = height - 285
    legend_w = 660
    legend_h = 225 if chance == "pr" else 205

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(w=width, h=height))
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    title_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 44px; font-weight: 800; fill: #282828; letter-spacing: 2px'
    lines.append('<text x="{x}" y="48" text-anchor="middle" style="{style}">{title}</text>'.format(x=width / 2, style=title_style, title=html.escape(title)))
    lines.append('<text x="{x}" y="94" text-anchor="middle" style="{style}">{subtitle}</text>'.format(x=width / 2, style=title_style, subtitle=html.escape(subtitle)))

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

    if chance == "roc":
        lines.append('<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#7f7f7f" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
            x0=map_x(0.0),
            y0=map_y(0.0),
            x1=map_x(1.0),
            y1=map_y(1.0),
        ))
    elif chance == "pr":
        sa_summary = next(item for item in summaries if item.evaluation.key == "zero_shot_sa")
        benin_summary = next(item for item in summaries if item.evaluation.key == "benin_source")
        sa_prev = prevalence(sa_summary.fold_evaluations)
        benin_prev = prevalence(benin_summary.fold_evaluations)
        lines.append('<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="#7f7f7f" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
            x0=map_x(0.0),
            x1=map_x(1.0),
            y=map_y(sa_prev),
        ))
        lines.append('<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="#7f7f7f" stroke-width="2.5" stroke-dasharray="2 9"/>'.format(
            x0=map_x(0.0),
            x1=map_x(1.0),
            y=map_y(benin_prev),
        ))

    for summary in summaries:
        lower_points = getattr(summary, lower_attr)
        upper_points = getattr(summary, upper_attr)
        lines.append('<polygon points="{points}" fill="{color}" opacity="0.055"/>'.format(
            points=polygon_points(lower_points, upper_points, map_x, map_y),
            color=summary.evaluation.color,
        ))

    for summary in summaries:
        for fold_eval in summary.fold_evaluations:
            dash_attr = ' stroke-dasharray="{dash}"'.format(dash=summary.evaluation.dash) if summary.evaluation.dash else ""
            lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-opacity="0.16"{dash}/>'.format(
                points=svg_polyline(step_points(getattr(fold_eval, fold_points_attr)), map_x, map_y),
                color=summary.evaluation.color,
                dash=dash_attr,
            ))

    for summary in summaries:
        dash_attr = ' stroke-dasharray="{dash}"'.format(dash=summary.evaluation.dash) if summary.evaluation.dash else ""
        lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"{dash}/>'.format(
            points=svg_polyline(getattr(summary, mean_attr), map_x, map_y),
            color=summary.evaluation.color,
            dash=dash_attr,
        ))

    lines.append('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2"/>'.format(x=margin_left, y1=margin_top, y2=margin_top + plot_h, color=axis_color))
    lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2"/>'.format(x1=margin_left, x2=margin_left + plot_w, y=margin_top + plot_h, color=axis_color))

    tick_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 30px; fill: #333333'
    for i in range(6):
        value = i / 5.0
        x = map_x(value)
        y = map_y(value)
        lines.append('<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"/>'.format(x=x, y1=margin_top + plot_h, y2=margin_top + plot_h + 8, color=axis_color))
        lines.append('<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" style="{style}">{value:.1f}</text>'.format(x=x, y=margin_top + plot_h + 45, style=tick_style, value=value))
        lines.append('<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/>'.format(x1=margin_left - 8, x2=margin_left, y=y, color=axis_color))
        lines.append('<text x="{x:.2f}" y="{y:.2f}" text-anchor="end" dominant-baseline="middle" style="{style}">{value:.1f}</text>'.format(x=margin_left - 20, y=y, style=tick_style, value=value))

    label_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 38px; font-weight: 800; fill: #282828; letter-spacing: 2px'
    lines.append('<text x="{x}" y="{y}" text-anchor="middle" style="{style}">{label}</text>'.format(
        x=margin_left + plot_w / 2,
        y=height - 30,
        style=label_style,
        label=html.escape(x_label),
    ))
    lines.append('<text x="42" y="{y}" text-anchor="middle" transform="rotate(-90 42 {y})" style="{style}">{label}</text>'.format(
        y=margin_top + plot_h / 2,
        style=label_style,
        label=html.escape(y_label),
    ))

    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#000000" opacity="0.18"/>'.format(x=legend_x + 8, y=legend_y + 8, w=legend_w, h=legend_h))
    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#c7c7c7" stroke-width="2"/>'.format(x=legend_x, y=legend_y, w=legend_w, h=legend_h))

    legend_font_size = 21 if chance == "pr" else 24
    legend_style = 'font-family: Arial, Helvetica, sans-serif; font-size: {size}px; fill: #2d2d2d'.format(size=legend_font_size)
    legend_rows: List[Tuple[str, Optional[str], str]] = []
    for summary in summaries:
        values = [getattr(fold_eval, metric_attr) for fold_eval in summary.fold_evaluations]
        legend_rows.append((summary.evaluation.color, summary.evaluation.dash, "{label}: {metric}={value}".format(
            label=summary.evaluation.label,
            metric=metric_label,
            value=fmt_mean_sd(values),
        )))
    if chance == "roc":
        legend_rows.append(("#7f7f7f", "9 7", "Chance: AUROC=0.50"))
    elif chance == "pr":
        sa_summary = next(item for item in summaries if item.evaluation.key == "zero_shot_sa")
        benin_summary = next(item for item in summaries if item.evaluation.key == "benin_source")
        legend_rows.append(("#7f7f7f", "9 7", "SA prevalence={:.2f}".format(prevalence(sa_summary.fold_evaluations))))
        legend_rows.append(("#7f7f7f", "2 9", "Benin prevalence={:.2f}".format(prevalence(benin_summary.fold_evaluations))))
    legend_rows.append(("#666666", None, "Thin lines: folds; shading: +/- 1 SD"))

    row_gap = 27 if chance == "pr" else 31
    for offset, (color, dash, text) in enumerate(legend_rows):
        y = legend_y + 29 + offset * row_gap
        if offset < len(legend_rows) - 1:
            dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
            lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="5"{dash}/>'.format(
                x1=legend_x + 20,
                x2=legend_x + 84,
                y=y,
                color=color,
                dash=dash_attr,
            ))
        else:
            lines.append('<rect x="{x}" y="{y}" width="64" height="12" fill="#999999" opacity="0.18"/>'.format(x=legend_x + 20, y=y - 6))
        lines.append('<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{text}</text>'.format(
            x=legend_x + 105,
            y=y,
            style=legend_style,
            text=html.escape(text),
        ))

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summaries(source_run_dir: Path, finetune_runs: Dict[int, Path], grid: Sequence[float]) -> List[CurveSummary]:
    source_folds = source_fold_dirs(source_run_dir)
    common_folds = sorted(set(source_folds.keys()) & set(finetune_runs.keys()))
    if len(common_folds) != 5:
        raise ValueError("Expected 5 common folds, found {folds}".format(folds=common_folds))

    by_key: Dict[str, List[FoldEvaluation]] = {evaluation.key: [] for evaluation in EVALUATIONS}
    for fold in common_folds:
        csv_paths = {
            "benin_source": source_folds[fold] / "final_results" / "test_predictions.csv",
            "zero_shot_sa": finetune_runs[fold] / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv",
            "sa_fine_tuned": finetune_runs[fold] / "results" / "finetune_full" / "finetuned_full_patient_predictions.csv",
            "source_after_sa_ft": finetune_runs[fold] / "results" / "source_test_evaluation" / "finetuned_source_test_patient_predictions.csv",
        }
        for evaluation in EVALUATIONS:
            by_key[evaluation.key].append(load_evaluation(fold, evaluation, csv_paths[evaluation.key]))

    summaries: List[CurveSummary] = []
    for evaluation in EVALUATIONS:
        fold_evaluations = by_key[evaluation.key]
        roc_mean, roc_lower, roc_upper = mean_curve([item.roc_points for item in fold_evaluations], grid, force_endpoints=True)
        pr_mean, pr_lower, pr_upper = mean_curve([item.pr_points for item in fold_evaluations], grid, force_endpoints=False)
        summaries.append(CurveSummary(evaluation, fold_evaluations, roc_mean, roc_lower, roc_upper, pr_mean, pr_lower, pr_upper))
    return summaries


def write_metrics(output_dir: Path, summaries: Sequence[CurveSummary], grid: Sequence[float]) -> Dict[str, object]:
    metrics_rows: List[Dict[str, object]] = []
    for summary in summaries:
        for fold_eval in summary.fold_evaluations:
            metrics_rows.append(
                {
                    "fold": fold_eval.fold,
                    "evaluation": summary.evaluation.key,
                    "label": summary.evaluation.label,
                    "cohort": summary.evaluation.cohort,
                    "prediction_csv": str(fold_eval.prediction_csv),
                    "auroc": fold_eval.auroc,
                    "auprc": fold_eval.auprc,
                    "n_samples": fold_eval.n_samples,
                    "n_positive": fold_eval.n_positive,
                    "n_negative": fold_eval.n_negative,
                }
            )
    metrics_csv = output_dir / "sa_simclr_transfer_fold_metrics.csv"
    write_csv(metrics_csv, metrics_rows)

    roc_rows: List[Dict[str, object]] = []
    pr_rows: List[Dict[str, object]] = []
    for summary in summaries:
        for index, x_value in enumerate(grid):
            roc_rows.append(
                {
                    "evaluation": summary.evaluation.key,
                    "label": summary.evaluation.label,
                    "point_index": index,
                    "fpr": x_value,
                    "mean_tpr": summary.roc_mean[index][1],
                    "lower_1sd_tpr": summary.roc_lower[index][1],
                    "upper_1sd_tpr": summary.roc_upper[index][1],
                }
            )
            pr_rows.append(
                {
                    "evaluation": summary.evaluation.key,
                    "label": summary.evaluation.label,
                    "point_index": index,
                    "recall": x_value,
                    "mean_precision": summary.pr_mean[index][1],
                    "lower_1sd_precision": summary.pr_lower[index][1],
                    "upper_1sd_precision": summary.pr_upper[index][1],
                }
            )
    roc_points_csv = output_dir / "sa_simclr_transfer_roc_mean_points.csv"
    pr_points_csv = output_dir / "sa_simclr_transfer_pr_mean_points.csv"
    write_csv(roc_points_csv, roc_rows)
    write_csv(pr_points_csv, pr_rows)

    metric_summary: Dict[str, object] = {}
    for summary in summaries:
        aurocs = [item.auroc for item in summary.fold_evaluations]
        auprcs = [item.auprc for item in summary.fold_evaluations]
        metric_summary[summary.evaluation.key] = {
            "label": summary.evaluation.label,
            "cohort": summary.evaluation.cohort,
            "auroc_mean": statistics.mean(aurocs),
            "auroc_sd": statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0,
            "auprc_mean": statistics.mean(auprcs),
            "auprc_sd": statistics.stdev(auprcs) if len(auprcs) > 1 else 0.0,
            "n_samples_total": sum(item.n_samples for item in summary.fold_evaluations),
            "n_positive_total": sum(item.n_positive for item in summary.fold_evaluations),
            "n_negative_total": sum(item.n_negative for item in summary.fold_evaluations),
            "prevalence": prevalence(summary.fold_evaluations),
        }

    return {
        "fold_metrics_csv": str(metrics_csv),
        "roc_mean_points_csv": str(roc_points_csv),
        "pr_mean_points_csv": str(pr_points_csv),
        "metrics": metric_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--finetune-run-root", type=Path, default=DEFAULT_FINETUNE_RUN_ROOT)
    parser.add_argument("--finetune-family-glob", default=DEFAULT_FINETUNE_FAMILY_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=201, help="Number of grid points for mean curve interpolation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    finetune_runs = discover_finetune_runs(args.finetune_run_root, args.finetune_family_glob)
    if not finetune_runs:
        raise FileNotFoundError("No completed fine-tune runs matched {pattern}".format(pattern=args.finetune_family_glob))

    grid_size = max(int(args.grid_size), 2)
    grid = [index / float(grid_size - 1) for index in range(grid_size)]
    summaries = build_summaries(source_run_dir, finetune_runs, grid)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_svg = output_dir / "sa_simclr_transfer_roc_mean_across_folds.svg"
    roc_png = output_dir / "sa_simclr_transfer_roc_mean_across_folds.png"
    pr_svg = output_dir / "sa_simclr_transfer_pr_mean_across_folds.svg"
    pr_png = output_dir / "sa_simclr_transfer_pr_mean_across_folds.png"
    summary_json = output_dir / "sa_simclr_transfer_curve_summary.json"
    manifest_json = output_dir / "manifest.json"

    write_multicurve_svg(
        roc_svg,
        "SA-SimCLR Transfer: ROC Curves",
        "Mean Across Five Benin Source Folds",
        "False Positive Rate (1 - Specificity)",
        "True Positive Rate (Sensitivity)",
        summaries,
        metric_attr="auroc",
        metric_label="AUROC",
        mean_attr="roc_mean",
        lower_attr="roc_lower",
        upper_attr="roc_upper",
        fold_points_attr="roc_points",
        chance="roc",
    )
    write_multicurve_svg(
        pr_svg,
        "SA-SimCLR Transfer: Precision-Recall",
        "Mean Across Five Benin Source Folds",
        "Recall (Sensitivity)",
        "Precision",
        summaries,
        metric_attr="auprc",
        metric_label="AUPRC",
        mean_attr="pr_mean",
        lower_attr="pr_lower",
        upper_attr="pr_upper",
        fold_points_attr="pr_points",
        chance="pr",
    )
    roc_png_written = convert_svg_to_png(roc_svg, roc_png)
    pr_png_written = convert_svg_to_png(pr_svg, pr_png)

    config = simple_yaml_values(
        next(iter(finetune_runs.values())) / "resolved_finetune_config.yaml",
        ["freeze_backbone", "clip_unfreeze_last_n_layers", "model_name", "val_size", "oversample_positive_class"],
    )
    output_info = write_metrics(output_dir, summaries, grid)
    summary = {
        "analysis": "sa_simclr_transfer_curves",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(source_run_dir),
        "finetune_run_root": str(args.finetune_run_root),
        "finetune_family_glob": args.finetune_family_glob,
        "finetune_runs": {str(fold): str(path) for fold, path in finetune_runs.items()},
        "folds": [item.fold for item in summaries[0].fold_evaluations],
        "config_sample": config,
        "outputs": {
            "roc_svg": str(roc_svg),
            "roc_png": str(roc_png) if roc_png_written else None,
            "pr_svg": str(pr_svg),
            "pr_png": str(pr_png) if pr_png_written else None,
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
            **{key: value for key, value in output_info.items() if key != "metrics"},
        },
        "metrics": output_info["metrics"],
        "interpretation_note": "SA curves are model/checkpoint averages over five Benin source-fold checkpoints evaluated on the same SA test split. Benin source curves use held-out Benin test folds. Source-after-FT evaluates each SA-fine-tuned checkpoint back on its corresponding Benin source test fold.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Fine-tune runs: {runs}".format(runs=", ".join(path.name for path in finetune_runs.values())))
    for item in EVALUATIONS:
        values = summary["metrics"][item.key]
        print(
            "{label}: AUROC {auroc:.2f}; AUPRC {auprc:.2f}".format(
                label=values["label"],
                auroc=values["auroc_mean"],
                auprc=values["auprc_mean"],
            )
        )
    print("ROC PNG: {path}".format(path=roc_png if roc_png_written else "not written"))
    print("PR PNG: {path}".format(path=pr_png if pr_png_written else "not written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
