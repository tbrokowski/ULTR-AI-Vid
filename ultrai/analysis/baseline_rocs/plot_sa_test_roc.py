#!/usr/bin/env python3
"""Plot SA test ROC curves for Benin zero-shot vs SA fine-tuned baselines.

This intentionally avoids pandas/matplotlib so it runs in the lean CSCS
analysis environment. It reads patient-level prediction CSVs from a finetune
run, writes a slide-ready SVG, converts it to PNG when ImageMagick is present,
and records a manifest with the run/config provenance.
"""

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRATCH_ROOT = Path("/capstor/scratch/cscs/lxflk/ULTR-AI-Vid")
DEFAULT_RUN_ROOT = SCRATCH_ROOT / "runs" / "finetune" / "benin_to_sa"
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / "runs" / "baseline_roc_comparison" / "benin_to_sa"


class RocCurve:
    def __init__(self, label: str, color: str, dash: Optional[str], points: List[Tuple[float, float]], auroc: float) -> None:
        self.label = label
        self.color = color
        self.dash = dash
        self.points = points
        self.auroc = auroc


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_label(value: object) -> Optional[int]:
    parsed = parse_float(value)
    if parsed is None:
        return None
    if abs(parsed - 1.0) < 1e-8:
        return 1
    if abs(parsed) < 1e-8:
        return 0
    return None


def pick_column(columns: Sequence[str], candidates: Sequence[str], purpose: str) -> str:
    lower_to_actual = {column.lower(): column for column in columns}
    for candidate in candidates:
        actual = lower_to_actual.get(candidate.lower())
        if actual is not None:
            return actual
    raise ValueError(
        "Could not find {purpose} column. Available columns: {columns}".format(
            purpose=purpose,
            columns=", ".join(columns),
        )
    )


def load_labels_scores(path: Path) -> Tuple[List[int], List[float]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("Prediction CSV is empty: {path}".format(path=path))

    columns = list(rows[0].keys())
    label_column = pick_column(
        columns,
        ["tb_label", "tb_target", "TB Label", "label", "target", "y_true", "true_label"],
        "label",
    )
    score_column = pick_column(
        columns,
        ["tb_prob", "tb_probability", "prob", "probability", "score", "prediction", "y_score", "tb_score"],
        "score",
    )

    labels: List[int] = []
    scores: List[float] = []
    for row in rows:
        label = parse_label(row.get(label_column))
        score = parse_float(row.get(score_column))
        if label is None or score is None:
            continue
        labels.append(label)
        scores.append(score)

    if not labels:
        raise ValueError("No usable label/score rows in: {path}".format(path=path))
    if not any(label == 1 for label in labels) or not any(label == 0 for label in labels):
        raise ValueError("ROC requires both positive and negative labels: {path}".format(path=path))
    return labels, scores


def compute_roc(labels: Sequence[int], scores: Sequence[float]) -> Tuple[List[Tuple[float, float]], float]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = float(sum(labels))
    negatives = float(len(labels) - sum(labels))

    points: List[Tuple[float, float]] = [(0.0, 0.0)]
    tp = 0.0
    fp = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                tp += 1.0
            else:
                fp += 1.0
            index += 1
        points.append((fp / negatives, tp / positives))

    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))

    auroc = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        auroc += (x1 - x0) * (y0 + y1) / 2.0
    return points, auroc


def step_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    stepped: List[Tuple[float, float]] = [points[0]]
    previous_x, previous_y = points[0]
    for x, y in points[1:]:
        stepped.append((x, previous_y))
        stepped.append((x, y))
        previous_x, previous_y = x, y
    return stepped


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_unfrozen_fixed_baseline(run_root: Path) -> Path:
    candidates: List[Path] = []
    if run_root.exists():
        for path in run_root.iterdir():
            if not path.is_dir():
                continue
            name = path.name.lower()
            if "baseline-fixednorm-retrained" not in name:
                continue
            if "freezeclip" in name:
                continue
            if (path / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv").exists() and (
                path / "results" / "finetune_full" / "finetuned_full_patient_predictions.csv"
            ).exists():
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No completed baseline-fixednorm-retrained runs found under {root}".format(root=run_root)
        )
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def simple_yaml_values(path: Path, keys: Iterable[str]) -> Dict[str, object]:
    wanted = set(keys)
    values: Dict[str, object] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if key not in wanted:
                continue
            value = raw_value.strip()
            if value.lower() == "true":
                values[key] = True
            elif value.lower() == "false":
                values[key] = False
            else:
                number = parse_float(value)
                if number is not None:
                    values[key] = int(number) if abs(number - int(number)) < 1e-8 else number
                else:
                    values[key] = value
    return values


def fmt_auroc(value: float) -> str:
    return "{:.2f}".format(value)


def svg_polyline(points: Sequence[Tuple[float, float]], map_x, map_y) -> str:
    return " ".join("{:.2f},{:.2f}".format(map_x(x), map_y(y)) for x, y in points)


def write_svg(path: Path, zero_shot: RocCurve, fine_tuned: RocCurve, run_name: str) -> None:
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

    curves = [zero_shot, fine_tuned]
    legend_x = width - 585
    legend_y = height - 255
    legend_w = 555
    legend_h = 145

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=width,
            h=height,
        )
    )
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    title_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 44px; font-weight: 800; fill: #282828; letter-spacing: 2px'
    lines.append('<text x="{x}" y="48" text-anchor="middle" style="{style}">ROC Curve: SA Test Set Performance</text>'.format(x=width / 2, style=title_style))
    lines.append('<text x="{x}" y="94" text-anchor="middle" style="{style}">Benin Zero-shot vs SA Fine-tuned</text>'.format(x=width / 2, style=title_style))

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

    lines.append('<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#7f7f7f" stroke-width="2.5" stroke-dasharray="9 7"/>'.format(
        x0=map_x(0.0),
        y0=map_y(0.0),
        x1=map_x(1.0),
        y1=map_y(1.0),
    ))

    for curve in curves:
        dash = ' stroke-dasharray="{dash}"'.format(dash=curve.dash) if curve.dash else ""
        lines.append('<polyline points="{points}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="miter" stroke-linecap="butt"{dash}/>'.format(
            points=svg_polyline(step_points(curve.points), map_x, map_y),
            color=curve.color,
            dash=dash,
        ))

    lines.append('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2"/>'.format(
        x=margin_left,
        y1=margin_top,
        y2=margin_top + plot_h,
        color=axis_color,
    ))
    lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2"/>'.format(
        x1=margin_left,
        x2=margin_left + plot_w,
        y=margin_top + plot_h,
        color=axis_color,
    ))

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

    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#000000" opacity="0.20"/>'.format(
        x=legend_x + 8,
        y=legend_y + 8,
        w=legend_w,
        h=legend_h,
    ))
    lines.append('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="#c7c7c7" stroke-width="2"/>'.format(
        x=legend_x,
        y=legend_y,
        w=legend_w,
        h=legend_h,
    ))
    legend_style = 'font-family: Arial, Helvetica, sans-serif; font-size: 30px; fill: #2d2d2d'
    legend_rows = [
        (zero_shot.color, zero_shot.dash, "Benin Zero-shot: AUROC={}".format(fmt_auroc(zero_shot.auroc))),
        (fine_tuned.color, fine_tuned.dash, "SA Fine-tuned: AUROC={}".format(fmt_auroc(fine_tuned.auroc))),
        ("#7f7f7f", "9 7", "Chance (AUROC=0.50)"),
    ]
    for offset, (color, dash, text) in enumerate(legend_rows):
        y = legend_y + 36 + offset * 42
        dash_attr = ' stroke-dasharray="{dash}"'.format(dash=dash) if dash else ""
        lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="5"{dash}/>'.format(
            x1=legend_x + 20,
            x2=legend_x + 80,
            y=y,
            color=color,
            dash=dash_attr,
        ))
        lines.append('<text x="{x}" y="{y}" dominant-baseline="middle" style="{style}">{text}</text>'.format(
            x=legend_x + 105,
            y=y,
            style=legend_style,
            text=html.escape(text),
        ))

    lines.append('<!-- Run: {run_name} -->'.format(run_name=html.escape(run_name)))
    lines.append("</svg>")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_svg_to_png(svg_path: Path, png_path: Path) -> bool:
    converter = shutil.which("convert") or shutil.which("magick")
    if converter is None:
        return False
    command = [converter]
    if Path(converter).name == "magick":
        command.append("convert")
    command.extend(["-density", "180", str(svg_path), str(png_path)])
    subprocess.check_call(command)
    return True


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_curves(run_dir: Path) -> Tuple[RocCurve, RocCurve, Dict[str, object]]:
    zero_csv = run_dir / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv"
    fine_csv = run_dir / "results" / "finetune_full" / "finetuned_full_patient_predictions.csv"
    if not zero_csv.exists():
        raise FileNotFoundError(zero_csv)
    if not fine_csv.exists():
        raise FileNotFoundError(fine_csv)

    zero_labels, zero_scores = load_labels_scores(zero_csv)
    fine_labels, fine_scores = load_labels_scores(fine_csv)
    zero_points, zero_auroc = compute_roc(zero_labels, zero_scores)
    fine_points, fine_auroc = compute_roc(fine_labels, fine_scores)

    zero_results = read_json(run_dir / "results" / "source_zero_shot" / "source_zero_shot_results.json")
    fine_results = read_json(run_dir / "results" / "finetune_full" / "results.json")
    config = simple_yaml_values(
        run_dir / "resolved_finetune_config.yaml",
        [
            "model_name",
            "freeze_backbone",
            "clip_unfreeze_last_n_layers",
            "val_size",
            "oversample_positive_class",
            "positive_class_multiplier",
            "use_pathology_loss",
            "pathology_weight",
        ],
    )

    provenance: Dict[str, object] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "zero_shot_csv": str(zero_csv),
        "fine_tuned_csv": str(fine_csv),
        "n_zero_shot": len(zero_labels),
        "n_fine_tuned": len(fine_labels),
        "n_positive": int(sum(fine_labels)),
        "n_negative": int(len(fine_labels) - sum(fine_labels)),
        "zero_shot_auroc_computed": zero_auroc,
        "fine_tuned_auroc_computed": fine_auroc,
        "zero_shot_auroc_reported": zero_results.get("auroc"),
        "fine_tuned_auroc_reported": fine_results.get("auroc"),
        "fine_tuned_best_metric": fine_results.get("best_metric"),
        "fine_tuned_best_epoch": fine_results.get("best_epoch"),
        "n_train": fine_results.get("n_train"),
        "n_val": fine_results.get("n_val"),
        "n_test": fine_results.get("n_test"),
        "config": config,
    }

    return (
        RocCurve("Benin Zero-shot", "#d62728", "25 14", zero_points, zero_auroc),
        RocCurve("SA Fine-tuned", "#2ca02c", None, fine_points, fine_auroc),
        provenance,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Fine-tune run directory. Defaults to the latest completed unfrozen baseline-fixednorm-retrained run.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Root used for auto-detecting the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to runs/baseline_roc_comparison/benin_to_sa/<run-name>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or find_latest_unfrozen_fixed_baseline(args.run_root)
    run_dir = run_dir.resolve()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_shot, fine_tuned, provenance = build_curves(run_dir)
    svg_path = output_dir / "sa_test_roc_zero_shot_vs_finetuned.svg"
    png_path = output_dir / "sa_test_roc_zero_shot_vs_finetuned.png"
    summary_path = output_dir / "sa_test_roc_summary.json"
    points_path = output_dir / "sa_test_roc_points.csv"
    manifest_path = output_dir / "manifest.json"

    write_svg(svg_path, zero_shot, fine_tuned, run_dir.name)
    png_written = convert_svg_to_png(svg_path, png_path)

    rows: List[Dict[str, object]] = []
    for curve in (zero_shot, fine_tuned):
        for point_index, (fpr, tpr) in enumerate(curve.points):
            rows.append(
                {
                    "model": curve.label,
                    "point_index": point_index,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auroc": curve.auroc,
                }
            )
    write_csv(points_path, rows)

    summary = dict(provenance)
    summary.update(
        {
            "svg_path": str(svg_path),
            "png_path": str(png_path) if png_written else None,
            "points_csv": str(points_path),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "analysis": "baseline_roc_comparison",
        "cohort_transfer": "benin_to_sa",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "outputs": {
            "svg": str(svg_path),
            "png": str(png_path) if png_written else None,
            "summary": str(summary_path),
            "points_csv": str(points_path),
        },
        "metrics": {
            "benin_zero_shot_auroc": zero_shot.auroc,
            "sa_fine_tuned_auroc": fine_tuned.auroc,
            "delta_auroc": fine_tuned.auroc - zero_shot.auroc,
        },
        "config": provenance.get("config", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Run: {run}".format(run=run_dir))
    print("Benin zero-shot AUROC: {value:.4f}".format(value=zero_shot.auroc))
    print("SA fine-tuned AUROC: {value:.4f}".format(value=fine_tuned.auroc))
    print("SVG: {path}".format(path=svg_path))
    if png_written:
        print("PNG: {path}".format(path=png_path))
    else:
        print("PNG: not written; ImageMagick convert/magick not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
