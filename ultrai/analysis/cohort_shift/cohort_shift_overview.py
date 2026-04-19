#!/usr/bin/env python3
"""Create a Benin-vs-South-Africa cohort-shift overview figure.

The script deliberately avoids pandas/matplotlib so it can run in the lean
CSCS analysis environment used by the project. It writes a slide-ready SVG,
optionally a PNG when ImageMagick is available, and machine-readable summaries.
"""

import argparse
import csv
import datetime as dt
import html
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_ROOT = Path("/capstor/scratch/cscs/lxflk/ULTR-AI-Vid")


class Metric:
    def __init__(
        self,
        group: str,
        label: str,
        benin: Optional[float],
        sa: Optional[float],
        benin_num: Optional[int],
        benin_den: Optional[int],
        sa_num: Optional[int],
        sa_den: Optional[int],
        note: str = "",
    ) -> None:
        self.group = group
        self.label = label
        self.benin = benin
        self.sa = sa
        self.benin_num = benin_num
        self.benin_den = benin_den
        self.sa_num = sa_num
        self.sa_den = sa_den
        self.note = note

    @property
    def diff(self) -> Optional[float]:
        if self.benin is None or self.sa is None:
            return None
        return self.sa - self.benin


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def clean(value: object) -> str:
    return str(value or "").strip()


def parse_float(value: object) -> Optional[float]:
    value = clean(value)
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_intish(value: object) -> Optional[int]:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def percent(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return 100.0 * num / den


def value_in_range(value: Optional[float], lower: float, upper: float) -> Optional[float]:
    if value is None or value < lower or value > upper:
        return None
    return value


def patient_ids_from_rows(rows: Iterable[Dict[str, str]], column: str) -> Set[str]:
    return {clean(row.get(column)) for row in rows if clean(row.get(column))}


def rows_by_id(rows: Iterable[Dict[str, str]], column: str) -> Dict[str, Dict[str, str]]:
    return {clean(row.get(column)): row for row in rows if clean(row.get(column))}


def pathology_patient_any(
    ids: Set[str],
    label_rows_by_id: Dict[str, Dict[str, str]],
    suffixes: Sequence[str],
) -> Tuple[int, int]:
    suffixes = set(suffixes)
    num = 0
    den = 0
    for patient_id in sorted(ids):
        row = label_rows_by_id.get(patient_id)
        if not row:
            continue
        relevant = [
            key
            for key in row
            if "_" in key and key.split("_", 1)[1] in suffixes
        ]
        if not relevant:
            continue
        den += 1
        if any(parse_intish(row.get(key)) == 1 for key in relevant):
            num += 1
    return num, den


def label_positive(
    ids: Set[str],
    label_rows_by_id: Dict[str, Dict[str, str]],
    column: str,
    positive_values: Set[int],
) -> Tuple[int, int]:
    num = 0
    den = 0
    for patient_id in sorted(ids):
        row = label_rows_by_id.get(patient_id)
        if not row:
            continue
        value = parse_intish(row.get(column))
        if value is None:
            continue
        den += 1
        if value in positive_values:
            num += 1
    return num, den


def clinical_binary(
    ids: Set[str],
    clinical_rows_by_id: Dict[str, Dict[str, str]],
    column: str,
    positive_values: Set[int],
    known_values: Set[int],
) -> Tuple[int, int]:
    num = 0
    den = 0
    for patient_id in sorted(ids):
        row = clinical_rows_by_id.get(patient_id)
        if not row:
            continue
        value = parse_intish(row.get(column))
        if value not in known_values:
            continue
        den += 1
        if value in positive_values:
            num += 1
    return num, den


def clinical_threshold(
    ids: Set[str],
    clinical_rows_by_id: Dict[str, Dict[str, str]],
    column: str,
    valid_range: Tuple[float, float],
    predicate: Callable[[float], bool],
) -> Tuple[int, int]:
    num = 0
    den = 0
    for patient_id in sorted(ids):
        row = clinical_rows_by_id.get(patient_id)
        if not row:
            continue
        value = value_in_range(parse_float(row.get(column)), *valid_range)
        if value is None:
            continue
        den += 1
        if predicate(value):
            num += 1
    return num, den


def collect_continuous(
    ids: Set[str],
    clinical_rows_by_id: Dict[str, Dict[str, str]],
    column: str,
    valid_range: Tuple[float, float],
) -> List[float]:
    values: List[float] = []
    for patient_id in sorted(ids):
        row = clinical_rows_by_id.get(patient_id)
        if not row:
            continue
        value = value_in_range(parse_float(row.get(column)), *valid_range)
        if value is not None:
            values.append(value)
    return values


def file_rows_for_ids(rows: Iterable[Dict[str, str]], ids: Set[str]) -> List[Dict[str, str]]:
    return [row for row in rows if clean(row.get("Patient ID")) in ids]


def file_row_binary(
    rows: Iterable[Dict[str, str]],
    predicate: Callable[[Dict[str, str]], Optional[bool]],
) -> Tuple[int, int]:
    num = 0
    den = 0
    for row in rows:
        value = predicate(row)
        if value is None:
            continue
        den += 1
        if value:
            num += 1
    return num, den


def summarize_file_counts(rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    counts = Counter(clean(row.get("Patient ID")) for row in rows if clean(row.get("Patient ID")))
    values = list(counts.values())
    if not values:
        return {"patients": 0, "rows": 0, "median_rows_per_patient": math.nan, "mean_rows_per_patient": math.nan}
    return {
        "patients": len(values),
        "rows": sum(values),
        "median_rows_per_patient": float(median(values)),
        "mean_rows_per_patient": float(sum(values) / len(values)),
    }


def format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 10:
        return f"{value:.0f}%"
    return f"{value:.1f}%"


def format_diff(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f} pp"


def metric_dict(metric: Metric) -> Dict[str, object]:
    return {
        "group": metric.group,
        "label": metric.label,
        "benin_percent": metric.benin,
        "sa_percent": metric.sa,
        "sa_minus_benin_pp": metric.diff,
        "benin_num": metric.benin_num,
        "benin_den": metric.benin_den,
        "sa_num": metric.sa_num,
        "sa_den": metric.sa_den,
        "note": metric.note,
    }


def add_metric(
    metrics: List[Metric],
    group: str,
    label: str,
    benin_counts: Tuple[int, int],
    sa_counts: Tuple[int, int],
    note: str = "",
) -> None:
    b_num, b_den = benin_counts
    s_num, s_den = sa_counts
    metrics.append(
        Metric(
            group=group,
            label=label,
            benin=percent(b_num, b_den),
            sa=percent(s_num, s_den),
            benin_num=b_num,
            benin_den=b_den,
            sa_num=s_num,
            sa_den=s_den,
            note=note,
        )
    )


def numeric_age_column(rows: Sequence[Dict[str, str]], candidates: Sequence[str]) -> Optional[str]:
    if not rows:
        return None
    for column in candidates:
        if column not in rows[0]:
            continue
        values = [
            value_in_range(parse_float(row.get(column)), 15, 100)
            for row in rows
        ]
        values = [value for value in values if value is not None]
        # Screening flags such as sc_age are often constant 0/1; require plausible
        # adult ages and some spread before treating a column as age.
        if len(values) >= 20 and max(values) >= 25 and len(set(values)) >= 5:
            return column
    return None


def build_metrics(
    benin_ids: Set[str],
    sa_ids: Set[str],
    benin_labels: Dict[str, Dict[str, str]],
    sa_labels: Dict[str, Dict[str, str]],
    benin_clinical: Dict[str, Dict[str, str]],
    sa_clinical: Dict[str, Dict[str, str]],
    benin_clinical_rows: Sequence[Dict[str, str]],
    sa_clinical_rows: Sequence[Dict[str, str]],
    benin_file_rows: Sequence[Dict[str, str]],
    sa_file_rows: Sequence[Dict[str, str]],
) -> Tuple[List[Metric], List[str], Dict[str, object]]:
    metrics: List[Metric] = []
    warnings: List[str] = []

    add_metric(
        metrics,
        "Patient mix",
        "TB positive",
        label_positive(benin_ids, benin_labels, "TB Label", {1}),
        label_positive(sa_ids, sa_labels, "TB Label", {1}),
        "Patient-level target label.",
    )
    add_metric(
        metrics,
        "Patient mix",
        "HIV positive",
        clinical_binary(benin_ids, benin_clinical, "hiv", {2}, {1, 2}),
        clinical_binary(sa_ids, sa_clinical, "hiv", {1}, {0, 1}),
        "Cohort-specific coding: Benin 2=positive/1=negative; SA 1=positive/0=negative.",
    )
    add_metric(
        metrics,
        "Patient mix",
        "Prior TB",
        clinical_binary(benin_ids, benin_clinical, "previous_tb_diagnosis", {2, 4}, {1, 2, 4}),
        clinical_binary(sa_ids, sa_clinical, "previous_tb_diagnosis", {2, 4}, {1, 2, 4}),
        "previous_tb_diagnosis: 1=no, 2=resolved, 4=active.",
    )
    add_metric(
        metrics,
        "Patient mix",
        "Underweight BMI <18.5",
        clinical_threshold(benin_ids, benin_clinical, "bmi", (10, 60), lambda value: value < 18.5),
        clinical_threshold(sa_ids, sa_clinical, "bmi", (10, 60), lambda value: value < 18.5),
        "BMI values outside 10-60 kg/m2 were treated as data-entry outliers.",
    )
    add_metric(
        metrics,
        "Patient mix",
        "Low SpO2 <95%",
        clinical_threshold(benin_ids, benin_clinical, "saturation_en_oxygene_spo2", (50, 100), lambda value: value < 95),
        clinical_threshold(sa_ids, sa_clinical, "saturation_en_oxygene_spo2", (50, 100), lambda value: value < 95),
        "Pulse oximetry values outside 50-100% were excluded.",
    )
    add_metric(
        metrics,
        "Patient mix",
        "Resp. rate >=24/min",
        clinical_threshold(benin_ids, benin_clinical, "frequence_respiratoire", (5, 80), lambda value: value >= 24),
        clinical_threshold(sa_ids, sa_clinical, "frequence_respiratoire", (5, 80), lambda value: value >= 24),
        "Respiratory rates outside 5-80/min were excluded.",
    )
    add_metric(
        metrics,
        "Patient mix",
        "Diabetes",
        clinical_binary(benin_ids, benin_clinical, "diabetes", {1}, {0, 1}),
        clinical_binary(sa_ids, sa_clinical, "diabetes", {1}, {0, 1}),
    )

    benin_age_col = numeric_age_column(benin_clinical_rows, ["age", "sc_age"])
    sa_age_col = numeric_age_column(sa_clinical_rows, ["age", "sc_age"])
    if benin_age_col and sa_age_col:
        add_metric(
            metrics,
            "Patient mix",
            "Age >=50 years",
            clinical_threshold(benin_ids, benin_clinical, benin_age_col, (15, 100), lambda value: value >= 50),
            clinical_threshold(sa_ids, sa_clinical, sa_age_col, (15, 100), lambda value: value >= 50),
            f"Age columns: Benin={benin_age_col}, SA={sa_age_col}.",
        )
    else:
        warnings.append(
            "Age was not plotted: the local SA clinical export contains no numeric age column "
            "(sc_age appears to be a screening flag)."
        )

    add_metric(
        metrics,
        "Auxiliary LUS pathology labels",
        "A-lines anywhere",
        pathology_patient_any(benin_ids, benin_labels, ["A-line"]),
        pathology_patient_any(sa_ids, sa_labels, ["A-line"]),
        "Patient has at least one annotated site with this finding.",
    )
    add_metric(
        metrics,
        "Auxiliary LUS pathology labels",
        "Large consolidation anywhere",
        pathology_patient_any(benin_ids, benin_labels, ["large Consolidations"]),
        pathology_patient_any(sa_ids, sa_labels, ["large Consolidations"]),
    )
    add_metric(
        metrics,
        "Auxiliary LUS pathology labels",
        "Pleural effusion anywhere",
        pathology_patient_any(benin_ids, benin_labels, ["Pleural effusion"]),
        pathology_patient_any(sa_ids, sa_labels, ["Pleural effusion"]),
    )
    add_metric(
        metrics,
        "Auxiliary LUS pathology labels",
        "Other pathology anywhere",
        pathology_patient_any(
            benin_ids,
            benin_labels,
            ["B-lines", "Confluent B-lines", "small Consolidations or Nodules", "Pattern A' (pneumothorax)"],
        ),
        pathology_patient_any(
            sa_ids,
            sa_labels,
            ["B-lines", "Confluent B-lines", "small Consolidations or Nodules", "Pattern A' (pneumothorax)"],
        ),
        "Matches the model's aggregated 'other_pathology' auxiliary head.",
    )

    add_metric(
        metrics,
        "Acquisition / protocol",
        "Video file rows",
        file_row_binary(benin_file_rows, lambda row: clean(row.get("type") or row.get("Type")).lower() == "video"),
        file_row_binary(sa_file_rows, lambda row: clean(row.get("type") or row.get("Type")).lower() == "video"),
    )
    add_metric(
        metrics,
        "Acquisition / protocol",
        "Unknown / non-standard site",
        file_row_binary(benin_file_rows, lambda row: clean(row.get("Site")).upper().startswith("UNKNOWN") or clean(row.get("Site")) == ""),
        file_row_binary(sa_file_rows, lambda row: clean(row.get("Site")).upper().startswith("UNKNOWN") or clean(row.get("Site")) == ""),
    )
    add_metric(
        metrics,
        "Acquisition / protocol",
        "Depth not 5 or 15 cm",
        file_row_binary(benin_file_rows, lambda row: (parse_intish(row.get("Depth")) not in {5, 15}) if parse_intish(row.get("Depth")) is not None else None),
        file_row_binary(sa_file_rows, lambda row: (parse_intish(row.get("Depth")) not in {5, 15}) if parse_intish(row.get("Depth")) is not None else None),
        "Depth is a proxy for image scale/field-of-view differences.",
    )

    continuous = {
        "benin": {
            "bmi": collect_continuous(benin_ids, benin_clinical, "bmi", (10, 60)),
            "spo2": collect_continuous(benin_ids, benin_clinical, "saturation_en_oxygene_spo2", (50, 100)),
            "respiratory_rate": collect_continuous(benin_ids, benin_clinical, "frequence_respiratoire", (5, 80)),
        },
        "sa": {
            "bmi": collect_continuous(sa_ids, sa_clinical, "bmi", (10, 60)),
            "spo2": collect_continuous(sa_ids, sa_clinical, "saturation_en_oxygene_spo2", (50, 100)),
            "respiratory_rate": collect_continuous(sa_ids, sa_clinical, "frequence_respiratoire", (5, 80)),
        },
    }

    extra_summary = {
        "continuous_medians": {
            domain: {
                key: {"n": len(values), "median": median(values) if values else None}
                for key, values in domain_values.items()
            }
            for domain, domain_values in continuous.items()
        },
        "file_counts": {
            "benin": summarize_file_counts(benin_file_rows),
            "sa": summarize_file_counts(sa_file_rows),
        },
    }

    return metrics, warnings, extra_summary


def svg_text(x: float, y: float, text: str, size: int = 20, color: str = "#333333", weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">'
        f"{html.escape(text)}</text>"
    )


def svg_value_text(x: float, y: float, text: str, size: int, color: str, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{color}" font-weight="700" text-anchor="{anchor}" '
        f'paint-order="stroke" stroke="#FFFFFF" stroke-width="5" stroke-linejoin="round">'
        f"{html.escape(text)}</text>"
    )


def render_svg(metrics: Sequence[Metric], output_path: Path) -> None:
    width = 1680
    left = 455
    right = 1360
    diff_x = right + 90
    top = 150
    row_h = 54
    group_h = 42
    bottom = 80
    benin_color = "#58B8AA"
    sa_color = "#E8C83F"
    line_color = "#B7C2C7"
    text_color = "#283238"
    light = "#F4F7F8"
    grid = "#DCE4E7"
    group_order: List[str] = []
    for metric in metrics:
        if metric.group not in group_order:
            group_order.append(metric.group)

    rows = 0
    for group in group_order:
        rows += 1
        rows += sum(1 for metric in metrics if metric.group == group)
    height = top + rows * row_h + bottom + 30

    def xmap(value: float) -> float:
        return left + (right - left) * (value / 100.0)

    def add_value_labels(parts: List[str], metric: Metric, xb: float, xs: float, y: float) -> None:
        benin_text = format_metric_value(metric.benin)
        sa_text = format_metric_value(metric.sa)
        close_pair = abs(xb - xs) < 90

        if close_pair:
            # When the two values are close, horizontal labels collide. Stack them
            # around the connector while keeping each label anchored to its own dot.
            parts.append(svg_value_text(xb, y - 18, benin_text, 17, "#1F5E55", "middle"))
            parts.append(svg_value_text(xs, y + 23, sa_text, 17, "#7C6511", "middle"))
            return

        benin_anchor = "end" if metric.benin > metric.sa else "start"
        sa_anchor = "start" if metric.benin > metric.sa else "end"
        benin_offset = -18 if benin_anchor == "end" else 18
        sa_offset = 18 if sa_anchor == "start" else -18
        parts.append(svg_value_text(xb + benin_offset, y + 7, benin_text, 17, "#1F5E55", benin_anchor))
        parts.append(svg_value_text(xs + sa_offset, y + 7, sa_text, 17, "#7C6511", sa_anchor))

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(60, 54, "Benin vs South Africa: cohort and acquisition shift", 34, text_color, "700"),
        svg_text(60, 88, "All rows are percentages; dots show cohort values and connectors show the direction of shift.", 20, "#5B6770"),
    ]

    legend_y = 58
    parts.extend(
        [
            f'<circle cx="{right - 240}" cy="{legend_y}" r="9" fill="{benin_color}" stroke="#1F5E55" stroke-width="2"/>',
            svg_text(right - 224, legend_y + 7, "Benin", 20, text_color),
            f'<circle cx="{right - 125}" cy="{legend_y}" r="9" fill="{sa_color}" stroke="#9E8216" stroke-width="2"/>',
            svg_text(right - 109, legend_y + 7, "South Africa", 20, text_color),
        ]
    )

    for tick in [0, 25, 50, 75, 100]:
        x = xmap(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - bottom}" stroke="{grid}" stroke-width="1"/>')
        parts.append(svg_text(x, top - 32, f"{tick}%", 16, "#67737B", anchor="middle"))

    parts.append(svg_text(diff_x, top - 32, "SA-Benin", 16, "#67737B", "700", anchor="middle"))

    y = top
    for group in group_order:
        parts.append(f'<rect x="42" y="{y - 27}" width="{width - 84}" height="34" rx="6" fill="{light}"/>')
        parts.append(svg_text(60, y - 3, group, 20, "#37434A", "700"))
        y += group_h
        for metric in [m for m in metrics if m.group == group]:
            label_y = y + 6
            parts.append(svg_text(80, label_y, metric.label, 20, text_color, "600"))
            if metric.benin is not None and metric.sa is not None:
                xb = xmap(metric.benin)
                xs = xmap(metric.sa)
                parts.append(f'<line x1="{xb:.1f}" y1="{y:.1f}" x2="{xs:.1f}" y2="{y:.1f}" stroke="{line_color}" stroke-width="7" stroke-linecap="round"/>')
                parts.append(f'<circle cx="{xb:.1f}" cy="{y:.1f}" r="11" fill="{benin_color}" stroke="#1F5E55" stroke-width="2"/>')
                parts.append(f'<circle cx="{xs:.1f}" cy="{y:.1f}" r="11" fill="{sa_color}" stroke="#9E8216" stroke-width="2"/>')

                add_value_labels(parts, metric, xb, xs, y)

                diff_color = "#A43F32" if metric.diff and metric.diff > 0 else "#216E63"
                parts.append(svg_value_text(diff_x, y + 7, format_diff(metric.diff), 17, diff_color, "middle"))
            else:
                parts.append(svg_text(left, y + 7, "Not available", 17, "#7A858C"))
            y += row_h

    axis_y = height - bottom + 10
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}" stroke="#809099" stroke-width="2"/>')
    for tick in [0, 25, 50, 75, 100]:
        x = xmap(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 8}" stroke="#809099" stroke-width="2"/>')
        parts.append(svg_text(x, axis_y + 30, f"{tick}%", 16, "#67737B", anchor="middle"))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_metrics_csv(metrics: Sequence[Metric], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "label",
                "benin_percent",
                "sa_percent",
                "sa_minus_benin_pp",
                "benin_num",
                "benin_den",
                "sa_num",
                "sa_den",
                "note",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric_dict(metric))


def maybe_convert_png(svg_path: Path, png_path: Path) -> Optional[str]:
    converter = shutil.which("convert")
    if converter is None:
        return "ImageMagick 'convert' not found; PNG was not generated."
    command = [converter, "-density", "180", str(svg_path), str(png_path)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        return f"PNG conversion failed: {exc.stderr.strip() or exc.stdout.strip()}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benin-labels", type=Path, default=REPO_ROOT / "Data/labels/labels_multidiagnosis.csv")
    parser.add_argument("--sa-labels", type=Path, default=REPO_ROOT / "Data/labels/rsa_pathology_labels.csv")
    parser.add_argument("--benin-clinical", type=Path, default=REPO_ROOT / "Data/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv")
    parser.add_argument("--sa-clinical", type=Path, default=REPO_ROOT / "Data/clinical_data/TrUSTSouthAfrica_DATA_2026-01-23_1807.csv")
    parser.add_argument("--benin-files", type=Path, default=REPO_ROOT / "Data/processed_files_2.csv")
    parser.add_argument("--sa-files", type=Path, default=Path("/capstor/scratch/cscs/tbrokowski/ultr-ai/RSA_Videos/cleaned/processed_files.csv"))
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "runs/cohort_shift_overview/benin_vs_sa")
    parser.add_argument("--no-png", action="store_true", help="Only write SVG and summary files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benin_label_rows = read_csv_rows(args.benin_labels)
    sa_label_rows = read_csv_rows(args.sa_labels)
    benin_clinical_rows = read_csv_rows(args.benin_clinical)
    sa_clinical_rows = read_csv_rows(args.sa_clinical)
    benin_file_rows_all = read_csv_rows(args.benin_files)
    sa_file_rows_all = read_csv_rows(args.sa_files)

    benin_labels_by_id = rows_by_id(benin_label_rows, "record_id")
    sa_labels_by_id = rows_by_id(sa_label_rows, "record_id")
    benin_clinical_by_id = rows_by_id(benin_clinical_rows, "record_id")
    sa_clinical_by_id = rows_by_id(sa_clinical_rows, "record_id")

    benin_ids = patient_ids_from_rows(benin_label_rows, "record_id") & patient_ids_from_rows(benin_file_rows_all, "Patient ID")
    sa_ids = patient_ids_from_rows(sa_label_rows, "record_id") & patient_ids_from_rows(sa_file_rows_all, "Patient ID")
    benin_file_rows = file_rows_for_ids(benin_file_rows_all, benin_ids)
    sa_file_rows = file_rows_for_ids(sa_file_rows_all, sa_ids)

    metrics, warnings, extra_summary = build_metrics(
        benin_ids=benin_ids,
        sa_ids=sa_ids,
        benin_labels=benin_labels_by_id,
        sa_labels=sa_labels_by_id,
        benin_clinical=benin_clinical_by_id,
        sa_clinical=sa_clinical_by_id,
        benin_clinical_rows=benin_clinical_rows,
        sa_clinical_rows=sa_clinical_rows,
        benin_file_rows=benin_file_rows,
        sa_file_rows=sa_file_rows,
    )

    svg_path = args.output_dir / "cohort_shift_dashboard.svg"
    png_path = args.output_dir / "cohort_shift_dashboard.png"
    metrics_csv_path = args.output_dir / "cohort_shift_metrics.csv"
    summary_json_path = args.output_dir / "cohort_shift_summary.json"
    manifest_path = args.output_dir / "manifest.json"

    render_svg(metrics, svg_path)
    write_metrics_csv(metrics, metrics_csv_path)

    png_warning = None
    if not args.no_png:
        png_warning = maybe_convert_png(svg_path, png_path)
        if png_warning:
            warnings.append(png_warning)

    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cohorts": {
            "benin": {
                "n_lus_patients": len(benin_ids),
                "n_clinical_matched": len(benin_ids & set(benin_clinical_by_id)),
                "n_file_rows": len(benin_file_rows),
            },
            "sa": {
                "n_lus_patients": len(sa_ids),
                "n_clinical_matched": len(sa_ids & set(sa_clinical_by_id)),
                "n_file_rows": len(sa_file_rows),
            },
        },
        "metrics": [metric_dict(metric) for metric in metrics],
        "extra_summary": extra_summary,
        "warnings": warnings,
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "benin_labels": str(args.benin_labels),
            "sa_labels": str(args.sa_labels),
            "benin_clinical": str(args.benin_clinical),
            "sa_clinical": str(args.sa_clinical),
            "benin_files": str(args.benin_files),
            "sa_files": str(args.sa_files),
        },
        "outputs": {
            "svg": str(svg_path),
            "png": str(png_path) if png_path.exists() else None,
            "metrics_csv": str(metrics_csv_path),
            "summary_json": str(summary_json_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {svg_path}")
    if png_path.exists():
        print(f"Wrote {png_path}")
    print(f"Wrote {metrics_csv_path}")
    print(f"Wrote {summary_json_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
