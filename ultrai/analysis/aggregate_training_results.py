#!/usr/bin/env python3
"""Aggregate per-fold supervised training final_results into run-level CSVs."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _fold_dirs(run_root: Path, expected_folds: Iterable[int]) -> List[Path]:
    fold_dirs = []
    for fold in expected_folds:
        fold_dir = run_root / f"fold{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing expected fold directory: {fold_dir}")
        fold_dirs.append(fold_dir)
    return fold_dirs


def _aggregate_rows(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(group_key, "")) for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    output_rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        summary: Dict[str, object] = {
            group_key: key[index]
            for index, group_key in enumerate(group_keys)
        }
        summary["num_folds"] = len(group_rows)

        numeric_columns = sorted(
            {
                column
                for row in group_rows
                for column, value in row.items()
                if column not in {"fold", *group_keys} and _as_float(value) is not None
            }
        )
        for column in numeric_columns:
            values = []
            for row in group_rows:
                numeric = _as_float(row.get(column))
                if numeric is not None:
                    values.append(numeric)
            if not values:
                continue
            summary[f"{column}_mean"] = mean(values)
            summary[f"{column}_std"] = stdev(values) if len(values) > 1 else 0.0
            summary[f"{column}_min"] = min(values)
            summary[f"{column}_max"] = max(values)
        output_rows.append(summary)

    return output_rows


def aggregate_tb(run_root: Path, fold_dirs: List[Path]) -> None:
    fold_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for fold_dir in fold_dirs:
        fold = fold_dir.name.replace("fold", "")
        summary_path = fold_dir / "final_results" / "tb_results" / "tb_metrics_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing fold TB summary CSV: {summary_path}")

        for row in _read_csv(summary_path):
            output_row: Dict[str, object] = {"fold": fold}
            output_row.update(row)
            fold_rows.append(output_row)
            if str(row.get("Split", "")).lower() == "test":
                test_rows.append(output_row.copy())

    output_dir = run_root / "final_results" / "tb_results"
    _write_csv(output_dir / "tb_metrics_by_fold.csv", fold_rows)
    _write_csv(output_dir / "tb_metrics_summary.csv", _aggregate_rows(fold_rows, ["Split"]))
    _write_csv(output_dir / "tb_test_metrics_by_fold.csv", test_rows)
    _write_csv(output_dir / "tb_test_metrics_summary.csv", _aggregate_rows(test_rows, []))


def aggregate_pathology(run_root: Path, fold_dirs: List[Path]) -> None:
    fold_rows: List[Dict[str, object]] = []

    for fold_dir in fold_dirs:
        fold = fold_dir.name.replace("fold", "")
        summary_path = fold_dir / "final_results" / "pathology_results" / "all_pathologies_summary.csv"
        if not summary_path.exists():
            continue
        for row in _read_csv(summary_path):
            output_row: Dict[str, object] = {"fold": fold}
            output_row.update(row)
            fold_rows.append(output_row)

    if not fold_rows:
        return

    output_dir = run_root / "final_results" / "pathology_results"
    _write_csv(output_dir / "all_pathologies_by_fold.csv", fold_rows)
    _write_csv(
        output_dir / "all_pathologies_summary.csv",
        _aggregate_rows(fold_rows, ["Split", "Pathology"]),
    )


def aggregate_predictions(run_root: Path, fold_dirs: List[Path]) -> None:
    for split_name in ["test", "val", "train"]:
        rows: List[Dict[str, object]] = []
        for fold_dir in fold_dirs:
            fold = fold_dir.name.replace("fold", "")
            predictions_path = fold_dir / "final_results" / f"{split_name}_predictions.csv"
            if not predictions_path.exists():
                continue
            for row in _read_csv(predictions_path):
                output_row: Dict[str, object] = {"fold": fold}
                output_row.update(row)
                rows.append(output_row)
        if rows:
            _write_csv(run_root / "final_results" / f"{split_name}_predictions_all_folds.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Training run root containing fold*/final_results")
    parser.add_argument("--expected-folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    fold_dirs = _fold_dirs(run_root, args.expected_folds)

    aggregate_tb(run_root, fold_dirs)
    aggregate_pathology(run_root, fold_dirs)
    aggregate_predictions(run_root, fold_dirs)

    print(f"Aggregated final results under {run_root / 'final_results'}", flush=True)


if __name__ == "__main__":
    main()
