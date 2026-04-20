#!/usr/bin/env python3
"""Plot ROC and precision-recall curves for the DANN transfer runs.

The figures compare four evaluation views across the five Benin source folds:
the original Benin source model on held-out Benin data, the same source model
zero-shot on SA, the DANN-adapted model on SA, and the DANN-adapted model
evaluated back on its Benin source test fold.
"""

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence


try:
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        SCRATCH_ROOT,
        convert_svg_to_png,
        simple_yaml_values,
        write_csv,
    )
    from ultrai.analysis.baseline_rocs.plot_sa_simclr_transfer_curves import (
        CurveSummary,
        Evaluation,
        FoldEvaluation,
        fold_number_from_name,
        load_evaluation,
        mean_curve,
        prevalence,
        source_fold_dirs,
        write_multicurve_svg,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from ultrai.analysis.baseline_rocs.plot_sa_test_roc import (
        SCRATCH_ROOT,
        convert_svg_to_png,
        simple_yaml_values,
        write_csv,
    )
    from ultrai.analysis.baseline_rocs.plot_sa_simclr_transfer_curves import (
        CurveSummary,
        Evaluation,
        FoldEvaluation,
        fold_number_from_name,
        load_evaluation,
        mean_curve,
        prevalence,
        source_fold_dirs,
        write_multicurve_svg,
    )


DEFAULT_SOURCE_RUN_DIR = SCRATCH_ROOT / "runs" / "train" / "benin" / "20260404_150537__train__benin__all-folds__baseline"
DEFAULT_FINETUNE_RUN_ROOT = SCRATCH_ROOT / "runs" / "finetune" / "benin_to_sa"
DEFAULT_DANN_RUN_GLOB = "*__finetune__benin_to_sa__src-fold*__dann"
DEFAULT_OUTPUT_DIR = (
    SCRATCH_ROOT
    / "runs"
    / "baseline_roc_comparison"
    / "benin_to_sa"
    / "20260413__dann__source-fold-average"
)


EVALUATIONS = [
    Evaluation("benin_source", "Benin source", "Benin", "#00897b", None),
    Evaluation("zero_shot_sa", "Zero-shot SA", "South Africa", "#d62728", "25 14"),
    Evaluation("sa_fine_tuned", "SA fine-tuned", "South Africa", "#2ca02c", None),
    Evaluation("source_after_sa_ft", "Source after SA FT", "Benin", "#1f77b4", "4 12"),
]


def discover_dann_runs(run_root: Path, run_glob: str) -> Dict[int, Path]:
    """Discover one completed DANN run per source fold.

    If multiple DANN runs exist for a fold, the lexicographically latest run
    name is used. In the current artifacts this selects the April 13 folds and
    the earlier April 12 fold 2 run, which is the only completed fold 2 DANN
    run present.
    """
    runs: Dict[int, Path] = {}
    for run_dir in sorted(path for path in run_root.glob(run_glob) if path.is_dir()):
        fold = fold_number_from_name(run_dir.name)
        if fold is None:
            continue
        required = [
            run_dir / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv",
            run_dir / "results" / "dann_full" / "dann_full_patient_predictions.csv",
            run_dir / "results" / "source_test_evaluation" / "finetuned_source_test_patient_predictions.csv",
        ]
        if all(path.exists() for path in required):
            runs[fold] = run_dir
    return dict(sorted(runs.items()))


def build_dann_summaries(source_run_dir: Path, dann_runs: Dict[int, Path], grid: Sequence[float]) -> List[CurveSummary]:
    source_folds = source_fold_dirs(source_run_dir)
    common_folds = sorted(set(source_folds.keys()) & set(dann_runs.keys()))
    if len(common_folds) != 5:
        raise ValueError("Expected 5 common folds, found {folds}".format(folds=common_folds))

    by_key: Dict[str, List[FoldEvaluation]] = {evaluation.key: [] for evaluation in EVALUATIONS}
    for fold in common_folds:
        csv_paths = {
            "benin_source": source_folds[fold] / "final_results" / "test_predictions.csv",
            "zero_shot_sa": dann_runs[fold] / "results" / "source_zero_shot" / "source_zero_shot_patient_predictions.csv",
            "sa_fine_tuned": dann_runs[fold] / "results" / "dann_full" / "dann_full_patient_predictions.csv",
            "source_after_sa_ft": dann_runs[fold] / "results" / "source_test_evaluation" / "finetuned_source_test_patient_predictions.csv",
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


def write_dann_metrics(output_dir: Path, summaries: Sequence[CurveSummary], grid: Sequence[float]) -> Dict[str, object]:
    metric_rows: List[Dict[str, object]] = []
    for summary in summaries:
        for fold_eval in summary.fold_evaluations:
            metric_rows.append(
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
    metrics_csv = output_dir / "dann_transfer_fold_metrics.csv"
    write_csv(metrics_csv, metric_rows)

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
    roc_points_csv = output_dir / "dann_transfer_roc_mean_points.csv"
    pr_points_csv = output_dir / "dann_transfer_pr_mean_points.csv"
    write_csv(roc_points_csv, roc_rows)
    write_csv(pr_points_csv, pr_rows)

    metrics: Dict[str, object] = {}
    for summary in summaries:
        aurocs = [item.auroc for item in summary.fold_evaluations]
        auprcs = [item.auprc for item in summary.fold_evaluations]
        metrics[summary.evaluation.key] = {
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
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--finetune-run-root", type=Path, default=DEFAULT_FINETUNE_RUN_ROOT)
    parser.add_argument("--dann-run-glob", default=DEFAULT_DANN_RUN_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=201, help="Number of grid points for mean curve interpolation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    dann_runs = discover_dann_runs(args.finetune_run_root, args.dann_run_glob)
    if not dann_runs:
        raise FileNotFoundError("No completed DANN runs matched {pattern}".format(pattern=args.dann_run_glob))

    grid_size = max(int(args.grid_size), 2)
    grid = [index / float(grid_size - 1) for index in range(grid_size)]
    summaries = build_dann_summaries(source_run_dir, dann_runs, grid)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_svg = output_dir / "dann_transfer_roc_mean_across_folds.svg"
    roc_png = output_dir / "dann_transfer_roc_mean_across_folds.png"
    pr_svg = output_dir / "dann_transfer_pr_mean_across_folds.svg"
    pr_png = output_dir / "dann_transfer_pr_mean_across_folds.png"
    summary_json = output_dir / "dann_transfer_curve_summary.json"
    manifest_json = output_dir / "manifest.json"

    write_multicurve_svg(
        roc_svg,
        "DANN Transfer: ROC Curves",
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
        "DANN Transfer: Precision-Recall",
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
        next(iter(dann_runs.values())) / "resolved_finetune_config.yaml",
        [
            "freeze_backbone",
            "clip_unfreeze_last_n_layers",
            "model_name",
            "val_size",
            "oversample_positive_class",
            "dann_lambda",
            "dann_domain_loss_weight",
            "dann_source_task_weight",
            "dann_target_task_weight",
        ],
    )
    output_info = write_dann_metrics(output_dir, summaries, grid)
    summary = {
        "analysis": "dann_transfer_curves",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(source_run_dir),
        "finetune_run_root": str(args.finetune_run_root),
        "dann_run_glob": args.dann_run_glob,
        "dann_runs": {str(fold): str(path) for fold, path in dann_runs.items()},
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
        "interpretation_note": "SA curves are model/checkpoint averages over five Benin source-fold checkpoints evaluated on the same SA test split. Benin source curves use held-out Benin test folds. Source-after-FT evaluates each DANN-adapted checkpoint back on its corresponding Benin source test fold.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("DANN runs: {runs}".format(runs=", ".join(path.name for path in dann_runs.values())))
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
