#!/usr/bin/env python3
"""Generate the plain SA fine-tuned ROC/PR figure with legends below."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultrai.analysis.domain_shift.metric_curve_plots import plot_run_roots_roc_pr


SOURCE_BASELINE_RUN_ROOT = Path(
    "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/"
    "20260506_090813__train__benin__all-folds__hmv-mil-legacy-parity-repro"
)

NORMAL_FINETUNE_RUN_ROOTS_BY_FOLD = {
    0: Path(
        "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/"
        "20260506_171838__finetune__benin_to_sa__src-fold0__normal-balanceclip2pos4-lowlr-repro"
    ),
    1: Path(
        "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/"
        "20260506_171838__finetune__benin_to_sa__src-fold1__normal-balanceclip2pos4-lowlr-repro"
    ),
    2: Path(
        "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/"
        "20260506_171838__finetune__benin_to_sa__src-fold2__normal-balanceclip2pos4-lowlr-repro"
    ),
    3: Path(
        "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/"
        "20260506_171838__finetune__benin_to_sa__src-fold3__normal-balanceclip2pos4-lowlr-repro"
    ),
    4: Path(
        "/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/"
        "20260506_171838__finetune__benin_to_sa__src-fold4__normal-balanceclip2pos4-lowlr-repro"
    ),
}

OUTPUT_SVG = (
    REPO_ROOT
    / "results_other_experiments"
    / "results"
    / "domain_shift_figures"
    / "plain_finetune_sa_target_roc_pr_mean_ci_legend_below.svg"
)


def main() -> None:
    summary = plot_run_roots_roc_pr(
        NORMAL_FINETUNE_RUN_ROOTS_BY_FOLD,
        "results",
        "finetune_full",
        "finetuned_full_patient_predictions.csv",
        evaluation_label="Plain SA Fine-tuned",
        title="Plain SA Fine-tuned Test Set Performance",
        subtitle="Mean ROC and Precision-Recall Curves Across Five Source Folds",
        output_svg=OUTPUT_SVG,
        color="#2ca02c",
        roc_legend_position="below",
        pr_legend_position="below",
        lines_to_show=("Zero-Shot SA", "Finetuned SA", "Prevalence", "Source Retention"),
        source_run_root=SOURCE_BASELINE_RUN_ROOT,
        show=False,
    )

    print(f"Saved Plain SA fine-tuned AUROC/AUPRC graph to {summary['output_svg']}")
    print(f"Plotted lines: {', '.join(summary['plotted_lines'])}")
    if summary["prevalence_lines"]:
        print(f"Prevalence lines: {', '.join(summary['prevalence_lines'])}")
    if summary["skipped_lines"]:
        print(f"Skipped missing lines: {', '.join(summary['skipped_lines'])}")
    print(
        "Plain SA fine-tuned curves: "
        f"AUROC {summary['auroc']['mean']:.4f} "
        f"(95% CI {summary['auroc']['ci_lower']:.4f}-{summary['auroc']['ci_upper']:.4f}); "
        f"AUPRC {summary['auprc']['mean']:.4f} "
        f"(95% CI {summary['auprc']['ci_lower']:.4f}-{summary['auprc']['ci_upper']:.4f})"
    )


if __name__ == "__main__":
    main()
