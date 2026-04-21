# SA Test ROC Baseline Plot

Create a slide-ready ROC comparison for the SA test set:

```bash
python3 ultrai/analysis/baseline_rocs/plot_sa_test_roc.py
```

By default the script auto-selects the latest completed unfrozen
`baseline-fixednorm-retrained` Benin-to-SA fine-tuning run under:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa`

Outputs are written analogously to the other analysis results:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/baseline_roc_comparison/benin_to_sa/<run-name>/`

The directory contains a PNG, SVG, ROC point CSV, summary JSON, and manifest.

To generate the mean ROC across the five fixed-baseline source folds:

```bash
python3 ultrai/analysis/baseline_rocs/plot_sa_test_roc_fold_average.py
```

This computes a model/checkpoint average over the five Benin source folds by
interpolating each ROC onto a shared FPR grid. It does not pool repeated SA test
patients as independent observations.

To generate the original Benin source-model ROC across all held-out Benin test
folds:

```bash
python3 ultrai/analysis/baseline_rocs/plot_benin_source_roc_fold_average.py
```

By default this reads:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/20260404_150537__train__benin__all-folds__baseline`

and writes:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/baseline_roc_comparison/benin/20260404_150537__train__benin__all-folds__baseline/`

To generate the SA-SimCLR transfer ROC and precision-recall curves:

```bash
python3 ultrai/analysis/baseline_rocs/plot_sa_simclr_transfer_curves.py
```

By default this combines the Benin source model trained with the SA-SimCLR
backbone:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/20260418__train__benin__all-folds__sa-simclr-temporal-v1`

with the freeze-CLIP Benin-to-SA fine-tuning family:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/20260418__finetune__benin_to_sa__src-fold*__sa-simclr-temporal-v1-freezeclip`

and writes:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/baseline_roc_comparison/benin_to_sa/20260418__sa-simclr-temporal-v1-freezeclip__source-fold-average/`

To generate the DANN transfer ROC and precision-recall curves:

```bash
python3 ultrai/analysis/baseline_rocs/plot_dann_transfer_curves.py
```

By default this combines the original Benin source model:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/20260404_150537__train__benin__all-folds__baseline`

with the completed DANN Benin-to-SA fine-tuning family:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/20260420__finetune__benin_to_sa__src-fold*__dann-v3-active-weak-freezeclip`

and writes:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/baseline_roc_comparison/benin_to_sa/20260420__dann-v3-active-weak-freezeclip__source-fold-average/`

To generate the FixMatch transfer ROC and precision-recall curves:

```bash
python3 ultrai/analysis/baseline_rocs/plot_fixmatch_transfer_curves.py
```

By default this combines the original Benin source model:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/train/benin/20260404_150537__train__benin__all-folds__baseline`

with the completed FixMatch Benin-to-SA fine-tuning family:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/finetune/benin_to_sa/*__finetune__benin_to_sa__src-fold*__fixmatch`

and writes:

`/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/baseline_roc_comparison/benin_to_sa/20260413__fixmatch__source-fold-average/`
