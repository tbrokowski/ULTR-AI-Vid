# Cohort Shift Overview

Generate a slide-ready Benin-vs-South-Africa cohort-shift graphic:

```bash
python3 ultrai/analysis/cohort_shift/cohort_shift_overview.py
```

Default outputs are written under:

```text
/capstor/scratch/cscs/lxflk/ULTR-AI-Vid/runs/cohort_shift_overview/benin_vs_sa/
```

The script writes:

- `cohort_shift_dashboard.svg`
- `cohort_shift_dashboard.png` when ImageMagick `convert` is available
- `cohort_shift_metrics.csv`
- `cohort_shift_summary.json`
- `manifest.json`

All plotted variables are percentages so they can be shown on one common axis.
Continuous variables are represented by clinically interpretable thresholds,
for example BMI `<18.5` and SpO2 `<95%`.

