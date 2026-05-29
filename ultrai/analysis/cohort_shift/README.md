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

For the domain-adaptation report figure, use the compact report profile:

```bash
python3 ultrai/analysis/cohort_shift/cohort_shift_overview.py \
  --profile report \
  --stem 04fig_cohort_shift \
  --output-dir /users/lxflk/domain-adaption-report/figures/cohort_shift
```

This profile omits lower-priority rows and writes report-ready SVG, PNG, and
CSV outputs using the same underlying cohort-shift calculations.
