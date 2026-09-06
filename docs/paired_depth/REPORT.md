# Benin paired-depth adaptation: internship handover

## Study status

No completed, frozen-test Benin results are available yet. The email's approximately +0.03 AUROC improvement has not been reproduced. No South African evaluation was performed. Twenty software tests and GPU checks on A100 40 GB, V100 32 GB and A100 80 GB passed. Private Hugging Face authentication is needed to finish revision verification before research training.

## Method

The study uses exact 5 cm and 15 cm Benin ultrasound recordings to learn representations that are less sensitive to acquisition depth. HMV-MIL's existing CLIP ViT-B/32 backbone, attention frame selector, pathology heads, anatomical integration, and patient attention pooling are retained. The source model uses exact 15 cm recordings. Subsequent training uses available TB and pathology labels at both depths, so this is **supervised within-Benin training**.

The ablations add scan-level and patient-level gradient reversal, paired cosine feature consistency, masked pathology and patient prediction consistency, detached TB-prediction conditioning, training-only class-frequency correction, and synthetic acquisition styles. The low-frequency Fourier transformation mixes amplitudes with a training-patient donor while preserving source phase. The transformations share parameters across each clip. Matching recording identifiers establish paired acquisitions, not synchronized frames; no frame-to-frame correspondence is assumed.

Mechanisms: [DANN](https://www.jmlr.org/beta/papers/v17/15-239.html), [conditional adversarial adaptation](https://papers.nips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html), [Fourier domain adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html).

## Data reconciliation and evaluation

The original 6,015 count combines image and video metadata matching groups. The available labeled video inventory yields 2,807 pairs across 470 patients before ambiguity and decoding checks. The current audit retains **2774 pairs across 467 patients**. The private dataset is pinned at `12cd0de88e25f39adf279e0105b8cdaf138593b4`; revision verification is **False**. Detailed exclusion counts are in the aggregate audit summary.

All recordings and synthetic derivatives inherit their patient's original partition. The five training/validation partitions share the same 101-patient test cohort; they are not independent test folds. Cross-depth evaluation uses matched-site patient bags at each depth. Ordinary 15 cm evaluation uses all eligible sites. Missing pathology annotations are masked, rather than treated as negative.

The protocol requires freezing configurations and checkpoint hashes using validation data before test evaluation. AUROC and average precision are calculated per patient. Paired 95% bootstrap intervals jointly resample the same patients across every model run. Partition and seed variation must be shown separately from patient-sampling uncertainty. A +0.03 cross-depth AUROC change is a target, not an acceptance criterion for selecting test results. A difference whose confidence interval includes zero is inconclusive. Stable 15 cm performance means no more than a 0.01 AUROC loss; reaching 0.82 absolute AUROC is reported separately.

## Evidence table

| Evidence type | Value | Interpretation |
|---|---|---|
| Historical report: source | AUROC 0.8853 ± 0.0128; AP 0.8729 ± 0.0247 | Earlier report values; not measured by this pipeline or directly comparable without protocol reconciliation. |
| Historical report: DANN, SA | AUROC 0.6932 ± 0.0647; AP 0.1803 ± 0.0718 | Earlier SA experiment; no SA data used in the present study. |
| Historical report: Benin retention | AUROC 0.8321 ± 0.0398; AP 0.7589 ± 0.0519 | Earlier reported result, not present-study replication. |
| August 29 email claim | About +0.03 Benin cross-depth AUROC | Unverified until the new frozen-test comparisons finish. |
| Newly measured Benin results | Pending. | Only completed pipeline outputs qualify as measured evidence. |
| Email SA forecast | AUROC about 0.75; AP about 0.27 | Untested expectation. Benin experiments cannot establish these values. |

## Reproduction and limitations

Start at Git commit `dba7e2d4b9318df8662fc2eeef8437d1b772beb0` and use branch `codex/benin-paired-depth-dann`. Follow `docs/paired_depth/REPRODUCE.md` for exact commands. Each run stores its resolved configuration, dataset-manifest hash, seed, source-checkpoint hash, software versions, model, optimizer, scheduler, scaler, random-number state, sampler position, and accumulated gradients in restricted LIGHT scratch. Public outputs contain aggregates only.

The compute limit is 200 allocated GPU-hours with at most three GPUs, including a 20 GPU-hour reserve for final evaluation. Epoch targets may be curtailed by that cap; incomplete or under-trained runs must be identified explicitly. This study does not validate geographic transfer, clinical deployment, or South African performance. An unlabeled SA adapter is a future extension, subject to separate access and evaluation protocols.

## Core reproduction commands

Run inside the pinned runtime from the repository root, after authenticating the private inventory and completing the preparation commands in the reproduction guide. This is the partition-0, seed-42 source and matched baseline sequence; final test evaluation additionally requires the immutable freeze file and saved selection.

```bash
BENIN_STUDY_ROOT=/scratch/users/falke/benin-paired-depth-dann
python -m rcp.paired_depth.experiments configure --output "$BENIN_STUDY_ROOT/artifacts/configs" --partition 0 --seed 42 --epochs 20
python -m ultrai.paired_depth train --config "$BENIN_STUDY_ROOT/artifacts/configs/source-p0-s42.json"
python -m ultrai.paired_depth train --config "$BENIN_STUDY_ROOT/artifacts/configs/source15-p0-s42.json"
python -m ultrai.paired_depth evaluate --config "$BENIN_STUDY_ROOT/artifacts/configs/source15-p0-s42.json" --checkpoint "$BENIN_STUDY_ROOT/artifacts/runs/source15-p0-s42/best.pt" --split test --freeze-file "$BENIN_STUDY_ROOT/artifacts/frozen-test-protocol.json"
```
