# Benin paired-depth training: method handover

## Purpose and scope

This implementation studies sensitivity to acquisition depth using Benin lung
ultrasound videos recorded at exactly 5 and 15 cm. It retains the existing HMV-MIL
attention-pooling architecture and provides controlled comparisons of supervision,
paired consistency, domain adversarial losses and synthetic acquisition styles.
Available TB and pathology labels are used at both depths: this is supervised
within-Benin training. South African data are not an input to this workflow.

The implementation branches from `dba7e2d4b9318df8662fc2eeef8437d1b772beb0`.
The separate `ultrai.paired_depth` entrypoint preserves the existing training
commands and model state-dict names.

## Inputs and patient bags

The dataset is `TrustBeninVideos/Trust-Benin-Videos`, pinned at
`12cd0de88e25f39adf279e0105b8cdaf138593b4`. Preparation verifies existing file hashes
before downloading missing videos, fully decodes recordings and joins the original
acquisition metadata, patient labels and split files. It deduplicates identical
records and excludes unresolved conflicts, missing labels/files and undecodable
videos. The generated audit records the exclusion reasons and input hashes.

Pairs require the same patient, anatomical site and recording counter, with one
exact 5 cm and one exact 15 cm acquisition. These identifiers establish paired
acquisitions, not synchronized frames. The model processes uniformly sampled clips
and compares scan representations; it does not impose frame-to-frame alignment.

Each bag contains all eligible recordings for its view. Paired bags use matched
acquisitions at both depths; ordinary 15 cm bags include all eligible sites. Clips
contain 32 uniformly sampled frames, resized to 224 by 224 and normalized consistently
for training and evaluation. Padding masks exclude absent scans. Missing pathology
annotations remain unknown rather than becoming negative labels. Every recording
and synthetic derivative follows its patient's existing partition.

## Architecture and training stages

CLIP ViT-B/32 encodes the frames, using model revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. HMV-MIL retains its learned attention
frame selector, scan representations, pathology heads, anatomical integration,
patient attention pooling and TB classification head.

Source training uses exact 15 cm videos and available Benin labels. It retains the
source parameter groups and learning rates in `configs/cscs/train.yaml`, including
CLIP fine-tuning. Pathology, patient and backbone phases retain the legacy gradient
routing. Microbatch one with accumulation targets an effective batch of 280 patients;
the final partial accumulation window uses its actual patient count.

All comparison arms then load the same matching source checkpoint for a given
partition and seed. The CLIP encoder is frozen; downstream representation and
classification layers remain trainable. The 15 cm supervised comparison receives
the same continuation budget as both-depth training. It is distinct from the
source-only checkpoint. Comparisons use a common epoch schedule selected before
viewing their validation scores.

| Configuration | Additions to the matched continuation |
|---|---|
| `source15` | Exact 15 cm supervision using all eligible sites. |
| `both_supervised` | Supervision on matched bags at both depths. |
| `consistency` | Both-depth supervision plus paired feature and prediction consistency. |
| `dann` | Consistency plus scan and patient domain adversarial losses. |
| `conditional` | Detached TB-prediction conditioning and training-class balancing. |
| `full` | Conditional training plus gain, speckle, resampling and Fourier styles. |

## Objectives and transformations

The supervised objective combines patient TB classification and masked pathology
classification. Additional terms have independent weights in `losses`:
`scan_domain`, `patient_domain`, `feature` and `prediction`. Zero weights disable
the corresponding terms. `grl_max` sets the gradient-reversal schedule's maximum;
`conditioning` and `class_balance` switch the corresponding domain mechanisms.

The domain heads predict acquisition depth from differentiable scan and patient
representations. Gradient reversal negates and scales the representation gradient
while the domain classifiers learn to minimize their classification loss. Scan
losses are first averaged within patients and then across domains, avoiding extra
weight for patients with more recordings. Detached diagnostic features cannot supply
these training losses. This follows the mechanism of
[Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html).

Conditioning forms an outer product of the representation and the detached binary
TB prediction probabilities. The conditioning path cannot update TB predictions.
Class weights use training-only TB frequencies within each domain. They are not
renormalized by each microbatch's weight sum, which would cancel balancing at batch
size one. This implements a binary prediction-conditioning variant inspired by
[Conditional Adversarial Domain Adaptation](https://papers.nips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html).
A separate prevalence-stress option changes the patients contributing to the domain
losses by class and depth; the supervised patient population remains unchanged.

Feature consistency uses cosine distance between matched scan representations,
averaged with equal patient weight. Prediction consistency compares patient TB
probabilities and corresponding known pathology predictions, using pair mappings
and validity masks. It does not compare unaligned individual frames.

Style transformations use clip-wide parameters: multiplicative gain, a shared
speckle field, downsampling followed by upsampling, and low-frequency Fourier
amplitude mixing. Fourier donors are drawn only from training patients, with
amplitude averaged over donor time. The Fourier mixing step preserves source phase;
the final transformed intensities are clipped to the valid input range. The
mechanism follows [Fourier Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html).
Each transformation has a YAML switch; Fourier extent and blending have explicit
settings. Donors never come from validation or test patients.

<!-- pagebreak -->

## Evaluation protocol

Preserve the five original training/validation partitions, which share one test
cohort. They are not independent test folds. Evaluate patient TB predictions on
matched-site bags at each depth and on all eligible 15 cm sites. Select settings
using partition-0 validation and a predefined search space, then retain that choice
for other partitions and seeds. Freeze every final configuration and checkpoint
hash before opening the test cohort.

Patient AUROC and average precision are computed for each run. Paired differences
use joint patient-bootstrap samples across all compared models; report 95% intervals.
An interval containing zero leaves the difference inconclusive. Model variation
across partitions within a seed and across seeds within a partition is reported
separately. Do not pool different validation cohorts into the shared-test analysis.
Retention uses a maximum 0.01 ordinary 15 cm AUROC loss relative to the matched
baseline; absolute 15 cm AUROC 0.82 is a separate protocol criterion.

The study settings define at most three concurrent single-GPU jobs and 200 allocated
GPU-hours, including 20 hours reserved for evaluation. A portable manual run needs
scheduler accounting to enforce the shared cap; the trainer itself enforces a
per-process deadline. Under-budget or interrupted comparisons must be identified
explicitly.

## Reproduction and continuation

Follow `docs/paired_depth/REPRODUCE.md` for environment installation, private input
verification, optional frame caching, portable configuration generation, source
training, matched comparisons, validation selection, freezing and evaluation.
The main command is `python -m ultrai.paired_depth`; configuration generation is
`python -m scripts.paired_depth.experiments`. RCP is optional.

Each run records resolved settings, seeds, input/code/checkpoint hashes and installed
package versions. Checkpoints also save optimizer, scheduler, scaler, random-number,
sampler and accumulated-gradient state. Resume using the identical code snapshot,
configuration and storage paths. Missing validation classes and incompatible
checkpoints fail explicitly.

A future unlabeled target-domain adapter can provide target videos, explicit domain
identifiers and supervision masks to the data/loss interfaces. It must exclude
unlabeled target records from supervised TB/pathology losses, prevent target test
data from entering training or donor pools, and define a separate evaluation
protocol. The current Benin CLI requires labeled patients; it does not implement
an unlabeled South Africa ingestion path.
