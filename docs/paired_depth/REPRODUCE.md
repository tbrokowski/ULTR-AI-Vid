# Benin-only paired-depth DANN

This entrypoint is separate from the existing SA DANN trainer. Existing commands
and model state-dict keys remain unchanged. This study uses supervised Benin TB
and pathology labels at both depths. It does not evaluate South Africa.

## Current verified status

- Base commit: `dba7e2d4b9318df8662fc2eeef8437d1b772beb0`.
- Dataset revision: `12cd0de88e25f39adf279e0105b8cdaf138593b4`.
- Existing RCP videos: 9,995; 8,087,878,316 bytes; all fully decoded successfully.
- Mixed image/video metadata matching groups: 6,015.
- Available labeled pairs before conflict checks: 2,807 across 470 patients.
- After excluding unresolved acquisition metadata conflicts: 2,774 pairs across
  467 patients. The matched test cohort has 563 pairs across 94 patients.
- The five patient split files share one 101-patient test cohort. Report these as
  training/validation partitions, never independent test folds.
- Private HF authentication and revision hash verification are still required
  before research training. No new scientific AUROC/AP result is claimed.

## Environment and access

Connect the EPFL VPN and authenticate Run:AI on `jumphost.rcp.epfl.ch`. The current
verified SSH route is:

```bash
ssh -o HostName=10.92.14.13 -o HostKeyAlias=haas001.rcp.epfl.ch \
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no rcp
runai login
```

The IP is an observed September 6, 2026 route, not a permanent DNS replacement.
The known host key was verified; do not disable SSH host-key verification.

The study uses project `light-falke`, namespace `runai-light-falke`. Jumphost
storage is `/mnt/light/scratch/users/falke/benin-paired-depth-dann`; the corresponding
container directory is `/scratch/users/falke/benin-paired-depth-dann`. Write only
within that directory. Benin input is mounted read-only at
`/benin/datasets/ULTR-AI/LusBeninVideos`.

`rcp/paired_depth/job.py` renders a supported `TrainingWorkload` with the existing
`light-scratch` PVC. This cluster disallows direct NFS volumes and direct user
creation of `RunaiJob` objects. Its workload schema does not support PVC subpaths;
the PVC has separate read-only input and writable scratch mounts. The writable
mount exposes LIGHT scratch, but study commands write only in the user's isolated
directory. The container starts in `/tmp` and then changes directory after UID/GID
setup to avoid NFS root-squash startup failures. Verify the rendered pod's `/benin`
volume mount has `readOnly: true`.

Runtime image:

```text
pytorch/pytorch@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69
```

This contains Python 3.11.13, PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126 and
CUDA 12.6. Supporting dependencies are fully version-locked in
`rcp/paired_depth/support.lock`; each run additionally saves the installed package
versions. `rcp/paired_depth/setup.sh` creates a private scratch virtual environment.
The CLIP revision is pinned to `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
GPU compatibility is checked with an actual forward/backward before using a pool.
The A100 40 GB and V100 32 GB checks passed. Do not use the historical NVIDIA
25.08 image for V100.

For a standalone image, build with
`docker build -f rcp/paired_depth/Dockerfile -t benin-paired-depth:torch2.8-cu126 .`.
The Dockerfile-specific ignore file excludes clinical data, predictions and weights
from the build context. On RCP the equivalent pinned base plus scratch environment
was used, avoiding a new container registry dependency.

## Inventory, manifest, and decoding

Authenticate HF with a read token that can access the private dataset. Keep the
token out of scripts, Git, logs and job arguments. On the jumphost the prepared CLI
is `/mnt/light/scratch/users/falke/benin-paired-depth-dann/env/hf/bin/hf`.

Inside the prepared runtime, from the repository root:

```bash
python -m ultrai.paired_depth inventory \
  --output /scratch/users/falke/benin-paired-depth-dann/artifacts/hf-inventory.json
python -m ultrai.paired_depth prepare \
  --videos /benin/datasets/ULTR-AI/LusBeninVideos \
  --inventory /scratch/users/falke/benin-paired-depth-dann/artifacts/hf-inventory.json \
  --output /scratch/users/falke/benin-paired-depth-dann/artifacts/data \
  --workers 8
```

The first command can also run on the authenticated jumphost with its `env/hf`
Python and `PYTHONPATH` set to the study checkout; inventory generation imports no
PyTorch. Transfer only the resulting inventory JSON to the runtime, not the token.
The present CPU audit omitted `--inventory`, so its manifest explicitly remains
unverified and cannot train.

If pinned files are missing, first verify every present file and create an overlay:

```bash
python -m ultrai.paired_depth.inventory \
  --existing /benin/datasets/ULTR-AI/LusBeninVideos \
  --destination /scratch/users/falke/benin-paired-depth-dann/artifacts/videos \
  --inventory /scratch/users/falke/benin-paired-depth-dann/artifacts/hf-inventory.json
```

The overlay links existing verified files and downloads only missing files at the
pinned revision. Rerun `prepare` on that overlay and set `videos` accordingly.
Do not rewrite the existing data. Full frame decoding checks truncation as well
as empty videos. Conflicting original acquisition filenames are excluded even
when their renamed file keys match. Matching identifiers do not imply temporal
synchronization.

To avoid repeating video decoding in every epoch, the CPU-only frame cache stores
the same deterministically resized uint8 pixels, keyed by source video SHA256,
frame count and image size:

```bash
python -m ultrai.paired_depth.cache_frames \
  --manifest /scratch/users/falke/benin-paired-depth-dann/artifacts/data/manifest.json \
  --videos /benin/datasets/ULTR-AI/LusBeninVideos \
  --output /scratch/users/falke/benin-paired-depth-dann/artifacts/frames32
```

This reproducible restricted cache uses approximately 45 GB for the current
eligible videos and contains no learned statistics. Tests verify exact decoded
pixel equivalence. The `frame_cache` YAML setting controls its use.

## Tests and GPU checks

```bash
python -m pytest tests/paired_depth -q
python -m ultrai.paired_depth.verification \
  --output /scratch/users/falke/benin-paired-depth-dann/artifacts/verification/a100-40g
```

Tests cover exact depths, identical duplicates, conflicting acquisitions, missing
and corrupt files, split leakage, pair mappings, padding, training-only donors,
Fourier phase and temporal consistency, GRL signs, detached conditioning,
class-frequency weights, nonzero representation gradients, disabled-loss
equivalence, checkpoint reload, actual-loop signal interruption and exact resume,
and paired bootstrap behavior. Synthetic overfit/GPU outputs are software checks,
not research results. Follow them with a two-microbatch real-data smoke run using
`python -m ultrai.paired_depth smoke --config ...` after revision verification.

## Experiment configurations and launch

The six comparison definitions and the finite weight grid are recorded before
test access in `configs/paired_depth/study.yaml`. Generate resolved JSON (also
valid YAML) configurations with:

```bash
python -m rcp.paired_depth.experiments configure \
  --output /scratch/users/falke/benin-paired-depth-dann/artifacts/configs \
  --partition 0 --seed 42 --epochs 20
```

The 20-epoch adaptation default must be confirmed or reduced from timing-only
smoke evidence before examining validation scores, so all comparisons can finish
with equal training budgets. Source models target the existing 200-epoch schedule,
bounded by 10 GPU-hours per source run. The evaluation baseline is a 15 cm
supervised continuation of that same source checkpoint, matched to the adaptation
stage's updates and frozen backbone. The both-depth supervised baseline shares
the initialization, normalization and update schedule with the adaptation arms.

Each patient bag includes every eligible recording for its view. Source training
uses all exact 15 cm recordings; paired training uses the matched 5/15 acquisitions.
All arms use 32 uniformly sampled frames at 224 x 224 and the existing deterministic
ImageNet normalization. This removes the legacy train/evaluation normalization
mismatch. Missing pathology labels are masked. No validation or test patient can
supply a style donor.

The source retains the architecture, parameter-group learning rates, pathology
weights and cosine restart schedule from `configs/cscs/train.yaml`. With one GPU,
microbatch one and accumulation preserve an effective 280-patient batch. Legacy
pathology, patient, and representation phase gradient routing is retained; the
four disjoint pathology heads share one forward. All groups step at a common
accumulation boundary and the partial final batch is normalized by its actual
size. Representation updates occur each microbatch, fixing the legacy alternating
backbone update condition that can miss an accumulation boundary. These are
explicit implementation corrections; the run is not a bitwise replay of the old
distributed source trainer.

On the jumphost, submit through the budget ledger (not raw `runai submit`):

```bash
python3 /mnt/light/scratch/users/falke/benin-paired-depth-dann/code/rcp/paired_depth/launch.py submit \
  --name bd-source-p0-s42 --hours 10 --pool a100-40g -- \
  /scratch/users/falke/benin-paired-depth-dann/env/runtime/bin/python \
  -m ultrai.paired_depth train \
  --config /scratch/users/falke/benin-paired-depth-dann/artifacts/configs/source-p0-s42.json
python3 /mnt/light/scratch/users/falke/benin-paired-depth-dann/code/rcp/paired_depth/launch.py watch
```

The watcher reserves at most 180 GPU-hours for training and retains 20 for final
evaluation. It meters GPU allocation from scheduling through termination, including
startup, and enforces at most three study/project GPU allocations. Per-job `timeout`
also bounds execution independently. The watcher must remain running and CLI
authentication must remain valid. It never deletes jobs outside this study ledger.
To resume a stopped run, submit a new job name with the same config and
`--resume /.../runs/<run>/last.pt`. The model, optimizer, scheduler, AMP scaler,
accumulated gradients, RNG state, and sampler cursor are restored. Changing the
resolved config or manifest is an error.

Priority: source and six partition-0 comparisons, the predefined grid, selected
method plus both supervised baselines across partitions 1-4, then extra seeds 43
and 44 while the cap permits. Run prevalence stress separately for ordinary and
class-balanced conditional DANN. That stress deterministically retains half the
positive 5 cm domain examples and half the negative 15 cm domain examples in the
adversarial stream; TB supervision stays available. Weights use only the resulting
training-domain counts. This isolates domain-class-prior confounding.

## Freeze and final evaluation

Select on partition 0 validation only:

```bash
python -m rcp.paired_depth.experiments select \
  --run-root /scratch/users/falke/benin-paired-depth-dann/artifacts/runs \
  --output /scratch/users/falke/benin-paired-depth-dann/artifacts/selection.json
```

This enforces the 0.01 15 cm validation retention constraint when feasible, then
maximizes matched-5 cm AUROC, breaking ties by fewer components. If no candidate
meets retention, that failure is explicit. Incomplete or unequal-epoch comparisons
cannot win selection. Fix the selected settings for the remaining partitions and
seeds; do not run a new search on them.

Use `ultrai.paired_depth.reporting.freeze(config_paths, output, selection)` to save
the resolved configs, validation evidence and exact checkpoint hashes before
opening the shared test cohort. The freeze file cannot be overwritten. Then:

```bash
python -m ultrai.paired_depth freeze --configs /.../run1.json /.../run2.json \
  --selection /.../selection.json --output /.../frozen-test-protocol.json
```

```bash
python -m ultrai.paired_depth evaluate --config /.../config.json \
  --checkpoint /.../runs/<run>/best.pt --split test \
  --freeze-file /.../frozen-test-protocol.json
python -m ultrai.paired_depth.reporting --summary /.../data/summary.json \
  --comparisons /.../paired-run-directories.json --output /.../aggregate-report
```

`paired-run-directories.json` is a list of `[baseline_directory, candidate_directory]`
for matching partitions and seeds. Reporting rejects patient-cohort mismatches;
it never silently intersects predictions. It jointly bootstraps patients across
all runs, reports average per-run metrics rather than an ensemble, and separates
variation across partitions within seeds from variation across seeds within
partitions. Regenerate metrics from saved checkpoints to verify the report.

## Restricted outputs and future SA extension

Keep manifests, acquisition audits, predictions, checkpoints and clinical metadata
in restricted storage. Commit only source, configuration templates, aggregate
counts/results and the English report. No report or messages are sent by this
pipeline.

Future unlabeled SA data can implement the `PatientBags` input contract with
`tb_labels = -1` and pathology labels `-1`, a separate target domain, no paired
indices, and target losses disabled where labels are absent. Its patient split,
normalization, model selection and label-shift policy require a separate protocol.
The present CLI deliberately has no SA data path or SA evaluation mode.
