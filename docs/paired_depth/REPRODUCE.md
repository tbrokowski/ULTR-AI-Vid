# Reproducing Benin paired-depth training

Run these commands from the repository root on a Linux NVIDIA GPU machine or
inside a cluster job. No RCP account or scheduler is required. See [the method](REPORT.md)
for the architecture and [optional RCP execution](../../rcp/paired_depth/README.md)
for that deployment route.

## 1. Environment

Use Python 3.11, PyTorch 2.8.0, torchvision 0.23.0 and CUDA 12.6 wheels. The
[official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v2-8-0)
document this wheel combination. Supporting dependencies are pinned in
`requirements/paired-depth-support.lock`. A compatible NVIDIA driver is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/paired-depth-cu126.txt
python -m pip check
python -m ultrai.paired_depth --help
python -m pytest tests/paired_depth -q
```

Use one GPU per process, with enough memory for every eligible scan in a patient
bag. Microbatch one preserves the effective optimizer batch through accumulation.
Keep sites and frame counts matched across comparison arms.

The optional digest-pinned container works without RCP:

```bash
docker build -f rcp/paired_depth/Dockerfile -t benin-paired-depth:torch2.8-cu126 .
docker run --rm --gpus all benin-paired-depth:torch2.8-cu126 --help
```

Mount input data read-only and a separate writable study directory. Generate
configurations using paths visible inside the container and keep mounts stable.
Mount metadata and splits for preparation: the image excludes `Data/` and patient
outputs. Docker is optional; the Python commands below work directly in the venv.

## 2. Paths and access

Replace these examples with absolute paths on your machine. Keep the study directory
outside the checkout and on storage appropriate for patient data.

```bash
umask 077
export BENIN_STUDY_ROOT=/absolute/path/to/benin-study
export BENIN_EXISTING_VIDEOS=/absolute/path/to/existing/BeninVideos
export BENIN_METADATA=/absolute/path/to/processed_files_2.csv
export BENIN_LABELS=/absolute/path/to/labels_multidiagnosis.csv
export BENIN_SPLITS=/absolute/path/to/test_files
mkdir -p "$BENIN_STUDY_ROOT/artifacts"
hf auth login
```

Use a read token authorized for `TrustBeninVideos/Trust-Benin-Videos`. The pinned
dataset revision is `12cd0de88e25f39adf279e0105b8cdaf138593b4`. Use Hugging Face's
credential store; do not put tokens in configurations or command history. CLIP
downloads from a pinned public revision. Offline setups can set `clip_snapshot`
to a complete local copy of that revision before training.

| Input | Required structure |
|---|---|
| Videos | MP4 files referenced by acquisition metadata and verified against the pinned inventory. |
| Metadata | Existing `processed_files_2.csv` schema, including original/renamed filenames and depth; see `manifest.py` for parsing. |
| Labels | Existing `labels_multidiagnosis.csv` schema with patient TB labels and site pathology columns; missing pathology is masked. |
| Splits | `Fold_0.csv` through `Fold_4.csv`, with `train_ids`, `valid_ids`, `test_ids`. Nonempty, patient-disjoint partitions sharing the same test cohort. |

The preparation CLI also has defaults for the repository's original `Data/` paths.
Explicit paths make input provenance and deployment clearer.

## 3. Verify and prepare

```bash
python -m ultrai.paired_depth inventory \
  --output "$BENIN_STUDY_ROOT/artifacts/hf-inventory.json"
python -m ultrai.paired_depth.inventory \
  --existing "$BENIN_EXISTING_VIDEOS" \
  --destination "$BENIN_STUDY_ROOT/artifacts/videos" \
  --inventory "$BENIN_STUDY_ROOT/artifacts/hf-inventory.json"
export BENIN_VIDEOS="$BENIN_STUDY_ROOT/artifacts/videos"
python -m ultrai.paired_depth prepare \
  --metadata "$BENIN_METADATA" --labels "$BENIN_LABELS" \
  --splits "$BENIN_SPLITS" --videos "$BENIN_VIDEOS" \
  --inventory "$BENIN_STUDY_ROOT/artifacts/hf-inventory.json" \
  --output "$BENIN_STUDY_ROOT/artifacts/data" --workers 8
```

The overlay verifies all existing files before downloading missing ones. It links
verified videos without rewriting originals and rejects conflicting files. An empty
existing directory can be used for an initial download. Preserve original paths:
the overlay uses symbolic links.

Preparation decodes every frame, verifies hashes, deduplicates identical metadata,
excludes unresolved conflicts/corrupt recordings, joins labels and splits, and pairs
exact depths by patient, anatomical site and recording counter. Inspect `data/summary.json`
and restricted `audit.json`. A `--skip-decode` manifest cannot train. Matching identifiers
do not establish synchronized frames.

Optional CPU cache with identical deterministic pixels to video decoding:

```bash
python -m ultrai.paired_depth.cache_frames \
  --manifest "$BENIN_STUDY_ROOT/artifacts/data/manifest.json" \
  --videos "$BENIN_VIDEOS" \
  --output "$BENIN_STUDY_ROOT/artifacts/frames32" --workers 8
```

The resized uint8 cache is keyed by video hash, frame count and image size. It has
no learned statistics and can exceed compressed video size substantially. Omit
`--frame-cache` below to decode videos directly.

## 4. Verify the GPU and generate configurations

```bash
python -m ultrai.paired_depth.verification \
  --output "$BENIN_STUDY_ROOT/artifacts/verification"
python -m scripts.paired_depth.experiments configure \
  --root "$BENIN_STUDY_ROOT" --videos "$BENIN_VIDEOS" \
  --frame-cache "$BENIN_STUDY_ROOT/artifacts/frames32" \
  --partition 0 --seed 42 --epochs 6
```

Verification exercises CUDA forward/backward, real HMV-MIL representation gradients,
synthetic classifier overfit and exact checkpoint prediction regeneration. These
are software checks, not patient model evaluations.

Configuration generation writes resolved JSON (valid YAML) in `artifacts/configs`;
it neither submits jobs nor overwrites files. Comparisons share a source checkpoint,
six epochs and a frozen CLIP encoder. Source pretraining targets at most 200 epochs,
stops after 50 non-improving epochs and has a ten-hour process limit. Comparisons
have a three-hour process limit. Source architecture/settings come from
`configs/cscs/train.yaml`; study settings and arms are in `configs/paired_depth/`.

For a new protocol, choose an epoch schedule using timing checks before viewing
comparison validation scores. Keep preprocessing, initialization and update budgets
matched. Time-limited runs may not finish their schedule: inspect `progress.json`,
including applied and AMP-skipped updates, and identify under-budget runs explicitly.

## 5. Train and resume

```bash
python -m ultrai.paired_depth train \
  --config "$BENIN_STUDY_ROOT/artifacts/configs/source-p0-s42.json"
for arm in source15 both_supervised consistency dann conditional full grid0 grid1 grid2 stress-dann stress-conditional; do
  python -m ultrai.paired_depth train \
    --config "$BENIN_STUDY_ROOT/artifacts/configs/${arm}-p0-s42.json" || break
done
```

Start comparisons after the source has completed validated epochs and saved
`best.pt` and `last.pt`. The `source15` comparison is a supervised continuation,
with the same frozen encoder and budget as adaptation. The source-only checkpoint
is a separate stage, not the matched comparison baseline.

Validation runs after each epoch. Best checkpoints maximize all-site 15 cm AUROC
for `all15` training and matched 5 cm AUROC for paired training. Test evaluation
requires a separate explicit command and protocol freeze.

Send SIGTERM, SIGINT or SIGUSR1 to interrupt and allow checkpoint-writing time.
Resume using the original configuration:

```bash
python -m ultrai.paired_depth train \
  --config "$BENIN_STUDY_ROOT/artifacts/configs/consistency-p0-s42.json" \
  --resume "$BENIN_STUDY_ROOT/artifacts/runs/consistency-p0-s42/last.pt"
```

State includes model/domain heads, optimizer, scheduler, scaler, random generators,
sampler position, accumulated gradients and elapsed process time. Resumption does
not grant a new time budget. Loading requires the original code snapshot and exact
resolved configuration, paths and manifest. Preserve the original software snapshot
with older checkpoints for exact resumption and evaluation.

## 6. Select and replicate

After all predefined partition-0 comparisons finish:

```bash
python -m scripts.paired_depth.experiments select \
  --run-root "$BENIN_STUDY_ROOT/artifacts/runs" \
  --output "$BENIN_STUDY_ROOT/artifacts/selection.json"
```

The fixed rule considers completed candidates, applies a maximum 0.01 ordinary
15 cm AUROC loss against `source15`, then maximizes matched 5 cm validation AUROC
with a simpler-component tie break. If none passes retention, the immutable output
records `retention_constraint_satisfied: false`; that fallback is not a successful
retention result. Prevalence-stress arms are excluded from selection.

Repeat the selected method and both supervised baselines over partitions 1-4, then
prioritize seeds 43 and 44 across all five partitions. Generate each set with the
appropriate `--partition` and `--seed`. Train its matching source first. Do not
reselect on later partitions. If a grid entry is selected, copy its exact loss/style
settings into new candidate configurations before training; preserve the matching
source checkpoint and all other settings.

Wrap the same `train` command with SLURM, Kubernetes or another scheduler if needed.
Request one GPU per process and allow checkpoint time before scheduler termination.
The trainer limits process time; it does **not** enforce a shared allocation budget
across independent manual jobs. Study policy is at most three GPUs and 200 allocated
GPU-hours, reserving 20 hours for final evaluation. Account for setup, failures and
idle allocations through the scheduler. The optional RCP controller implements this.

## 7. Freeze and evaluate

List every completed configuration for final comparisons across all included
partitions/seeds. Freeze the full list before test access. This example shows one
pair; `consistency` here is a command example, not a stated selection outcome:

```bash
python -m ultrai.paired_depth freeze \
  --configs "$BENIN_STUDY_ROOT/artifacts/configs/source15-p0-s42.json" \
            "$BENIN_STUDY_ROOT/artifacts/configs/consistency-p0-s42.json" \
  --selection "$BENIN_STUDY_ROOT/artifacts/selection.json" \
  --output "$BENIN_STUDY_ROOT/artifacts/frozen-test-protocol.json"
for arm in source15 consistency; do
  python -m ultrai.paired_depth evaluate \
    --config "$BENIN_STUDY_ROOT/artifacts/configs/${arm}-p0-s42.json" \
    --checkpoint "$BENIN_STUDY_ROOT/artifacts/runs/${arm}-p0-s42/best.pt" \
    --split test \
    --freeze-file "$BENIN_STUDY_ROOT/artifacts/frozen-test-protocol.json" || break
done
```

The freeze records configurations, validation selection/metrics and checkpoint
hashes. It refuses overwrite and freezing after test predictions already exist.
Test evaluation rejects a checkpoint absent from the freeze. To verify regeneration
before test access, evaluate with `--split valid` and no freeze argument; compare
against a preserved copy of the original predictions/metrics. Keep paths and code
unchanged.

## 8. Generate separate analysis artifacts

Create a restricted JSON list of `[baseline_run_directory, candidate_run_directory]`
pairs using absolute paths, one pair per matching partition/seed. Then run:

```bash
python -m ultrai.paired_depth.reporting \
  --summary "$BENIN_STUDY_ROOT/artifacts/data/summary.json" \
  --comparisons "$BENIN_STUDY_ROOT/artifacts/paired-run-directories.json" \
  --output "$BENIN_STUDY_ROOT/artifacts/analysis"
```

Aggregation requires identical patients and labels across runs within each view;
it rejects duplicate predictions and unequal cohorts rather than intersecting them.
It reports patient AUROC/AP, paired differences and 95% intervals from 2,000 joint
patient-bootstrap samples. Metrics are averaged across runs, not ensembled.
Partition variation within seeds and seed variation within partitions are separate
from patient uncertainty. Do not pool differing validation cohorts with this shared-test
aggregator. An interval including zero makes a difference inconclusive. Report the
0.01 retention threshold and absolute 15 cm AUROC 0.82 criterion separately.

## Artifacts and verification boundaries

Run directories save `resolved_config.json`, `provenance.json`, `executions.json`,
`training_domain_frequencies.json`, `progress.json`, `history.json`, checkpoints and
validation outputs. Explicit evaluation adds metrics and prediction files. Preserve
the complete code snapshot, including uncommitted changes: a commit alone does not
identify an edited checkout. Keep manifests, predictions, weights, credentials and
generated analyses outside the handover branch.

Tests cover exact depths, conflicts, missing/corrupt files, patient separation,
padding/pairs, donor exclusion, GRL signs, detached conditioning, class weights,
scan gradients, disabled-loss equivalence, interruption/resumption, AMP recovery
and joint patient bootstrap behavior. Passing tests verifies these software properties;
it does not establish training convergence or a performance outcome.
