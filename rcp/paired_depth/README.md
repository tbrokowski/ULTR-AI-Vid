# Optional EPFL RCP execution

The [portable guide](../../docs/paired_depth/REPRODUCE.md) defines the model workflow.
These scripts supply RCP storage, authentication, scheduling and allocation controls.
They are unnecessary for ordinary Python training on another machine.

## Storage and environment

Use project `light-falke`, namespace `runai-light-falke`, through
`jumphost.rcp.epfl.ch`, with EPFL network/VPN access and authenticated Run:AI CLI.
Authenticate on the jumphost with `runai login`. Keep credentials out of jobs/logs.
The study defaults are:

| Context | Path |
|---|---|
| Jumphost study root | `/mnt/light/scratch/users/falke/benin-paired-depth-dann` |
| Container study root | `/scratch/users/falke/benin-paired-depth-dann` |
| Read-only video mount | `/benin/datasets/ULTR-AI/LusBeninVideos` |
| Runtime Python | `/scratch/users/falke/benin-paired-depth-dann/env/runtime/bin/python` |

`job.py` renders a supported TrainingWorkload against the existing `light-scratch`
PVC. Input is mounted read-only; generated artifacts go into the isolated user
study directory. The writable PVC exposes wider scratch, so commands must retain
the isolated output paths. The container establishes UID/GID before entering the
checkout to accommodate NFS root squash. Keep the code under the study root's
`code/` directory and do not replace a snapshot used by active training.

`setup.sh` prepares the pinned runtime environment. Run inventory and preparation
explicitly with the portable guide, including `--inventory`; a manifest generated
without revision verification cannot train. The digest-pinned Dockerfile and Python
support lock are shared with the portable environment.

## Single jobs and finite plans

From the prepared container runtime, configuration generation with
`python -m rcp.paired_depth.experiments configure --output ... --partition 0 --seed 42 --epochs 6`
uses these RCP path defaults. It delegates to the portable generator. Submit one
GPU per training process using `launch.py`; inspect its CLI help for `submit`,
`watch` and the configured pool names. Verify each pool before scheduling it and
save `gpu_verification.json` under `artifacts/verification/<pool>/`.

For a bounded initial comparison queue, `adaptation_preflight.py` measures training
throughput without test access and `freeze_queue.py` freezes a finite configuration
set before comparison validation is observed. `queue.py --plan <plan.json>` runs it
through the ledger. Each job retains its own timeout. Missing/incomplete runs,
configuration drift and failures stop further submissions; the plan is immutable.
Do not use the initial tuning queue to reselect settings after replication.

After validation selection, `replication.py prepare` pins the selected method and
source dependencies, and `replication.py run --plan <plan.json>` schedules the plan.
Defaults cover partitions 1-4, seed 42. Additional seeds use `--seed 43` or `--seed 44`
with `--partitions 0 1 2 3 4`. The `series --plans <seed43-plan> <seed44-plan>` command
finishes the higher-priority seed before starting the next. Grid-entry selection
requires an explicit copied candidate configuration; the replication helper rejects
unsupported choices instead of substituting another method.

Run controllers in a persistent terminal or detached process on one jumphost.
Restart the same immutable plan after a controller interruption. Do not start
concurrent controllers for the same plan or regenerate its files.

## Accounting and process health

The ledger and watcher meter scheduled-to-terminated GPU allocation, including setup,
failures and idle allocation. Training retains the 20-hour evaluation reserve within
the overall 200-hour cap and enforces at most three simultaneous GPU allocations.
Per-job process deadlines remain active independently of the watcher. Checkpoint and
stop at limits; preserve unfinished runs and report their actual completed budgets.

Controller locks use `/var/tmp/benin-paired-depth-dann-<uid>` on the jumphost's local
disk, with a namespace derived from the scratch root. This avoids NFS advisory-lock
stalls. Ledger files and plans remain on restricted LIGHT scratch. All controllers
must run on the same jumphost to share these local locks. Lock and Kubernetes requests
have finite timeouts. Compare ledger and plan-status modification times with the
current time: an existing process ID alone does not show that accounting advances.

Patient predictions and checkpoints stay on restricted storage. CPU-only
`validation_review.py` and `replication_review.py` produce separate analysis outputs;
they are not part of this method handover. Final test evaluation still requires the
portable workflow's explicit checkpoint freeze.
