# Implementation and evidence status - September 6, 2026

The implementation is under review. Research training and frozen-test evaluation
have **not** been completed. The approximately +0.03 AUROC email claim is unverified.

Completed:

- New branch from the requested base commit, with separate preparation, training,
  evaluation, experiment configuration, budget control and reporting modules.
- Full decode audit of 9,995 existing videos: zero decoding failures; 2,774 valid
  pairs across 467 patients after unresolved acquisition conflicts are excluded.
- Twenty passing CPU tests for protocol invariants, real HMV-MIL gradient paths and loss-disabled
  equivalence, actual training-loop interruption/resume, and metrics regeneration.
- GPU verification with pinned PyTorch 2.8.0/CUDA 12.6 on A100 40 GB, V100 32 GB
  and A100 80 GB (the current `default` pool).
- Synthetic classifier overfit from BCE about 0.65 to about 0.0004; exact
  checkpoint-to-prediction regeneration. These are software checks only.
- Restricted scratch environment and a GPU budget ledger/watcher; all verification
  jobs have exited. Verification consumed about 0.0664 allocated GPU-hours.
- Deterministic frame cache prepared on CPU: 9,427 eligible recordings,
  45,410,085,248 bytes. It contains resized uint8 frames, with exact pixel
  equivalence tested against uncached decoding.

Outstanding:

1. Authenticate to the private HF dataset on the jumphost, export the pinned file
   inventory, and verify every local SHA256. The current manifest explicitly
   records `revision_verified: false`, and research training fails on it.
2. Run real-data source and adaptation smoke/timing checks, freeze the common
   adaptation epoch budget using timing only, and execute the prescribed comparisons.
3. Select the method on validation, repeat the selected method and both baselines
   over the remaining partitions, and add seeds within the budget.
4. Freeze checkpoints before final shared-cohort testing; regenerate paired
   AUROC/AP and patient-bootstrap intervals; replace the pending-results report.

South African data and forecasts remain outside the empirical study. No messages
or reports have been sent to Thomas or anyone else.
