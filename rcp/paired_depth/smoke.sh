#!/usr/bin/env bash
set -euo pipefail

# Run inside the prepared single-GPU environment. Outputs are software checks.
study_root=${1:-/scratch/users/falke/benin-paired-depth-dann}
run_tag=${2:-real}
python_bin="$study_root/env/runtime/bin/python"
cd "$study_root/code"

"$python_bin" - "$study_root" "$run_tag" <<'PY'
import json
import sys
from pathlib import Path

from ultrai.paired_depth.manifest import write_json

root = Path(sys.argv[1])
tag = sys.argv[2]
if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
    raise ValueError("Use a simple alphanumeric smoke-run tag")
for label in ("source", "full"):
    config = json.loads((root / f"artifacts/configs/{label}-p0-s42.json").read_text())
    config.update(output=str(root / f"artifacts/verification/{tag}-{label}"),
                  source_checkpoint=None, max_hours=0.5, effective_batch_size=2,
                  checkpoint_interval_seconds=30)
    write_json(root / f"artifacts/configs/smoke-{tag}-{label}-p0-s42.json", config)
PY

"$python_bin" -m ultrai.paired_depth smoke --config "$study_root/artifacts/configs/smoke-$run_tag-source-p0-s42.json"
"$python_bin" -m ultrai.paired_depth smoke --config "$study_root/artifacts/configs/smoke-$run_tag-source-p0-s42.json" --resume "$study_root/artifacts/verification/$run_tag-source/last.pt"
"$python_bin" -m ultrai.paired_depth smoke --config "$study_root/artifacts/configs/smoke-$run_tag-full-p0-s42.json"
