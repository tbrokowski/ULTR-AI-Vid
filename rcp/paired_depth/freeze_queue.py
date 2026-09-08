"""Freeze partition-0 comparisons using timing only; no validation-based tuning."""
import argparse
import datetime
import json
import math
from pathlib import Path

from .queue import CONTAINER_ROOT, host_path, sha256
from . import launch

ARMS = ("source15", "both_supervised", "consistency", "dann", "conditional",
        "full", "grid0", "grid1", "grid2", "stress-dann", "stress-conditional")


def timing_epochs(epoch_seconds, hours=3, maximum=20):
    # Retain five minutes for startup/checkpoints and 35% throughput headroom.
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("Invalid timing measurement")
    epochs = min(maximum, math.floor((hours * 3600 - 300) / (epoch_seconds * 1.35)))
    if epochs < 1:
        raise ValueError("One full epoch cannot fit the bounded run")
    return epochs


def freeze(drafts, preflight):
    drafts, preflight = Path(drafts), Path(preflight)
    timing = json.loads(preflight.read_text())
    if not timing["validation_regenerated_exactly"] or timing["test_data_read"]:
        raise ValueError("Preflight verification did not pass")
    epochs = timing_epochs(timing["estimated_full_epoch_seconds"])
    folder = launch.ROOT / "artifacts/configs/adaptation-p0"
    plan_path = launch.ROOT / "artifacts/scheduler/adaptation-p0-plan.json"
    if plan_path.exists() or folder.exists():
        raise FileExistsError("Adaptation freeze is immutable; use the existing plan")
    configs = {arm: json.loads((drafts / f"{arm}-p0-s42.json").read_text()) for arm in ARMS}
    common = configs["source15"]
    manifest = json.loads(host_path(common["manifest"]).read_text())
    train = set(manifest["partitions"][0]["train"])
    counts = {"all15": len({r["patient"] for r in manifest["records"]
                           if r["depth"] == 15 and r["patient"] in train}),
              "paired": len({p["patient"] for p in manifest["pairs"] if p["patient"] in train})}
    if sha256(host_path(common["source_checkpoint"])) != timing["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed since verification")
    folder.mkdir(parents=True)
    jobs = []
    for arm, cfg in configs.items():
        cfg.update(epochs=epochs, max_hours=3)
        config_path = folder / f"{arm}-p0-s42.json"
        launch.save(config_path, cfg)
        jobs.append({"name": "bd-" + arm.replace("_", "-") + "-p0-s42",
                     "config": str(CONTAINER_ROOT / config_path.relative_to(launch.ROOT)),
                     "config_sha256": sha256(config_path), "output": cfg["output"],
                     "epochs": epochs, "hours": 3, "pool": "a100-40g",
                     "expected_updates": epochs * math.ceil(counts[cfg["training_view"]] / cfg["effective_batch_size"])})
    plan = {"schema": 1, "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": "partition 0 seed 42 supervised continuation and adaptation comparisons",
            "source_checkpoint": common["source_checkpoint"],
            "source_checkpoint_sha256": timing["source_checkpoint_sha256"],
            "implementation_sha256": common["implementation_sha256"],
            "preflight_sha256": sha256(preflight), "epochs": epochs,
            "timing_rule": "min(20, floor((3*3600 - 300)/(full_epoch_seconds*1.35)))",
            "measured_estimated_full_epoch_seconds": timing["estimated_full_epoch_seconds"],
            "selection_basis": "training timing only; source validation used for checkpoint verification",
            "adaptation_validation_observed": False, "test_data_read": False, "jobs": jobs}
    launch.save(plan_path, plan)
    print(json.dumps({"plan": str(plan_path), "epochs": epochs, "jobs": len(jobs),
                      "expected_full_run_hours": epochs * timing["estimated_full_epoch_seconds"] / 3600}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", required=True)
    parser.add_argument("--preflight", required=True)
    args = parser.parse_args()
    freeze(args.drafts, args.preflight)
