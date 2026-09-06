"""Predefined experiment queue and validation-only choice. Run from repository root.

Write configurations before launching; selection never reads test predictions.
The job controller enforces the global budget, including all smoke runs.
"""
import argparse
import json
from pathlib import Path
import time

import yaml

from ultrai.paired_depth.engine import load_config, merge
from ultrai.paired_depth.manifest import digest, write_json

CONTAINER_ROOT = "/scratch/users/falke/benin-paired-depth-dann"


def configurations(output, partition=0, seed=42, epochs=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common = load_config("configs/paired_depth/base.yaml")
    common.update(partition=partition, seed=seed)
    source_name = f"source-p{partition}-s{seed}"
    common["output"] = f"{CONTAINER_ROOT}/artifacts/runs/{source_name}"
    write_json(output / f"{source_name}.json", common)
    source_checkpoint = common["output"] + "/best.pt"
    study = yaml.safe_load(Path("configs/paired_depth/study.yaml").read_text())
    configs = {}
    for name, components in study["experiments"].items():
        # Each evaluated baseline/method continues from the same source checkpoint.
        # The source15 comparison receives the same adaptation-stage update budget.
        run_name = f"{name}-p{partition}-s{seed}"
        cfg = merge(common, {"output": f"{CONTAINER_ROOT}/artifacts/runs/{run_name}",
                             "source_checkpoint": source_checkpoint, "freeze_backbone": True,
                             "max_hours": 3, "epochs": epochs, "training_view": components["training_view"]})
        for key in cfg["losses"]:
            if key in components:
                cfg["losses"][key] = components[key]
        cfg["styles"] = merge(cfg["styles"], components.get("styles", {}))
        write_json(output / f"{run_name}.json", cfg)
        configs[name] = cfg
    if partition == 0 and seed == 42:
        for i, weights in enumerate(study["selection"]["weight_grid"]):
            cfg = merge(configs["conditional"], {"output": f"{CONTAINER_ROOT}/artifacts/runs/grid{i}-p0-s42",
                        "losses": {"scan_domain": weights["adversarial"], "patient_domain": weights["adversarial"],
                                   "feature": weights["feature"], "prediction": weights["prediction"]}})
            write_json(output / f"grid{i}-p0-s42.json", cfg)
        for name in ("dann", "conditional"):
            cfg = merge(configs[name], {"output": f"{CONTAINER_ROOT}/artifacts/runs/stress-{name}-p0-s42",
                                       "prevalence_stress": {"enabled": True}})
            write_json(output / f"stress-{name}-p0-s42.json", cfg)
    return configs


def select(run_root, output):
    run_root = Path(run_root)
    baseline = json.loads((run_root / "source15-p0-s42" / "valid_metrics.json").read_text())
    baseline_progress = json.loads((run_root / "source15-p0-s42" / "progress.json").read_text())
    if baseline_progress["status"] != "complete":
        raise ValueError("The matching 15 cm baseline did not complete its predefined budget")
    baseline_epochs = len(baseline_progress["history"])
    candidates = []
    for complexity, name in enumerate(("consistency", "dann", "conditional", "full", "grid0", "grid1", "grid2")):
        root = run_root / f"{name}-p0-s42"
        if not (root / "best.pt").exists() or not (root / "valid_metrics.json").exists():
            continue
        metrics = json.loads((root / "valid_metrics.json").read_text())
        history = json.loads((root / "history.json").read_text())
        progress = json.loads((root / "progress.json").read_text())
        if progress["status"] != "complete" or len(history) != baseline_epochs:
            continue  # Under-budget runs are documented as unfinished, never selected.
        candidates.append({"method": name, "validation": metrics, "complexity": complexity,
                           "completed_epochs": len(history), "checkpoint_sha256": digest(root / "best.pt"),
                           "stable15": metrics["all15"]["auroc"] - baseline["all15"]["auroc"] >= -0.01})
    if not candidates:
        raise ValueError("No completed validation candidates")
    feasible = [c for c in candidates if c["stable15"]]
    selected = max(feasible or candidates, key=lambda c: (c["validation"]["matched5"]["auroc"], -c["complexity"]))
    result = {"selected": selected, "candidates": candidates, "retention_constraint_satisfied": bool(feasible),
              "selection_data": "partition 0 validation only", "test_data_read": False}
    if Path(output).exists():
        raise FileExistsError("Selection is immutable; do not reselect after test evaluation")
    write_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("configure")
    c.add_argument("--output", required=True)
    c.add_argument("--partition", type=int, default=0)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--epochs", type=int, default=20, help="Freeze after timing smoke, before viewing validation scores")
    s = sub.add_parser("select")
    s.add_argument("--run-root", required=True)
    s.add_argument("--output", required=True)
    a = p.parse_args()
    if a.action == "configure":
        configurations(a.output, a.partition, a.seed, a.epochs)
    else:
        print(json.dumps(select(a.run_root, a.output), indent=2))
