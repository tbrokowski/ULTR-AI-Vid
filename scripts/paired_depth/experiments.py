"""Portable experiment configurations and validation-only choice.

Write configurations before launching; selection never reads test predictions.
Run from the repository root. Configure storage before running on any scheduler.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from ultrai.paired_depth.manifest import digest, write_json


def configurations(output, partition=0, seed=42, epochs=6, *, root, videos, manifest=None, frame_cache=None):
    import yaml
    from ultrai.paired_depth.engine import load_config, merge

    if partition not in range(5) or seed < 0 or epochs < 1:
        raise ValueError("Use a partition in 0-4, a nonnegative seed and positive epochs")
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = load_config("configs/paired_depth/base.yaml")
    common.update(partition=partition, seed=seed,
                  videos=str(Path(videos).expanduser().resolve()),
                  manifest=str(Path(manifest).expanduser().resolve()) if manifest else str(root / "artifacts/data/manifest.json"),
                  frame_cache=str(Path(frame_cache).expanduser().resolve()) if frame_cache else None)
    source_name = f"source-p{partition}-s{seed}"
    common["output"] = str(root / "artifacts/runs" / source_name)
    pending = {source_name: common}
    source_checkpoint = common["output"] + "/best.pt"
    study = yaml.safe_load(Path("configs/paired_depth/study.yaml").read_text())
    configs = {}
    for name, components in study["experiments"].items():
        # Each evaluated baseline/method continues from the same source checkpoint.
        # The source15 comparison receives the same adaptation-stage update budget.
        run_name = f"{name}-p{partition}-s{seed}"
        cfg = merge(deepcopy(common), {"output": str(root / "artifacts/runs" / run_name),
                             "source_checkpoint": source_checkpoint, "freeze_backbone": True,
                             "max_hours": 3, "epochs": epochs, "training_view": components["training_view"]})
        for key in cfg["losses"]:
            if key in components:
                cfg["losses"][key] = components[key]
        cfg["styles"] = merge(cfg["styles"], components.get("styles", {}))
        pending[run_name] = cfg
        configs[name] = cfg
    if partition == 0 and seed == 42:
        for i, weights in enumerate(study["selection"]["weight_grid"]):
            cfg = merge(configs["conditional"], {"output": str(root / f"artifacts/runs/grid{i}-p0-s42"),
                        "losses": {"scan_domain": weights["adversarial"], "patient_domain": weights["adversarial"],
                                   "feature": weights["feature"], "prediction": weights["prediction"]}})
            pending[f"grid{i}-p0-s42"] = cfg
        for name in ("dann", "conditional"):
            cfg = merge(configs[name], {"output": str(root / f"artifacts/runs/stress-{name}-p0-s42"),
                                       "prevalence_stress": {"enabled": True}})
            pending[f"stress-{name}-p0-s42"] = cfg
    existing = [name for name in pending if (output / f"{name}.json").exists()]
    if existing:
        raise FileExistsError(f"Configurations are immutable; reuse them or choose another directory: {existing}")
    for name, cfg in pending.items():
        write_json(output / f"{name}.json", cfg)
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
    c.add_argument("--root", required=True, help="Absolute study directory on the machine running training")
    c.add_argument("--videos", required=True, help="Verified video directory, readable by training jobs")
    c.add_argument("--manifest", help="Defaults to ROOT/artifacts/data/manifest.json")
    c.add_argument("--frame-cache", help="Optional cache created by cache_frames; omitted means decode videos")
    c.add_argument("--output", help="Defaults to ROOT/artifacts/configs")
    c.add_argument("--partition", type=int, default=0)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--epochs", type=int, required=True, help="Common comparison schedule; choose before viewing validation scores")
    s = sub.add_parser("select")
    s.add_argument("--run-root", required=True)
    s.add_argument("--output", required=True)
    a = p.parse_args()
    if a.action == "configure":
        configurations(a.output or Path(a.root) / "artifacts/configs", a.partition, a.seed, a.epochs,
                       root=a.root, videos=a.videos, manifest=a.manifest, frame_cache=a.frame_cache)
    else:
        print(json.dumps(select(a.run_root, a.output), indent=2))
