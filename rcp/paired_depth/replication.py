"""Finite source-to-comparison pipelines for predefined partitions and seeds.

The validation choice is immutable. Sources receive their original ten-hour
limits; dependent comparisons retain the six-epoch partition-0 schedule.
"""
import argparse
import json
import math
from pathlib import Path
import time

from . import launch, queue


def prepare(drafts, selection, seed=42, partitions=None):
    drafts, selection = Path(drafts), Path(selection)
    partitions = list(partitions if partitions is not None else range(1, 5))
    if seed not in (42, 43, 44) or not partitions or len(set(partitions)) != len(partitions) or any(p not in range(5) for p in partitions):
        raise ValueError("Use the predefined seeds and unique partition indices 0-4")
    prefix = "replication" if seed == 42 and partitions == [1, 2, 3, 4] else f"replication-s{seed}"
    selected = json.loads(selection.read_text())
    method = selected["selected"]["method"]
    p0 = json.loads((launch.ROOT / "artifacts/scheduler/adaptation-p0-plan.json").read_text())
    epochs = p0["epochs"]
    directory = launch.ROOT / "artifacts/configs" / prefix
    master = launch.ROOT / "artifacts/scheduler" / f"{prefix}-plan.json"
    if directory.exists() or master.exists():
        raise FileExistsError("Replication configurations and plan are immutable")
    if method not in ("consistency", "dann", "conditional", "full"):
        raise ValueError("Copy the selected grid configuration explicitly before replication")
    configs = {}
    for partition in partitions:
        for arm in ("source", "source15", "both_supervised", method):
            name = f"{arm}-p{partition}-s{seed}"
            cfg = json.loads((drafts / f"{name}.json").read_text())
            if cfg["partition"] != partition or cfg["seed"] != seed:
                raise ValueError("Replication partition/seed mismatch")
            if arm != "source" and cfg["epochs"] != epochs:
                raise ValueError("Replication must retain the partition-0 epoch budget")
            configs[name] = cfg
    common = next(iter(configs.values()))
    manifest = json.loads(queue.host_path(common["manifest"]).read_text())
    directory.mkdir(parents=True)
    jobs = {}
    for name, cfg in configs.items():
        config_path = directory / f"{name}.json"
        launch.save(config_path, cfg)
        job = {"name": "bd-" + name.replace("_", "-"),
               "config": str(queue.CONTAINER_ROOT / config_path.relative_to(launch.ROOT)),
               "config_sha256": queue.sha256(config_path), "output": cfg["output"],
               "epochs": cfg["epochs"], "hours": cfg["max_hours"], "pool": "a100-40g"}
        if name.startswith("source-p"):
            job["completion"] = "source_pretraining"
        else:
            allowed = set(manifest["partitions"][cfg["partition"]]["train"])
            if cfg["training_view"] == "all15":
                ids = {r["patient"] for r in manifest["records"] if r["depth"] == 15 and r["patient"] in allowed}
            else:
                ids = {p["patient"] for p in manifest["pairs"] if p["patient"] in allowed}
            job["expected_updates"] = epochs * math.ceil(len(ids) / cfg["effective_batch_size"])
        jobs[name] = job
    sources = {"source_checkpoint": None, "implementation_sha256": common["implementation_sha256"],
               "jobs": [jobs[f"source-p{p}-s{seed}"] for p in partitions]}
    plan = {"schema": 1, "frozen_unix": time.time(), "selection": str(selection),
            "selection_sha256": queue.sha256(selection), "selected_method": method,
            "retention_constraint_satisfied": selected["retention_constraint_satisfied"],
            "epochs": epochs, "seed": seed, "partitions": partitions, "source_plan": sources,
            "comparisons": {str(p): [jobs[f"{arm}-p{p}-s{seed}"] for arm in ("source15", "both_supervised", method)]
                            for p in partitions},
            "source_checkpoints": {str(p): configs[f"source15-p{p}-s{seed}"]["source_checkpoint"] for p in partitions}}
    queue.check_plan(sources)
    launch.save(master, plan)
    print(json.dumps({"plan": str(master), "selected_method": method, "source_jobs": len(partitions),
                      "comparison_jobs": 3 * len(partitions), "retention_constraint_satisfied": plan["retention_constraint_satisfied"]}))


def ready_partitions(plan, completed_sources):
    seed = plan.get("seed", 42)
    return [p for p in plan["comparisons"] if f"bd-source-p{p}-s{seed}" in completed_sources]


def run(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    status_path = path.with_name(path.stem + "-status.json")
    plan_hash = queue.sha256(path)
    expected_comparisons = sum(len(jobs) for jobs in plan["comparisons"].values())
    prefix = path.stem.removesuffix("-plan")
    if queue.sha256(plan["selection"]) != plan["selection_sha256"]:
        raise ValueError("Frozen validation choice changed")
    with launch.controller_lock(str(path), wait_seconds=0):
        if status_path.exists() and json.loads(status_path.read_text())["plan_sha256"] != plan_hash:
            raise ValueError("Replication plan changed after execution started")
        queue.check_plan(plan["source_plan"])
        comparison_plans = {}
        while True:
            try:
                source_status = queue.tick(plan["source_plan"])
                submitted = list(source_status["newly_submitted"])
                completed = []
                for partition in ready_partitions(plan, source_status["completed_jobs"]):
                    if partition not in comparison_plans:
                        cp = path.with_name(f"{prefix}-p{partition}-comparison-plan.json")
                        if cp.exists():
                            child = json.loads(cp.read_text())
                            if child["parent_plan_sha256"] != plan_hash:
                                raise ValueError("Dependent comparison plan has a different parent")
                        else:
                            checkpoint = plan["source_checkpoints"][partition]
                            child = {"parent_plan_sha256": plan_hash, "source_checkpoint": checkpoint,
                                     "source_checkpoint_sha256": queue.sha256(queue.host_path(checkpoint)),
                                     "implementation_sha256": plan["source_plan"]["implementation_sha256"],
                                     "jobs": plan["comparisons"][partition]}
                            launch.save(cp, child)
                        queue.check_plan(child)
                        comparison_plans[partition] = child
                    child_status = queue.tick(comparison_plans[partition])
                    submitted.extend(child_status["newly_submitted"])
                    completed.extend(child_status["completed_jobs"])
                status = {"status": "complete" if len(completed) == expected_comparisons else "running",
                          "plan_sha256": plan_hash, "selected_method": plan["selected_method"],
                          "seed": plan.get("seed", 42),
                          "completed_sources": source_status["completed_jobs"],
                          "completed_comparisons": completed, "newly_submitted": submitted,
                          "updated_unix": time.time()}
            except Exception as error:
                launch.save(status_path, {"status": "needs_attention", "error": str(error),
                            "plan_sha256": plan_hash, "updated_unix": time.time()})
                raise
            launch.save(status_path, status)
            if submitted or status["status"] == "complete":
                print(json.dumps(status), flush=True)
            if status["status"] == "complete":
                return
            time.sleep(30)


def run_series(paths):
    """Finish the higher-priority seed before starting the next finite plan."""
    for path in paths:
        run(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--drafts", required=True)
    prepare_parser.add_argument("--selection", required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--partitions", nargs="+", type=int)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    series_parser = commands.add_parser("series")
    series_parser.add_argument("--plans", nargs="+", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.drafts, args.selection, args.seed, args.partitions)
    elif args.action == "series":
        run_series(args.plans)
    else:
        run(args.plan)
