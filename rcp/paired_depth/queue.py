"""Run an immutable, finite training queue through the existing GPU budget controller.

Stops submitting on any failed or incomplete comparison. Existing jobs retain
their own deadlines. This process never reads validation scores or test data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from . import launch

CONTAINER_ROOT = Path("/scratch/users/falke/benin-paired-depth-dann")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def host_path(path):
    return launch.ROOT / Path(path).relative_to(CONTAINER_ROOT)


def check_completion(job):
    path = host_path(job["output"]) / "progress.json"
    if not path.exists():
        raise RuntimeError(f"{job['name']} exited without training progress")
    progress = json.loads(path.read_text())
    if job.get("completion") == "source_pretraining":
        if progress["status"] not in ("complete", "budget_exhausted") or not progress["history"]:
            raise RuntimeError(f"{job['name']} has no completed source training epoch")
        if (progress["status"] == "budget_exhausted"
                and progress["elapsed_seconds"] < job["hours"] * 3600 - 185):
            raise RuntimeError(f"{job['name']} stopped before its source time budget")
        for name in ("best.pt", "last.pt"):
            if not (path.parent / name).is_file():
                raise RuntimeError(f"{job['name']} has no {name}")
        return
    if (progress["status"] != "complete" or len(progress["history"]) != job["epochs"]
            or progress["updates"] != job["expected_updates"]):
        raise RuntimeError(f"{job['name']} did not finish the predefined epoch/update budget")


def check_plan(plan):
    jobs = plan["jobs"]
    if not jobs or len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("Queue job names must be nonempty and unique")
    if (plan["source_checkpoint"] is not None
            and sha256(host_path(plan["source_checkpoint"])) != plan["source_checkpoint_sha256"]):
        raise ValueError("Source checkpoint changed after queue freeze")
    for job in jobs:
        path = host_path(job["config"])
        if sha256(path) != job["config_sha256"]:
            raise ValueError(f"Frozen configuration changed: {job['name']}")
        config = json.loads(path.read_text())
        if plan["source_checkpoint"] is None and (
                job.get("completion") != "source_pretraining" or config["freeze_backbone"]
                or config["training_view"] != "all15"):
            raise ValueError("Only source pretraining can omit its source checkpoint")
        if (config["output"] != job["output"] or config["epochs"] != job["epochs"]
                or config["max_hours"] != job["hours"]
                or config["source_checkpoint"] != plan["source_checkpoint"]):
            raise ValueError(f"Queue/configuration mismatch: {job['name']}")
    for name, expected in plan["implementation_sha256"].items():
        if sha256(launch.ROOT / "code" / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")


def tick(plan):
    directory = launch.ROOT / "artifacts/scheduler"
    with launch.controller_lock("ledger"):
        ledger = json.loads((directory / "ledger.json").read_text())
        pods = json.loads(launch.kubectl("get", "pods", "-o", "json"))["items"]
        launch.refresh(ledger, pods)
        launch.save(directory / "ledger.json", ledger)
    records = {r["name"]: r for r in ledger["jobs"]}
    completed = []
    for job in plan["jobs"]:
        record = records.get(job["name"])
        if record is None:
            continue
        if record["status"] in ("failed", "budget_stopped"):
            raise RuntimeError(f"Queue stopped after {job['name']}: {record['status']}")
        if record["status"] == "complete":
            check_completion(job)
            completed.append(job["name"])
    remaining = [j for j in plan["jobs"] if j["name"] not in records]
    submitted = []
    if remaining:
        for name, expected in plan.get("implementation_sha256", {}).items():
            if sha256(launch.ROOT / "code" / name) != expected:
                raise ValueError(f"Frozen implementation changed: {name}")
    for job in remaining:
        if sha256(host_path(job["config"])) != job["config_sha256"]:
            raise ValueError(f"Frozen configuration changed: {job['name']}")
        command = [str(CONTAINER_ROOT / "env/runtime/bin/python"), "-m", "ultrai.paired_depth",
                   "train", "--config", job["config"]]
        try:
            launch.launch(job["name"], command, job["hours"], job["pool"])
        except RuntimeError as error:
            if "Three-GPU concurrency limit reached" in str(error):
                break
            raise
        submitted.append(job["name"])
    return {"status": "complete" if len(completed) == len(plan["jobs"]) else "running",
            "completed_jobs": completed, "newly_submitted": submitted,
            "unsubmitted_jobs": [j["name"] for j in remaining if j["name"] not in submitted],
            "updated_unix": time.time()}


def run(path, once=False):
    path = Path(path)
    plan = json.loads(path.read_text())
    plan_hash = sha256(path)
    status_path = path.with_name(path.stem + "-status.json")
    with launch.controller_lock(str(path), wait_seconds=0):
        if status_path.exists() and json.loads(status_path.read_text())["plan_sha256"] != plan_hash:
            raise ValueError("Queue plan changed after execution started")
        check_plan(plan)
        while True:
            try:
                status = tick(plan)
            except Exception as error:
                launch.save(status_path, {"status": "needs_attention", "error": str(error),
                            "plan_sha256": plan_hash, "updated_unix": time.time()})
                raise
            status["plan_sha256"] = plan_hash
            launch.save(status_path, status)
            if status["newly_submitted"] or status["status"] == "complete":
                print(json.dumps(status), flush=True)
            if once or status["status"] == "complete":
                return
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run(args.plan, args.once)
