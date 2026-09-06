"""Run on the jumphost. Persist reservations and meter allocated, not busy, GPU time.

All launches go through this controller. It checks existing project allocations,
reserves each job's upper bound plus startup margin, and checkpoints/stops jobs
before the study cap. The final 20 GPU-hours cannot be spent on training.
"""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("job", HERE / "job.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ROOT = Path("/mnt/light/scratch/users/falke/benin-paired-depth-dann")
NAMESPACE = "runai-light-falke"


def kubectl(*args, stdin=None):
    return subprocess.check_output(["kubectl", *args, "-n", NAMESPACE], input=stdin, text=True)


def save(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def seconds(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def refresh(ledger, pods):
    now = time.time()
    for record in ledger["jobs"]:
        matching = [p for p in pods if p["metadata"].get("labels", {}).get("release") == record["name"]]
        pod_hours = record.setdefault("pod_hours", {})
        running = False
        for pod in matching:
            scheduled = next((c for c in pod.get("status", {}).get("conditions", []) if c["type"] == "PodScheduled" and c["status"] == "True"), None)
            if not scheduled:
                continue
            start = seconds(scheduled["lastTransitionTime"])
            states = [c.get("state", {}) for c in pod.get("status", {}).get("containerStatuses", [])]
            finishes = [seconds(s["terminated"]["finishedAt"]) for s in states if "terminated" in s]
            active = pod.get("status", {}).get("phase") not in ("Succeeded", "Failed")
            finish = now if active or not finishes else max(finishes)
            # Keep finished/preempted pod charges even after Kubernetes removes
            # those pods. A replacement pod must not reset the run's GPU meter.
            uid = pod["metadata"].get("uid", pod["metadata"].get("name", record["name"]))
            pod_hours[uid] = max(pod_hours.get(uid, 0), max(0, finish - start) * record["gpus"] / 3600)
            running |= active
        record["allocated_hours"] = max(record.get("allocated_hours", 0), sum(pod_hours.values()))
        if matching and all(p.get("status", {}).get("phase") in ("Succeeded", "Failed") for p in matching):
            record["status"] = "complete" if all(p["status"]["phase"] == "Succeeded" for p in matching) else "failed"
        if running:
            record["status"] = "running"
    return ledger


def launch(name, command, hours, pool, final=False, gpu=1):
    directory = ROOT / "artifacts" / "scheduler"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ledger.json"
    with (directory / "ledger.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {"cap": 200, "evaluation_reserve": 20, "jobs": []}
        pods = json.loads(kubectl("get", "pods", "-o", "json"))["items"]
        refresh(ledger, pods)
        if any(r["name"] == name for r in ledger["jobs"]):
            raise ValueError("Job names are unique; resume under a new name")
        pending = [r for r in ledger["jobs"] if r["status"] in ("submitted", "running")]
        # Include queued study work to prevent oversubmitting beyond three GPUs.
        project_allocations = sum(float(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
                                  for p in pods if p.get("status", {}).get("phase") not in ("Succeeded", "Failed")
                                  for c in p["spec"]["containers"])
        if max(sum(r["gpus"] for r in pending), project_allocations) + gpu > 3:
            raise RuntimeError("Three-GPU concurrency limit reached")
        used_or_reserved = sum(r["allocated_hours"] if r["status"] in ("complete", "failed") else
                               max(r.get("allocated_hours", 0), r["reserved_hours"]) for r in ledger["jobs"])
        reservation = gpu * (hours + 0.25)  # Include image/container startup and shutdown.
        cap = ledger["cap"] if final else ledger["cap"] - ledger["evaluation_reserve"]
        if used_or_reserved + reservation > cap:
            raise RuntimeError(f"GPU budget unavailable: {used_or_reserved:.3f} + {reservation:.3f} > {cap}")
        is_verification = "ultrai.paired_depth.verification" in command
        if gpu and not is_verification and not (ROOT / "artifacts" / "verification" / pool / "gpu_verification.json").exists():
            raise RuntimeError("Verify this GPU pool before research jobs")
        job = module.job(name, command, gpu, hours, pool)
        record = {"name": name, "gpus": gpu, "pool": pool, "hours": hours, "reserved_hours": reservation,
                  "allocated_hours": 0, "status": "submitted", "submitted_at": time.time(), "final_evaluation": final}
        ledger["jobs"].append(record)
        save(directory / f"{name}.json", job)
        save(path, ledger)  # Reserve before submitting, including uncertain outcomes.
        kubectl("create", "-f", "-", stdin=json.dumps(job))
        return record


def watch(once=False):
    directory = ROOT / "artifacts" / "scheduler"
    path = directory / "ledger.json"
    while True:
        with (directory / "ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(path.read_text())
            pods = json.loads(kubectl("get", "pods", "-o", "json"))["items"]
            refresh(ledger, pods)
            used = sum(r["allocated_hours"] for r in ledger["jobs"])
            training_used = sum(r["allocated_hours"] for r in ledger["jobs"] if not r["final_evaluation"])
            for record in ledger["jobs"]:
                if record["status"] not in ("running", "submitted"):
                    continue
                if not record["gpus"]:
                    continue  # CPU jobs have no GPU reservation; their timeout still applies.
                if record["allocated_hours"] >= record["reserved_hours"] or used >= 199.85 or (not record["final_evaluation"] and training_used >= 179.85):
                    # Deleting our workload delivers SIGTERM; the trainer saves full state.
                    kubectl("delete", "trainingworkload", record["name"], "--wait=false")
                    record["status"] = "budget_stopped"
            save(path, ledger)
        if once:
            print(json.dumps(ledger, indent=2))
            return
        time.sleep(15)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--once", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--hours", type=float, required=True)
    s.add_argument("--pool", default="a100-40g")
    s.add_argument("--gpu", type=int, default=1)
    s.add_argument("--final-evaluation", action="store_true")
    s.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.action == "watch":
        watch(a.once)
    else:
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        print(json.dumps(launch(a.name, command, a.hours, a.pool, a.final_evaluation, a.gpu)))
