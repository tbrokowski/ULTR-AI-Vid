import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest


def launcher(monkeypatch, tmp_path, records=None):
    spec = importlib.util.spec_from_file_location("study_launcher", "rcp/paired_depth/launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "artifacts" / "scheduler"
    directory.mkdir(parents=True)
    (directory / "ledger.json").write_text(json.dumps({"cap": 200, "evaluation_reserve": 20, "jobs": records or []}))
    verified = tmp_path / "artifacts" / "verification" / "a100-40g"
    verified.mkdir(parents=True)
    (verified / "gpu_verification.json").write_text("{}")
    calls = []
    def kubectl(*args, stdin=None):
        calls.append(args)
        return '{"items": []}' if args[:2] == ("get", "pods") else "ok"
    monkeypatch.setattr(module, "kubectl", kubectl)
    return module, calls


def test_training_cannot_spend_evaluation_reserve(monkeypatch, tmp_path):
    record = {"name": "old", "status": "complete", "allocated_hours": 179, "gpus": 1}
    module, calls = launcher(monkeypatch, tmp_path, [record])
    with pytest.raises(RuntimeError, match="budget"):
        module.launch("training", ["python", "train.py"], 1, "a100-40g")
    assert not any(c[0] == "create" for c in calls)
    module.launch("evaluation", ["python", "evaluate.py"], 1, "a100-40g", final=True)
    assert any(c[0] == "create" for c in calls)


def test_three_gpu_limit_counts_queued_reservations(monkeypatch, tmp_path):
    records = [{"name": f"old-{i}", "status": "submitted", "allocated_hours": 0,
                "reserved_hours": 2, "gpus": 1} for i in range(3)]
    module, calls = launcher(monkeypatch, tmp_path, records)
    with pytest.raises(RuntimeError, match="concurrency"):
        module.launch("fourth", ["python", "train.py"], 1, "a100-40g")


def test_cpu_jobs_are_not_killed_for_zero_gpu_reservation(monkeypatch, tmp_path):
    record = {"name": "cpu", "status": "submitted", "allocated_hours": 0,
              "reserved_hours": 0, "gpus": 0, "final_evaluation": False}
    module, calls = launcher(monkeypatch, tmp_path, [record])
    module.watch(once=True)
    assert not any(c[0] == "delete" for c in calls)


def test_allocated_time_includes_image_startup(monkeypatch, tmp_path):
    module, _ = launcher(monkeypatch, tmp_path)
    timestamp = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat().replace("+00:00", "Z")
    ledger = {"jobs": [{"name": "ours", "gpus": 1, "status": "submitted"}]}
    pod = {"metadata": {"labels": {"release": "ours"}}, "status": {"phase": "Succeeded",
           "conditions": [{"type": "PodScheduled", "status": "True", "lastTransitionTime": timestamp(1000)}],
           "containerStatuses": [{"state": {"terminated": {"startedAt": timestamp(1300), "finishedAt": timestamp(1600)}}}]}}
    module.refresh(ledger, [pod])
    assert ledger["jobs"][0]["allocated_hours"] == pytest.approx(600 / 3600)
    assert ledger["jobs"][0]["status"] == "complete"
    pod["metadata"]["uid"] = "replacement-pod"
    module.refresh(ledger, [pod])
    assert ledger["jobs"][0]["allocated_hours"] == pytest.approx(1200 / 3600)
