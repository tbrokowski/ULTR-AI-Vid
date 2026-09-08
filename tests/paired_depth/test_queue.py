"""The finite queue must stop on incomplete training and never duplicate a job."""
import json

import pytest

from rcp.paired_depth import queue


def setup_queue(tmp_path, monkeypatch, records, progress=None):
    monkeypatch.setattr(queue.launch, "ROOT", tmp_path)
    monkeypatch.setattr(queue.launch, "CONTROL_DIRECTORY", tmp_path / "local-control")
    directory = tmp_path / "artifacts/scheduler"
    directory.mkdir(parents=True)
    (directory / "ledger.json").write_text(json.dumps({"jobs": records}))
    monkeypatch.setattr(queue.launch, "kubectl", lambda *args: '{"items": []}')
    monkeypatch.setattr(queue.launch, "refresh", lambda ledger, pods: ledger)
    config = tmp_path / "config.json"
    config.write_text("{}")
    job = {"name": "bd-comparison", "config": str(queue.CONTAINER_ROOT / "config.json"),
           "config_sha256": queue.sha256(config), "output": str(queue.CONTAINER_ROOT / "run"),
           "epochs": 20, "expected_updates": 40, "hours": 3, "pool": "a100-40g"}
    if progress is not None:
        (tmp_path / "run").mkdir()
        (tmp_path / "run/progress.json").write_text(json.dumps(progress))
    calls = []
    monkeypatch.setattr(queue.launch, "launch", lambda *args: calls.append(args))
    return {"jobs": [job]}, calls


@pytest.mark.parametrize("status,epochs,updates", [
    ("budget_exhausted", 19, 38), ("complete", 19, 38), ("complete", 20, 39)])
def test_successful_pod_with_incomplete_training_stops_queue(tmp_path, monkeypatch, status, epochs, updates):
    plan, calls = setup_queue(tmp_path, monkeypatch,
                             [{"name": "bd-comparison", "status": "complete"}],
                             {"status": status, "history": [{}] * epochs, "updates": updates})
    with pytest.raises(RuntimeError, match="epoch/update budget"):
        queue.tick(plan)
    assert calls == []


def test_restart_does_not_resubmit_recorded_job(tmp_path, monkeypatch):
    plan, calls = setup_queue(tmp_path, monkeypatch,
                             [{"name": "bd-comparison", "status": "running"}])
    assert queue.tick(plan)["status"] == "running"
    assert calls == []


def test_changed_config_is_rejected_before_submission(tmp_path, monkeypatch):
    plan, calls = setup_queue(tmp_path, monkeypatch, [])
    (tmp_path / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Frozen configuration changed"):
        queue.tick(plan)
    assert calls == []


def test_full_gpu_allocation_waits_without_failing_queue(tmp_path, monkeypatch):
    plan, _ = setup_queue(tmp_path, monkeypatch, [])
    def full(*args):
        raise RuntimeError("Three-GPU concurrency limit reached")
    monkeypatch.setattr(queue.launch, "launch", full)
    result = queue.tick(plan)
    assert result["status"] == "running"
    assert result["newly_submitted"] == []
    assert result["unsubmitted_jobs"] == ["bd-comparison"]


def test_failed_pod_stops_new_submissions(tmp_path, monkeypatch):
    plan, calls = setup_queue(tmp_path, monkeypatch,
                             [{"name": "bd-comparison", "status": "failed"}])
    with pytest.raises(RuntimeError, match="Queue stopped"):
        queue.tick(plan)
    assert calls == []


def test_implementation_drift_stops_submission(tmp_path, monkeypatch):
    plan, calls = setup_queue(tmp_path, monkeypatch, [])
    (tmp_path / "code").mkdir()
    code = tmp_path / "code/trainer.py"
    code.write_text("original")
    plan["implementation_sha256"] = {"trainer.py": queue.sha256(code)}
    code.write_text("changed")
    with pytest.raises(ValueError, match="Frozen implementation changed"):
        queue.tick(plan)
    assert calls == []


def test_timing_freeze_is_immutable_and_uses_eligible_training_bags(tmp_path, monkeypatch):
    from rcp.paired_depth import freeze_queue
    monkeypatch.setattr(queue.launch, "ROOT", tmp_path)
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (tmp_path / "artifacts/scheduler").mkdir(parents=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest = {"partitions": [{"train": ["a", "b", "c"]}],
                "records": [{"patient": p, "depth": 15} for p in ["a", "b", "c", "held-out"]],
                "pairs": [{"patient": p} for p in ["a", "held-out"]]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for arm in freeze_queue.ARMS:
        cfg = {"manifest": str(queue.CONTAINER_ROOT / "manifest.json"),
               "source_checkpoint": str(queue.CONTAINER_ROOT / "best.pt"),
               "implementation_sha256": {}, "effective_batch_size": 2,
               "training_view": "all15" if arm == "source15" else "paired",
               "output": str(queue.CONTAINER_ROOT / "runs" / arm)}
        (drafts / f"{arm}-p0-s42.json").write_text(json.dumps(cfg))
    evidence = tmp_path / "preflight.json"
    evidence.write_text(json.dumps({"validation_regenerated_exactly": True,
                        "test_data_read": False, "estimated_full_epoch_seconds": 1200,
                        "source_checkpoint_sha256": queue.sha256(checkpoint)}))
    freeze_queue.freeze(drafts, evidence)
    plan = json.loads((tmp_path / "artifacts/scheduler/adaptation-p0-plan.json").read_text())
    assert plan["epochs"] == 6
    assert len(plan["jobs"]) == 11
    assert plan["jobs"][0]["expected_updates"] == 12
    assert plan["jobs"][1]["expected_updates"] == 6
    queue.check_plan(plan)
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_queue.freeze(drafts, evidence)


def test_source_deadline_is_allowed_only_with_saved_checkpoints(tmp_path, monkeypatch):
    plan, _ = setup_queue(tmp_path, monkeypatch,
                         [{"name": "bd-comparison", "status": "complete"}],
                         {"status": "budget_exhausted", "history": [{}] * 80,
                          "updates": 160, "elapsed_seconds": 35820})
    job = plan["jobs"][0]
    job.update(completion="source_pretraining", hours=10, epochs=200)
    with pytest.raises(RuntimeError, match="no best.pt"):
        queue.tick(plan)
    for name in ("best.pt", "last.pt"):
        (tmp_path / "run" / name).write_bytes(b"synthetic checkpoint")
    assert queue.tick(plan)["status"] == "complete"


def test_premature_source_deadline_is_rejected(tmp_path, monkeypatch):
    plan, _ = setup_queue(tmp_path, monkeypatch,
                         [{"name": "bd-comparison", "status": "complete"}],
                         {"status": "budget_exhausted", "history": [{}],
                          "updates": 2, "elapsed_seconds": 600})
    plan["jobs"][0].update(completion="source_pretraining", hours=10)
    with pytest.raises(RuntimeError, match="before its source time budget"):
        queue.tick(plan)


def test_replication_only_releases_partitions_with_completed_sources():
    from rcp.paired_depth.replication import ready_partitions
    plan = {"comparisons": {str(p): [] for p in range(1, 5)}}
    assert ready_partitions(plan, []) == []
    assert ready_partitions(plan, ["bd-source-p2-s42"]) == ["2"]


def test_replication_never_uses_another_seeds_source():
    from rcp.paired_depth.replication import ready_partitions
    plan = {"seed": 43, "comparisons": {"0": [], "4": []}}
    assert ready_partitions(plan, ["bd-source-p0-s42", "bd-source-p4-s44"]) == []
    assert ready_partitions(plan, ["bd-source-p4-s43"]) == ["4"]


def test_seed_series_preserves_priority_and_stops_after_failure(monkeypatch):
    from rcp.paired_depth import replication
    calls = []
    monkeypatch.setattr(replication, "run", calls.append)
    replication.run_series(["seed43", "seed44"])
    assert calls == ["seed43", "seed44"]
    calls.clear()
    def fail(path):
        calls.append(path)
        raise RuntimeError("budget exhausted")
    monkeypatch.setattr(replication, "run", fail)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        replication.run_series(["seed43", "seed44"])
    assert calls == ["seed43"]


def test_additional_seed_plan_keeps_source_dependencies_and_existing_plan(tmp_path, monkeypatch):
    from rcp.paired_depth import replication
    monkeypatch.setattr(queue.launch, "ROOT", tmp_path)
    scheduler = tmp_path / "artifacts/scheduler"
    scheduler.mkdir(parents=True)
    (scheduler / "adaptation-p0-plan.json").write_text('{"epochs": 6}')
    original = scheduler / "replication-plan.json"
    original.write_text("previous immutable seed-42 plan")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": {"method": "consistency"},
                                    "retention_constraint_satisfied": False}))
    manifest = {"partitions": [{"train": ["a", "b"]}] * 5,
                "records": [{"patient": p, "depth": 15} for p in ("a", "b", "held-out")],
                "pairs": [{"patient": p} for p in ("a", "held-out")]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    for partition in (0, 4):
        for arm in ("source", "source15", "both_supervised", "consistency"):
            source = arm == "source"
            cfg = {"partition": partition, "seed": 43, "epochs": 200 if source else 6,
                   "max_hours": 10 if source else 3, "freeze_backbone": not source,
                   "training_view": "all15" if arm in ("source", "source15") else "paired",
                   "effective_batch_size": 1, "implementation_sha256": {},
                   "manifest": str(queue.CONTAINER_ROOT / "manifest.json"),
                   "source_checkpoint": None if source else str(queue.CONTAINER_ROOT / f"runs/source-p{partition}-s43/best.pt"),
                   "output": str(queue.CONTAINER_ROOT / f"runs/{arm}-p{partition}-s43")}
            (drafts / f"{arm}-p{partition}-s43.json").write_text(json.dumps(cfg))
    replication.prepare(drafts, selection, seed=43, partitions=[0, 4])
    plan = json.loads((scheduler / "replication-s43-plan.json").read_text())
    assert original.read_text() == "previous immutable seed-42 plan"
    assert plan["seed"] == 43 and plan["partitions"] == [0, 4]
    assert len(plan["source_plan"]["jobs"]) == 2
    assert plan["comparisons"]["0"][0]["expected_updates"] == 12
    assert plan["comparisons"]["0"][1]["expected_updates"] == 6
    assert plan["source_checkpoints"]["4"].endswith("source-p4-s43/best.pt")
    assert plan["retention_constraint_satisfied"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        replication.prepare(drafts, selection, seed=43, partitions=[0, 4])


def test_controller_lock_uses_local_disk_and_excludes_duplicate_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.launch, "ROOT", tmp_path / "network-scratch")
    local = tmp_path / "local-control"
    monkeypatch.setattr(queue.launch, "CONTROL_DIRECTORY", local)
    with queue.launch.controller_lock("ledger", wait_seconds=0):
        assert list(local.rglob("*.lock"))
        assert not (tmp_path / "network-scratch").exists()
        with pytest.raises(TimeoutError, match="lock busy"):
            with queue.launch.controller_lock("ledger", wait_seconds=0):
                pytest.fail("Duplicate lock owner")
    with queue.launch.controller_lock("ledger", wait_seconds=0):
        pass


def test_controller_lock_released_after_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.launch, "CONTROL_DIRECTORY", tmp_path)
    with pytest.raises(RuntimeError, match="simulated"):
        with queue.launch.controller_lock("ledger", wait_seconds=0):
            raise RuntimeError("simulated controller failure")
    with queue.launch.controller_lock("ledger", wait_seconds=0):
        pass


def test_stale_meter_recovers_finished_pod_time_without_charging_idle_hours():
    record = {"name": "bd-example", "gpus": 1, "allocated_hours": 0.5, "status": "running"}
    pod = {"metadata": {"labels": {"release": "bd-example"}, "uid": "pod-one"},
           "status": {"phase": "Succeeded", "conditions": [{"type": "PodScheduled", "status": "True",
                      "lastTransitionTime": "2026-09-07T12:00:00Z"}],
                      "containerStatuses": [{"state": {"terminated": {"finishedAt": "2026-09-07T13:30:00Z"}}}]}}
    queue.launch.refresh({"jobs": [record]}, [pod])
    assert record["status"] == "complete"
    assert record["allocated_hours"] == 1.5
    queue.launch.refresh({"jobs": [record]}, [pod])
    assert record["allocated_hours"] == 1.5
