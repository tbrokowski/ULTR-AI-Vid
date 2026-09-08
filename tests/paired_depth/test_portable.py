"""Portable configuration must preserve matched arms and exclude study outputs."""
import json
import pytest

from scripts.paired_depth.experiments import configurations
from ultrai.paired_depth.reporting import aggregate, markdown


def test_portable_configs_keep_supervised_and_adversarial_arms_independent(tmp_path):
    root = tmp_path / "study with spaces"
    output = root / "artifacts/configs"
    configurations(output, root=root, videos=tmp_path / "videos", epochs=6)
    configs = {p.stem: json.loads(p.read_text()) for p in output.glob("*.json")}
    assert len(configs) == 12
    source = configs["source-p0-s42"]
    assert source["source_checkpoint"] is None and not source["freeze_backbone"]
    assert source["epochs"] == 200 and source["frame_cache"] is None
    assert source["manifest"] == str(root / "artifacts/data/manifest.json")
    assert source["losses"]["scan_domain"] == source["losses"]["feature"] == 0
    for name, cfg in configs.items():
        assert cfg["output"].startswith(str(root / "artifacts/runs"))
        assert cfg["videos"] == str(tmp_path / "videos")
        assert "/scratch/users/" not in json.dumps(cfg)
        if name != "source-p0-s42":
            assert cfg["freeze_backbone"] and cfg["epochs"] == 6
            assert cfg["source_checkpoint"] == source["output"] + "/best.pt"
    baseline = configs["both_supervised-p0-s42"]
    assert all(baseline["losses"][k] == 0 for k in ("scan_domain", "patient_domain", "feature", "prediction"))
    assert not any(baseline["styles"][k] for k in ("gain", "speckle", "resampling", "fourier"))
    assert configs["consistency-p0-s42"]["losses"]["feature"] == 0.1
    assert not configs["dann-p0-s42"]["losses"]["conditioning"]
    assert configs["conditional-p0-s42"]["losses"]["conditioning"]
    assert configs["full-p0-s42"]["styles"]["fourier"]
    assert configs["grid2-p0-s42"]["losses"]["feature"] == 0.5
    assert not configs["stress-dann-p0-s42"]["losses"]["conditioning"]
    assert not configs["stress-dann-p0-s42"]["losses"]["class_balance"]
    assert configs["stress-conditional-p0-s42"]["losses"]["conditioning"]


def test_portable_configs_refuse_overwrite_and_keep_seed_dependencies(tmp_path):
    output = tmp_path / "configs"
    kwargs = dict(root=tmp_path, videos=tmp_path / "videos", frame_cache=tmp_path / "frames")
    configurations(output, partition=4, seed=43, **kwargs)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="immutable"):
        configurations(output, partition=4, seed=43, epochs=12, **kwargs)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    cfg = json.loads((output / "consistency-p4-s43.json").read_text())
    assert cfg["source_checkpoint"].endswith("source-p4-s43/best.pt")
    assert cfg["frame_cache"] == str(tmp_path / "frames")
    assert not any("grid" in name for name in before)


def test_rcp_wrapper_retains_existing_cluster_paths(tmp_path):
    from rcp.paired_depth.experiments import configurations as cluster_configs, CONTAINER_ROOT
    cluster_configs(tmp_path, partition=2, seed=44, epochs=6)
    cfg = json.loads((tmp_path / "source-p2-s44.json").read_text())
    assert cfg["manifest"] == CONTAINER_ROOT + "/artifacts/data/manifest.json"
    assert cfg["frame_cache"] == CONTAINER_ROOT + "/artifacts/frames32"
    assert cfg["videos"] == "/benin/datasets/ULTR-AI/LusBeninVideos"


def test_report_does_not_invent_scores_or_verification_claims():
    report = markdown({"dataset": "synthetic", "revision": "fixture", "eligible_pairs": 7, "paired_patients": 3})
    assert "Eligible acquisition pairs: 7" in report
    assert "No model evaluation was supplied" in report
    assert "Revision verified: False" in report
    for text in ("Historical report", "forecast", "email", "GPU checks", "passed"):
        assert text not in report


def test_report_aggregation_refuses_mismatched_patient_cohorts(tmp_path):
    directories = [tmp_path / "baseline", tmp_path / "candidate"]
    for index, directory in enumerate(directories):
        directory.mkdir()
        rows = [{"patient": p, "view": "matched5", "label": y, "probability": 0.2 + 0.6*y,
                 "partition": 0, "seed": 42} for p, y in [("a", 0), ("b" if index == 0 else "c", 1)]]
        (directory / "test_predictions.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="same patient cohort"):
        aggregate([directories], samples=10)
