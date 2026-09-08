"""Exercise the actual loop, signal handling, validation, and mid-epoch resume."""
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from ultrai.paired_depth import engine
from ultrai.paired_depth.data import collate


def test_amp_overflow_skips_update_and_recovers():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    scaler.scale(parameter.square().sum()).backward()
    parameter.grad.fill_(float("inf"))
    assert not engine.optimizer_step(optimizer, scaler)
    assert parameter.item() == 1.0
    assert parameter.grad is None
    assert scaler.get_scale() == 512.0
    scaler.scale(parameter.square().sum()).backward()
    assert engine.optimizer_step(optimizer, scaler)
    assert parameter.item() < 1.0
    assert scaler.get_scale() == 512.0


def test_nonfinite_full_precision_gradient_fails_without_updating():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError, match="without mixed-precision"):
        engine.optimizer_step(optimizer, torch.amp.GradScaler("cpu", enabled=False))
    assert parameter.item() == 1.0


class LoopModel(torch.nn.Module):
    hidden_dim = 4

    def __init__(self):
        super().__init__()
        self.vision_encoder = torch.nn.Linear(3, 4)
        self.frame_selector = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.2))
        self.tb_classifier = torch.nn.Linear(4, 1)
        self.pathology_modules = torch.nn.Linear(4, 4)

    def forward(self, x):
        features = self.frame_selector(x["site_videos"].mean((2, 4, 5)))
        patient = features.mean(1)
        logits = self.tb_classifier(patient).squeeze(-1)
        return {"site_features": features, "patient_features": patient, "tb_logits": logits,
                "pathology_scores": self.pathology_modules(features), "task_logits": {"TB Label": logits}}

    def compute_losses(self, out, target, weights):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["tb_logits"], target["tb_labels"])
        return loss, {}


class LoopData:
    interrupt = False

    def __init__(self, manifest, root, partition, split, view, *args, **kwargs):
        self.ids = ["a", "b", "c", "d"]
        self.split, self.view, self.epoch = split, view, 0

    def __len__(self):
        return 4

    def __getitem__(self, i):
        if self.interrupt and self.split == "train":
            LoopData.interrupt = False
            os.kill(os.getpid(), signal.SIGUSR1)
        return {"patient": self.ids[i], "tb": i % 2,
                "views": {d: {"site_videos": torch.full((1, 3, 3, 4, 4), float(i)), "site_indices": torch.tensor([1]),
                               "pathology_labels": torch.zeros(1, 4)} for d in ((5, 15) if self.view == "paired" else (15,))}}


def test_signal_resume_and_metrics_regeneration(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "PatientBags", LoopData)
    source = {f"{g}_{k}": value for g in ("backbone", "pathology", "classifier", "integration", "patient_pipeline")
              for k, value in (("lr", 0.01), ("weight_decay", 0.0))}
    source.update(backbone_T_0=4, backbone_T_mult=2, backbone_eta_min=1e-6)
    monkeypatch.setattr(engine, "make_model", lambda cfg: (LoopModel(), source))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"patients": {p: {"tb": i % 2} for i, p in enumerate("abcd")}}))
    cfg = engine.load_config("configs/paired_depth/base.yaml")
    cfg.update(manifest=str(manifest), device="cpu", amp=False, freeze_backbone=True,
               training_view="paired", epochs=2, effective_batch_size=3, max_hours=1)
    config = tmp_path / "config.json"
    def args(stage="train", **kwargs):
        return SimpleNamespace(config=str(config), stage=stage, resume=None, checkpoint=None, split="valid", freeze_file=None, **kwargs)
    cfg["output"] = str(tmp_path / "uninterrupted")
    config.write_text(json.dumps(cfg))
    engine.run(args())
    expected = torch.load(Path(cfg["output"]) / "last.pt", weights_only=False)
    cfg["output"] = str(tmp_path / "resumed")
    config.write_text(json.dumps(cfg))
    LoopData.interrupt = True
    engine.run(args())
    interrupted = torch.load(Path(cfg["output"]) / "last.pt", weights_only=False)
    assert interrupted["progress"]["status"] == "interrupted"
    assert interrupted["sampler"]["cursor"] == 1
    resume_args = args()
    resume_args.resume = str(Path(cfg["output"]) / "last.pt")
    engine.run(resume_args)
    actual = torch.load(Path(cfg["output"]) / "last.pt", weights_only=False)
    assert all(torch.equal(v, actual["model_state_dict"][k]) for k, v in expected["model_state_dict"].items())
    assert actual["progress"]["history"] == expected["progress"]["history"]
    evaluation = args("evaluate")
    evaluation.checkpoint = str(Path(cfg["output"]) / "best.pt")
    before = json.loads((Path(cfg["output"]) / "valid_metrics.json").read_text())
    engine.run(evaluation)
    after = json.loads((Path(cfg["output"]) / "valid_metrics.json").read_text())
    assert before == after
