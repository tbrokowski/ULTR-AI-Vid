"""Regenerate source validation and time training-only full-method bags.

This software check never selects a model, saves trained weights, or reads test data.
"""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch

from ultrai.paired_depth.data import PatientBags, collate
from ultrai.paired_depth.engine import (evaluate, gpu_check, load_config, losses,
                                       make_model, merge, seed_all, to_device)
from ultrai.paired_depth.manifest import digest, write_json
from ultrai.paired_depth.objectives import DomainHeads, class_weights


def run(source_run, output, samples=48):
    os.umask(0o077)
    source_run, output = Path(source_run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_cfg = load_config(source_run / "resolved_config.json")
    manifest = json.loads(Path(source_cfg["manifest"]).read_text())
    checkpoint = source_run / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"] != source_cfg or state["manifest_sha256"] != digest(source_cfg["manifest"]):
        raise ValueError("Source checkpoint configuration or manifest changed")
    hardware = gpu_check()
    seed_all(source_cfg["seed"])
    cfg = merge(source_cfg, {"freeze_backbone": True, "training_view": "paired",
                "losses": {"scan_domain": 0.01, "patient_domain": 0.01,
                           "feature": 0.1, "prediction": 0.1,
                           "conditioning": True, "class_balance": True},
                "styles": {"gain": True, "speckle": True, "resampling": True, "fourier": True}})
    model, _ = make_model(cfg)
    model.load_state_dict(state["model_state_dict"], strict=True)
    del state
    started = time.monotonic()
    metrics, predictions = evaluate(model, manifest, cfg, split="valid")
    validation_seconds = time.monotonic() - started
    expected_metrics = json.loads((source_run / "valid_metrics.json").read_text())
    expected_predictions = json.loads((source_run / "valid_predictions.json").read_text())
    if metrics != expected_metrics or predictions != expected_predictions:
        raise AssertionError("Source validation did not regenerate exactly")
    print(json.dumps({"validation_regenerated_exactly": True,
                      "validation_seconds": validation_seconds}), flush=True)
    ds = PatientBags(manifest, cfg["videos"], cfg["partition"], "train", "paired",
                     cfg["seed"], cfg["frames"], cfg["size"], cfg["styles"],
                     frame_cache=cfg.get("frame_cache"))
    labels = [manifest["patients"][p]["tb"] for p in ds.ids]
    weights = class_weights([labels, labels]).to(cfg["device"])
    heads = DomainHeads(model.hidden_dim, True).to(cfg["device"])
    model.train()
    model.vision_encoder.eval()
    heads.train()
    indices = random.Random(cfg["seed"]).sample(range(len(ds)), min(samples, len(ds)))
    durations = []
    gradient_l1 = 0.0
    for n, i in enumerate(indices):
        model.zero_grad(set_to_none=True)
        heads.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        batch = to_device(collate([ds[i]]), cfg["device"])
        with torch.autocast(device_type="cuda", enabled=cfg["amp"]):
            loss, _ = losses(model, heads, batch, cfg, 1.0, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite timing-check loss")
        (loss * cfg["amp_initial_scale"] / cfg["effective_batch_size"]).backward()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - started)
        if n == len(indices) - 1:
            gradient = model.frame_selector.output_projection[0].weight.grad
            if gradient is None or not gradient.isfinite().all() or not gradient.abs().sum() > 0:
                raise AssertionError("Missing or invalid downstream representation gradient")
            gradient_l1 = gradient.abs().sum().item()
        if any(p.grad is not None for p in model.vision_encoder.parameters()):
            raise AssertionError("Frozen CLIP received gradients")
        del batch, loss
        if (n + 1) % 8 == 0:
            print(json.dumps({"timed_training_patients": n + 1,
                              "mean_patient_seconds": statistics.mean(durations)}), flush=True)
    epoch_seconds = statistics.mean(durations) * len(ds) + validation_seconds
    result = {"source_checkpoint_sha256": digest(checkpoint), "hardware": hardware,
              "validation_regenerated_exactly": True, "test_data_read": False,
              "source_validation": metrics, "sampled_training_patients": len(indices),
              "eligible_training_patients": len(ds), "timing_sample_seed": cfg["seed"],
              "mean_patient_seconds": statistics.mean(durations),
              "stdev_patient_seconds": statistics.stdev(durations) if len(durations) > 1 else 0,
              "validation_seconds": validation_seconds, "estimated_full_epoch_seconds": epoch_seconds,
              "estimated_20_epoch_hours": epoch_seconds * 20 / 3600,
              "downstream_gradient_l1": gradient_l1, "frozen_clip_gradients": False,
              "optimizer_updates": 0, "purpose": "timing and checkpoint verification only"}
    write_json(output / "preflight.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    run(args.source_run, args.output, args.samples)
