"""GPU architecture, overfit and exact interrupted/resumed checks on synthetic bags.

These are software checks, never scientific performance results.
"""
import copy
import json
import time
from pathlib import Path

import torch

from .data import StatefulOrder
from .engine import (gpu_check, load_checkpoint, load_config, losses, make_model,
                     optimizers, save_checkpoint, seed_all)
from .manifest import write_json
from .objectives import DomainHeads


def verify(output):
    started = time.monotonic()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    hardware = gpu_check()
    seed_all(42)
    cfg = load_config("configs/paired_depth/base.yaml")
    cfg.update(freeze_backbone=True, gradient_checkpointing=False)
    model, source = make_model(cfg)
    heads = DomainHeads(512, True).cuda()
    model.eval()
    # Two deliberately distinguishable synthetic patients, two scans per depth.
    labels = torch.tensor([0., 1.], device="cuda")
    clips = torch.randn(2, 2, 4, 3, 224, 224, device="cuda") * 0.1
    clips[1] += 1.0
    views = {d: {"site_videos": clips + (0.03 if d == 5 else 0),
                 "site_indices": torch.tensor([[1, 2], [1, 2]], device="cuda"),
                 "site_masks": torch.ones(2, 2, dtype=torch.bool, device="cuda"),
                 "pathology_labels": torch.tensor([[[0., 0, -1, 0]] * 2, [[1., 1, -1, 1]] * 2], device="cuda")}
             for d in (5, 15)}
    batch = {"patients": ["synthetic-negative", "synthetic-positive"], "tb_labels": labels,
             "views": views, "pairs": torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], device="cuda")}
    cfg["losses"].update(scan_domain=0.01, patient_domain=0.01, feature=0.1, prediction=0.1, conditioning=True)
    total, _ = losses(model, heads, batch, cfg, 1, None)
    total.backward()
    gradient = model.frame_selector.output_projection[0].weight.grad.abs().sum().item()
    assert gradient > 0
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    # Tiny-data overfit of the real downstream representation, with cached frozen
    # CLIP outputs to keep this verification small and deterministic.
    with torch.no_grad():
        feature = model(views[15])["patient_features"].detach()
    optimizer = torch.optim.AdamW(model.tb_classifier.parameters(), lr=0.003)
    initial = torch.nn.functional.binary_cross_entropy_with_logits(model.tb_classifier(feature).squeeze(-1), labels).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.tb_classifier(feature).squeeze(-1), labels)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.binary_cross_entropy_with_logits(model.tb_classifier(feature).squeeze(-1), labels).item()
    assert final < initial * 0.1 and final < 0.05, (initial, final)
    # Verify checkpoint reload reproduces predictions without re-training.
    optimizer, scheduler = optimizers(model, heads, source, cfg)
    scaler = torch.amp.GradScaler("cuda")
    sampler = StatefulOrder(2, 42)
    save_checkpoint(out / "synthetic.pt", model, heads, optimizer, scheduler, scaler, sampler, cfg, "synthetic-only", {})
    with torch.no_grad():
        before = model(views[15])["tb_logits"].clone()
    with torch.no_grad():
        model.tb_classifier[-1].weight.add_(1)
    load_checkpoint(out / "synthetic.pt", model, heads, manifest_hash="synthetic-only")
    with torch.no_grad():
        after = model(views[15])["tb_logits"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    result = {"hardware": hardware, "scan_gradient_l1": gradient, "tiny_overfit_initial_loss": initial,
              "tiny_overfit_final_loss": final, "checkpoint_predictions_exact": True,
              "seconds": time.monotonic() - started, "scientific_results": False}
    write_json(out / "gpu_verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    verify(p.parse_args().output)
