from types import SimpleNamespace

import torch

from ultrai.paired_depth.engine import losses, load_config, make_model
from ultrai.paired_depth.objectives import DomainHeads


class TinyVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 768)

    def forward(self, x):
        return SimpleNamespace(pooler_output=self.projection(x.mean((-2, -1))))


def test_real_hmv_architecture_gradients_and_disabled_equivalence(monkeypatch):
    import NetworkArchitecture.CLIP_DRL_Aug11 as original
    monkeypatch.setattr(original, "create_clip_vision_encoder", lambda **kwargs: TinyVision())
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: "/test-only/fake-clip")
    torch.set_num_threads(2)
    cfg = load_config("configs/paired_depth/base.yaml")
    cfg.update(device="cpu", freeze_backbone=True, gradient_checkpointing=False, amp=False)
    model, _ = make_model(cfg)
    model.eval()  # Deterministic comparison, with autograd enabled.
    views = {d: {"site_videos": torch.randn(2, 2, 4, 3, 8, 8), "site_indices": torch.tensor([[1, 2], [2, 0]]),
                 "site_masks": torch.tensor([[True, True], [True, False]]),
                 "pathology_labels": torch.tensor([[[1., 0, -1, 0], [0, 1, 0, -1]], [[1, 0, 0, 0], [-1, -1, -1, -1]]])}
             for d in (5, 15)}
    batch = {"views": views, "tb_labels": torch.tensor([0., 1.]), "pairs": torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 0]])}
    heads = DomainHeads(512)
    loss, _ = losses(model, heads, batch, cfg, 1, None)
    reference = sum(model.compute_losses(model(v), {"tb_labels": batch["tb_labels"], "pathology_labels": v["pathology_labels"]}, {"TB Label": 2.0})[0] for v in views.values()) / 2
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    cfg["losses"].update(scan_domain=0.1, patient_domain=0.1, feature=0.1, prediction=0.1)
    total, _ = losses(model, heads, batch, cfg, 1, None)
    total.backward()
    assert model.frame_selector.output_projection[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision_encoder.parameters())
