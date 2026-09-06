import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultrai.paired_depth.data import PatientBags, StatefulOrder, collate
from ultrai.paired_depth.manifest import partitions, site_labels
from ultrai.paired_depth.objectives import DomainHeads, Reverse, class_weights, condition, consistency
from ultrai.paired_depth.styles import fourier_style, synthetic_style


def test_grl_and_detached_conditioning():
    x = torch.randn(3, 4, requires_grad=True)
    Reverse.apply(x, 0.7).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, -0.7))
    x.grad = None
    logits = torch.randn(3, requires_grad=True)
    condition(x, logits).square().sum().backward()
    assert x.grad.abs().sum() > 0
    assert logits.grad is None


def test_training_class_frequency_weights():
    weights = class_weights([[0, 0, 0, 1], [0, 1, 1, 1]])
    torch.testing.assert_close(weights, torch.tensor([[2 / 3, 2], [2, 2 / 3]]))
    with pytest.raises(ValueError):
        class_weights([[0, 0], [0, 1]])


def output(batch=2, scans=3, dim=8):
    return {"patient_features": torch.randn(batch, dim, requires_grad=True),
            "site_features": torch.randn(batch, scans, dim, requires_grad=True),
            "tb_logits": torch.randn(batch, requires_grad=True),
            "pathology_scores": torch.randn(batch, scans, 4, requires_grad=True)}


def test_domain_gradients_and_padding():
    torch.manual_seed(42)
    a, b = output(), output()
    masks = torch.tensor([[True, True, False], [True, False, False]])
    heads = DomainHeads(8, conditional=True)
    losses = heads([a, b], [masks, masks], [torch.tensor([0., 1.])] * 2)
    sum(losses.values()).backward()
    for x in (a, b):
        assert x["site_features"].grad[masks].abs().sum() > 0
        assert x["site_features"].grad[~masks].abs().sum() == 0
        assert x["patient_features"].grad.abs().sum() > 0
        assert x["tb_logits"].grad is None


def test_pair_mapping_and_prediction_masks():
    a, b = output(), output()
    pairs = torch.tensor([[0, 0, 2], [1, 1, 0]])
    valid = torch.zeros(2, 3, 4, dtype=torch.bool)
    losses = consistency(a, b, pairs, valid)
    sum(losses.values()).backward()
    assert a["site_features"].grad[0, 0].abs().sum() > 0
    assert b["site_features"].grad[0, 2].abs().sum() > 0
    assert a["site_features"].grad[0, 1:].abs().sum() == 0
    assert a["pathology_scores"].grad.abs().sum() == 0


def test_temporal_styles_and_fourier_phase():
    frame = torch.rand(1, 3, 32, 32)
    clip = frame.expand(8, -1, -1, -1).clone()
    donor = torch.rand_like(clip)
    result = synthetic_style(clip, torch.Generator().manual_seed(4),
                             dict(gain=True, speckle=True, resampling=True, fourier=True), donor)
    torch.testing.assert_close(result[0], result[7])
    styled = fourier_style(clip, donor)
    fa, fb = torch.fft.fft2(clip), torch.fft.fft2(styled)
    nonzero = (fa.abs() > 1e-4) & (fb.abs() > 1e-4)
    torch.testing.assert_close((fa / fa.abs())[nonzero], (fb / fb.abs())[nonzero], atol=2e-4, rtol=2e-4)


def synthetic_manifest():
    records = [{"patient": p, "site": "QAID", "site_index": 1, "counter": "1", "depth": depth,
                "file": f"{p}-{depth}.mp4", "pathology": [0, 1, -1, 0]}
               for p in ("train", "valid", "test") for depth in (5, 15)]
    return {"revision_verified": True, "decode_verified": True, "records": records,
            "patients": {p: {"tb": 0} for p in ("train", "valid", "test")},
            "pairs": [{"patient": p, "five": 2 * i, "fifteen": 2 * i + 1} for i, p in enumerate(("train", "valid", "test"))],
            "partitions": [{"train": ["train"], "valid": ["valid"], "test": ["test"]}]}


def test_donors_and_fail_closed_data():
    m = synthetic_manifest()
    ds = PatientBags(m, ".", 0, "train", styles={"fourier": True})
    assert {m["records"][i]["patient"] for i in ds.donors} == {"train"}
    with pytest.raises(ValueError, match="restricted"):
        PatientBags(m, ".", 0, "valid", styles={"fourier": True})
    m["partitions"][0]["valid"] = []
    with pytest.raises(ValueError, match="No eligible"):
        PatientBags(m, ".", 0, "valid")
    m["revision_verified"] = False
    with pytest.raises(ValueError, match="verified"):
        PatientBags(m, ".", 0, "train")


def test_collation_keeps_valid_prefix_and_alignment():
    items = []
    for count in (3, 1):
        views = {depth: {"site_videos": torch.ones(count, 3, 3, 8, 8), "site_indices": torch.arange(count) + 1,
                         "pathology_labels": torch.ones(count, 4)} for depth in (5, 15)}
        items.append({"patient": str(count), "tb": 0, "views": views})
    batch = collate(items)
    assert batch["pairs"].tolist() == [[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 0]]
    assert batch["views"][5]["site_masks"].tolist() == [[True, True, True], [True, False, False]]
    assert (batch["views"][5]["pathology_labels"][1, 1:] == -1).all()
    items[0]["views"][15]["site_indices"][0] = 20
    with pytest.raises(ValueError, match="order"):
        collate(items)


def test_pathology_missing_is_not_negative():
    assert site_labels({}, "QAID") == [-1] * 4
    assert site_labels({"QAID_B-lines": "1"}, "QAID")[3] == 1
    assert site_labels({"QLID_A-line": "1"}, "QLD")[0] == 1
    assert site_labels({"QAID_Not measured": "1", "QAID_A-line": "1"}, "QAID") == [-1] * 4


def test_existing_patient_partitions_are_shared_test():
    folds = partitions("Data/test_files")
    assert len(folds[0]["test"]) == 101
    assert all(f["test"] == folds[0]["test"] for f in folds)


def test_sampler_resume():
    sampler = StatefulOrder(13, 42)
    initial = sampler.take(4)
    sampler.advance(4)
    resumed = StatefulOrder(13, 42)
    resumed.load_state_dict(sampler.state_dict())
    assert sampler.take(4) == resumed.take(4)
    assert not set(initial) & set(resumed.take(4))
    sampler.next_epoch()
    resumed.next_epoch()
    assert sampler.order == resumed.order
