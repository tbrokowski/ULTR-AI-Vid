"""Scan/patient DANN, detached prediction conditioning, and paired consistency."""
import torch
from torch import nn
from torch.nn import functional as F


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def condition(features, logits):
    """Binary CDAN multilinear map; conditioning must not train TB predictions."""
    p = logits.sigmoid().detach()
    probs = torch.stack((1 - p, p), dim=-1)
    return (features.unsqueeze(-1) * probs.unsqueeze(-2)).flatten(-2)


def class_weights(labels_by_domain):
    """Equal mass for each observed class in each training domain."""
    result = []
    for labels in labels_by_domain:
        labels = torch.as_tensor(labels, dtype=torch.long)
        counts = torch.bincount(labels, minlength=2).float()
        if (counts == 0).any():
            raise ValueError("Both TB classes are required in each training domain")
        result.append(len(labels) / (2 * counts))
    return torch.stack(result)


class DomainHeads(nn.Module):
    def __init__(self, dim=512, conditional=False):
        super().__init__()
        self.conditional = conditional
        def head():
            return nn.Sequential(nn.Linear(dim * (2 if conditional else 1), 256), nn.LayerNorm(256),
                                 nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.LayerNorm(256),
                                 nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 2))
        self.scan, self.patient = head(), head()

    def forward(self, outputs, masks, labels, strength=1.0, weights=None, domain_valid=None, keep_rates=None):
        losses = {"scan": [], "patient": []}
        for domain, (out, mask, y) in enumerate(zip(outputs, masks, labels)):
            patient = out["patient_features"]
            scans = out["site_features"]
            if not scans.requires_grad and self.training:
                raise RuntimeError("Detached scan representations cannot train domain losses")
            if self.conditional:
                patient = condition(patient, out["tb_logits"])
                scans = condition(scans, out["tb_logits"][:, None].expand(mask.shape))
            for name, features in [("scan", scans), ("patient", patient)]:
                logits = getattr(self, name)(Reverse.apply(features, strength))
                target = torch.full(logits.shape[:-1], domain, device=logits.device, dtype=torch.long)
                element = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1), reduction="none").reshape(target.shape)
                w = torch.ones_like(y) if weights is None else weights[domain, y.long()]
                if name == "scan":
                    # First average scans within patients, then patients within domains.
                    element = (element * mask).sum(1) / mask.sum(1).clamp_min(1)
                if domain_valid is not None:
                    w = w * domain_valid[domain] / keep_rates[domain]
                # Population weights have expectation one. Dividing by the current
                # microbatch weight sum would cancel weighting at batch size one.
                losses[name].append((element * w).mean())
        return {k: torch.stack(v).mean() for k, v in losses.items()}


def consistency(a, b, pairs, pathology_valid):
    """Mappings are (batch, five-position, fifteen-position), never assumed frame alignment."""
    zero = (a["patient_features"].sum() + b["patient_features"].sum()) * 0
    if pairs.numel() == 0:
        return {"feature": zero, "prediction": zero}
    batch, ia, ib = pairs.unbind(1)
    fa, fb = a["site_features"][batch, ia], b["site_features"][batch, ib]
    distances = 1 - F.cosine_similarity(fa, fb, dim=-1)
    # Equal patient weight despite variable scan counts.
    feature = torch.stack([distances[batch == i].mean() for i in batch.unique()]).mean()
    pa = a["pathology_scores"][batch, ia].sigmoid()
    pb = b["pathology_scores"][batch, ib].sigmoid()
    valid = pathology_valid[batch, ia]
    path = ((pa - pb).square() * valid).sum() / valid.sum().clamp_min(1)
    tb = (a["tb_logits"].sigmoid() - b["tb_logits"].sigmoid()).square().mean()
    return {"feature": feature, "prediction": (path + tb) / 2}
