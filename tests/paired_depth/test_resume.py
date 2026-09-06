import torch

from ultrai.paired_depth.data import StatefulOrder
from ultrai.paired_depth.engine import load_checkpoint, save_checkpoint, seed_all
from ultrai.paired_depth.objectives import DomainHeads


def test_resume_mid_accumulation_is_bitwise_identical(tmp_path):
    seed_all(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
    heads = DomainHeads(8)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = StatefulOrder(5, 42)
    x = torch.ones(2, 4)
    model(x).square().mean().backward()
    sampler.advance(1)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, heads, opt, sched, scaler, sampler, {"seed": 42}, "manifest", {})
    model(x).square().mean().backward()
    opt.step()
    sched.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    load_checkpoint(path, model, heads, opt, sched, scaler, sampler, {"seed": 42}, "manifest")
    model(x).square().mean().backward()
    opt.step()
    sched.step()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
    assert sampler.cursor == 1
