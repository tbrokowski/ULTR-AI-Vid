"""Bounded single-GPU training with exact restart and validation-only selection."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .data import PatientBags, StatefulOrder, collate, seed_for
from .manifest import digest, write_json
from .metrics import scores
from .objectives import DomainHeads, class_weights, consistency


def merge(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
    return result


def load_config(path):
    override = yaml.safe_load(Path(path).read_text())
    base = yaml.safe_load(Path("configs/paired_depth/base.yaml").read_text())
    cfg = merge(base, override)
    cfg["schema"] = 1
    implementation = sorted(Path("ultrai/paired_depth").glob("*.py")) + sorted(Path("NetworkArchitecture").glob("*.py"))
    cfg["implementation_sha256"] = {str(p): digest(p) for p in implementation}
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def gpu_check():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this run")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"Unsupported GPU {arch}: {torch.cuda.get_arch_list()}")
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    (x @ x).square().mean().backward()
    torch.cuda.synchronize()
    return {"name": torch.cuda.get_device_name(), "capability": arch, "torch": torch.__version__,
            "cuda": torch.version.cuda, "supported_architectures": torch.cuda.get_arch_list()}


def make_model(cfg):
    from NetworkArchitecture.ablation_models import AttentionPoolMultiTaskModel
    from ultrai.training.config import MultiTaskConfig
    settings = dataclasses.asdict(MultiTaskConfig())
    settings.update(yaml.safe_load(Path(cfg["source_config"]).read_text()))
    settings.update(device=cfg["device"], freeze_backbone=cfg["freeze_backbone"], local_weights_dir=None)
    # The official public CLIP revision is pinned by the downloaded snapshot path.
    from huggingface_hub import snapshot_download
    settings["clip_model_name"] = cfg.get("clip_snapshot") or snapshot_download(
        "openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors"])
    model = AttentionPoolMultiTaskModel(SimpleNamespace(**settings))
    if cfg["gradient_checkpointing"] and not cfg["freeze_backbone"]:
        model.vision_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(cfg["device"]), settings


def optimizers(model, heads, source, cfg):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = ("backbone" if name.startswith(("vision_encoder", "frame_selector")) else
                 "pathology" if name.startswith("pathology_modules") else
                 "classifier" if name.startswith(("tb_classifier", "task_classifiers")) else
                 "integration" if name.startswith("site_integration") else "patient_pipeline")
        groups.setdefault(group, []).append(p)
    parameters = [{"params": params, "lr": float(source[f"{group}_lr"]),
                   "weight_decay": float(source[f"{group}_weight_decay"]), "name": group}
                  for group, params in groups.items()]
    parameters.append({"params": list(heads.parameters()), "lr": cfg["domain_learning_rate"], "weight_decay": 1e-5, "name": "domain"})
    opt = torch.optim.AdamW(parameters)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=int(source["backbone_T_0"]), T_mult=int(source["backbone_T_mult"]), eta_min=float(source["backbone_eta_min"]))
    return opt, scheduler


def to_device(batch, device):
    return {"patients": batch["patients"], "tb_labels": batch["tb_labels"].to(device), "pairs": batch["pairs"].to(device),
            "views": {d: {k: v.to(device) for k, v in b.items()} for d, b in batch["views"].items()}}


def optimizer_step(optimizer, scaler):
    """Let AMP skip overflowed updates; reject nonfinite full-precision gradients."""
    scaler.unscale_(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not parameters:
        raise RuntimeError("No gradients reached the optimizer")
    finite = torch.stack([p.grad.isfinite().all() for p in parameters]).all().item()
    if finite:
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    elif not scaler.is_enabled():
        raise FloatingPointError("Nonfinite gradients without mixed-precision scaling")
    # unscale_ has recorded overflows, so step skips the update and update lowers
    # the scale. Do not clip infinite gradients or count a skipped optimizer step.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if not finite and scaler.get_scale() < 1:
        raise FloatingPointError("Nonfinite gradients persist below AMP loss scale 1")
    return bool(finite)


def losses(model, heads, batch, cfg, strength, weights, domain_selection=None):
    outputs, supervised = {}, []
    for depth, inputs in batch["views"].items():
        out = model(inputs)
        outputs[depth] = out
        target = {"tb_labels": batch["tb_labels"], "pathology_labels": inputs["pathology_labels"]}
        loss, _ = model.compute_losses(out, target, {"TB Label": 2.0})
        supervised.append(loss)
    total = torch.stack(supervised).mean()
    details = {"supervised": total.detach().item()}
    components = cfg["losses"]
    if 5 in outputs and 15 in outputs:
        if components["feature"] or components["prediction"]:
            c = consistency(outputs[5], outputs[15], batch["pairs"], batch["views"][5]["pathology_labels"] >= 0)
            for key, loss in c.items():
                total = total + float(components[key]) * loss
                details[key] = loss.detach().item()
        if components["scan_domain"] or components["patient_domain"]:
            valid, rates = None, None
            if domain_selection is not None:
                valid = [batch["tb_labels"].new_tensor([p in ids for p in batch["patients"]]) for ids in domain_selection["ids"]]
                rates = domain_selection["keep_rates"]
            d = heads([outputs[5], outputs[15]], [batch["views"][i]["site_masks"] for i in (5, 15)],
                      [batch["tb_labels"]] * 2, strength, weights, valid, rates)
            for key, loss in d.items():
                total = total + float(components[f"{key}_domain"]) * loss
                details[f"{key}_domain"] = loss.detach().item()
    return total, details


def phase_source_backward(model, batch, cfg, source, scaler, divisor):
    """Legacy source phase routing with accumulated gradients kept for each group.

    Four independent pathology heads share one forward (their parameters are
    disjoint). Patient and backbone phases use separate forwards. All groups step
    at the same accumulation boundary, including a partial final batch.
    """
    if set(batch["views"]) != {15}:
        raise ValueError("Source phase training requires exact 15 cm bags")
    original = {name: p.requires_grad for name, p in model.named_parameters()}
    inputs = batch["views"][15]
    target = {"tb_labels": batch["tb_labels"], "pathology_labels": inputs["pathology_labels"]}
    value = 0.0
    for phase in ("pathology", "patient", "backbone"):
        for name, p in model.named_parameters():
            category = ("pathology" if name.startswith("pathology_modules") else
                        "backbone" if name.startswith(("vision_encoder", "frame_selector")) else "patient")
            p.requires_grad_(original[name] and category == phase)
        model.vision_encoder.train(phase == "backbone" and not cfg["freeze_backbone"])
        with torch.autocast(device_type=cfg["device"], enabled=cfg["amp"] and cfg["device"] == "cuda"):
            out = model(inputs)
            if phase == "pathology":
                loss = out["pathology_scores"].sum() * 0
                for i in range(4):
                    y = inputs["pathology_labels"][..., i]
                    valid = y >= 0
                    if valid.any():
                        loss = loss + F.binary_cross_entropy_with_logits(out["pathology_scores"][..., i][valid], y[valid],
                                  pos_weight=y.new_tensor(source["pathology_pos_weights"][i]))
            else:
                loss, _ = model.compute_losses(out, target, {"TB Label": 2.0})
            scaler.scale(loss / divisor).backward()
        value += loss.detach().item()
        del out, loss
    for name, p in model.named_parameters():
        p.requires_grad_(original[name])
    return value / 3


@torch.no_grad()
def evaluate(model, manifest, cfg, split="valid"):
    model.eval()
    result, predictions = {}, []
    for view in ("paired", "all15"):
        ds = PatientBags(manifest, cfg["videos"], cfg["partition"], split, view, frames=cfg["frames"], size=cfg["size"], frame_cache=cfg.get("frame_cache"))
        by_depth = {5: [], 15: []}
        for i in range(len(ds)):
            batch = to_device(collate([ds[i]]), cfg["device"])
            for depth, inputs in batch["views"].items():
                with torch.autocast(device_type=cfg["device"], enabled=cfg["amp"] and cfg["device"] == "cuda"):
                    probability = model(inputs)["tb_logits"].float().sigmoid().item()
                record = {"patient": ds.ids[i], "label": int(batch["tb_labels"].item()), "probability": probability,
                          "view": f"matched{depth}" if view == "paired" else "all15", "partition": cfg["partition"], "seed": cfg["seed"]}
                by_depth[depth].append(record)
                predictions.append(record)
        for depth, records in by_depth.items():
            if records:
                result[records[0]["view"]] = scores([r["label"] for r in records], [r["probability"] for r in records])
    return result, predictions


def save_checkpoint(path, model, heads, optimizer, scheduler, scaler, sampler, cfg, manifest_hash, progress):
    path = Path(path)
    payload = {"schema": 1, "model_state_dict": model.state_dict(), "domain_state_dict": heads.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
               "scaler_state_dict": scaler.state_dict(), "rng": rng_state(), "sampler": sampler.state_dict(),
               "gradients": {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None},
               "domain_gradients": {k: p.grad.detach().cpu() for k, p in heads.named_parameters() if p.grad is not None},
               "config": cfg, "manifest_sha256": manifest_hash, "progress": progress}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.chmod(0o600)
    temporary.replace(path)


def load_checkpoint(path, model, heads, optimizer=None, scheduler=None, scaler=None, sampler=None, cfg=None, manifest_hash=None):
    # Only load trusted, locally produced checkpoints; torch pickle is executable.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1:
        raise ValueError("Incompatible checkpoint schema; use an explicitly verified import")
    if manifest_hash and state["manifest_sha256"] != manifest_hash:
        raise ValueError("Checkpoint manifest differs from the audited dataset")
    if cfg is not None and state["config"] != cfg:
        raise ValueError("Checkpoint requires the exact resolved configuration")
    model.load_state_dict(state["model_state_dict"], strict=True)
    heads.load_state_dict(state["domain_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        sampler.load_state_dict(state["sampler"])
        for module, key in [(model, "gradients"), (heads, "domain_gradients")]:
            for name, p in module.named_parameters():
                p.grad = state[key][name].to(p.device) if name in state[key] else None
        restore_rng(state["rng"])
    return state


def _run(args):
    os.umask(0o077)
    cfg = load_config(args.config)
    started = time.monotonic()
    seed_all(cfg["seed"])
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if args.stage != "evaluate" and (out / "last.pt").exists() and not args.resume:
        raise FileExistsError("Run already exists; explicitly resume it or choose a new output directory")
    manifest_hash = digest(cfg["manifest"])
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    gpu = gpu_check() if cfg["device"] == "cuda" else {"name": "CPU"}
    model, source = make_model(cfg)
    # Heads are initialized AFTER the model in every arm, preserving model RNG initialization.
    heads = DomainHeads(model.hidden_dim, cfg["losses"]["conditioning"]).to(cfg["device"])
    if cfg.get("source_checkpoint") and not args.resume:
        state = torch.load(cfg["source_checkpoint"], map_location="cpu", weights_only=False)
        if state.get("manifest_sha256") != manifest_hash or state["config"]["partition"] != cfg["partition"] or state["config"]["seed"] != cfg["seed"]:
            raise ValueError("Source checkpoint provenance does not match partition/seed/manifest")
        model.load_state_dict(state["model_state_dict"], strict=True)
    write_json(out / "resolved_config.json", cfg)
    code_files = sorted(Path("ultrai/paired_depth").glob("*.py")) + sorted(Path("NetworkArchitecture").glob("*.py"))
    provenance = {"manifest_sha256": manifest_hash, "source_config": source, "gpu": gpu,
                  "base_commit": "dba7e2d4b9318df8662fc2eeef8437d1b772beb0",
                  "code_sha256": {str(p): digest(p) for p in code_files},
                  "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                  "source_checkpoint_sha256": digest(cfg["source_checkpoint"]) if cfg.get("source_checkpoint") else None}
    if not (out / "provenance.json").exists():
        write_json(out / "provenance.json", provenance)
    executions = out / "executions.json"
    segments = json.loads(executions.read_text()) if executions.exists() else []
    segments.append({"stage": args.stage, "started_unix": time.time(), "gpu": gpu,
                     "resume_checkpoint_sha256": digest(args.resume) if args.resume else None,
                     "evaluation_checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
                     "manifest_sha256": manifest_hash, "code_sha256": provenance["code_sha256"]})
    write_json(executions, segments)
    if args.stage == "evaluate":
        if not args.checkpoint:
            raise ValueError("Evaluation requires --checkpoint")
        if args.split == "test":
            if not args.freeze_file:
                raise ValueError("Test evaluation requires a validation-selected freeze file")
            freeze = json.loads(Path(args.freeze_file).read_text())
            if digest(args.checkpoint) not in freeze["checkpoint_sha256"]:
                raise ValueError("Checkpoint was not frozen before opening the test cohort")
        load_checkpoint(args.checkpoint, model, heads, cfg=cfg, manifest_hash=manifest_hash)
        metrics, predictions = evaluate(model, manifest, cfg, args.split)
        write_json(out / f"{args.split}_metrics.json", metrics)
        write_json(out / f"{args.split}_predictions.json", predictions)
        print(json.dumps(metrics), flush=True)
        return
    train = PatientBags(manifest, cfg["videos"], cfg["partition"], "train", cfg["training_view"], cfg["seed"], cfg["frames"], cfg["size"], cfg["styles"], frame_cache=cfg.get("frame_cache"))
    # Validate before any optimizer step; no fallback to training data.
    for view in ("paired", "all15"):
        valid = PatientBags(manifest, cfg["videos"], cfg["partition"], "valid", view, frames=cfg["frames"], size=cfg["size"])
        if len({manifest["patients"][p]["tb"] for p in valid.ids}) != 2:
            raise ValueError("Validation cohort must contain both TB classes")
    domain_ids = [set(train.ids), set(train.ids)]
    stress = cfg.get("prevalence_stress", {})
    if stress.get("enabled"):
        for domain in (0, 1):
            target = 1 if domain == 0 else 0
            fraction = stress["five_positive_keep" if domain == 0 else "fifteen_negative_keep"]
            if not 0 < fraction <= 1:
                raise ValueError("Domain prevalence retention fractions must be in (0, 1]")
            candidates = sorted((p for p in train.ids if manifest["patients"][p]["tb"] == target),
                                key=lambda p: seed_for(cfg["seed"], 0, p, f"prevalence-{domain}"))
            removed = set(candidates[max(1, round(len(candidates) * fraction)):])
            domain_ids[domain] -= removed
    labels_by_domain = [[manifest["patients"][p]["tb"] for p in sorted(ids)] for ids in domain_ids]
    weights = class_weights(labels_by_domain).to(cfg["device"]) if cfg["losses"]["class_balance"] else None
    domain_selection = {"ids": domain_ids, "keep_rates": [len(ids) / len(train) for ids in domain_ids]}
    write_json(out / "training_domain_frequencies.json", {"class_counts": [[ys.count(0), ys.count(1)] for ys in labels_by_domain],
                                                         "weights": weights.tolist() if weights is not None else None,
                                                         "keep_rates": domain_selection["keep_rates"]})
    optimizer, scheduler = optimizers(model, heads, source, cfg)
    scaler = torch.amp.GradScaler(cfg["device"], enabled=cfg["amp"] and cfg["device"] == "cuda",
                                 init_scale=float(cfg["amp_initial_scale"]))
    sampler = StatefulOrder(len(train), cfg["seed"])
    progress = {"microsteps": 0, "updates": 0, "accumulated": 0, "best": -1.0, "bad_epochs": 0,
                "elapsed_seconds": 0.0, "history": [], "status": "running", "amp_skipped_updates": 0}
    if args.resume:
        state = load_checkpoint(args.resume, model, heads, optimizer, scheduler, scaler, sampler, cfg, manifest_hash)
        progress = state["progress"]
    prior_seconds = progress["elapsed_seconds"]
    stop = {"requested": False}
    def interrupted(signum, frame):
        stop["requested"] = True
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, interrupted)
    last_saved = time.monotonic()
    while sampler.epoch < cfg["epochs"]:
        model.train()
        heads.train()
        if cfg["freeze_backbone"]:
            model.vision_encoder.eval()
        train.epoch = sampler.epoch
        indices = sampler.take(cfg["batch_size"])
        # Match model dropout/noise draws across arms independently of head shape
        # and how many adversarial forward calls the preceding microbatch made.
        seed_all(seed_for(cfg["seed"], sampler.epoch, str(sampler.cursor), "model"))
        batch = to_device(collate([train[i] for i in indices]), cfg["device"])
        # A short final optimizer batch is normalized by its actual patient count.
        window_start = sampler.cursor - progress["accumulated"]
        divisor = min(cfg["effective_batch_size"], len(train) - window_start) / len(indices)
        strength = cfg["losses"]["grl_max"] * (2 / (1 + math.exp(-10 * (sampler.epoch + sampler.cursor / len(train)) / cfg["epochs"])) - 1)
        if not cfg.get("source_checkpoint") and cfg["training_view"] == "all15" and source.get("legacy_hmv_mil_phase_updates"):
            details = {"source_phase_mean": phase_source_backward(model, batch, cfg, source, scaler, divisor)}
        else:
            with torch.autocast(device_type=cfg["device"], enabled=cfg["amp"] and cfg["device"] == "cuda"):
                loss, details = losses(model, heads, batch, cfg, strength, weights, domain_selection)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {details}")
            scaler.scale(loss / divisor).backward()
            del loss
        sampler.advance(len(indices))
        progress["microsteps"] += 1
        progress["accumulated"] += len(indices)
        if progress["accumulated"] >= cfg["effective_batch_size"] or sampler.cursor == len(train):
            applied = optimizer_step(optimizer, scaler)
            progress["accumulated"] = 0
            if applied:
                progress["updates"] += 1
                scheduler.step(sampler.epoch + sampler.cursor / len(train))
            else:
                progress["amp_skipped_updates"] += 1
                print(json.dumps({"amp_overflow": True, "loss_scale": scaler.get_scale(),
                                  "skipped_updates": progress["amp_skipped_updates"]}), flush=True)
        del batch
        if progress["microsteps"] % 10 == 0:
            print(json.dumps({"epoch": sampler.epoch, "patient_cursor": sampler.cursor, "updates": progress["updates"], "loss": details}), flush=True)
        if sampler.cursor == len(train):
            metrics, predictions = evaluate(model, manifest, cfg)
            value = metrics["all15" if cfg["training_view"] == "all15" else "matched5"]["auroc"]
            progress["history"].append({"epoch": sampler.epoch, "metrics": metrics, "updates": progress["updates"]})
            improved = value > progress["best"]
            progress["best"] = max(progress["best"], value)
            progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
            sampler.next_epoch()
            progress["elapsed_seconds"] = prior_seconds + time.monotonic() - started
            if improved:
                save_checkpoint(out / "best.pt", model, heads, optimizer, scheduler, scaler, sampler, cfg, manifest_hash, progress)
                write_json(out / "valid_metrics.json", metrics)
                write_json(out / "valid_predictions.json", predictions)
            write_json(out / "history.json", progress["history"])
            print(json.dumps({"completed_epoch": sampler.epoch, "validation": metrics}), flush=True)
        progress["elapsed_seconds"] = prior_seconds + time.monotonic() - started
        budget_stop = progress["elapsed_seconds"] >= cfg["max_hours"] * 3600 - 180
        smoke_stop = args.stage == "smoke" and progress["microsteps"] >= 2 and progress["updates"] >= 1
        finished = sampler.epoch >= cfg["epochs"] or progress["bad_epochs"] >= cfg["early_stopping_patience"]
        if stop["requested"] or budget_stop or smoke_stop or finished or time.monotonic() - last_saved >= cfg["checkpoint_interval_seconds"]:
            progress["status"] = "interrupted" if stop["requested"] else "budget_exhausted" if budget_stop else "complete" if finished else "smoke_complete" if smoke_stop else "running"
            save_checkpoint(out / "last.pt", model, heads, optimizer, scheduler, scaler, sampler, cfg, manifest_hash, progress)
            write_json(out / "progress.json", progress)
            last_saved = time.monotonic()
        if stop["requested"] or budget_stop or smoke_stop or finished:
            break
    print(json.dumps({"status": progress["status"], "epochs": sampler.epoch, "updates": progress["updates"], "gpu_hours_process": progress["elapsed_seconds"] / 3600}), flush=True)


def run(args):
    from filelock import FileLock
    cfg = load_config(args.config)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    # Independent experiments may run concurrently; two writers to one run may not.
    with FileLock(out / ".run.lock", timeout=0):
        return _run(args)
