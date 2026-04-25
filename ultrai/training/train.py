#!/usr/bin/env python3
"""
Public entrypoint for supervised core-model training.

This is the dataset-aware launcher used by `train_job.sh`. It keeps the public
surface simple:
- select the dataset adapter (`benin` or `sa`)
- forward the resolved training config to the core training engine

The launcher routes core CLIP+RL configs to `ultrai.training.core_engine` and
ablation/HMV-MIL configs to `ultrai.training.ablation_engine`.
"""

import argparse
from pathlib import Path

import yaml

from ultrai.data.registry import load_dataset_adapter


ABLATION_MODEL_TYPES = {
    "no_rl",
    "mean_pool",
    "attention_pool",
    "single_task",
    "3d_cnn",
    "cnn_lstm",
    "video_transformer",
    "vivit",
    "uniform",
    "no_rl_full_train",
    "no_pathology",
    "no_keyframe",
    "distilled_edge",
    "multi_backbone_ensemble",
    "dinov3_bert",
}


def _model_type_from_forwarded_args(forwarded_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--model_type")
    args, _ = parser.parse_known_args(forwarded_args)

    if args.model_type:
        return args.model_type

    if not args.config:
        return None

    config_path = Path(args.config)
    if not config_path.exists():
        return None

    with config_path.open("r") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        return None
    return config.get("model_type")


def run(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch supervised core-model training with the selected dataset adapter."
    )
    parser.add_argument(
        "--dataset",
        choices=["benin", "sa"],
        default="benin",
        help="Dataset adapter to use. Default: benin.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "core", "ablation"],
        default="auto",
        help="Training engine. Auto routes ablation model types through ablation_engine.",
    )
    args, forwarded_args = parser.parse_known_args(argv)

    adapter = load_dataset_adapter(args.dataset)
    model_type = _model_type_from_forwarded_args(forwarded_args)
    engine = args.engine
    if engine == "auto":
        engine = "ablation" if model_type in ABLATION_MODEL_TYPES else "core"

    if engine == "ablation":
        from ultrai.training import ablation_engine

        ablation_engine.set_dataset_adapter(adapter)
        print(
            f"Using dataset adapter: {args.dataset} ({adapter.__name__}); "
            f"engine: ablation; model_type: {model_type}",
            flush=True,
        )
        return ablation_engine.main(forwarded_args)

    from ultrai.training import core_engine

    core_engine.set_dataset_adapter(adapter)
    print(
        f"Using dataset adapter: {args.dataset} ({adapter.__name__}); "
        f"engine: core; model_type: {model_type}",
        flush=True,
    )
    return core_engine.main(forwarded_args)


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    main()
