#!/usr/bin/env python3
"""
Public entrypoint for supervised core-model training.

This is the dataset-aware launcher used by `train_job.sh`. It keeps the public
surface simple:
- select the dataset adapter (`benin` or `sa`)
- forward the resolved training config to the core training engine

The actual CLIP+RL training logic stays in `ultrai.training.core_engine`.
"""

import argparse

from ultrai.data.registry import load_dataset_adapter
from ultrai.training import core_engine


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
    args, forwarded_args = parser.parse_known_args(argv)

    adapter = load_dataset_adapter(args.dataset)
    core_engine.set_dataset_adapter(adapter)
    print(f"Using dataset adapter: {args.dataset} ({adapter.__name__})", flush=True)
    return core_engine.main(forwarded_args)


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    main()
