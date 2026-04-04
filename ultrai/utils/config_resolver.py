#!/usr/bin/env python3
"""
Resolve a run config by shallow-merging YAML files in order.

Examples:
    python3 -m ultrai.utils.config_resolver \
        --base configs/cscs/train.yaml \
        --override configs/cscs/benin_fold3.yaml \
        --set experiment_dir=/tmp/run \
        --output /tmp/run/resolved_config.yaml
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def parse_set_arg(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE format, got: {raw}")
    key, value = raw.split("=", 1)
    return key, yaml.safe_load(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a config by merging YAML files and key overrides.")
    parser.add_argument("--base", required=True, help="Base YAML config")
    parser.add_argument("--override", action="append", default=[], help="Override YAML config(s), applied in order")
    parser.add_argument("--set", dest="sets", action="append", default=[], help="Final KEY=VALUE overrides")
    parser.add_argument("--output", required=True, help="Where to write the resolved YAML")
    args = parser.parse_args()

    resolved = load_yaml(args.base)
    for override_path in args.override:
        resolved.update(load_yaml(override_path))
    for raw in args.sets:
        key, value = parse_set_arg(raw)
        resolved[key] = value

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)


if __name__ == "__main__":
    main()
