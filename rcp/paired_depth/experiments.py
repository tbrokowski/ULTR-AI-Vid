"""RCP defaults for the portable experiment generator.

Resolved files remain compatible with existing cluster launch and queue commands.
"""
import argparse
import json

from scripts.paired_depth.experiments import configurations as portable_configurations, select

CONTAINER_ROOT = "/scratch/users/falke/benin-paired-depth-dann"


def configurations(output, partition=0, seed=42, epochs=20):
    return portable_configurations(output, partition, seed, epochs, root=CONTAINER_ROOT,
        videos="/benin/datasets/ULTR-AI/LusBeninVideos",
        frame_cache=CONTAINER_ROOT + "/artifacts/frames32")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("configure")
    c.add_argument("--output", required=True)
    c.add_argument("--partition", type=int, default=0)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--epochs", type=int, default=20, help="Freeze after timing smoke, before viewing validation scores")
    s = sub.add_parser("select")
    s.add_argument("--run-root", required=True)
    s.add_argument("--output", required=True)
    a = p.parse_args()
    if a.action == "configure":
        configurations(a.output, a.partition, a.seed, a.epochs)
    else:
        print(json.dumps(select(a.run_root, a.output), indent=2))
