import argparse


def main():
    p = argparse.ArgumentParser(description="Benin paired-depth study: restricted artifacts only")
    sub = p.add_subparsers(dest="stage", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--output", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--configs", nargs="+", required=True)
    freeze.add_argument("--selection", required=True)
    freeze.add_argument("--output", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--metadata", default="Data/processed_files_2.csv")
    prep.add_argument("--labels", default="Data/labels/labels_multidiagnosis.csv")
    prep.add_argument("--splits", default="Data/test_files")
    prep.add_argument("--videos", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--inventory")
    prep.add_argument("--workers", type=int, default=8)
    prep.add_argument("--skip-decode", action="store_true", help="Audit only; resulting manifest cannot train")
    for stage in ("train", "evaluate", "smoke"):
        s = sub.add_parser(stage)
        s.add_argument("--config", required=True)
        s.add_argument("--resume")
        s.add_argument("--checkpoint")
        s.add_argument("--split", choices=["valid", "test"], default="valid")
        s.add_argument("--freeze-file")
    a = p.parse_args()
    if a.stage == "inventory":
        from .manifest import hf_inventory
        hf_inventory(a.output)
    elif a.stage == "freeze":
        import json
        from pathlib import Path
        from .reporting import freeze
        freeze(a.configs, a.output, json.loads(Path(a.selection).read_text()))
    elif a.stage == "prepare":
        from .manifest import prepare
        prepare(a.metadata, a.labels, a.splits, a.videos, a.output, a.inventory, a.workers, not a.skip_decode)
    else:
        from .engine import run
        run(a)


if __name__ == "__main__":
    main()
