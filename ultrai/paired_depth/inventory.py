"""Verify the existing read-only copy before downloading missing pinned files."""
import argparse
import json
from pathlib import Path

from .manifest import DATASET, REVISION, digest, hf_inventory, write_json


def materialize(existing, destination, inventory):
    from huggingface_hub import hf_hub_download
    existing, destination = Path(existing), Path(destination)
    inv = json.loads(Path(inventory).read_text())
    if inv["dataset"] != DATASET or inv["revision"] != REVISION:
        raise ValueError("Incorrect pinned inventory")
    destination.mkdir(parents=True, exist_ok=True)
    by_name, missing, verified = {}, [], []
    for remote, expected in sorted(inv["files"].items()):
        name = Path(remote).name
        if name in by_name:
            raise ValueError("Ambiguous basenames in dataset repository")
        by_name[name] = remote
        path = existing / name
        if path.exists():
            if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
                raise ValueError(f"Existing file differs from pinned revision: {name}; do not overwrite it")
            verified.append((name, path.resolve()))
        else:
            missing.append((name, remote, expected))
    # Only after verifying every present file, download the missing files.
    for name, remote, expected in missing:
        downloaded = Path(hf_hub_download(DATASET, remote, repo_type="dataset", revision=REVISION,
                                         local_dir=destination / "downloaded"))
        if downloaded.stat().st_size != expected["size"] or digest(downloaded) != expected["sha256"]:
            raise ValueError(f"Downloaded file integrity failure: {name}")
        verified.append((name, downloaded.resolve()))
    for name, target in verified:
        link = destination / name
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Conflicting overlay link: {name}")
        elif link.exists():
            raise ValueError(f"Refusing to replace overlay file: {name}")
        else:
            link.symlink_to(target)
    write_json(destination / "verification.json", {"dataset": DATASET, "revision": REVISION,
               "existing_verified": len(verified) - len(missing), "downloaded": len(missing)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--existing", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--inventory", required=True)
    a = p.parse_args()
    materialize(a.existing, a.destination, a.inventory)
