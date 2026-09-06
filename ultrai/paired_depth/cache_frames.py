"""Cache deterministic uint8 frame decoding on CPU, keyed by video SHA256."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .data import read_clip
from .manifest import write_json


def cache_one(task):
    root, output, record, frames, size = task
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    path = Path(output) / f"{record['sha256']}-{frames}-{size}.npy"
    if not path.exists():
        clip = read_clip(Path(root) / record["file"], frames, size)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as f:
            np.save(f, (clip * 255).round().to(torch.uint8).numpy(), allow_pickle=False)
        temporary.chmod(0o600)
        temporary.replace(path)
    return path.stat().st_size


def build(manifest, root, output, frames=32, size=224, workers=8):
    os.umask(0o077)
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m = json.loads(Path(manifest).read_text())
    tasks = [(str(root), str(output), r, frames, size) for r in m["records"]]
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, count in enumerate(pool.map(cache_one, tasks, chunksize=8), 1):
            total += count
            if i % 500 == 0:
                print(json.dumps({"cached": i, "total": len(tasks)}), flush=True)
    write_json(output / "summary.json", {"records": len(tasks), "bytes": total, "frames": frames,
               "size": size, "seconds": time.monotonic() - started, "label_dependent": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    build(a.manifest, a.videos, a.output, workers=a.workers)
