"""Patient bags with exact depth, explicit pair mappings and valid-prefix padding."""
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .styles import synthetic_style


def seed_for(seed, epoch, patient, stream=""):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{patient}:{stream}".encode()).digest()[:8], "little") % (2**31)


def read_clip(path, frames=32, size=224, cache=None, sha=None):
    if cache and sha:
        cached = Path(cache) / f"{sha}-{frames}-{size}.npy"
        if cached.exists():
            array = np.load(cached, allow_pickle=False)
            if array.shape != (frames, 3, size, size) or array.dtype != np.uint8:
                raise ValueError(f"Invalid decoded-frame cache: {cached}")
            return torch.from_numpy(array).float().div_(255)
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        raise RuntimeError(f"Undecodable video: {path}")
    selected = np.linspace(0, count - 1, frames).astype(int)
    wanted, images, i = set(selected.tolist()), {}, 0
    while i <= selected[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frame = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
            images[i] = torch.from_numpy(frame.copy()).permute(2, 0, 1)
        i += 1
    cap.release()
    if not wanted.issubset(images):
        raise RuntimeError(f"Video changed or truncated after manifest audit: {path}")
    return torch.stack([images[i] for i in selected]).float().div_(255)


class PatientBags(Dataset):
    def __init__(self, manifest, root, partition, split, view="paired", seed=42, frames=32,
                 size=224, styles=None, require_verified=True, frame_cache=None):
        if require_verified and not (manifest["revision_verified"] and manifest["decode_verified"]):
            raise ValueError("Training requires a revision-verified and decoded manifest")
        self.manifest, self.root = manifest, Path(root)
        self.seed, self.epoch, self.frames, self.size = seed, 0, frames, size
        self.styles = styles or {}
        self.frame_cache = frame_cache
        self.split, self.view = split, view
        allowed = set(manifest["partitions"][partition][split])
        training = set(manifest["partitions"][partition]["train"])
        records = manifest["records"]
        self.by_patient = {}
        for p in sorted(allowed):
            if view == "paired":
                pairs = [x for x in manifest["pairs"] if x["patient"] == p]
                if pairs:
                    self.by_patient[p] = {5: [x["five"] for x in pairs], 15: [x["fifteen"] for x in pairs]}
            elif view == "all15":
                indices = [i for i, r in enumerate(records) if r["patient"] == p and r["depth"] == 15]
                if indices:
                    self.by_patient[p] = {15: indices}
            else:
                raise ValueError(view)
        self.ids = sorted(self.by_patient)
        if not self.ids:
            raise ValueError(f"No eligible {split} patients for {view}")
        self.donors = [i for i, r in enumerate(records) if r["patient"] in training]
        if split != "train" and any(self.styles.get(k) for k in ("gain", "speckle", "resampling", "fourier")):
            raise ValueError("Synthetic styles are restricted to training")
        if self.styles.get("fourier") and not self.donors:
            raise ValueError("No training-only Fourier donors")
        assert all(records[i]["patient"] in training for i in self.donors)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        p = self.ids[index]
        g = torch.Generator().manual_seed(seed_for(self.seed, self.epoch, p))
        records = self.manifest["records"]
        result = {"patient": p, "tb": self.manifest["patients"][p]["tb"], "views": {}}
        for depth, indices in self.by_patient[p].items():
            clips, sites, labels = [], [], []
            for i in indices:
                r = records[i]
                clip = read_clip(self.root / r["file"], self.frames, self.size, self.frame_cache, r.get("sha256"))
                donor = None
                if self.styles.get("fourier"):
                    donor_index = self.donors[torch.randint(len(self.donors), (), generator=g).item()]
                    donor_record = records[donor_index]
                    donor = read_clip(self.root / donor_record["file"], self.frames, self.size, self.frame_cache, donor_record.get("sha256"))
                if self.split == "train":
                    clip = synthetic_style(clip, g, self.styles, donor)
                mean = clip.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = clip.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                clips.append((clip - mean) / std)
                sites.append(r["site_index"])
                labels.append(r["pathology"])
            result["views"][depth] = {"site_videos": torch.stack(clips), "site_indices": torch.tensor(sites),
                                      "pathology_labels": torch.tensor(labels, dtype=torch.float32)}
        return result


def collate(items):
    result = {"patients": [x["patient"] for x in items], "tb_labels": torch.tensor([x["tb"] for x in items], dtype=torch.float32), "views": {}}
    depths = items[0]["views"].keys()
    for depth in depths:
        n = max(x["views"][depth]["site_videos"].shape[0] for x in items)
        sample = items[0]["views"][depth]["site_videos"]
        videos = sample.new_zeros((len(items), n, *sample.shape[1:]))
        sites = torch.zeros((len(items), n), dtype=torch.long)
        masks = torch.zeros((len(items), n), dtype=torch.bool)
        labels = torch.full((len(items), n, 4), -1.0)
        for b, x in enumerate(items):
            view = x["views"][depth]
            count = len(view["site_indices"])
            videos[b, :count] = view["site_videos"]
            sites[b, :count] = view["site_indices"]
            labels[b, :count] = view["pathology_labels"]
            masks[b, :count] = True
        result["views"][depth] = {"site_videos": videos, "site_indices": sites, "site_masks": masks,
                                  "pathology_labels": labels, "domain": torch.full((len(items),), int(depth == 15))}
    mappings = []
    if 5 in depths and 15 in depths:
        for b, item in enumerate(items):
            a, c = item["views"][5], item["views"][15]
            if not torch.equal(a["site_indices"], c["site_indices"]):
                raise ValueError("Paired scan order differs between depths")
            mappings.extend((b, i, i) for i in range(len(a["site_indices"])))
    result["pairs"] = torch.tensor(mappings, dtype=torch.long).reshape(-1, 3)
    return result


class StatefulOrder:
    """Consume only after a completed microbatch. No hidden DataLoader prefetch state."""
    def __init__(self, length, seed):
        self.length, self.seed, self.epoch, self.cursor = length, seed, 0, 0
        self.order = self._order()

    def _order(self):
        return torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()

    def take(self, batch_size):
        return self.order[self.cursor:self.cursor + batch_size]

    def advance(self, count):
        self.cursor += count

    def next_epoch(self):
        self.epoch += 1
        self.cursor = 0
        self.order = self._order()

    def state_dict(self):
        return dict(length=self.length, seed=self.seed, epoch=self.epoch, cursor=self.cursor, order=self.order)

    def load_state_dict(self, state):
        if (state["length"], state["seed"]) != (self.length, self.seed):
            raise ValueError("Incompatible sampler checkpoint")
        self.__dict__.update(state)
