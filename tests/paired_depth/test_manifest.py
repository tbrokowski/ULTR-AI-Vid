import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from ultrai.paired_depth.manifest import prepare, inspect_video, digest, partitions
from ultrai.paired_depth.data import read_clip
from ultrai.paired_depth.cache_frames import cache_one


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_exact_depth_dedup_conflict_missing_and_decoding(tmp_path):
    root = tmp_path / "videos"
    root.mkdir()
    rows = []
    for p, site, depth, counter in [("a", "QAID", 5, 1), ("a", "QAID", 15, 1), ("a", "QAID", 6, 1),
                                    ("b", "QAID", 5, 1), ("b", "QAID", 15, 1), ("c", "QAID", 5, 1),
                                    ("c", "QAID", 15, 1), ("c", "QAID", 15, 2)]:
        name = f"{p}_{site}_{depth}_{counter}.mp4"
        rows.append({"Patient ID": p, "Site": site, "Depth": depth, "Count": counter, "New File Name": name,
                     "Original File Name": name, "type": "video"})
        if p == "c" and counter == 2:
            continue  # Missing file.
        if p == "c" and depth == 15:
            (root / name).write_bytes(b"broken")
            continue
        writer = cv2.VideoWriter(str(root / name), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16))
        for i in range(5):
            writer.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
        writer.release()
    rows.append(dict(rows[0]))  # Exact duplicate is harmless.
    rows.append({**rows[3], "Original File Name": "conflicting-acquisition.mp4"})
    write_csv(tmp_path / "metadata.csv", rows)
    write_csv(tmp_path / "labels.csv", [{"record_id": p, "TB Label": y} for p, y in [("a", 0), ("b", 1), ("c", 0)]])
    split = tmp_path / "splits"
    split.mkdir()
    for i in range(5):
        write_csv(split / f"Fold_{i}.csv", [{"train_ids": "a", "valid_ids": "b", "test_ids": "c"}])
    manifest, summary = prepare(tmp_path / "metadata.csv", tmp_path / "labels.csv", split, root, tmp_path / "out", workers=1)
    assert len(manifest["pairs"]) == 1
    assert manifest["pairs"][0]["patient"] == "a"
    assert {r["depth"] for r in manifest["records"]} == {5, 15}
    assert summary["identical_metadata_duplicates"] == 1
    assert summary["excluded"]["metadata_conflict"] == 1
    assert summary["excluded"]["missing_file"] == 1
    assert summary["excluded"]["decode_failure_or_truncation"] == 1
    name = rows[0]["New File Name"]
    bad_hash = inspect_video((name, root / name, {"sha256": "wrong", "size": 0}, True))
    assert bad_hash["error"] == "revision_hash_mismatch"
    assert not manifest["revision_verified"]
    cache = tmp_path / "frames"
    cache.mkdir()
    record = manifest["records"][0]
    cache_one((root, cache, record, 4, 16))
    original = read_clip(root / record["file"], 4, 16)
    cached = read_clip(root / record["file"], 4, 16, cache, record["sha256"])
    assert np.array_equal(original.numpy(), cached.numpy())


def test_patient_leakage_is_an_error(tmp_path):
    for i in range(5):
        write_csv(tmp_path / f"Fold_{i}.csv", [{"train_ids": "a", "valid_ids": "a", "test_ids": "b"}])
    with pytest.raises(ValueError, match="leakage"):
        partitions(tmp_path)
