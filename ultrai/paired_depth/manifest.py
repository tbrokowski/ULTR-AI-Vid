"""Strict inventory, patient partitions, pairing, and decode audit. No SA data."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATASET = "TrustBeninVideos/Trust-Benin-Videos"
REVISION = "12cd0de88e25f39adf279e0105b8cdaf138593b4"
SITES = ["<PAD>", "QAID", "QAIG", "QASD", "QASG", "QLD", "QLG", "QPID",
         "QPIG", "QPSD", "QPSG", "APXD", "APXG", "QSLD", "QSLG", "SAD",
         "SLD", "SAG", "SLG", "SPD", "SPG"]
ALIASES = {"QLID": "QLD", "QLIG": "QLG"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def binary(value):
    try:
        x = float(value)
        return int(x) if x in (0, 1) else -1
    except (TypeError, ValueError):
        return -1


def site_labels(row, site):
    raw_site = {v: k for k, v in ALIASES.items()}.get(site, site)
    def value(suffix):
        values = {binary(row[k]) for k in (f"{site}_{suffix}", f"{raw_site}_{suffix}") if k in row}
        return values.pop() if len(values) == 1 else -1
    if value("Not measured") == 1:
        return [-1] * 4
    other = [value(x) for x in ("B-lines", "Confluent B-lines", "small Consolidations or Nodules")]
    combined = 1 if 1 in other else (0 if all(x == 0 for x in other) else -1)
    return [value("A-line"), value("large Consolidations"), value("Pleural effusion"), combined]


def partitions(directory):
    result = []
    for i in range(5):
        rows = read_csv(Path(directory) / f"Fold_{i}.csv")
        fold = {split: sorted({r[col].strip() for r in rows if r.get(col, "").strip()})
                for split, col in [("train", "train_ids"), ("valid", "valid_ids"), ("test", "test_ids")]}
        if any(not ids for ids in fold.values()):
            raise ValueError(f"Empty patient partition in fold {i}")
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            if set(fold[a]) & set(fold[b]):
                raise ValueError(f"Patient leakage in fold {i}: {a}/{b}")
        result.append(fold)
    if any(f["test"] != result[0]["test"] for f in result):
        raise ValueError("Expected the existing shared test cohort across all five partitions")
    return result


def inspect_video(task):
    """Decode every frame, not only a container header or first frame."""
    import cv2
    cv2.setNumThreads(1)
    name, path, expected, decode = task
    result = {"file": name, "bytes": Path(path).stat().st_size, "sha256": digest(path)}
    if expected and (result["sha256"] != expected["sha256"] or result["bytes"] != expected["size"]):
        result["error"] = "revision_hash_mismatch"
        return result
    if decode:
        cap = cv2.VideoCapture(str(path))
        advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count, shape, bad = 0, None, False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if frame is None or frame.size == 0:
                bad = True
                break
            current = list(frame.shape)
            if shape is not None and shape != current:
                bad = True
            shape = current
        cap.release()
        result.update(frames=count, shape=shape, advertised_frames=advertised)
        if bad or count == 0 or (advertised > 0 and count != advertised):
            result["error"] = "decode_failure_or_truncation"
    return result


def hf_inventory(output):
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    files = {}
    for f in info.siblings:
        if f.rfilename.lower().endswith(".mp4"):
            if not f.lfs:
                raise ValueError(f"No SHA256 available for {f.rfilename}")
            files[f.rfilename] = {"sha256": f.lfs.sha256, "size": f.size}
    write_json(output, {"dataset": DATASET, "revision": info.sha, "files": files})


def prepare(metadata, labels, splits, videos, output, inventory=None, workers=8, decode=True):
    out, root = Path(output), Path(videos)
    rows, labels_rows = read_csv(metadata), read_csv(labels)
    folds = partitions(splits)
    label_groups = defaultdict(list)
    for row in labels_rows:
        label_groups[row["record_id"]].append(row)
    label_map = {p: rs[0] for p, rs in label_groups.items()
                 if len({json.dumps(r, sort_keys=True) for r in rs}) == 1 and binary(rs[0].get("TB Label")) >= 0}
    inv = json.loads(Path(inventory).read_text()) if inventory else None
    if inv and (inv["revision"] != REVISION or inv["dataset"] != DATASET):
        raise ValueError("Unexpected Hugging Face dataset or revision")
    expected = {}
    for remote, meta in (inv or {}).get("files", {}).items():
        name = Path(remote).name
        if name in expected and expected[name] != meta:
            raise ValueError(f"Conflicting repository file basenames: {name}")
        expected[name] = meta
    disk = {p.name: p for p in root.glob("*.mp4")}
    if not disk:
        raise ValueError(f"No videos in {root}")
    missing = sorted(set(expected) - set(disk))
    metadata_groups, mixed = defaultdict(list), defaultdict(set)
    exact_duplicates = 0
    seen = set()
    for row in rows:
        patient, raw_site = row["Patient ID"].strip(), row["Site"].strip()
        site = ALIASES.get(raw_site, raw_site)
        try:
            depth = float(row["Depth"])
        except ValueError:
            continue
        mixed[(patient, site, row["Count"])].add(depth)
        if row["type"] != "video" or depth not in (5, 15):
            continue
        signature = json.dumps(row, sort_keys=True)
        if signature in seen:
            exact_duplicates += 1
            continue
        seen.add(signature)
        # Keep the metadata's actual filename; do not manufacture identifiers.
        key = (patient, site, row["Count"].strip(), int(depth))
        metadata_groups[key].append(row)
    tasks = [(name, str(path), expected.get(name), decode) for name, path in sorted(disk.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audits = list(pool.map(inspect_video, tasks, chunksize=8))
    audit_map = {r["file"]: r for r in audits}
    excluded, records = [], []
    for (patient, site, counter, depth), group in sorted(metadata_groups.items()):
        files = {r["New File Name"] for r in group}
        originals = {r["Original File Name"] for r in group}
        if len(files) != 1 or len(originals) != 1:
            excluded.append({"patient": patient, "site": site, "counter": counter, "depth": depth, "reason": "metadata_conflict"})
            continue
        name = next(iter(files))
        reason = ("missing_file" if name not in disk else
                  "unverified_revision_file" if inv and name not in expected else
                  audit_map[name].get("error") if name in disk else None)
        if reason or patient not in label_map or site not in SITES:
            excluded.append({"file": name, "reason": reason or "missing_label_or_unknown_site"})
            continue
        records.append({"patient": patient, "site": site, "site_index": SITES.index(site),
                        "counter": counter, "depth": depth, "file": name,
                        "pathology": site_labels(label_map[patient], site),
                        "sha256": audit_map[name]["sha256"], "frames": audit_map[name].get("frames")})
    grouped = defaultdict(dict)
    for i, r in enumerate(records):
        grouped[(r["patient"], r["site"], r["counter"])][r["depth"]] = i
    pairs = [{"patient": p, "site": s, "counter": c, "five": rs[5], "fifteen": rs[15]}
             for (p, s, c), rs in sorted(grouped.items()) if 5 in rs and 15 in rs]
    patients = {p: {"tb": binary(row["TB Label"])} for p, row in label_map.items()}
    result = {"schema": 1, "dataset": DATASET, "revision": REVISION,
              "revision_verified": bool(inv) and not missing and not any(r.get("error") == "revision_hash_mismatch" for r in audits),
              "decode_verified": decode, "records": records, "pairs": pairs, "patients": patients, "partitions": folds,
              "input_sha256": {"metadata": digest(metadata), "labels": digest(labels),
                               **{f"fold_{i}": digest(Path(splits) / f"Fold_{i}.csv") for i in range(5)}}}
    paired_patients = {p["patient"] for p in pairs}
    summary = {"dataset": DATASET, "revision": REVISION, "revision_verified": result["revision_verified"],
               "metadata_mixed_matching_groups": sum(5 in v and 15 in v for v in mixed.values()),
               "historical_available_labeled_pairs_before_checks": 2807,
               "historical_available_paired_patients_before_checks": 470,
               "disk_videos": len(disk), "disk_bytes": sum(a["bytes"] for a in audits),
               "eligible_records": len(records), "eligible_pairs": len(pairs), "paired_patients": len(paired_patients),
               "identical_metadata_duplicates": exact_duplicates, "excluded": dict(Counter(x["reason"] for x in excluded)),
               "decode_errors": sum(bool(a.get("error")) for a in audits), "missing_revision_files": len(missing) if inv else None,
               "partitions": [{s: {"patients": len(set(f[s]) & paired_patients),
                                    "pairs": sum(p["patient"] in set(f[s]) for p in pairs)} for s in f} for f in folds],
               "cautions": ["Matching identifiers do not establish synchronized frames.",
                            "Five training/validation partitions share one test cohort."]}
    write_json(out / "manifest.json", result)
    write_json(out / "audit.json", {"videos": audits, "exclusions": excluded, "missing_revision_files": missing})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return result, summary
