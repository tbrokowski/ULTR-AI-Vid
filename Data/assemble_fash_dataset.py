#!/usr/bin/env python3
"""Verify protected Butterfly downloads and assemble a private HF staging tree.

The input manifests contain sensitive source metadata.  This program uses that
metadata only for validation and writes a deliberately smaller public schema:
pseudonymous patient IDs, archive names, derived stable keys, checksums, sizes,
relative paths, and duplicate-group information.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Iterable, Sequence
import uuid


FORMAT_VERSION = 1
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
SAFE_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,10}")
STUDY_KEY_RE = re.compile(r"[0-9a-f]{12}")
CAPTURE_KEY_RE = re.compile(r"[0-9a-f]{16}")
CAPTURE_METADATA_FIELDS = frozenset(
    {
        "archive",
        "capture_key",
        "content_duplicate_capture_count",
        "content_duplicate_group",
        "content_duplicate_patient_count",
        "content_duplicate_study_count",
        "path",
        "patient_id",
        "resource_duplicate_group",
        "resource_duplicate_capture_count",
        "resource_duplicate_content_hash_count",
        "resource_duplicate_patient_count",
        "resource_duplicate_study_count",
        "sha256",
        "size_bytes",
        "study_key",
    }
)
STUDY_METADATA_FIELDS = frozenset(
    {
        "archive",
        "capture_count",
        "content_duplicate_groups",
        "patient_id",
        "resource_duplicate_groups",
        "size_bytes",
        "study_key",
    }
)
SUMMARY_FIELDS = frozenset(
    {
        "archive_counts",
        "capture_count",
        "content_duplicate_capture_count",
        "content_duplicate_group_count",
        "content_duplicate_groups_across_patients",
        "content_duplicate_groups_across_studies",
        "content_redundant_capture_count",
        "format_version",
        "patient_count",
        "resource_duplicate_capture_count",
        "resource_duplicate_group_count",
        "resource_duplicate_groups_across_patients",
        "resource_duplicate_groups_across_studies",
        "resource_duplicate_groups_with_content_conflicts",
        "resource_redundant_capture_count",
        "selection_sha256",
        "size_bytes",
        "staging_transfer_counts",
        "study_count",
        "unique_content_count",
        "unique_resource_count",
    }
)
COPY_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EPERM,
    errno.EACCES,
    errno.EMLINK,
    getattr(errno, "ENOTSUP", errno.EPERM),
    getattr(errno, "EOPNOTSUPP", errno.EPERM),
}


class AssemblyError(RuntimeError):
    """Raised when an input or output safety invariant is not satisfied."""


@dataclass(frozen=True)
class SelectedStudy:
    archive: str
    patient_id: str
    study_resource_id: str
    study_key: str

    @property
    def identity(self) -> tuple[str, str]:
        return self.archive, self.study_resource_id


@dataclass(frozen=True)
class ManifestState:
    status: str
    record: dict
    manifest_path: Path
    raw_root: Path
    line_number: int
    sequence: int

    @property
    def location(self) -> str:
        return f"{self.manifest_path}:{self.line_number}"


@dataclass(frozen=True)
class VerifiedCapture:
    study: SelectedStudy
    capture_key: str
    capture_resource_id: str
    source_path: Path
    source_name: str
    size_bytes: int
    sha256: str
    source_signature: tuple[int, int, int, int]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(info: os.stat_result) -> tuple[int, int, int, int]:
    """Return fields that reveal a source replacement or in-place modification."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    )


def _validate_private_file_info(info: os.stat_result, path: Path, label: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise AssemblyError(f"{label} is not a regular file: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise AssemblyError(
            f"{label} must not be accessible by group or other users: {path}"
        )


def _read_private_file_snapshot(path: Path, label: str) -> bytes:
    """Read one stable private-file snapshot for hashing and parsing."""
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            _validate_private_file_info(before, path, label)
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AssemblyError(f"Cannot read {label} {path}: {exc}") from exc
    if file_signature(before) != file_signature(after) or len(payload) != after.st_size:
        raise AssemblyError(f"{label} changed while it was being read: {path}")
    return payload


def _jsonl_records(
    path: Path, label: str, payload: bytes
) -> Iterable[tuple[int, dict]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"{path}: {label} is not valid UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssemblyError(
                f"{path}:{line_number}: invalid JSON in {label}"
            ) from exc
        if not isinstance(record, dict):
            raise AssemblyError(
                f"{path}:{line_number}: each {label} row must be a JSON object"
            )
        yield line_number, record


def study_key_for_resource(resource_id: str) -> str:
    return sha256_text(resource_id)[:12]


def capture_key_for_resource(resource_id: str) -> str:
    return sha256_text(resource_id)[:16]


def _required_string(record: dict, field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise AssemblyError(f"{location}: {field} must be a non-empty string")
    return value


def _required_count(record: dict, field: str, location: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssemblyError(f"{location}: {field} must be a non-negative integer")
    return value


def _read_jsonl(path: Path, label: str) -> Iterable[tuple[int, dict]]:
    yield from _jsonl_records(path, label, _read_private_file_snapshot(path, label))


def _require_private_file(path: Path, label: str) -> None:
    try:
        info = path.stat()
    except OSError as exc:
        raise AssemblyError(f"Cannot read {label} {path}: {exc}") from exc
    _validate_private_file_info(info, path, label)


def _require_private_raw_root(path: Path) -> None:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise AssemblyError(f"Cannot inspect raw download directory {path}: {exc}") from exc
    if mode & 0o077:
        raise AssemblyError(
            f"Raw download directory must be private (mode 0700 or stricter): {path}"
        )


def read_selection(path: Path, snapshot: bytes | None = None) -> list[SelectedStudy]:
    """Read and validate the protected download selection."""
    resolved = path.expanduser().resolve()
    selected: list[SelectedStudy] = []
    identities: set[tuple[str, str]] = set()
    keys: dict[str, tuple[str, str]] = {}
    records = (
        _read_jsonl(resolved, "selection file")
        if snapshot is None
        else _jsonl_records(resolved, "selection file", snapshot)
    )
    for line_number, record in records:
        location = f"{resolved}:{line_number}"
        archive = _required_string(record, "archive", location)
        patient_id = _required_string(record, "patient_id", location)
        resource_id = _required_string(record, "study_resource_id", location)
        study_key = _required_string(record, "study_key", location)
        if not SAFE_COMPONENT_RE.fullmatch(patient_id):
            raise AssemblyError(
                f"{location}: patient_id is not a safe pseudonymous path component"
            )
        expected_key = study_key_for_resource(resource_id)
        if study_key != expected_key:
            raise AssemblyError(
                f"{location}: study_key does not match study_resource_id "
                f"(expected {expected_key})"
            )
        identity = (archive, resource_id)
        if identity in identities:
            raise AssemblyError(f"{location}: duplicate selected study")
        if study_key in keys and keys[study_key] != identity:
            raise AssemblyError(f"{location}: colliding study_key in selection")
        identities.add(identity)
        keys[study_key] = identity
        selected.append(
            SelectedStudy(
                archive=archive,
                patient_id=patient_id,
                study_resource_id=resource_id,
                study_key=study_key,
            )
        )
    if not selected:
        raise AssemblyError(f"Selection file contains no studies: {resolved}")
    return selected


def _validate_manifest_identity(
    state: ManifestState, study: SelectedStudy
) -> None:
    record = state.record
    location = state.location
    if _required_string(record, "archive", location) != study.archive:
        raise AssemblyError(f"{location}: archive does not match selection")
    if _required_string(record, "study_resource_id", location) != study.study_resource_id:
        raise AssemblyError(f"{location}: study_resource_id does not match selection")
    if _required_string(record, "patient_id", location) != study.patient_id:
        raise AssemblyError(f"{location}: patient_id does not match selection")
    if _required_string(record, "study_key", location) != study.study_key:
        raise AssemblyError(f"{location}: study_key does not match selection")


def _safe_source_path(state: ManifestState) -> tuple[Path, str]:
    filename = _required_string(state.record, "filename", state.location)
    relative = Path(filename)
    if relative.is_absolute():
        raise AssemblyError(f"{state.location}: capture filename must be relative")
    raw_root = state.raw_root.resolve()
    candidate = (raw_root / relative).resolve()
    try:
        candidate.relative_to(raw_root)
    except ValueError as exc:
        raise AssemblyError(
            f"{state.location}: capture filename escapes its raw download directory"
        ) from exc
    return candidate, filename


def load_and_verify_captures(
    selection: Sequence[SelectedStudy], manifest_paths: Sequence[Path]
) -> list[VerifiedCapture]:
    """Resolve latest manifest state and verify completeness, size, and SHA-256."""
    if not manifest_paths:
        raise AssemblyError("At least one download manifest is required")
    selected_by_identity = {study.identity: study for study in selection}
    selected_by_key = {study.study_key: study for study in selection}
    latest_captures: dict[tuple[str, str], dict[str, ManifestState]] = defaultdict(dict)
    latest_markers: dict[tuple[str, str], ManifestState] = {}
    capture_resources_by_key: dict[tuple[tuple[str, str], str], str] = {}
    sequence = 0

    for manifest_argument in manifest_paths:
        manifest_path = manifest_argument.expanduser().resolve()
        _require_private_file(manifest_path, "download manifest")
        raw_root = manifest_path.parent.resolve()
        _require_private_raw_root(raw_root)
        for line_number, record in _read_jsonl(manifest_path, "download manifest"):
            sequence += 1
            status = record.get("status")
            if not isinstance(status, str) or status not in {
                "downloaded",
                "failed",
                "study_complete",
                "study_sample_complete",
            }:
                continue
            archive = record.get("archive")
            resource_id = record.get("study_resource_id")
            identity = (archive, resource_id)
            study = (
                selected_by_identity.get(identity)
                if isinstance(archive, str) and isinstance(resource_id, str)
                else None
            )
            # A stable key that belongs to the selection makes the row relevant
            # even when another identity field was corrupted.  This prevents a
            # bad row from being silently treated as an unrelated study.
            raw_study_key = record.get("study_key")
            key_study = (
                selected_by_key.get(raw_study_key)
                if isinstance(raw_study_key, str)
                else None
            )
            if study is None:
                study = key_study
            elif key_study is not None and key_study != study:
                raise AssemblyError(
                    f"{manifest_path}:{line_number}: manifest identity fields "
                    "refer to different selected studies"
                )
            if study is None:
                continue
            state = ManifestState(
                status=status,
                record=record,
                manifest_path=manifest_path,
                raw_root=raw_root,
                line_number=line_number,
                sequence=sequence,
            )
            _validate_manifest_identity(state, study)
            if status in {"study_complete", "study_sample_complete"}:
                latest_markers[study.identity] = state
                continue

            capture_key = _required_string(record, "capture_key", state.location)
            capture_resource_id = _required_string(
                record, "capture_resource_id", state.location
            )
            expected_capture_key = capture_key_for_resource(capture_resource_id)
            if capture_key != expected_capture_key:
                raise AssemblyError(
                    f"{state.location}: capture_key does not match capture_resource_id "
                    f"(expected {expected_capture_key})"
                )
            collision_key = (identity, capture_key)
            previous_resource = capture_resources_by_key.get(collision_key)
            if previous_resource is not None and previous_resource != capture_resource_id:
                raise AssemblyError(f"{state.location}: colliding capture_key in study")
            capture_resources_by_key[collision_key] = capture_resource_id
            latest_captures[identity][capture_key] = state

    verified: list[VerifiedCapture] = []
    for study in selection:
        identity = study.identity
        marker = latest_markers.get(identity)
        if marker is None:
            raise AssemblyError(
                f"Selected study {study.study_key} has no completion record"
            )
        if marker.status != "study_complete":
            raise AssemblyError(
                f"Selected study {study.study_key} has a latest sample completion record"
            )
        states = latest_captures.get(identity, {})
        incomplete = [key for key, state in states.items() if state.status != "downloaded"]
        if incomplete:
            raise AssemblyError(
                f"Selected study {study.study_key} has {len(incomplete)} latest failed "
                "capture record(s)"
            )
        capture_count = len(states)
        videos_available = _required_count(
            marker.record, "videos_available", marker.location
        )
        videos_selected = _required_count(
            marker.record, "videos_selected", marker.location
        )
        if videos_available == 0:
            raise AssemblyError(
                f"{marker.location}: zero-capture completion is ambiguous and cannot "
                "be certified"
            )
        if videos_available != capture_count or videos_selected != capture_count:
            raise AssemblyError(
                f"{marker.location}: completion counts ({videos_available} available, "
                f"{videos_selected} selected) do not match {capture_count} capture rows"
            )
        if states and marker.sequence <= max(state.sequence for state in states.values()):
            raise AssemblyError(
                f"{marker.location}: completion record is older than a latest capture row"
            )

        for capture_key, state in sorted(states.items()):
            record = state.record
            candidate, source_name = _safe_source_path(state)
            expected_size = _required_count(record, "size_bytes", state.location)
            if expected_size == 0:
                raise AssemblyError(
                    f"{state.location}: size_bytes must be positive for a capture"
                )
            expected_sha = _required_string(record, "sha256", state.location)
            if not SHA256_RE.fullmatch(expected_sha):
                raise AssemblyError(f"{state.location}: sha256 is not lowercase SHA-256")
            if not candidate.is_file():
                raise AssemblyError(f"{state.location}: capture file is missing: {candidate}")
            before_hash = candidate.stat()
            actual_size = before_hash.st_size
            if actual_size != expected_size:
                raise AssemblyError(
                    f"{state.location}: capture size mismatch for {candidate} "
                    f"(expected {expected_size}, got {actual_size})"
                )
            actual_sha = sha256_file(candidate)
            after_hash = candidate.stat()
            if file_signature(before_hash) != file_signature(after_hash):
                raise AssemblyError(
                    f"{state.location}: capture changed while it was being verified: "
                    f"{candidate}"
                )
            if actual_sha != expected_sha:
                raise AssemblyError(
                    f"{state.location}: capture SHA-256 mismatch for {candidate}"
                )
            verified.append(
                VerifiedCapture(
                    study=study,
                    capture_key=capture_key,
                    capture_resource_id=_required_string(
                        record, "capture_resource_id", state.location
                    ),
                    source_path=candidate,
                    source_name=source_name,
                    size_bytes=expected_size,
                    sha256=expected_sha,
                    source_signature=file_signature(after_hash),
                )
            )
    return verified


def safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if SAFE_EXTENSION_RE.fullmatch(suffix) else ".bin"


def _private_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    os.chmod(path, PRIVATE_DIRECTORY_MODE)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        os.chmod(path, PRIVATE_FILE_MODE)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        os.chmod(path, PRIVATE_FILE_MODE)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _readme_text(summary: dict) -> str:
    return f"""---
pretty_name: FASH ultrasound captures
---

# FASH ultrasound captures

This private dataset contains {summary['capture_count']} verified ultrasound
captures from {summary['study_count']} studies and {summary['patient_count']}
pseudonymous patients. It is intended for approved research use only.

Files are stored as `data/<patient_id>/<study_key>/<capture_key>.<ext>`.
`metadata/captures.jsonl` contains one row per capture and
`metadata/studies.jsonl` contains one row per selected study. Source titles,
URLs, raw resource identifiers, source filenames, and acquisition timestamps
are deliberately excluded.

Every capture was checked against the byte size and SHA-256 recorded by the
downloader. Duplicate-group fields identify repeated resource identifiers and
identical file content; duplicate captures have not been silently discarded.
Keep the source manifests separate because they contain sensitive source
metadata.
"""


def _write_readme(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        os.chmod(path, PRIVATE_FILE_MODE)
        handle.write(_readme_text(summary))


def _duplicate_info(captures: Sequence[VerifiedCapture], attribute: str) -> dict[str, dict]:
    grouped: dict[str, list[VerifiedCapture]] = defaultdict(list)
    for capture in captures:
        grouped[str(getattr(capture, attribute))].append(capture)
    result: dict[str, dict] = {}
    for value, members in grouped.items():
        if len(members) < 2:
            continue
        studies = {member.study.study_key for member in members}
        patients = {member.study.patient_id for member in members}
        result[value] = {
            "capture_count": len(members),
            "content_hash_count": len({member.sha256 for member in members}),
            "study_count": len(studies),
            "patient_count": len(patients),
            "across_studies": len(studies) > 1,
            "across_patients": len(patients) > 1,
        }
    return result


def _group_id(kind: str, value: str) -> str:
    # Values never expose a raw Butterfly identifier. Content group IDs use a
    # hash that is already present in the permitted sha256 field.
    digest = value if kind == "content" else sha256_text(value)
    return f"{kind}-{digest}"


def _stage_capture(source: Path, destination: Path) -> str:
    """Hardlink a private source when possible, otherwise make a private copy."""
    source_mode = stat.S_IMODE(source.stat().st_mode)
    if source_mode & 0o077:
        shutil.copyfile(source, destination)
        os.chmod(destination, PRIVATE_FILE_MODE)
        return "copied_for_permissions"
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError as exc:
        if exc.errno not in COPY_FALLBACK_ERRNOS:
            raise
    shutil.copyfile(source, destination)
    os.chmod(destination, PRIVATE_FILE_MODE)
    return "copied"


def _build_tree(
    root: Path,
    selection: Sequence[SelectedStudy],
    captures: Sequence[VerifiedCapture],
    selection_sha256: str,
) -> dict:
    _private_mkdir(root)
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    _private_mkdir(data_dir)
    _private_mkdir(metadata_dir)

    resource_duplicates = _duplicate_info(captures, "capture_resource_id")
    content_duplicates = _duplicate_info(captures, "sha256")
    capture_rows: list[dict] = []
    captures_by_study: dict[str, list[dict]] = defaultdict(list)
    transfer_counts: Counter[str] = Counter()

    sorted_captures = sorted(
        captures,
        key=lambda item: (
            item.study.patient_id,
            item.study.archive,
            item.study.study_key,
            item.capture_key,
        ),
    )
    for capture in sorted_captures:
        if file_signature(capture.source_path.stat()) != capture.source_signature:
            raise AssemblyError(
                f"Verified source changed before staging: {capture.source_path}"
            )
        extension = safe_extension(capture.source_name)
        relative = (
            Path("data")
            / capture.study.patient_id
            / capture.study.study_key
            / f"{capture.capture_key}{extension}"
        )
        destination = root / relative
        _private_mkdir(destination.parent)
        transfer_counts[_stage_capture(capture.source_path, destination)] += 1
        if destination.stat().st_size != capture.size_bytes:
            raise AssemblyError(f"Staged capture has wrong size: {destination}")
        if os.path.samefile(capture.source_path, destination):
            if file_signature(destination.stat()) != capture.source_signature:
                raise AssemblyError(f"Hardlinked capture changed while staging: {destination}")
        else:
            if sha256_file(destination) != capture.sha256:
                raise AssemblyError(f"Staged capture has wrong SHA-256: {destination}")

        resource_info = resource_duplicates.get(capture.capture_resource_id)
        content_info = content_duplicates.get(capture.sha256)
        row = {
            "archive": capture.study.archive,
            "capture_key": capture.capture_key,
            "content_duplicate_group": (
                _group_id("content", capture.sha256) if content_info else None
            ),
            "content_duplicate_capture_count": (
                content_info["capture_count"] if content_info else 1
            ),
            "content_duplicate_patient_count": (
                content_info["patient_count"] if content_info else 1
            ),
            "content_duplicate_study_count": (
                content_info["study_count"] if content_info else 1
            ),
            "path": relative.as_posix(),
            "patient_id": capture.study.patient_id,
            "resource_duplicate_group": (
                _group_id("resource", capture.capture_resource_id)
                if resource_info
                else None
            ),
            "resource_duplicate_capture_count": (
                resource_info["capture_count"] if resource_info else 1
            ),
            "resource_duplicate_content_hash_count": (
                resource_info["content_hash_count"] if resource_info else 1
            ),
            "resource_duplicate_patient_count": (
                resource_info["patient_count"] if resource_info else 1
            ),
            "resource_duplicate_study_count": (
                resource_info["study_count"] if resource_info else 1
            ),
            "sha256": capture.sha256,
            "size_bytes": capture.size_bytes,
            "study_key": capture.study.study_key,
        }
        capture_rows.append(row)
        captures_by_study[capture.study.study_key].append(row)

    study_rows: list[dict] = []
    for study in sorted(
        selection, key=lambda item: (item.patient_id, item.archive, item.study_key)
    ):
        rows = captures_by_study.get(study.study_key, [])
        study_rows.append(
            {
                "archive": study.archive,
                "capture_count": len(rows),
                "content_duplicate_groups": sorted(
                    {
                        row["content_duplicate_group"]
                        for row in rows
                        if row["content_duplicate_group"] is not None
                    }
                ),
                "patient_id": study.patient_id,
                "resource_duplicate_groups": sorted(
                    {
                        row["resource_duplicate_group"]
                        for row in rows
                        if row["resource_duplicate_group"] is not None
                    }
                ),
                "size_bytes": sum(row["size_bytes"] for row in rows),
                "study_key": study.study_key,
            }
        )

    by_archive: dict[str, dict] = {}
    for archive in sorted({study.archive for study in selection}):
        archive_studies = [study for study in selection if study.archive == archive]
        archive_captures = [row for row in capture_rows if row["archive"] == archive]
        by_archive[archive] = {
            "capture_count": len(archive_captures),
            "patient_count": len({study.patient_id for study in archive_studies}),
            "size_bytes": sum(row["size_bytes"] for row in archive_captures),
            "study_count": len(archive_studies),
        }

    summary = {
        "archive_counts": by_archive,
        "capture_count": len(capture_rows),
        "content_duplicate_capture_count": sum(
            info["capture_count"] for info in content_duplicates.values()
        ),
        "content_duplicate_group_count": len(content_duplicates),
        "content_duplicate_groups_across_patients": sum(
            info["across_patients"] for info in content_duplicates.values()
        ),
        "content_duplicate_groups_across_studies": sum(
            info["across_studies"] for info in content_duplicates.values()
        ),
        "content_redundant_capture_count": sum(
            info["capture_count"] - 1 for info in content_duplicates.values()
        ),
        "format_version": FORMAT_VERSION,
        "patient_count": len({study.patient_id for study in selection}),
        "resource_duplicate_capture_count": sum(
            info["capture_count"] for info in resource_duplicates.values()
        ),
        "resource_duplicate_group_count": len(resource_duplicates),
        "resource_duplicate_groups_across_patients": sum(
            info["across_patients"] for info in resource_duplicates.values()
        ),
        "resource_duplicate_groups_across_studies": sum(
            info["across_studies"] for info in resource_duplicates.values()
        ),
        "resource_duplicate_groups_with_content_conflicts": sum(
            info["content_hash_count"] > 1 for info in resource_duplicates.values()
        ),
        "resource_redundant_capture_count": sum(
            info["capture_count"] - 1 for info in resource_duplicates.values()
        ),
        "selection_sha256": selection_sha256,
        "size_bytes": sum(row["size_bytes"] for row in capture_rows),
        "staging_transfer_counts": dict(sorted(transfer_counts.items())),
        "study_count": len(selection),
        "unique_content_count": len({row["sha256"] for row in capture_rows}),
        "unique_resource_count": len(
            {capture.capture_resource_id for capture in captures}
        ),
    }
    _write_jsonl(metadata_dir / "captures.jsonl", capture_rows)
    _write_jsonl(metadata_dir / "studies.jsonl", study_rows)
    _write_json(root / "summary.json", summary)
    _write_readme(root / "README.md", summary)
    return summary


def _require_exact_fields(record: dict, expected: frozenset[str], location: str) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssemblyError(
            f"{location}: metadata fields do not match the private-dataset schema "
            f"(missing={missing}, extra={extra})"
        )


def _strictly_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strictly_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strictly_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _read_staging_jsonl(path: Path, label: str) -> list[tuple[int, dict]]:
    return list(
        _jsonl_records(path, label, _read_private_file_snapshot(path, label))
    )


def _read_staging_json(path: Path, label: str) -> dict:
    payload = _read_private_file_snapshot(path, label)
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"{path}: invalid JSON in {label}") from exc
    if not isinstance(record, dict):
        raise AssemblyError(f"{path}: {label} must be a JSON object")
    return record


def _scan_staging_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    try:
        candidates = list(root.rglob("*"))
    except OSError as exc:
        raise AssemblyError(f"Cannot enumerate staging tree {root}: {exc}") from exc
    for candidate in candidates:
        relative = candidate.relative_to(root).as_posix()
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise AssemblyError(f"Cannot inspect staged path {candidate}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise AssemblyError(f"Staging tree must not contain symlinks: {candidate}")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AssemblyError(
                f"Staged paths must not be accessible by group or other users: {candidate}"
            )
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(info.st_mode):
            files.add(relative)
        else:
            raise AssemblyError(f"Staging tree contains a non-file entry: {candidate}")
    return files, directories


def _expected_staging_directories(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for filename in files:
        parent = PurePosixPath(filename).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _capture_duplicate_expectations(
    rows: Sequence[dict], group_field: str
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group = row[group_field]
        if group is not None:
            grouped[group].append(row)
    result: dict[str, dict[str, int]] = {}
    for group, members in grouped.items():
        if len(members) < 2:
            raise AssemblyError(f"Duplicate group {group!r} contains fewer than two captures")
        result[group] = {
            "capture_count": len(members),
            "content_hash_count": len({member["sha256"] for member in members}),
            "patient_count": len({member["patient_id"] for member in members}),
            "study_count": len({member["study_key"] for member in members}),
        }
    return result


def verify_staging_tree(
    root_path: Path, expected_selection_sha256: str | None = None
) -> dict:
    """Fail closed unless a staging tree exactly matches its minimized metadata."""
    lexical_root = Path(os.path.abspath(root_path.expanduser()))
    try:
        root_info = lexical_root.lstat()
    except OSError as exc:
        raise AssemblyError(f"Cannot inspect staging directory {lexical_root}: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise AssemblyError(f"Staging path must be a real directory: {lexical_root}")
    if stat.S_IMODE(root_info.st_mode) & 0o077:
        raise AssemblyError(
            f"Staging directory must be private (mode 0700 or stricter): {lexical_root}"
        )
    root = lexical_root.resolve()
    actual_files, actual_directories = _scan_staging_tree(root)

    mandatory_files = {
        "README.md",
        "metadata/captures.jsonl",
        "metadata/studies.jsonl",
        "summary.json",
    }
    missing_mandatory = mandatory_files - actual_files
    if missing_mandatory:
        raise AssemblyError(
            f"Staging tree is missing required files: {sorted(missing_mandatory)}"
        )

    capture_records = _read_staging_jsonl(
        root / "metadata" / "captures.jsonl", "capture metadata"
    )
    if not capture_records:
        raise AssemblyError("Capture metadata contains no captures")
    captures: list[dict] = []
    capture_paths: set[str] = set()
    for line_number, row in capture_records:
        location = f"{root / 'metadata' / 'captures.jsonl'}:{line_number}"
        _require_exact_fields(row, CAPTURE_METADATA_FIELDS, location)
        archive = _required_string(row, "archive", location)
        patient_id = _required_string(row, "patient_id", location)
        study_key = _required_string(row, "study_key", location)
        capture_key = _required_string(row, "capture_key", location)
        relative_text = _required_string(row, "path", location)
        expected_sha = _required_string(row, "sha256", location)
        expected_size = _required_count(row, "size_bytes", location)
        if not SAFE_COMPONENT_RE.fullmatch(patient_id):
            raise AssemblyError(f"{location}: patient_id is not a safe path component")
        if not STUDY_KEY_RE.fullmatch(study_key):
            raise AssemblyError(f"{location}: study_key is not a derived study key")
        if not CAPTURE_KEY_RE.fullmatch(capture_key):
            raise AssemblyError(f"{location}: capture_key is not a derived capture key")
        if not SHA256_RE.fullmatch(expected_sha):
            raise AssemblyError(f"{location}: sha256 is not lowercase SHA-256")
        if expected_size == 0:
            raise AssemblyError(f"{location}: size_bytes must be positive for a capture")
        relative = PurePosixPath(relative_text)
        suffix = relative.suffix.lower()
        expected_path = PurePosixPath(
            "data", patient_id, study_key, f"{capture_key}{suffix}"
        )
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or not SAFE_EXTENSION_RE.fullmatch(suffix)
            or relative != expected_path
        ):
            raise AssemblyError(f"{location}: capture path is not canonical")
        if relative_text in capture_paths:
            raise AssemblyError(f"{location}: duplicate capture path")
        capture_paths.add(relative_text)

        for field in (
            "content_duplicate_capture_count",
            "content_duplicate_patient_count",
            "content_duplicate_study_count",
            "resource_duplicate_capture_count",
            "resource_duplicate_content_hash_count",
            "resource_duplicate_patient_count",
            "resource_duplicate_study_count",
        ):
            if _required_count(row, field, location) < 1:
                raise AssemblyError(f"{location}: {field} must be positive")
        for field, prefix in (
            ("content_duplicate_group", "content-"),
            ("resource_duplicate_group", "resource-"),
        ):
            value = row[field]
            if value is not None and (
                not isinstance(value, str)
                or not value.startswith(prefix)
                or not SHA256_RE.fullmatch(value[len(prefix) :])
            ):
                raise AssemblyError(f"{location}: {field} is not a derived group ID")

        staged = root / Path(*relative.parts)
        try:
            before_hash = staged.stat()
        except OSError as exc:
            raise AssemblyError(f"{location}: cannot inspect staged capture {staged}") from exc
        if before_hash.st_size != expected_size:
            raise AssemblyError(f"{location}: staged capture size does not match metadata")
        actual_sha = sha256_file(staged)
        after_hash = staged.stat()
        if file_signature(before_hash) != file_signature(after_hash):
            raise AssemblyError(f"{location}: staged capture changed while being verified")
        if actual_sha != expected_sha:
            raise AssemblyError(f"{location}: staged capture SHA-256 does not match metadata")
        captures.append(row)

    content_by_sha: dict[str, list[dict]] = defaultdict(list)
    for row in captures:
        content_by_sha[row["sha256"]].append(row)
    for row in captures:
        members = content_by_sha[row["sha256"]]
        duplicate = len(members) > 1
        expected_group = f"content-{row['sha256']}" if duplicate else None
        expected_values = {
            "content_duplicate_group": expected_group,
            "content_duplicate_capture_count": len(members),
            "content_duplicate_patient_count": len(
                {member["patient_id"] for member in members}
            ),
            "content_duplicate_study_count": len(
                {member["study_key"] for member in members}
            ),
        }
        for field, expected in expected_values.items():
            if row[field] != expected:
                raise AssemblyError(f"Capture metadata has inconsistent {field}")

    resource_groups = _capture_duplicate_expectations(
        captures, "resource_duplicate_group"
    )
    for row in captures:
        group = row["resource_duplicate_group"]
        if group is None:
            expected = {
                "resource_duplicate_capture_count": 1,
                "resource_duplicate_content_hash_count": 1,
                "resource_duplicate_patient_count": 1,
                "resource_duplicate_study_count": 1,
            }
        else:
            info = resource_groups[group]
            expected = {
                "resource_duplicate_capture_count": info["capture_count"],
                "resource_duplicate_content_hash_count": info["content_hash_count"],
                "resource_duplicate_patient_count": info["patient_count"],
                "resource_duplicate_study_count": info["study_count"],
            }
        for field, value in expected.items():
            if row[field] != value:
                raise AssemblyError(f"Capture metadata has inconsistent {field}")

    study_records = _read_staging_jsonl(
        root / "metadata" / "studies.jsonl", "study metadata"
    )
    studies: dict[str, dict] = {}
    captures_by_study: dict[str, list[dict]] = defaultdict(list)
    for capture in captures:
        captures_by_study[capture["study_key"]].append(capture)
    for line_number, row in study_records:
        location = f"{root / 'metadata' / 'studies.jsonl'}:{line_number}"
        _require_exact_fields(row, STUDY_METADATA_FIELDS, location)
        study_key = _required_string(row, "study_key", location)
        archive = _required_string(row, "archive", location)
        patient_id = _required_string(row, "patient_id", location)
        if not STUDY_KEY_RE.fullmatch(study_key) or study_key in studies:
            raise AssemblyError(f"{location}: invalid or duplicate study_key")
        members = captures_by_study.get(study_key, [])
        if not members:
            raise AssemblyError(f"{location}: study has no staged captures")
        if {member["archive"] for member in members} != {archive}:
            raise AssemblyError(f"{location}: study archive disagrees with captures")
        if {member["patient_id"] for member in members} != {patient_id}:
            raise AssemblyError(f"{location}: study patient_id disagrees with captures")
        expected_study = {
            "capture_count": len(members),
            "content_duplicate_groups": sorted(
                {
                    member["content_duplicate_group"]
                    for member in members
                    if member["content_duplicate_group"] is not None
                }
            ),
            "resource_duplicate_groups": sorted(
                {
                    member["resource_duplicate_group"]
                    for member in members
                    if member["resource_duplicate_group"] is not None
                }
            ),
            "size_bytes": sum(member["size_bytes"] for member in members),
        }
        for field, expected in expected_study.items():
            if not _strictly_equal(row[field], expected):
                raise AssemblyError(f"{location}: inconsistent {field}")
        studies[study_key] = row
    if set(studies) != set(captures_by_study):
        raise AssemblyError("Study metadata does not cover the exact capture study set")

    summary_path = root / "summary.json"
    summary = _read_staging_json(summary_path, "dataset summary")
    _require_exact_fields(summary, SUMMARY_FIELDS, str(summary_path))
    selection_sha256 = _required_string(summary, "selection_sha256", str(summary_path))
    if not SHA256_RE.fullmatch(selection_sha256):
        raise AssemblyError("Dataset summary has an invalid selection SHA-256")
    if expected_selection_sha256 is not None:
        if not SHA256_RE.fullmatch(expected_selection_sha256):
            raise AssemblyError("Expected selection SHA-256 must be 64 lowercase hex digits")
        if selection_sha256 != expected_selection_sha256:
            raise AssemblyError(
                "Staging selection SHA-256 does not match the audited baseline"
            )

    content_groups = {
        f"content-{digest}": members
        for digest, members in content_by_sha.items()
        if len(members) > 1
    }
    archive_counts: dict[str, dict] = {}
    for archive in sorted({study["archive"] for study in studies.values()}):
        archive_studies = [
            study for study in studies.values() if study["archive"] == archive
        ]
        archive_captures = [row for row in captures if row["archive"] == archive]
        archive_counts[archive] = {
            "capture_count": len(archive_captures),
            "patient_count": len({study["patient_id"] for study in archive_studies}),
            "size_bytes": sum(row["size_bytes"] for row in archive_captures),
            "study_count": len(archive_studies),
        }
    expected_summary = {
        "archive_counts": archive_counts,
        "capture_count": len(captures),
        "content_duplicate_capture_count": sum(
            len(members) for members in content_groups.values()
        ),
        "content_duplicate_group_count": len(content_groups),
        "content_duplicate_groups_across_patients": sum(
            len({member["patient_id"] for member in members}) > 1
            for members in content_groups.values()
        ),
        "content_duplicate_groups_across_studies": sum(
            len({member["study_key"] for member in members}) > 1
            for members in content_groups.values()
        ),
        "content_redundant_capture_count": sum(
            len(members) - 1 for members in content_groups.values()
        ),
        "format_version": FORMAT_VERSION,
        "patient_count": len({study["patient_id"] for study in studies.values()}),
        "resource_duplicate_capture_count": sum(
            info["capture_count"] for info in resource_groups.values()
        ),
        "resource_duplicate_group_count": len(resource_groups),
        "resource_duplicate_groups_across_patients": sum(
            info["patient_count"] > 1 for info in resource_groups.values()
        ),
        "resource_duplicate_groups_across_studies": sum(
            info["study_count"] > 1 for info in resource_groups.values()
        ),
        "resource_duplicate_groups_with_content_conflicts": sum(
            info["content_hash_count"] > 1 for info in resource_groups.values()
        ),
        "resource_redundant_capture_count": sum(
            info["capture_count"] - 1 for info in resource_groups.values()
        ),
        "selection_sha256": selection_sha256,
        "size_bytes": sum(row["size_bytes"] for row in captures),
        "study_count": len(studies),
        "unique_content_count": len(content_by_sha),
        "unique_resource_count": sum(
            row["resource_duplicate_group"] is None for row in captures
        )
        + len(resource_groups),
    }
    for field, expected in expected_summary.items():
        if not _strictly_equal(summary[field], expected):
            raise AssemblyError(f"Dataset summary has inconsistent {field}")
    transfer_counts = summary["staging_transfer_counts"]
    if (
        not isinstance(transfer_counts, dict)
        or not transfer_counts
        or not set(transfer_counts) <= {
            "copied",
            "copied_for_permissions",
            "hardlinked",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in transfer_counts.values()
        )
        or sum(transfer_counts.values()) != len(captures)
    ):
        raise AssemblyError("Dataset summary has invalid staging_transfer_counts")

    readme = _read_private_file_snapshot(root / "README.md", "dataset README")
    if readme != _readme_text(summary).encode("utf-8"):
        raise AssemblyError("Dataset README does not match the verified summary")

    expected_files = mandatory_files | capture_paths
    if actual_files != expected_files:
        raise AssemblyError(
            "Staging file set does not match metadata "
            f"(missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)})"
        )
    expected_directories = _expected_staging_directories(expected_files)
    if actual_directories != expected_directories:
        raise AssemblyError(
            "Staging directory set does not match metadata "
            f"(missing={sorted(expected_directories - actual_directories)}, "
            f"extra={sorted(actual_directories - expected_directories)})"
        )
    return summary


def _output_lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _validate_output_target(
    output: Path,
    selection_path: Path,
    manifest_paths: Sequence[Path],
    replace_existing: bool,
) -> os.stat_result | None:
    forbidden_roots = {Path(output.anchor), Path.home().resolve(), Path.cwd().resolve()}
    if output in forbidden_roots or len(output.parts) < 3:
        raise AssemblyError(f"Refusing unsafe output directory: {output}")
    for protected_input in [selection_path, *manifest_paths]:
        resolved_input = protected_input.expanduser().resolve()
        try:
            resolved_input.relative_to(output)
        except ValueError:
            pass
        else:
            raise AssemblyError(
                f"Output directory must not contain a protected input: {resolved_input}"
            )
    initial = _output_lstat(output)
    if initial is None:
        return None
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISDIR(initial.st_mode):
        raise AssemblyError(f"Output path must be a real directory: {output}")
    if any(output.iterdir()) and not replace_existing:
        raise AssemblyError(
            f"Output directory is nonempty; pass --replace-existing explicitly: {output}"
        )
    return initial


def _same_object(path: Path, expected: os.stat_result | None) -> bool:
    current = _output_lstat(path)
    if current is None or expected is None:
        return current is None and expected is None
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


def _remove_generated_tree(path: Path, parent: Path, prefix: str) -> None:
    # Deletion is limited to a UUID-named sibling created by this process.
    if path.parent != parent or not path.name.startswith(prefix) or path.is_symlink():
        raise AssemblyError(f"Refusing unsafe generated-tree removal: {path}")
    if path.exists():
        shutil.rmtree(path)


def assemble_dataset(
    selection_path: Path,
    manifest_paths: Sequence[Path],
    output_dir: Path,
    replace_existing: bool = False,
    expected_selection_sha256: str | None = None,
) -> dict:
    """Validate inputs and atomically install a privacy-minimized staging tree."""
    selection_path = selection_path.expanduser().resolve()
    manifests = [path.expanduser().resolve() for path in manifest_paths]
    lexical_output = Path(os.path.abspath(output_dir.expanduser()))
    lexical_info = _output_lstat(lexical_output)
    if lexical_info is not None and stat.S_ISLNK(lexical_info.st_mode):
        raise AssemblyError(f"Output path must not be a symbolic link: {lexical_output}")
    output = lexical_output.resolve()
    initial_output = _validate_output_target(
        output, selection_path, manifests, replace_existing
    )
    selection_snapshot = _read_private_file_snapshot(selection_path, "selection file")
    selection_sha256 = hashlib.sha256(selection_snapshot).hexdigest()
    if expected_selection_sha256 is not None:
        if not SHA256_RE.fullmatch(expected_selection_sha256):
            raise AssemblyError("Expected selection SHA-256 must be 64 lowercase hex digits")
        if selection_sha256 != expected_selection_sha256:
            raise AssemblyError(
                "Protected selection SHA-256 does not match the audited baseline"
            )
    selection = read_selection(selection_path, snapshot=selection_snapshot)
    captures = load_and_verify_captures(selection, manifests)

    output.parent.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    backup = output.parent / f".{output.name}.replaced-{uuid.uuid4().hex}"
    old_umask = os.umask(0o077)
    installed = False
    try:
        summary = _build_tree(temporary, selection, captures, selection_sha256)
        if not _same_object(output, initial_output):
            raise AssemblyError(f"Output directory changed while assembling: {output}")
        if initial_output is not None:
            os.replace(output, backup)
            try:
                os.replace(temporary, output)
                installed = True
            except BaseException:
                os.replace(backup, output)
                raise
            _remove_generated_tree(backup, output.parent, f".{output.name}.replaced-")
        else:
            os.replace(temporary, output)
            installed = True
        os.chmod(output, PRIVATE_DIRECTORY_MODE)
        return summary
    finally:
        os.umask(old_umask)
        if not installed and temporary.exists():
            _remove_generated_tree(
                temporary, output.parent, f".{output.name}.building-"
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--selection", type=Path, help="Protected selection JSONL"
    )
    result.add_argument(
        "--expected-selection-sha256",
        required=True,
        help="Audited SHA-256 of the exact protected selection JSONL",
    )
    result.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help=(
            "Protected download-manifest.jsonl (repeat for each raw root). "
            "Later arguments/rows take precedence."
        ),
    )
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--verify-staging",
        type=Path,
        help="Verify an existing staging tree without assembling it",
    )
    result.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace an existing nonempty output after a complete staged build",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.verify_staging is not None:
            if any(
                value is not None
                for value in (args.selection, args.manifest, args.output)
            ) or args.replace_existing:
                argument_parser.error(
                    "--verify-staging cannot be combined with assembly arguments"
                )
            summary = verify_staging_tree(
                args.verify_staging,
                expected_selection_sha256=args.expected_selection_sha256,
            )
            print("staging_verified=true")
            print(f"studies={summary['study_count']}")
            print(f"patients={summary['patient_count']}")
            print(f"captures={summary['capture_count']}")
            print(f"bytes={summary['size_bytes']}")
            print(f"output={args.verify_staging.expanduser().resolve()}")
            return 0
        missing = [
            flag
            for flag, value in (
                ("--selection", args.selection),
                ("--manifest", args.manifest),
                ("--output", args.output),
            )
            if value is None
        ]
        if missing:
            argument_parser.error(
                f"assembly requires {', '.join(missing)}"
            )
        summary = assemble_dataset(
            args.selection,
            args.manifest,
            args.output,
            replace_existing=args.replace_existing,
            expected_selection_sha256=args.expected_selection_sha256,
        )
    except AssemblyError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"studies={summary['study_count']}")
    print(f"patients={summary['patient_count']}")
    print(f"captures={summary['capture_count']}")
    print(f"bytes={summary['size_bytes']}")
    print(f"output={args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
