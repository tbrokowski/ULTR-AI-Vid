#!/usr/bin/env python3
"""Build a protected, deduplicated Butterfly download plan for the Benin cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse


DEFAULT_PRIMARY_ARCHIVE = "TrUST Bénin"
DEFAULT_SUPPLEMENT_ARCHIVE = "Contact study Bénin"


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def patient_id_from_title(title: str) -> str:
    match = re.search(r"(?<!\d)(\d{2}-\d+)(?!\d)", title)
    return match.group(1) if match else ""


def resource_id_from_url(value: str) -> str:
    path = unquote(urlparse(value).path).rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""


def parse_inventory_argument(value: str) -> tuple[str, Path]:
    try:
        archive, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use ARCHIVE=/path/to/inventory.jsonl") from exc
    archive = archive.strip()
    if not archive or not raw_path.strip():
        raise argparse.ArgumentTypeError("archive and inventory path must both be non-empty")
    return archive, Path(raw_path).expanduser().resolve()


def parse_expected_inventory_count(value: str) -> tuple[str, int]:
    try:
        archive, raw_count = value.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use ARCHIVE=N") from exc
    archive = archive.strip()
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("inventory count must be an integer") from exc
    if not archive or count < 0:
        raise argparse.ArgumentTypeError(
            "archive must be non-empty and inventory count must be non-negative"
        )
    return archive, count


def read_roster(path: Path, column: str) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"Roster {path} has no {column!r} column")
        roster = {str(row.get(column, "")).strip() for row in reader}
    roster.discard("")
    if not roster:
        raise ValueError(f"Roster {path} contains no IDs")
    return roster


def read_inventory_with_count(archive: str, path: Path) -> tuple[list[dict], int]:
    """Return matched selection records plus the complete nonblank row count."""
    records: list[dict] = []
    inventory_count = 0
    seen_resource_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Inventory line {line_number} of {path} is not an object")
        title = str(record.get("title", ""))
        url = str(record.get("url", ""))
        resource_id = str(record.get("study_resource_id", "")) or resource_id_from_url(url)
        if not resource_id:
            raise ValueError(f"Inventory line {line_number} of {path} has no resource ID")
        if resource_id in seen_resource_ids:
            raise ValueError(
                f"Inventory line {line_number} of {path} duplicates resource ID"
            )
        seen_resource_ids.add(resource_id)
        inventory_count += 1
        if not record.get("matched"):
            continue
        records.append(
            {
                "archive": archive,
                "study_resource_id": resource_id,
                "study_key": hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:12],
                "patient_id": str(record.get("patient_id", ""))
                or patient_id_from_title(title),
                "title_fingerprint": hashlib.sha256(
                    normalize_text(title).encode("utf-8")
                ).hexdigest(),
                "page_number": int(record.get("page_number", 1)),
                "row_index": int(record.get("row_index", 0)),
            }
        )
    return records, inventory_count


def read_inventory(archive: str, path: Path) -> list[dict]:
    records, _inventory_count = read_inventory_with_count(archive, path)
    return records


def validate_inventory_counts(
    actual: Mapping[str, int], expected: Mapping[str, int]
) -> None:
    """Require one audited count for every supplied inventory and no extras."""
    missing = sorted(set(actual) - set(expected))
    extra = sorted(set(expected) - set(actual))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                "missing expected counts for " + ", ".join(repr(x) for x in missing)
            )
        if extra:
            details.append(
                "expected counts without inventories for "
                + ", ".join(repr(x) for x in extra)
            )
        raise ValueError("Inventory count coverage mismatch: " + "; ".join(details))
    mismatches = [
        (archive, actual[archive], expected[archive])
        for archive in sorted(actual)
        if actual[archive] != expected[archive]
    ]
    if mismatches:
        formatted = "; ".join(
            f"{archive!r}: found {found}, expected {wanted}"
            for archive, found, wanted in mismatches
        )
        raise ValueError("Complete inventory count mismatch: " + formatted)


def build_selection(
    inventory_records: Iterable[dict],
    roster: set[str],
    primary_archive: str = DEFAULT_PRIMARY_ARCHIVE,
    supplement_archive: str = DEFAULT_SUPPLEMENT_ARCHIVE,
    inventory_counts: Mapping[str, int] | None = None,
) -> tuple[list[dict], dict]:
    records = list(inventory_records)
    raw_by_archive = Counter(record["archive"] for record in records)
    roster_records = [record for record in records if record["patient_id"] in roster]
    roster_by_archive = Counter(record["archive"] for record in roster_records)

    patient_archives: dict[str, set[str]] = defaultdict(set)
    for record in roster_records:
        patient_archives[record["patient_id"]].add(record["archive"])

    # TrUST is the model cohort. Contact is used only for roster members absent
    # from TrUST, avoiding a second copy of shared patient directories.
    primary_patients = {
        record["patient_id"]
        for record in roster_records
        if normalize_text(record["archive"]) == normalize_text(primary_archive)
    }
    eligible: list[dict] = []
    for record in roster_records:
        archive = normalize_text(record["archive"])
        if archive == normalize_text(primary_archive):
            eligible.append(record)
        elif (
            archive == normalize_text(supplement_archive)
            and record["patient_id"] not in primary_patients
        ):
            eligible.append(record)

    archive_priority = {
        normalize_text(primary_archive): 0,
        normalize_text(supplement_archive): 1,
    }
    eligible.sort(
        key=lambda record: (
            archive_priority.get(normalize_text(record["archive"]), 99),
            record["page_number"],
            record["row_index"],
            record["study_resource_id"],
        )
    )
    selected_by_resource: dict[str, dict] = {}
    for record in eligible:
        selected_by_resource.setdefault(record["study_resource_id"], record)
    selected = list(selected_by_resource.values())

    selected_patient_ids = {record["patient_id"] for record in selected}
    resource_archives: dict[str, set[str]] = defaultdict(set)
    title_archives: dict[str, set[str]] = defaultdict(set)
    for record in roster_records:
        resource_archives[record["study_resource_id"]].add(record["archive"])
        title_archives[record["title_fingerprint"]].add(record["archive"])

    summary = {
        "roster_ids": len(roster),
        "inventory_studies_by_archive": dict(sorted((inventory_counts or {}).items())),
        "raw_matched_studies_by_archive": dict(sorted(raw_by_archive.items())),
        "roster_matched_studies_by_archive": dict(sorted(roster_by_archive.items())),
        "roster_patients_seen_by_archive": {
            archive: len(
                {
                    record["patient_id"]
                    for record in roster_records
                    if record["archive"] == archive
                }
            )
            for archive in sorted(raw_by_archive)
        },
        "patients_seen_in_multiple_archives": sum(
            len(archives) > 1 for archives in patient_archives.values()
        ),
        "study_resources_seen_in_multiple_archives": sum(
            len(archives) > 1 for archives in resource_archives.values()
        ),
        "title_fingerprints_seen_in_multiple_archives": sum(
            len(archives) > 1 for archives in title_archives.values()
        ),
        "selected_studies": len(selected),
        "selected_patients": len(selected_patient_ids),
        "selected_studies_by_archive": dict(
            sorted(Counter(record["archive"] for record in selected).items())
        ),
        "roster_ids_without_selected_study": sorted(roster - selected_patient_ids),
        "selection_policy": {
            "primary_archive": primary_archive,
            "supplement_archive": supplement_archive,
            "supplement_only_when_patient_absent_from_primary": True,
            "deduplicate_by_study_resource_id": True,
        },
    }
    return selected, summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        for record in records:
            public_record = {
                key: record[key]
                for key in (
                    "archive",
                    "study_resource_id",
                    "study_key",
                    "patient_id",
                    "page_number",
                    "row_index",
                )
            }
            handle.write(json.dumps(public_record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--inventory",
        action="append",
        required=True,
        type=parse_inventory_argument,
        metavar="ARCHIVE=PATH",
    )
    result.add_argument(
        "--expected-inventory-count",
        action="append",
        required=True,
        type=parse_expected_inventory_count,
        metavar="ARCHIVE=N",
        help="Audited complete row count; repeat exactly once per --inventory",
    )
    result.add_argument("--roster", required=True, type=Path)
    result.add_argument("--roster-column", default="record_id")
    result.add_argument("--primary-archive", default=DEFAULT_PRIMARY_ARCHIVE)
    result.add_argument("--supplement-archive", default=DEFAULT_SUPPLEMENT_ARCHIVE)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--summary", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    roster = read_roster(args.roster.expanduser().resolve(), args.roster_column)
    records: list[dict] = []
    inventory_counts: dict[str, int] = {}
    for archive, path in args.inventory:
        if archive in inventory_counts:
            raise ValueError(f"Duplicate --inventory archive: {archive!r}")
        matched_records, inventory_count = read_inventory_with_count(archive, path)
        records.extend(matched_records)
        inventory_counts[archive] = inventory_count
    expected_counts: dict[str, int] = {}
    for archive, count in args.expected_inventory_count:
        if archive in expected_counts:
            raise ValueError(f"Duplicate --expected-inventory-count archive: {archive!r}")
        expected_counts[archive] = count
    validate_inventory_counts(inventory_counts, expected_counts)
    selected, summary = build_selection(
        records,
        roster,
        primary_archive=args.primary_archive,
        supplement_archive=args.supplement_archive,
        inventory_counts=inventory_counts,
    )
    write_jsonl(args.output.expanduser().resolve(), selected)
    write_json(args.summary.expanduser().resolve(), summary)
    print(f"selected_studies={summary['selected_studies']}")
    print(f"selected_patients={summary['selected_patients']}")
    print(f"missing_roster_ids={len(summary['roster_ids_without_selected_study'])}")
    print(f"selection_file={args.output.expanduser().resolve()}")
    print(f"summary_file={args.summary.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
