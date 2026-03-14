#!/usr/bin/env python3

"""Utilities shared by domain-shift plotting scripts."""

import csv
from pathlib import Path
from typing import Dict, List, Optional


SITE_CODE_TO_INDEX = {
    "<PAD>": 0,
    "QAID": 1,
    "QAIG": 2,
    "QASD": 3,
    "QASG": 4,
    "QLD": 5,
    "QLG": 6,
    "QPID": 7,
    "QPIG": 8,
    "QPSD": 9,
    "QPSG": 10,
    "APXD": 11,
    "APXG": 12,
    "QSLD": 13,
    "QSLG": 14,
    "SAD": 15,
    "SLD": 16,
    "SAG": 17,
    "SLG": 18,
    "SPD": 19,
    "SPG": 20,
    "UNKNOWN": 21,
}
SITE_INDEX_TO_CODE = {index: code for code, index in SITE_CODE_TO_INDEX.items()}

RIGHT_SITES = {
    "QAID",
    "QASD",
    "QLD",
    "QPID",
    "QPSD",
    "APXD",
    "QSLD",
    "SAD",
    "SLD",
    "SPD",
}
LEFT_SITES = {
    "QAIG",
    "QASG",
    "QLG",
    "QPIG",
    "QPSG",
    "APXG",
    "QSLG",
    "SAG",
    "SLG",
    "SPG",
}
ANTERIOR_SITES = {"QASD", "QASG", "QAID", "QAIG", "SAD", "SAG"}
LATERAL_SITES = {"QLD", "QLG", "QSLD", "QSLG", "SLD", "SLG"}
POSTERIOR_SITES = {"QPSD", "QPSG", "QPID", "QPIG", "SPD", "SPG"}
APICAL_SITES = {"APXD", "APXG"}
SWEEP_SITES = {"SAD", "SLD", "SAG", "SLG", "SPD", "SPG"}


def _unique_preserving_order(values: List[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def patient_id_variants(patient_id: object) -> List[str]:
    pid = str(patient_id).strip()
    if not pid:
        return []

    variants = [pid]
    if pid.isdigit():
        variants.append(pid.zfill(3))

    if "-" in pid:
        suffix = pid.split("-")[-1].strip()
        if suffix:
            variants.append(suffix)
            if suffix.isdigit():
                variants.append(suffix.zfill(3))

    return _unique_preserving_order(variants)


def load_tb_label_lookup(labels_csv: Path) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_pid = row.get("record_id", "")
            raw_label = row.get("TB Label", "")
            if raw_pid is None or raw_label is None:
                continue
            raw_pid = str(raw_pid).strip()
            raw_label = str(raw_label).strip()
            if not raw_pid or raw_label == "":
                continue

            try:
                label = int(float(raw_label))
            except ValueError:
                continue

            for variant in patient_id_variants(raw_pid):
                if variant not in lookup:
                    lookup[variant] = label
    return lookup


def lookup_tb_label(patient_id: object, label_lookup: Dict[str, int]) -> Optional[int]:
    for variant in patient_id_variants(patient_id):
        if variant in label_lookup:
            return label_lookup[variant]
    return None


def site_code_from_index(site_index: object) -> str:
    try:
        index = int(site_index)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return SITE_INDEX_TO_CODE.get(index, "UNKNOWN")


def site_metadata_from_code(site_code: object) -> Dict[str, str]:
    code = str(site_code or "UNKNOWN").strip().upper() or "UNKNOWN"
    if code in RIGHT_SITES:
        laterality = "right"
    elif code in LEFT_SITES:
        laterality = "left"
    else:
        laterality = "unknown"

    if code in ANTERIOR_SITES:
        region = "anterior"
    elif code in LATERAL_SITES:
        region = "lateral"
    elif code in POSTERIOR_SITES:
        region = "posterior"
    elif code in APICAL_SITES:
        region = "apical"
    else:
        region = "unknown"

    if code == "UNKNOWN":
        family = "unknown"
    elif code in SWEEP_SITES:
        family = "sweep"
    elif code == "<PAD>":
        family = "padding"
    else:
        family = "standard"

    return {
        "site_code": code,
        "site_laterality": laterality,
        "site_region": region,
        "site_family": family,
    }


def site_metadata_from_index(site_index: object) -> Dict[str, str]:
    return site_metadata_from_code(site_code_from_index(site_index))


def default_site_groups(group_by: str, include_unknown: bool = False) -> List[str]:
    if group_by == "laterality":
        groups = ["right", "left"]
        if include_unknown:
            groups.append("unknown")
        return groups
    if group_by == "family":
        groups = ["standard", "sweep"]
        if include_unknown:
            groups.append("unknown")
        return groups
    if group_by == "region":
        groups = ["anterior", "lateral", "posterior", "apical"]
        if include_unknown:
            groups.append("unknown")
        return groups
    if group_by == "site_code":
        codes = [
            code
            for code in SITE_CODE_TO_INDEX
            if code not in {"<PAD>"} and (include_unknown or code != "UNKNOWN")
        ]
        return codes
    return []


def site_group_value(row: Dict[str, object], group_by: str) -> str:
    if group_by == "laterality":
        return str(row.get("site_laterality", "unknown"))
    if group_by == "family":
        return str(row.get("site_family", "unknown"))
    if group_by == "region":
        return str(row.get("site_region", "unknown"))
    if group_by == "site_code":
        return str(row.get("site_code", "UNKNOWN"))
    raise ValueError("Unsupported site grouping: {0}".format(group_by))
