import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from build_fash_selection import (
    build_selection,
    main,
    parse_expected_inventory_count,
    read_inventory,
    read_inventory_with_count,
    read_roster,
    validate_inventory_counts,
)


class BuildFashSelectionTest(unittest.TestCase):
    def test_primary_archive_wins_and_contact_supplements_missing_patient(self):
        records = [
            self.record("TrUST Bénin", "study-a", "99-9001"),
            self.record("Contact study Bénin", "study-copy", "99-9001"),
            self.record("Contact study Bénin", "study-b", "99-9002"),
            self.record("3P pediatrie Benin", "study-c", "99-9003"),
        ]
        selected, summary = build_selection(
            records,
            {"99-9001", "99-9002", "99-9003", "99-9004"},
            inventory_counts={
                "TrUST Bénin": 1007,
                "Contact study Bénin": 219,
                "3P pediatrie Benin": 92,
            },
        )
        self.assertEqual(
            [(record["archive"], record["patient_id"]) for record in selected],
            [("TrUST Bénin", "99-9001"), ("Contact study Bénin", "99-9002")],
        )
        self.assertEqual(summary["selected_patients"], 2)
        self.assertEqual(
            summary["inventory_studies_by_archive"],
            {
                "3P pediatrie Benin": 92,
                "Contact study Bénin": 219,
                "TrUST Bénin": 1007,
            },
        )
        self.assertEqual(
            summary["roster_ids_without_selected_study"], ["99-9003", "99-9004"]
        )

    def test_inventory_and_roster_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster = root / "roster.csv"
            with roster.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["record_id"])
                writer.writeheader()
                writer.writerow({"record_id": "99-9001"})
            inventory = root / "inventory.jsonl"
            inventory.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in [
                        {
                            "matched": True,
                            "title": "99-9001, FASH",
                            "url": "https://example.test/exams/study-a",
                            "page_number": 2,
                            "row_index": 3,
                        },
                        {
                            "matched": False,
                            "title": "not selected",
                            "url": "https://example.test/exams/study-b",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(read_roster(roster, "record_id"), {"99-9001"})
            records = read_inventory("TrUST Bénin", inventory)
            self.assertEqual(records[0]["patient_id"], "99-9001")
            self.assertEqual(records[0]["study_resource_id"], "study-a")
            counted_records, inventory_count = read_inventory_with_count(
                "TrUST Bénin", inventory
            )
            self.assertEqual(counted_records, records)
            self.assertEqual(inventory_count, 2)

            duplicate = {
                "matched": False,
                "title": "duplicate",
                "url": "https://example.test/exams/study-a",
            }
            with inventory.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(duplicate) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicates resource ID"):
                read_inventory_with_count("TrUST Bénin", inventory)

    def test_expected_inventory_counts_are_exact_and_cover_every_archive(self):
        actual = {"TrUST Bénin": 1007, "Contact study Bénin": 219}
        validate_inventory_counts(actual, dict(actual))
        with self.assertRaisesRegex(ValueError, "found 1007, expected 1006"):
            validate_inventory_counts(
                actual, {"TrUST Bénin": 1006, "Contact study Bénin": 219}
            )
        with self.assertRaisesRegex(ValueError, "missing expected counts"):
            validate_inventory_counts(actual, {"TrUST Bénin": 1007})
        with self.assertRaisesRegex(ValueError, "without inventories"):
            validate_inventory_counts(
                actual,
                {
                    "TrUST Bénin": 1007,
                    "Contact study Bénin": 219,
                    "3P pediatrie Benin": 92,
                },
            )

    def test_expected_inventory_count_parser(self):
        self.assertEqual(
            parse_expected_inventory_count("TrUST Bénin=1007"),
            ("TrUST Bénin", 1007),
        )
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "ARCHIVE=N"):
            parse_expected_inventory_count("TrUST Bénin")

    def test_main_refuses_wrong_complete_inventory_count_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster = root / "roster.csv"
            roster.write_text("record_id\n99-9001\n", encoding="utf-8")
            inventory = root / "inventory.jsonl"
            inventory.write_text(
                json.dumps(
                    {
                        "matched": True,
                        "title": "99-9001, FASH",
                        "url": "https://example.test/exams/study-a",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "selection.jsonl"
            with self.assertRaisesRegex(ValueError, "found 1, expected 2"):
                main(
                    [
                        "--inventory",
                        f"TrUST Bénin={inventory}",
                        "--expected-inventory-count",
                        "TrUST Bénin=2",
                        "--roster",
                        str(roster),
                        "--output",
                        str(output),
                        "--summary",
                        str(root / "summary.json"),
                    ]
                )
            self.assertFalse(output.exists())

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--inventory",
                        f"TrUST Bénin={inventory}",
                        "--expected-inventory-count",
                        "TrUST Bénin=1",
                        "--roster",
                        str(roster),
                        "--output",
                        str(output),
                        "--summary",
                        str(root / "summary.json"),
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["inventory_studies_by_archive"], {"TrUST Bénin": 1}
            )

    @staticmethod
    def record(archive: str, study_resource_id: str, patient_id: str) -> dict:
        return {
            "archive": archive,
            "study_resource_id": study_resource_id,
            "study_key": study_resource_id,
            "patient_id": patient_id,
            "title_fingerprint": study_resource_id,
            "page_number": 1,
            "row_index": 0,
        }


if __name__ == "__main__":
    unittest.main()
