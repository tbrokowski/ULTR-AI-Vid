import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from assemble_fash_dataset import (
    AssemblyError,
    _read_private_file_snapshot,
    assemble_dataset,
    capture_key_for_resource,
    study_key_for_resource,
    verify_staging_tree,
)


class AssembleFashDatasetTest(unittest.TestCase):
    def test_combines_multiple_protected_manifest_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = [
                ("TrUST Bénin", "99-9001", "study-resource-a"),
                ("Contact study Bénin", "99-9002", "study-resource-b"),
            ]
            selection = root / "selection.jsonl"
            self.write_private_jsonl(
                selection, [self.selection_record(*study) for study in studies]
            )
            manifests = []
            for index, study in enumerate(studies):
                raw = root / f"raw-{index}"
                raw.mkdir(mode=0o700)
                source = raw / "studies" / str(index) / "capture.mp4"
                source.parent.mkdir(parents=True)
                source.write_bytes(f"capture-{index}".encode())
                source.chmod(0o600)
                manifest = raw / "download-manifest.jsonl"
                self.write_private_jsonl(
                    manifest,
                    [
                        self.capture_record(study, f"capture-{index}", source, raw),
                        self.completion_record(study, 1),
                    ],
                )
                manifests.append(manifest)

            summary = assemble_dataset(selection, manifests, root / "output")

            self.assertEqual(summary["study_count"], 2)
            self.assertEqual(summary["capture_count"], 2)
            self.assertEqual(len(summary["archive_counts"]), 2)

    def test_builds_private_minimized_tree_and_reports_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, sources = self.complete_fixture(root)
            output = root / "hf-staging"

            summary = assemble_dataset(selection, [manifest], output)
            self.assertEqual(
                verify_staging_tree(output, summary["selection_sha256"]), summary
            )

            self.assertEqual(summary["study_count"], 2)
            self.assertEqual(summary["patient_count"], 2)
            self.assertEqual(summary["capture_count"], 3)
            self.assertEqual(summary["unique_resource_count"], 2)
            self.assertEqual(summary["unique_content_count"], 1)
            self.assertEqual(summary["resource_duplicate_group_count"], 1)
            self.assertEqual(summary["content_duplicate_group_count"], 1)
            self.assertEqual(summary["resource_duplicate_capture_count"], 2)
            self.assertEqual(summary["content_duplicate_capture_count"], 3)
            self.assertEqual(summary["resource_redundant_capture_count"], 1)
            self.assertEqual(summary["content_redundant_capture_count"], 2)
            self.assertEqual(
                summary["resource_duplicate_groups_with_content_conflicts"], 0
            )
            self.assertEqual(summary["resource_duplicate_groups_across_patients"], 1)
            self.assertEqual(summary["content_duplicate_groups_across_patients"], 1)

            captures = self.read_jsonl(output / "metadata" / "captures.jsonl")
            self.assertEqual(len(captures), 3)
            for row in captures:
                self.assertEqual(
                    set(row),
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
                    },
                )
                staged = output / row["path"]
                self.assertTrue(staged.is_file())
                self.assertEqual(staged.suffix, ".mp4")
                self.assertEqual(staged.stat().st_mode & 0o077, 0)
            shared_rows = [
                row
                for row in captures
                if row["capture_key"] == capture_key_for_resource("capture-shared")
            ]
            self.assertEqual(len(shared_rows), 2)
            self.assertEqual(
                {row["resource_duplicate_patient_count"] for row in shared_rows}, {2}
            )
            self.assertEqual(
                {row["resource_duplicate_content_hash_count"] for row in shared_rows},
                {1},
            )
            self.assertTrue(all(row["resource_duplicate_group"] for row in shared_rows))
            self.assertTrue(all(row["content_duplicate_group"] for row in captures))
            self.assertEqual(
                {row["content_duplicate_capture_count"] for row in captures}, {3}
            )

            # Private sources on the same filesystem should be hardlinked.
            first_staged = output / captures[0]["path"]
            matching_source = next(
                source
                for source in sources
                if hashlib.sha256(source.read_bytes()).hexdigest()
                == captures[0]["sha256"]
            )
            self.assertTrue(os.path.samefile(first_staged, matching_source))
            self.assertEqual(summary["staging_transfer_counts"], {"hardlinked": 3})

            studies = self.read_jsonl(output / "metadata" / "studies.jsonl")
            self.assertEqual([row["capture_count"] for row in studies], [2, 1])
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            for metadata_file in [
                output / "metadata" / "captures.jsonl",
                output / "metadata" / "studies.jsonl",
                output / "summary.json",
                output / "README.md",
            ]:
                self.assertEqual(metadata_file.stat().st_mode & 0o777, 0o600)

            public_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.suffix in {".json", ".jsonl", ".md"}
            )
            for forbidden in [
                "Raw patient title",
                "https://cloud.example/study",
                "study-resource-a",
                "study-resource-b",
                "capture-shared",
                "capture-alias",
                "source-a.MP4",
            ]:
                self.assertNotIn(forbidden, public_text)

    def test_latest_capture_row_must_match_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            records = self.read_jsonl(manifest)
            bad_latest = dict(records[0], sha256="0" * 64)
            self.write_private_jsonl(manifest, [records[0], bad_latest, records[1]])

            with self.assertRaisesRegex(AssemblyError, "SHA-256 mismatch"):
                assemble_dataset(selection, [manifest], root / "output")
            self.assertFalse((root / "output").exists())

    def test_latest_failed_capture_prevents_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            records = self.read_jsonl(manifest)
            failed = {
                key: records[0][key]
                for key in (
                    "archive",
                    "patient_id",
                    "study_key",
                    "study_resource_id",
                    "capture_key",
                    "capture_resource_id",
                )
            }
            failed["status"] = "failed"
            self.write_private_jsonl(manifest, [records[0], failed, records[1]])

            with self.assertRaisesRegex(AssemblyError, "latest failed"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_full_completion_counts_must_match_latest_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            records = self.read_jsonl(manifest)
            records[1]["videos_available"] = 2
            self.write_private_jsonl(manifest, records)

            with self.assertRaisesRegex(AssemblyError, "completion counts"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_zero_capture_completion_is_rejected_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir(mode=0o700)
            study = ("TrUST Bénin", "99-9001", "study-resource-a")
            selection = root / "selection.jsonl"
            self.write_private_jsonl(selection, [self.selection_record(*study)])
            manifest = raw / "download-manifest.jsonl"
            self.write_private_jsonl(manifest, [self.completion_record(study, 0)])

            with self.assertRaisesRegex(AssemblyError, "zero-capture completion"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_sample_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            records = self.read_jsonl(manifest)
            records[1]["status"] = "study_sample_complete"
            self.write_private_jsonl(manifest, records)

            with self.assertRaisesRegex(AssemblyError, "sample completion"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_latest_sample_completion_supersedes_older_full_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            records = self.read_jsonl(manifest)
            latest_sample = dict(records[1])
            latest_sample.update(
                status="study_sample_complete",
                videos_available=2,
                videos_selected=1,
            )
            self.write_private_jsonl(manifest, [*records, latest_sample])

            with self.assertRaisesRegex(AssemblyError, "latest sample completion"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_zero_byte_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, source = self.single_study_fixture(root)
            source.write_bytes(b"")
            source.chmod(0o600)
            records = self.read_jsonl(manifest)
            records[0]["size_bytes"] = 0
            records[0]["sha256"] = hashlib.sha256(b"").hexdigest()
            self.write_private_jsonl(manifest, records)

            with self.assertRaisesRegex(AssemblyError, "size_bytes must be positive"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_audited_selection_hash_is_enforced_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            expected = hashlib.sha256(selection.read_bytes()).hexdigest()

            with self.assertRaisesRegex(AssemblyError, "audited baseline"):
                assemble_dataset(
                    selection,
                    [manifest],
                    root / "wrong-output",
                    expected_selection_sha256="0" * 64,
                )

            output = root / "output"
            summary = assemble_dataset(
                selection,
                [manifest],
                output,
                expected_selection_sha256=expected,
            )
            self.assertEqual(summary["selection_sha256"], expected)
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["selection_sha256"],
                expected,
            )

    def test_selection_hash_and_parser_use_the_same_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            original_bytes = selection.read_bytes()
            expected = hashlib.sha256(original_bytes).hexdigest()
            replacement = self.selection_record(
                "TrUST Bénin", "99-9999", "replacement-study"
            )
            original_snapshot_reader = _read_private_file_snapshot
            replaced = False

            def replace_after_snapshot(path: Path, label: str) -> bytes:
                nonlocal replaced
                payload = original_snapshot_reader(path, label)
                if label == "selection file" and not replaced:
                    replaced = True
                    self.write_private_jsonl(selection, [replacement])
                return payload

            output = root / "output"
            with mock.patch(
                "assemble_fash_dataset._read_private_file_snapshot",
                side_effect=replace_after_snapshot,
            ):
                summary = assemble_dataset(
                    selection,
                    [manifest],
                    output,
                    expected_selection_sha256=expected,
                )

            study = self.read_jsonl(output / "metadata" / "studies.jsonl")[0]
            self.assertEqual(study["patient_id"], "99-9001")
            self.assertEqual(summary["selection_sha256"], expected)

    def test_manifest_identity_must_match_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            records = self.read_jsonl(manifest)
            records[0]["patient_id"] = "99-9999"
            self.write_private_jsonl(manifest, records)

            with self.assertRaisesRegex(AssemblyError, "patient_id does not match"):
                assemble_dataset(selection, [manifest], root / "output")

    def test_nonempty_output_requires_explicit_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            output = root / "output"
            output.mkdir()
            sentinel = output / "unrelated.txt"
            sentinel.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(AssemblyError, "--replace-existing"):
                assemble_dataset(selection, [manifest], output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me")

            summary = assemble_dataset(
                selection, [manifest], output, replace_existing=True
            )
            self.assertEqual(summary["capture_count"], 1)
            self.assertFalse(sentinel.exists())
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.building-*")), []
            )
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.replaced-*")), []
            )

    def test_output_symlink_is_rejected_without_replacing_its_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            target = root / "unrelated-target"
            target.mkdir()
            sentinel = target / "preserve.txt"
            sentinel.write_text("preserve me", encoding="utf-8")
            output_link = root / "output-link"
            output_link.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(AssemblyError, "symbolic link"):
                assemble_dataset(
                    selection,
                    [manifest],
                    output_link,
                    replace_existing=True,
                )
            self.assertTrue(output_link.is_symlink())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me")

    def test_staging_verifier_rejects_drift_and_forbidden_metadata(self):
        mutations = ("deleted", "mutated", "extra", "metadata", "summary")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                selection, manifest, _ = self.single_study_fixture(root)
                output = root / "output"
                summary = assemble_dataset(selection, [manifest], output)
                verify_staging_tree(output, summary["selection_sha256"])
                capture_row = self.read_jsonl(
                    output / "metadata" / "captures.jsonl"
                )[0]
                capture_path = output / capture_row["path"]

                if mutation == "deleted":
                    capture_path.unlink()
                elif mutation == "mutated":
                    capture_path.write_bytes(b"X" * capture_row["size_bytes"])
                elif mutation == "extra":
                    unexpected = output / "raw-manifest.jsonl"
                    unexpected.write_text("patient title", encoding="utf-8")
                    unexpected.chmod(0o600)
                elif mutation == "metadata":
                    capture_row["study_title"] = "Raw patient title"
                    self.write_private_jsonl(
                        output / "metadata" / "captures.jsonl", [capture_row]
                    )
                else:
                    summary["capture_count"] += 1
                    summary_path = output / "summary.json"
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    summary_path.chmod(0o600)

                with self.assertRaises(AssemblyError):
                    verify_staging_tree(output, summary["selection_sha256"])

    def test_cross_device_hardlink_failure_uses_verified_private_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            output = root / "output"
            with mock.patch(
                "assemble_fash_dataset.os.link",
                side_effect=OSError(errno.EXDEV, "cross-device link"),
            ):
                summary = assemble_dataset(selection, [manifest], output)

            self.assertEqual(summary["staging_transfer_counts"], {"copied": 1})
            capture = self.read_jsonl(output / "metadata" / "captures.jsonl")[0]
            staged = output / capture["path"]
            self.assertEqual(staged.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                hashlib.sha256(staged.read_bytes()).hexdigest(), capture["sha256"]
            )

    def test_rejects_unprotected_selection_and_output_containing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, manifest, _ = self.single_study_fixture(root)
            selection.chmod(0o644)
            with self.assertRaisesRegex(AssemblyError, "group or other"):
                assemble_dataset(selection, [manifest], root / "output")
            selection.chmod(0o600)

            with self.assertRaisesRegex(AssemblyError, "must not contain"):
                assemble_dataset(selection, [manifest], root)

    def complete_fixture(self, root: Path):
        raw = root / "raw"
        raw.mkdir(mode=0o700)
        studies = [
            ("TrUST Bénin", "99-9001", "study-resource-a"),
            ("Contact study Bénin", "99-9002", "study-resource-b"),
        ]
        selection = root / "selection.jsonl"
        self.write_private_jsonl(
            selection,
            [self.selection_record(*study) for study in studies],
        )
        source_a = raw / "studies" / "a" / "source-a.MP4"
        source_alias = raw / "studies" / "a" / "source-alias.mp4"
        source_b = raw / "studies" / "b" / "source-b.mp4"
        for source in (source_a, source_alias, source_b):
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"identical ultrasound bytes")
            source.chmod(0o600)
        records = [
            self.capture_record(studies[0], "capture-shared", source_a, raw),
            self.capture_record(studies[0], "capture-alias", source_alias, raw),
            self.completion_record(studies[0], 2),
            self.capture_record(studies[1], "capture-shared", source_b, raw),
            self.completion_record(studies[1], 1),
        ]
        manifest = raw / "download-manifest.jsonl"
        self.write_private_jsonl(manifest, records)
        return selection, manifest, [source_a, source_alias, source_b]

    def single_study_fixture(self, root: Path):
        raw = root / "raw"
        raw.mkdir(mode=0o700)
        study = ("TrUST Bénin", "99-9001", "study-resource-a")
        selection = root / "selection.jsonl"
        self.write_private_jsonl(selection, [self.selection_record(*study)])
        source = raw / "studies" / "a" / "capture.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"ultrasound")
        source.chmod(0o600)
        manifest = raw / "download-manifest.jsonl"
        self.write_private_jsonl(
            manifest,
            [
                self.capture_record(study, "capture-a", source, raw),
                self.completion_record(study, 1),
            ],
        )
        return selection, manifest, source

    @staticmethod
    def selection_record(archive: str, patient_id: str, resource_id: str) -> dict:
        return {
            "archive": archive,
            "patient_id": patient_id,
            "study_resource_id": resource_id,
            "study_key": study_key_for_resource(resource_id),
        }

    @staticmethod
    def capture_record(study: tuple[str, str, str], capture_id: str, source: Path, raw: Path):
        archive, patient_id, resource_id = study
        return {
            "status": "downloaded",
            "archive": archive,
            "patient_id": patient_id,
            "study_key": study_key_for_resource(resource_id),
            "study_resource_id": resource_id,
            "study_title": "Raw patient title",
            "study_url": "https://cloud.example/study",
            "capture_key": capture_key_for_resource(capture_id),
            "capture_resource_id": capture_id,
            "video_url": "https://cloud.example/capture",
            "filename": source.relative_to(raw).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

    @staticmethod
    def completion_record(study: tuple[str, str, str], count: int) -> dict:
        archive, patient_id, resource_id = study
        return {
            "status": "study_complete",
            "archive": archive,
            "patient_id": patient_id,
            "study_key": study_key_for_resource(resource_id),
            "study_resource_id": resource_id,
            "study_title": "Raw patient title",
            "study_url": "https://cloud.example/study",
            "videos_available": count,
            "videos_selected": count,
        }

    @staticmethod
    def write_private_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        path.chmod(0o600)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
