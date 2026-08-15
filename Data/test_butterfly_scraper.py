import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from butterfly_scraper import (
    ButterflyScraper,
    DownloadStartError,
    GridPage,
    ScraperError,
    Study,
    canonical_url_path,
    matching_studies,
    download_video_with_retry,
    open_study_tab_with_retry,
    paginator_expected_row_count,
    paginator_range,
    patient_id_from_title,
    positive_int,
    resource_id_from_url,
    resource_key_from_url,
    safe_component,
    same_resource_url,
    selected_studies_from_file,
    shard_studies,
    title_fields,
    title_matches,
    validate_selected_match_freshness,
    verified_completed_pairs,
    write_jsonl,
)


class ButterflyScraperHelpersTest(unittest.TestCase):
    def test_sample_caps_require_positive_integers(self):
        self.assertEqual(positive_int("1"), 1)
        for value in ("0", "-1", "not-a-number"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_int(value)

    def test_required_capture_discovery_rejects_ambiguous_empty_page(self):
        class EmptyCandidates:
            @property
            def first(self):
                return self

            def wait_for(self, **_kwargs):
                return None

            def count(self):
                return 0

        class EmptyPage:
            url = "https://example.test/exams/study-a"

            def locator(self, _selector):
                return EmptyCandidates()

            def reload(self, **_kwargs):
                return None

        scraper = ButterflyScraper(EmptyPage(), timeout_ms=1_000)
        with self.assertRaisesRegex(ScraperError, "ambiguous empty study"):
            scraper.current_video_urls(require_nonempty=True)

    def test_grid_stability_waits_for_the_expected_complete_row_count(self):
        class SnapshotRows:
            def __init__(self):
                self.snapshots = [
                    [["https://example.test/exams/a", "A", "FASH", 0]],
                    [["https://example.test/exams/a", "A", "FASH", 0]],
                    [
                        ["https://example.test/exams/a", "A", "FASH", 0],
                        ["https://example.test/exams/b", "B", "FASH", 1],
                    ],
                    [
                        ["https://example.test/exams/a", "A", "FASH", 0],
                        ["https://example.test/exams/b", "B", "FASH", 1],
                    ],
                    [
                        ["https://example.test/exams/a", "A", "FASH", 0],
                        ["https://example.test/exams/b", "B", "FASH", 1],
                    ],
                ]
                self.calls = 0

            @property
            def first(self):
                return self

            def wait_for(self, **_kwargs):
                return None

            def evaluate_all(self, _script):
                snapshot = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
                self.calls += 1
                return {
                    "rows": snapshot,
                    "paginatorText": "1-2 of 2",
                    "nextPresent": True,
                    "nextDisabled": True,
                }

        class SnapshotPage:
            def __init__(self, rows):
                self.rows = rows

            def locator(self, _selector):
                return self.rows

            def wait_for_timeout(self, _milliseconds):
                return None

        rows = SnapshotRows()
        scraper = ButterflyScraper(SnapshotPage(rows), timeout_ms=1_000)
        grid_page = scraper._wait_for_grid_stable(expected_row_count=2)
        self.assertEqual(rows.calls, 5)
        self.assertEqual((grid_page.start, grid_page.end, grid_page.total), (1, 2, 2))

    def test_paginator_range_drives_full_and_short_page_counts(self):
        self.assertEqual(paginator_range("1-25 of 1,007"), (1, 25, 1007))
        self.assertEqual(paginator_expected_row_count("1,001–1,007 of 1,007"), 7)
        self.assertEqual(paginator_expected_row_count("76-92 sur 92"), 17)
        self.assertIsNone(paginator_range("loading"))
        self.assertIsNone(paginator_range("1-24 of 1,007"))

    def test_grid_stability_rejects_partial_and_blank_title_snapshots(self):
        def page_rows(count, *, blank_last=False):
            return [
                [
                    f"https://example.test/exams/synthetic-{index}",
                    "" if blank_last and index == count - 1 else f"Synthetic {index}",
                    "FASH",
                    index,
                ]
                for index in range(count)
            ]

        class SnapshotRows:
            def __init__(self):
                full = page_rows(25)
                self.states = [
                    {"rows": page_rows(24), "paginatorText": "1-25 of 1,007", "nextPresent": True, "nextDisabled": False},
                    {"rows": page_rows(25, blank_last=True), "paginatorText": "1-25 of 1,007", "nextPresent": True, "nextDisabled": False},
                    {"rows": full, "paginatorText": "1-25 of 1,007", "nextPresent": True, "nextDisabled": False},
                    {"rows": full, "paginatorText": "1-25 of 1,007", "nextPresent": True, "nextDisabled": False},
                    {"rows": full, "paginatorText": "1-25 of 1,007", "nextPresent": True, "nextDisabled": False},
                ]
                self.calls = 0

            @property
            def first(self):
                return self

            def wait_for(self, **_kwargs):
                return None

            def evaluate_all(self, _script):
                state = self.states[min(self.calls, len(self.states) - 1)]
                self.calls += 1
                return state

        class SnapshotPage:
            def __init__(self, rows):
                self.rows = rows

            def locator(self, _selector):
                return self.rows

            def wait_for_timeout(self, _milliseconds):
                return None

        rows = SnapshotRows()
        grid_page = ButterflyScraper(
            SnapshotPage(rows), timeout_ms=1_000
        )._wait_for_grid_stable()
        self.assertEqual(len(grid_page.rows), 25)
        self.assertEqual(rows.calls, 5)

    def test_grid_stability_rejects_enabled_next_with_apparent_final_range(self):
        full = [
            [
                f"https://example.test/exams/synthetic-{index}",
                f"Synthetic {index}",
                "FASH",
                index,
            ]
            for index in range(25)
        ]

        class SnapshotRows:
            def __init__(self):
                self.calls = 0

            @property
            def first(self):
                return self

            def wait_for(self, **_kwargs):
                return None

            def evaluate_all(self, _script):
                self.calls += 1
                return {
                    "rows": full,
                    "paginatorText": "1-25 of 25",
                    "nextPresent": True,
                    "nextDisabled": self.calls > 1,
                }

        class SnapshotPage:
            def __init__(self, rows):
                self.rows = rows

            def locator(self, _selector):
                return self.rows

            def wait_for_timeout(self, _milliseconds):
                return None

        rows = SnapshotRows()
        grid_page = ButterflyScraper(
            SnapshotPage(rows), timeout_ms=1_000
        )._wait_for_grid_stable()
        self.assertTrue(grid_page.next_disabled)
        self.assertEqual(rows.calls, 4)

    def test_collection_rejects_nonadjacent_repeated_page_resources(self):
        def grid_rows(offset):
            return tuple(
                (
                    f"https://example.test/exams/synthetic-{offset + index}",
                    f"Synthetic {offset + index}",
                    "FASH",
                    index,
                )
                for index in range(25)
            )

        pages = iter(
            [
                GridPage(grid_rows(25), 26, 50, 75, False),
                GridPage(grid_rows(0), 51, 75, 75, True),
            ]
        )

        class Page:
            url = "https://example.test/archives/synthetic"

        scraper = ButterflyScraper(Page(), timeout_ms=1_000)
        scraper._wait_for_grid_stable = lambda **_kwargs: GridPage(
            grid_rows(0), 1, 25, 75, False
        )
        scraper._advance_page = lambda _current: next(pages)
        with self.assertRaisesRegex(ScraperError, "repeated"):
            scraper.collect_studies(max_pages=3)

    def test_child_study_timeout_retries_validated_click(self):
        expected_page = object()

        class RetryPage:
            def __init__(self):
                self.waits = 0

            def wait_for_timeout(self, _milliseconds):
                self.waits += 1

        class RetryScraper:
            def __init__(self):
                self.calls = 0
                self.page = RetryPage()

            def open_study_tab(self, _study):
                self.calls += 1
                if self.calls < 3:
                    raise PlaywrightTimeoutError("transient")
                return expected_page

        from butterfly_scraper import PlaywrightTimeoutError

        scraper = RetryScraper()
        actual = open_study_tab_with_retry(
            scraper, Study("synthetic", "https://example.test/exams/study-a")
        )
        self.assertIs(actual, expected_page)
        self.assertEqual(scraper.calls, 3)
        self.assertEqual(scraper.page.waits, 2)

    def test_download_start_failure_reloads_and_retries(self):
        expected = (Path("capture.mp4"), 1.25)

        class ReloadPage:
            def __init__(self):
                self.reloads = 0

            def reload(self, **_kwargs):
                self.reloads += 1

            def locator(self, _selector):
                return self

            def wait_for(self, **_kwargs):
                return None

        class RetryDownloadScraper:
            timeout_ms = 1_000

            def __init__(self):
                self.calls = 0
                self.page = ReloadPage()

            def download_video(self, *_args):
                self.calls += 1
                if self.calls < 3:
                    raise DownloadStartError("transient")
                return expected

        scraper = RetryDownloadScraper()
        actual = download_video_with_retry(
            scraper,
            "https://example.test/captures/capture-a",
            Path("unused"),
            1,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(scraper.calls, 3)
        self.assertEqual(scraper.page.reloads, 2)

    def test_field_matching_is_case_and_whitespace_insensitive(self):
        self.assertTrue(title_matches("patient-1,  FASH ", ["FASH"], "field"))
        self.assertTrue(title_matches("patient-2 | lus et fash", ["LUS ET FASH"], "field"))
        self.assertFalse(title_matches("patient-3, not-fash", ["FASH"], "field"))

    def test_contains_mode_is_explicitly_broader(self):
        self.assertTrue(title_matches("patient-3, follow-up FASH exam", ["FASH"], "contains"))
        self.assertFalse(title_matches("patient-3, follow-up FASH exam", ["FASH"], "field"))

    def test_title_fields(self):
        self.assertEqual(title_fields("A, B / C; D | E"), ["a", "b", "c", "d", "e"])

    def test_safe_component(self):
        self.assertEqual(safe_component("  A/B:C  "), "A_B_C")

    def test_study_key_is_stable(self):
        self.assertEqual(
            Study("one", "https://example.test/1").key,
            Study("two", "https://example.test/1").key,
        )

    def test_study_key_uses_resource_id_across_route_aliases(self):
        self.assertEqual(
            Study("one", "https://example.test/a/exams/exam-123?sort=asc").key,
            Study("two", "https://example.test/b/archive/exam-123").key,
        )

    def test_resource_identity_helpers(self):
        url = "https://example.test/a/exams/exam%2D123/?sort=asc"
        self.assertEqual(canonical_url_path(url), "/a/exams/exam-123")
        self.assertEqual(resource_id_from_url(url), "exam-123")
        self.assertEqual(
            resource_key_from_url(url),
            resource_key_from_url("https://example.test/other/exam-123"),
        )

    def test_patient_id_from_title(self):
        self.assertEqual(patient_id_from_title("99-9001, LUS et FASH"), "99-9001")
        self.assertEqual(patient_id_from_title("prefix 99-9002 / FASH"), "99-9002")
        self.assertEqual(patient_id_from_title("FASH without roster id"), "")

    def test_resource_url_ignores_query_and_route_prefix(self):
        self.assertTrue(
            same_resource_url(
                "https://example.test/a/exams/exam-123?sort=asc",
                "https://example.test/b/archive/exam-123",
            )
        )
        self.assertFalse(
            same_resource_url(
                "https://example.test/a/exams/exam-123",
                "https://example.test/a/exams/exam-456",
            )
        )

    def test_exam_type_matching_does_not_use_patient_title(self):
        studies = [Study("patient-1", "https://example.test/1", "LUS et FASH")]
        self.assertEqual(
            matching_studies(studies, ["FASH"], "contains", "exam-type"), studies
        )
        self.assertEqual(matching_studies(studies, ["FASH"], "contains", "title"), [])

    def test_inventory_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.jsonl"
            write_jsonl(path, [{"ok": True}])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_resume_requires_present_file_with_matching_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "studies" / "one" / "capture.mp4"
            good.parent.mkdir(parents=True)
            good.write_bytes(b"verified")
            manifest = root / "download-manifest.jsonl"
            records = [
                {
                    "status": "downloaded",
                    "study_key": "one",
                    "video_url": "https://example.test/captures/good",
                    "filename": str(good.relative_to(root)),
                    "size_bytes": good.stat().st_size,
                    "sha256": hashlib.sha256(good.read_bytes()).hexdigest(),
                },
                {
                    "status": "downloaded",
                    "study_key": "one",
                    "video_url": "https://example.test/captures/missing",
                    "filename": "missing.mp4",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                },
            ]
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            completed, invalid = verified_completed_pairs(manifest, root)
            self.assertEqual(
                completed,
                {("one", resource_key_from_url("https://example.test/captures/good"))},
            )
            self.assertEqual(invalid, 1)

    def test_exact_cross_archive_selection(self):
        studies = [
            Study("99-9001, FASH", "https://example.test/exams/study-a", page_number=2),
            Study("99-9002, FASH", "https://example.test/exams/study-b", page_number=3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "archive": "TrUST Bénin",
                        "study_resource_id": "study-b",
                        "patient_id": "99-9002",
                    },
                    {
                        "archive": "Contact study Bénin",
                        "study_resource_id": "study-a",
                        "patient_id": "99-9001",
                    },
                ],
            )
            self.assertEqual(
                selected_studies_from_file(studies, "TrUST Bénin", path),
                [studies[1]],
            )

    def test_exact_selection_must_still_match_fash_title(self):
        with self.assertRaisesRegex(ScraperError, "no longer match"):
            validate_selected_match_freshness(
                [Study("99-9001, OTHER", "https://example.test/exams/study-a")],
                ["FASH", "LUS ET FASH"],
                "contains",
                "title",
            )

    def test_shards_are_disjoint_and_complete(self):
        studies = [
            Study(str(index), f"https://example.test/exams/study-{index}")
            for index in range(20)
        ]
        shards = [shard_studies(studies, 4, index) for index in range(4)]
        keys = [{study.resource_id for study in shard} for shard in shards]
        self.assertEqual(set().union(*keys), {study.resource_id for study in studies})
        self.assertEqual(sum(len(key_set) for key_set in keys), len(studies))


if __name__ == "__main__":
    unittest.main()
