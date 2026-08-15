#!/usr/bin/env python3
"""Inventory and selectively download studies from Butterfly Cloud.

Credentials are never stored in source code. Inventory is separate from
downloading, FASH matching is explicit, downloads are resumable, and sample
downloads are capped by default.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import getpass
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote, urljoin, urlparse

from playwright.sync_api import (
    Browser,
    Locator,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


BUTTERFLY_URL = "https://cloud.butterflynetwork.com/"
DEFAULT_LABELS = ("FASH", "LUS ET FASH")
GRID_PAGE_SIZE = 25
LOGGER = logging.getLogger("butterfly_scraper")
GridSnapshotRow = tuple[str, str, str, int]


class ScraperError(RuntimeError):
    """Raised when a safe scraper invariant cannot be satisfied."""


class AuthenticationError(ScraperError):
    """Raised when Butterfly rejects the supplied credentials."""


class StaleGridError(ScraperError):
    """Raised when a previously observed archive row has been re-rendered away."""


class DownloadStartError(ScraperError):
    """Raised when Butterfly does not begin an explicitly requested download."""


@dataclass(frozen=True)
class GridPage:
    rows: tuple[GridSnapshotRow, ...]
    start: int
    end: int
    total: int
    next_disabled: bool


@dataclass(frozen=True)
class Study:
    title: str
    url: str
    exam_types: str = ""
    page_number: int = 1
    row_index: int = 0

    @property
    def resource_id(self) -> str:
        return resource_id_from_url(self.url)

    @property
    def key(self) -> str:
        identity = self.resource_id or canonical_url_path(self.url)
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class DownloadResult:
    archive: str
    patient_id: str
    study_key: str
    study_resource_id: str
    study_title: str
    study_url: str
    capture_key: str
    capture_resource_id: str
    video_url: str
    filename: str
    size_bytes: int
    sha256: str
    seconds: float
    completed_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def paginator_range(text: str) -> tuple[int, int, int] | None:
    """Parse the three numeric fields in a localized Butterfly paginator."""
    tokens = re.findall(r"\d(?:[\d,.\u00a0\u202f']*\d)?|\d", text)
    if len(tokens) != 3:
        return None
    start, end, total = tuple(
        int(re.sub(r"\D", "", token))
        for token in tokens
    )
    if not (1 <= start <= end <= total):
        return None
    count = end - start + 1
    if count > GRID_PAGE_SIZE:
        return None
    if end < total and count != GRID_PAGE_SIZE:
        return None
    return start, end, total


def paginator_expected_row_count(text: str) -> int | None:
    """Return the complete row count encoded by a Butterfly paginator range.

    The UI text is localized, so only the three numeric fields are used. A
    valid range is either a full non-final page or the (possibly short) final
    page. Ambiguous text fails closed.
    """
    parsed = paginator_range(text)
    return parsed[1] - parsed[0] + 1 if parsed else None


def title_fields(title: str) -> list[str]:
    return [
        normalize_text(part)
        for part in re.split(r"[,;/|]+", title)
        if normalize_text(part)
    ]


def title_matches(title: str, labels: Sequence[str], mode: str) -> bool:
    normalized_labels = [normalize_text(label) for label in labels]
    normalized_title = normalize_text(title)
    if mode == "contains":
        return any(label in normalized_title for label in normalized_labels)
    if mode == "field":
        fields = title_fields(title)
        return any(label == field for label in normalized_labels for field in fields)
    raise ValueError(f"Unsupported match mode: {mode}")


def canonical_url_path(value: str) -> str:
    """Return a query-free, normalized path for a Butterfly resource URL."""
    return unquote(urlparse(value).path).rstrip("/")


def resource_id_from_url(value: str) -> str:
    """Extract Butterfly's opaque resource identifier from a URL."""
    path = canonical_url_path(value)
    return path.rsplit("/", 1)[-1] if path else ""


def resource_key_from_url(value: str, length: int = 16) -> str:
    identity = resource_id_from_url(value) or canonical_url_path(value)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:length]


def patient_id_from_title(title: str) -> str:
    """Extract the pseudonymous study participant ID used by the Benin cohort."""
    match = re.search(r"(?<!\d)(\d{2}-\d+)(?!\d)", title)
    return match.group(1) if match else ""


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def same_resource_url(left: str, right: str) -> bool:
    """Compare Butterfly links without depending on transient query parameters."""
    left_path = canonical_url_path(left)
    right_path = canonical_url_path(right)
    if left_path == right_path:
        return True
    left_id = resource_id_from_url(left)
    right_id = resource_id_from_url(right)
    return bool(left_id and left_id == right_id)


def safe_component(value: str, fallback: str = "unnamed", limit: int = 120) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^\w .-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip(" ._-")
    return (value or fallback)[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch(mode=0o600)
    os.chmod(path, 0o600)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, payloads: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def verified_completed_pairs(manifest: Path, output_root: Path) -> tuple[set[tuple[str, str]], int]:
    """Return only manifest downloads whose local size and SHA-256 still match."""
    latest: dict[tuple[str, str], dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        capture_key = str(record.get("capture_key", ""))
        if not capture_key and record.get("video_url"):
            capture_key = resource_key_from_url(str(record["video_url"]))
        pair = (str(record.get("study_key", "")), capture_key)
        if not all(pair):
            continue
        latest[pair] = record

    completed: set[tuple[str, str]] = set()
    invalid = 0
    resolved_root = output_root.resolve()
    for pair, record in latest.items():
        if record.get("status") != "downloaded":
            continue
        candidate = (output_root / str(record.get("filename", ""))).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            invalid += 1
            continue
        expected_size = record.get("size_bytes")
        expected_sha256 = str(record.get("sha256", ""))
        if (
            not candidate.is_file()
            or not isinstance(expected_size, int)
            or candidate.stat().st_size != expected_size
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or sha256_file(candidate) != expected_sha256
        ):
            invalid += 1
            continue
        completed.add(pair)
    return completed, invalid


def _locator_text(locator: Locator) -> str:
    try:
        return " ".join(locator.inner_text(timeout=1_500).split())
    except PlaywrightTimeoutError:
        return ""


class ButterflyScraper:
    def __init__(self, page: Page, timeout_ms: int) -> None:
        self.page = page
        self.timeout_ms = timeout_ms
        self.home_url = BUTTERFLY_URL

    def login(self, username: str, password: str) -> None:
        self.page.goto(BUTTERFLY_URL, wait_until="domcontentloaded", timeout=self.timeout_ms)
        email = self.page.locator("input[data-bni-id='emailField']")
        password_field = self.page.locator("input[data-bni-id='passwordField']")
        email.wait_for(state="visible", timeout=self.timeout_ms)
        email.fill(username)
        password_field.fill(password)
        self.page.locator("button[data-bni-id='loginButton']").click()

        deadline = time.monotonic() + self.timeout_ms / 1_000
        while time.monotonic() < deadline:
            if email.count() == 0 or not email.is_visible():
                LOGGER.info("Butterfly authentication succeeded")
                self._settle()
                self.home_url = self.page.url
                return
            body = normalize_text(self.page.locator("body").inner_text())
            if any(
                marker in body
                for marker in (
                    "cannot be found in our system",
                    "incorrect password",
                    "invalid password",
                    "login failed",
                )
            ):
                raise AuthenticationError("Butterfly rejected the supplied credentials")
            self.page.wait_for_timeout(500)
        raise AuthenticationError("Butterfly login did not leave the login page before timeout")

    def list_archives(self) -> list[tuple[str, str]]:
        self._settle()
        archives: dict[str, str] = {}
        anchors = self.page.locator("a[href]")
        for index in range(anchors.count()):
            anchor = anchors.nth(index)
            text = _locator_text(anchor)
            href = anchor.get_attribute("href") or ""
            if not text or not href:
                continue
            absolute = urljoin(self.page.url, href)
            if urlparse(absolute).hostname != "cloud.butterflynetwork.com":
                continue
            if not re.search(r"/archives/[^/]+/?$", urlparse(absolute).path):
                continue
            if anchor.locator("span").count() == 0:
                continue
            if normalize_text(text) in {
                "settings",
                "help",
                "tele-guidance",
                "tele-guidance™",
                "home",
                "log out",
            }:
                continue
            archives.setdefault(text, absolute)
        return sorted(archives.items(), key=lambda item: normalize_text(item[0]))

    def open_archive(self, archive_name: str) -> None:
        exact = self.page.get_by_text(archive_name, exact=True)
        if exact.count() == 0:
            available = [name for name, _ in self.list_archives()]
            normalized_wanted = normalize_text(archive_name)
            matches = [name for name in available if normalized_wanted in normalize_text(name)]
            if len(matches) == 1:
                exact = self.page.get_by_text(matches[0], exact=True)
            else:
                raise ScraperError(
                    f"Archive {archive_name!r} was not found. "
                    "Run the 'archives' command to list accessible archives."
                )
        link = exact.first.locator("xpath=ancestor::a[1]")
        if link.count() == 0:
            raise ScraperError(f"Archive label {archive_name!r} is not inside a link")
        link.click()
        self.page.locator("tr[data-bni-id='DataGridTableRow']").first.wait_for(
            state="visible", timeout=self.timeout_ms
        )

    def collect_studies(self, max_pages: int) -> list[Study]:
        studies: dict[str, Study] = {}
        seen_identities: set[str] = set()
        grid_page = self._wait_for_grid_stable(expected_start=1)
        for _ in range(max_pages):
            page_number = (grid_page.start - 1) // GRID_PAGE_SIZE + 1
            page_studies = self._studies_on_current_page(
                None, page_number, stable_rows=grid_page.rows
            )
            page_identities = {
                study.resource_id or canonical_url_path(study.url)
                for study in page_studies
            }
            if seen_identities & page_identities:
                raise ScraperError(
                    "Archive pagination repeated a previously collected study; "
                    "stop and retry the inventory"
                )
            if grid_page.start != len(studies) + 1:
                raise ScraperError(
                    "Archive paginator range is not contiguous with collected studies"
                )
            for study in page_studies:
                identity = study.resource_id or canonical_url_path(study.url)
                studies[identity] = study
            seen_identities.update(page_identities)
            if grid_page.end == grid_page.total:
                if len(studies) != grid_page.total:
                    raise ScraperError(
                        "Collected unique-study count does not match the archive paginator total"
                    )
                return list(studies.values())
            try:
                grid_page = self._advance_page(grid_page)
            except (PlaywrightTimeoutError, ScraperError) as exc:
                raise ScraperError(
                    f"Pagination did not advance after page {page_number}; stopping safely"
                ) from exc
        raise ScraperError(
            f"Reached --max-pages={max_pages}; increase it after checking the inventory"
        )

    def video_urls(self, study: Study) -> list[str]:
        self.page.goto(study.url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        self._settle()
        return self.current_video_urls()

    def open_study(self, archive_name: str, study: Study) -> None:
        """Open a study through UI clicks because direct deep links are redirected."""
        self.page.goto(self.home_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        self._settle()
        self.open_archive(archive_name)
        self._rewind_pages()
        for _ in range(1, study.page_number):
            self._advance_page()
        link = self._study_link_on_current_page(study)
        link.click()
        self.page.locator("[data-bni-id='ExamPageSidebar']").wait_for(
            state="visible", timeout=self.timeout_ms
        )

    def open_study_tab(self, study: Study) -> Page:
        """Open a validated grid row in a child tab while preserving the archive page."""
        link = self._study_link_on_current_page(study)
        with self.page.context.expect_page(timeout=self.timeout_ms) as page_info:
            link.click(button="middle")
        child = page_info.value
        try:
            child.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
            child.locator("[data-bni-id='ExamPageSidebar']").wait_for(
                state="visible", timeout=self.timeout_ms
            )
            if not same_resource_url(child.url, study.url):
                raise ScraperError("The opened study does not match the validated archive row")
            return child
        except Exception:
            child.close()
            raise

    def current_video_urls(self, *, require_nonempty: bool = False) -> list[str]:
        attempts = 3 if require_nonempty else 1
        for attempt in range(1, attempts + 1):
            candidates = self.page.locator(
                "div[class*='AspectRatioBox'] a[href], div.group a[href]"
            )
            try:
                candidates.first.wait_for(
                    state="attached",
                    timeout=(self.timeout_ms if require_nonempty else min(self.timeout_ms, 15_000)),
                )
            except PlaywrightTimeoutError:
                pass
            urls: list[str] = []
            for index in range(candidates.count()):
                href = candidates.nth(index).get_attribute("href")
                if not href:
                    continue
                absolute = urljoin(self.page.url, href)
                if absolute not in urls:
                    urls.append(absolute)
            if urls or not require_nonempty:
                return urls
            if attempt < attempts:
                LOGGER.warning(
                    "No capture links found; reloading study page before retry %d/%d",
                    attempt + 1,
                    attempts,
                )
                self.page.reload(wait_until="domcontentloaded", timeout=self.timeout_ms)
                self.page.locator("[data-bni-id='ExamPageSidebar']").wait_for(
                    state="visible", timeout=self.timeout_ms
                )
        raise ScraperError(
            "No capture links appeared after three study-page attempts; "
            "refusing to certify an ambiguous empty study"
        )

    def download_video(self, video_url: str, destination_dir: Path, index: int) -> tuple[Path, float]:
        if self.page.url != video_url:
            self._click_url(video_url)
        download_button = self.page.locator("[data-bni-id='DownloadButton']")
        download_button.wait_for(state="visible", timeout=self.timeout_ms)

        started = time.monotonic()
        try:
            with self.page.expect_download(timeout=max(self.timeout_ms, 120_000)) as download_info:
                download_button.click()
                submit = self.page.locator("button[type='submit']")
                if submit.count() and submit.first.is_visible():
                    submit.first.click()
            download = download_info.value
        except PlaywrightTimeoutError as exc:
            raise DownloadStartError(
                f"Butterfly did not start a download for video {index}"
            ) from exc

        destination_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(destination_dir, 0o700)
        suggested = safe_component(download.suggested_filename, fallback="capture", limit=160)
        suffix = Path(suggested).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ".bin"
        capture_key = resource_key_from_url(video_url)
        destination = destination_dir / f"{index:04d}-{capture_key}{suffix}"
        if destination.exists():
            destination = destination_dir / (
                f"{index:04d}-{capture_key}-retry-{time.time_ns()}{suffix}"
            )
        download.save_as(destination)
        os.chmod(destination, 0o600)
        return destination, time.monotonic() - started

    def _study_from_row(
        self,
        row: Locator,
        exam_type_column: int | None,
        page_number: int,
        row_index: int,
    ) -> Study | None:
        anchors = row.locator("a[data-bni-id='DataGridRowLink'][href]")
        if anchors.count() == 0:
            anchors = row.locator("a[href]")
        if anchors.count() == 0:
            return None
        anchor = anchors.first
        title_locator = row.locator(".flex-grow.font-bold.truncate")
        title = _locator_text(title_locator.first) if title_locator.count() else _locator_text(anchor)
        href = anchor.get_attribute("href") or ""
        if not title or not href:
            return None
        exam_types = ""
        cells = row.locator("td")
        if exam_type_column is not None and exam_type_column < cells.count():
            exam_types = _locator_text(cells.nth(exam_type_column))
        return Study(
            title=title,
            url=urljoin(self.page.url, href),
            exam_types=exam_types,
            page_number=page_number,
            row_index=row_index,
        )

    def _studies_on_current_page(
        self,
        exam_type_column: int | None,
        page_number: int,
        stable_rows: Sequence[GridSnapshotRow] | None = None,
    ) -> list[Study]:
        if stable_rows is not None:
            return [
                Study(
                    title=title,
                    url=urljoin(self.page.url, href),
                    exam_types=exam_types,
                    page_number=page_number,
                    row_index=row_index,
                )
                for href, title, exam_types, row_index in stable_rows
            ]
        rows = self.page.locator("tr[data-bni-id='DataGridTableRow']")
        payloads = rows.evaluate_all(
            r"""(elements, examTypeColumn) => elements.map((row, rowIndex) => {
                const anchor = row.querySelector("a[data-bni-id='DataGridRowLink'][href]")
                    || row.querySelector("a[href]");
                const titleNode = row.querySelector(".flex-grow.font-bold.truncate") || anchor;
                const cells = Array.from(row.querySelectorAll("td"));
                const clean = node => node ? (node.innerText || "").replace(/\s+/g, " ").trim() : "";
                return {
                    title: clean(titleNode),
                    url: anchor ? anchor.href : "",
                    exam_types: Number.isInteger(examTypeColumn)
                        ? clean(cells[examTypeColumn])
                        : "",
                    row_index: rowIndex,
                };
            })""",
            exam_type_column,
        )
        return [
            Study(
                title=str(payload["title"]),
                url=urljoin(self.page.url, str(payload["url"])),
                exam_types=str(payload["exam_types"]),
                page_number=page_number,
                row_index=int(payload["row_index"]),
            )
            for payload in payloads
            if payload.get("title") and payload.get("url")
        ]

    def _exam_type_column_index(self) -> int | None:
        headers = self.page.locator("thead th")
        for index in range(headers.count()):
            header = headers.nth(index)
            if header.get_attribute("data-bni-id") == "DataGridColumn:examTypes":
                return index
            if normalize_text(_locator_text(header)) in {"exam type", "exam types"}:
                return index
        return None

    def _study_link_on_current_page(self, study: Study) -> Locator:
        """Resolve a study on the current page without ever clicking an unverified row."""
        rows = self.page.locator("tr[data-bni-id='DataGridTableRow']")
        row_count = rows.count()
        preferred = [study.row_index] if 0 <= study.row_index < row_count else []
        indexes = preferred + [index for index in range(row_count) if index not in preferred]
        for row_index in indexes:
            link = rows.nth(row_index).locator(
                "a[data-bni-id='DataGridRowLink'][href]"
            ).first
            if link.count() == 0:
                continue
            href = link.get_attribute("href")
            if href and same_resource_url(urljoin(self.page.url, href), study.url):
                return link
        raise StaleGridError(
            "The stored study is no longer on its inventoried archive page; "
            "refresh the inventory before downloading"
        )

    def _advance_page(self, current_page: GridPage | None = None) -> GridPage:
        if current_page is None:
            current_page = self._wait_for_grid_stable()
        if current_page.end >= current_page.total:
            raise ScraperError("Cannot advance beyond the archive's final page")
        next_button = self.page.locator("button[data-testid='next-btn']")
        if next_button.count() == 0 or next_button.is_disabled():
            raise ScraperError("Cannot advance to the stored study page")
        if next_button.get_attribute("aria-disabled") == "true":
            raise ScraperError("Cannot advance to the stored study page")
        previous_resource_ids = frozenset(
            resource_id_from_url(href) or canonical_url_path(href)
            for href, _title, _exam_types, _row_index in current_page.rows
        )
        next_button.click()
        return self._wait_for_grid_stable(
            expected_start=current_page.end + 1,
            expected_total=current_page.total,
            previous_resource_ids=previous_resource_ids,
        )

    def _wait_for_grid_stable(
        self,
        expected_row_count: int | None = None,
        expected_start: int | None = None,
        expected_total: int | None = None,
        previous_resource_ids: frozenset[str] = frozenset(),
    ) -> GridPage:
        """Wait for the paginator contract and three identical complete row snapshots."""
        rows = self.page.locator("tr[data-bni-id='DataGridTableRow']")
        rows.first.wait_for(state="visible", timeout=self.timeout_ms)
        previous: tuple[
            tuple[int, int, int], tuple[GridSnapshotRow, ...], bool
        ] | None = None
        stable_samples = 0
        deadline = time.monotonic() + self.timeout_ms / 1_000
        while time.monotonic() < deadline:
            state = rows.evaluate_all(
                """elements => {
                    const headers = Array.from(document.querySelectorAll("thead th"));
                    const examTypeColumn = headers.findIndex(header => {
                        const dataId = header.getAttribute("data-bni-id");
                        const text = (header.innerText || "")
                            .replace(/\\s+/g, " ").trim().toLowerCase();
                        return dataId === "DataGridColumn:examTypes"
                            || text === "exam type"
                            || text === "exam types";
                    });
                    const next = document.querySelector("button[data-testid='next-btn']");
                    let paginator = next;
                    while (
                        paginator
                        && !paginator.querySelector("button[data-testid='prev-btn']")
                    ) {
                        paginator = paginator.parentElement;
                    }
                    const clean = node => node
                        ? (node.innerText || "").replace(/\\s+/g, " ").trim()
                        : "";
                    return {
                        rows: elements.map((row, rowIndex) => {
                            const anchor = row.querySelector(
                                "a[data-bni-id='DataGridRowLink'][href]"
                            ) || row.querySelector("a[href]");
                            const title = row.querySelector(
                                ".flex-grow.font-bold.truncate"
                            ) || anchor;
                            const cells = Array.from(row.querySelectorAll("td"));
                            return [
                                anchor ? anchor.href : "",
                                clean(title),
                                examTypeColumn >= 0 ? clean(cells[examTypeColumn]) : "",
                                rowIndex,
                            ];
                        }),
                        paginatorText: clean(paginator),
                        nextPresent: Boolean(next),
                        nextDisabled: next
                            ? Boolean(
                                next.disabled
                                || next.getAttribute("aria-disabled") === "true"
                            )
                            : false,
                    };
                }"""
            )
            snapshot = tuple(
                (str(row[0]), str(row[1]), str(row[2]), int(row[3]))
                for row in state["rows"]
            )
            parsed_range = paginator_range(str(state["paginatorText"]))
            inferred_count = (
                parsed_range[1] - parsed_range[0] + 1 if parsed_range else None
            )
            required_count = (
                expected_row_count if expected_row_count is not None else inferred_count
            )
            identities = frozenset(
                resource_id_from_url(href) or canonical_url_path(href)
                for href, _title, _exam_types, _row_index in snapshot
                if href
            )
            complete = (
                bool(snapshot)
                and parsed_range is not None
                and bool(state["nextPresent"])
                and required_count is not None
                and required_count == inferred_count
                and len(snapshot) == required_count
                and len(identities) == len(snapshot)
                and identities.isdisjoint(previous_resource_ids)
                and (expected_start is None or parsed_range[0] == expected_start)
                and (expected_total is None or parsed_range[2] == expected_total)
                and (
                    bool(state["nextDisabled"])
                    == (parsed_range[1] == parsed_range[2])
                )
                and all(
                    href and title
                    for href, title, _exam_types, _row_index in snapshot
                )
            )
            current = (
                parsed_range, snapshot, bool(state["nextDisabled"])
            ) if parsed_range else None
            if complete and current == previous:
                stable_samples += 1
                if stable_samples >= 2:
                    return GridPage(
                        snapshot, *parsed_range, bool(state["nextDisabled"])
                    )
            else:
                stable_samples = 0
            previous = current
            self.page.wait_for_timeout(250)
        raise ScraperError("The Butterfly study grid did not stabilize before timeout")

    def _rewind_pages(self) -> None:
        """Return a remembered archive paginator to its first page."""
        previous = self.page.locator(
            "button[data-testid='prev-btn'], "
            "button[data-testid='previous-btn'], "
            "button[aria-label*='previous' i]"
        ).first
        for _ in range(1_000):
            if previous.count() == 0 or previous.is_disabled():
                return
            if previous.get_attribute("aria-disabled") == "true":
                return
            rows = self.page.locator("tr[data-bni-id='DataGridTableRow']")
            first_href = self._first_row_href(rows)
            previous.click()
            self.page.wait_for_function(
                """oldHref => {
                    const row = document.querySelector("tr[data-bni-id='DataGridTableRow']");
                    const anchor = row && row.querySelector("a[data-bni-id='DataGridRowLink'][href]");
                    return anchor && anchor.href !== oldHref;
                }""",
                arg=first_href,
                timeout=self.timeout_ms,
            )
        raise ScraperError("Archive pagination could not be rewound safely")

    def _click_url(self, target_url: str) -> None:
        anchors = self.page.locator("a[href]")
        target_index: int | None = None
        for index in range(anchors.count()):
            href = anchors.nth(index).get_attribute("href")
            if href and urljoin(self.page.url, href) == target_url:
                target_index = index
                break
        if target_index is None:
            raise ScraperError("The selected capture link is no longer present in the study")
        anchors.nth(target_index).click()
        target_path = urlparse(target_url).path
        self.page.wait_for_function(
            "targetPath => window.location.pathname === targetPath",
            arg=target_path,
            timeout=self.timeout_ms,
        )

    @staticmethod
    def _first_row_href(rows: Locator) -> str:
        if rows.count() == 0:
            return ""
        anchor = rows.first.locator("a[data-bni-id='DataGridRowLink'][href]")
        if anchor.count() == 0:
            anchor = rows.first.locator("a[href]")
        anchor = anchor.first
        return anchor.get_attribute("href") or ""

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 10_000))
        except PlaywrightTimeoutError:
            pass


def credentials(args: argparse.Namespace) -> tuple[str, str]:
    username = args.username or os.environ.get("BUTTERFLY_USERNAME")
    if not username:
        username = input("Butterfly username: ").strip()
    if not username:
        raise ScraperError("Butterfly username is required")

    password_file = args.password_file or os.environ.get("BUTTERFLY_PASSWORD_FILE")
    if password_file:
        password = Path(password_file).read_text(encoding="utf-8").strip()
    else:
        password = os.environ.get("BUTTERFLY_PASSWORD") or getpass.getpass("Butterfly password: ")
    if not password:
        raise ScraperError("Butterfly password is required")
    return username, password


def matching_studies(
    studies: Sequence[Study], labels: Sequence[str], mode: str, location: str = "title"
) -> list[Study]:
    if location == "exam-type":
        return [study for study in studies if title_matches(study.exam_types, labels, mode)]
    return [study for study in studies if title_matches(study.title, labels, mode)]


def validate_selected_match_freshness(
    studies: Sequence[Study], labels: Sequence[str], mode: str, location: str
) -> list[Study]:
    fresh = matching_studies(studies, labels, mode, location)
    if len(fresh) != len(studies):
        raise ScraperError(
            f"{len(studies) - len(fresh)} selected studies no longer match the "
            "requested FASH criterion; rebuild and review the protected selection"
        )
    return list(studies)


def open_study_tab_with_retry(
    scraper: ButterflyScraper, study: Study, attempts: int = 3
) -> Page:
    """Retry transient child-page timeouts without weakening row validation."""
    for attempt in range(1, attempts + 1):
        try:
            return scraper.open_study_tab(study)
        except PlaywrightTimeoutError:
            if attempt == attempts:
                raise
            LOGGER.warning(
                "Study page timed out; retrying validated row click %d/%d",
                attempt + 1,
                attempts,
            )
            scraper.page.wait_for_timeout(500)
    raise AssertionError("unreachable")


def download_video_with_retry(
    scraper: ButterflyScraper,
    video_url: str,
    destination_dir: Path,
    index: int,
    attempts: int = 3,
) -> tuple[Path, float]:
    """Retry only the explicit transient where a download never begins."""
    for attempt in range(1, attempts + 1):
        try:
            return scraper.download_video(video_url, destination_dir, index)
        except DownloadStartError:
            if attempt == attempts:
                raise
            LOGGER.warning(
                "Capture download did not start; reloading before retry %d/%d",
                attempt + 1,
                attempts,
            )
            scraper.page.reload(wait_until="domcontentloaded", timeout=scraper.timeout_ms)
            scraper.page.locator("[data-bni-id='ExamPageSidebar']").wait_for(
                state="visible", timeout=scraper.timeout_ms
            )
    raise AssertionError("unreachable")


def selected_studies_from_file(
    studies: Sequence[Study], archive_name: str, selection_file: Path
) -> list[Study]:
    """Select exact stable resource IDs from a protected cross-archive plan."""
    wanted: dict[str, str] = {}
    for line_number, line in enumerate(
        selection_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScraperError(
                f"Invalid JSON on line {line_number} of {selection_file}"
            ) from exc
        record_archive = str(record.get("archive", ""))
        if normalize_text(record_archive) != normalize_text(archive_name):
            continue
        resource_id = str(record.get("study_resource_id", ""))
        if not resource_id and record.get("url"):
            resource_id = resource_id_from_url(str(record["url"]))
        if not resource_id:
            raise ScraperError(
                f"Selection line {line_number} for {archive_name!r} has no study resource ID"
            )
        expected_patient_id = str(record.get("patient_id", ""))
        if resource_id in wanted and wanted[resource_id] != expected_patient_id:
            raise ScraperError(
                f"Selection contains conflicting patient IDs for resource {resource_id}"
            )
        wanted[resource_id] = expected_patient_id

    if not wanted:
        raise ScraperError(f"The selection file contains no studies for {archive_name!r}")
    available = {study.resource_id: study for study in studies}
    missing = wanted.keys() - available.keys()
    if missing:
        raise ScraperError(
            f"{len(missing)} selected studies are absent from the fresh archive inventory; "
            "rebuild the selection before downloading"
        )
    mismatched = [
        resource_id
        for resource_id, expected_patient_id in wanted.items()
        if expected_patient_id
        and patient_id_from_title(available[resource_id].title) != expected_patient_id
    ]
    if mismatched:
        raise ScraperError(
            f"{len(mismatched)} selected studies changed patient identity; "
            "rebuild and review the protected selection"
        )
    return [study for study in studies if study.resource_id in wanted]


def shard_studies(studies: Sequence[Study], shard_count: int, shard_index: int) -> list[Study]:
    if shard_count < 1:
        raise ScraperError("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ScraperError("--shard-index must be between 0 and --shard-count - 1")
    if shard_count == 1:
        return list(studies)
    return [
        study
        for study in studies
        if int(hashlib.sha256(study.resource_id.encode("utf-8")).hexdigest(), 16)
        % shard_count
        == shard_index
    ]


def inventory_payload(
    study: Study,
    matched: bool,
    video_count: int | None = None,
    inspected: bool = False,
    page_matched: bool = False,
) -> dict:
    payload = {
        "study_key": study.key,
        "study_resource_id": study.resource_id,
        "patient_id": patient_id_from_title(study.title),
        "title": study.title,
        "url": study.url,
        "exam_types": study.exam_types,
        "page_number": study.page_number,
        "row_index": study.row_index,
        "matched": matched,
        "inspected": inspected,
        "page_matched": page_matched,
    }
    if video_count is not None:
        payload["video_count"] = video_count
    return payload


def run_inventory(scraper: ButterflyScraper, args: argparse.Namespace) -> int:
    scraper.open_archive(args.archive)
    studies = scraper.collect_studies(args.max_pages)
    metadata_location = "exam-type" if args.match_location == "exam-type" else "title"
    title_matches_found = matching_studies(
        studies, args.labels, args.match_mode, location=metadata_location
    )
    title_match_keys = {study.key for study in title_matches_found}
    counts: dict[str, int] = {}
    inspected_keys: set[str] = set()
    page_match_keys: set[str] = set()

    if args.match_location in {"page", "either"}:
        inspect = studies
    elif args.count_videos:
        inspect = title_matches_found
    else:
        inspect = []
    if args.max_study_inspections:
        inspect = inspect[: args.max_study_inspections]

    if inspect:
        normalized_labels = [normalize_text(label) for label in args.labels]
        for number, study in enumerate(inspect, 1):
            LOGGER.info("Inspecting study page %d/%d (key=%s)", number, len(inspect), study.key)
            video_count = len(scraper.video_urls(study))
            body = normalize_text(scraper.page.locator("body").inner_text())
            inspected_keys.add(study.key)
            if any(label in body for label in normalized_labels):
                page_match_keys.add(study.key)
            if args.count_videos:
                counts[study.key] = video_count

    if args.match_location == "title":
        match_keys = title_match_keys
    elif args.match_location == "page":
        match_keys = page_match_keys
    else:
        match_keys = title_match_keys | page_match_keys
    matches = [study for study in studies if study.key in match_keys]

    output = Path(args.output).expanduser().resolve()
    write_jsonl(
        output,
        (
            inventory_payload(
                study,
                study.key in match_keys,
                counts.get(study.key),
                study.key in inspected_keys,
                study.key in page_match_keys,
            )
            for study in studies
        ),
    )
    print(f"studies_total={len(studies)}")
    print(f"studies_matched={len(matches)}")
    print(f"inventory_file={output}")
    if inspected_keys:
        print(f"study_pages_inspected={len(inspected_keys)}")
        print(f"study_pages_with_labels={len(page_match_keys)}")
        if args.match_location in {"page", "either"} and len(inspected_keys) < len(studies):
            print("matching_status=partial")
        else:
            print("matching_status=complete")
    if counts:
        print(f"videos_counted={sum(counts.values())}")
    if args.show_matched_titles:
        for study in matches:
            print(study.title)
    return 0


def run_download(scraper: ButterflyScraper, args: argparse.Namespace) -> int:
    scraper.open_archive(args.archive)
    studies = scraper.collect_studies(args.max_pages)
    if args.selection_file:
        matches = selected_studies_from_file(
            studies, args.archive, Path(args.selection_file).expanduser().resolve()
        )
        matches = validate_selected_match_freshness(
            matches, args.labels, args.match_mode, args.match_location
        )
    else:
        matches = matching_studies(studies, args.labels, args.match_mode, args.match_location)
    selected = matches if args.all else matches[: args.max_studies]
    selected = shard_studies(selected, args.shard_count, args.shard_index)
    selected = sorted(selected, key=lambda study: (study.page_number, study.row_index))
    output_root = Path(args.output_dir).expanduser().resolve()
    manifest = output_root / "download-manifest.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)

    completed_pairs: set[tuple[str, str]] = set()
    if manifest.exists():
        completed_pairs, invalid_completed = verified_completed_pairs(manifest, output_root)
        LOGGER.info(
            "Verified %d completed downloads; %d manifest entries require redownload",
            len(completed_pairs),
            invalid_completed,
        )

    downloaded = 0
    videos_available = 0
    bytes_downloaded = 0
    seconds_downloading = 0.0
    # Collection ends on the archive's final page. Reopening once is both more
    # reliable and much cheaper than rewinding or reopening for every study.
    scraper.page.goto(scraper.home_url, wait_until="domcontentloaded", timeout=scraper.timeout_ms)
    scraper._settle()
    scraper.open_archive(args.archive)
    def download_selected_study(study: Study, study_number: int) -> None:
        nonlocal downloaded, videos_available, bytes_downloaded, seconds_downloading
        LOGGER.info("Processing matched study %d/%d (key=%s)", study_number, len(selected), study.key)
        study_page = open_study_tab_with_retry(scraper, study)
        study_scraper = ButterflyScraper(study_page, timeout_ms=scraper.timeout_ms)
        try:
            urls = study_scraper.current_video_urls(require_nonempty=True)
            videos_available_for_study = len(urls)
            videos_available += videos_available_for_study
            if not args.all:
                urls = urls[: args.max_videos_per_study]
            study_dir = output_root / "studies" / study.key

            for video_number, video_url in enumerate(urls, 1):
                completed_pair = (study.key, resource_key_from_url(video_url))
                if completed_pair in completed_pairs:
                    LOGGER.info("Skipping completed video %d for study %s", video_number, study.key)
                    continue
                try:
                    path, elapsed = download_video_with_retry(
                        study_scraper,
                        video_url, study_dir, video_number
                    )
                    result = DownloadResult(
                        archive=args.archive,
                        patient_id=patient_id_from_title(study.title),
                        study_key=study.key,
                        study_resource_id=study.resource_id,
                        study_title=study.title,
                        study_url=study.url,
                        capture_key=resource_key_from_url(video_url),
                        capture_resource_id=resource_id_from_url(video_url),
                        video_url=video_url,
                        filename=str(path.relative_to(output_root)),
                        size_bytes=path.stat().st_size,
                        sha256=sha256_file(path),
                        seconds=round(elapsed, 3),
                        completed_at=utc_now(),
                    )
                    append_jsonl(manifest, {"status": "downloaded", **asdict(result)})
                    completed_pairs.add(completed_pair)
                    downloaded += 1
                    bytes_downloaded += result.size_bytes
                    seconds_downloading += result.seconds
                except Exception as exc:
                    append_jsonl(
                        manifest,
                        {
                            "status": "failed",
                            "archive": args.archive,
                            "patient_id": patient_id_from_title(study.title),
                            "study_key": study.key,
                            "study_resource_id": study.resource_id,
                            "study_title": study.title,
                            "study_url": study.url,
                            "capture_key": resource_key_from_url(video_url),
                            "capture_resource_id": resource_id_from_url(video_url),
                            "video_url": video_url,
                            "error": type(exc).__name__,
                            "completed_at": utc_now(),
                        },
                    )
                    raise
            append_jsonl(
                manifest,
                {
                    "status": "study_complete" if args.all else "study_sample_complete",
                    "archive": args.archive,
                    "patient_id": patient_id_from_title(study.title),
                    "study_key": study.key,
                    "study_resource_id": study.resource_id,
                    "study_title": study.title,
                    "study_url": study.url,
                    "videos_available": videos_available_for_study,
                    "videos_selected": len(urls),
                    "completed_at": utc_now(),
                },
            )
        finally:
            study_page.close()

    pending = {study.resource_id: study for study in selected}
    study_number = 0
    grid_page = scraper._wait_for_grid_stable(expected_start=1)
    for _ in range(args.max_pages):
        page_number = (grid_page.start - 1) // GRID_PAGE_SIZE + 1
        stale_retries = 0
        while True:
            grid_page = scraper._wait_for_grid_stable(
                expected_start=grid_page.start,
                expected_total=grid_page.total,
            )
            page_studies = scraper._studies_on_current_page(
                None, page_number, stable_rows=grid_page.rows
            )
            current_study = next(
                (
                    study
                    for study in page_studies
                    if study.resource_id in pending
                ),
                None,
            )
            if current_study is None:
                break
            planned_study = pending[current_study.resource_id]
            if patient_id_from_title(current_study.title) != patient_id_from_title(
                planned_study.title
            ):
                raise ScraperError(
                    "A selected study changed patient identity during traversal; "
                    "stop and refresh the protected inventory"
                )
            if not matching_studies(
                [current_study],
                args.labels,
                args.match_mode,
                args.match_location,
            ):
                raise ScraperError(
                    "A selected study no longer matches the requested FASH criterion; "
                    "stop and refresh the protected inventory"
                )
            try:
                download_selected_study(current_study, study_number + 1)
            except StaleGridError:
                stale_retries += 1
                if stale_retries > 5:
                    raise
                LOGGER.info("Archive row re-rendered; refreshing the current page snapshot")
                scraper.page.wait_for_timeout(500)
                continue
            pending.pop(current_study.resource_id)
            study_number += 1
            stale_retries = 0

        if grid_page.end == grid_page.total:
            break
        grid_page = scraper._advance_page(grid_page)
    else:
        raise ScraperError(
            f"Reached --max-pages={args.max_pages} during the download traversal"
        )

    if pending:
        raise ScraperError(
            f"{len(pending)} selected studies disappeared between inventory and download traversal"
        )

    print(f"studies_total={len(studies)}")
    print(f"studies_matched={len(matches)}")
    print(f"studies_selected={len(selected)}")
    print(f"videos_available_in_selected_studies={videos_available}")
    print(f"videos_downloaded={downloaded}")
    print(f"bytes_downloaded={bytes_downloaded}")
    print(f"download_seconds={seconds_downloading:.3f}")
    print(f"manifest_file={manifest}")
    return 0


def run_survey(scraper: ButterflyScraper, args: argparse.Namespace) -> int:
    """Report aggregate label counts across selected archives."""
    home_url = scraper.page.url
    available = [name for name, _ in scraper.list_archives()]
    pattern = re.compile(args.archive_pattern, flags=re.IGNORECASE)
    selected = [name for name in available if pattern.search(name)]
    if not selected:
        raise ScraperError(f"No archive names matched --archive-pattern={args.archive_pattern!r}")

    print(f"archives_selected={len(selected)}")
    for archive_number, archive_name in enumerate(selected, 1):
        if archive_number > 1:
            scraper.page.goto(home_url, wait_until="domcontentloaded", timeout=scraper.timeout_ms)
            scraper._settle()
        LOGGER.info("Surveying archive %d/%d: %s", archive_number, len(selected), archive_name)
        try:
            scraper.open_archive(archive_name)
            studies = scraper.collect_studies(args.max_pages)
        except (ScraperError, PlaywrightTimeoutError) as exc:
            print(
                "archive_survey="
                + json.dumps(
                    {
                        "archive": archive_name,
                        "status": "unreadable",
                        "error": type(exc).__name__,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            continue
        exam_matches = matching_studies(studies, args.labels, "contains", "exam-type")
        title_matches_found = matching_studies(studies, args.labels, "contains", "title")
        payload = {
            "archive": archive_name,
            "studies": len(studies),
            "exam_type_values_present": sum(bool(study.exam_types) for study in studies),
            "exam_type_matches": len(exam_matches),
            "title_matches": len(title_matches_found),
        }
        print("archive_survey=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def run_probe(scraper: ButterflyScraper, args: argparse.Namespace) -> int:
    """Locate label evidence without emitting titles, URLs, or response bodies."""
    live_click = bool(args.archive)
    grid_shape: dict | None = None
    if live_click:
        scraper.open_archive(args.archive)
        for _ in range(1, args.study_page):
            scraper._advance_page()
        rows = scraper.page.locator("tr[data-bni-id='DataGridTableRow']")
        if args.study_index < 0 or args.study_index >= rows.count():
            raise ScraperError(f"--study-index must be between 0 and {rows.count() - 1} on this page")
        selected_row = rows.nth(args.study_index)
        if args.show_dom_shape:
            headers = scraper.page.locator("thead th")
            header_shape = headers.evaluate_all(
                """els => els.map((e, index) => ({
                    index,
                    data_bni_id: e.getAttribute('data-bni-id'),
                    text: (e.innerText || '').trim()
                }))"""
            )
            exam_type_index = next(
                (
                    item["index"]
                    for item in header_shape
                    if item["data_bni_id"] == "DataGridColumn:examTypes"
                ),
                None,
            )
            row_cells = selected_row.locator(":scope > td")
            grid_shape = {
                "headers": header_shape,
                "exam_type_index": exam_type_index,
                "exam_type_cell_text": (
                    _locator_text(row_cells.nth(exam_type_index))
                    if exam_type_index is not None and exam_type_index < row_cells.count()
                    else None
                ),
                "exam_type_column_nodes": scraper.page.locator(
                    "[data-bni-id='DataGridColumn:examTypes']"
                ).evaluate_all(
                    """els => els.map(e => ({
                        tag: e.tagName,
                        role: e.getAttribute('role'),
                        aria_colindex: e.getAttribute('aria-colindex'),
                        class_name: String(e.className)
                    }))"""
                ),
                "row_children": selected_row.locator(":scope > *").evaluate_all(
                    """els => els.map(e => ({
                        tag: e.tagName,
                        role: e.getAttribute('role'),
                        aria_colindex: e.getAttribute('aria-colindex'),
                        data_bni_id: e.getAttribute('data-bni-id'),
                        class_name: String(e.className),
                        contains_fash: /fash/i.test(e.innerText || '')
                    }))"""
                ),
                "row_data_ids": selected_row.locator("[data-bni-id]").evaluate_all(
                    "els => els.map(e => e.getAttribute('data-bni-id')).filter(Boolean)"
                ),
            }
        link = selected_row.locator("a[data-bni-id='DataGridRowLink'][href]").first
        link.click()
        scraper.page.wait_for_timeout(3_000)
        records = [{"title": "redacted", "url": scraper.page.url}]
        start_index = 0
        end_index = 1
    else:
        records = [
            json.loads(line)
            for line in Path(args.inventory).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ScraperError("The supplied inventory is empty")
        if args.study_index < 0 or args.study_index >= len(records):
            raise ScraperError(f"--study-index must be between 0 and {len(records) - 1}")
        start_index = args.study_index
        end_index = min(len(records), args.study_index + args.study_count)
    normalized_labels = [normalize_text(label) for label in args.labels]
    response_paths: set[tuple[str, str, int]] = set()

    def record_response(response) -> None:
        if response.request.resource_type not in {"xhr", "fetch"}:
            return
        parsed = urlparse(response.url)
        response_paths.add((response.request.method, parsed.path, response.status))

    scraper.page.on("response", record_response)
    if grid_shape is not None:
        print("grid_shape=" + json.dumps(grid_shape, sort_keys=True))

    for study_index in range(start_index, end_index):
        record = records[study_index]
        study = Study(record["title"], record["url"])
        video_urls = scraper.current_video_urls() if live_click else scraper.video_urls(study)
        study_body = normalize_text(scraper.page.locator("body").inner_text())
        study_html = normalize_text(scraper.page.content())
        if args.show_dom_shape:
            data_ids = sorted(
                set(
                    scraper.page.locator("[data-bni-id]").evaluate_all(
                        "els => els.map(e => e.getAttribute('data-bni-id')).filter(Boolean)"
                    )
                )
            )
            class_names = sorted(
                set(
                    scraper.page.locator("[class]").evaluate_all(
                        """els => els.flatMap(e => String(e.className).split(/\\s+/))
                            .filter(c => /aspect|group|thumb|image|clip|scan|exam|media/i.test(c))"""
                    )
                )
            )
            anchor_paths: dict[str, int] = {}
            for href in scraper.page.locator("a[href]").evaluate_all(
                "els => els.map(e => e.href).filter(Boolean)"
            ):
                path = urlparse(href).path
                shape = re.sub(r"[A-Za-z0-9_-]{12,}", ":id", path)
                anchor_paths[shape] = anchor_paths.get(shape, 0) + 1
            element_counts = {
                tag: scraper.page.locator(tag).count()
                for tag in ("a", "button", "canvas", "img", "video")
            }
            print("data_bni_ids=" + json.dumps(data_ids))
            print("relevant_classes=" + json.dumps(class_names))
            print("anchor_path_shapes=" + json.dumps(anchor_paths, sort_keys=True))
            print("element_counts=" + json.dumps(element_counts, sort_keys=True))
        video_body_matches = 0
        video_html_matches = 0
        for video_url in video_urls[: args.max_videos]:
            scraper.page.goto(video_url, wait_until="domcontentloaded", timeout=scraper.timeout_ms)
            scraper._settle()
            body = normalize_text(scraper.page.locator("body").inner_text())
            html = normalize_text(scraper.page.content())
            video_body_matches += int(any(label in body for label in normalized_labels))
            video_html_matches += int(any(label in html for label in normalized_labels))

        print(f"study_index={study_index}")
        print(f"study_key={study.key}")
        current_path = urlparse(scraper.page.url).path
        current_path_shape = re.sub(r"[A-Za-z0-9_-]{12,}", ":id", current_path)
        print(f"current_path_shape={current_path_shape}")
        print(f"current_url_matches_inventory={str(scraper.page.url == study.url).lower()}")
        print(f"video_links={len(video_urls)}")
        print(f"study_body_has_label={str(any(label in study_body for label in normalized_labels)).lower()}")
        print(f"study_html_has_label={str(any(label in study_html for label in normalized_labels)).lower()}")
        print(f"video_pages_inspected={min(len(video_urls), args.max_videos)}")
        print(f"video_bodies_with_labels={video_body_matches}")
        print(f"video_html_with_labels={video_html_matches}")
    if args.show_response_paths:
        for method, path, status in sorted(response_paths):
            print(f"response={method} {status} {path}")
    return 0


def launch_browser(playwright: Playwright, args: argparse.Namespace) -> Browser:
    kwargs: dict = {"headless": not args.headed, "slow_mo": args.slow_mo}
    if args.browser_channel:
        kwargs["channel"] = args.browser_channel
    return playwright.chromium.launch(**kwargs)


def add_inventory_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", required=True)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--match-mode", choices=("field", "contains"), default="field")
    parser.add_argument("--max-pages", type=positive_int, default=1_000)
    parser.add_argument(
        "--match-location",
        choices=("exam-type", "title"),
        default="exam-type",
        help="Match labels in Butterfly's Exam Type column or the patient-directory title",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--username", help="Prefer BUTTERFLY_USERNAME to keep it out of shell history")
    result.add_argument("--password-file", help="Read the password from a mounted secret file")
    result.add_argument("--browser-channel", help="Use 'chrome' for a local Google Chrome installation")
    result.add_argument("--headed", action="store_true", help="Show the browser window")
    result.add_argument("--timeout", type=float, default=45.0, help="UI timeout in seconds")
    result.add_argument("--slow-mo", type=int, default=0, help="Browser action delay in milliseconds")
    result.add_argument("--verbose", action="store_true")

    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("archives", help="List accessible Butterfly archives")

    survey = subparsers.add_parser("survey", help="Aggregate FASH counts across matching archives")
    survey.add_argument(
        "--archive-pattern",
        default=r".*",
        help="Regex selecting archive names; the default audits every accessible archive",
    )
    survey.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    survey.add_argument("--max-pages", type=positive_int, default=1_000)

    inventory = subparsers.add_parser("inventory", help="Inventory studies without downloading media")
    add_inventory_options(inventory)
    inventory.add_argument("--output", required=True, help="Protected JSONL inventory path")
    inventory.add_argument("--count-videos", action="store_true")
    inventory.add_argument("--max-study-inspections", type=positive_int)
    inventory.add_argument("--show-matched-titles", action="store_true")

    download = subparsers.add_parser("download", help="Download a capped sample by default")
    add_inventory_options(download)
    download.add_argument(
        "--selection-file",
        help="Protected JSONL plan with archive and study_resource_id fields",
    )
    download.add_argument("--shard-count", type=int, default=1)
    download.add_argument("--shard-index", type=int, default=0)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--max-studies", type=positive_int, default=1)
    download.add_argument("--max-videos-per-study", type=positive_int, default=1)
    download.add_argument(
        "--all",
        action="store_true",
        help="Remove sample caps and download all matched media (explicit safety switch)",
    )

    probe = subparsers.add_parser(
        "probe",
        help="Inspect one inventoried study for label locations without printing patient data",
    )
    probe_source = probe.add_mutually_exclusive_group(required=True)
    probe_source.add_argument("--inventory")
    probe_source.add_argument("--archive", help="Click a visible row instead of directly opening an inventory URL")
    probe.add_argument("--study-index", type=int, default=0)
    probe.add_argument("--study-page", type=positive_int, default=1)
    probe.add_argument("--study-count", type=positive_int, default=1)
    probe.add_argument("--max-videos", type=positive_int, default=3)
    probe.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    probe.add_argument("--show-response-paths", action="store_true")
    probe.add_argument("--show-dom-shape", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        username, password = credentials(args)
        with sync_playwright() as playwright:
            browser = launch_browser(playwright, args)
            try:
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()
                scraper = ButterflyScraper(page, timeout_ms=int(args.timeout * 1_000))
                scraper.login(username, password)
                password = ""
                if args.command == "archives":
                    archives = scraper.list_archives()
                    print(f"archives_found={len(archives)}")
                    for name, _ in archives:
                        print(name)
                    return 0
                if args.command == "inventory":
                    return run_inventory(scraper, args)
                if args.command == "survey":
                    return run_survey(scraper, args)
                if args.command == "download":
                    return run_download(scraper, args)
                if args.command == "probe":
                    return run_probe(scraper, args)
                raise ScraperError(f"Unsupported command: {args.command}")
            finally:
                with suppress(Exception):
                    browser.close()
    except (ScraperError, PlaywrightTimeoutError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
