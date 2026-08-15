# Butterfly FASH dataset workflow

This workflow inventories Butterfly Cloud, selects the Benin HMV-MIL cohort,
downloads every capture resumably to EPFL RCP, and builds a privacy-minimized
tree for the private Hugging Face dataset `lxflk/fash`.

Never commit or upload the protected inventories, cohort roster, selection,
selection summary, raw download manifests, authentication state, or any
intermediate files. Butterfly credentials are requested interactively; do not
put them in source, command-line arguments, shell history, logs, or the shared
PVC.

## Selection policy

The model roster defines the cohort. `TrUST Bénin` is the primary archive.
`Contact study Bénin` supplements it only for roster members who have no
matching study in the primary archive. This avoids mixing the separate cohorts
from other archives into the HMV-MIL dataset.

`build_fash_selection.py` applies that policy to records already marked as
matching by the protected inventories and:

- retains only records whose parsed pseudonymous ID is in the model roster;
- keeps every primary-archive study for an eligible roster member;
- admits a supplement-archive study only when that member is absent from the
  primary archive; and
- deduplicates selected entries by Butterfly's stable study resource ID.

The resulting JSONL is an exact download plan, not another fuzzy title query.
The downloader checks its archive, stable resource ID, and pseudonymous
participant identity against a fresh archive inventory, and confirms the live
row still matches the requested FASH title criterion, before it downloads
anything. The assembler independently validates each derived key.

The final full read-only audit on 2026-08-15 traversed 2,461 studies across all
readable archive grids. The four archives with FASH-title matches contained
1,007 studies in `TrUST Bénin`, 219 in `Contact study Bénin`, 92 in
`3P pediatrie Benin`, and 132 in `RSA-Tintswalo-RuralUS Library`. Within those
complete grids, the FASH-title match counts were respectively 500, 105, 23,
and 1: 629 matching archive entries in total. Every other readable archive had
zero FASH-title matches. `Maison de Santé` exposed no readable study grid and
is recorded as unreadable, not as a zero. Neither the 2,461 grid rows nor the
629 matches are patient counts or the training scope: archive overlap and the
504-ID model roster are resolved by the selection policy above.

Re-run the complete read-only archive audit with the default all-archive
pattern when the source may have changed:

```bash
kubectl exec -it -n runai-light-falke fash-scraper-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/butterfly_scraper.py \
  --timeout 60 survey
```

## RCP setup and protected paths

Connect to the EPFL VPN, authenticate Run:ai through browser SSO, select the
`light-falke` project, and confirm the expected account before starting a job.
Never record a browser verification code.

Start a CPU-only Playwright container with the shared scratch PVC:

```bash
runai submit --name fash-scraper -p light-falke \
  -i mcr.microsoft.com/playwright/python:v1.61.0-noble \
  --cpu 2 --memory 4G \
  --existing-pvc claimname=light-scratch,path=/mloscratch \
  --large-shm --run-as-uid 315954 --run-as-gid 84257 \
  --create-home-dir --interactive --command -- bash -lc 'sleep infinity'
```

Install the pinned scraper dependencies in every new scraper pod:

```bash
kubectl exec -n runai-light-falke fash-scraper-0-0 -- \
  python -m pip install --user \
  -r /mloscratch/fash-tintswalo/tools/requirements-scraper.txt
```

Create private working directories before placing the roster or generating
metadata. The assembler refuses a selection, manifest, or raw root that is
accessible by group or other users.

```bash
kubectl exec -n runai-light-falke fash-scraper-0-0 -- bash -lc '
  umask 077
  install -d -m 700 \
    /mloscratch/fash-data \
    /mloscratch/fash-data/inventories \
    /mloscratch/fash-data/protected \
    /mloscratch/fash-data/raw-shards
'
```

Place the approved roster in the protected directory using an approved secure
transfer, then set its mode before reading it:

```bash
kubectl exec -n runai-light-falke fash-scraper-0-0 -- chmod 600 \
  /mloscratch/fash-data/protected/labels_multidiagnosis.csv
```

Inventories contain source metadata and therefore belong only in the protected
RCP area. Run each command with an interactive TTY so the scraper can prompt
for the username and hidden password:

```bash
kubectl exec -it -n runai-light-falke fash-scraper-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/butterfly_scraper.py \
  --timeout 60 inventory \
  --archive 'TrUST Bénin' \
  --match-location title --match-mode contains \
  --output /mloscratch/fash-data/inventories/trust-benin-stable.jsonl

kubectl exec -it -n runai-light-falke fash-scraper-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/butterfly_scraper.py \
  --timeout 60 inventory \
  --archive 'Contact study Bénin' \
  --match-location title --match-mode contains \
  --output /mloscratch/fash-data/inventories/contact-benin-stable.jsonl
```

The defaults match both supported FASH labels. Do not use
`--show-matched-titles` in an RCP session log. Re-running `inventory` replaces
the selected output file, so review aggregate counts before using a refreshed
inventory.

Build and protect the exact cross-archive plan:

```bash
kubectl exec -n runai-light-falke fash-scraper-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/build_fash_selection.py \
  --inventory 'TrUST Bénin=/mloscratch/fash-data/inventories/trust-benin-stable.jsonl' \
  --inventory 'Contact study Bénin=/mloscratch/fash-data/inventories/contact-benin-stable.jsonl' \
  --expected-inventory-count 'TrUST Bénin=1007' \
  --expected-inventory-count 'Contact study Bénin=219' \
  --roster /mloscratch/fash-data/protected/labels_multidiagnosis.csv \
  --roster-column record_id \
  --output /mloscratch/fash-data/protected/selection-final.jsonl \
  --summary /mloscratch/fash-data/protected/selection-final-summary.json

kubectl exec -n runai-light-falke fash-scraper-0-0 -- chmod 600 \
  /mloscratch/fash-data/inventories/trust-benin-stable.jsonl \
  /mloscratch/fash-data/inventories/contact-benin-stable.jsonl \
  /mloscratch/fash-data/protected/labels_multidiagnosis.csv \
  /mloscratch/fash-data/protected/selection-final.jsonl \
  /mloscratch/fash-data/protected/selection-final-summary.json
```

The expected counts are mandatory and cover every supplied inventory. They
count all inventory rows, not only FASH matches. The builder fails before
writing a selection if an inventory is truncated, contains duplicate resource
rows, is duplicated on the command line, or lacks an exact audited count. Its
protected summary records the complete counts as
`inventory_studies_by_archive` alongside the matched and selected counts.

Treat an audited selection as immutable during a download attempt. If an exact
selected resource disappears or its parsed identity changes, the downloader
stops; refresh and review the inventories and selection instead of weakening
that check. If a complete re-audit requires a replacement selection, first let
active attempts finish or stop them, atomically install and hash the reviewed
replacement, and then rerun every TrUST shard plus the Contact command with the
same shard count and raw output roots. Each downloader reads the selection once
at startup. The stable resource sharding and verified manifests retain valid
captures, while newly selected studies are downloaded on the rerun. Do not
assemble or upload until every rerun completes against the replacement.

For the final audited Benin run on 2026-08-15, stop if the protected summary
does not report exactly 497 studies for 494 roster participants: 495 studies
from `TrUST Bénin`, 2 from `Contact study Bénin`, and 10 of the 504 roster IDs
with no selected FASH study. The SHA-256 of that exact audited selection JSONL
is `0e68ab6ed194820ccbca352e2af5d76a94ce2590096df2ab6ad65ecedca36775`.
The assembler command below binds completion to this hash, so a valid but
truncated replacement selection cannot be declared complete accidentally.

## Exact, sharded, resumable download

Use the same shard count for the entire run. The stable study resource ID is
hashed modulo `--shard-count`, so changing the count changes the assignment.
Run indices `0`, `1`, `2`, and `3` in four separate Playwright pods, with one
private output root per index. For example, the index-zero command is:

```bash
SCRAPER_POD=fash-scraper-0-0
SHARD_INDEX=0

kubectl exec -it -n runai-light-falke "$SCRAPER_POD" -- \
  python /mloscratch/fash-tintswalo/tools/butterfly_scraper.py \
  --timeout 60 download \
  --archive 'TrUST Bénin' \
  --match-location title --match-mode contains \
  --selection-file /mloscratch/fash-data/protected/selection-final.jsonl \
  --shard-count 4 --shard-index "$SHARD_INDEX" \
  --output-dir "/mloscratch/fash-data/raw-shards/shard-$SHARD_INDEX" \
  --all
```

Run the supplement selection without sharding after the primary shards (or in
another private output root):

```bash
kubectl exec -it -n runai-light-falke fash-scraper-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/butterfly_scraper.py \
  --timeout 60 download \
  --archive 'Contact study Bénin' \
  --match-location title --match-mode contains \
  --selection-file /mloscratch/fash-data/protected/selection-final.jsonl \
  --output-dir /mloscratch/fash-data/raw-shards/contact \
  --all
```

`--all` is the explicit safety switch that removes sample caps and writes full
study-completion records. Re-run the identical command, including shard count,
index, selection, and output directory, after an interruption. On startup the
scraper reads `download-manifest.jsonl` and skips a capture only when its file
still matches the recorded byte size and SHA-256; missing or changed files are
downloaded again.

Each raw root is mode `0700`, each capture and manifest is mode `0600`, and
capture paths use derived stable keys rather than source titles. The manifest
still contains sensitive source metadata needed for audit and resume. Keep it
beside the raw files on RCP; never upload it to Hugging Face.

## Verify and assemble the Hugging Face staging tree

`assemble_fash_dataset.py` is the boundary between protected source metadata
and the private dataset. It requires a full completion marker for every exact
selected study, rejects a latest failed capture, checks completion counts,
safe relative paths, file sizes, and every SHA-256, and detects a source file
that changes during verification. It then installs a new staging tree
atomically.

Pass every shard manifest and the supplement manifest:

```bash
kubectl exec -n runai-light-falke fash-scraper-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/assemble_fash_dataset.py \
  --selection /mloscratch/fash-data/protected/selection-final.jsonl \
  --expected-selection-sha256 0e68ab6ed194820ccbca352e2af5d76a94ce2590096df2ab6ad65ecedca36775 \
  --manifest /mloscratch/fash-data/raw-shards/shard-0/download-manifest.jsonl \
  --manifest /mloscratch/fash-data/raw-shards/shard-1/download-manifest.jsonl \
  --manifest /mloscratch/fash-data/raw-shards/shard-2/download-manifest.jsonl \
  --manifest /mloscratch/fash-data/raw-shards/shard-3/download-manifest.jsonl \
  --manifest /mloscratch/fash-data/raw-shards/contact/download-manifest.jsonl \
  --output /mloscratch/fash-data/hf-staging
```

Omit `--replace-existing` for the first build. On subsequent builds it permits
replacement only after the new tree has been assembled completely. Staging
directories remain mode `0700` and staged files mode `0600`. The output
contains only:

- `data/` with the private captures under pseudonymous, derived path keys;
- `metadata/captures.jsonl` and `metadata/studies.jsonl` with the minimized
  training/audit schema;
- `summary.json` with counts, sizes, and duplicate statistics; and
- a generated private-dataset `README.md`.

Source titles, URLs, raw resource IDs, source filenames, and acquisition
timestamps are excluded. Do not manually copy anything from a raw root into
this staging tree. A quick metadata-boundary check should produce no matches:

```bash
rg -n '"(study_title|study_url|video_url|study_resource_id|capture_resource_id|completed_at)"' \
  /mloscratch/fash-data/hf-staging/metadata \
  /mloscratch/fash-data/hf-staging/summary.json \
  /mloscratch/fash-data/hf-staging/README.md
```

The assembler retains repeated captures rather than silently discarding them.
It records duplicate groups separately for repeated source resources and
identical SHA-256 content, including whether a group crosses studies or
pseudonymous participants. Review those statistics in `summary.json` before
upload. Because same-filesystem staging uses private hardlinks for efficiency,
the reusable verifier must be run immediately before and after the upload. It
rejects any post-assembly addition, deletion, symlink, permission widening,
capture size/content change, forbidden metadata field, or inconsistent study
or summary count.

## Replace and verify the private Hugging Face dataset

Use a short-lived pod so the Hub token and cache never land on the PVC:

```bash
runai submit --name fash-hf-upload -p light-falke \
  -i python:3.12-slim \
  --cpu 2 --memory 4G \
  --existing-pvc claimname=light-scratch,path=/mloscratch \
  --run-as-uid 315954 --run-as-gid 84257 \
  --create-home-dir --interactive --command -- bash -lc 'sleep infinity'

kubectl exec -n runai-light-falke fash-hf-upload-0-0 -- \
  python -m pip install --user 'huggingface_hub>=1.3,<2'

kubectl exec -n runai-light-falke fash-hf-upload-0-0 -- \
  install -d -m 700 /tmp/fash-hf-home

kubectl exec -it -n runai-light-falke fash-hf-upload-0-0 -- \
  env HF_HOME=/tmp/fash-hf-home hf auth login --force
```

Complete the browser/device authorization interactively and confirm the
intended Hub account. Never paste the token or device code into a script, log,
manifest, or repository.

Run the local staging verifier in the upload pod immediately before the upload:

```bash
kubectl exec -n runai-light-falke fash-hf-upload-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/assemble_fash_dataset.py \
  --verify-staging /mloscratch/fash-data/hf-staging \
  --expected-selection-sha256 0e68ab6ed194820ccbca352e2af5d76a94ce2590096df2ab6ad65ecedca36775
```

The following upload keeps the repository private and deletes the exact old
remote file list while uploading the completed staging tree. Hugging Face
preserves `.gitattributes` even when it appears in `delete_patterns`.

```bash
kubectl exec -i -n runai-light-falke fash-hf-upload-0-0 -- \
  env HF_HOME=/tmp/fash-hf-home python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
repo_id = "lxflk/fash"
repo_type = "dataset"

assert api.whoami().get("name") == "lxflk"
api.update_repo_settings(repo_id, repo_type=repo_type, private=True)
old_files = [
    path
    for path in api.list_repo_files(repo_id, repo_type=repo_type)
    if path != ".gitattributes"
]
api.upload_folder(
    folder_path="/mloscratch/fash-data/hf-staging",
    repo_id=repo_id,
    repo_type=repo_type,
    delete_patterns=old_files or None,
    commit_message="Replace with verified FASH dataset",
)
PY
```

After the upload, verify privacy, the exact staged path set, file sizes, total
bytes, and every available LFS SHA-256 against local content:

```bash
kubectl exec -i -n runai-light-falke fash-hf-upload-0-0 -- \
  env HF_HOME=/tmp/fash-hf-home python - <<'PY'
from hashlib import sha1, sha256
import json
from pathlib import Path

from huggingface_hub import HfApi


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


api = HfApi()
repo_id = "lxflk/fash"
repo_type = "dataset"
root = Path("/mloscratch/fash-data/hf-staging")
local = {
    path.relative_to(root).as_posix(): path
    for path in root.rglob("*")
    if path.is_file()
}
remote = {
    item.path: item
    for item in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True)
    if hasattr(item, "size")
}
capture_rows = [
    json.loads(line)
    for line in (root / "metadata" / "captures.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
capture_metadata = {row["path"]: row for row in capture_rows}
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

assert api.whoami().get("name") == "lxflk"
assert api.repo_info(repo_id, repo_type=repo_type).private is True
assert set(local) - set(remote) == set()
assert set(remote) - set(local) <= {".gitattributes"}

lfs_files_checked = 0
capture_files = set(capture_metadata)
assert capture_files == {relative for relative in local if relative.startswith("data/")}
assert len(capture_files) == summary["capture_count"]
assert sum(row["size_bytes"] for row in capture_rows) == summary["size_bytes"]
for relative, path in local.items():
    entry = remote[relative]
    assert entry.size == path.stat().st_size
    if relative in capture_files:
        assert entry.lfs is not None
    if entry.lfs is not None:
        lfs_sha = (
            entry.lfs["sha256"]
            if isinstance(entry.lfs, dict)
            else entry.lfs.sha256
        )
        assert lfs_sha == file_sha256(path)
        if relative in capture_files:
            assert entry.size == capture_metadata[relative]["size_bytes"]
            assert lfs_sha == capture_metadata[relative]["sha256"]
            lfs_files_checked += 1
    else:
        assert entry.blob_id == git_blob_sha1(path)

local_bytes = sum(path.stat().st_size for path in local.values())
remote_bytes = sum(remote[relative].size for relative in local)
assert local_bytes == remote_bytes
assert lfs_files_checked == len(capture_files)
print("repo_private=true")
print(f"files={len(local)}")
print(f"bytes={local_bytes}")
print(f"lfs_sha256_checked={lfs_files_checked}")
PY
```

Re-run the same local staging verifier after the remote assertions, before
removing credentials or jobs:

```bash
kubectl exec -n runai-light-falke fash-hf-upload-0-0 -- \
  python /mloscratch/fash-tintswalo/tools/assemble_fash_dataset.py \
  --verify-staging /mloscratch/fash-data/hf-staging \
  --expected-selection-sha256 0e68ab6ed194820ccbca352e2af5d76a94ce2590096df2ab6ad65ecedca36775
```

All capture checksums were already verified by the assembler; the Hub check
adds remote path, size, and LFS-object verification. Investigate any assertion
failure before deleting raw RCP data.

Finally, remove authentication and cache state from the explicit ephemeral
path, then delete the upload pod and all scraper jobs:

```bash
kubectl exec -n runai-light-falke fash-hf-upload-0-0 -- \
  env HF_HOME=/tmp/fash-hf-home hf auth logout
kubectl exec -n runai-light-falke fash-hf-upload-0-0 -- \
  rm -rf -- /tmp/fash-hf-home
runai delete job fash-hf-upload -p light-falke
runai delete job fash-scraper -p light-falke
```

Delete additional shard jobs by their exact names. Keep the protected roster,
inventories, selection, raw manifests, captures, and final staging tree on RCP
until the remote verification and research handoff are complete; remove only
known temporary pods, authentication state, smoke-test outputs, and abandoned
partial staging trees.
