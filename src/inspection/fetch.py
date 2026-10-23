"""Official MVTec AD download, verification and safe extraction (stdlib tarfile plus httpx)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tarfile
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DOWNLOADS_PAGE = "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads"
ALLOWED_HOST = "https://www.mydrive.ch/"
CATEGORY_ORDER = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip


def discover_links(client: httpx.Client | None = None) -> dict[str, str]:
    """Read the official downloads page and return {archive stem: url}."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        html = client.get(DOWNLOADS_PAGE).text
    finally:
        if own:
            client.close()
    links: dict[str, str] = {}
    for url in re.findall(r"https://www\.mydrive\.ch[^\"'\s<]+", html):
        m = re.search(r"/([A-Za-z_]+)\.tar\.xz$", url)
        if m and url.startswith(ALLOWED_HOST):
            links[m.group(1)] = url
    return links


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class DownloadUnavailable(RuntimeError):
    """The official link answered with a permanent client error (for example 404)."""


MANUAL_HELP = (
    "Download the archives yourself from the official page "
    "(https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads), put them in "
    "data/raw/ (per-category <category>.tar.xz files or the full "
    "mvtec_anomaly_detection.tar.xz) and run `mvinspect fetch-data` again; local archives are "
    "used before any download is attempted."
)


def download(
    url: str, dest: Path, retries: int = 6, transport: httpx.BaseTransport | None = None
) -> Path:
    """Download with HTTP Range resume and exponential backoff. Transient failures (network
    errors, 5xx, 429) are retried; a permanent client error such as 404 fails at once."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with (
                httpx.Client(
                    follow_redirects=True,
                    timeout=httpx.Timeout(60, read=120),
                    transport=transport,
                ) as c,
                c.stream("GET", url, headers=headers) as r,
            ):
                if r.status_code == 416:  # already complete
                    break
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise DownloadUnavailable(
                        f"the official link returned HTTP {r.status_code}: {url}. {MANUAL_HELP}"
                    )
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(part, mode) as fh:
                    for block in r.iter_bytes(1 << 20):
                        fh.write(block)
            break
        except (httpx.HTTPError, OSError) as exc:
            wait = min(60, 2**attempt)
            log.warning("download attempt %d failed (%s); retry in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"download failed after {retries} attempts: {url}. {MANUAL_HELP}")
    part.replace(dest)
    return dest


def safe_extract(archive: Path, out_dir: Path) -> int:
    """Extract an xz tar, rejecting absolute paths, '..', links and device files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    n = 0
    with tarfile.open(archive, "r:xz") as tar:
        for m in tar:
            name = m.name
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in archive: {name}")
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                raise ValueError(f"unsafe member type in archive: {name}")
            target = (root / name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"path escapes target: {name}")
            tar.extract(m, root, filter="data")
            n += m.isfile()
    return n


def verify_archive(archive: Path) -> bool:
    """Equivalent of `tar -tJf`: read the whole archive index."""
    try:
        with tarfile.open(archive, "r:xz") as tar:
            for _ in tar:
                pass
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


FULL_ARCHIVE = "mvtec_anomaly_detection.tar.xz"


def _save_record(downloads_json: Path, record: dict[str, str]) -> None:
    downloads_json.parent.mkdir(parents=True, exist_ok=True)
    downloads_json.write_text(json.dumps(record, indent=2))


def fetch_all(
    data_dir: Path,
    raw_dir: Path,
    categories: list[str] | None = None,
    downloads_json: Path | None = None,
) -> dict[str, str]:
    """Get the dataset in the order the project specifies:

    1. categories already extracted (and recorded) are skipped;
    2. archives found in raw_dir are verified and extracted without any network access
       (per-category <category>.tar.xz first, then the full mvtec_anomaly_detection.tar.xz);
    3. only what is still missing is downloaded from the links on the official page.

    Every archive's SHA-256 is recorded in downloads.json. Archives this function downloaded
    are deleted after a verified extraction; archives placed by the user are kept."""
    cats = categories or CATEGORY_ORDER
    downloads_json = downloads_json or data_dir.parent / "downloads.json"
    record: dict[str, str] = {}
    if downloads_json.exists():
        record = json.loads(downloads_json.read_text())

    def done(c: str) -> bool:
        return (data_dir / c / "test").is_dir() and c in record

    for c in cats:
        if done(c):
            log.info("%s already extracted, skipping", c)

    def extract(arc: Path, keys: list[str], delete: bool) -> None:
        if not verify_archive(arc):
            if delete:
                arc.unlink()
            raise RuntimeError(f"archive failed verification: {arc.name}")
        sha = sha256_file(arc)
        safe_extract(arc, data_dir)
        for k in keys:
            record[k] = sha
        _save_record(downloads_json, record)
        if delete:
            arc.unlink()
        log.info("%s extracted", arc.name)

    # 2. local archives, no network
    for c in [c for c in cats if not done(c)]:
        arc = raw_dir / f"{c}.tar.xz"
        if arc.exists():
            extract(arc, [c], delete=False)
    full = raw_dir / FULL_ARCHIVE
    if full.exists() and any(not done(c) for c in cats):
        extract(full, [], delete=False)
        for c in cats:
            if (data_dir / c / "test").is_dir() and c not in record:
                record[c] = "full:" + sha256_file(full)
        _save_record(downloads_json, record)

    todo = [c for c in cats if not done(c)]
    if not todo:
        return record

    # 3. download what is still missing
    links = discover_links()
    missing = [c for c in todo if c not in links]
    if missing:
        raise RuntimeError(
            f"no official link found on the downloads page for: {missing}. {MANUAL_HELP}"
        )
    # archive plus extracted files: about 2 GB per category at most, 12 GB for the full set
    need_gb = min(12, 2 * len(todo))
    free = shutil.disk_usage(raw_dir.parent if raw_dir.parent.exists() else Path.cwd()).free
    if free < need_gb * 1024**3:
        raise RuntimeError(f"need about {need_gb} GB free, have {free / 1024**3:.1f} GB")
    for cat in todo:
        arc = raw_dir / f"{cat}.tar.xz"
        log.info("downloading %s", cat)
        download(links[cat], arc)
        extract(arc, [cat], delete=True)
    return record
