"""
Unified downloader for all public OCT datasets.

Supports: Kaggle API, Google Drive (gdown), direct URL, Mendeley, Zenodo, GitHub releases.
"""

from __future__ import annotations

import stat
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

from .registry import DatasetInfo, DownloadSource, DATASET_REGISTRY


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with partial.open("xb") as handle, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for chunk in resp.iter_content(chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    pbar.update(len(chunk))
        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _safe_download_filename(filename: str) -> str:
    normalized = str(filename or "").strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in {"", ".", ".."}
        or ":" in path.name
    ):
        raise ValueError(f"unsafe download filename: {filename!r}")
    return path.name


def _archive_target(dest_dir: Path, member_name: str) -> Path | None:
    """Resolve an archive member without allowing it to escape ``dest_dir``."""
    normalized = str(member_name or "").replace("\\", "/")
    member = PurePosixPath(normalized)
    parts = tuple(part for part in member.parts if part not in {"", "."})
    if not parts:
        return None
    if member.is_absolute() or ".." in parts or ":" in parts[0]:
        raise ValueError(f"unsafe archive member: {member_name!r}")
    root = dest_dir.resolve()
    target = (root / Path(*parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"archive member escapes destination: {member_name!r}") from exc
    return target


def _prepare_archive_parent(root: Path, target: Path) -> None:
    """Create target parents while rejecting pre-existing symlink hops."""
    relative_parent = target.parent.relative_to(root)
    current = root
    for part in relative_parent.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"archive path traverses symlink: {current}")
        current.mkdir(exist_ok=True)
    if target.exists() and target.is_symlink():
        raise ValueError(f"archive target is a symlink: {target}")


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                target = _archive_target(root, member.filename)
                if target is None:
                    continue
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ValueError(f"archive symlink is not allowed: {member.filename!r}")
                if member.is_dir():
                    _prepare_archive_parent(root, target)
                    target.mkdir(exist_ok=True)
                    continue
                _prepare_archive_parent(root, target)
                if target.exists() and target.is_dir():
                    raise ValueError(f"archive file conflicts with directory: {member.filename!r}")
                with zf.open(member, "r") as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    elif archive_path.suffix in (".tar", ".gz", ".tgz", ".bz2"):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                target = _archive_target(root, member.name)
                if target is None:
                    continue
                if member.isdir():
                    _prepare_archive_parent(root, target)
                    target.mkdir(exist_ok=True)
                    continue
                if not member.isreg():
                    raise ValueError(
                        f"unsupported archive member type: {member.name!r}"
                    )
                source = tf.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                _prepare_archive_parent(root, target)
                if target.exists() and target.is_dir():
                    raise ValueError(f"archive file conflicts with directory: {member.name!r}")
                with source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    else:
        print(f"[WARN] Unknown archive format: {archive_path.suffix}, skipping extraction.")


def download_from_kaggle(dataset_id: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset_id, "-p", str(dest_dir), "--unzip",
    ]
    print(f"[Kaggle] Downloading {dataset_id} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Kaggle] stdout: {result.stdout}")
        print(f"[Kaggle] stderr: {result.stderr}")
        raise RuntimeError(
            f"Kaggle download failed for {dataset_id}. "
            "Ensure KAGGLE_USERNAME and KAGGLE_KEY are set, "
            "or place kaggle.json in ~/.kaggle/"
        )
    print(f"[Kaggle] Downloaded to {dest_dir}")


def download_from_gdrive(file_id: str, dest_dir: Path, filename: str = "data.zip") -> Path:
    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)
    output = dest_dir / _safe_download_filename(filename)
    partial = output.with_name(f"{output.name}.part")
    partial.unlink(missing_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        result = gdown.download(url, str(partial), quiet=False)
        if not result or not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Google Drive download failed for file {file_id}")
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if output.suffix == ".zip":
        _extract_archive(output, dest_dir)
    return output


def download_from_url(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    filename = _safe_download_filename(filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    output = dest_dir / filename
    if output.exists():
        print(f"[URL] {output} already exists, skipping.")
        return output
    _download_file(url, output)
    if output.suffix in (".zip", ".tar", ".gz", ".tgz"):
        _extract_archive(output, dest_dir)
    return output


def download_from_github(repo_url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(dest_dir)]
    print(f"[GitHub] Cloning {repo_url} ...")
    subprocess.run(cmd, check=True)


def download_dataset(name: str, root: str | Path = "datasets") -> Path:
    root = Path(root)
    info = DATASET_REGISTRY[name]
    dest = root / name

    if dest.exists() and any(dest.iterdir()):
        print(f"[{name}] Already downloaded at {dest}")
        return dest

    print(f"\n{'='*60}")
    print(f"Downloading: {info.name}")
    print(f"Description: {info.description}")
    print(f"Size: {info.size}")
    print(f"Source: {info.source.value}")
    print(f"{'='*60}\n")

    if info.source == DownloadSource.KAGGLE:
        download_from_kaggle(info.source_id, dest)

    elif info.source == DownloadSource.GDRIVE:
        download_from_gdrive(info.source_id, dest)

    elif info.source == DownloadSource.DIRECT_URL:
        download_from_url(info.url, dest)

    elif info.source == DownloadSource.GITHUB:
        download_from_github(info.url, dest)

    elif info.source == DownloadSource.ZENODO:
        if info.url:
            download_from_url(info.url, dest)
        else:
            _print_manual_instructions(info)

    elif info.source in (DownloadSource.MENDELEY, DownloadSource.INSTITUTIONAL):
        _print_manual_instructions(info)

    return dest


def _print_manual_instructions(info: DatasetInfo) -> None:
    print(f"\n[MANUAL DOWNLOAD REQUIRED] {info.name}")
    print(f"  URL: {info.url}")
    print(f"  Notes: {info.notes}")
    print(f"  Citation: {info.citation}")
    print(f"  Please download manually and place in datasets/{info.name.lower().replace(' ', '_')}/\n")


def download_all(
    root: str | Path = "datasets",
    skip_manual: bool = True,
) -> dict[str, Path]:
    root = Path(root)
    results = {}
    auto_sources = {DownloadSource.KAGGLE, DownloadSource.GDRIVE, DownloadSource.DIRECT_URL, DownloadSource.GITHUB}

    for name, info in DATASET_REGISTRY.items():
        if skip_manual and info.source not in auto_sources:
            _print_manual_instructions(info)
            continue
        try:
            path = download_dataset(name, root)
            results[name] = path
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Downloaded {len(results)} / {len(DATASET_REGISTRY)} datasets")
    print(f"{'='*60}")
    return results
