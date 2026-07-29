"""Build and audit a source-only OphAgent public release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MANIFEST_NAME = "RELEASE_MANIFEST.json"
BLOCKED_PATH_PARTS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".openai",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "annotator",
    "build",
    "cache",
    "checkpoints",
    "config",
    "datasets",
    "dist",
    "external",
    "inputs",
    "logs",
    "paper",
    "reports",
    "results",
    "review_workspace",
    "runtime",
    "tmp",
    "validation",
    "venv",
    "weights",
}
BLOCKED_SUFFIXES = {
    ".7z",
    ".bmp",
    ".ckpt",
    ".dcm",
    ".gif",
    ".jpeg",
    ".jpg",
    ".key",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tif",
    ".tiff",
    ".zip",
}
SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "GitHub token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"""(?ix)
        \b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b
        \s*[:=]\s*
        ["'][^"'{}\r\n]{8,}["']
        """
    ),
    "environment credential value": re.compile(
        r"""(?mx)
        ^[A-Z][A-Z0-9_]*
        (?:API_KEY|TOKEN|PASSWORD|SECRET)
        [A-Z0-9_]*=[A-Za-z0-9_./+=-]{8,}\s*$
        """
    ),
}
LOCAL_PATH_PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    "Windows network path": re.compile(r"(?<!\\)\\{2}[A-Za-z0-9._-]+[\\/]"),
    "macOS user path": re.compile("/" + r"Users/[^/\s]+/"),
    "Linux user path": re.compile("/" + r"home/[^/\s]+/"),
}
VERIFY_TRANSIENT_PATH_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
VERIFY_TRANSIENT_DIR_SUFFIXES = {".dist-info", ".egg-info"}
VERIFY_TRANSIENT_SUFFIXES = {".pyc", ".pyo"}


class ReleaseError(RuntimeError):
    """A public release cannot be built or verified safely."""


def _git(repo_root: Path, *args: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
            errors=None if binary else "replace",
        )
    except FileNotFoundError as exc:
        raise ReleaseError("Git is required to build a public release") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            detail = stderr.decode("utf-8", errors="replace")
        else:
            detail = stderr or str(exc)
        raise ReleaseError(detail.strip()) from exc
    return completed.stdout


def find_repo_root(start: Path | None = None) -> Path:
    root = Path(start or Path.cwd()).resolve()
    output = _git(root, "rev-parse", "--show-toplevel")
    return Path(str(output).strip()).resolve()


def _git_paths(repo_root: Path, *, include_untracked: bool) -> list[PurePosixPath]:
    args = ["ls-files", "-z", "--cached"]
    if include_untracked:
        args.extend(["--others", "--exclude-standard"])
    raw = _git(repo_root, *args, binary=True)
    assert isinstance(raw, bytes)
    paths = {
        PurePosixPath(item.decode("utf-8", errors="strict"))
        for item in raw.split(b"\0")
        if item
    }
    return sorted(paths, key=lambda path: path.as_posix())


def _worktree_state(repo_root: Path) -> tuple[bool, str]:
    status = str(_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"))
    return bool(status.strip()), status


def _blocked_suffix(path: PurePosixPath) -> str | None:
    name = path.name.casefold()
    for suffix in sorted(BLOCKED_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return None


def _safe_relative_path(path: PurePosixPath) -> str | None:
    text = path.as_posix()
    if path.is_absolute() or not text or ".." in path.parts or "\\" in text:
        return "path is not a portable repository-relative path"
    return None


def _read_text_for_scan(path: Path) -> str | None:
    if path.stat().st_size > 5 * 1024 * 1024:
        return None
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _local_path_markers(repo_root: Path) -> list[str]:
    candidates = {str(repo_root), str(Path.home())}
    markers: set[str] = set()
    for candidate in candidates:
        markers.add(candidate.casefold())
        markers.add(candidate.replace("\\", "/").casefold())
    return sorted(marker for marker in markers if marker)


def audit_files(
    root: Path,
    relative_paths: Iterable[PurePosixPath],
    *,
    local_reference_root: Path | None = None,
) -> list[dict[str, str]]:
    root = root.resolve()
    findings: list[dict[str, str]] = []
    local_markers = _local_path_markers(local_reference_root or root)
    for relative in sorted(set(relative_paths), key=lambda path: path.as_posix()):
        display = relative.as_posix()
        path_error = _safe_relative_path(relative)
        if path_error:
            findings.append({"path": display, "kind": "unsafe-path", "detail": path_error})
            continue
        lower_parts = {part.casefold() for part in relative.parts}
        blocked_parts = sorted(lower_parts & BLOCKED_PATH_PARTS)
        if blocked_parts:
            findings.append(
                {
                    "path": display,
                    "kind": "blocked-directory",
                    "detail": ", ".join(blocked_parts),
                }
            )
            continue
        if relative.name.casefold() == ".env":
            findings.append(
                {"path": display, "kind": "credential-file", "detail": ".env is private"}
            )
            continue
        suffix = _blocked_suffix(relative)
        if suffix:
            findings.append(
                {
                    "path": display,
                    "kind": "blocked-file-type",
                    "detail": suffix,
                }
            )
            continue

        source = (root / Path(*relative.parts)).resolve()
        if source != root and root not in source.parents:
            findings.append(
                {
                    "path": display,
                    "kind": "unsafe-path",
                    "detail": "resolved path escapes release root",
                }
            )
            continue
        if source.is_symlink():
            findings.append(
                {
                    "path": display,
                    "kind": "symlink",
                    "detail": "symlinks are not included in public releases",
                }
            )
            continue
        if not source.is_file():
            findings.append(
                {
                    "path": display,
                    "kind": "missing-file",
                    "detail": "candidate is not a regular file",
                }
            )
            continue

        text = _read_text_for_scan(source)
        if text is None:
            continue
        folded = text.casefold()
        for label, pattern in LOCAL_PATH_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    {"path": display, "kind": "local-path", "detail": label}
                )
                break
        for marker in local_markers:
            if marker in folded:
                findings.append(
                    {
                        "path": display,
                        "kind": "local-path",
                        "detail": "contains a developer workstation path",
                    }
                )
                break
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    {"path": display, "kind": "secret", "detail": label}
                )
    return findings


def audit_tree(
    root: Path,
    *,
    local_reference_root: Path | None = None,
) -> list[dict[str, str]]:
    root = root.resolve()
    paths = [
        PurePosixPath(path.relative_to(root).as_posix())
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    return audit_files(
        root,
        paths,
        local_reference_root=local_reference_root,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verify_transient(path: str | PurePosixPath) -> bool:
    relative = path if isinstance(path, PurePosixPath) else PurePosixPath(path)
    folded_parts = {part.casefold() for part in relative.parts}
    if folded_parts & VERIFY_TRANSIENT_PATH_PARTS:
        return True
    if any(
        part.endswith(tuple(VERIFY_TRANSIENT_DIR_SUFFIXES))
        for part in folded_parts
    ):
        return True
    return relative.suffix.casefold() in VERIFY_TRANSIENT_SUFFIXES


def _release_metadata(
    repo_root: Path,
    release_root: Path,
    paths: Iterable[PurePosixPath],
    *,
    dirty: bool,
) -> dict[str, Any]:
    files = []
    for relative in sorted(paths, key=lambda path: path.as_posix()):
        target = release_root / Path(*relative.parts)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    return {
        "schema_version": 1,
        "project": "OphAgent",
        "release_version": _project_version(repo_root),
        "source_commit": str(_git(repo_root, "rev-parse", "HEAD")).strip(),
        "source_dirty": dirty,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": "tracked-source-only",
        "excluded_categories": [
            "credentials",
            "model weights",
            "external source checkouts",
            "datasets and clinical inputs",
            "sessions, logs, reports, caches, and generated results",
            "Git history and developer metadata",
        ],
        "file_count": len(files),
        "files": files,
    }


def _project_version(repo_root: Path) -> str:
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', pyproject)
    if not match:
        raise ReleaseError("Cannot determine project version from pyproject.toml")
    return match.group(1)


def _write_zip(release_root: Path) -> Path:
    # Versioned release directories commonly end in a dotted semantic version
    # (for example, ``OphAgent-0.1.0``). ``Path.with_suffix`` would interpret
    # ``.0`` as a file extension and silently truncate the archive name.
    archive = release_root.with_name(f"{release_root.name}.zip")
    if archive.exists():
        raise ReleaseError(f"Archive already exists: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(release_root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(release_root.parent)
                output.write(path, relative.as_posix())
    return archive


def build_release(
    *,
    repo_root: Path,
    output: Path,
    allow_dirty: bool = False,
    make_zip: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    if output == repo_root or repo_root in output.parents or output in repo_root.parents:
        raise ReleaseError("Release output must be outside the source repository")
    if output.exists():
        raise ReleaseError(f"Release output already exists: {output}")

    dirty, status = _worktree_state(repo_root)
    if dirty and not allow_dirty:
        raise ReleaseError(
            "The Git worktree is not clean. Commit or remove pending files before "
            "building a formal release; use --allow-dirty only for a local preview."
        )
    paths = _git_paths(repo_root, include_untracked=allow_dirty)
    findings = audit_files(repo_root, paths, local_reference_root=repo_root)
    if findings:
        raise ReleaseError(
            "Public release audit failed:\n"
            + "\n".join(
                f"- {item['path']}: {item['kind']} ({item['detail']})"
                for item in findings
            )
        )

    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if partial.exists():
        raise ReleaseError(f"Temporary release path already exists: {partial}")
    partial.mkdir(parents=True)
    try:
        for relative in paths:
            source = repo_root / Path(*relative.parts)
            destination = partial / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        metadata = _release_metadata(
            repo_root,
            partial,
            paths,
            dirty=dirty,
        )
        (partial / MANIFEST_NAME).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        post_findings = audit_tree(partial, local_reference_root=repo_root)
        if post_findings:
            raise ReleaseError(
                "Staged release audit failed:\n"
                + "\n".join(
                    f"- {item['path']}: {item['kind']} ({item['detail']})"
                    for item in post_findings
                )
            )
        partial.replace(output)
    except Exception:
        resolved_partial = partial.resolve()
        if (
            resolved_partial.exists()
            and resolved_partial.parent == output.parent
            and resolved_partial.name.startswith(f".{output.name}.partial-")
        ):
            shutil.rmtree(resolved_partial)
        raise

    verification = verify_release(output)
    archive = _write_zip(output) if make_zip else None
    return {
        "release_root": str(output),
        "archive": str(archive) if archive else None,
        "source_dirty": dirty,
        "worktree_status": status if dirty else "",
        **verification,
    }


def verify_release(release_root: Path) -> dict[str, Any]:
    release_root = release_root.resolve()
    manifest_path = release_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ReleaseError(f"Missing {MANIFEST_NAME}: {release_root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Invalid release manifest: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ReleaseError("Unsupported release manifest schema")

    expected = {
        str(item["path"]): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    actual = {
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file()
        and not _is_verify_transient(path.relative_to(release_root).as_posix())
    }
    unlisted = sorted(actual - set(expected) - {MANIFEST_NAME})
    missing = sorted(set(expected) - actual)
    mismatches: list[str] = []
    for relative, item in expected.items():
        path = release_root / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            continue
        if path.stat().st_size != item.get("size_bytes"):
            mismatches.append(f"{relative}: size")
        elif _sha256(path) != item.get("sha256"):
            mismatches.append(f"{relative}: SHA-256")

    findings = [
        finding
        for finding in audit_tree(release_root, local_reference_root=release_root)
        if not _is_verify_transient(finding["path"])
    ]
    if unlisted or missing or mismatches or findings:
        details = []
        details.extend(f"unlisted file: {item}" for item in unlisted)
        details.extend(f"missing file: {item}" for item in missing)
        details.extend(f"integrity mismatch: {item}" for item in mismatches)
        details.extend(
            f"audit finding: {item['path']} ({item['kind']}: {item['detail']})"
            for item in findings
        )
        raise ReleaseError("Release verification failed:\n" + "\n".join(details))
    return {
        "verified": True,
        "file_count": len(expected),
        "source_commit": manifest.get("source_commit"),
        "source_dirty": bool(manifest.get("source_dirty")),
    }


def _print_findings(findings: list[dict[str, str]]) -> None:
    if not findings:
        print("Public release audit: PASS")
        return
    print("Public release audit: FAIL")
    for finding in findings:
        print(
            f"- {finding['path']}: {finding['kind']} "
            f"({finding['detail']})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ophagent-release",
        description="Audit, build, and verify a source-only public release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Audit tracked and non-ignored source candidates",
    )
    audit.add_argument("--repo", type=Path, default=None)
    audit.add_argument("--json", action="store_true", dest="as_json")

    build = subparsers.add_parser(
        "build",
        help="Copy approved source into a new directory outside the repository",
    )
    build.add_argument("--repo", type=Path, default=None)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--zip", action="store_true", dest="make_zip")
    build.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build a marked local preview from modified/untracked source files",
    )
    build.add_argument("--json", action="store_true", dest="as_json")

    verify = subparsers.add_parser(
        "verify",
        help="Verify a built release against its integrity manifest",
    )
    verify.add_argument("release_root", type=Path)
    verify.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            repo_root = find_repo_root(args.repo)
            paths = _git_paths(repo_root, include_untracked=True)
            findings = audit_files(
                repo_root,
                paths,
                local_reference_root=repo_root,
            )
            payload = {
                "ok": not findings,
                "repo_root": str(repo_root),
                "candidate_files": len(paths),
                "findings": findings,
            }
            if args.as_json:
                print(json.dumps(payload, indent=2))
            else:
                _print_findings(findings)
            return 0 if not findings else 1

        if args.command == "build":
            repo_root = find_repo_root(args.repo)
            payload = build_release(
                repo_root=repo_root,
                output=args.output,
                allow_dirty=args.allow_dirty,
                make_zip=args.make_zip,
            )
        else:
            payload = verify_release(args.release_root)
        if args.as_json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                if value not in (None, ""):
                    print(f"{key}: {value}")
        return 0
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
