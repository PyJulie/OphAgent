"""Path helpers for release-safe optional model dependencies.

OphAgent wraps several external ophthalmic model repositories. The public code
must not contain workstation-specific absolute paths, so adapters resolve those
locations through environment variables with repository-relative fallbacks.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(os.environ.get("OPHAGENT_ROOT", str(REPO_ROOT))).resolve()
BUNDLED_COMPONENTS_DIR = Path(__file__).resolve().parents[1] / "components"
DEFAULT_RUNTIME_DIR = (Path.home() / ".ophagent").resolve()

# Keep mutable and private state outside the source checkout by default. Local
# deployments can point one variable at a different runtime directory containing
# credentials, weights, external model sources, caches, and generated output.
RUNTIME_DIR = Path(
    os.environ.get("OPHAGENT_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))
).resolve()
CKPT_DIR = Path(
    os.environ.get("OPHAGENT_CKPT_DIR", str(RUNTIME_DIR / "checkpoints"))
).resolve()
EXTERNAL_DIR = Path(
    os.environ.get("OPHAGENT_EXTERNAL_DIR", str(RUNTIME_DIR / "external"))
).resolve()
OUTPUT_DIR = Path(
    os.environ.get("OPHAGENT_OUTPUT_DIR", str(RUNTIME_DIR / "reports"))
).resolve()
CACHE_DIR = Path(
    os.environ.get("OPHAGENT_CACHE_DIR", str(RUNTIME_DIR / "cache"))
).resolve()
ENV_FILE = Path(
    os.environ.get("OPHAGENT_ENV_FILE", str(RUNTIME_DIR / ".env"))
).resolve()


def _expand(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def repo_path(*parts: str) -> Path:
    """Return a path inside the current OphAgent checkout."""
    return (RELEASE_ROOT / Path(*parts)).resolve()


def runtime_path(*parts: str) -> Path:
    """Return a path inside the deployment's private runtime directory."""
    return (RUNTIME_DIR / Path(*parts)).resolve()


def output_path(*parts: str) -> Path:
    """Return a path under the configured generated-output directory."""
    return (OUTPUT_DIR / Path(*parts)).resolve()


def cache_path(*parts: str) -> Path:
    """Return a path under the configured runtime cache directory."""
    return (CACHE_DIR / Path(*parts)).resolve()


def checkpoint_file(env_var: str, *default_parts: str) -> Path:
    """Resolve a model file under checkpoints/, with an env override."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return _expand(value)
    return (CKPT_DIR / Path(*default_parts)).resolve()


def external_dir(env_var: str, *default_parts: str) -> Path:
    """Resolve a bundled external runtime source tree."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return _expand(value)
    return (EXTERNAL_DIR / Path(*default_parts)).resolve()


def bundled_component_dir(env_var: str, *default_parts: str) -> Path:
    """Resolve bundled inference source, with an authorised local override."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return _expand(value)
    return (BUNDLED_COMPONENTS_DIR / Path(*default_parts)).resolve()


def external_path(env_var: str, *default_parts: str) -> Path:
    """Resolve an optional external project directory.

    If ``env_var`` is set, it wins. Otherwise a repository-relative default is
    returned, usually under ``external/``. The path is not required to exist at
    import time; adapters raise a clear error only when the tool is loaded.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return _expand(value)
    return repo_path(*default_parts)


def external_file(env_var: str, *default_parts: str) -> Path:
    """Resolve an optional external file such as a checkpoint."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return _expand(value)
    return repo_path(*default_parts)


def first_existing_path(env_var: str, candidates: list[Path]) -> Path:
    """Return an env override, the first existing candidate, or the first path.

    Returning the first candidate when none exists lets the caller raise a
    deterministic FileNotFoundError that explains the expected location.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return _expand(value)
    for path in candidates:
        if path.exists():
            return path
    if not candidates:
        raise ValueError("first_existing_path() requires at least one candidate")
    return candidates[0]
