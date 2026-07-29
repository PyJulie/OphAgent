"""
Sandboxed Python execution for derived-metric computation.

The agent uses this when a clinical question requires combining outputs
of multiple prior tool calls in a way that isn't covered by any single
adapter — e.g. "lesion area within 3 mm of the macula", "hemispheric
asymmetry of RNFL", "fluid pocket distance from the foveal centre".

Design choices
--------------
* `exec(code, restricted_globals, {})` rather than `subprocess` keeps large
  arrays in-process. The exposed scientific modules are capability-restricted
  proxies, not the original modules.
* `__builtins__` is replaced with a small allow-list (no `open`, `eval`,
  `exec`, `compile`, `__import__`, `getattr`/`setattr` etc.).
* AST-level deny-list catches obvious escape attempts (`__class__`,
  `import`, dunders) before execution.
* A trace deadline interrupts pure-Python loops, while code length, AST size,
  and captured output are bounded.
* Read-only access to prior tool outputs through `tools` / `masks` /
  `landmarks` / `figures` namespaces, all materialised from
  `session.context.analyses` at call time.
* The single permitted side-effect is `save_figure(arr_or_pil, name)`
  which writes a PNG into `<session>/derived/` and surfaces the path
  back to the chat UI.
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import re
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any


log = logging.getLogger(__name__)


# Dangerous BARE NAMES (modules + escape-prone builtins). Checked via AST
# `Name` nodes so we match `os.getcwd()` / bare `eval` precisely WITHOUT the
# old substring false-positives (`pos.`, `cos.`, `re.compile(`, `Image.open(`
# are now fine — they are attribute access on safe objects, not these names).
_DANGER_NAMES = frozenset({
    "os", "sys", "socket", "subprocess", "shutil", "importlib", "ctypes",
    "builtins", "open", "input", "breakpoint", "eval", "exec", "compile",
    "globals", "locals", "vars", "delattr", "setattr", "getattr",
    "__import__", "__builtins__", "exit", "quit",
})
# Dangerous ATTRIBUTE names — the classic introspection-escape chain.
_DANGER_ATTRS = frozenset({
    "__class__", "__bases__", "__base__", "__mro__", "__subclasses__",
    "__globals__", "__builtins__", "__dict__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__import__", "__subclasshook__",
    "__init_subclass__", "__code__", "__closure__",
    # File, process, native-library, and serialization surfaces reachable via
    # numpy arrays or scientific modules even when bare open() is unavailable.
    "open", "read", "write", "fromfile", "tofile", "fromregex",
    "load", "loads", "loadtxt", "genfromtxt", "save", "savez",
    "savez_compressed", "savetxt", "dump", "dumps", "memmap",
    "open_memmap", "datasource", "ctypes", "ctypeslib", "load_library",
    "f2py", "imread", "imwrite", "videocapture", "system", "popen",
    "read_csv", "read_excel", "read_json", "read_html", "read_parquet",
    "read_pickle", "to_csv", "to_excel", "to_json", "to_parquet",
    "to_pickle", "urlopen", "request",
})

_MAX_CODE_CHARS = 20_000
_MAX_AST_NODES = 2_000
_MAX_STDOUT_CHARS = 200_000


def _check_code_safety(code: str) -> str | None:
    """Static scan. Returns a violation message or None if OK.

    Not bullet-proof against a determined adversary — but combined with
    the empty __builtins__ + whitelisted-import sandbox below it raises the
    bar far enough that an honest LLM never trips and a hostile one needs
    nontrivial obfuscation.

    Imports are unnecessary because the safe ``np`` and ``ndi`` proxies are
    already present, and are blocked to avoid exposing module file/network
    helpers through otherwise scientific packages.
    """
    if len(code) > _MAX_CODE_CHARS:
        return f"code exceeds {_MAX_CODE_CHARS} characters"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # let exec surface the real syntax error with a clear msg
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        return f"code exceeds {_MAX_AST_NODES} syntax nodes"
    for node in nodes:
        # Dangerous bare names (os/sys/eval/open/...) — precise, no substrings.
        if isinstance(node, ast.Name) and node.id in _DANGER_NAMES:
            return f"forbidden name in code: {node.id!r}"
        # Introspection-escape attribute access (x.__class__, ...).
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr.lower() in _DANGER_ATTRS
        ):
            return f"forbidden attribute access: {node.attr!r}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are disabled; use the provided np and ndi namespaces"
        if isinstance(node, ast.While):
            return "while loops are disabled in compute()"
    return None


class _SafeModuleProxy:
    """Expose numeric module attributes while denying capability-bearing APIs."""

    __slots__ = ("_module", "_submodules", "_name")

    def __init__(self, module: ModuleType, *, submodules: tuple[str, ...] = ()):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_submodules", frozenset(submodules))
        object.__setattr__(self, "_name", module.__name__)

    def __getattr__(self, name: str):
        lower = name.lower()
        if name.startswith("_") or lower in _DANGER_ATTRS:
            raise AttributeError(f"{self._name}.{name} is unavailable in compute()")
        value = getattr(self._module, name)
        if isinstance(value, ModuleType):
            if name not in self._submodules:
                raise AttributeError(f"{self._name}.{name} is unavailable in compute()")
            return _SafeModuleProxy(value)
        return value

    def __repr__(self) -> str:
        return f"<safe module {self._name}>"


class _LimitedTextIO(io.StringIO):
    def __init__(self, limit: int):
        super().__init__()
        self._limit = limit
        self._written = 0

    def write(self, value: str) -> int:
        text = str(value)
        remaining = self._limit - self._written
        if remaining <= 0:
            raise RuntimeError("compute output limit exceeded")
        if len(text) > remaining:
            super().write(text[:remaining])
            self._written = self._limit
            raise RuntimeError("compute output limit exceeded")
        written = super().write(text)
        self._written += written
        return written


def _safe_builtins() -> dict[str, Any]:
    """Hand-rolled allow-list of the Python builtins LLMs need for numeric work."""
    import builtins as _b

    keep = (
        "abs", "all", "any", "bool", "callable", "chr", "complex", "dict",
        "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "next", "object", "oct", "ord",
        "pow", "print", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None",
        # Exception classes — needed for try/except blocks the LLM writes.
        # These are inert classes (no capability); the introspection-escape
        # paths off them (__subclasses__ etc.) are blocked by _DANGER_ATTRS.
        "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "ZeroDivisionError", "ArithmeticError",
        "OverflowError", "FloatingPointError", "RuntimeError", "StopIteration",
        "AssertionError", "NotImplementedError", "NameError", "LookupError",
        "ImportError", "Warning", "RuntimeWarning", "UserWarning",
    )
    out: dict[str, Any] = {k: getattr(_b, k) for k in keep if hasattr(_b, k)}
    return out


# ── context materialisation ────────────────────────────────────────────────
def _gather_landmarks_and_masks(
    analyses: dict[str, dict[str, dict]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Walk `session.context.analyses` and pull out
       (landmarks, mask_paths) — both keyed by short canonical names.

    analyses schema:  {image_path: {tool_name: jsonified_AdapterResult, ...}, ...}
    """
    landmarks: dict[str, Any] = {}
    mask_paths: dict[str, str] = {}

    for img_path, by_tool in analyses.items():
        landmarks.setdefault("image_paths", {})[Path(img_path).stem] = img_path
        for tool_name, payload in by_tool.items():
            if not isinstance(payload, dict):
                continue
            preds = payload.get("predictions") or {}
            figs  = payload.get("figures") or {}

            # ── CFP od/macula detector — extract centroids ──
            if tool_name == "cfp_od_detection":
                # Prefer the post-processed `best` dict — it already handles the
                # joint-bbox fovea-inference fallback when direct detection is
                # missing or low-confidence.
                best = preds.get("best") or {}
                if "OD" in best and best["OD"].get("center"):
                    xy = tuple(best["OD"]["center"])
                    landmarks["od_center_px"] = xy
                    landmarks["od_center_xy"] = xy
                    landmarks["od_center_yx"] = (xy[1], xy[0])
                if "Fovea" in best and best["Fovea"].get("center"):
                    xy = tuple(best["Fovea"]["center"])
                    landmarks["macula_center_px"] = xy
                    landmarks["macula_center_xy"] = xy
                    landmarks["macula_center_yx"] = (xy[1], xy[0])
                    if best["Fovea"].get("inferred_from"):
                        landmarks["macula_center_inferred"] = True
                if "joint" in best and best["joint"].get("center"):
                    landmarks["od_macula_joint_center_px"] = tuple(best["joint"]["center"])

            # ── retsam mask channels (PNG files on disk, binary) ──
            if tool_name == "cfp_retsam_segmentation":
                mf = preds.get("mask_files") or {}
                for ch, path in mf.items():
                    mask_paths[f"retsam.{ch}"] = path

            # ── fluid / layer seg — int mask saved as .npy ──
            if tool_name in ("oct_fluid_segmentation", "oct_layer_segmentation"):
                if "mask_npy" in figs:
                    short = tool_name.replace("oct_", "")
                    mask_paths[short] = figs["mask_npy"]

            # ── OCT volume disc analysis — landmarks ──
            if tool_name == "oct_volume_disc":
                if preds.get("disc_centroid_px"):
                    landmarks["oct_disc_centroid_px"] = tuple(preds["disc_centroid_px"])
                if preds.get("cpRNFLT_sectors"):
                    landmarks["cpRNFLT_sectors"] = dict(preds["cpRNFLT_sectors"])
                if preds.get("rnfl_tsni"):
                    landmarks["rnfl_tsni"] = dict(preds["rnfl_tsni"])

    return landmarks, mask_paths


def _load_mask(path: str):
    """Load a mask file — supports .npy and .png. Returns numpy array."""
    import numpy as np
    from PIL import Image
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mask file missing: {path}")
    if p.suffix.lower() == ".npy":
        return np.load(p)
    arr = np.asarray(Image.open(p))
    # PNG masks are typically uint8 binary (0/255). Boolean-ify if so.
    if arr.ndim == 2 and arr.dtype == np.uint8 and arr.max() <= 1:
        return arr.astype(bool)
    if arr.ndim == 2 and arr.dtype == np.uint8 and set(np.unique(arr).tolist()) <= {0, 255}:
        return (arr > 127)
    return arr


# ── main entry ─────────────────────────────────────────────────────────────
def run_compute(
    code: str,
    session,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Execute `code` in a sandbox with prior-tool context.

    Returns a JSON-friendly dict:
      {
        "ok": bool,
        "stdout": str,
        "error": str | None,
        "saved_figures": {name: path, ...},
        "exposed": {"masks": [names], "landmarks": [names], "tools": [names]},
        "elapsed_s": float,
      }
    """
    timeout_s = max(0.1, min(float(timeout_s), 30.0))
    violation = _check_code_safety(code)
    if violation:
        return {"ok": False, "error": violation, "stdout": "",
                "saved_figures": {}, "exposed": {}, "elapsed_s": 0.0}

    import numpy as np
    import scipy.ndimage as ndi   # noqa: F401  - exposed to sandbox

    analyses = getattr(getattr(session, "context", None), "analyses", {}) or {}
    landmarks, mask_paths = _gather_landmarks_and_masks(analyses)
    tools_view = {
        tool_name: (payload.get("predictions") or {})
        for img_payload in analyses.values()
        for tool_name, payload in img_payload.items()
        if isinstance(payload, dict)
    }
    figures_view = {
        tool_name: (payload.get("figures") or {})
        for img_payload in analyses.values()
        for tool_name, payload in img_payload.items()
        if isinstance(payload, dict)
    }

    # Lazy-load masks on first attribute access via a dict subclass.
    class _MaskDict(dict):
        def __getitem__(self, key):
            v = super().__getitem__(key)
            if isinstance(v, str):
                arr = _load_mask(v)
                super().__setitem__(key, arr)
                return arr
            return v
        def get(self, key, default=None):
            if key not in self:
                return default
            return self[key]
        def keys(self):
            return list(super().keys())
        def __repr__(self):
            return f"<masks: {list(super().keys())}>"
    masks = _MaskDict(mask_paths)

    # Saved-figure sink
    derived_dir = Path(session.workspace) / session.session_id / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    saved_figures: dict[str, str] = {}

    def _safe_figure_name(name: str) -> str:
        raw = Path(str(name or "figure")).name
        stem = Path(raw).stem or "figure"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
        return (stem or "figure")[:80]

    def save_figure(arr_or_pil, name: str):
        """Persist a derived image into <session>/derived/<name>.png and
        register it so the chat UI renders it."""
        from PIL import Image
        if hasattr(arr_or_pil, "save") and not hasattr(arr_or_pil, "shape"):
            img = arr_or_pil
        else:
            a = np.asarray(arr_or_pil)
            if a.dtype == bool:
                a = (a.astype(np.uint8) * 255)
            elif a.dtype != np.uint8:
                a = a.astype(np.float32)
                a -= a.min()
                if a.max() > 0:
                    a = (255.0 * a / a.max()).astype(np.uint8)
                else:
                    a = a.astype(np.uint8)
            img = Image.fromarray(a)
        safe_name = _safe_figure_name(name)
        path = (derived_dir / f"{safe_name}.png").resolve()
        try:
            path.relative_to(derived_dir.resolve())
        except ValueError:
            raise ValueError("figure name escapes the derived directory")
        img.save(path)
        saved_figures[safe_name] = str(path)
        return str(path)

    # Pre-load the session's CURRENT image as a numpy RGB array so the
    # agent can do "overlay X on the original" without needing PIL/cv2
    # imports (which the AST scanner forbids). Falls back to None if no
    # image is registered. Sized HxWx3 uint8 RGB.
    original_image = None
    orig_image_path = (
        session.context.current_image
        if session is not None and getattr(session, "context", None) is not None
        else None
    )
    if orig_image_path:
        try:
            from PIL import Image as _PILImage
            resolved_image_path = session.resolve_session_file(orig_image_path)
            _arr = np.array(_PILImage.open(resolved_image_path).convert("RGB"))
            original_image = _arr  # RGB uint8, HxWx3
            orig_image_path = session.session_file_reference(resolved_image_path)
        except Exception as _e:
            log.warning(f"compute: failed to load original image: {_e}")

    def load_image(path):
        """Load a session-authorized image path → HxWx3 uint8 RGB numpy array.
        Useful when the agent wants to overlay on a non-focus image."""
        from PIL import Image as _PILImage
        safe_path = session.resolve_session_file(path)
        return np.array(_PILImage.open(safe_path).convert("RGB"))

    def blend(base, color_layer, alpha=0.5, where=None):
        """Convenience: alpha-blend `color_layer` onto `base`.
        `color_layer` may be a full HxWx3 image OR a single RGB triplet like
        [255, 220, 0] (broadcast to the base shape). If `where` mask (HxW bool)
        is given, blend only there; outside stays as base. Returns uint8 RGB."""
        b = base.astype(np.float32)
        c = np.asarray(color_layer, dtype=np.float32)
        # A single colour (shape (3,) or scalar) → broadcast to base's shape so
        # `c[where]` works for a 2-D mask. Without this, indexing a 1-D colour
        # with a 2-D mask raises "too many indices for array: 1-dimensional".
        if c.ndim < b.ndim:
            c = np.broadcast_to(c, b.shape).astype(np.float32)
        out = b.copy()
        if where is None:
            out = (1 - alpha) * b + alpha * c
        else:
            out[where] = (1 - alpha) * b[where] + alpha * c[where]
        return np.clip(out, 0, 255).astype(np.uint8)

    sandbox_globals: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "np": _SafeModuleProxy(
            np,
            submodules=("linalg", "fft", "random", "ma", "char", "polynomial"),
        ),
        "ndi": _SafeModuleProxy(ndi),
        "tools": tools_view,
        "figures": figures_view,
        "masks": masks,
        "landmarks": landmarks,
        "original_image": original_image,          # HxWx3 RGB uint8 or None
        "original_image_path": orig_image_path,
        "load_image": load_image,
        "blend": blend,
        "save_figure": save_figure,
    }
    sandbox_locals: dict[str, Any] = {}

    stdout_buf = _LimitedTextIO(_MAX_STDOUT_CHARS)
    holder: dict[str, Any] = {"err": None}
    deadline = time.monotonic() + timeout_s

    def runner():
        import sys as _sys

        def deadline_trace(frame, event, arg):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"compute exceeded {timeout_s:.1f}s")
            return deadline_trace

        _sys.settrace(deadline_trace)
        try:
            with contextlib.redirect_stdout(stdout_buf):
                with contextlib.redirect_stderr(stdout_buf):
                    exec(code, sandbox_globals, sandbox_locals)
        except SystemExit as e:
            holder["err"] = f"SystemExit blocked: {e}"
        except Exception as e:
            holder["err"] = f"{type(e).__name__}: {e}"
        finally:
            _sys.settrace(None)

    t0 = time.time()
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    th.join(timeout=timeout_s + 0.25)
    elapsed = time.time() - t0
    if th.is_alive():
        holder["err"] = f"TimeoutError: compute exceeded {timeout_s:.0f}s"

    return {
        "ok": holder["err"] is None,
        "error": holder["err"],
        "stdout": stdout_buf.getvalue(),
        "saved_figures": saved_figures,
        "exposed": {
            "masks": list(mask_paths.keys()),
            "landmarks": list(landmarks.keys()),
            "tools": list(tools_view.keys()),
        },
        "elapsed_s": round(elapsed, 3),
    }
