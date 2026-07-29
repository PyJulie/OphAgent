"""
Adapter: ReT-SAM 2.0 full-fundus multi-task segmentation and post-processing.

Invokes the bundled `scripts/infer.py` and `scripts/quantify.py` entry points.
The source can be overridden with ``OPHAGENT_RETSAM_SRC`` for development;
the separately distributed checkpoint is configured with
``OPHAGENT_RETSAM_CKPT``.

Outputs (per image):
  * 8 segmentation masks (artery/vein, OD/OC, tessellation, myopia,
    lesion_s1/s2/s3, possible_lesions)
  * Quantitative biomarkers via the `postprocess/` module: vessel CRAE/CRVE,
    A/V ratio, disc-cup geometry (CDR, ISNT compliance, DDLS), lesion areas
    by class + ETDRS zone, tessellation severity, myopia.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import CACHE_DIR, bundled_component_dir, checkpoint_file


RETSAM_SRC = bundled_component_dir("OPHAGENT_RETSAM_SRC", "retsam_2")
RETSAM_CKPT = checkpoint_file("OPHAGENT_RETSAM_CKPT", "cfp", "retsam.ckpt")

# 8-task output channel config (from retsam-2.0 README / scripts/infer.py)
DEFAULT_OUTPUT_CHANNELS = "(2,3,2,4,6,11,6,2)"

# ──────────────────────────────────────────────────────────────────────────
# Content-addressed mask cache
# ──────────────────────────────────────────────────────────────────────────
# Profiling (single 640² image, RTX-class GPU) showed a retsam call spends
# ~19.5 s of FIXED overhead and only ~0.7 s of real work:
#     import torch            2.1 s
#     import model + monai   10.5 s   (monai lazily pulls in tf/tensorboard/cupy)
#     load 10.8 GB ckpt→GPU   6.9 s
#     forward (all 8 heads)   0.72 s  ← the actual "branches"
#     quantify (all modules)  1.2 s
# i.e. the multi-task forward is cheap; the cost is reloading the 10.8 GB
# checkpoint in a fresh process on EVERY image. Skipping a task head saves
# almost nothing — but skipping the reload saves ~20 s/image.
#
# The segmentation MASKS depend only on the image bytes (the coordinate head
# is fixed-off, output_channels fixed), so they are content-addressable. We
# cache masks + infer_summary keyed on md5(image bytes). On a hit we skip the
# infer subprocess entirely and only re-run quantify (CPU, ~1 s) on the cached
# masks — quantify is re-run every call so eye_side / pixel_spacing stay exact.
# Pre-warm the cache once (scripts/prewarm_retsam.py loads the model once and
# batches every image) and an agent eval that calls retsam per image collapses
# from ~21 s/image to ~1.2 s/image.
#
# Cache is content-addressed → always returns identical masks; safe to leave on
# for webchat (repeat uploads become instant). Disable with RETSAM_CACHE=0.
RETSAM_CACHE_DIR = Path(os.environ.get("RETSAM_CACHE_DIR",
                                       str(CACHE_DIR / "retsam")))
RETSAM_CACHE_ON = os.environ.get("RETSAM_CACHE", "1") != "0"


# ──────────────────────────────────────────────────────────────────────────
# GPU concurrency control
# ──────────────────────────────────────────────────────────────────────────
# retsam runs as a Python subprocess that loads ~2-3 GB of weights into its
# OWN CUDA context. If multiple webchat sessions trigger retsam concurrently,
# the GPU sees 2×, 3×, 4× duplicate model copies stacked on top of every
# other model already resident (in-process CLIPs, Ollama's hot LLM, etc.) —
# the previously reported `retsam infer failed (rc=1) ... CUDA kernel
# launch failure` came from exactly this contention.
#
# Fix: serialise retsam invocations across all threads in this process.
# Concurrent users still get parallel CFP classifier calls (those are
# in-process and share already-loaded weights); only retsam is forced to
# queue. Average retsam wall-time is ~8-12 s, so up to 3 simultaneous
# users see at most ~24-36 s of queue wait — well within tolerance and
# vastly better than an OOM crash.
#
# `_RETSAM_LOCK` is module-scoped (one lock per worker process); a wait
# longer than `_RETSAM_LOCK_TIMEOUT_S` returns a graceful "busy" error
# rather than blocking forever.
_RETSAM_LOCK = threading.Lock()
_RETSAM_LOCK_TIMEOUT_S = 180   # 3 min — covers ~10 queued callers worst-case
_RETSAM_INFLIGHT = 0
_RETSAM_INFLIGHT_LOCK = threading.Lock()


def _retsam_inflight_count() -> int:
    """Best-effort counter for telemetry / debugging."""
    with _RETSAM_INFLIGHT_LOCK:
        return _RETSAM_INFLIGHT


class _RetsamGate:
    """Context manager that acquires the retsam lock with a timeout and
    tracks an inflight counter. Raises TimeoutError if the queue is too
    long."""

    def __init__(self, timeout_s: float = _RETSAM_LOCK_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._acquired = False
        self.waited_s = 0.0

    def __enter__(self):
        global _RETSAM_INFLIGHT
        t0 = time.time()
        self._acquired = _RETSAM_LOCK.acquire(timeout=self.timeout_s)
        self.waited_s = time.time() - t0
        if not self._acquired:
            raise TimeoutError(
                f"retsam queue is full — waited {self.waited_s:.0f}s for the "
                f"GPU lock and gave up. Try again in a few seconds."
            )
        with _RETSAM_INFLIGHT_LOCK:
            _RETSAM_INFLIGHT += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        global _RETSAM_INFLIGHT
        if self._acquired:
            with _RETSAM_INFLIGHT_LOCK:
                _RETSAM_INFLIGHT = max(0, _RETSAM_INFLIGHT - 1)
            _RETSAM_LOCK.release()
        return False


@register
class RetsamAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_retsam_segmentation",
        modality="CFP",
        task="segmentation",
        description=(
            "Multi-task fundus segmentation (retsam-2.0): artery/vein, optic "
            "disc & cup, tessellation, myopic features, and three lesion "
            "groups (DR/AMD/general). Followed by a quantitative biomarker "
            "module: vessel calibre (CRAE/CRVE), A/V ratio, vertical CDR, "
            "ISNT compliance, lesion counts/areas, tessellation grade. "
            "NOTE: macula localisation is disabled (unreliable), so "
            "macula-relative metrics (ETDRS-zone areas, distance-from-macula, "
            "*_involves_macula) are NOT reported — use cfp_od_detection / OCT "
            "for fovea-relative analysis."
        ),
        input_size=(640, 640),
        labels=[
            "artery_vein", "od_oc", "tessellation", "myopia",
            "lesion_s1", "lesion_s2", "lesion_s3", "possible_lesions",
        ],
        confidence_threshold=0.0,   # masks are pixel-wise; no scalar conf
        limitations=[
            "Macula localisation DISABLED (coordinate head off — was unreliable): no fovea coords, no ETDRS-zone / distance-from-macula metrics. Use cfp_od_detection / OCT for fovea.",
            "Heavyweight; ~5-15s per image on GPU; subprocess invocation",
            "Lesion confidence depends on image quality — check via cfp_eyeq first",
        ],
        cost_class="slow",
        source_dir=str(RETSAM_SRC),
    )

    def _load_impl(self) -> None:
        # No model is loaded in-process; we shell out. Just verify ckpt exists.
        if not RETSAM_CKPT.exists():
            raise FileNotFoundError(
                "ReT-SAM checkpoint not found. Set OPHAGENT_RETSAM_CKPT. "
                f"Expected: {RETSAM_CKPT}"
            )
        infer_py = RETSAM_SRC / "scripts" / "infer.py"
        quantify_py = RETSAM_SRC / "scripts" / "quantify.py"
        if not infer_py.exists() or not quantify_py.exists():
            raise FileNotFoundError(
                "ReT-SAM source scripts not found. Set OPHAGENT_RETSAM_SRC. "
                f"Expected: {infer_py} and {quantify_py}"
            )
        self._impl = "subprocess"

    def _predict_impl(self,
                      image_path: str,
                      eye_side: str | None = None,
                      pixel_spacing_um: float | None = None,
                      run_quantify: bool = True,
                      output_dir: str | None = None,
                      quantify_modules: list[str] | None = None,
                      **_) -> AdapterResult:
        """
        eye_side: "OS" or "OD" — required for full ETDRS zone analysis (otherwise
                   spatial stats degrade gracefully to None).
        pixel_spacing_um: physical scale — if known, CRAE/CRVE reported in µm.
        quantify_modules: restrict the biomarker post-processing to a subset of
                   {vessels, disc_cup, lesions, myopia, tessellation} via
                   quantify.py --only. For a single-disease screen pass just the
                   relevant module (glaucoma → ["disc_cup"], DR → ["lesions"]);
                   None = all modules. The segmentation masks are unaffected
                   (one shared backbone forward always produces all 8 heads), so
                   this only trims CPU post-processing + keeps the surfaced
                   evidence on-topic. Mask caching is independent of this.
        """
        image_path = str(Path(image_path).resolve())

        # Content hash of the image BYTES → cache key. The masks depend only on
        # the image (coordinate head fixed-off, output_channels fixed), so
        # identical bytes always produce identical masks → safe to cache.
        try:
            content_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
        except Exception:
            content_hash = None

        # Stage the image under an ASCII-safe name — retsam's cv2.imread can't
        # read CJK / non-latin paths on Windows. Used by the infer subprocess
        # (miss) AND by quantify (hit + miss, to reload the original image).
        in_root = Path(tempfile.mkdtemp(prefix="retsam_in_"))
        suffix = Path(image_path).suffix.lower() or ".png"
        stem_safe = "img_" + (content_hash[:10] if content_hash
                              else hashlib.md5(Path(image_path).stem.encode("utf-8")).hexdigest()[:10])
        staged = in_root / f"{stem_safe}{suffix}"
        shutil.copy2(image_path, staged)
        out_root = Path(tempfile.mkdtemp(prefix="retsam_out_"))
        image_outdir = out_root / stem_safe

        cache_entry = (RETSAM_CACHE_DIR / content_hash) if (RETSAM_CACHE_ON and content_hash) else None
        cache_hit = bool(cache_entry and (cache_entry / "masks").is_dir()
                         and any((cache_entry / "masks").glob("*.png")))

        gpu_wait_s = 0.0
        env = os.environ.copy()
        env["PYTHONPATH"] = str(RETSAM_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        # Force UTF-8 in BOTH directions: child emits UTF-8 (PYTHONIOENCODING)
        # and our pipe decodes UTF-8 with replacement (a stray byte never
        # collapses stderr to None on Windows GBK consoles).
        env["PYTHONIOENCODING"] = "utf-8"

        # ── Cache hit: reconstruct a working dir from cached masks; skip the
        # ~20 s model reload entirely. quantify still runs below on these masks
        # with THIS call's eye_side / pixel_spacing, so results stay exact. ──
        if cache_hit:
            try:
                (image_outdir / "masks").mkdir(parents=True, exist_ok=True)
                for p in (cache_entry / "masks").glob("*.png"):
                    shutil.copy2(p, image_outdir / "masks" / p.name)
                coords_src = cache_entry / "coords.json"
                if coords_src.exists():
                    shutil.copy2(coords_src, image_outdir / "coords.json")
                # Rewrite infer_summary.json so its image_path points at the
                # freshly staged ASCII copy (the original cached staged path is
                # long gone) — quantify reloads the original via cv2.imread.
                summary: dict[str, Any] = {}
                cs = cache_entry / "infer_summary.json"
                if cs.exists():
                    try:
                        summary = json.load(open(cs, encoding="utf-8"))
                    except Exception:
                        summary = {}
                summary["image_path"] = str(staged)
                with open(image_outdir / "infer_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False)
            except Exception:
                cache_hit = False  # fall through to a fresh infer

        # ── Cache miss: run the infer subprocess under the GPU gate, then
        # populate the cache for next time. ──
        if not cache_hit:
            infer_cmd = [
                sys.executable,
                str(RETSAM_SRC / "scripts" / "infer.py"),
                "--input_dir", str(in_root),
                "--output_dir", str(out_root),
                "--model_path", str(RETSAM_CKPT),
                "--output_channels", DEFAULT_OUTPUT_CHANNELS,
                # Macula-localisation branch DISABLED. retsam's coordinate-head
                # macula centre is unreliable (verified wrong on real cases,
                # esp. high-myopia / atrophic fundi), and it was driving the
                # ETDRS-zone / "X DD from macula" metrics. Turning the head off
                # means coords.json is not written → quantify.py degrades
                # gracefully (macula_center_yx=None, all *_involves_macula /
                # *_within_1dd_of_macula fields → None). Fovea localisation
                # should come from cfp_od_detection / OCT instead.
                # Rollback: restore "--has_coordinate_head", "--num_coordinates","2".
                "--no_coordinate_head",
                "--batch_size", "1",
                "--num_workers", "0",
                "--fp16",
            ]
            # ── GPU gate: serialise retsam subprocesses across all threads ──
            # See `_RetsamGate` docstring above (we used to OOM the GPU when two
            # webchat users triggered retsam in parallel). quantify is moved
            # OUTSIDE the gate (CPU-only) so the GPU frees as soon as infer ends.
            try:
                gate = _RetsamGate()
                with gate:
                    proc = subprocess.run(
                        infer_cmd,
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                        cwd=str(RETSAM_SRC), env=env, timeout=600,
                    )
                    gpu_wait_s = gate.waited_s
                    if proc.returncode != 0:
                        shutil.rmtree(out_root, ignore_errors=True)
                        shutil.rmtree(in_root, ignore_errors=True)
                        return AdapterResult(
                            success=False,
                            tool=self.metadata.name, modality="CFP", task="segmentation",
                            error=f"retsam infer failed (rc={proc.returncode}): "
                                  f"{((proc.stderr or proc.stdout or '') or '').strip()[-500:]}",
                            metadata={"gpu_wait_s": round(gate.waited_s, 2)},
                        )
                    # Find the per-image output subdir (infer.py names it by stem).
                    summaries = list(out_root.rglob("infer_summary.json"))
                    if not summaries:
                        shutil.rmtree(out_root, ignore_errors=True)
                        return AdapterResult(
                            success=False, tool=self.metadata.name, modality="CFP",
                            task="segmentation",
                            error=f"no infer_summary.json under {out_root}",
                            metadata={"gpu_wait_s": round(gate.waited_s, 2)},
                        )
                    image_outdir = summaries[0].parent
            except TimeoutError as e:
                # Graceful "busy" response — never block the chat loop forever.
                shutil.rmtree(out_root, ignore_errors=True)
                shutil.rmtree(in_root, ignore_errors=True)
                return AdapterResult(
                    success=False,
                    tool=self.metadata.name, modality="CFP", task="segmentation",
                    error=str(e),
                    metadata={"gpu_wait_s": round(_RETSAM_LOCK_TIMEOUT_S, 0),
                              "inflight": _retsam_inflight_count()},
                )
            # Populate the cache (pure file I/O, outside the gate).
            if cache_entry is not None:
                try:
                    self._save_to_cache(image_outdir, cache_entry)
                except Exception:
                    pass

        # ── quantify.py (CPU/numpy, no GPU) — run for BOTH hit & miss so
        # eye_side / pixel_spacing are always honoured. `quantify_modules`
        # restricts post-processing to the relevant module(s) for a
        # single-disease screen. ──
        quantify_data: dict[str, Any] = {}
        if run_quantify:
            qcmd = [
                sys.executable,
                str(RETSAM_SRC / "scripts" / "quantify.py"),
                "--infer_dir", str(out_root),
                "--no_viz",
                "--recursive",
            ]
            _mods = [m for m in (quantify_modules or []) if m]
            if _mods:
                qcmd += ["--only", *_mods]
            if eye_side:
                qcmd += ["--eye_side", eye_side]
            if pixel_spacing_um is not None:
                qcmd += ["--pixel_spacing_um", str(pixel_spacing_um)]
            qproc = subprocess.run(
                qcmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=str(RETSAM_SRC), env=env, timeout=300,
            )
            if qproc.returncode == 0:
                analysis_paths = list(image_outdir.glob("analysis.json"))
                if analysis_paths:
                    try:
                        with open(analysis_paths[0], encoding="utf-8") as f:
                            quantify_data = json.load(f)
                    except Exception:
                        pass

        # Pick up the mask files for visualisation
        mask_dir = image_outdir / "masks"
        mask_files = {p.stem: str(p) for p in mask_dir.glob("*.png")} if mask_dir.exists() else {}

        # If the caller gave us a persistent output_dir under the project root,
        # render colour overlays there so the chat UI can display them.
        overlays: dict[str, str] = {}
        if output_dir and mask_files:
            try:
                overlays = self._render_overlays(
                    image_path=image_path,
                    mask_files=mask_files,
                    out_dir=Path(output_dir) / f"retsam_{Path(image_path).stem}",
                )
            except Exception as _e:
                overlays = {"error": f"overlay rendering failed: {_e}"}

        # Build the high-level summary
        predictions: dict[str, Any] = {
            "mask_files": mask_files,
            "n_masks": len(mask_files),
            "overlay_files": overlays,
        }
        if quantify_data:
            # analysis.json schema: {meta, summary (flat top-line), modules: {...}}
            modules = quantify_data.get("modules", {})
            top_summary = quantify_data.get("summary", {})

            def _summarise(name: str) -> dict:
                m = modules.get(name, {}) or {}
                if not isinstance(m, dict):
                    return {}
                # Prefer the module's own 'summary' if present, else the full module dict
                return m.get("summary", m if name in ("disc_cup", "lesions", "myopia", "tessellation") else m)

            predictions["quantitative"] = {
                "top_line": top_summary,   # CRAE_px, CRVE_px, AVR, fractal dims, density
                "vessels": _summarise("vessels"),
                "disc_cup": _summarise("disc_cup"),
                "lesions": _summarise("lesions"),
                "tessellation": _summarise("tessellation"),
                "myopia": _summarise("myopia"),
            }
            predictions["meta"] = quantify_data.get("meta", {})

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="segmentation",
            predictions=predictions,
            figures={k: v for k, v in overlays.items() if k != "error"},
            metadata={
                "output_dir": str(image_outdir),
                "eye_side": eye_side,
                "pixel_spacing_um": pixel_spacing_um,
                # GPU concurrency telemetry: how long this call waited for the
                # retsam lock + how many callers are queued behind. cache_hit
                # means the ~20 s model reload was skipped (gpu_wait_s == 0).
                "gpu_wait_s": round(gpu_wait_s, 2),
                "cache_hit": cache_hit,
                "inflight": _retsam_inflight_count(),
            },
        )

    @staticmethod
    def _save_to_cache(image_outdir: "Path", cache_entry: "Path") -> None:
        """Copy this image's masks + infer_summary + coords into the
        content-addressed cache. Writes to a sibling temp dir then atomically
        renames into place, so a concurrent reader never sees a half-written
        entry. No-op if the entry already exists (another caller won the race).
        """
        if cache_entry.exists():
            return
        cache_entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=cache_entry.name + ".tmp",
                                    dir=str(cache_entry.parent)))
        try:
            (tmp / "masks").mkdir(parents=True, exist_ok=True)
            src_masks = image_outdir / "masks"
            if src_masks.is_dir():
                for p in src_masks.glob("*.png"):
                    shutil.copy2(p, tmp / "masks" / p.name)
            for fn in ("infer_summary.json", "coords.json"):
                s = image_outdir / fn
                if s.exists():
                    shutil.copy2(s, tmp / fn)
            try:
                os.replace(str(tmp), str(cache_entry))  # atomic rename
                tmp = None
            except OSError:
                pass  # lost the race / target exists — cached entry is fine
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _imread_unicode(path: str, flags: int | None = None):
        """`cv2.imread`-compatible loader that survives Windows CJK paths.

        OpenCV on Windows uses the narrow-char ANSI API for fopen() and
        chokes on any non-latin character in the path. We read the file
        bytes via Python (which handles utf-8 paths fine) and decode the
        buffer with `cv2.imdecode`. Returns ndarray or None.
        """
        import cv2
        import numpy as np
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        try:
            buf = np.fromfile(str(p), dtype=np.uint8)
            return cv2.imdecode(buf, flags if flags is not None else cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _render_overlays(image_path: str,
                         mask_files: dict[str, str],
                         out_dir: Path) -> dict[str, str]:
        """Composite each relevant mask onto the original CFP using a
        per-mask colour map and save PNGs to `out_dir`. Returns a dict
        {task_name: absolute_path}."""
        import cv2
        import numpy as np
        out_dir.mkdir(parents=True, exist_ok=True)

        original = RetsamAdapter._imread_unicode(image_path)
        if original is None:
            return {}
        H, W = original.shape[:2]

        # Per-task colour palette (BGR) — keys are mask filename stems
        palette = {
            "od_oc": [(0, 0, 0), (40, 220, 40), (0, 165, 255)],     # bg / disc-rim / cup
            "artery_vein": [(0, 0, 0), (60, 60, 220), (220, 60, 60)],  # bg / artery red / vein blue
            "tessellation": [(0, 0, 0), (200, 150, 255)],
            "myopia": [(0, 0, 0), (255, 200, 50), (50, 220, 200), (200, 50, 220)],
            "lesion_s1": [(0, 0, 0)] + [
                (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
                for _ in range(20)
            ],
            "lesion_s2": [(0, 0, 0)] + [
                (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
                for _ in range(20)
            ],
            "lesion_s3": [(0, 0, 0)] + [
                (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
                for _ in range(20)
            ],
            "possible_lesions": [(0, 0, 0), (60, 200, 220)],
        }
        alpha = 0.45
        out: dict[str, str] = {}

        for stem, mask_path in mask_files.items():
            mask = RetsamAdapter._imread_unicode(mask_path, flags=cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            # Resize mask to original
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # Pick a palette. Values are clipped into [0, 255] — numpy 2.x
            # refuses out-of-range Python ints in uint8 assignments.
            def _clip3(rgb):
                return tuple(max(0, min(255, int(v))) for v in rgb)
            raw_colours = palette.get(stem, [(0, 0, 0)] + [
                (60 + 30 * i, 200, 220 - 20 * i) for i in range(1, 21)
            ])
            colours = [_clip3(c) for c in raw_colours]
            colour_mask = np.zeros_like(original, dtype=np.uint8)
            for cls_idx, c in enumerate(colours):
                colour_mask[mask == cls_idx] = c

            # Only colour the foreground; keep background = original
            foreground = (mask > 0)[..., None]
            blended = cv2.addWeighted(original, 1.0 - alpha, colour_mask, alpha, 0)
            comp = np.where(foreground, blended, original).astype(np.uint8)

            dst = out_dir / f"{stem}_overlay.png"
            # Use imencode + tofile instead of imwrite — OpenCV's C++
            # fopen uses Windows ANSI codec, which mangles non-ASCII
            # paths (e.g. CJK uploads). imencode goes through numpy
            # buffer + Python's UTF-8-aware open, which is unicode-safe.
            cv2.imencode(".png", comp)[1].tofile(str(dst))
            out[stem] = str(dst)

        return out
