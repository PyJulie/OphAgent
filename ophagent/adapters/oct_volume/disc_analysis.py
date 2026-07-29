"""
Adapter: OCT volume → optic-disc cpRNFLT quantification.

Wraps the g-disc_OCT4 pipeline (Tohoku SLSS layer model + UCL W-Net disc
centroid + annular cpRNFLT integration) by subprocess-calling its CLI.

Inputs accepted (single file):
  - `.dcm`  Topcon DICOM volume
  - `.fda`  Topcon raw (requires `oct-converter` installed)
  - `.npy`  pre-processed 3D volume

Outputs returned to the agent (structured):
  - `cpRNFLT_sectors`: 12 clock-hour RNFL thicknesses S1..S12 (µm)
  - `cpRNFLT_total`, `upper_half`, `lower_half`
  - `tsni`: T / S / N / I quadrant means for RNFL
  - `per_layer`: same 12-sector breakdown for GCL / IPL / INL / OPL / ONL /
    PR / Choroid + GCC composite
  - `eye`: OD or OS (auto-detected from DICOM tags)
  - `disc_centroid_px`: (x, y) on the en-face image
  - `figures`: paths to en-face overlay, sector plots, TSNI plots

Runtime: ~10s for a 49-slice volume on a 24 GB GPU (CUDA 12+).
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import bundled_component_dir, checkpoint_file, output_path


log = logging.getLogger(__name__)


G_DISC_ROOT = bundled_component_dir("OPHAGENT_G_DISC_ROOT", "g_disc_oct4")
G_DISC_ASSETS = {
    "layer segmentation weights": checkpoint_file(
        "OPHAGENT_G_DISC_SEGMENTATION_WEIGHTS",
        "oct_volume",
        "g_disc",
        "tohoku_full_slss_013_fold0_3.522%.t7",
    ),
    "disc localisation weights": checkpoint_file(
        "OPHAGENT_G_DISC_WNET_WEIGHTS",
        "oct_volume",
        "g_disc",
        "wnet_disc512_best_UCL-new3-0.793.pth",
    ),
    "3D OCT histogram calibration": checkpoint_file(
        "OPHAGENT_G_DISC_3DOCT_HISTOGRAM",
        "oct_volume",
        "g_disc",
        "3DOCT_hist.cdf",
    ),
    "Triton histogram calibration": checkpoint_file(
        "OPHAGENT_G_DISC_TRITON_HISTOGRAM",
        "oct_volume",
        "g_disc",
        "Triton_hist.cdf",
    ),
}
OUT_BASE = output_path("adapter_figures", "oct_volume_disc")

PIPELINE_BY_SUFFIX = {".dcm": "dicom", ".fda": "fda", ".npy": "vol_npy"}

RNFL_LAYER_NAME = "RNFL"
LAYERS_OF_INTEREST = ["RNFL", "GCL", "IPL", "INL", "OPL", "ONL", "PR", "Choroid", "GCC"]


def _check_environment() -> None:
    """Raise a clear error if G-DISC source, assets, or CuPy are missing."""
    if not G_DISC_ROOT.exists():
        raise RuntimeError(
            "g-disc_OCT4 source missing. Set OPHAGENT_G_DISC_ROOT. "
            f"Expected: {G_DISC_ROOT}"
        )
    src_dir = G_DISC_ROOT / "src"
    if not src_dir.exists():
        raise RuntimeError(
            "g-disc src dir missing. Set OPHAGENT_G_DISC_ROOT to a valid checkout. "
            f"Expected: {src_dir}"
        )
    missing_assets = [
        f"{label}: {path}"
        for label, path in G_DISC_ASSETS.items()
        if not path.is_file()
    ]
    if missing_assets:
        details = "\n".join(f"- {item}" for item in missing_assets)
        raise FileNotFoundError(
            "G-DISC requires external model and calibration assets. "
            "Place the verified files under checkpoints/oct_volume/g_disc "
            "or set the corresponding OPHAGENT_G_DISC_* variables:\n"
            f"{details}"
        )
    # Quick cupy import sanity (heavy dep)
    try:
        import cupy as _cp   # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "cupy missing — install with `pip install cupy-cuda12x` first."
        ) from e


def _safe_float(v: Any) -> float | None:
    """Convert v to float; return None on empty / NaN / unparseable."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "na", "n/a"}:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    # treat -1 / -1.0 as "missing" — the upstream W-Net uses it as a sentinel
    if f == -1 or (f != f):    # NaN check
        return None
    return f


def _parse_cprnflt12(csv_path: Path, pid: str) -> dict[str, Any]:
    """Parse cpRNFLT_12.csv → {S1..S12, Total, UH, LH, Source, Eye}.
    Missing / unparseable cells become None — the LLM can still report
    'unavailable' rather than crashing the whole tool."""
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if r.get("PID") == pid]
    if not matches:
        return {}
    row = matches[-1]
    out: dict[str, Any] = {
        "Source": row.get("Source"),
        "Eye": row.get("Eye"),
    }
    for i in range(1, 13):
        out[f"S{i}"] = _safe_float(row.get(f"S{i}"))
    for key in ("Total", "UH", "LH"):
        out[key] = _safe_float(row.get(key))
    return out


def _parse_annulus(csv_path: Path, pid: str) -> dict[str, dict[str, float | None]]:
    """Parse annulus_summary.csv → {layer_name: {1h..12h, global, T,S,N,I, upper, lower}}.
    Tolerates empty cells (returned as None)."""
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        if r.get("core_id") != pid:
            continue
        layer = r.get("layer")
        if layer not in LAYERS_OF_INTEREST:
            continue
        entry: dict[str, float | None] = {
            "global": _safe_float(r.get("global")),
            "upper":  _safe_float(r.get("upper")),
            "lower":  _safe_float(r.get("lower")),
            "T":      _safe_float(r.get("T")),
            "S":      _safe_float(r.get("S")),
            "N":      _safe_float(r.get("N")),
            "I":      _safe_float(r.get("I")),
        }
        for i in range(1, 13):
            entry[f"{i}h"] = _safe_float(r.get(f"{i}h"))
        out[layer] = entry
    return out


def _parse_centroid(csv_path: Path, pid: str) -> tuple[int, int] | None:
    """Schema: filename,x,y — pick the first valid (x != -1) row matching this pid.
    BM-based extraction can fail and return (-1, -1); ALL-based usually succeeds."""
    if not csv_path.exists():
        return None
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        name = r.get("filename") or r.get("image") or r.get("file") or ""
        if pid not in name:
            continue
        for kx, ky in [("x", "y"), ("cx", "cy"), ("X", "Y")]:
            if kx in r and ky in r:
                try:
                    cx = int(float(r[kx]))
                    cy = int(float(r[ky]))
                except ValueError:
                    continue
                if cx >= 0 and cy >= 0:
                    return cx, cy
    return None


def _collect_figures(out_dir: Path, pid: str) -> dict[str, str]:
    """Pick the most user-facing figures only — avoid dumping 32+ PNGs."""
    figs: dict[str, str] = {}
    # En-face overlay (BM preferred, else ALL)
    overlay_dir = out_dir / "centroid_out" / "overlay"
    for cand in [f"{pid}_enface_BM_overlay.png", f"{pid}_enface_ALL_overlay.png"]:
        p = overlay_dir / cand
        if p.exists():
            figs["disc_centroid_overlay"] = str(p)
            break
    # RNFL sector + TSNI plot (the headline figures clinicians read)
    res_root = out_dir / "results" / "ai" / pid
    for layer in (RNFL_LAYER_NAME, "GCC"):
        s12 = res_root / f"{layer}_sector12.png"
        if s12.exists():
            figs[f"{layer.lower()}_sector12"] = str(s12)
        tsni = res_root / f"{layer}_tsni.png"
        if tsni.exists():
            figs[f"{layer.lower()}_tsni"] = str(tsni)
        ul = res_root / f"{layer}_upper_lower.png"
        if ul.exists():
            figs[f"{layer.lower()}_upper_lower"] = str(ul)
    return figs


def _interpret_rnfl(sectors: dict[str, Any]) -> dict[str, Any]:
    """Light-touch interpretive hints. NOT a diagnosis — surfaces patterns.
    Robust to missing values (None) — only flags what it can compute."""
    if not sectors:
        return {"summary_notes": ["RNFL data unavailable — quantitative analysis could not complete."]}

    s_vals = [sectors.get(f"S{i}") for i in range(1, 13)]
    s_valid = [(i + 1, v) for i, v in enumerate(s_vals) if v is not None]
    total = sectors.get("Total")
    uh = sectors.get("UH")
    lh = sectors.get("LH")

    notes: list[str] = []
    if total is None:
        notes.append("Global RNFL could not be computed — likely disc localisation failed or scan geometry unexpected.")
    else:
        if total < 70:
            notes.append("Global RNFL is markedly thin (<70 um) — concerning for advanced glaucomatous loss.")
        elif total < 85:
            notes.append("Global RNFL is below average (<85 um) — suggestive of generalised thinning.")

    if uh is not None and lh is not None:
        if (uh - lh) > 15:
            notes.append("Upper hemisphere notably thicker than lower — atypical for ISNT.")
        elif (lh - uh) > 15:
            notes.append("Inferior thinning relative to superior — common pattern in early glaucoma.")

    if total is not None and total > 0 and s_valid:
        focal = [(i, v) for i, v in s_valid if v < 0.7 * total]
        if focal:
            spots = ", ".join(f"S{i}={v:.0f}" for i, v in focal)
            notes.append(f"Focal thinning sectors (<70% of global): {spots}.")

    if not notes:
        notes = ["RNFL appears within or above normal range; review per-sector plot."]

    return {
        "global_um": total,
        "upper_um": uh,
        "lower_um": lh,
        "n_sectors_valid": len(s_valid),
        "summary_notes": notes,
    }


@register
class OCTVolumeDiscAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="oct_volume_disc",
        modality="OCT",
        task="quantification",
        description=(
            "Optic-disc cpRNFLT quantification on a full Topcon OCT volume "
            "(.dcm / .fda / .npy). Runs the Tohoku SLSS 10-layer segmentor → "
            "RNFL en-face → UCL W-Net disc-centroid → 12-clock-hour annular "
            "integration around the disc. Returns the clinical cpRNFLT vector "
            "(S1..S12, Total, UH, LH, TSNI), per-layer thickness breakdown "
            "(RNFL/GCL/IPL/INL/OPL/ONL/PR/Choroid + GCC), the detected eye "
            "laterality, and overlay figures. Use this for glaucoma follow-up "
            "or whenever the user has a 3D OCT volume rather than a single "
            "B-scan."
        ),
        input_size=None,
        labels=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12",
                "Total", "UH", "LH", "T", "S", "N", "I"],
        confidence_threshold=0.0,    # quantitative output, no class probability
        limitations=[
            "Requires the external g-disc_OCT4 pipeline + cupy-cuda12x.",
            ".fda input additionally requires `oct-converter`.",
            "Trained on Topcon volumes — other vendors (Heidelberg, Zeiss) may "
            "need a different histogram CDF or shape preset.",
            "Runtime ~10s for 49 slices; larger volumes scale linearly.",
            "Normative range for global RNFL is ~95-110 µm in healthy adults; "
            "interpretation must consider age, axial length, and refractive error.",
        ],
        cost_class="slow",
        source_dir=str(G_DISC_ROOT),
    )

    def _load_impl(self) -> None:
        _check_environment()
        OUT_BASE.mkdir(parents=True, exist_ok=True)
        self._impl = "external-cli"

    def _predict_impl(
        self,
        image_path: str,
        target: str = "disc",
        gpu: str = "0",
        **_,
    ) -> AdapterResult:
        src = Path(image_path).resolve()
        if not src.exists():
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="quantification",
                error=f"input does not exist: {src}",
            )
        suffix = src.suffix.lower()
        pipeline = PIPELINE_BY_SUFFIX.get(suffix)
        if pipeline is None:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="quantification",
                error=(f"unsupported OCT volume suffix '{suffix}'. "
                       "Expected one of .dcm / .fda / .npy"),
            )

        pid = src.stem
        run_token = f"{pid}_{int(time.time())}"
        out_dir = OUT_BASE / run_token
        out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = os.pathsep.join([
            str(G_DISC_ROOT / "src"),
            str(G_DISC_ROOT),
            env.get("PYTHONPATH", ""),
        ])

        # Map agent's `target` arg → octseg's `--target`
        target_map = {"disc": "disc", "disc2": "disc2", "macula": "macula"}
        oct_target = target_map.get(target, "disc")

        cmd = [
            sys.executable, "-m", "octseg", "run",
            "--pipeline", pipeline,
            "--input", str(src),
            "--output", str(out_dir),
            "--gpu", str(gpu),
            "--target", oct_target,
        ]
        log.info(f"[oct_volume_disc] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(G_DISC_ROOT), env=env,
                              capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="quantification",
                error=f"octseg CLI failed (rc={proc.returncode}): "
                      f"{(proc.stderr or proc.stdout)[-1200:]}",
                metadata={"cmd": cmd, "stdout_tail": (proc.stdout or '')[-500:]},
            )

        # Parse outputs (resilient — any cell can be None)
        cprnflt_row = _parse_cprnflt12(out_dir / "results" / "cpRNFLT_12.csv", pid)
        per_layer = _parse_annulus(out_dir / "results" / "annulus_summary.csv", pid)
        centroid = _parse_centroid(out_dir / "centroid_out" / "centroids.csv", pid)
        figures = _collect_figures(out_dir, pid)

        rnfl_layer = per_layer.get(RNFL_LAYER_NAME, {})
        tsni = {k: rnfl_layer[k] for k in ("T", "S", "N", "I")
                if k in rnfl_layer and rnfl_layer[k] is not None}

        interpretation = _interpret_rnfl(cprnflt_row)

        # Strip Nones from the sector dict so the LLM doesn't see noise
        sectors_clean = {f"S{i}": cprnflt_row.get(f"S{i}")
                         for i in range(1, 13)
                         if cprnflt_row.get(f"S{i}") is not None}

        # If literally nothing usable came back, fail loudly
        nothing_usable = (
            not sectors_clean
            and cprnflt_row.get("Total") is None
            and not per_layer
        )
        if nothing_usable:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="quantification",
                error=(
                    "Pipeline ran but produced no parseable cpRNFLT values. "
                    "Likely causes: (1) disc localisation failed because the "
                    "volume is macula-centred, not disc-centred; (2) "
                    "scan-geometry metadata in the DICOM is missing; "
                    "(3) volume shape outside the supported presets. "
                    "Try a disc-centred scan or set --allow-any-shape on the NPY pipeline."
                ),
                metadata={"out_dir": str(out_dir), "pid": pid},
            )

        partial = (
            cprnflt_row.get("Total") is None
            or len(sectors_clean) < 12
            or centroid is None
        )

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="quantification",
            predictions={
                "eye": cprnflt_row.get("Eye"),
                "segmentation_source": cprnflt_row.get("Source"),
                "cpRNFLT_total_um": cprnflt_row.get("Total"),
                "upper_half_um": cprnflt_row.get("UH"),
                "lower_half_um": cprnflt_row.get("LH"),
                "cpRNFLT_sectors": sectors_clean,
                "rnfl_tsni": tsni,
                "per_layer_thickness": per_layer,
                "disc_centroid_px": list(centroid) if centroid else None,
                "interpretation": interpretation,
                "partial_result": partial,
            },
            confidence=1.0 if not partial else 0.5,
            figures=figures,
            metadata={
                "pipeline": pipeline,
                "target": oct_target,
                "out_dir": str(out_dir),
                "g_disc_root": str(G_DISC_ROOT),
            },
        )
