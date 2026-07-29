#!/usr/bin/env python3
"""
retsam-2.0 — one-shot orchestrator that runs every postprocess module.

Given a directory produced by ``scripts/infer.py`` (one subfolder per image
with ``masks/`` + ``infer_summary.json`` + ``coords.json``), this script:

    1. Loads the original image + every mask it needs.
    2. Runs **vessels**, **disc_cup**, **lesions**, **myopia**, and
       **tessellation** in sequence.
    3. Writes each module's JSON (``vessels.json``, ``disc_cup.json``, …).
    4. Renders each module's visualization panels into
       ``visualizations/<module>/``.
    5. Writes a single ``analysis.json`` that aggregates all module results
       (meta + per-module trees + a flat high-level summary).

Modules can be skipped via ``--skip``, visualization rendering can be
turned off globally via ``--no_viz``.

Example:
    python scripts/quantify.py --infer_dir ./out --eye_side OS --pixel_spacing_um 7.5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from postprocess.vessels import (  # noqa: E402
    VesselAnalysisConfig,
    VesselVizConfig,
    analyze_vessels,
    render_vessel_visualizations,
)
from postprocess.disc_cup import (  # noqa: E402
    DiscCupAnalysisConfig,
    DiscCupVizConfig,
    analyze_disc_cup,
    render_disc_cup_visualizations,
)
from postprocess.lesions import (  # noqa: E402
    LesionAnalysisConfig,
    LesionVizConfig,
    analyze_lesions,
    load_lesion_masks_from_dir,
    render_lesion_visualizations,
)
from postprocess.myopia import (  # noqa: E402
    MyopiaAnalysisConfig,
    MyopiaVizConfig,
    analyze_myopia,
    render_myopia_visualizations,
)
from postprocess.tessellation import (  # noqa: E402
    TessellationAnalysisConfig,
    TessellationVizConfig,
    analyze_tessellation,
    render_tessellation_visualizations,
)


MODULES = ("vessels", "disc_cup", "lesions", "myopia", "tessellation")


# --------------------------------------------------------------------------- I/O

def _read_single_channel(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img


def _load_original(image_dir: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    summary_path = os.path.join(image_dir, "infer_summary.json")
    if not os.path.exists(summary_path):
        return None, None
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    p = summary.get("image_path")
    if not p or not os.path.exists(p):
        return None, p
    return cv2.imread(p), p


def _load_macula(image_dir: str) -> Optional[Tuple[float, float]]:
    path = os.path.join(image_dir, "coords.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f).get("macula_center")
    if isinstance(d, dict) and "y" in d and "x" in d:
        return float(d["y"]), float(d["x"])
    return None


def _load_eye_side(image_dir: str) -> Optional[str]:
    path = os.path.join(image_dir, "eye_side.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        s = json.load(f).get("eye_side")
    return s if s in ("OS", "OD") else None


# --------------------------------------------------------------------------- per-module drivers

def _run_vessels(image_dir: str, image_bgr, od_oc, fundus, macula, eye_side,
                 pixel_spacing_um, write_viz) -> Optional[Dict[str, Any]]:
    av = _read_single_channel(os.path.join(image_dir, "masks", "artery_vein_mask.png"))
    if av is None:
        return None
    artery = ((av == 1) | (av == 3)).astype(np.uint8)
    vein = ((av == 2) | (av == 3)).astype(np.uint8)
    disc_bin = (od_oc > 0).astype(np.uint8)
    result = analyze_vessels(
        artery_mask=artery, vein_mask=vein, disc_mask=disc_bin,
        fundus_mask=fundus, pixel_spacing_um=pixel_spacing_um,
        config=VesselAnalysisConfig(),
    )
    if write_viz:
        render_vessel_visualizations(
            original_image_bgr=image_bgr,
            artery_mask=artery, vein_mask=vein, disc_mask=disc_bin,
            fundus_mask=fundus, macula_center_yx=macula,
            analysis_result=result,
            output_dir=os.path.join(image_dir, "visualizations", "vessels"),
            config=VesselVizConfig(),
        )
    return result


def _run_disc_cup(image_dir: str, image_bgr, od_oc, fundus, macula, eye_side,
                  pixel_spacing_um, write_viz) -> Optional[Dict[str, Any]]:
    # fundus is unused — disc_cup runs entirely off od_oc — but we keep the
    # uniform runner signature so the orchestrator can dispatch generically.
    del fundus
    result = analyze_disc_cup(
        od_oc_mask=od_oc, eye_side=eye_side,
        pixel_spacing_um=pixel_spacing_um,
        config=DiscCupAnalysisConfig(),
    )
    if write_viz:
        render_disc_cup_visualizations(
            original_image_bgr=image_bgr, od_oc_mask=od_oc,
            analysis_result=result,
            output_dir=os.path.join(image_dir, "visualizations", "disc_cup"),
            macula_center_yx=macula,
            config=DiscCupVizConfig(),
        )
    return result


def _run_lesions(image_dir: str, image_bgr, od_oc, fundus, macula, eye_side,
                 pixel_spacing_um, write_viz) -> Optional[Dict[str, Any]]:
    masks_dir = os.path.join(image_dir, "masks")
    lesion_masks = load_lesion_masks_from_dir(masks_dir)
    result = analyze_lesions(
        lesion_masks=lesion_masks, od_oc_mask=od_oc,
        fundus_mask=fundus, macula_center_yx=macula,
        eye_side=eye_side, pixel_spacing_um=pixel_spacing_um,
        config=LesionAnalysisConfig(),
    )
    if write_viz:
        render_lesion_visualizations(
            original_image_bgr=image_bgr, lesion_masks=lesion_masks,
            analysis_result=result,
            output_dir=os.path.join(image_dir, "visualizations", "lesions"),
            disc_mask=(od_oc > 0).astype(np.uint8),
            fundus_mask=fundus, macula_center_yx=macula, eye_side=eye_side,
            config=LesionVizConfig(),
        )
    return result


def _run_myopia(image_dir: str, image_bgr, od_oc, fundus, macula, eye_side,
                pixel_spacing_um, write_viz) -> Optional[Dict[str, Any]]:
    myopia = _read_single_channel(os.path.join(image_dir, "masks", "myopia_mask.png"))
    if myopia is None:
        return None
    result = analyze_myopia(
        myopia_mask=myopia, od_oc_mask=od_oc,
        fundus_mask=fundus, macula_center_yx=macula,
        eye_side=eye_side, pixel_spacing_um=pixel_spacing_um,
        config=MyopiaAnalysisConfig(),
    )
    if write_viz:
        render_myopia_visualizations(
            original_image_bgr=image_bgr, myopia_mask=myopia,
            analysis_result=result,
            output_dir=os.path.join(image_dir, "visualizations", "myopia"),
            disc_mask=(od_oc > 0).astype(np.uint8),
            macula_center_yx=macula,
            config=MyopiaVizConfig(),
        )
    return result


def _run_tessellation(image_dir: str, image_bgr, od_oc, fundus, macula, eye_side,
                      pixel_spacing_um, write_viz) -> Optional[Dict[str, Any]]:
    tess = _read_single_channel(os.path.join(image_dir, "masks", "tessellation_mask.png"))
    if tess is None:
        return None
    result = analyze_tessellation(
        tessellation_mask=tess, od_oc_mask=od_oc,
        fundus_mask=fundus, macula_center_yx=macula,
        eye_side=eye_side, pixel_spacing_um=pixel_spacing_um,
        config=TessellationAnalysisConfig(),
    )
    if write_viz:
        render_tessellation_visualizations(
            original_image_bgr=image_bgr, tessellation_mask=tess,
            analysis_result=result,
            output_dir=os.path.join(image_dir, "visualizations", "tessellation"),
            disc_mask=(od_oc > 0).astype(np.uint8),
            macula_center_yx=macula,
            config=TessellationVizConfig(),
        )
    return result


_RUNNERS = {
    "vessels": _run_vessels,
    "disc_cup": _run_disc_cup,
    "lesions": _run_lesions,
    "myopia": _run_myopia,
    "tessellation": _run_tessellation,
}


# --------------------------------------------------------------------------- flat summary

def _flat_summary(per_module: Dict[str, Any]) -> Dict[str, Any]:
    """Curated single-dict summary of the most useful numbers. Intended for
    quick screening or CSV export; full detail lives inside each module's tree.
    """
    v = per_module.get("vessels") or {}
    dc = per_module.get("disc_cup") or {}
    l = per_module.get("lesions") or {}
    m = per_module.get("myopia") or {}
    t = per_module.get("tessellation") or {}

    summary: Dict[str, Any] = {}
    if v:
        summary["CRAE_px"] = v["crae"]["value_px"]
        summary["CRVE_px"] = v["crve"]["value_px"]
        summary["AVR"] = v["avr"]["value"]
        summary["fractal_dimension_artery"] = v["fractal_dimension"]["artery"]["value"]
        summary["fractal_dimension_vein"] = v["fractal_dimension"]["vein"]["value"]
        summary["tortuosity_artery_mean_df"] = v["tortuosity"]["artery"]["mean_df"]
        summary["tortuosity_vein_mean_df"] = v["tortuosity"]["vein"]["mean_df"]
        summary["vessel_density_total"] = v["density"]["total_ratio"]
    if dc:
        summary["vCDR"] = dc["cdr"]["vertical"]
        summary["hCDR"] = dc["cdr"]["horizontal"]
        summary["aCDR"] = dc["cdr"]["area"]
        summary["rim_disc_area_ratio"] = dc["rim"]["rim_disc_area_ratio"]
        summary["disc_tilt_deg"] = dc["disc"]["tilt_angle_deg"]
        summary["disc_ovality"] = dc["disc"]["ovality"]
        summary["ISNT_follows"] = dc["rim"]["isnt"]["follows_rule"]
        summary["ddls_grade"] = (dc["ddls"] or {}).get("grade")
    if l:
        summary["lesions_classes_detected"] = l["global_summary"]["n_classes_detected"]
        summary["lesions_total_coverage_ratio"] = l["global_summary"]["total_coverage_ratio"]
        dr = (l["severity_inputs"] or {}).get("dr")
        if dr:
            summary["DR_hemorrhage_total_count"] = dr["hemorrhage_total_count"]
            summary["DR_cotton_wool_spot_count"] = dr["cotton_wool_spot_count"]
            summary["DR_exudate_count"] = dr["exudate_count"]
            summary["DR_four_quadrants_heavy"] = dr.get("four_quadrants_heavy")
        amd = (l["severity_inputs"] or {}).get("amd")
        if amd:
            summary["AMD_drusen_total_count"] = amd["drusen_total_count"]
            summary["AMD_drusen_within_1dd_of_macula"] = amd["drusen_count_within_1dd_of_macula"]
            summary["AMD_patch_hemorrhage_present"] = amd["patch_hemorrhage_present"]
    if m:
        sev = m["severity_inputs"]
        summary["myopia_arc_present"] = sev["arc_lesion_present"]
        summary["myopia_arc_angular_coverage_deg"] = sev["arc_angular_coverage_deg"]
        summary["myopia_diffuse_atrophy_present"] = sev["diffuse_atrophy_present"]
        summary["myopia_diffuse_involves_macula"] = sev["diffuse_involves_macula"]
        summary["myopia_patchy_atrophy_count"] = sev["patchy_count"]
        summary["myopia_patchy_involves_macula"] = sev["patchy_involves_macula"]
    if t:
        summary["tessellation_coverage_ratio"] = t["tessellation"]["coverage_ratio"]
        summary["tessellation_severity"] = t["tessellation"]["severity"]
        summary["tessellation_involves_macula"] = t["tessellation"]["involves_macula"]
    return summary


# --------------------------------------------------------------------------- per-image driver

def _should_skip(image_dir: str, modules: List[str]) -> bool:
    """Resume helper: skip if ``analysis.json`` is newer than every input
    (masks, infer_summary, coords) it depends on. Conservative — missing
    input timestamps disable the skip, which is safer.
    """
    analysis_path = os.path.join(image_dir, "analysis.json")
    if not os.path.exists(analysis_path):
        return False
    try:
        analysis_mtime = os.path.getmtime(analysis_path)
    except OSError:
        return False

    dep_files = [
        os.path.join(image_dir, "infer_summary.json"),
        os.path.join(image_dir, "coords.json"),
    ]
    masks_dir = os.path.join(image_dir, "masks")
    if os.path.isdir(masks_dir):
        dep_files += [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
                      if f.endswith(".png")]

    for dep in dep_files:
        if not os.path.exists(dep):
            continue
        try:
            if os.path.getmtime(dep) > analysis_mtime:
                return False
        except OSError:
            return False
    return True


def process_one(image_dir: str, modules: List[str], write_viz: bool,
                eye_side_cli: Optional[str],
                pixel_spacing_um: Optional[float]) -> Dict[str, Any]:
    masks_dir = os.path.join(image_dir, "masks")
    od_oc = _read_single_channel(os.path.join(masks_dir, "od_oc_mask.png"))
    if od_oc is None:
        return {"error": "missing od_oc_mask.png", "meta": {"image_dir": image_dir}}
    image_bgr, image_path = _load_original(image_dir)
    if image_bgr is None:
        return {"error": f"cannot load original image ({image_path})",
                "meta": {"image_dir": image_dir}}
    fundus_raw = _read_single_channel(os.path.join(masks_dir, "fundus_mask.png"))
    fundus = (fundus_raw > 0).astype(np.uint8) if fundus_raw is not None else None
    macula = _load_macula(image_dir)
    eye_side = _load_eye_side(image_dir) or eye_side_cli

    per_module: Dict[str, Any] = {}
    module_errors: Dict[str, str] = {}
    for name in modules:
        runner = _RUNNERS[name]
        try:
            result = runner(image_dir, image_bgr, od_oc, fundus, macula,
                            eye_side, pixel_spacing_um, write_viz)
            if result is not None:
                per_module[name] = result
            else:
                module_errors[name] = "required masks not found"
        except Exception as e:
            module_errors[name] = f"{type(e).__name__}: {e}"

    aggregate = {
        "meta": {
            "image_dir": image_dir,
            "image_path": image_path,
            "eye_side": eye_side,
            "pixel_spacing_um": pixel_spacing_um,
            "macula_center_yx": list(macula) if macula else None,
            "modules_attempted": list(modules),
            "modules_succeeded": list(per_module.keys()),
            "module_errors": module_errors,
            "schema_version": "2.0.0",
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        },
        "summary": _flat_summary(per_module),
        "modules": per_module,
    }

    out_path = os.path.join(image_dir, "analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    aggregate["__out_path"] = out_path
    return aggregate


# --------------------------------------------------------------------------- CLI

def _find(root: str, recursive: bool = False) -> List[str]:
    """Locate every directory under ``root`` that holds an
    ``infer_summary.json`` — i.e. a per-image inference output folder.

    With ``recursive=True``, descends into any subdirectory that does not
    itself look like a per-image folder. This is what lets us quantify
    datasets with a split/class structure (e.g. PAPILA's
    ``train/anormal/RET027OD``) from a single entry point.
    """
    if not os.path.isdir(root):
        return []
    if not recursive:
        return [os.path.join(root, e) for e in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, e)) and
                os.path.exists(os.path.join(root, e, "infer_summary.json"))]
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "infer_summary.json" in filenames:
            found.append(dirpath)
            # Don't descend further — this is a leaf image-output dir.
            dirnames[:] = []
    return sorted(found)


def main() -> int:
    p = argparse.ArgumentParser(
        description="retsam-2.0 one-shot quantification (all modules)",
    )
    p.add_argument("--infer_dir", required=True,
                   help="Root output dir from scripts/infer.py")
    p.add_argument("--eye_side", choices=["OS", "OD"], default=None,
                   help="Applied to all images unless overridden by <image>/eye_side.json")
    p.add_argument("--pixel_spacing_um", type=float, default=None)
    p.add_argument("--skip", nargs="*", default=[],
                   help=f"Modules to skip. Options: {', '.join(MODULES)}")
    p.add_argument("--only", nargs="*", default=None,
                   help="Whitelist of modules to run (overrides --skip).")
    p.add_argument("--no_viz", action="store_true",
                   help="Skip visualization rendering — JSON only.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes. 1 = serial. "
                        "Typically set to min(CPU_count, num_images). Multiprocessing "
                        "is per-image: each worker processes one image through all "
                        "modules.")
    p.add_argument("--resume", action="store_true",
                   help="Skip images whose analysis.json is newer than every input "
                        "(masks, summary, coords).")
    p.add_argument("--recursive", action="store_true",
                   help="Walk into every subfolder under --infer_dir to find "
                        "per-image output folders (use with nested dataset layouts).")
    args = p.parse_args()

    root = os.path.abspath(args.infer_dir)
    dirs = _find(root, recursive=args.recursive)
    if not dirs:
        print(f"No image subfolders under {root}", file=sys.stderr)
        return 2

    if args.only:
        modules = [m for m in args.only if m in MODULES]
        unknown = [m for m in args.only if m not in MODULES]
        if unknown:
            print(f"[warn] ignoring unknown --only values: {unknown}", file=sys.stderr)
    else:
        modules = [m for m in MODULES if m not in args.skip]
    if not modules:
        print("No modules left after filter", file=sys.stderr)
        return 2
    print(f"Running modules: {modules}", file=sys.stderr)

    # Resume: drop already-completed dirs before dispatching work
    original_count = len(dirs)
    if args.resume:
        dirs = [d for d in dirs if not _should_skip(d, modules)]
        skipped = original_count - len(dirs)
        if skipped > 0:
            print(f"Resumed: skipping {skipped}/{original_count} images with "
                  f"up-to-date analysis.json", file=sys.stderr)
    else:
        skipped = 0

    if not dirs:
        print(f"Nothing to do (skipped {skipped}, total {original_count}).")
        return 0

    ok = failed = partial_ok = 0
    t0 = time.time()
    worker_fn = partial(
        process_one, modules=modules, write_viz=not args.no_viz,
        eye_side_cli=args.eye_side, pixel_spacing_um=args.pixel_spacing_um,
    )

    workers = max(1, int(args.workers))
    pool = None
    try:
        if workers == 1:
            results_iter = (worker_fn(d) for d in dirs)
        else:
            # Multiprocessing per-image. imap_unordered overlaps I/O with
            # compute and avoids holding the whole result set in memory. Use
            # spawn for Windows compatibility.
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=workers)
            results_iter = pool.imap_unordered(worker_fn, dirs)

        for agg in tqdm_friendly(results_iter, total=len(dirs)):
            # Each result's image_dir is carried inside meta so order from
            # imap_unordered doesn't matter.
            image_dir = agg.get("meta", {}).get("image_dir", "<unknown>")
            errs = agg.get("meta", {}).get("module_errors", {})
            if "error" in agg:
                failed += 1
                print(f"[fail] {image_dir}: {agg['error']}", file=sys.stderr)
            elif errs:
                partial_ok += 1
                print(f"[partial] {image_dir}: OK={agg['meta']['modules_succeeded']} errs={errs}",
                      file=sys.stderr)
            else:
                ok += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    dt_s = time.time() - t0
    thr = (ok + partial_ok + failed) / dt_s if dt_s > 0 else 0.0
    print(
        f"Done. full_ok={ok} partial={partial_ok} failed={failed} "
        f"skipped={skipped} total={original_count}  "
        f"wall_time={dt_s:.1f}s  throughput={thr:.2f} img/s",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


def tqdm_friendly(iterable, total=None):
    """Try to wrap with tqdm; fall back to plain iterator if tqdm unavailable.

    Multiprocessing + tqdm can interact poorly with stdout on Windows; keep
    it out of the hot loop itself and just emit progress-safe prints.
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc="Quantify", unit="img")
    except ImportError:
        return iterable


if __name__ == "__main__":
    raise SystemExit(main())
