from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_pipeline_config
from .layout import OutputLayout
from .legacy import PROJECT_ROOT, resolve_project_path
from .manifests import read_manifest_entries


PIPELINE_ALIASES = {
    "dicom": "dicom",
    "dcm": "dicom",
    "fda": "fda",
    "vol_npy": "vol_npy",
    "npy": "vol_npy",
}

PIPELINE_SUFFIXES = {
    "dicom": (".dcm",),
    "fda": (".fda",),
    "vol_npy": (".npy",),
}

PIPELINE_FORMAT_FLAGS = {
    "dicom": "--dcm",
    "fda": "--fda",
    "vol_npy": "--npy",
}


@dataclass
class PipelineRequest:
    pipeline: str
    input_path: Path | None
    manifest_path: Path | None
    output_root: Path
    gpu: str
    target: str
    segmentation_source: str | None
    slice_overlays: str
    volume_preproc: str
    allow_any_shape: bool
    rotate_npy: str | None
    config_path: Path | None
    dry_run: bool


@dataclass
class InputSelection:
    selected_paths: list[Path]
    selected_ids: list[str]
    manifest_entries: list[str]
    staging_rows: list[dict[str, Any]]
    active_input_path: Path
    oct_root: Path | None


def _canonical_pipeline(pipeline: str) -> str:
    try:
        return PIPELINE_ALIASES[pipeline]
    except KeyError as exc:
        raise ValueError(f"Unsupported pipeline: {pipeline}") from exc


def _canonical_segmentation_source(source: str | None, pipeline: str) -> str:
    requested = (source or "").strip().lower()
    if requested == "topcon":
        requested = "vendor"
    if not requested:
        requested = "auto" if pipeline == "fda" else "ai"

    if pipeline != "fda":
        if requested == "vendor":
            raise ValueError(f"{pipeline} does not support vendor segmentation. Use --segmentation-source ai.")
        if requested == "auto":
            return "ai"
    if requested not in {"ai", "vendor", "auto"}:
        raise ValueError(f"Unsupported segmentation source: {source}")
    return requested


def _quoted(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    dry_run: bool,
    command_log: list[dict[str, Any]],
    env: dict[str, str] | None = None,
) -> None:
    rendered = _quoted(cmd)
    print(f"[RUN] {label}")
    print(f"      {rendered}")
    command_log.append({"label": label, "cwd": str(cwd), "cmd": cmd})
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _resolve_python_cmd(config: dict[str, Any]) -> str:
    python_cmd = config.get("python_cmd")
    return str(python_cmd) if python_cmd else sys.executable


def _resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    raw_paths = config["paths"]
    return {key: resolve_project_path(value) for key, value in raw_paths.items()}


def _pipeline_targets(target: str) -> list[str]:
    if target == "all":
        return ["disc", "disc2", "macula"]
    return [target]


def _resolve_runtime_dirs(layout: OutputLayout) -> tuple[Path, Path]:
    run_token = uuid.uuid4().hex[:10]
    runtime_input = layout.tmp_input / f"run_{run_token}"
    runtime_work = layout.work_segmentation / f"run_{run_token}"
    runtime_input.mkdir(parents=True, exist_ok=True)
    runtime_work.mkdir(parents=True, exist_ok=True)
    return runtime_input, runtime_work


def _remove_flag(args: list[str], flag: str) -> list[str]:
    return [item for item in args if item != flag]


def _remove_option(args: list[str], flag: str, *, optional_value: bool = False) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item != flag:
            normalized.append(item)
            index += 1
            continue
        index += 1
        if index < len(args) and (optional_value or not args[index].startswith("--")):
            index += 1
    return normalized


def _toggle_flag(args: list[str], flag: str, enabled: bool) -> list[str]:
    normalized = _remove_flag(args, flag)
    if enabled:
        normalized.append(flag)
    return normalized


def _upsert_option(args: list[str], flag: str, value: str, *, optional_value: bool = False) -> list[str]:
    normalized = _remove_option(args, flag, optional_value=optional_value)
    normalized.extend([flag, value])
    return normalized


def _pipeline_histogram_key(pipeline: str) -> str:
    return "histogram_triton" if pipeline == "fda" else "histogram_3doct"


def _segmentation_flags_for_source(pipeline: str, segmentation_source: str) -> list[str]:
    if pipeline != "fda":
        return ["--ai"]
    if segmentation_source == "vendor":
        return ["--topcon"]
    if segmentation_source == "auto":
        return ["--ai", "--topcon"]
    return ["--ai"]


def _apply_request_overrides(
    config: dict[str, Any],
    request: PipelineRequest,
    paths: dict[str, Path],
    *,
    pipeline: str,
    segmentation_source: str,
) -> dict[str, Any]:
    segmentation_args = list(config["segmentation"]["extra_args"])
    analysis_base_args = list(config["analysis"]["base_args"])
    enface_args = list(config.get("enface", {}).get("extra_args", []))

    for flag in ("--fda", "--dcm", "--npy"):
        segmentation_args = _remove_flag(segmentation_args, flag)
    segmentation_args.append(PIPELINE_FORMAT_FLAGS[pipeline])

    for flag in ("--ai", "--topcon"):
        segmentation_args = _remove_flag(segmentation_args, flag)
    segmentation_args.extend(_segmentation_flags_for_source(pipeline, segmentation_source))

    if request.slice_overlays != "profile":
        segmentation_args = _toggle_flag(segmentation_args, "--overlay_slice", request.slice_overlays == "on")

    if request.volume_preproc != "profile":
        segmentation_args = _remove_option(segmentation_args, "--hist_match", optional_value=True)
        segmentation_args = _remove_option(segmentation_args, "--avg")
        segmentation_args = _remove_flag(segmentation_args, "--clahe3d")
        if request.volume_preproc == "on":
            histogram_key = _pipeline_histogram_key(pipeline)
            segmentation_args.extend(
                [
                    "--hist_match",
                    str(paths[histogram_key]),
                    "--avg",
                    "1",
                    "--clahe3d",
                ]
            )

    if request.rotate_npy:
        if pipeline != "vol_npy":
            raise ValueError("--rotate-npy is only supported for vol_npy runs.")
        segmentation_args = _upsert_option(segmentation_args, "--rotate_npy", request.rotate_npy)

    if pipeline in {"dicom", "fda"}:
        analysis_seg_mode = "topcon" if segmentation_source == "vendor" else segmentation_source
        analysis_base_args = _upsert_option(analysis_base_args, "--seg_mode", analysis_seg_mode)

    if pipeline == "vol_npy" and request.allow_any_shape:
        enface_args = _toggle_flag(enface_args, "--allow_any_shape", True)
        enface_args = _remove_flag(enface_args, "--allow_h512")
        enface_args = _remove_flag(enface_args, "--allow_h885")

    config["segmentation"]["extra_args"] = segmentation_args
    config["analysis"]["base_args"] = analysis_base_args
    if "enface" in config:
        config["enface"]["extra_args"] = enface_args
    return config


def _normalize_path_args(args: list[str], *, flags: set[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        normalized.append(item)
        if item in flags and index + 1 < len(args):
            normalized.append(str(resolve_project_path(args[index + 1])))
            index += 2
            continue
        index += 1
    return normalized


def _prepare_segmentation_command(
    *,
    python_cmd: str,
    seg_script: Path,
    seg_ckpt: Path,
    input_path: Path,
    gpu: str,
    extra_args: list[str],
) -> list[str]:
    normalized_args = _normalize_path_args(extra_args, flags={"--hist_match"})
    return [
        python_cmd,
        str(seg_script),
        "--src",
        str(input_path),
        "--ckpt",
        str(seg_ckpt),
        "--gpu",
        gpu,
        *normalized_args,
    ]


def _prepare_analysis_command(
    *,
    python_cmd: str,
    analysis_script: Path,
    thick_dir: Path,
    out_dir: Path,
    gpu: str,
    wnet_ckpt: Path,
    ai_ckpt: Path,
    base_args: list[str],
    extra_args: list[str],
    oct_root: Path | None = None,
    enface_dir: Path | None = None,
    center_out: Path | None = None,
) -> list[str]:
    cmd = [
        python_cmd,
        str(analysis_script),
        "--thick_dir",
        str(thick_dir),
        "--out_dir",
        str(out_dir),
        "--gpu",
        gpu,
        "--wnet_ckpt",
        str(wnet_ckpt),
        "--ai_ckpt",
        str(ai_ckpt),
        *base_args,
    ]
    if oct_root is not None:
        cmd.extend(["--oct_root", str(oct_root)])
    if enface_dir is not None:
        cmd.extend(["--enface_dir", str(enface_dir)])
    if center_out is not None:
        cmd.extend(["--center_out", str(center_out)])
    cmd.extend(extra_args)
    return cmd


def _prepare_enface_command(
    *,
    python_cmd: str,
    enface_script: Path,
    vol_root: Path,
    seg_root: Path,
    out_dir: Path,
    cdf_path: Path,
    extra_args: list[str],
) -> list[str]:
    normalized_args = _normalize_path_args(extra_args, flags={"--cdf"})
    return [
        python_cmd,
        str(enface_script),
        "--vol_root",
        str(vol_root),
        "--seg_root",
        str(seg_root),
        "--out_dir",
        str(out_dir),
        "--cdf",
        str(cdf_path),
        *normalized_args,
    ]


def _stage_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
        mode = "symlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"
    return {"source": str(source), "staged_path": str(destination), "mode": mode}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen_paths: set[Path] = set()
    seen_ids: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        case_id = path.stem
        if case_id in seen_ids and seen_ids[case_id] != resolved:
            raise ValueError(f"Duplicate case id '{case_id}' resolved to multiple files.")
        seen_paths.add(resolved)
        seen_ids[case_id] = resolved
        ordered.append(resolved)
    return ordered


def _candidate_manifest_paths(
    entry: str,
    *,
    manifest_path: Path,
    input_root: Path | None,
    allowed_suffixes: tuple[str, ...],
) -> list[Path]:
    token_path = Path(entry).expanduser()
    candidates: list[Path] = []

    if token_path.is_absolute():
        candidates.append(token_path)
    else:
        candidates.append(manifest_path.parent / token_path)
        if not token_path.suffix:
            for suffix in allowed_suffixes:
                candidates.append(manifest_path.parent / f"{entry}{suffix}")
        if input_root is not None:
            candidates.append(input_root / token_path)
            if not token_path.suffix:
                for suffix in allowed_suffixes:
                    candidates.append(input_root / f"{entry}{suffix}")
            else:
                candidates.append(input_root / token_path.name)

    return candidates


def _resolve_manifest_paths(
    entries: list[str],
    *,
    manifest_path: Path,
    input_root: Path | None,
    allowed_suffixes: tuple[str, ...],
) -> list[Path]:
    resolved: list[Path] = []
    for entry in entries:
        chosen: Path | None = None
        for candidate in _candidate_manifest_paths(
            entry,
            manifest_path=manifest_path,
            input_root=input_root,
            allowed_suffixes=allowed_suffixes,
        ):
            candidate = candidate.expanduser()
            if not candidate.exists() or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in allowed_suffixes:
                continue
            chosen = candidate.resolve()
            break
        if chosen is None:
            raise FileNotFoundError(f"Could not resolve manifest entry '{entry}' for {manifest_path.name}.")
        resolved.append(chosen)
    return _dedupe_paths(resolved)


def _collect_input_paths(request: PipelineRequest, *, pipeline: str) -> tuple[list[Path], list[str]]:
    allowed_suffixes = PIPELINE_SUFFIXES[pipeline]
    if request.manifest_path is not None:
        if request.input_path is not None and request.input_path.is_file():
            raise ValueError("--input must be a directory when used together with --manifest.")
        manifest_entries = read_manifest_entries(request.manifest_path)
        input_root = request.input_path if request.input_path is not None else None
        paths = _resolve_manifest_paths(
            manifest_entries,
            manifest_path=request.manifest_path,
            input_root=input_root,
            allowed_suffixes=allowed_suffixes,
        )
        return paths, manifest_entries

    if request.input_path is None:
        raise ValueError("Either --input or --manifest must be provided.")

    input_path = request.input_path
    if input_path.is_dir():
        collected = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
        if not collected:
            raise FileNotFoundError(f"No {allowed_suffixes} files found under {input_path}.")
        return _dedupe_paths(collected), []

    if input_path.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"{input_path} does not match the expected suffixes for {pipeline}: {allowed_suffixes}")
    return [input_path.resolve()], []


def _prepare_standard_input(
    *,
    selected_paths: list[Path],
    manifest_entries: list[str],
    request: PipelineRequest,
    runtime_input_dir: Path,
) -> InputSelection:
    selected_ids = [path.stem for path in selected_paths]
    if request.manifest_path is None:
        if request.input_path is None:
            raise ValueError("input_path is required for non-manifest runs.")
        if request.input_path.is_dir():
            return InputSelection(
                selected_paths=selected_paths,
                selected_ids=selected_ids,
                manifest_entries=manifest_entries,
                staging_rows=[],
                active_input_path=request.input_path,
                oct_root=request.input_path,
            )
        return InputSelection(
            selected_paths=selected_paths,
            selected_ids=selected_ids,
            manifest_entries=manifest_entries,
            staging_rows=[],
            active_input_path=selected_paths[0],
            oct_root=selected_paths[0].parent,
        )

    staging_rows = [_stage_file(path, runtime_input_dir / path.name) for path in selected_paths]
    return InputSelection(
        selected_paths=selected_paths,
        selected_ids=selected_ids,
        manifest_entries=manifest_entries,
        staging_rows=staging_rows,
        active_input_path=runtime_input_dir,
        oct_root=runtime_input_dir,
    )


def _collect_valid_volumes(
    volume_paths: list[Path],
    staging_dir: Path,
    *,
    allowed_heights: list[int],
    allow_any_shape: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    collected_ids: list[str] = []
    inspection_rows: list[dict[str, Any]] = []

    for path in volume_paths:
        record: dict[str, Any] = {"file": str(path), "accepted": False}
        try:
            array = np.load(path, mmap_mode="r")
        except Exception as exc:
            record["error"] = str(exc)
            inspection_rows.append(record)
            continue

        record["shape"] = list(array.shape)
        record["dtype"] = str(array.dtype)

        if array.ndim != 3:
            record["reason"] = "ndim != 3"
            inspection_rows.append(record)
            continue

        if not allow_any_shape:
            if array.shape[2] != 512:
                record["reason"] = "width != 512"
                inspection_rows.append(record)
                continue
            if int(array.shape[1]) not in allowed_heights:
                record["reason"] = f"height not in {allowed_heights}"
                inspection_rows.append(record)
                continue

        stage_info = _stage_file(path, staging_dir / path.name)
        record["accepted"] = True
        record["staged_path"] = stage_info["staged_path"]
        record["stage_mode"] = stage_info["mode"]
        record["case_id"] = path.stem
        collected_ids.append(path.stem)
        inspection_rows.append(record)

    return collected_ids, inspection_rows


def _collect_segmentation_outputs(
    source_dir: Path,
    output_dir: Path,
    expected_ids: list[str],
) -> list[str]:
    moved: list[str] = []
    expected = set(expected_ids)

    for path in sorted(source_dir.glob("*_seg.npy")):
        stem_id = path.stem.split("_")[0]
        if expected and stem_id not in expected:
            continue
        target = output_dir / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))
        moved.append(target.name)

    return moved


def _profile_summary(
    request: PipelineRequest,
    *,
    pipeline: str,
    segmentation_source: str,
    config: dict[str, Any],
    layout: OutputLayout,
    command_log: list[dict[str, Any]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "requested_pipeline": request.pipeline,
        "input_path": str(request.input_path) if request.input_path else None,
        "manifest_path": str(request.manifest_path) if request.manifest_path else None,
        "output_root": str(request.output_root),
        "gpu": request.gpu,
        "target": request.target,
        "segmentation_source": segmentation_source,
        "slice_overlays": request.slice_overlays,
        "volume_preproc": request.volume_preproc,
        "allow_any_shape": request.allow_any_shape,
        "rotate_npy": request.rotate_npy,
        "dry_run": request.dry_run,
        "project_root": str(PROJECT_ROOT),
        "layout": {
            "tmp_input": str(layout.tmp_input),
            "npy_seg": str(layout.npy_seg),
            "enface": str(layout.enface),
            "centroid_out": str(layout.centroid_out),
            "results": str(layout.results),
            "logs": str(layout.logs),
            "manifests": str(layout.manifests),
            "work_segmentation": str(layout.work_segmentation),
        },
        "config": config,
        "commands": command_log,
        **extra,
    }


def run_pipeline(request: PipelineRequest) -> Path:
    pipeline = _canonical_pipeline(request.pipeline)
    segmentation_source = _canonical_segmentation_source(request.segmentation_source, pipeline)

    config = load_pipeline_config(pipeline, request.config_path)
    paths = _resolve_paths(config)
    config = _apply_request_overrides(
        config,
        request,
        paths,
        pipeline=pipeline,
        segmentation_source=segmentation_source,
    )

    selected_paths, manifest_entries = _collect_input_paths(request, pipeline=pipeline)
    python_cmd = _resolve_python_cmd(config)
    layout = OutputLayout.create(request.output_root)
    layout.ensure(include_tmp_input=True)
    runtime_input_dir, runtime_work_dir = _resolve_runtime_dirs(layout)

    command_env = os.environ.copy()
    command_env["CUDA_VISIBLE_DEVICES"] = request.gpu
    pythonpath_parts = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
    existing_pythonpath = command_env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    command_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    command_log: list[dict[str, Any]] = []
    manifest_path = layout.manifests / "run_manifest.json"

    targets = _pipeline_targets(request.target)
    analysis_cfg = config["analysis"]
    seg_cfg = config["segmentation"]

    extra: dict[str, Any] = {
        "targets": targets,
        "resolved_inputs": [str(path) for path in selected_paths],
        "manifest_entries": manifest_entries,
        "runtime_input_dir": str(runtime_input_dir),
        "runtime_work_dir": str(runtime_work_dir),
    }

    if pipeline in {"dicom", "fda"}:
        selection = _prepare_standard_input(
            selected_paths=selected_paths,
            manifest_entries=manifest_entries,
            request=request,
            runtime_input_dir=runtime_input_dir,
        )
        extra["selected_ids"] = selection.selected_ids
        if selection.staging_rows:
            extra["staging"] = selection.staging_rows
            _write_json(layout.manifests / "selected_inputs.json", {"files": selection.staging_rows})

        seg_cmd = _prepare_segmentation_command(
            python_cmd=python_cmd,
            seg_script=paths["segmentation_script"],
            seg_ckpt=paths["segmentation_checkpoint"],
            input_path=selection.active_input_path,
            gpu=request.gpu,
            extra_args=seg_cfg["extra_args"],
        )
        _run_command(
            seg_cmd,
            cwd=runtime_work_dir,
            label="segmentation",
            dry_run=request.dry_run,
            command_log=command_log,
            env=command_env,
        )

        seg_source = runtime_work_dir / "data"
        moved = [] if request.dry_run else _collect_segmentation_outputs(seg_source, layout.npy_seg, selection.selected_ids)
        extra["segmentation_outputs"] = moved

        for target in targets:
            script_key = f"{target}_script" if target != "disc2" else "disc2_script"
            analysis_script = paths[analysis_cfg["scripts"][script_key]]
            target_args = analysis_cfg["target_args"][target]
            cmd = _prepare_analysis_command(
                python_cmd=python_cmd,
                analysis_script=analysis_script,
                thick_dir=layout.npy_seg,
                out_dir=layout.results,
                gpu=request.gpu,
                wnet_ckpt=paths["wnet_checkpoint"],
                ai_ckpt=paths["segmentation_checkpoint"],
                base_args=analysis_cfg["base_args"],
                extra_args=target_args,
                oct_root=selection.oct_root,
                center_out=layout.centroid_out,
            )
            _run_command(
                cmd,
                cwd=PROJECT_ROOT,
                label=f"analysis:{target}",
                dry_run=request.dry_run,
                command_log=command_log,
                env=command_env,
            )

    elif pipeline == "vol_npy":
        allowed_heights = [int(height) for height in config["staging"]["allowed_heights"]]
        ids, inspection_rows = _collect_valid_volumes(
            selected_paths,
            runtime_input_dir,
            allowed_heights=allowed_heights,
            allow_any_shape=request.allow_any_shape,
        )
        if not ids:
            raise ValueError("No valid vol_npy inputs were accepted. Check the manifest and shape constraints.")
        extra["selected_ids"] = ids
        extra["staging"] = inspection_rows
        _write_json(layout.manifests / "volumes.json", {"volumes": inspection_rows})

        seg_cmd = _prepare_segmentation_command(
            python_cmd=python_cmd,
            seg_script=paths["segmentation_script"],
            seg_ckpt=paths["segmentation_checkpoint"],
            input_path=runtime_input_dir,
            gpu=request.gpu,
            extra_args=seg_cfg["extra_args"],
        )
        _run_command(
            seg_cmd,
            cwd=runtime_work_dir,
            label="segmentation",
            dry_run=request.dry_run,
            command_log=command_log,
            env=command_env,
        )

        seg_source = runtime_work_dir / "data"
        moved = [] if request.dry_run else _collect_segmentation_outputs(seg_source, layout.npy_seg, ids)
        extra["segmentation_outputs"] = moved

        enface_cmd = _prepare_enface_command(
            python_cmd=python_cmd,
            enface_script=paths["enface_flexible_script"],
            vol_root=runtime_input_dir,
            seg_root=layout.npy_seg,
            out_dir=layout.enface,
            cdf_path=paths["histogram_3doct"],
            extra_args=config["enface"]["extra_args"],
        )
        _run_command(
            enface_cmd,
            cwd=PROJECT_ROOT,
            label="enface",
            dry_run=request.dry_run,
            command_log=command_log,
            env=command_env,
        )

        for target in targets:
            script_key = f"{target}_script" if target != "disc2" else "disc2_script"
            analysis_script = paths[analysis_cfg["scripts"][script_key]]
            target_args = analysis_cfg["target_args"][target]
            cmd = _prepare_analysis_command(
                python_cmd=python_cmd,
                analysis_script=analysis_script,
                thick_dir=layout.npy_seg,
                out_dir=layout.results,
                gpu=request.gpu,
                wnet_ckpt=paths["wnet_checkpoint"],
                ai_ckpt=paths["segmentation_checkpoint"],
                base_args=analysis_cfg["base_args"],
                extra_args=target_args,
                enface_dir=layout.enface,
                center_out=layout.centroid_out,
            )
            _run_command(
                cmd,
                cwd=PROJECT_ROOT,
                label=f"analysis:{target}",
                dry_run=request.dry_run,
                command_log=command_log,
                env=command_env,
            )
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    summary = _profile_summary(
        request,
        pipeline=pipeline,
        segmentation_source=segmentation_source,
        config=config,
        layout=layout,
        command_log=command_log,
        extra=extra,
    )
    _write_json(manifest_path, summary)
    return manifest_path
