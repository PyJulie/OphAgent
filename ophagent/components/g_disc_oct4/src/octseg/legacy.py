from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _asset_path(env_var: str, filename: str) -> Path:
    override = os.environ.get(env_var, "").strip()
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override))).resolve()
    runtime = Path(
        os.environ.get(
            "OPHAGENT_RUNTIME_DIR",
            str(Path.home() / ".ophagent"),
        )
    ).expanduser()
    checkpoint_root = Path(
        os.environ.get(
            "OPHAGENT_CKPT_DIR",
            str(runtime / "checkpoints"),
        )
    ).expanduser()
    return (checkpoint_root / "oct_volume" / "g_disc" / filename).resolve()


LEGACY_FILES = {
    "segmentation_script": project_path(
        "legacy_pipeline", "seg1d2d_aspectfix.py"
    ),
    "analysis_script": project_path(
        "legacy_pipeline", "oct_all_in_one_patched.py"
    ),
    "analysis_disc2_script": project_path(
        "legacy_pipeline", "oct_all_in_one-bk3.py"
    ),
    "enface_flexible_script": project_path(
        "legacy_pipeline", "make_enface_from_volnpy.py"
    ),
    "enface_885_script": project_path(
        "legacy_pipeline", "make_enface_from_volnpy.py"
    ),
    "predict_disc_script": project_path(
        "legacy_pipeline", "predict_disc_centroid.py"
    ),
    "segmentation_checkpoint": _asset_path(
        "OPHAGENT_G_DISC_SEGMENTATION_WEIGHTS",
        "tohoku_full_slss_013_fold0_3.522%.t7",
    ),
    "wnet_checkpoint": _asset_path(
        "OPHAGENT_G_DISC_WNET_WEIGHTS",
        "wnet_disc512_best_UCL-new3-0.793.pth",
    ),
    "histogram_3doct": _asset_path(
        "OPHAGENT_G_DISC_3DOCT_HISTOGRAM",
        "3DOCT_hist.cdf",
    ),
    "histogram_triton": _asset_path(
        "OPHAGENT_G_DISC_TRITON_HISTOGRAM",
        "Triton_hist.cdf",
    ),
}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_path(str(path))
