#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as skt
from skimage import exposure

matplotlib.use("Agg")


EF_SIZE = 512
TARGET_H = 512
TARGET_W = 992
SLAB_OFFSET = 5

try:
    from octseg.legacy import LEGACY_FILES

    DEFAULT_CDF = LEGACY_FILES["histogram_3doct"]
except ImportError:
    _cdf_override = os.environ.get("OPHAGENT_G_DISC_3DOCT_HISTOGRAM", "")
    DEFAULT_CDF = Path(_cdf_override) if _cdf_override else Path("3DOCT_hist.cdf")

SURFS = [
    "ILM", "RNFL_GCL", "GCL_IPL", "IPL_INL", "INL_OPL",
    "OPL_ONL", "ONL_ELM", "ELM", "RPE_BM", "CHOROID_OUT",
]
IDX = {name: index for index, name in enumerate(SURFS)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _average_neighbor(volume: np.ndarray, k: int = 0) -> np.ndarray:
    if k <= 0:
        return volume
    padded = np.pad(volume, ((k, k), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(volume, dtype=np.float32)
    for index in range(volume.shape[0]):
        out[index] = padded[index:index + 2 * k + 1].mean(axis=0)
    return out


def _hist_match_volume(volume: np.ndarray, ref_cdf: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    ref_cdf = ref_cdf.astype(np.float32)
    ref_cdf /= ref_cdf[-1]
    ref_inv = np.interp(np.linspace(0, 1, 256), ref_cdf, np.arange(256))

    volume_uint8 = np.clip(volume, 0, 255).astype(np.uint8)
    src_hist = np.bincount(volume_uint8.ravel(), minlength=256).astype(np.float32)
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1]
    lut = np.interp(src_cdf, np.linspace(0, 1, 256), ref_inv).astype(np.float32)

    out = volume.astype(np.float32, copy=True)
    for z_index in range(out.shape[0]):
        mapped = lut[volume_uint8[z_index]]
        out[z_index] = (1.0 - alpha) * out[z_index] + alpha * mapped
    return out


def _clahe3d(
    volume: np.ndarray,
    alpha: float = 0.2,
    clip: float = 0.01,
    z_tile: int = 4,
    tile_factor: int = 6,
) -> np.ndarray:
    v = volume.astype(np.float32) / 255.0
    z_size, height, width = v.shape
    v_clahe = exposure.equalize_adapthist(
        v,
        kernel_size=(z_tile, max(1, height // tile_factor), max(1, width // tile_factor)),
        clip_limit=clip,
    )
    v_mix = (1.0 - alpha) * v + alpha * v_clahe
    return (v_mix * 255.0).astype(np.float32)


def load_vol_npy(
    vol_path: Path,
    *,
    allow_h512: bool,
    allow_h885: bool,
    allow_any_shape: bool,
    verbose: bool,
) -> Optional[np.ndarray]:
    try:
        volume = np.load(vol_path)
    except Exception as exc:
        if verbose:
            print(f"[WARN] failed to load {vol_path}: {exc}")
        return None

    if volume.ndim != 3:
        if verbose:
            print(f"[SKIP] {vol_path.name}: ndim={volume.ndim}, expected 3")
        return None

    if allow_any_shape:
        if verbose:
            print(f"[OK] loaded {vol_path.name}: shape={volume.shape} [allow_any_shape]")
        return volume.astype(np.float32)

    _, height, width = volume.shape
    allowed_shapes: list[tuple[int, int]] = []
    if allow_h512:
        allowed_shapes.append((512, 512))
    if allow_h885:
        allowed_shapes.append((885, 512))

    if (height, width) not in allowed_shapes:
        if verbose:
            print(f"[SKIP] {vol_path.name}: shape={volume.shape}, allowed HW={allowed_shapes}")
        return None

    if verbose:
        print(f"[OK] loaded {vol_path.name}: shape={volume.shape}")

    return volume.astype(np.float32)


def preprocess_and_resize(
    volume_nhw: np.ndarray,
    *,
    cdf_path: Optional[Path],
    avg_neighbor: int,
) -> np.ndarray:
    volume = _average_neighbor(volume_nhw, avg_neighbor)

    if cdf_path and cdf_path.exists():
        try:
            ref_cdf = joblib.load(cdf_path)
        except Exception:
            ref_cdf = np.loadtxt(cdf_path, dtype=np.float32)
        volume = _hist_match_volume(volume, ref_cdf, alpha=0.4)

    volume = _clahe3d(volume, alpha=0.2, clip=0.01, z_tile=4, tile_factor=6)
    volume_hwn = volume.transpose(1, 2, 0)

    if volume_hwn.shape[0] != TARGET_H or volume_hwn.shape[1] != TARGET_W:
        resized = []
        for z_index in range(volume_hwn.shape[2]):
            slice_2d = volume_hwn[:, :, z_index]
            resized_slice = skt.resize(
                slice_2d,
                (TARGET_H, TARGET_W),
                preserve_range=True,
                order=1,
                anti_aliasing=True,
            )
            resized.append(resized_slice.astype(np.float32))
        volume_hwn = np.stack(resized, axis=-1)

    return volume_hwn.astype(np.float32)


def slab_enface(volume_hwn: np.ndarray, top_wz: np.ndarray, bot_wz: np.ndarray) -> np.ndarray:
    height, width, depth = volume_hwn.shape
    enface = np.zeros((width, depth), np.float32)

    for z_index in range(depth):
        for x_index in range(width):
            y_top = float(top_wz[x_index, z_index])
            y_bottom = float(bot_wz[x_index, z_index]) + SLAB_OFFSET

            if not np.isfinite(y_top) or not np.isfinite(y_bottom):
                continue

            y_top_i = int(round(y_top))
            y_bottom_i = int(round(y_bottom))
            if y_top_i < 0 or y_bottom_i < 0 or y_top_i >= height or y_bottom_i >= height or y_top_i >= y_bottom_i:
                continue

            enface[x_index, z_index] = float(volume_hwn[y_top_i:y_bottom_i + 1, x_index, z_index].mean())

    return enface


def save_enface(image_2d: np.ndarray, out_png: Path) -> None:
    arr = image_2d.astype(np.float32)
    arr = arr - np.nanmin(arr)
    max_value = np.nanmax(arr)
    if max_value > 0:
        arr = arr / max_value
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    arr = skt.resize(
        arr,
        (EF_SIZE, EF_SIZE),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    ensure_dir(out_png.parent)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap="gray", origin="upper")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def resolve_allowed_heights(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.allow_h512 or args.allow_h885:
        return args.allow_h512, args.allow_h885
    if args.mode == "885":
        return False, True
    return True, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol_root", required=True, help="vol_npy root (ID.npy)")
    parser.add_argument("--seg_root", required=True, help="npy_seg root (ID*_seg.npy)")
    parser.add_argument("--out_dir", required=True, help="enface output dir")
    parser.add_argument("--cdf", default=str(DEFAULT_CDF), help="histogram reference cdf")
    parser.add_argument("--avg", type=int, default=0, help="±avg slices mean (0=off)")
    parser.add_argument("--prefer_bm", action="store_true", help="BMが作れない場合にALLのみでも保存")
    parser.add_argument("--allow_h512", action="store_true", help="(N,512,512) を許可")
    parser.add_argument("--allow_h885", action="store_true", help="(N,885,512) を許可")
    parser.add_argument("--allow_any_shape", action="store_true", help="任意の3D shapeを許可して内部で resize する")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mode", choices=["flexible", "885"], default="flexible", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    vol_root = Path(args.vol_root)
    seg_root = Path(args.seg_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    cdf_path = Path(args.cdf) if args.cdf else None
    allow_h512, allow_h885 = resolve_allowed_heights(args)

    seg_paths = sorted(seg_root.glob("*_seg.npy"))
    if not seg_paths:
        raise SystemExit(f"no *_seg.npy under {seg_root}")

    processed = 0
    skipped_vol_missing = 0
    skipped_vol_shape = 0
    skipped_seg_shape = 0

    for seg_path in seg_paths:
        case_id = seg_path.stem.split("_")[0]
        vol_path = vol_root / f"{case_id}.npy"

        if not vol_path.exists():
            skipped_vol_missing += 1
            if args.verbose:
                print(f"[SKIP] vol missing: {vol_path}")
            continue

        volume_nhw = load_vol_npy(
            vol_path,
            allow_h512=allow_h512,
            allow_h885=allow_h885,
            allow_any_shape=args.allow_any_shape,
            verbose=args.verbose,
        )
        if volume_nhw is None:
            skipped_vol_shape += 1
            continue

        surfaces = np.load(seg_path)
        if surfaces.ndim != 3 or surfaces.shape[1] != TARGET_W:
            skipped_seg_shape += 1
            if args.verbose:
                print(f"[SKIP] seg shape invalid: {seg_path.name}, shape={surfaces.shape}")
            continue

        volume_hwn = preprocess_and_resize(
            volume_nhw,
            cdf_path=cdf_path,
            avg_neighbor=int(args.avg),
        )

        full = np.rot90(volume_hwn.mean(axis=0), 1)
        save_enface(full, out_dir / f"{case_id}_enface_ALL.png")

        if ("ELM" in IDX) and ("RPE_BM" in IDX) and surfaces.shape[0] > IDX["RPE_BM"]:
            top = surfaces[IDX["ELM"]]
            bottom = surfaces[IDX["RPE_BM"]]
            bm = slab_enface(volume_hwn, top, bottom)
            bm = np.rot90(bm, 1)
            save_enface(bm, out_dir / f"{case_id}_enface_BM.png")
        elif args.prefer_bm and args.verbose:
            print(f"[INFO] BM slab not available for {case_id}")

        processed += 1

    print(f"[DONE] enface -> {out_dir}")
    print(f"[INFO] processed={processed}")
    print(f"[INFO] skipped_vol_missing={skipped_vol_missing}")
    print(f"[INFO] skipped_vol_shape={skipped_vol_shape}")
    print(f"[INFO] skipped_seg_shape={skipped_seg_shape}")


if __name__ == "__main__":
    main()
