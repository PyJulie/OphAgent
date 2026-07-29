#!/usr/bin/env python3
"""
make_enface_bm.py  (v3 — 2025‑06‑28)

Topcon OCT En‑face Generator (ELM–RPE_BM slab)
==============================================

OVERVIEW
    This utility creates two kinds of grey‑scale en‑face projections from
    volumetric OCT data:
        • *ELM–RPE_BM slab* – mean‑intensity projection between the ELM and
          Bruch’s membrane (configurable offset; default +5 px below RPE_BM).
        • *Full‑volume average* – simple mean of all voxels, exported as a
          reference image (`*_enface_ALL.png`).

    Supported input formats are
        • Topcon STEP files (`*.fda`)
        • multi‑frame DICOM (`*.dcm`)
    Single‑frame DICOMs are detected and silently skipped with a warning.

WHAT’S NEW IN v3
    • Robust skip of single‑frame DICOM – prevents crashes on ophthalmic
      reports accidentally saved as 2‑D DICOMs.
    • `load_volume()` returns *None* rather than raising, letting the caller
      decide how to proceed.
    • `process_one()` now exits early when `load_volume()` returns None.
    • Added full‑volume en‑face output (`*_enface_ALL.png`).

QUICK EXAMPLES
    # Auto‑select Topcon surfaces if available, otherwise fall back to AI.
    python make_enface_bm.py ./folder_with_fda_and_dcm --seg auto \
        --ckpt model.ckpt --gpu 0

    # Process files listed in an Excel sheet, Topcon‑only segmentation.
    python make_enface_bm.py patient_list.xlsx --seg topcon

COMMAND‑LINE ARGUMENTS
    src
        File, folder, or Excel list of IDs to process.

    --seg {topcon|ai|auto}
        Segmentation source:
            topcon – use surfaces embedded in *.fda only.
            ai     – run the AI model everywhere.
            auto   – try Topcon first; fall back to AI if missing.
        Default `auto`.

    --ckpt <FILE>
        PyTorch checkpoint for AI segmentation. Required when --seg is `ai`
        or when `auto` cannot find Topcon surfaces.

    --gpu <IDX>
        GPU index for AI inference. Default 0.

    --root <DIR>
        Root directory containing patient files referenced by an Excel list.
        Defaults to the current directory.

OUTPUTS (per patient)
    ./enface_out_bm/<VID>_enface_BM.png     – ELM–RPE_BM slab
    ./enface_out_bm/<VID>_enface_ALL.png    – full‑volume mean

IMPORTANT CONSTANTS
    EF_SIZE        = 512    final en‑face size (px × px)
    TARGET_H/W     = 512 / 992  resizing target for B‑scans
    AVG_NEIGHBOR   = 0      ±1‑slice averaging. Change to 1 for smoothing.
    HIST_ALPHA     = 0.4    blend weight for global histogram matching.
    CLIP_LIMIT     = 0.01   3‑D CLAHE clip limit.
    CDF_PATH       = path to reference cumulative histogram.
    SLAB_OFFSET    = 5      extra pixels included *below* RPE_BM.

    Modify these in the *constants* section near the top of the script.

CLAHE / PRE‑PROCESSING PIPELINE
    1. ±{AVG_NEIGHBOR}‑slice mean (if AVG_NEIGHBOR > 0)
    2. Global histogram matching against `CDF_PATH` (α = HIST_ALPHA)
    3. Volume‑wide 3‑D CLAHE
       Parameters are hard‑coded inside `_clahe3d()`:
           alpha       0.2   blend with original volume
           clip        0.01  CLAHE clip limit
           z_tile      4     number of tiles along Z
           tile_factor 6     XY tile divisor (tile = H//factor × W//factor)

SURFACE DETECTION
    The script needs *two* surfaces:
        • ELM (External Limiting Membrane)
        • RPE_BM (Bruch’s membrane)

    Surface names vary across devices and algorithms. The alias map
    `ALIAS` resolves common alternatives (e.g. `ONL_ELM`, `BM`, `RPE`).
    If either surface is missing after both Topcon and AI attempts, the
    file is skipped.

NOTES & LIMITATIONS
    • AI inference uses a minimal dataset wrapper when a single *.fda* or
      *.dcm* file is passed – no extra disk writes.
    • Resizing logic always forces width to 992 px. Adjust `TARGET_W` to
      match different acquisition geometries.
    • `--seg ai` on an *.fda* file ignores any built‑in surfaces.

Copyright 2025 Takahiro Ninomiya.
The underlying 1D+2D U-Net architecture was developed by Tsubasa Konno.

"""

from __future__ import annotations
import os, sys, argparse, gc, warnings, math
from pathlib import Path
import numpy as np
import joblib  # ← binary CDF loader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.transform as skt
import skimage.exposure as ex
import pandas as pd
from tqdm import tqdm
from oct_converter.readers import FDA
import pydicom  # DICOM support

# ────────── 定数 ──────────
EF_SIZE, TARGET_H, TARGET_W = 512, 512, 992
CLIP_LIMIT = 0.01
OUT_DIR = Path("./enface_out_bm")
SURF_KEYS = ("ELM", "RPE_BM")
try:
    from octseg.legacy import LEGACY_FILES

    CDF_PATH = LEGACY_FILES["histogram_triton"]
except ImportError:
    _cdf_override = os.environ.get("OPHAGENT_G_DISC_TRITON_HISTOGRAM", "")
    CDF_PATH = Path(_cdf_override) if _cdf_override else Path("Triton_hist.cdf")
AVG_NEIGHBOR = 0            # ±1 枚平均
HIST_ALPHA   = 0.4          # ヒストグラムマッチング強度

# 置き換え用コード -------------------------------------------------
ALIAS = {
    "ELM"    : ["ELM", "ONL_ELM"],
    "RPE_BM" : ["RPE_BM", "BM", "IZ_RPE", "RPE"],  # BM を最優先
}

def pick_surface(seg: dict, cand: list[str]) -> str | None:
    for n in cand:
        if n in seg:
            return n
    return None

# ────────── 前処理ユーティリティ ──────────

def _hist_match_volume(vol: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """apply histogram‑matching with global α=HIST_ALPHA"""
    ref_cdf = ref_cdf.astype(np.float32); ref_cdf /= ref_cdf[-1]
    ref_inv = np.interp(np.linspace(0, 1, 256), ref_cdf, np.arange(256))
    vol_uint8 = np.clip(vol, 0, 255).astype(np.uint8)
    src_hist = np.bincount(vol_uint8.ravel(), minlength=256).astype(np.float32)
    src_cdf  = np.cumsum(src_hist); src_cdf /= src_cdf[-1]
    lut = np.interp(src_cdf, np.linspace(0, 1, 256), ref_inv).astype(np.float32)
    out = vol.astype(np.float32, copy=True)
    for z in range(out.shape[0]):
        mapped = lut[vol_uint8[z]]
        out[z] = (1.0 - HIST_ALPHA) * out[z] + HIST_ALPHA * mapped
    return out


def _clahe3d(vol: np.ndarray, *, alpha: float = 0.2, clip: float = 0.01,
             z_tile: int = 4, tile_factor: int = 6) -> np.ndarray:
    """3‑D CLAHE 実装 (γ=1.0 固定)"""
    v = vol.astype(np.float32) / 255.0
    from skimage import exposure
    z, h, w = v.shape
    v_clahe = exposure.equalize_adapthist(
        v,
        kernel_size=(z_tile, h // tile_factor, w // tile_factor),
        clip_limit=clip,
    )
    v_mix = (1.0 - alpha) * v + alpha * v_clahe
    return (v_mix * 255.0).astype(np.float32)


def _average_neighbor(vol: np.ndarray, k: int = 1) -> np.ndarray:
    if k <= 0:
        return vol
    padded = np.pad(vol, ((k, k), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(vol)
    for i in range(vol.shape[0]):
        out[i] = padded[i:i + 2 * k + 1].mean(axis=0)
    return out

# ────────── raw volume reader ──────────

def _read_raw_volume(fpath: Path) -> np.ndarray:
    """return (N,H,W) uint8/float32 or raise ValueError"""
    if fpath.suffix.lower() == ".fda":
        oct_vol = FDA(str(fpath), printing=False).read_oct_volume()
        bscans = oct_vol.volume if hasattr(oct_vol, "volume") else oct_vol.volume_data
        vol = np.stack([
            np.asarray(getattr(bs, "pixel_array", bs.data)) for bs in bscans
        ], axis=0)
        return vol.astype(np.float32)

    if fpath.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(str(fpath))
        vol = getattr(ds, "pixel_array", None)
        if vol is None or getattr(vol, "ndim", 0) != 3:
            raise ValueError("not multi‑frame DICOM")
        return vol.astype(np.float32)

    raise ValueError("unsupported file type")

# ────────── 強度ボリューム with global preproc ──────────

def load_volume(src_path: Path) -> np.ndarray | None:
    """read volume, apply avg / hist‑match / CLAHE, finally (H,W,N)"""
    try:
        vol_nhw = _read_raw_volume(src_path)  # (N,H,W)
    except ValueError as e:
        warnings.warn(f"{src_path.name}: {e}; skipped")
        return None

    # 1) ±1 枚平均
    vol_nhw = _average_neighbor(vol_nhw, AVG_NEIGHBOR)

    # 2) ヒストグラムマッチ (global)
    if CDF_PATH.exists():
        try:
            ref_cdf = joblib.load(CDF_PATH)  # binary .cdf (pickle)
        except Exception:
            # フォールバック: ASCII テキスト (.csv/.txt)
            ref_cdf = np.loadtxt(CDF_PATH, dtype=np.float32)

        # try/except を抜けたあとで必ず適用
        vol_nhw = _hist_match_volume(vol_nhw, ref_cdf)

    # 3) 3‑D CLAHE
    vol_nhw = _clahe3d(vol_nhw)

    # to (H,W,N)
    vol = vol_nhw.transpose(1, 2, 0)

    # resize width to 992 px
    if vol.shape[1] != TARGET_W:
        vol = np.stack([
            skt.resize(s, (TARGET_H, TARGET_W), preserve_range=True,
                        order=1, anti_aliasing=True)
            for s in vol.transpose(2, 0, 1)
        ], axis=-1)

    return vol.astype(np.float32)

# ────────── TOPCON セグ ──────────

# def read_topcon_surface(fda_path: Path) -> dict[str, np.ndarray]:
#     surf, seg = {}, FDA(str(fda_path), printing=False).read_segmentation()
#     for key, cand in ALIAS.items():
#         n = pick_surface(seg, cand)
#         if n is None:
#             continue
#         arr = np.vstack([np.asarray(s, np.float32) if s is not None else np.full((1, TARGET_H), np.nan) for s in seg[n]]).T
#         if arr.shape[0] != TARGET_W:
#             arr = skt.resize(arr, (TARGET_W, arr.shape[1]), order=1, preserve_range=True, anti_aliasing=True)
#         surf[key] = arr
#     return surf

def read_topcon_surface(fda_path: Path) -> dict[str, np.ndarray]:
    fda_obj = FDA(str(fda_path), printing=False)
    
    # --- 1. 元画像の高さを取得 ---
    oct_data = fda_obj.read_oct_volume()
    raw_vol = oct_data.volume if hasattr(oct_data, "volume") else oct_data.volume_data
    if isinstance(raw_vol, list):
        original_h = raw_vol[0].shape[0]
    else:
        original_h = raw_vol[0].shape[0]
        
    scale_y = TARGET_H / original_h

    # --- 2. セグメンテーション読み込み ---
    surf, seg = {}, fda_obj.read_segmentation()
    
    for key, cand in ALIAS.items():
        n = pick_surface(seg, cand)
        if n is None:
            continue
        
        # 座標データを取得 (W, N)
        # Topconのデータ構造によっては一部欠損があるため、ない場合はNaNで埋める
        arr = np.vstack([np.asarray(s, np.float32) if s is not None else np.full((1, TARGET_H), np.nan) for s in seg[n]]).T
        
        # ★★★ 修正ポイント: 上下反転 ＋ スケール補正 ★★★
        # Topcon座標(下=0) を 画像座標(上=0) に変換してから、512pxスケールに合わせる
        # 式: y_new = (元の高さ - 1 - y_old) * スケール
        arr = (original_h - 1 - arr) * scale_y

        # 横幅のリサイズ (TARGET_W = 992 に合わせる)
        if arr.shape[0] != TARGET_W:
            arr = skt.resize(arr, (TARGET_W, arr.shape[1]), order=1, preserve_range=True, anti_aliasing=True)
            
        surf[key] = arr
        
    return surf

# ────────── AI セグ ──────────

def infer_surfaces_ai(src_path: Path, ckpt: Path, gpu: str) -> dict[str, np.ndarray]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from dataset import FDADataset
    from topology import correct_surface_topology

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net = ck["model"] if isinstance(ck, dict) else ck
    net = net.module if hasattr(net, "module") else net
    net.eval().to(dev)

    # dataset 判定
    if src_path.is_dir():
        ds = FDADataset(str(src_path))
    elif src_path.suffix.lower() == ".fda":
        class SingleFDADataset(torch.utils.data.Dataset):
            def __init__(self, fpath: Path):
                vol = _read_raw_volume(fpath)  # (N,H,W)
                self.vol = vol
            def __len__(self):
                return self.vol.shape[0]
            def __getitem__(self, i):
                img = self.vol[i]
                if img.shape[1] != TARGET_W:
                    img = skt.resize(img, (TARGET_H, TARGET_W), order=1, preserve_range=True, anti_aliasing=True)
                return torch.from_numpy(img).unsqueeze(0).float(), i
        ds = SingleFDADataset(src_path)
    elif src_path.suffix.lower() == ".dcm":
        class SingleDICOMDataset(torch.utils.data.Dataset):
            def __init__(self, fpath: Path):
                vol = _read_raw_volume(fpath)
                self.vol = vol
            def __len__(self):
                return self.vol.shape[0]
            def __getitem__(self, i):
                img = self.vol[i]
                if img.shape[1] != TARGET_W:
                    img = skt.resize(img, (TARGET_H, TARGET_W), order=1, preserve_range=True, anti_aliasing=True)
                return torch.from_numpy(img).unsqueeze(0).float(), i
        ds = SingleDICOMDataset(src_path)
    else:
        raise ValueError("unsupported src")

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    surfs = []
    for sl, _ in loader:
        with torch.no_grad():
            out = net(sl.to(dev))
        if isinstance(out, dict):
            out = out.get("refined_final_surface") or out["final_surfaces"]
        from topology import correct_surface_topology
        surfs.append(correct_surface_topology(out, min_gap=2).cpu().short().numpy())

    vol = np.concatenate(surfs, axis=0).transpose(1, 2, 0)  # (lay,W,N)
    if vol.shape[1] != TARGET_W:
        import scipy.ndimage as ndi
        vol = ndi.zoom(vol, (1, TARGET_W/vol.shape[1], 1), order=1)

    idx = {"ELM": 7, "RPE_BM": 8}
    return {k: vol[i] for k, i in idx.items() if i < vol.shape[0]}

# ────────── スラブ投影 & save ──────────
SLAB_OFFSET = 5

def slab_enface(vol, top, bot, method="mean"):
    H, W, N = vol.shape
    en = np.zeros((W, N), np.float32)
    for z in range(N):
        for x in range(W):
            yt = int(round(top[x, z])); yb = int(round(bot[x, z])) + SLAB_OFFSET
            if math.isnan(yt) or math.isnan(yb) or yt >= yb:
                continue
            slab = vol[yt: yb + 1, x, z]
            en[x, z] = slab.mean() if method == "mean" else slab.max()
    return en


def save_enface(img: np.ndarray, out_path: Path):
    img = skt.resize(img, (EF_SIZE, EF_SIZE), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    plt.figure(figsize=(4, 4)); plt.imshow(img, cmap="gray", origin="upper")
    plt.axis("off"); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(); print("✔", out_path)

# ────────── 処理本体 ──────────

def process_one(src_path: Path, seg_mode: str, ckpt: Path | None, gpu: str):
    vid = src_path.stem
    raw_vol = load_volume(src_path)
    if raw_vol is None:
        return  # skip bad file
    vol = raw_vol  # すでに global preproc 済み

    surf: dict[str, np.ndarray] = {}

    # Topcon セグ (fda only)
    if src_path.suffix.lower() == ".fda" and seg_mode in ("topcon", "auto"):
        try:
            surf.update(read_topcon_surface(src_path))
        except Exception as e:
            warnings.warn(f"{vid}: Topcon seg fail {e}")

    # AI セグ (残欠 or dcm)
    if seg_mode in ("ai", "auto") and (set(SURF_KEYS) - surf.keys()):
        if ckpt is None:
            warnings.warn(f"{vid}: --ckpt required for AI segmentation")
            return
        try:
            surf.update(infer_surfaces_ai(src_path, ckpt, gpu))
        except Exception as e:
            warnings.warn(f"{vid}: AI seg fail {e}")
            return

    if set(SURF_KEYS) - surf.keys():
        print(f"{vid}: ELM/RPE_BM missing → skip"); return

    # --- Safety guards (逆転/形状の取り違い対策) ---
    if np.nanmedian(surf["ELM"] - surf["RPE_BM"]) > 0:
        surf["ELM"], surf["RPE_BM"] = surf["RPE_BM"], surf["ELM"]
    for k in ("ELM", "RPE_BM"):
        if surf[k].shape != (vol.shape[1], vol.shape[2]):
            surf[k] = surf[k].T if surf[k].shape == (vol.shape[2], vol.shape[1]) else surf[k]
    # ------------------------------------------------
    en = slab_enface(vol, surf["ELM"], surf["RPE_BM"])
    en = np.rot90(en, 1)
    save_enface(en, OUT_DIR / f"{vid}_enface_BM.png")

    # === 全層 en‑face (平均投影) =============================
    full = np.rot90(vol.mean(axis=0), 1)   # (N,992) → 90°CCW
    save_enface(full, OUT_DIR / f"{vid}_enface_ALL.png")

# ────────── CLI ──────────

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("src", help=".fda / .dcm file / folder / xlsx")
    p.add_argument("--seg", choices=["topcon", "ai", "auto"], default="auto")
    p.add_argument("--ckpt", help="AI ckpt (.t7)")
    p.add_argument("--gpu", default="0")
    p.add_argument("--root", default=".")
    return p.parse_args()


def main():
    a = parse()
    src_path = Path(a.src)

    # 1. xlsx list
    if src_path.suffix.lower() == ".xlsx":
        ids = pd.read_excel(src_path, header=None, usecols=[0]).iloc[:, 0].dropna().astype(str).str.strip()
        root = Path(a.root)
        for vid in tqdm(ids, desc="xlsx"):
            for ext in (".fda", ".dcm"):
                fp = root / f"{vid}{ext}"
                if fp.exists():
                    process_one(fp, a.seg, Path(a.ckpt) if a.ckpt else None, a.gpu)
                    break
            else:
                print(f"[{vid}] neither .fda nor .dcm found → skip")
        return

    # 2. folder
    if src_path.is_dir():
        files = sorted(list(src_path.glob("*.fda")) + list(src_path.glob("*.dcm")))
        for fp in tqdm(files, desc="folder"):
            process_one(fp, a.seg, Path(a.ckpt) if a.ckpt else None, a.gpu)
        return

    # 3. single file
    process_one(src_path, a.seg, Path(a.ckpt) if a.ckpt else None, a.gpu)

if __name__ == "__main__":
    main()
