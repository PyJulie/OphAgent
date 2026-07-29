#!/usr/bin/env python3
"""
seg1d2d_aspectfix.py  (based on seg1d2d.py v4.2 — 2025-07-08)

Topcon OCT Three‑Dimensional Retina Analysis Tool
=================================================

This edition preserves the input B‑scan aspect ratio for slice overlay JPEGs.
Heat‑map generation (256×256 / 10×10) remains unchanged (square).

Changes from v4.2 relevant to aspect ratio:
    • save_slice_overlays(): DO NOT resize the B‑scan to the surface width.
      Instead, we resample the surface curves horizontally to the B‑scan
      width and draw on a canvas with the original H×W of the B‑scan.
    • run_topcon(): when preparing B‑scans for overlays, DO NOT force the
      width to TARGET_W. Keep the raw B‑scan width; the overlay function will
      resample the curves accordingly.
"""

from __future__ import annotations

# ───────────── standard lib ─────────────
import argparse, gc, os
from pathlib import Path
from typing import List, Tuple, Optional

# ───────────── third-party ──────────────
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None
from skimage import exposure
import joblib
import cupy as cp

# ───────────── project local ────────────
import medial                       # GPU-EDT helper
import calculation
from calculation import append_cprnflt12

# ───────────── 定義類 ─────────────
TARGET_W = 992            # 目標横幅（px）

# Histogram-matching strength  (0 = 無効, 1 = フルマッチ)
HIST_ALPHA = 0.4          # ← 好みに応じて 0.2–0.6 程度で調整
SURFACE_NAMES = [
    "ILM", "RNFL_GCL", "GCL_IPL", "IPL_INL", "INL_OPL",
    "OPL_ONL", "ONL_ELM", "ELM", "RPE_BM", "CHOROID_OUT",
]

LAYER_DEF: List[Tuple[str, str, str]] = [
    ("RNFL",    "ILM",       "RNFL_GCL"),
    ("GCL",     "RNFL_GCL",  "GCL_IPL"),
    ("IPL",     "GCL_IPL",   "IPL_INL"),
    ("INL",     "IPL_INL",   "INL_OPL"),
    ("OPL",     "INL_OPL",   "OPL_ONL"),
    ("ONL",     "OPL_ONL",   "ONL_ELM"),
    ("PR",      "ELM",       "RPE_BM"),
    ("Choroid", "RPE_BM",    "CHOROID_OUT"),
    ("GCC",     "ILM",       "IPL_INL"),
]

VOXEL_Y      = 2.6                       # µm / pixel
SURFACE_ROOT = Path("./surface_save")
OUT_ROOT     = Path("./data")
LOG_CSV   = OUT_ROOT / "processed_files.csv"   # data/processed_files.csv

LAYER_VMAX = {
    "RNFL": 150,
    "GCL":   90, "IPL": 90, "GCC": 250,
    "INL":   60, "OPL": 60,
    "ONL":  140, "PR":  40,
    "Choroid": 500,
}

# ───────────── helpers ─────────────
def block_mean(img: np.ndarray, ny: int, nx: int) -> np.ndarray:
    H, W = img.shape
    ys = np.linspace(0, H, ny + 1, dtype=int)
    xs = np.linspace(0, W, nx + 1, dtype=int)
    return np.array([[img[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
                      for j in range(nx)] for i in range(ny)])

def save_heat(arr: np.ndarray, title: str, out_path: Path, *, vmax: float = 100):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(arr, cmap="jet", origin="upper", vmin=0, vmax=vmax)
    plt.title(title); plt.colorbar(label="thickness [µm]")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print("✔", out_path)

def _upsample_x2(arr: np.ndarray, axis: int = 2):
    import scipy.ndimage as ndi
    zoom = [1] * arr.ndim; zoom[axis] = 2
    return ndi.zoom(arr, zoom=zoom, order=1)

def load_id_list(xlsx: Path) -> set[str]:
    if not xlsx.exists():
        return set()
    df = pd.read_excel(xlsx, header=None, usecols=[0])
    return set(df.iloc[:, 0].astype(str).str.strip())

def preprocess(img, gamma=1,
                     use_clahe=True, clahe_alpha=0.2,
                     clahe_clip=0.01, tile_factor=6):
    """
    img: uint8 or float32 (0-255 想定)
    戻値: float32, 0-255 スケールのまま
    """
    im = img.astype(np.float32)          # 0-255 のまま

    # ---- ガンマ（0-255 スケールで直接） -----------------------------
    im_gamma = ((im / 255.0) ** gamma) * 255.0

    if not use_clahe or clahe_alpha <= 0:
        return im_gamma

    # ---- CLAHE も 0-1 スケールで実行してから 255 戻し --------------
    from skimage import exposure
    H, W = im.shape
    im01 = im / 255.0
    im_clahe = exposure.equalize_adapthist(
        im01,
        kernel_size=(H // tile_factor, W // tile_factor),
        clip_limit=clahe_clip
    ) * 255.0

    out = (1 - clahe_alpha) * im_gamma + clahe_alpha * im_clahe
    return out.astype(np.float32)

def clahe3d_volume(
        vol: np.ndarray,
        *,                       # 以降はキーワード専用
        alpha: float = 0.2,      # 0 = 元画像そのまま, 1 = 100 % CLAHE
        gamma: float = 1.0,      # 必要ならコントラストを γ 補正
        clip: float  = 0.1,     # CLAHE の clip_limit
        z_tile: int   = 6,
        tile_factor: int = 6     # y_tile = H//tile_factor, x_tile 同様
    ) -> np.ndarray:
    """
    vol : (Z, H, W)  0–255 スケール (uint8/float32)
    戻り : 同 shape  0–255 float32
    """
    # 0–1 へ正規化
    v = vol.astype(np.float32) / 255.0

    # γ 補正
    v_gamma = v ** gamma

    # 3-D CLAHE
    z, h, w = v.shape
    v_clahe = exposure.equalize_adapthist(
        v,
        kernel_size=(z_tile, h // tile_factor, w // tile_factor),
        clip_limit=clip
    )

    # α ブレンド
    v_mix = (1.0 - alpha) * v_gamma + alpha * v_clahe

    # 0–255 へ戻す
    return (v_mix * 255.0).astype(np.float32)

def save_segmentation_npy(vol: np.ndarray, vid: str, *, suffix: str = "") -> Path:
    """
    vol : (n_layer, W, N) など
    vid : 患者 ID / ファイル名 stem
    戻  : 保存した .npy の Path
    """
    out_path = OUT_ROOT / f"{vid}{suffix}_seg.npy"     # data 直下に一直線
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vol.astype(np.int16))    # short のまま保存
    print(f"✔ segmentation → {out_path}")
    return out_path


def append_log_csv(src_path: Path, src_type: str, npy_path: Path,
                   orig_shape: tuple[int, ...], npy_shape: tuple[int, ...]):
    """
    処理結果を 1 行ずつ追記。ヘッダは初回だけ書く。
    src_path  : 入力ファイルの絶対パス
    src_type  : 'DICOM' / 'NPY' / 'FDA'
    npy_path  : save_segmentation_npy() が返した Path
    orig_shape: 読み込み直後の volume.shape
    npy_shape : 保存した vol.shape
    """
    import csv, os
    header = ["source_path", "type", "npy_file",
              "orig_res", "npy_shape"]
    need_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow([
            os.path.abspath(src_path),
            src_type,
            npy_path.name,
            f"{orig_shape[1]}x{orig_shape[0]}" if len(orig_shape) >= 2 else "",  # 例: 992x512
            "x".join(map(str, npy_shape)),        # 例: 11x992x512
        ])

# ───────────── histogram-matching helper ─────────────
def hist_match_volume(vol: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """
    vol      : (Z, H, W) 0-255 スケール (uint8/float32/float64 可)
    ref_cdf  : Triton など参照 CDF, shape=(256,)
    返り値   : float32, 同 shape
    """
    # ---- 参照 CDF → 逆関数 (u∈[0,1] → intensity) -----------
    ref_cdf = ref_cdf.astype(np.float32)
    ref_cdf /= ref_cdf[-1]
    ref_inv = np.interp(np.linspace(0, 1, 256), ref_cdf, np.arange(256))

    # ---- 入力側 LUT 構築 (256 要素) -------------------------
    vol_uint8 = np.clip(vol, 0, 255).astype(np.uint8)
    src_hist  = np.bincount(vol_uint8.ravel(), minlength=256).astype(np.float32)
    src_cdf   = np.cumsum(src_hist); src_cdf /= src_cdf[-1]
    lut       = np.interp(src_cdf, np.linspace(0, 1, 256), ref_inv).astype(np.float32)

    # ---- 出力バッファ (float32) ----------------------------
    out = vol.astype(np.float32, copy=True)

    # ---- スライス毎に LUT 適用して α ブレンド --------------
    for z in range(out.shape[0]):            # Z 枚繰り返し
        mapped = lut[vol_uint8[z]]           # ここだけ 2 MB 程度
        out[z] = (1.0 - HIST_ALPHA) * out[z] + HIST_ALPHA * mapped

    return out

# ───────── slice overlay ──────────
def save_slice_overlays(
    surfs : np.ndarray,                       # (n_layer, W_surf, N)
    bscans: List["torch.Tensor"] | List[np.ndarray],
    vid   : str,
    *,
    out_subdir: str = "slice",
    line_thickness: int = 1,
    jpg_quality: int = 95,
    flip_y: bool = False
):
    """
    B-scan に層境界を重ねて JPEG 出力する。
    変更点（重要）:
      • B-scan の H×W をそのまま採用（入力縦横比を保持）。
      • surf 曲線（横方向 W_surf）を B-scan の横幅 W_img へ線形補間してから描画。
    - out_subdir: 出力先サブフォルダ名（例: slice_ai / slice_topcon）
    - flip_y: True の場合は y 座標を上下反転（Topcon 表記: 下端=0 → 画像座標: 上端=0）
    """
    import cv2
    import numpy as _np

    out_dir = OUT_ROOT / vid / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layer, W_surf, N = surfs.shape

    colors_bgr = [
        (255, 128, 0),   # 青系
        (0, 165, 255),   # 橙
        (0, 0, 255),     # 赤
        (180, 105, 255), # 桃
        (211, 0, 148),   # 紫
        (0, 255, 255),   # 黄
        (255, 255, 0),   # 水
        (0, 255, 0),     # 緑
        (255, 0, 0),     # 青
        (128, 128, 128), # 灰
    ]

    # 予め x 軸の補間元を準備
    x_src = _np.arange(W_surf, dtype=_np.float32)

    for z in range(min(N, len(bscans))):
        img = bscans[z]

        # ---- Tensor → numpy & 値域調整（0-255）----
        _is_tensor = False
        try:
            import torch as _torch
            _is_tensor = isinstance(img, _torch.Tensor)
        except Exception:
            _is_tensor = False
        if _is_tensor:
            img = img.squeeze().detach().cpu().numpy()

        img = img.astype(_np.float32)
        if img.max() <= 1.0:
            img *= 255.0

        # ---- 入力 B-scan のサイズをそのまま使用 ----
        H, W_img = img.shape
        H = int(H); W_img = int(W_img)

        # ---- グレースケール → 3ch (BGR) へ変換 ----
        canvas = img.clip(0, 255).astype(_np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # ---- surf 曲線を W_img に補間して重ね描き ----
        x_tgt = _np.linspace(0, W_surf - 1, W_img, dtype=_np.float32)

        for l in range(n_layer):
            y_src = surfs[l, :, z].astype(_np.float32)
            # 線形補間して W_img 長に
            y = _np.interp(x_tgt, x_src, y_src)
            if flip_y:
                y = (H - 1) - y
            y = _np.clip(y, 0, H - 1)

            pts = _np.stack([_np.arange(W_img, dtype=_np.int32), y.astype(_np.int32)], axis=1)
            pts = pts.reshape((-1, 1, 2))
            color = colors_bgr[l % len(colors_bgr)]
            import cv2 as _cv2
            _cv2.polylines(canvas, [pts], isClosed=False, color=color,
                           thickness=line_thickness, lineType=_cv2.LINE_AA)

        # ---- JPEG 保存 ----
        out_path = out_dir / f"{z+1:03d}.jpg"
        import cv2 as _cv2
        _cv2.imencode(".jpg", canvas, [int(_cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])[1].tofile(str(out_path))

    print(f"✔ slice overlays → {out_dir}")

# ───────── map creation ──────────
def create_maps(
    vol: np.ndarray,         # (n_layer,512,256)
    vid: str,
    *, suffix: str, with10: bool, use_medial: bool,
    cpr_flag: bool = False, fda_path: Path | None = None,
):
    if vol.shape[2] == 128:
        vol = _upsample_x2(vol, axis=2)
        print(f"[{vid}] upsampled 512×128 → 512×256")
    # --- shape チェック（横幅 992 が必須）---
    if vol.shape[1:] != (TARGET_W, 256):
        print(f"[{vid}] skipped (unsupported shape {vol.shape[1:]})")
        return

    idx = {n: i for i, n in enumerate(SURFACE_NAMES) if i < vol.shape[0]}
    out_dir = OUT_ROOT / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, top, bot in LAYER_DEF:
        if top not in idx or bot not in idx:
            continue
        if use_medial:
            cp._default_memory_pool.free_all_blocks()
            vol_med = vol
            if vol_med.shape[1] != 512:            # 幅が 992→512 などの場合
                import scipy.ndimage as ndi
                zoom = (1, 512 / vol_med.shape[1], 1)   # (layer, H, W)
                vol_med = ndi.zoom(vol_med, zoom=zoom, order=1)
            mat = medial.medial_thickness_layer(vol_med, idx[top], idx[bot])
            mat = np.flipud(np.rot90(mat, 1))
            if suffix.endswith("_topcon"):
                mat = -mat
        else:
            mat = (vol[idx[bot]] - vol[idx[top]]).T * VOXEL_Y
            if suffix.endswith("_topcon") or (mat < 0).mean() > 0.5:
                mat = -mat
            mat = np.clip(mat, 0, None)

        tag  = "_med" if use_medial else ""
        base = f"{label}{suffix}{tag}"
        vmax = LAYER_VMAX.get(label, 100)

        save_heat(block_mean(mat, 256, 256),
                  f"{label} 256×256 [{vid}]{suffix}{tag}",
                  out_dir / f"{base}_256x256.png",
                  vmax=vmax)

        if with10:
            save_heat(block_mean(mat, 10, 10),
                      f"{label} 10×10 [{vid}]{suffix}{tag}",
                      out_dir / f"{base}_10x10.png",
                      vmax=vmax)

        if label == "RNFL" and cpr_flag and fda_path is not None:
            src_tag = f"{suffix.lstrip('_')}{tag}"
            append_cprnflt12(mat.T, fda_path, src_tag)

# ───────── surface-only helper ─────────
def load_surface_volume(pid: str) -> Optional[np.ndarray]:
    files = sorted((SURFACE_ROOT / pid).glob(f"{pid}_*.npy"))
    return None if not files else np.stack([np.load(f) for f in files], axis=-1)

# ─────────── AI pipeline ───────────
def run_ai(
    files: List[Path], ckpt: Path,
    *, gpu: str, batch: int, with10: bool, topology: bool,
    min_gap: Optional[int], medial: bool, cpr_flag: bool,
    overlay: bool, avg: int, no_preproc: bool,
    clahe3d: bool, hist_match: Optional[Path],
    rotate_npy: Optional[str] = None,
    allowed_ids: set[str] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from dataset  import FDADataset
    from topology import correct_surface_topology
    from tqdm import tqdm

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_data = torch.load(ckpt, map_location=dev, weights_only=False)
    net = ckpt_data["model"] if isinstance(ckpt_data, dict) else ckpt_data
    net = net.module if hasattr(net, "module") else net
    net.eval().to(dev)

    for fp in tqdm(files, desc="patients"):
        if not fp.exists():
            continue
        vid = fp.stem
        if allowed_ids and vid not in allowed_ids:
            continue
        if (OUT_ROOT / vid / "RNFL_ai_256x256.png").exists() and not overlay:
            continue

        # ---------- 入力種別ごとに Dataset 準備 ----------
        if fp.suffix.lower() == ".fda":
            from oct_converter.readers import FDA as _FDAReader
            import numpy as _np
            import skimage.transform as skt
            try:
                _vol = _FDAReader(str(fp), printing=False).read_oct_volume().volume
            except Exception as e:
                print(f"[{vid}] skipped (FDA read error: {e})")
                continue
            if isinstance(_vol, list):
                vol_np = _np.stack([_np.asarray(v) for v in _vol], axis=0)
            elif not isinstance(_vol, _np.ndarray):
                vol_np = _np.asarray(_vol)
            else:
                vol_np = _vol
            vol_np = vol_np.astype(_np.float32)
            if vol_np.shape[1:] != (TARGET_W, 512):
                print(f"[{vid}] resizing {vol_np.shape[1:]} → (512, {TARGET_W})")
                vol_np = _np.stack([
                    skt.resize(s, (512, TARGET_W), preserve_range=True, order=1, anti_aliasing=True).astype(_np.float32)
                    for s in vol_np
                ])
            if hist_match:
                ref_cdf = joblib.load(hist_match)
                print(f"[{vid}] histogram-match → Triton CDF")
                vol_np = hist_match_volume(vol_np, ref_cdf)
            if clahe3d:
                print(f"[{vid}] apply 3-D CLAHE")
                vol_np = clahe3d_volume(vol_np)
            import torch as _torch
            class FDASliceDataset(_torch.utils.data.Dataset):
                def __init__(self, vol, avg, no_pp):
                    self.vol, self.avg, self.no_pp = vol, avg, no_pp
                def __len__(self):
                    return self.vol.shape[0]
                def __getitem__(self, i):
                    a = self.avg
                    if a > 0:
                        lo = max(0, i - a); hi = min(len(self.vol), i + a + 1)
                        img = self.vol[lo:hi].mean(axis=0)
                    else:
                        img = self.vol[i]
                    if not self.no_pp:
                        img = preprocess(img)
                    return _torch.from_numpy(img).unsqueeze(0).float(), i
            ds = FDASliceDataset(vol_np, avg, no_preproc)
        elif fp.suffix.lower() == ".npy":
            import skimage.transform as skt
            vol_np = np.load(fp)

            # rotate option は残す
            if rotate_npy == "cw":
                vol_np = np.rot90(vol_np, k=-1, axes=(1, 2))   # 90°右回転
            elif rotate_npy == "ccw":
                vol_np = np.rot90(vol_np, k=1, axes=(1, 2))    # 90°左回転

            if vol_np.ndim != 3:
                print(f"[{vid}] skipped (not 3-D .npy)")
                continue

            def _prepare_npy_slice(s: np.ndarray) -> np.ndarray:
                s = s.astype(np.float32)

                # 885,512 は H,W としてそのまま扱う
                # 高さだけ 885 -> 992 に変更し、幅 512 は維持
                return skt.resize(
                    s,
                    (TARGET_W, 512),   # -> (992, 512)
                    preserve_range=True,
                    order=1,
                    anti_aliasing=True
                ).astype(np.float32)

            if vol_np.shape[1:] != (TARGET_W, 512):
                print(f"[{vid}] resizing {vol_np.shape[1:]} → ({TARGET_W}, 512) [keep H,W]")
                vol_np = np.stack([
                    _prepare_npy_slice(s)
                    for s in vol_np
                ])
            if hist_match:
                ref_cdf = joblib.load(hist_match)
                print(f"[{vid}] histogram-match → Triton CDF")
                vol_np = hist_match_volume(vol_np, ref_cdf)
            if clahe3d:
                print(f"[{vid}] apply 3-D CLAHE")
                vol_np = clahe3d_volume(vol_np)
            class NPYSliceDataset(torch.utils.data.Dataset):
                def __init__(self, vol, avg, no_pp):
                    self.vol, self.avg, self.no_pp = vol, avg, no_pp
                def __len__(self):
                    return self.vol.shape[0]
                def __getitem__(self, i):
                    a = self.avg
                    if a > 0:
                        lo = max(0, i - a); hi = min(len(self.vol), i + a + 1)
                        img = self.vol[lo:hi].mean(axis=0)
                    else:
                        img = self.vol[i]
                    if not self.no_pp:
                        img = preprocess(img)
                    return torch.from_numpy(img).unsqueeze(0).float(), i
            ds = NPYSliceDataset(vol_np, avg, no_preproc)
        elif fp.suffix.lower() == ".dcm":
            import pydicom, skimage.transform as skt
            ds_dcm = pydicom.dcmread(str(fp))
            vol_np = ds_dcm.pixel_array
            if vol_np.ndim != 3:
                print(f"[{vid}] skipped (not 3-D DICOM)"); continue
            if vol_np.shape[1:] != (TARGET_W, 512):
                print(f"[{vid}] resizing {vol_np.shape[1:]} → (512, {TARGET_W})")
                vol_np = np.stack([
                    skt.resize(s, (512, TARGET_W), preserve_range=True, order=1, anti_aliasing=True).astype(np.float32)
                    for s in vol_np
                ])
            if hist_match:
                ref_cdf = joblib.load(hist_match)
                print(f"[{vid}] histogram-match → Triton CDF")
                vol_np = hist_match_volume(vol_np, ref_cdf)
            if clahe3d:
                print(f"[{vid}] apply 3-D CLAHE")
                vol_np = clahe3d_volume(vol_np)
            class DICOMSliceDataset(torch.utils.data.Dataset):
                def __init__(self, vol, avg, no_pp):
                    self.vol, self.avg, self.no_pp = vol, avg, no_pp
                def __len__(self):
                    return self.vol.shape[0]
                def __getitem__(self, i):
                    a = self.avg
                    if a > 0:
                        lo = max(0, i - a); hi = min(len(self.vol), i + a + 1)
                        img = self.vol[lo:hi].mean(axis=0)
                    else:
                        img = self.vol[i]
                    if not self.no_pp:
                        img = preprocess(img)
                    return torch.from_numpy(img).unsqueeze(0).float(), i
            ds = DICOMSliceDataset(vol_np, avg, no_preproc)
        else:
            print(f"[{vid}] skipped (unsupported ext {fp.suffix})"); continue

        total_slices = len(ds)
        from tqdm import tqdm as _tqdm
        inner = _tqdm(total=total_slices,
                      desc=f"[{vid}] segmentation",
                      bar_format="segmentation {n_fmt}/{total_fmt} → {postfix}",
                      position=1, leave=True, dynamic_ncols=True)
        inner.set_postfix_str(str(OUT_ROOT / f"{vid}_ai_seg.npy"))

        slices, surfs = [], []
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        for sl, idx in loader:
            sl = sl.to(dev)
            with torch.no_grad():
                out = net(sl)
            if isinstance(out, dict):
                out = out.get("refined_final_surface") or out.get("final_surfaces")
            if topology:
                out = correct_surface_topology(out, min_gap=min_gap)
            surfs.append(out.cpu().short().numpy())
            for s in sl.cpu():
                slices.append(s)
            inner.update(sl.size(0))
        inner.close()

        vol = np.concatenate(surfs, axis=0).transpose(1, 2, 0)  # (n_layer, W, N)
        if vol.shape[1] != TARGET_W:
            import scipy.ndimage as ndi
            zoom = (1, TARGET_W / vol.shape[1], 1)
            vol  = ndi.zoom(vol, zoom=zoom, order=1)
        fda_path = fp if fp.suffix.lower() == ".fda" else None

        # === ここで .npy 保存 & CSV 追記（1行／ファイル） ===
        npy_path = save_segmentation_npy(vol, vid, suffix="_ai")
        append_log_csv(
            src_path=fp,
            src_type=fp.suffix.upper().lstrip(".") or "UNKNOWN",
            npy_path=npy_path,
            orig_shape=(512, TARGET_W, total_slices),
            npy_shape=vol.shape,
        )

        try:
            create_maps(vol, vid, suffix="_ai",
                        with10=with10, use_medial=medial,
                        cpr_flag=cpr_flag, fda_path=fda_path)
            if overlay:
                save_slice_overlays(vol, slices, vid, out_subdir="slice_ai", flip_y=False)
        except (IndexError, ValueError) as e:
            print(f"[{vid}] skipped (shape/medial error: {e})"); continue
        gc.collect()

def run_topcon(
    files: List[Path], *, allowed_ids:set[str], with10:bool,
    medial:bool, cpr_flag:bool, overlay:bool,
):
    """
    Topcon pipeline that accepts either a list of FDA files or files collected from a directory.
    Robust to missing seg["BSCAN"] by falling back to FDA.read_oct_volume().
    """
    from oct_converter.readers import FDA
    import skimage.transform as skt

    for fda_path in tqdm(list(files), desc="patients"):
        vid = fda_path.stem
        if allowed_ids and vid not in allowed_ids:
            continue
        if (OUT_ROOT/vid/"RNFL_topcon_256x256.png").exists() and not overlay:
            continue

        # --- read Topcon segmentation (surfaces) ---
        try:
            seg = FDA(str(fda_path), printing=False).read_segmentation()
        except Exception as e:
            print(f"[{vid}] skipped ({e})")
            continue

        # --- build surface volume (n_layer, W, N) ---
        stack = []
        for n in SURFACE_NAMES:
            if n not in seg:
                continue
            arr = np.vstack([
                np.asarray(s, np.float32) if s is not None else np.full((1, 512), np.nan, dtype=np.float32)
                for s in seg[n]
            ]).T  # (W, N)
            stack.append(arr)
        if len(stack) < 2:
            print(f"[{vid}] skipped (insufficient surfaces)")
            continue
        vol = np.stack(stack, axis=0)  # (n_layer, W, N)

        # --- ensure width = TARGET_W for downstream maps ---
        if vol.shape[1] != TARGET_W:
            import scipy.ndimage as ndi
            zoom = (1, TARGET_W / vol.shape[1], 1)
            vol  = ndi.zoom(vol, zoom=zoom, order=1)

        # --- obtain B-scans for logging/overlay ---
        bscan_hwz = None  # (H, W, N)
        if isinstance(seg, dict) and "BSCAN" in seg and hasattr(seg["BSCAN"], "shape"):
            bscan_hwz = seg["BSCAN"]
        else:
            # Fallback: read the raw OCT volume (N, H, W) -> transpose to (H, W, N)
            try:
                oct_vol = FDA(str(fda_path), printing=False).read_oct_volume().volume
                if isinstance(oct_vol, list):
                    oct_vol = np.stack([np.asarray(v) for v in oct_vol], axis=0)
                elif not isinstance(oct_vol, np.ndarray):
                    oct_vol = np.asarray(oct_vol)
                if oct_vol.ndim == 3:
                    bscan_hwz = np.transpose(oct_vol.astype(np.float32), (1, 2, 0))
            except Exception:
                bscan_hwz = None

        # --- save .npy & append CSV log ---
        npy_path = save_segmentation_npy(vol, vid, suffix="_topcon")
        orig_shape = (512, TARGET_W, vol.shape[2])
        append_log_csv(
            src_path=fda_path,
            src_type="FDA",
            npy_path=npy_path,
            orig_shape=orig_shape,     # (H, W, N)
            npy_shape=vol.shape,       # (n_layer, W, N)
        )

        # --- generate maps ---
        try:
            create_maps(
                vol, vid, suffix="_topcon",
                with10=with10, use_medial=medial,
                cpr_flag=cpr_flag, fda_path=fda_path
            )
        except (IndexError, ValueError) as e:
            print(f"[{vid}] skipped (shape/medial error: {e})")
            continue

        # --- slice overlays ---
        if overlay:
            if bscan_hwz is None:
                print(f"[{vid}] overlay skipped (BSCAN not present; OCT read failed)")
            else:
                bscans = []
                for z in range(bscan_hwz.shape[2]):
                    # 重要: 元の B-scan の横幅を保持（TARGET_W へはリサイズしない）
                    img = bscan_hwz[..., z].astype(np.float32)
                    bscans.append(img)
                save_slice_overlays(vol, bscans, vid, out_subdir="slice_topcon", flip_y=True)

def parse()->argparse.Namespace:
    p=argparse.ArgumentParser(description="Generate layer-thickness maps")
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pid")
    src.add_argument("--src", help="file or folder")
    # 入力フォーマットフラグ
    fm=p.add_mutually_exclusive_group()
    fm.add_argument("--fda", action="store_true")
    fm.add_argument("--npy", action="store_true")
    fm.add_argument("--dcm", action="store_true")
    p.add_argument("--ucl", action="store_true", help=argparse.SUPPRESS)  # 旧互換
    # 処理種別
    p.add_argument("--ai", action="store_true")
    p.add_argument("--topcon", action="store_true")
    p.add_argument("--ckpt")
    p.add_argument("--gpu", default="0"); p.add_argument("--batch", type=int, default=1)
    p.add_argument("--topology", action="store_true"); p.add_argument("--min_gap", type=int)
    p.add_argument("--with10", action="store_true")
    p.add_argument("--medial_ai", action="store_true")
    p.add_argument("--medial_topcon", action="store_true")
    p.add_argument("--cprnflt12", action="store_true")
    p.add_argument("--overlay_slice", action="store_true")
    p.add_argument("--list_xlsx", default="Disc_list.xlsx")
    p.add_argument("--avg", type=int, default=0,
               help="各 B-scan を前後 ±AVG 枚で平均 (0 で無効)")
    p.add_argument("--no_preproc", action="store_true",
                help="preprocess() を無効化")
    p.add_argument("--clahe3d", action="store_true",
               help="3-D CLAHE を volume 全体に一度だけ適用する")
    p.add_argument("--hist_match", nargs="?", const="Triton_hist.cdf",
                help="CDF ファイルを指定してヒストグラムマッチングを適用")
    p.add_argument("--rotate_npy", choices=["cw","ccw"], help="rotate npy input 90deg before processing")
    return p.parse_args()

def main()->None:
    a=parse()
    if not a.ai and not a.topcon: a.ai=True           # default
    id_set=load_id_list(Path(a.list_xlsx))
    # surface-only mode
    if a.pid:
        vol=load_surface_volume(a.pid)
        if vol is not None:
            create_maps(vol, a.pid, suffix="_ai",
                        with10=a.with10, use_medial=a.medial_ai)
        return
    # input path
    src_path=Path(a.src);   assert src_path.exists()
    # フォルダ入力時の拡張子決定
    if src_path.is_dir():
        if a.npy or a.ucl:      ext="*.npy"
        elif a.dcm:             ext="*.dcm"
        else:                   ext="*.fda"
        in_files=sorted(src_path.glob(ext))
    else:
        in_files=[src_path]

    # ---------- AI ----------
    if a.ai:
        if not a.ckpt:
            raise ValueError("--ckpt required for AI segmentation")

        run_ai(
            in_files, Path(a.ckpt),
            allowed_ids=id_set, gpu=a.gpu, batch=a.batch,
            with10=a.with10, topology=a.topology, min_gap=a.min_gap,
            medial=a.medial_ai, cpr_flag=a.cprnflt12, overlay=a.overlay_slice,
            avg=a.avg, no_preproc=a.no_preproc,
            clahe3d=a.clahe3d,
            hist_match=Path(a.hist_match) if a.hist_match else None,
            rotate_npy=a.rotate_npy,
        )

    # ---------- Topcon ----------
    if a.topcon:
        # Accept both a folder of .fda and a single .fda file
        if src_path.is_dir():
            fda_files = sorted(src_path.glob("*.fda"))
        else:
            fda_files = [src_path] if src_path.suffix.lower() == ".fda" else []
        if not fda_files:
            print("[Topcon] .fda ファイルが見つかりません（--src は .fda 単体または .fda を含むフォルダ）")
        else:
            run_topcon(fda_files, allowed_ids=id_set,
                       with10=a.with10, medial=a.medial_topcon,
                       cpr_flag=a.cprnflt12, overlay=a.overlay_slice)

if __name__ == "__main__":
    main()
