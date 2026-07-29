#!/usr/bin/env python
"""
calculation.py – cpRNFLT 12-sector  + Total / UH / LH を 1 行保存
-----------------------------------------------------------------
* 入力: axis-aligned RNFL マップ (512×256, µm), .fda パス, source タグ
* 出力: ./cpRNFLT_12.csv  (UTF-8)
        1 行 = PID, Source, S1…S12, Total, UH, LH   ← 列数 17
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from oct_converter.readers import FDA

# ─────────── 設定 ───────────
CSV_PATH = Path("./cpRNFLT_12.csv")
COLS = ["PID", "Source"] + [f"S{i}" for i in range(1, 13)] + ["Total", "UH", "LH"]

RADIUS_MM = 3.4          # cp circle 3.4 mm
VOXEL_X = 17.6           # µm / A-scan
R_PX = RADIUS_MM * 1_000 / VOXEL_X   # ≈ 193 px

ANGLES = np.deg2rad(np.arange(0, 360 + 30, 30))  # 0,30,…360  (13 本 → 12 区間)

# ─────────── ヘッダー保証 ───────────
def _ensure_header() -> None:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLS)

# ─────────── ディスク中心 (OCT 座標) ───────────
def disc_center_oct(fda_path: Path) -> Tuple[float, float]:
    meta = FDA(str(fda_path), printing=False).read_all_metadata(False)
    x0, y0, w, h = map(float, meta["regist_info"]["bounding_box_in_fundus_pixels"])
    cx = ( (x0 + w/2) - x0 ) / w * 512
    cz = ( (y0 + h/2) - y0 ) / h * 256
    return cx, cz

# ─────────── リングサンプリング ───────────
def ring_sample(rnfl: np.ndarray, cx: float, cz: float,
                n=720) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = cx + R_PX*np.cos(theta); zs = cz - R_PX*np.sin(theta)
    xs = np.clip(xs, 0, 511);     zs = np.clip(zs, 0, 255)
    return map_coordinates(rnfl, np.vstack([xs, zs]), order=1, mode="nearest")

# ─────────── 12 セクター平均 ───────────
def sector_means(ring: np.ndarray) -> list[float]:
    pts = ring.size // 12
    means = [float(np.nanmean(ring[i*pts:(i+1)*pts])) for i in range(12)]
    # NaN → 0 µm
    return [0.0 if np.isnan(m) else m for m in means]

# ─────────── メイン呼び出し関数 ───────────
def append_cprnflt12(rnfl_map: np.ndarray,
                     fda_path: Path,
                     source_tag: str) -> None:
    """
    rnfl_map  : (512,256) axis-aligned RNFL 厚み [µm]
    fda_path  : 当該 .fda ファイル Path
    source_tag: ai / ai_med / topcon / topcon_med
    """
    pid = fda_path.stem
    cx, cz = disc_center_oct(fda_path)
    ring   = ring_sample(rnfl_map, cx, cz)          # 720 points

    s_means = sector_means(ring)                    # S1–S12
    total   = float(np.nanmean(ring))
    uh      = float(np.nanmean(ring[180:540]))      # 上半分 90–270°
    lh      = float(np.nanmean(np.concatenate([ring[:180], ring[540:]])))

    _ensure_header()
    row = [pid, source_tag] + [f"{m:.2f}" for m in s_means] + \
          [f"{total:.2f}", f"{uh:.2f}", f"{lh:.2f}"]

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
