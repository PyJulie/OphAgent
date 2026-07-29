#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oct_all_in_one.py  —  厚み/表面 .npy → en‑face → 乳頭中心 → 環状解析＋cpRNFLT／Macula／Disc2（ワンストップ）

更新 (2025-10-23):
  • `--target disc2` を復活（2.8–3.46mm / 3.46–4.4mm の二重リングで面積平均を算出）
      - 全体・上下・TSNI 四象限・12時計エリアを **内側(2.8–3.46)** と **外側(3.46–4.4)** それぞれで計算。
      - 図は 3 本円 + 45°/135°/… 境界、12 分割ガイド、四象限ラベルに「内/外」の 2 値を表示。
      - CSV は disc2_summary.csv として、S1..S12 などは *_in / *_out の 2 列で出力。
  • それ以外の `disc` / `macula` の処理は **変更なし**（Eye/Source を CSV に含める等）。
"""

from __future__ import annotations
import os, sys, glob, argparse, subprocess, warnings, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ---- 既存モジュール（処理はそのまま使用） -----------------------------------
import make_enface_bm as ef                               # en‑face 生成（直呼び）
from annulus_analysis import analyse_annulus              # 図＆統計（disc 用。既存関数そのまま）

# Laterality: readers
try:
    from oct_converter.readers import FDA
except Exception:
    FDA = None
try:
    import pydicom
except Exception:
    pydicom = None

# calculation.py と同じ定数（リング半径）
CSV_COLS = ["PID", "Source", "Eye"] + [f"S{i}" for i in range(1, 13)] + ["Total", "UH", "LH"]
RADIUS_MM = 3.4
VOXEL_X   = 17.6
R_PX      = RADIUS_MM * 1_000 / VOXEL_X   # ≈193 px

# ---- batch_annulus.py の SURFS / LAYERS / VMAP を忠実トレース ----------------
SURFS = ["ILM","RNFL_GCL","GCL_IPL","IPL_INL","INL_OPL",
         "OPL_ONL","ONL_ELM","ELM","RPE_BM","CHOROID_OUT"]

LAYERS_FROM_SURFS = {
    "RNFL":   ("ILM","RNFL_GCL"),
    "GCL":    ("RNFL_GCL","GCL_IPL"),
    "IPL":    ("GCL_IPL","IPL_INL"),
    "INL":    ("IPL_INL","INL_OPL"),
    "OPL":    ("INL_OPL","OPL_ONL"),
    "ONL":    ("OPL_ONL","ONL_ELM"),
    "PR":     ("ELM","RPE_BM"),
    "Choroid":("RPE_BM","CHOROID_OUT"),
    "GCC":    ("ILM","IPL_INL"),
    "Retina": ("ILM","RPE_BM")
}

VMAP = {
    "RNFL":    (0, 180),
    "GCL":     (0,  50),
    "IPL":     (0,  50),
    "INL":     (0,  50),
    "OPL":     (0, 150),
    "ONL":     (0,  50),
    "PR":      (0,  70),
    "Choroid": (0, 500),
    "GCC":     (0, 230),
    "Retina":  (0, 500),
}
VMAP_6_ONLY = {
    "GCL+": (0, 150),
    "GCC":  (0, 230),
}

# ---- ユーティリティ ---------------------------------------------------------
def id_from_path(p: Path) -> str:
    return p.stem.split("_")[0]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_npys(thick_dir: Path, pattern: str) -> List[Path]:
    return sorted(thick_dir.rglob(pattern) if ("*" in pattern or "?" in pattern) else thick_dir.glob(pattern))

def guess_npy_kind(path: Path) -> str:
    stem = path.stem.lower()
    return "seg" if ("_seg" in stem or stem.endswith("seg")) else "thick"

def is_ai_seg(path: Path) -> bool:
    s = path.stem.lower()
    return ("ai_seg" in s) or ("_ai_" in s)

def source_tag_for(path: Path, kind: str) -> str:
    if kind == "seg":
        return "ai" if is_ai_seg(path) else "topcon"
    return "thick"

# ---- laterality helpers -----------------------------------------------------
def _iter_kv(d: Any, prefix: str = ""):
    """ネストした dict/list から (keypath, value) を列挙"""
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            yield full, v
            yield from _iter_kv(v, full)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            full = f"{prefix}[{i}]"
            yield full, v
            yield from _iter_kv(v, full)

_KEY_PATS = [
    re.compile(r"(?:^|\.)laterality$"),
    re.compile(r"(?:^|\.)image[_\-]?laterality$"),
    re.compile(r"(?:^|\.)eye$"),
    re.compile(r"(?:^|\.)od[_\-]?os$"),
    re.compile(r"(?:^|\.)od[_\-]?os[_\-]?flag$"),
    re.compile(r"(?:^|\.)exam[_\-]?side$"),
    re.compile(r"(?:^|\.)eye[_\-]?side$"),
    re.compile(r"(?:^|\.)patient[_\-]?eye$"),
    re.compile(r"(?:^|\.)rl$"),
]
def _find_first_by_key_candidates(meta: Dict[str, Any], key_regexes=_KEY_PATS) -> Optional[Tuple[str, Any]]:
    for keypath, value in _iter_kv(meta):
        low = keypath.lower()
        for pat in key_regexes:
            if pat.search(low):
                return keypath, value
    return None

def _normalize_laterality(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    s = str(val).strip().lower()
    if s in {"od", "o.d.", "o d"}:
        return "OD"
    if s in {"os", "o.s.", "o s"}:
        return "OS"
    if s in {"r", "right", "rt", "右", "od/右", "right eye", "r-eye"}:
        return "OD"
    if s in {"l", "left", "lt", "左", "os/左", "left eye", "l-eye"}:
        return "OS"
    if s.startswith("r-") or s.endswith("-r"):
        return "OD"
    if s.startswith("l-") or s.endswith("-l"):
        return "OS"
    if "right" in s:
        return "OD"
    if "left" in s:
        return "OS"
    if s.endswith(" od") or s.endswith(" os"):
        return s[-2:].upper().strip()
    return None

def determine_od_os_from_fda(fda_path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    dbg = {"hit_key": None, "hit_value": None, "candidates_preview": [], "method": None}
    if FDA is None:
        return None, dbg
    try:
        meta = FDA(str(fda_path), printing=False).read_all_metadata(False)
    except Exception:
        return None, dbg

    # 1) Primary: capture_info_02.eye  (0→OD, 1→OS)
    try:
        ci2 = meta.get("capture_info_02", None)
        if ci2 is not None and ("eye" in ci2):
            val = ci2["eye"]
            if isinstance(val, (list, tuple)) and val:
                val = val[0]
            iv = None
            try:
                iv = int(val)
            except Exception:
                iv = None
            if iv in (0, 1):
                lat = "OD" if iv == 0 else "OS"
                dbg["hit_key"] = "capture_info_02.eye"
                dbg["hit_value"] = val
                dbg["method"] = "capture_info_02.eye"
                return lat, dbg
    except Exception:
        pass

    # 2) Generic key search (broader patterns)
    hit = _find_first_by_key_candidates(meta)
    if hit:
        keypath, value = hit
        lat = _normalize_laterality(value)
        dbg["hit_key"] = keypath; dbg["hit_value"] = value; dbg["method"] = "generic"
        if lat in {"OD","OS"}:
            return lat, dbg

    # 3) Preview candidates for debugging
    previews = []
    for keypath, value in _iter_kv(meta):
        if any(k in keypath.lower() for k in ["eye","lat","od","os","side"]):
            s = str(value)
            if len(s) <= 80:
                previews.append((keypath, s))
    dbg["candidates_preview"] = previews[:50]
    return None, dbg

def determine_od_os_from_dicom(dcm_path: Path) -> Optional[str]:
    if pydicom is None:
        return None
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    for tag in [(0x0020,0x0060), (0x0020,0x0062)]:  # Laterality, ImageLaterality
        if tag in ds:
            lat = _normalize_laterality(str(ds[tag].value))
            if lat in {"OD","OS"}:
                return lat
    for elem in ds:
        name = elem.keyword or str(elem.tag)
        if re.search(r"laterality|eye|side", name.lower()):
            lat = _normalize_laterality(str(elem.value))
            if lat in {"OD","OS"}:
                return lat
    return None

def load_eye_csv(csv_path: Path) -> Dict[str, str]:
    m: Dict[str,str] = {}
    df = pd.read_csv(csv_path, header=None)
    for _, row in df.iterrows():
        try:
            cid = str(row.iloc[0]).split("_")[0]
            lat = _normalize_laterality(row.iloc[1])
            if lat in {"OD","OS"}:
                m[cid] = lat
        except Exception:
            continue
    return m

def build_id2eye(ids: List[str], oct_root: Optional[Path], eye_csv: Optional[Path]) -> Dict[str,str]:
    id2eye: Dict[str,str] = {}
    csvmap = load_eye_csv(eye_csv) if eye_csv else {}
    for vid in ids:
        lat = None
        src_hint = "default"
        if vid in csvmap:
            lat = csvmap[vid]; src_hint = "csv"
        elif oct_root is not None:
            fda = oct_root / f"{vid}.fda"
            dcm = oct_root / f"{vid}.dcm"
            if fda.exists():
                lat, dbg = determine_od_os_from_fda(fda); src_hint = f"fda:{dbg.get('method','?')}"
            if (lat is None) and dcm.exists():
                lat = determine_od_os_from_dicom(dcm); src_hint = "dcm"
        if lat not in {"OD","OS"}:
            lat = "OD"; src_hint = "fallback"
        id2eye[vid] = lat
        print(f"[EYE] {vid}: {lat}  ({src_hint})")
    return id2eye

# ---- 512×512（annulus 用） & RNFL (512×256) --------------------------------
def resize_512x512(map_zx: np.ndarray) -> np.ndarray:
    zoom = (512 / map_zx.shape[0], 512 / map_zx.shape[1])
    return ndi.zoom(map_zx, zoom, order=1)

def make_thick512_from_surfs(vol_surfs: np.ndarray, top_idx: int, bot_idx: int, voxel_y: float) -> np.ndarray:
    t_zx = (vol_surfs[bot_idx] - vol_surfs[top_idx]).T * voxel_y
    return resize_512x512(t_zx)

def thick_to_512x512_from_wz(layer_wz: np.ndarray, scale_um_per_unit: float) -> np.ndarray:
    t_zx = layer_wz.T * scale_um_per_unit
    return resize_512x512(t_zx)

def rnfl_wz_to_xz512x256(rnfl_wz: np.ndarray, scale_um_per_unit: float) -> np.ndarray:
    rnfl_wz = rnfl_wz * scale_um_per_unit
    zoom = (512 / rnfl_wz.shape[0], 256 / rnfl_wz.shape[1])
    rnfl_xz = ndi.zoom(rnfl_wz, zoom, order=1)
    return rnfl_xz

def rnfl_from_surfs_to_xz(vol_surfs: np.ndarray, voxel_y: float) -> np.ndarray:
    ilm_idx = SURFS.index("ILM")
    rg_idx  = SURFS.index("RNFL_GCL")
    rnfl_wz = (vol_surfs[rg_idx] - vol_surfs[ilm_idx]) * voxel_y
    return rnfl_wz_to_xz512x256(rnfl_wz, scale_um_per_unit=1.0)

# ---- Topcon 6面：ラベリングの頑健化 ----------------------------------------
def _order_six_by_depth(vol_surfs: np.ndarray):
    med = [float(np.nanmedian(vol_surfs[i])) for i in range(6)]
    asc = np.argsort(med).tolist()
    dsc = asc[::-1]

    def score(order):
        idx = {
            "ILM": order[0],
            "RNFL_GCL": order[1],
            "GCL_IPL": order[2],
            "IPL_INL": order[3],
            "RPE_BM": order[4],
            "CHOROID_OUT": order[5],
        }
        rnfl_med   = float(np.nanmedian(vol_surfs[idx["RNFL_GCL"]] - vol_surfs[idx["ILM"]]))
        gcl_med    = float(np.nanmedian(vol_surfs[idx["GCL_IPL"]] - vol_surfs[idx["RNFL_GCL"]]))
        retina_med = float(np.nanmedian(vol_surfs[idx["RPE_BM"]] - vol_surfs[idx["ILM"]]))
        gcc_med    = float(np.nanmedian(vol_surfs[idx["GCL_IPL"]] - vol_surfs[idx["ILM"]]))  # 1–3

        ok_rnfl   = (rnfl_med > 0)
        ok_retina = (retina_med >= max(gcc_med, rnfl_med + gcl_med) - 10.0)  # tol=10µm
        ok_order  = ok_rnfl and ok_retina
        base = (rnfl_med + gcl_med) if ok_order else -1e9
        return base, idx, {"rnfl": rnfl_med, "gcl+": gcl_med, "gcc": gcc_med, "retina": retina_med}

    sc1, idx1, dbg1 = score(asc)
    sc2, idx2, dbg2 = score(dsc)
    if sc1 >= sc2:
        return asc, idx1, dbg1, med
    else:
        return dsc, idx2, dbg2, med

def label_topcon6_surfaces(vol_surfs: np.ndarray):
    if vol_surfs.shape[0] != 6:
        return None, {}
    order, idx, dbg, med = _order_six_by_depth(vol_surfs)
    return idx, {"order": order, "med_depths": med, **dbg}

# ---- en‑face 生成（make_enface_bm の処理をそのまま利用） --------------------
def make_enface_for_ids(ids: List[str], oct_root: Path, enface_dir: Path,
                        seg_mode: str, ai_ckpt: Path | None, gpu: str) -> None:
    ef.OUT_DIR = enface_dir
    ensure_dir(enface_dir)
    for vid in sorted(set(ids)):
        src = None
        for ext in (".fda", ".dcm"):
            cand = (Path(oct_root) / f"{vid}{ext}")
            if cand.exists():
                src = cand; break
        if src is None:
            print(f"[WARN] {vid}: neither .fda nor .dcm under {oct_root} → skip en‑face"); continue
        try:
            ef.process_one(src_path=src, seg_mode=seg_mode, ckpt=(ai_ckpt if ai_ckpt else None), gpu=gpu)
        except Exception as e:
            warnings.warn(f"{vid}: en‑face generation failed: {e}")

# ---- 乳頭中心（W‑Net or CSV） ----------------------------------------------
def load_centers_csv(csv_path: Path) -> Dict[str, Tuple[float,float]]:
    df = pd.read_csv(csv_path, header=None)
    centers = {}
    for _, row in df.iterrows():
        try:
            cid = str(row.iloc[0]).split("_")[0]
            x, y = float(row.iloc[1]), float(row.iloc[2])
            centers[cid] = (x, y)
        except Exception:
            continue
    return centers

def run_predict_wnet_on_dir(img_dir: Path, ckpt: Path, out_dir: Path) -> Dict[str, Tuple[float,float]]:
    ensure_dir(out_dir)
    cmd = [sys.executable, str(Path(__file__).parent / "predict_disc_centroid.py"),
           "--ckpt", str(ckpt), "--img_dir", str(img_dir), "--out", str(out_dir)]
    print(">> run:", " ".join(cmd)); subprocess.run(cmd, check=True)
    df = pd.read_csv(out_dir / "centroids.csv")
    id2xy = {}
    for _, r in df.iterrows():
        fname = Path(r["filename"]).name
        cid = fname.replace("_enface_BM.png","").replace("_enface_ALL.png","")
        id2xy[cid] = (float(r["x"]), float(r["y"]))
    print(f"[CENTER] loaded {len(id2xy)} centres from {out_dir/'centroids.csv'}")
    return id2xy

# ---- CLI --------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="厚み/表面 .npy → en‑face → 乳頭中心 → 環状解析＋cpRNFLT／Macula／Disc2（ワンストップ）")
    ap.add_argument("--thick_dir", required=True, help=".npy フォルダ（*_seg.npy または 厚み Layers×992×(512|256)）")
    ap.add_argument("--pattern", default="*.npy", help="検索パターン")
    ap.add_argument("--npy_kind", choices=["auto","seg","thick"], default="auto", help="入力 .npy の種別")
    ap.add_argument("--npy_units", choices=["um","px"], default="um", help="--npy_kind=thick のときの単位")
    ap.add_argument("--voxel_y", type=float, default=2.6*885/992, help="px→µm 換算（Topcon 既定）")
    ap.add_argument("--gclplus_mode", choices=["gcl", "gclipl"], default="gcl",
                    help="GCL+ の定義: gcl=2–3（GCLのみ, 既定） / gclipl=2–4（GCL+IPL）")
    ap.add_argument("--layer_order", default="RNFL,GCL,IPL,INL,OPL,ONL,PR,Choroid,GCC,Retina",
                    help="--npy_kind=thick の層順（左から index=0..）")
    ap.add_argument("--rnfl_idx", type=int, default=0, help="--npy_kind=thick の RNFL インデックス")

    ap.add_argument("--oct_root", help=".fda/.dcm のルート（<ID>.fda / <ID>.dcm を想定。en‑face と中心推定で使用）")
    ap.add_argument("--enface_dir", help="既存 en‑face のフォルダ（指定時は生成をスキップ）")
    ap.add_argument("--seg_mode", choices=["topcon","ai","auto"], default="auto")
    ap.add_argument("--ai_ckpt", help="--seg_mode ai/auto 用 ckpt")
    ap.add_argument("--gpu", default="0")

    ap.add_argument("--center_mode", choices=["wnet","csv"], default="wnet", help="中心座標の取得方法（disc/disc2）")
    ap.add_argument("--wnet_ckpt", help="W‑Net .pth（--center_mode=wnet）")
    ap.add_argument("--center_out", default="centroid_out", help="W‑Net 出力フォルダ")
    ap.add_argument("--center_csv", help="中心 CSV（--center_mode=csv）")

    ap.add_argument("--eye_csv", help="Laterality CSV（ID,Eye）。Eye は OD/OS/右/左/R/L 等を許容")
    ap.add_argument("--out_dir", default="output", help="出力先（<out>/<source>/<ID>/ に保存）")
    ap.add_argument("--outer", type=float, default=3.46, help="測定外径 [mm]（disc）")
    ap.add_argument("--inner", type=float, default=3.46, help="測定内径 [mm]（disc）")
    ap.add_argument("--scan_range_mm", type=float, default=6.0, help="解析面 512px の一辺 [mm]")

    ap.add_argument("--cp_append", action="store_true",
                    help="cpRNFLT_12.csv を追記モードで出力（既定は新規作成/上書き）")

    # ---- disc2 専用 ----
    ap.add_argument("--d2_inner", type=float, default=2.8, help="disc2: 内側直径 [mm]（例: 2.8）")
    ap.add_argument("--d2_mid",   type=float, default=3.46, help="disc2: 中間直径 [mm]（例: 3.46）")
    ap.add_argument("--d2_outer", type=float, default=4.4, help="disc2: 外側直径 [mm]（例: 4.4）")

    ap.add_argument("--target", choices=["disc","macula","disc2"], default="disc",
                    help="解析対象: 視神経乳頭 (disc) / 黄斑 (macula) / 乳頭3重リング (disc2)")
    return ap.parse_args()

# ---- 1 core につき **複数**選択（ai と topcon を両方採用） -------------------
# PATCH: add 'require_992' so macula can relax the rigid 992×(512|256) filter.
def select_per_core_multi(npy_paths: List[Path], npy_kind_arg: str, require_992: bool=True) -> List[Tuple[str, Path, str, int, str]]:
    groups = defaultdict(list)
    for p in npy_paths:
        groups[id_from_path(p)].append(p)

    selected: List[Tuple[str, Path, str, int, str]] = []
    for cid, paths in groups.items():
        cand_by_src = defaultdict(list)
        for p in paths:
            k = npy_kind_arg if npy_kind_arg != "auto" else guess_npy_kind(p)
            try:
                vol = np.load(p, mmap_mode="r"); shape = vol.shape
                ok = (len(shape) == 3)
                if require_992:
                    ok = ok and (shape[1] == 992 and shape[2] in (512, 256))
                nlay = shape[0] if ok else 0
            except Exception:
                nlay = 0
            src = source_tag_for(p, k)
            score = -1
            if k == "seg":
                if nlay >= 9: score = 300
                elif nlay == 6: score = 200
            elif k == "thick":
                score = 100
            cand_by_src[src].append((score, p, k, nlay))

        picked_any = False
        for src in ("ai", "topcon"):
            if cand_by_src.get(src):
                srt = sorted(cand_by_src[src], reverse=True)
                score, p, k, nlay = srt[0]
                if score >= 0:
                    selected.append((cid, p, k, nlay, src)); picked_any = True

        if (not picked_any) and cand_by_src.get("thick"):
            srt = sorted(cand_by_src["thick"], reverse=True)
            score, p, k, nlay = srt[0]
            if score >= 0:
                selected.append((cid, p, k, nlay, "thick"))
    return selected

# ---- 共通: px/角度・半径ユーティリティ --------------------------------------
def _mm_to_px_radius(mm_radius: float, scan_range_mm: float) -> float:
    """6mm スキャン → 512px の対応。引数は **半径[mm]**。"""
    return (mm_radius / scan_range_mm) * 512.0

def _polar(shape, cx, cy):
    H, W = shape
    yy, xx = np.indices((H, W))
    dx = xx - cx
    dy = cy - yy  # 画面座標補正（上が +）
    r = np.hypot(dx, dy)
    theta = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0  # 0°=右(TEM), 90°=上(SUP)
    return r, theta, xx, yy

def _mask_sector(mask, theta, lo, hi):
    """角度 [lo,hi) の扇形。lo>hi は 360 を跨ぐ。"""
    if lo < hi:
        return mask & (theta >= lo) & (theta < hi)
    else:
        return mask & ((theta >= lo) | (theta < hi))

def _mean(arr, m):
    v = arr[m]
    if v.size == 0:
        return np.nan
    return float(np.nanmean(v))

# ---- disc2: 3 重円の面積解析 + 描画 -----------------------------------------
def analyse_disc2(core_id: str, layer_name: str, thick512: np.ndarray, eye: str,
                  disc_center_xy: Tuple[float,float], scan_range_mm: float,
                  diam_in: float, diam_mid: float, diam_out: float,
                  vmin: float=0.0, vmax: float=500.0):
    """
    2.8–3.46mm (inner annulus) / 3.46–4.4mm (outer annulus) を面積平均で評価。
    戻り値: (fig: Figure, df: DataFrame[1行])
    出力列は *_in / *_out の 2 系列（全体/上下/TSNI/12時計）。
    """
    cx, cy = disc_center_xy
    r, theta, xx, yy = _polar(thick512.shape, cx, cy)

    r1 = _mm_to_px_radius(diam_in/2.0, scan_range_mm)
    r2 = _mm_to_px_radius(diam_mid/2.0, scan_range_mm)
    r3 = _mm_to_px_radius(diam_out/2.0, scan_range_mm)

    ann_in  = (r > r1) & (r <= r2)
    ann_out = (r > r2) & (r <= r3)

    # halves (上下)
    upper = yy < cy
    lower = ~upper

    # TSNI quadrants (45°/135° 境界)
    T_in  = _mean(thick512, _mask_sector(ann_in,  theta, 315, 360) | _mask_sector(ann_in,  theta, 0, 45))
    S_in  = _mean(thick512, _mask_sector(ann_in,  theta,  45, 135))
    N_in  = _mean(thick512, _mask_sector(ann_in,  theta, 135, 225))
    I_in  = _mean(thick512, _mask_sector(ann_in,  theta, 225, 315))

    T_out = _mean(thick512, _mask_sector(ann_out, theta, 315, 360) | _mask_sector(ann_out, theta, 0, 45))
    S_out = _mean(thick512, _mask_sector(ann_out, theta,  45, 135))
    N_out = _mean(thick512, _mask_sector(ann_out, theta, 135, 225))
    I_out = _mean(thick512, _mask_sector(ann_out, theta, 225, 315))

    # 全体/上下
    All_in = _mean(thick512, ann_in)
    UH_in  = _mean(thick512, ann_in & upper)
    LH_in  = _mean(thick512, ann_in & lower)

    All_out = _mean(thick512, ann_out)
    UH_out  = _mean(thick512, ann_out & upper)
    LH_out  = _mean(thick512, ann_out & lower)

    # 12 時計（各 30°）— [0,30), [30,60), ... in/out 別々
    H_in  = []
    H_out = []
    for k in range(12):
        lo = (k*30) % 360
        hi = ((k+1)*30) % 360
        H_in.append(_mean(thick512, _mask_sector(ann_in, theta, lo, hi)))
        H_out.append(_mean(thick512, _mask_sector(ann_out, theta, lo, hi)))

    # -------- 図 --------
    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = plt.gca()
    im = ax.imshow(thick512, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")

    # 3 circles
    ax.add_patch(plt.Circle((cx,cy), r1,  fill=False, color="w", lw=1.5))
    ax.add_patch(plt.Circle((cx,cy), r2,  fill=False, color="w", lw=1.0, ls="--"))
    ax.add_patch(plt.Circle((cx,cy), r3,  fill=False, color="w", lw=1.0, ls="--"))

    # 45/135/… lines（中心の穴は描かない）
    for ang in (45, 135, 225, 315):
        rad = np.deg2rad(ang); ux, uy = np.cos(rad), -np.sin(rad)
        ax.plot([cx+ux*r1, cx+ux*r3], [cy-uy*r1, cy-uy*r3], 'w:', lw=1.0)

    # 12 本のガイド（薄め）
    for ang in range(0, 360, 30):
        rad = np.deg2rad(ang); ux, uy = np.cos(rad), -np.sin(rad)
        ax.plot([cx+ux*r1, cx+ux*r3], [cy-uy*r1, cy-uy*r3], 'w:', lw=0.5, alpha=0.5)

    # 四象限ラベル（「内/外」）
    ax.text(cx, cy-r3*0.85, f"S  {S_in:.0f}/{S_out:.0f}", color="w", ha="center", va="bottom", fontsize=10)
    ax.text(cx, cy+r3*0.85, f"I  {I_in:.0f}/{I_out:.0f}", color="w", ha="center", va="top",    fontsize=10)
    ax.text(cx+r3*0.85, cy, f"T  {T_in:.0f}/{T_out:.0f}", color="w", ha="left",   va="center", fontsize=10)
    ax.text(cx-r3*0.85, cy, f"N  {N_in:.0f}/{N_out:.0f}", color="w", ha="right",  va="center", fontsize=10)

    # 12 時計の数字（外側リングの外周に寄せて表示。内/外は「内/外」の順）
    for k in range(12):
        ang = k*30 + 15  # セクタ中央
        rad = np.deg2rad(ang); ux, uy = np.cos(rad), -np.sin(rad)
        rr = (r2 + r3) / 2.0
        ax.text(cx+ux*rr, cy-uy*rr, f"{H_in[k]:.0f}/{H_out[k]:.0f}", color="w",
                ha="center", va="center", fontsize=8)

    ax.set_title(f"{core_id} {layer_name}  disc2 ({diam_in}–{diam_mid} / {diam_mid}–{diam_out} mm) ({eye})")
    ax.set_xlim(0,512); ax.set_ylim(512,0); ax.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # -------- DataFrame --------
    row = {
        "core_id": core_id,
        "layer": layer_name,
        # 全体/上下
        "ALL_in": f"{All_in:.2f}", "UH_in": f"{UH_in:.2f}", "LH_in": f"{LH_in:.2f}",
        "ALL_out": f"{All_out:.2f}", "UH_out": f"{UH_out:.2f}", "LH_out": f"{LH_out:.2f}",
        # TSNI
        "T_in": f"{T_in:.2f}", "S_in": f"{S_in:.2f}", "N_in": f"{N_in:.2f}", "I_in": f"{I_in:.2f}",
        "T_out": f"{T_out:.2f}", "S_out": f"{S_out:.2f}", "N_out": f"{N_out:.2f}", "I_out": f"{I_out:.2f}",
    }
    for k in range(12):
        row[f"H{k+1:02d}_in"]  = f"{H_in[k]:.2f}"
        row[f"H{k+1:02d}_out"] = f"{H_out[k]:.2f}"

    df = pd.DataFrame([row])
    return fig, df

# ---- laterality-aware wrappers ---------------------------------------------
def flip_if_os_512(thick512: np.ndarray, cx: float, eye: str) -> Tuple[np.ndarray, float]:
    """OS の場合は左右反転して右眼基準に統一"""
    if eye == "OS":
        return np.fliplr(thick512), (512 - 1 - cx)
    return thick512, cx

def flip_if_os_rnfl_xz(rnfl_xz: np.ndarray, cx: float, eye: str) -> Tuple[np.ndarray, float]:
    if eye == "OS":
        return rnfl_xz[::-1, :], (512 - 1 - cx)
    return rnfl_xz, cx

# ---- メイン ------------------------------------------------------------------
def main():
    args = parse_args()
    THICK_DIR = Path(args.thick_dir)
    OUT_DIR   = Path(args.out_dir); ensure_dir(OUT_DIR)
    ENFACE_DIR= Path(args.enface_dir) if args.enface_dir else (OUT_DIR / "enface_bm"); ensure_dir(ENFACE_DIR)
    CENTER_OUT= Path(args.center_out); ensure_dir(CENTER_OUT)
    OCT_ROOT  = Path(args.oct_root) if args.oct_root else None

    print("=== oct_all_in_one.py (OD/OS-aware, verbose) ===")
    print(f"[CFG] out_dir={OUT_DIR}  cp_append={args.cp_append}  seg_mode={args.seg_mode}  scan_range_mm={args.scan_range_mm}  target={args.target}")

    all_paths = list_npys(THICK_DIR, args.pattern)
    if not all_paths:
        print(f"[ERROR] no npy found under {THICK_DIR} with '{args.pattern}'"); sys.exit(1)

    # PATCH: 992 固定を解除
    selected = select_per_core_multi(all_paths, args.npy_kind, require_992=False)
    if not selected:
        print("[ERROR] no usable npy after selection (see warnings above)"); sys.exit(1)
    ids = sorted({cid for cid, *_ in selected})
    print(f"[INFO] selected cores: {len(ids)}  (total items={len(selected)})")

    # en‑face生成と中心推定（disc/disc2 のときだけ必要）
    if args.target in ('disc','disc2'):
        if not args.enface_dir:
            if OCT_ROOT is None:
                print("[ERROR] en‑face を作るには --oct_root が必要です"); sys.exit(1)
            make_enface_for_ids(ids=ids, oct_root=OCT_ROOT, enface_dir=ENFACE_DIR,
                                seg_mode=args.seg_mode, ai_ckpt=(Path(args.ai_ckpt) if args.ai_ckpt else None),
                                gpu=args.gpu)
        if args.center_mode == 'csv':
            if not args.center_csv:
                print("[ERROR] --center_mode csv には --center_csv が必要です"); sys.exit(1)
            id2center = load_centers_csv(Path(args.center_csv))
            print(f"[CENTER] loaded {len(id2center)} centres from {args.center_csv}")
        else:
            if not args.wnet_ckpt:
                print("[ERROR] --center_mode wnet には --wnet_ckpt が必要です"); sys.exit(1)
            id2center = run_predict_wnet_on_dir(img_dir=ENFACE_DIR, ckpt=Path(args.wnet_ckpt), out_dir=CENTER_OUT)
    else:
        # macula: center is image center unless CSV provided
        if args.center_csv:
            id2center = load_centers_csv(Path(args.center_csv))
            print(f"[CENTER] loaded {len(id2center)} macula centres from {args.center_csv}")
        else:
            id2center = {cid: (256.0, 256.0) for cid in ids}

    # Laterality
    id2eye = build_id2eye(ids, OCT_ROOT, Path(args.eye_csv) if args.eye_csv else None)
    dist = {k: list(id2eye.values()).count(k) for k in ("OD","OS")}
    print(f"[EYE] OD={dist.get('OD',0)}  OS={dist.get('OS',0)}")

    # 収集用
    rows_annulus: List[pd.DataFrame] = []   # disc 用
    rows_disc2 : List[pd.DataFrame] = []    # disc2 用
    cp_rows: List[Dict[str,str]] = []
    unit_scale_thick = args.voxel_y if args.npy_units == "px" else 1.0
    layer_names_thick = [s.strip() for s in args.layer_order.split(",")]

    for core_id, npy_path, npy_kind, nlay, src_tag in selected:
        eye = id2eye.get(core_id, "OD")
        print(f"== process {core_id} [{src_tag}/{eye}] ({Path(npy_path).name}, nlay={nlay}) ==")
        cx, cy = id2center.get(core_id, (256,256))
        if not np.isfinite(cx) or not np.isfinite(cy) or cx < 0 or cy < 0:
            cx, cy = 256.0, 256.0

        vol = np.load(npy_path)  # shape (L, W, Z)
        # DICOM / Topcon とも、3D (L×W×Z) であれば受け入れて、
        # 後段で 512×512 / 512×256 にリサンプリングする
        if vol.ndim != 3:
            print(f"[WARN] shape {vol.shape} ≠ L×W×Z (3D) → skip"); continue

        out_dir = OUT_DIR / src_tag / core_id
        ensure_dir(out_dir)

        rnfl_xz = None; cx_cpr = cx

        # ---- MACULA ----
        if args.target == 'macula':
            if npy_kind == 'seg':
                if nlay >= 9:
                    for lname,(top,bot) in LAYERS_FROM_SURFS.items():
                        ti, bi = SURFS.index(top), SURFS.index(bot)
                        if max(ti, bi) >= vol.shape[0]:
                            print(f"[WARN] {core_id}: need {top}/{bot} but have only {vol.shape[0]} surfaces → skip {lname}")
                            continue
                        thick512 = make_thick512_from_surfs(vol, ti, bi, args.voxel_y)
                        thick512, cx_eff = flip_if_os_512(thick512, cx, eye)
                        vmin, vmax = VMAP.get(lname, (0, 500))
                        figs, df = analyse_macula(core_id=core_id, layer_name=lname, thick_map=thick512, eye=eye,
                                                  scan_range_mm=args.scan_range_mm, cx=cx_eff, cy=cy, vmin=vmin, vmax=vmax)
                        df.insert(1, 'Eye', eye); df.insert(2, 'Source', src_tag)
                        for key, fig in figs.items():
                            fig.savefig(out_dir / f"{lname}_{key}.png", dpi=300); plt.close(fig)
                        rows_annulus.append(df)  # macula_summary として後で出力
                elif nlay == 6:
                    idx_map, dbg = label_topcon6_surfaces(vol)
                    if not idx_map:
                        print(f"[WARN] {core_id}: cannot label 6 surfaces reliably → skip macula")
                    else:
                        if args.gclplus_mode == 'gclipl':
                            L6 = {'RNFL':('ILM','RNFL_GCL'),'GCL+':('RNFL_GCL','IPL_INL'),
                                  'GCC':('ILM','GCL_IPL'),'Retina':('ILM','RPE_BM'),'Choroid':('RPE_BM','CHOROID_OUT')}
                        else:
                            L6 = {'RNFL':('ILM','RNFL_GCL'),'GCL+':('RNFL_GCL','GCL_IPL'),
                                  'GCC':('ILM','GCL_IPL'),'Retina':('ILM','RPE_BM'),'Choroid':('RPE_BM','CHOROID_OUT')}
                        for lname,(top,bot) in L6.items():
                            ti, bi = idx_map[top], idx_map[bot]
                            thick_wz = (vol[bi] - vol[ti]) * args.voxel_y
                            thick512 = resize_512x512(thick_wz.T)
                            thick512, cx_eff = flip_if_os_512(thick512, cx, eye)
                            vmin, vmax = (VMAP_6_ONLY.get(lname) or VMAP.get(lname, (0, 500)))
                            figs, df = analyse_macula(core_id=core_id, layer_name=lname, thick_map=thick512, eye=eye,
                                                      scan_range_mm=args.scan_range_mm, cx=cx_eff, cy=cy, vmin=vmin, vmax=vmax)
                            df.insert(1, 'Eye', eye); df.insert(2, 'Source', src_tag)
                            for key, fig in figs.items():
                                fig.savefig(out_dir / f"{lname}_{key}.png", dpi=300); plt.close(fig)
                            rows_annulus.append(df)
                else:
                    print(f"[WARN] {core_id}: seg with {nlay} surfaces is unsupported (<6) → skip macula")
            else:
                for li, lname in enumerate(layer_names_thick[:vol.shape[0]]):
                    layer_wz = vol[li]
                    thick512 = thick_to_512x512_from_wz(layer_wz, unit_scale_thick)
                    thick512, cx_eff = flip_if_os_512(thick512, cx, eye)
                    vmin, vmax = VMAP.get(lname, (0, 500))
                    figs, df = analyse_macula(core_id=core_id, layer_name=lname, thick_map=thick512, eye=eye,
                                              scan_range_mm=args.scan_range_mm, cx=cx_eff, cy=cy, vmin=vmin, vmax=vmax)
                    df.insert(1, 'Eye', eye); df.insert(2, 'Source', src_tag)
                    for key, fig in figs.items():
                        fig.savefig(out_dir / f"{lname}_{key}.png", dpi=300); plt.close(fig)
                    rows_annulus.append(df)
            # macula は cpRNFLT 無し
            continue

        # ---- DISC / DISC2 共通: thickness map 生成 ----
        if npy_kind == "seg":
            if nlay >= 9:
                layer_pairs = list(LAYERS_FROM_SURFS.items())
                rnfl_src_pair = ("RNFL", ("ILM","RNFL_GCL"))
            elif nlay == 6:
                idx_map, dbg = label_topcon6_surfaces(vol)
                if not idx_map:
                    print(f"[WARN] {core_id}: cannot label 6 surfaces reliably → skip")
                    continue
                if args.gclplus_mode == "gclipl":
                    L6 = {"RNFL":("ILM","RNFL_GCL"), "GCL+":("RNFL_GCL","IPL_INL"),
                          "GCC":("ILM","GCL_IPL"), "Retina":("ILM","RPE_BM"), "Choroid":("RPE_BM","CHOROID_OUT")}
                else:
                    L6 = {"RNFL":("ILM","RNFL_GCL"), "GCL+":("RNFL_GCL","GCL_IPL"),
                          "GCC":("ILM","GCL_IPL"), "Retina":("ILM","RPE_BM"), "Choroid":("RPE_BM","CHOROID_OUT")}
                layer_pairs = list(L6.items())
                rnfl_src_pair = ("RNFL", ("ILM","RNFL_GCL"))
            else:
                print(f"[WARN] {core_id}: seg with {nlay} surfaces is unsupported (<6) → skip")
                continue

            # ループ（disc / disc2 の出力に分岐）
            for lname,(top,bot) in layer_pairs:
                if nlay >= 9:
                    ti, bi = SURFS.index(top), SURFS.index(bot)
                else:
                    ti, bi = idx_map[top], idx_map[bot]
                if max(ti, bi) >= vol.shape[0]:
                    print(f"[WARN] {core_id}: need {top}/{bot} but have only {vol.shape[0]} surfaces → skip {lname}")
                    continue
                thick_wz = (vol[bi] - vol[ti]) * args.voxel_y
                thick512 = resize_512x512(thick_wz.T)
                thick512, cx_eff = flip_if_os_512(thick512, cx, eye)
                vmin, vmax = (VMAP_6_ONLY.get(lname) or VMAP.get(lname, (0, 500)))

                if args.target == 'disc':
                    figs, df = analyse_annulus(
                        core_id        = core_id,
                        layer_name     = lname,
                        thick_map      = thick512,
                        disc_center_xy = (cx_eff, cy),
                        outer_diam_mm  = args.outer,
                        inner_diam_mm  = args.inner,
                        scan_range_mm  = args.scan_range_mm,
                        vmin=vmin, vmax=vmax
                    )
                    df.insert(1, "Eye", eye)
                    df.insert(2, "Source", src_tag)
                    for key, fig in figs.items():
                        fig.savefig(out_dir / f"{lname}_{key}.png", dpi=300); plt.close(fig)
                    rows_annulus.append(df)
                else:  # disc2
                    fig, df = analyse_disc2(
                        core_id=core_id, layer_name=lname, thick512=thick512, eye=eye,
                        disc_center_xy=(cx_eff, cy), scan_range_mm=args.scan_range_mm,
                        diam_in=args.d2_inner, diam_mid=args.d2_mid, diam_out=args.d2_outer,
                        vmin=vmin, vmax=vmax
                    )
                    df.insert(1, "Eye", eye)
                    df.insert(2, "Source", src_tag)
                    fig.savefig(out_dir / f"{lname}_disc2.png", dpi=300); plt.close(fig)
                    rows_disc2.append(df)

            # RNFL xz for cpRNFLT (共通; 3.4mm)
            if nlay >= 9:
                rnfl_xz = rnfl_from_surfs_to_xz(vol, args.voxel_y)
            else:
                rnfl_wz = (vol[idx_map["RNFL_GCL"]] - vol[idx_map["ILM"]]) * args.voxel_y
                rnfl_xz = rnfl_wz_to_xz512x256(rnfl_wz, 1.0)
            rnfl_xz, cx_cpr = flip_if_os_rnfl_xz(rnfl_xz, cx, eye)

        else:
            # thick（Layers×992×512/256）
            for li, lname in enumerate(layer_names_thick[:vol.shape[0]]):
                layer_wz = vol[li]
                thick512 = thick_to_512x512_from_wz(layer_wz, unit_scale_thick)
                thick512, cx_eff = flip_if_os_512(thick512, cx, eye)
                vmin, vmax = VMAP.get(lname, (0, 500))

                if args.target == 'disc':
                    figs, df = analyse_annulus(
                        core_id        = core_id,
                        layer_name     = lname,
                        thick_map      = thick512,
                        disc_center_xy = (cx_eff, cy),
                        outer_diam_mm  = args.outer,
                        inner_diam_mm  = args.inner,
                        scan_range_mm  = args.scan_range_mm,
                        vmin=vmin, vmax=vmax
                    )
                    df.insert(1, "Eye", eye)
                    df.insert(2, "Source", src_tag)
                    for key, fig in figs.items():
                        fig.savefig(out_dir / f"{lname}_{key}.png", dpi=300); plt.close(fig)
                    rows_annulus.append(df)
                else:  # disc2
                    fig, df = analyse_disc2(
                        core_id=core_id, layer_name=lname, thick512=thick512, eye=eye,
                        disc_center_xy=(cx_eff, cy), scan_range_mm=args.scan_range_mm,
                        diam_in=args.d2_inner, diam_mid=args.d2_mid, diam_out=args.d2_outer,
                        vmin=vmin, vmax=vmax
                    )
                    df.insert(1, "Eye", eye)
                    df.insert(2, "Source", src_tag)
                    fig.savefig(out_dir / f"{lname}_disc2.png", dpi=300); plt.close(fig)
                    rows_disc2.append(df)

            rnfl_idx = args.rnfl_idx
            if rnfl_idx < vol.shape[0]:
                rnfl_xz = rnfl_wz_to_xz512x256(vol[rnfl_idx], unit_scale_thick)
                rnfl_xz, cx_cpr = flip_if_os_rnfl_xz(rnfl_xz, cx, eye)
            else:
                print(f"[WARN] rnfl_idx out of range → cpRNFLT skip"); rnfl_xz = None

        # ---- cpRNFLT（disc / disc2 共通） ---------------------------------
        if rnfl_xz is not None:
            cz = cy * (rnfl_xz.shape[1] / 512.0)   # 512→256 に換算
            row = compute_cprnflt12_row(core_id, src_tag, eye, rnfl_xz, cx_cpr, cz)
            cp_rows.append(row)
            print(f"  [cpRNFLT] added row for {core_id} [{src_tag}/{eye}]")

    # === CSV 出力 ==========================================================
    if args.target == 'macula':
        if rows_annulus:
            final_df = pd.concat(rows_annulus, ignore_index=True)
            if 'Eye' not in final_df.columns:   final_df.insert(1, 'Eye', 'OD')
            if 'Source' not in final_df.columns:final_df.insert(2, 'Source', 'ai')
            csv_path = Path(args.out_dir) / 'macula_summary.csv'
            final_df.to_csv(csv_path, index=False)
            print('✔ macula_summary.csv saved:', csv_path)
            print('[macula_summary.csv] columns:', list(final_df.columns))
    elif args.target == 'disc':
        if rows_annulus:
            final_df = pd.concat(rows_annulus, ignore_index=True)
            if 'Eye' not in final_df.columns:   final_df.insert(1, 'Eye', 'OD')
            if 'Source' not in final_df.columns:final_df.insert(2, 'Source', 'ai')
            ann_csv = Path(args.out_dir) / 'annulus_summary.csv'
            final_df.to_csv(ann_csv, index=False)
            print('✔ annulus_summary.csv saved:', ann_csv)
            print('[annulus_summary.csv] columns:', list(final_df.columns))
    else:  # disc2
        if rows_disc2:
            final_df = pd.concat(rows_disc2, ignore_index=True)
            if 'Eye' not in final_df.columns:   final_df.insert(1, 'Eye', 'OD')
            if 'Source' not in final_df.columns:final_df.insert(2, 'Source', 'ai')
            d2_csv = Path(args.out_dir) / 'disc2_summary.csv'
            final_df.to_csv(d2_csv, index=False)
            print('✔ disc2_summary.csv saved:', d2_csv)
            print('[disc2_summary.csv] columns:', list(final_df.columns))

    if cp_rows:
        cp_csv = Path(args.out_dir) / 'cpRNFLT_12.csv'
        if not args.cp_append:
            with open(cp_csv, 'w', encoding='utf-8') as f:
                f.write(','.join(CSV_COLS) + '\n')
                for r in cp_rows:
                    f.write(','.join([r[c] for c in CSV_COLS]) + '\n')
            mode = 'overwrite'
        else:
            header_exists = cp_csv.exists() and cp_csv.stat().st_size > 0
            if header_exists:
                try:
                    with open(cp_csv, 'r', encoding='utf-8') as f:
                        first = f.readline().strip()
                    if first.split(',') != CSV_COLS:
                        print(f"[WARN] existing header differs from expected: {first}")
                except Exception:
                    pass
            with open(cp_csv, 'a', encoding='utf-8') as f:
                if not header_exists:
                    f.write(','.join(CSV_COLS) + '\n')
                for r in cp_rows:
                    f.write(','.join([r[c] for c in CSV_COLS]) + '\n')
            mode = 'append'
        print(f'✔ cpRNFLT_12.csv saved ({mode}):', cp_csv)
        try:
            head = pd.read_csv(cp_csv, nrows=0)
            print('[cpRNFLT_12.csv] columns:', list(head.columns))
        except Exception:
            pass

# ---- Macula analysis helpers（現行版そのまま） -------------------------------
def _macula_masks(shape, cx, cy, scan_range_mm):
    H, W = shape
    yy, xx = np.indices((H, W))
    dx = xx - cx
    dy = cy - yy  # up is + (screen y down)
    r = np.hypot(dx, dy)
    theta = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0  # 0°=right(TEM), 90°=up(SUP)

    def _mm_to_px(mm: float, scan_range_mm: float) -> float:
        return (mm / scan_range_mm) * 512.0

    r0 = _mm_to_px(0.5, scan_range_mm)  # 1mm diameter
    r1 = _mm_to_px(1.5, scan_range_mm)  # 3mm diameter
    r2 = _mm_to_px(3.0, scan_range_mm)  # 6mm diameter

    mC   = (r <= r0)
    rin  = (r > r0) & (r <= r1)
    rout = (r > r1) & (r <= r2)

    # Temporal: -45..45 -> 315..360 + 0..45
    def sector(mask, lo, hi):
        if lo < hi:
            return mask & (theta >= lo) & (theta < hi)
        else:  # wrap around 360
            return mask & ((theta >= lo) | (theta < hi))

    iTEM = sector(rin, 315, 360) | sector(rin, 0, 45)
    iSUP = sector(rin, 45, 135)
    iNAS = sector(rin, 135, 225)
    iINF = sector(rin, 225, 315)

    oTEM = sector(rout, 315, 360) | sector(rout, 0, 45)
    oSUP = sector(rout, 45, 135)
    oNAS = sector(rout, 135, 225)
    oINF = sector(rout, 225, 315)

    # halves
    upper = (yy < cy)
    lower = (yy >= cy)

    # 4x4 grid
    xs = np.linspace(0, 512, 5).astype(int)
    ys = np.linspace(0, 512, 5).astype(int)
    grids = []
    for gi in range(4):
        row = []
        for gj in range(4):
            m = (yy >= ys[gi]) & (yy < ys[gi+1]) & (xx >= xs[gj]) & (xx < xs[gj+1])
            row.append(m)
        grids.append(row)

    masks = {
        "C": mC,
        "iSUP": iSUP, "iNAS": iNAS, "iINF": iINF, "iTEM": iTEM,
        "oSUP": oSUP, "oNAS": oNAS, "oINF": oINF, "oTEM": oTEM,
        "upper": upper, "lower": lower,
        "grids": grids,
        "radii_px": (r0, r1, r2),
        "theta": theta
    }
    return masks

def _region_mean(arr, mask):
    v = arr[mask]
    if v.size == 0:
        return np.nan
    return float(np.nanmean(v))

def analyse_macula(core_id: str, layer_name: str, thick_map: np.ndarray,
                   eye: str, scan_range_mm: float, cx: float=256.0, cy: float=256.0,
                   vmin: float=0.0, vmax: float=500.0):
    # masks
    masks = _macula_masks(thick_map.shape, cx, cy, scan_range_mm)

    # global / halves
    global_mean = float(np.nanmean(thick_map))
    upper_mean  = _region_mean(thick_map, masks["upper"])
    lower_mean  = _region_mean(thick_map, masks["lower"])

    # 4x4 grid
    grid_means = {}
    for i in range(4):
        for j in range(4):
            key = f"G{i+1}{j+1}"
            grid_means[key] = _region_mean(thick_map, masks["grids"][i][j])

    # ETDRS
    etdrs_keys = ["C","iSUP","iNAS","iINF","iTEM","oSUP","oNAS","oINF","oTEM"]
    etdrs_means = {k: _region_mean(thick_map, masks[k]) for k in etdrs_keys}

    # ----- figures -----
    figs = {}

    # 1) ETDRS overlay（45/135 分割）
    fig1 = plt.figure(figsize=(6,6), dpi=150)
    ax1 = plt.gca()
    im = ax1.imshow(thick_map, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    r0, r1, r2 = masks["radii_px"]
    circ = plt.Circle((cx,cy), r0, fill=False, color="w", lw=1.5)
    ax1.add_patch(circ)
    for rr in [r1, r2]:
        ax1.add_patch(plt.Circle((cx,cy), rr, fill=False, color="w", lw=1.0, ls="--"))
    # 45/135/… lines（中央 1mm 円の内側は引かない）
    for ang in (45, 135, 225, 315):
        rad = np.deg2rad(ang); ux, uy = np.cos(rad), -np.sin(rad)
        ax1.plot([cx+ux*r0, cx+ux*r2], [cy-uy*r0, cy-uy*r2], 'w--', lw=1.0)
    # ラベル
    ax1.text(cx+r1*0.65, cy, "iTEM", color="w", ha="left", va="center", fontsize=8)
    ax1.text(cx-r1*0.65, cy, "iNAS", color="w", ha="right", va="center", fontsize=8)
    ax1.text(cx, cy-r1*0.65, "iSUP", color="w", ha="center", fontsize=8)
    ax1.text(cx, cy+r1*0.65, "iINF", color="w", ha="center", va="top", fontsize=8)
    ax1.text(cx+r2*0.8, cy, "oTEM", color="w", ha="left", va="center", fontsize=8)
    ax1.text(cx-r2*0.8, cy, "oNAS", color="w", ha="right", va="center", fontsize=8)
    ax1.text(cx, cy-r2*0.8, "oSUP", color="w", ha="center", va="bottom", fontsize=8)
    ax1.text(cx, cy+r2*0.8, "oINF", color="w", ha="center", va="top", fontsize=8)
    ax1.set_title(f"{core_id} {layer_name}  ETDRS ({eye})")
    ax1.set_xlim(0,512); ax1.set_ylim(512,0)
    ax1.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    figs["macula_etdrs"] = fig1

    # 2) 4x4 grid
    fig2 = plt.figure(figsize=(6,6), dpi=150)
    ax2 = plt.gca()
    im2 = ax2.imshow(thick_map, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    xs = np.linspace(0, 512, 5)
    ys = np.linspace(0, 512, 5)
    for x in xs: ax2.plot([x,x],[0,512],"w-",lw=0.6)
    for y in ys: ax2.plot([0,512],[y,y],"w-",lw=0.6)
    for i in range(4):
        for j in range(4):
            key = f"G{i+1}{j+1}"
            xm = (xs[j]+xs[j+1])/2; ym = (ys[i]+ys[i+1])/2
            ax2.text(xm, ym, f"{grid_means[key]:.0f}", color="w", ha="center", va="center", fontsize=8)
    ax2.set_title(f"{core_id} {layer_name}  4x4 grid ({eye})")
    ax2.set_xlim(0,512); ax2.set_ylim(512,0)
    ax2.axis("off")
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    figs["macula_grid16"] = fig2

    # 3) upper/lower halves
    fig3 = plt.figure(figsize=(6,6), dpi=150)
    ax3 = plt.gca()
    im3 = ax3.imshow(thick_map, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    ax3.plot([0,512],[cy,cy],"w-",lw=1.0)
    ax3.text(256, cy-10, f"iSUP: {upper_mean:.1f}", color="w", ha="center", va="bottom", fontsize=10)
    ax3.text(256, cy+10, f"iINF: {lower_mean:.1f}", color="w", ha="center", va="top", fontsize=10)
    ax3.set_title(f"{core_id} {layer_name}  halves ({eye})")
    ax3.set_xlim(0,512); ax3.set_ylim(512,0); ax3.axis("off")
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    figs["macula_half"] = fig3

    # ----- dataframe row -----
    row = {
        "core_id": core_id,
        "layer": layer_name,
        "global": f"{global_mean:.2f}",
        "upper": f"{upper_mean:.2f}",
        "lower": f"{lower_mean:.2f}",
    }
    for k in [f"G{i}{j}" for i in range(1,5) for j in range(1,5)]:
        row[k] = f"{grid_means[k]:.2f}"
    row.update({k: f"{etdrs_means[k]:.2f}" for k in ["C","iSUP","iNAS","iINF","iTEM","oSUP","oNAS","oINF","oTEM"]})
    df = pd.DataFrame([row])
    return figs, df

# ---- cpRNFLT 内部関数（calculation.py と等価） -------------------------------
def _ring_sample_xz(rnfl_xz: np.ndarray, cx: float, cz: float, n=720) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = cx + R_PX*np.cos(theta); zs = cz - R_PX*np.sin(theta)
    xs = np.clip(xs, 0, rnfl_xz.shape[0]-1); zs = np.clip(zs, 0, rnfl_xz.shape[1]-1)
    return ndi.map_coordinates(rnfl_xz, np.vstack([xs, zs]), order=1, mode="nearest")

def _sector_means_12(ring: np.ndarray) -> List[float]:
    pts = ring.size // 12
    means = [float(np.nanmean(ring[i*pts:(i+1)*pts])) for i in range(12)]
    return [0.0 if (np.isnan(m) or not np.isfinite(m)) else m for m in means]

def compute_cprnflt12_row(pid: str, source_tag: str, eye: str, rnfl_xz: np.ndarray, cx: float, cz: float) -> Dict[str, str]:
    ring   = _ring_sample_xz(rnfl_xz, cx, cz)
    s_means = _sector_means_12(ring)
    total   = float(np.nanmean(ring))
    uh      = float(np.nanmean(ring[180:540]))
    lh      = float(np.nanmean(np.concatenate([ring[:180], ring[540:]])))
    return {
        "PID": pid,
        "Source": source_tag,
        "Eye": eye,
        **{f"S{i}": f"{m:.2f}" for i, m in enumerate(s_means, 1)},
        "Total": f"{total:.2f}",
        "UH": f"{uh:.2f}",
        "LH": f"{lh:.2f}",
    }

if __name__ == "__main__":
    main()
