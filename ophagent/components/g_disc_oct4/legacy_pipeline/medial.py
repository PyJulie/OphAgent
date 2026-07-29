# medial.py  ―  GPU-EDT を用いたメディアル厚み計算ヘルパ
# ===========================================================

from __future__ import annotations
import time, gc, subprocess, sys
from typing import Tuple
import numpy as np

# ---------------- ensure cupy -------------------------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt
except ImportError:
    print("→ installing cupy-cuda12x …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt

# ---------------- public API --------------------------------
def medial_thickness_layer(
    vol: np.ndarray,
    top_idx: int,
    bot_idx: int,
    *,
    voxel_y: float = 2.6,
    tol_px: int   = 1,
) -> np.ndarray:
    """
    Parameters
    ----------
    vol : np.ndarray
        shape (n_layer, 512, 256)  ; Y-index at each (x,z)
    top_idx / bot_idx : int
        indices of top / bottom boundary in vol
    voxel_y : float
        micron per pixel in Y-axis
    tol_px : int
        tolerance for |Dt - Db| to regard a voxel as medial (in pixel)

    Returns
    -------
    thickness_map : (512,256) float32 micron
        NaN where medial voxel is not found (should be rare)
    """
    H = 992                               # OCT-Cube height (Topcon 992px)
    W, Z = 512, 256
    y_top    = vol[top_idx]               # (W,Z)
    y_bottom = vol[bot_idx]

    # -------- 1. voxel-wise mask / surface -------------------
    inside_mask   = np.zeros((H, W, Z),   dtype=bool)
    surf_top_bool = np.zeros_like(inside_mask)
    surf_bot_bool = np.zeros_like(inside_mask)

    x = np.arange(W)
    for z in range(Z):
        yt = y_top   [:, z].astype(np.int32)
        yb = y_bottom[:, z].astype(np.int32)
        # skip invalid columns
        valid = (~np.isnan(yt)) & (~np.isnan(yb)) & (yt < yb)
        if not valid.any():
            continue
        yt = yt[valid]; yb = yb[valid]; xv = x[valid]
        for col, t, b in zip(xv, yt, yb):
            surf_top_bool [t, col, z] = True
            surf_bot_bool [b, col, z] = True
            inside_mask   [t:b+1, col, z] = True   # inclusive

    if not inside_mask.any():
        # fall back to axis-aligned if mask is empty
        return (y_bottom - y_top) * voxel_y

    # -------- 2. GPU distance transform ----------------------
    t0 = time.time()
    Dt_px_gpu = distance_transform_edt(cp.asarray(~surf_top_bool))
    Db_px_gpu = distance_transform_edt(cp.asarray(~surf_bot_bool))
    cp.cuda.Stream.null.synchronize()
    print(f"[medial] GPU EDT done in {time.time()-t0:.2f} s")

    Dt = (Dt_px_gpu * voxel_y).get().astype(np.float32)
    Db = (Db_px_gpu * voxel_y).get().astype(np.float32)

    # free GPU mem
    del Dt_px_gpu, Db_px_gpu
    cp.get_default_memory_pool().free_all_blocks(); gc.collect()

    # -------- 3. medial condition & thickness ---------------
    medial_mask = np.abs(Dt - Db) <= (tol_px * voxel_y)
    thk = (Dt + Db) * medial_mask
    thk[~medial_mask] = np.nan          # outside or undefined

    # 2D thickness map (x,z)  :=  nanmin along y
    return np.nanmin(thk, axis=0)       # shape (W,Z)
