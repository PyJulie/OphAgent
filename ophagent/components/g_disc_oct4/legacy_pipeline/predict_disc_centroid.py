#!/usr/bin/env python
"""
predict_disc_centroid.py  (v1 — 2025‑07‑08)

Optic‑Disc Centroid Predictor with W‑Net
=======================================

OVERVIEW
    This script uses a pre‑trained **W‑Net** model to segment the optic disc
    in colour fundus photographs and returns its centroid. For every input
    image it writes
        • an overlay PNG showing the predicted mask (blue), optional ground
          truth (red) and the centroid marker (red cross), and
        • a consolidated CSV file (`centroids.csv`) containing the (x, y)
          co‑ordinates of each prediction.

SUPPORTED INPUTS
    • A single image file (`--img`)
    • A folder of images (`--img_dir`)
    • Optional ground‑truth masks (`--msk_dir`) – file names must match the
      image stems.
    Accepted formats are those readable by OpenCV (PNG, JPG, TIFF …).

USAGE EXAMPLES
    # Batch inference with overlays + CSV summary
    python predict_disc_centroid.py \
        --ckpt wnet_disc512_best.pth \
        --img_dir ./data/image \
        --msk_dir ./data/mask \
        --out ./result

    # Test a single image (no GT masks)
    python predict_disc_centroid.py \
        --ckpt wnet_disc512_best.pth \
        --img sample.png \
        --out ./single_out

COMMAND‑LINE ARGUMENTS
    --ckpt <FILE>
        Required. Path to the W‑Net checkpoint (`.pth`).

    --img <FILE>
        Single image to process. Mutually exclusive with `--img_dir`.

    --img_dir <DIR>
        Directory containing images to process. Mutually exclusive with
        `--img`.

    --msk_dir <DIR>
        Optional directory with binary ground‑truth masks. When a matching
        mask is found it is rendered in *red* on the overlay for visual QC.

    --out <DIR>
        Output folder (default `./out`). Two sub‑artifacts are created:
            out/overlay/   – PNG overlays per image
            out/centroids.csv – consolidated centroids

OUTPUT FILES
    overlay/<stem>_overlay.png   – side‑by‑side RGB overlay (512×512 px)
    centroids.csv                – filename, x, y (‑1,‑1 if nothing found)

KEY CONSTANTS & ALGORITHM DETAILS
    • **Input size**          512 × 512 px – images / masks are resized with
      bicubic (images) or nearest‑neighbour (masks) interpolation.

    • **Normalisation**       Per‑channel ImageNet statistics
      (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).

    • **Threshold**           Probability > 0.5 → foreground mask.

    • **Post‑processing**
        – Keep only the largest connected component.
        – Fill internal holes.
        – Compute centroid via `scipy.ndimage.center_of_mass`.

    • **Overlay colours**
        – Prediction mask      → blue channel boosted (0.9)
        – Ground‑truth mask    → red channel boosted (0.3)
        – Centroid marker      → red tilted cross (OpenCV).

    • **CSV sentinel**        If no disc is detected the script records
      (x, y) = (‑1, ‑1) to signal “not found”.

MODIFYING BEHAVIOUR
    Change `preprocess()` for different input scales or normalisation.
    Adjust the threshold or component filtering logic in `post_process()`.

LIMITATIONS
    • The network expects a centred optic disc; extreme off‑axis cases may
      fail. Consider retraining W‑Net if you process highly variable data.
    • Only GPU index 0 is used; edit the `--gpu` option in `get_arch()` or
      set `CUDA_VISIBLE_DEVICES` externally for multi‑GPU setups.

Copyright 2025 Takahiro Ninomiya (t.ninomiya@ucl.ac.uk / ninomiya@tohoku.ac.jp)

"""


import os, sys, argparse, glob, csv
import cv2, torch, numpy as np, pandas as pd
import scipy.ndimage as ndi
from skimage import measure
from pathlib import Path

# ---------------- 引数 ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="学習済み .pth")
ap.add_argument("--img", help="単一画像パス")
ap.add_argument("--img_dir", help="画像フォルダパス")
ap.add_argument("--msk_dir", help="（任意）GT マスクフォルダ")
ap.add_argument("--out", default="out", help="出力フォルダ")
args = ap.parse_args()

# ---------------- モデル読込 ----------------
sys.path.append(".")                      # model_pkg が同階層にある想定
import model_pkg.get_model as gm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gm.get_arch("wnet", in_c=3, n_classes=1).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
model.eval()

# ---------------- 入力画像リスト ----------------
if args.img_dir:
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*")))
elif args.img:
    img_paths = [args.img]
else:
    ap.error("--img または --img_dir のどちらかを指定してください")

# マスク辞書（任意）
stem2msk = {}
if args.msk_dir:
    for mp in glob.glob(os.path.join(args.msk_dir, "*")):
        stem2msk[Path(mp).stem] = mp

# ---------------- 補助関数 ----------------
def preprocess(bgr):
    bgr = cv2.resize(bgr, (512,512), interpolation=cv2.INTER_CUBIC)
    rgb = bgr[:,:,::-1].astype(np.float32)/255.
    rgb = (rgb - [0.485,0.456,0.406])/[0.229,0.224,0.225]
    tensor = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float().to(dev)
    return tensor, bgr

def post_process(bin_mask):
    lbl = measure.label(bin_mask, connectivity=2)
    if lbl.max()==0: return bin_mask
    areas = ndi.sum(bin_mask, lbl, index=range(1,lbl.max()+1))
    largest = 1 + np.argmax(areas)
    return ndi.binary_fill_holes(lbl==largest).astype(np.uint8)

def centroid(mask):
    if mask.sum()==0: return None
    cy,cx = ndi.center_of_mass(mask)
    return int(round(cx)), int(round(cy))

# ---------------- 出力準備 ----------------
ov_dir = Path(args.out)/"overlay"; ov_dir.mkdir(parents=True, exist_ok=True)
records = []

# ---------------- ループ ----------------
for ip in img_paths:
    name = Path(ip).name
    stem = Path(ip).stem
    tensor, bgr = preprocess(cv2.imread(ip))

    with torch.no_grad():
        logit = model(tensor)
        if isinstance(logit,(list,tuple)): logit = logit[-1]
        prob = torch.sigmoid(logit)[0,0].cpu().numpy()

    pred = post_process((prob>0.5).astype(np.uint8))
    cen  = centroid(pred)
    cx, cy = cen if cen else (-1, -1)

    # ---- 可視化 ----
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    overlay = rgb.copy()
    overlay[...,2] = np.maximum(overlay[...,2], pred*0.9)   # 青 Pred

    if stem in stem2msk:                    # GT があれば重ねる
        gt = cv2.resize((cv2.imread(stem2msk[stem],0)>127).astype(np.uint8),
                        (512,512), interpolation=cv2.INTER_NEAREST)
        overlay[...,0] = np.maximum(overlay[...,0], gt*0.3) # 赤 GT

    if cen:
        cv2.drawMarker(overlay, cen, color=(1,0,0),
                       markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=18, thickness=2)

    out_png = ov_dir/f"{stem}_overlay.png"
    cv2.imwrite(str(out_png), (overlay*255).astype(np.uint8)[:,:,::-1])  # 保存はBGR

    records.append({"filename": name, "x": cx, "y": cy})
    print(f"{name:30s}  centroid = ({cx},{cy})  -> {out_png.name}")

# ---------------- CSV 出力 ----------------
csv_path = Path(args.out)/"centroids.csv"
pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nCSV saved to {csv_path.resolve()}")
