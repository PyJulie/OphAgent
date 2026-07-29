# postprocess/vessels

Clean-room rewrite of the retinal vessel quantification layer. Takes binary
artery/vein masks (plus an OD/OC disc mask) and returns a structured JSON
with CRAE/CRVE, AVR, fractal dimension, tortuosity, and vessel density.

## Entry point

```python
from postprocess.vessels import analyze_vessels, VesselAnalysisConfig

result = analyze_vessels(
    artery_mask,       # HxW uint8, non-zero = artery
    vein_mask,         # HxW uint8, non-zero = vein
    disc_mask,         # HxW uint8, non-zero = disc region (disc ∪ cup). REQUIRED.
    fundus_mask=None,  # HxW bool, optional; falls back to full-image denominator
    pixel_spacing_um=None,   # provide if you know μm/px; outputs in μm when set
    config=None,       # optional VesselAnalysisConfig
)
```

All masks must already be in the **same coordinate frame** (e.g. all at the
640×640 analysis resolution after the same black-edge crop).

## Metrics

| Metric | File | Method |
|---|---|---|
| Disc geometry | `zones.py` | Centroid of disc mask; radius = √(area/π). No convex hull. |
| Measurement zone | `zones.py` | Annulus in disc-diameters from centre. Default `[0.5 DD, 1.0 DD]` per Knudtson 2003. |
| Per-vessel diameter | `diameter.py` | Zhang-Suen skeleton → break at bifurcations (≥3 8-conn neighbours) → keep arcs crossing zone B → median of 2·EDT along zone-B pixels. |
| CRAE / CRVE | `crae_crve.py` | Knudtson 2003 iterative pair-combining with k=0.88 (artery), k=0.95 (vein), top-6 diameters. Odd-count: middle value passes through. |
| AVR | `api.py` | CRAE / CRVE. |
| Fractal dimension | `fractal.py` | Box counting on skeleton, log-spaced box sizes from 2 to min(H,W)/4. Slope of log N(s) vs log(1/s). R² reported for QC. |
| Tortuosity | `tortuosity.py` | Distance Factor DF = arc_length / chord_length, per-segment, filtered by min length 40 px. Mean, median, length-weighted. |
| Vessel density | `density.py` | Foreground pixels / fundus area (or full image if no fundus mask). |

## Design choices

1. **Use the OD/OC disc mask, don't re-estimate.** The previous pipeline
   located the disc from vessel density, which is less accurate than the
   model-produced mask. Here we require `disc_mask` and raise `NoDiscError`
   if it is empty.
2. **Equivalent-area radius.** Disc radius = √(area/π) rather than bounding-box
   half-width. Robust to non-circular disc segmentations.
3. **Analysis runs in the original image frame (isotropic pixels).** The
   segmentation model runs on an anisotropically-resized 640×640 copy (same
   as its training), so disc gets stretched into an ellipse inside the model.
   We NN-upsample the output masks back to the original shape, where camera
   pixels are square again, and run every geometric measurement in that frame.
4. **All length thresholds are DD-normalised.** `min_segment_length_dd=0.5`
   means half a disc diameter. The per-image threshold in pixels is derived
   from the disc mask at run time. Across images of very different resolution
   or magnification, the threshold captures the same anatomical scale.
5. **Every length field is reported in three units:** `_px` (pixels in the
   analysis frame), `_dd` (disc diameters — cross-image comparable), `_um`
   (micrometres, non-null only when caller provides `pixel_spacing_um`).
6. **Fundus mask is a first-class input.** Density denominators and the
   skeleton-fg QC ratio use the illuminated retinal area, not the full
   canvas. `scripts/infer.py` writes `masks/fundus_mask.png` for each image
   so downstream postprocess modules can share the same denominator.
7. **Diameter = median of 2·EDT along skeleton in zone B.** Median avoids the
   distance-transform dip near skeleton termini and is not dominated by a
   single outlier pixel.
8. **Knudtson's original pair-combining order** is kept (big + small pairs
   iteratively).
9. **Fail loud.** `disc_mask=None` → `NoDiscError`. Low skeleton coverage,
   poor fractal R², reduced CRAE/CRVE sample, missing fundus mask → QC
   messages, never silent zero-fill.

## Output schema (v2 — triple-unit + DD-normalised thresholds)

```json
{
  "units": {
    "pixel_spacing_um": 7.5 | null,
    "disc_diameter_px": 83.5,
    "analysis_frame": "original_image",
    "length_fields_suffixes": ["_px", "_dd", "_um (if pixel_spacing_um given)"]
  },
  "qc": {
    "passed": true,
    "messages": [],
    "skeleton_fg_rate": 0.034,
    "fundus_area_source": "fundus_mask" | "full_image"
  },
  "disc": {
    "center_yx": [y, x],
    "radius_px": 45.0, "radius_dd": 0.5, "radius_um": 337.5,
    "area_px": 6400,
    "source": "od_oc_mask"
  },
  "measurement_zone": {
    "method": "knudtson_zone_b",
    "inner_dd": 0.5, "outer_dd": 1.0,
    "inner_px": 45.0, "outer_px": 90.0,
    "inner_um": 337.5, "outer_um": 675.0,
    "area_px": 12345
  },
  "crae": {
    "value_px": 18.96, "value_dd": 0.227, "value_um": 142.3,
    "method": "knudtson_2003", "k": 0.88,
    "n_vessels_used": 6,
    "top_diameters_px": [...], "top_diameters_dd": [...], "top_diameters_um": [...]
  },
  "crve": { "value_px": 29.29, "value_dd": 0.351, "value_um": 219.7, ... },
  "avr": { "value": 0.647, "definition": "CRAE / CRVE (Knudtson equivalents)" },
  "fractal_dimension": {
    "artery": { "value": 1.452, "r2": 0.996, "method": "box_counting_hausdorff",
                "box_sizes_px": [2,3,...], "box_counts": [...] },
    "vein":   { ... }
  },
  "tortuosity": {
    "artery": { "mean_df": 1.082, "median_df": 1.045, "length_weighted_df": 1.061,
                "n_segments": 23, "method": "distance_factor_per_segment",
                "min_segment_length_dd": 0.5, "min_segment_length_px": 42 },
    "vein":   { ... }
  },
  "density": {
    "artery_ratio": 0.065, "vein_ratio": 0.082, "total_ratio": 0.141,
    "fundus_area_px": 245000, "denominator_source": "fundus_mask"
  }
}
```

## Visualizations

```python
from postprocess.vessels import (
    analyze_vessels, render_vessel_visualizations, VesselVizConfig
)

result = analyze_vessels(artery, vein, disc, fundus_mask=fundus)
written = render_vessel_visualizations(
    original_image_bgr=cv2.imread("image.png"),
    artery_mask=artery, vein_mask=vein, disc_mask=disc, fundus_mask=fundus,
    analysis_result=result,
    output_dir="out/vessels",
    macula_center_yx=(y, x),           # optional
    config=VesselVizConfig(density=False),  # toggle any panel off
)
# written is {'overview': path, 'crae_crve': path, ..., 'combined': path}
```

### Panels

| Name | What it shows |
|---|---|
| `overview` | Original image + artery (red) + vein (blue) + disc outline (green) + Zone B annulus + macula marker |
| `crae_crve` | Same base dimmed; **top-N artery/vein segments selected by Knudtson** highlighted with diameter labels. Text panel carries CRAE/CRVE/AVR |
| `diameter_heatmap` | Viridis heatmap of `2·EDT` along the vessel union (bright = wider) with a colorbar |
| `tortuosity` | Each qualifying segment coloured by DF (green → red), chord drawn as gray line, DF labels on top-N most tortuous |
| `fractal` | Left: off-disc skeletons (artery red, vein blue) with overlay box-counting grid. Right: log–log scatter + fit line (D and R² in legend) |
| `density` | Grayscale base + vessel overlay + fundus outline + bar chart inset of artery/vein/total coverage |
| `combined` | 2×3 matplotlib mosaic of the above with a title bar carrying key metrics |

### CLI

```bash
python scripts/quantify_vessels.py --infer_dir ./out

# whitelist specific panels only
python scripts/quantify_vessels.py --infer_dir ./out --enable overview crae_crve

# blacklist specific panels
python scripts/quantify_vessels.py --infer_dir ./out --disable fractal density

# attach μm calibration if known
python scripts/quantify_vessels.py --infer_dir ./out --pixel_spacing_um 7.5
```

Output per image: `<image_dir>/vessels.json` and `<image_dir>/visualizations/vessels/vessels_*.png`.

### A note on the combined mosaic

`combined` is a 2×3 matplotlib assembly of the six base panels. If you ask
for the mosaic while any base panel is disabled, the empty cells show up as
blanks and the old full mosaic on disk gets overwritten — people have been
burned by this. To avoid silent surprises:

- The library call (`render_vessel_visualizations`) issues a Python
  ``UserWarning`` when this happens.
- The CLI prints a `[warn]` line to stderr with the missing panel names, so
  you can decide whether to re-run with them enabled or explicitly pass
  `--disable combined` to skip the mosaic.

## References

- Knudtson MD, Lee KE, Hubbard LD, Wong TY, Klein R, Klein BEK.
  *Revised formulas for summarizing retinal vessel diameters.*
  Curr Eye Res 27(3):143–9, 2003. doi:10.1076/ceyr.27.3.143.16049
- Hart WE, Goldbaum M, Côté B, Kube P, Nelson MR.
  *Measurement and classification of retinal vascular tortuosity.*
  Int J Med Inform 53(2-3):239–52, 1999.
- Mandelbrot BB. *The Fractal Geometry of Nature.* WH Freeman, 1982
  (box-counting dimension).
