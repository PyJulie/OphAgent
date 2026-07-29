# postprocess/tessellation

Quantification of the tessellation (豹纹样变) task — a single foreground
class reflecting visible choroidal pattern through a thin RPE, associated
with axial myopia.

## Entry point

```python
from postprocess.tessellation import analyze_tessellation

result = analyze_tessellation(
    tessellation_mask,       # HxW {0=bg, 1=tessellation}
    od_oc_mask,              # for DD and disc geometry
    fundus_mask=fundus,      # for coverage denominator
    macula_center_yx=(y, x), # optional — enables macula involvement + zones
    eye_side="OS",           # optional — enables nasal/temporal quadrant split
    pixel_spacing_um=7.5,    # enables μm² / mm² fields
)
```

## Metrics

* **coverage_ratio** — primary clinical signal (tessellation area / fundus area)
* **severity** — one of `{minimal, mild, moderate, severe}` via coverage
  thresholds (defaults `(0.05, 0.15, 0.30)`; overridable via config)
* **count**, **area (px, DD², μm²)**, **shape** (circularity, aspect ratio)
* **size_distribution** — small/medium/large bucketed in DD²
* **spatial** — macula-centred quadrant counts, distance-to-disc,
  distance-to-macula, macula-zone (1 / 2 DD) counts & areas
* **involves_macula** — any tessellation pixel within N × DD of macula
  (default 1.0 DD)

`severity_inputs` exposes a clean contract for a future myopic maculopathy
grading layer: coverage, severity bucket, macular involvement.

## Visualization panels

| Panel | What it shows |
|---|---|
| `overview` | Fundus with tessellation highlighted; title text with severity |
| `heatmap` | Gaussian-smoothed density map (magma colormap) |
| `macula_zones` | 1/2 DD concentric rings around macula + per-zone counts |
| `severity` | Info card: coverage, severity bucket, size distribution, a scale bar with the current value marked against the thresholds |
| `combined` | 2×2 mosaic |

## CLI

```bash
python scripts/quantify_tessellation.py --infer_dir ./out --eye_side OS
python scripts/quantify_tessellation.py --infer_dir ./out --disable heatmap
```
