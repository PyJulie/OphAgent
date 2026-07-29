# postprocess/lesions

Clean-room lesion quantification. Takes per-class binary lesion masks (as
saved by ``scripts/infer.py``) together with the OD/OC mask, fundus mask,
macula coordinate and eye side, and returns a comprehensive nested JSON
describing every class plus group- and global-level summaries.

## Entry point

```python
from postprocess.lesions import (
    analyze_lesions, LesionAnalysisConfig, load_lesion_masks_from_dir,
)

lesion_masks = load_lesion_masks_from_dir("out/<image>/masks/")
result = analyze_lesions(
    lesion_masks=lesion_masks,
    od_oc_mask=od_oc,                  # HxW {0=bg, 1=rim, 2=cup}
    fundus_mask=fundus_mask,           # HxW uint8, illuminated retina
    macula_center_yx=(y, x),           # from coords.json (optional)
    eye_side="OS",                     # "OS"/"OD"/None (this model does not auto-derive)
    pixel_spacing_um=7.5,              # enables μm²/mm² outputs
)
```

## Class coverage

| Group | Classes |
|---|---|
| `lesion_dr` | hemorrhage, exudate, cotton_wool_spot, laser_spot |
| `lesion_amd` | drusen, patch_hemorrhage |
| `lesion_others` | epiretinal_membrane, artifact, retinal_scar, macular_hole |
| `possible_lesions` | other_lesions, myelinated_nerve_fiber, venous_tortuosity |

Every class is reported in the output even if absent (zeros) — downstream
consumers don't have to special-case missing classes.

## Per-class metrics

* **count** — connected components above the size threshold
* **area** — `_px`, `_dd2` (disc-diameter²), `_um2` (when spacing given)
* **coverage_ratio** — area / fundus area
* **shape** — mean circularity & aspect ratio
* **size_distribution** — components bucketed into small/medium/large with
  DD²-normalised thresholds
* **spatial**
  * macula-centred **quadrant counts** (null when eye_side or macula absent)
  * **distance-to-disc** and **distance-to-macula** (min/mean/median/max in DD)
  * **macula zones** — 0–1 DD, 1–2 DD, >2 DD counts and areas
* **components** — raw per-lesion centroid/area/shape list (opt-out via
  `include_components=False`)

## Group- and global-level summaries

Each group reports `total_count`, `total_area`, and `coverage_ratio`. The
`global_summary` reports `total_area_px`, `total_coverage_ratio`, and
`n_classes_detected`.

## Severity inputs (inputs only, NOT a grade)

* **DR** — ETDRS-style `hemorrhage_per_quadrant`, `heavy_hemorrhage_quadrants_count`
  (threshold configurable, default 20), `four_quadrants_heavy` boolean,
  cotton_wool_spot / exudate / laser_spot counts
* **AMD** — total drusen count, drusen within 1 DD of macula, patch
  hemorrhage presence
* **Other findings** — presence flags for ERM, retinal scar, macular hole,
  MNF, venous tortuosity, plus unspecified-lesions count

This module does not produce clinical grades — grading schemes vary and
should be decided by a clinician. The inputs are exposed cleanly so a
future disease-classification layer can consume them unambiguously.

## Design choices

1. **Masks are the record.** No convex hulls, no morphology. Noise is
   controlled solely by ``min_component_size_px`` + DD²-normalised size
   thresholds.
2. **DD-normalised thresholds.** A "small" lesion means the same clinical
   thing on a 600×600 scaled fundus as on a 3000×3000 native photo.
3. **Graceful degradation on missing inputs.** No macula → quadrant counts
   are null, macula zones are null. No eye_side → DR severity inputs are
   null. We do not guess.
4. **Every class always reported.** Zero-count classes still appear in the
   output so schemas are stable across images.

## CLI

```bash
python scripts/quantify_lesions.py --infer_dir ./out --eye_side OS --pixel_spacing_um 7.5

# Per-image eye_side via <image_dir>/eye_side.json (dict with "eye_side")
# overrides --eye_side.

# Panel whitelist / blacklist
python scripts/quantify_lesions.py --infer_dir ./out --enable overview burden
python scripts/quantify_lesions.py --infer_dir ./out --disable macula_zones
```

## Visualization panels

| Panel | What it shows |
|---|---|
| `overview` | Original + every detected class in its distinctive colour; legend with per-class counts |
| `per_group` | 2×2 small multiples, one cell per lesion group |
| `burden` | Horizontal bar charts: count + fundus-coverage % per class |
| `spatial` | Lesion centroids on fundus, dot size ∝ lesion area, macula-centred quadrant dividers |
| `macula_zones` | Concentric ETDRS-style rings at 1 / 2 DD from macula + per-zone counts |
| `size_distribution` | Stacked horizontal bars showing small / medium / large counts per class |
| `combined` | 2×3 mosaic with a title-line summary |

Every length/area in the schema carries `_px` / `_dd2` / `_um2` triples
where meaningful.
