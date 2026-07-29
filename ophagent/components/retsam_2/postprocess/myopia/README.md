# postprocess/myopia

Quantification of the myopia task (3-class labelmap): arc_lesion,
diffuse_chorioretinal_atrophy, patchy_chorioretinal_atrophy.

## Entry point

```python
from postprocess.myopia import analyze_myopia, MyopiaAnalysisConfig

result = analyze_myopia(
    myopia_mask,               # HxW labelmap: 0=bg, 1=arc, 2=diffuse, 3=patchy
    od_oc_mask,                # for disc geometry
    fundus_mask=fundus,        # for coverage denominator
    macula_center_yx=(y, x),   # optional
    eye_side="OS",             # optional (OS / OD / None)
    pixel_spacing_um=7.5,
)
```

## Class-specific reporting

### arc_lesion (peripapillary crescent)
In addition to the generic per-class block (count / area / spatial) it carries
a disc-relative geometry sub-dict:
- `angular_coverage_deg` — how many degrees of the disc circumference have
  crescent pixels (search zone: 0 to 2 DD outward from the disc edge)
- `sector_involvement` — booleans for I / S / N / T (N and T null when
  eye_side unknown; raw east/west booleans still exposed)
- `max_radial_extent_dd` — farthest arc pixel from the disc edge, in DD

### diffuse_chorioretinal_atrophy
Continuous background atrophy — reported as coverage, macula involvement,
and distance-to-disc / distance-to-macula distributions.

### patchy_chorioretinal_atrophy
Discrete atrophy patches — component-level metrics (count, size dist,
quadrants, macula zones, macula involvement).

## Severity inputs

`severity_inputs` exposes the inputs that META-PM / ATN grading schemes
would consume (arc presence + coverage, diffuse/patchy presence, macular
involvement). **No grade is produced** — grading is clinical.

## Visualization panels

| Panel | What it shows |
|---|---|
| `overview` | All 3 classes colour-coded on dimmed fundus + disc/macula |
| `arc_profile` | Polar plot around the disc with shaded involved sectors |
| `atrophy_map` | Diffuse + patchy highlighted (no arc); macula & disc markers |
| `macula_zones` | 1 DD / 2 DD concentric rings + per-zone class aggregates |
| `burden` | Per-class count and coverage % bar charts |
| `components_map` | Patchy atrophy (and arc) component centroids sized by area |
| `combined` | 2×3 mosaic with a title-line summary |

## CLI

```bash
python scripts/quantify_myopia.py --infer_dir ./out --eye_side OS
python scripts/quantify_myopia.py --infer_dir ./out --disable burden
```

## Design choices

- **DD-normalised** thresholds everywhere; size buckets slightly bigger than
  for lesions since atrophy patches run larger.
- **Arc geometry from pixel binning**, not ray walking — simpler and works
  for arcs not connected to the disc border.
- **fail loud** on bad inputs; no silent eye_side guessing.
