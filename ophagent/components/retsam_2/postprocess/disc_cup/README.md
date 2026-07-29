# postprocess/disc_cup

Clean-room optic disc / optic cup quantification. Takes the OD/OC label map
produced by the segmentation model and returns CDR, ISNT rule compliance,
rim geometry at 4 cardinal axes + full 360° scan, disc tilt / ovality, and a
simplified Bayer–Nicolela DDLS grade.

## Entry point

```python
from postprocess.disc_cup import analyze_disc_cup, DiscCupAnalysisConfig

result = analyze_disc_cup(
    od_oc_mask,             # HxW uint8; 0=bg, 1=rim, 2=cup
    eye_side="OS",          # "OS" / "OD" / None. THIS MODEL DOES NOT auto-derive.
    pixel_spacing_um=7.5,   # enables mm² outputs and DDLS
)
```

### Mask convention

The segmentation model emits a single 3-valued label map where the **cup
label (2) is nested inside the rim label (1)**. In this layer we always
treat:

* ``disc = mask > 0``  — the union of rim and cup (the full disc region).
* ``cup  = mask == 2`` — just the cup.
* ``rim  = mask == 1`` — the annular rim tissue (computed as
  ``disc & ¬cup`` in practice).

No convex hulls, no morphological cleanup — the mask is the record.

## Metrics

| Metric | File | Method |
|---|---|---|
| Disc/cup centroid, bbox diameters, area | `geometry.py` | centroid + bbox extents; no convex hull |
| Equivalent diameter | `geometry.py` | 2·√(area / π) |
| Tilt angle, ovality | `geometry.py` | cv2.fitEllipse on the largest contour; angle reported from vertical |
| Vertical / horizontal / area CDR | `api.py` | bbox-based, **strict y- and x-axis** (not "longest disc diameter") |
| Cardinal rim widths (I, S, east, west, and → N/T by eye_side) | `rim.py` | rim = pixels that are disc & ¬cup along a cardinal ray from disc centre |
| ISNT rule compliance | `rim.py` | ordered check I ≥ S ≥ N ≥ T, returns explicit violation list |
| 360° rim profile | `rim.py` | per-degree rim width; min location + sector label; longest contiguous rim-absent arc |
| Per-quadrant rim area | `rim.py` | 90° wedges around disc centre; mapped to I/S/N/T when eye_side is known |
| DDLS grade (1–10) | `ddls.py` | Bayer & Nicolela, simplified. Thresholds per disc-size bucket (small/avg/large); rim-absent arcs override ratio grades |

Length fields are reported as ``{_px, _dd, _um}`` triples where ``_dd``
divides by the disc vertical diameter (DD-normalisation for cross-image
comparability) and ``_um`` is populated only when ``pixel_spacing_um`` is
given.

## QC

* ``cup_enclosed_by_rim`` — cup pixels with a bg 4-neighbour count as
  "touching the disc outer boundary"; more than the configured tolerance
  flips this to ``false`` with a ``qc.messages`` entry.
* ``cup_present`` — ``false`` when the cup mask is empty (CDR, ISNT widths
  and DDLS then come back as 0 / null).
* ``eye_side_source`` — ``user_provided`` or ``unknown``; when unknown the
  nasal/temporal fields are ``null`` (we do NOT guess from image position).

## Design choices

1. **Eye side is an explicit input, never auto-derived.** This module simply
   takes ``eye_side`` as a string or ``None``. Another model is responsible
   for producing it. When ``None``, nasal/temporal splits and ISNT
   compliance are reported as ``null``.
2. **vCDR is measured on the strict y-axis** (bbox height of cup / bbox
   height of disc), never along "whichever axis the disc happens to be
   longest along". Tilted discs therefore still get a clinically-meaningful
   vertical CDR.
3. **Rim width = pixels on the ray that lie in disc ∧ ¬cup**, counted
   outward from the disc centre. This definition is well-defined regardless
   of whether the ray passes through the cup and does not require the cup to
   contain the disc centroid.
4. **360° sector scan** drives both the minimum-rim location for DDLS and
   the rim-absent-arc detection for grades 6–10.
5. **DDLS thresholds are exposed via the config** so they can be tuned
   without editing code.

## Visualization panels

```python
from postprocess.disc_cup import render_disc_cup_visualizations, DiscCupVizConfig

written = render_disc_cup_visualizations(
    original_image_bgr=cv2.imread("image.png"),
    od_oc_mask=mask,
    analysis_result=result,
    output_dir="out/disc_cup",
    macula_center_yx=(y, x),  # optional
    config=DiscCupVizConfig(ddls=False),
)
```

| Panel | What it shows |
|---|---|
| `overview` | Original + disc (green) + cup (amber) + centres + macula + eye-side tag |
| `cdr` | Close-up of the disc region with explicit vertical & horizontal diameter lines, cup extent bracketed; vCDR / hCDR / aCDR / rim-disc ratio in a legend |
| `isnt` | Disc + cup + 4 cardinal rays labelled with rim widths; ISNT rule status ✓/✗ with the violating pairs listed |
| `rim_profile` | Matplotlib polar plot of rim width vs angle; I/S/N/T sectors shaded; min-rim point in red; rim-absent-arc size in the title |
| `tilt` | Disc outline + fitted ellipse + major axis + tilt angle / ovality text |
| `ddls` | Info card: disc size, min rim/disc, rim-absent arc, resulting DDLS grade + reasoning, plus a 0→0.5 colour scale with a marker at the measured ratio |
| `combined` | 2×3 matplotlib mosaic with a title bar of key metrics |

### A note on the combined mosaic

If you render with ``combined=True`` while any base panel is disabled the
resulting PNG has blank cells for them and **overwrites** any previously
full mosaic on disk — the library emits a ``UserWarning`` and the CLI
prints a ``[warn]`` line to stderr.

## CLI

```bash
# All panels, eye side applied to every image
python scripts/quantify_disc_cup.py --infer_dir ./out --eye_side OS

# Attach μm calibration to get mm² disc area and DDLS grading
python scripts/quantify_disc_cup.py --infer_dir ./out --eye_side OS --pixel_spacing_um 7.5

# Per-image eye side via sidecar: drop {"eye_side": "OS"} into
# <image_dir>/eye_side.json — that file takes priority over --eye_side.

# Panel whitelist / blacklist
python scripts/quantify_disc_cup.py --infer_dir ./out --enable overview cdr isnt
python scripts/quantify_disc_cup.py --infer_dir ./out --disable ddls
```

Output per image: ``<image_dir>/disc_cup.json`` and
``<image_dir>/visualizations/disc_cup/disc_cup_*.png``.

## References

* Bayer AU, Nicolela MT. *Disc damage likelihood scale (DDLS).*
  Br J Ophthalmol 2006;90:1045.
* Spaeth GL, et al. *The disc damage likelihood scale: reproducibility of a
  new method of estimating the amount of optic nerve damage caused by
  glaucoma.* Trans Am Ophthalmol Soc 2002;100:181–6.
* Jonas JB, et al. *Optic disk morphology in normal-pressure glaucoma.*
  Am J Ophthalmol 1988;106:316–22 (ISNT rule context).
