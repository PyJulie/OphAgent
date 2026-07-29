"""
Vessel analysis visualizations.

Given the masks and the output of ``analyze_vessels``, render a set of
human-readable PNG panels that show *where* each metric came from. Each
panel is independently toggleable via ``VesselVizConfig``. The panels:

    * overview          — original image + artery/vein + disc + fundus + Zone B
    * crae_crve         — top-N arteries/veins selected by Knudtson highlighted
    * diameter_heatmap  — per-pixel 2·EDT diameter heatmap along vessels
    * tortuosity        — per-segment DF colouring with chord overlays
    * fractal           — off-disc skeletons + box-counting log–log plot
    * density           — fundus outline + vessel area breakdown
    * combined          — 2×3 mosaic of all of the above

All single-panel renderers return a BGR uint8 numpy array so they can be
saved with cv2.imwrite or embedded into the combined mosaic.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .diameter import (
    VesselSegment,
    extract_all_skeleton_segments,
    extract_vessel_segments,
)
from .zones import DiscGeometry, build_zone_b, compute_disc_geometry


# --------------------------------------------------------------------------- config

@dataclass
class VesselVizConfig:
    """Toggles and styling for vessel visualizations."""

    # Panel toggles
    overview: bool = True
    crae_crve: bool = True
    diameter_heatmap: bool = True
    tortuosity: bool = True
    fractal: bool = True
    density: bool = True
    combined: bool = True          # 2×3 mosaic assembled from the panels above

    # Styling / options
    alpha_vessel: float = 0.45     # artery/vein overlay transparency
    alpha_zone: float = 0.18       # Zone B fill transparency
    topn_label_count: int = 10     # for tortuosity panel (to avoid clutter)
    fractal_subgrid_size: Optional[int] = None  # overlaid grid size; None = auto
    dpi: int = 150
    jpeg_quality: int = 92         # unused with png
    combined_title: bool = True    # add global title bar with key metrics


# --------------------------------------------------------------------------- palette

# BGR colour constants (OpenCV channel order)
_COLOR = {
    "artery_dim":       (60, 60, 230),      # desaturated red
    "artery_bright":    (0, 0, 255),
    "vein_dim":         (230, 90, 30),      # desaturated blue
    "vein_bright":      (255, 60, 0),
    "overlap":          (200, 0, 200),      # purple where artery ∩ vein
    "disc_outline":     (0, 255, 0),
    "fundus_outline":   (255, 255, 255),
    "zone_fill":        (220, 255, 255),    # pale cyan fill
    "zone_edge":        (255, 255, 255),
    "disc_center":      (0, 255, 255),      # yellow-ish
    "macula":           (255, 255, 255),
    "chord":            (180, 180, 180),
    "text_fg":          (255, 255, 255),
    "text_bg":          (0, 0, 0),
}


# --------------------------------------------------------------------------- helpers

def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure a (H, W, 3) uint8 BGR image."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _blend_overlay(base: np.ndarray, color_bgr: Tuple[int, int, int],
                   mask: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha blend a solid colour into ``base`` where ``mask`` is truthy."""
    out = base.copy()
    m = mask.astype(bool)
    if not m.any():
        return out
    colour = np.array(color_bgr, dtype=np.float32)
    layer = out[m].astype(np.float32)
    out[m] = (layer * (1.0 - alpha) + colour * alpha).astype(np.uint8)
    return out


def _draw_mask_outline(canvas: np.ndarray, mask: np.ndarray,
                       color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Draw the outer contours of a binary mask on ``canvas`` (in place)."""
    if mask is None or mask.size == 0:
        return
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness=thickness)


# --------------------------------------------------------------------------- ASCII safety

_ASCII_MAP = str.maketrans({
    "°": "deg", "≥": ">=", "≤": "<=", "→": "->",
    "—": "-",  "–": "-",  "↔": "<->", "−": "-",
    "²": "^2", "³": "^3",
    "μ": "u",  "σ": "sd", "π": "pi",  "Δ": "Delta",
    "×": "x",  "±": "+/-",
    "✓": "[OK]", "✔": "[OK]",
    "✗": "[X]",  "✘": "[X]",
})


def _ascii_safe(text: str) -> str:
    """cv2.putText with FONT_HERSHEY_SIMPLEX renders non-ASCII as '???'.
    Translate the common scientific/clinical glyphs to ASCII, then replace
    anything else remaining with '?' as a last resort.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(_ASCII_MAP)
    return text.encode("ascii", errors="replace").decode("ascii")


def _put_text_with_bg(canvas: np.ndarray, text: str, pos: Tuple[int, int],
                      scale: float = 0.55, color: Tuple[int, int, int] = (255, 255, 255),
                      bg: Tuple[int, int, int] = (0, 0, 0),
                      thickness: int = 1, pad: int = 3) -> None:
    """Draw small text with a dark translucent background box for legibility."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    # Draw background rectangle
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - h - pad - 1),
        (x + w + pad, y + baseline + pad),
        bg,
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, _ascii_safe(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, lineType=cv2.LINE_AA)


def _text_block(canvas: np.ndarray, lines: Sequence[str],
                anchor: str = "tl", scale: float = 0.55,
                pad: int = 6, line_spacing: int = 6) -> None:
    """Paint a multi-line text block in one of the four corners."""
    h, w = canvas.shape[:2]
    text_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for line in lines
    ]
    box_w = max((tw for tw, _ in text_sizes), default=0) + 2 * pad
    box_h = sum(th for _, th in text_sizes) + line_spacing * (len(lines) - 1) + 2 * pad
    if anchor == "tl":
        x0, y0 = 8, 8
    elif anchor == "tr":
        x0, y0 = w - box_w - 8, 8
    elif anchor == "bl":
        x0, y0 = 8, h - box_h - 8
    else:  # "br"
        x0, y0 = w - box_w - 8, h - box_h - 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    y = y0 + pad
    for line, (tw, th) in zip(lines, text_sizes):
        y += th
        cv2.putText(canvas, _ascii_safe(line), (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, _COLOR["text_fg"], 1, cv2.LINE_AA)
        y += line_spacing


def _build_rdylgn_lut() -> np.ndarray:
    """Green → yellow → red 256-entry BGR LUT for tortuosity colouring."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            # green (0,255,0) → yellow (0,255,255)
            b, g, r = 0, 255, int(255 * t * 2)
        else:
            # yellow (0,255,255) → red (0,0,255)
            tt = (t - 0.5) * 2
            b, g, r = 0, int(255 * (1 - tt)), 255
        lut[i, 0] = (b, g, r)
    return lut


_RDYLGN_LUT = _build_rdylgn_lut()


def _colorbar_strip(width: int, height: int, cmap: str) -> np.ndarray:
    """Return a horizontal colorbar strip (BGR)."""
    gradient = np.linspace(0, 255, width, dtype=np.uint8).reshape(1, -1)
    gradient = np.repeat(gradient, height, axis=0)
    if cmap == "viridis":
        return cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    if cmap == "rdylgn":
        return cv2.LUT(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), _RDYLGN_LUT)
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- panels

def render_overview(
    image_bgr: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    disc: np.ndarray,
    fundus: Optional[np.ndarray],
    analysis: Dict[str, Any],
    config: VesselVizConfig,
    macula_yx: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """A+B: base layer + AV + disc + fundus + Zone B + macula."""
    canvas = _to_bgr_uint8(image_bgr).copy()

    canvas = _blend_overlay(canvas, _COLOR["vein_dim"], vein, config.alpha_vessel)
    canvas = _blend_overlay(canvas, _COLOR["artery_dim"], artery, config.alpha_vessel)
    overlap = (artery > 0) & (vein > 0)
    canvas = _blend_overlay(canvas, _COLOR["overlap"], overlap, config.alpha_vessel)

    # Zone B annulus
    disc_cx = float(analysis["disc"]["center_yx"][1])
    disc_cy = float(analysis["disc"]["center_yx"][0])
    inner_px = float(analysis["measurement_zone"]["inner_px"])
    outer_px = float(analysis["measurement_zone"]["outer_px"])
    zone_yy, zone_xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    dist = np.sqrt((zone_yy - disc_cy) ** 2 + (zone_xx - disc_cx) ** 2)
    zone_mask = (dist >= inner_px) & (dist <= outer_px)
    canvas = _blend_overlay(canvas, _COLOR["zone_fill"], zone_mask, config.alpha_zone)
    cv2.circle(canvas, (int(disc_cx), int(disc_cy)), int(inner_px),
               _COLOR["zone_edge"], 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (int(disc_cx), int(disc_cy)), int(outer_px),
               _COLOR["zone_edge"], 1, lineType=cv2.LINE_AA)

    # Disc and fundus outlines
    _draw_mask_outline(canvas, disc, _COLOR["disc_outline"], thickness=2)
    if fundus is not None:
        _draw_mask_outline(canvas, fundus, _COLOR["fundus_outline"], thickness=1)

    # Disc centre and macula markers
    cv2.drawMarker(canvas, (int(disc_cx), int(disc_cy)),
                   _COLOR["disc_center"], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), _COLOR["macula"], cv2.MARKER_TILTED_CROSS,
                       14, 2, cv2.LINE_AA)
        _put_text_with_bg(canvas, "macula", (mx + 8, my - 6), scale=0.4)

    dd = float(analysis["units"]["disc_diameter_px"])
    _text_block(canvas, [
        "Overview",
        f"disc centre: ({int(disc_cy)}, {int(disc_cx)})",
        f"DD: {dd:.1f} px    zone: [{inner_px:.0f}, {outer_px:.0f}] px",
        "artery=red    vein=blue    disc=green    macula=×",
    ], anchor="tl")
    return canvas


def _draw_segment_polyline(canvas: np.ndarray, seg: VesselSegment,
                           color: Tuple[int, int, int], thickness: int = 2) -> None:
    if len(seg.points_yx) < 2:
        return
    pts = seg.points_yx[:, [1, 0]].astype(np.int32)  # convert to (x, y) for cv2
    cv2.polylines(canvas, [pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _select_topn_segments(segments: List[VesselSegment], n: int) -> List[VesselSegment]:
    return sorted(segments, key=lambda s: s.diameter_px, reverse=True)[:n]


def render_crae_crve(
    image_bgr: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    disc: np.ndarray,
    analysis: Dict[str, Any],
    artery_zone_segments: List[VesselSegment],
    vein_zone_segments: List[VesselSegment],
    config: VesselVizConfig,
) -> np.ndarray:
    """C: highlight the top-N artery/vein segments used by Knudtson."""
    # Start from a darkened grayscale base so the top-N highlights pop.
    gray = cv2.cvtColor(_to_bgr_uint8(image_bgr), cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor((gray * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Dim the full artery/vein masks
    canvas = _blend_overlay(canvas, _COLOR["vein_dim"], vein, 0.30)
    canvas = _blend_overlay(canvas, _COLOR["artery_dim"], artery, 0.30)

    # Zone B outline
    disc_cx = float(analysis["disc"]["center_yx"][1])
    disc_cy = float(analysis["disc"]["center_yx"][0])
    inner_px = float(analysis["measurement_zone"]["inner_px"])
    outer_px = float(analysis["measurement_zone"]["outer_px"])
    cv2.circle(canvas, (int(disc_cx), int(disc_cy)), int(inner_px),
               (200, 200, 200), 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (int(disc_cx), int(disc_cy)), int(outer_px),
               (200, 200, 200), 1, lineType=cv2.LINE_AA)

    _draw_mask_outline(canvas, disc, _COLOR["disc_outline"], thickness=1)

    # Top-N highlights
    n_top = int(analysis["crae"]["n_vessels_used"])
    top_a = _select_topn_segments(artery_zone_segments, n_top)
    top_v = _select_topn_segments(vein_zone_segments, n_top)

    for seg in top_a:
        _draw_segment_polyline(canvas, seg, _COLOR["artery_bright"], thickness=3)
    for seg in top_v:
        _draw_segment_polyline(canvas, seg, _COLOR["vein_bright"], thickness=3)

    # Diameter labels near each top segment midpoint
    def _label(segs: List[VesselSegment]) -> None:
        for seg in segs:
            mid = seg.points_yx[len(seg.points_yx) // 2]
            pos = (int(mid[1]) + 5, int(mid[0]) - 5)
            _put_text_with_bg(canvas, f"{seg.diameter_px:.1f}px", pos, scale=0.38)

    _label(top_a)
    _label(top_v)

    crae = analysis["crae"]
    crve = analysis["crve"]
    avr = analysis["avr"]
    unit_suffix = ""
    if analysis["units"]["pixel_spacing_um"] is not None:
        unit_suffix = f"    ({crae['value_um']:.1f} μm / {crve['value_um']:.1f} μm)"
    _text_block(canvas, [
        "CRAE / CRVE (Knudtson 2003)",
        f"CRAE: {crae['value_px']:.2f} px  ({crae['value_dd']:.4f} DD){unit_suffix}",
        f"CRVE: {crve['value_px']:.2f} px  ({crve['value_dd']:.4f} DD)",
        f"AVR : {avr['value']:.4f}",
        f"n (top): artery={crae['n_vessels_used']}  vein={crve['n_vessels_used']}",
    ], anchor="tl")
    return canvas


def render_diameter_heatmap(
    image_bgr: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    disc: np.ndarray,
    fundus: Optional[np.ndarray],
    analysis: Dict[str, Any],
    config: VesselVizConfig,
) -> np.ndarray:
    """D: 2·EDT diameter heatmap along the combined vessel mask."""
    from scipy.ndimage import distance_transform_edt

    canvas = cv2.cvtColor(
        (cv2.cvtColor(_to_bgr_uint8(image_bgr), cv2.COLOR_BGR2GRAY) * 0.35).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )

    combined = ((artery > 0) | (vein > 0)).astype(np.uint8)
    edt = distance_transform_edt(combined)
    diam = 2.0 * edt

    vmax = float(np.percentile(diam[combined > 0], 99)) if combined.any() else 1.0
    vmax = max(vmax, 1.0)
    norm = np.clip(diam / vmax * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    vessel_mask = combined.astype(bool)
    out = canvas.copy()
    out[vessel_mask] = heat_bgr[vessel_mask]
    canvas = out

    # Zone B outline
    disc_cx = float(analysis["disc"]["center_yx"][1])
    disc_cy = float(analysis["disc"]["center_yx"][0])
    inner_px = float(analysis["measurement_zone"]["inner_px"])
    outer_px = float(analysis["measurement_zone"]["outer_px"])
    cv2.circle(canvas, (int(disc_cx), int(disc_cy)), int(inner_px),
               (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (int(disc_cx), int(disc_cy)), int(outer_px),
               (255, 255, 255), 1, lineType=cv2.LINE_AA)

    _draw_mask_outline(canvas, disc, _COLOR["disc_outline"], thickness=1)

    # Colorbar bottom-right
    cb_w, cb_h = 180, 14
    strip = _colorbar_strip(cb_w, cb_h, "viridis")
    h, w = canvas.shape[:2]
    x0, y0 = w - cb_w - 12, h - 36
    canvas[y0:y0 + cb_h, x0:x0 + cb_w] = strip
    _put_text_with_bg(canvas, "0", (x0 - 2, y0 + cb_h + 12), scale=0.4)
    _put_text_with_bg(canvas, f"{vmax:.1f} px", (x0 + cb_w - 40, y0 + cb_h + 12), scale=0.4)

    _text_block(canvas, [
        "Diameter heatmap (2·EDT)",
        f"peak: {vmax:.2f} px",
        "bright = wider vessel",
    ], anchor="tl")
    return canvas


def render_tortuosity(
    image_bgr: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    disc: np.ndarray,
    analysis: Dict[str, Any],
    artery_all_segments: List[VesselSegment],
    vein_all_segments: List[VesselSegment],
    config: VesselVizConfig,
) -> np.ndarray:
    """E: colour each qualifying segment by its distance-factor."""
    gray = cv2.cvtColor(_to_bgr_uint8(image_bgr), cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor((gray * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    min_px = int(analysis["tortuosity"]["artery"]["min_segment_length_px"])

    # DF range for colour mapping: 1.0 (straight) → 1.3 (very tortuous)
    df_min, df_max = 1.0, 1.3

    def df_of(seg: VesselSegment) -> float:
        if len(seg.points_yx) < 2:
            return 0.0
        pts = seg.points_yx.astype(np.float64)
        diffs = np.diff(pts, axis=0)
        arc = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        return arc / chord if chord > 0 else 0.0

    labelled: List[Tuple[VesselSegment, float]] = []

    for seg in artery_all_segments + vein_all_segments:
        if seg.length_px < min_px:
            continue
        df = df_of(seg)
        if df <= 0:
            continue
        t = np.clip((df - df_min) / (df_max - df_min), 0.0, 1.0)
        # map t→LUT index
        color = _RDYLGN_LUT[int(t * 255), 0].tolist()
        _draw_segment_polyline(canvas, seg, tuple(int(c) for c in color), thickness=2)
        # draw chord as dashed-ish gray line
        p0 = (int(seg.points_yx[0, 1]), int(seg.points_yx[0, 0]))
        p1 = (int(seg.points_yx[-1, 1]), int(seg.points_yx[-1, 0]))
        cv2.line(canvas, p0, p1, _COLOR["chord"], 1, lineType=cv2.LINE_AA)
        labelled.append((seg, df))

    # Label the N most tortuous segments
    labelled.sort(key=lambda x: x[1], reverse=True)
    for seg, df in labelled[: config.topn_label_count]:
        mid = seg.points_yx[len(seg.points_yx) // 2]
        _put_text_with_bg(canvas, f"{df:.2f}",
                          (int(mid[1]) + 4, int(mid[0]) - 4), scale=0.38)

    _draw_mask_outline(canvas, disc, _COLOR["disc_outline"], thickness=1)

    # Colorbar
    cb_w, cb_h = 180, 14
    strip = _colorbar_strip(cb_w, cb_h, "rdylgn")
    h, w = canvas.shape[:2]
    x0, y0 = w - cb_w - 12, h - 36
    canvas[y0:y0 + cb_h, x0:x0 + cb_w] = strip
    _put_text_with_bg(canvas, f"DF={df_min:.2f}", (x0 - 2, y0 + cb_h + 12), scale=0.4)
    _put_text_with_bg(canvas, f"DF≥{df_max:.2f}",
                      (x0 + cb_w - 60, y0 + cb_h + 12), scale=0.4)

    ta = analysis["tortuosity"]["artery"]
    tv = analysis["tortuosity"]["vein"]
    _text_block(canvas, [
        "Tortuosity (distance factor)",
        f"artery: mean={ta['mean_df']:.3f}  med={ta['median_df']:.3f}  n={ta['n_segments']}",
        f"vein  : mean={tv['mean_df']:.3f}  med={tv['median_df']:.3f}  n={tv['n_segments']}",
        f"min_segment: {min_px} px ({ta['min_segment_length_dd']:.2f} DD)",
        "chord drawn as gray line; label = DF",
    ], anchor="tl")
    return canvas


def _matplotlib_to_bgr(fig) -> np.ndarray:
    """Render a matplotlib Figure to a BGR uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _panel_bg_color(img: np.ndarray) -> Tuple[int, int, int]:
    """Pick a reasonable padding background from the panel's four corners.

    Median of the four corner pixels — works for both dark image-based panels
    (fundus base is black in the corners) and light matplotlib panels (white
    margins).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return (0, 0, 0)
    corners = np.stack([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
    med = np.median(corners, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _pad_to_aspect(img: np.ndarray, target_aspect_wh: float,
                   bg: Tuple[int, int, int]) -> np.ndarray:
    """Letterbox the panel to a target width/height ratio using ``bg`` as fill.

    Content is never cropped or scaled; only the padded borders change. This
    is used when composing panels of different native aspect ratios into the
    uniform cells of the combined mosaic, so each panel fills its cell
    without distortion.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    current = w / h
    if abs(current - target_aspect_wh) < 1e-3:
        return img
    if current < target_aspect_wh:
        new_w = int(round(h * target_aspect_wh))
        pad = max(0, new_w - w)
        left = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=bg)
    new_h = int(round(w / target_aspect_wh))
    pad = max(0, new_h - h)
    top = pad // 2
    bot = pad - top
    return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=bg)


def render_fractal(
    image_bgr: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    disc: np.ndarray,
    analysis: Dict[str, Any],
    config: VesselVizConfig,
) -> np.ndarray:
    """F: skeletons (off-disc) + log–log box-counting fit.

    Composition: left = dark canvas with coloured skeletons and grid overlay,
    right = matplotlib log-log plot. The two are concatenated horizontally.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = artery.shape
    disc_bin = (disc > 0)
    skel_a = skeletonize(artery.astype(bool)) & ~disc_bin
    skel_v = skeletonize(vein.astype(bool)) & ~disc_bin

    # Left panel: dark background + skeletons
    left = np.zeros((h, w, 3), dtype=np.uint8)
    left[skel_v] = _COLOR["vein_bright"]
    left[skel_a] = _COLOR["artery_bright"]

    # Overlay a box-counting grid to illustrate the technique
    box_sizes = analysis["fractal_dimension"]["artery"].get("box_sizes_px") or []
    if box_sizes:
        if config.fractal_subgrid_size is not None:
            grid_size = int(config.fractal_subgrid_size)
        else:
            # pick a representative middle box size
            grid_size = int(box_sizes[len(box_sizes) // 2])
        for y in range(0, h, grid_size):
            cv2.line(left, (0, y), (w, y), (60, 60, 60), 1)
        for x in range(0, w, grid_size):
            cv2.line(left, (x, 0), (x, h), (60, 60, 60), 1)
        # Highlight boxes containing skeleton
        combined_skel = skel_a | skel_v
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                if combined_skel[y:y + grid_size, x:x + grid_size].any():
                    cv2.rectangle(left, (x, y),
                                  (min(x + grid_size, w - 1), min(y + grid_size, h - 1)),
                                  (120, 120, 120), 1)
        _text_block(left, [
            "Skeleton + box grid",
            f"grid: {grid_size} px",
            "artery=red  vein=blue",
        ], anchor="tl")
    else:
        _text_block(left, ["Skeleton (no grid overlay)"], anchor="tl")

    # Right panel: log-log plot via matplotlib
    fd_a = analysis["fractal_dimension"]["artery"]
    fd_v = analysis["fractal_dimension"]["vein"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=config.dpi)

    def _plot_series(info: Dict[str, Any], color: str, label: str) -> None:
        sizes = np.array(info.get("box_sizes_px") or [], dtype=float)
        counts = np.array(info.get("box_counts") or [], dtype=float)
        mask = counts > 0
        if not mask.any():
            return
        x = np.log(1.0 / sizes[mask])
        y = np.log(counts[mask])
        ax.scatter(x, y, c=color, s=28, label=f"{label}  D={info['value']:.3f} R²={info['r2']:.3f}")
        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, coef[0] * xs + coef[1], color=color, linewidth=1, alpha=0.7)

    _plot_series(fd_a, "tab:red", "artery")
    _plot_series(fd_v, "tab:blue", "vein")
    ax.set_xlabel("log(1 / box_size)")
    ax.set_ylabel("log(N(box_size))")
    ax.set_title("Box-counting fractal fit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    right = _matplotlib_to_bgr(fig)
    plt.close(fig)

    # Resize right panel to match left height
    rh, rw = right.shape[:2]
    target_h = h
    target_w = int(rw * target_h / rh)
    right_resized = cv2.resize(right, (target_w, target_h))

    canvas = np.hstack([left, right_resized])
    return canvas


def render_density(
    image_bgr: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    disc: np.ndarray,
    fundus: Optional[np.ndarray],
    analysis: Dict[str, Any],
    config: VesselVizConfig,
) -> np.ndarray:
    """G: fundus outline + vessel area breakdown as both overlay and bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas = _to_bgr_uint8(image_bgr).copy()
    canvas = cv2.cvtColor(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    canvas = (canvas * 0.55).astype(np.uint8)

    canvas = _blend_overlay(canvas, _COLOR["vein_bright"], vein, 0.55)
    canvas = _blend_overlay(canvas, _COLOR["artery_bright"], artery, 0.55)
    if fundus is not None:
        _draw_mask_outline(canvas, fundus, _COLOR["fundus_outline"], thickness=1)
    _draw_mask_outline(canvas, disc, _COLOR["disc_outline"], thickness=1)

    d = analysis["density"]
    _text_block(canvas, [
        "Density (ratio to fundus)",
        f"artery: {d['artery_ratio']*100:.2f} %",
        f"vein  : {d['vein_ratio']*100:.2f} %",
        f"total : {d['total_ratio']*100:.2f} %",
        f"fundus area: {d['fundus_area_px']} px ({d['denominator_source']})",
    ], anchor="tl")

    # Bottom-right bar chart via matplotlib (small inset)
    fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=config.dpi)
    labels = ["artery", "vein", "total"]
    values = [d["artery_ratio"] * 100, d["vein_ratio"] * 100, d["total_ratio"] * 100]
    colors = ["tab:red", "tab:blue", "tab:gray"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("%")
    ax.set_title("vessel coverage", fontsize=9)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(8)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    chart = _matplotlib_to_bgr(fig)
    plt.close(fig)

    ch, cw = chart.shape[:2]
    target_w = min(cw, canvas.shape[1] // 3)
    target_h = int(ch * target_w / cw)
    chart_resized = cv2.resize(chart, (target_w, target_h))
    h_c, w_c = canvas.shape[:2]
    y0 = h_c - target_h - 12
    x0 = w_c - target_w - 12
    canvas[y0:y0 + target_h, x0:x0 + target_w] = chart_resized
    return canvas


def render_combined_panel(panels: Dict[str, np.ndarray],
                          analysis: Dict[str, Any],
                          config: VesselVizConfig) -> np.ndarray:
    """Assemble the individual panels into a 2×3 mosaic with a title bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["overview", "crae_crve", "diameter_heatmap",
             "tortuosity", "fractal", "density"]
    titles = {
        "overview": "Overview",
        "crae_crve": "CRAE / CRVE (top-N)",
        "diameter_heatmap": "Diameter heatmap",
        "tortuosity": "Tortuosity (DF)",
        "fractal": "Fractal dimension",
        "density": "Vessel density",
    }

    available = [k for k in order if k in panels]
    rows = 2
    cols = 3
    cell_w_in, cell_h_in = 5.0, 4.0
    cell_aspect = cell_w_in / cell_h_in  # matches the subplot rectangle
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * cell_w_in, rows * cell_h_in),
                             dpi=config.dpi)
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    # Letterbox each panel to the cell aspect so every one fills its subplot
    # without distortion. Panels with unusual native aspects (e.g. fractal's
    # skeleton+logplot is ~2:1 wide) get letterboxed vertically with their
    # own background colour; square panels get horizontal letterboxing.
    for ax, key in zip(axes, available):
        panel = panels[key]
        bg = _panel_bg_color(panel)
        padded = _pad_to_aspect(panel, cell_aspect, bg)
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(titles.get(key, key), fontsize=11)
        ax.axis("off")

    if config.combined_title:
        crae_px = analysis["crae"]["value_px"]
        crae_dd = analysis["crae"]["value_dd"]
        crve_px = analysis["crve"]["value_px"]
        crve_dd = analysis["crve"]["value_dd"]
        avr = analysis["avr"]["value"] or 0.0
        fda = analysis["fractal_dimension"]["artery"]["value"]
        fdv = analysis["fractal_dimension"]["vein"]["value"]
        title = (
            f"Vessel analysis  |  CRAE={crae_px:.2f}px ({crae_dd:.4f} DD)  "
            f"CRVE={crve_px:.2f}px ({crve_dd:.4f} DD)  AVR={avr:.4f}  "
            f"FD(a/v)={fda:.3f}/{fdv:.3f}"
        )
        fig.suptitle(title, fontsize=12, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if config.combined_title else None)
    out = _matplotlib_to_bgr(fig)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- main entry

def render_vessel_visualizations(
    original_image_bgr: np.ndarray,
    artery_mask: np.ndarray,
    vein_mask: np.ndarray,
    disc_mask: np.ndarray,
    analysis_result: Dict[str, Any],
    output_dir: str,
    fundus_mask: Optional[np.ndarray] = None,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    config: Optional[VesselVizConfig] = None,
) -> Dict[str, str]:
    """Render all configured panels + the combined mosaic, save as PNGs.

    Returns a dict mapping panel name → written file path.
    """
    cfg = config or VesselVizConfig()
    os.makedirs(output_dir, exist_ok=True)

    # If the user asked for the 2×3 mosaic but disabled some base panels, the
    # resulting combined image would be a partial mosaic with blank cells —
    # which has burned people in practice (they regenerate a subset, forget
    # it overwrote the combined, and later think "why is it half empty?").
    # Warn explicitly so the surprise doesn't get silently baked onto disk.
    _BASE_PANELS = ("overview", "crae_crve", "diameter_heatmap",
                    "tortuosity", "fractal", "density")
    if cfg.combined:
        missing = [p for p in _BASE_PANELS if not getattr(cfg, p)]
        if missing:
            warnings.warn(
                "render_vessel_visualizations: combined=True but these base "
                f"panels are disabled: {missing}. The combined mosaic will "
                "contain blank cells for them.",
                stacklevel=2,
            )

    artery_bin = (artery_mask > 0).astype(np.uint8)
    vein_bin = (vein_mask > 0).astype(np.uint8)
    disc_bin = (disc_mask > 0).astype(np.uint8)
    fundus_bin = (fundus_mask > 0).astype(np.uint8) if fundus_mask is not None else None

    # Build Zone B mask once (for segment extraction)
    disc_geom = compute_disc_geometry(disc_bin)
    zone_inner = float(analysis_result["measurement_zone"]["inner_dd"])
    zone_outer = float(analysis_result["measurement_zone"]["outer_dd"])
    zone = build_zone_b(artery_bin.shape, disc_geom, zone_inner, zone_outer)

    artery_zone_segments = extract_vessel_segments(artery_bin, disc_bin, zone)
    vein_zone_segments = extract_vessel_segments(vein_bin, disc_bin, zone)
    artery_all_segments = extract_all_skeleton_segments(artery_bin, disc_bin)
    vein_all_segments = extract_all_skeleton_segments(vein_bin, disc_bin)

    image_bgr = _to_bgr_uint8(original_image_bgr)
    panels: Dict[str, np.ndarray] = {}

    if cfg.overview:
        panels["overview"] = render_overview(
            image_bgr, artery_bin, vein_bin, disc_bin, fundus_bin,
            analysis_result, cfg, macula_yx=macula_center_yx,
        )
    if cfg.crae_crve:
        panels["crae_crve"] = render_crae_crve(
            image_bgr, artery_bin, vein_bin, disc_bin,
            analysis_result, artery_zone_segments, vein_zone_segments, cfg,
        )
    if cfg.diameter_heatmap:
        panels["diameter_heatmap"] = render_diameter_heatmap(
            image_bgr, artery_bin, vein_bin, disc_bin, fundus_bin,
            analysis_result, cfg,
        )
    if cfg.tortuosity:
        panels["tortuosity"] = render_tortuosity(
            image_bgr, artery_bin, vein_bin, disc_bin,
            analysis_result, artery_all_segments, vein_all_segments, cfg,
        )
    if cfg.fractal:
        panels["fractal"] = render_fractal(
            image_bgr, artery_bin, vein_bin, disc_bin, analysis_result, cfg,
        )
    if cfg.density:
        panels["density"] = render_density(
            image_bgr, artery_bin, vein_bin, disc_bin, fundus_bin,
            analysis_result, cfg,
        )

    written: Dict[str, str] = {}
    for name, panel in panels.items():
        path = os.path.join(output_dir, f"vessels_{name}.png")
        cv2.imwrite(path, panel)
        written[name] = path

    if cfg.combined and panels:
        combined = render_combined_panel(panels, analysis_result, cfg)
        path = os.path.join(output_dir, "vessels_combined.png")
        cv2.imwrite(path, combined)
        written["combined"] = path

    return written
