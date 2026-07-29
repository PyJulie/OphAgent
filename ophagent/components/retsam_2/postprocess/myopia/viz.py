"""
Myopia analysis visualizations.

Panels:

    * overview         — fundus with all 3 classes in their own colours
    * arc_profile      — polar plot of arc_lesion angular coverage around the disc
    * atrophy_map      — diffuse + patchy highlighted + disc/macula markers + rings
    * macula_zones     — 1/2 DD concentric rings with atrophy overlay and counts
    * burden           — per-class coverage / count bar charts
    * components_map   — patchy atrophy centroids sized by area
    * combined         — 2×3 mosaic
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


MYOPIA_COLORS: Dict[str, Tuple[int, int, int]] = {
    "arc_lesion": (0, 200, 255),                        # yellow-orange crescent
    "diffuse_chorioretinal_atrophy": (180, 100, 160),    # muted purple
    "patchy_chorioretinal_atrophy": (255, 120, 60),      # saturated orange-blue
}


def _bgr_to_mpl_rgb(color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert a (0-255) BGR tuple into matplotlib's (0-1) RGB."""
    b, g, r = color_bgr
    return (r / 255.0, g / 255.0, b / 255.0)


@dataclass
class MyopiaVizConfig:
    overview: bool = True
    arc_profile: bool = True
    atrophy_map: bool = True
    macula_zones: bool = True
    burden: bool = True
    components_map: bool = True
    combined: bool = True

    alpha: float = 0.55
    dim_factor: float = 0.45
    dpi: int = 150
    combined_title: bool = True


# --------------------------------------------------------------------------- helpers (duplicated so module stays self-contained)

def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _blend_overlay(base: np.ndarray, color: Tuple[int, int, int],
                   mask: np.ndarray, alpha: float) -> np.ndarray:
    out = base.copy()
    m = mask.astype(bool)
    if not m.any():
        return out
    colour = np.array(color, dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + colour * alpha).astype(np.uint8)
    return out


def _draw_mask_outline(canvas: np.ndarray, mask: np.ndarray,
                       color: Tuple[int, int, int], thickness: int = 2) -> None:
    if mask is None or mask.size == 0:
        return
    bm = (mask > 0).astype(np.uint8)
    if bm.sum() == 0:
        return
    contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness=thickness, lineType=cv2.LINE_AA)


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
                      scale: float = 0.55, color=(255, 255, 255),
                      bg=(0, 0, 0), thickness: int = 1, pad: int = 3) -> None:
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - pad, y - h - pad - 1),
                  (x + w + pad, y + baseline + pad), bg, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, _ascii_safe(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _text_block(canvas: np.ndarray, lines: Sequence[str], anchor: str = "tl",
                scale: float = 0.5, pad: int = 6, line_spacing: int = 6) -> None:
    h, w = canvas.shape[:2]
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for line in lines]
    box_w = max((tw for tw, _ in sizes), default=0) + 2 * pad
    box_h = sum(th for _, th in sizes) + line_spacing * max(0, len(lines) - 1) + 2 * pad
    x0, y0 = (8, 8) if anchor == "tl" else \
             (w - box_w - 8, 8) if anchor == "tr" else \
             (8, h - box_h - 8) if anchor == "bl" else (w - box_w - 8, h - box_h - 8)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    y = y0 + pad
    for line, (tw, th) in zip(lines, sizes):
        y += th
        cv2.putText(canvas, _ascii_safe(line), (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_spacing


def _matplotlib_to_bgr(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)


def _panel_bg_color(img: np.ndarray) -> Tuple[int, int, int]:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return (0, 0, 0)
    corners = np.stack([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
    med = np.median(corners, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _pad_to_aspect(img: np.ndarray, target_aspect_wh: float,
                   bg: Tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    current = w / h
    if abs(current - target_aspect_wh) < 1e-3:
        return img
    if current < target_aspect_wh:
        new_w = int(round(h * target_aspect_wh))
        pad = max(0, new_w - w); l = pad // 2; r = pad - l
        return cv2.copyMakeBorder(img, 0, 0, l, r, cv2.BORDER_CONSTANT, value=bg)
    new_h = int(round(w / target_aspect_wh))
    pad = max(0, new_h - h); t = pad // 2; b = pad - t
    return cv2.copyMakeBorder(img, t, b, 0, 0, cv2.BORDER_CONSTANT, value=bg)


def _dim(img: np.ndarray, factor: float) -> np.ndarray:
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- panels

def render_overview(image_bgr: np.ndarray, myopia_mask: np.ndarray,
                    analysis: Dict[str, Any], disc_mask: Optional[np.ndarray],
                    macula_yx: Optional[Tuple[float, float]],
                    config: MyopiaVizConfig) -> np.ndarray:
    canvas = _dim(_to_bgr_uint8(image_bgr), config.dim_factor)
    for cls_id, cls_name in ((1, "arc_lesion"),
                             (2, "diffuse_chorioretinal_atrophy"),
                             (3, "patchy_chorioretinal_atrophy")):
        bin_mask = (myopia_mask == cls_id).astype(np.uint8)
        canvas = _blend_overlay(canvas, MYOPIA_COLORS[cls_name], bin_mask, config.alpha)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), 2)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    cls = analysis["classes"]
    _text_block(canvas, [
        "Myopia overview",
        f"arc_lesion:  count={cls['arc_lesion']['count']}  coverage={cls['arc_lesion']['coverage_ratio']*100:.3f}%",
        f"diffuse:     count={cls['diffuse_chorioretinal_atrophy']['count']}  coverage={cls['diffuse_chorioretinal_atrophy']['coverage_ratio']*100:.3f}%",
        f"patchy:      count={cls['patchy_chorioretinal_atrophy']['count']}  coverage={cls['patchy_chorioretinal_atrophy']['coverage_ratio']*100:.3f}%",
    ], anchor="tl")
    return canvas


def _render_arc_thumbnail(image_bgr: np.ndarray, arc_mask: np.ndarray,
                          disc_mask: Optional[np.ndarray],
                          analysis: Dict[str, Any],
                          config: MyopiaVizConfig) -> np.ndarray:
    """Disc-centered cartesian thumbnail showing the actual arc_lesion pixels.

    A ~3 DD × 3 DD window centred on the disc, with arc mask overlaid, disc
    outline drawn, and I/S/N/T (or I/S/east/west) arrows pointing outward.
    """
    h, w = image_bgr.shape[:2]
    dcy, dcx = analysis["disc"]["center_yx"]
    DD = float(analysis["units"]["disc_diameter_px"])
    win = int(round(1.75 * DD))
    y0 = max(0, int(dcy) - win); y1 = min(h, int(dcy) + win)
    x0 = max(0, int(dcx) - win); x1 = min(w, int(dcx) + win)
    crop = _to_bgr_uint8(image_bgr)[y0:y1, x0:x1].copy()
    crop = _dim(crop, 0.45)

    arc_crop = arc_mask[y0:y1, x0:x1]
    crop = _blend_overlay(crop, MYOPIA_COLORS["arc_lesion"], arc_crop, 0.7)

    if disc_mask is not None:
        disc_crop = disc_mask[y0:y1, x0:x1]
        _draw_mask_outline(crop, disc_crop, (0, 255, 0), thickness=2)

    # Local disc centre in the crop
    lcy = int(dcy) - y0
    lcx = int(dcx) - x0
    cv2.drawMarker(crop, (lcx, lcy), (255, 255, 255),
                   cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # 4 cardinal labels pointing OUTWARD from the disc centre
    eye_side = analysis["rim"]["isnt"].get("eye_side") if "rim" in analysis else None
    # In myopia analysis dict, eye_side is in qc not rim. Fall back to qc.
    if eye_side is None:
        eye_side_src = analysis.get("qc", {}).get("eye_side_source")
        if eye_side_src == "user_provided":
            # look in severity_inputs → arc_sector_involvement for known side
            si = analysis.get("severity_inputs", {})
            arc_si = (analysis["classes"]["arc_lesion"]["disc_relative"]
                      ["sector_involvement"])
            if "nasal" in arc_si and arc_si.get("nasal") is not None:
                # eye_side is known inside arc_si — we can't recover it here,
                # so leave eye_side None and label east/west
                pass

    # Fallback: read eye_side from the analysis.qc.eye_side_source is not enough;
    # we actually need the label. We look up whether nasal/temporal are not None
    # in sector_involvement — if they are None → unknown.
    arc_si = analysis["classes"]["arc_lesion"].get("disc_relative", {}).get(
        "sector_involvement", {}
    )
    side_known = arc_si.get("nasal") is not None

    # Arrow length: halfway to crop edge, measured from disc centre
    arrow_r = max(40, min(crop.shape[:2]) // 2 - 20)

    def _arrow(label: str, angle_deg: float, color=(255, 255, 0)):
        theta = np.deg2rad(angle_deg)
        dx = np.cos(theta); dy = np.sin(theta)
        tip = (int(lcx + dx * arrow_r), int(lcy + dy * arrow_r))
        cv2.arrowedLine(crop, (lcx, lcy), tip, color, 1, cv2.LINE_AA, tipLength=0.12)
        _put_text_with_bg(crop, label,
                          (tip[0] + int(dx * 6) - 6, tip[1] + int(dy * 6) + 4),
                          scale=0.55, color=color)

    # In image coords: 90° = inferior (down), 270° = superior (up)
    _arrow("I", 90)
    _arrow("S", 270)
    if side_known:
        if arc_si.get("nasal") is not None:
            # we need to know which side is nasal — infer from sector_involvement keys
            # severity_inputs carries it directly
            pass
        # eye_side isn't directly available here; look in severity_inputs
        sev = analysis.get("severity_inputs", {})
        # We actually need the literal OS/OD label. Read it off the sector
        # involvement block by checking whether east_raw or west_raw is nasal.
        # Simpler: infer from sector booleans — if nasal True while east_raw True → OS.
        arc_si_full = analysis["classes"]["arc_lesion"]["disc_relative"]["sector_involvement"]
        is_os = (arc_si_full.get("east_raw") is True
                 and arc_si_full.get("nasal") is True)
        is_od = (arc_si_full.get("west_raw") is True
                 and arc_si_full.get("nasal") is True)
        if is_os:
            _arrow("N", 0); _arrow("T", 180)
        elif is_od:
            _arrow("T", 0); _arrow("N", 180)
        else:
            # presence pattern inconclusive — fall back to east/west
            _arrow("E", 0); _arrow("W", 180)
    else:
        _arrow("E", 0); _arrow("W", 180)

    disc_rel = analysis["classes"]["arc_lesion"].get("disc_relative", {})
    coverage = disc_rel.get("angular_coverage_deg", 0)
    max_ext = disc_rel.get("max_radial_extent_dd", 0.0)
    mean_ext = disc_rel.get("mean_radial_extent_dd", 0.0)
    _text_block(crop, [
        "Arc lesion (disc close-up)",
        f"angular coverage: {coverage}°",
        f"max extent: {max_ext:.3f} DD",
        f"mean extent: {mean_ext:.3f} DD",
        f"arrows: disc→outward along I/S/"
        f"{'N/T' if side_known else 'E/W'}",
    ], anchor="tl", scale=0.42)
    return crop


def _render_arc_polar_bars(analysis: Dict[str, Any],
                           config: MyopiaVizConfig) -> np.ndarray:
    """Polar bar plot of per-degree radial extent around the disc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arc = analysis["classes"]["arc_lesion"]
    disc_rel = arc.get("disc_relative", {})
    profile = disc_rel.get("angular_profile") or []

    fig = plt.figure(figsize=(5.5, 5.5), dpi=config.dpi)
    ax = fig.add_subplot(111, projection="polar")
    # Image convention: 0° = east (+x), 90° = south (+y, inferior),
    # 180° = west, 270° = north (superior). Matplotlib defaults place 0° east
    # and go counter-clockwise; we invert direction so angles grow CW like
    # image y grows down.
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    if arc["count"] == 0 or not profile:
        ax.text(0, 0, "no arc_lesion detected", ha="center", va="center",
                fontsize=12, transform=ax.transData)
        ax.set_yticks([]); ax.set_xticks([])
        out = _matplotlib_to_bgr(fig); plt.close(fig); return out

    angles_deg = np.array([p[0] for p in profile], dtype=np.float64)
    extents = np.array([p[1] for p in profile], dtype=np.float64)
    theta = np.deg2rad(angles_deg)
    width = 2 * np.pi / len(angles_deg)

    # Light grey shading for I / S wedges (eye-side-independent)
    rmax_data = float(max(extents.max(), 0.05))
    rmax = max(rmax_data * 1.25, 0.2)
    for a0, a1, lab in [(45, 135, "I"), (225, 315, "S")]:
        ts = np.deg2rad(np.arange(a0, a1 + 1))
        ax.fill_between(ts, np.zeros_like(ts, dtype=float),
                        np.full_like(ts, rmax, dtype=float),
                        color="#dddddd", alpha=0.45, zorder=0)

    # N/T wedges if eye_side known, else east/west
    si = disc_rel.get("sector_involvement", {})
    side_known = si.get("nasal") is not None
    if side_known:
        # Determine which side is nasal by looking at east_raw/west_raw + nasal
        is_os = (si.get("east_raw") is True and si.get("nasal") is True)
        # Fall back to using the fact that nasal depends on eye_side: if nasal
        # is True and east_raw is True → OS, if nasal True and west_raw True → OD
        nasal_at_east = bool(si.get("east_raw")) and si.get("nasal") is True
        nasal_at_west = bool(si.get("west_raw")) and si.get("nasal") is True
        if nasal_at_east or (not nasal_at_west and not nasal_at_east):
            # default or OS: nasal east, temporal west
            lab_east, lab_west = "N", "T"
        else:
            lab_east, lab_west = "T", "N"
    else:
        lab_east, lab_west = "E", "W"

    # Light wedge shading for east/west too
    for a0, a1 in [(315, 405), (135, 225)]:  # 315..45 (wrap) + 135..225
        ts = (np.deg2rad(np.arange(a0, a1 + 1)) if a1 <= 360
              else np.deg2rad(np.concatenate([np.arange(a0, 360), np.arange(0, a1 - 360 + 1)])))
        ax.fill_between(ts, np.zeros_like(ts, dtype=float),
                        np.full_like(ts, rmax, dtype=float),
                        color="#eeeeee", alpha=0.3, zorder=0)

    # The actual per-degree data as polar bars
    mpl_color = _bgr_to_mpl_rgb(MYOPIA_COLORS["arc_lesion"])
    ax.bar(theta, extents, width=width, bottom=0.0, color=mpl_color,
           edgecolor=mpl_color, alpha=0.9, zorder=3)

    # Cardinal labels at rmax
    ax.text(np.deg2rad(90), rmax, "I", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#333")
    ax.text(np.deg2rad(270), rmax, "S", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#333")
    ax.text(np.deg2rad(0), rmax, lab_east, ha="left", va="center",
            fontsize=12, fontweight="bold", color="#333")
    ax.text(np.deg2rad(180), rmax, lab_west, ha="right", va="center",
            fontsize=12, fontweight="bold", color="#333")

    ax.set_rmax(rmax)
    ax.set_rlabel_position(135)
    ax.set_yticks([v for v in np.linspace(0, rmax, 4)[1:]])
    ax.set_yticklabels([f"{v:.2f} DD" for v in np.linspace(0, rmax, 4)[1:]],
                       fontsize=8)
    ax.grid(alpha=0.35)
    ax.set_title(
        f"Arc radial extent vs angle  (coverage={disc_rel.get('angular_coverage_deg', 0)}°)",
        pad=14, fontsize=11,
    )
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


def render_arc_profile(image_bgr: np.ndarray, arc_mask: np.ndarray,
                       disc_mask: Optional[np.ndarray],
                       analysis: Dict[str, Any],
                       config: MyopiaVizConfig) -> np.ndarray:
    """Left: disc-centred cartesian thumbnail showing the actual arc pixels.
    Right: polar bar plot of per-degree radial extent. Concatenated horizontally.

    If the arc lesion is not present, both panels still render (with a clear
    "no arc" placeholder on the polar side) so the mosaic layout stays stable.
    """
    left = _render_arc_thumbnail(image_bgr, arc_mask, disc_mask, analysis, config)
    right = _render_arc_polar_bars(analysis, config)

    # Align heights for side-by-side concat
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    target_h = max(h_left, h_right)

    def _pad_height(img: np.ndarray, th: int) -> np.ndarray:
        if img.shape[0] >= th:
            return img
        pad = th - img.shape[0]
        return cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0,
                                  cv2.BORDER_CONSTANT, value=_panel_bg_color(img))

    left_p = _pad_height(left, target_h)
    right_p = _pad_height(right, target_h)
    return np.hstack([left_p, right_p])


def render_atrophy_map(image_bgr: np.ndarray, myopia_mask: np.ndarray,
                       analysis: Dict[str, Any], disc_mask: Optional[np.ndarray],
                       macula_yx: Optional[Tuple[float, float]],
                       config: MyopiaVizConfig) -> np.ndarray:
    """Diffuse + patchy highlighted; no arc (arc lives in its own polar panel)."""
    canvas = _dim(_to_bgr_uint8(image_bgr), 0.35)
    diffuse = (myopia_mask == 2).astype(np.uint8)
    patchy = (myopia_mask == 3).astype(np.uint8)
    canvas = _blend_overlay(canvas, MYOPIA_COLORS["diffuse_chorioretinal_atrophy"],
                            diffuse, config.alpha)
    canvas = _blend_overlay(canvas, MYOPIA_COLORS["patchy_chorioretinal_atrophy"],
                            patchy, config.alpha)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), 2)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    cls = analysis["classes"]
    d = cls["diffuse_chorioretinal_atrophy"]
    p = cls["patchy_chorioretinal_atrophy"]
    lines = [
        "Atrophy map",
        f"diffuse:  coverage={d['coverage_ratio']*100:.3f}%  involves_macula={d['involves_macula']}",
        f"patchy:   count={p['count']}  involves_macula={p['involves_macula']}",
    ]
    _text_block(canvas, lines, anchor="tl")
    return canvas


def render_macula_zones(image_bgr: np.ndarray, myopia_mask: np.ndarray,
                        analysis: Dict[str, Any],
                        disc_mask: Optional[np.ndarray],
                        macula_yx: Optional[Tuple[float, float]],
                        config: MyopiaVizConfig) -> np.ndarray:
    canvas = _dim(_to_bgr_uint8(image_bgr), 0.45)
    # Overlay all 3 classes
    for cls_id, cls_name in ((1, "arc_lesion"),
                             (2, "diffuse_chorioretinal_atrophy"),
                             (3, "patchy_chorioretinal_atrophy")):
        bin_mask = (myopia_mask == cls_id).astype(np.uint8)
        canvas = _blend_overlay(canvas, MYOPIA_COLORS[cls_name], bin_mask, config.alpha)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), 2)

    DD = float(analysis["units"]["disc_diameter_px"])
    if macula_yx is None or DD <= 0:
        _text_block(canvas, ["Macula zones",
                             "macula_center not provided → zones omitted"], anchor="tl")
        return canvas

    my, mx = int(macula_yx[0]), int(macula_yx[1])
    for b in (1.0, 2.0):
        r = int(round(b * DD))
        cv2.circle(canvas, (mx, my), r, (255, 255, 255), 1, cv2.LINE_AA)
        _put_text_with_bg(canvas, f"{b:g} DD", (mx + r + 4, my - 6), scale=0.42)
    cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                   cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    # Aggregate zone counts across classes (from 'patchy' which is typically
    # componentised; diffuse is continuous so component counts are 1 anyway).
    lines = ["Macula-centred zones"]
    zone_agg: Dict[str, int] = {}
    for cls in analysis["classes"].values():
        zones = cls["spatial"].get("macula_zone_counts")
        if isinstance(zones, dict):
            for k, v in zones.items():
                zone_agg[k] = zone_agg.get(k, 0) + int(v)
    for k, v in zone_agg.items():
        lines.append(f"{k}: {v}")
    _text_block(canvas, lines, anchor="tl")
    return canvas


def render_burden(analysis: Dict[str, Any], config: MyopiaVizConfig) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["arc_lesion", "diffuse_chorioretinal_atrophy", "patchy_chorioretinal_atrophy"]
    counts = [analysis["classes"][n]["count"] for n in names]
    coverages = [analysis["classes"][n]["coverage_ratio"] * 100 for n in names]
    colors = [_bgr_to_mpl_rgb(MYOPIA_COLORS[n]) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), dpi=config.dpi)
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, counts, color=colors)
    ax1.set_yticks(y_pos, [n.replace("_chorioretinal_atrophy", "") for n in names])
    ax1.invert_yaxis()
    ax1.set_xlabel("count")
    ax1.set_title("Count per class")
    for i, v in enumerate(counts):
        ax1.text(v, i, f" {v}", va="center", fontsize=9)

    ax2.barh(y_pos, coverages, color=colors)
    ax2.set_yticks(y_pos, ["" for _ in names])
    ax2.set_xlabel("coverage %")
    ax2.set_title("Fundus coverage per class")
    for i, v in enumerate(coverages):
        ax2.text(v, i, f" {v:.3f}%", va="center", fontsize=9)

    fig.tight_layout()
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


def render_components_map(image_bgr: np.ndarray, analysis: Dict[str, Any],
                          disc_mask: Optional[np.ndarray],
                          macula_yx: Optional[Tuple[float, float]],
                          config: MyopiaVizConfig) -> np.ndarray:
    """Patchy atrophy centroids sized by area, overlaid on dim fundus."""
    canvas = _dim(_to_bgr_uint8(image_bgr), 0.65)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), 2)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    for cls_name in ("patchy_chorioretinal_atrophy", "arc_lesion"):
        comps = analysis["classes"][cls_name].get("components") or []
        color = MYOPIA_COLORS[cls_name]
        for comp in comps:
            y, x = comp["centroid_yx"]
            r = max(3, int(round(np.sqrt(comp["area_px"] / np.pi))))
            cv2.circle(canvas, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)

    n_patchy = analysis["classes"]["patchy_chorioretinal_atrophy"]["count"]
    n_arc = analysis["classes"]["arc_lesion"]["count"]
    _text_block(canvas, [
        "Components map",
        f"patchy: {n_patchy} components",
        f"arc: {n_arc} components (if split)",
        "(diffuse atrophy omitted — it's continuous, not component-like)",
    ], anchor="tl")
    return canvas


def render_combined_panel(panels: Dict[str, np.ndarray],
                          analysis: Dict[str, Any],
                          config: MyopiaVizConfig) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["overview", "arc_profile", "atrophy_map",
             "macula_zones", "burden", "components_map"]
    titles = {
        "overview": "Overview", "arc_profile": "Arc lesion",
        "atrophy_map": "Atrophy map", "macula_zones": "Macula zones",
        "burden": "Burden", "components_map": "Components",
    }
    rows, cols = 2, 3
    cell_w_in, cell_h_in = 5.0, 4.0
    cell_aspect = cell_w_in / cell_h_in
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * cell_w_in, rows * cell_h_in),
                             dpi=config.dpi)
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for ax, key in zip(axes, order):
        if key not in panels:
            continue
        panel = panels[key]
        bg = _panel_bg_color(panel)
        padded = _pad_to_aspect(panel, cell_aspect, bg)
        ax.imshow(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        ax.set_title(titles.get(key, key), fontsize=11)
        ax.axis("off")

    if config.combined_title:
        gs = analysis["global_summary"]
        sev = analysis["severity_inputs"]
        title = (
            f"Myopia analysis  |  classes: {gs['n_classes_detected']}  "
            f"coverage: {gs['total_coverage_ratio']*100:.3f}%  "
            f"arc: {sev['arc_angular_coverage_deg']}°  "
            f"diffuse: {'+' if sev['diffuse_atrophy_present'] else '-'}  "
            f"patchy: {sev['patchy_count']}"
        )
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if config.combined_title else None)
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


# --------------------------------------------------------------------------- main entry

def render_myopia_visualizations(
    original_image_bgr: np.ndarray,
    myopia_mask: np.ndarray,
    analysis_result: Dict[str, Any],
    output_dir: str,
    disc_mask: Optional[np.ndarray] = None,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    config: Optional[MyopiaVizConfig] = None,
) -> Dict[str, str]:
    cfg = config or MyopiaVizConfig()
    os.makedirs(output_dir, exist_ok=True)

    _BASE_PANELS = ("overview", "arc_profile", "atrophy_map",
                    "macula_zones", "burden", "components_map")
    if cfg.combined:
        missing = [p for p in _BASE_PANELS if not getattr(cfg, p)]
        if missing:
            warnings.warn(
                "render_myopia_visualizations: combined=True but these base "
                f"panels are disabled: {missing}. The combined mosaic will "
                "contain blank cells for them.",
                stacklevel=2,
            )

    image_bgr = _to_bgr_uint8(original_image_bgr)
    panels: Dict[str, np.ndarray] = {}
    if cfg.overview:
        panels["overview"] = render_overview(
            image_bgr, myopia_mask, analysis_result, disc_mask, macula_center_yx, cfg
        )
    if cfg.arc_profile:
        arc_mask = (myopia_mask == 1).astype(np.uint8)
        panels["arc_profile"] = render_arc_profile(
            image_bgr, arc_mask, disc_mask, analysis_result, cfg,
        )
    if cfg.atrophy_map:
        panels["atrophy_map"] = render_atrophy_map(
            image_bgr, myopia_mask, analysis_result, disc_mask, macula_center_yx, cfg
        )
    if cfg.macula_zones:
        panels["macula_zones"] = render_macula_zones(
            image_bgr, myopia_mask, analysis_result, disc_mask, macula_center_yx, cfg
        )
    if cfg.burden:
        panels["burden"] = render_burden(analysis_result, cfg)
    if cfg.components_map:
        panels["components_map"] = render_components_map(
            image_bgr, analysis_result, disc_mask, macula_center_yx, cfg
        )

    written: Dict[str, str] = {}
    for name, panel in panels.items():
        path = os.path.join(output_dir, f"myopia_{name}.png")
        cv2.imwrite(path, panel)
        written[name] = path
    if cfg.combined and panels:
        combined = render_combined_panel(panels, analysis_result, cfg)
        path = os.path.join(output_dir, "myopia_combined.png")
        cv2.imwrite(path, combined)
        written["combined"] = path
    return written
