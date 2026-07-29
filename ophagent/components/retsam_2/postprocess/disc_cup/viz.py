"""
Disc / cup analysis visualizations.

Panels (each independently toggleable via ``DiscCupVizConfig``):

    * overview      — original + disc + cup + centres + macula + eye-side tag
    * cdr           — close-up of the disc region with explicit vertical /
                      horizontal diameter lines overlaid, labelled with vCDR,
                      hCDR, area-CDR and rim-disc ratio
    * isnt          — disc/cup outlines + 4 cardinal rim-width rays (with the
                      disc / cup intersection pairs highlighted), rim-width
                      annotations, ISNT compliance status
    * rim_profile   — matplotlib polar plot of rim width vs angle with the
                      I/S/N/T sectors shaded and the minimum marked
    * tilt          — disc mask + ellipse fit overlay + major axis + tilt
                      angle and ovality legend
    * ddls          — small info card: disc-size bucket, rim-absent arc,
                      min rim/disc ratio, resulting DDLS grade and reasoning
    * combined      — 2×3 matplotlib mosaic of the above

All individual panels return BGR uint8 arrays so they can be saved with
cv2.imwrite or composed into the mosaic.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import from_od_oc_labelmap


# --------------------------------------------------------------------------- config

@dataclass
class DiscCupVizConfig:
    overview: bool = True
    cdr: bool = True
    isnt: bool = True
    rim_profile: bool = True
    tilt: bool = True
    ddls: bool = True
    combined: bool = True

    alpha_disc: float = 0.28
    alpha_cup: float = 0.35
    close_up_padding_dd: float = 0.6   # pad around disc in CDR close-up, in DD units
    dpi: int = 150
    combined_title: bool = True


# --------------------------------------------------------------------------- palette

_COLOR = {
    "disc":          (80, 255, 80),     # green
    "disc_outline":  (0, 220, 0),
    "cup":           (40, 200, 255),    # amber
    "cup_outline":   (0, 180, 255),
    "rim_outline":   (0, 255, 255),
    "disc_center":   (0, 255, 255),
    "cup_center":    (255, 255, 0),
    "macula":        (255, 255, 255),
    "vline":         (0, 255, 255),     # vertical CDR line
    "hline":         (255, 180, 0),     # horizontal CDR line
    "text_fg":       (255, 255, 255),
    "text_bg":       (0, 0, 0),
    "ray_disc":      (0, 255, 0),
    "ray_cup":       (0, 180, 255),
    "ray_rim":       (255, 255, 0),
    "fit_ellipse":   (255, 0, 255),
    "tilt_axis":     (255, 0, 255),
    "absent_arc":    (0, 0, 255),
}


# --------------------------------------------------------------------------- helpers

def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _blend_overlay(base: np.ndarray, color_bgr: Tuple[int, int, int],
                   mask: np.ndarray, alpha: float) -> np.ndarray:
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
    if mask is None or mask.size == 0:
        return
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                      scale: float = 0.55, color: Tuple[int, int, int] = (255, 255, 255),
                      bg: Tuple[int, int, int] = (0, 0, 0),
                      thickness: int = 1, pad: int = 3) -> None:
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - pad, y - h - pad - 1),
                  (x + w + pad, y + baseline + pad), bg, thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, _ascii_safe(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, lineType=cv2.LINE_AA)


def _text_block(canvas: np.ndarray, lines: Sequence[str],
                anchor: str = "tl", scale: float = 0.55,
                pad: int = 6, line_spacing: int = 6) -> None:
    h, w = canvas.shape[:2]
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for line in lines]
    box_w = max((tw for tw, _ in sizes), default=0) + 2 * pad
    box_h = sum(th for _, th in sizes) + line_spacing * (len(lines) - 1) + 2 * pad
    if anchor == "tl":
        x0, y0 = 8, 8
    elif anchor == "tr":
        x0, y0 = w - box_w - 8, 8
    elif anchor == "bl":
        x0, y0 = 8, h - box_h - 8
    else:
        x0, y0 = w - box_w - 8, h - box_h - 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    y = y0 + pad
    for line, (tw, th) in zip(lines, sizes):
        y += th
        cv2.putText(canvas, _ascii_safe(line), (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, _COLOR["text_fg"], 1, cv2.LINE_AA)
        y += line_spacing


def _matplotlib_to_bgr(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _panel_bg_color(img: np.ndarray) -> Tuple[int, int, int]:
    """Median of the four corner pixels; blends naturally into dark or light panels."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return (0, 0, 0)
    corners = np.stack([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
    med = np.median(corners, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _pad_to_aspect(img: np.ndarray, target_aspect_wh: float,
                   bg: Tuple[int, int, int]) -> np.ndarray:
    """Letterbox a panel to a target w/h ratio; content is never scaled or cropped."""
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


def _extent_ys_in_col(mask: np.ndarray, col: int) -> Optional[Tuple[int, int]]:
    if col < 0 or col >= mask.shape[1]:
        return None
    ys = np.nonzero(mask[:, col])[0]
    return (int(ys.min()), int(ys.max())) if ys.size else None


def _extent_xs_in_row(mask: np.ndarray, row: int) -> Optional[Tuple[int, int]]:
    if row < 0 or row >= mask.shape[0]:
        return None
    xs = np.nonzero(mask[row, :])[0]
    return (int(xs.min()), int(xs.max())) if xs.size else None


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# --------------------------------------------------------------------------- panels

def render_overview(image_bgr: np.ndarray, disc_bin: np.ndarray, cup_bin: np.ndarray,
                    analysis: Dict[str, Any], config: DiscCupVizConfig,
                    macula_yx: Optional[Tuple[float, float]] = None) -> np.ndarray:
    canvas = _to_bgr_uint8(image_bgr).copy()
    canvas = _blend_overlay(canvas, _COLOR["disc"], disc_bin, config.alpha_disc)
    canvas = _blend_overlay(canvas, _COLOR["cup"], cup_bin, config.alpha_cup)
    _draw_mask_outline(canvas, disc_bin, _COLOR["disc_outline"], thickness=2)
    _draw_mask_outline(canvas, cup_bin, _COLOR["cup_outline"], thickness=2)

    dcy, dcx = analysis["disc"]["center_yx"]
    cv2.drawMarker(canvas, (int(dcx), int(dcy)),
                   _COLOR["disc_center"], cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    if analysis.get("cup") is not None:
        ccy, ccx = analysis["cup"]["center_yx"]
        cv2.drawMarker(canvas, (int(ccx), int(ccy)),
                       _COLOR["cup_center"], cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), _COLOR["macula"], cv2.MARKER_TILTED_CROSS,
                       14, 2, cv2.LINE_AA)
        _put_text_with_bg(canvas, "macula", (mx + 8, my - 6), scale=0.4)

    eye = analysis["rim"]["isnt"].get("eye_side") if analysis.get("rim") else None
    eye_tag = eye if eye else "eye side: unknown"
    _text_block(canvas, [
        "Overview",
        f"disc area: {analysis['disc']['area_px']} px",
        f"DD (vertical): {analysis['units']['disc_diameter_px']:.1f} px",
        f"tilt: {analysis['disc']['tilt_angle_deg']:+.1f}°   ovality: {analysis['disc']['ovality']:.3f}",
        f"{eye_tag}",
    ], anchor="tl")
    return canvas


def render_cdr(image_bgr: np.ndarray, disc_bin: np.ndarray, cup_bin: np.ndarray,
               analysis: Dict[str, Any], config: DiscCupVizConfig) -> np.ndarray:
    """Close-up on disc region with explicit vCDR/hCDR measurement lines."""
    canvas = _to_bgr_uint8(image_bgr).copy()

    # Dim everything outside the disc ROI
    canvas = (canvas * 0.5).astype(np.uint8)
    disc_yx_yxmin = None
    if disc_bin.sum() == 0:
        return canvas
    ymin, ymax, xmin, xmax = _bbox(disc_bin > 0)
    DD = analysis["units"]["disc_diameter_px"]
    pad = int(config.close_up_padding_dd * DD)
    y0 = max(0, ymin - pad)
    y1 = min(canvas.shape[0], ymax + pad)
    x0 = max(0, xmin - pad)
    x1 = min(canvas.shape[1], xmax + pad)
    # Un-dim the ROI
    canvas[y0:y1, x0:x1] = _to_bgr_uint8(image_bgr)[y0:y1, x0:x1]

    canvas = _blend_overlay(canvas, _COLOR["disc"], disc_bin, config.alpha_disc)
    canvas = _blend_overlay(canvas, _COLOR["cup"], cup_bin, config.alpha_cup)
    _draw_mask_outline(canvas, disc_bin, _COLOR["disc_outline"], thickness=2)
    _draw_mask_outline(canvas, cup_bin, _COLOR["cup_outline"], thickness=2)

    # Vertical CDR line: through cup vertical-midline x, if cup exists; else disc center
    dcy, dcx = analysis["disc"]["center_yx"]
    if analysis.get("cup") is not None:
        ccy, ccx = analysis["cup"]["center_yx"]
    else:
        ccy, ccx = dcy, dcx

    disc_v = _extent_ys_in_col(disc_bin > 0, int(ccx))
    cup_v = _extent_ys_in_col(cup_bin > 0, int(ccx)) if cup_bin.sum() else None
    if disc_v is not None:
        cv2.line(canvas, (int(ccx), disc_v[0] - 3), (int(ccx), disc_v[1] + 3),
                 _COLOR["vline"], 2, cv2.LINE_AA)
    if cup_v is not None:
        cv2.line(canvas, (int(ccx) - 6, cup_v[0]), (int(ccx) + 6, cup_v[0]),
                 _COLOR["vline"], 2, cv2.LINE_AA)
        cv2.line(canvas, (int(ccx) - 6, cup_v[1]), (int(ccx) + 6, cup_v[1]),
                 _COLOR["vline"], 2, cv2.LINE_AA)

    disc_h = _extent_xs_in_row(disc_bin > 0, int(ccy))
    cup_h = _extent_xs_in_row(cup_bin > 0, int(ccy)) if cup_bin.sum() else None
    if disc_h is not None:
        cv2.line(canvas, (disc_h[0] - 3, int(ccy)), (disc_h[1] + 3, int(ccy)),
                 _COLOR["hline"], 2, cv2.LINE_AA)
    if cup_h is not None:
        cv2.line(canvas, (cup_h[0], int(ccy) - 6), (cup_h[0], int(ccy) + 6),
                 _COLOR["hline"], 2, cv2.LINE_AA)
        cv2.line(canvas, (cup_h[1], int(ccy) - 6), (cup_h[1], int(ccy) + 6),
                 _COLOR["hline"], 2, cv2.LINE_AA)

    cdr = analysis["cdr"]
    rim = analysis["rim"]
    _text_block(canvas, [
        "CDR",
        f"vCDR: {cdr['vertical']:.4f}",
        f"hCDR: {cdr['horizontal']:.4f}",
        f"aCDR: {cdr['area']:.4f}",
        f"rim/disc area: {rim['rim_disc_area_ratio']:.4f}",
    ], anchor="tl")
    return canvas


def _polar_ray_end(center_yx: Tuple[float, float], angle_deg: float, length: float) -> Tuple[int, int]:
    theta = np.deg2rad(angle_deg)
    return (int(round(center_yx[1] + length * np.cos(theta))),
            int(round(center_yx[0] + length * np.sin(theta))))


def render_isnt(image_bgr: np.ndarray, disc_bin: np.ndarray, cup_bin: np.ndarray,
                analysis: Dict[str, Any], config: DiscCupVizConfig) -> np.ndarray:
    canvas = _to_bgr_uint8(image_bgr).copy()
    canvas = (canvas * 0.55).astype(np.uint8)
    canvas = _blend_overlay(canvas, _COLOR["disc"], disc_bin, config.alpha_disc)
    canvas = _blend_overlay(canvas, _COLOR["cup"], cup_bin, config.alpha_cup)
    _draw_mask_outline(canvas, disc_bin, _COLOR["disc_outline"], thickness=2)
    _draw_mask_outline(canvas, cup_bin, _COLOR["cup_outline"], thickness=2)

    cardinal = analysis["rim"].get("cardinal_widths")
    dcy, dcx = analysis["disc"]["center_yx"]

    # Draw rays with width labels. Use the raw east/west fields so labels are
    # visible even when eye_side is unknown; also add I/S/N/T text when known.
    sides = [
        ("inferior",  90.0, "I", cardinal["inferior"]["px"] if cardinal else None),
        ("superior", 270.0, "S", cardinal["superior"]["px"] if cardinal else None),
        ("east_raw",   0.0, "E", cardinal["east_raw"]["px"] if cardinal else None),
        ("west_raw", 180.0, "W", cardinal["west_raw"]["px"] if cardinal else None),
    ]
    eye_side = analysis["rim"]["isnt"].get("eye_side")
    if eye_side == "OD":
        sides[2] = ("east_raw", 0.0, "T", cardinal["east_raw"]["px"] if cardinal else None)
        sides[3] = ("west_raw", 180.0, "N", cardinal["west_raw"]["px"] if cardinal else None)
    elif eye_side == "OS":
        sides[2] = ("east_raw", 0.0, "N", cardinal["east_raw"]["px"] if cardinal else None)
        sides[3] = ("west_raw", 180.0, "T", cardinal["west_raw"]["px"] if cardinal else None)

    h, w = canvas.shape[:2]
    DD = analysis["units"]["disc_diameter_px"]
    ray_len = DD  # ray length of one disc diameter is enough to reach the edge
    for name, angle, label, width_px in sides:
        end = _polar_ray_end((dcy, dcx), angle, ray_len)
        cv2.line(canvas, (int(dcx), int(dcy)), end, _COLOR["ray_rim"], 1, cv2.LINE_AA)
        lbl = f"{label}"
        if width_px is not None:
            lbl += f": {width_px:.1f}px"
        _put_text_with_bg(canvas, lbl, (end[0], end[1]), scale=0.45)

    # ISNT status
    isnt = analysis["rim"]["isnt"]
    lines = ["ISNT rule (I ≥ S ≥ N ≥ T)"]
    if isnt.get("follows_rule") is None:
        lines.append("eye_side unknown → N/T undetermined")
    elif isnt["follows_rule"]:
        lines.append("✓ rule respected")
    else:
        lines.append("✗ rule violated")
        lines.append(f"violations: {', '.join(isnt['violations'])}")
    if cardinal and isnt.get("eye_side") in ("OS", "OD"):
        lines.append(f"I={cardinal['inferior']['px']:.1f}  S={cardinal['superior']['px']:.1f}  "
                     f"N={cardinal['nasal']['px']:.1f}  T={cardinal['temporal']['px']:.1f}")
    _text_block(canvas, lines, anchor="tl")
    return canvas


def render_rim_profile(analysis: Dict[str, Any], config: DiscCupVizConfig) -> np.ndarray:
    """Matplotlib polar plot of rim width vs angle.

    Image frame is rendered purely from metadata (no base image needed), so
    this panel doubles as a standalone chart you can drop into a report.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scan = analysis["rim"].get("sector_scan")
    if scan is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=config.dpi)
        ax.text(0.5, 0.5, "rim profile unavailable (no cup)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        out = _matplotlib_to_bgr(fig)
        plt.close(fig)
        return out

    angles = np.array([p[0] for p in scan["profile_deg_to_width_px"]])
    widths = np.array([p[1] for p in scan["profile_deg_to_width_px"]])
    DD = analysis["units"]["disc_diameter_px"]
    rims_dd = widths / DD if DD > 0 else widths

    fig = plt.figure(figsize=(5.5, 5.5), dpi=config.dpi)
    ax = fig.add_subplot(111, projection="polar")
    # Convert our image-plane angle (0°=east, 90°=south) into matplotlib polar
    # (0°=east, 90°=north, counter-clockwise). In polar, "south" is -y which
    # corresponds to matplotlib 270° with default orientation. We rotate the
    # polar plot so that 90° (inferior) sits at the bottom of the figure,
    # matching the image convention.
    ax.set_theta_zero_location("E")       # 0° at east
    ax.set_theta_direction(-1)            # clockwise — matches image y-down
    theta = np.deg2rad(angles)
    ax.plot(theta, rims_dd, color="tab:green", linewidth=1.5)
    ax.fill(theta, rims_dd, color="tab:green", alpha=0.15)

    # Shade I / S / N / T sectors for context
    eye_side = analysis["rim"]["isnt"].get("eye_side")
    sector_wedges = [
        ("inferior", 45, 135, "#cccccc"),
        ("superior", 225, 315, "#cccccc"),
    ]
    if eye_side == "OD":
        sector_wedges += [("temporal", 315, 45, "#ffe9a8"),
                          ("nasal", 135, 225, "#a8d4ff")]
    elif eye_side == "OS":
        sector_wedges += [("nasal", 315, 45, "#a8d4ff"),
                          ("temporal", 135, 225, "#ffe9a8")]

    rmax = float(max(rims_dd.max() * 1.1, 0.05))
    for name, a0, a1, fill in sector_wedges:
        if a1 < a0:  # wraps past 360
            ts = np.concatenate([np.arange(a0, 360), np.arange(0, a1 + 1)])
        else:
            ts = np.arange(a0, a1 + 1)
        ax.fill_between(np.deg2rad(ts),
                        np.zeros_like(ts, dtype=float),
                        np.full_like(ts, rmax, dtype=float),
                        color=fill, alpha=0.15, zorder=0)
        # Label at centre angle
        if a1 < a0:
            mid = ((a0 + a1 + 360) / 2) % 360
        else:
            mid = (a0 + a1) / 2
        ax.text(np.deg2rad(mid), rmax, name[0].upper(),
                fontsize=10, color="#333333", ha="center", va="center")

    # Mark min-rim angle
    min_angle = scan["min_rim_angle_deg"]
    min_dd = scan["min_rim_width"]["dd"]
    ax.plot([np.deg2rad(min_angle)], [min_dd], "ro", markersize=8, zorder=5)
    ax.annotate(f"min  {min_dd:.3f} DD",
                xy=(np.deg2rad(min_angle), min_dd),
                xytext=(8, 4), textcoords="offset points",
                color="red", fontsize=9, zorder=6)

    if scan["zero_run_deg"] > 0:
        ax.set_title(f"Rim profile (rim-absent arc: {scan['zero_run_deg']}°)", pad=14)
    else:
        ax.set_title("Rim profile (polar — radius = rim / DD)", pad=14)
    ax.set_rlabel_position(45)
    ax.grid(alpha=0.3)

    out = _matplotlib_to_bgr(fig)
    plt.close(fig)
    return out


def render_tilt(image_bgr: np.ndarray, disc_bin: np.ndarray,
                analysis: Dict[str, Any], config: DiscCupVizConfig) -> np.ndarray:
    canvas = _to_bgr_uint8(image_bgr).copy()
    canvas = (canvas * 0.5).astype(np.uint8)
    canvas = _blend_overlay(canvas, _COLOR["disc"], disc_bin, config.alpha_disc)
    _draw_mask_outline(canvas, disc_bin, _COLOR["disc_outline"], thickness=2)

    dcy, dcx = analysis["disc"]["center_yx"]
    v_ext = analysis["disc"]["diameter_vertical"]["px"]
    h_ext = analysis["disc"]["diameter_horizontal"]["px"]

    # Draw bbox-based vertical and horizontal diameters
    y0 = int(dcy - v_ext / 2)
    y1 = int(dcy + v_ext / 2)
    cv2.line(canvas, (int(dcx), y0), (int(dcx), y1), _COLOR["vline"], 1, cv2.LINE_AA)
    x0 = int(dcx - h_ext / 2)
    x1 = int(dcx + h_ext / 2)
    cv2.line(canvas, (x0, int(dcy)), (x1, int(dcy)), _COLOR["hline"], 1, cv2.LINE_AA)

    # Ellipse overlay + major axis
    # Try to recompute ellipse here so we have the full cv2 params
    mask_u8 = (disc_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=lambda c: len(c))
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(canvas, ellipse, _COLOR["fit_ellipse"], 2, cv2.LINE_AA)
            (cx, cy), (minor, major), angle = ellipse
            # Draw major axis
            a = np.deg2rad(angle - 90.0)
            half_major = major / 2.0
            p1 = (int(cx + half_major * np.cos(a)), int(cy + half_major * np.sin(a)))
            p2 = (int(cx - half_major * np.cos(a)), int(cy - half_major * np.sin(a)))
            cv2.line(canvas, p1, p2, _COLOR["tilt_axis"], 2, cv2.LINE_AA)

    _text_block(canvas, [
        "Disc tilt & ovality",
        f"tilt: {analysis['disc']['tilt_angle_deg']:+.2f}° from vertical",
        f"ovality: {analysis['disc']['ovality']:.4f}  (vertical/horizontal)",
        f"v-diameter: {v_ext:.1f} px    h-diameter: {h_ext:.1f} px",
        "green ellipse = cv2.fitEllipse major axis",
    ], anchor="tl")
    return canvas


def render_ddls(image_bgr: np.ndarray, analysis: Dict[str, Any],
                config: DiscCupVizConfig) -> np.ndarray:
    """A small info card summarising DDLS grading. Uses matplotlib for the
    layout since the content is mostly text + a horizontal scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ddls = analysis.get("ddls")
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=config.dpi)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title = "DDLS (Bayer–Nicolela, simplified)"
    ax.text(0.02, 0.95, title, fontsize=14, fontweight="bold", va="top")

    if ddls is None:
        ax.text(0.02, 0.75, "Rim profile unavailable (no cup). DDLS = N/A.",
                fontsize=11, va="top")
        out = _matplotlib_to_bgr(fig)
        plt.close(fig)
        return out

    grade = ddls.get("grade")
    disc_size = ddls.get("disc_size")
    min_ratio = ddls.get("min_rim_disc_ratio")
    zero_run = ddls.get("zero_run_deg")
    disc_area_mm2 = ddls.get("disc_area_mm2")
    reasoning = ddls.get("reasoning", "")

    lines = [
        f"grade: {'N/A' if grade is None else grade}",
        f"disc size: {disc_size or 'unknown'}" +
            (f"  (area = {disc_area_mm2:.2f} mm²)" if disc_area_mm2 is not None else ""),
        f"min rim / disc: {min_ratio:.4f}" if min_ratio is not None else "min rim / disc: n/a",
        f"rim-absent arc: {zero_run}°" if zero_run is not None else "",
        "",
        f"reasoning:  {reasoning}",
    ]
    y = 0.82
    for line in lines:
        ax.text(0.02, y, line, fontsize=11, va="top")
        y -= 0.07

    # Ratio scale bar
    if min_ratio is not None:
        ax.text(0.02, 0.22, "min rim / disc scale:", fontsize=10, va="top")
        # Colour bar from 0 (red) → 0.5 (green)
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, extent=(0.02, 0.98, 0.08, 0.14), aspect="auto",
                  cmap="RdYlGn", clip_on=False)
        pos = 0.02 + 0.96 * min(max(min_ratio / 0.5, 0.0), 1.0)
        ax.plot([pos, pos], [0.08, 0.18], color="black", linewidth=2)
        ax.text(pos, 0.20, f"{min_ratio:.3f}", ha="center", fontsize=10)
        ax.text(0.02, 0.05, "0.00", fontsize=9, ha="left")
        ax.text(0.98, 0.05, "≥0.50", fontsize=9, ha="right")

    out = _matplotlib_to_bgr(fig)
    plt.close(fig)
    return out


def render_combined_panel(panels: Dict[str, np.ndarray],
                          analysis: Dict[str, Any],
                          config: DiscCupVizConfig) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["overview", "cdr", "isnt", "rim_profile", "tilt", "ddls"]
    titles = {
        "overview": "Overview", "cdr": "CDR", "isnt": "ISNT rule",
        "rim_profile": "Rim profile", "tilt": "Tilt & ovality",
        "ddls": "DDLS",
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
    # Letterbox each panel to cell aspect so every one fills its subplot
    # without distortion; panel content is preserved intact.
    for ax, key in zip(axes, order):
        if key not in panels:
            continue
        panel = panels[key]
        bg = _panel_bg_color(panel)
        padded = _pad_to_aspect(panel, cell_aspect, bg)
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(titles.get(key, key), fontsize=11)
        ax.axis("off")

    if config.combined_title:
        cdr = analysis["cdr"]
        disc_area_mm2 = analysis["disc"].get("area_mm2")
        area_tag = f"{disc_area_mm2:.2f} mm²" if disc_area_mm2 else f"{analysis['disc']['area_px']} px²"
        ddls = analysis.get("ddls")
        ddls_tag = ""
        if ddls and ddls.get("grade") is not None:
            ddls_tag = f"    DDLS = {ddls['grade']}"
        title = (
            f"Disc/Cup analysis  |  vCDR={cdr['vertical']:.3f}  "
            f"hCDR={cdr['horizontal']:.3f}  aCDR={cdr['area']:.3f}  "
            f"disc={area_tag}{ddls_tag}"
        )
        fig.suptitle(title, fontsize=12, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if config.combined_title else None)
    out = _matplotlib_to_bgr(fig)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- main entry

def render_disc_cup_visualizations(
    original_image_bgr: np.ndarray,
    od_oc_mask: np.ndarray,
    analysis_result: Dict[str, Any],
    output_dir: str,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    config: Optional[DiscCupVizConfig] = None,
) -> Dict[str, str]:
    """Render all enabled panels + the combined mosaic, write PNGs.

    Returns a dict mapping panel name → written file path.
    """
    cfg = config or DiscCupVizConfig()
    os.makedirs(output_dir, exist_ok=True)

    _BASE_PANELS = ("overview", "cdr", "isnt", "rim_profile", "tilt", "ddls")
    if cfg.combined:
        missing = [p for p in _BASE_PANELS if not getattr(cfg, p)]
        if missing:
            warnings.warn(
                "render_disc_cup_visualizations: combined=True but these base "
                f"panels are disabled: {missing}. The combined mosaic will "
                "contain blank cells for them.",
                stacklevel=2,
            )

    disc_bin, cup_bin = from_od_oc_labelmap(od_oc_mask)
    image_bgr = _to_bgr_uint8(original_image_bgr)
    panels: Dict[str, np.ndarray] = {}

    if cfg.overview:
        panels["overview"] = render_overview(
            image_bgr, disc_bin, cup_bin, analysis_result, cfg, macula_yx=macula_center_yx
        )
    if cfg.cdr:
        panels["cdr"] = render_cdr(image_bgr, disc_bin, cup_bin, analysis_result, cfg)
    if cfg.isnt:
        panels["isnt"] = render_isnt(image_bgr, disc_bin, cup_bin, analysis_result, cfg)
    if cfg.rim_profile:
        panels["rim_profile"] = render_rim_profile(analysis_result, cfg)
    if cfg.tilt:
        panels["tilt"] = render_tilt(image_bgr, disc_bin, analysis_result, cfg)
    if cfg.ddls:
        panels["ddls"] = render_ddls(image_bgr, analysis_result, cfg)

    written: Dict[str, str] = {}
    for name, panel in panels.items():
        path = os.path.join(output_dir, f"disc_cup_{name}.png")
        cv2.imwrite(path, panel)
        written[name] = path

    if cfg.combined and panels:
        combined = render_combined_panel(panels, analysis_result, cfg)
        path = os.path.join(output_dir, "disc_cup_combined.png")
        cv2.imwrite(path, combined)
        written["combined"] = path

    return written
