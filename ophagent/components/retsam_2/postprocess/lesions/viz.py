"""
Lesion analysis visualizations.

Panels (each independently toggleable via ``LesionVizConfig``):

    * overview          — original + all classes in their own colours + legend
    * per_group         — small multiples, one sub-panel per group
    * burden            — bar charts of count + area per class
    * spatial           — centroids on fundus + macula-quadrant dividers
    * macula_zones      — concentric ETDRS-style rings + per-zone counts
    * size_distribution — stacked histograms of lesion sizes per class
    * combined          — 2×3 mosaic
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- palette

# BGR colours — consistent across all lesion panels. Picked so clinically-
# similar lesions have similar hues (reds for blood, yellows for exudate-type).
LESION_COLORS: Dict[str, Tuple[int, int, int]] = {
    # DR lesions
    "hemorrhage":           (0, 0, 220),       # saturated red
    "exudate":              (0, 220, 230),     # yellow
    "cotton_wool_spot":     (220, 220, 220),   # off-white
    "laser_spot":           (255, 100, 0),     # bright blue
    # AMD lesions
    "drusen":               (100, 200, 255),   # soft orange
    "patch_hemorrhage":     (0, 50, 150),      # dark red
    # Others
    "epiretinal_membrane":  (200, 100, 255),   # pink-purple
    "artifact":             (60, 60, 160),     # muddy red
    "retinal_scar":         (150, 150, 60),    # olive
    "macular_hole":         (150, 0, 150),     # magenta
    # Possible
    "other_lesions":        (100, 100, 100),   # grey
    "myelinated_nerve_fiber": (180, 220, 120), # pale green
    "venous_tortuosity":    (255, 50, 200),    # bright magenta
}

GROUP_COLORS: Dict[str, Tuple[int, int, int]] = {
    "lesion_dr":        (40, 40, 220),
    "lesion_amd":       (40, 140, 230),
    "lesion_others":    (200, 100, 100),
    "possible_lesions": (120, 120, 120),
}


# --------------------------------------------------------------------------- config

@dataclass
class LesionVizConfig:
    overview: bool = True
    per_group: bool = True
    burden: bool = True
    spatial: bool = True
    macula_zones: bool = True
    size_distribution: bool = True
    combined: bool = True

    alpha_lesion: float = 0.55
    dim_factor: float = 0.45          # darken base image for coloured overlays
    dpi: int = 150
    combined_title: bool = True


# --------------------------------------------------------------------------- helpers

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
                  (x + w + pad, y + baseline + pad), bg, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, _ascii_safe(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _text_block(canvas: np.ndarray, lines: Sequence[str],
                anchor: str = "tl", scale: float = 0.55,
                pad: int = 6, line_spacing: int = 6) -> None:
    h, w = canvas.shape[:2]
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for line in lines]
    box_w = max((tw for tw, _ in sizes), default=0) + 2 * pad
    box_h = sum(th for _, th in sizes) + line_spacing * max(0, len(lines) - 1) + 2 * pad
    if anchor == "tl":
        x0, y0 = 8, 8
    elif anchor == "tr":
        x0, y0 = w - box_w - 8, 8
    elif anchor == "bl":
        x0, y0 = 8, h - box_h - 8
    else:
        x0, y0 = w - box_w - 8, h - box_h - 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    y = y0 + pad
    for line, (tw, th) in zip(lines, sizes):
        y += th
        cv2.putText(canvas, _ascii_safe(line), (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_spacing


def _matplotlib_to_bgr(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
        pad = max(0, new_w - w)
        l = pad // 2
        r = pad - l
        return cv2.copyMakeBorder(img, 0, 0, l, r, cv2.BORDER_CONSTANT, value=bg)
    new_h = int(round(w / target_aspect_wh))
    pad = max(0, new_h - h)
    t = pad // 2
    b = pad - t
    return cv2.copyMakeBorder(img, t, b, 0, 0, cv2.BORDER_CONSTANT, value=bg)


def _dim(img: np.ndarray, factor: float) -> np.ndarray:
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def _iter_nonempty_classes(analysis: Dict[str, Any]):
    for group_name, group in analysis["groups"].items():
        for class_name, cls in group["classes"].items():
            if cls["count"] > 0:
                yield group_name, class_name, cls


# --------------------------------------------------------------------------- panels

def render_overview(image_bgr: np.ndarray,
                    lesion_masks: Dict[str, Dict[str, np.ndarray]],
                    analysis: Dict[str, Any],
                    disc_mask: Optional[np.ndarray],
                    fundus_mask: Optional[np.ndarray],
                    macula_yx: Optional[Tuple[float, float]],
                    config: LesionVizConfig) -> np.ndarray:
    canvas = _dim(_to_bgr_uint8(image_bgr), config.dim_factor)

    # Overlay each detected class
    detected_legend: List[Tuple[str, int, Tuple[int, int, int]]] = []
    for group_name, classes in lesion_masks.items():
        for class_name, mask in classes.items():
            if mask is None or mask.sum() == 0:
                continue
            color = LESION_COLORS.get(class_name, (200, 200, 200))
            canvas = _blend_overlay(canvas, color, mask, config.alpha_lesion)
            count = analysis["groups"].get(group_name, {}).get("classes", {}).get(class_name, {}).get("count", 0)
            if count > 0:
                detected_legend.append((class_name, count, color))

    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), thickness=2)
    if fundus_mask is not None:
        _draw_mask_outline(canvas, fundus_mask, (255, 255, 255), thickness=1)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255), cv2.MARKER_TILTED_CROSS,
                       16, 2, cv2.LINE_AA)

    title_lines = [
        "Lesion overview",
        f"classes detected: {analysis['global_summary']['n_classes_detected']}",
        f"total coverage: {analysis['global_summary']['total_coverage_ratio']*100:.3f} %",
    ]
    _text_block(canvas, title_lines, anchor="tl")

    # Bottom-right colour legend for detected classes
    if detected_legend:
        h, w = canvas.shape[:2]
        line_h = 22
        box_h = line_h * len(detected_legend) + 8
        box_w = 260
        x0 = w - box_w - 8
        y0 = h - box_h - 8
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
        for i, (name, count, colour) in enumerate(detected_legend):
            swatch_x0 = x0 + 8
            swatch_y0 = y0 + 8 + i * line_h
            cv2.rectangle(canvas, (swatch_x0, swatch_y0),
                          (swatch_x0 + 16, swatch_y0 + 14), colour, -1)
            cv2.putText(canvas, _ascii_safe(f"{name}  (n={count})"),
                        (swatch_x0 + 24, swatch_y0 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_per_group(image_bgr: np.ndarray,
                     lesion_masks: Dict[str, Dict[str, np.ndarray]],
                     analysis: Dict[str, Any],
                     disc_mask: Optional[np.ndarray],
                     config: LesionVizConfig) -> np.ndarray:
    """2×2 small multiples: one cell per lesion GROUP, with all its classes
    mixed in their own colours."""
    order = ["lesion_dr", "lesion_amd", "lesion_others", "possible_lesions"]
    h, w = image_bgr.shape[:2]
    cells: List[np.ndarray] = []
    for group_name in order:
        cell = _dim(_to_bgr_uint8(image_bgr), config.dim_factor)
        classes = lesion_masks.get(group_name, {})
        per_group_detected = []
        for class_name, mask in classes.items():
            if mask is None or mask.sum() == 0:
                continue
            color = LESION_COLORS.get(class_name, (200, 200, 200))
            cell = _blend_overlay(cell, color, mask, config.alpha_lesion)
            count = analysis["groups"].get(group_name, {}).get("classes", {}).get(class_name, {}).get("count", 0)
            per_group_detected.append((class_name, count, color))
        if disc_mask is not None:
            _draw_mask_outline(cell, disc_mask, (0, 255, 0), thickness=1)
        group_total = analysis["groups"].get(group_name, {}).get("summary", {}).get("total_count", 0)
        lines = [f"{group_name}", f"total count: {group_total}"]
        lines += [f"{n} (n={c})" for n, c, _ in per_group_detected[:6]]
        _text_block(cell, lines, anchor="tl", scale=0.45)
        cells.append(cell)
    # Arrange into 2×2
    top = np.hstack(cells[:2])
    bot = np.hstack(cells[2:])
    return np.vstack([top, bot])


def render_burden(analysis: Dict[str, Any], config: LesionVizConfig) -> np.ndarray:
    """Bar chart: count + coverage % per detected class, grouped by group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[Tuple[str, str, int, float, Tuple[int, int, int]]] = []
    for group_name, group in analysis["groups"].items():
        for class_name, cls in group["classes"].items():
            count = cls["count"]
            if count == 0:
                continue
            coverage_pct = cls["coverage_ratio"] * 100
            color_bgr = LESION_COLORS.get(class_name, (200, 200, 200))
            rows.append((group_name, class_name, count, coverage_pct, color_bgr))

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=config.dpi)
        ax.text(0.5, 0.5, "No lesions detected", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        out = _matplotlib_to_bgr(fig); plt.close(fig); return out

    # Sort by (group, name) for stable order
    rows.sort(key=lambda r: (r[0], r[1]))
    labels = [f"{r[1]}" for r in rows]
    counts = [r[2] for r in rows]
    coverages = [r[3] for r in rows]
    colors = [(b / 255, g / 255, r_ / 255) for (_, _, _, _, (b, g, r_)) in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3.5, len(rows) * 0.36)),
                                   dpi=config.dpi)
    y_pos = np.arange(len(rows))
    ax1.barh(y_pos, counts, color=colors)
    ax1.set_yticks(y_pos, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("count")
    ax1.set_title("Lesion count per class")
    for i, v in enumerate(counts):
        ax1.text(v, i, f" {v}", va="center", fontsize=9)

    ax2.barh(y_pos, coverages, color=colors)
    ax2.set_yticks(y_pos, ["" for _ in labels])
    ax2.set_xlabel("coverage %")
    ax2.set_title("Fundus coverage per class")
    for i, v in enumerate(coverages):
        ax2.text(v, i, f" {v:.3f}%", va="center", fontsize=9)

    fig.tight_layout()
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


def render_spatial(image_bgr: np.ndarray,
                   analysis: Dict[str, Any],
                   disc_mask: Optional[np.ndarray],
                   macula_yx: Optional[Tuple[float, float]],
                   eye_side: Optional[str],
                   config: LesionVizConfig) -> np.ndarray:
    """Fundus with lesion centroids coloured by class + macula-centred
    quadrant dividers (when eye_side + macula are available)."""
    canvas = _dim(_to_bgr_uint8(image_bgr), 0.65)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), thickness=2)

    h, w = canvas.shape[:2]
    # Draw quadrant dividers through macula
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.line(canvas, (0, my), (w, my), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(canvas, (mx, 0), (mx, h), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    # Centroid dots, sized by lesion area
    for group_name, group in analysis["groups"].items():
        for class_name, cls in group["classes"].items():
            if cls["count"] == 0 or not cls.get("components"):
                continue
            color = LESION_COLORS.get(class_name, (200, 200, 200))
            for comp in cls["components"]:
                y, x = comp["centroid_yx"]
                r = max(2, int(round(np.sqrt(comp["area_px"] / np.pi))))
                cv2.circle(canvas, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)

    # Text: per-quadrant totals if known
    eye_tag = f"eye side: {eye_side}" if eye_side else "eye side: unknown"
    q_counts_text: List[str] = []
    if eye_side in ("OS", "OD") and macula_yx is not None:
        # Aggregate over all classes
        aggregate = {
            "superior_temporal": 0, "superior_nasal": 0,
            "inferior_temporal": 0, "inferior_nasal": 0,
        }
        for group in analysis["groups"].values():
            for cls in group["classes"].values():
                q = cls.get("spatial", {}).get("quadrant_counts") or {}
                for k, v in q.items():
                    if k in aggregate:
                        aggregate[k] += int(v)
        q_counts_text = [f"{k}: {v}" for k, v in aggregate.items()]
    _text_block(canvas, ["Spatial distribution", eye_tag] + q_counts_text, anchor="tl")
    return canvas


def render_macula_zones(image_bgr: np.ndarray,
                        analysis: Dict[str, Any],
                        disc_mask: Optional[np.ndarray],
                        macula_yx: Optional[Tuple[float, float]],
                        zone_boundaries_dd: Sequence[float],
                        config: LesionVizConfig) -> np.ndarray:
    """Concentric rings around macula with lesion overlay and per-zone counts.

    Without a macula centre the panel still shows a readable placeholder
    explaining why the rings are omitted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas = _dim(_to_bgr_uint8(image_bgr), 0.5)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), thickness=2)

    DD = float(analysis["units"]["disc_diameter_px"])
    if macula_yx is not None and DD > 0:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        for b in zone_boundaries_dd:
            r = int(round(float(b) * DD))
            cv2.circle(canvas, (mx, my), r, (255, 255, 255), 1, cv2.LINE_AA)
            _put_text_with_bg(canvas, f"{b:g} DD", (mx + r + 4, my - 6), scale=0.42)
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    else:
        _text_block(canvas,
                    ["Macula zones",
                     "macula_center not provided → zones omitted"],
                    anchor="tl")
        return canvas

    # Overlay lesions (colour per class)
    for group in analysis["groups"].values():
        for class_name, cls in group["classes"].items():
            if cls["count"] == 0 or not cls.get("components"):
                continue
            color = LESION_COLORS.get(class_name, (200, 200, 200))
            for comp in cls["components"]:
                y, x = comp["centroid_yx"]
                r = max(2, int(round(np.sqrt(comp["area_px"] / np.pi))))
                cv2.circle(canvas, (int(x), int(y)), r, color, -1, cv2.LINE_AA)

    # Aggregate per-zone counts across classes
    zone_counts_agg: Dict[str, int] = {}
    for group in analysis["groups"].values():
        for cls in group["classes"].values():
            zones = cls.get("spatial", {}).get("macula_zone_counts")
            if not isinstance(zones, dict):
                continue
            for key, val in zones.items():
                zone_counts_agg[key] = zone_counts_agg.get(key, 0) + int(val)

    lines = ["Macula-centred zones"]
    for key in zone_counts_agg:
        lines.append(f"{key}: {zone_counts_agg[key]} lesions")
    _text_block(canvas, lines, anchor="tl")
    return canvas


def render_size_distribution(analysis: Dict[str, Any],
                             config: LesionVizConfig) -> np.ndarray:
    """Horizontal stacked bars: small / medium / large counts per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[Tuple[str, Dict[str, int], Tuple[int, int, int]]] = []
    for group in analysis["groups"].values():
        for class_name, cls in group["classes"].items():
            if cls["count"] == 0:
                continue
            counts = cls["size_distribution"]["counts"]
            rows.append((class_name, counts, LESION_COLORS.get(class_name, (200, 200, 200))))

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=config.dpi)
        ax.text(0.5, 0.5, "No lesions detected", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        out = _matplotlib_to_bgr(fig); plt.close(fig); return out

    labels = [r[0] for r in rows]
    small = np.array([r[1].get("small", 0) for r in rows])
    med = np.array([r[1].get("medium", 0) for r in rows])
    large = np.array([r[1].get("large", 0) for r in rows])

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(rows) * 0.4)), dpi=config.dpi)
    y_pos = np.arange(len(rows))
    ax.barh(y_pos, small, color="#9ecae1", label="small")
    ax.barh(y_pos, med, left=small, color="#4292c6", label="medium")
    ax.barh(y_pos, large, left=small + med, color="#08519c", label="large")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("count")

    # Display the DD² threshold definitions as subtitle
    first_row = next((g["classes"] for g in analysis["groups"].values()
                      if any(c["count"] > 0 for c in g["classes"].values())), None)
    small_thr, med_thr = None, None
    if first_row:
        for cls in first_row.values():
            t = cls.get("size_distribution", {}).get("thresholds_dd2")
            if t:
                small_thr, med_thr = t
                break
    subtitle = (
        f"Size buckets in DD²: small ≤ {small_thr:.4f}, "
        f"medium ≤ {med_thr:.4f}, large > {med_thr:.4f}"
        if small_thr is not None else "Size buckets in DD²"
    )
    ax.set_title("Lesion size distribution per class\n" + subtitle, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    for i, tot in enumerate(small + med + large):
        ax.text(tot, i, f" {tot}", va="center", fontsize=9)
    fig.tight_layout()
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


def render_combined_panel(panels: Dict[str, np.ndarray],
                          analysis: Dict[str, Any],
                          config: LesionVizConfig) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["overview", "per_group", "burden",
             "spatial", "macula_zones", "size_distribution"]
    titles = {
        "overview": "Overview", "per_group": "Per group",
        "burden": "Burden", "spatial": "Spatial",
        "macula_zones": "Macula zones", "size_distribution": "Size distribution",
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
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(titles.get(key, key), fontsize=11)
        ax.axis("off")

    if config.combined_title:
        gs = analysis["global_summary"]
        total_cov = gs["total_coverage_ratio"] * 100
        title = (
            f"Lesion analysis  |  classes detected: {gs['n_classes_detected']}  "
            f"total coverage: {total_cov:.3f}%"
        )
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if config.combined_title else None)
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


# --------------------------------------------------------------------------- main entry

def render_lesion_visualizations(
    original_image_bgr: np.ndarray,
    lesion_masks: Dict[str, Dict[str, np.ndarray]],
    analysis_result: Dict[str, Any],
    output_dir: str,
    disc_mask: Optional[np.ndarray] = None,
    fundus_mask: Optional[np.ndarray] = None,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    eye_side: Optional[str] = None,
    config: Optional[LesionVizConfig] = None,
) -> Dict[str, str]:
    """Render all enabled panels + the combined mosaic."""
    cfg = config or LesionVizConfig()
    os.makedirs(output_dir, exist_ok=True)

    _BASE_PANELS = ("overview", "per_group", "burden",
                    "spatial", "macula_zones", "size_distribution")
    if cfg.combined:
        missing = [p for p in _BASE_PANELS if not getattr(cfg, p)]
        if missing:
            warnings.warn(
                "render_lesion_visualizations: combined=True but these base "
                f"panels are disabled: {missing}. The combined mosaic will "
                "contain blank cells for them.",
                stacklevel=2,
            )

    image_bgr = _to_bgr_uint8(original_image_bgr)
    zone_boundaries_dd = tuple(analysis_result["groups"]["lesion_dr"]["classes"]
                               .get("hemorrhage", {})
                               .get("spatial", {})
                               .get("zone_boundaries_dd")
                               or (1.0, 2.0))

    panels: Dict[str, np.ndarray] = {}
    if cfg.overview:
        panels["overview"] = render_overview(
            image_bgr, lesion_masks, analysis_result,
            disc_mask=disc_mask, fundus_mask=fundus_mask,
            macula_yx=macula_center_yx, config=cfg,
        )
    if cfg.per_group:
        panels["per_group"] = render_per_group(
            image_bgr, lesion_masks, analysis_result,
            disc_mask=disc_mask, config=cfg,
        )
    if cfg.burden:
        panels["burden"] = render_burden(analysis_result, cfg)
    if cfg.spatial:
        panels["spatial"] = render_spatial(
            image_bgr, analysis_result,
            disc_mask=disc_mask, macula_yx=macula_center_yx,
            eye_side=eye_side, config=cfg,
        )
    if cfg.macula_zones:
        panels["macula_zones"] = render_macula_zones(
            image_bgr, analysis_result,
            disc_mask=disc_mask, macula_yx=macula_center_yx,
            zone_boundaries_dd=zone_boundaries_dd, config=cfg,
        )
    if cfg.size_distribution:
        panels["size_distribution"] = render_size_distribution(analysis_result, cfg)

    written: Dict[str, str] = {}
    for name, panel in panels.items():
        path = os.path.join(output_dir, f"lesions_{name}.png")
        cv2.imwrite(path, panel)
        written[name] = path

    if cfg.combined and panels:
        combined = render_combined_panel(panels, analysis_result, cfg)
        path = os.path.join(output_dir, "lesions_combined.png")
        cv2.imwrite(path, combined)
        written["combined"] = path

    return written
