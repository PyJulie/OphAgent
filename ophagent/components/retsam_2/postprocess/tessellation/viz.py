"""
Tessellation visualizations — 4 panels + combined.

    * overview       — fundus with tessellation highlighted
    * heatmap        — density heatmap of tessellation pixels (Gaussian blur)
    * macula_zones   — same mask with 1/2 DD rings from macula + per-zone counts
    * severity       — information card: coverage, severity bucket, size distribution
    * combined       — 2×2 mosaic
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TESS_COLOR: Tuple[int, int, int] = (255, 200, 120)   # BGR pastel orange


@dataclass
class TessellationVizConfig:
    overview: bool = True
    heatmap: bool = True
    macula_zones: bool = True
    severity: bool = True
    combined: bool = True

    alpha: float = 0.55
    dim_factor: float = 0.45
    dpi: int = 150
    combined_title: bool = True


# --------------------------------------------------------------------------- helpers (self-contained)

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
    col = np.array(color, dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + col * alpha).astype(np.uint8)
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


def _put_text_with_bg(canvas, text, pos, scale=0.55, color=(255, 255, 255),
                      bg=(0, 0, 0), thickness=1, pad=3):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - pad, y - h - pad - 1),
                  (x + w + pad, y + baseline + pad), bg, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, _ascii_safe(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _text_block(canvas, lines, anchor="tl", scale=0.55, pad=6, line_spacing=6):
    h, w = canvas.shape[:2]
    sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for l in lines]
    box_w = max((tw for tw, _ in sizes), default=0) + 2 * pad
    box_h = sum(th for _, th in sizes) + line_spacing * max(0, len(lines) - 1) + 2 * pad
    x0, y0 = (8, 8) if anchor == "tl" else (w - box_w - 8, 8) if anchor == "tr" \
             else (8, h - box_h - 8) if anchor == "bl" else (w - box_w - 8, h - box_h - 8)
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


def _panel_bg_color(img):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return (0, 0, 0)
    corners = np.stack([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
    med = np.median(corners, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _pad_to_aspect(img, target_aspect_wh, bg):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    cur = w / h
    if abs(cur - target_aspect_wh) < 1e-3:
        return img
    if cur < target_aspect_wh:
        new_w = int(round(h * target_aspect_wh)); pad = max(0, new_w - w)
        l = pad // 2; r = pad - l
        return cv2.copyMakeBorder(img, 0, 0, l, r, cv2.BORDER_CONSTANT, value=bg)
    new_h = int(round(w / target_aspect_wh)); pad = max(0, new_h - h)
    t = pad // 2; b = pad - t
    return cv2.copyMakeBorder(img, t, b, 0, 0, cv2.BORDER_CONSTANT, value=bg)


def _dim(img, factor):
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- panels

def render_overview(image_bgr, tessellation_mask, analysis, disc_mask,
                    macula_yx, config):
    canvas = _dim(_to_bgr_uint8(image_bgr), config.dim_factor)
    canvas = _blend_overlay(canvas, TESS_COLOR, tessellation_mask, config.alpha)
    if disc_mask is not None:
        _draw_mask_outline(canvas, disc_mask, (0, 255, 0), 2)
    if macula_yx is not None:
        my, mx = int(macula_yx[0]), int(macula_yx[1])
        cv2.drawMarker(canvas, (mx, my), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    tess = analysis["tessellation"]
    _text_block(canvas, [
        "Tessellation overview",
        f"coverage: {tess['coverage_ratio']*100:.3f}%",
        f"severity: {tess['severity']}",
        f"components: {tess['count']}  (involves macula: {tess['involves_macula']})",
    ], anchor="tl")
    return canvas


def render_heatmap(image_bgr, tessellation_mask, analysis,
                   disc_mask, config):
    """Gaussian-smoothed density heatmap of tessellation presence."""
    canvas = _dim(_to_bgr_uint8(image_bgr), 0.35)
    bin_mask = (tessellation_mask > 0).astype(np.float32)
    if bin_mask.sum() == 0:
        _text_block(canvas, ["Heatmap", "no tessellation detected"], anchor="tl")
        return canvas
    sigma = max(8, int(min(canvas.shape[:2]) * 0.02))
    density = cv2.GaussianBlur(bin_mask, (0, 0), sigma)
    density /= max(density.max(), 1e-6)
    density_u8 = np.clip(density * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(density_u8, cv2.COLORMAP_MAGMA)
    # Blend where density is non-trivial
    mask = density_u8 > 20
    out = canvas.copy()
    alpha = 0.65
    out[mask] = (canvas[mask].astype(np.float32) * (1 - alpha)
                 + heat[mask].astype(np.float32) * alpha).astype(np.uint8)
    if disc_mask is not None:
        _draw_mask_outline(out, disc_mask, (0, 255, 0), 2)
    _text_block(out, ["Tessellation density heatmap", f"σ = {sigma} px"], anchor="tl")
    return out


def render_macula_zones(image_bgr, tessellation_mask, analysis, disc_mask,
                        macula_yx, config):
    canvas = _dim(_to_bgr_uint8(image_bgr), 0.45)
    canvas = _blend_overlay(canvas, TESS_COLOR, tessellation_mask, config.alpha)
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

    zones = analysis["tessellation"]["spatial"].get("macula_zone_counts") or {}
    lines = ["Macula-centred zones",
             f"involves macula: {analysis['tessellation']['involves_macula']}"]
    for k, v in zones.items():
        lines.append(f"{k}: {v}")
    _text_block(canvas, lines, anchor="tl")
    return canvas


def render_severity(analysis, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tess = analysis["tessellation"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=config.dpi)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.95, "Tessellation severity", fontsize=14, fontweight="bold", va="top")

    cov = tess["coverage_ratio"] * 100
    lines = [
        f"coverage: {cov:.3f} %",
        f"severity bucket: {tess['severity']}",
        f"components (small / medium / large): "
        f"{tess['size_distribution']['counts'].get('small', 0)} / "
        f"{tess['size_distribution']['counts'].get('medium', 0)} / "
        f"{tess['size_distribution']['counts'].get('large', 0)}",
        f"shape:   circularity={tess['shape']['mean_circularity']:.3f}, "
        f"aspect_ratio={tess['shape']['mean_aspect_ratio']:.3f}",
        f"involves macula: {tess['involves_macula']}",
    ]
    y = 0.82
    for line in lines:
        ax.text(0.02, y, line, fontsize=11, va="top")
        y -= 0.08

    # Severity scale bar
    thr = tess["severity_thresholds_ratio"]
    ax.text(0.02, 0.30, "Severity scale (coverage_ratio):", fontsize=10, va="top")
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=(0.02, 0.98, 0.16, 0.22), aspect="auto",
              cmap="YlOrRd", clip_on=False)
    # Place the current coverage marker
    max_display = max(thr[-1] * 1.5, cov / 100.0)
    pos = 0.02 + 0.96 * min(max((cov / 100.0) / max_display, 0.0), 1.0)
    ax.plot([pos, pos], [0.16, 0.26], color="black", linewidth=2)
    ax.text(pos, 0.28, f"{cov:.2f}%", ha="center", fontsize=10)
    # Threshold ticks
    for t in thr:
        tx = 0.02 + 0.96 * min(max(t / max_display, 0.0), 1.0)
        ax.plot([tx, tx], [0.14, 0.16], color="gray", linewidth=1)
        ax.text(tx, 0.12, f"{t*100:g}%", ha="center", fontsize=8, color="gray")
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


def render_combined_panel(panels, analysis, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["overview", "heatmap", "macula_zones", "severity"]
    titles = {"overview": "Overview", "heatmap": "Density heatmap",
              "macula_zones": "Macula zones", "severity": "Severity card"}
    rows, cols = 2, 2
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
        t = analysis["tessellation"]
        title = (f"Tessellation  |  coverage: {t['coverage_ratio']*100:.3f}%  "
                 f"severity: {t['severity']}  components: {t['count']}  "
                 f"involves macula: {t['involves_macula']}")
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if config.combined_title else None)
    out = _matplotlib_to_bgr(fig); plt.close(fig); return out


# --------------------------------------------------------------------------- main

def render_tessellation_visualizations(
    original_image_bgr: np.ndarray,
    tessellation_mask: np.ndarray,
    analysis_result: Dict[str, Any],
    output_dir: str,
    disc_mask: Optional[np.ndarray] = None,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    config: Optional[TessellationVizConfig] = None,
) -> Dict[str, str]:
    cfg = config or TessellationVizConfig()
    os.makedirs(output_dir, exist_ok=True)

    _BASE_PANELS = ("overview", "heatmap", "macula_zones", "severity")
    if cfg.combined:
        missing = [p for p in _BASE_PANELS if not getattr(cfg, p)]
        if missing:
            warnings.warn(
                "render_tessellation_visualizations: combined=True but these "
                f"base panels are disabled: {missing}. Combined mosaic will "
                "have blank cells.",
                stacklevel=2,
            )

    image_bgr = _to_bgr_uint8(original_image_bgr)
    bin_mask = (tessellation_mask > 0).astype(np.uint8)
    panels: Dict[str, np.ndarray] = {}
    if cfg.overview:
        panels["overview"] = render_overview(image_bgr, bin_mask, analysis_result,
                                             disc_mask, macula_center_yx, cfg)
    if cfg.heatmap:
        panels["heatmap"] = render_heatmap(image_bgr, bin_mask, analysis_result,
                                           disc_mask, cfg)
    if cfg.macula_zones:
        panels["macula_zones"] = render_macula_zones(image_bgr, bin_mask,
                                                     analysis_result, disc_mask,
                                                     macula_center_yx, cfg)
    if cfg.severity:
        panels["severity"] = render_severity(analysis_result, cfg)

    written: Dict[str, str] = {}
    for name, panel in panels.items():
        path = os.path.join(output_dir, f"tessellation_{name}.png")
        cv2.imwrite(path, panel)
        written[name] = path
    if cfg.combined and panels:
        combined = render_combined_panel(panels, analysis_result, cfg)
        path = os.path.join(output_dir, "tessellation_combined.png")
        cv2.imwrite(path, combined)
        written["combined"] = path
    return written
