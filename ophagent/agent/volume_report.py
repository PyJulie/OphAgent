"""
Volume-aware report builder.

Extends the single-slice report with:
  - cover summary of the cube
  - en-face fluid heat map (z-projection of fluid masks)
  - per-slice classification timeline (which slices voted for which class)
  - foveal slice deep-dive (Grad-CAM + fluid bbox + layer overlay)
  - representative slice montage

Outputs an HTML + a print-styled PDF.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..data.preprocessing.transforms import get_classification_transforms
from ..inference.model_registry import ModelRegistry
from ..inference.predictor import OphPredictor
from ..visualization.visualizer import (
    FLUID_COLORS, LAYER_COLORS,
    GradCAM, heatmap_overlay, segmentation_overlay,
    boxes_from_heatmap, boxes_from_mask, draw_boxes, save_image,
)
from .pdf_styler import PRINT_CSS
from .report_builder import _render_pdf
from .volume_processor import VolumeAnalysis


def render_enface(enface: np.ndarray, target_h: int = 200) -> np.ndarray:
    """Render the en-face fluid map (N_slices × W) as a BGR heatmap."""
    arr = enface.astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    arr = arr.astype(np.uint8)

    # upsample so the en-face has a usable display size
    if arr.shape[0] < target_h:
        scale = target_h / arr.shape[0]
        new_w = int(arr.shape[1] * scale)
        arr = cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    color = cv2.applyColorMap(arr, cv2.COLORMAP_HOT)
    return color


def build_slice_montage(
    slices,
    volume_arr: np.ndarray,
    n_rows: int = 2, n_cols: int = 4,
    border: int = 2,
) -> np.ndarray:
    """Pick evenly-spaced slices and tile them as a montage."""
    n_total = volume_arr.shape[0]
    n_thumbs = n_rows * n_cols
    indices = np.linspace(0, n_total - 1, n_thumbs, dtype=int)

    thumbs = []
    h, w = volume_arr.shape[1:3]
    target_w = 256
    target_h = int(h * (target_w / w))

    for i in indices:
        img = cv2.cvtColor(volume_arr[i], cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (target_w, target_h))
        cv2.putText(img, f"slice {i}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # find this slice's fluid status
        sr = next((s for s in slices if s.index == i), None)
        if sr and sr.fluid_has_any:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                          (66, 134, 245), 2)
        thumbs.append(img)

    rows = []
    for r in range(n_rows):
        row = thumbs[r * n_cols:(r + 1) * n_cols]
        sep = np.full((row[0].shape[0], border, 3), 255, dtype=np.uint8)
        row_img = row[0]
        for img in row[1:]:
            row_img = np.hstack([row_img, sep, img])
        rows.append(row_img)
    vsep = np.full((border, rows[0].shape[1], 3), 255, dtype=np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, vsep, r])
    return out


def classification_timeline(slices, classes: list[str], height: int = 80) -> np.ndarray:
    """Render a row-per-class barcode of which slice voted for which class."""
    if not slices:
        return np.zeros((height, 100, 3), dtype=np.uint8)
    n = len(slices)
    palette = [
        (66, 135, 245), (66, 245, 161), (245, 134, 66), (245, 66, 156),
        (156, 66, 245), (245, 232, 66), (66, 245, 245), (245, 66, 66),
    ]
    class_to_color = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

    cell_w = max(8, 800 // n)
    img = np.full((height, cell_w * n, 3), 245, dtype=np.uint8)
    for col, sr in enumerate(slices):
        if not sr.classification or "predicted_class" not in sr.classification:
            continue
        cls = sr.classification["predicted_class"]
        color = class_to_color.get(cls, (180, 180, 180))
        cv2.rectangle(img, (col * cell_w, 10), ((col + 1) * cell_w, height - 10),
                      color, -1)
    return img, class_to_color


def build_volume_report(
    volume_path: str,
    analysis: VolumeAnalysis,
    registry: ModelRegistry,
    predictor: OphPredictor,
    clinical_report_md: str,
    output_dir: str | Path,
    classifier_model_name: str = "oct_classifier_octdl",
    patient_context: str = "",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    vol = analysis.volume

    # ── 1. En-face fluid map
    enface_img = render_enface(analysis.enface_fluid)
    save_image(fig_dir / "enface_fluid.png", enface_img)

    # ── 2. Slice montage
    montage = build_slice_montage(analysis.slices, vol.volume)
    save_image(fig_dir / "slice_montage.png", montage)

    # ── 3. Classification timeline
    classes = list(analysis.classification_consensus.keys())
    timeline_img, class_colors = classification_timeline(analysis.slices, classes)
    save_image(fig_dir / "classification_timeline.png", timeline_img)

    # ── 4. Foveal-slice deep dive
    fov = analysis.foveal_slice_idx
    fov_slice_obj = next((s for s in analysis.slices if s.index == fov), None)
    fov_img = vol.slice(fov)
    save_image(fig_dir / "foveal_original.png", fov_img)

    # Grad-CAM on foveal slice
    gradcam_summary = None
    try:
        card = registry.get_card(classifier_model_name)
        model = registry.load_model(classifier_model_name)
        tf = get_classification_transforms(card.input_size, is_training=False)
        img3 = np.stack([fov_img] * 3, axis=-1)
        tensor = tf(image=img3)["image"].unsqueeze(0)
        gradcam = GradCAM(model)
        heat, cls_idx, conf = gradcam.compute(tensor)
        gradcam.remove_hooks()

        heat_full = cv2.resize(heat, (fov_img.shape[1], fov_img.shape[0]))
        overlay = heatmap_overlay(fov_img, heat_full, alpha=0.45)
        boxes = boxes_from_heatmap(heat_full, threshold=0.55)
        overlay_boxed = draw_boxes(
            overlay, boxes, color=(0, 255, 255), thickness=2,
            label=card.class_names[cls_idx] if card.class_names else f"cls{cls_idx}",
        )
        save_image(fig_dir / "foveal_gradcam.png", overlay)
        save_image(fig_dir / "foveal_gradcam_boxes.png", overlay_boxed)
        gradcam_summary = {
            "predicted_class": card.class_names[cls_idx] if card.class_names else str(cls_idx),
            "confidence": conf,
            "boxes": boxes,
        }
    except Exception as e:
        gradcam_summary = {"error": str(e)}

    # Fluid + layer overlays on foveal slice
    fluid_summary = None
    if fov_slice_obj and fov_slice_obj.fluid_mask is not None:
        fluid_overlay = segmentation_overlay(
            fov_img, fov_slice_obj.fluid_mask, FLUID_COLORS, alpha=0.5
        )
        fluid_boxes_dict = boxes_from_mask(
            fov_slice_obj.fluid_mask, class_indices=[1, 2, 3], min_area=20
        )
        fluid_class_names = registry.get_card("oct_fluid_segmentor").class_names
        fluid_name_map = {i: n for i, n in enumerate(fluid_class_names)}

        boxed = fluid_overlay.copy()
        box_list = []
        for cls, boxes in fluid_boxes_dict.items():
            color = FLUID_COLORS[cls]
            name = fluid_name_map.get(cls, f"cls{cls}")
            boxed = draw_boxes(boxed, boxes, color=color, thickness=2, label=name)
            for x, y, w, h in boxes:
                box_list.append({"class": name, "x": x, "y": y, "w": w, "h": h})

        save_image(fig_dir / "foveal_fluid_overlay.png", fluid_overlay)
        save_image(fig_dir / "foveal_fluid_boxes.png", boxed)
        fluid_summary = {
            "class_areas": fov_slice_obj.fluid_class_areas,
            "boxes": box_list,
            "class_names": fluid_class_names,
        }

    layer_summary = None
    if fov_slice_obj and fov_slice_obj.layer_mask is not None:
        layer_overlay = segmentation_overlay(
            fov_img, fov_slice_obj.layer_mask, LAYER_COLORS, alpha=0.45
        )
        save_image(fig_dir / "foveal_layer_overlay.png", layer_overlay)
        layer_summary = {
            "class_areas": fov_slice_obj.layer_class_areas,
            "class_names": registry.get_card("oct_layer_segmentor").class_names,
        }

    # ── 5. Render HTML
    print_html_path = output_dir / "report_print.html"
    html_text = _render_volume_html(
        volume=vol, analysis=analysis,
        clinical_report_md=clinical_report_md,
        patient_context=patient_context,
        gradcam_summary=gradcam_summary,
        fluid_summary=fluid_summary,
        layer_summary=layer_summary,
        class_colors=class_colors,
    )
    print_html_path.write_text(html_text, encoding="utf-8")

    pdf_path = output_dir / "report.pdf"
    try:
        _render_pdf(print_html_path, pdf_path)
    except Exception as e:
        print(f"  [WARN] PDF rendering failed: {e}")
        pdf_path = None

    return {
        "report_html": str(print_html_path),
        "report_pdf": str(pdf_path) if pdf_path else None,
        "figures_dir": str(fig_dir),
        "n_slices": vol.n_slices,
        "foveal_idx": fov,
        "slices_with_fluid": analysis.slice_with_fluid_count,
        "classification_consensus": analysis.classification_consensus,
        "total_fluid_voxels": analysis.total_fluid_voxels,
    }


def _palette_legend(palette: dict, names: list[str]) -> str:
    parts = []
    for i, name in enumerate(names):
        if i == 0:
            continue
        color = palette.get(i)
        if not color:
            continue
        rgb = f"rgb({color[2]},{color[1]},{color[0]})"
        parts.append(
            f'<span class="item"><span class="swatch" style="background:{rgb}"></span>{html.escape(name)}</span>'
        )
    return "".join(parts)


def _render_volume_html(
    volume,
    analysis: VolumeAnalysis,
    clinical_report_md: str,
    patient_context: str,
    gradcam_summary: dict | None,
    fluid_summary: dict | None,
    layer_summary: dict | None,
    class_colors: dict,
) -> str:
    iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = datetime.now().strftime("OCT-VOL-%Y%m%d-%H%M%S")
    image_name = Path(volume.source).name

    meta = volume.metadata or {}
    n_slices = volume.n_slices
    h, w = volume.shape[1:3]

    n_with_fluid = analysis.slice_with_fluid_count
    pct_with_fluid = (n_with_fluid / n_slices * 100) if n_slices else 0
    fov = analysis.foveal_slice_idx

    fluid_total = analysis.total_fluid_voxels
    fluid_total_str = ", ".join(f"{k}: {v:,} px" for k, v in fluid_total.items()) or "none"

    top_class, top_votes = next(iter(analysis.classification_consensus.items()), (None, 0))
    top_pct = (top_votes / n_slices * 100) if n_slices else 0

    consensus_rows = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{v}</td>'
        f'<td class="num">{(v/n_slices*100):.1f}%</td></tr>'
        for k, v in analysis.classification_consensus.items()
    )

    # legend for the timeline barcode
    timeline_legend = " &nbsp;".join(
        f'<span class="item"><span class="swatch" style="background:rgb({c[2]},{c[1]},{c[0]})"></span>{html.escape(k)}</span>'
        for k, c in class_colors.items()
    )

    # ── Cover ──
    cover = f"""
    <section class="cover">
      <div class="cover-header">
        <div class="eyebrow">OCT Volume · Macular Cube Analysis</div>
        <h1>OphAgent OCT Volume Report</h1>
      </div>

      <dl class="cover-meta">
        <dt>Report ID</dt><dd>{report_id}</dd><br>
        <dt>Generated</dt><dd>{iso}</dd><br>
        <dt>Source</dt><dd><code>{html.escape(image_name)}</code></dd><br>
        <dt>Modality</dt><dd>{html.escape(volume.modality)} ({html.escape(meta.get("Manufacturer",""))} {html.escape(meta.get("Model",""))})</dd><br>
        <dt>Volume size</dt><dd>{n_slices} B-scans × {h} × {w} pixels</dd><br>
        <dt>Patient context</dt><dd>{html.escape(patient_context or "—")}</dd>
      </dl>

      <div class="summary-card">
        <h3>Volume Findings at a Glance</h3>
        <div class="summary-row">
          <span class="label">Total slices analyzed</span><span class="value">{n_slices}</span>
        </div>
        <div class="summary-row">
          <span class="label">Slices with detected fluid</span>
          <span class="value">{n_with_fluid} / {n_slices} <small>({pct_with_fluid:.1f}%)</small></span>
        </div>
        <div class="summary-row">
          <span class="label">Top per-slice prediction (consensus)</span>
          <span class="value">{html.escape(str(top_class or "—"))} <small>({top_votes} slices · {top_pct:.0f}%)</small></span>
        </div>
        <div class="summary-row">
          <span class="label">Total fluid voxels (cube)</span><span class="value">{fluid_total_str}</span>
        </div>
        <div class="summary-row">
          <span class="label">Foveal-like slice selected</span><span class="value">#{fov}</span>
        </div>
      </div>

      <div class="disclaimer">
        <b>This report is decision-support output for research and educational use only,</b>
        not a substitute for clinical judgment. Findings are produced by trained
        discriminative models applied per B-scan and aggregated across the cube.
        Voxel counts are pixel-based and not converted to physical units unless DICOM
        spacing was present.
      </div>
    </section>
    """

    # ── Volume overview ──
    section_overview = f"""
    <section class="section">
      <h2>1. Volume Overview</h2>

      <h3>En-face fluid distribution</h3>
      <figure class="figure">
        <img src="figures/enface_fluid.png">
        <figcaption class="figcap"><span class="fignum">Figure 1.</span>
        Z-projection en-face map of detected fluid. Each row corresponds to one
        B-scan (top = first slice, bottom = last), each column is an A-scan
        position. Hotter colors = more fluid pixels along that column.
        </figcaption>
      </figure>

      <h3>Per-slice classification consensus</h3>
      <table class="data">
        <caption>Number of slices voting for each class (classifier: {html.escape(analysis.classifier_name)})</caption>
        <thead><tr><th>Class</th><th class="num">Slice count</th><th class="num">Share</th></tr></thead>
        <tbody>{consensus_rows}</tbody>
      </table>

      <h3>Classification timeline (slice 0 → {n_slices - 1})</h3>
      <div class="legend">{timeline_legend}</div>
      <figure class="figure">
        <img src="figures/classification_timeline.png" style="max-height:30mm;">
        <figcaption class="figcap"><span class="fignum">Figure 2.</span>
        Each column represents one B-scan, colored by its top predicted class.
        </figcaption>
      </figure>

      <h3>Representative slice montage</h3>
      <figure class="figure">
        <img src="figures/slice_montage.png">
        <figcaption class="figcap"><span class="fignum">Figure 3.</span>
        Eight evenly-spaced B-scans across the volume. Blue border = fluid was detected on that slice.
        </figcaption>
      </figure>
    </section>
    """

    # ── Foveal slice deep dive ──
    gradcam_html = ""
    if gradcam_summary and "error" not in gradcam_summary:
        gc_pred = gradcam_summary.get("predicted_class", "—")
        gc_conf = gradcam_summary.get("confidence", 0)
        gc_n = len(gradcam_summary.get("boxes", []))
        gradcam_html = f"""
        <h3>Class-activation visualization on the foveal-like slice (#{fov})</h3>
        <div class="figure-pair">
          <figure class="figure">
            <img src="figures/foveal_gradcam.png">
            <figcaption class="figcap"><span class="fignum">Figure 4.</span>
            Grad-CAM heatmap; classifier predicts <b>{html.escape(str(gc_pred))}</b>
            ({gc_conf*100:.1f}% confidence).</figcaption>
          </figure>
          <figure class="figure">
            <img src="figures/foveal_gradcam_boxes.png">
            <figcaption class="figcap"><span class="fignum">Figure 5.</span>
            Detection boxes from heatmap hotspots ({gc_n} region{"s" if gc_n != 1 else ""}).
            </figcaption>
          </figure>
        </div>
        """

    fluid_html = ""
    if fluid_summary:
        areas = fluid_summary.get("class_areas", {})
        area_rows = "".join(
            f'<tr><td>{html.escape(k)}</td><td class="num">{v:,} px</td></tr>'
            for k, v in areas.items()
        )
        box_rows = "".join(
            f'<tr><td>{html.escape(b["class"])}</td>'
            f'<td class="num">{b["x"]}</td><td class="num">{b["y"]}</td>'
            f'<td class="num">{b["w"]}</td><td class="num">{b["h"]}</td></tr>'
            for b in fluid_summary.get("boxes", [])
        ) or '<tr><td colspan="5" style="text-align:center;color:#9ca3af">No boxes</td></tr>'
        fluid_html = f"""
        <h3>Fluid segmentation on the foveal-like slice</h3>
        <div class="legend">{_palette_legend(FLUID_COLORS, fluid_summary["class_names"])}</div>
        <div class="figure-pair">
          <figure class="figure">
            <img src="figures/foveal_fluid_overlay.png">
            <figcaption class="figcap"><span class="fignum">Figure 6.</span> Fluid mask.</figcaption>
          </figure>
          <figure class="figure">
            <img src="figures/foveal_fluid_boxes.png">
            <figcaption class="figcap"><span class="fignum">Figure 7.</span> Bounding boxes.</figcaption>
          </figure>
        </div>
        <div class="figure-pair" style="margin-top:0;">
          <table class="data" style="flex:1;">
            <caption>Class areas on slice #{fov}</caption>
            <thead><tr><th>Class</th><th class="num">Area</th></tr></thead>
            <tbody>{area_rows}</tbody>
          </table>
          <table class="data" style="flex:1.5;">
            <caption>Detected boxes</caption>
            <thead><tr><th>Class</th><th class="num">x</th><th class="num">y</th><th class="num">w</th><th class="num">h</th></tr></thead>
            <tbody>{box_rows}</tbody>
          </table>
        </div>
        """

    layer_html = ""
    if layer_summary:
        names = layer_summary["class_names"]
        layer_html = f"""
        <h3>Retinal layer overlay on the foveal-like slice</h3>
        <div class="legend">{_palette_legend(LAYER_COLORS, names)}</div>
        <div class="figure-pair">
          <figure class="figure">
            <img src="figures/foveal_original.png">
            <figcaption class="figcap"><span class="fignum">Figure 8.</span> Original B-scan #{fov}.</figcaption>
          </figure>
          <figure class="figure">
            <img src="figures/foveal_layer_overlay.png">
            <figcaption class="figcap"><span class="fignum">Figure 9.</span> Layer segmentation overlay.</figcaption>
          </figure>
        </div>
        """

    section_fovea = f"""
    <section class="section">
      <h2>2. Foveal-like Slice Deep Dive (slice #{fov})</h2>
      {gradcam_html}
      {fluid_html}
      {layer_html}
    </section>
    """

    # ── Clinical narrative ──
    try:
        import markdown as _md
        report_html_body = _md.markdown(clinical_report_md or "", extensions=["tables", "fenced_code"])
    except ImportError:
        report_html_body = "<pre>" + html.escape(clinical_report_md or "") + "</pre>"

    section_clinical = f"""
    <section class="section">
      <h2>3. Clinical Interpretation</h2>
      <div class="report-text">{report_html_body}</div>
    </section>
    """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>OphAgent OCT Volume Report — {html.escape(report_id)}</title>
<style>{PRINT_CSS}</style>
</head><body>
{cover}
{section_overview}
{section_fovea}
{section_clinical}
</body></html>
"""
