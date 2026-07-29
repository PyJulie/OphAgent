"""
Visual report builder for the OCT agent.

Composes:
  - Original image
  - Grad-CAM heatmap + extracted detection boxes (from classifier)
  - Fluid segmentation overlay + per-class bounding boxes
  - Layer segmentation overlay
  - Clinical text (LLM-generated)

Produces a styled HTML report + an output folder with all figures.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..inference.predictor import OphPredictor
from ..inference.model_registry import ModelRegistry
from ..data.preprocessing.transforms import get_classification_transforms
from ..visualization.visualizer import (
    FLUID_COLORS, LAYER_COLORS,
    GradCAM, heatmap_overlay, segmentation_overlay,
    boxes_from_heatmap, boxes_from_mask, draw_boxes, save_image, load_image_gray,
)


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  margin: 0; padding: 32px; background: #f5f6fa; color: #1f2333;
}
.container { max-width: 1200px; margin: 0 auto; }
.header {
  background: linear-gradient(135deg, #2d4a7c 0%, #4a6fa5 100%);
  color: white; padding: 28px 36px; border-radius: 12px;
  margin-bottom: 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.header h1 { margin: 0 0 6px; font-size: 26px; font-weight: 600; }
.header .meta { font-size: 13px; opacity: 0.9; }
.card {
  background: white; border-radius: 12px; padding: 24px 28px;
  margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.card h2 {
  margin: 0 0 16px; font-size: 18px; font-weight: 600; color: #2d4a7c;
  padding-bottom: 10px; border-bottom: 2px solid #e8ecf3;
}
.card h3 { margin: 14px 0 8px; font-size: 15px; color: #1f2333; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.fig { text-align: center; }
.fig img {
  width: 100%; height: auto; border-radius: 8px; border: 1px solid #e0e4ec;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.fig figcaption {
  margin-top: 8px; font-size: 13px; color: #555a6e; font-weight: 500;
}
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
table th, table td {
  padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee;
  font-size: 14px;
}
table th { background: #f5f6fa; font-weight: 600; color: #2d4a7c; }
.tag {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-size: 12px; font-weight: 600;
}
.tag-high { background: #d4f5d4; color: #1d6b1d; }
.tag-low  { background: #f5d4d4; color: #6b1d1d; }
.tag-mid  { background: #f5e8c4; color: #806814; }
.pill {
  display: inline-block; padding: 4px 10px; border-radius: 4px;
  background: #eef1f8; color: #2d4a7c; font-size: 13px; margin-right: 6px;
}
.report-text {
  font-size: 14px; line-height: 1.65; color: #2c2f3e;
}
.report-text h1, .report-text h2 { color: #2d4a7c; }
.report-text h3 { font-size: 15px; color: #444; margin: 14px 0 6px; }
.report-text table { font-size: 13px; }
.bbox-list {
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.bbox-list .pill { font-family: monospace; font-size: 12px; }
.swatch {
  display: inline-block; width: 14px; height: 14px; vertical-align: middle;
  border-radius: 3px; margin-right: 6px; border: 1px solid rgba(0,0,0,0.2);
}
.muted { color: #888; font-size: 12px; }
"""


def _img_to_data_uri(img: np.ndarray, fmt: str = ".png") -> str:
    success, buf = cv2.imencode(fmt, img)
    if not success:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _markdown_to_html(text: str) -> str:
    """Tiny markdown-ish renderer (just for the LLM-generated report)."""
    try:
        import markdown  # if available, use it
        return markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        pass

    # very minimal fallback
    out = html.escape(text)
    # headings
    lines = []
    for line in out.split("\n"):
        if line.startswith("### "):
            lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            lines.append(f"<li>{line[2:]}</li>")
        else:
            lines.append(line)
    return "<br>".join(lines)


def _palette_legend(palette: dict, names: dict, skip_zero: bool = True) -> str:
    parts = []
    for cls, color in palette.items():
        if skip_zero and cls == 0:
            continue
        rgb = f"rgb({color[2]},{color[1]},{color[0]})"  # BGR -> RGB for CSS
        name = names.get(cls, f"class {cls}")
        parts.append(f'<span class="swatch" style="background:{rgb}"></span>{html.escape(name)}')
    return ' &nbsp; '.join(parts)


def build_visual_report(
    image_path: str,
    registry: ModelRegistry,
    predictor: OphPredictor,
    findings: dict[str, Any],
    clinical_report_md: str,
    output_dir: str | Path,
    classifier_model_name: str = "oct_classifier_octdl",
    grad_cam_threshold: float = 0.55,
    fluid_class_names: list[str] = None,
    layer_class_names: list[str] = None,
    patient_context: str = "",
) -> dict[str, Any]:
    """Build the full visual HTML report and save assets to output_dir.

    Args:
        image_path: original OCT image path
        registry / predictor: needed to (re)load the classifier for Grad-CAM
        findings: dict containing earlier model outputs (esp. fluid/layer masks)
        clinical_report_md: markdown-formatted clinical text from the LLM
        output_dir: where to write the report.html + figures/

    Returns:
        dict with paths and a summary of what was generated.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_gray(image_path)
    save_image(fig_dir / "original.png", image)

    # ── 1. Run classifier + Grad-CAM ────────────────────────────────────────
    grad_panel = None
    try:
        card = registry.get_card(classifier_model_name)
        model = registry.load_model(classifier_model_name)
        transform = get_classification_transforms(card.input_size, is_training=False)
        img3 = np.stack([image] * 3, axis=-1)
        tensor = transform(image=img3)["image"].unsqueeze(0)

        gradcam = GradCAM(model)
        heat, cls_idx, conf = gradcam.compute(tensor)
        gradcam.remove_hooks()

        # heatmap is at input_size; resize to original
        heat_full = cv2.resize(heat, (image.shape[1], image.shape[0]))
        overlay = heatmap_overlay(image, heat_full, alpha=0.45)
        boxes = boxes_from_heatmap(heat_full, threshold=grad_cam_threshold)
        overlay_boxed = draw_boxes(
            overlay, boxes, color=(0, 255, 255), thickness=2,
            label=card.class_names[cls_idx] if card.class_names else f"cls{cls_idx}",
        )

        save_image(fig_dir / "gradcam.png", overlay)
        save_image(fig_dir / "gradcam_boxes.png", overlay_boxed)

        grad_panel = {
            "predicted_class": card.class_names[cls_idx] if card.class_names else str(cls_idx),
            "confidence": conf,
            "boxes": boxes,
            "model": classifier_model_name,
        }
    except Exception as e:
        grad_panel = {"error": str(e)}

    # ── 2. Fluid segmentation overlay + boxes ───────────────────────────────
    fluid_panel = None
    try:
        fluid_res = predictor.predict("oct_fluid_segmentor", str(image_path))
        fluid_mask = fluid_res["mask"]
        if fluid_class_names is None:
            fluid_class_names = registry.get_card("oct_fluid_segmentor").class_names
        fluid_name_map = {i: n for i, n in enumerate(fluid_class_names)}

        seg_overlay = segmentation_overlay(image, fluid_mask, FLUID_COLORS, alpha=0.5)
        # bounding boxes for IRF/SRF/PED
        fluid_boxes = boxes_from_mask(fluid_mask, class_indices=[1, 2, 3], min_area=30)

        boxed = seg_overlay.copy()
        box_summary = []
        for cls, boxes in fluid_boxes.items():
            color = FLUID_COLORS[cls]
            name = fluid_name_map.get(cls, f"cls{cls}")
            boxed = draw_boxes(boxed, boxes, color=color, thickness=2, label=name)
            for x, y, w, h in boxes:
                box_summary.append({"class": name, "x": x, "y": y, "w": w, "h": h})

        save_image(fig_dir / "fluid_overlay.png", seg_overlay)
        save_image(fig_dir / "fluid_boxes.png", boxed)

        fluid_panel = {
            "class_areas": fluid_res["class_areas"],
            "boxes": box_summary,
            "class_names": fluid_class_names,
        }
    except Exception as e:
        fluid_panel = {"error": str(e)}

    # ── 3. Layer segmentation overlay ───────────────────────────────────────
    layer_panel = None
    try:
        layer_res = predictor.predict("oct_layer_segmentor", str(image_path))
        layer_mask = layer_res["mask"]
        if layer_class_names is None:
            layer_class_names = registry.get_card("oct_layer_segmentor").class_names
        layer_seg = segmentation_overlay(image, layer_mask, LAYER_COLORS, alpha=0.45)
        save_image(fig_dir / "layer_overlay.png", layer_seg)
        layer_panel = {
            "class_areas": layer_res["class_areas"],
            "class_names": layer_class_names,
        }
    except Exception as e:
        layer_panel = {"error": str(e)}

    # ── 4. Compose HTML ─────────────────────────────────────────────────────
    html_path = output_dir / "report.html"
    html_text = _render_html(
        image_path=image_path,
        findings=findings,
        clinical_report_md=clinical_report_md,
        grad_panel=grad_panel,
        fluid_panel=fluid_panel,
        layer_panel=layer_panel,
        fig_dir=fig_dir,
    )
    html_path.write_text(html_text, encoding="utf-8")

    # ── 5. Render print-optimized HTML, then convert to PDF ─────────────────
    from .pdf_styler import render_print_html

    print_html_path = output_dir / "report_print.html"
    print_html = render_print_html(
        image_path=image_path,
        findings=findings,
        clinical_report_md=clinical_report_md,
        grad_panel=grad_panel,
        fluid_panel=fluid_panel,
        layer_panel=layer_panel,
        patient_context=patient_context,
        fluid_palette=FLUID_COLORS,
        layer_palette=LAYER_COLORS,
    )
    print_html_path.write_text(print_html, encoding="utf-8")

    pdf_path = output_dir / "report.pdf"
    try:
        _render_pdf(print_html_path, pdf_path)
    except Exception as e:
        # Non-fatal: keep HTML output even if PDF rendering fails
        print(f"  [WARN] PDF rendering failed: {e}")
        pdf_path = None

    # ── 6. Save findings JSON next to it ────────────────────────────────────
    (output_dir / "findings.json").write_text(
        json.dumps({
            "image_path": str(image_path),
            "findings": findings,
            "grad_cam": grad_panel,
            "fluid_segmentation": fluid_panel,
            "layer_segmentation": layer_panel,
        }, default=str, indent=2),
        encoding="utf-8",
    )

    return {
        "report_html": str(html_path),
        "report_pdf": str(pdf_path) if pdf_path else None,
        "figures_dir": str(fig_dir),
        "grad_cam": grad_panel,
        "fluid": fluid_panel,
        "layer": layer_panel,
    }


def _render_pdf(html_path: Path, pdf_path: Path) -> None:
    """Render the HTML report to PDF using headless Chromium (playwright).

    Adds header/footer with page numbers via Chromium's display_header_footer.
    """
    header_tpl = """
    <div style="font-family: Helvetica, Arial, sans-serif; font-size: 8pt;
                color: #6b7280; width: 100%; padding: 0 12mm;
                display: flex; justify-content: space-between;">
      <span>OphAgent Analysis Report</span>
      <span class="date"></span>
    </div>
    """
    footer_tpl = """
    <div style="font-family: Helvetica, Arial, sans-serif; font-size: 8pt;
                color: #9ca3af; width: 100%; padding: 0 12mm;
                display: flex; justify-content: space-between;">
      <span>AI-assisted analysis — clinical correlation required</span>
      <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
    </div>
    """

    # ① playwright sync API
    try:
        from playwright.sync_api import sync_playwright

        html_uri = html_path.resolve().as_uri()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context()
            page = ctx.new_page()
            page.goto(html_uri)
            page.emulate_media(media="print")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template=header_tpl,
                footer_template=footer_tpl,
                margin={"top": "20mm", "bottom": "18mm",
                        "left": "0mm", "right": "0mm"},
            )
            browser.close()
        return
    except Exception as e_pw:
        import traceback
        traceback.print_exc()
        last_err = e_pw

    # ② weasyprint (Linux/Mac; on Windows needs GTK)
    try:
        import weasyprint  # type: ignore
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return
    except Exception as e_wp:
        last_err = e_wp

    raise RuntimeError(f"No working HTML→PDF backend: {last_err}")


def _render_html(
    image_path: str,
    findings: dict[str, Any],
    clinical_report_md: str,
    grad_panel: dict | None,
    fluid_panel: dict | None,
    layer_panel: dict | None,
    fig_dir: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = Path(image_path).name

    # Quality
    q = findings.get("quality_assessment") or findings.get("assess_quality") or {}
    quality = q.get("quality", "unknown")
    q_conf = q.get("confidence")

    # Classification (broad)
    cls = findings.get("classify_disease") or findings.get("classification") or {}
    pred = cls.get("predicted_class", "?")
    pred_conf = cls.get("confidence")
    probs = cls.get("probabilities", {})

    # Build prob table
    prob_rows = ""
    if probs:
        for k, v in sorted(probs.items(), key=lambda x: -x[1])[:8]:
            prob_rows += f"<tr><td>{html.escape(k)}</td><td>{v*100:.2f}%</td></tr>"

    # Quality badge
    q_class = {"high": "tag-high", "low": "tag-low"}.get(quality, "tag-mid")

    # Grad-CAM section
    gradcam_section = ""
    if grad_panel and "error" not in grad_panel:
        gc_pred = grad_panel.get("predicted_class", "?")
        gc_conf = grad_panel.get("confidence", 0)
        boxes = grad_panel.get("boxes", [])
        boxes_html = "".join(
            f'<span class="pill">[{x},{y} {w}×{h}]</span>'
            for x, y, w, h in boxes
        ) or '<span class="muted">No high-confidence regions above threshold</span>'
        gradcam_section = f"""
        <div class="card">
          <h2>Grad-CAM Visualization & Detection Boxes</h2>
          <p>Highlights the regions the classifier used for its decision; boxes are
          drawn around connected hotspots ≥ threshold.</p>
          <div class="grid-2">
            <figure class="fig">
              <img src="figures/gradcam.png">
              <figcaption>Grad-CAM heatmap (predicted: <b>{html.escape(str(gc_pred))}</b>, conf {gc_conf*100:.1f}%)</figcaption>
            </figure>
            <figure class="fig">
              <img src="figures/gradcam_boxes.png">
              <figcaption>Detection boxes from heatmap hotspots</figcaption>
            </figure>
          </div>
          <h3>Detected attention regions (x, y, w, h)</h3>
          <div class="bbox-list">{boxes_html}</div>
        </div>
        """

    # Fluid section
    fluid_section = ""
    if fluid_panel and "error" not in fluid_panel:
        names = fluid_panel.get("class_names", [])
        legend = _palette_legend(FLUID_COLORS, dict(enumerate(names)))
        areas = fluid_panel.get("class_areas", {})
        area_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v:,} px</td></tr>"
            for k, v in areas.items()
        )
        boxes = fluid_panel.get("boxes", [])
        box_rows = "".join(
            f'<tr><td>{html.escape(b["class"])}</td>'
            f'<td>{b["x"]}</td><td>{b["y"]}</td>'
            f'<td>{b["w"]}</td><td>{b["h"]}</td></tr>'
            for b in boxes
        )
        if not box_rows:
            box_rows = '<tr><td colspan="5" class="muted">No fluid regions detected</td></tr>'
        fluid_section = f"""
        <div class="card">
          <h2>Fluid Segmentation (RETOUCH model)</h2>
          <p class="muted">Legend: {legend}</p>
          <div class="grid-2">
            <figure class="fig">
              <img src="figures/fluid_overlay.png">
              <figcaption>Pixel-wise fluid mask overlay</figcaption>
            </figure>
            <figure class="fig">
              <img src="figures/fluid_boxes.png">
              <figcaption>Bounding boxes per fluid class</figcaption>
            </figure>
          </div>
          <div class="grid-2">
            <div>
              <h3>Area per class</h3>
              <table><tr><th>Class</th><th>Area</th></tr>{area_rows}</table>
            </div>
            <div>
              <h3>Detected boxes</h3>
              <table>
                <tr><th>Class</th><th>x</th><th>y</th><th>w</th><th>h</th></tr>
                {box_rows}
              </table>
            </div>
          </div>
        </div>
        """

    # Layer section
    layer_section = ""
    if layer_panel and "error" not in layer_panel:
        names = layer_panel.get("class_names", [])
        legend = _palette_legend(LAYER_COLORS, dict(enumerate(names)))
        areas = layer_panel.get("class_areas", {})
        area_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v:,} px</td></tr>"
            for k, v in areas.items()
        )
        layer_section = f"""
        <div class="card">
          <h2>Retinal Layer Segmentation (Duke DME model)</h2>
          <p class="muted">Legend: {legend}</p>
          <div class="grid-2">
            <figure class="fig">
              <img src="figures/original.png">
              <figcaption>Original B-scan</figcaption>
            </figure>
            <figure class="fig">
              <img src="figures/layer_overlay.png">
              <figcaption>Layer mask overlay</figcaption>
            </figure>
          </div>
          <h3>Area per layer</h3>
          <table><tr><th>Layer</th><th>Area</th></tr>{area_rows}</table>
        </div>
        """

    quality_chip = f'<span class="tag {q_class}">{html.escape(quality)}</span>'
    quality_conf = f"({q_conf*100:.2f}%)" if isinstance(q_conf, (int, float)) else ""
    pred_chip = f'<span class="pill">{html.escape(str(pred))}</span>'
    pred_conf_chip = f"<b>{pred_conf*100:.1f}%</b>" if isinstance(pred_conf, (int, float)) else ""

    report_html_body = _markdown_to_html(clinical_report_md or "")

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>OphAgent Report — {html.escape(name)}</title>
<style>{CSS}</style></head><body>
<div class="container">

  <div class="header">
    <h1>OphAgent Analysis Report</h1>
    <div class="meta">Image: <code>{html.escape(name)}</code> &nbsp;·&nbsp; Generated: {now}</div>
  </div>

  <div class="card">
    <h2>Summary</h2>
    <div class="grid-3">
      <div>
        <h3>Image quality</h3>
        {quality_chip} <span class="muted">{quality_conf}</span>
      </div>
      <div>
        <h3>Top prediction</h3>
        {pred_chip} {pred_conf_chip}
      </div>
      <div>
        <h3>Tools used</h3>
        <span class="pill">classifier</span>
        <span class="pill">fluid-seg</span>
        <span class="pill">layer-seg</span>
        <span class="pill">grad-cam</span>
        <span class="pill">vision-caption</span>
      </div>
    </div>
    <h3>Top class probabilities</h3>
    <table><tr><th>Class</th><th>Probability</th></tr>{prob_rows}</table>
  </div>

  {gradcam_section}
  {fluid_section}
  {layer_section}

  <div class="card">
    <h2>Clinical Report</h2>
    <div class="report-text">{report_html_body}</div>
  </div>

  <div class="card">
    <h2>Original Image</h2>
    <figure class="fig" style="max-width:600px; margin:auto;">
      <img src="figures/original.png">
      <figcaption>Source B-scan</figcaption>
    </figure>
  </div>

  <p class="muted" style="text-align:center; margin-top: 20px;">
    Generated by OphAgent. For research and educational use only.
  </p>
</div>
</body></html>
"""
    return html_doc
