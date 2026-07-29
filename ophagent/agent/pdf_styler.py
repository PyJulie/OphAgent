"""
PDF-optimized HTML renderer for the OCT Agent visual report.

Designed for print (A4):
  - Cover page with key findings card
  - Section-per-page layout with controlled page breaks
  - Header + footer with page numbers via @page CSS
  - Clean print typography (serif body, sans headings)
  - Clinical color palette
  - Numbered figures and tables
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any


PRINT_CSS = r"""
@page {
  size: A4;
  margin: 18mm 16mm 22mm 16mm;
}

@page :first { margin: 16mm 16mm 16mm 16mm; }

* { box-sizing: border-box; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

body {
  font-family: 'Georgia', 'Times New Roman', serif;
  color: #1f2937;
  margin: 0;
  font-size: 10.5pt;
  line-height: 1.55;
}

h1, h2, h3, h4 { font-family: 'Helvetica', 'Arial', sans-serif; color: #0b2545; }
h1 { font-size: 26pt; font-weight: 700; margin: 0 0 8mm; letter-spacing: -0.5px; }
h2 { font-size: 16pt; font-weight: 600; margin: 0 0 6mm;
     padding-bottom: 2mm; border-bottom: 1.5pt solid #0b2545; }
h3 { font-size: 12pt; font-weight: 600; margin: 5mm 0 2mm; color: #1d3557; }
h4 { font-size: 10.5pt; font-weight: 600; margin: 3mm 0 1mm; color: #374151; }

p { margin: 0 0 3mm; }

.cover {
  page-break-after: always;
}
.cover-header {
  border-bottom: 2pt solid #0b2545;
  padding-bottom: 4mm;
  margin-bottom: 6mm;
}
.cover-header .eyebrow {
  font-family: 'Helvetica', sans-serif; font-size: 8.5pt;
  letter-spacing: 2px; text-transform: uppercase;
  color: #6b7280; margin-bottom: 2mm;
}
.cover-header h1 { font-size: 22pt; margin: 0; }
.cover-meta {
  margin: 0 0 5mm; font-size: 9.5pt; color: #4b5563;
}
.cover-meta dt { font-weight: 600; color: #1f2937;
                  display: inline-block; width: 32mm; }
.cover-meta dd { display: inline; margin: 0; }
.cover-meta dl { margin: 0 0 1.5mm; }

.summary-card {
  border: 1pt solid #cbd5e1;
  border-radius: 4pt;
  padding: 5mm 7mm;
  margin: 0 0 6mm;
  page-break-inside: avoid;
  background: #f8fafc;
}
.summary-card h3 { margin: 0 0 3mm; color: #0b2545;
                   font-size: 10.5pt; text-transform: uppercase; letter-spacing: 1px; }
.summary-row { display: flex; justify-content: space-between;
               padding: 1.5mm 0; border-bottom: 1pt dashed #e2e8f0; font-size: 9.5pt; }
.summary-row:last-child { border-bottom: none; }
.summary-row .label { color: #4b5563; }
.summary-row .value { color: #0b2545; font-weight: 600; }

.preview-box {
  text-align: center;
  border: 1pt solid #cbd5e1;
  border-radius: 4pt;
  padding: 4mm;
  margin: 0 0 6mm;
}
.preview-box img { max-width: 100%; max-height: 60mm;
                    border: 1pt solid #e5e7eb; }
.preview-box .caption { margin-top: 2mm; font-size: 8.5pt; color: #6b7280; font-style: italic; }

.disclaimer {
  border-top: 1pt solid #e5e7eb;
  padding-top: 3mm;
  font-size: 8pt;
  color: #6b7280;
  text-align: justify;
  line-height: 1.4;
}

.section { page-break-before: always; }
.section:first-of-type { page-break-before: auto; }

.figure { text-align: center; margin: 4mm 0; page-break-inside: avoid; }
.figure img {
  max-width: 100%; height: auto;
  border: 1pt solid #d1d5db; border-radius: 2pt;
}
.figure .figcap {
  margin-top: 2mm;
  font-family: 'Helvetica', sans-serif;
  font-size: 8.5pt; color: #4b5563;
  font-style: normal;
}
.figure .figcap .fignum {
  font-weight: 600; color: #0b2545;
}

.figure-pair {
  display: flex; gap: 4mm; align-items: stretch;
  page-break-inside: avoid; margin: 4mm 0;
}
.figure-pair .figure { flex: 1; margin: 0; }

table.data {
  width: 100%; border-collapse: collapse;
  font-family: 'Helvetica', sans-serif; font-size: 9.5pt;
  margin: 3mm 0;
  page-break-inside: avoid;
}
table.data caption {
  text-align: left; font-weight: 600; color: #0b2545;
  margin-bottom: 1.5mm; font-size: 9.5pt;
}
table.data thead th {
  background: #0b2545; color: white;
  font-weight: 600; padding: 1.5mm 3mm;
  font-size: 9pt; letter-spacing: 0.3px;
}
table.data tbody td {
  padding: 1.5mm 3mm;
  border-bottom: 0.5pt solid #e5e7eb;
}
table.data tbody tr:nth-child(even) td { background: #f8fafc; }
table.data tbody tr:last-child td { border-bottom: 1pt solid #cbd5e1; }
table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }

.legend {
  margin: 2mm 0 4mm; font-size: 8.5pt;
  font-family: 'Helvetica', sans-serif; color: #4b5563;
}
.legend .swatch {
  display: inline-block; width: 3mm; height: 3mm;
  border: 0.5pt solid rgba(0,0,0,0.3); vertical-align: middle;
  margin-right: 1mm; border-radius: 1pt;
}
.legend .item { display: inline-block; margin-right: 4mm; }

.report-text {
  font-family: 'Georgia', serif;
  font-size: 10.5pt; line-height: 1.65;
  text-align: justify;
}
.report-text h1, .report-text h2 { page-break-after: avoid; }
.report-text h2 { font-size: 14pt; margin-top: 6mm; border-bottom: none;
                  padding-bottom: 0; }
.report-text h3 { font-size: 11pt; }
.report-text table { font-size: 9.5pt; border-collapse: collapse; width: 100%;
                     margin: 3mm 0; }
.report-text table th, .report-text table td {
  border-bottom: 0.5pt solid #d1d5db; padding: 1.5mm 3mm; text-align: left;
}
.report-text table th { background: #f1f5f9; font-weight: 600; }
.report-text ul, .report-text ol { margin: 2mm 0 3mm 6mm; }
.report-text li { margin: 1mm 0; }

.tag {
  display: inline-block; padding: 1mm 3mm; border-radius: 8pt;
  font-family: 'Helvetica', sans-serif; font-size: 8.5pt;
  font-weight: 600;
}
.tag-high { background: #dcfce7; color: #166534; border: 0.5pt solid #86efac; }
.tag-low  { background: #fee2e2; color: #991b1b; border: 0.5pt solid #fca5a5; }
.tag-mid  { background: #fef3c7; color: #92400e; border: 0.5pt solid #fcd34d; }

.bbox-pills { font-family: 'Courier New', monospace; font-size: 8.5pt;
              color: #475569; }
.bbox-pills span { display: inline-block; padding: 0.5mm 2mm; margin: 0 1mm 1mm 0;
                    background: #eef2f7; border-radius: 2pt; }

.kvgrid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 3mm 8mm;
  margin: 2mm 0;
}
.kvgrid .kv { font-size: 10pt; padding: 1.5mm 0; border-bottom: 0.5pt solid #e5e7eb; }
.kvgrid .kv .k { color: #6b7280; font-family: 'Helvetica', sans-serif;
                  font-size: 9pt; }
.kvgrid .kv .v { color: #0b2545; font-weight: 600; }
"""


def render_print_html(
    image_path: str,
    findings: dict[str, Any],
    clinical_report_md: str,
    grad_panel: dict | None,
    fluid_panel: dict | None,
    layer_panel: dict | None,
    patient_context: str = "",
    fluid_palette: dict | None = None,
    layer_palette: dict | None = None,
) -> str:
    """Render the print-optimized HTML for PDF rendering."""
    from datetime import datetime as _dt
    now = _dt.now()
    iso = now.strftime("%Y-%m-%d %H:%M:%S")
    report_id = now.strftime("OCT-%Y%m%d-%H%M%S")
    image_name = Path(image_path).name

    quality = findings.get("quality_assessment", {}).get("quality") \
        or findings.get("assess_quality", {}).get("quality", "unknown")
    q_conf = findings.get("quality_assessment", {}).get("confidence") \
        or findings.get("assess_quality", {}).get("confidence")
    cls = findings.get("classify_disease") or findings.get("classification") or {}
    pred = cls.get("predicted_class", "—")
    pred_conf = cls.get("confidence")
    probs = cls.get("probabilities", {})

    q_class = {"high": "tag-high", "low": "tag-low"}.get(quality, "tag-mid")
    q_conf_str = f"{q_conf*100:.2f}%" if isinstance(q_conf, (int, float)) else ""
    pred_conf_str = f"{pred_conf*100:.1f}%" if isinstance(pred_conf, (int, float)) else ""

    prob_rows = ""
    for k, v in sorted(probs.items(), key=lambda x: -x[1])[:8]:
        prob_rows += (f'<tr><td>{html.escape(k)}</td>'
                      f'<td class="num">{v*100:.2f}%</td></tr>')

    fig_no = [0]
    def figcap(text: str) -> str:
        fig_no[0] += 1
        return (f'<figcaption class="figcap">'
                f'<span class="fignum">Figure {fig_no[0]}.</span> {html.escape(text)}'
                f'</figcaption>')

    # ── Cover page ──────────────────────────────────────────────────────────
    cover = f"""
    <section class="cover">
      <div class="cover-header">
        <div class="eyebrow">Optical Coherence Tomography · Automated Analysis</div>
        <h1>OphAgent Analysis Report</h1>
      </div>

      <dl class="cover-meta">
        <dt>Report ID</dt><dd>{report_id}</dd><br>
        <dt>Generated</dt><dd>{iso}</dd><br>
        <dt>Source image</dt><dd><code>{html.escape(image_name)}</code></dd><br>
        <dt>Patient context</dt><dd>{html.escape(patient_context or "—")}</dd>
      </dl>

      <div class="preview-box">
        <img src="figures/original.png" alt="OCT B-scan">
        <div class="caption">Source OCT B-scan</div>
      </div>

      <div class="summary-card">
        <h3>Key Findings at a Glance</h3>
        <div class="summary-row">
          <span class="label">Image quality</span>
          <span class="value"><span class="tag {q_class}">{html.escape(quality)}</span> {q_conf_str}</span>
        </div>
        <div class="summary-row">
          <span class="label">Top prediction</span>
          <span class="value">{html.escape(str(pred))} &nbsp; <small>{pred_conf_str}</small></span>
        </div>
        <div class="summary-row">
          <span class="label">Grad-CAM hotspot regions</span>
          <span class="value">{len(grad_panel.get("boxes", [])) if grad_panel and "error" not in grad_panel else 0}</span>
        </div>
        <div class="summary-row">
          <span class="label">Fluid regions detected</span>
          <span class="value">{len(fluid_panel.get("boxes", [])) if fluid_panel and "error" not in fluid_panel else 0}</span>
        </div>
        <div class="summary-row">
          <span class="label">Retinal layers segmented</span>
          <span class="value">{len(layer_panel.get("class_areas", {})) if layer_panel and "error" not in layer_panel else 0}</span>
        </div>
      </div>

      <div class="disclaimer">
        <b>This report is decision-support output for research and educational use only,</b>
        not a substitute for clinical judgment. All findings require review by a qualified
        ophthalmologist before any clinical decision. Clinical text is generated by a
        large language model based on trained model outputs (classification, fluid and
        layer segmentation, vision-language captioning).
      </div>
    </section>
    """

    # ── Classification section ──────────────────────────────────────────────
    if grad_panel and "error" not in grad_panel:
        gc_pred = grad_panel.get("predicted_class", "—")
        gc_conf = grad_panel.get("confidence", 0)
        gc_boxes = grad_panel.get("boxes", [])
        bbox_pills = "".join(
            f'<span>[{x}, {y}, {w}×{h}]</span>'
            for x, y, w, h in gc_boxes
        ) or '<span style="background:#f3f4f6">—</span>'
        gradcam_html = f"""
        <h3>Class-activation Visualization</h3>
        <div class="figure-pair">
          <figure class="figure">
            <img src="figures/gradcam.png">
            {figcap(f"Grad-CAM heatmap from the classifier. Predicted: {gc_pred} ({gc_conf*100:.1f}% confidence).")}
          </figure>
          <figure class="figure">
            <img src="figures/gradcam_boxes.png">
            {figcap("Detection boxes extracted from heatmap hotspots above the activation threshold.")}
          </figure>
        </div>
        <h3>Detected attention regions</h3>
        <div class="bbox-pills">{bbox_pills}</div>
        """
    else:
        gradcam_html = "<p>Grad-CAM visualization unavailable.</p>"

    section_class = f"""
    <section class="section">
      <h2>1. Disease Classification</h2>

      <div class="kvgrid">
        <div class="kv"><div class="k">Predicted class</div><div class="v">{html.escape(str(pred))}</div></div>
        <div class="kv"><div class="k">Confidence</div><div class="v">{pred_conf_str}</div></div>
      </div>

      <table class="data">
        <caption>Top class probabilities</caption>
        <thead><tr><th>Class</th><th class="num">Probability</th></tr></thead>
        <tbody>{prob_rows}</tbody>
      </table>

      {gradcam_html}
    </section>
    """

    # ── Fluid segmentation section ──────────────────────────────────────────
    fluid_palette = fluid_palette or {}
    layer_palette = layer_palette or {}

    def palette_legend(palette: dict, names: list[str]) -> str:
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

    if fluid_panel and "error" not in fluid_panel:
        fluid_names = fluid_panel.get("class_names", [])
        fluid_areas = fluid_panel.get("class_areas", {})
        area_rows = "".join(
            f'<tr><td>{html.escape(k)}</td><td class="num">{v:,} px</td></tr>'
            for k, v in fluid_areas.items()
        )
        fluid_boxes = fluid_panel.get("boxes", [])
        box_rows = "".join(
            f'<tr><td>{html.escape(b["class"])}</td>'
            f'<td class="num">{b["x"]}</td><td class="num">{b["y"]}</td>'
            f'<td class="num">{b["w"]}</td><td class="num">{b["h"]}</td></tr>'
            for b in fluid_boxes
        ) or '<tr><td colspan="5" style="text-align:center; color:#9ca3af;">No fluid detected</td></tr>'

        section_fluid = f"""
        <section class="section">
          <h2>2. Fluid Segmentation</h2>
          <div class="legend">{palette_legend(fluid_palette, fluid_names)}</div>
          <div class="figure-pair">
            <figure class="figure">
              <img src="figures/fluid_overlay.png">
              {figcap("Pixel-wise fluid mask overlaid on the B-scan.")}
            </figure>
            <figure class="figure">
              <img src="figures/fluid_boxes.png">
              {figcap("Bounding boxes for each detected fluid class.")}
            </figure>
          </div>

          <table class="data">
            <caption>Class areas (pixels)</caption>
            <thead><tr><th>Class</th><th class="num">Area</th></tr></thead>
            <tbody>{area_rows}</tbody>
          </table>

          <table class="data">
            <caption>Detected bounding boxes</caption>
            <thead><tr><th>Class</th><th class="num">x</th><th class="num">y</th>
                       <th class="num">w</th><th class="num">h</th></tr></thead>
            <tbody>{box_rows}</tbody>
          </table>
        </section>
        """
    else:
        section_fluid = ""

    # ── Layer segmentation section ──────────────────────────────────────────
    if layer_panel and "error" not in layer_panel:
        layer_names = layer_panel.get("class_names", [])
        layer_areas = layer_panel.get("class_areas", {})
        area_rows = "".join(
            f'<tr><td>{html.escape(k)}</td><td class="num">{v:,} px</td></tr>'
            for k, v in layer_areas.items()
        )
        section_layer = f"""
        <section class="section">
          <h2>3. Retinal Layer Segmentation</h2>
          <div class="legend">{palette_legend(layer_palette, layer_names)}</div>
          <div class="figure-pair">
            <figure class="figure">
              <img src="figures/original.png">
              {figcap("Original B-scan.")}
            </figure>
            <figure class="figure">
              <img src="figures/layer_overlay.png">
              {figcap("Predicted retinal layer mask.")}
            </figure>
          </div>

          <table class="data">
            <caption>Area per segmented layer (pixels)</caption>
            <thead><tr><th>Region</th><th class="num">Area</th></tr></thead>
            <tbody>{area_rows}</tbody>
          </table>
        </section>
        """
    else:
        section_layer = ""

    # ── Clinical narrative ──────────────────────────────────────────────────
    try:
        import markdown as _md
        report_html_body = _md.markdown(clinical_report_md or "", extensions=["tables", "fenced_code"])
    except ImportError:
        report_html_body = "<pre>" + html.escape(clinical_report_md or "") + "</pre>"

    section_clinical = f"""
    <section class="section">
      <h2>4. Clinical Interpretation</h2>
      <div class="report-text">{report_html_body}</div>
    </section>
    """

    # ── Compose final HTML ──────────────────────────────────────────────────
    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>OphAgent Report — {html.escape(report_id)}</title>
<style>
@page {{ counter-reset: page 1; }}
html {{ string-set: genDate "{iso}"; }}
{PRINT_CSS}
</style>
</head><body>
{cover}
{section_class}
{section_fluid}
{section_layer}
{section_clinical}
</body></html>
"""
    return doc
