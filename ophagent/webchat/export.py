"""
Session export → a single, self-contained, good-looking HTML clinical report.

Design goals:
  * ONE portable file — every image (uploads + generated figures) is inlined
    as a base64 data URI, so the report opens anywhere with no /files server.
  * Clinical-report layout (not chat bubbles): metadata header, image gallery,
    Q&A transcript with properly rendered Markdown tables, a collapsible tool
    trace appendix, and a safety disclaimer.
  * Print-friendly — open in a browser and Ctrl/Cmd+P → clean A4 PDF.

The only server-side dependency is python-markdown (already installed).
"""
from __future__ import annotations

import base64
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import markdown as _md
try:
    import bleach as _bleach
except Exception:  # pragma: no cover - optional dependency fallback
    _bleach = None


# Images larger than this are skipped (kept as a placeholder) so a stray huge
# DICOM-derived PNG can't bloat the report to hundreds of MB.
_MAX_EMBED_BYTES = 12 * 1024 * 1024  # 12 MB

_MIME_BY_SUFFIX = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}


_ALLOWED_TAGS = {
    "a", "abbr", "b", "blockquote", "br", "code", "del", "div", "em", "h1",
    "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "li", "ol", "p", "pre",
    "span", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul",
}
_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "title"],
    "th": ["align"],
    "td": ["align"],
}
_ALLOWED_PROTOCOLS = {"http", "https", "mailto", "data"}


def _render_markdown(text: str) -> str:
    """Markdown → HTML with the same feature set the live UI uses (GFM
    tables, fenced code, line breaks, sane lists)."""
    if not text:
        return ""
    rendered = _md.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
        output_format="html5",
    )
    if _bleach is None:
        return html.escape(rendered)
    return _bleach.clean(
        rendered,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )


def _img_data_uri(abs_path: Path) -> str | None:
    """Read an image file and return a base64 data URI, or None on failure /
    oversize."""
    try:
        if not abs_path.exists() or not abs_path.is_file():
            return None
        size = abs_path.stat().st_size
        if size == 0 or size > _MAX_EMBED_BYTES:
            return None
        mime = _MIME_BY_SUFFIX.get(abs_path.suffix.lower())
        if mime is None:
            return None
        b64 = base64.b64encode(abs_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _inline_file_images(html_fragment: str, project_root: Path,
                        session: Any | None = None) -> str:
    """Replace <img src="/files/REL"> (and bare /files/REL hrefs) in rendered
    HTML with embedded base64 data URIs read from PROJECT_ROOT/REL."""
    if "/files/" not in html_fragment:
        return html_fragment

    def _repl(match: re.Match) -> str:
        prefix, url, suffix = match.group(1), match.group(2), match.group(3)
        rel = url[len("/files/"):]
        # Defang path traversal, then apply the session file boundary.
        try:
            target = (project_root / rel).resolve()
            target.relative_to(project_root.resolve())
            if session is not None:
                target = session.resolve_session_file(target)
        except (ValueError, OSError):
            return match.group(0)
        except Exception:
            return match.group(0)
        data = _img_data_uri(target)
        if data is None:
            return match.group(0)
        return f"{prefix}{data}{suffix}"

    # src="/files/..." or src='/files/...'
    html_fragment = re.sub(
        r'(src=["\'])(/files/[^"\']+)(["\'])', _repl, html_fragment
    )
    return html_fragment


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


# ── tool-trace helpers ─────────────────────────────────────────────────────
def _retsam_headline_rows(content: str) -> list[str]:
    """If a tool message is a retsam result, pull a few key metrics out of
    its llm_headline for the trace table. Best-effort; returns [] otherwise."""
    import json
    try:
        data = json.loads(content)
    except Exception:
        return []
    head = data.get("llm_headline") if isinstance(data, dict) else None
    if not isinstance(head, dict):
        return []
    rows = []
    for line in (head.get("natural_language_summary") or [])[:6]:
        rows.append(_esc(line))
    return rows


def _tool_preview(name: str, content: str) -> str:
    """A compact, human-readable one-liner for the trace table."""
    import json
    try:
        data = json.loads(content)
    except Exception:
        return _esc(content[:160])
    if not isinstance(data, dict):
        return _esc(str(data)[:160])
    if data.get("error"):
        return f'<span class="trace-err">error: {_esc(str(data["error"])[:140])}</span>'
    if data.get("skipped"):
        return f'<span class="trace-skip">skipped — {_esc(data.get("reason",""))[:140]}</span>'
    head = data.get("llm_headline")
    if isinstance(head, dict):
        od = head.get("optic_disc", {}) or {}
        dr = head.get("diabetic_retinopathy_signs", {}) or {}
        return _esc(
            f"vCDR={od.get('vCDR')} ({od.get('glaucoma_morphology_flag','?')}); "
            f"DR hemorrhage×{dr.get('hemorrhage_count',0)}; "
            f"exudate×{dr.get('exudate_count',0)}"
        )
    preds = data.get("predictions", {}) or {}
    for k in ("predicted_class", "category", "primary_diagnosis", "quality"):
        if k in preds:
            conf = data.get("confidence")
            tail = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
            return _esc(f"{preds[k]}{tail}")
    return _esc(str(preds)[:140])


# ── main builder ───────────────────────────────────────────────────────────
def build_session_html(
    session: Any,
    *,
    user: str | None,
    project_root: Path,
    title: str | None = None,
) -> str:
    """Render `session` (an OphSession) into a standalone HTML report string."""
    s = session
    project_root = Path(project_root)

    now = datetime.now(timezone.utc).astimezone()
    gen_ts = now.strftime("%Y-%m-%d %H:%M %Z")

    # ── attached images gallery ──────────────────────────────────────────
    gallery_items: list[str] = []
    for entry in getattr(s.context, "attached_images", []) or []:
        ap = Path(entry.get("path", ""))
        try:
            ap = s.resolve_session_file(ap)
        except Exception:
            data = None
        else:
            data = _img_data_uri(ap)
        cap = f'{_esc(entry.get("filename") or ap.name)} · <b>{_esc(entry.get("modality") or "?")}</b>'
        if data:
            gallery_items.append(
                f'<figure class="shot"><img src="{data}" alt="{_esc(ap.name)}">'
                f'<figcaption>{cap}</figcaption></figure>'
            )
        else:
            gallery_items.append(
                f'<figure class="shot shot-missing"><div class="noimg">image unavailable</div>'
                f'<figcaption>{cap}</figcaption></figure>'
            )
    gallery_html = (
        f'<section class="gallery"><h2>Images</h2><div class="shots">'
        + "".join(gallery_items) + "</div></section>"
        if gallery_items else ""
    )

    # ── transcript + tool trace ──────────────────────────────────────────
    blocks: list[str] = []
    trace_rows: list[dict[str, Any]] = []
    trace_by_id: dict[str, dict[str, Any]] = {}
    step = 0
    for m in s.messages:
        role = m.get("role")
        if role == "user":
            content = m.get("content")
            # content can be a string or a structured list — coerce to text
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            txt = (content or "").strip()
            # Skip internal steering nudges that older sessions persisted
            # with role="user" — they were never typed by the user.
            if txt.startswith("STOP CALLING ") or txt.startswith(
                    "STOP. Your previous reply leaked"):
                continue
            if txt:
                blocks.append(
                    f'<div class="turn user"><div class="role">Question</div>'
                    f'<div class="qtext">{_esc(txt)}</div></div>'
                )
        elif role == "assistant":
            txt = (m.get("content") or "").strip()
            if txt:
                rendered = _inline_file_images(
                    _render_markdown(txt), project_root, session
                )
                blocks.append(
                    f'<div class="turn assistant"><div class="role">OphAgent</div>'
                    f'<div class="answer">{rendered}</div></div>'
                )
            for tc in (m.get("tool_calls") or []):
                fn = (tc.get("function") or {})
                step += 1
                row = {
                    "step": step,
                    "id": tc.get("id"),
                    "name": fn.get("name", ""),
                    "preview": None,
                }
                trace_rows.append(row)
                if row["id"]:
                    trace_by_id[row["id"]] = row
        elif role == "tool":
            # Attach the result preview to the exact tool call. Assistant
            # messages may contain multiple tool calls, and results can arrive
            # interleaved; using the most recent row corrupts exported traces.
            preview = _tool_preview(m.get("name", ""), m.get("content", "") or "")
            row = trace_by_id.get(m.get("tool_call_id"))
            if row is None:
                for cand in reversed(trace_rows):
                    if cand.get("preview") is None and cand.get("name") == m.get("name"):
                        row = cand
                        break
            if row is None:
                step += 1
                trace_rows.append({
                    "step": step,
                    "id": m.get("tool_call_id"),
                    "name": m.get("name", ""),
                    "preview": preview,
                })
            else:
                row["preview"] = preview

    transcript_html = (
        '<section class="transcript"><h2>Analysis</h2>'
        + ("".join(blocks) if blocks else '<p class="muted">No conversation yet.</p>')
        + "</section>"
    )

    trace_html = ""
    if trace_rows:
        trace_body: list[str] = []
        for row in trace_rows:
            preview = row.get("preview") or "<em>calling...</em>"
            trace_body.append(
                f'<tr><td class="step">{row["step"]}</td>'
                f'<td class="tname">{_esc(row.get("name", ""))}</td>'
                f'<td class="tprev">{preview}</td></tr>'
            )
        trace_html = (
            '<section class="appendix"><details><summary>Tool trace '
            f'({len(trace_rows)} calls)</summary>'
            '<table class="trace"><thead><tr><th>#</th><th>Tool</th>'
            '<th>Result</th></tr></thead><tbody>'
            + "".join(trace_body) + "</tbody></table></details></section>"
        )

    # ── metadata header ──────────────────────────────────────────────────
    report_title = title or f"OphAgent report · {s.session_id}"
    meta_rows = [
        ("Session", s.session_id),
        ("Generated", gen_ts),
        ("Model", f"{s.model}  ·  {s.backend}"),
        ("Effort", str(s.effort).upper()),
        ("Modality", getattr(s.context, "current_modality", None) or "—"),
        ("Analyst", user or s.owner or "—"),
    ]
    meta_html = "".join(
        f'<div class="meta-item"><span class="meta-k">{_esc(k)}</span>'
        f'<span class="meta-v">{_esc(v)}</span></div>'
        for k, v in meta_rows
    )

    return _PAGE_TEMPLATE.format(
        title=_esc(report_title),
        css=_REPORT_CSS,
        meta=meta_html,
        gallery=gallery_html,
        transcript=transcript_html,
        trace=trace_html,
        gen_ts=_esc(gen_ts),
    )


# ── presentation ───────────────────────────────────────────────────────────
_REPORT_CSS = """
:root{
  --ink:#1a2332; --muted:#5b6b82; --line:#e3e8f0; --bg:#f4f6fb;
  --accent:#2563c9; --accent-soft:#eaf1fd; --user:#f3f5f9; --ok:#0b8a4b;
  --warn:#b4690e; --err:#c0392b;
}
*{box-sizing:border-box}
html{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
       "Hiragino Sans GB","Microsoft YaHei",Roboto,Helvetica,Arial,sans-serif;
}
.page{ max-width:860px; margin:0 auto; padding:32px 20px 64px; }
.report{ background:#fff; border:1px solid var(--line); border-radius:16px;
  overflow:hidden; box-shadow:0 10px 40px rgba(20,40,80,.06); }

/* header */
.head{ padding:28px 34px; background:linear-gradient(135deg,#1f3a6e,#2563c9);
  color:#fff; }
.brand{ display:flex; align-items:center; gap:10px; font-size:20px;
  font-weight:700; letter-spacing:.2px; }
.brand .eye{ font-size:24px; }
.tagline{ margin:4px 0 0; opacity:.85; font-size:12.5px; }
.meta{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px 22px;
  margin-top:20px; }
.meta-item{ display:flex; flex-direction:column; gap:1px; }
.meta-k{ font-size:10.5px; text-transform:uppercase; letter-spacing:.7px;
  opacity:.7; }
.meta-v{ font-size:13.5px; font-weight:600; word-break:break-word; }

/* body sections */
section{ padding:22px 34px; border-top:1px solid var(--line); }
section h2{ margin:0 0 14px; font-size:13px; text-transform:uppercase;
  letter-spacing:1px; color:var(--accent); }

/* image gallery */
.shots{ display:flex; flex-wrap:wrap; gap:14px; }
.shot{ margin:0; width:200px; }
.shot img{ width:200px; height:200px; object-fit:cover; border-radius:10px;
  border:1px solid var(--line); background:#000; }
.shot figcaption{ font-size:11.5px; color:var(--muted); margin-top:6px;
  text-align:center; }
.shot-missing .noimg{ width:200px; height:200px; border-radius:10px;
  border:1px dashed var(--line); display:flex; align-items:center;
  justify-content:center; color:var(--muted); font-size:12px; }

/* transcript */
.turn{ margin:0 0 18px; }
.turn .role{ font-size:11px; text-transform:uppercase; letter-spacing:.6px;
  color:var(--muted); margin-bottom:5px; font-weight:700; }
.turn.user .qtext{ background:var(--user); border:1px solid var(--line);
  border-radius:10px; padding:10px 14px; white-space:pre-wrap; }
.turn.assistant{ border-left:3px solid var(--accent-soft); padding-left:16px; }
.answer{ }
.answer h1,.answer h2,.answer h3{ color:var(--ink); margin:18px 0 8px;
  padding:0; border:0; text-transform:none; letter-spacing:0; font-size:1.05em; }
.answer h2{ font-size:1.18em; }
.answer p{ margin:8px 0; }
.answer ul,.answer ol{ margin:8px 0; padding-left:22px; }
.answer li{ margin:3px 0; }
.answer code{ background:#eef2f8; padding:1px 5px;
  border-radius:4px; font-size:.88em; }
.answer pre{ background:#0f172a; color:#e2e8f0; padding:12px 14px;
  border-radius:8px; overflow:auto; font-size:12.5px; }
.answer pre code{ background:transparent; color:inherit; padding:0; }
.answer blockquote{ margin:8px 0; padding:6px 14px; border-left:3px solid var(--accent);
  background:var(--accent-soft); color:#23406e; border-radius:0 6px 6px 0; }
.answer img{ max-width:100%; border-radius:8px; border:1px solid var(--line);
  margin:8px 0; }
.answer table{ border-collapse:collapse; width:100%; margin:12px 0;
  font-size:13px; }
.answer th,.answer td{ border:1px solid var(--line); padding:7px 10px;
  text-align:left; vertical-align:top; }
.answer thead th{ background:var(--accent-soft); color:#23406e;
  font-weight:700; }
.answer tbody tr:nth-child(even){ background:#fafbfe; }

/* tool trace appendix */
.appendix details{ border:1px solid var(--line); border-radius:10px;
  padding:4px 14px; }
.appendix summary{ cursor:pointer; font-weight:600; color:var(--muted);
  padding:6px 0; }
table.trace{ border-collapse:collapse; width:100%; margin:8px 0; font-size:12.5px; }
table.trace th,table.trace td{ border-bottom:1px solid var(--line);
  padding:6px 8px; text-align:left; vertical-align:top; }
table.trace .step{ width:32px; color:var(--muted); }
table.trace .tname{ font-family:ui-monospace,Menlo,Consolas,monospace;
  color:var(--accent); white-space:nowrap; }
.trace-err{ color:var(--err); }
.trace-skip{ color:var(--warn); }

.muted{ color:var(--muted); }

/* footer / disclaimer */
.foot{ padding:18px 34px 26px; color:var(--muted); font-size:12px;
  border-top:1px solid var(--line); background:#fafbfe; }
.foot .disc{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
  border-radius:8px; padding:10px 14px; margin-bottom:10px; }

/* floating print button (hidden on print) */
.print-bar{ position:fixed; top:16px; right:16px; }
.print-bar button{ background:var(--accent); color:#fff; border:0;
  border-radius:9px; padding:10px 16px; font-size:13.5px; font-weight:600;
  cursor:pointer; box-shadow:0 4px 14px rgba(37,99,201,.35); }
@media print{
  body{ background:#fff; }
  .page{ padding:0; max-width:none; }
  .report{ border:0; border-radius:0; box-shadow:none; }
  .print-bar{ display:none; }
  section,.foot{ break-inside:avoid; }
  .turn{ break-inside:avoid; }
}
"""

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="print-bar"><button onclick="window.print()">⎙ Save as PDF</button></div>
<div class="page">
  <div class="report">
    <header class="head">
      <div class="brand"><span class="eye">👁</span> OphAgent</div>
      <p class="tagline">Multi-modal ophthalmology AI — CFP · OCT · UWF · FFA</p>
      <div class="meta">{meta}</div>
    </header>
    {gallery}
    {transcript}
    {trace}
    <footer class="foot">
      <div class="disc"><b>For research / decision-support only.</b> This
      AI-generated report is not a medical diagnosis. All findings must be
      confirmed by a qualified ophthalmologist alongside the patient's history
      and examination.</div>
      Generated by OphAgent on {gen_ts}.
    </footer>
  </div>
</div>
</body>
</html>"""
