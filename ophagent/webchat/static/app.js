// OphAgent Web client.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const chatEl = $("#chat");
const chatScroll = $("#chat-scroll");
const sessionList = $("#session-list");
const promptEl = $("#prompt");
const sendBtn = $("#send-btn");
const attachBtn = $("#attach-btn");
const fileInput = $("#file-input");
const attachmentsEl = $("#attachments");
const composer = $("#composer");
const composerWrap = $(".composer-wrap");
const dropHint = $("#drop-hint");
const newChatBtn = $("#new-chat");
const historySidebar = $("#history-sidebar");
const sidebarToggle = $("#sidebar-toggle");
const sidebarClose = $("#sidebar-close");
const sidebarBackdrop = $("#sidebar-backdrop");

const ctxImage = $("#ctx-image");
const ctxVolume = $("#ctx-volume");
const lastReportEl = $("#last-report");
const exportBtn = $("#export-btn");
const chatTitle = $("#chat-title");
const personalizationSummary = $("#personalization-summary");
const effortSeg = $("#effort-seg");
const viewSeg = $("#view-seg");
const userPill = $("#user-pill");

// ── View mode: how trace + reply are laid out ─────────────────────────
// "clean"  -> collapsed trace above reply (default)
// "inline" -> expanded trace above reply (debug-style)
// Persisted per-browser in localStorage; applied as body class.
const VIEW_MODES = ["clean", "split", "inline"];
let turnSeq = 0;
function nextTurnId() {
  turnSeq += 1;
  return `turn-${turnSeq}`;
}
function reflowTraceReplyOrder(mode = getViewMode()) {
  if (!chatEl) return;
  chatEl.querySelectorAll(".trace-msg[data-turn-id]").forEach((traceWrap) => {
    const id = traceWrap.dataset.turnId;
    if (!id) return;
    const replyWrap = chatEl.querySelector(`.reply-msg[data-turn-id="${id}"]`);
    if (!replyWrap || replyWrap.parentNode !== traceWrap.parentNode) return;
    if (mode === "clean" || mode === "inline") {
      if (traceWrap.nextSibling !== replyWrap) {
        traceWrap.parentNode.insertBefore(traceWrap, replyWrap);
      }
    } else {
      if (replyWrap.nextSibling !== traceWrap) {
        traceWrap.parentNode.insertBefore(traceWrap, replyWrap.nextSibling);
      }
    }
  });
}
function getViewMode() {
  const m = localStorage.getItem("ophagent.view") || "clean";
  return VIEW_MODES.includes(m) ? m : "clean";
}
function applyViewMode(mode) {
  if (!VIEW_MODES.includes(mode)) mode = "clean";
  localStorage.setItem("ophagent.view", mode);
  document.body.classList.remove("view-clean", "view-inline", "view-split");
  document.body.classList.add("view-" + mode);
  reflowTraceReplyOrder(mode);
  if (viewSeg) {
    viewSeg.querySelectorAll(".view-mode-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === mode));
  }
  // Sync the evidence-pane content for split mode (or clear it).
  if (mode === "split") rebuildEvidencePane();
}
// Initial application — runs as soon as the script loads.
applyViewMode(getViewMode());
// Wire the segmented control.
if (viewSeg) {
  viewSeg.addEventListener("click", (e) => {
    const btn = e.target.closest(".view-mode-btn");
    if (btn && btn.dataset.view) applyViewMode(btn.dataset.view);
  });
}
// Make finished trace heads clickable to toggle .expanded.
document.addEventListener("click", (e) => {
  const head = e.target.closest(".trace.done .trace-head");
  if (!head) return;
  const trace = head.closest(".trace");
  if (trace) trace.classList.toggle("expanded");
});

// Make each step row clickable to reveal its per-tool details panel.
// Catches both running (live) and done (finished) steps.
document.addEventListener("click", (e) => {
  const row = e.target.closest(".step .step-row");
  if (!row) return;
  const step = row.closest(".step");
  if (!step) return;
  const details = step.querySelector(".step-details");
  if (!details) return;
  // Empty details → nothing to show
  if (!details.children.length) return;
  const open = step.classList.toggle("step-open");
  details.hidden = !open;
  const expander = row.querySelector(".step-expander");
  if (expander) expander.textContent = open ? "▼" : "▶";
});

// Build the per-step drill-down content: figure thumbnails + a compact
// JSON view of the tool's predictions (or error). Truncates obvious
// noise (mask file paths, base64 blobs, very long arrays) so the user
// gets a readable summary not a 50 KB dump.
function populateStepDetails(detailsEl, ev) {
  if (!detailsEl) return;
  detailsEl.innerHTML = "";

  // 1) Error block (if any) — show prominently
  if (ev.error) {
    const errBlock = document.createElement("div");
    errBlock.className = "step-detail-error";
    errBlock.textContent = "Error: " + String(ev.error).slice(0, 600);
    detailsEl.appendChild(errBlock);
  }

  // 1b) Human-readable model output (LLM/text tools: vision_impression's
  //     gestalt read + differential, verify_findings' recommendation).
  if (ev.detail_md && String(ev.detail_md).trim()) {
    const sec = document.createElement("div");
    sec.className = "step-md";
    const head = document.createElement("div");
    head.className = "step-section-head";
    head.textContent = "Model read";
    sec.appendChild(head);
    const body = document.createElement("div");
    body.className = "step-md-body";
    body.innerHTML = md(String(ev.detail_md));
    body.querySelectorAll("a").forEach((a) => (a.target = "_blank"));
    sec.appendChild(body);
    detailsEl.appendChild(sec);
  }

  // 2) Figures — render as clickable thumbnails opening in new tab
  if (ev.figure_urls && Object.keys(ev.figure_urls).length) {
    const figs = document.createElement("div");
    figs.className = "step-figs";
    const head = document.createElement("div");
    head.className = "step-section-head";
    head.textContent = "Figures";
    figs.appendChild(head);
    const grid = document.createElement("div");
    grid.className = "step-fig-grid";
    for (const [name, url] of Object.entries(ev.figure_urls)) {
      const a = document.createElement("a");
      a.href = url; a.target = "_blank"; a.title = name;
      a.className = "step-fig";
      a.innerHTML = `<img alt="${escHtml(name)}"><div class="step-fig-name">${escHtml(name)}</div>`;
      // Auth-aware blob load — bare <img src> fails 401 silently in
      // Chrome on Basic-Auth /files paths.
      const im = a.querySelector("img");
      fetchAsBlobUrl(url).then((b) => { if (im && b) im.src = b; });
      grid.appendChild(a);
    }
    figs.appendChild(grid);
    detailsEl.appendChild(figs);
  }

  // 3) Predictions — render as a syntax-highlighted JSON block (capped)
  if (ev.predictions !== undefined && ev.predictions !== null) {
    const preds = document.createElement("div");
    preds.className = "step-preds";
    const head = document.createElement("div");
    head.className = "step-section-head";
    head.textContent = "Key predictions";
    preds.appendChild(head);
    const pre = document.createElement("pre");
    pre.className = "step-json";
    let txt;
    try {
      txt = JSON.stringify(_compactForUi(ev.predictions), null, 2);
    } catch {
      txt = String(ev.predictions);
    }
    if (txt.length > 4000) txt = txt.slice(0, 4000) + "\n… (truncated)";
    pre.textContent = txt;
    preds.appendChild(pre);
    detailsEl.appendChild(preds);
  }

  // If empty, give a quiet placeholder
  if (!detailsEl.children.length) {
    const empty = document.createElement("div");
    empty.className = "step-detail-empty";
    empty.textContent = "No additional structured output for this step.";
    detailsEl.appendChild(empty);
  }
}

// Strip large arrays / known noisy fields before showing predictions
// in the UI. Recursive.
function _compactForUi(v, depth = 0) {
  if (depth > 6) return "<deep>";
  if (Array.isArray(v)) {
    if (v.length > 12) return [...v.slice(0, 12).map((x) => _compactForUi(x, depth + 1)),
                                  `… (${v.length - 12} more)`];
    return v.map((x) => _compactForUi(x, depth + 1));
  }
  if (v && typeof v === "object") {
    const out = {};
    for (const k of Object.keys(v)) {
      // Drop noise keys
      if (["mask_files", "components", "raw_pixel_data",
            "_thresholds", "overlay_files"].includes(k)) {
        const inner = v[k];
        out[k] = Array.isArray(inner) ? `<${inner.length} items elided>`
                  : typeof inner === "object" ? `<elided>` : inner;
        continue;
      }
      out[k] = _compactForUi(v[k], depth + 1);
    }
    return out;
  }
  if (typeof v === "string" && v.length > 200) {
    return v.slice(0, 200) + "… (truncated)";
  }
  return v;
}

// ── Evidence pane support (used by Split mode) ────────────────────────
// In split mode the inline .trace blocks are hidden via CSS and a cloned
// list of tool steps is shown in #evidence-list. Re-rendered when the
// user switches into split mode, and when a new trace finishes.
function rebuildEvidencePane() {
  const list = document.getElementById("evidence-list");
  const empty = document.querySelector("#evidence-pane .evidence-empty");
  if (!list) return;
  list.innerHTML = "";
  const traces = document.querySelectorAll(".chat .trace");
  if (traces.length === 0) {
    if (empty) empty.classList.remove("hidden");
    return;
  }
  if (empty) empty.classList.add("hidden");
  traces.forEach((t, idx) => {
    const card = document.createElement("div");
    card.className = "evidence-card";
    const head = document.createElement("div");
    head.className = "evidence-card-head";
    const steps = t.querySelectorAll(".step");
    const ok = t.querySelectorAll(".step.done").length;
    const fail = t.querySelectorAll(".step.error").length;
    head.textContent = `Turn ${idx + 1}  ·  ${steps.length} steps · ${ok} ok · ${fail} fail`;
    card.appendChild(head);
    const body = document.createElement("div");
    body.className = "evidence-card-body";
    // Clone each step into the side panel
    steps.forEach((s) => {
      const row = document.createElement("div");
      row.className = "evidence-step";
      const icon = s.querySelector(".step-icon")?.textContent || "•";
      const name = s.querySelector(".step-name")?.textContent || "";
      const detail = s.querySelector(".step-detail")?.textContent || "";
      row.innerHTML = `<span class="ev-icon">${escHtml(icon)}</span>
                        <span class="ev-name">${escHtml(name)}</span>
                        <span class="ev-detail">${escHtml(detail)}</span>`;
      body.appendChild(row);
    });
    card.appendChild(body);
    list.appendChild(card);
  });
}
// Auto-rebuild evidence pane every time a new trace finishes (rough
// trigger: poll after each tool_result via the existing event stream).
// We expose a helper that the trace controller can call; see onEvent.
window.__rebuildEvidence = rebuildEvidencePane;

function applyUserPill(you, isAdmin) {
  if (!userPill) return;
  if (!you) { userPill.hidden = true; return; }
  userPill.hidden = false;
  userPill.innerHTML = `<span class="who">${escHtml(you)}</span>` +
    (isAdmin ? '<span class="role">· admin</span>' : '');
  userPill.title = `Signed in as ${you}${isAdmin ? ' (runtime administrator)' : ''}`;
}

let state = {
  sessionId: null,
  // Multi-attachment: list of pending files to send with the next message.
  // Each: {filename, path, kind, modality}
  pendingAttachments: [],
};

/**
 * Per-session state — keyed by session_id.
 *   {
 *     messages: [...],
 *     chatHTML: "<html cache for the chat scroll>",
 *     activeTrace: traceController | null,
 *     running: bool,
 *     context: {current_image, current_volume},
 *     lastReport: {pdf, html, dir} | null,
 *   }
 */
const sessions = new Map();

function getSessionState(sid) {
  if (!sessions.has(sid)) {
    sessions.set(sid, {
      messages: [], chatHTML: null, activeTrace: null,
      running: false, context: null, lastReport: null,
      effort: "low", runtime: null,
    });
  }
  return sessions.get(sid);
}

// ── markdown rendering ─────────────────────────────────────────────────
if (window.marked?.setOptions) {
  window.marked.setOptions({
    breaks: true,
    gfm: true,
  });
}
function md(text) {
  if (!text) return "";
  if (window.marked?.parse && window.DOMPurify?.sanitize) {
    return window.DOMPurify.sanitize(window.marked.parse(text), {
      ADD_ATTR: ["target"],
    });
  }
  return `<p>${escHtml(text).replace(/\n/g, "<br>")}</p>`;
}

function escHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function safeClassToken(value) {
  return String(value ?? "").replace(/[^A-Za-z0-9_-]/g, "_");
}

// ── tool pretty-names (shared by the live trace AND history replay) ─────
const TOOL_PRETTY = {
  // session / generic
  set_current_image: "Loading image",
  set_current_volume: "Loading volume",
  detect_modality: "Detecting modality",
  vision_impression: "Visual impression",
  analyze_image: "Routine analysis",
  verify_findings: "Cross-checking findings",
  compute: "Computing derived metric",
  // CFP
  cfp_eyeq: "Image quality (EyeQ)",
  cfp_efiqa: "Image quality (EFIQA)",
  cfp_quality_robust: "Image quality (robust)",
  cfp_dr_workup: "DR workup",
  cfp_pdr_cascade: "PDR cascade",
  cfp_dynamic_clip: "Dynamic CLIP",
  cfp_clip_ensemble: "CLIP ensemble (3 models)",
  cfp_clip_multi_disease: "CLIP — ViLReF",
  cfp_retizero: "CLIP — RetiZero",
  cfp_flair: "CLIP — FLAIR",
  cfp_glaucoma_workup: "Glaucoma workup",
  cfp_glaucoma: "Glaucoma classifier",
  cfp_od_detection: "Optic disc detection",
  cfp_retsam_segmentation: "Full-fundus segmentation",
  cfp_paired5: "Joint 5-class classifier",
  // OCT
  oct_quality: "OCT image quality",
  oct_fmue_16class: "OCT 16-class diagnosis",
  oct_fluid_segmentation: "Segmenting fluid (IRF/SRF/PED)",
  oct_layer_segmentation: "Segmenting retinal layers",
  oct_volume_disc: "Analysing disc volume",
  // UWF / FFA / cross
  uwf_multi_disease: "UWF multi-disease",
  uwf_vessel_segmentation: "UWF vessel segmentation",
  ffa_classification: "FFA classification",
  ffa_paired5: "FFA joint classifier",
  cross_cfp_oct: "Cross CFP+OCT",
  cross_cfp_ffa: "Cross CFP+FFA",
  paired_bilingual_report: "Bilingual report",
};
function prettyTool(name) { return TOOL_PRETTY[name] || name; }

// ── API ────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const t = await r.text();
    let detail = t;
    try {
      const parsed = JSON.parse(t);
      detail = typeof parsed.detail === "string" ? parsed.detail : t;
    } catch (_) {
      // Non-JSON responses (for example a reverse-proxy error) stay readable.
    }
    throw new Error(detail ? `${r.status} ${detail}` : `Request failed (${r.status})`);
  }
  return r.json();
}

async function createSession() {
  const data = await api("/api/sessions", { method: "POST" });
  return data;
}

async function listSessions() {
  return api("/api/sessions");
}

async function loadSession(sid) {
  return api(`/api/sessions/${sid}`);
}

async function sendChat(sid, text) {
  return api(`/api/sessions/${sid}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

async function sendChatStream(sid, text, onEvent) {
  const resp = await fetch(`/api/sessions/${sid}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, nl);
      buf = buf.slice(nl + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) {
          try { onEvent(JSON.parse(line.slice(6))); }
          catch (e) { /* ignore parse errors */ }
        }
      }
    }
  }
}

async function uploadFile(sid, file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("kind", "auto");
  const r = await fetch(`/api/sessions/${sid}/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function deleteSession(sid) {
  return api(`/api/sessions/${sid}`, { method: "DELETE" });
}

// ── Render ─────────────────────────────────────────────────────────────
function clearChat() {
  chatEl.innerHTML = "";
}

// Replay a saved conversation so it looks the SAME as the live run:
// consecutive tool-call / tool-result messages are collapsed into a single
// finished "trace" panel (✓ steps + result previews), exactly like the live
// SSE trace — instead of a scatter of raw "→ name" / "name ✓" pills.
function renderMessages(messages) {
  clearChat();
  let pending = [];   // [{name, preview}] accumulated for the current trace
  let pendingTurnId = null;

  const flush = () => {
    if (!pending.length) return null;
    const traceWrap = appendFinishedTrace(pending, pendingTurnId);
    pending = [];
    pendingTurnId = null;
    return traceWrap;
  };

  for (const m of messages) {
    const role = m.role;
    if (role === "user") {
      flush();
      // Re-attach any image thumbnails the server resolved for this turn
      // (so a reloaded session shows the picture, matching the live view).
      if (m.images && m.images.length) {
        for (const im of m.images) {
          appendImagePreview("user", im.thumb,
            (im.modality ? "[" + im.modality + "] " : "") + (im.name || ""));
        }
      }
      appendBubble("user", m.content);
    } else if (role === "assistant") {
      // An assistant turn may carry tool_calls and/or final text.
      if (m.tool_calls && m.tool_calls.length) {
        if (!pendingTurnId) pendingTurnId = nextTurnId();
        for (const tc of m.tool_calls) pending.push({ name: tc.name, preview: null });
      }
      if (m.content && m.content.trim()) {
        const traceWrap = flush();      // close the trace before the answer
        if (traceWrap) appendAssistantAfter(traceWrap, m.content);
        else appendBubble("assistant", m.content);
      }
    } else if (role === "tool") {
      // Attach the result preview + drill-down data to the first matching
      // open step (so the reloaded trace is expandable like the live one).
      const detail = {
        preview: m.preview || "",
        predictions: m.predictions,
        figure_urls: m.figure_urls,
        error: m.error,
        detail_md: m.detail_md,
      };
      const step = pending.find(s => s.name === m.name && s.preview === null);
      if (step) Object.assign(step, detail);
      else pending.push({ name: m.name, ...detail });
    }
  }
  flush();
  reflowTraceReplyOrder(getViewMode());
  scrollToBottom();
}

// Used when a single message arrives live (rare path). Kept for compatibility.
function renderMessage(m, animate = true) {
  const role = m.role;
  if (role === "user") {
    appendBubble("user", m.content);
  } else if (role === "assistant") {
    if (m.content && m.content.trim()) appendBubble("assistant", m.content);
    if (m.tool_calls && m.tool_calls.length) {
      appendFinishedTrace(m.tool_calls.map(tc => ({ name: tc.name, preview: null })));
    }
  } else if (role === "tool") {
    appendFinishedTrace([{ name: m.name, preview: m.preview || "" }]);
  }
}

// Render a finished (non-animated) trace panel matching the live one.
function appendFinishedTrace(steps, turnId = null) {
  if (!steps || !steps.length) return;
  const wrap = document.createElement("div");
  wrap.className = "msg assistant trace-msg";
  if (turnId) wrap.dataset.turnId = turnId;
  const trace = document.createElement("div");
  trace.className = "trace done";

  const stepsEl = document.createElement("div");
  stepsEl.className = "trace-steps";
  for (const s of steps) {
    // Build the SAME structure the live trace uses (step-row + expander +
    // hidden step-details) so the global click handler makes it expandable.
    const step = document.createElement("div");
    step.className = "step done";
    step.innerHTML = `
      <div class="step-row">
        <span class="step-icon">✓</span>
        <span class="step-name"></span>
        <span class="step-detail"></span>
        <span class="step-expander" title="Click to expand details">▶</span>
      </div>
      <div class="step-details" hidden></div>`;
    step.querySelector(".step-name").textContent = prettyTool(s.name);
    const det = step.querySelector(".step-detail");
    if (s.preview) det.textContent = "— " + s.preview;   // textContent = auto-escaped
    // Populate the drill-down from the carried fields (same renderer as live).
    populateStepDetails(step.querySelector(".step-details"), {
      preview: s.preview,
      predictions: s.predictions,
      figure_urls: s.figure_urls,
      error: s.error,
      detail_md: s.detail_md,
    });
    stepsEl.appendChild(step);
  }

  const head = document.createElement("div");
  head.className = "trace-head";
  const status = document.createElement("span");
  status.className = "trace-status";
  status.innerHTML = `<span style="color:var(--success);">✓</span> ${steps.length} step${steps.length > 1 ? "s" : ""}`;
  head.appendChild(status);

  trace.append(head, stepsEl);
  wrap.appendChild(trace);
  chatEl.appendChild(wrap);
  return wrap;
}

function appendBubble(role, content) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = md(content);
  // Open links in new tab
  bubble.querySelectorAll("a").forEach((a) => (a.target = "_blank"));
  // Auth-aware load for any /files/ images embedded in the markdown.
  rewriteFileImagesToBlob(bubble);
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function appendImagePreview(role, url, name) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = `<img alt="${escHtml(name)}"><div class="muted" style="font-size:12px; color:var(--text-secondary); margin-top:4px;">${escHtml(name)}</div>`;
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  // Load via fetch (credentials: include) → blob URL. The bare <img src>
  // path doesn't always re-send Basic Auth on Chrome, so a public-served
  // image at /files/... can fail with 401 silently. Fetching by JS forces
  // the credentials to be attached.
  fetchAsBlobUrl(url).then((blobUrl) => {
    const im = bubble.querySelector("img");
    if (im && blobUrl) im.src = blobUrl;
  }).catch(() => {});
  scrollToBottom();
}

// ── Auth-aware image loader ────────────────────────────────────────────
// Fetches an image URL with credentials and returns an object-URL the
// browser can show in an <img> tag without further auth round-trips.
// Cached per-URL so repeat-render of the same upload doesn't refetch.
const _blobUrlCache = new Map();
async function fetchAsBlobUrl(url) {
  if (_blobUrlCache.has(url)) return _blobUrlCache.get(url);
  try {
    const resp = await fetch(url, { credentials: "include" });
    if (!resp.ok) return null;
    const blob = await resp.blob();
    const objectUrl = URL.createObjectURL(blob);
    _blobUrlCache.set(url, objectUrl);
    return objectUrl;
  } catch (e) {
    return null;
  }
}

// Post-process a freshly-rendered markdown container: find any <img> that
// points to a /files/... URL and reload it via fetchAsBlobUrl so it
// renders even when the browser drops Basic-Auth on async image loads.
function rewriteFileImagesToBlob(rootEl) {
  if (!rootEl) return;
  rootEl.querySelectorAll('img').forEach((im) => {
    const src = im.getAttribute('src');
    if (!src) return;
    // Match relative or absolute /files/ URLs
    if (src.startsWith('/files/') || /^https?:\/\/[^/]+\/files\//.test(src)) {
      im.removeAttribute('src');
      fetchAsBlobUrl(src).then((b) => { if (b) im.src = b; });
    }
  });
}

function appendToolBadge(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg tool";
  const tag = document.createElement("div");
  tag.className = "tool-call";
  tag.innerHTML = `<span class="tool-name">${escHtml(text)}</span>`;
  wrap.appendChild(tag);
  chatEl.appendChild(wrap);
}

function appendTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant typing-wrap";
  wrap.innerHTML = `<div class="msg-bubble" style="padding: 4px 8px;"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chatEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

/**
 * Render a horizontal scrolling gallery of tool-produced figures.
 */
function appendFigureGallery(toolName, urls) {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `
    <div class="figure-gallery">
      <div class="figure-gallery-head">${escHtml(toolName)} — figures</div>
      <div class="figure-gallery-scroll"></div>
    </div>
  `;
  const scroll = wrap.querySelector(".figure-gallery-scroll");
  for (const [label, url] of Object.entries(urls)) {
    if (!url) continue;
    const fig = document.createElement("figure");
    fig.className = "fig-thumb";
    fig.innerHTML = `
      <a href="${escHtml(url)}" target="_blank"><img alt="${escHtml(label)}"></a>
      <figcaption>${escHtml(label)}</figcaption>
    `;
    const im = fig.querySelector("img");
    fetchAsBlobUrl(url).then((b) => { if (im && b) im.src = b; });
    scroll.appendChild(fig);
  }
  chatEl.appendChild(wrap);
  scrollToBottom();
}


/**
 * Append a live "trace" panel that gets populated as SSE events arrive.
 * Returns a controller object so the caller can attach events to it.
 */
function appendTrace() {
  const turnId = nextTurnId();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant trace-msg";
  wrap.dataset.turnId = turnId;
  const trace = document.createElement("div");
  trace.className = "trace";
  trace.innerHTML = `
    <div class="trace-head">
      <div class="spinner"></div>
      <span class="trace-status">Thinking…</span>
    </div>
    <div class="trace-steps"></div>
  `;
  wrap.appendChild(trace);
  chatEl.appendChild(wrap);
  scrollToBottom();

  const statusEl = trace.querySelector(".trace-status");
  const stepsEl = trace.querySelector(".trace-steps");
  const startedAt = Date.now();
  let lastStep = null; // {name, label, kind, el, startedAt}
  let thinkingCount = 0;
  let toolCount = 0;

  const pretty = prettyTool;
  function elapsed() {
    const sec = (Date.now() - startedAt) / 1000;
    return sec < 60 ? `${sec.toFixed(1)}s`
                    : `${Math.floor(sec/60)}m ${Math.floor(sec%60)}s`;
  }

  function setStepElapsed(step) {
    if (!step || !step.el) return;
    const stepSec = ((Date.now() - step.startedAt) / 1000).toFixed(1);
    const elapsedEl = step.el.querySelector(".step-elapsed");
    if (elapsedEl) elapsedEl.textContent = `${stepSec}s`;
  }

  function finishStep(cssClass = "done", iconText = "✓") {
    if (!lastStep || !lastStep.el) return;
    setStepElapsed(lastStep);
    lastStep.el.classList.remove("running");
    lastStep.el.classList.add(cssClass);
    const i = lastStep.el.querySelector(".step-icon");
    if (i) i.textContent = iconText;
    lastStep = null;
  }

  function startStep(name, opts = {}) {
    const kind = opts.kind || "tool";
    const detail = opts.detail || "";
    const expandable = opts.expandable !== false;
    finishStep();
    const step = document.createElement("div");
    step.className = "step running";
    step.dataset.kind = kind;
    step.innerHTML = `
      <div class="step-row">
        <span class="step-icon">◐</span>
        <span class="step-name">${escHtml(name)}</span>
        <span class="step-detail">${detail ? "— " + escHtml(detail) : ""}</span>
        <span class="step-elapsed">0.0s</span>
        ${expandable ? '<span class="step-expander" title="Click to expand details">▶</span>' : '<span class="step-expander step-expander-empty"></span>'}
      </div>
      <div class="step-details" hidden></div>
    `;
    stepsEl.appendChild(step);
    lastStep = { name, label: name, kind, el: step, startedAt: Date.now() };
    statusEl.textContent = name;
    scrollToBottom();
    return step;
  }

  function tick() {
    if (!trace.isConnected) return;
    setStepElapsed(lastStep);
    const stepLabel = lastStep ? (lastStep.label || lastStep.name) : "";
    statusEl.textContent = lastStep
      ? `${stepLabel} (${elapsed()})`
      : `Thinking… (${elapsed()})`;
  }
  const ticker = setInterval(tick, 200);

  return {
    onEvent(ev) {
      if (ev.type === "thinking") {
        thinkingCount += 1;
        const label = thinkingCount === 1
          ? "LLM planning"
          : (toolCount > 0 ? "LLM re-planning / synthesis" : "LLM thinking");
        startStep(label, {
          kind: "llm",
          detail: ev.note || "model call",
          expandable: false,
        });
      } else if (ev.type === "tool_call") {
        toolCount += 1;
        startStep(pretty(ev.name), { kind: "tool", expandable: true });
        if (lastStep) {
          lastStep.name = ev.name;
          lastStep.label = pretty(ev.name);
        }
      } else if (ev.type === "tool_result") {
        if (lastStep && lastStep.name === ev.name) {
          const d = lastStep.el.querySelector(".step-detail");
          if (d && ev.preview) d.textContent = "— " + ev.preview;
          // Fill in the per-step drill-down (figures + raw predictions /
          // error). The container starts hidden; user toggles with the ▶
          // expander on the row.
          const detailsEl = lastStep.el.querySelector(".step-details");
          if (detailsEl) populateStepDetails(detailsEl, ev);
          finishStep("done", "✓");
        }
        // Keep the Split-mode evidence panel live.
        if (window.__rebuildEvidence) window.__rebuildEvidence();
        // Note: we no longer auto-render a gallery of every tool figure —
        // the LLM's response embeds the relevant ones inline via markdown,
        // and dumping 22 thumbnails was noisy. The URLs are still passed to
        // the LLM via the tool_result, so it can pick which to embed.
      } else if (ev.type === "text") {
        // Final text arrived. Close any running LLM synthesis row and let the
        // caller render the answer bubble.
        finishStep("done", "✓");
      } else if (ev.type === "error") {
        finishStep("error", "✕");
        statusEl.textContent = "Error: " + ev.message;
        trace.dataset.errored = "1";
        // Append a visible error step row so it survives finish()
        const errRow = document.createElement("div");
        errRow.className = "step error";
        errRow.innerHTML = `<span class="step-icon">✕</span><span class="step-detail" style="color:var(--danger);">${escHtml(ev.message)}</span>`;
        stepsEl.appendChild(errRow);
      }
    },
    finish() {
      clearInterval(ticker);
      trace.classList.add("done");
      finishStep(trace.dataset.errored === "1" ? "error" : "done",
                 trace.dataset.errored === "1" ? "✕" : "✓");
      const head = trace.querySelector(".trace-head");
      if (head) {
        const spinner = head.querySelector(".spinner");
        if (spinner) spinner.remove();
        if (trace.dataset.errored === "1") {
          statusEl.innerHTML = `<span style="color:var(--danger);">✕</span> Failed after ${elapsed()}`;
        } else {
          statusEl.innerHTML = `<span style="color:var(--success);">✓</span> Done in ${elapsed()}`;
        }
      }
    },
    remove() {
      clearInterval(ticker);
      wrap.remove();
    },
    element: wrap,
    turnId,
  };
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  });
}

function updateContext(ctx, lastReport) {
  const ctxModality = document.getElementById("ctx-modality");
  if (ctx && ctx.current_modality) {
    const m = ctx.current_modality;
    const color = {CFP: "#34c759", OCT: "#007aff", UWF: "#af52de", FFA: "#ff9500"}[m] || "#86868b";
    ctxModality.hidden = false;
    ctxModality.style.color = color;
    ctxModality.style.borderColor = color + "55";
    ctxModality.innerHTML = `<b>${escHtml(m)}</b>`;
  } else if (ctxModality) {
    ctxModality.hidden = true;
  }
  if (ctx && ctx.current_image) {
    const name = ctx.current_image.split("/").pop();
    ctxImage.hidden = false;
    ctxImage.className = "context-pill image";
    ctxImage.innerHTML = `🖼 <a href="${escHtml(ctx.current_image)}" target="_blank">${escHtml(name)}</a>`;
  } else {
    ctxImage.hidden = true;
  }
  if (ctx && ctx.current_volume) {
    const name = ctx.current_volume.split("/").pop();
    ctxVolume.hidden = false;
    ctxVolume.className = "context-pill volume";
    ctxVolume.innerHTML = `🧊 <a href="${escHtml(ctx.current_volume)}" target="_blank">${escHtml(name)}</a>`;
  } else {
    ctxVolume.hidden = true;
  }
  if (lastReport && lastReport.pdf) {
    lastReportEl.hidden = false;
    lastReportEl.innerHTML = `📄 <a href="${escHtml(lastReport.pdf)}" target="_blank">report.pdf</a>`;
  } else {
    lastReportEl.hidden = true;
  }
  updateExportBtn();
}

// Show the Export button only when the active session has real content.
function updateExportBtn() {
  if (!exportBtn) return;
  const st = state.sessionId ? sessions.get(state.sessionId) : null;
  const hasContent = !!(st && Array.isArray(st.messages) && st.messages.some(
    m => (m.role === "user" || m.role === "assistant") && (m.content || "").trim()
  ));
  exportBtn.hidden = !hasContent;
}

// Download the current session as a self-contained HTML report.
async function exportSession() {
  const sid = state.sessionId;
  if (!sid || !exportBtn) return;
  const prev = exportBtn.innerHTML;
  exportBtn.disabled = true;
  exportBtn.innerHTML = '<span class="ico">⏳</span> Exporting…';
  try {
    const r = await fetch(`/api/sessions/${sid}/export`, { credentials: "same-origin" });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    const blob = await r.blob();
    let fname = `ophagent_${sid}.html`;
    const cd = r.headers.get("Content-Disposition") || "";
    const mt = cd.match(/filename="?([^"]+)"?/);
    if (mt) fname = mt[1];
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  } catch (e) {
    alert("Export failed: " + (e && e.message ? e.message : e));
  } finally {
    exportBtn.disabled = false;
    exportBtn.innerHTML = prev;
  }
}

// ── Sessions sidebar ──────────────────────────────────────────────────
async function refreshSessions() {
  // Defend against persisted placeholders created by older server builds.
  // The current backend already omits these, but filtering here keeps the UI
  // clean when it is used with a mixed-version deployment.
  const items = (await listSessions()).filter(it => Number(it.n_messages) > 0);
  // Only surface unsaved sessions that are actively running.
  // Empty "new chat" placeholders should never clutter the sidebar.
  const seen = new Set(items.map(i => i.session_id));
  for (const [sid, st] of sessions) {
    if (!seen.has(sid) && st.running) {
      items.unshift({ session_id: sid, title: "Running…",
                      n_messages: 0, updated_at: Date.now()/1000 });
    }
  }
  sessionList.innerHTML = "";
  for (const it of items) {
    const div = document.createElement("div");
    div.className = "session-item";
    div.dataset.sessionId = it.session_id;
    if (it.session_id === state.sessionId) div.classList.add("active");

    const sst = sessions.get(it.session_id);
    if (sst && sst.running) div.classList.add("running");

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "session-open";

    const titleSpan = document.createElement("span");
    titleSpan.className = "session-title";
    titleSpan.textContent = it.title || "(empty)";
    const updated = new Date(it.updated_at * 1000);
    const updatedText = Number.isNaN(updated.getTime())
      ? "Unknown date"
      : updated.toLocaleString([], {
          year: updated.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
          month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        });
    const fullTitle = it.full_title || it.title || "(empty)";
    openBtn.title = `${fullTitle}\n${it.n_messages} messages · updated ${updatedText}`;
    openBtn.setAttribute("aria-label", `${fullTitle}. ${it.n_messages} messages. Updated ${updatedText}`);
    if (it.session_id === state.sessionId) openBtn.setAttribute("aria-current", "page");
    openBtn.onclick = async () => {
      closeHistorySidebar();
      await switchSession(it.session_id);
    };

    const meta = document.createElement("span");
    meta.className = "session-meta";

    if (it.attachment_label && it.modality) {
      const modality = document.createElement("span");
      modality.className = `session-modality modality-${safeClassToken(String(it.modality).toLowerCase())}`;
      modality.textContent = it.modality;
      meta.appendChild(modality);
    }

    const source = document.createElement("span");
    source.className = "session-meta-source";
    if (it.attachment_label) {
      source.classList.add("attachment");
      source.title = it.attachment_label;
      const attachmentIcon = document.createElement("span");
      attachmentIcon.className = "session-meta-icon";
      attachmentIcon.setAttribute("aria-hidden", "true");
      attachmentIcon.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16.5 6L9.4 13.1c-1.8 1.8-1.8 4.7 0 6.4s4.7 1.8 6.4 0L22.9 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M19.5 9L12.4 16.1c-.8.8-.8 2.1 0 2.8s2.1.8 2.8 0L22 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
      const attachmentText = document.createElement("span");
      attachmentText.className = "session-meta-filename";
      attachmentText.textContent = it.attachment_label;
      source.appendChild(attachmentIcon);
      source.appendChild(attachmentText);
    } else {
      source.classList.add("message-count");
      source.textContent = `${it.n_messages} messages`;
    }
    const time = document.createElement("time");
    time.className = "session-meta-time";
    time.textContent = updatedText;
    if (!Number.isNaN(updated.getTime())) time.dateTime = updated.toISOString();
    meta.appendChild(source);
    meta.appendChild(time);
    openBtn.appendChild(titleSpan);
    openBtn.appendChild(meta);

    const delBtn = document.createElement("button");
    delBtn.className = "delete-btn";
    delBtn.innerHTML = "×";
    delBtn.title = "Delete this chat";
    delBtn.setAttribute("aria-label", `Delete chat: ${fullTitle}`);
    delBtn.onclick = async (e) => {
      e.stopPropagation();
      if (sst && sst.running) {
        if (!confirm(`This chat is still running. Delete it anyway?`)) return;
      } else if (!confirm(`Delete chat "${it.title}"?`)) {
        return;
      }
      try { await deleteSession(it.session_id); } catch (err) { /* tolerate 404 */ }
      sessions.delete(it.session_id);
      if (it.session_id === state.sessionId) {
        // The deleted session was the one we were viewing — open a fresh chat.
        state.sessionId = null;
        await newSession();
      } else {
        await refreshSessions();
      }
    };

    div.appendChild(openBtn);
    div.appendChild(delBtn);
    sessionList.appendChild(div);
  }
}

function setHistorySidebarOpen(open) {
  if (!historySidebar || !sidebarToggle || !sidebarBackdrop) return;
  historySidebar.classList.toggle("open", open);
  sidebarBackdrop.classList.toggle("open", open);
  sidebarBackdrop.setAttribute("aria-hidden", String(!open));
  sidebarToggle.setAttribute("aria-expanded", String(open));
  sidebarToggle.setAttribute("aria-label", open ? "Close chat history" : "Open chat history");
  sidebarToggle.title = open ? "Close chat history" : "Open chat history";
  document.body.classList.toggle("sidebar-open", open);
}

function closeHistorySidebar() {
  setHistorySidebarOpen(false);
}

// Cache the current chat scroll DOM into the session's state before switching.
// IMPORTANT: do NOT resurrect a session that was just deleted — only cache
// into an entry that already exists.
function cacheCurrentChat() {
  if (!state.sessionId) return;
  const sst = sessions.get(state.sessionId);
  if (!sst) return;
  sst.chatHTML = chatEl.innerHTML;
}

// Drop any sessions from the in-memory map that have no messages, are not
// running, and are not the one currently visible.
function pruneEmptySessions() {
  const toDrop = [];
  for (const [sid, st] of sessions) {
    if (sid === state.sessionId) continue;
    if (st.running) continue;
    const hasMessages = Array.isArray(st.messages) && st.messages.length > 0;
    if (!hasMessages) toDrop.push(sid);
  }
  for (const sid of toDrop) {
    sessions.delete(sid);
    // also clean up the backend's in-memory entry (best-effort)
    deleteSession(sid).catch(() => {});
  }
}

async function switchSession(sid) {
  if (sid === state.sessionId) return;
  cacheCurrentChat();

  const sst = getSessionState(sid);

  // If we already have cached DOM for this session, restore it instantly
  if (sst.chatHTML !== null) {
    chatEl.innerHTML = sst.chatHTML;
    reflowTraceReplyOrder(getViewMode());
  } else {
    // Otherwise fetch from server
    const data = await loadSession(sid);
    sst.messages = data.messages || [];
    sst.context = data.context;
    chatEl.innerHTML = "";
    if (sst.messages.length === 0) {
      showWelcome();
    } else {
      renderMessages(sst.messages);   // grouped trace replay (matches live)
    }
    if (!sst.model) sst.model = data.model;
    if (!sst.backend) sst.backend = data.backend;
    if (data.effort) sst.effort = data.effort;
    if (data.runtime) sst.runtime = data.runtime;
    if (data.you !== undefined) { sst.you = data.you; sst.isAdmin = !!data.is_admin; }
  }
  state.sessionId = sid;
  updateContext(sst.context, sst.lastReport);
  chatTitle.textContent = (sst.messages.find?.(m => m.role === "user")?.content || "New chat").slice(0, 80);
  applyEffortUI(sst.effort || "low");
  updatePersonalizationSummary(sst);
  applyUserPill(sst.you, sst.isAdmin);
  updateSendState();
  scrollToBottom();
  await refreshSessions();
}

async function newSession() {
  cacheCurrentChat();
  pruneEmptySessions();
  // If the current session has no real messages and isn't running, just reuse it.
  if (state.sessionId) {
    const cur = sessions.get(state.sessionId);
    if (cur && !cur.running && (!cur.messages || cur.messages.length === 0)) {
      state.pendingAttachment = null;
      renderAttachments();
      clearChat();
      showWelcome();
      updateContext(null, null);
      chatTitle.textContent = "New chat";
      updateSendState();
      await refreshSessions();
      return;
    }
  }
  const data = await createSession();
  const sst = getSessionState(data.session_id);
  sst.model = data.model;
  sst.backend = data.backend;
  sst.effort = data.effort || "low";
  sst.runtime = data.runtime || null;
  state.sessionId = data.session_id;
  state.pendingAttachment = null;
  renderAttachments();
  clearChat();
  showWelcome();
  updateContext(null, null);
  chatTitle.textContent = "New chat";
  applyEffortUI(data.effort || "low");
  updatePersonalizationSummary(sst);
  if (data.owner) { sst.you = data.owner; }
  applyUserPill(sst.you, sst.isAdmin);
  updateSendState();
  await refreshSessions();
}

function showWelcome() {
  chatEl.innerHTML = `
    <div class="welcome">
      <div class="welcome-card">
        <h2>👁 Welcome to OphAgent</h2>
        <p>Multi-modal ophthalmology assistant — colour fundus (CFP), OCT, ultra-wide-field (UWF), and fluorescein angiography (FFA).</p>
        <p class="muted">Drop an ophthalmic image or DICOM volume into the composer below, or use the 📎 button. The modality is auto-detected and the appropriate tool chain runs automatically with verifier checks.</p>
        <div class="suggest-grid">
          <button class="suggest" data-text="Analyse this image for retinal pathology — quality check first, then disease classification.">Run full pathology workup</button>
          <button class="suggest" data-text="Detect optic disc and measure cup-to-disc ratio.">Optic disc & CDR</button>
          <button class="suggest" data-text="Check for diabetic retinopathy and grade the severity. Look for laser scars or neovascularisation.">DR grading + PDR check</button>
          <button class="suggest" data-text="Verify the findings across all tools before writing the final report.">Cross-check + verify</button>
        </div>
      </div>
    </div>
  `;
  // Re-bind suggest buttons
  chatEl.querySelectorAll(".suggest").forEach(btn => {
    btn.onclick = () => {
      promptEl.value = btn.dataset.text;
      updateSendState();
      promptEl.focus();
    };
  });
}

// ── Composer ──────────────────────────────────────────────────────────
function updateSendState() {
  const hasText = promptEl.value.trim().length > 0;
  // Only "ready" (non-temp, non-error) attachments count for send-enable.
  const readyAttachments = state.pendingAttachments.filter(
    (a) => !a._temp && !a._error
  );
  const hasAttachment = readyAttachments.length > 0;
  const uploadInFlight = state.pendingAttachments.some((a) => a._temp);
  const sst = state.sessionId ? sessions.get(state.sessionId) : null;
  const busy = sst ? sst.running : false;

  // Mode logic:
  //   busy + stopping → STOPPING mode: spinner, disabled (waiting for abort)
  //   busy            → STOP mode: button is the abort control, always enabled
  //   uploadInFlight  → WAITING mode: spinner, disabled
  //   else            → SEND mode: normal arrow, enabled iff hasText|hasAtt
  const stopping = sst ? sst.stopping : false;
  sendBtn.classList.toggle("waiting-for-upload", uploadInFlight && !busy);
  sendBtn.classList.toggle("stop-mode", busy && !stopping);
  sendBtn.classList.toggle("stopping", busy && stopping);
  if (busy && stopping) {
    sendBtn.disabled = true;
    sendBtn.title = "Stopping… (current step must finish first)";
  } else if (busy) {
    sendBtn.disabled = false;
    sendBtn.title = "Stop the agent (interrupt)";
  } else {
    sendBtn.disabled = !(hasText || hasAttachment) || uploadInFlight;
    sendBtn.title = uploadInFlight
      ? "Waiting for upload to finish…"
      : "Send";
  }
}

// Send button now does double duty: send a message when idle, abort the
// in-flight agent when busy. The session's `running` flag is the switch.
async function handleSendOrStop() {
  const sst = state.sessionId ? sessions.get(state.sessionId) : null;
  if (sst && sst.running) {
    await handleAbort();
  } else {
    await handleSend();
  }
}

async function handleAbort() {
  if (!state.sessionId) return;
  // Immediate visual ack — the abort signal travels async and the agent
  // may take up to ~30s to actually stop (it has to finish the current
  // LLM round-trip or tool call). Mark the button so the user knows the
  // click registered.
  const sst = sessions.get(state.sessionId);
  if (sst) {
    sst.stopping = true;
    sendBtn.classList.add("stopping");
    sendBtn.disabled = true;
    sendBtn.title = "Stopping… (waiting for current step to finish)";
  }
  try {
    await fetch(`/api/sessions/${state.sessionId}/abort`, {
      method: "POST",
      credentials: "include",
    });
  } catch (e) {
    console.warn("abort request failed:", e);
  }
  // The streaming loop will get an "Interrupted" assistant message + done
  // event from the server, which clears sst.running via the existing path.
}

function autoGrow() {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 220) + "px";
}

promptEl.addEventListener("input", () => {
  autoGrow();
  updateSendState();
});

promptEl.addEventListener("keydown", (e) => {
  // Enter sends, shift+Enter or alt+Enter newlines
  if (e.key === "Enter" && !e.shiftKey && !e.altKey && !e.isComposing) {
    e.preventDefault();
    handleSend();
  } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    handleSend();
  }
});

attachBtn.onclick = () => fileInput.click();

fileInput.addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) await attachFile(f);
  fileInput.value = "";
});

async function attachFile(file) {
  if (!state.sessionId) await newSession();
  // Insert an "uploading…" placeholder chip immediately so the user gets
  // feedback during long uploads (~10s+ over Cloudflare Tunnel). Replace
  // it with the real chip when the upload completes, or with an error
  // chip if it fails.
  const tempId = "tmp_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
  state.pendingAttachments.push({
    _temp: true,
    _id: tempId,
    filename: file.name,
    kind: "image",          // refined after upload
    modality: null,
    path: null,
    thumb_path: null,
  });
  renderAttachments();
  updateSendState();

  try {
    const result = await uploadFile(state.sessionId, file);
    const idx = state.pendingAttachments.findIndex((a) => a._id === tempId);
    if (idx < 0) return;     // user removed it while uploading
    state.pendingAttachments[idx] = {
      filename: result.filename,
      path: result.path,                              // analysis-grade original
      thumb_path: result.thumb_path || result.path,   // baseline JPEG for UI
      kind: result.kind,
      modality: result.modality || null,
    };
    renderAttachments();
    updateSendState();
  } catch (e) {
    const idx = state.pendingAttachments.findIndex((a) => a._id === tempId);
    if (idx >= 0) {
      state.pendingAttachments[idx]._error = e.message || "upload failed";
      state.pendingAttachments[idx]._temp = false;
      renderAttachments();
    }
  }
}

function renderAttachments() {
  attachmentsEl.innerHTML = "";
  for (let i = 0; i < state.pendingAttachments.length; i++) {
    const a = state.pendingAttachments[i];
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    const filename = String(a.filename || "");
    const ext = filename.split(".").pop().toLowerCase();
    const modalityTag = a.modality
      ? `<span class="modality-tag mod-${safeClassToken(a.modality)}">${escHtml(a.modality)}</span>`
      : "";

    // Three visual states for a chip:
    //   _temp  → uploading; show a small spinner + "Uploading…"
    //   _error → failed; show a red ⚠ icon + tooltip on the error
    //   normal → uploaded; render thumb + modality + filename
    if (a._temp) {
      chip.classList.add("uploading");
      chip.innerHTML = `<span class="chip-spinner"></span>`
        + `<span class="chip-name" title="${escHtml(filename)}">${escHtml(filename)}</span>`
        + `<span class="chip-status">Uploading…</span>`
        + `<button class="chip-remove" title="Cancel">×</button>`;
    } else if (a._error) {
      chip.classList.add("upload-failed");
      chip.innerHTML = `<span class="chip-fail-icon">⚠</span>`
        + `<span class="chip-name" title="${escHtml(a._error)}">${escHtml(filename)}</span>`
        + `<span class="chip-status">Failed</span>`
        + `<button class="chip-remove" title="Remove">×</button>`;
    } else if (["png", "jpg", "jpeg", "bmp"].includes(ext) && a.kind === "image") {
      chip.innerHTML = `<img alt="">${modalityTag}<span class="chip-name">${escHtml(filename)}</span><button class="chip-remove" title="Remove">×</button>`;
      // Use the SERVER-GENERATED baseline-JPEG thumbnail. Some uploads
      // (Optos UWF etc.) use a non-baseline JPEG profile Chrome can't
      // decode; the thumbnail is always a clean Pillow-emitted JPEG so
      // the chip preview always renders. Falls back to a.path if the
      // server didn't manage to make a thumb.
      const im = chip.querySelector("img");
      fetchAsBlobUrl(a.thumb_path || a.path).then((b) => {
        if (im && b) im.src = b;
      });
    } else {
      const icon = a.kind === "volume" ? "🧊" : "📎";
      chip.innerHTML = `<span style="margin: 0 4px 0 6px;">${icon}</span>${modalityTag}<span class="chip-name">${escHtml(filename)}</span><button class="chip-remove" title="Remove">×</button>`;
    }
    chip.querySelector(".chip-remove").onclick = () => {
      state.pendingAttachments.splice(i, 1);
      renderAttachments();
      updateSendState();
    };
    attachmentsEl.appendChild(chip);
  }
}

sendBtn.onclick = handleSendOrStop;

async function handleSend() {
  const text = promptEl.value.trim();
  // Only include ready attachments (skip in-flight + failed ones)
  const atts = state.pendingAttachments.filter((a) => !a._temp && !a._error);
  if (!text && atts.length === 0) return;

  if (!state.sessionId) await newSession();

  const sid = state.sessionId;
  const sst = getSessionState(sid);
  if (sst.running) return;  // can't double-send within same session

  // hide welcome card if present
  const welcome = chatEl.querySelector(".welcome");
  if (welcome) welcome.remove();

  // Show every attached image / file
  for (const a of atts) {
    if (a.kind === "image") {
      // Display uses the safe thumbnail (baseline JPEG produced server-side);
      // analysis tools still see the original via a.path in the chat-API request.
      appendImagePreview("user", a.thumb_path || a.path, `${a.modality ? "[" + a.modality + "] " : ""}${a.filename}`);
    } else {
      appendBubble("user", `📎 attached: \`${a.filename}\``);
    }
  }
  if (text) appendBubble("user", text);

  // Compose what to actually send to the LLM.
  let modelText = text || "What can you tell me about these?";
  if (atts.length === 1) {
    const a = atts[0];
    const kindLabel =
      a.kind === "volume" ? "an OCT volume" :
      a.modality ? `a ${a.modality} image` : "an ophthalmic image";
    if (!text) {
      modelText = `I attached ${kindLabel} (${a.filename}). Please analyse it.`;
    } else {
      modelText = `${text}\n\n(Attached ${kindLabel}: ${a.filename})`;
    }
  } else if (atts.length > 1) {
    const listing = atts.map(a =>
      `  - [${a.modality || "?"}] ${a.filename} → \`${a.path.replace(/^\/files\//, "")}\``
    ).join("\n");
    modelText = (text || "Please analyse these.") +
      `\n\n(Attached ${atts.length} files:\n${listing}\n)`;
  }

  promptEl.value = "";
  state.pendingAttachments = [];
  renderAttachments();
  autoGrow();
  sst.running = true;
  sst.stopping = false;
  await refreshSessions();
  updateSendState();

  const trace = appendTrace();
  sst.activeTrace = trace;
  let finalText = "";
  let finalContext = null;
  let finalReport = null;
  let finalMessages = null;

  // Run the stream in the background so the user can switch sessions freely.
  (async () => {
    try {
      await sendChatStream(sid, modelText, (ev) => {
        trace.onEvent(ev);
        if (ev.type === "text") {
          finalText = ev.content || "";
        } else if (ev.type === "done") {
          finalContext = ev.context;
          finalReport = ev.last_report;
          finalMessages = ev.messages;
        }
      });
      trace.finish();

      if (finalText && finalText.trim()) {
        // appendBubble appends to chatEl — but the user may have switched away.
        // Inject the final bubble inside the trace's container instead so it's
        // captured by the session's DOM cache regardless of current view.
        appendAssistantAfter(trace.element, finalText);
      }
      if (finalMessages) sst.messages = finalMessages;
      sst.context = finalContext;
      sst.lastReport = finalReport;

      // Update visible UI only if the user is currently looking at this session
      if (state.sessionId === sid) {
        updateContext(finalContext, finalReport);
        chatTitle.textContent = (sst.messages.find?.(m => m.role === "user")?.content || "Chat").slice(0, 80);
      }
    } catch (e) {
      trace.remove();
      const wrap = document.createElement("div");
      wrap.className = "msg assistant";
      wrap.innerHTML = `<div class="msg-bubble">${md("**Error:** " + (e.message || e))}</div>`;
      // place into the cached DOM (in case of session switch)
      if (state.sessionId === sid) chatEl.appendChild(wrap);
      else {
        const tmp = document.createElement("div");
        tmp.innerHTML = sst.chatHTML || "";
        tmp.appendChild(wrap);
        sst.chatHTML = tmp.innerHTML;
      }
    } finally {
      sst.running = false;
      sst.stopping = false;
      sst.activeTrace = null;
      // Cache the current state of this session's chat DOM
      if (state.sessionId === sid) {
        sst.chatHTML = chatEl.innerHTML;
      }
      updateSendState();
      if (state.sessionId === sid) scrollToBottom();
      await refreshSessions();
    }
  })();
}

// Insert an assistant bubble immediately after the given element node.
function appendAssistantAfter(anchorEl, markdown) {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant reply-msg";
  if (anchorEl?.dataset?.turnId) wrap.dataset.turnId = anchorEl.dataset.turnId;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = md(markdown);
  bubble.querySelectorAll("a").forEach((a) => (a.target = "_blank"));
  rewriteFileImagesToBlob(bubble);
  wrap.appendChild(bubble);

  // Insert after the trace, then let reflowTraceReplyOrder keep the
  // current view mode's ordering. Clean collapses the trace; Inline
  // keeps the same order but expands every step.
  anchorEl.parentNode?.insertBefore(wrap, anchorEl.nextSibling);
  reflowTraceReplyOrder(getViewMode());
  scrollToBottom();
}

// ── Drag and drop ──────────────────────────────────────────────────────
["dragenter", "dragover"].forEach(ev => {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    composer.classList.add("dragging");
    composerWrap.classList.add("dragging");
    dropHint.textContent = "Release to attach";
  });
});
["dragleave", "drop"].forEach(ev => {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "dragleave" && e.relatedTarget) return;
    composer.classList.remove("dragging");
    composerWrap.classList.remove("dragging");
    dropHint.textContent = "Drop image or volume to attach";
  });
});
document.addEventListener("drop", async (e) => {
  e.preventDefault();
  composer.classList.remove("dragging");
  composerWrap.classList.remove("dragging");
  const f = e.dataTransfer.files[0];
  if (f) await attachFile(f);
});

// ── Personalization ────────────────────────────────────────────────────
const personalizationBtn = document.getElementById("personalization-btn");
const personalizationModal = document.getElementById("personalization-modal");
const personalizationModalClose = document.getElementById("personalization-modal-close");
const settingsNav = document.getElementById("settings-nav");
const settingsPagePanels = personalizationModal.querySelectorAll("[data-settings-page-panel]");
const settingsToolsTab = document.getElementById("settings-tab-tools");
const providerTabs = document.getElementById("provider-tabs");
const apiProviderTabs = document.getElementById("api-provider-tabs");
const modelList = document.getElementById("model-list");
const modelSectionTitle = document.getElementById("model-section-title");
const apiConnectionTitle = document.getElementById("api-connection-title");
const apiStatus = document.getElementById("api-status");
const apiKeyInput = document.getElementById("api-key-input");
const apiBaseUrlInput = document.getElementById("api-base-url-input");
const apiKeyVisibility = document.getElementById("api-key-visibility");
const apiSaveBtn = document.getElementById("api-save-btn");
const apiCheckBtn = document.getElementById("api-check-btn");
const apiClearBtn = document.getElementById("api-clear-btn");
const apiFeedback = document.getElementById("api-feedback");
const checkpointSummary = document.getElementById("checkpoint-summary");
const checkpointRestart = document.getElementById("checkpoint-restart");
const checkpointFilters = document.getElementById("checkpoint-filters");
const checkpointList = document.getElementById("checkpoint-list");

let catalog = null;          // { channels, providers, current_backend, current_model }
let pickerSelectedProvider = null;
let apiCredentialSettings = null;
let activeSettingsPage = "model";
let checkpointSettings = null;
let checkpointFilter = "All";
const expandedCheckpointGroups = new Set();
const checkpointDrafts = new Map();
const checkpointChecks = new Map();
const checkpointFeedbacks = new Map();
const checkpointBusy = new Set();

async function loadCatalog() {
  if (catalog) return catalog;
  catalog = await api("/api/catalog");
  catalog.channels = Array.isArray(catalog.channels) && catalog.channels.length
    ? catalog.channels
    : [{id: "gateway", label: "Gateway"}];
  catalog.providers = Array.isArray(catalog.providers) ? catalog.providers : [];
  return catalog;
}

async function loadApiCredentialSettings(force = false) {
  if (!apiCredentialSettings || force) {
    try {
      apiCredentialSettings = await api("/api/settings/api");
    } catch (error) {
      apiCredentialSettings = {
        providers: [],
        error: error.message || String(error),
      };
    }
  }
  return apiCredentialSettings;
}

async function loadCheckpointSettings(force = false) {
  if (!checkpointSettings || force) {
    try {
      checkpointSettings = await api("/api/settings/checkpoints");
    } catch (error) {
      checkpointSettings = {
        can_manage: false,
        groups: [],
        summary: {},
        restart_required: false,
        error: error.message || String(error),
      };
    }
  }
  return checkpointSettings;
}

function currentApiSetting() {
  return apiCredentialSettings?.providers?.find(
    p => p.id === pickerSelectedProvider
  ) || null;
}

function replaceApiSetting(setting) {
  if (!apiCredentialSettings) apiCredentialSettings = {providers: []};
  const index = apiCredentialSettings.providers.findIndex(p => p.id === setting.id);
  if (index >= 0) apiCredentialSettings.providers[index] = setting;
  else apiCredentialSettings.providers.push(setting);
}

function setApiFeedback(message = "", stateName = "") {
  apiFeedback.textContent = message;
  apiFeedback.dataset.state = stateName;
  apiFeedback.hidden = !message;
}

function updateApiActionState() {
  const setting = currentApiSetting();
  const hasTypedKey = Boolean(apiKeyInput.value.trim());
  apiSaveBtn.disabled = !setting;
  apiCheckBtn.disabled = !setting || (!setting.configured && !hasTypedKey);
  apiClearBtn.disabled = !setting;
}

function renderApiSettings() {
  const setting = currentApiSetting();
  if (apiConnectionTitle) {
    apiConnectionTitle.textContent = setting
      ? `Connection · ${providerLabel(setting.id)}`
      : "Connection";
  }
  if (!setting) {
    apiStatus.textContent = "Unavailable";
    apiStatus.dataset.state = "missing";
    apiKeyInput.value = "";
    apiBaseUrlInput.value = "";
    apiSaveBtn.disabled = true;
    apiCheckBtn.disabled = true;
    apiClearBtn.hidden = true;
    setApiFeedback();
    return;
  }
  const statusLabels = {
    personal: "Configured · Personal",
    environment: "Configured · Server",
    missing: "Not configured",
  };
  apiStatus.textContent = statusLabels[setting.source] || setting.source;
  apiStatus.dataset.state = setting.source;
  apiStatus.title = setting.source === "personal"
    ? "Saved for your account on this server"
    : setting.source === "environment"
      ? "Configured by the server environment"
      : "No key is available for this provider";
  apiKeyInput.type = "password";
  apiKeyVisibility.setAttribute("aria-label", "Show API key");
  apiKeyVisibility.title = "Show API key";
  apiKeyInput.value = "";
  apiKeyInput.placeholder = setting.configured
    ? "••••••••••••  configured"
    : "Enter API key";
  apiKeyInput.title = setting.configured
    ? "A key is configured. Enter a new key only to replace it."
    : "Enter an API key for this provider";
  apiBaseUrlInput.value = setting.base_url || "";
  apiClearBtn.hidden = !setting.has_personal_key;
  setApiFeedback();
  updateApiActionState();
}

async function saveApiSettings(clearKey = false) {
  const provider = pickerSelectedProvider;
  const setting = currentApiSetting();
  if (!provider || !setting) return;

  const typedKey = apiKeyInput.value.trim();
  if (!clearKey && !typedKey && !setting.has_personal_key
      && !setting.configured) {
    setApiFeedback("Enter an API key before saving", "error");
    apiKeyInput.focus();
    return;
  }

  apiSaveBtn.disabled = true;
  apiCheckBtn.disabled = true;
  apiClearBtn.disabled = true;
  setApiFeedback(clearKey ? "Removing…" : "Saving…");
  try {
    const payload = {
      base_url: apiBaseUrlInput.value.trim(),
      clear_key: clearKey,
    };
    if (typedKey && !clearKey) payload.api_key = typedKey;
    const saved = await api(`/api/settings/api/${encodeURIComponent(provider)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    replaceApiSetting(saved);
    renderApiSettings();
    setApiFeedback(clearKey ? "Personal key removed" : "Saved", "success");
  } catch (error) {
    setApiFeedback(error.message || String(error), "error");
  } finally {
    updateApiActionState();
  }
}

async function checkApiSettings() {
  const provider = pickerSelectedProvider;
  if (!provider || !currentApiSetting()) return;

  apiCheckBtn.disabled = true;
  apiSaveBtn.disabled = true;
  apiClearBtn.disabled = true;
  setApiFeedback("Checking…");
  try {
    const typedKey = apiKeyInput.value.trim();
    const payload = {base_url: apiBaseUrlInput.value.trim()};
    if (typedKey) payload.api_key = typedKey;
    const result = await api(
      `/api/settings/api/${encodeURIComponent(provider)}/check`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      },
    );
    const suffix = result.ok && Number.isInteger(result.model_count)
      ? ` · ${result.model_count} models`
      : "";
    setApiFeedback(
      `${result.message}${suffix}`,
      result.ok ? "success" : "error",
    );
  } catch (error) {
    setApiFeedback(error.message || String(error), "error");
  } finally {
    updateApiActionState();
  }
}

const CHECKPOINT_STATUS_LABELS = {
  ready: "Ready",
  verified: "Verified",
  missing: "Missing",
  mismatch: "Mismatch",
  disabled: "Disabled",
  checking: "Checking",
  unchecked: "Not checked",
};

const CHECKPOINT_SOURCE_LABELS = {
  saved: "Saved path",
  environment: "Environment path",
  default: "Default path",
  draft: "Unsaved path",
};

function checkpointGroup(groupId) {
  return checkpointSettings?.groups?.find(group => group.id === groupId) || null;
}

function replaceCheckpointGroup(group) {
  if (!checkpointSettings) return;
  const index = checkpointSettings.groups.findIndex(item => item.id === group.id);
  if (index >= 0) checkpointSettings.groups[index] = group;
  else checkpointSettings.groups.push(group);
}

function checkpointPaths(group) {
  const draft = checkpointDrafts.get(group.id);
  if (draft) return {...draft};
  return Object.fromEntries(group.resources.map(resource => [resource.id, resource.path]));
}

function checkpointFeedbackFor(group, check, busy) {
  const saved = checkpointFeedbacks.get(group.id);
  if (saved) return saved;
  if (busy) return {message: "Checking file signatures…", state: ""};
  if (!check) return {message: "", state: ""};
  if (check.status === "verified") {
    return {message: "All resources verified", state: "success"};
  }
  if (check.status === "ready") {
    return {message: "Paths valid; no checksum for one or more resources", state: "success"};
  }
  const failed = check.resources.find(resource =>
    resource.status === "missing" || resource.status === "mismatch"
  );
  return {
    message: failed ? `${failed.label}: ${failed.message}` : "Verification failed",
    state: "error",
  };
}

function renderCheckpointSettings() {
  const canManage = Boolean(checkpointSettings?.can_manage);
  settingsToolsTab.hidden = !canManage;
  if (!canManage) {
    checkpointSummary.textContent = "Unavailable";
    checkpointList.innerHTML = "";
    checkpointRestart.hidden = true;
    return;
  }

  const allGroups = checkpointSettings.groups || [];
  const enabledGroups = allGroups.filter(group => group.enabled);
  const readyGroups = enabledGroups.filter(group =>
    group.status === "ready" || group.status === "verified"
  );
  checkpointSummary.textContent = `${readyGroups.length}/${enabledGroups.length} ready`;
  checkpointRestart.hidden = !checkpointSettings.restart_required;

  checkpointFilters.querySelectorAll("[data-checkpoint-filter]").forEach(button => {
    const active = button.dataset.checkpointFilter === checkpointFilter;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });

  const groups = checkpointFilter === "All"
    ? allGroups
    : allGroups.filter(group => group.modality === checkpointFilter);
  if (!groups.length) {
    checkpointList.innerHTML = '<div class="checkpoint-empty">No tools in this modality</div>';
    return;
  }

  checkpointList.innerHTML = groups.map(group => {
    const expanded = expandedCheckpointGroups.has(group.id);
    const busy = checkpointBusy.has(group.id);
    const check = checkpointChecks.get(group.id) || null;
    const draft = checkpointDrafts.get(group.id) || null;
    const hasUncheckedDraft = Boolean(draft && !check);
    const status = busy
      ? "checking"
      : (hasUncheckedDraft ? "unchecked" : (check?.status || group.status));
    const paths = checkpointPaths(group);
    const checkedResources = new Map((check?.resources || []).map(item => [item.id, item]));
    const feedback = checkpointFeedbackFor(group, check, busy);
    const resources = group.resources.map(resource => {
      const checked = checkedResources.get(resource.id);
      const pathChanged = draft && draft[resource.id] !== resource.path;
      const shown = checked || (pathChanged
        ? {...resource, status: "unchecked", source: "draft", message: "Not checked"}
        : resource);
      const source = CHECKPOINT_SOURCE_LABELS[shown.source] || shown.source;
      const statusLabel = CHECKPOINT_STATUS_LABELS[shown.status] || shown.status;
      return `
        <div class="checkpoint-resource-row">
          <label class="checkpoint-resource-label" for="checkpoint-${escHtml(group.id)}-${escHtml(resource.id)}"
                 title="${escHtml(resource.label)}">${escHtml(resource.label)}</label>
          <input class="checkpoint-path-input" id="checkpoint-${escHtml(group.id)}-${escHtml(resource.id)}"
                 type="text" spellcheck="false" autocomplete="off"
                 data-checkpoint-path data-resource-id="${escHtml(resource.id)}"
                 value="${escHtml(paths[resource.id] || "")}"
                 title="${escHtml(source)}">
          <span class="checkpoint-resource-status" data-state="${safeClassToken(shown.status)}"
                title="${escHtml(`${source} · ${shown.message || statusLabel}`)}">${escHtml(statusLabel)}</span>
        </div>`;
    }).join("");
    return `
      <article class="checkpoint-card" data-checkpoint-group="${escHtml(group.id)}">
        <div class="checkpoint-card-head">
          <button class="checkpoint-expand" type="button" data-checkpoint-action="expand"
                  aria-expanded="${String(expanded)}">
            <span class="checkpoint-name-line">
              <span class="checkpoint-name" title="${escHtml(group.label)}">${escHtml(group.label)}</span>
              <span class="checkpoint-modality">${escHtml(group.modality)}</span>
              <span class="checkpoint-status" data-state="${safeClassToken(status)}">${escHtml(CHECKPOINT_STATUS_LABELS[status] || status)}</span>
            </span>
            <span class="checkpoint-meta">${group.tool_count} tools · ${group.resource_count} resources</span>
          </button>
          <button class="checkpoint-check-btn" type="button" data-checkpoint-action="check"
                  ${busy ? "disabled" : ""}>Check</button>
          <label class="checkpoint-switch" title="${group.enabled ? "Disable" : "Enable"} ${escHtml(group.label)}">
            <input type="checkbox" data-checkpoint-action="toggle"
                   aria-label="Enable ${escHtml(group.label)}" ${group.enabled ? "checked" : ""}
                   ${busy ? "disabled" : ""}>
            <span class="checkpoint-switch-track"></span>
          </label>
        </div>
        <div class="checkpoint-card-body" ${expanded ? "" : "hidden"}>
          ${resources}
          <div class="checkpoint-card-actions">
            <span class="checkpoint-feedback" data-state="${escHtml(feedback.state)}">${escHtml(feedback.message)}</span>
            <button class="checkpoint-save-btn" type="button" data-checkpoint-action="save"
                    ${busy ? "disabled" : ""}>Save</button>
          </div>
        </div>
      </article>`;
  }).join("");
}

async function saveCheckpointGroup(groupId, payload, {clearDraft = false} = {}) {
  const group = checkpointGroup(groupId);
  if (!group || checkpointBusy.has(groupId)) return;
  checkpointBusy.add(groupId);
  checkpointFeedbacks.set(groupId, {message: "Saving…", state: ""});
  renderCheckpointSettings();
  try {
    const result = await api(`/api/settings/checkpoints/${encodeURIComponent(groupId)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    replaceCheckpointGroup(result.group);
    checkpointSettings.restart_required = result.restart_required;
    if (clearDraft) checkpointDrafts.delete(groupId);
    checkpointChecks.delete(groupId);
    checkpointFeedbacks.set(groupId, {
      message: result.paths_changed ? "Saved · restart required" : "Saved",
      state: "success",
    });
  } catch (error) {
    checkpointFeedbacks.set(groupId, {
      message: error.message || String(error),
      state: "error",
    });
  } finally {
    checkpointBusy.delete(groupId);
    renderCheckpointSettings();
  }
}

async function checkCheckpointGroup(groupId) {
  const group = checkpointGroup(groupId);
  if (!group || checkpointBusy.has(groupId)) return;
  const draftPaths = checkpointDrafts.get(groupId) || null;
  checkpointBusy.add(groupId);
  checkpointFeedbacks.delete(groupId);
  renderCheckpointSettings();
  try {
    const result = await api(
      `/api/settings/checkpoints/${encodeURIComponent(groupId)}/check`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({paths: draftPaths}),
      },
    );
    checkpointChecks.set(groupId, result);
  } catch (error) {
    checkpointFeedbacks.set(groupId, {
      message: error.message || String(error),
      state: "error",
    });
  } finally {
    checkpointBusy.delete(groupId);
    renderCheckpointSettings();
  }
}

const PROVIDER_LABELS = {
  dashscope: "DashScope",
  aigcbest: "AIGCBest",
  openrouter: "OpenRouter",
  openai: "OpenAI",
  anthropic: "Anthropic Claude",
  gemini: "Google Gemini",
};
const EFFORT_LABELS = {
  low: "Low", medium: "Med", high: "High", max: "Max", ultra: "Ultra",
};

function providerLabel(value) {
  const provider = catalog?.providers?.find(p => p.id === value);
  return provider?.label
    || PROVIDER_LABELS[String(value || "").toLowerCase()]
    || value
    || "Provider";
}

function providerChannel(providerId) {
  return catalog?.providers?.find(p => p.id === providerId)?.channel
    || catalog?.channels?.[0]?.id
    || "gateway";
}

function providersForChannel(channelId) {
  return (catalog?.providers || []).filter(
    p => (p.channel || catalog?.channels?.[0]?.id) === channelId
  );
}

function modelLabel(backend, modelId) {
  const provider = catalog?.providers?.find(p => p.id === backend);
  const model = provider?.models?.find(m => m.id === modelId);
  return model?.name || String(modelId || "Model").split("/").pop();
}

function updatePersonalizationSummary(sessionState = null) {
  if (!personalizationSummary) return;
  const sst = sessionState || (state.sessionId ? sessions.get(state.sessionId) : null);
  const backend = sst?.backend || catalog?.current_backend || "";
  const model = sst?.model || catalog?.current_model || "";
  const effort = sst?.effort || "low";
  const summary = `${providerLabel(backend)} · ${modelLabel(backend, model)} · ${EFFORT_LABELS[effort] || effort}`;
  personalizationSummary.textContent = summary;
  if (personalizationBtn) {
    const vision = sst?.runtime?.components?.vision;
    const visionText = vision
      ? `${providerLabel(vision.backend)} / ${vision.model || "Unavailable"}`
      : "Not resolved";
    personalizationBtn.title = `Provider: ${providerLabel(backend)}\nModel: ${model || "Not selected"}\nVision: ${visionText}\nEffort: ${EFFORT_LABELS[effort] || effort}`;
  }
}

async function openPersonalization() {
  await Promise.all([
    loadCatalog(),
    loadApiCredentialSettings(true),
    loadCheckpointSettings(true),
  ]);
  const sst = state.sessionId ? sessions.get(state.sessionId) : null;
  const curBackend = (sst && sst.backend) || catalog.current_backend;
  pickerSelectedProvider = curBackend;
  applyEffortUI((sst && sst.effort) || "low");
  updatePersonalizationSummary(sst);
  renderProviderTabs();
  renderCheckpointSettings();
  if (activeSettingsPage === "tools" && !checkpointSettings?.can_manage) {
    activeSettingsPage = "model";
  }
  closeHistorySidebar();
  personalizationModal.hidden = false;
  setSettingsPage(activeSettingsPage);
  personalizationModalClose.focus();
}

function setSettingsPage(page, {focus = false} = {}) {
  if (page === "tools" && !checkpointSettings?.can_manage) page = "model";
  const tab = settingsNav.querySelector(`[data-settings-page="${page}"]`);
  if (!tab) return;
  activeSettingsPage = page;
  settingsNav.querySelectorAll("[data-settings-page]").forEach((button) => {
    const selected = button.dataset.settingsPage === page;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  settingsPagePanels.forEach((panel) => {
    const selected = panel.dataset.settingsPagePanel === page;
    panel.hidden = !selected;
    panel.classList.toggle("active", selected);
  });
  if (page === "api") renderApiSettings();
  if (page === "model") renderModelList();
  if (page === "tools") renderCheckpointSettings();
  if (focus) tab.focus();
}

function selectPickerProvider(providerId) {
  const provider = catalog.providers.find(p => p.id === providerId);
  if (!provider) return;
  pickerSelectedProvider = providerId;
  renderProviderTabs();
  if (activeSettingsPage === "api") renderApiSettings();
  if (activeSettingsPage === "model") renderModelList();
}

function renderProviderTabs() {
  for (const container of [providerTabs, apiProviderTabs]) {
    if (!container) continue;
    container.innerHTML = "";
    for (const channel of catalog.channels) {
      const providers = providersForChannel(channel.id);
      if (!providers.length) continue;

      const group = document.createElement("section");
      group.className = "provider-group";
      group.dataset.channel = channel.id;

      const heading = document.createElement("div");
      heading.className = "provider-group-head";
      heading.innerHTML = `
        <span class="provider-group-title">${escHtml(channel.label || channel.id)}</span>
        <span class="provider-group-description">${escHtml(channel.description || "")}</span>
      `;
      group.appendChild(heading);

      const options = document.createElement("div");
      options.className = "provider-options";
      options.setAttribute("role", "group");
      options.setAttribute("aria-label", channel.label || channel.id);

      for (const p of providers) {
        const selected = p.id === pickerSelectedProvider;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "provider-option" + (selected ? " active" : "");
        btn.title = p.id;
        btn.setAttribute("aria-pressed", String(selected));
        btn.innerHTML = `
          <span class="provider-option-name">${escHtml(providerLabel(p.id))}</span>
          ${selected ? '<span class="provider-option-check" aria-hidden="true">✓</span>' : ""}
        `;
        btn.onclick = () => selectPickerProvider(p.id);
        options.appendChild(btn);
      }
      group.appendChild(options);
      container.appendChild(group);
    }
  }
}

function renderModelList() {
  modelList.innerHTML = "";
  modelList.scrollTop = 0;
  const p = catalog.providers.find(x => x.id === pickerSelectedProvider);
  if (!p) {
    if (modelSectionTitle) modelSectionTitle.textContent = "Models";
    return;
  }
  if (modelSectionTitle) {
    modelSectionTitle.textContent = `Models · ${providerLabel(p.id)}`;
  }
  const sst = state.sessionId ? sessions.get(state.sessionId) : null;
  const curModel = (sst && sst.model) || catalog.current_model;
  const curBackend = (sst && sst.backend) || catalog.current_backend;
  for (const m of p.models) {
    const selected = p.id === curBackend && m.id === curModel;
    const toolCapable = m.tools !== false;
    const card = document.createElement("button");
    card.type = "button";
    card.className = "model-card"
      + (selected ? " selected" : "")
      + (toolCapable ? "" : " unavailable");
    card.setAttribute("aria-pressed", String(selected));
    if (!toolCapable) {
      card.disabled = true;
      card.setAttribute("aria-disabled", "true");
    }
    const tags = [];
    if (m.reasoning) tags.push(`<span class="tag-mini tag-reasoning">reasoning</span>`);
    if (m.vision)    tags.push(`<span class="tag-mini tag-vision">vision</span>`);
    tags.push(`<span class="tag-mini tag-cost-${safeClassToken(m.cost)}">${escHtml(m.cost)}</span>`);
    card.innerHTML = `
      <div class="model-card-header">
        <div style="flex:1; min-width:0;">
          <div class="model-card-name" title="${escHtml(m.name)}">${escHtml(m.name)}</div>
          <div class="model-card-id" title="${escHtml(m.id)}">${escHtml(m.id)}</div>
        </div>
        <div class="model-card-meta">${selected ? '<span class="model-selected-check" aria-hidden="true">✓</span>' : ""}${tags.join("")}</div>
      </div>
      <div class="model-card-note" title="${escHtml(m.note || "")}">${escHtml(m.note || "")}</div>
    `;
    card.onclick = toolCapable ? async () => {
      card.disabled = true;
      try {
        await switchSessionModel(pickerSelectedProvider, m.id);
        renderProviderTabs();
        renderModelList();
      } catch (error) {
        card.disabled = false;
        alert(`Could not switch model: ${error.message || error}`);
      }
    } : null;
    modelList.appendChild(card);
  }
  requestAnimationFrame(() => {
    const selected = modelList.querySelector(".model-card.selected");
    if (!selected || modelList.clientHeight === 0) return;
    const listRect = modelList.getBoundingClientRect();
    const selectedRect = selected.getBoundingClientRect();
    const target = modelList.scrollTop + (selectedRect.top - listRect.top)
      - Math.max(0, (modelList.clientHeight - selectedRect.height) / 2);
    modelList.scrollTop = Math.max(0, target);
  });
}

async function switchSessionModel(backend, modelId) {
  if (!state.sessionId) await newSession();
  const res = await api(`/api/sessions/${state.sessionId}/model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend, model: modelId }),
  });
  const sst = getSessionState(state.sessionId);
  sst.backend = res.backend;
  sst.model = res.model;
  sst.runtime = res.runtime || null;
  updatePersonalizationSummary(sst);
}

async function setSessionEffort(effort) {
  if (!state.sessionId) {
    await newSession();
    // newSession() resets effort UI to 'low' — re-apply the user's choice
    // before we POST so the segmented control doesn't flicker back.
    applyEffortUI(effort);
  }
  const res = await api(`/api/sessions/${state.sessionId}/model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ effort }),
  });
  const sst = getSessionState(state.sessionId);
  sst.effort = res.effort;
  sst.runtime = res.runtime || null;
  applyEffortUI(res.effort);
  updatePersonalizationSummary(sst);
}

function applyEffortUI(effort) {
  if (effortSeg) {
    effortSeg.querySelectorAll(".effort-btn").forEach((b) => {
      const active = b.dataset.effort === effort;
      b.classList.toggle("active", active);
      b.setAttribute("aria-pressed", String(active));
    });
  }
  updatePersonalizationSummary();
}

if (effortSeg) {
  effortSeg.addEventListener("click", async (e) => {
    const btn = e.target.closest(".effort-btn");
    if (!btn) return;
    const newEffort = btn.dataset.effort;
    const sst = state.sessionId ? getSessionState(state.sessionId) : null;
    const previousEffort = sst?.effort || "low";
    if (sst) sst.effort = newEffort;
    applyEffortUI(newEffort);
    try {
      await setSessionEffort(newEffort);
    } catch (error) {
      if (sst) sst.effort = previousEffort;
      applyEffortUI(previousEffort);
      alert(`Could not change effort: ${error.message || error}`);
    }
  });
}

if (exportBtn) exportBtn.addEventListener("click", exportSession);

personalizationBtn.onclick = openPersonalization;
personalizationModalClose.onclick = () => { personalizationModal.hidden = true; };
personalizationModal.querySelector(".modal-backdrop").onclick = () => {
  personalizationModal.hidden = true;
};
settingsNav.addEventListener("click", (event) => {
  const button = event.target.closest("[data-settings-page]");
  if (button) setSettingsPage(button.dataset.settingsPage);
});
settingsNav.addEventListener("keydown", (event) => {
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const buttons = [...settingsNav.querySelectorAll("[data-settings-page]")]
    .filter(button => !button.hidden);
  const current = buttons.findIndex(button => button.dataset.settingsPage === activeSettingsPage);
  if (current < 0) return;
  event.preventDefault();
  const forward = event.key === "ArrowDown" || event.key === "ArrowRight";
  const next = (current + (forward ? 1 : -1) + buttons.length) % buttons.length;
  setSettingsPage(buttons[next].dataset.settingsPage, {focus: true});
});
checkpointFilters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-checkpoint-filter]");
  if (!button) return;
  checkpointFilter = button.dataset.checkpointFilter;
  renderCheckpointSettings();
});
checkpointList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-checkpoint-action]");
  const card = event.target.closest("[data-checkpoint-group]");
  if (!button || !card) return;
  const groupId = card.dataset.checkpointGroup;
  const action = button.dataset.checkpointAction;
  if (action === "expand") {
    const expanded = !expandedCheckpointGroups.has(groupId);
    if (expanded) expandedCheckpointGroups.add(groupId);
    else expandedCheckpointGroups.delete(groupId);
    button.setAttribute("aria-expanded", String(expanded));
    const body = card.querySelector(".checkpoint-card-body");
    if (body) body.hidden = !expanded;
  } else if (action === "check") {
    checkCheckpointGroup(groupId);
  } else if (action === "save") {
    const group = checkpointGroup(groupId);
    if (group) {
      saveCheckpointGroup(
        groupId,
        {paths: checkpointPaths(group)},
        {clearDraft: true},
      );
    }
  }
});
checkpointList.addEventListener("change", (event) => {
  const toggle = event.target.closest('[data-checkpoint-action="toggle"]');
  const card = event.target.closest("[data-checkpoint-group]");
  if (!toggle || !card) return;
  saveCheckpointGroup(card.dataset.checkpointGroup, {enabled: toggle.checked});
});
checkpointList.addEventListener("input", (event) => {
  const input = event.target.closest("[data-checkpoint-path]");
  const card = event.target.closest("[data-checkpoint-group]");
  if (!input || !card) return;
  const groupId = card.dataset.checkpointGroup;
  const group = checkpointGroup(groupId);
  if (!group) return;
  const draft = checkpointDrafts.get(groupId) || checkpointPaths(group);
  draft[input.dataset.resourceId] = input.value;
  checkpointDrafts.set(groupId, draft);
  checkpointChecks.delete(groupId);
  checkpointFeedbacks.set(groupId, {message: "Unsaved changes", state: ""});
  const groupStatus = card.querySelector(".checkpoint-status");
  if (groupStatus) {
    groupStatus.textContent = "Not checked";
    groupStatus.dataset.state = "unchecked";
  }
  const resourceStatus = input.closest(".checkpoint-resource-row")
    ?.querySelector(".checkpoint-resource-status");
  if (resourceStatus) {
    resourceStatus.textContent = "Not checked";
    resourceStatus.dataset.state = "unchecked";
    resourceStatus.title = "Unsaved path · Not checked";
  }
  const feedback = card.querySelector(".checkpoint-feedback");
  if (feedback) {
    feedback.textContent = "Unsaved changes";
    feedback.dataset.state = "";
  }
});
apiSaveBtn.onclick = () => saveApiSettings(false);
apiClearBtn.onclick = () => saveApiSettings(true);
apiCheckBtn.onclick = checkApiSettings;
apiKeyVisibility.onclick = () => {
  const reveal = apiKeyInput.type === "password";
  apiKeyInput.type = reveal ? "text" : "password";
  apiKeyVisibility.setAttribute("aria-label", reveal ? "Hide API key" : "Show API key");
  apiKeyVisibility.title = reveal ? "Hide API key" : "Show API key";
};
apiKeyInput.addEventListener("input", () => {
  setApiFeedback();
  updateApiActionState();
});
apiBaseUrlInput.addEventListener("input", () => setApiFeedback());
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!personalizationModal.hidden) personalizationModal.hidden = true;
  closeHistorySidebar();
});


// ── Init ───────────────────────────────────────────────────────────────
newChatBtn.onclick = async () => {
  closeHistorySidebar();
  await newSession();
};
if (sidebarToggle) {
  sidebarToggle.onclick = () => {
    setHistorySidebarOpen(!historySidebar.classList.contains("open"));
  };
}
if (sidebarClose) sidebarClose.onclick = closeHistorySidebar;
if (sidebarBackdrop) sidebarBackdrop.onclick = closeHistorySidebar;
const narrowLayout = window.matchMedia("(max-width: 760px)");
narrowLayout.addEventListener("change", (event) => {
  if (!event.matches) closeHistorySidebar();
});

(async function init() {
  await newSession();
  // wire up suggestions on first load
  chatEl.querySelectorAll(".suggest").forEach(btn => {
    btn.onclick = () => {
      promptEl.value = btn.dataset.text;
      updateSendState();
      promptEl.focus();
    };
  });
})();
