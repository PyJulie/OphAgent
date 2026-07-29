"""
OphToolKit — multi-modal ophthalmology toolkit for the chat agent.

Wraps the adapter registry (`ophagent.adapters.GLOBAL_REGISTRY`) so each
adapter is exposed to the LLM as a callable function with a JSON schema.

Also adds session-scoped tools (set_current_image, get_modality, etc.) and
a `verify_findings` tool the planner can call as part of the Planner →
Executor → Verifier loop.
"""

from __future__ import annotations

import json
import os
import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .oph_session import OphSession

from ..adapters import GLOBAL_REGISTRY, ToolMetadata
from ..agent.tools.oct_tools import Tool, ToolParameter  # reuse existing Tool dataclass
from ..utils.paths import OUTPUT_DIR
from .run_policy import get_effort_policy
# llm_classify_modality is defined later in this module; the forward reference
# below uses the symbol via the module namespace at call time.


# ── path helpers (web serving) ───────────────────────────────────────────
_ADAPTER_OUT_ROOT = OUTPUT_DIR / "adapter_figures"
_ADAPTER_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _adapter_output_dir(adapter_name: str, session: "OphSession" | None = None) -> str:
    """Return a session-owned output directory for web calls.

    CLI/benchmark calls retain the historical global adapter output directory.
    Web sessions must keep generated figures below their own workspace so the
    ``/files`` endpoint can enforce ownership.
    """
    if session is not None and session._web_file_restrictions_enabled():
        target = (
            Path(session.workspace)
            / session.session_id
            / "tool_figures"
            / adapter_name
        )
        target.mkdir(parents=True, exist_ok=True)
        return str(target)
    return str(_ADAPTER_OUT_ROOT / adapter_name)


def _scope_web_figures(
    figures: dict[str, str],
    session: "OphSession" | None,
    namespace: str,
) -> dict[str, str]:
    """Copy out-of-session figures into an owner-gated web directory."""
    if session is None or not session._web_file_restrictions_enabled():
        return figures

    target_dir = (
        Path(session.workspace)
        / session.session_id
        / "tool_figures"
        / namespace
    ).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    scoped: dict[str, str] = {}
    for label, raw_path in figures.items():
        try:
            source = Path(raw_path).resolve()
            session.resolve_session_file(source)
            scoped[label] = str(source)
            continue
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            pass

        try:
            source = Path(raw_path).resolve(strict=True)
            if not source.is_file():
                continue
            suffix = source.suffix.lower()
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
            safe_label = "".join(
                char if char.isascii() and (char.isalnum() or char in "-_") else "_"
                for char in str(label)
            ).strip("_")[:48] or "figure"
            destination = target_dir / f"{safe_label}_{digest}{suffix}"
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
            scoped[label] = str(destination)
        except (OSError, ValueError):
            continue
    return scoped


def _figures_to_urls(
    figures: dict[str, str],
    session: "OphSession" | None = None,
    namespace: str = "figures",
) -> dict[str, str]:
    """Convert generated-output paths into owner-gated ``/files`` URLs.

    Path segments are URL-encoded (e.g. spaces → %20) so figures from images
    whose names contain spaces ('Retinal Detachment5.jpg') produce valid URLs
    that don't break in Markdown `![](...)` or the browser. FastAPI decodes
    the path param back before the whitelist lookup."""
    from urllib.parse import quote
    out: dict[str, str] = {}
    for name, abs_path in _scope_web_figures(figures, session, namespace).items():
        try:
            rel = Path(abs_path).resolve().relative_to(OUTPUT_DIR)
            out[name] = "/files/" + quote(rel.as_posix())
        except (ValueError, OSError):
            out[name] = None
    return out


# ── modality auto-detection ──────────────────────────────────────────────
_MODALITY_HEX_RE = None   # lazy re cache


def _is_hex_uuid_like(token: str) -> bool:
    """Long hex-only tokens (e.g. UUID fragments) get false-positive hits on
    short substrings like 'ffa' / 'oct' / 'cfp'. If the token is mostly hex
    and over 6 chars, treat it as opaque — don't match modality keywords."""
    global _MODALITY_HEX_RE
    if _MODALITY_HEX_RE is None:
        import re as _re
        _MODALITY_HEX_RE = _re.compile(r"^[0-9a-f]{6,}$")
    return bool(_MODALITY_HEX_RE.match(token))


def filename_modality_hint(image_path: str) -> str | None:
    """Cheap filename-only check. Returns CFP/OCT/UWF/FFA or None if uncertain.

    Uses token-level matching, NOT substring matching — `1ffa95f2-8d87-...`
    no longer matches FFA just because the hex happens to contain 'ffa'.
    Tokens are split on `[ _\\-.]`. We accept the modality keyword only if
    it is a standalone token OR a clear suffix/prefix with a separator.
    """
    import re
    name = os.path.basename(str(image_path)).lower()
    # Strip extension before tokenising
    stem, _, ext = name.rpartition(".")
    if not stem:                       # no extension
        stem = name
    if name.endswith(".dcm"):          # explicit OCT volume container
        return "OCT"
    # Tokenise — common separators in clinical filenames
    raw_tokens = re.split(r"[ _\-.()]+", stem)
    tokens = [t for t in raw_tokens if t]

    # Discard hex-UUID-shaped tokens (they cause false positives on 'ffa'/'oct'/'cfp')
    safe_tokens = [t for t in tokens if not _is_hex_uuid_like(t)]
    token_set = set(safe_tokens)

    # Multi-word phrases — these are unambiguous and may contain hyphens
    def has_phrase(*phrases: str) -> bool:
        return any(phrase in stem for phrase in phrases)

    # OCT — explicit cues only
    if ("oct" in token_set or "octcube" in token_set or "bscan" in token_set
        or has_phrase("b-scan", "_b_scan", "octcube")):
        return "OCT"

    # UWF
    if ("uwf" in token_set or "optos" in token_set or "ultrawide" in token_set
        or has_phrase("ultra-wide")):
        return "UWF"

    # FFA — token must be exactly 'ffa' or recognisable phrase
    if ("ffa" in token_set or "angio" in token_set or "fluorescein" in token_set
        or has_phrase("angiogram", "angiography", "fa-")):
        return "FFA"

    # CFP — most permissive; allowed substring 'fundus' is still distinctive
    if ("cfp" in token_set or "fundus" in token_set):
        return "CFP"

    return None


def pixel_modality_hint(image_path: str) -> str:
    """Cheap pixel-based fallback. CFP-biased."""
    try:
        import numpy as np
        from PIL import Image
        img = np.asarray(Image.open(image_path).convert("RGB"))
        h, w = img.shape[:2]
        aspect = w / max(1, h)
        r, g, b = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
        mean_chroma = (np.abs(r - g) + np.abs(g - b) + np.abs(b - r)).mean()
        if mean_chroma < 8:
            return "OCT" if aspect > 1.8 else "FFA"
        return "CFP"
    except Exception:
        return "CFP"


# ── Local 4-class modality classifier (ResNet18, 100% val acc) ─────────────
# Trained on Paired_600 (CFP, FFA) + OCTDL (OCT) + uwf dataset + FFAIR.
# Runs <50 ms on CPU, so we can use it as the FAST + accurate path before
# falling back to the LLM. Loaded lazily on first call.
_MODALITY_MODEL = None


def _load_modality_model():
    global _MODALITY_MODEL
    from ..checkpoint_config import group_is_enabled

    if not group_is_enabled("modality_detector"):
        return "disabled"
    if _MODALITY_MODEL is not None:
        return _MODALITY_MODEL
    import torch
    from ophagent.utils.paths import checkpoint_file
    ckpt_path = checkpoint_file(
        "OPHAGENT_MODALITY_CLASSIFIER_WEIGHTS",
        "_shared", "modality_classifier", "best.pt",
    )
    if not ckpt_path.exists():
        _MODALITY_MODEL = "missing"   # cache the negative result
        return _MODALITY_MODEL
    try:
        import timm
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        classes = ck.get("classes", ["CFP", "OCT", "UWF", "FFA"])
        m = timm.create_model("resnet18", pretrained=False, num_classes=len(classes))
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
        # Move to GPU if available — inference is still ~10 ms either way
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(device)
        from torchvision import transforms as T
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        _MODALITY_MODEL = {
            "model": m, "device": device, "tf": tf, "classes": classes,
            "val_acc": ck.get("val_acc"),
        }
        return _MODALITY_MODEL
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"modality classifier load failed: {e}")
        _MODALITY_MODEL = "broken"
        return _MODALITY_MODEL


def cnn_modality_hint(image_path: str,
                       min_margin: float = 0.05,
                       min_top_prob: float = 0.85,
                       max_energy: float = -2.17,
                       ) -> str | None:
    """Local ResNet18 4-class predictor with built-in OOD detection.

    Returns the argmax class if ALL three confidence signals pass:
      1. Top-2 margin ≥ `min_margin` (not a coin-flip between two classes)
      2. Top-1 probability ≥ `min_top_prob` (network is confident)
      3. Energy ≤ `max_energy` (input is "in-distribution" per
         Liu et al. 2020 NeurIPS energy-based OOD score)
    Otherwise returns None, letting the caller escalate to the vision LLM.

    The thresholds were empirically calibrated by running this CNN on:
      - 48 real eye images (12 each of CFP / OCT / UWF / FFA, randomly
        sampled from MFIDDR, OCT-bscan, UWF-IQA, and the FFA paired set)
      - 8 obvious out-of-distribution synthetic inputs (solid colours,
        random noise, gradient, checkerboard, text screenshot, VF-like
        dot grid, blue-red block)

    Calibration outcome (seed=42):
      - ID samples correctly passed:  45/48 (94%)
      - OOD samples correctly blocked: 8/8 (100%)
      - False-OOD ID rate of 6% is acceptable — those cases escalate to the
        vision LLM, which still classifies them correctly.

    See scripts/_calibrate_modality_ood.py for the calibration script."""
    cfg = _load_modality_model()
    if not isinstance(cfg, dict):
        return None
    try:
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        x = cfg["tf"](img).unsqueeze(0).to(cfg["device"])
        with torch.no_grad():
            logits = cfg["model"](x)
        # Energy-based OOD score (NeurIPS 2020). Stored as the negative of
        # logsumexp so that lower = more in-distribution.
        energy = -torch.logsumexp(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        ranked = sorted(range(len(probs)), key=lambda i: -probs[i])
        top_p = probs[ranked[0]]
        second_p = probs[ranked[1]] if len(ranked) > 1 else 0.0
        # Three independent guards. Any one failing → escalate to LLM.
        if (top_p - second_p) < min_margin:
            return None
        if top_p < min_top_prob:
            return None
        if energy > max_energy:
            return None
        return cfg["classes"][ranked[0]]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"cnn_modality_hint failed: {e}")
        return None


def llm_classify_modality(image_path: str, client, model: str,
                          max_tokens: int = 1024) -> str | None:
    """1024 token budget. Smaller would be cheaper, but reasoning models
    like gpt-5 burn most of their tokens on internal CoT and need a
    generous budget to leave any room for the actual label output. Vision
    models without internal CoT (qwen-vl, etc.) only use ~10 tokens for
    the label so the overhead is minimal."""
    """Ask a vision LLM 'what modality is this?'. One-shot, ~$0.001/call.

    Returns one of:
      "CFP", "OCT", "UWF", "FFA"             — IN-SCOPE, has trained tools
      "OPHTHALMOLOGIC_OTHER:<modality_name>" — ophthalmologic but no tools
                                                (VF, OCTA, FAF, ASOCT,
                                                 slit-lamp, B-scan US, ICGA,
                                                 topography, etc.)
      "NON_OPHTHALMOLOGIC"                   — not an eye image at all
      None                                    — LLM call failed
    """
    import base64
    try:
        suffix = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
        b64 = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        prompt = (
            "What is this image? Reply with EXACTLY one of the following "
            "labels:\n"
            "  CFP, OCT, UWF, FFA, OPHTHALMOLOGIC_OTHER:<name>, "
            "NON_OPHTHALMOLOGIC.\n\n"
            "Strict definitions (USE THESE — don't infer from colour alone):\n"
            "  CFP = standard colour fundus photo, 30-60° field. Optic disc "
            "AND macula are clearly visible together OR the macula is "
            "centred. Posterior pole only. Default to CFP when uncertain "
            "between CFP and UWF.\n"
            "  UWF = ultra-wide-field (Optos / Mirante), 100-200° field. "
            "MUST show far peripheral retina (vortex veins, ora serrata, "
            "or extensive periphery beyond the equator). Optic disc is "
            "small and off-centre. Often pseudocolour green/red channels.\n"
            "  OCT = grayscale cross-sectional B-scan strip (wide rectangle), "
            "NOT en-face.\n"
            "  FFA = monochrome / mostly-grayscale fluorescein angiography "
            "of the fundus, vessels appear bright on dark background.\n\n"
            "Use OPHTHALMOLOGIC_OTHER:<name> for eye-related images that "
            "are NOT one of the four above. Replace <name> with a short "
            "lowercase identifier. Common cases:\n"
            "  OPHTHALMOLOGIC_OTHER:visual_field   — perimetry / VF in any "
            "form: a Humphrey/Octopus printout (grayscale defect map + "
            "reliability indices), OR just the numeric deviation grid "
            "(an array of dB values arranged in a cross-shaped pattern "
            "with negative numbers indicating defects), OR a single panel "
            "showing MD/PSD/VFI summary statistics. Any image containing "
            "a clearly perimetric numeric grid is visual_field even "
            "without surrounding labels.\n"
            "  OPHTHALMOLOGIC_OTHER:octa           — OCT angiography en-face "
            "(grayscale square showing capillary network)\n"
            "  OPHTHALMOLOGIC_OTHER:faf            — fundus autofluorescence "
            "(grayscale fundus, hyper/hypo-AF without dye injection)\n"
            "  OPHTHALMOLOGIC_OTHER:icga           — indocyanine green "
            "angiography (deep choroidal angiogram)\n"
            "  OPHTHALMOLOGIC_OTHER:asoct          — anterior-segment OCT "
            "(cornea / iris / anterior chamber cross-section)\n"
            "  OPHTHALMOLOGIC_OTHER:slit_lamp      — slit-lamp photo of "
            "cornea, conjunctiva, iris, lens\n"
            "  OPHTHALMOLOGIC_OTHER:bscan_us       — B-scan ultrasonography "
            "(grayscale arc-shaped acoustic image)\n"
            "  OPHTHALMOLOGIC_OTHER:topography     — corneal topography / "
            "Pentacam map (false-colour curvature/elevation)\n"
            "  OPHTHALMOLOGIC_OTHER:unknown_ophth  — clearly ophthalmologic "
            "but doesn't fit any of the above\n\n"
            "Use NON_OPHTHALMOLOGIC for anything that is NOT an eye image "
            "(natural photos, screenshots, diagrams, animals, food, body "
            "parts other than the eye, solid colour blocks, abstract art).\n\n"
            "If the image is a degraded / blurry / atypical-looking real "
            "ophthalmologic image, prefer the most likely modality over "
            "NON_OPHTHALMOLOGIC — only refuse if it is clearly NOT an eye.\n\n"
            "Reply with the label and nothing else."
        )
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
        text = (resp.choices[0].message.content or "").strip()
        upper = text.upper()
        # Exact in-scope tokens win first
        for cand in ("CFP", "OCT", "UWF", "FFA"):
            if upper.startswith(cand) and not upper.startswith(cand + ":"):
                # Don't false-match "OCT" against "OPHTHALMOLOGIC_OTHER:octa"
                if not upper.startswith("OPHTHALMOLOGIC"):
                    return cand
        if "NON_OPHTHALMOLOGIC" in upper:
            return "NON_OPHTHALMOLOGIC"
        if upper.startswith("OPHTHALMOLOGIC_OTHER"):
            # Preserve the original-case sub-label after the colon
            sublabel = text.split(":", 1)[1].strip().lower() if ":" in text else "unknown_ophth"
            # Sanitise to a safe identifier
            sublabel = "".join(c for c in sublabel if c.isalnum() or c == "_")[:32]
            return f"OPHTHALMOLOGIC_OTHER:{sublabel or 'unknown_ophth'}"
        # Bare in-scope token search as last resort
        for cand in ("CFP", "OCT", "UWF", "FFA"):
            if cand in upper:
                return cand
        return None
    except Exception:
        return None


def auto_detect_modality(image_path: str, client=None, model: str | None = None) -> str:
    """Detect modality. Returns one of:
      "CFP", "OCT", "UWF", "FFA"              — in-scope (full tool pipeline)
      "OPHTHALMOLOGIC_OTHER:<name>"           — eye image, no trained tools
      "NON_OPHTHALMOLOGIC"                    — refuse and do not analyse
      "UNVERIFIED_INPUT"                      — scope could not be established

    Detection priority depends on what's available:
      1. Filename hint (free, instant — only fires on explicit CFP/OCT/UWF/FFA
         token in the filename; never returns OPHTH_OTHER or NON_OPH)
      2. Vision LLM (~1s, $0.001) — the ONLY source of truth for
         OPHTHALMOLOGIC_OTHER and NON_OPHTHALMOLOGIC. We try this BEFORE the
         local CNN because the CNN was trained on 4 classes only and will
         misclassify any non-medical input into one of them.
      3. Local CNN (ResNet18, ~50 ms, 4-way) — fallback when no LLM.
      4. Conservative refusal when neither detector can establish scope.
    """
    hint = filename_modality_hint(image_path)
    if hint is not None:
        return hint
    if client is not None and model is not None:
        llm = llm_classify_modality(image_path, client, model)
        if llm is not None:
            return llm
    # LLM unavailable — fall back to CNN. Note: CNN cannot detect non-eye
    # inputs so a cat photo will get classified as one of CFP/OCT/UWF/FFA.
    # This is documented as a limitation; running without an LLM is also
    # a degraded mode that the user must opt into.
    cnn = cnn_modality_hint(image_path)
    if cnn is not None:
        return cnn
    return "UNVERIFIED_INPUT"


# ── retsam payload compaction ─────────────────────────────────────────────
# The raw cfp_retsam_segmentation result is ~50-70 KB because it carries the
# per-lesion `components` array for up to 100+ lesions. The LLM drowns in
# this and ends up hallucinating zeros for key clinical metrics
# (e.g. it told the user "DR hemorrhage area = 0.0" when retsam actually
# detected 43 hemorrhages totalling 10,158 px²). To fix this we:
#   1. Hoist a compact `llm_headline` dict to the TOP of the result with the
#      key clinical numbers in plain prose — the LLM cannot miss it.
#   2. Strip the per-lesion `components` arrays from the deep tree (LLM does
#      not need each lesion's centroid; the counts/areas are what matter).

def _round(v, n=3):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


def _make_retsam_headline(out: dict) -> dict:
    """Build a compact, LLM-readable summary of the most important retsam
    clinical metrics. The LLM is instructed (via field names + the
    `instructions_for_llm` line) to quote these numbers directly instead of
    fishing through the nested JSON."""
    pred = out.get("predictions", {}) or {}
    q = pred.get("quantitative", {}) or {}
    tl = q.get("top_line", {}) or {}
    les = q.get("lesions", {}) or {}
    groups = les.get("groups", {}) or {}
    quant_meta = pred.get("meta", {}) or {}
    module_errors = quant_meta.get("module_errors", {}) or {}
    modules_succeeded = quant_meta.get("modules_succeeded", []) or []
    quant_has_data = any(
        bool(q.get(name))
        for name in ("top_line", "vessels", "disc_cup", "lesions", "tessellation", "myopia")
    )
    quant_unavailable = bool(module_errors) and not quant_has_data

    # A missing optic-disc mask prevents the formal post-processor from
    # producing disc-normalised counts and spatial measurements. The raw task
    # masks remain valid model outputs, so preserve their disc-independent
    # burden rather than silently turning unavailable measurements into zero.
    #
    # Disease-labelled segmentation heads are not independent etiologic
    # diagnoses. In particular, a single macular hemorrhage can activate both
    # the DR-hemorrhage and AMD-patch-hemorrhage heads. Quantify that overlap
    # for every run so downstream reasoning does not count one lesion twice.
    raw_mask_evidence: dict[str, object] = {}
    hemorrhage_etiology: dict[str, object] = {
        "status": "not_assessed",
    }
    mask_files = pred.get("mask_files", {}) or {}
    try:
        import numpy as _np
        from PIL import Image as _Image

        def _read_mask(name: str):
            path = mask_files.get(name)
            if not (path and os.path.exists(path)):
                return None
            return _np.asarray(_Image.open(path))

        fundus = _read_mask("fundus_mask")
        fundus_area = int((fundus > 0).sum()) if fundus is not None else 0
        if quant_unavailable and fundus_area:
            raw_mask_evidence["fundus_area_px"] = fundus_area

            def _store(name: str, mask, values=None) -> None:
                if mask is None:
                    return
                selected = _np.isin(mask, values) if values is not None else mask > 0
                area = int(selected.sum())
                raw_mask_evidence[f"{name}_area_px"] = area
                raw_mask_evidence[f"{name}_fundus_ratio"] = round(
                    area / fundus_area, 4
                )

            myopia_mask = _read_mask("myopia_mask")
            _store("myopic_arc", myopia_mask, [1])
            _store("diffuse_atrophy", myopia_mask, [2])
            _store("patchy_atrophy", myopia_mask, [3])
            _store("chorioretinal_atrophy", myopia_mask, [2, 3])
            _store("tessellation", _read_mask("tessellation_mask"))
            _store("dr_hemorrhage", _read_mask("lesion_dr_hemorrhage"))
            _store("dr_exudate", _read_mask("lesion_dr_exudate"))
            _store("dr_cotton_wool", _read_mask("lesion_dr_cotton_wool_spot"))
            _store("amd_drusen", _read_mask("lesion_amd_drusen"))
            _store("amd_patch_hemorrhage", _read_mask("lesion_amd_patch_hemorrhage"))
            _store("epiretinal_membrane", _read_mask("lesion_others_epiretinal_membrane"))

        dr_hem_mask = _read_mask("lesion_dr_hemorrhage")
        amd_hem_mask = _read_mask("lesion_amd_patch_hemorrhage")
        if dr_hem_mask is not None and amd_hem_mask is not None:
            dr_fg = dr_hem_mask > 0
            amd_fg = amd_hem_mask > 0
            dr_px = int(dr_fg.sum())
            amd_px = int(amd_fg.sum())
            if dr_px and amd_px:
                overlap_px = int((dr_fg & amd_fg).sum())
                union_px = int((dr_fg | amd_fg).sum())
                overlap_of_dr = overlap_px / dr_px
                overlap_of_amd = overlap_px / amd_px
                overlap_of_smaller = overlap_px / min(dr_px, amd_px)
                ambiguous = overlap_px >= 20 and overlap_of_smaller >= 0.5
                hemorrhage_etiology = {
                    "status": "ambiguous" if ambiguous else "separable",
                    "dr_head_area_px": dr_px,
                    "amd_head_area_px": amd_px,
                    "overlap_area_px": overlap_px,
                    "overlap_fraction_of_dr_head": round(overlap_of_dr, 4),
                    "overlap_fraction_of_amd_head": round(overlap_of_amd, 4),
                    "intersection_over_union": round(
                        overlap_px / max(union_px, 1), 4
                    ),
                    "interpretation": (
                        "The DR and AMD hemorrhage heads substantially overlap. "
                        "Treat this as one hemorrhagic lesion with unresolved "
                        "etiology; do not use the DR-head component count as "
                        "independent evidence of diabetic retinopathy."
                        if ambiguous else
                        "The DR and AMD hemorrhage masks are sufficiently "
                        "separate for independent morphologic review."
                    ),
                }
    except Exception:
        raw_mask_evidence = {}
        hemorrhage_etiology = {"status": "not_assessed"}

    def _cls(group, name):
        return ((groups.get(group, {}) or {}).get("classes", {}) or {}).get(name, {}) or {}

    hem = _cls("lesion_dr", "hemorrhage")
    exd = _cls("lesion_dr", "exudate")
    cws = _cls("lesion_dr", "cotton_wool_spot")
    las = _cls("lesion_dr", "laser_spot")
    drusen = _cls("lesion_amd", "drusen")
    amd_hem = _cls("lesion_amd", "patch_hemorrhage")
    erm = _cls("lesion_others", "epiretinal_membrane")
    scar = _cls("lesion_others", "retinal_scar")
    mh = _cls("lesion_others", "macular_hole")
    vt = _cls("possible_lesions", "venous_tortuosity")

    def _ct(d):
        try:
            return int(d.get("count", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _area(d):
        a = d.get("area")
        if isinstance(a, dict):
            return _round(a.get("px"), 1)
        return _round(d.get("area_px"), 1)

    vcdr = _round(tl.get("vCDR"))
    hcdr = _round(tl.get("hCDR"))

    # Spatial: how many hemorrhages near the macula?
    hem_macula = ((hem.get("spatial", {}) or {})
                  .get("macula_zone_counts", {}) or {})

    n_hem = _ct(hem)
    n_exd = _ct(exd)
    n_cws = _ct(cws)
    n_drusen = _ct(drusen)
    n_amd_hem = _ct(amd_hem)
    n_mh = _ct(mh)
    n_erm = _ct(erm)
    n_scar = _ct(scar)
    n_vt = _ct(vt)
    n_las = _ct(las)

    # Glaucoma morphology flag — vCDR >= 0.55 is suspicious for glaucoma
    # (lowered from 0.70 because vCDR 0.55-0.70 cases were being missed by
    # the LLM even though the morphology adapter computed real numbers).
    vcdr_flag = (
        "suspicious_for_glaucoma" if (isinstance(vcdr, (int, float)) and vcdr >= 0.55)
        else "normal" if isinstance(vcdr, (int, float))
        else "unknown"
    )

    # ── DR signal confidence tier ─────────────────────────────────────────
    # Conservative tiers to defend against retsam false positives (camera
    # dust, isolated noise pixels mis-detected as microaneurysms).
    #
    # high   - multiple lesions OR clustering near macula OR accompanied by
    #          other DR-like signs, provided the hemorrhage mask does not
    #          substantially overlap the AMD patch-hemorrhage head.
    # ambiguous - DR-like morphology is present, but the same pixels activate
    #          the AMD hemorrhage head, so disease attribution is unresolved.
    # low    - 1-2 isolated hemorrhages with NO macular clustering and NO
    #          accompanying sign. Possible early NPDR, but indistinguishable
    #          from imaging artefact. Report as "suspicious; recommend
    #          follow-up"; do NOT flip a Normal call to DR on this alone.
    # absent - zero hemorrhages, zero other DR signs.
    #
    # Tuned for new_dst dataset: NPDR I subfolder typically has 2-5
    # microaneurysms / dot hemorrhages; mild noise on a clean image is
    # usually 0-1 false positive. The 2-with-macular-clustering or 3-anywhere
    # gate threads the needle.
    near_macula = (hem_macula.get("0_1_dd", 0) or 0) + (hem_macula.get("1_2_dd", 0) or 0)
    if n_hem == 0 and n_exd == 0 and n_cws == 0 and n_las == 0:
        dr_signal_confidence = "absent"
    elif (n_hem >= 3
          or (n_hem >= 2 and near_macula >= 1)
          or (n_hem >= 1 and (n_exd > 0 or n_cws > 0))
          or n_exd >= 2
          or n_cws >= 1
          or n_las >= 1):
        dr_signal_confidence = "high"
    elif n_hem >= 1:
        dr_signal_confidence = "low"  # 1-2 isolated dots, no support
    else:
        dr_signal_confidence = "absent"
    dr_morphology_signal_confidence = dr_signal_confidence
    if (hemorrhage_etiology.get("status") == "ambiguous"
            and dr_signal_confidence in {"high", "low"}):
        dr_signal_confidence = "ambiguous"

    lines = []
    if quant_unavailable:
        lines.append(
            "ReT-SAM formal quantification unavailable because the optic disc "
            "was not localised; disc-normalised counts and spatial metrics are "
            "unavailable, not zero."
        )
    if n_hem:
        macula_near = hem_macula.get("0_1_dd", 0) + hem_macula.get("1_2_dd", 0)
        macula_part = f" ({macula_near} within 2 disc-diameters of macula)" if macula_near else ""
        if hemorrhage_etiology.get("status") == "ambiguous":
            overlap_pct = 100 * float(
                hemorrhage_etiology.get("overlap_fraction_of_dr_head") or 0
            )
            lines.append(
                f"HEMORRHAGE PRESENT: the DR head split it into {n_hem} "
                f"components totaling {_area(hem)} px2{macula_part}, but "
                f"{overlap_pct:.0f}% of its pixels overlap the AMD "
                "patch-hemorrhage head. Count this as one hemorrhagic region "
                "with unresolved etiology, NOT as independent proof of DR."
            )
        else:
            lines.append(
                f"DR HEMORRHAGES PRESENT: {n_hem} lesions, total "
                f"{_area(hem)} px2{macula_part} - strong DR morphology signal."
            )
    if n_exd:
        if hemorrhage_etiology.get("status") == "ambiguous":
            lines.append(
                f"Exudate-like regions detected by the DR head: {n_exd} "
                f"components, total {_area(exd)} px2. In this overlapping "
                "macular lesion these are morphologic findings, not "
                "disease-specific proof of DR."
            )
        else:
            lines.append(
                f"DR HARD EXUDATES PRESENT: {n_exd} lesions, "
                f"total {_area(exd)} px2."
            )
    if n_cws:
        lines.append(f"COTTON-WOOL SPOTS PRESENT: {n_cws} lesions.")
    if n_las:
        lines.append(f"Laser scars present: {n_las} lesions (suggests prior PRP).")
    if n_drusen:
        lines.append(f"AMD DRUSEN PRESENT: {n_drusen} lesions.")
    if n_amd_hem:
        if hemorrhage_etiology.get("status") == "ambiguous":
            lines.append(
                "AMD-head patch/subretinal hemorrhage is present and overlaps "
                "the DR-head hemorrhage. Prioritise a macular neovascular "
                "differential (nAMD/PCV or myopic CNV when clinically "
                "appropriate) over automatic DR attribution."
            )
        else:
            lines.append(
                f"AMD-pattern subretinal hemorrhage present: {n_amd_hem} lesions."
            )
    if n_mh:
        lines.append(f"Macular-hole-like lesion present: {n_mh}.")
    if n_erm:
        lines.append(f"Epiretinal-membrane-like lesion present: {n_erm}.")
    if n_scar:
        lines.append(f"Retinal scar present: {n_scar} lesions.")
    if n_vt:
        lines.append(f"Venous tortuosity present: {n_vt} segments.")
    if isinstance(vcdr, (int, float)):
        lines.append(
            f"Optic disc cup-to-disc: vCDR={vcdr}, hCDR={hcdr} → {vcdr_flag.replace('_',' ')}"
            + (" (consider glaucoma)." if vcdr_flag == "suspicious_for_glaucoma" else ".")
        )
    # Tessellation + myopic-atrophy live in their OWN quantitative sub-modules
    # (quantitative["tessellation"]["tessellation"] + quantitative["myopia"]
    # ["severity_inputs"]), NOT in top_line. The original code read them from
    # top_line → always None → the myopia lines never reached the headline.
    # The corrected reads are GATED behind RETSAM_MYOPIA_FIX=1 so production
    # keeps the original behaviour by default and we can A/B the fix cleanly.
    tess = (q.get("tessellation", {}) or {}).get("tessellation", {}) or {}
    myo = (q.get("myopia", {}) or {}).get("severity_inputs", {}) or {}
    _myopia_fix = os.environ.get("RETSAM_MYOPIA_FIX") == "1"
    if _myopia_fix:
        tess_sev = tess.get("severity")
        if tess_sev and tess_sev != "none":
            cov = tess.get("coverage_ratio")
            lines.append(
                f"Tessellation severity: {tess_sev}"
                + (f" (coverage {cov*100:.0f}%)" if isinstance(cov, (int, float)) else "")
                + (" — involves macula" if tess.get("involves_macula") else "")
                + "."
            )
        _myo_signs = []
        if myo.get("diffuse_atrophy_present"):
            _myo_signs.append("diffuse chorioretinal atrophy")
        if (myo.get("patchy_count") or 0):
            _myo_signs.append(f"{myo.get('patchy_count')} patchy-atrophy lesion(s)")
        if myo.get("arc_lesion_present"):
            _myo_signs.append("peripapillary atrophic arc")
        if _myo_signs:
            lines.append(
                "Myopic-degeneration signs: " + ", ".join(_myo_signs)
                + " (chorioretinal atrophy / patchy atrophy support pathologic myopia / MMD; "
                  "tessellation or a peripapillary crescent ALONE does not)."
            )
    else:
        # ORIGINAL behaviour: reads top_line (keys absent there → no lines added)
        if tl.get("tessellation_severity"):
            lines.append(
                f"Tessellation severity: {tl.get('tessellation_severity')}"
                + (" — involves macula" if tl.get("tessellation_involves_macula") else "")
                + "."
            )
        if tl.get("myopia_diffuse_atrophy_present") or tl.get("myopia_patchy_atrophy_count"):
            lines.append("Myopic atrophy detected.")

    # CLINICAL PM-SIGNAL (OPH_PM_FIX=1, default OFF): deterministically apply the
    # clinician-calibrated META-PM rule from retsam fields, so the PM call does NOT
    # depend on the LLM mis-applying numeric thresholds. PM-positive if ANY of:
    #   (1) diffuse OR patchy chorioretinal atrophy            (META-PM >=2)
    #   (2) LARGE peripapillary atrophy arc (>=0.8 disc-areas OR >=180 deg)
    #   (3) SEVERE tessellation + a corroborator (peripapillary arc OR tilted disc)
    # The peripapillary arc is CENTRAL -> robust to poor PERIPHERAL image quality,
    # so it rescues high-myopia eyes whose tessellation is under-measured on a bad
    # photo. Thresholds (0.8 dd^2, 180 deg, 'severe') are first-pass, to be tuned.
    pm_signal = None
    if os.environ.get("OPH_PM_FIX") == "1":
        _arc = ((q.get("myopia", {}) or {}).get("classes", {}) or {}).get("arc_lesion", {}) or {}
        _arc_dd2 = _arc.get("area_dd2") or 0
        _arc_deg = myo.get("arc_angular_coverage_deg") or 0
        _tsev = tess.get("severity")
        _tilt = tl.get("disc_tilt_deg") or 0
        _tcov = tess.get("coverage_ratio") or 0
        _arc_present = bool(myo.get("arc_lesion_present")) or _arc_dd2 > 0
        _reasons = []
        # Trigger A — REAL chorioretinal atrophy (META-PM >=2). Strongest, standalone.
        if myo.get("diffuse_atrophy_present"):
            _reasons.append("diffuse chorioretinal atrophy (META-PM>=2)")
        if (myo.get("patchy_count") or 0):
            _reasons.append(f"{myo.get('patchy_count')} patchy-atrophy lesion(s) (META-PM>=2)")
        # Trigger B — myopic CONFIGURATION = tessellation AND peripapillary arc (NOT
        # either alone: a large arc with no tessellation is often NOT a myopic fundus,
        # per clinician review of FP_large_arc). Severe tessellation needs only an arc
        # present; moderate tessellation (>=25%, often under-measured on poor-quality
        # peripheries) needs a LARGE arc (>=0.8 disc-areas). AREA, not angular spread,
        # discriminates the arc (benign eyes have wide-but-thin rings).
        if (_tsev == "severe" and _arc_present) or (_tcov >= 0.25 and _arc_dd2 >= 0.8):
            _reasons.append(f"tessellation {_tcov*100:.0f}% ({_tsev}) AND peripapillary arc {_arc_dd2:.2f} disc-areas [myopic configuration]")
        # Layer 2 — quantify EMPTY (NoDisc: disc localisation failed, common on severe
        # atrophic eyes) but segmentation masks exist -> read raw atrophy/tessellation
        # masks DIRECTLY (disc-independent) so valid atrophy isn't silently discarded.
        _quantify_empty = (not tl) and (not myo) and (not tess)
        _retsam_blank = False
        if not _reasons and _quantify_empty:
            mf = pred.get("mask_files") or {}
            if mf.get("myopia_mask") or mf.get("tessellation_mask"):
                try:
                    import numpy as _np
                    from PIL import Image as _Img
                    _fund = mf.get("fundus_mask")
                    _fa = int((_np.array(_Img.open(_fund)) > 0).sum()) if (_fund and os.path.exists(_fund)) else 0
                    def _mcov(key, vals):
                        p = mf.get(key)
                        if not (p and os.path.exists(p)) or not _fa:
                            return 0.0
                        return int(_np.isin(_np.array(_Img.open(p)), vals).sum()) / _fa
                    _atr = _mcov("myopia_mask", [2, 3])     # diffuse + patchy chorioretinal atrophy
                    _tes = _mcov("tessellation_mask", [1])
                    _arcm = _mcov("myopia_mask", [1])
                    if _atr >= 0.05:
                        _reasons.append(f"chorioretinal atrophy mask {_atr*100:.0f}% (disc-free fallback, META-PM>=2)")
                    if _tes >= 0.30 and _arcm > 0.005:
                        _reasons.append(f"severe tessellation mask {_tes*100:.0f}% AND peripapillary arc (disc-free fallback)")
                    if not _reasons and _atr < 0.02 and _tes < 0.10:
                        _retsam_blank = True
                except Exception:
                    pass
            else:
                _retsam_blank = True
        pm_signal = bool(_reasons)
        if pm_signal:
            _sig = "PRESENT — " + "; ".join(_reasons) + ". [report PM(MMD)=1; do NOT require macular-centre involvement.]"
        elif _retsam_blank:
            # Layer 3 — retsam produced no usable myopia data at all -> defer to CLIP
            # (disc- and segmentation-independent appearance classifier).
            _sig = ("UNAVAILABLE (retsam gave no usable myopia data: disc not localised, atrophy/"
                    "tessellation masks empty). FALL BACK to cfp_clip_ensemble — if its top-1 = "
                    "Pathological myopia (prob >=0.5) -> report PM(MMD)=1.")
        else:
            _sig = "absent (no chorioretinal atrophy; no tessellation+arc configuration)."
        lines.append("PATHOLOGIC-MYOPIA SIGNAL: " + _sig)

    # NoDisc LESION fallback (OPH_PM_FIX=1, default OFF): generalises the PM
    # Layer-2 rescue to DR / AMD / ERM. When the disc fails to localise, EVERY
    # disc-normalised quantify module — including `lesions` — returns nothing, so
    # n_hem/n_drusen/n_erm are all 0 and the headline would wrongly read "clean",
    # even though the per-class lesion masks (segmentation is disc-INDEPENDENT)
    # still hold real lesions. Read those masks directly and surface their
    # coverage as EVIDENCE for the LLM to judge. Conservative on purpose: DR/AMD/
    # ERM have no clinician-validated deterministic threshold yet (unlike PM), so
    # this informs rather than forces the call — and explicitly warns the LLM not
    # to treat missing counts as "absent". Gate `(not tl) and (not les)` fires
    # only on quantify failure (a normal disc populates top_line → skipped).
    if os.environ.get("OPH_PM_FIX") == "1" and (not tl) and (not les):
        mf = pred.get("mask_files") or {}
        if any(mf.get(k) for k in ("lesion_dr_mask", "lesion_amd_mask", "lesion_others_mask")):
            try:
                import numpy as _np
                from PIL import Image as _Img
                _fund = mf.get("fundus_mask")
                _fa = int((_np.array(_Img.open(_fund)) > 0).sum()) if (_fund and os.path.exists(_fund)) else 0

                def _fg(key):  # per-class BINARY mask → (coverage_ratio, px)
                    p = mf.get(key)
                    if not (p and os.path.exists(p)):
                        return 0.0, 0
                    px = int((_np.array(_Img.open(p)) > 0).sum())
                    return (px / _fa if _fa else 0.0), px

                # DR = active lesions only (haemorrhage + exudate + cotton-wool;
                # laser scar is treatment, not active DR → excluded).
                _dr_px = sum(_fg(k)[1] for k in (
                    "lesion_dr_hemorrhage", "lesion_dr_exudate", "lesion_dr_cotton_wool_spot"))
                _dru_r, _dru_px = _fg("lesion_amd_drusen")
                _aph_r, _aph_px = _fg("lesion_amd_patch_hemorrhage")
                _erm_r, _erm_px = _fg("lesion_others_epiretinal_membrane")
                _amd_px = _dru_px + _aph_px
                _avail = []
                if _dr_px >= 20:
                    _avail.append(f"DR lesions (haem/exudate/CWS) {_dr_px}px ({_dr_px/max(_fa,1)*100:.2f}% fundus)")
                if _amd_px >= 20:
                    _avail.append(f"AMD drusen/patch-haemorrhage {_amd_px}px ({_amd_px/max(_fa,1)*100:.2f}%)")
                if _erm_px >= 20:
                    _avail.append(f"ERM membrane {_erm_px}px ({_erm_r*100:.2f}%)")
                if _avail:
                    lines.append(
                        "RETSAM QUANTIFY UNAVAILABLE (disc not localised — lesion counts missing) "
                        "BUT raw lesion masks are still present: " + "; ".join(_avail) + ". "
                        "Judge DR / AMD / ERM from these masks + the image directly; do NOT read "
                        "the missing counts as 'absent'.")
            except Exception:
                pass

    # DISC-PRESENT / CUP-ABSENT signal (OPH_PM_FIX=1, default OFF): clinician note.
    # If retsam localises the optic DISC but cannot delineate a CUP, the cup is
    # either OBLITERATED (disc swelling — papilloedema / optic neuritis / AION /
    # malignant hypertension, a RED FLAG) or physiologically absent (a small
    # "crowded" disc). Either way it ARGUES AGAINST glaucoma, which ENLARGES the
    # cup — so this can NEVER suppress a true glaucoma (those have a LARGE cup).
    # Surface it so the LLM does not false-call glaucoma and can flag a
    # non-glaucomatous disc abnormality (outside the 7-class set). od_oc_mask:
    # value 1 = disc, value 2 = cup.
    if os.environ.get("OPH_PM_FIX") == "1":
        _odp = (pred.get("mask_files") or {}).get("od_oc_mask")
        if _odp and os.path.exists(_odp):
            try:
                import numpy as _np
                from PIL import Image as _Img
                _od = _np.array(_Img.open(_odp))
                _disc_px = int((_od >= 1).sum())
                _cup_px = int((_od == 2).sum())
                if _disc_px >= 200 and _cup_px <= max(8, 0.02 * _disc_px):
                    lines.append(
                        f"OPTIC DISC PRESENT but CUP NOT DELINEABLE (cup~0 vs disc {_disc_px}px) "
                        "— cup obliterated/absent. ARGUES AGAINST glaucoma (glaucoma ENLARGES the "
                        "cup); do NOT call Glaucoma on this. Consider disc swelling (papilloedema / "
                        "optic neuritis / AION / hypertensive) or a small crowded disc — flag as a "
                        "non-glaucomatous disc finding if the disc looks elevated/blurred.")
            except Exception:
                pass

    if not lines:
        lines.append("No lesions or morphology abnormalities detected by retsam.")

    def _available(value):
        return None if quant_unavailable else value

    return {
        "instructions_for_llm": (
            "Quote the numbers in this `llm_headline` block directly in your report. "
            "Do NOT claim a metric is 0 unless the field below explicitly shows 0. "
            "If you write a sentence like 'no DR hemorrhages detected' or 'vCDR=0.0', "
            "first re-read these numbers. A segmentation-head name describes "
            "morphology, not proven disease etiology. When "
            "`hemorrhage_etiology.status` is `ambiguous`, count the overlapping "
            "DR/AMD masks as one lesion and do NOT diagnose concurrent DR from "
            "the hemorrhage or exudate component counts alone."
        ),
        "natural_language_summary": lines,
        "quantification_status": {
            "status": "unavailable" if quant_unavailable else "available",
            "modules_succeeded": list(modules_succeeded),
            "module_errors": module_errors,
        },
        "raw_mask_evidence": raw_mask_evidence,
        "hemorrhage_etiology": hemorrhage_etiology,
        "optic_disc": {
            "vCDR": vcdr, "hCDR": hcdr,
            "rim_disc_area_ratio": _round(tl.get("rim_disc_area_ratio")),
            "glaucoma_morphology_flag": vcdr_flag,
        },
        "diabetic_retinopathy_signs": {
            "hemorrhage_count": _available(n_hem),
            "hemorrhage_area_px": _available(_area(hem) if n_hem else 0),
            "hemorrhage_near_macula_count": _available(near_macula),
            "hemorrhage_present": _available(n_hem > 0),
            "exudate_count": _available(n_exd),
            "exudate_present": _available(n_exd > 0),
            "cotton_wool_count": _available(n_cws),
            "laser_spot_count": _available(n_las),
            "any_dr_lesion": _available((n_hem + n_exd + n_cws + n_las) > 0),
            # Conservative confidence tier — see _make_retsam_headline
            # docstring for thresholds. Use this, NOT the raw count, to
            # decide whether to flip a Normal call to DR.
            "dr_signal_confidence": (
                "unavailable" if quant_unavailable else dr_signal_confidence
            ),
            "morphology_signal_confidence_before_etiology_guard": (
                "unavailable"
                if quant_unavailable else dr_morphology_signal_confidence
            ),
        },
        "amd_signs": {
            "drusen_count": _available(n_drusen),
            "patch_hemorrhage_count": _available(n_amd_hem),
            "patch_hemorrhage_present": _available(
                bool(tl.get("AMD_patch_hemorrhage_present")) or n_amd_hem > 0
            ),
            "any_amd_lesion": _available((n_drusen + n_amd_hem) > 0),
        },
        "other_findings": {
            "macular_hole_count": _available(n_mh),
            "venous_tortuosity_count": _available(n_vt),
            "epiretinal_membrane_count": _available(n_erm),
            "retinal_scar_count": _available(n_scar),
            "tessellation_severity": (tess.get("severity") if _myopia_fix else tl.get("tessellation_severity")),
            "tessellation_involves_macula": _available(bool(tess.get("involves_macula") if _myopia_fix else tl.get("tessellation_involves_macula"))),
            "myopia_arc_present": _available(bool(myo.get("arc_lesion_present") if _myopia_fix else tl.get("myopia_arc_present"))),
            "myopia_diffuse_atrophy_present": _available(bool(myo.get("diffuse_atrophy_present") if _myopia_fix else tl.get("myopia_diffuse_atrophy_present"))),
            "myopia_patchy_atrophy_count": _available(
                int(myo.get("patchy_count") or 0) if _myopia_fix else None
            ),
        },
        "vessels": {
            "CRAE_px": _round(tl.get("CRAE_px"), 2),
            "CRVE_px": _round(tl.get("CRVE_px"), 2),
            "AVR": _round(tl.get("AVR")),
            "vessel_density_total": _round(tl.get("vessel_density_total")),
            "fractal_dimension_artery": _round(tl.get("fractal_dimension_artery")),
        },
    }


# Field names whose values are per-degree / per-blob debug arrays. Stripping
# them from the LLM-facing payload saves ~15-20 KB without losing any
# clinically actionable info — the summaries in `llm_headline` carry the
# important numbers.
_STRIP_KEYS = {
    "components",                 # per-lesion centroids/bboxes
    "profile_deg_to_width_px",   # disc rim per-degree scan
    "angular_profile",            # myopic-arc per-angle scan
}


def _strip_noisy_keys(o):
    if isinstance(o, dict):
        return {k: _strip_noisy_keys(v) for k, v in o.items() if k not in _STRIP_KEYS}
    if isinstance(o, list):
        return [_strip_noisy_keys(v) for v in o]
    return o


def _compact_retsam_payload(out: dict) -> dict:
    """Add an llm_headline at the top and strip large debug arrays."""
    headline = _make_retsam_headline(out)
    preds = _strip_noisy_keys(out.get("predictions", {}) or {})
    # COST OPTION (RETSAM_TRIM_PAYLOAD=1, default OFF): the raw `quantitative`
    # module dicts (~13k tokens) are re-sent in every later orchestrator turn at
    # ~full price (prompt caching barely fires across the slow agent loop). The
    # `llm_headline` already carries every actionable number (vCDR, CRAE/CRVE/AVR,
    # lesion counts, tessellation severity, myopia flags), so for BATCH screening
    # we drop the raw quantitative from the LLM-facing payload. The full
    # analysis.json stays on disk (predictions.meta / output_dir) for lazy lookup.
    # Default OFF keeps the interactive/production payload complete.
    if os.environ.get("RETSAM_TRIM_PAYLOAD") == "1":
        preds.pop("quantitative", None)
    out["predictions"] = preds
    # Re-order so headline is the first key in the JSON the LLM sees.
    return {"llm_headline": headline, **out}


# ── verifier escalation: adaptive visual differential ─────────────────────
# When the specialised classifiers deadlock at low confidence, the replanner
# escalates to ONE targeted vision-LLM differential instead of re-running the
# whole classifier battery. The focus prompt is built adaptively from the tied
# candidate diagnoses + their distinguishing clinical signs.
_DDX_SIGNS = {
    "diabetic retinopathy": "microaneurysms, dot/blot hemorrhages, hard exudates "
        "(often circinate), cotton-wool spots; neovascularisation if proliferative",
    "dr": "microaneurysms, dot/blot hemorrhages, hard exudates, cotton-wool spots",
    "neovascular amd": "macula-centred subretinal or sub-RPE hemorrhage, "
        "gray-green neovascular tissue, exudation, or a hemorrhagic PED",
    "pcv": "macula-centred or peripapillary subretinal hemorrhage, orange-red "
        "polypoidal nodules, or a hemorrhagic pigment epithelial detachment",
    "myopic cnv": "focal macular hemorrhage or a gray CNV lesion accompanied "
        "by independent high-myopia signs such as lacquer cracks or patchy atrophy",
    "age-related macular degeneration": "drusen, RPE pigmentary mottling, "
        "geographic atrophy, or sub-retinal fluid/hemorrhage at the macula",
    "amd": "drusen, RPE mottling, macular atrophy",
    "retinal vein occlusion": "sectoral flame hemorrhages following a vein, "
        "dilated tortuous veins, cotton-wool spots in the affected territory",
    "rvo": "flame hemorrhages along a vein, venous dilation/tortuosity",
    "central serous": "well-demarcated serous macular detachment, no hard exudates",
    "csc": "serous macular elevation, absent exudates",
    "pathologic myopia": "peripapillary atrophy, tessellated fundus, lacquer "
        "cracks, posterior staphyloma",
    "pathological myopia": "peripapillary atrophy, tessellation, lacquer cracks",
    "glaucoma": "enlarged cup-to-disc ratio, neuroretinal rim thinning, RNFL "
        "wedge defects, disc hemorrhage",
    "retinal detachment": "billowing/elevated retina, demarcation line, "
        "corrugated retinal surface",
    "cataract": "diffuse media opacity / haze obscuring fundus detail",
}


_QUALITY_TOOL_NAMES = {
    "cfp_eyeq", "cfp_efiqa", "cfp_quality_robust", "oct_quality",
}

# Tools on this list emit a primary, mutually competing disease label. Other
# tools are deliberately excluded because they operate on orthogonal axes
# (quality, DR stage, glaucoma referability, lesion burden) and can legitimately
# be positive together as comorbid findings.
_PRIMARY_DIAGNOSTIC_TOOLS = {
    "cfp_clip_multi_disease", "cfp_retizero", "cfp_flair",
    "cfp_clip_ensemble", "cfp_paired5", "cfp_openvocab_zeroshot",
    "oct_fmue_16class", "oct_volume_octcubem", "oct_volume_macular",
    "uwf_disease_7class", "uwf_multi_disease",
}
_CFP_ENSEMBLE_COMPONENTS = {
    "cfp_clip_multi_disease", "cfp_retizero", "cfp_flair",
}


def _canonical_diagnostic_family(label: Any) -> str | None:
    if isinstance(label, dict):
        label = label.get("label") or label.get("label_en")
    text = str(label or "").strip().lower()
    if not text:
        return None
    if text in {"normal", "no dr", "no dr (normal)", "healthy"}:
        return "normal"
    if any(token in text for token in (
        "diabetic", "npdr", "pdr", "hard exudate", "microaneurysm",
    )) or text == "dr":
        return "diabetic retinopathy"
    if "macular degeneration" in text or text in {"amd", "namd", "cnv"}:
        return "age-related macular degeneration"
    if "vein occlusion" in text or text in {"rvo", "brvo", "crvo"}:
        return "retinal vein occlusion"
    if "central serous" in text or text == "csc":
        return "central serous chorioretinopathy"
    if "glaucoma" in text:
        return "glaucoma"
    if "myopia" in text:
        return "pathologic myopia"
    if "retinal detachment" in text:
        return "retinal detachment"
    return text


def _entry_diagnostic_family(entry: dict[str, Any]) -> str | None:
    """Extract a positive disease-level vote without mixing quality/negatives."""
    tool = str(entry.get("tool") or "")
    if tool in _QUALITY_TOOL_NAMES:
        return None
    predictions = entry.get("predictions") or {}
    if not isinstance(predictions, dict):
        return None

    if predictions.get("referable_glaucoma") is True:
        return "glaucoma"
    if predictions.get("referable_glaucoma") is False:
        return None

    label = (
        predictions.get("fused_top1")
        or predictions.get("top1")
        or predictions.get("primary_diagnosis")
        or predictions.get("predicted_class")
        or predictions.get("severity_label")
    )
    if not label:
        clip_top3 = predictions.get("clip_top3") or []
        if clip_top3 and isinstance(clip_top3[0], dict):
            label = clip_top3[0].get("label_en") or clip_top3[0].get("label")
    if not label:
        ranked = (
            predictions.get("fused_top3")
            or predictions.get("top_3")
            or predictions.get("top3")
            or []
        )
        if ranked and isinstance(ranked[0], dict):
            label = ranked[0].get("label_en") or ranked[0].get("label")

    # A negative PDR result is not a separate disease vote. A positive PDR
    # category belongs to the DR family.
    pdr_category = str(predictions.get("pdr_category") or "").strip().lower()
    if not label and pdr_category and pdr_category not in {
        "no pdr", "non-pdr", "无pdr", "none",
    }:
        label = pdr_category

    return _canonical_diagnostic_family(label)


def _diagnostic_family_votes(data: dict[str, Any]) -> dict[str, list[str]]:
    votes: dict[str, list[str]] = {}
    for entry in data.get("results", []):
        if not isinstance(entry, dict):
            continue
        confidence = entry.get("confidence")
        try:
            if confidence is not None and float(confidence) < 0.3:
                continue
        except (TypeError, ValueError):
            pass
        family = _entry_diagnostic_family(entry)
        if family:
            votes.setdefault(family, []).append(str(entry.get("tool") or "?"))
    return votes


def _primary_diagnostic_votes(data: dict[str, Any]) -> dict[str, list[str]]:
    """Return only mutually competing primary-label votes.

    When the CFP ensemble is present, its three constituent CLIPs are omitted:
    their disagreement is already represented by the ensemble's agreement and
    confidence fields and must not be counted again as an unresolved conflict.
    """
    entries = [
        entry for entry in data.get("results", [])
        if isinstance(entry, dict)
    ]
    tools_present = {str(entry.get("tool") or "") for entry in entries}
    use_ensemble = "cfp_clip_ensemble" in tools_present
    votes: dict[str, list[str]] = {}
    for entry in entries:
        tool = str(entry.get("tool") or "")
        if tool not in _PRIMARY_DIAGNOSTIC_TOOLS:
            continue
        if use_ensemble and tool in _CFP_ENSEMBLE_COMPONENTS:
            continue
        confidence = entry.get("confidence")
        try:
            if confidence is not None and float(confidence) < 0.3:
                continue
        except (TypeError, ValueError):
            pass
        family = _entry_diagnostic_family(entry)
        if family:
            votes.setdefault(family, []).append(tool)
    return votes


def _verifier_top_candidates(data: dict) -> list[str]:
    """Distinct top differential labels from the ensemble / paired classifiers."""
    res = data.get("results", [])
    cands: list[str] = []

    dr421 = next(
        (e for e in res if e.get("tool") == "cfp_dr_421_assessment"),
        None,
    )
    if dr421:
        pred = dr421.get("predictions") or {}
        guard = pred.get("etiology_guard") or {}
        if (pred.get("eligible_for_dr_grading") is False
                or guard.get("status") == "ambiguous"):
            cands.extend(["Neovascular AMD/PCV", "Myopic CNV"])

    retsam = next(
        (e for e in res if e.get("tool") == "cfp_retsam_segmentation"),
        None,
    )
    if retsam:
        pred = retsam.get("predictions") or {}
        headline = retsam.get("llm_headline") or pred.get("llm_headline") or pred
        guard = headline.get("hemorrhage_etiology") or {}
        if guard.get("status") == "ambiguous":
            cands.extend(["Neovascular AMD/PCV", "Myopic CNV"])

    cle = next((e for e in res if e.get("tool") == "cfp_clip_ensemble"), None)
    if cle:
        for h in ((cle.get("predictions") or {}).get("fused_top3") or [])[:3]:
            lab = h.get("label_en") or h.get("label")
            if lab:
                cands.append(lab)
    p5 = next((e for e in res if e.get("tool") == "cfp_paired5"), None)
    if p5:
        for h in ((p5.get("predictions") or {}).get("top_3") or [])[:2]:
            lab = h.get("label")
            if lab and lab.lower() != "normal":
                cands.append(lab)
    seen, out = set(), []
    for c in cands:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:3]


def _cfp_summary_md(stage1: dict, stage2: dict, reliability: str,
                    field_issues: list, cross_issues: list) -> str:
    """Render the two-stage CFP vision output as compact Markdown for
    backward-compatible `impression_markdown` consumers (chat UI, agent
    LLM context). The structured JSON is the real source of truth."""
    lines = []
    lines.append(f"**Reliability**: {reliability}"
                 + (" (validation issues below)" if (field_issues or cross_issues) else ""))
    if stage1:
        q = stage1.get("image_quality", "?")
        qreason = stage1.get("image_quality_reason", "")
        lat = stage1.get("laterality_guess", "?")
        lines.append(f"**Quality**: {q}"
                     + (f" — {qreason}" if qreason else "")
                     + f"  •  laterality: {lat}")
        feat = stage1.get("one_phrase_impression", "")
        if feat:
            lines.append(f"**Most striking feature**: {feat}")
        # v2 gestalt key observations
        bullets = []
        for fld, label in [
            ("overall_pattern", "Overall"),
            ("disc_appearance", "Disc"),
            ("macula_appearance", "Macula"),
            ("peripheral_appearance", "Periphery"),
            ("hemorrhage_predominant_shape", "Hem-shape"),
            ("macular_star_present", "Macular star"),
            ("prominent_AV_nicking_or_arteriolar_narrowing", "AV-nicking"),
        ]:
            v = stage1.get(fld)
            # Fields are expected to be short strings, but some vision models
            # wrap them in a nested object (e.g. {"value": ..., "confidence":
            # ...}) or a list. Coerce to a hashable scalar before the
            # membership test — a raw dict here would raise
            # "TypeError: unhashable type: 'dict'".
            if isinstance(v, dict):
                v = (v.get("value") or v.get("finding") or v.get("label")
                     or json.dumps(v, ensure_ascii=False))
            elif isinstance(v, list):
                v = ", ".join(str(x) for x in v if x)
            # Only surface fields with notable values
            if v and v not in {"normal", "normal-appearing", "absent", "none",
                                "cannot_assess"}:
                bullets.append(f"{label}: {v}")
        if bullets:
            lines.append("**Key gestalt**:")
            for b in bullets:
                lines.append(f"- {b}")
    if isinstance(stage2, dict):
        if stage2.get("single_line_impression"):
            lines.append(f"\n**Impression**: {stage2['single_line_impression']}")
        top3 = stage2.get("top3_differential") or []
        if top3:
            lines.append("\n**Top differential**:")
            for entry in top3[:3]:
                if isinstance(entry, dict):
                    dx = entry.get("diagnosis", "?")
                    lk = entry.get("likelihood", "?")
                    lines.append(f"- {dx} ({lk})")
        ruled = stage2.get("ruled_out") or []
        if ruled:
            lines.append("\n**Ruled out**:")
            for entry in ruled[:5]:
                if isinstance(entry, dict):
                    dx = entry.get("diagnosis", "?")
                    reason = entry.get("reason", "")[:140]
                    lines.append(f"- {dx} — {reason}")
        if stage2.get("recommended_followup_imaging"):
            lines.append(f"\n**Followup**: {', '.join(stage2['recommended_followup_imaging'])}")
    if field_issues:
        lines.append(f"\n_Field validation_: {len(field_issues)} issue(s)")
    if cross_issues:
        lines.append(f"_Cross-tool disagreements_: {len(cross_issues)} issue(s)")
    return "\n".join(lines)


def _build_differential_focus(candidates: list[str]) -> str:
    cand = [c for c in candidates if c][:3]
    if not cand:
        return ("The specialised classifiers could not agree. State the single "
                "most likely diagnosis with the specific clinical signs that "
                "support it, or say it is indeterminate on this image.")
    lines = []
    for c in cand:
        cl = c.lower()
        sign = next((v for k, v in _DDX_SIGNS.items() if k in cl), None)
        lines.append(f"- {c}: look for {sign}" if sign else f"- {c}")
    return (
        "The specialised classifiers are split, at LOW confidence, between: "
        + ", ".join(cand) + ". Adjudicate using the SPECIFIC clinical signs "
        "visible in THIS image:\n" + "\n".join(lines) +
        "\nState which single diagnosis the visible signs best support and why, "
        "or say it is genuinely indeterminate on a single colour photo."
    )


_INDEP_VERIFIER_SYS = (
    "You are an INDEPENDENT verification agent in a clinical ophthalmology "
    "pipeline. You are SEPARATE from the diagnostic planner: you did NOT pick "
    "these tools and have NOT seen the planner's reasoning or its proposed "
    "diagnosis.\n"
    "IMPORTANT: STRUCTURED rule-based consistency checks have ALREADY been "
    "applied to these outputs and their results are given to you below. Your "
    "job is NOT to re-litigate from scratch — it is to catch a GENUINE, "
    "SPECIFIC contradiction the rule checks MISSED.\n"
    "Verdict definitions:\n"
    "  (1) conflict — one output DIRECTLY and concretely contradicts another "
    "(e.g. a classifier asserts a disease the segmentation/quantification "
    "clearly refutes). Use ONLY for a specific, nameable contradiction.\n"
    "  (2) insufficient — evidence is genuinely too thin for ANY diagnosis.\n"
    "  (3) consistent — outputs cohere. This is the DEFAULT when the rule "
    "checks found no issues and you cannot name a concrete contradiction.\n"
    "BIAS TOWARD 'consistent'. Do NOT flag conflict merely for low confidence, "
    "ambiguity, a single uncertain tool, or a borderline probability — the "
    "planner already weighs those. Only override the rule checks with a "
    "specific, defensible contradiction. If unsure, say consistent.\n"
    "If a conflict truly needs resolving, name the SINGLE most useful tool.\n"
    "Respond with ONLY a JSON object:\n"
    '{"verdict": "consistent"|"conflict"|"insufficient", '
    '"conflicts": ["..."], "suggested_tool": "<tool name or empty>", '
    '"note": "<one-sentence justification>"}'
)


def _independent_verifier_review(session, data: dict,
                                 rule_issues=None, rule_warnings=None) -> dict | None:
    """A genuinely independent Verifier *agent*: a SEPARATE LLM pass that did
    not plan the case and sees ONLY the raw tool outputs (not the planner's
    chat history), then independently judges consistency / sufficiency. This
    is the second reasoning LLM in the Planner–Verifier architecture (the
    Planner is the orchestrator loop; this is the Verifier).

    Returns a parsed verdict dict, or None when unavailable (no client /
    text model / disabled / error) — in which case the rule-based checks in
    `verify_findings` still stand on their own. Disable via
    OPH_VERIFIER_LLM=0 or session.verifier_llm=False.
    """
    import os
    if session is None:
        return None
    # On by DEFAULT at HIGH effort (1-shot independent review). MAX / ULTRA use
    # the multi-agent debate panel instead (see _run_debate). Overridable for
    # controlled ablations: session.verifier_llm (True/False) or env
    # OPH_VERIFIER_LLM (1/0) force it on/off regardless of effort.
    force_env = os.environ.get("OPH_VERIFIER_LLM")        # "1" / "0" / None
    force_attr = getattr(session, "verifier_llm", None)   # True / False / None
    if force_env == "0" or force_attr is False:
        return None
    effort_on = (
        get_effort_policy(getattr(session, "effort", "medium")).verifier_mode
        == "independent_llm"
    )
    if not (force_env == "1" or force_attr is True or effort_on):
        return None
    backend = getattr(session, "verifier_backend", None) or getattr(session, "backend", None)
    try:
        if hasattr(session, "_client_for_backend"):
            client = session._client_for_backend(backend, "_verifier_client_obj")
        else:
            session._ensure_client()
            client = getattr(session, "_client", None)
    except Exception:
        return None
    # Per-role backbone override: the Verifier agent's LLM may be specified
    # separately from the orchestrator (session.verifier_model), defaulting to
    # the orchestrator's model. Enables the heterogeneous-backbone ablation.
    model = getattr(session, "verifier_model", None) or getattr(session, "model", None)
    if client is None or not model:
        return None

    lines = _evidence_lines(data)
    if not lines:
        return None
    modality = getattr(getattr(session, "context", None), "current_modality", None)
    # Feed the rule-based checks' verdict in, so the LLM verifier defers to the
    # structured checks and only adds a GENUINE missed contradiction (rather
    # than re-judging from scratch and over-flagging conflicts → over-escalation).
    ri = "; ".join(rule_issues) if rule_issues else "none"
    rw = "; ".join(rule_warnings) if rule_warnings else "none"
    user = (f"Modality: {modality}\n"
            f"Rule-based consistency checks already run — issues: {ri} | "
            f"warnings (soft/low-confidence): {rw}.\n"
            f"Raw tool outputs:\n" + "\n".join(lines))
    out = _json_llm_call(client, model, _INDEP_VERIFIER_SYS, user)
    if out is None:
        return None
    out["_model"] = model
    out["_backend"] = backend
    return out


# ── Shared LLM-agent helpers (verifier + debate) ─────────────────────────
# Tools the verifier/debate panel may REQUEST as a bounded escalation when a
# conflict is unresolved (known-safe, cheap, more-informative re-checks).
_SAFE_ESCALATIONS = {
    "cfp_clip_ensemble", "vision_impression",
    "oct_volume_octcubem", "ffa_classification",
}


def _evidence_lines(data: dict) -> list[str]:
    """Compact, neutral one-line-per-tool summary of the raw tool outputs."""
    lines: list[str] = []
    for e in data.get("results", []):
        if not isinstance(e, dict):
            continue
        snippet = json.dumps(e.get("predictions") or {}, ensure_ascii=False)
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        und = " [undetermined]" if e.get("undetermined") else ""
        lines.append(f"- {e.get('tool', '?')}{und} (conf={e.get('confidence')}): {snippet}")
    return lines


def _seed_diagnosis(data: dict) -> str:
    """Pick the strongest classifier signal to seed the debate's proposition."""
    best, best_conf = None, -1.0
    for e in data.get("results", []):
        if not isinstance(e, dict):
            continue
        try:
            conf = float(e.get("confidence")) if e.get("confidence") is not None else -1.0
        except Exception:
            conf = -1.0
        p = e.get("predictions") or {}
        label = (p.get("fused_top1") or p.get("predicted_class")
                 or p.get("primary_diagnosis"))
        if not label:
            t3 = p.get("fused_top3") or p.get("top_3") or p.get("top3") or []
            if t3 and isinstance(t3[0], dict):
                label = t3[0].get("label_en") or t3[0].get("label")
        if label and conf >= best_conf:
            best, best_conf = label, conf
    return best or "uncertain"


def _json_llm_call(client, model: str, system: str, user: str) -> dict | None:
    """One LLM turn returning a parsed JSON object, or None on failure.
    max_tokens is generous because reasoning models (gpt-5) spend the cap on
    hidden reasoning tokens first; a small cap returns empty visible content.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=4000,
        )
        txt = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
    import re as _re
    m = _re.search(r"\{.*\}", txt, _re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except Exception:
        return None
    return out if isinstance(out, dict) else None


_DEBATE_CHALLENGER_SYS = (
    "You are the CHALLENGER in a clinical ophthalmology case debate. A planner "
    "has proposed a diagnosis. ATTACK it: using ONLY the raw tool outputs, give "
    "the strongest reasons it could be WRONG, and name the single most plausible "
    "ALTERNATIVE the evidence supports. Do not be agreeable — your value is "
    "surfacing what was missed or over-read. Respond with ONLY JSON:\n"
    '{"challenge": "<strongest case against the proposed diagnosis>", '
    '"alternative": "<most plausible competing diagnosis, or empty>", '
    '"evidence_gap": "<what evidence would settle it, if any>"}'
)
_DEBATE_DEFENDER_SYS = (
    "You are the DEFENDER in a clinical ophthalmology case debate, advocating "
    "for the proposed diagnosis. A challenger has argued against it. Using ONLY "
    "the raw tool outputs, either (a) REBUT the challenge with specific "
    "evidence, or (b) if the challenge is stronger, CONCEDE and adopt the "
    "better diagnosis — honesty over winning. Respond with ONLY JSON:\n"
    '{"defense": "<evidence-based explanation>", '
    '"revised_diagnosis": "<keep original, or the conceded diagnosis>", '
    '"conceded": true|false}'
)
_DEBATE_JUDGE_SYS = (
    "You are the IMPARTIAL JUDGE of a clinical ophthalmology case debate. You "
    "argued neither side. Given the raw tool outputs, the proposed diagnosis, "
    "the challenge, and the defense, decide the outcome on the EVIDENCE alone. "
    "Respond with ONLY JSON:\n"
    '{"final_diagnosis": "<diagnosis the evidence best supports>", '
    '"resolved": true|false, '
    '"reason": "<one-sentence justification grounded in the tool outputs>", '
    '"request_tool": "<single tool to resolve a still-open conflict, or empty>"}'
)


def _run_debate(session, data: dict) -> dict | None:
    """Multi-agent debate verification for the MAX / ULTRA tiers: a CHALLENGER,
    a DEFENDER, and an impartial JUDGE — three SEPARATE LLM agents, none of
    which has seen the orchestrator's reasoning — argue the differential over
    the raw tool outputs for up to 2 rounds. Returns the judge's verdict
    (with a `_transcript`), or None (graceful: rule-based checks still stand).

    Per-role backbone override: session.debate_model (defaults to the
    orchestrator's model) — enables the heterogeneous-backbone ablation.
    Disable entirely via env OPH_DEBATE=0.
    """
    import os
    if session is None or os.environ.get("OPH_DEBATE", "1") == "0":
        return None
    backend = getattr(session, "debate_backend", None) or getattr(session, "backend", None)
    try:
        if hasattr(session, "_client_for_backend"):
            client = session._client_for_backend(backend, "_debate_client_obj")
        else:
            session._ensure_client()
            client = getattr(session, "_client", None)
    except Exception:
        return None
    model = getattr(session, "debate_model", None) or getattr(session, "model", None)
    if client is None or not model:
        return None
    lines = _evidence_lines(data)
    if not lines:
        return None
    evidence = "\n".join(lines)
    modality = getattr(getattr(session, "context", None), "current_modality", None)
    proposed = _seed_diagnosis(data)

    transcript: list[dict] = []
    verdict: dict | None = None
    for _ in range(2):  # bounded: at most 2 argumentation rounds
        ctx = (f"Modality: {modality}\nRaw tool outputs:\n{evidence}\n"
               f"Proposed diagnosis: {proposed}")
        if transcript:
            ctx += ("\nPrior round (judge ruled unresolved): "
                    + json.dumps(transcript[-1]["judge"], ensure_ascii=False))
        challenge = _json_llm_call(client, model, _DEBATE_CHALLENGER_SYS, ctx)
        if challenge is None:
            break
        defense = _json_llm_call(
            client, model, _DEBATE_DEFENDER_SYS,
            ctx + "\nChallenge: " + json.dumps(challenge, ensure_ascii=False))
        if defense is None:
            break
        judge = _json_llm_call(
            client, model, _DEBATE_JUDGE_SYS,
            ctx + "\nChallenge: " + json.dumps(challenge, ensure_ascii=False)
            + "\nDefense: " + json.dumps(defense, ensure_ascii=False))
        if judge is None:
            break
        verdict = judge
        transcript.append({"challenge": challenge, "defense": defense, "judge": judge})
        proposed = judge.get("final_diagnosis") or proposed
        # Stop if resolved, or if it needs a tool (cross-call escalation), else
        # run one more pure-argumentation round.
        if judge.get("resolved") or (judge.get("request_tool") or "").strip():
            break
    if verdict is None:
        return None
    # Build a FRESH dict — do NOT mutate `judge` in place: `verdict` IS the
    # last `judge` object and `transcript[-1]["judge"]` is the same object, so
    # assigning verdict["_transcript"]=transcript would create a circular
    # reference that json.dumps rejects.
    return {**verdict, "_rounds": len(transcript),
            "_model": model, "_backend": backend, "_transcript": transcript}


# ── adapter → Tool mapping ───────────────────────────────────────────────
def _make_adapter_tool(
    meta: ToolMetadata,
    session: "OphSession" | None = None,
) -> Tool:
    """Wrap an adapter as a Tool the LLM can call."""

    description = (
        f"[{meta.modality}] {meta.description}\n"
        f"Confidence threshold: {meta.confidence_threshold}.\n"
    )
    if meta.limitations:
        description += "Known limitations: " + "; ".join(meta.limitations) + "."
    if meta.requires_tools:
        description += f" Depends on: {', '.join(meta.requires_tools)}."

    params = [
        ToolParameter(
            "image_path", "string",
            f"Path to the {meta.modality} image to analyse",
            required=False, default="",
        ),
    ]
    # Per-tool extra args
    if meta.name == "cfp_retsam_segmentation":
        params.append(ToolParameter(
            "eye_side", "string", "Eye laterality (OS = left, OD = right) for ETDRS zoning",
            required=False, default="", enum=["OS", "OD"],
        ))
    if meta.name == "oct_volume_macular":
        params.extend([
            ToolParameter(
                "classifier_model", "string",
                "Per-slice classifier for the volume pass. Use "
                "'oct_fmue_16class' when a fine-grained OCT disease label is "
                "needed; other options are oct_classifier_broad, "
                "oct_classifier_octdl, and oct_classifier_kermany.",
                required=False, default="oct_fmue_16class",
                enum=[
                    "oct_fmue_16class", "oct_classifier_broad",
                    "oct_classifier_octdl", "oct_classifier_kermany",
                ],
            ),
            ToolParameter(
                "stride", "string",
                "Analyze every N-th B-scan. Use 1 for small 25-slice benchmark "
                "volumes; use larger values for very large clinical cubes.",
                required=False, default="1",
            ),
            ToolParameter(
                "segment", "string",
                "Whether to run fluid/layer segmentation in addition to "
                "classification. Use false for fast classification-only "
                "benchmarking unless fluid burden is needed.",
                required=False, default="false",
                enum=["true", "false"],
            ),
        ])
    if meta.name == "cfp_dynamic_clip":
        params.extend([
            ToolParameter(
                "candidates_json", "string",
                "JSON list defining the candidate set. Each item can be a string "
                "or an object like {\"label\":\"ICDR grade 3 severe NPDR\", "
                "\"texts\":[\"severe non-proliferative diabetic retinopathy\", "
                "\"4-2-1 rule signs with extensive hemorrhages, venous beading, "
                "or IRMA\"]}. Include an explicit normal/negative comparator "
                "when appropriate.",
                required=False, default="",
            ),
            ToolParameter(
                "candidate_texts", "string",
                "Fallback plain-text candidate list, one line per candidate. "
                "Format: label: prompt text 1 | prompt text 2. Use only if "
                "candidates_json is not convenient.",
                required=False, default="",
            ),
            ToolParameter(
                "task_hint", "string",
                "Short task label, e.g. 'ICDR DR severity grading 0-4' or "
                "'focused differential: RVO vs DR vs hypertensive retinopathy'.",
                required=False, default="",
            ),
            ToolParameter(
                "models", "string",
                "Comma-separated CLIP backends to use. Default: retizero,flair.",
                required=False, default="retizero,flair",
            ),
            ToolParameter(
                "top_k", "string",
                "Number of fused candidates to return, 1-10. Default: 5.",
                required=False, default="5",
            ),
        ])
    # Multi-modal tools: second image path
    if meta.name in {"cross_cfp_ffa", "cross_cfp_ffa_softvote",
                     "cross_cfp_ffa_paired", "paired_bilingual_report"}:
        params.append(ToolParameter(
            "ffa_path", "string", "Path to the paired FFA image (same eye as CFP).",
            required=True, default="",
        ))
    if meta.name == "cross_cfp_oct":
        params.append(ToolParameter(
            "oct_path", "string", "Path to the paired OCT B-scan (same eye as CFP).",
            required=True, default="",
        ))
    if meta.name == "paired_bilingual_report":
        params.append(ToolParameter(
            "languages", "string",
            "Comma-separated language codes from {en,zh,ja,ko,es,de,fr}. "
            "Default: 'en,zh'.",
            required=False, default="en,zh",
        ))

    def fn(image_path: str = "", **kwargs) -> dict:
        if not image_path:
            raise ValueError(f"{meta.name} requires image_path")
        # Capture the owning session in this tool closure. A process-global
        # ``_current_session`` caused concurrent users to resolve files and API
        # clients against whichever toolkit happened to be created last.
        sess = session
        if sess is not None:
            try:
                if meta.name in {
                    "oct_volume_macular", "oct_volume_octcubem", "oct_volume_disc",
                }:
                    image_path = str(
                        sess.resolve_session_path(image_path, allow_dir=True)
                    )
                else:
                    image_path = str(sess.resolve_session_file(image_path))
                for extra_path_arg in ("ffa_path", "oct_path"):
                    if kwargs.get(extra_path_arg):
                        kwargs[extra_path_arg] = str(
                            sess.resolve_session_file(kwargs[extra_path_arg])
                        )
            except Exception as e:
                return {
                    "tool": meta.name,
                    "success": False,
                    "error": f"file access denied: {e}",
                }
        # Tools that render figures (retsam, UWF vessel seg, …) get a
        # persistent output dir under the report root so figures are served
        # by the web UI as /files URLs.
        if (meta.name in ("cfp_retsam_segmentation", "uwf_vessel_segmentation",
                          "ffa_lesion_detection")
                and "output_dir" not in kwargs):
            kwargs["output_dir"] = _adapter_output_dir(meta.name, sess)
        # Focused single-disease screen: scope retsam's biomarker post-
        # processing to the relevant module(s) (glaucoma → disc_cup, DR →
        # lesions, …) when the session set `_focus_quantify_modules`. Keeps
        # the surfaced evidence on-topic and trims CPU; masks are unaffected.
        if meta.name == "cfp_retsam_segmentation" and "quantify_modules" not in kwargs:
            mods = getattr(sess, "_focus_quantify_modules", None) if sess else None
            if mods:
                kwargs["quantify_modules"] = list(mods)
        # Lesion-aware quality tool needs a vision LLM for second opinion;
        # inject the session's client + model when available.
        if meta.name == "cfp_quality_robust" and "llm_client" not in kwargs:
            if sess is not None:
                try:
                    vision_model = sess.vision_model
                    if vision_model is not None:
                        kwargs["llm_client"] = sess._vision_client()
                        kwargs["llm_model"] = vision_model
                except Exception:
                    pass
        # Bilingual-report tool calls the chat LLM itself — pass the session.
        if meta.name == "paired_bilingual_report" and "session" not in kwargs:
            if sess is not None:
                kwargs["session"] = sess
        result = GLOBAL_REGISTRY.predict(meta.name, image_path, **kwargs)
        out = result.to_jsonable()
        # Convert filesystem paths in `figures` to web-served URLs
        if result.figures:
            out["figures"] = result.figures
            out["figure_urls"] = _figures_to_urls(
                result.figures,
                sess,
                namespace=meta.name,
            )
        # Front-load retsam with a compact, LLM-readable headline +
        # strip the heavy per-lesion components arrays. Otherwise the LLM
        # buries itself in the 50-70 KB nested JSON and hallucinates zeros
        # for key clinical metrics like vCDR and DR-hemorrhage area.
        if meta.name == "cfp_retsam_segmentation" and out.get("success") is not False:
            try:
                out = _compact_retsam_payload(out)
            except Exception as _e:
                # Compaction must never break the tool call — fall back to raw.
                out["llm_headline_error"] = f"{type(_e).__name__}: {_e}"
        return out

    return Tool(
        name=meta.name, description=description,
        parameters=params, function=fn,
    )


# ── toolkit ──────────────────────────────────────────────────────────────
class OphToolKit:
    """All-modalities toolkit. Self-registers session-scoped tools too."""

    def __init__(self, session: "OphSession" = None,
                 report_output_root: str | None = None):
        from ..checkpoint_config import disabled_tool_names

        self.session = session
        self.report_output_root = report_output_root or str(OUTPUT_DIR)
        self.last_report_dir: str | None = None
        self.tools: dict[str, Tool] = {}
        self.disabled_adapter_tools = disabled_tool_names()
        self._register_all()

    # ── public API for the agent loop ──────────────────────────────────
    def get_tool(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Tool {name!r} not in {list(self.tools)}")
        return self.tools[name]

    def get_all_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> dict:
        return self.get_tool(tool_name).function(**kwargs)

    # ── tool registration ──────────────────────────────────────────────
    def _register_all(self):
        # 1. Adapter-backed tools (CFP/OCT/UWF/FFA)
        for meta in GLOBAL_REGISTRY.list_tools():
            if meta.name in self.disabled_adapter_tools:
                continue
            self.tools[meta.name] = _make_adapter_tool(meta, self.session)

        # 2. Session-scoped meta tools
        self._add_session_tools()

        # 3. Visual report generation (HTML + PDF)
        self._add_visual_report_tool()

        # 4. Verifier tool — pure LLM helper
        self._add_verifier_tool()

        # 5. Effort-aware meta tool: routes to N tools based on session.effort
        self._add_analyze_image_tool()

        # 6. Sandboxed Python compute — derived metrics from prior tool masks
        self._add_compute_tool()

        # 6b. Modality-aware Grad-CAM heatmap explainer
        self._add_gradcam_tool()

        # 7. Vision LLM "overall impression" — catches obvious lesions that
        # bespoke classifiers miss (e.g. hypertensive retinopathy, papilledema).
        self._add_vision_impression_tool()

    def _add_session_tools(self):
        from ..agent.tools.oct_tools import Tool, ToolParameter

        def _set_current_image(path: str) -> dict:
            try:
                p = (self.session.resolve_session_file(path)
                     if self.session else Path(path).resolve())
            except Exception as e:
                return {"status": "error", "error": f"file access denied: {e}"}
            if not p.exists():
                return {"status": "error", "error": f"file not found: {path}"}
            if self.session:
                self.session.set_image(str(p))
            return {"status": "ok", "image_path": str(p.resolve()),
                    "detected_modality": auto_detect_modality(str(p))}

        self.tools["set_current_image"] = Tool(
            name="set_current_image",
            description=(
                "Register an ophthalmic image as the current focus and auto-detect "
                "its modality (CFP / OCT / UWF / FFA). Subsequent diagnostic tools "
                "may omit `image_path` and operate on this registered image."
            ),
            parameters=[ToolParameter("path", "string", "Absolute path to the image")],
            function=_set_current_image,
        )

        def _detect_modality(path: str) -> dict:
            # Cascade: filename → local CNN classifier → vision LLM → pixel.
            # The CNN (100% val acc) is fast AND accurate, so we only fall
            # through to the vision LLM when the CNN is uncertain (margin <
            # 0.05 between top-2 classes). This avoids over-ruling a
            # confident CNN call with an unreliable LLM read.
            local_path = path
            if self.session is not None:
                # Web prompts intentionally expose a relative /files-style
                # reference instead of the host path. Resolve it through the
                # session boundary before any pixel/model reader opens it.
                local_path = str(self.session.resolve_session_file(path))
            filename = filename_modality_hint(local_path)
            cnn = cnn_modality_hint(local_path)   # None if margin too tight
            verified = filename or cnn
            volume_hint_used = False
            if self.session is not None:
                # A mounted volume is explicit session context. For OCT-volume
                # benchmarks, do not let a single representative B-scan inside
                # that folder overwrite the known OCT modality.
                try:
                    current_modality = (
                        getattr(self.session.context, "current_modality", None) or ""
                    ).upper()
                    current_volume = getattr(self.session.context, "current_volume", None)
                    if current_modality == "OCT" and current_volume:
                        p = Path(local_path).resolve()
                        v = Path(current_volume).resolve()
                        if p == v or v in p.parents:
                            verified = "OCT"
                            volume_hint_used = True
                except Exception:
                    pass
            llm_used = False
            llm_verdict = None
            # Only consult the vision LLM if BOTH filename and CNN punted.
            if verified is None and self.session is not None:
                try:
                    vision_model = self.session.vision_model
                    if vision_model is not None:
                        llm_verdict = llm_classify_modality(
                            local_path,
                            self.session._vision_client(),
                            vision_model,
                        )
                        if llm_verdict:
                            verified = llm_verdict
                            llm_used = True
                except Exception:
                    pass
            # Final fall-through to pixel heuristic
            if verified is None:
                verified = pixel_modality_hint(local_path)
            # Update session state so downstream tools route correctly
            if self.session is not None:
                self.session.context.current_modality = verified
            return {
                "modality": verified,
                "filename_hint": filename,
                "cnn_hint": cnn,
                "llm_verdict": llm_verdict,
                "llm_used": llm_used,
                "volume_hint_used": volume_hint_used,
                "path": path,
            }

        self.tools["detect_modality"] = Tool(
            name="detect_modality",
            description=(
                "Classify an image as CFP, OCT, UWF, or FFA. Uses a vision LLM "
                "for accurate detection (slow ~3 s), with a filename/pixel "
                "fallback. Call this when you're unsure which modality the "
                "current image is, especially before invoking a modality-"
                "specific tool."
            ),
            parameters=[ToolParameter("path", "string", "Path to the image")],
            function=_detect_modality,
        )

    def _add_visual_report_tool(self):
        """Register build_visual_report tool which produces an HTML + PDF
        clinical report with embedded figures."""
        from ..agent.tools.oct_tools import Tool, ToolParameter
        from ..agent.report_builder import build_visual_report
        from ..inference.model_registry import create_default_registry
        from ..inference.predictor import OphPredictor

        # Reuse the existing OCT predictor (works for OCT classifiers + segmentors).
        # For pure CFP images the report still has value: shows GradCAM of any OCT
        # classifier if requested, fluid masks, etc. We pass-through `findings_json`
        # so the agent's accumulated multi-modal findings get embedded in the PDF.
        registry = create_default_registry()
        predictor = OphPredictor(registry)
        out_root = Path(self.report_output_root)
        out_root.mkdir(parents=True, exist_ok=True)

        def _build(image_path: str = "",
                   report_markdown: str = "",
                   findings_json: str = "{}",
                   classifier: str = "oct_classifier_octdl") -> dict:
            if not image_path:
                return {"status": "error", "error": "image_path required"}
            try:
                findings = json.loads(findings_json) if findings_json else {}
            except json.JSONDecodeError:
                findings = {}
            import time as _t
            stem = Path(image_path).stem
            run_dir = out_root / f"{stem}_{int(_t.time())}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = build_visual_report(
                    image_path=image_path,
                    registry=registry,
                    predictor=predictor,
                    findings=findings,
                    clinical_report_md=report_markdown,
                    output_dir=run_dir,
                    classifier_model_name=classifier,
                )
                self.last_report_dir = str(run_dir)
                if self.session:
                    self.session.context.last_report = {
                        "html": result.get("report_html"),
                        "pdf": result.get("report_pdf"),
                    }
                return {
                    "status": "success",
                    "report_html": result.get("report_html"),
                    "report_pdf": result.get("report_pdf"),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        self.tools["build_visual_report"] = Tool(
            name="build_visual_report",
            description=(
                "Generate a styled HTML + PDF clinical report with embedded "
                "figures (original image, Grad-CAM heatmap, segmentation "
                "overlays, detection boxes, and your written clinical text). "
                "Call this AFTER all diagnostic tools have run AND you have "
                "written the clinical markdown. Pass markdown text as "
                "`report_markdown` and the accumulated tool findings as "
                "`findings_json`."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Image path",
                              required=False, default=""),
                ToolParameter("report_markdown", "string", "Clinical markdown text"),
                ToolParameter("findings_json", "string",
                              "JSON of all findings", required=False, default="{}"),
                ToolParameter("classifier", "string",
                              "OCT classifier for Grad-CAM",
                              required=False, default="oct_classifier_octdl",
                              enum=["oct_classifier_octdl", "oct_classifier_kermany",
                                    "oct_classifier_broad"]),
            ],
            function=_build,
        )

    def _add_verifier_tool(self):
        from ..agent.tools.oct_tools import Tool, ToolParameter

        def _verify_findings(findings_json: str = "", image_path: str = "",
                              **_) -> dict:
            """Cross-check finding consistency. Pure rule-based + heuristics.
            The LLM itself does the deeper natural-language verification.

            Resilient to the LLM forgetting (or omitting) `findings_json`:
            we auto-reconstruct it from `session.context.analyses` when
            empty so the chain doesn't dead-lock on tool-call malformation.
            """
            # Auto-reconstruct findings from the session if the caller didn't
            # supply them — common with DeepSeek / Sonnet which sometimes
            # strip the JSON arg.
            parsed_findings = None
            if isinstance(findings_json, str):
                if findings_json.strip():
                    try:
                        parsed_findings = json.loads(findings_json)
                    except Exception:
                        return {
                            "status": "warning",
                            "verify_passed": False,
                            "error": (
                                "findings_json was not valid JSON; verifier "
                                "could not verify the evidence"
                            ),
                            "recommendation": (
                                "Re-call verify_findings with findings_json as "
                                "a JSON object {tools_run: [...], results: [...]} "
                                "containing the tool outputs. If you cannot "
                                "construct it, pass an empty string and the "
                                "verifier will read from session context itself."
                            ),
                        }
            else:
                parsed_findings = findings_json

            findings_are_empty = (
                parsed_findings is None
                or parsed_findings == {}
                or parsed_findings == []
                or (
                    isinstance(parsed_findings, dict)
                    and not parsed_findings.get("tools_run")
                    and not parsed_findings.get("results")
                )
            )
            if findings_are_empty:
                results = []
                tools_run: list[str] = []
                if self.session is not None:
                    analyses = getattr(self.session.context, "analyses", {}) or {}
                    img = image_path or self.session.context.current_image or ""
                    if hasattr(self.session, "_analyses_for_image"):
                        by_tool = self.session._analyses_for_image(img)
                    else:
                        by_tool = analyses.get(img) or {}
                    # If the specified path has no analyses, fall back to
                    # the most recent image's results.
                    if not by_tool and analyses:
                        most_recent_img = next(iter(reversed(analyses)))
                        by_tool = analyses[most_recent_img]
                    for tname, tres in by_tool.items():
                        if tname == "verify_findings":
                            continue
                        tools_run.append(tname)
                        if isinstance(tres, dict):
                            results.append({
                                "tool": tname,
                                "predictions": tres.get("predictions") or {},
                                "confidence": tres.get("confidence"),
                                "undetermined": tres.get("undetermined", False),
                            })
                data = {"tools_run": tools_run, "results": results}
                auto_constructed = True
            else:
                if not isinstance(parsed_findings, dict):
                    return {
                        "status": "warning",
                        "verify_passed": False,
                        "error": (
                            "findings_json must decode to a JSON object with "
                            "tools_run and results fields"
                        ),
                        "recommendation": (
                            "Re-call verify_findings with findings_json as a "
                            "JSON object {tools_run: [...], results: [...]} "
                            "containing the tool outputs. If you cannot "
                            "construct it, pass an empty string and the "
                            "verifier will read from session context itself."
                        ),
                    }
                data = parsed_findings
                auto_constructed = False

            # Normalize `results`: the LLM sometimes formats it as an OBJECT
            # keyed by tool name ({"ffa_classification": {...}}) instead of the
            # expected LIST of entries. Iterating a dict yields its keys
            # (strings), so downstream `entry.get(...)` would raise
            # "'str' object has no attribute 'get'". Convert to the list shape,
            # then drop any non-dict stragglers so every loop below is safe.
            if isinstance(data, dict) and isinstance(data.get("results"), dict):
                data["results"] = [
                    {
                        "tool": tname,
                        "predictions": (p.get("predictions")
                                        if isinstance(p, dict) and "predictions" in p
                                        else (p if isinstance(p, dict) else {})),
                        "confidence": (p.get("confidence")
                                       if isinstance(p, dict) else None),
                        "undetermined": (p.get("undetermined", False)
                                         if isinstance(p, dict) else False),
                    }
                    for tname, p in data["results"].items()
                ]
            if isinstance(data, dict):
                data["results"] = [e for e in data.get("results", [])
                                   if isinstance(e, dict)]

            issues: list[str] = []
            warnings: list[str] = []
            warning_categories: dict[str, list[str]] = {
                "quality": [],
                "confidence": [],
                "diagnostic_conflict": [],
                "other": [],
            }
            evidence: dict[str, Any] = {}

            if not data.get("results"):
                issues.append(
                    "No structured tool results were available to verify."
                )

            def add_warning(category: str, message: str) -> None:
                warnings.append(message)
                warning_categories.setdefault(category, []).append(message)

            # Prefer the cached structured tool result over a lossy
            # planner-written findings summary. This keeps safety fields such as
            # quality rejection and etiology guards available to the verifier.
            def _cached_result(tool_name: str) -> dict:
                if self.session is None:
                    return {}
                analyses = getattr(self.session.context, "analyses", {}) or {}
                img = image_path or self.session.context.current_image or ""
                if hasattr(self.session, "_analyses_for_image"):
                    by_tool = self.session._analyses_for_image(img)
                else:
                    by_tool = analyses.get(img) or {}
                if not by_tool and analyses:
                    by_tool = analyses[next(iter(reversed(analyses)))]
                result = by_tool.get(tool_name)
                return result if isinstance(result, dict) else {}

            def _entry(tool_name: str) -> dict:
                return next(
                    (entry for entry in data.get("results", [])
                     if entry.get("tool") == tool_name),
                    {},
                )

            def _payload(tool_name: str) -> tuple[dict, dict]:
                entry = _entry(tool_name)
                cached = _cached_result(tool_name)
                source = cached or entry
                predictions = source.get("predictions")
                if not isinstance(predictions, dict):
                    predictions = {}
                merged = dict(predictions)
                for container in (source, entry):
                    for key, value in container.items():
                        if key not in {"tool", "predictions"}:
                            merged.setdefault(key, value)
                return merged, entry

            # Collect all tools mentioned
            tools_run = data.get("tools_run", [])
            # Check for low-confidence (undetermined) results
            for entry in data.get("results", []):
                tname = entry.get("tool", "?")
                conf = entry.get("confidence")
                if entry.get("undetermined"):
                    add_warning(
                        "confidence",
                        f"{tname} reports undetermined output (conf<threshold)",
                    )
                if conf is not None:
                    try:
                        numeric_conf = float(conf)
                    except (TypeError, ValueError):
                        numeric_conf = None
                    if numeric_conf is not None and numeric_conf < 0.3:
                        add_warning(
                            "confidence",
                            f"{tname} confidence is very low ({numeric_conf:.2f})",
                        )

            # Quality limitations remain scoped to acquisition quality, while
            # still being visible to the final report and over-confidence checks.
            quality_reject = None
            for quality_tool in (
                    "cfp_efiqa", "cfp_eyeq", "cfp_quality_robust"):
                quality_payload, quality_entry = _payload(quality_tool)
                if not quality_payload and not quality_entry:
                    continue
                quality_label = str(
                    quality_payload.get("quality")
                    or quality_payload.get("quality_label")
                    or ""
                ).strip().lower()
                rejected = bool(
                    quality_payload.get("is_rejected")
                    or quality_payload.get("rejected")
                    or quality_label in {"reject", "rejected", "poor"}
                )
                if rejected:
                    quality_reject = {
                        "tool": quality_tool,
                        "quality": quality_label or "reject",
                        "usable_area_ratio": quality_payload.get(
                            "usable_area_ratio"
                        ),
                    }
                    add_warning(
                        "quality",
                        f"{quality_tool} rejected the CFP image; disease "
                        "attribution must remain limited-confidence and should "
                        "be confirmed on reacquired or complementary imaging.",
                    )
                    evidence["image_quality"] = quality_reject
                    break

            if quality_reject:
                for entry in data.get("results", []):
                    if entry.get("tool") == quality_reject["tool"]:
                        continue
                    if (entry.get("confidence") or 0) > 0.7:
                        add_warning(
                            "confidence",
                            f"{quality_reject['tool']} rejected the image but "
                            f"{entry.get('tool', '?')} is highly confident; "
                            "check for downstream over-confidence.",
                        )

            clip_payload, _ = _payload("cfp_clip_ensemble")
            clip_agreement = str(
                clip_payload.get("agreement_level") or ""
            ).strip().lower()
            clip_probability = clip_payload.get("fused_top1_probability")
            if clip_agreement == "low":
                add_warning(
                    "confidence",
                    "cfp_clip_ensemble reports low inter-model agreement; "
                    "its fused top-1 is a differential signal, not a diagnosis.",
                )
                evidence["clip_ensemble"] = {
                    "agreement_level": clip_agreement,
                    "fused_top1": clip_payload.get("fused_top1"),
                    "fused_top1_probability": clip_probability,
                }
            if (isinstance(clip_probability, (int, float))
                    and clip_probability < 0.3):
                add_warning(
                    "confidence",
                    "cfp_clip_ensemble fused top-1 probability is very low "
                    f"({clip_probability:.2f}).",
                )

            retsam_payload, retsam_entry = _payload(
                "cfp_retsam_segmentation"
            )
            retsam_headline = retsam_payload.get("llm_headline")
            if not isinstance(retsam_headline, dict):
                retsam_headline = retsam_payload
            hemorrhage_etiology = retsam_headline.get(
                "hemorrhage_etiology"
            )
            if not isinstance(hemorrhage_etiology, dict):
                hemorrhage_etiology = {}
            dr_signs = retsam_headline.get(
                "diabetic_retinopathy_signs"
            )
            if not isinstance(dr_signs, dict):
                dr_signs = {}
            dr_signal = (
                dr_signs.get("dr_signal_confidence")
                or retsam_entry.get("dr_signal_confidence")
            )
            amd_signs = retsam_headline.get("amd_signs")
            if not isinstance(amd_signs, dict):
                amd_signs = {}
            amd_patch_present = bool(
                amd_signs.get("patch_hemorrhage_present")
                or retsam_entry.get("AMD_patch_hemorrhage_present")
            )
            hemorrhage_ambiguous = (
                hemorrhage_etiology.get("status") == "ambiguous"
                or str(dr_signal).lower() == "ambiguous"
            )
            if hemorrhage_ambiguous:
                add_warning(
                    "diagnostic_conflict",
                    "ReT-SAM's DR-hemorrhage and AMD-patch-hemorrhage "
                    "heads substantially overlap. Treat them as one "
                    "hemorrhagic lesion with unresolved etiology; do not "
                    "diagnose concurrent DR from component counts alone. "
                    "Explicitly consider neovascular AMD/PCV and, when "
                    "independent high-myopia signs are present, myopic CNV. "
                    "Keep the active hemorrhagic macular lesion separate from "
                    "background myopic atrophy in the primary impression.",
                )
                evidence["hemorrhage_etiology"] = hemorrhage_etiology
                evidence["macular_hemorrhage_differential"] = [
                    "neovascular AMD/PCV",
                    "myopic CNV if high myopia is independently supported",
                ]
            elif str(dr_signal).lower() == "high" and amd_patch_present:
                add_warning(
                    "diagnostic_conflict",
                    "ReT-SAM reports both a high DR-like morphology signal "
                    "and AMD-pattern patch hemorrhage. Confirm lesion "
                    "distribution or obtain OCT/angiography before assigning "
                    "the hemorrhage to DR.",
                )

            dr421_payload, _ = _payload("cfp_dr_421_assessment")
            dr421_guard = dr421_payload.get("etiology_guard")
            if not isinstance(dr421_guard, dict):
                dr421_guard = {}
            if (dr421_payload.get("eligible_for_dr_grading") is False
                    or dr421_guard.get("status") == "ambiguous"):
                add_warning(
                    "diagnostic_conflict",
                    "cfp_dr_421_assessment suppressed its DR severity proxy "
                    "because the hemorrhage cannot be assigned to DR. Do not "
                    "recover severe NPDR/PDR from the unadjusted lesion counts.",
                )
                evidence["dr_421_etiology_guard"] = {
                    "eligible_for_dr_grading": False,
                    "etiology_guard": dr421_guard,
                }

            dr_workup_payload, _ = _payload("cfp_dr_workup")
            if (dr_workup_payload.get("do_not_report_as_pdr")
                    or dr_workup_payload.get("pdr_eligible_for_reporting") is False):
                add_warning(
                    "diagnostic_conflict",
                    "cfp_dr_workup marked the PDR cascade as strongly "
                    "confounded by non-DR pathology. Its raw PDR category is "
                    "audit-only and must not be reported as a positive finding.",
                )
                evidence["pdr_confound_guard"] = {
                    "pdr_confound_severity": dr_workup_payload.get(
                        "pdr_confound_severity"
                    ),
                    "raw_pdr_category": dr_workup_payload.get(
                        "raw_pdr_category_before_confound_guard"
                    ),
                }

            # A quality rejection is a scoped acquisition limitation, not a
            # disease-level contradiction. Lesion-heavy CFPs can be rejected by
            # EyeQ even when independent diagnostic tools consistently detect
            # pathology.
            eyeq = next((e for e in data.get("results", []) if e.get("tool") == "cfp_eyeq"), None)
            if eyeq and eyeq.get("predictions", {}).get("is_rejected"):
                pathology_votes = _diagnostic_family_votes(data)
                if any(family != "normal" for family in pathology_votes):
                    add_warning(
                        "quality",
                        "EyeQ flagged the image as Reject while independent "
                        "disease tools detected pathology. Treat image quality "
                        "as a scoped limitation; lesion burden may have "
                        "confounded the quality classifier.",
                    )

            # Cross-modality consistency: if FMUE OCT says nAMD high conf, but
            # there's no fluid finding from a fluid model, warn.
            fmue = next((e for e in data.get("results", []) if e.get("tool") == "oct_fmue_16class"), None)
            if fmue and fmue.get("predictions", {}).get("predicted_class") in ("nAMD", "PCV", "DME"):
                fluid = next((e for e in data.get("results", []) if e.get("tool") == "oct_fluid_segmentor"), None)
                if fluid:
                    areas = fluid.get("predictions", {}).get("class_areas", {})
                    total_fluid = sum(v for k, v in areas.items() if k.lower() != "background")
                    if total_fluid < 100:
                        add_warning(
                            "diagnostic_conflict",
                            f"FMUE predicted {fmue['predictions']['predicted_class']} "
                            f"(exudative pathology) but fluid segmentor found minimal fluid "
                            f"({total_fluid} px). Consider re-checking foveal slice."
                        )

            # ── PDR confound cross-check ──────────────────────────────────
            # If PDR cascade ran and any CLIP top-1 is a non-DR class with
            # moderate-high probability, suggest the CLIP ensemble.
            # NOTE: RVO is intentionally NOT in this list — DR and RVO often
            # coexist (hypertensive diabetics), so a high CLIP-RVO score does
            # not invalidate PDR. Same logic as the cfp_dr_workup confound.
            NON_DR = {"Pathological myopia", "Pathologic Myopia",
                      "Retinal detachment", "Suspected cataract", "Cataract",
                      "Age-related macular degeneration",
                      "Retinitis pigmentosa"}
            pdr_e = next((e for e in data.get("results", [])
                          if e.get("tool") == "cfp_pdr_cascade"), None)
            clip_e = next((e for e in data.get("results", [])
                           if e.get("tool") == "cfp_clip_multi_disease"), None)
            if pdr_e and clip_e:
                pdr_cat = (pdr_e.get("predictions") or {}).get("category") or ""
                top3 = (clip_e.get("predictions") or {}).get("top_3") or []
                non_dr_hits = [
                    h for h in top3
                    if (h.get("label_en") or h.get("label") or "") in NON_DR
                    and float(h.get("probability") or 0) >= 0.25
                ]
                if "PDR" in pdr_cat and non_dr_hits:
                    add_warning(
                        "diagnostic_conflict",
                        f"PDR cascade said '{pdr_cat}' but CLIP suggests "
                        f"non-DR pathology: {non_dr_hits[0].get('label_en')} "
                        f"({non_dr_hits[0].get('probability'):.0%}). "
                        "PDR cascade's training set lacked these negatives; "
                        "treat the PDR label as suspect."
                    )

            # ── Decide next action (ESCALATING replanner, not repeating) ──
            # Principle: a replan should escalate to a DIFFERENT, cheaper,
            # more-informative action — never re-run the same battery (that
            # can't manufacture agreement and just churns). The ladder, used
            # at most ONCE total:
            #   1. classifiers deadlock + no ensemble yet → cfp_clip_ensemble
            #   2. classifiers deadlock + ensemble done + vision available →
            #      ONE targeted vision-LLM differential (adaptive prompt)
            #   3. otherwise → stop escalating, finalise as undetermined
            next_actions: list[dict] = []
            tools_already = {e.get("tool") for e in data.get("results", [])}
            had_clip_ensemble = "cfp_clip_ensemble" in tools_already
            effort = getattr(self.session, "effort", "medium") \
                if self.session else "medium"
            vision_block_reason = None
            if self.session is not None and hasattr(
                    self.session, "_vision_policy_block_reason"):
                vision_block_reason = self.session._vision_policy_block_reason(
                    image_path or getattr(self.session.context, "current_image", None),
                    for_escalation=True,
                )

            # Count how many verify rounds have happened (incl. this one) so we
            # escalate at most once and never loop.
            n_verify_rounds = 1
            if self.session is not None:
                n_verify_rounds = sum(
                    1
                    for m in getattr(self.session, "messages", [])
                    if m.get("role") == "assistant"
                    for tc in (m.get("tool_calls") or [])
                    if (tc.get("function", {}) or {}).get("name") == "verify_findings"
                ) or 1
            first_round = n_verify_rounds <= 1

            diagnostic_votes = _diagnostic_family_votes(data)
            primary_votes = _primary_diagnostic_votes(data)
            if len(primary_votes) > 1:
                vote_summary = "; ".join(
                    f"{family}: {', '.join(tools)}"
                    for family, tools in primary_votes.items()
                )
                add_warning(
                    "diagnostic_conflict",
                    "Mutually exclusive primary-classification outputs "
                    f"disagree ({vote_summary}).",
                )
            # Quality and confidence limitations stay scoped to those levels.
            # Only a concrete disease-level contradiction opens escalation.
            deadlock = bool(
                issues or warning_categories["diagnostic_conflict"]
            )
            vision_ok = bool(getattr(self.session, "vision_available", False)) \
                if self.session is not None else False

            if first_round and deadlock:
                if not had_clip_ensemble:
                    next_actions.append({
                        "tool": "cfp_clip_ensemble",
                        "reason": (
                            "Three independent CLIPs will tiebreak the "
                            "differential before finalising."
                        ),
                    })
                elif vision_ok and not vision_block_reason:
                    cands = _verifier_top_candidates(data)
                    next_actions.append({
                        "tool": "vision_impression",
                        "args": {"focus": _build_differential_focus(cands)},
                        "reason": (
                            "Specialised classifiers deadlocked at low "
                            "confidence. Run ONE targeted visual differential "
                            "(cheaper and more informative than re-running the "
                            "classifiers) to adjudicate between: "
                            + (", ".join(cands) or "the top candidates") + "."
                        ),
                    })
                elif vision_ok and vision_block_reason:
                    add_warning(
                        "other",
                        "Vision escalation skipped by effort policy: "
                        f"{vision_block_reason}",
                    )
                # else: vision unavailable / escalation exhausted -> finalise
                # the differential honestly as undetermined (no next_action).

            # ── Independent Verifier agent (a SEPARATE LLM second opinion) ──
            # Distinct from the orchestrator/Planner LLM: it sees ONLY the raw
            # tool outputs and judges consistency/sufficiency on their own
            # merits. Augments the rule-based checks above; degrades to
            # rule-only when no LLM is available.
            # LLM verification, gated by effort tier. Both paths are SEPARATE
            # LLM agents that see ONLY the raw tool outputs (not the planner's
            # chat) and degrade to rule-only when no LLM is available:
            #   high          → 1-shot independent Verifier (second opinion)
            #   max / ultra    → multi-agent debate panel (challenger/defender/judge)
            independent_review = None
            debate_review = None
            verifier_mode = get_effort_policy(effort).verifier_mode
            if verifier_mode == "debate":
                debate_review = _run_debate(self.session, data)
                if debate_review and not debate_review.get("resolved"):
                    reason = (debate_review.get("reason") or "").strip()
                    if reason:
                        add_warning(
                            "diagnostic_conflict",
                            f"Debate panel (unresolved): {reason}",
                        )
                    sug = (debate_review.get("request_tool") or "").strip()
                    if (first_round and not next_actions
                            and sug in _SAFE_ESCALATIONS
                            and sug not in tools_already
                            and not (sug == "vision_impression"
                                     and vision_block_reason)):
                        next_actions.append({
                            "tool": sug,
                            "reason": f"Debate panel requested: {reason}",
                        })
            else:
                # Only a genuine CONFLICT influences the gate (the Verifier's
                # core job); an "insufficient" verdict is recorded but NOT
                # forced into the undetermined path, to avoid over-hedging on a
                # forced-choice benchmark.
                independent_review = _independent_verifier_review(
                    self.session, data, rule_issues=issues, rule_warnings=warnings)
                if independent_review and independent_review.get("verdict") == "conflict":
                    inote = (independent_review.get("note") or "").strip()
                    if inote:
                        add_warning(
                            "diagnostic_conflict",
                            f"Independent verifier (conflict): {inote}",
                        )
                    sug = (independent_review.get("suggested_tool") or "").strip()
                    if (first_round and not next_actions
                            and sug in _SAFE_ESCALATIONS
                            and sug not in tools_already
                            and not (sug == "vision_impression"
                                     and vision_block_reason)):
                        next_actions.append({
                            "tool": sug,
                            "reason": f"Independent verifier flagged conflict: {inote}",
                        })

            # verify_passed allows finalising once there is nothing left to
            # escalate (soft low-confidence warnings alone must NOT block
            # forever — they convert to an "undetermined" caveat in the report).
            verify_passed = not issues and not next_actions

            if next_actions:
                rec = ("DO NOT FINALISE YET. Run the single suggested "
                       "next_action (a targeted differential / ensemble), then "
                       "call verify_findings once more.")
            elif warning_categories["diagnostic_conflict"]:
                rec = (
                    "Escalation is exhausted and disease-level evidence remains "
                    "conflicting. Finalise with an undetermined / low-confidence "
                    "differential and recommend confirmatory imaging. Do not "
                    "force a single high-confidence label."
                )
            elif warnings:
                rec = (
                    "Disease-level findings are sufficiently consistent to "
                    "finalise. Preserve the listed quality or confidence "
                    "limitations at their own level; do not convert them into "
                    "overall diagnostic uncertainty."
                )
            else:
                rec = "Findings are mutually consistent — you may finalise."

            return {
                "status": "ok",
                "input_source": "session_cache" if auto_constructed else "provided",
                "n_tools_run": len(data.get("results", [])),
                "issues": issues,
                "warnings": warnings,
                "warning_categories": warning_categories,
                "diagnostic_votes": diagnostic_votes,
                "primary_diagnostic_votes": primary_votes,
                "diagnostic_status": (
                    "conflict" if warning_categories["diagnostic_conflict"]
                    else "consistent"
                ),
                "evidence": evidence,
                "verify_passed": verify_passed,
                "next_actions": next_actions,
                "recommendation": rec,
                "independent_review": independent_review,
                "debate_review": debate_review,
            }

        self.tools["verify_findings"] = Tool(
            name="verify_findings",
            description=(
                "Cross-check the tool outputs collected so far for mutual "
                "consistency. Returns {verify_passed, warnings, next_actions, "
                "recommendation}. CALL THIS BEFORE FINAL REPORT GENERATION; "
                "if verify_passed=False or next_actions is non-empty, DO NOT "
                "finalise — execute next_actions then call verify again.\n\n"
                "Args:\n"
                "  findings_json (optional) — JSON string of "
                "{tools_run:[...], results:[...]}. If you pass an empty "
                "string, an empty object, or omit it, the verifier "
                "auto-reconstructs from the session's cached tool results.\n"
                "  image_path (optional) — which image these findings refer to."
            ),
            parameters=[
                ToolParameter("findings_json", "string",
                              ("Optional JSON-encoded findings. Pass empty "
                               "string '' or {} and the verifier will read "
                               "the session's accumulated tool results itself."),
                              required=False, default=""),
                ToolParameter("image_path", "string",
                              "Optional: image these findings refer to",
                              required=False, default=""),
            ],
            function=_verify_findings,
        )

    def _add_analyze_image_tool(self):
        """`analyze_image(modality, task, image_path, ffa_path?)` — runs
        N compatible tools per the session's effort setting and returns
        all results merged for the LLM to summarise."""
        from ..agent.tools.oct_tools import Tool, ToolParameter

        def _analyze(modality: str, task: str, image_path: str = "",
                     ffa_path: str = "") -> dict:
            modality = modality.upper()
            task = task.lower()
            effort = getattr(self.session, "effort", "low") if self.session else "low"
            effort_policy = get_effort_policy(effort)

            # Pick image path if omitted
            img = image_path or (
                self.session.context.current_image if self.session else ""
            )
            if not img:
                return {"error": "no image_path given and session has no current image"}
            if self.session:
                try:
                    img = str(self.session.resolve_session_file(img))
                    if ffa_path:
                        ffa_path = str(self.session.resolve_session_file(ffa_path))
                except Exception as e:
                    return {"error": f"file access denied: {e}"}

            # Cross-modal case
            if modality in {"MULTI", "CROSS", "CFP+FFA"} and ffa_path:
                tools_meta = GLOBAL_REGISTRY.cross_modal_tools_for(["CFP", "FFA"])
            else:
                tools_meta = GLOBAL_REGISTRY.tools_for(modality, task)

            tools_meta = [
                meta for meta in tools_meta
                if meta.name not in self.disabled_adapter_tools
            ]

            if not tools_meta:
                return {"error": f"no tools for modality={modality} task={task}"}

            chosen = (
                tools_meta
                if effort_policy.meta_tool_limit is None
                else tools_meta[:effort_policy.meta_tool_limit]
            )
            results: list[dict] = []
            for meta in chosen:
                try:
                    if meta.modality == "multi":
                        r = GLOBAL_REGISTRY.predict(meta.name, img, ffa_path=ffa_path)
                    else:
                        r = GLOBAL_REGISTRY.predict(meta.name, img)
                    results.append({"tool": meta.name, "result": r.to_jsonable()})
                except Exception as e:
                    results.append({"tool": meta.name, "error": f"{type(e).__name__}: {e}"})

            # Escalation rule for LOW: re-run if top tool conf < 0.6
            if effort == "low" and results:
                top_conf = (results[0].get("result", {}) or {}).get("confidence") or 0
                if top_conf < 0.6 and len(tools_meta) > 1:
                    nxt = tools_meta[1]
                    try:
                        r2 = (GLOBAL_REGISTRY.predict(nxt.name, img, ffa_path=ffa_path)
                              if nxt.modality == "multi"
                              else GLOBAL_REGISTRY.predict(nxt.name, img))
                        results.append({"tool": nxt.name, "result": r2.to_jsonable(),
                                        "escalated": True,
                                        "reason": f"top conf {top_conf:.2f} < 0.60"})
                    except Exception as e:
                        results.append({"tool": nxt.name, "error": str(e),
                                        "escalated": True})

            return {
                "modality": modality, "task": task, "effort": effort,
                "tools_run": [r["tool"] for r in results],
                "results": results,
            }

        self.tools["analyze_image"] = Tool(
            name="analyze_image",
            description=(
                "Run the best tool(s) for a given (modality, task) pair on the "
                "current image, honouring the session's thinking-effort "
                "setting. low=1 tool (escalates if conf<0.6), medium=2 tools, "
                "high=all compatible tools. Use this when you need a routine "
                "answer and don't want to micro-manage which model. For "
                "paired CFP+FFA, set modality='multi' and pass both paths."
            ),
            parameters=[
                ToolParameter("modality", "string",
                              "CFP | OCT | UWF | FFA | multi",
                              enum=["CFP", "OCT", "UWF", "FFA", "multi"]),
                ToolParameter("task", "string",
                              "classification | segmentation | detection | quality",
                              enum=["classification", "segmentation", "detection", "quality"]),
                ToolParameter("image_path", "string",
                              "Primary image (CFP/OCT/UWF/FFA depending on modality)",
                              required=False, default=""),
                ToolParameter("ffa_path", "string",
                              "Paired FFA path when modality='multi' and both CFP+FFA available",
                              required=False, default=""),
            ],
            function=_analyze,
        )

    def _add_compute_tool(self):
        """Sandboxed Python execution for derived metrics that combine
        previous tool outputs (masks + landmarks)."""
        from ..agent.tools.oct_tools import Tool, ToolParameter
        from .compute_sandbox import run_compute

        def _compute(code: str) -> dict:
            if self.session is None:
                return {"ok": False, "error": "no active session"}
            out = run_compute(code, self.session)
            # Surface saved figures to the web UI by translating to URLs.
            figs = out.get("saved_figures") or {}
            if figs:
                out["figure_urls"] = _figures_to_urls(
                    figs,
                    self.session,
                    namespace="compute",
                )
            return out

        self.tools["compute"] = Tool(
            name="compute",
            description=(
                "Run a short Python snippet to compute a derived metric from "
                "the masks + landmarks of previously called tools. Use this "
                "whenever the user asks for a quantity that is a *combination* "
                "of existing tool outputs (lesion area within N mm of a "
                "landmark, hemispheric asymmetry of a thickness map, distance "
                "between two structures, custom ROI averages, etc.).\n\n"
                "EXPOSED IN THE SANDBOX (read-only):\n"
                "  np, ndi (scipy.ndimage)\n"
                "  tools[<tool_name>]    — flat dict of each tool's "
                "`predictions` from this session\n"
                "  figures[<tool_name>]  — file paths to that tool's saved PNGs\n"
                "  masks[<short_name>]   — numpy arrays, lazy-loaded:\n"
                "      • 'retsam.<channel>' for CFP retsam binary masks\n"
                "      • 'fluid_segmentation' (int H×W, 0/1/2/3=BG/IRF/SRF/PED)\n"
                "      • 'layer_segmentation' (int H×W, 0..9 retinal regions)\n"
                "  original_image        — HxWx3 uint8 RGB numpy array of the\n"
                "      session's CURRENT image (None if no image registered).\n"
                "      Use this for any 'overlay X on the original' task (the\n"
                "      `blend`/`save_figure` helpers are easier than raw PIL).\n"
                "  original_image_path   — session-relative reference to it\n"
                "  load_image(path)      — convenience: load any RGB image\n"
                "      (HxWx3 uint8) attached to this session — for multi-image overlays\n"
                "  blend(base, color, alpha=0.5, where=None)  — alpha-blend\n"
                "      two HxWx3 images; if `where` (HxW bool) given, blend\n"
                "      only inside. Returns uint8 RGB.\n"
                "  landmarks             — auto-extracted named coordinates:\n"
                "      • macula_center_xy, od_center_xy are (x, y) pixels.\n"
                "      • macula_center_yx, od_center_yx are (y, x) for numpy row/col indexing.\n"
                "      • macula_center_px, od_center_px are legacy aliases for *_xy.\n"
                "      • oct_disc_centroid_px, cpRNFLT_sectors, rnfl_tsni\n"
                "  save_figure(arr_or_pil, name)  — persist a derived image\n"
                "      so it shows up in the chat output\n\n"
                "RESTRICTIONS:\n"
                "  `np` (numpy) and `ndi` (scipy.ndimage) are ALREADY available "
                "through restricted numeric interfaces — do not import modules. "
                "There is no arbitrary file, network, process, native-library, "
                "open()/eval()/exec(), or serialization access. 30s maximum. "
                "Use `print(...)` to communicate values back to yourself.\n\n"
                "EXAMPLE — 'lesion area within 3 mm of the macula on a CFP':\n"
                "  cx, cy = landmarks['macula_center_px']\n"
                "  lesion = masks['retsam.lesion_amd'] | masks['retsam.lesion_general']\n"
                "  H, W = lesion.shape\n"
                "  pixel_um = 7.5   # typical 45-deg CFP at 1024 px; override if known\n"
                "  r_px = 3000 / pixel_um\n"
                "  y, x = np.ogrid[:H, :W]\n"
                "  roi = (x - cx)**2 + (y - cy)**2 <= r_px**2\n"
                "  area_mm2 = (lesion & roi).sum() * (pixel_um**2) / 1e6\n"
                "  print(f'Lesion area within 3 mm of macula: {area_mm2:.3f} mm^2')\n"
                "  save_figure(lesion & roi, 'macula_3mm_lesion')\n\n"
                "CALL ONLY AFTER the underlying segmentation/detection tools "
                "have already populated the session — otherwise `masks` / "
                "`landmarks` will be empty."
            ),
            parameters=[
                ToolParameter(
                    "code", "string",
                    "Python source. Multi-line allowed. Use `print()` to "
                    "report numbers; call `save_figure(arr, name)` for overlays.",
                ),
            ],
            function=_compute,
        )

    def _add_gradcam_tool(self):
        """Modality-aware Grad-CAM heatmap explainer.

        Routes to a CNN-based main classifier per the session's current
        modality; reuses the existing visualization.GradCAM machinery.
        ViT classifiers are explicitly rejected in v1 (need attention
        rollout, not last-Conv2d hook).
        """
        from ..agent.tools.oct_tools import Tool, ToolParameter

        # Modality → (adapter_name, friendly_note)
        _DEFAULT_PER_MODALITY = {
            "CFP":  "cfp_paired5",          # ResNet-50
            "OCT":  "oct_fmue_16class",     # CNN
            "UWF":  "uwf_disease_7class",   # ConvNeXt-Tiny
            "FFA":  "ffa_paired5",          # ResNet-50
        }
        _VIT_CLASSIFIERS = {
            "cfp_clip_multi_disease", "cfp_retizero", "cfp_flair",
            "cfp_clip_ensemble", "cfp_glaucoma", "cfp_glaucoma_workup",
            "cfp_pdr_cascade", "cfp_dr_workup", "cfp_dynamic_clip",
        }

        def _resolve_class_idx(adapter, target_class):
            """Map canonical disease label → class index in the adapter's
            labels list. Handles loose matches (case-insensitive,
            substring) so the agent can pass user-friendly names."""
            labels = list(adapter.metadata.labels or [])
            if not labels:
                return None, "adapter has empty labels list"
            if target_class is None:
                return None, None  # let GradCAM use predicted top-1
            tc = target_class.strip()
            # exact
            for i, lab in enumerate(labels):
                if lab == tc:
                    return i, None
            # case-insensitive
            for i, lab in enumerate(labels):
                if lab.lower() == tc.lower():
                    return i, None
            # substring (helps "diabetic retinopathy" → "DR")
            for i, lab in enumerate(labels):
                if tc.lower() in lab.lower() or lab.lower() in tc.lower():
                    return i, None
            return None, (f"target_class={target_class!r} not in adapter "
                          f"labels {labels}. Use one of them exactly.")

        def _adapter_to_tensor(adapter, image_path):
            """Get a (1,C,H,W) tensor using whatever preprocessing the
            adapter uses internally. Falls back through known patterns."""
            from PIL import Image
            # Pattern 1: adapter._transform (torchvision Compose)
            if hasattr(adapter, "_transform") and adapter._transform is not None:
                img = Image.open(image_path).convert("RGB")
                return adapter._transform(img).unsqueeze(0).to(adapter.device)
            # Pattern 2: module-level _preprocess in adapters/paired
            adapter_mod = type(adapter).__module__
            if "paired.classifiers" in adapter_mod:
                from ..adapters.paired.classifiers import _preprocess as _pp
                return _pp(image_path, adapter.device, tta=False)
            raise RuntimeError(
                f"don't know how to build input tensor for adapter "
                f"{type(adapter).__name__}; expected ._transform or "
                f"a registered _preprocess pattern"
            )

        def _has_conv2d(model):
            import torch.nn as nn
            return any(isinstance(m, nn.Conv2d) for m in model.modules())

        def _gradcam(
            target_class: str | None = None,
            classifier: str | None = None,
            image_path: str | None = None,
        ) -> dict:
            from pathlib import Path
            import cv2
            import numpy as np
            # Resolve image
            img_path = image_path or (
                self.session.context.current_image if self.session else None
            )
            if self.session and img_path:
                try:
                    img_path = str(self.session.resolve_session_file(img_path))
                except Exception as e:
                    return {"error": f"file access denied: {e}"}
            if not img_path or not Path(img_path).exists():
                return {"error": "no image: pass image_path or set a "
                                  "current image first via set_current_image"}

            # Resolve modality (use session, fallback to filename hint)
            modality = (self.session.context.current_modality
                        if self.session else None) or filename_modality_hint(img_path)
            if modality is None:
                modality = "CFP"  # conservative default

            # Pick classifier
            picked = classifier or _DEFAULT_PER_MODALITY.get(modality)
            if not picked:
                return {"error": f"no default Grad-CAM classifier for "
                                  f"modality {modality}"}
            if picked in _VIT_CLASSIFIERS:
                return {
                    "error": f"{picked!r} is a ViT-based classifier; "
                             f"Grad-CAM v1 only supports CNN-based "
                             f"classifiers. Default CNN for {modality} is "
                             f"{_DEFAULT_PER_MODALITY.get(modality)!r}.",
                    "hint": "call gradcam() without the `classifier` arg "
                            "to use the default.",
                }

            # Load adapter
            try:
                adapter = GLOBAL_REGISTRY.get(picked)
            except (KeyError, RuntimeError) as e:
                return {"error": f"classifier {picked!r} unavailable: {e}"}
            try:
                adapter.load()
            except Exception as e:
                return {"error": f"failed to load {picked}: {type(e).__name__}: {e}"}

            model = getattr(adapter, "_impl", None)
            if model is None:
                return {"error": f"adapter {picked} has no ._impl model"}
            if not _has_conv2d(model):
                return {
                    "error": f"adapter {picked} has no Conv2d layers — "
                             f"likely a transformer. v1 Grad-CAM is "
                             f"Conv2d-only; ViT attention rollout TODO.",
                }

            cls_idx, err = _resolve_class_idx(adapter, target_class)
            if err:
                return {"error": err,
                        "available_labels": list(adapter.metadata.labels or [])}

            # Build input tensor
            try:
                tensor = _adapter_to_tensor(adapter, img_path)
            except Exception as e:
                return {"error": f"preprocessing failed: {type(e).__name__}: {e}"}
            # paired adapters may stack a TTA flip (B=2); take first
            if tensor.dim() == 4 and tensor.shape[0] > 1:
                tensor = tensor[:1]

            # Run Grad-CAM (lazy import — keeps module-import cheap)
            from ..visualization.visualizer import (
                GradCAM, heatmap_overlay, boxes_from_heatmap,
            )
            try:
                cam = GradCAM(model)
                try:
                    heat, predicted_idx, confidence = cam.compute(tensor, class_idx=cls_idx)
                finally:
                    cam.remove_hooks()
            except Exception as e:
                return {"error": f"GradCAM compute failed: "
                                  f"{type(e).__name__}: {e}"}

            # Load original image for overlay (BGR for cv2)
            original = cv2.imdecode(
                np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if original is None:
                return {"error": f"cv2 could not load image: {img_path}"}

            # Resize heatmap to original resolution
            H, W = original.shape[:2]
            heat_full = cv2.resize(heat, (W, H))
            overlay = heatmap_overlay(original, heat_full, alpha=0.45)
            boxes = boxes_from_heatmap(heat_full, threshold=0.5, min_area=200)
            overlay_boxed = overlay.copy()
            for (x, y, w_, h_) in boxes:
                cv2.rectangle(overlay_boxed, (x, y), (x + w_, y + h_),
                              (0, 255, 0), 3)

            # Save into the session workspace. We hash the original stem to
            # an ASCII-only prefix because OpenCV's imwrite + Windows' GBK
            # default codec mangle non-ASCII paths (e.g. Chinese filenames
            # like '微信截图...' get written as '寰俊鎴浘...' on disk while
            # the agent's URL encodes proper UTF-8 — the mismatch breaks the
            # inline image render in the chat UI).
            import hashlib
            labels = list(adapter.metadata.labels or [])
            predicted_label = (labels[predicted_idx] if 0 <= predicted_idx < len(labels)
                               else f"idx_{predicted_idx}")
            target_label = (labels[cls_idx] if cls_idx is not None and 0 <= cls_idx < len(labels)
                            else predicted_label)
            out_dir = (Path(self.session.workspace) / self.session.session_id
                       / "gradcam") if self.session else Path("./gradcam")
            out_dir.mkdir(parents=True, exist_ok=True)
            orig_stem = Path(img_path).stem
            stem_hash = hashlib.md5(orig_stem.encode("utf-8")).hexdigest()[:10]
            safe_stem = f"img_{stem_hash}"
            # Sanitise label too (no spaces / non-ASCII in URL component)
            safe_target = "".join(
                c if (c.isascii() and (c.isalnum() or c in "-_")) else "_"
                for c in target_label
            )
            overlay_path = out_dir / f"{safe_stem}_{picked}_{safe_target}_heatmap.png"
            boxed_path = out_dir / f"{safe_stem}_{picked}_{safe_target}_heatmap_bbox.png"
            # Use OpenCV via numpy buffer to defeat Windows code-page paths
            cv2.imencode(".png", overlay)[1].tofile(str(overlay_path))
            cv2.imencode(".png", overlay_boxed)[1].tofile(str(boxed_path))

            figures = {
                "heatmap_overlay": str(overlay_path),
                "heatmap_with_boxes": str(boxed_path),
            }
            out = {
                "success": True,
                "tool": "gradcam",
                "predictions": {
                    "classifier": picked,
                    "modality": modality,
                    "target_class": target_label,
                    "predicted_class": predicted_label,
                    "confidence": float(confidence),
                    "boxes": [list(b) for b in boxes],
                    "n_hotspots": len(boxes),
                },
                "figures": figures,
                "figure_urls": _figures_to_urls(
                    figures,
                    self.session,
                    namespace="gradcam",
                ),
            }
            return out

        self.tools["gradcam"] = Tool(
            name="gradcam",
            description=(
                "Generate a Grad-CAM heatmap that highlights which image "
                "regions drove a classifier's prediction. Use when the "
                "user asks 'where is the X' / 'show heatmap' / "
                "'explain the diagnosis' / 'Grad-CAM'.\n\n"
                "Auto-routes by current modality:\n"
                "  CFP  → cfp_paired5         (5-class: Normal/DR/RVO/AMD/CSC)\n"
                "  OCT  → oct_fmue_16class    (16 OCT B-scan classes)\n"
                "  UWF  → uwf_disease_7class  (7-class UWF disease)\n"
                "  FFA  → ffa_paired5         (5-class FFA disease)\n\n"
                "All args optional:\n"
                "  target_class  — canonical disease label (e.g. 'AMD', 'DR'). "
                "If omitted, uses the classifier's top-1 prediction.\n"
                "  classifier    — override the default. Must be CNN-based "
                "(ResNet / ConvNeXt). ViT classifiers (cfp_clip_*, "
                "cfp_glaucoma, cfp_pdr_cascade) return an error.\n"
                "  image_path    — defaults to the session's current image.\n\n"
                "Returns `figure_urls` with `heatmap_overlay` (jet overlay) "
                "and `heatmap_with_boxes` (overlay with green bboxes around "
                "the hot regions ≥ threshold). Embed those URLs inline as "
                "`![label](url)` so the user sees the heatmap."
            ),
            parameters=[
                ToolParameter("target_class", "string",
                              "Canonical disease label (e.g. 'AMD', 'DR'). "
                              "Optional — defaults to the classifier's top-1.",
                              required=False),
                ToolParameter("classifier", "string",
                              "Override the default classifier. Must be CNN-"
                              "based. Optional.",
                              required=False),
                ToolParameter("image_path", "string",
                              "Override the session's current image. Optional.",
                              required=False),
            ],
            function=_gradcam,
        )

    def _add_vision_impression_tool(self):
        """Direct vision-LLM read of the image. Covers the long tail of
        clinical findings our specialised classifiers don't have heads for
        (hypertensive retinopathy, papilledema, drusen patterns, intra-
        operative landmarks, post-laser status, etc.)."""
        from ..agent.tools.oct_tools import Tool, ToolParameter
        import base64

        def _vision_impression(image_path: str = "", focus: str = "") -> dict:
            """Two-stage vision impression with literature-grounded rubric
            and self-consistency / cross-tool validation.

            Stage 1 — structured morphology (strict JSON, schema-validated).
            Stage 2 — differential diagnosis citing only stage-1 observations,
                       grounded in a clinical-evidence rubric.

            Falls back to the legacy single-pass markdown impression for
            modalities other than CFP (until per-modality rubrics ship).
            """
            if self.session is None:
                return {"error": "no active session"}
            img = image_path or (self.session.context.current_image or "")
            if not img:
                return {"error": "no image_path and no current image"}
            try:
                p = self.session.resolve_session_file(img)
            except Exception as e:
                return {"error": f"file access denied: {e}"}

            vision_model, reason = self.session._resolve_vision()
            if vision_model is None:
                return {
                    "tool": "vision_impression",
                    "skipped": True,
                    "reason": reason,
                    "note": (
                        "No vision-capable model available. Do NOT fabricate "
                        "a visual read; rely on the specialist classifiers "
                        "(cfp_pdr_cascade, cfp_efiqa, cfp_retsam_segmentation, "
                        "etc.) which read the pixels independently of the LLM."
                    ),
                }
            client = self.session._vision_client()

            # Capability gating + runtime probe for unknown models
            from .vision_prompts import (
                cfp as cfp_prompts,
                oct as oct_prompts,
                uwf as uwf_prompts,
                run_validators, cross_check, parse_json_lenient,
            )
            import os as _os_vi
            from .vision_prompts.capability_probe import ensure_capability

            cap, probe_note = ensure_capability(self.session, vision_model)
            if cap == "none":
                return {
                    "tool": "vision_impression",
                    "skipped": True,
                    "reason": f"model {vision_model} failed vision capability "
                              f"probe ({probe_note})",
                    "note": "Treating as text-only — using classifier-only path.",
                }

            suffix = p.suffix.lower().lstrip(".") or "png"
            mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            except Exception as e:
                return {"error": f"read failed: {e}"}
            data_url = f"data:{mime};base64,{b64}"

            modality = (self.session.context.current_modality or "").upper()
            # Only CFP has a full two-stage rubric right now. Detect the
            # modality from filename or session context; if it doesn't look
            # like CFP, fall back to a single-stage markdown impression.
            # Prefer the KNOWN session modality. Only sniff the filename when
            # the modality is unknown — otherwise the project directory name
            # ("…/oct/…") makes every path match the "oct" tag and forces the
            # legacy path even for CFP-detected images.
            is_cfp = (modality == "CFP" or
                      (not modality
                       and not any(tag in str(img).lower()
                                   for tag in ("oct", "bscan", "ffa", "uwf", "octcube"))))

            # ─── Two-stage CFP path ──────────────────────────────────────
            if is_cfp:
                # Stage 1: morphology
                s1_system = cfp_prompts.STAGE1_SYSTEM
                s1_user_text = cfp_prompts.stage1_user_prompt(focus)
                try:
                    s1_resp = client.chat.completions.create(
                        model=vision_model,
                        max_tokens=6000,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": s1_system},
                            {"role": "user", "content": [
                                {"type": "text", "text": s1_user_text},
                                {"type": "image_url",
                                 "image_url": {"url": data_url}},
                            ]},
                        ],
                    )
                    s1_raw = (s1_resp.choices[0].message.content or "").strip()
                except Exception as e:
                    return {"error": f"stage1 LLM call failed: "
                                     f"{type(e).__name__}: {e}"}

                s1, parse_err = parse_json_lenient(s1_raw)
                if s1 is None or not isinstance(s1, dict):
                    return {
                        "tool": "vision_impression",
                        "stage1_parse_error": parse_err,
                        "stage1_raw": s1_raw[:2000],
                        "reliability": "failed",
                    }

                field_issues = run_validators(s1, cfp_prompts.VALIDATORS)
                analyses = self.session._analyses_for_image(str(p))
                cross_issues = cross_check(
                    s1, analyses, cfp_prompts.CROSS_TOOL_FIELDS)

                # Reliability gate
                model_self_conf = (s1.get("model_self_assessment", {})
                                     .get("confidence_overall", "moderate"))
                n_issues = len(field_issues) + len(cross_issues)
                if n_issues == 0 and cap == "good" and model_self_conf == "high":
                    reliability = "high"
                elif n_issues <= 1 and cap in ("good", "unknown") and \
                        model_self_conf in ("high", "moderate"):
                    reliability = "moderate"
                else:
                    reliability = "degraded"

                # Stage 2: differential diagnosis grounded in stage 1.
                # Note: the rubric is ~10k chars (~2.5k tokens). Combined
                # with the reasoning-token cost on gpt-5.x-pro, we need a
                # generous max_tokens so the model has budget left to
                # actually emit output (was hitting finish_reason='length'
                # with content='' at 2200). 8000 is the working default.
                # response_format=json_object is intentionally omitted —
                # it occasionally interacts badly with the gateway routing
                # for reasoning models; the lenient parser handles fences.
                s2_system = cfp_prompts.stage2_system_prompt()
                s2_user_text = cfp_prompts.stage2_user_prompt(s1)
                try:
                    s2_resp = client.chat.completions.create(
                        model=vision_model,
                        max_tokens=8000,
                        messages=[
                            {"role": "system", "content": s2_system},
                            # Stage 2 does NOT need the image — it
                            # works from stage1's JSON. This forces grounding.
                            {"role": "user", "content": s2_user_text},
                        ],
                    )
                    s2_raw = (s2_resp.choices[0].message.content or "").strip()
                except Exception as e:
                    return {
                        "tool": "vision_impression",
                        "stage1_morphology": s1,
                        "stage1_validation": {
                            "field_issues": field_issues,
                            "cross_tool_disagreements": cross_issues,
                        },
                        "reliability": reliability,
                        "stage2_error": f"{type(e).__name__}: {e}",
                    }

                s2, parse_err2 = parse_json_lenient(s2_raw)
                if s2 is None:
                    s2 = {"parse_error": parse_err2, "raw": s2_raw[:1500]}

                # Build a backward-compatible markdown summary. This is purely
                # cosmetic — the structured JSON above is the source of truth —
                # so never let a rendering glitch fail the whole observer.
                try:
                    md_summary = _cfp_summary_md(s1, s2, reliability,
                                                 field_issues, cross_issues)
                except Exception as e:
                    md_summary = (f"_(summary rendering failed: "
                                  f"{type(e).__name__}: {e}; see structured "
                                  f"stage1/stage2 fields)_")

                return {
                    "tool": "vision_impression",
                    "modality_input": "CFP",
                    "model": vision_model,
                    "reliability": reliability,
                    "stage1_morphology": s1,
                    "stage2_differential": s2,
                    "stage1_validation": {
                        "field_issues": field_issues,
                        "cross_tool_disagreements": cross_issues,
                    },
                    "capability_probe": probe_note,
                    # Backward-compat for callers that expect markdown
                    "impression_markdown": md_summary,
                }

            # ─── Two-stage OCT path (OPH_OCT_VISION_V2, default ON) ──────
            is_oct = (modality == "OCT" or
                      (not modality and any(t in str(img).lower()
                                            for t in ("oct", "bscan", "octcube"))))
            if is_oct and _os_vi.environ.get("OPH_OCT_VISION_V2", "1") == "1":
                s1_resp = None
                try:
                    s1_resp = client.chat.completions.create(
                        model=vision_model, max_tokens=6000,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": oct_prompts.STAGE1_SYSTEM},
                            {"role": "user", "content": [
                                {"type": "text", "text": oct_prompts.stage1_user_prompt(focus)},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ]},
                        ],
                    )
                    s1_raw = (s1_resp.choices[0].message.content or "").strip()
                except Exception as e:
                    return {"error": f"OCT stage1 LLM call failed: {type(e).__name__}: {e}"}
                s1, parse_err = parse_json_lenient(s1_raw)
                if s1 is None or not isinstance(s1, dict):
                    return {"tool": "vision_impression", "modality_input": "OCT",
                            "stage1_parse_error": parse_err, "stage1_raw": s1_raw[:2000],
                            "reliability": "failed"}
                field_issues = run_validators(s1, oct_prompts.VALIDATORS)
                analyses = self.session._analyses_for_image(str(p))
                cross_issues = cross_check(s1, analyses, oct_prompts.CROSS_TOOL_FIELDS)
                msc = (s1.get("model_self_assessment", {}) or {}).get("confidence_overall", "moderate")
                n_issues = len(field_issues) + len(cross_issues)
                if n_issues == 0 and cap == "good" and msc == "high":
                    reliability = "high"
                elif n_issues <= 1 and cap in ("good", "unknown") and msc in ("high", "moderate"):
                    reliability = "moderate"
                else:
                    reliability = "degraded"
                # Tool context for grounding: FMUE top class + fluid-seg verdict
                tc = []
                for tname in ("oct_fmue_16class", "oct_volume_octcubem",
                              "oct_fluid_segmentation", "oct_layer_segmentation"):
                    a = analyses.get(tname)
                    if a:
                        preds = a.get("predictions", a) if isinstance(a, dict) else a
                        tc.append(f"{tname}: {json.dumps(preds, ensure_ascii=False)[:600]}")
                tool_context = "\n".join(tc)
                try:
                    s2_resp = client.chat.completions.create(
                        model=vision_model, max_tokens=8000,
                        messages=[
                            {"role": "system", "content": oct_prompts.stage2_system_prompt()},
                            {"role": "user", "content": oct_prompts.stage2_user_prompt(s1, tool_context)},
                        ],
                    )
                    s2_raw = (s2_resp.choices[0].message.content or "").strip()
                    s2, _ = parse_json_lenient(s2_raw)
                    if s2 is None:
                        s2 = {"raw": s2_raw[:1500]}
                except Exception as e:
                    s2 = {"stage2_error": f"{type(e).__name__}: {e}"}
                return {
                    "tool": "vision_impression", "modality_input": "OCT",
                    "model": vision_model, "reliability": reliability,
                    "stage1_morphology": s1, "stage2_differential": s2,
                    "stage1_validation": {"field_issues": field_issues,
                                          "cross_tool_disagreements": cross_issues},
                    "capability_probe": probe_note,
                }

            # ─── Two-stage UWF path (OPH_UWF_VISION_V2, default ON) ──────
            is_uwf = (modality == "UWF" or
                      (not modality and any(t in str(img).lower()
                                            for t in ("uwf", "optos", "widefield",
                                                      "ultrawide"))))
            if is_uwf and _os_vi.environ.get("OPH_UWF_VISION_V2", "1") == "1":
                try:
                    s1_resp = client.chat.completions.create(
                        model=vision_model, max_tokens=6000,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": uwf_prompts.STAGE1_SYSTEM},
                            {"role": "user", "content": [
                                {"type": "text", "text": uwf_prompts.stage1_user_prompt(focus)},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ]},
                        ],
                    )
                    s1_raw = (s1_resp.choices[0].message.content or "").strip()
                except Exception as e:
                    return {"error": f"UWF stage1 LLM call failed: {type(e).__name__}: {e}"}
                s1, parse_err = parse_json_lenient(s1_raw)
                if s1 is None or not isinstance(s1, dict):
                    return {"tool": "vision_impression", "modality_input": "UWF",
                            "stage1_parse_error": parse_err, "stage1_raw": s1_raw[:2000],
                            "reliability": "failed"}
                field_issues = run_validators(s1, uwf_prompts.VALIDATORS)
                analyses = self.session._analyses_for_image(str(p))
                cross_issues = cross_check(s1, analyses, uwf_prompts.CROSS_TOOL_FIELDS)
                msc = (s1.get("model_self_assessment", {}) or {}).get("confidence_overall", "moderate")
                n_issues = len(field_issues) + len(cross_issues)
                if n_issues == 0 and cap == "good" and msc == "high":
                    reliability = "high"
                elif n_issues <= 1 and cap in ("good", "unknown") and msc in ("high", "moderate"):
                    reliability = "moderate"
                else:
                    reliability = "degraded"
                # Tool context for grounding: both UWF classifiers' outputs
                tc = []
                for tname in ("uwf_multi_disease", "uwf_disease_7class"):
                    a = analyses.get(tname)
                    if a:
                        preds = a.get("predictions", a) if isinstance(a, dict) else a
                        tc.append(f"{tname}: {json.dumps(preds, ensure_ascii=False)[:600]}")
                tool_context = "\n".join(tc)
                try:
                    s2_resp = client.chat.completions.create(
                        model=vision_model, max_tokens=8000,
                        messages=[
                            {"role": "system", "content": uwf_prompts.stage2_system_prompt()},
                            {"role": "user", "content": uwf_prompts.stage2_user_prompt(s1, tool_context)},
                        ],
                    )
                    s2_raw = (s2_resp.choices[0].message.content or "").strip()
                    s2, _ = parse_json_lenient(s2_raw)
                    if s2 is None:
                        s2 = {"raw": s2_raw[:1500]}
                except Exception as e:
                    s2 = {"stage2_error": f"{type(e).__name__}: {e}"}
                return {
                    "tool": "vision_impression", "modality_input": "UWF",
                    "model": vision_model, "reliability": reliability,
                    "stage1_morphology": s1, "stage2_differential": s2,
                    "stage1_validation": {"field_issues": field_issues,
                                          "cross_tool_disagreements": cross_issues},
                    "capability_probe": probe_note,
                }

            # ─── Fallback: legacy single-stage markdown (non-CFP) ────────
            focus_line = (f"User focus: {focus}\n" if focus else "")
            system = (
                "You are a retinal subspecialist giving a structured overall "
                "visual impression of a single ophthalmic image. You are NOT "
                "writing a final diagnosis — your job is to surface what the "
                "image LOOKS like so a downstream classifier ensemble can "
                "consider it.\n\n"
                "Always return Markdown with EXACTLY these short sections "
                "(omit a section if not applicable):\n"
                "  **Modality**: CFP | OCT | UWF | FFA | other\n"
                "  **Quality**: usable / borderline / poor + 1-line reason\n"
                "  **Optic disc / Layers**: as relevant to modality\n"
                "  **Vessels / Vasculature**: as relevant\n"
                "  **Macula / Central**: as relevant\n"
                "  **Periphery**: as relevant\n"
                "  **Conspicuous findings**: 1-3 bullets — only if striking\n"
                "  **Top differential**: 1-3 entries with rough confidence"
            )
            try:
                resp = client.chat.completions.create(
                    model=vision_model,
                    # Reasoning models (gpt-5.x) spend hidden tokens before any
                    # visible content; 900 could be fully consumed → empty
                    # output. Give generous headroom so the markdown survives.
                    max_tokens=3000,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": [
                            {"type": "text",
                             "text": focus_line + "Provide the structured "
                                     "visual impression now."},
                            {"type": "image_url",
                             "image_url": {"url": data_url}},
                        ]},
                    ],
                )
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                finish = getattr(resp.choices[0], "finish_reason", None)
            except Exception as e:
                return {"error": f"vision LLM call failed: "
                                 f"{type(e).__name__}: {e}"}

            if not content:
                # Empty visible output (often a reasoning model hitting the
                # token cap before emitting text). Report it honestly rather
                # than silently returning "(no impression)".
                return {
                    "tool": "vision_impression",
                    "modality_input": modality or "unknown",
                    "skipped": True,
                    "reason": (f"vision model {vision_model} returned no visible "
                               f"content (finish_reason={finish}). Likely the "
                               "token budget was consumed by reasoning."),
                    "note": ("No gestalt read available — rely on the "
                             "calibrated classifiers for this image."),
                    "model": vision_model,
                }

            return {
                "tool": "vision_impression",
                "modality_input": modality or "unknown",
                "impression_markdown": content,
                "model": vision_model,
                "reliability": "unrated (legacy path; CFP-only schema)",
            }

        self.tools["vision_impression"] = Tool(
            name="vision_impression",
            description=(
                "Ask a vision model to LOOK at the image and return a "
                "structured visual impression (modality, quality, disc, "
                "vessels, macula, periphery, conspicuous findings, top "
                "differential). LOW effort must skip this. MEDIUM effort uses "
                "it only as a single bounded escalation after objective "
                "quality/classifier/measurement tools are available; do not "
                "call it as the default first observer or repeat it. HIGH and "
                "MAX retain objective-first ordering but may use it for a "
                "specific unresolved differential; ULTRA includes it in its "
                "exhaustive compatible-tool pass. It "
                "catches conditions our trained models don't have heads for "
                "(hypertensive retinopathy, papilledema, drusen patterns, "
                "post-laser changes, atypical pigmentation). NOTE: if no "
                "vision-capable "
                "model is available (text-only chat backbone, no "
                "OPH_WEB_VISION_MODEL configured) this returns "
                "{skipped: true} — in that case do NOT fabricate a visual "
                "read; rely on the classifiers + retsam metrics."
            ),
            parameters=[
                ToolParameter("image_path", "string",
                              "Image to look at. Defaults to current session image.",
                              required=False, default=""),
                ToolParameter("focus", "string",
                              "Optional hint about what to pay attention to "
                              "(e.g. 'look for hypertensive retinopathy signs')",
                              required=False, default=""),
            ],
            function=_vision_impression,
        )
