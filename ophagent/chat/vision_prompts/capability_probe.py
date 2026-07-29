"""
Model vision-capability gating.

Three buckets:
  • "good"     — known vision-capable, ship as-is
  • "none"     — known text-only, short-circuit the vision_impression tool
  • "unknown"  — fall back to a one-shot probe with a known-answer image

The probe sends a small CFP with an obvious finding (PRP laser scars) and
checks the response contains the expected keyword. If it doesn't, the model
is downgraded to "none" for this session.

Caching: probe results are cached on the session object (`_vision_probe_cache`)
so we don't pay the probe cost more than once per session.
"""
from __future__ import annotations

import base64
from pathlib import Path


# ── Static knowledge of which models can see — kept short on purpose ──────
# Lower-case keys; we match by substring so versioned variants pass.
VISION_CAPABLE = {
    # OpenAI / GPT family
    "gpt-5", "gpt-4o", "gpt-4.1", "gpt-4-turbo",
    # Anthropic
    "claude-opus", "claude-sonnet", "claude-haiku",
    # Google Gemini
    "gemini", "gemini-pro", "gemini-flash",
    # Qwen-VL
    "qwen-vl", "qwen3-vl", "qwen2-vl", "qwen2.5-vl",
    # xAI Grok
    "grok-4-vision", "grok-vision",
    # Open-source vision-capable
    "internvl", "minicpm-v", "llava",
}

TEXT_ONLY = {
    "deepseek-chat", "deepseek-v3", "deepseek-r1",
    "qwen3-coder", "qwen-text",
    "mistral-medium", "mistral-large",
    # Plain (non-vision) LLMs that share names with vision variants — note
    # we check VISION_CAPABLE first, so these don't collide if the user
    # picks the vision-named SKU.
}


def vision_capability(model_id: str) -> str:
    """Return one of: 'good' | 'none' | 'unknown'.

    Matched by lowercase substring against VISION_CAPABLE / TEXT_ONLY.
    'unknown' callers should fall back to runtime probing.
    """
    if not model_id:
        return "unknown"
    m = model_id.lower()
    for hit in VISION_CAPABLE:
        if hit in m:
            return "good"
    for miss in TEXT_ONLY:
        if miss in m:
            return "none"
    return "unknown"


# ── Runtime probe — used when vision_capability() returns 'unknown' ──────
PROBE_IMAGE = Path(__file__).resolve().parent / "_probe_image.jpg"
PROBE_EXPECTED_KEYWORDS = {
    "fundus", "retina", "optic disc", "vessel", "hemorrhage",
    "视网膜", "视盘", "眼底",
}
PROBE_PROMPT = (
    "What ophthalmic imaging modality is this and what is the single most "
    "obvious abnormality? Reply in 30 words or less."
)


def probe_vision(client, model_id: str) -> tuple[bool, str]:
    """Send a known image to the model and check the response contains
    fundus-related vocabulary. Returns (passed, raw_response)."""
    if not PROBE_IMAGE.exists():
        return True, "(probe image missing — assuming capable)"
    b64 = base64.b64encode(PROBE_IMAGE.read_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").lower()
    except Exception as e:
        return False, f"probe call failed: {type(e).__name__}: {e}"
    passed = any(k in text for k in PROBE_EXPECTED_KEYWORDS)
    return passed, text


def ensure_capability(session, model_id: str) -> tuple[str, str | None]:
    """High-level entry: get capability, falling back to a runtime probe
    cached on the session. Returns (capability, probe_note_or_None)."""
    cap = vision_capability(model_id)
    if cap != "unknown":
        return cap, None
    cache = getattr(session, "_vision_probe_cache", None) or {}
    key = model_id.lower()
    if key in cache:
        return cache[key]["cap"], cache[key].get("note")
    # Probe
    try:
        session._ensure_client()
        client = session._client
        passed, note = probe_vision(client, model_id)
        cap = "good" if passed else "none"
        if not hasattr(session, "_vision_probe_cache"):
            session._vision_probe_cache = {}
        session._vision_probe_cache[key] = {"cap": cap, "note": note}
        return cap, note
    except Exception as e:
        return "none", f"probe exception: {type(e).__name__}: {e}"
