"""
Curated catalog of LLM models per provider, surfaced in the UI's model picker.

Per-model entry:
  - id          : the API model id (passed to OpenAI-compatible chat.completions)
  - name        : human-friendly display label
  - family      : "openai" | "anthropic" | "google" | "xai" | "deepseek" | "qwen"
  - reasoning   : True if it's a CoT/reasoning model (burns more tokens)
  - vision      : True if it supports image input
  - cost        : "high" | "medium" | "low" — relative cost hint for the UI
  - paper_backbone: canonical manuscript backbone name, when applicable
  - note        : short caveat shown as tooltip
"""

from __future__ import annotations


CATALOG: dict[str, list[dict]] = {

    # ── aigcbest (api2.aigcbest.top) — plain model ids ─────────────────────
    "aigcbest": [
        # GPT-5 family — verified live (probe_gpt5.py). Reasoning models;
        # need max_tokens >= ~2000 to surface output past the CoT burn.
        {"id": "gpt-5",             "name": "GPT-5",                "family": "openai",   "reasoning": True,  "vision": True, "cost": "high",
         "paper_backbone": "GPT-5",
         "note": "High-capability reasoning model. Allow enough output tokens for reasoning and tool calls."},
        {"id": "gemini-3-pro-preview", "name": "Gemini 3.0 Pro",    "family": "google",   "reasoning": True,  "vision": True, "cost": "medium",
         "paper_backbone": "Gemini-3.0-Pro",
         "note": "Multimodal reasoning model with vision and tool-calling support."},
        {"id": "qwen3-vl-235b-a22b-instruct", "name": "Qwen3-VL 235B", "family": "qwen", "reasoning": False, "vision": True, "cost": "medium",
         "paper_backbone": "Qwen-3-VL",
         "note": "Vision-capable 235B instruct model with tool-calling support."},
        {"id": "DeepSeek-V3",       "name": "DeepSeek V3",          "family": "deepseek", "reasoning": False, "vision": False, "cost": "low",
         "paper_backbone": "DeepSeek-V3",
         "note": "Text-only planner; pair with a vision model for open-ended image reading."},
        {"id": "gpt-5-mini",        "name": "GPT-5 Mini",           "family": "openai",   "reasoning": True,  "vision": True, "cost": "medium",
         "note": "Cheaper gpt-5 variant. Still reasoning; good cost/quality balance."},
        {"id": "gpt-5-pro",         "name": "GPT-5 Pro",            "family": "openai",   "reasoning": True,  "vision": True, "cost": "high",
         "note": "Highest-tier gpt-5. Most expensive; may require larger wallet pre-deduction."},
        # GPT-5.5 family — newer aigcbest aliases
        {"id": "gpt-5.5-pro",       "name": "GPT-5.5 Pro",          "family": "openai",   "reasoning": True,  "vision": True, "cost": "high",
         "note": "Strongest reasoning, slow & expensive. Best for hard cases."},
        {"id": "gpt-5.5",           "name": "GPT-5.5",              "family": "openai",   "reasoning": False, "vision": True, "cost": "medium",
         "note": "Flagship non-reasoning. Fast, solid quality."},
        {"id": "gpt-5.4",           "name": "GPT-5.4",              "family": "openai",   "reasoning": False, "vision": True, "cost": "medium",
         "note": "Previous flagship."},
        {"id": "gpt-5.4-mini",      "name": "GPT-5.4 Mini",         "family": "openai",   "reasoning": False, "vision": True, "cost": "low",
         "note": "Cheap & fast vision model. Good for captioning / quick checks."},
        {"id": "claude-opus-4-7",   "name": "Claude Opus 4.7",      "family": "anthropic","reasoning": False, "vision": True, "cost": "high",
         "note": "Anthropic top-tier. Strong instruction following & tool use."},
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6",    "family": "anthropic","reasoning": False, "vision": True, "cost": "medium",
         "note": "Balanced Claude. Reliable for clinical reports."},
        {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash",  "family": "google",   "reasoning": False, "vision": True, "cost": "low",
         "note": "Cheap & very fast."},
        {"id": "grok-4",            "name": "Grok 4",               "family": "xai",      "reasoning": False, "vision": True, "cost": "medium",
         "note": "xAI flagship."},
        {"id": "DeepSeek-V3.2",     "name": "DeepSeek V3.2",        "family": "deepseek", "reasoning": False, "vision": False,"cost": "low",
         "note": "Open-source-style; text only, no images. Cheap."},
    ],

    # ── OpenRouter — vendor-prefixed ids ───────────────────────────────────
    "openrouter": [
        {"id": "openai/gpt-5",                    "name": "GPT-5",          "family": "openai",   "reasoning": True,  "vision": True, "cost": "high",
         "paper_backbone": "GPT-5",
         "note": "High-capability reasoning and vision model through OpenRouter."},
        {"id": "qwen/qwen3-vl-235b-a22b-instruct", "name": "Qwen3-VL 235B", "family": "qwen",     "reasoning": False, "vision": True, "cost": "medium",
         "paper_backbone": "Qwen-3-VL",
         "note": "Vision-capable 235B instruct model through OpenRouter."},
        {"id": "openai/gpt-5.5-pro",            "name": "GPT-5.5 Pro",      "family": "openai",   "reasoning": True,  "vision": True, "cost": "high",
         "note": "Strongest reasoning, slow & expensive."},
        {"id": "openai/gpt-5.5",                "name": "GPT-5.5",          "family": "openai",   "reasoning": False, "vision": True, "cost": "medium",
         "note": "Flagship non-reasoning."},
        {"id": "openai/gpt-5.4-mini",           "name": "GPT-5.4 Mini",     "family": "openai",   "reasoning": False, "vision": True, "cost": "low",
         "note": "Cheap & fast vision."},
        {"id": "anthropic/claude-opus-4.7",     "name": "Claude Opus 4.7",  "family": "anthropic","reasoning": False, "vision": True, "cost": "high",
         "note": "Anthropic top-tier."},
        {"id": "anthropic/claude-sonnet-4.6",   "name": "Claude Sonnet 4.6","family": "anthropic","reasoning": False, "vision": True, "cost": "medium",
         "note": "Balanced Claude."},
        {"id": "google/gemini-3-pro-preview",   "name": "Gemini 3 Pro",     "family": "google",   "reasoning": False, "vision": True, "cost": "medium",
         "note": "Google flagship."},
        {"id": "google/gemini-3-flash-preview", "name": "Gemini 3 Flash",   "family": "google",   "reasoning": False, "vision": True, "cost": "low",
         "note": "Cheap & fast."},
        {"id": "x-ai/grok-4",                   "name": "Grok 4",           "family": "xai",      "reasoning": False, "vision": True, "cost": "medium",
         "note": "xAI flagship."},
        {"id": "deepseek/deepseek-v3.2",        "name": "DeepSeek V3.2",    "family": "deepseek", "reasoning": False, "vision": False,"cost": "low",
         "note": "Open-source. Text only."},
    ],

    # ── First-party OpenAI API ──────────────────────────────────────────────
    "openai": [
        {"id": "gpt-5.6",       "name": "GPT-5.6 Sol",   "family": "openai", "reasoning": True, "vision": True, "tools": True, "cost": "high",
         "note": "OpenAI flagship for complex professional work; the gpt-5.6 alias resolves to Sol."},
        {"id": "gpt-5.6-terra", "name": "GPT-5.6 Terra", "family": "openai", "reasoning": True, "vision": True, "tools": True, "cost": "medium",
         "note": "Balanced intelligence and cost for routine OphAgent use."},
        {"id": "gpt-5.6-luna",  "name": "GPT-5.6 Luna",  "family": "openai", "reasoning": True, "vision": True, "tools": True, "cost": "low",
         "note": "Cost-sensitive option for higher-volume workloads."},
        {"id": "gpt-5",         "name": "GPT-5",          "family": "openai", "reasoning": True, "vision": True, "tools": True, "cost": "high",
         "paper_backbone": "GPT-5",
         "note": "High-capability reasoning and vision model through the official OpenAI API."},
    ],

    # ── First-party Anthropic Claude API (official compatibility endpoint) ──
    "anthropic": [
        {"id": "claude-sonnet-5", "name": "Claude Sonnet 5", "family": "anthropic", "reasoning": True, "vision": True, "tools": True, "cost": "medium",
         "note": "Balanced Claude model for speed, intelligence, vision, and tool use."},
        {"id": "claude-opus-5",   "name": "Claude Opus 5",   "family": "anthropic", "reasoning": True, "vision": True, "tools": True, "cost": "high",
         "note": "High-capability model for complex agentic and enterprise workflows."},
        {"id": "claude-fable-5",  "name": "Claude Fable 5",  "family": "anthropic", "reasoning": True, "vision": True, "tools": True, "cost": "high",
         "note": "Anthropic's most capable widely released model; highest-cost option."},
        {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5", "family": "anthropic", "reasoning": False, "vision": True, "tools": True, "cost": "low",
         "note": "Fast, lower-cost Claude option with image and tool support."},
    ],

    # ── First-party Google Gemini API (official OpenAI-compatible endpoint) ──
    "gemini": [
        {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "family": "google", "reasoning": True, "vision": True, "tools": True, "cost": "medium",
         "note": "Stable default balancing speed and intelligence for agentic multimodal tasks."},
        {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview", "family": "google", "reasoning": True, "vision": True, "tools": True, "cost": "high",
         "note": "Preview model for complex reasoning and precise multi-step tool use."},
        {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash", "family": "google", "reasoning": True, "vision": True, "tools": True, "cost": "medium",
         "note": "Stable frontier model for sustained agentic and coding workloads."},
        {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite", "family": "google", "reasoning": True, "vision": True, "tools": True, "cost": "low",
         "note": "Stable, fast, and cost-effective option for high-throughput execution."},
    ],

    # ── DashScope / 百炼 (Alibaba Model Studio) — OpenAI-compatible ─────────
    # Capabilities below are VERIFIED against the live DashScope endpoint
    # (function-calling + image input each probed directly), not guessed.
    # The chat model drives OphAgent's tool loop, so it MUST have tools:True.
    # NOTE: the bare `qwen-vl-max` alias resolves to an OLD snapshot that can't
    # call tools — use `qwen3-vl-plus` (newer, does tools AND vision) instead.
    "dashscope": [
        {"id": "qwen3-vl-plus",            "name": "Qwen3-VL Plus",      "family": "qwen",     "reasoning": False, "vision": True,  "tools": True,  "cost": "medium",
         "note": "Default — newest Qwen3 vision model: calls tools AND sees images. One model runs everything."},
        {"id": "qwen3-vl-235b-a22b-instruct", "name": "Qwen3-VL 235B",    "family": "qwen",     "reasoning": False, "vision": True,  "tools": True,  "cost": "medium",
         "paper_backbone": "Qwen-3-VL",
         "note": "Vision and tool-calling support through DashScope."},
        {"id": "deepseek-v3",              "name": "DeepSeek V3",        "family": "deepseek", "reasoning": False, "vision": False, "tools": True,  "cost": "low",
         "paper_backbone": "DeepSeek-V3",
         "note": "Text-only planner with tool-calling support through DashScope."},
        {"id": "qwen3.7-max",              "name": "Qwen3.7 Max",        "family": "qwen",     "reasoning": False, "vision": False, "tools": True,  "cost": "high",
         "note": "Current flagship brain — strongest reasoning. Text only; set a VL vision model for image reads."},
        {"id": "qwen3-vl-flash",           "name": "Qwen3-VL Flash",     "family": "qwen",     "reasoning": False, "vision": True,  "tools": True,  "cost": "low",
         "note": "Fast & cheap vision+tools. Good default for quick/bulk runs."},
        {"id": "qwen-max",                 "name": "Qwen Max",           "family": "qwen",     "reasoning": False, "vision": False, "tools": True,  "cost": "medium",
         "note": "Stable text flagship alias. Tools ✓; text only."},
        {"id": "deepseek-v4-pro",          "name": "DeepSeek V4 Pro",    "family": "deepseek", "reasoning": False, "vision": False, "tools": True,  "cost": "medium",
         "note": "Latest DeepSeek flagship (hosted by Alibaba). Strong reasoning, tools ✓; text only — set a VL vision model."},
        {"id": "deepseek-v4-flash",        "name": "DeepSeek V4 Flash",  "family": "deepseek", "reasoning": False, "vision": False, "tools": True,  "cost": "low",
         "note": "DeepSeek V4 fast/cheap variant. Tools ✓; text only."},
        {"id": "deepseek-r1",              "name": "DeepSeek R1",        "family": "deepseek", "reasoning": True,  "vision": False, "tools": True,  "cost": "medium",
         "note": "DeepSeek reasoning. Strong on hard cases; text only."},
        {"id": "qwen-vl-max",              "name": "Qwen-VL Max (vision-only)", "family": "qwen", "reasoning": False, "vision": True, "tools": False, "cost": "medium",
         "note": "⚠ Bare alias = old snapshot, CANNOT call tools. Don't pick as chat model; only usable as OPH_WEB_VISION_MODEL."},
    ],
}


def list_providers() -> list[str]:
    return list(CATALOG.keys())


def list_models(provider: str) -> list[dict]:
    return CATALOG.get(provider, [])


def default_model(provider: str) -> str:
    models = CATALOG.get(provider, [])
    return models[0]["id"] if models else ""


def model_supports_tools(provider: str, model: str) -> bool:
    """Whether this model can drive the agent's tool loop (function-calling).
    Defaults to True for unknown models — only KNOWN tool-incapable models
    (catalog entry with tools=False, e.g. qwen-vl-max) return False."""
    for m in CATALOG.get(provider, []):
        if m["id"] == model:
            return m.get("tools", True)
    return True
