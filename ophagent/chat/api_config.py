"""Provider connection settings shared by the Web UI and OphSession.

Per-user overrides are supplied by the Web server at runtime. Environment
variables remain the fallback for CLI and batch entry points.
"""

from __future__ import annotations

import os
from typing import Any


API_CHANNELS: dict[str, dict[str, str]] = {
    "gateway": {
        "label": "Multi-model gateways",
        "description": "One endpoint for multiple model families",
        "default_provider": "openrouter",
    },
    "official": {
        "label": "Official providers",
        "description": "Direct connections to model providers",
        "default_provider": "openai",
    },
}


PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "aigcbest": {
        "label": "AIGCBest",
        "channel": "gateway",
        "api_key_env": "AIGCBEST_API_KEY",
        "base_url_env": "AIGCBEST_BASE_URL",
        "default_base_url": "https://api2.aigcbest.top/v1",
        "timeout": 240.0,
        "max_retries": 1,
    },
    "openrouter": {
        "label": "OpenRouter",
        "channel": "gateway",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_headers": {"X-Title": "oph-agent"},
    },
    "dashscope": {
        "label": "DashScope (Alibaba)",
        "channel": "official",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "timeout": 120.0,
        "max_retries": 2,
    },
    "openai": {
        "label": "OpenAI",
        "channel": "official",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "timeout": 120.0,
        "max_retries": 2,
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "channel": "official",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "default_base_url": "https://api.anthropic.com/v1",
        "timeout": 180.0,
        "max_retries": 2,
    },
    "gemini": {
        "label": "Google Gemini",
        "channel": "official",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "timeout": 180.0,
        "max_retries": 2,
    },
}

# Defaults used only when OPH_WEB_MODEL is absent. Keeping them beside the
# provider connection map prevents Web startup and preflight from drifting.
DEFAULT_WEB_MODELS: dict[str, str] = {
    "aigcbest": "claude-sonnet-4-6",
    "openrouter": "anthropic/claude-sonnet-4-6",
    "dashscope": "qwen3-vl-plus",
    "openai": "gpt-5.6",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.6-flash",
}


def list_api_channels() -> list[dict[str, str]]:
    """Return the ordered API-channel metadata used by Web and CLI pickers."""
    return [{"id": channel_id, **metadata} for channel_id, metadata in API_CHANNELS.items()]


def provider_channel(provider: str) -> str:
    if provider not in PROVIDER_SPECS:
        raise ValueError(f"unsupported API provider: {provider}")
    return str(PROVIDER_SPECS[provider]["channel"])


def providers_for_channel(channel: str) -> list[str]:
    if channel not in API_CHANNELS:
        raise ValueError(f"unsupported API channel: {channel}")
    return [
        provider
        for provider, spec in PROVIDER_SPECS.items()
        if spec["channel"] == channel
    ]


def resolve_provider_connection(
    provider: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve a provider without exposing the environment to callers."""
    if provider not in PROVIDER_SPECS:
        raise ValueError(f"unsupported API provider: {provider}")
    spec = PROVIDER_SPECS[provider]
    overrides = overrides or {}
    personal_key = str(overrides.get("api_key") or "").strip()
    environment_key = os.environ.get(spec["api_key_env"], "").strip()
    personal_base = str(overrides.get("base_url") or "").strip().rstrip("/")
    environment_base = os.environ.get(spec["base_url_env"], "").strip().rstrip("/")
    base_url = personal_base or environment_base or spec.get("default_base_url")
    return {
        "provider": provider,
        "label": spec["label"],
        "channel": spec["channel"],
        "api_key": personal_key or environment_key,
        "base_url": base_url,
        "source": "personal" if personal_key else ("environment" if environment_key else "missing"),
        "has_personal_key": bool(personal_key),
        "has_custom_base_url": bool(personal_base),
    }


def create_provider_client(
    provider: str,
    overrides: dict[str, str] | None = None,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
):
    """Create an OpenAI-compatible client for a configured provider."""
    import openai

    resolved = resolve_provider_connection(provider, overrides)
    if not resolved["api_key"]:
        env_name = PROVIDER_SPECS[provider]["api_key_env"]
        raise RuntimeError(f"{env_name} env var or a Personalization API key is required")

    spec = PROVIDER_SPECS[provider]
    kwargs: dict[str, Any] = {"api_key": resolved["api_key"]}
    if resolved["base_url"]:
        kwargs["base_url"] = resolved["base_url"]
    resolved_timeout = timeout if timeout is not None else spec.get("timeout")
    resolved_retries = (
        max_retries if max_retries is not None else spec.get("max_retries")
    )
    if resolved_timeout is not None:
        kwargs["timeout"] = resolved_timeout
    if resolved_retries is not None:
        kwargs["max_retries"] = resolved_retries
    if spec.get("default_headers"):
        kwargs["default_headers"] = dict(spec["default_headers"])
    return openai.OpenAI(**kwargs)
