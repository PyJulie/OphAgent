from __future__ import annotations

import openai

from ophagent.chat.api_config import (
    API_CHANNELS,
    DEFAULT_WEB_MODELS,
    PROVIDER_SPECS,
    create_provider_client,
    list_api_channels,
    provider_channel,
    providers_for_channel,
    resolve_provider_connection,
)
from ophagent.webchat import models_catalog


def test_provider_hierarchy_separates_gateways_and_official_apis() -> None:
    assert list(API_CHANNELS) == ["gateway", "official"]
    assert providers_for_channel("gateway") == ["aigcbest", "openrouter"]
    assert providers_for_channel("official") == [
        "dashscope",
        "openai",
        "anthropic",
        "gemini",
    ]
    assert provider_channel("openrouter") == "gateway"
    assert provider_channel("anthropic") == "official"
    assert [item["id"] for item in list_api_channels()] == [
        "gateway",
        "official",
    ]


def test_official_provider_connections_use_first_party_endpoints(
    monkeypatch,
) -> None:
    expected = {
        "openai": (
            "OPENAI_API_KEY",
            "https://api.openai.com/v1",
        ),
        "anthropic": (
            "ANTHROPIC_API_KEY",
            "https://api.anthropic.com/v1",
        ),
        "gemini": (
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    }
    for provider, (env_name, base_url) in expected.items():
        monkeypatch.delenv(env_name, raising=False)
        monkeypatch.delenv(PROVIDER_SPECS[provider]["base_url_env"], raising=False)
        resolved = resolve_provider_connection(
            provider,
            {"api_key": f"{provider}-test-key"},
        )
        assert resolved["channel"] == "official"
        assert resolved["source"] == "personal"
        assert resolved["base_url"] == base_url


def test_official_catalog_defaults_are_tool_and_vision_capable() -> None:
    for provider in ("openai", "anthropic", "gemini"):
        model_id = DEFAULT_WEB_MODELS[provider]
        model = next(
            item
            for item in models_catalog.list_models(provider)
            if item["id"] == model_id
        )
        assert model["tools"] is True
        assert model["vision"] is True
        assert models_catalog.default_model(provider) == model_id


def test_manuscript_ophagent_backbones_are_all_selectable() -> None:
    paper_models = {
        item["paper_backbone"]: item
        for item in models_catalog.list_models("aigcbest")
        if item.get("paper_backbone")
    }
    assert set(paper_models) == {
        "GPT-5",
        "Gemini-3.0-Pro",
        "Qwen-3-VL",
        "DeepSeek-V3",
    }
    assert all(model.get("tools", True) for model in paper_models.values())


def test_first_party_catalogs_include_available_manuscript_backbones() -> None:
    openai_models = {
        item["id"]: item for item in models_catalog.list_models("openai")
    }
    dashscope_models = {
        item["id"]: item for item in models_catalog.list_models("dashscope")
    }
    assert openai_models["gpt-5"]["paper_backbone"] == "GPT-5"
    assert (
        dashscope_models["qwen3-vl-235b-a22b-instruct"]["paper_backbone"]
        == "Qwen-3-VL"
    )
    assert (
        dashscope_models["deepseek-v3"]["paper_backbone"]
        == "DeepSeek-V3"
    )


def test_official_clients_use_openai_compatible_provider_urls(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    def fake_openai(**kwargs):
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    for provider in ("openai", "anthropic", "gemini"):
        create_provider_client(
            provider,
            {"api_key": f"{provider}-test-key"},
        )

    assert [item["base_url"] for item in captured] == [
        "https://api.openai.com/v1",
        "https://api.anthropic.com/v1",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    ]
