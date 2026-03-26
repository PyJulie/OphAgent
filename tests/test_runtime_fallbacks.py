"""Regression tests for runtime protocol and fallback paths."""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata
from ophagent.tools.scheduler import ToolScheduler


class FallbackTool(BaseTool):
    def run(self, inputs):
        raise RuntimeError("boom")

    def fallback_run(self, inputs, error):
        return {"handled": True, "error": str(error)}


class FastAPIClientTool(FastAPIToolMixin, BaseTool):
    def run(self, inputs):
        return self._post(8110, "/run", {"image_b64": "abc", "params": {}})


def runtime_settings(mode: str = "graceful"):
    return SimpleNamespace(
        allow_fallbacks=(mode != "strict"),
        runtime=SimpleNamespace(mode=mode, is_strict=(mode == "strict")),
        llm=SimpleNamespace(
            provider="openai",
            model_id="gpt-5",
            api_key=None,
            local_model_url="http://localhost:11434/v1",
        ),
    )


def make_metadata(tool_id: str, mode: str = "inline") -> ToolMetadata:
    return ToolMetadata(
        tool_id=tool_id,
        name=tool_id,
        description="test",
        modality="CFP",
        task="classification",
        scheduling_mode=mode,
        input_type="image",
        output_type="dict",
        fastapi_port=8110 if mode == "fastapi" else None,
    )


def test_base_tool_uses_fallback_run():
    tool = FallbackTool(make_metadata("fallback_tool"))
    with patch("config.settings.get_settings", return_value=runtime_settings("graceful")):
        result = tool({"image_path": "img.jpg"})
    assert result["handled"] is True
    assert result["used_fallback"] is True
    assert result["degraded"] is True
    assert result["success"] is True
    assert result["tool_id"] == "fallback_tool"


def test_base_tool_raises_in_strict_mode():
    tool = FallbackTool(make_metadata("fallback_tool"))
    with patch("config.settings.get_settings", return_value=runtime_settings("strict")):
        with pytest.raises(RuntimeError, match="boom"):
            tool({"image_path": "img.jpg"})


def test_fastapi_mixin_unwraps_service_envelope():
    tool = FastAPIClientTool(make_metadata("client_tool", mode="fastapi"))
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "success": True,
        "result": {"quality_score": 0.8},
        "latency_ms": 12.5,
        "model_id": "cfp_quality",
    }
    with patch("httpx.post", return_value=fake_response):
        result = tool({"image_path": "img.jpg"})
    assert result["quality_score"] == 0.8
    assert result["service_latency_ms"] == 12.5
    assert result["service_model_id"] == "cfp_quality"


def test_scheduler_prefers_tool_wrapper_for_fastapi_tools():
    registry = MagicMock()
    meta = make_metadata("cfp_quality", mode="fastapi")
    registry.exists.return_value = True
    registry.get.return_value = meta
    scheduler = ToolScheduler(registry=registry)
    fake_tool = MagicMock(return_value={"wrapped": True})
    with patch.object(scheduler, "_get_or_load_tool", return_value=fake_tool):
        result = scheduler.run("cfp_quality", {"image_path": "img.jpg"})
    fake_tool.assert_called_once_with({"image_path": "img.jpg"})
    assert result == {"wrapped": True}


def test_service_spec_marks_multimodal_service_correctly():
    pytest.importorskip("fastapi")
    from ophagent.models.inference.service import _service_spec

    spec = _service_spec("cfp_ffa_multimodal")
    assert spec.dual_image is True


def test_llm_backbone_raises_in_strict_mode_when_provider_init_fails():
    from ophagent.llm.backbone import LLMBackbone

    with patch("config.settings.get_settings", return_value=runtime_settings("strict")):
        with patch("ophagent.llm.backbone.OpenAILLM", side_effect=RuntimeError("init fail")):
            with pytest.raises(RuntimeError, match="strict mode"):
                LLMBackbone()
