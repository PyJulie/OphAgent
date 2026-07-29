from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from ophagent.chat.oph_tools import auto_detect_modality
from ophagent.chat.oph_session import OphSession, _effort_directive
from ophagent.preflight import (
    check_llm_backend,
    resolve_runtime_config,
)
from ophagent.webchat.server import _session_runtime_config


_RUNTIME_ENV = (
    "OPH_WEB_BACKEND",
    "OPH_WEB_MODEL",
    "OPH_WEB_VISION_BACKEND",
    "OPH_WEB_VISION_MODEL",
    "OPH_WEB_EFFORT",
)


def _clear_runtime_env(monkeypatch) -> None:
    for name in _RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)


def test_runtime_config_uses_shared_defaults(monkeypatch):
    _clear_runtime_env(monkeypatch)
    config = resolve_runtime_config(backend="dashscope")
    assert config["planner"]["model"] == "qwen3-vl-plus"
    assert config["vision"]["model"] == "qwen3-vl-plus"
    assert config["vision"]["inherits_planner_model"] is True
    assert config["errors"] == []


def test_runtime_arguments_override_environment(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("OPH_WEB_BACKEND", "openrouter")
    monkeypatch.setenv("OPH_WEB_MODEL", "environment-model")
    monkeypatch.setenv("OPH_WEB_EFFORT", "medium")
    config = resolve_runtime_config(
        backend="aigcbest",
        model="argument-model",
        vision_backend="dashscope",
        vision_model="qwen3-vl-plus",
        effort="high",
    )
    assert config["planner"] == {
        "backend": "aigcbest",
        "model": "argument-model",
        "backend_source": "argument",
        "model_source": "argument",
    }
    assert config["vision"]["backend"] == "dashscope"
    assert config["vision"]["model"] == "qwen3-vl-plus"
    assert config["vision"]["inherits_planner_model"] is False
    assert config["effort"] == "high"


def test_quick_llm_check_requires_selected_provider_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = check_llm_backend(
        "openrouter", "anthropic/claude-sonnet-4.6", probe=False
    )
    assert result.status == "FAIL"
    assert "OPENROUTER_API_KEY" in result.detail


def test_full_llm_check_probes_exact_selected_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setattr(
        "ophagent.preflight.create_provider_client",
        lambda *args, **kwargs: fake_client,
    )
    result = check_llm_backend(
        "openrouter", "vendor/selected-model", probe=True
    )
    assert result.status == "OK"
    assert calls[0]["model"] == "vendor/selected-model"


def test_session_runtime_exposes_resolved_vision_and_policy(monkeypatch, tmp_path):
    monkeypatch.delenv("OPH_WEB_VISION_BACKEND", raising=False)
    monkeypatch.delenv("OPH_WEB_VISION_MODEL", raising=False)
    session = OphSession.new(
        backend="dashscope",
        model="deepseek-v4-pro",
        effort="high",
        workspace=str(tmp_path),
        vision_backend="aigcbest",
        vision_model_override="gpt-5-mini",
    )
    runtime = _session_runtime_config(session)
    assert runtime["status"] == "configured"
    assert "ready" not in runtime
    assert runtime["components"]["planner"] == {
        "backend": "dashscope",
        "model": "deepseek-v4-pro",
    }
    assert runtime["components"]["vision"]["backend"] == "aigcbest"
    assert runtime["components"]["vision"]["model"] == "gpt-5-mini"
    assert runtime["components"]["vision"]["available"] is True
    assert runtime["components"]["verifier"]["mode"] == "independent_llm"


def test_low_effort_does_not_request_missing_uwf_quality_adapter():
    directive = _effort_directive(
        "low", vision_available=True, attached_modalities=["UWF"]
    )
    assert "`uwf_disease_7class`" in directive
    assert "do NOT call `analyze_image`" in directive
    assert "ALWAYS run the quality check" not in directive


def test_low_effort_keeps_available_cfp_quality_adapter():
    directive = _effort_directive(
        "low", vision_available=True, attached_modalities=["CFP"]
    )
    assert "`cfp_clip_ensemble`" in directive
    assert "`cfp_efiqa`" in directive


def test_unverified_image_scope_refuses_without_diagnostic_fields(
    monkeypatch, tmp_path
):
    path = tmp_path / "scope_unknown.jpg"
    Image.new("RGB", (64, 64), color=(32, 128, 224)).save(path)
    monkeypatch.setattr(
        "ophagent.chat.oph_tools.filename_modality_hint", lambda _: None
    )
    monkeypatch.setattr(
        "ophagent.chat.oph_tools.cnn_modality_hint", lambda _: None
    )
    session = OphSession.new(
        backend="dashscope",
        model="deepseek-v3",
        workspace=str(tmp_path),
    )
    session.set_image(str(path))
    assert session.context.modality_scope == "unverified_input"
    reply = session.chat("Please analyze this image.")
    assert "===UNVERIFIED_INPUT===" in reply
    assert "===FINAL===" not in reply
    assert '"top1"' not in reply
    assert '"confidence"' not in reply
    assert '"probability"' not in reply


def test_modality_detection_has_no_pixel_guess_fallback(monkeypatch):
    monkeypatch.setattr(
        "ophagent.chat.oph_tools.filename_modality_hint", lambda _: None
    )
    monkeypatch.setattr(
        "ophagent.chat.oph_tools.cnn_modality_hint", lambda _: None
    )
    assert auto_detect_modality("unknown.jpg") == "UNVERIFIED_INPUT"
