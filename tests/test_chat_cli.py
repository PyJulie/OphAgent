from __future__ import annotations

import json

from ophagent.chat.cli import (
    _handle_slash,
    _interactive_provider_selection,
    _load_session,
    _resolve_cli_selection,
    _session_configuration,
)
from ophagent.chat.oph_session import OphSession


def test_cli_loads_current_ophsession(tmp_path) -> None:
    session = OphSession.new(
        backend="openrouter",
        model="example/model",
        effort="medium",
        prompt_profile="standard",
        workspace=str(tmp_path),
    )
    session.messages.append({"role": "user", "content": "hello"})
    saved = session.save(tmp_path / "current.json")

    loaded = _load_session(saved)

    assert loaded.session_id == session.session_id
    assert loaded.model == "example/model"
    assert loaded.effort == "medium"
    assert loaded.messages == session.messages


def test_cli_migrates_legacy_oct_session(tmp_path) -> None:
    image = tmp_path / "slice.png"
    image.write_bytes(b"not-decoded-during-load")
    legacy = {
        "session_id": "legacy123",
        "backend": "openrouter",
        "model": "openai/gpt-5.5-pro",
        "caption_model": "openai/gpt-5.4",
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": "review this OCT"}],
        "context": {
            "current_image": str(image),
            "current_volume": None,
            "analyses": {},
            "volume_analyses": {"ignored": {"status": "legacy"}},
            "last_report": None,
        },
        "created_at": 123.0,
        "workspace": str(tmp_path),
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = _load_session(path)

    assert isinstance(loaded, OphSession)
    assert loaded.session_id == "legacy123"
    assert loaded.context.current_image == str(image)
    assert loaded.vision_backend == "openrouter"
    assert loaded.vision_model_override == "openai/gpt-5.4"
    assert loaded.prompt_profile == "standard"


def test_cli_volume_registration_and_reset_preserve_runtime(tmp_path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    session = OphSession.new(
        backend="openai",
        model="gpt-5",
        effort="high",
        vision_backend="openai",
        vision_model_override="gpt-5",
        workspace=str(tmp_path),
    )

    active = _handle_slash(session, f"/volume {volume}")
    assert active is session
    assert session.context.current_volume == str(volume.resolve())

    reset = _handle_slash(session, "/reset")
    assert isinstance(reset, OphSession)
    assert reset.session_id != session.session_id
    assert _session_configuration(reset) == _session_configuration(session)
    assert reset.context.current_volume is None


def test_cli_model_switch_invalidates_cached_clients() -> None:
    session = OphSession.new(model="old-model")
    session._client = object()
    session._vision_resolved = ("cached-model", "configured")

    active = _handle_slash(session, "/model new-model")

    assert active is session
    assert session.model == "new-model"
    assert session._client is None
    assert session._vision_resolved is None


def test_cli_resolves_official_provider_default_model() -> None:
    channel, provider, model = _resolve_cli_selection(
        "official",
        "anthropic",
        None,
    )

    assert channel == "official"
    assert provider == "anthropic"
    assert model == "claude-sonnet-5"


def test_cli_preserves_legacy_default_without_hierarchy_flags() -> None:
    assert _resolve_cli_selection(None, None, None) == (
        "gateway",
        "openrouter",
        "openai/gpt-5.5-pro",
    )


def test_cli_interactive_picker_selects_provider_then_model() -> None:
    answers = iter(["", ""])

    channel, provider, model = _interactive_provider_selection(
        input_fn=lambda _prompt: next(answers)
    )

    assert channel == "official"
    assert provider == "openai"
    assert model == "gpt-5.6"


def test_cli_reset_preserves_ephemeral_api_credentials(tmp_path) -> None:
    session = OphSession.new(workspace=str(tmp_path))
    session._api_credentials = {
        "anthropic": {"api_key": "not-persisted-secret"}
    }

    reset = _handle_slash(session, "/reset")

    assert reset is not None
    assert reset._api_credentials == session._api_credentials
    saved = reset.save(tmp_path / "reset.json")
    assert "not-persisted-secret" not in saved.read_text(encoding="utf-8")
