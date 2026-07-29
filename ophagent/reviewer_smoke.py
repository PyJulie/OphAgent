"""Deterministic offline checks for reviewer-facing safety behavior.

Run from the repository root with ``python -m ophagent.reviewer_smoke``.
No API key, model checkpoint, or clinical dataset is used by these checks.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ophagent.chat.oph_session import OphSession


logging.disable(logging.CRITICAL)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _new_session(workspace: str) -> OphSession:
    return OphSession.new(
        backend="aigcbest",
        model="gpt-5-mini",
        workspace=workspace,
    )


def _assert_no_diagnostic_fields(reply: str, name: str) -> None:
    _assert("===FINAL===" not in reply, f"{name} produced FINAL")
    for field in ('"top1"', '"confidence"', '"probability"'):
        _assert(field not in reply, f"{name} produced {field}")


def test_missing_file_refusal(workspace: str) -> dict:
    session = _new_session(workspace)
    session.set_image(str(Path(workspace) / "missing.jpg"))
    _assert(session.context.modality_scope == "invalid_input", "missing file scope")
    reply = session.chat("Please analyze this image.")
    _assert("===INVALID_INPUT===" in reply, "missing file marker")
    _assert_no_diagnostic_fields(reply, "missing file")
    return {"name": "missing_file_refusal", "ok": True}


def test_non_image_file_refusal(workspace: str) -> dict:
    path = Path(workspace) / "not_an_image.jpg"
    path.write_text("this is not a jpeg", encoding="utf-8")
    session = _new_session(workspace)
    session.set_image(str(path))
    _assert(session.context.modality_scope == "invalid_input", "non-image scope")
    reply = session.chat("Please analyze this image.")
    _assert("===INVALID_INPUT===" in reply, "non-image marker")
    _assert_no_diagnostic_fields(reply, "non-image")
    return {"name": "non_image_file_refusal", "ok": True}


def test_non_ophthalmic_refusal_without_key(workspace: str) -> dict:
    session = _new_session(workspace)
    session.context.current_image = str(Path(workspace) / "dummy.jpg")
    session.context.current_modality = "NON_OPHTHALMOLOGIC"
    session.context.modality_scope = "non_ophth"
    reply = session.chat("What is the diagnosis?")
    _assert("===NOT_OPHTHALMOLOGIC===" in reply, "non-ophthalmic marker")
    _assert_no_diagnostic_fields(reply, "non-ophthalmic input")
    return {"name": "non_ophthalmic_refusal_without_key", "ok": True}


def test_unverified_scope_refusal(workspace: str) -> dict:
    path = Path(workspace) / "scope_unknown.jpg"
    Image.new("RGB", (64, 64), color=(32, 128, 224)).save(path)
    session = OphSession.new(
        backend="dashscope",
        model="deepseek-v3",
        workspace=workspace,
    )
    with (
        patch("ophagent.chat.oph_tools.filename_modality_hint", return_value=None),
        patch("ophagent.chat.oph_tools.cnn_modality_hint", return_value=None),
    ):
        session.set_image(str(path))
    _assert(session.context.modality_scope == "unverified_input", "unverified scope")
    reply = session.chat("Please analyze this image.")
    _assert("===UNVERIFIED_INPUT===" in reply, "unverified-input marker")
    _assert_no_diagnostic_fields(reply, "unverified input")
    return {"name": "unverified_scope_refusal", "ok": True}


def test_core_failure_suppresses_diagnosis(workspace: str) -> dict:
    session = _new_session(workspace)
    session.context.current_image = str(Path(workspace) / "dummy.jpg")
    session.context.current_modality = "CFP"
    outcomes = {
        name: "fail" for name in session._CORE_TOOLS_BY_MODALITY["CFP"]
    }
    reply = session._finalize_reply(
        '===FINAL===\n{"top1":"DR","confidence":0.99}',
        outcomes,
    )
    _assert("===INSUFFICIENT_DATA===" in reply, "insufficient-data marker")
    _assert_no_diagnostic_fields(reply, "core failure")
    return {"name": "core_failure_suppresses_diagnosis", "ok": True}


def test_single_modality_path_is_unchanged(workspace: str) -> dict:
    session = _new_session(workspace)
    cfp = str(Path(workspace) / "single_cfp.jpg")
    session.context.current_image = cfp
    session.context.current_modality = "CFP"
    session.context.attached_images = [
        {"path": cfp, "modality": "CFP", "filename": "single_cfp.jpg"},
    ]
    original = "Follow-up answer from cached context."
    _assert(
        session._finalize_reply(original, {}) == original,
        "single-image finalization changed",
    )
    return {"name": "single_modality_path_is_unchanged", "ok": True}


def main() -> int:
    for key in (
        "AIGCBEST_API_KEY",
        "OPENROUTER_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        os.environ.pop(key, None)

    results = []
    with tempfile.TemporaryDirectory(prefix="ophagent_smoke_") as workspace:
        for check in (
            test_missing_file_refusal,
            test_non_image_file_refusal,
            test_non_ophthalmic_refusal_without_key,
            test_unverified_scope_refusal,
            test_core_failure_suppresses_diagnosis,
            test_single_modality_path_is_unchanged,
        ):
            results.append(check(workspace))

    print(json.dumps({"ok": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
