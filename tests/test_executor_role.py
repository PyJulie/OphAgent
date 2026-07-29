from __future__ import annotations

import json
from types import SimpleNamespace

from ophagent.chat.executor_role import (
    parse_executor_repair,
    repairable_invocation_failure,
    validate_tool_arguments,
)
from ophagent.chat.oph_session import OphSession


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_image",
        "description": "Classify one image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "threshold": {"type": "number"},
            },
            "required": ["image_path"],
        },
    },
}


def _repair_response(arguments: dict):
    call = SimpleNamespace(
        function=SimpleNamespace(arguments=json.dumps(arguments)),
    )
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_only_invocation_errors_are_repairable() -> None:
    assert repairable_invocation_failure(
        {"error": "missing 1 required positional argument: image_path"}
    )
    assert repairable_invocation_failure(
        {"error": "invalid parameter: threshold"}
    )
    assert repairable_invocation_failure(
        {"error": "anything"},
        arguments_parse_failed=True,
    )
    assert not repairable_invocation_failure(
        {"error": "CUDA out of memory"}
    )
    assert not repairable_invocation_failure(
        {"error": "checkpoint download failed"}
    )


def test_argument_validation_is_schema_constrained() -> None:
    assert validate_tool_arguments(
        TOOL_SCHEMA,
        {"image_path": "image.png", "threshold": 0.5},
    ) == []
    assert "missing required argument: image_path" in validate_tool_arguments(
        TOOL_SCHEMA,
        {"threshold": 0.5},
    )
    assert "threshold must be number" in validate_tool_arguments(
        TOOL_SCHEMA,
        {"image_path": "image.png", "threshold": "high"},
    )
    assert "unknown argument: shell_command" in validate_tool_arguments(
        TOOL_SCHEMA,
        {"image_path": "image.png", "shell_command": "anything"},
    )


def test_repair_cannot_substitute_another_tool() -> None:
    payload = {
        "action": "retry",
        "tool_name": "different_tool",
        "arguments": {"image_path": "image.png"},
        "reason": "substitute",
    }
    assert parse_executor_repair(
        json.dumps(payload),
        attempted_tool="classify_image",
        tool_schema=TOOL_SCHEMA,
    ) is None


def test_executor_repairs_failed_arguments_with_role_model() -> None:
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _repair_response({
                "action": "retry",
                "tool_name": "classify_image",
                "arguments": {
                    "image_path": "image.png",
                    "threshold": 0.5,
                },
                "reason": "restore the required image path",
            })

    session = OphSession.new(
        backend="openrouter",
        model="planner-model",
        executor_model="executor-model",
    )
    session._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    events: list[dict] = []

    repair = session._executor_repair_tool_call(
        attempted_tool="classify_image",
        attempted_arguments={"threshold": 0.5},
        attempted_arguments_raw='{"threshold": 0.5}',
        result={"error": "missing required argument: image_path"},
        arguments_parse_failed=False,
        tool_schemas=[TOOL_SCHEMA],
        emit=events.append,
    )

    assert repair is not None
    assert repair.arguments["image_path"] == "image.png"
    assert calls[0]["model"] == "executor-model"
    assert events[-1]["status"] == "retrying"


def test_successful_tool_result_never_calls_executor_llm() -> None:
    class FailingCompletions:
        def create(self, **kwargs):
            raise AssertionError("Executor LLM must not run on a successful call")

    session = OphSession.new()
    session._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )

    repair = session._executor_repair_tool_call(
        attempted_tool="classify_image",
        attempted_arguments={"image_path": "image.png"},
        attempted_arguments_raw='{"image_path": "image.png"}',
        result={"success": True, "prediction": "normal"},
        arguments_parse_failed=False,
        tool_schemas=[TOOL_SCHEMA],
        emit=lambda event: None,
    )

    assert repair is None

