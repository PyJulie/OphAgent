"""Bounded LLM-assisted repair for failed tool invocations.

The Executor may repair a malformed invocation, but it never executes free-form
commands. A repaired call must target the same registered tool and pass the
tool's deterministic argument schema before the runtime can retry it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


_REPAIRABLE_ERROR_MARKERS = (
    "missing required",
    "required positional argument",
    "unexpected keyword argument",
    "invalid argument",
    "invalid parameter",
    "malformed argument",
    "must be ",
    "should be ",
    "not in enum",
)


@dataclass(frozen=True)
class ExecutorRepair:
    action: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str = ""


def repairable_invocation_failure(
    result: Any,
    *,
    arguments_parse_failed: bool = False,
) -> bool:
    """Return whether an LLM repair may safely address this failure."""
    if arguments_parse_failed:
        return True
    if not isinstance(result, dict):
        return False
    error = str(result.get("error") or "").lower()
    return bool(error) and any(marker in error for marker in _REPAIRABLE_ERROR_MARKERS)


def schema_for_tool(
    tool_schemas: list[dict[str, Any]],
    tool_name: str,
) -> dict[str, Any] | None:
    for schema in tool_schemas:
        function = schema.get("function") or {}
        if function.get("name") == tool_name:
            return schema
    return None


def validate_tool_arguments(
    tool_schema: dict[str, Any],
    arguments: dict[str, Any],
) -> list[str]:
    """Apply the deterministic subset of JSON Schema used by OphAgent tools."""
    if not isinstance(arguments, dict):
        return ["arguments must be an object"]

    function = tool_schema.get("function") or {}
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    errors: list[str] = []

    for name in required:
        if name not in arguments or arguments[name] in (None, ""):
            errors.append(f"missing required argument: {name}")

    expected_python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for name, value in arguments.items():
        definition = properties.get(name)
        if definition is None:
            errors.append(f"unknown argument: {name}")
            continue
        expected_name = definition.get("type")
        expected_type = expected_python_types.get(expected_name)
        if expected_type is not None:
            if expected_name in {"integer", "number"} and isinstance(value, bool):
                errors.append(f"{name} must be {expected_name}")
            elif not isinstance(value, expected_type):
                errors.append(f"{name} must be {expected_name}")
        enum = definition.get("enum")
        if enum and value not in enum:
            errors.append(f"{name} must be one of {enum}")
    return errors


def parse_executor_repair(
    raw_arguments: str,
    *,
    attempted_tool: str,
    tool_schema: dict[str, Any],
) -> ExecutorRepair | None:
    """Parse and validate the Executor LLM's structured repair proposal."""
    try:
        payload = json.loads(raw_arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    action = str(payload.get("action") or "").lower()
    if action == "abort":
        return ExecutorRepair(
            action="abort",
            tool_name=attempted_tool,
            arguments={},
            reason=str(payload.get("reason") or ""),
        )
    if action != "retry" or payload.get("tool_name") != attempted_tool:
        return None

    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    if validate_tool_arguments(tool_schema, arguments):
        return None
    return ExecutorRepair(
        action="retry",
        tool_name=attempted_tool,
        arguments=arguments,
        reason=str(payload.get("reason") or ""),
    )


def executor_repair_tool_schema() -> dict[str, Any]:
    """Synthetic function schema used only for the Executor repair decision."""
    return {
        "type": "function",
        "function": {
            "name": "repair_tool_invocation",
            "description": (
                "Repair the arguments for the attempted registered tool call, "
                "or abort when the failure cannot be fixed by changing arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["retry", "abort"],
                    },
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "required": ["action", "tool_name", "arguments", "reason"],
            },
        },
    }

