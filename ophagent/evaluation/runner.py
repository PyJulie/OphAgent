"""Execution helpers for protocol-based OphAgent evaluations."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .protocols import DR_SEVERITY_ICDR_SINGLE_IMAGE, TaskProtocol


_FINAL_MARKER = "===FINAL==="


def _extract_json_object(text: str, start: int = 0) -> tuple[str | None, str | None]:
    """Return the first balanced JSON object at or after ``start``.

    A regex is not safe here because the final schema contains nested objects.
    """

    obj_start = text.find("{", start)
    if obj_start < 0:
        return None, "no_json_object"

    depth = 0
    in_string = False
    escape = False
    for idx in range(obj_start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[obj_start:idx + 1], None

    return None, "unterminated_json_object"


@dataclass
class EvaluationRunConfig:
    """Runtime configuration for one protocol-based evaluation run."""

    backend: str = "aigcbest"
    model: str = "gpt-5"
    effort: str = "medium"
    max_tool_steps: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    workspace: str = "reports/ophagent_evaluation_sessions"
    vision_backend: str | None = None
    vision_model: str | None = None
    native_effort: bool | None = None
    architecture_arm: str = "full"
    prompt_profile: str = "standard"


@dataclass
class EvaluationRunResult:
    """Serializable result for one image."""

    image_path: str
    protocol_id: str
    backend: str
    model: str
    effort: str
    elapsed_s: float
    tool_calls: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    analyses: dict[str, Any] = field(default_factory=dict)
    intermediate_results_version: int = 1
    parsed_final: dict[str, Any] = field(default_factory=dict)
    raw_reply_tail: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_max_tool_steps(effort: str) -> int:
    """Bounded defaults for protocol-based evaluations.

    These are deliberately tighter than the historical high/max/ultra defaults:
    evaluation effort represents evidence depth and verification strength, not
    unlimited tool-loop expansion.
    """

    return {
        "low": 8,
        "medium": 14,
        "high": 18,
        "max": 20,
        "ultra": 28,
    }.get(effort, 14)


def architecture_ablation_flags(arm: str) -> tuple[bool, bool]:
    """Return ``(ablate_planner, ablate_verifier)`` for a named agent arm."""

    arms = {
        "single": (True, True),
        "planner": (False, True),
        "full": (False, False),
    }
    try:
        return arms[arm]
    except KeyError as exc:
        raise ValueError(f"unknown architecture_arm {arm!r}; expected one of {sorted(arms)}") from exc


def parse_final(text: str | None) -> dict[str, Any]:
    """Parse the v2 final block without trusting the surrounding prose."""

    if not text:
        return {"parse_method": "empty", "grade": None}

    marker_idx = text.rfind(_FINAL_MARKER)
    if marker_idx < 0:
        return {
            "parse_method": "failed",
            "parse_error": "no_final_marker",
            "grade": None,
        }

    block, extract_error = _extract_json_object(text, marker_idx + len(_FINAL_MARKER))
    if block:
        try:
            data = json.loads(block)
            grade = data.get("grade")
            if isinstance(grade, str) and grade.isdigit():
                grade = int(grade)
            data["grade"] = grade if isinstance(grade, int) and 0 <= grade <= 4 else None
            data["parse_method"] = "json_block"
            return data
        except Exception as exc:
            parse_error = f"json: {type(exc).__name__}: {exc}"
    else:
        parse_error = extract_error or "no_json_object"

    return {
        "parse_method": "failed",
        "parse_error": parse_error,
        "grade": None,
    }


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        return str(obj)


def _current_analyses(session: Any | None) -> dict[str, Any]:
    if session is None:
        return {}
    img = getattr(session.context, "current_image", None)
    analyses = getattr(session.context, "analyses", {}) or {}
    by_tool = analyses.get(img, {}) if img else {}
    if not by_tool and analyses:
        by_tool = next(iter(reversed(analyses.values())))
    return {
        "current_image": img,
        "by_tool": _json_safe(by_tool),
    }


@contextmanager
def _temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    old: dict[str, str | None] = {}
    for key, value in overrides.items():
        old[key] = os.environ.get(key)
        if value is None:
            continue
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_image(
    image_path: str | Path,
    *,
    protocol: TaskProtocol = DR_SEVERITY_ICDR_SINGLE_IMAGE,
    config: EvaluationRunConfig | None = None,
    user_question: str | None = None,
) -> EvaluationRunResult:
    """Run one image through a declared evaluation protocol."""

    cfg = config or EvaluationRunConfig()
    image = str(Path(image_path))
    prompt = protocol.build_user_prompt(cfg.effort, user_question=user_question)
    max_steps = cfg.max_tool_steps or default_max_tool_steps(cfg.effort)

    env = {
        "OPH_WEB_VISION_BACKEND": cfg.vision_backend,
        "OPH_WEB_VISION_MODEL": cfg.vision_model,
    }
    if cfg.native_effort is not None:
        env["OPH_NATIVE_EFFORT"] = "1" if cfg.native_effort else "0"

    tool_calls: list[str] = []
    tool_events: list[dict[str, Any]] = []
    session = None
    t0 = time.time()
    try:
        with _temporary_env(env):
            # Import lazily so evaluation utilities can be inspected without
            # loading the heavy adapter registry or API clients.
            import ophagent.adapters  # noqa: F401
            from ophagent.chat.oph_session import OphSession

            session = OphSession.new(
                backend=cfg.backend,
                model=cfg.model,
                effort=cfg.effort,
                workspace=cfg.workspace,
                prompt_profile=cfg.prompt_profile,
                **({"max_tokens": cfg.max_tokens} if cfg.max_tokens else {}),
                **({"temperature": cfg.temperature} if cfg.temperature is not None else {}),
            )
            ablate_planner, ablate_verifier = architecture_ablation_flags(cfg.architecture_arm)
            session._ablate_planner = ablate_planner
            session._ablate_verifier = ablate_verifier
            session.set_image(image)

            def _on_event(event: dict[str, Any]) -> None:
                if event.get("type") in {"tool_call", "tool_result"}:
                    item = _json_safe(event)
                    if isinstance(item, dict):
                        item["t_rel_s"] = round(time.time() - t0, 3)
                    tool_events.append(item)
                if event.get("type") == "tool_call":
                    tool_calls.append(str(event.get("name") or "?"))

            reply = session.chat(prompt, on_event=_on_event, max_tool_steps=max_steps)
            parsed = parse_final(reply)
            return EvaluationRunResult(
                image_path=image,
                protocol_id=protocol.task_id,
                backend=cfg.backend,
                model=cfg.model,
                effort=cfg.effort,
                elapsed_s=round(time.time() - t0, 3),
                tool_calls=tool_calls,
                tool_events=tool_events,
                analyses=_current_analyses(session),
                parsed_final=parsed,
                raw_reply_tail=(reply or "")[-2000:],
            )
    except Exception as exc:
        return EvaluationRunResult(
            image_path=image,
            protocol_id=protocol.task_id,
            backend=cfg.backend,
            model=cfg.model,
            effort=cfg.effort,
            elapsed_s=round(time.time() - t0, 3),
            tool_calls=tool_calls,
            tool_events=tool_events,
            analyses=_current_analyses(session),
            parsed_final={},
            error=f"{type(exc).__name__}: {exc}",
        )
