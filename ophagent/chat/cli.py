"""Interactive command-line interface for the full OphAgent workflow.

Usage:
    python -m ophagent.chat.cli
    python demos/chat.py
"""

from __future__ import annotations

import argparse
import getpass
import json
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any

from .api_config import (
    API_CHANNELS,
    DEFAULT_WEB_MODELS,
    PROVIDER_SPECS,
    provider_channel,
    providers_for_channel,
    resolve_provider_connection,
)
from .oph_session import OphContext, OphSession
from .prompt_profiles import SUPPORTED_PROFILES
from ..utils.paths import output_path
from ..webchat import models_catalog


EFFORT_LEVELS = ("low", "medium", "high", "max", "ultra")
DEFAULT_CLI_BACKEND = "openrouter"
DEFAULT_CLI_MODEL = "openai/gpt-5.5-pro"

HELP = """\
slash commands:
  /open <path>       register an ophthalmic image (CFP/OCT/UWF/FFA)
  /volume <path>     register an OCT volume (DICOM/NIfTI/NPY/folder)
  /save [path]       persist session JSON
  /load <path>       reload a saved OphSession or legacy CLI session
  /history           print message history
  /context           print session context
  /config            show the active runtime configuration
  /clear             wipe history (preserve image and analysis context)
  /reset             start a fresh session with the same runtime configuration
  /model <name>      switch the Planner model
  /effort <level>    set low/medium/high/max/ultra execution policy
  /tokens <n>        set max_tokens
  /help              show this list
  /quit | /exit      leave"""


def _prompt_menu(
    title: str,
    options: list[tuple[str, str, str]],
    *,
    default_index: int = 0,
    input_fn=None,
) -> str:
    """Prompt for one item without adding a CLI UI dependency."""
    if not options:
        raise ValueError(f"no options available for {title.lower()}")
    input_fn = input_fn or input
    print(f"\n{title}")
    for index, (_, label, detail) in enumerate(options, start=1):
        suffix = f" — {detail}" if detail else ""
        marker = " (default)" if index - 1 == default_index else ""
        print(f"  {index}. {label}{marker}{suffix}")
    while True:
        raw = input_fn(f"Select [default {default_index + 1}]: ").strip()
        if not raw:
            return options[default_index][0]
        try:
            selected = int(raw) - 1
        except ValueError:
            selected = -1
        if 0 <= selected < len(options):
            return options[selected][0]
        print(f"  Enter a number from 1 to {len(options)}.")


def _resolve_cli_selection(
    channel: str | None,
    backend: str | None,
    model: str | None,
) -> tuple[str, str, str]:
    """Resolve optional hierarchical flags while preserving legacy defaults."""
    if channel and channel not in API_CHANNELS:
        raise ValueError(f"unknown API channel: {channel}")

    if backend is None:
        if channel:
            preferred = API_CHANNELS[channel].get("default_provider", "")
            available = providers_for_channel(channel)
            backend = preferred if preferred in available else available[0]
        else:
            backend = DEFAULT_CLI_BACKEND
    if backend not in PROVIDER_SPECS:
        raise ValueError(f"unknown API provider: {backend}")

    resolved_channel = provider_channel(backend)
    if channel and resolved_channel != channel:
        raise ValueError(
            f"provider '{backend}' belongs to channel '{resolved_channel}', "
            f"not '{channel}'"
        )

    if model is None:
        if not channel and backend == DEFAULT_CLI_BACKEND:
            model = DEFAULT_CLI_MODEL
        else:
            model = (
                DEFAULT_WEB_MODELS.get(backend)
                or models_catalog.default_model(backend)
            )
    if not model:
        raise ValueError(f"no default model is available for provider '{backend}'")
    return resolved_channel, backend, model


def _interactive_provider_selection(input_fn=None) -> tuple[str, str, str]:
    providers = [
        provider
        for channel in API_CHANNELS
        for provider in providers_for_channel(channel)
        if provider in models_catalog.list_providers()
    ]
    provider_options = [
        (
            provider,
            str(PROVIDER_SPECS[provider]["label"]),
            (
                f"{API_CHANNELS[provider_channel(provider)]['label']} · "
                f"{PROVIDER_SPECS[provider]['api_key_env']}"
            ),
        )
        for provider in providers
    ]
    preferred_provider = str(API_CHANNELS["official"]["default_provider"])
    provider_default = providers.index(preferred_provider)
    backend = _prompt_menu(
        "1. Provider",
        provider_options,
        default_index=provider_default,
        input_fn=input_fn,
    )
    channel = provider_channel(backend)

    models = [
        model
        for model in models_catalog.list_models(backend)
        if model.get("tools", True)
    ]
    preferred_model = DEFAULT_WEB_MODELS.get(backend, "")
    model_options = [
        (
            str(model["id"]),
            str(model["name"]),
            str(model.get("note", "")),
        )
        for model in models
    ]
    model_ids = [item[0] for item in model_options]
    model_default = (
        model_ids.index(preferred_model)
        if preferred_model in model_ids
        else 0
    )
    model = _prompt_menu(
        "2. Model",
        model_options,
        default_index=model_default,
        input_fn=input_fn,
    )
    return channel, backend, model


def _interactive_api_credentials(provider: str) -> dict[str, dict[str, str]]:
    """Collect a missing key for this process only; never persist it."""
    resolved = resolve_provider_connection(provider)
    if resolved["api_key"]:
        print(
            f"\nUsing {PROVIDER_SPECS[provider]['api_key_env']} from the "
            "current environment."
        )
        return {}
    env_name = str(PROVIDER_SPECS[provider]["api_key_env"])
    key = getpass.getpass(
        f"\n{env_name} (hidden; used for this run only): "
    ).strip()
    if not key:
        raise ValueError(f"{env_name} is required for provider '{provider}'")
    return {provider: {"api_key": key}}


def _invalidate_clients(session: OphSession) -> None:
    """Discard provider clients after a runtime model assignment changes."""
    for attr in (
        "_client",
        "_vision_client_obj",
        "_executor_client_obj",
        "_verifier_client_obj",
        "_debate_client_obj",
    ):
        if hasattr(session, attr):
            setattr(session, attr, None)
    session._vision_resolved = None


def _session_configuration(session: OphSession) -> dict[str, Any]:
    return {
        "backend": session.backend,
        "model": session.model,
        "max_tokens": session.max_tokens,
        "effort": session.effort,
        "prompt_profile": session.prompt_profile,
        "vision_backend": session.vision_backend,
        "vision_model_override": session.vision_model_override,
        "executor_backend": session.executor_backend,
        "executor_model": session.executor_model,
        "executor_repair_enabled": session.executor_repair_enabled,
        "verifier_backend": session.verifier_backend,
        "verifier_model": session.verifier_model,
        "debate_backend": session.debate_backend,
        "debate_model": session.debate_model,
        "temperature": session.temperature,
        "workspace": session.workspace,
    }


def _load_session(path: str | Path) -> OphSession:
    """Load OphSession JSON and migrate the former OCT-only CLI schema."""
    source = Path(path).expanduser()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("session JSON must contain an object")

    context_data = data.pop("context", {})
    if not isinstance(context_data, dict):
        raise ValueError("session context must contain an object")

    session_fields = {
        item.name for item in fields(OphSession) if not item.name.startswith("_")
    }
    context_fields = {item.name for item in fields(OphContext)}

    legacy_caption_model = data.pop("caption_model", None)
    session_data = {
        key: value for key, value in data.items() if key in session_fields
    }
    if not session_data.get("session_id"):
        raise ValueError("session JSON does not contain a session_id")

    if legacy_caption_model and not session_data.get("vision_model_override"):
        session_data["vision_backend"] = session_data.get("backend", "openrouter")
        session_data["vision_model_override"] = legacy_caption_model

    migrated_context = {
        key: value for key, value in context_data.items() if key in context_fields
    }
    session = OphSession(**session_data)
    session.context = OphContext(**migrated_context)
    return session


def _print_header(session: OphSession) -> None:
    print("\n" + "=" * 68)
    print("  OphAgent - multimodal ophthalmology analysis assistant")
    print(
        f"  session={session.session_id}  backend={session.backend}  "
        f"model={session.model}  effort={session.effort}"
    )
    print("  type /help for commands, /quit to exit")
    print("=" * 68 + "\n")


def _message_preview(content: Any, limit: int = 200) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)
    return text[:limit].replace("\n", " ")


def _handle_slash(session: OphSession, line: str) -> OphSession | None:
    """Handle one slash command and return the active session."""
    parts = line.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return None

    if cmd == "/help":
        print(HELP)
        return session

    if cmd == "/open":
        if not arg:
            print("usage: /open <path>")
            return session
        try:
            session.set_image(arg)
            print(
                "  registered image: "
                f"{session.context.current_image} "
                f"(modality={session.context.current_modality or 'unresolved'})"
            )
        except Exception as exc:
            print(f"  image registration failed: {exc}")
        return session

    if cmd == "/volume":
        if not arg:
            print("usage: /volume <path>")
            return session
        try:
            volume = Path(arg).expanduser()
            if not volume.exists():
                raise FileNotFoundError(volume)
            session.set_volume(str(volume))
            print(f"  registered OCT volume: {session.context.current_volume}")
        except Exception as exc:
            print(f"  volume registration failed: {exc}")
        return session

    if cmd == "/save":
        path = session.save(arg) if arg else session.save()
        print(f"  saved to {path}")
        return session

    if cmd == "/load":
        if not arg:
            print("usage: /load <path>")
            return session
        try:
            loaded = _load_session(arg)
        except Exception as exc:
            print(f"  session load failed: {exc}")
            return session
        print(
            f"  loaded session {loaded.session_id} "
            f"with {len(loaded.messages)} messages"
        )
        return loaded

    if cmd == "/history":
        if not session.messages:
            print("  (empty)")
        for index, message in enumerate(session.messages):
            role = str(message.get("role", "?"))
            preview = _message_preview(message.get("content") or "")
            if role == "tool":
                print(
                    f"  [{index:02d}] tool:"
                    f"{str(message.get('name', '?')):20s} -> {preview}"
                )
            else:
                tool_calls = message.get("tool_calls") or []
                note = f"  +{len(tool_calls)} tool_calls" if tool_calls else ""
                print(f"  [{index:02d}] {role:9s} {preview}{note}")
        return session

    if cmd == "/context":
        print(session._context_note() or "  (empty)")
        return session

    if cmd == "/config":
        for key, value in _session_configuration(session).items():
            print(f"  {key}: {value}")
        return session

    if cmd == "/clear":
        session.messages.clear()
        print("  message history cleared (image and analysis context preserved)")
        return session

    if cmd == "/reset":
        new_session = OphSession.new(**_session_configuration(session))
        new_session._api_credentials = dict(session._api_credentials)
        print(f"  new session {new_session.session_id}")
        return new_session

    if cmd == "/model":
        if not arg:
            print(f"  current model: {session.model}")
            return session
        session.model = arg
        _invalidate_clients(session)
        print(f"  model set to {arg}")
        return session

    if cmd == "/effort":
        if not arg:
            print(f"  current effort: {session.effort}")
            return session
        effort = arg.lower()
        if effort not in EFFORT_LEVELS:
            print("usage: /effort low|medium|high|max|ultra")
            return session
        session.effort = effort
        print(f"  effort set to {effort}")
        return session

    if cmd == "/tokens":
        try:
            max_tokens = int(arg)
            if max_tokens <= 0:
                raise ValueError
            session.max_tokens = max_tokens
            print(f"  max_tokens={session.max_tokens}")
        except ValueError:
            print("usage: /tokens <positive integer>")
        return session

    print(f"  unknown command: {cmd} (try /help)")
    return session


def repl(session: OphSession) -> None:
    _print_header(session)

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            new_session = _handle_slash(session, line)
            if new_session is None:
                break
            session = new_session
            continue

        try:
            reply = session.chat(line)
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        print(f"\nassistant> {reply}\n")

    try:
        path = session.save()
        print(f"(session autosaved to {path})")
    except Exception as exc:
        print(f"(autosave failed: {exc})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the full multimodal OphAgent workflow."
    )
    parser.add_argument("--load", help="Load a saved session JSON and resume")
    parser.add_argument(
        "--channel",
        choices=tuple(API_CHANNELS),
        help="API channel: gateway or official",
    )
    parser.add_argument(
        "--backend", "--provider",
        dest="backend",
        choices=sorted(PROVIDER_SPECS),
        help="Provider within the selected API channel",
    )
    parser.add_argument(
        "--configure-provider",
        action="store_true",
        help="Interactively choose a provider, model, and a missing API key",
    )
    parser.add_argument("--model", help="Provider-specific model id")
    parser.add_argument("--workspace", default=str(output_path("oph_sessions")))
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--effort", choices=EFFORT_LEVELS, default="low")
    parser.add_argument(
        "--prompt-profile",
        choices=SUPPORTED_PROFILES,
        default="standard",
    )
    parser.add_argument("--vision-backend", choices=sorted(PROVIDER_SPECS))
    parser.add_argument("--vision-model")
    parser.add_argument("--executor-backend", choices=sorted(PROVIDER_SPECS))
    parser.add_argument("--executor-model")
    parser.add_argument(
        "--disable-executor-repair",
        action="store_true",
        help="Disable the schema-constrained Executor LLM repair role.",
    )
    parser.add_argument("--verifier-backend", choices=sorted(PROVIDER_SPECS))
    parser.add_argument("--verifier-model")
    parser.add_argument("--debate-backend", choices=sorted(PROVIDER_SPECS))
    parser.add_argument("--debate-model")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Decoding temperature; omit to use the configured OphSession default.",
    )
    parser.add_argument(
        "--caption-model",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.load:
        session = _load_session(args.load)
    else:
        try:
            if args.configure_provider:
                _, backend, model = _interactive_provider_selection()
                api_credentials = _interactive_api_credentials(backend)
            else:
                _, backend, model = _resolve_cli_selection(
                    args.channel,
                    args.backend,
                    args.model,
                )
                api_credentials = {}
        except ValueError as exc:
            parser.error(str(exc))

        kwargs: dict[str, Any] = {
            "backend": backend,
            "model": model,
            "workspace": args.workspace,
            "max_tokens": args.max_tokens,
            "effort": args.effort,
            "prompt_profile": args.prompt_profile,
            "vision_backend": args.vision_backend,
            "vision_model_override": args.vision_model or args.caption_model,
            "executor_backend": args.executor_backend,
            "executor_model": args.executor_model,
            "executor_repair_enabled": not args.disable_executor_repair,
            "verifier_backend": args.verifier_backend,
            "verifier_model": args.verifier_model,
            "debate_backend": args.debate_backend,
            "debate_model": args.debate_model,
            "_api_credentials": api_credentials,
        }
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        session = OphSession.new(**kwargs)
    repl(session)


if __name__ == "__main__":
    main()
