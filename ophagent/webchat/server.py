"""
FastAPI server for the OphAgent Web UI.

Routes:
  GET  /                        → index.html (the chat UI)
  POST /api/sessions            → create new session, returns id
  GET  /api/sessions            → list saved sessions
  GET  /api/sessions/{id}       → metadata + messages for a session
  POST /api/sessions/{id}/chat  → send user text, returns assistant reply
  POST /api/sessions/{id}/upload → upload image/volume, registered as current
  DELETE /api/sessions/{id}     → delete a saved session
  GET  /files/...               → serve uploaded files & generated reports
"""

from __future__ import annotations

import os
import json
import logging
import re
import hashlib
import ipaddress
import shutil
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request as UrlRequest, build_opener

import asyncio
import queue as _queue
import threading

# Load API keys before importing the chat modules, because some adapters create
# clients or resolve paths at module import time. Deployments should keep this
# file under OPHAGENT_RUNTIME_DIR (or set OPHAGENT_ENV_FILE explicitly); a
# checkout-local .env remains a backwards-compatible development fallback.
_PROJ_ROOT_FOR_ENV = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv as _load_dotenv
    _explicit_env = os.environ.get("OPHAGENT_ENV_FILE", "").strip()
    _runtime_root = os.environ.get("OPHAGENT_RUNTIME_DIR", "").strip()
    _env_candidates = []
    if _explicit_env:
        _env_candidates.append(Path(_explicit_env).expanduser())
    if _runtime_root:
        _env_candidates.append(Path(_runtime_root).expanduser() / ".env")
        _env_candidates.append(Path(_runtime_root).expanduser() / ".env.local")
    _env_candidates.append(_PROJ_ROOT_FOR_ENV / ".env")
    _dotenv_paths = []
    for _candidate in _env_candidates:
        if _candidate.is_file():
            _resolved = _candidate.resolve()
            if _resolved not in _dotenv_paths:
                _dotenv_paths.append(_resolved)
    if _dotenv_paths:
        for _dotenv_path in _dotenv_paths:
            _load_dotenv(dotenv_path=_dotenv_path, override=False)
        print(f"[server] loaded env from {', '.join(map(str, _dotenv_paths))}")
    else:
        checked = ", ".join(str(p) for p in _env_candidates)
        print(f"[server] no .env found ({checked}); keys must come from shell env")
except ImportError:
    print("[server] python-dotenv not installed; keys must come from shell env")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets as _secrets

from ..checkpoint_config import (
    apply_saved_checkpoint_environment,
    check_checkpoint_group,
    checkpoint_group_view,
    checkpoint_restart_required,
    checkpoint_settings_view,
    update_checkpoint_group,
)

# Path overrides must be applied before OphSession imports OphToolKit and the
# adapter modules resolve their checkpoint constants.
apply_saved_checkpoint_environment()

from ..utils.paths import OUTPUT_DIR, RELEASE_ROOT, runtime_path
from ..chat.session import ChatSession
from ..chat.oph_session import OphSession
from ..chat.api_config import (
    DEFAULT_WEB_MODELS,
    PROVIDER_SPECS,
    list_api_channels,
    resolve_provider_connection,
)
from ..chat.run_policy import get_effort_policy
from . import models_catalog

log = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = RELEASE_ROOT
WEB_ROOT = Path(__file__).parent / "static"
FILE_ROOT = OUTPUT_DIR
WORKSPACE = FILE_ROOT / "webchat_sessions"
SESSIONS_DIR = WORKSPACE / "_sessions"
UPLOADS_DIR = WORKSPACE / "_uploads"
API_CREDENTIALS_DIR = runtime_path("config", "web_api_credentials")
WORKSPACE.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
API_CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
_API_CREDENTIALS_LOCK = threading.RLock()
_SID_RE = re.compile(r"^[0-9a-f]{12}$")
_LEGACY_OUTPUT_ROOTS = tuple(
    Path(value).expanduser().resolve()
    for value in os.environ.get("OPHAGENT_LEGACY_OUTPUT_ROOTS", "").split(os.pathsep)
    if value.strip()
)


# ── Security: Cloudflare Access JWT (primary) + Basic Auth (fallback) ────
# When the server sits behind Cloudflare Access (Tunnel deployment), every
# request comes with a signed JWT in `Cf-Access-Jwt-Assertion`. The JWT's
# `email` claim is the identity authenticated by Cloudflare. We trust it
# only after RS256 signature verification against the team's JWKS.
#
# Required env (set in .env for the public deployment):
#   CF_ACCESS_TEAM_DOMAIN  e.g. your-team.cloudflareaccess.com
#   CF_ACCESS_AUD          the Application Audience tag from Access settings
#
# Fallback (when JWT absent OR misconfigured): Basic Auth via
# WEB_USERNAME / WEB_PASSWORD. Lets you log in directly from localhost
# without a Cloudflare round-trip during development.
import jwt as _jwt
from jwt import PyJWKClient as _PyJWKClient
from fastapi import Header as _Header

_WEB_USERNAME = os.environ.get("WEB_USERNAME", "")
_WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "")
_BASIC_AUTH_ENABLED = bool(_WEB_USERNAME and _WEB_PASSWORD)

_CF_TEAM_DOMAIN = os.environ.get("CF_ACCESS_TEAM_DOMAIN", "").strip().rstrip("/")
_CF_ACCESS_AUD  = os.environ.get("CF_ACCESS_AUD", "").strip()
_CF_ACCESS_ENABLED = bool(_CF_TEAM_DOMAIN and _CF_ACCESS_AUD)
_AUTH_ENABLED = _CF_ACCESS_ENABLED or _BASIC_AUTH_ENABLED

# Cache the JWKS client so we don't fetch keys on every request.
_jwks_client: _PyJWKClient | None = None
if _CF_ACCESS_ENABLED:
    _certs_url = f"https://{_CF_TEAM_DOMAIN}/cdn-cgi/access/certs"
    _jwks_client = _PyJWKClient(_certs_url, lifespan=3600, cache_keys=True)

_basic_auth = HTTPBasic(auto_error=False)
_LOCAL_FILE_BYPASS = object()


def _request_is_loopback(request: Request) -> bool:
    host = ((request.client.host if request.client else "") or "").strip().lower()
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    # IPv4-mapped IPv6 loopback, used by some local ASGI/proxy stacks.
    return host == "::ffff:127.0.0.1"


def _request_has_proxy_headers(request: Request) -> bool:
    """Return true when a loopback connection was forwarded by a proxy."""
    return any(
        (request.headers.get(name) or "").strip()
        for name in (
            "Cf-Connecting-Ip",
            "Cf-Ray",
            "X-Forwarded-For",
            "Forwarded",
        )
    )


def _verify_cf_access(token: str) -> str | None:
    """Verify a Cloudflare Access JWT and return the email claim. None on
    any failure (signature, audience, expiry, missing email)."""
    if not _jwks_client or not token:
        return None
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token).key
        payload = _jwt.decode(
            token, signing_key,
            algorithms=["RS256"],
            audience=_CF_ACCESS_AUD,
            issuer=f"https://{_CF_TEAM_DOMAIN}",
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except Exception as e:
        log.debug(f"[auth] CF Access JWT verification failed: {e}")
        return None
    email = payload.get("email") or payload.get("identity_nonce")
    return email or None


def require_auth(
    request: Request,
    cf_jwt: str | None = _Header(default=None, alias="Cf-Access-Jwt-Assertion"),
    creds: HTTPBasicCredentials | None = Depends(_basic_auth),
):
    """Auth dependency with two acceptable paths:
      1. Valid Cloudflare Access JWT (preferred — covers collaborators).
      2. Valid Basic Auth (fallback for localhost / config recovery).
    Returns the authenticated email (Access) or username (Basic Auth).
    Raises 401 if neither passes."""
    if not _AUTH_ENABLED:
        if _request_is_loopback(request) and not _request_has_proxy_headers(request):
            return None
        raise HTTPException(
            status_code=401,
            detail="remote access requires authentication",
        )
    # Try CF Access first
    if _CF_ACCESS_ENABLED and cf_jwt:
        email = _verify_cf_access(cf_jwt)
        if email:
            return email
    # Fall back to Basic Auth
    if _BASIC_AUTH_ENABLED and creds is not None:
        ok_user = _secrets.compare_digest(creds.username, _WEB_USERNAME)
        ok_pass = _secrets.compare_digest(creds.password, _WEB_PASSWORD)
        if ok_user and ok_pass:
            return creds.username
    # Nothing worked → 401
    raise HTTPException(
        status_code=401, detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )


# Generated web assets are stored under ``webchat_sessions/<sid>`` and are
# owner-gated below. Global adapter output directories must not be public in a
# multi-user deployment because they can contain source-image derivatives.
_PUBLIC_FILE_WHITELIST: tuple[Path, ...] = ()


def _validate_sid(sid: str) -> str:
    if not _SID_RE.fullmatch(str(sid or "")):
        raise HTTPException(status_code=404, detail="session not found")
    return sid


def _resolve_project_file(rel_path: str) -> tuple[Path, Path] | None:
    """Resolve ``rel_path`` against the generated-output root.

    Source files, credentials, model weights, and external repositories are
    outside this tree and can never be served through ``/files``.
    """
    candidate = (FILE_ROOT / rel_path).resolve()
    file_root_resolved = FILE_ROOT.resolve()
    # Reject anything that escapes the output root (path traversal).
    try:
        rel_to_root = candidate.relative_to(file_root_resolved)
    except ValueError:
        return None
    # Reject dotfiles at any depth (catches `.env`, `.git/`, etc.).
    for part in rel_to_root.parts:
        if part.startswith("."):
            return None
    return (candidate, rel_to_root)


def _resolve_whitelisted(rel_path: str, *, user: Any = None) -> Path | None:
    resolved = _resolve_project_file(rel_path)
    if resolved is None:
        return None
    candidate, rel_to_root = resolved
    if not candidate.is_file():
        return None

    # Owner-gated webchat assets:
    #   webchat_sessions/_uploads/<sid>/...
    #   webchat_sessions/<sid>/...
    # Explicitly deny state/private dirs such as _sessions and _prefs.
    try:
        rel_to_workspace = candidate.relative_to(WORKSPACE.resolve())
    except ValueError:
        rel_to_workspace = None
    if rel_to_workspace is not None:
        parts = rel_to_workspace.parts
        if not parts:
            return None
        if parts[0] in {"_sessions", "_prefs"}:
            return None
        if parts[0] == "_uploads":
            if len(parts) < 3:
                return None
            sid = _validate_sid(parts[1])
        else:
            sid = _validate_sid(parts[0])
        if user is not _LOCAL_FILE_BYPASS:
            manager.get(sid, user=user)  # enforces owner/admin access
        return candidate

    # Public, non-session report directories.
    for allowed in _PUBLIC_FILE_WHITELIST:
        try:
            rel_to_root.relative_to(allowed)
            return candidate
        except ValueError:
            continue
    return None


# Rate limit — 30 requests / minute per IP across the whole API. Adjust via env.
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

_RATE_LIMIT = os.environ.get("WEB_RATE_LIMIT", "30/minute")


def _rate_limit_key(request: Request) -> str:
    # A Cloudflare tunnel terminates locally, so get_remote_address() would
    # collapse every collaborator into 127.0.0.1 and one user could throttle
    # everyone. Trust Cloudflare's client address only when Access is enabled.
    if _CF_ACCESS_ENABLED:
        address = (request.headers.get("Cf-Connecting-Ip") or "").strip()
        if address:
            return address
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key, default_limits=[_RATE_LIMIT])


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum or value > maximum:
        log.warning(
            "%s=%d is outside [%d, %d]; using %d",
            name, value, minimum, maximum, default,
        )
        return default
    return value


DEFAULT_BACKEND = os.environ.get("OPH_WEB_BACKEND", "openrouter")
# Note: model id format differs per backend:
#   - openrouter: "openai/gpt-5.5-pro"   (vendor-prefixed)
#   - aigcbest:   "gpt-5.5-pro"          (plain model id)
DEFAULT_MODEL = os.environ.get(
    "OPH_WEB_MODEL",
    DEFAULT_WEB_MODELS.get(DEFAULT_BACKEND, ""),
)
DEFAULT_CAPTION = os.environ.get("OPH_WEB_CAPTION_MODEL", "openai/gpt-5.4")
# Reasoning models (gpt-5.x-pro) need lots of headroom for internal CoT, but
# this is also the upper-bound that triggers OpenRouter 402 when credits run
# low. The safe_completion path auto-backs off on 402.
DEFAULT_MAX_TOKENS = _bounded_env_int("OPH_WEB_MAX_TOKENS", 6000, 256, 100_000)
DEFAULT_EFFORT = os.environ.get("OPH_WEB_EFFORT", "low")


# ── Per-user preferences (remember last backend / model / effort) ────────────
# A new chat should open with the model + effort the user last chose, not the
# global default. Stored as one small JSON per user under _prefs/.
_PREFS_DIR = WORKSPACE / "_prefs"
_PREFS_DIR.mkdir(parents=True, exist_ok=True)
_PREFS_LOCK = threading.RLock()


def _pref_path(user: str | None) -> Path:
    value = user or "_local"
    safe = "".join(c if c.isalnum() or c in "_.@-" else "_" for c in value)
    safe = (safe or "_local")[:48]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return _PREFS_DIR / f"{safe}-{digest}.json"


def _legacy_pref_path(user: str | None) -> Path:
    value = user or "_local"
    safe = "".join(c if c.isalnum() or c in "_.@-" else "_" for c in value)
    return _PREFS_DIR / f"{safe or '_local'}.json"


def load_user_pref(user: str | None) -> dict:
    """Return {backend, model, effort} for this user, or {} if none/invalid.
    Validates against the live catalog so a removed model can't stick."""
    p = _pref_path(user)
    if not p.exists():
        p = _legacy_pref_path(user)
    with _PREFS_LOCK:
        if not p.exists():
            return {}
        try:
            pref = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    backend = pref.get("backend")
    model = pref.get("model")
    if backend and backend not in models_catalog.list_providers():
        return {}   # provider gone — fall back to defaults
    if backend and model:
        valid = {m["id"] for m in models_catalog.list_models(backend)}
        if valid and model not in valid:
            # keep the backend but let it pick that provider's default model
            pref["model"] = models_catalog.default_model(backend)
    if pref.get("effort") not in {"low", "medium", "high", "max", "ultra"}:
        pref["effort"] = DEFAULT_EFFORT
    return pref


def save_user_pref(user: str | None, backend: str, model: str, effort: str) -> None:
    try:
        path = _pref_path(user)
        with _PREFS_LOCK:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"backend": backend, "model": model, "effort": effort}),
                encoding="utf-8",
            )
            tmp.replace(path)
    except Exception:
        pass


# ── Per-user API credentials ────────────────────────────────────────────────
def _safe_user_token(user: str | None) -> str:
    value = user or "_local"
    safe = "".join(c if c.isalnum() or c in "_.@-" else "_" for c in value)
    safe = (safe or "_local")[:48]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}"


def _api_credentials_path(user: str | None) -> Path:
    return API_CREDENTIALS_DIR / f"{_safe_user_token(user)}.json"


def load_user_api_credentials(user: str | None) -> dict[str, dict[str, str]]:
    path = _api_credentials_path(user)
    with _API_CREDENTIALS_LOCK:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for provider, values in raw.items():
        if provider not in PROVIDER_SPECS or not isinstance(values, dict):
            continue
        entry: dict[str, str] = {}
        key = str(values.get("api_key") or "").strip()
        base_url = str(values.get("base_url") or "").strip().rstrip("/")
        if key:
            entry["api_key"] = key[:8192]
        if base_url:
            try:
                entry["base_url"] = _validate_api_base_url(
                    base_url[:2048],
                    allow_private=(not _AUTH_ENABLED or _is_admin(user)),
                    resolve_host=bool(_AUTH_ENABLED and not _is_admin(user)),
                )
            except HTTPException:
                pass
        if entry:
            cleaned[provider] = entry
    return cleaned


def save_user_api_credentials(
    user: str | None,
    credentials: dict[str, dict[str, str]],
) -> None:
    path = _api_credentials_path(user)
    with _API_CREDENTIALS_LOCK:
        if not credentials:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(credentials), encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(path)


def _api_host_is_private(
    hostname: str,
    port: int | None,
    *,
    resolve_host: bool,
) -> bool | None:
    hostname = hostname.strip().strip("[]").lower()
    if hostname == "localhost":
        return True
    addresses: set[str] = set()
    try:
        addresses.add(str(ipaddress.ip_address(hostname)))
    except ValueError:
        if not resolve_host:
            return False
        try:
            for item in socket.getaddrinfo(
                hostname,
                port or 443,
                type=socket.SOCK_STREAM,
            ):
                addresses.add(item[4][0].split("%", 1)[0])
        except OSError:
            return None
    if resolve_host and not addresses:
        return None
    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
                return True
        except ValueError:
            continue
    return False


def _validate_api_base_url(
    value: str,
    *,
    allow_private: bool = False,
    resolve_host: bool = True,
) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""
    if len(value) > 2048:
        raise HTTPException(400, "base URL is too long")
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HTTPException(400, "base URL must not contain credentials, query, or fragment")
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise HTTPException(400, "base URL must be an absolute HTTP(S) URL")
    try:
        port = parsed.port
    except ValueError:
        raise HTTPException(400, "base URL contains an invalid port")
    private_target = _api_host_is_private(
        parsed.hostname,
        port,
        resolve_host=resolve_host,
    )
    if private_target is None and not allow_private:
        raise HTTPException(400, "base URL hostname could not be resolved safely")
    if private_target and not allow_private:
        raise HTTPException(
            400,
            "private or loopback API endpoints require an administrator",
        )
    if parsed.scheme == "http" and not private_target:
        raise HTTPException(400, "non-loopback API endpoints must use HTTPS")
    return value


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep credential checks on the already-validated endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _api_provider_status(user: str | None, provider: str) -> dict[str, Any]:
    credentials = load_user_api_credentials(user)
    resolved = resolve_provider_connection(provider, credentials.get(provider))
    return {
        "id": provider,
        "label": resolved["label"],
        "channel": resolved["channel"],
        "configured": bool(resolved["api_key"]),
        "source": resolved["source"],
        "has_personal_key": resolved["has_personal_key"],
        "base_url": resolved["base_url"] or "",
        "has_custom_base_url": resolved["has_custom_base_url"],
    }


def _invalidate_session_clients(session: OphSession) -> None:
    for attr in (
        "_client", "_vision_client_obj", "_executor_client_obj",
        "_verifier_client_obj",
        "_debate_client_obj",
    ):
        if hasattr(session, attr):
            setattr(session, attr, None)
    session._vision_resolved = None


def _session_runtime_config(session: OphSession) -> dict[str, Any]:
    """Expose effective role assignments without implying component readiness."""
    resolved = session._resolved_run_policy(get_effort_policy(session.effort))
    components = resolved.pop("components")
    return {
        "status": "configured",
        "components": components,
        "policy": resolved,
        "last_run_policy": session.context.last_run_policy,
    }


# ── Session manager ─────────────────────────────────────────────────────────
# Admins can manage shared runtime settings. Session contents remain private to
# their recorded owner; administrative status never grants conversation access.
# Set WEB_ADMIN_EMAILS in .env (comma-separated) to elevate specific CF Access
# emails; the Basic-Auth username is always a runtime administrator too.
_ADMIN_EMAILS = {
    e.strip().lower() for e in os.environ.get("WEB_ADMIN_EMAILS", "").split(",")
    if e.strip()
}


def _is_admin(user: str | None) -> bool:
    if not user:
        return False
    if _WEB_USERNAME and user == _WEB_USERNAME:
        return True
    return user.lower() in _ADMIN_EMAILS


def _can_access_session(user: str | None, owner: str | None) -> bool:
    """Authorisation rule for an individual session record."""
    if user is None:           # auth disabled entirely (local trusted)
        return True
    if owner is None:          # legacy unowned records stay hidden remotely
        return False
    return _secrets.compare_digest(owner, user)


class SessionManager:
    """Keeps live ChatSession objects in memory keyed by id, persists to disk."""

    def __init__(self):
        self._live: dict[str, ChatSession] = {}
        self._run_locks: dict[str, threading.Lock] = {}
        self._state_lock = threading.RLock()

    def create(self, owner: str | None = None) -> OphSession:
        # Open a new chat with the user's LAST-used backend/model/effort so it
        # doesn't reset to the global default every time. Falls back to the
        # defaults when the user has no saved preference yet.
        pref = load_user_pref(owner)
        s = OphSession.new(
            backend=pref.get("backend") or DEFAULT_BACKEND,
            model=pref.get("model") or DEFAULT_MODEL,
            effort=pref.get("effort") or DEFAULT_EFFORT,
            workspace=str(WORKSPACE),
            max_tokens=DEFAULT_MAX_TOKENS,
            owner=owner,
        )
        s._api_credentials = load_user_api_credentials(owner)
        with self._state_lock:
            self._live[s.session_id] = s
        # don't persist to disk yet — only after first real interaction
        return s

    @staticmethod
    def _relocate_loaded_session(s: OphSession) -> OphSession:
        """Map persisted paths from an older output root into ``FILE_ROOT``.

        Remapping is enabled only through ``OPHAGENT_LEGACY_OUTPUT_ROOTS`` and
        only when the corresponding destination exists. This keeps migration
        support portable without embedding workstation paths in public code.
        """
        if not _LEGACY_OUTPUT_ROOTS:
            s.workspace = str(WORKSPACE)
            return s

        def remap(value):
            if isinstance(value, str):
                try:
                    source = Path(value).expanduser().resolve()
                except (OSError, ValueError):
                    return value
                for old_root in _LEGACY_OUTPUT_ROOTS:
                    try:
                        relative = source.relative_to(old_root)
                    except ValueError:
                        continue
                    candidate = (FILE_ROOT / relative).resolve()
                    return str(candidate) if candidate.exists() else value
                return value
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                return {remap(key): remap(item) for key, item in value.items()}
            return value

        ctx = s.context
        ctx.current_image = remap(ctx.current_image)
        ctx.current_volume = remap(ctx.current_volume)
        ctx.attached_images = remap(ctx.attached_images)
        ctx.analyses = remap(ctx.analyses)
        ctx.last_report = remap(ctx.last_report)
        for message in s.messages:
            if (message.get("role") != "tool"
                    or not isinstance(message.get("content"), str)):
                continue
            try:
                payload = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            message["content"] = json.dumps(
                remap(payload), default=str, ensure_ascii=False)
        s.workspace = str(WORKSPACE)
        return s

    def get(self, sid: str, *, user: str | None = None) -> OphSession:
        sid = _validate_sid(sid)
        if sid in self._live:
            s = self._live[sid]
        else:
            path = SESSIONS_DIR / f"{sid}.json"
            if not path.exists():
                raise HTTPException(404, f"session {sid} not found")
            s = self._relocate_loaded_session(OphSession.load(path))
            self._live[sid] = s
        # Authorisation gate
        if not _can_access_session(user, s.owner):
            raise HTTPException(403, "you do not own this session")
        credential_user = s.owner if s.owner is not None else user
        credentials = load_user_api_credentials(credential_user)
        if s._api_credentials != credentials:
            s._api_credentials = credentials
            _invalidate_session_clients(s)
        return s

    def refresh_api_credentials(self, user: str | None) -> None:
        credentials = load_user_api_credentials(user)
        for session in self._live.values():
            if session.owner == user or (session.owner is None and user is None):
                session._api_credentials = credentials
                _invalidate_session_clients(session)

    def has_active_runs(self) -> bool:
        with self._state_lock:
            return any(lock.locked() for lock in self._run_locks.values())

    @contextmanager
    def configuration_guard(self):
        """Block new analyses while a global tool policy is being changed."""
        with self._state_lock:
            if any(lock.locked() for lock in self._run_locks.values()):
                raise HTTPException(
                    409,
                    "wait for active analyses to finish before changing tools",
                )
            yield

    def refresh_checkpoint_config(self) -> None:
        """Drop cached toolkits/models after an enablement policy change."""
        try:
            from ..adapters import GLOBAL_REGISTRY

            GLOBAL_REGISTRY.unload_all()
        except Exception:
            log.exception("failed to unload adapters after checkpoint update")
        try:
            from ..chat import oph_tools

            oph_tools._MODALITY_MODEL = None
        except Exception:
            pass
        for session in self._live.values():
            session._toolkit = None

    def try_acquire_run(self, sid: str) -> bool:
        sid = _validate_sid(sid)
        with self._state_lock:
            lock = self._run_locks.setdefault(sid, threading.Lock())
            return lock.acquire(blocking=False)

    def release_run(self, sid: str) -> None:
        sid = _validate_sid(sid)
        with self._state_lock:
            lock = self._run_locks.get(sid)
            if lock and lock.locked():
                lock.release()

    def list(self, *, user: str | None = None) -> list[dict]:
        out = []
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = data.get("session_id")
            if not sid:
                continue
            owner = data.get("owner")
            if not _can_access_session(user, owner):
                continue
            msgs = data.get("messages", [])
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            # A newly opened chat is deliberately kept only in memory until
            # its first interaction. Older builds persisted those placeholders,
            # however, so historical directories can contain many zero-message
            # "New chat" files. They are not conversations and must not be
            # surfaced in the history list.
            if not user_msgs:
                continue
            raw = _sanitize_title(user_msgs[0].get("content", ""))
            full_title = " ".join(raw.split())
            if not full_title:
                continue
            title = (
                full_title[:120] + "…"
                if len(full_title) > 120
                else full_title
            )
            # Sort/display key: prefer the explicit last-chat timestamp; fall
            # back to created_at, then file mtime (legacy sessions). This
            # makes the sidebar reflect "last chatted", immune to mtime bumps
            # from model-switches, uploads, backups, git, or antivirus.
            activity = (
                data.get("last_active")
                or data.get("created_at")
                or f.stat().st_mtime
            )
            out.append({
                "session_id": sid,
                "title": title,
                "full_title": full_title[:1000],
                "attachment_label": _session_attachment_label(data),
                "modality": _session_modality(data),
                "n_messages": len(msgs),
                "updated_at": activity,
                "owner": owner,
            })
        out.sort(key=lambda r: -(r.get("updated_at") or 0))
        return out

    def save(self, sid: str):
        # Internal — already authorised by the calling endpoint.
        sid = _validate_sid(sid)
        s = self._live.get(sid)
        if s is None:
            path = SESSIONS_DIR / f"{sid}.json"
            if not path.exists():
                raise HTTPException(404, f"session {sid} not found")
            s = self._relocate_loaded_session(OphSession.load(path))
            self._live[sid] = s
        s.save(SESSIONS_DIR / f"{sid}.json")

    def delete(self, sid: str, *, user: str | None = None) -> bool:
        # Authorisation: only the recorded owner can delete.
        sid = _validate_sid(sid)
        with self._state_lock:
            path = SESSIONS_DIR / f"{sid}.json"
            existed = path.exists() or sid in self._live
            if not existed:
                return False
            lock = self._run_locks.get(sid)
            if lock and lock.locked():
                raise HTTPException(409, "cannot delete a session while chat is running")
            owner = None
            if path.exists():
                try:
                    owner = json.loads(path.read_text(encoding="utf-8")).get("owner")
                except Exception:
                    owner = None
            elif sid in self._live:
                owner = self._live[sid].owner
            if not _can_access_session(user, owner):
                raise HTTPException(403, "you do not own this session")
            # Deleting a conversation also removes its uploaded clinical files
            # and generated derivatives. Verify both recursive targets remain
            # direct children of their intended roots before removal.
            for root in (UPLOADS_DIR, WORKSPACE):
                root_resolved = root.resolve()
                target = (root_resolved / sid).resolve()
                if target.parent != root_resolved:
                    raise HTTPException(500, "refusing unsafe session cleanup path")
                if target.is_dir():
                    shutil.rmtree(target)
            self._live.pop(sid, None)
            self._run_locks.pop(sid, None)
            if path.exists():
                path.unlink()
            return True


def _sanitize_title(s: str) -> str:
    """Strip lone surrogates / NULs / control bytes that came from encoding mishaps."""
    if not s:
        return ""
    out = []
    for ch in s:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            continue  # lone surrogate
        if cp < 0x20 and ch not in ("\n", "\t"):
            continue
        if ch == "�":
            continue  # replacement char from mojibake
        out.append(ch)
    return "".join(out).strip()


def _session_attachment_label(data: dict) -> str | None:
    """Return a safe, compact attachment name for the history subtitle."""
    context = data.get("context")
    if not isinstance(context, dict):
        return None

    names: list[str] = []
    attached = context.get("attached_images")
    if isinstance(attached, list):
        for item in attached:
            if not isinstance(item, dict):
                continue
            value = item.get("filename") or item.get("path")
            if value:
                name = _sanitize_title(Path(str(value)).name)
                if name and name not in names:
                    names.append(name)

    if not names:
        for key in ("current_image", "current_volume"):
            value = context.get(key)
            if value:
                name = _sanitize_title(Path(str(value)).name)
                if name:
                    names.append(name)
                    break

    if not names:
        return None
    label = names[0]
    if len(names) > 1:
        label += f" +{len(names) - 1}"
    return label[:160]


def _session_modality(data: dict) -> str | None:
    """Return the session modality for a compact history-list badge."""
    context = data.get("context")
    if not isinstance(context, dict):
        return None
    value = _sanitize_title(str(context.get("current_modality") or ""))
    return value[:24].upper() or None


manager = SessionManager()
app = FastAPI(title="OphAgent Web")

# Hook the rate limiter into the app
app.state.limiter = limiter
def _rate_limit_handler(_request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "rate limit exceeded",
                                                   "detail": str(exc.detail)})
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

# Reject oversize uploads up front (≤ 80 MB unless overridden)
_MAX_UPLOAD_MB = _bounded_env_int("WEB_MAX_UPLOAD_MB", 80, 1, 4096)
_MAX_IMAGE_PIXELS = _bounded_env_int(
    "WEB_MAX_IMAGE_PIXELS", 80_000_000, 1_000_000, 250_000_000
)


# ── UI ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@limiter.exempt
def index(_user: str | None = Depends(require_auth)):
    """Serve index.html with cache-busting query strings appended to JS/CSS
    URLs so the browser always reloads them when the file's mtime changes."""
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    js_v = int((WEB_ROOT / "app.js").stat().st_mtime)
    css_v = int((WEB_ROOT / "style.css").stat().st_mtime)
    html = html.replace('/static/app.js"', f'/static/app.js?v={js_v}"')
    html = html.replace('/static/style.css"', f'/static/style.css?v={css_v}"')
    return html


# Static assets (the app shell — JS/CSS). These are safe to serve unauth'd
# because they contain no secrets and refusing them would just break the
# native browser Basic-Auth prompt flow.
app.mount("/static", StaticFiles(directory=str(WEB_ROOT)), name="static")


# Safe replacement for the old `app.mount("/files", PROJECT_ROOT)` — that
# version allowed reading ANY file under the project, including `.env`,
# source code, and checkpoints. Now: explicit allowlist + path-traversal
# defense + dotfile rejection.
#
# Auth handling on this route is special-cased: Chrome (and other Chromium
# browsers) intermittently DROP cached Basic-Auth credentials on async
# subresource fetches (the <img src> loads triggered by attaching an
# upload). The result is that uploaded image thumbnails return 401 even
# though the user has logged in successfully for the /api routes. To avoid
# that footgun we relax auth for /files when the request originates from
# loopback (127.0.0.1 / ::1) — the user is already at the keyboard and
# the host-binding limits exposure to local processes only.
def require_auth_relax_loopback(
    request: Request,
    cf_jwt: str | None = _Header(default=None, alias="Cf-Access-Jwt-Assertion"),
    cf_conn_ip: str | None = _Header(default=None, alias="Cf-Connecting-Ip"),
    xff: str | None = _Header(default=None, alias="X-Forwarded-For"),
    creds: HTTPBasicCredentials | None = Depends(_basic_auth),
):
    if not _AUTH_ENABLED:
        if _request_is_loopback(request) and not _request_has_proxy_headers(request):
            return None
        raise HTTPException(
            status_code=401,
            detail="remote access requires authentication",
        )
    # CF JWT verification takes priority — if the header is present at
    # all, we're behind Cloudflare Access and MUST validate it. We do NOT
    # fall back to the loopback bypass for CF-tunnelled traffic, even
    # though `cloudflared` reverse-proxies via 127.0.0.1.
    if _CF_ACCESS_ENABLED and cf_jwt:
        try:
            email = _verify_cf_access(cf_jwt)
        except Exception:
            email = None
        if email:
            return email
        # CF JWT was sent but malformed or signature failed → fail with a
        # clean 401, never silent loopback downgrade.
        raise HTTPException(status_code=401,
                             detail="Cloudflare Access JWT invalid")
    # Headers set by Cloudflare even when JWT is absent — also bail out
    # of the loopback bypass if any of these are present.
    if cf_conn_ip or xff:
        if _BASIC_AUTH_ENABLED and creds is not None:
            ok_user = _secrets.compare_digest(creds.username, _WEB_USERNAME)
            ok_pass = _secrets.compare_digest(creds.password, _WEB_PASSWORD)
            if ok_user and ok_pass:
                return creds.username
        raise HTTPException(status_code=401, detail="Authentication required",
                             headers={"WWW-Authenticate": 'Basic realm="ophagent"'})

    # Prefer the real Basic identity when the browser/fetch sends it. This lets
    # `/files` reuse the same per-session owner checks as `/api`.
    if _BASIC_AUTH_ENABLED and creds is not None:
        ok_user = _secrets.compare_digest(creds.username, _WEB_USERNAME)
        ok_pass = _secrets.compare_digest(creds.password, _WEB_PASSWORD)
        if ok_user and ok_pass:
            return creds.username

    # Truly local request: no CF headers + loopback IP → bypass auth.
    # Static-file paths are already protected by the whitelist + path
    # traversal defence + host binding to 127.0.0.1.
    if _request_is_loopback(request):
        return _LOCAL_FILE_BYPASS

    # Remote-bound but no CF headers (e.g. host bound to 0.0.0.0 without
    # CF tunnel) — go through the strict auth chain.
    raise HTTPException(
        status_code=401, detail="Authentication required",
        headers={"WWW-Authenticate": 'Basic realm="ophagent"'},
    )


@app.get("/files/{rel_path:path}")
@limiter.exempt
def serve_whitelisted_file(
    rel_path: str,
    _user: str | None = Depends(require_auth_relax_loopback),
):
    target = _resolve_whitelisted(rel_path, user=_user)
    if target is None:
        raise HTTPException(status_code=404, detail="not found or out of scope")
    return FileResponse(str(target))


# ── API: sessions ───────────────────────────────────────────────────────────
@app.post("/api/sessions")
def api_create_session(_user: str | None = Depends(require_auth)):
    s = manager.create(owner=_user)
    return {"session_id": s.session_id, "owner": s.owner,
            "backend": s.backend, "model": s.model, "effort": s.effort,
            "runtime": _session_runtime_config(s)}


@app.get("/api/sessions")
def api_list_sessions(_user: str | None = Depends(require_auth)):
    return manager.list(user=_user)


@app.get("/api/sessions/{sid}")
def api_get_session(sid: str, _user: str | None = Depends(require_auth)):
    s = manager.get(sid, user=_user)
    return {
        "session_id": s.session_id,
        "owner": s.owner,
        "you": _user,
        "is_admin": _is_admin(_user),
        "backend": s.backend,
        "model": s.model,
        "effort": s.effort,
        "runtime": _session_runtime_config(s),
        "messages": _expose_messages(s.messages, s.context.attached_images),
        "context": {
            "current_image": _to_web_path(s.context.current_image),
            "current_volume": _to_web_path(s.context.current_volume),
            "current_modality": getattr(s.context, "current_modality", None),
            "n_analyses": len(s.context.analyses),
        },
    }


@app.delete("/api/sessions/{sid}")
def api_delete_session(sid: str, _user: str | None = Depends(require_auth)):
    ok = manager.delete(sid, user=_user)
    return {"deleted": ok}


@app.get("/api/sessions/{sid}/export")
def api_export_session(sid: str, _user: str | None = Depends(require_auth)):
    """Export a session as a single self-contained, print-ready HTML report
    (images embedded as base64, Markdown rendered, tool trace appended)."""
    from .export import build_session_html
    s = manager.get(sid, user=_user)   # enforces ownership / auth
    title = f"OphAgent report · {s.session_id}"
    html_str = build_session_html(
        s, user=_user, project_root=FILE_ROOT, title=title,
    )
    # Filename: ophagent_<modality>_<sid>_<date>.html
    from datetime import datetime
    modality = (getattr(s.context, "current_modality", None) or "report").lower()
    stamp = datetime.now().strftime("%Y%m%d")
    fname = f"ophagent_{modality}_{sid}_{stamp}.html"
    return HTMLResponse(
        content=html_str,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── API: model / provider catalog + per-session switching ───────────────────
@app.get("/api/catalog")
def api_catalog(_user: str | None = Depends(require_auth)):
    """Return the channel → provider → model hierarchy for the UI picker."""
    return {
        "channels": list_api_channels(),
        "providers": [
            {
                "id": p,
                "label": PROVIDER_SPECS[p]["label"],
                "channel": PROVIDER_SPECS[p]["channel"],
                "models": models_catalog.list_models(p),
            }
            for p in models_catalog.list_providers()
        ],
        "current_backend": DEFAULT_BACKEND,
        "current_model": DEFAULT_MODEL,
    }


class ApiProviderSettingsBody(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    clear_key: bool = False


def _web_api_providers() -> list[str]:
    return [p for p in models_catalog.list_providers() if p in PROVIDER_SPECS]


@app.get("/api/settings/api")
def api_get_api_settings(_user: str | None = Depends(require_auth)):
    return {"providers": [_api_provider_status(_user, p) for p in _web_api_providers()]}


@app.post("/api/settings/api/{provider}")
def api_save_api_settings(
    provider: str,
    body: ApiProviderSettingsBody,
    _user: str | None = Depends(require_auth),
):
    if provider not in _web_api_providers():
        raise HTTPException(400, f"unsupported API provider: {provider}")
    with manager.configuration_guard():
        with _API_CREDENTIALS_LOCK:
            credentials = load_user_api_credentials(_user)
            entry = dict(credentials.get(provider) or {})
            if body.clear_key:
                # A custom endpoint belongs to the personal credential. Removing
                # the key must also remove that endpoint so an environment secret
                # can never be redirected to a user-controlled URL.
                entry.clear()
            elif body.api_key is not None and body.api_key.strip():
                key = body.api_key.strip()
                if len(key) > 8192:
                    raise HTTPException(400, "API key is too long")
                entry["api_key"] = key
            if body.base_url is not None and not body.clear_key:
                base_url = _validate_api_base_url(
                    body.base_url,
                    allow_private=(not _AUTH_ENABLED or _is_admin(_user)),
                )
                trusted_base = str(
                    resolve_provider_connection(provider, None).get("base_url") or ""
                ).rstrip("/")
                if not base_url or base_url == trusted_base:
                    entry.pop("base_url", None)
                elif not entry.get("api_key"):
                    raise HTTPException(
                        400, "a custom Base URL requires a personal API key"
                    )
                else:
                    entry["base_url"] = base_url
            if entry:
                credentials[provider] = entry
            else:
                credentials.pop(provider, None)
            save_user_api_credentials(_user, credentials)
        manager.refresh_api_credentials(_user)
    return _api_provider_status(_user, provider)


@app.post("/api/settings/api/{provider}/check")
def api_check_api_settings(
    provider: str,
    body: ApiProviderSettingsBody | None = None,
    _user: str | None = Depends(require_auth),
):
    if provider not in _web_api_providers():
        raise HTTPException(400, f"unsupported API provider: {provider}")
    credentials = load_user_api_credentials(_user)
    saved = dict(credentials.get(provider) or {})
    transient_key = str((body.api_key if body else None) or "").strip()
    if len(transient_key) > 8192:
        raise HTTPException(400, "API key is too long")
    if transient_key:
        saved["api_key"] = transient_key
    resolved = resolve_provider_connection(provider, saved)
    if not resolved["api_key"]:
        return {"ok": False, "message": "API key is not configured"}
    if body and body.base_url is not None:
        transient_base = _validate_api_base_url(
            body.base_url,
            allow_private=(not _AUTH_ENABLED or _is_admin(_user)),
        )
        trusted_base = str(
            resolve_provider_connection(provider, None).get("base_url") or ""
        ).rstrip("/")
        has_personal_key = bool(saved.get("api_key"))
        if transient_base and transient_base != trusted_base and not has_personal_key:
            return {
                "ok": False,
                "message": "A custom Base URL requires a personal API key",
            }
        resolved["base_url"] = transient_base or trusted_base
    try:
        base_url = _validate_api_base_url(
            str(resolved["base_url"] or ""),
            allow_private=(not _AUTH_ENABLED or _is_admin(_user)),
        )
    except HTTPException as exc:
        return {"ok": False, "message": str(exc.detail)}
    if not base_url:
        return {"ok": False, "message": "Base URL is not configured"}
    request_headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "Accept": "application/json",
        "User-Agent": "OphAgent-credential-check/1.0",
    }
    if provider == "anthropic":
        request_headers.update({
            "x-api-key": resolved["api_key"],
            "anthropic-version": "2023-06-01",
        })
    request = UrlRequest(
        f"{base_url}/models",
        headers=request_headers,
        method="GET",
    )
    try:
        # A redirect to a private address must not bypass Base URL validation.
        with build_opener(_NoRedirectHandler()).open(request, timeout=15) as response:
            raw = response.read(2_000_000)
            payload = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            models = payload.get("data") if isinstance(payload, dict) else None
            count = len(models) if isinstance(models, list) else None
            return {"ok": True, "message": "Connection verified", "model_count": count}
    except HTTPError as exc:
        return {"ok": False, "message": f"Provider returned HTTP {exc.code}"}
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        return {"ok": False, "message": f"Connection failed: {type(exc).__name__}"}


class CheckpointGroupBody(BaseModel):
    enabled: bool | None = None
    paths: dict[str, str] | None = None


class CheckpointCheckBody(BaseModel):
    paths: dict[str, str] | None = None


def _can_manage_checkpoints(user: str | None) -> bool:
    # An unauthenticated deployment is permitted only on loopback by main().
    return not _AUTH_ENABLED or _is_admin(user)


@app.get("/api/settings/checkpoints")
def api_get_checkpoint_settings(_user: str | None = Depends(require_auth)):
    if not _can_manage_checkpoints(_user):
        return {
            "can_manage": False,
            "groups": [],
            "summary": {},
            "restart_required": False,
        }
    return {"can_manage": True, **checkpoint_settings_view()}


@app.post("/api/settings/checkpoints/{group_id}")
def api_save_checkpoint_group(
    group_id: str,
    body: CheckpointGroupBody,
    _user: str | None = Depends(require_auth),
):
    if not _can_manage_checkpoints(_user):
        raise HTTPException(403, "checkpoint configuration requires an administrator")
    if body.enabled is None and body.paths is None:
        raise HTTPException(400, "enabled or paths must be provided")
    with manager.configuration_guard():
        try:
            config, paths_changed, enabled_changed = update_checkpoint_group(
                group_id,
                enabled=body.enabled,
                paths=body.paths,
            )
        except KeyError:
            raise HTTPException(404, f"unknown checkpoint group: {group_id}")
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        if enabled_changed:
            manager.refresh_checkpoint_config()
    return {
        "group": checkpoint_group_view(group_id, config=config),
        "paths_changed": paths_changed,
        "enabled_changed": enabled_changed,
        "restart_required": checkpoint_restart_required(config),
    }


@app.post("/api/settings/checkpoints/{group_id}/check")
def api_check_checkpoint_group(
    group_id: str,
    body: CheckpointCheckBody | None = None,
    _user: str | None = Depends(require_auth),
):
    if not _can_manage_checkpoints(_user):
        raise HTTPException(403, "checkpoint verification requires an administrator")
    try:
        return check_checkpoint_group(group_id, (body.paths if body else None))
    except KeyError:
        raise HTTPException(404, f"unknown checkpoint group: {group_id}")
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc))


class ModelSwitchBody(BaseModel):
    backend: str | None = None
    model: str | None = None
    effort: str | None = None    # 'low' | 'medium' | 'high' | 'max' | 'ultra'


def _session_has_persistable_state(s: OphSession) -> bool:
    ctx = s.context
    return bool(
        s.messages
        or ctx.current_image
        or ctx.current_volume
        or ctx.attached_images
        or ctx.analyses
        or ctx.last_report
    )


@app.post("/api/sessions/{sid}/model")
def api_switch_model(sid: str, body: ModelSwitchBody,
                     _user: str | None = Depends(require_auth)):
    """Switch the backend and/or model and/or effort level for a session."""
    s = manager.get(sid, user=_user)
    if not manager.try_acquire_run(sid):
        raise HTTPException(409, "cannot change model while chat is running")
    try:
        proposed_backend = body.backend or s.backend
        if proposed_backend not in models_catalog.list_providers():
            raise HTTPException(400, f"unknown provider {proposed_backend}")
        proposed_model = body.model or (
            models_catalog.default_model(proposed_backend)
            if proposed_backend != s.backend
            else s.model
        )
        proposed_effort = body.effort or s.effort
        if proposed_effort not in {"low", "medium", "high", "max", "ultra"}:
            raise HTTPException(
                400,
                "effort must be low/medium/high/max/ultra, "
                f"got {proposed_effort}",
            )
        if not models_catalog.model_supports_tools(proposed_backend, proposed_model):
            raise HTTPException(
                400,
                f"'{proposed_model}' cannot call tools, so it cannot run the "
                "analysis pipeline. Pick a tool-capable model such as "
                "qwen3-vl-plus or qwen-max. To use it only for visual reads, "
                "set OPH_WEB_VISION_MODEL instead.",
            )

        changed_model = (
            proposed_backend != s.backend or proposed_model != s.model
        )
        s.backend = proposed_backend
        s.model = proposed_model
        s.effort = proposed_effort
        if changed_model:
            _invalidate_session_clients(s)

        # A model choice alone is a user preference, not a conversation. Do
        # not recreate the historical empty "New chat" files that the sidebar
        # deliberately filters out.
        session_path = SESSIONS_DIR / f"{sid}.json"
        if session_path.exists() or _session_has_persistable_state(s):
            manager.save(sid)
        save_user_pref(_user, s.backend, s.model, s.effort)
        return {
            "session_id": sid,
            "backend": s.backend,
            "model": s.model,
            "effort": s.effort,
            "runtime": _session_runtime_config(s),
        }
    finally:
        manager.release_run(sid)


# ── API: chat ───────────────────────────────────────────────────────────────
class ChatBody(BaseModel):
    text: str


@app.post("/api/sessions/{sid}/chat")
def api_chat(sid: str, body: ChatBody,
             _user: str | None = Depends(require_auth)):
    s = manager.get(sid, user=_user)
    if not manager.try_acquire_run(sid):
        raise HTTPException(409, "session already has a running chat")
    try:
        reply = s.chat(body.text)
    except Exception:
        log.exception("chat failed for session %s", sid)
        raise HTTPException(500, "chat failed")
    finally:
        manager.release_run(sid)
    manager.save(sid)
    return {
        "reply": reply,
        "messages": _expose_messages(s.messages, s.context.attached_images),
        "context": {
            "current_image": _to_web_path(s.context.current_image),
            "current_volume": _to_web_path(s.context.current_volume),
        },
        "last_report": _expose_last_report(s),
    }


@app.post("/api/sessions/{sid}/chat/stream")
async def api_chat_stream(sid: str, body: ChatBody,
                          _user: str | None = Depends(require_auth)):
    """Server-Sent Events: stream tool-use events as they happen.

    Emits NDJSON-style SSE 'data:' lines with these event types:
      - thinking
      - tool_call   {name, arguments}
      - tool_result {name, preview}
      - text        {content}
      - done        {context, last_report, messages}
      - error       {message}
    """
    s = manager.get(sid, user=_user)
    if not manager.try_acquire_run(sid):
        raise HTTPException(409, "session already has a running chat")
    q: "_queue.Queue[dict]" = _queue.Queue()
    _emitted_text = {"v": False}

    def on_event(ev: dict):
        if (isinstance(ev, dict) and ev.get("type") == "text"
                and (ev.get("content") or "").strip()):
            _emitted_text["v"] = True
        q.put(ev)

    def runner():
        reply = None
        try:
            reply = s.chat(body.text, on_event=on_event)
        except Exception as e:
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            # Belt-and-suspenders: a few chat() return paths (e.g. the
            # scope-routing NON_OPHTHALMOLOGIC refusal) return the reply
            # WITHOUT emitting a 'text' event, so the live SSE stream never
            # renders the answer bubble — it only appears after a manual
            # reload from persisted history. If no text was streamed this
            # turn, emit the returned reply here before 'done'.
            if reply and reply.strip() and not _emitted_text["v"]:
                q.put({"type": "text", "content": reply})
            try:
                manager.save(sid)
            except Exception:
                pass
            manager.release_run(sid)
            q.put({
                "type": "done",
                "context": {
                    "current_image": _to_web_path(s.context.current_image),
                    "current_volume": _to_web_path(s.context.current_volume),
                },
                "last_report": _expose_last_report(s),
                "messages": _expose_messages(s.messages, s.context.attached_images),
            })
            q.put(None)  # sentinel to end the stream

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    async def gen():
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Heartbeat every 15s: MUST be well under Cloudflare's ~100s
                # idle-stream timeout, or a single slow step (a long gpt-5 call
                # under load, or a max/ultra debate) goes >100s without bytes
                # and Cloudflare cancels the SSE stream → browser "network
                # error". 600s here was the bug.
                ev = await loop.run_in_executor(None, q.get, True, 15)
            except _queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                continue
            if ev is None:
                return
            yield f"data: {json.dumps(ev, default=str, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/sessions/{sid}/abort")
def api_abort(sid: str, _user: str | None = Depends(require_auth)):
    """Signal an in-flight chat() loop to stop after its current step.

    Sets `interrupt_requested` on the session's context; the chat loop
    polls this flag at the top of every iteration and exits cleanly with
    a "## Interrupted" assistant message. Returns immediately — the
    background runner thread continues to drain its current step and
    closes the SSE stream on its own.
    """
    s = manager.get(sid, user=_user)
    s.context.interrupt_requested = True
    return {"ok": True, "interrupted_session": sid}


# Extension whitelist — never accept executables / scripts via upload.
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_ALLOWED_VOLUME_EXT = {".dcm", ".nii", ".gz", ".npy", ".npz", ".fda"}


# ── API: upload ─────────────────────────────────────────────────────────────
@app.post("/api/sessions/{sid}/upload")
def api_upload(sid: str, file: UploadFile = File(...),
               kind: str = Form("auto"),
               _user: str | None = Depends(require_auth)):
    """Accept an image (PNG/JPG) or volume (DCM/NIfTI/NPY)."""
    s = manager.get(sid, user=_user)
    if not manager.try_acquire_run(sid):
        raise HTTPException(409, "cannot upload while chat is running")
    try:
        return _store_upload(sid, s, file, kind)
    finally:
        manager.release_run(sid)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _upload_filename(filename: str, kind: str) -> tuple[str, bool]:
    orig_name = Path(filename or "").name
    if not orig_name:
        raise HTTPException(400, "filename required")
    kind = str(kind or "auto").strip().lower()
    if kind not in {"auto", "image", "volume"}:
        raise HTTPException(400, "kind must be auto, image, or volume")

    path = Path(orig_name)
    ext_lower = path.suffix.lower()
    suffixes_lower = [suffix.lower() for suffix in path.suffixes]
    if ext_lower not in (_ALLOWED_IMAGE_EXT | _ALLOWED_VOLUME_EXT):
        raise HTTPException(415, f"unsupported file type: {ext_lower!r}")
    is_nifti_gz = suffixes_lower[-2:] == [".nii", ".gz"]
    if ext_lower == ".gz" and not is_nifti_gz:
        raise HTTPException(415, "only .nii.gz archives are accepted")
    is_volume = ext_lower in _ALLOWED_VOLUME_EXT
    if kind == "volume" and not is_volume:
        raise HTTPException(400, "the selected file is not a supported volume")
    if kind == "image" and is_volume:
        raise HTTPException(400, "the selected file is a volume, not an image")

    suffix = ".nii.gz" if is_nifti_gz else ext_lower
    source_stem = orig_name[:-len(suffix)] if suffix else orig_name
    safe_stem = source_stem.strip(" .")
    unsafe = (
        not safe_stem
        or not safe_stem.isascii()
        or any(c not in " ._-" and not c.isalnum() for c in safe_stem)
    )
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if unsafe or safe_stem.upper() in reserved:
        safe_stem = "file_" + hashlib.sha256(
            source_stem.encode("utf-8")
        ).hexdigest()[:12]
    return safe_stem + suffix, is_volume


def _store_upload(
    sid: str,
    s: OphSession,
    file: UploadFile,
    kind: str,
) -> dict[str, Any]:
    safe_name, is_volume = _upload_filename(file.filename or "", kind)
    s._ensure_toolkit()
    target_dir = UPLOADS_DIR / sid
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = target_dir / f".upload-{_secrets.token_hex(12)}.tmp"
    max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    incoming_hash = hashlib.sha256()
    try:
        file.file.seek(0)
        with temp_path.open("xb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        413,
                        f"file exceeds {_MAX_UPLOAD_MB} MB limit",
                    )
                incoming_hash.update(chunk)
                out.write(chunk)
        if total == 0:
            raise HTTPException(400, "empty files are not accepted")

        dest = target_dir / safe_name
        if dest.exists():
            suffix = (
                ".nii.gz"
                if safe_name.lower().endswith(".nii.gz")
                else Path(safe_name).suffix
            )
            stem = safe_name[:-len(suffix)] if suffix else safe_name
            incoming_digest = incoming_hash.hexdigest()
            counter = 0
            while True:
                candidate = (
                    target_dir / safe_name
                    if counter == 0
                    else target_dir / f"{stem}_{counter}{suffix}"
                )
                if not candidate.exists():
                    dest = candidate
                    temp_path.replace(dest)
                    break
                identical = (
                    candidate.stat().st_size == total
                    and _file_sha256(candidate) == incoming_digest
                )
                if identical:
                    dest = candidate
                    temp_path.unlink()
                    break
                counter += 1
        else:
            temp_path.replace(dest)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if is_volume:
        s.set_volume(str(dest))
        manager.save(sid)
        return {
            "kind": "volume",
            "path": _to_web_path(str(dest)),
            "filename": dest.name,
        }

    s.set_image(str(dest))
    if s.context.current_modality == "INVALID_INPUT":
        reason = s.context.invalid_input_reason or "file is not a readable image"
        dest.unlink(missing_ok=True)
        s.context.current_image = None
        s.context.current_modality = None
        s.context.modality_scope = "in_scope"
        s.context.invalid_input_reason = None
        status_code = 413 if "pixel limit" in reason else 400
        raise HTTPException(status_code, reason)
    manager.save(sid)
    thumb_url = _make_safe_thumbnail(dest)
    return {
        "kind": "image",
        "modality": s.context.current_modality,
        "path": _to_web_path(str(dest)),
        "thumb_path": thumb_url or _to_web_path(str(dest)),
        "filename": dest.name,
    }


def _make_safe_thumbnail(orig_path: Path, max_side: int = 384) -> str | None:
    """Render a small baseline-JPEG thumbnail next to the original.

    Returns the /files/... URL of the thumbnail on success, or None if
    Pillow can't decode the source (in which case the caller falls back
    to the original URL and the browser may still fail to render — but
    that's no worse than today's behaviour)."""
    try:
        from PIL import Image, ImageOps
        thumb_path = orig_path.with_name(orig_path.stem + "_thumb.jpg")
        if thumb_path.exists():
            return _to_web_path(str(thumb_path))
        with Image.open(str(orig_path)) as im:
            width, height = im.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"image dimensions {width}x{height} exceed the "
                    f"{_MAX_IMAGE_PIXELS:,}-pixel limit"
                )
            # Convert to RGB to drop alpha + exotic colour profiles.
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im = ImageOps.exif_transpose(im)
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            im.save(thumb_path, format="JPEG", quality=82, optimize=True,
                     progressive=False)
        return _to_web_path(str(thumb_path))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"thumbnail generation failed for {orig_path}: {e}")
        return None


# ── helpers ─────────────────────────────────────────────────────────────────
def _to_web_path(abs_path: str | None) -> str | None:
    """Convert an absolute generated-output path to a URL served by /files."""
    if not abs_path:
        return None
    p = Path(abs_path).resolve()
    try:
        rel = p.relative_to(FILE_ROOT)
    except ValueError:
        return None
    return "/files/" + rel.as_posix()


def _expose_last_report(s: ChatSession) -> dict | None:
    """Find the most recent report dir created during this session."""
    # ChatSession's toolkit drops reports under its run_dir
    if s._toolkit is None:
        return None
    root = Path(s._toolkit.report_output_root)
    if not root.exists():
        return None
    candidates = sorted(
        [d for d in root.glob("*/report.pdf")],
        key=lambda p: -p.stat().st_mtime,
    )
    if not candidates:
        return None
    pdf = candidates[0]
    html = pdf.with_name("report.html")
    return {
        "pdf": _to_web_path(str(pdf)),
        "html": _to_web_path(str(html)) if html.exists() else None,
        "dir": _to_web_path(str(pdf.parent)),
    }


def _tool_preview_for_history(name: str, content: str) -> str:
    """Build the SAME clean one-line preview the live trace shows (via
    _summarize_result), so reloading a session looks identical to the live
    run instead of dumping raw truncated JSON."""
    if not content:
        return ""
    try:
        parsed = json.loads(content)
    except Exception:
        return content[:160]
    try:
        from ..chat.oph_session import _summarize_result
        return _summarize_result(name, parsed)
    except Exception:
        return content[:160]


# Signatures of internal steering messages that were historically persisted
# with role="user". They must never render as if the user typed them.
_INTERNAL_STEERING_PREFIXES = (
    "STOP CALLING ",
    "STOP. Your previous reply leaked",
)


def _is_internal_steering(content) -> bool:
    if not isinstance(content, str):
        return False
    c = content.lstrip()
    return any(c.startswith(p) for p in _INTERNAL_STEERING_PREFIXES)


def _expose_messages(messages: list[dict],
                     attached_images: list[dict] | None = None) -> list[dict]:
    """Strip tool-call internals; keep user/assistant text + tool results that
    the user might want to see in the UI.

    `attached_images` (session.context.attached_images) lets us re-attach the
    image thumbnail to the user turn that referenced it, so a reloaded session
    shows the picture again (the live path renders it client-side, but the
    persisted OpenAI-format message only carries the filename as text).

    Association is by UPLOAD ORDER, not filename: the server stores a hashed
    name (img_<hash>.jpg) while the message text keeps the user's original
    filename, so they never match. Instead we walk user turns in order and
    consume attached_images sequentially — each "(Attached …)" turn takes the
    next N images. The original display name (for the label) is parsed back out
    of the message text when possible."""
    import re
    atts = attached_images or []
    img_ptr = 0
    out = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            content = m.get("content", "")
            # Hide internal steering nudges that older sessions baked into
            # the persisted history with role="user".
            if _is_internal_steering(content):
                continue
            entry = {"role": "user", "content": content}
            # How many files did this turn attach? "(Attached N files" → N;
            # a single "(Attached …)" → 1; otherwise 0.
            mobj = re.search(r"\(Attached\s+(\d+)\s+files", content)
            n_files = (int(mobj.group(1)) if mobj
                       else (1 if "(Attached " in content else 0))
            if n_files and img_ptr < len(atts):
                # Recover original display names from the text for nicer labels.
                names = re.findall(r"\(Attached [^:]*:\s*(.+)\)", content)
                names += re.findall(r"-\s*\[[^\]]*\]\s*(\S+)", content)
                imgs = []
                for k, a in enumerate(atts[img_ptr:img_ptr + n_files]):
                    try:
                        thumb = _make_safe_thumbnail(Path(a["path"])) \
                            or _to_web_path(a.get("path"))
                    except Exception:
                        thumb = _to_web_path(a.get("path"))
                    if thumb:
                        label = names[k] if k < len(names) else a.get("filename")
                        imgs.append({"name": label,
                                     "modality": a.get("modality"),
                                     "thumb": thumb})
                img_ptr += n_files
                if imgs:
                    entry["images"] = imgs
            out.append(entry)
        elif role == "assistant":
            entry = {"role": "assistant", "content": m.get("content", "") or ""}
            if m.get("tool_calls"):
                entry["tool_calls"] = [
                    {"name": tc["function"]["name"]} for tc in m["tool_calls"]
                ]
            out.append(entry)
        elif role == "tool":
            # surface the tool name + the SAME clean preview the live trace
            # uses, PLUS the per-step drill-down data (predictions / figures /
            # error) so a reloaded trace is expandable just like the live one.
            raw = m.get("content", "") or ""
            tool_entry = {
                "role": "tool",
                "name": m.get("name", ""),
                "preview": _tool_preview_for_history(m.get("name", ""), raw),
            }
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if parsed.get("predictions") is not None:
                    tool_entry["predictions"] = parsed["predictions"]
                fu = parsed.get("figure_urls")
                if fu and isinstance(fu, dict):
                    keep = {k: v for k, v in fu.items() if v}
                    if keep:
                        tool_entry["figure_urls"] = keep
                if parsed.get("error"):
                    tool_entry["error"] = parsed["error"]
                # Same human-readable markdown the live trace shows for
                # LLM/text tools (vision_impression, verify_findings).
                try:
                    from ..chat.oph_session import _tool_detail_md
                    md_txt = _tool_detail_md(m.get("name", ""), parsed)
                    if md_txt:
                        tool_entry["detail_md"] = md_txt
                except Exception:
                    pass
            out.append(tool_entry)
    return out


def main():
    import uvicorn
    host = os.environ.get("OPH_WEB_HOST", "127.0.0.1")
    port = _bounded_env_int("OPH_WEB_PORT", 8765, 1, 65535)

    # Refuse to bind to a public interface without Basic Auth configured —
    # this is the single biggest foot-gun for an Internet-facing deployment.
    public_bind = host not in {"127.0.0.1", "localhost", "::1"}
    if public_bind and not _AUTH_ENABLED:
        raise SystemExit(
            "REFUSING TO START: OPH_WEB_HOST is bound to a public interface "
            f"({host}) but WEB_USERNAME / WEB_PASSWORD are not set in the "
            ".env. Anyone on the Internet could call your tools and read "
            "your AIGCBEST credits. Set both vars (and use Cloudflare Tunnel "
            "+ Access for production) before starting in public mode."
        )
    print(f"\n  → open http://{host}:{port}/  in your browser")
    if _CF_ACCESS_ENABLED:
        print(f"  → auth: Cloudflare Access ({_CF_TEAM_DOMAIN}) + Basic Auth fallback")
    elif _BASIC_AUTH_ENABLED:
        print(f"  → auth: Basic Auth only (WEB_USERNAME / WEB_PASSWORD)")
    else:
        print(f"  → auth: OFF (local trusted mode)")
    print(f"  → rate limit: {_RATE_LIMIT}    max upload: {_MAX_UPLOAD_MB} MB\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
