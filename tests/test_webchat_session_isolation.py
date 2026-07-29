from __future__ import annotations

import json
from pathlib import Path

from ophagent.webchat import server


def _write_session(path: Path, sid: str, owner: str | None) -> None:
    path.write_text(
        json.dumps(
            {
                "session_id": sid,
                "owner": owner,
                "messages": [{"role": "user", "content": f"message-{sid}"}],
                "last_active": 1,
            }
        ),
        encoding="utf-8",
    )


def test_admin_status_does_not_grant_conversation_access(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "_WEB_USERNAME", "local-admin")
    monkeypatch.setattr(server, "_ADMIN_EMAILS", {"admin@example.com"})

    assert server._is_admin("local-admin")
    assert server._is_admin("admin@example.com")
    assert server._can_access_session(
        "admin@example.com", "admin@example.com"
    )
    assert not server._can_access_session(
        "admin@example.com", "reader@example.com"
    )
    assert not server._can_access_session("local-admin", "reader@example.com")
    assert not server._can_access_session("admin@example.com", None)


def test_session_list_contains_only_the_authenticated_owners_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(server, "_WEB_USERNAME", "local-admin")
    monkeypatch.setattr(server, "_ADMIN_EMAILS", {"admin@example.com"})

    _write_session(tmp_path / "own.json", "own", "admin@example.com")
    _write_session(tmp_path / "other.json", "other", "reader@example.com")
    _write_session(tmp_path / "legacy.json", "legacy", None)

    rows = server.SessionManager().list(user="admin@example.com")

    assert [row["session_id"] for row in rows] == ["own"]
