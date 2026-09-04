"""SQLite projection of runtime lifecycle events for inspectable Agent runs."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from nanobot.bus.runtime_events import (
    RuntimeEvent,
    RuntimeEventBus,
    SessionTurnPersisted,
    SessionTurnStarted,
    TurnCompleted,
    TurnRetryObserved,
    TurnRunStatusChanged,
    TurnRuntimeAdmitted,
)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    turn_id: str | None
    session_key: str
    channel: str
    chat_id: str
    status: str
    started_at_ms: int
    completed_at_ms: int | None
    latency_ms: int | None
    provider: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    tool_calls: int
    retries: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunStore:
    """Persist a read model without coupling the Agent loop to SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._active_by_session: dict[str, str] = {}

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 2000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    turn_id TEXT,
                    session_key TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at_ms INTEGER NOT NULL,
                    completed_at_ms INTEGER,
                    latency_ms INTEGER,
                    provider TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS agent_runs_session_time_idx
                    ON agent_runs(session_key, started_at_ms DESC);
                CREATE INDEX IF NOT EXISTS agent_runs_status_time_idx
                    ON agent_runs(status, started_at_ms DESC);
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(**dict(row))

    def get(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._record(row) if row else None

    def list(self, *, session_key: str | None = None, limit: int = 50) -> list[RunRecord]:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        query = "SELECT * FROM agent_runs"
        params: list[object] = []
        if session_key:
            query += " WHERE session_key = ?"
            params.append(session_key)
        query += " ORDER BY started_at_ms DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._record(row) for row in rows]

    def _active_id(self, session_key: str) -> str | None:
        return self._active_by_session.get(session_key)

    def increment_tool_calls(self, session_key: str) -> None:
        run_id = self._active_id(session_key)
        if run_id is None:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE agent_runs SET tool_calls = tool_calls + 1 WHERE run_id = ?",
                (run_id,),
            )

    def record_error(self, session_key: str, error: str) -> None:
        run_id = self._active_id(session_key)
        if run_id is None:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE agent_runs SET status = 'error', error = ? WHERE run_id = ?",
                (error[:4_000], run_id),
            )

    def handle(self, event: RuntimeEvent) -> None:
        """Project one runtime event; suitable as a RuntimeEventBus subscriber."""

        context = getattr(event, "context", None)
        if context is None:
            return
        session_key = context.session_key
        with self._lock, self._connect() as connection:
            if isinstance(event, SessionTurnStarted):
                run_id = uuid4().hex
                self._active_by_session[session_key] = run_id
                connection.execute(
                    """INSERT INTO agent_runs (
                        run_id, session_key, channel, chat_id, status, started_at_ms
                    ) VALUES (?, ?, ?, ?, 'running', ?)""",
                    (run_id, session_key, context.channel, context.chat_id, int(time.time() * 1000)),
                )
                return

            run_id = self._active_id(session_key)
            if run_id is None:
                return
            if isinstance(event, TurnRuntimeAdmitted):
                provider = type(event.runtime.provider).__name__
                connection.execute(
                    "UPDATE agent_runs SET provider = ?, model = ? WHERE run_id = ?",
                    (provider, event.runtime.model, run_id),
                )
            elif isinstance(event, TurnRunStatusChanged):
                connection.execute(
                    "UPDATE agent_runs SET status = ? WHERE run_id = ?",
                    (event.status[:80], run_id),
                )
            elif isinstance(event, TurnRetryObserved):
                connection.execute(
                    "UPDATE agent_runs SET retries = retries + 1 WHERE run_id = ?",
                    (run_id,),
                )
            elif isinstance(event, TurnCompleted):
                usage = event.usage
                connection.execute(
                    """UPDATE agent_runs SET
                    status = CASE WHEN error IS NULL THEN 'completed' ELSE 'error' END,
                    completed_at_ms = ?,
                    latency_ms = ?, input_tokens = ?, output_tokens = ?, total_tokens = ?
                    WHERE run_id = ?""",
                    (
                        int(time.time() * 1000),
                        event.latency_ms,
                        usage.input_tokens if usage else None,
                        usage.output_tokens if usage else None,
                        usage.total_tokens if usage else None,
                        run_id,
                    ),
                )
                record = connection.execute(
                    "SELECT turn_id FROM agent_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if record and record["turn_id"] is not None:
                    self._active_by_session.pop(session_key, None)
            elif isinstance(event, SessionTurnPersisted):
                connection.execute(
                    "UPDATE agent_runs SET turn_id = ? WHERE run_id = ?",
                    (event.turn_id, run_id),
                )
                record = connection.execute(
                    "SELECT completed_at_ms FROM agent_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if record and record["completed_at_ms"] is not None:
                    self._active_by_session.pop(session_key, None)

    def subscribe(self, bus: RuntimeEventBus):
        return bus.subscribe(self.handle)
