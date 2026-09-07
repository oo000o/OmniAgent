"""SQLite projection of runtime lifecycle events for inspectable Agent runs."""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections.abc import Sequence
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
from nanobot.observability.errors import (
    ERROR_POLICY,
    AgentErrorCode,
    extract_error_code,
    sanitize_error_message,
)

_FINISHED_STATUSES = frozenset({"completed", "error"})
_TOOL_TERMINAL = frozenset({"tool_succeeded", "tool_failed"})


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
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunEvent:
    event_id: str
    run_id: str
    parent_event_id: str | None
    event_type: str
    tool_name: str | None
    workflow_id: str | None
    plan_revision: int | None
    step: str | None
    status: str
    error_code: str | None
    retry_count: int
    started_at_ms: int | None
    completed_at_ms: int | None
    flags: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = [flag for flag in self.flags.split(",") if flag]
        return payload


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
                    error TEXT,
                    error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS agent_runs_session_time_idx
                    ON agent_runs(session_key, started_at_ms DESC);
                CREATE INDEX IF NOT EXISTS agent_runs_status_time_idx
                    ON agent_runs(status, started_at_ms DESC);
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_event_id TEXT,
                    event_type TEXT NOT NULL,
                    tool_name TEXT,
                    workflow_id TEXT,
                    plan_revision INTEGER,
                    step TEXT,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    started_at_ms INTEGER,
                    completed_at_ms INTEGER,
                    flags TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS agent_run_events_run_time_idx
                    ON agent_run_events(run_id, started_at_ms, event_id);
                """
            )
            self._ensure_column(connection, "agent_runs", "error_code", "TEXT")

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        names = {str(row[1]) for row in rows}
        if column not in names:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _record(row: sqlite3.Row) -> RunRecord:
        payload = dict(row)
        payload.setdefault("error_code", None)
        return RunRecord(**payload)

    @staticmethod
    def _event(row: sqlite3.Row) -> RunEvent:
        return RunEvent(**dict(row))

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

    def list_events(self, run_id: str) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM agent_run_events
                WHERE run_id = ?
                ORDER BY COALESCE(started_at_ms, 0), event_id""",
                (run_id,),
            ).fetchall()
        return [self._event(row) for row in rows]

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

    def record_error(
        self,
        session_key: str,
        error: str,
        error_code: str | AgentErrorCode | None = None,
    ) -> None:
        run_id = self._active_id(session_key)
        if run_id is None:
            return
        sanitized = sanitize_error_message(error)
        code = str(error_code) if error_code else None
        if code is None:
            extracted = extract_error_code(error)
            code = extracted.value if extracted else None
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE agent_runs
                SET status = 'error', error = ?, error_code = ?
                WHERE run_id = ?""",
                (sanitized, code, run_id),
            )

    def begin_tool_call(
        self,
        session_key: str,
        *,
        tool_name: str,
        workflow_id: str | None = None,
        plan_revision: int | None = None,
        step: str | None = None,
        flags: Sequence[str] = (),
    ) -> str | None:
        run_id = self._active_id(session_key)
        if run_id is None:
            return None
        event_id = uuid4().hex
        now = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_run_events (
                    event_id, run_id, parent_event_id, event_type, tool_name,
                    workflow_id, plan_revision, step, status, error_code,
                    retry_count, started_at_ms, completed_at_ms, flags
                ) VALUES (?, ?, NULL, 'tool_started', ?, ?, ?, ?, 'started', NULL, 0, ?, NULL, ?)""",
                (
                    event_id,
                    run_id,
                    tool_name[:200],
                    workflow_id,
                    plan_revision,
                    (step or tool_name)[:200],
                    now,
                    ",".join(flag for flag in flags if flag),
                ),
            )
        return event_id

    def finish_tool_call(
        self,
        event_id: str,
        *,
        status: str,
        error_code: str | AgentErrorCode | None = None,
        flags: Sequence[str] = (),
        workflow_id: str | None = None,
        plan_revision: int | None = None,
    ) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("tool status must be succeeded or failed")
        event_type = "tool_succeeded" if status == "succeeded" else "tool_failed"
        code = str(error_code) if error_code else None
        now = int(time.time() * 1000)
        merged_flags = ",".join(flag for flag in flags if flag)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT flags FROM agent_run_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                return
            existing = [flag for flag in str(row["flags"] or "").split(",") if flag]
            incoming = [flag for flag in merged_flags.split(",") if flag]
            combined = ",".join(dict.fromkeys([*existing, *incoming]))
            connection.execute(
                """UPDATE agent_run_events SET
                    event_type = ?, status = ?, error_code = ?, completed_at_ms = ?,
                    flags = ?,
                    workflow_id = COALESCE(?, workflow_id),
                    plan_revision = COALESCE(?, plan_revision)
                WHERE event_id = ?""",
                (event_type, status, code, now, combined, workflow_id, plan_revision, event_id),
            )

    def record_span(
        self,
        session_key: str,
        *,
        event_type: str,
        status: str,
        tool_name: str | None = None,
        workflow_id: str | None = None,
        plan_revision: int | None = None,
        error_code: str | AgentErrorCode | None = None,
        parent_event_id: str | None = None,
        flags: Sequence[str] = (),
    ) -> str | None:
        run_id = self._active_id(session_key)
        if run_id is None:
            return None
        event_id = uuid4().hex
        now = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_run_events (
                    event_id, run_id, parent_event_id, event_type, tool_name,
                    workflow_id, plan_revision, step, status, error_code,
                    retry_count, started_at_ms, completed_at_ms, flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (
                    event_id,
                    run_id,
                    parent_event_id,
                    event_type[:80],
                    tool_name[:200] if tool_name else None,
                    workflow_id,
                    plan_revision,
                    tool_name[:200] if tool_name else event_type[:80],
                    status[:40],
                    str(error_code) if error_code else None,
                    now,
                    now,
                    ",".join(flag for flag in flags if flag),
                ),
            )
        return event_id

    def aggregate(self, *, session_key: str | None = None) -> dict[str, Any]:
        query = "SELECT * FROM agent_runs"
        params: list[object] = []
        if session_key:
            query += " WHERE session_key = ?"
            params.append(session_key)
        with self._connect() as connection:
            run_rows = connection.execute(query, params).fetchall()
            if session_key:
                event_rows = connection.execute(
                    """SELECT e.* FROM agent_run_events e
                    JOIN agent_runs r ON r.run_id = e.run_id
                    WHERE r.session_key = ?""",
                    (session_key,),
                ).fetchall()
            else:
                event_rows = connection.execute("SELECT * FROM agent_run_events").fetchall()
        runs = [self._record(row) for row in run_rows]
        events = [self._event(row) for row in event_rows]
        finished = [run for run in runs if run.status in _FINISHED_STATUSES]
        succeeded_runs = [run for run in finished if run.status == "completed"]
        latencies = [run.latency_ms for run in finished if run.latency_ms is not None]
        tokens = [run.total_tokens for run in finished if run.total_tokens is not None]
        tool_totals = [run.tool_calls for run in finished]
        retries = [run.retries for run in finished]
        tool_done = [event for event in events if event.event_type in _TOOL_TERMINAL]
        tool_ok = [event for event in tool_done if event.event_type == "tool_succeeded"]
        recovery_terminal = [
            event.event_type
            for event in events
            if event.event_type in {"recovery_succeeded", "recovery_failed"}
        ]
        replan_terminal = [
            event.event_type
            for event in events
            if event.event_type in {"replan_accepted", "replan_exhausted"}
        ]
        run_ids_with = _run_ids_by_event(events)
        error_counts: dict[str, int] = {}
        for run in finished:
            if run.error_code:
                error_counts[run.error_code] = error_counts.get(run.error_code, 0) + 1
        for event in events:
            if event.error_code:
                error_counts[event.error_code] = error_counts.get(event.error_code, 0) + 1

        sample_count = len(finished)
        return {
            "sample_count": sample_count,
            "metrics": {
                "run_success_rate": _rate(len(succeeded_runs), sample_count),
                "avg_latency_ms": _mean(latencies),
                "p95_latency_ms": _percentile(latencies, 95),
                "avg_tokens": _mean(tokens),
                "avg_tool_calls": _mean(tool_totals),
                "avg_retries": _mean(retries),
                "retry_rate": _rate(sum(1 for run in finished if run.retries > 0), sample_count),
                "tool_success_rate": _rate(len(tool_ok), len(tool_done)),
                "recovery_success_rate": _rate(
                    recovery_terminal.count("recovery_succeeded"),
                    len(recovery_terminal),
                ),
                "replan_rate": _rate(len(run_ids_with["replan_started"]), sample_count),
                "replan_success_rate": _rate(
                    replan_terminal.count("replan_accepted"),
                    len(replan_terminal),
                ),
                "retrieval_fallback_rate": _rate(
                    len(run_ids_with["retrieval_fallback"]), sample_count
                ),
                "model_fallback_rate": _rate(
                    len(run_ids_with["model_fallback"]), sample_count
                ),
            },
            "error_counts": error_counts,
            "error_policy": {code.value: action for code, action in ERROR_POLICY.items()},
            "notes": (
                "P95 is a sample percentile for this workspace store; "
                "small n is not a production SLA."
            ),
        }

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


def _run_ids_by_event(events: Sequence[RunEvent]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {
        "replan_started": set(),
        "retrieval_fallback": set(),
        "model_fallback": set(),
    }
    for event in events:
        if event.event_type in grouped:
            grouped[event.event_type].add(event.run_id)
        if "replan" in event.flags.split(","):
            grouped["replan_started"].add(event.run_id)
        if "fallback" in event.flags.split(",") and event.event_type.startswith("tool_"):
            grouped["retrieval_fallback"].add(event.run_id)
    return grouped


def _mean(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]
