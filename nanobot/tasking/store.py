"""SQLite task store with idempotent creation and optimistic updates."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nanobot.tasking.models import Task, TaskCreate, TaskStatus, TaskUpdate

SCHEMA_VERSION = 1


class TaskNotFoundError(LookupError):
    pass


class TaskConflictError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dump_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class TaskStore:
    """Small process-safe SQLite repository for structured tasks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 2000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    due_at TEXT,
                    tags_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tasks_status_due_idx ON tasks(status, due_at);
                CREATE TABLE IF NOT EXISTS task_idempotency (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_mutation_receipts (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            task_id=row["task_id"],
            title=row["title"],
            description=row["description"],
            status=TaskStatus(row["status"]),
            priority=row["priority"],
            due_at=row["due_at"],
            tags=json.loads(row["tags_json"]),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
        )

    def create(self, request: TaskCreate, *, idempotency_key: str) -> Task:
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise ValueError("idempotency_key must contain 1-200 characters")
        payload = request.model_dump(mode="json")
        request_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT request_hash, task_id FROM task_idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if prior:
                if prior["request_hash"] != request_hash:
                    raise TaskConflictError("idempotency key was already used with another payload")
                row = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = ?", (prior["task_id"],)
                ).fetchone()
                if row is None:
                    raise TaskConflictError("idempotency record references a missing task")
                return self._row_to_task(row)

            task_id = str(uuid4())
            timestamp = _now().isoformat()
            connection.execute(
                """INSERT INTO tasks (
                    task_id, title, description, status, priority, due_at, tags_json,
                    source, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    task_id,
                    request.title,
                    request.description,
                    TaskStatus.TODO.value,
                    request.priority,
                    _dump_datetime(request.due_at),
                    json.dumps(request.tags, ensure_ascii=False),
                    request.source,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT INTO task_idempotency VALUES (?, 'create', ?, ?, ?)",
                (key, request_hash, task_id, timestamp),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            assert row is not None
            return self._row_to_task(row)

    def get(self, task_id: str) -> Task:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise TaskNotFoundError(f"task {task_id!r} was not found")
        return self._row_to_task(row)

    def get_created_by_idempotency_key(self, idempotency_key: str) -> Task | None:
        """Resolve an authoritative create receipt without replaying the mutation."""
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise ValueError("idempotency_key must contain 1-200 characters")
        with self._connect() as connection:
            receipt = connection.execute(
                """SELECT task_id FROM task_idempotency
                WHERE idempotency_key = ? AND operation = 'create'""",
                (key,),
            ).fetchone()
            if receipt is None:
                return None
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (receipt["task_id"],)
            ).fetchone()
        if row is None:
            raise TaskConflictError("idempotency record references a missing task")
        return self._row_to_task(row)

    def list(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Task]:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        if offset < 0 or offset > 10_000:
            raise ValueError("offset must be between 0 and 10000")
        query = "SELECT * FROM tasks"
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += (
            " ORDER BY status IN ('done', 'cancelled'), due_at IS NULL, due_at, "
            "priority, created_at LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_task(row) for row in rows]

    def update(
        self,
        task_id: str,
        request: TaskUpdate,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> Task:
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise ValueError("idempotency_key must contain 1-200 characters")
        changes = request.model_dump(exclude_none=True, exclude={"clear_due_at"})
        if request.clear_due_at:
            changes["due_at"] = None
        if not changes:
            return self.get(task_id)
        receipt_payload = {
            "task_id": task_id,
            "expected_version": expected_version,
            "changes": request.model_dump(mode="json"),
        }
        request_hash = hashlib.sha256(
            json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if "status" in changes:
            changes["status"] = changes["status"].value
        if "due_at" in changes:
            changes["due_at"] = _dump_datetime(changes["due_at"])
        if "tags" in changes:
            changes["tags_json"] = json.dumps(changes.pop("tags"), ensure_ascii=False)
        changes["updated_at"] = _now().isoformat()
        assignments = ", ".join(f"{column} = ?" for column in changes)
        values = [*changes.values(), task_id, expected_version]
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt = connection.execute(
                """SELECT request_hash, response_json FROM task_mutation_receipts
                WHERE idempotency_key = ? AND operation = 'update'""",
                (key,),
            ).fetchone()
            if receipt:
                if receipt["request_hash"] != request_hash:
                    raise TaskConflictError("idempotency key was already used with another payload")
                return Task.model_validate_json(receipt["response_json"])
            cursor = connection.execute(
                f"UPDATE tasks SET {assignments}, version = version + 1 "
                "WHERE task_id = ? AND version = ?",
                values,
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if exists is None:
                    raise TaskNotFoundError(f"task {task_id!r} was not found")
                raise TaskConflictError("task changed; reload it and retry with the latest version")
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            assert row is not None
            task = self._row_to_task(row)
            connection.execute(
                "INSERT INTO task_mutation_receipts VALUES (?, 'update', ?, ?, ?)",
                (key, request_hash, task.model_dump_json(), _now().isoformat()),
            )
            return task

    def cancel(
        self,
        task_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> Task:
        """Business deletion that preserves the task's audit trail."""

        return self.update(
            task_id,
            TaskUpdate(status=TaskStatus.CANCELLED),
            expected_version=expected_version,
            idempotency_key=idempotency_key,
        )
