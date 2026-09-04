"""Standalone stdio MCP server for persistent OmniAgent tasks."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import TypeAdapter

from nanobot.tasking.models import TaskCreate, TaskStatus, TaskUpdate
from nanobot.tasking.store import TaskStore

_DATETIME_ADAPTER = TypeAdapter(datetime)


def _parse_due_at(value: str | None) -> datetime | None:
    return None if value is None else _DATETIME_ADAPTER.validate_python(value)


def create_server(database_path: Path) -> FastMCP:
    store = TaskStore(database_path.expanduser().resolve(strict=False))
    store.initialize()
    server = FastMCP(
        "OmniAgent Tasks",
        instructions="Create and track durable personal tasks. Reuse idempotency keys on retries.",
    )

    @server.tool(description="Create one persistent task. Retried calls must reuse idempotency_key.")
    def task_create(
        title: str,
        idempotency_key: str,
        description: str = "",
        priority: int = 3,
        due_at: str | None = None,
        tags: list[str] | None = None,
        source: str = "agent",
    ) -> dict[str, Any]:
        request = TaskCreate(
            title=title,
            description=description,
            priority=priority,
            due_at=_parse_due_at(due_at),
            tags=tags or [],
            source=source,
        )
        return store.create(request, idempotency_key=idempotency_key).model_dump(mode="json")

    @server.tool(description="Get one task by its stable task_id.")
    def task_get(task_id: str) -> dict[str, Any]:
        return store.get(task_id).model_dump(mode="json")

    @server.tool(description="List tasks, optionally filtered by status.")
    def task_list(
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        parsed_status = TaskStatus(status) if status else None
        return [
            task.model_dump(mode="json")
            for task in store.list(status=parsed_status, limit=limit, offset=offset)
        ]

    @server.tool(description="Update a task using optimistic version checking.")
    def task_update(
        task_id: str,
        expected_version: int,
        idempotency_key: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        due_at: str | None = None,
        clear_due_at: bool = False,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        request = TaskUpdate(
            title=title,
            description=description,
            status=TaskStatus(status) if status else None,
            priority=priority,
            due_at=_parse_due_at(due_at),
            clear_due_at=clear_due_at,
            tags=tags,
        )
        return store.update(
            task_id,
            request,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
        ).model_dump(mode="json")

    @server.tool(
        description=(
            "Cancel a task while preserving its audit history. Retried calls must reuse "
            "idempotency_key."
        )
    )
    def task_cancel(
        task_id: str,
        expected_version: int,
        idempotency_key: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if not confirm:
            raise ValueError("task cancellation requires confirm=true")
        return store.cancel(
            task_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
        ).model_dump(mode="json")

    return server


def main() -> None:
    database = Path(os.environ.get("OMNIAGENT_TASK_DB", ".nanobot/tasks.db"))
    create_server(database).run(transport="stdio")


if __name__ == "__main__":
    main()
