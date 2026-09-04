from datetime import datetime, timezone

import pytest

from nanobot.tasking import (
    TaskConflictError,
    TaskCreate,
    TaskNotFoundError,
    TaskStatus,
    TaskStore,
    TaskUpdate,
)


def _store(tmp_path) -> TaskStore:
    store = TaskStore(tmp_path / "tasks.db")
    store.initialize()
    return store


def test_create_is_idempotent_for_same_key_and_payload(tmp_path) -> None:
    store = _store(tmp_path)
    request = TaskCreate(title="Review RAG", tags=["AI", "ai"])

    first = store.create(request, idempotency_key="turn-1-call-1")
    second = store.create(request, idempotency_key="turn-1-call-1")

    assert first.task_id == second.task_id
    assert first.tags == ["ai"]
    assert len(store.list()) == 1


def test_create_receipt_can_be_resolved_without_replaying_mutation(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(TaskCreate(title="Review RAG"), idempotency_key="career:rag")

    assert store.get_created_by_idempotency_key("career:rag") == task
    assert store.get_created_by_idempotency_key("career:missing") is None


def test_reused_key_with_different_payload_conflicts(tmp_path) -> None:
    store = _store(tmp_path)
    store.create(TaskCreate(title="First"), idempotency_key="same")

    with pytest.raises(TaskConflictError):
        store.create(TaskCreate(title="Different"), idempotency_key="same")


def test_update_uses_optimistic_version(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(TaskCreate(title="Draft"), idempotency_key="create-draft")

    updated = store.update(
        task.task_id,
        TaskUpdate(status=TaskStatus.DONE),
        expected_version=1,
        idempotency_key="finish-draft",
    )

    assert updated.status is TaskStatus.DONE
    assert updated.version == 2
    with pytest.raises(TaskConflictError):
        store.update(
            task.task_id,
            TaskUpdate(title="Stale"),
            expected_version=1,
            idempotency_key="stale-update",
        )


def test_due_at_requires_timezone() -> None:
    with pytest.raises(ValueError):
        TaskCreate(title="Bad date", due_at=datetime(2026, 9, 3, 20, 0))
    task = TaskCreate(
        title="Good date", due_at=datetime(2026, 9, 3, 20, 0, tzinfo=timezone.utc)
    )
    assert task.due_at is not None and task.due_at.tzinfo is not None


def test_list_filters_status_and_missing_get_raises(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(TaskCreate(title="Do it"), idempotency_key="do-it")
    store.update(
        task.task_id,
        TaskUpdate(status=TaskStatus.DONE),
        expected_version=1,
        idempotency_key="complete-do-it",
    )

    assert [item.task_id for item in store.list(status=TaskStatus.DONE)] == [task.task_id]
    assert store.list(status=TaskStatus.TODO) == []
    with pytest.raises(TaskNotFoundError):
        store.get("missing")


def test_update_and_cancel_are_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(TaskCreate(title="Draft"), idempotency_key="new-draft")

    first = store.update(
        task.task_id,
        TaskUpdate(title="Final"),
        expected_version=1,
        idempotency_key="rename-draft",
    )
    replay = store.update(
        task.task_id,
        TaskUpdate(title="Final"),
        expected_version=1,
        idempotency_key="rename-draft",
    )
    cancelled = store.cancel(
        task.task_id,
        expected_version=2,
        idempotency_key="cancel-draft",
    )

    assert first == replay
    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.version == 3


def test_list_supports_stable_pagination(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.create(TaskCreate(title="First"), idempotency_key="page-first")
    second = store.create(TaskCreate(title="Second"), idempotency_key="page-second")

    assert store.list(limit=1)[0].task_id == first.task_id
    assert store.list(limit=1, offset=1)[0].task_id == second.task_id
