"""SQLite store and transition rules for resumable career workflows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nanobot.career.models import (
    CareerCheckpoint,
    CareerWorkflow,
    CareerWorkflowCreate,
    CareerWorkflowState,
    CareerWorkflowTransition,
)


class CareerWorkflowNotFoundError(LookupError):
    pass


class CareerWorkflowConflictError(RuntimeError):
    pass


_ALLOWED_TRANSITIONS: dict[CareerWorkflowState, frozenset[CareerWorkflowState]] = {
    CareerWorkflowState.DOCUMENTS_READY: frozenset(
        {CareerWorkflowState.EVIDENCE_RETRIEVED, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.EVIDENCE_RETRIEVED: frozenset(
        {CareerWorkflowState.GAP_READY, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.GAP_READY: frozenset(
        {CareerWorkflowState.AWAITING_CONFIRMATION, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.AWAITING_CONFIRMATION: frozenset(
        {CareerWorkflowState.TASKS_CREATING, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.TASKS_CREATING: frozenset(
        {CareerWorkflowState.TASKS_CREATED, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.TASKS_CREATED: frozenset(
        {CareerWorkflowState.FOLLOWUP_SCHEDULED, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.FOLLOWUP_SCHEDULED: frozenset(
        {CareerWorkflowState.COMPLETED, CareerWorkflowState.FAILED}
    ),
    CareerWorkflowState.COMPLETED: frozenset(),
    CareerWorkflowState.FAILED: frozenset(
        {
            CareerWorkflowState.DOCUMENTS_READY,
            CareerWorkflowState.EVIDENCE_RETRIEVED,
            CareerWorkflowState.GAP_READY,
            CareerWorkflowState.AWAITING_CONFIRMATION,
            CareerWorkflowState.TASKS_CREATING,
            CareerWorkflowState.TASKS_CREATED,
            CareerWorkflowState.FOLLOWUP_SCHEDULED,
        }
    ),
}


class CareerWorkflowStore:
    """Persist versioned checkpoints and reject unsafe state transitions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

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
                CREATE TABLE IF NOT EXISTS career_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    resume_source TEXT NOT NULL,
                    jd_source TEXT NOT NULL,
                    state TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS career_workflows_state_updated_idx
                    ON career_workflows(state, updated_at DESC);
                CREATE TABLE IF NOT EXISTS career_workflow_receipts (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _row_to_workflow(row: sqlite3.Row) -> CareerWorkflow:
        return CareerWorkflow(
            workflow_id=row["workflow_id"],
            resume_source=row["resume_source"],
            jd_source=row["jd_source"],
            state=CareerWorkflowState(row["state"]),
            checkpoint=CareerCheckpoint.model_validate_json(row["checkpoint_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
        )

    @staticmethod
    def _request_hash(payload: object) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _validate_key(idempotency_key: str) -> str:
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise ValueError("idempotency_key must contain 1-200 characters")
        return key

    def create(
        self, request: CareerWorkflowCreate, *, idempotency_key: str
    ) -> CareerWorkflow:
        key = self._validate_key(idempotency_key)
        request_hash = self._request_hash(request.model_dump(mode="json"))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay(connection, key, "create", request_hash)
            if replay is not None:
                return replay
            now = datetime.now(timezone.utc)
            workflow = CareerWorkflow(
                workflow_id=str(uuid4()),
                resume_source=request.resume_source,
                jd_source=request.jd_source,
                state=CareerWorkflowState.DOCUMENTS_READY,
                checkpoint=CareerCheckpoint(),
                created_at=now,
                updated_at=now,
                version=1,
            )
            connection.execute(
                """INSERT INTO career_workflows VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    workflow.workflow_id,
                    workflow.resume_source,
                    workflow.jd_source,
                    workflow.state.value,
                    workflow.checkpoint.model_dump_json(),
                    workflow.created_at.isoformat(),
                    workflow.updated_at.isoformat(),
                    workflow.version,
                ),
            )
            self._save_receipt(connection, key, "create", request_hash, workflow)
            return workflow

    def get(self, workflow_id: str) -> CareerWorkflow:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM career_workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
        if row is None:
            raise CareerWorkflowNotFoundError(f"workflow {workflow_id!r} was not found")
        return self._row_to_workflow(row)

    def transition(
        self,
        workflow_id: str,
        request: CareerWorkflowTransition,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> CareerWorkflow:
        key = self._validate_key(idempotency_key)
        payload = {
            "workflow_id": workflow_id,
            "expected_version": expected_version,
            "request": request.model_dump(mode="json"),
        }
        request_hash = self._request_hash(payload)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay(connection, key, "transition", request_hash)
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM career_workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise CareerWorkflowNotFoundError(f"workflow {workflow_id!r} was not found")
            current = self._row_to_workflow(row)
            if current.version != expected_version:
                raise CareerWorkflowConflictError(
                    "workflow changed; reload it and retry with the latest version"
                )
            if request.target_state not in _ALLOWED_TRANSITIONS[current.state]:
                raise CareerWorkflowConflictError(
                    f"invalid workflow transition: {current.state.value} -> "
                    f"{request.target_state.value}"
                )
            if request.target_state is CareerWorkflowState.FAILED:
                if request.checkpoint.resume_state is not current.state:
                    raise ValueError("failed checkpoint must preserve the state to resume")
            elif current.state is CareerWorkflowState.FAILED:
                if request.target_state is not current.checkpoint.resume_state:
                    raise CareerWorkflowConflictError(
                        "failed workflow may only resume its recorded state"
                    )
            self._validate_checkpoint(request.target_state, request.checkpoint)
            updated_at = datetime.now(timezone.utc)
            connection.execute(
                """UPDATE career_workflows
                SET state = ?, checkpoint_json = ?, updated_at = ?, version = version + 1
                WHERE workflow_id = ? AND version = ?""",
                (
                    request.target_state.value,
                    request.checkpoint.model_dump_json(),
                    updated_at.isoformat(),
                    workflow_id,
                    expected_version,
                ),
            )
            workflow = CareerWorkflow(
                workflow_id=current.workflow_id,
                resume_source=current.resume_source,
                jd_source=current.jd_source,
                state=request.target_state,
                checkpoint=request.checkpoint,
                created_at=current.created_at,
                updated_at=updated_at,
                version=current.version + 1,
            )
            self._save_receipt(connection, key, "transition", request_hash, workflow)
            return workflow

    @staticmethod
    def _validate_checkpoint(state: CareerWorkflowState, checkpoint: CareerCheckpoint) -> None:
        if state in {
            CareerWorkflowState.EVIDENCE_RETRIEVED,
            CareerWorkflowState.GAP_READY,
            CareerWorkflowState.AWAITING_CONFIRMATION,
            CareerWorkflowState.TASKS_CREATING,
            CareerWorkflowState.TASKS_CREATED,
            CareerWorkflowState.FOLLOWUP_SCHEDULED,
            CareerWorkflowState.COMPLETED,
        } and not checkpoint.evidence:
            raise ValueError("retrieved workflow states require evidence")
        if state in {
            CareerWorkflowState.GAP_READY,
            CareerWorkflowState.AWAITING_CONFIRMATION,
            CareerWorkflowState.TASKS_CREATING,
            CareerWorkflowState.TASKS_CREATED,
            CareerWorkflowState.FOLLOWUP_SCHEDULED,
            CareerWorkflowState.COMPLETED,
        } and not checkpoint.gaps:
            raise ValueError("gap workflow states require gap items")
        if state in {
            CareerWorkflowState.AWAITING_CONFIRMATION,
            CareerWorkflowState.TASKS_CREATING,
            CareerWorkflowState.TASKS_CREATED,
            CareerWorkflowState.FOLLOWUP_SCHEDULED,
            CareerWorkflowState.COMPLETED,
        } and not checkpoint.plan:
            raise ValueError("planning workflow states require learning plan items")
        if state in {
            CareerWorkflowState.TASKS_CREATING,
            CareerWorkflowState.TASKS_CREATED,
            CareerWorkflowState.FOLLOWUP_SCHEDULED,
            CareerWorkflowState.COMPLETED,
        } and not checkpoint.confirmed:
            raise ValueError("task creation requires user confirmation")
        if state in {
            CareerWorkflowState.TASKS_CREATED,
            CareerWorkflowState.FOLLOWUP_SCHEDULED,
            CareerWorkflowState.COMPLETED,
        } and set(checkpoint.task_ids) != {item.item_id for item in checkpoint.plan}:
            raise ValueError("every plan item must have a persisted task ID")
        if state is CareerWorkflowState.FOLLOWUP_SCHEDULED and not checkpoint.followup_job_id:
            raise ValueError("scheduled follow-up requires a job ID")
        if state is CareerWorkflowState.FAILED and not checkpoint.error:
            raise ValueError("failed workflows require an error")
        if state is CareerWorkflowState.FAILED and checkpoint.resume_state is None:
            raise ValueError("failed workflows require a resume state")

    def _replay(
        self,
        connection: sqlite3.Connection,
        key: str,
        operation: str,
        request_hash: str,
    ) -> CareerWorkflow | None:
        row = connection.execute(
            "SELECT operation, request_hash, response_json FROM career_workflow_receipts "
            "WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["operation"] != operation or row["request_hash"] != request_hash:
            raise CareerWorkflowConflictError(
                "idempotency key was already used with another operation or payload"
            )
        return CareerWorkflow.model_validate_json(row["response_json"])

    @staticmethod
    def _save_receipt(
        connection: sqlite3.Connection,
        key: str,
        operation: str,
        request_hash: str,
        workflow: CareerWorkflow,
    ) -> None:
        connection.execute(
            "INSERT INTO career_workflow_receipts VALUES (?, ?, ?, ?, ?)",
            (
                key,
                operation,
                request_hash,
                workflow.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
