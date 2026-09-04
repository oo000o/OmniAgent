"""Tools for durable, user-confirmed career-planning workflows."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from pathlib import Path

from pydantic import Field, ValidationError, field_validator

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import RequestContext, ToolContext
from nanobot.agent.tools.path_utils import resolve_workspace_path
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema
from nanobot.career import (
    CareerCheckpoint,
    CareerWorkflowConflictError,
    CareerWorkflowCreate,
    CareerWorkflowNotFoundError,
    CareerWorkflowState,
    CareerWorkflowStore,
    CareerWorkflowTransition,
)
from nanobot.config_base import Base


class CareerToolsConfig(Base):
    enable: bool = True
    database_path: str = Field(default=".nanobot/career.db", min_length=1)

    @field_validator("database_path")
    @classmethod
    def database_must_be_relative(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("database_path must stay inside the workspace")
        return value


class _CareerTool(Tool):
    config_key = "career"

    @classmethod
    def config_cls(cls) -> type[CareerToolsConfig]:
        return CareerToolsConfig

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.career.enable

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        return cls(workspace=Path(ctx.workspace), config=ctx.config.career)

    def __init__(self, *, workspace: Path, config: CareerToolsConfig) -> None:
        self._workspace = workspace.expanduser().resolve(strict=False)
        database_path = (self._workspace / config.database_path).resolve(strict=False)
        try:
            database_path.relative_to(self._workspace)
        except ValueError as exc:
            raise ValueError("career database must stay inside the workspace") from exc
        self._store = CareerWorkflowStore(database_path)
        self._store.initialize()

    def _document(self, path: str) -> Path:
        source = resolve_workspace_path(path, self._workspace, self._workspace)
        if not source.is_file():
            raise ValueError(f"document {path!r} was not found")
        return source


@tool_parameters(
    tool_parameters_schema(
        resume_path=StringSchema("Resume path inside the workspace.", min_length=1),
        jd_path=StringSchema("Job-description path inside the workspace.", min_length=1),
        idempotency_key=StringSchema(
            "Stable key for safely replaying workflow creation.", min_length=1, max_length=200
        ),
        required=["resume_path", "jd_path", "idempotency_key"],
    )
)
class CareerWorkflowStartTool(_CareerTool):
    @property
    def name(self) -> str:
        return "career_workflow_start"

    @property
    def description(self) -> str:
        return (
            "Start a durable resume/JD gap-analysis workflow. This records verified document "
            "paths but does not create learning tasks."
        )

    async def execute(self, resume_path: str, jd_path: str, idempotency_key: str) -> str:
        try:
            resume = self._document(resume_path)
            jd = self._document(jd_path)
            workflow = self._store.create(
                CareerWorkflowCreate(
                    resume_source=str(resume.relative_to(self._workspace)),
                    jd_source=str(jd.relative_to(self._workspace)),
                ),
                idempotency_key=idempotency_key,
            )
        except (CareerWorkflowConflictError, OSError, ValueError) as exc:
            return ToolResult.error(f"Career workflow creation failed: {exc}")
        return workflow.model_dump_json()


@tool_parameters(
    tool_parameters_schema(
        workflow_id=StringSchema("Workflow ID returned by career_workflow_start.", min_length=1),
        required=["workflow_id"],
    )
)
class CareerWorkflowGetTool(_CareerTool):
    @property
    def name(self) -> str:
        return "career_workflow_get"

    @property
    def description(self) -> str:
        return "Load the authoritative state, checkpoint, task IDs, and version of a workflow."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, workflow_id: str) -> str:
        try:
            return self._store.get(workflow_id).model_dump_json()
        except CareerWorkflowNotFoundError as exc:
            return ToolResult.error(str(exc))


@tool_parameters(
    tool_parameters_schema(
        workflow_id=StringSchema("Workflow ID to update.", min_length=1),
        target_state=StringSchema(
            "Next workflow state.",
            enum=[state.value for state in CareerWorkflowState],
        ),
        checkpoint_json=StringSchema(
            "Complete validated CareerCheckpoint JSON for the next state.",
            min_length=2,
            max_length=50_000,
        ),
        expected_version=IntegerSchema(
            description="Version returned by the latest workflow read.", minimum=1
        ),
        idempotency_key=StringSchema(
            "Stable key for safely replaying this exact transition.",
            min_length=1,
            max_length=200,
        ),
        required=[
            "workflow_id",
            "target_state",
            "checkpoint_json",
            "expected_version",
            "idempotency_key",
        ],
    )
)
class CareerWorkflowTransitionTool(_CareerTool):
    @property
    def name(self) -> str:
        return "career_workflow_transition"

    @property
    def description(self) -> str:
        return (
            "Persist one legal career-workflow transition with optimistic locking and "
            "idempotency. Reload after a version conflict; never skip user confirmation."
        )

    async def execute(
        self,
        workflow_id: str,
        target_state: str,
        checkpoint_json: str,
        expected_version: int,
        idempotency_key: str,
    ) -> str:
        try:
            protected = {
                CareerWorkflowState.TASKS_CREATING,
                CareerWorkflowState.FOLLOWUP_SCHEDULED,
                CareerWorkflowState.COMPLETED,
            }
            parsed_state = CareerWorkflowState(target_state)
            if parsed_state in protected:
                raise ValueError(
                    "protected state requires a dedicated tool backed by a real user or tool result"
                )
            checkpoint = CareerCheckpoint.model_validate_json(checkpoint_json)
            request = CareerWorkflowTransition(
                target_state=parsed_state,
                checkpoint=checkpoint,
            )
            workflow = self._store.transition(
                workflow_id,
                request,
                expected_version=expected_version,
                idempotency_key=idempotency_key,
            )
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            ValidationError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career workflow transition failed: {exc}")
        return workflow.model_dump_json()


@tool_parameters(
    tool_parameters_schema(
        workflow_id=StringSchema("Workflow ID whose displayed plan the user confirmed."),
        expected_version=IntegerSchema(
            description="Version of the workflow shown to the user.", minimum=1
        ),
        idempotency_key=StringSchema(
            "Stable key for safely replaying this confirmation.", min_length=1, max_length=200
        ),
        required=["workflow_id", "expected_version", "idempotency_key"],
    )
)
class CareerWorkflowConfirmTool(_CareerTool):
    """Accept confirmation only from the runtime-bound original user message."""

    _CONFIRMATION_PHRASE = "确认创建学习任务"

    def __init__(self, *, workspace: Path, config: CareerToolsConfig) -> None:
        super().__init__(workspace=workspace, config=config)
        self._request_context: RequestContext | None = None

    def set_context(self, ctx: RequestContext) -> None:
        self._request_context = ctx

    @property
    def name(self) -> str:
        return "career_workflow_confirm"

    @property
    def description(self) -> str:
        return (
            "Confirm the displayed learning plan. The current user message must explicitly "
            f"contain {self._CONFIRMATION_PHRASE!r}; model-generated arguments cannot replace it."
        )

    async def execute(
        self, workflow_id: str, expected_version: int, idempotency_key: str
    ) -> str:
        original_text = (
            self._request_context.original_user_text if self._request_context is not None else None
        )
        if not original_text or self._CONFIRMATION_PHRASE not in original_text:
            return ToolResult.error(
                "Career workflow confirmation failed: explicit confirmation was not present "
                "in the original user message"
            )
        try:
            current = self._store.get(workflow_id)
            checkpoint = current.checkpoint.model_copy(
                update={"confirmed": True, "error": None, "resume_state": None}
            )
            workflow = self._store.transition(
                workflow_id,
                CareerWorkflowTransition(
                    target_state=CareerWorkflowState.TASKS_CREATING,
                    checkpoint=checkpoint,
                ),
                expected_version=expected_version,
                idempotency_key=idempotency_key,
            )
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            ValidationError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career workflow confirmation failed: {exc}")
        return workflow.model_dump_json()
