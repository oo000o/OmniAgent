"""Tools for durable, user-confirmed career-planning workflows."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import Field, ValidationError, field_validator

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import RequestContext, ToolContext, current_request_context
from nanobot.agent.tools.knowledge import KnowledgeToolsConfig
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
    EvidenceReference,
)
from nanobot.config_base import Base
from nanobot.knowledge import (
    EmbeddingProvider,
    EmbeddingProviderError,
    HybridKnowledgeRetriever,
    KnowledgeSearchResult,
    KnowledgeStore,
    OpenAICompatibleEmbeddingProvider,
)
from nanobot.knowledge.ingest import KnowledgeIngestionError, ingest_document
from nanobot.session.keys import UNIFIED_SESSION_KEY
from nanobot.tasking import TaskConflictError, TaskNotFoundError, TaskStatus, TaskStore

if TYPE_CHECKING:
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob


class CareerToolsConfig(Base):
    enable: bool = True
    database_path: str = Field(default=".nanobot/career.db", min_length=1)
    task_database_path: str = Field(default=".nanobot/tasks.db", min_length=1)

    @field_validator("database_path", "task_database_path")
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
        task_database_path = (self._workspace / config.task_database_path).resolve(strict=False)
        try:
            task_database_path.relative_to(self._workspace)
        except ValueError as exc:
            raise ValueError("task database must stay inside the workspace") from exc
        self._task_store = TaskStore(task_database_path)
        self._task_store.initialize()

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
        workflow_id=StringSchema("Workflow ID whose resume and JD should be searched."),
        queries_json=StringSchema(
            "JSON array of 1-20 concise competency queries derived from the target JD.",
            min_length=2,
            max_length=10_000,
        ),
        expected_version=IntegerSchema(description="Latest workflow version.", minimum=1),
        idempotency_key=StringSchema(
            "Stable key for safely replaying this retrieval step.", min_length=1, max_length=200
        ),
        required=["workflow_id", "queries_json", "expected_version", "idempotency_key"],
    )
)
class CareerWorkflowRetrieveTool(_CareerTool):
    """Index the verified documents and persist only genuine retrieval evidence."""

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        return cls(
            workspace=Path(ctx.workspace),
            config=ctx.config.career,
            knowledge_config=ctx.config.knowledge,
        )

    def __init__(
        self,
        *,
        workspace: Path,
        config: CareerToolsConfig,
        knowledge_config: KnowledgeToolsConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        super().__init__(workspace=workspace, config=config)
        database_path = (self._workspace / knowledge_config.database_path).resolve(strict=False)
        self._knowledge_store = KnowledgeStore(database_path)
        self._knowledge_store.initialize()
        self._knowledge_config = knowledge_config
        if knowledge_config.retrieval_mode == "hybrid":
            provider = embedding_provider or OpenAICompatibleEmbeddingProvider(
                model=knowledge_config.embedding_model,
                api_key=knowledge_config.embedding_api_key,
                base_url=knowledge_config.embedding_base_url,
                dimensions=knowledge_config.embedding_dimensions,
                batch_size=knowledge_config.embedding_batch_size,
            )
            self._retriever: HybridKnowledgeRetriever | None = HybridKnowledgeRetriever(
                self._knowledge_store, provider
            )
        else:
            self._retriever = None

    @property
    def name(self) -> str:
        return "career_workflow_retrieve"

    @property
    def description(self) -> str:
        return (
            "Index and search the workflow's verified resume and JD, persist genuine chunk "
            "references, and return untrusted evidence for gap analysis."
        )

    async def execute(
        self,
        workflow_id: str,
        queries_json: str,
        expected_version: int,
        idempotency_key: str,
    ) -> str:
        try:
            parsed_queries: object = json.loads(queries_json)
            if not isinstance(parsed_queries, list):
                raise ValueError("queries_json must contain 1-20 non-empty strings")
            raw_queries = cast(list[object], parsed_queries)
            if not 1 <= len(raw_queries) <= 20:
                raise ValueError("queries_json must contain 1-20 non-empty strings")
            queries: list[str] = []
            for query in raw_queries:
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("queries_json must contain 1-20 non-empty strings")
                queries.append(query.strip())
            current = self._store.get(workflow_id)
            resume = self._document(current.resume_source)
            jd = self._document(current.jd_source)
            for source in (resume, jd):
                await self._index(source)

            allowed_paths: dict[Path, Literal["resume", "jd"]] = {
                resume.resolve(): "resume",
                jd.resolve(): "jd",
            }
            unique_results: dict[str, KnowledgeSearchResult] = {}
            for query in queries:
                for result in await self._search(query):
                    if result.source_path.resolve() in allowed_paths:
                        unique_results.setdefault(result.chunk.chunk_id, result)
            if not unique_results:
                raise ValueError("retrieval produced no evidence from the workflow documents")

            evidence_payload: list[dict[str, object]] = []
            references: list[EvidenceReference] = []
            for index, result in enumerate(unique_results.values(), 1):
                source_type = allowed_paths[result.source_path.resolve()]
                evidence_id = f"K{index}"
                references.append(
                    EvidenceReference(
                        evidence_id=evidence_id,
                        source_type=source_type,
                        source_name=result.source_name,
                        chunk_id=result.chunk.chunk_id,
                    )
                )
                evidence_payload.append(
                    {
                        "evidence_id": evidence_id,
                        "source_type": source_type,
                        "source_name": result.source_name,
                        "chunk_id": result.chunk.chunk_id,
                        "text": result.chunk.text,
                    }
                )
            checkpoint = current.checkpoint.model_copy(update={"evidence": references})
            workflow = self._store.transition(
                workflow_id,
                CareerWorkflowTransition(
                    target_state=CareerWorkflowState.EVIDENCE_RETRIEVED,
                    checkpoint=checkpoint,
                ),
                expected_version=expected_version,
                idempotency_key=idempotency_key,
            )
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            EmbeddingProviderError,
            KnowledgeIngestionError,
            OSError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career evidence retrieval failed: {exc}")
        return json.dumps(
            {"workflow": workflow.model_dump(mode="json"), "evidence": evidence_payload},
            ensure_ascii=False,
        )

    async def _index(self, source: Path) -> None:
        if self._retriever is None:
            ingest_document(source, self._knowledge_store)
            return
        try:
            await self._retriever.index_document(source)
        except EmbeddingProviderError:
            if not self._knowledge_config.fallback_to_lexical:
                raise
            ingest_document(source, self._knowledge_store)

    async def _search(self, query: str) -> list[KnowledgeSearchResult]:
        if self._retriever is None:
            return self._knowledge_store.search_lexical(
                query, limit=self._knowledge_config.candidate_results
            )
        try:
            fused = await self._retriever.search(
                query,
                limit=self._knowledge_config.candidate_results,
                candidate_limit=self._knowledge_config.candidate_results,
            )
            return [item.result for item in fused]
        except EmbeddingProviderError:
            if not self._knowledge_config.fallback_to_lexical:
                raise
            return self._knowledge_store.search_lexical(
                query, limit=self._knowledge_config.candidate_results
            )


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
                CareerWorkflowState.TASKS_CREATED,
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
        workflow_id=StringSchema("Workflow awaiting verification of MCP-created tasks."),
        task_ids_json=StringSchema(
            "JSON object mapping every learning-plan item_id to the task_id returned by MCP.",
            min_length=2,
            max_length=20_000,
        ),
        expected_version=IntegerSchema(description="Latest workflow version.", minimum=1),
        idempotency_key=StringSchema(
            "Stable key for safely replaying verification.", min_length=1, max_length=200
        ),
        required=["workflow_id", "task_ids_json", "expected_version", "idempotency_key"],
    )
)
class CareerWorkflowRecordTasksTool(_CareerTool):
    """Accept task IDs only after verifying authoritative MCP task storage."""

    @property
    def name(self) -> str:
        return "career_workflow_record_tasks"

    @property
    def description(self) -> str:
        return (
            "Verify MCP-returned task IDs against the shared task database, then checkpoint "
            "them. Each task source must be career:<workflow_id>:<plan_item_id>."
        )

    async def execute(
        self,
        workflow_id: str,
        task_ids_json: str,
        expected_version: int,
        idempotency_key: str,
    ) -> str:
        try:
            parsed_mapping: object = json.loads(task_ids_json)
            if not isinstance(parsed_mapping, dict):
                raise ValueError("task_ids_json must be a string-to-string JSON object")
            raw_mapping = cast(dict[object, object], parsed_mapping)
            task_ids: dict[str, str] = {}
            for key, value in raw_mapping.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("task_ids_json must be a string-to-string JSON object")
                task_ids[key] = value
            current = self._store.get(workflow_id)
            if current.state not in {
                CareerWorkflowState.TASKS_CREATING,
                CareerWorkflowState.TASKS_CREATED,
            }:
                raise CareerWorkflowConflictError("workflow is not creating tasks")
            plan = {item.item_id: item for item in current.checkpoint.plan}
            if set(task_ids) != set(plan):
                raise ValueError("every learning-plan item must have exactly one task ID")
            for item_id, task_id in task_ids.items():
                task = self._task_store.get(task_id)
                expected_source = f"career:{workflow_id}:{item_id}"
                if task.source != expected_source:
                    raise ValueError(f"task for plan item {item_id!r} has an invalid source")
                if task.title != plan[item_id].title:
                    raise ValueError(f"task for plan item {item_id!r} has an unexpected title")
            checkpoint = current.checkpoint.model_copy(update={"task_ids": task_ids})
            workflow = self._store.transition(
                workflow_id,
                CareerWorkflowTransition(
                    target_state=CareerWorkflowState.TASKS_CREATED,
                    checkpoint=checkpoint,
                ),
                expected_version=expected_version,
                idempotency_key=idempotency_key,
            )
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            TaskNotFoundError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career task verification failed: {exc}")
        return workflow.model_dump_json()


@tool_parameters(
    tool_parameters_schema(
        workflow_id=StringSchema("Confirmed workflow whose task creation should resume."),
        required=["workflow_id"],
    )
)
class CareerWorkflowTaskManifestTool(_CareerTool):
    """Reconstruct completed and pending MCP task_create calls from durable receipts."""

    @property
    def name(self) -> str:
        return "career_workflow_task_manifest"

    @property
    def description(self) -> str:
        return (
            "After explicit confirmation, inspect authoritative task receipts and return "
            "completed task IDs plus exact pending mcp_omniagent_tasks_task_create arguments. "
            "Call the pending MCP tools, then verify all returned IDs with "
            "career_workflow_record_tasks."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, workflow_id: str) -> str:
        try:
            current = self._store.get(workflow_id)
            if current.state is not CareerWorkflowState.TASKS_CREATING:
                raise CareerWorkflowConflictError("workflow is not creating tasks")
            completed: dict[str, str] = {}
            pending: list[dict[str, object]] = []
            for item in current.checkpoint.plan:
                key = f"career:{workflow_id}:{item.item_id}"
                task = self._task_store.get_created_by_idempotency_key(key)
                if task is None:
                    pending.append(
                        {
                            "plan_item_id": item.item_id,
                            "tool": "mcp_omniagent_tasks_task_create",
                            "arguments": {
                                "title": item.title,
                                "description": item.description,
                                "priority": item.priority,
                                "tags": ["career-plan"],
                                "source": key,
                                "idempotency_key": key,
                            },
                        }
                    )
                    continue
                if task.source != key or task.title != item.title:
                    raise ValueError(
                        f"task receipt for plan item {item.item_id!r} does not match the plan"
                    )
                completed[item.item_id] = task.task_id
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            TaskConflictError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career task manifest failed: {exc}")
        return json.dumps(
            {
                "workflow_id": workflow_id,
                "workflow_version": current.version,
                "completed_task_ids": completed,
                "pending_calls": pending,
                "next_step": (
                    "call every pending MCP tool, merge returned task IDs with "
                    "completed_task_ids, then call career_workflow_record_tasks"
                ),
            },
            ensure_ascii=False,
        )


class _CareerCronTool(_CareerTool):
    _plugin_discoverable = False

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return super().enabled(ctx) and ctx.cron_service is not None

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        if ctx.cron_service is None:
            raise RuntimeError("career cron tools require an initialized cron service")
        return cls(
            workspace=Path(ctx.workspace),
            config=ctx.config.career,
            cron_service=ctx.cron_service,
        )

    def __init__(
        self,
        *,
        workspace: Path,
        config: CareerToolsConfig,
        cron_service: CronService,
    ) -> None:
        super().__init__(workspace=workspace, config=config)
        self._cron = cron_service

    @staticmethod
    def _job_name(workflow_id: str) -> str:
        return f"career-followup:{workflow_id}"

    def _matching_jobs(self, workflow_id: str) -> list[CronJob]:
        name = self._job_name(workflow_id)
        return [
            job
            for job in self._cron.list_jobs(include_disabled=True)
            if job.name == name and job.payload.kind == "agent_turn"
        ]


@tool_parameters(
    tool_parameters_schema(
        workflow_id=StringSchema("Workflow whose created tasks should be checked periodically."),
        every_seconds=IntegerSchema(
            description="Follow-up interval in seconds (minimum 300).", minimum=300
        ),
        expected_version=IntegerSchema(description="Latest workflow version.", minimum=1),
        idempotency_key=StringSchema(
            "Stable key for scheduling or recovering this follow-up.",
            min_length=1,
            max_length=200,
        ),
        required=["workflow_id", "every_seconds", "expected_version", "idempotency_key"],
    )
)
class CareerWorkflowScheduleTool(_CareerCronTool):
    """Create or recover one channel-bound Cron job and checkpoint its real ID."""

    _plugin_discoverable = True

    @property
    def name(self) -> str:
        return "career_workflow_schedule"

    @property
    def description(self) -> str:
        return (
            "After verified task creation, schedule one recurring progress check bound to "
            "the current chat. Replays recover the persisted Cron job instead of duplicating it."
        )

    async def execute(
        self,
        workflow_id: str,
        every_seconds: int,
        expected_version: int,
        idempotency_key: str,
    ) -> str:
        try:
            if every_seconds < 300:
                raise ValueError("every_seconds must be at least 300")
            current = self._store.get(workflow_id)
            if current.state not in {
                CareerWorkflowState.TASKS_CREATED,
                CareerWorkflowState.FOLLOWUP_SCHEDULED,
            }:
                raise CareerWorkflowConflictError("workflow tasks have not been created")
            ctx = current_request_context()
            if ctx is None or not ctx.channel or not ctx.chat_id:
                raise ValueError("follow-up scheduling requires an originating chat")
            raw_key = f"{ctx.channel}:{ctx.chat_id}"
            session_key = raw_key if ctx.session_key == UNIFIED_SESSION_KEY else ctx.session_key
            if not session_key:
                session_key = raw_key
            jobs = self._matching_jobs(workflow_id)
            if len(jobs) > 1:
                raise CareerWorkflowConflictError("multiple follow-up jobs require manual repair")
            message = (
                f"Check career workflow {workflow_id}. Read its authoritative checkpoint, "
                "call task_get for every persisted task ID, report progress to this chat, and "
                "call career_workflow_complete only when every task is done."
            )
            if jobs:
                job = jobs[0]
                if (
                    job.schedule.kind != "every"
                    or job.schedule.every_ms != every_seconds * 1000
                    or job.payload.message != message
                    or job.payload.origin_channel != ctx.channel
                    or job.payload.origin_chat_id != ctx.chat_id
                ):
                    raise CareerWorkflowConflictError(
                        "existing follow-up job does not match this workflow request"
                    )
            else:
                from nanobot.cron.types import CronSchedule

                job = self._cron.add_job(
                    name=self._job_name(workflow_id),
                    schedule=CronSchedule(kind="every", every_ms=every_seconds * 1000),
                    message=message,
                    session_key=session_key,
                    origin_channel=ctx.channel,
                    origin_chat_id=ctx.chat_id,
                    origin_metadata=dict(ctx.metadata or {}),
                )
            checkpoint = current.checkpoint.model_copy(update={"followup_job_id": job.id})
            workflow = self._store.transition(
                workflow_id,
                CareerWorkflowTransition(
                    target_state=CareerWorkflowState.FOLLOWUP_SCHEDULED,
                    checkpoint=checkpoint,
                ),
                expected_version=expected_version,
                idempotency_key=idempotency_key,
            )
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career follow-up scheduling failed: {exc}")
        return workflow.model_dump_json()


@tool_parameters(
    tool_parameters_schema(
        workflow_id=StringSchema("Scheduled workflow whose tasks may all be complete."),
        expected_version=IntegerSchema(description="Latest workflow version.", minimum=1),
        idempotency_key=StringSchema(
            "Stable key for completing this workflow.", min_length=1, max_length=200
        ),
        required=["workflow_id", "expected_version", "idempotency_key"],
    )
)
class CareerWorkflowCompleteTool(_CareerCronTool):
    """Complete only after authoritative tasks are done, then clean up the Cron job."""

    _plugin_discoverable = True

    @property
    def name(self) -> str:
        return "career_workflow_complete"

    @property
    def description(self) -> str:
        return (
            "Verify every checkpointed task is done, persist workflow completion, and remove "
            "its recurring Cron job. Safe to replay after interruption."
        )

    async def execute(
        self, workflow_id: str, expected_version: int, idempotency_key: str
    ) -> str:
        try:
            current = self._store.get(workflow_id)
            if current.state not in {
                CareerWorkflowState.FOLLOWUP_SCHEDULED,
                CareerWorkflowState.COMPLETED,
            }:
                raise CareerWorkflowConflictError("workflow has no scheduled follow-up")
            if any(
                self._task_store.get(task_id).status is not TaskStatus.DONE
                for task_id in current.checkpoint.task_ids.values()
            ):
                raise CareerWorkflowConflictError("not all learning tasks are done")
            job_id = current.checkpoint.followup_job_id
            if not job_id:
                raise ValueError("workflow has no persisted follow-up job ID")
            job = self._cron.get_job(job_id)
            if current.state is CareerWorkflowState.FOLLOWUP_SCHEDULED:
                if job is None or job.name != self._job_name(workflow_id):
                    raise CareerWorkflowConflictError(
                        "persisted follow-up job is missing or belongs to another workflow"
                    )
                workflow = self._store.transition(
                    workflow_id,
                    CareerWorkflowTransition(
                        target_state=CareerWorkflowState.COMPLETED,
                        checkpoint=current.checkpoint,
                    ),
                    expected_version=expected_version,
                    idempotency_key=idempotency_key,
                )
            else:
                workflow = self._store.transition(
                    workflow_id,
                    CareerWorkflowTransition(
                        target_state=CareerWorkflowState.COMPLETED,
                        checkpoint=current.checkpoint,
                    ),
                    expected_version=expected_version,
                    idempotency_key=idempotency_key,
                )
            removal = self._cron.remove_job(job_id)
            if removal not in {"removed", "not_found"}:
                raise RuntimeError("completed workflow follow-up job could not be removed")
        except (
            CareerWorkflowConflictError,
            CareerWorkflowNotFoundError,
            OSError,
            RuntimeError,
            TaskNotFoundError,
            ValueError,
        ) as exc:
            return ToolResult.error(f"Career workflow completion failed: {exc}")
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
