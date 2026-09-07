import json

from nanobot.agent.tools.career import (
    CareerToolsConfig,
    CareerWorkflowCompleteTool,
    CareerWorkflowConfirmTool,
    CareerWorkflowGetTool,
    CareerWorkflowRecordTasksTool,
    CareerWorkflowReplanTool,
    CareerWorkflowRetrieveTool,
    CareerWorkflowScheduleTool,
    CareerWorkflowStartTool,
    CareerWorkflowTaskManifestTool,
    CareerWorkflowTransitionTool,
    CareerWorkflowVerifyPlanTool,
)
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.knowledge import KnowledgeToolsConfig
from nanobot.career import (
    CareerCheckpoint,
    CareerWorkflow,
    CareerWorkflowCreate,
    CareerWorkflowState,
    CareerWorkflowStore,
    CareerWorkflowTransition,
    EvidenceReference,
    GapItem,
    GapStatus,
    LearningPlanItem,
)
from nanobot.cron.service import CronService
from nanobot.knowledge.embeddings import EmbeddingProviderError
from nanobot.session.keys import UNIFIED_SESSION_KEY
from nanobot.tasking import TaskCreate, TaskStatus, TaskStore, TaskUpdate


async def test_career_tools_start_replay_and_load(tmp_path) -> None:
    (tmp_path / "resume.md").write_text("Python project experience", encoding="utf-8")
    (tmp_path / "jd.md").write_text("RAG evaluation required", encoding="utf-8")
    config = CareerToolsConfig(database_path="state/career.db")
    start = CareerWorkflowStartTool(workspace=tmp_path, config=config)
    get = CareerWorkflowGetTool(workspace=tmp_path, config=config)

    first = json.loads(await start.execute("resume.md", "jd.md", "career-demo"))
    replay = json.loads(await start.execute("resume.md", "jd.md", "career-demo"))
    loaded = json.loads(await get.execute(first["workflow_id"]))

    assert replay["workflow_id"] == first["workflow_id"]
    assert loaded["state"] == CareerWorkflowState.DOCUMENTS_READY
    assert loaded["version"] == 1


async def test_career_start_rejects_documents_outside_workspace(tmp_path) -> None:
    config = CareerToolsConfig(database_path="state/career.db")
    start = CareerWorkflowStartTool(workspace=tmp_path, config=config)

    result = await start.execute("../resume.md", "jd.md", "career-demo")

    assert result.startswith("VALIDATION_ERROR:")
    assert "Career workflow creation failed:" in result


async def test_transition_tool_rejects_skipping_confirmation(tmp_path) -> None:
    (tmp_path / "resume.md").write_text("Python project experience", encoding="utf-8")
    (tmp_path / "jd.md").write_text("RAG evaluation required", encoding="utf-8")
    config = CareerToolsConfig(database_path="state/career.db")
    start = CareerWorkflowStartTool(workspace=tmp_path, config=config)
    transition = CareerWorkflowTransitionTool(workspace=tmp_path, config=config)
    workflow = json.loads(await start.execute("resume.md", "jd.md", "career-demo"))

    result = await transition.execute(
        workflow["workflow_id"],
        CareerWorkflowState.GAP_READY.value,
        """{"evidence":[{"evidence_id":"K1","source_type":"jd","source_name":"jd.md","chunk_id":"c1"}],
        "gaps":[{"competency":"RAG","status":"missing","rationale":"No evidence",
        "evidence_ids":["K1"]}]}""",
        1,
        "unsafe-skip",
    )

    assert result.startswith("VALIDATION_ERROR:")
    assert "Career workflow transition failed:" in result
    assert "invalid workflow transition" in result


async def test_retrieval_persists_only_real_resume_and_jd_chunks(tmp_path) -> None:
    (tmp_path / "resume.md").write_text(
        "Built Python APIs and Docker deployment automation.", encoding="utf-8"
    )
    (tmp_path / "jd.md").write_text(
        "The role requires Python, RAG evaluation, and Docker.", encoding="utf-8"
    )
    career_config = CareerToolsConfig(database_path="state/career.db")
    start = CareerWorkflowStartTool(workspace=tmp_path, config=career_config)
    workflow = json.loads(await start.execute("resume.md", "jd.md", "career-evidence"))
    retrieve = CareerWorkflowRetrieveTool(
        workspace=tmp_path,
        config=career_config,
        knowledge_config=KnowledgeToolsConfig(
            database_path="state/knowledge.db",
            retrieval_mode="lexical",
            candidate_results=10,
        ),
    )

    result = json.loads(
        await retrieve.execute(
            workflow["workflow_id"],
            json.dumps(["Python", "RAG evaluation", "Docker"]),
            workflow["version"],
            "retrieve-1",
        )
    )

    assert result["workflow"]["state"] == CareerWorkflowState.EVIDENCE_RETRIEVED
    assert result["retrieval_fallback"] is False
    assert {item["source_type"] for item in result["evidence"]} == {"resume", "jd"}
    assert all(item["chunk_id"] for item in result["evidence"])
    persisted = result["workflow"]["checkpoint"]["evidence"]
    assert {item["chunk_id"] for item in persisted} == {
        item["chunk_id"] for item in result["evidence"]
    }

    replay = json.loads(
        await retrieve.execute(
            workflow["workflow_id"],
            json.dumps(["Python", "RAG evaluation", "Docker"]),
            workflow["version"],
            "retrieve-1",
        )
    )
    assert replay["workflow"]["version"] == result["workflow"]["version"]


class _FailingEmbeddingProvider:
    @property
    def model_name(self) -> str:
        return "failing-v1"

    async def embed(self, texts):
        raise EmbeddingProviderError("backend unavailable")


async def test_retrieve_records_lexical_fallback_without_raising(tmp_path) -> None:
    (tmp_path / "resume.md").write_text(
        "Built Python APIs and Docker deployment automation.", encoding="utf-8"
    )
    (tmp_path / "jd.md").write_text(
        "The role requires Python, RAG evaluation, and Docker.", encoding="utf-8"
    )
    career_config = CareerToolsConfig(database_path="state/career.db")
    start = CareerWorkflowStartTool(workspace=tmp_path, config=career_config)
    workflow = json.loads(await start.execute("resume.md", "jd.md", "career-fallback"))
    retrieve = CareerWorkflowRetrieveTool(
        workspace=tmp_path,
        config=career_config,
        knowledge_config=KnowledgeToolsConfig(
            database_path="state/knowledge.db",
            retrieval_mode="hybrid",
            embedding_model="failing-v1",
            candidate_results=10,
        ),
        embedding_provider=_FailingEmbeddingProvider(),
    )

    result = json.loads(
        await retrieve.execute(
            workflow["workflow_id"],
            json.dumps(["Python", "RAG evaluation"]),
            workflow["version"],
            "retrieve-fallback",
        )
    )

    assert result["retrieval_fallback"] is True
    assert result["workflow"]["state"] == CareerWorkflowState.EVIDENCE_RETRIEVED


async def test_retrieved_evidence_drives_gap_and_plan_checkpoints(tmp_path) -> None:
    (tmp_path / "resume.md").write_text(
        "Built Python APIs and Docker deployment automation.", encoding="utf-8"
    )
    (tmp_path / "jd.md").write_text(
        "The role requires Python, RAG evaluation, and Docker.", encoding="utf-8"
    )
    config = CareerToolsConfig(database_path="state/career.db")
    start = CareerWorkflowStartTool(workspace=tmp_path, config=config)
    retrieve = CareerWorkflowRetrieveTool(
        workspace=tmp_path,
        config=config,
        knowledge_config=KnowledgeToolsConfig(
            database_path="state/knowledge.db", retrieval_mode="lexical"
        ),
    )
    transition = CareerWorkflowTransitionTool(workspace=tmp_path, config=config)
    workflow = json.loads(await start.execute("resume.md", "jd.md", "career-flow"))
    retrieved = json.loads(
        await retrieve.execute(
            workflow["workflow_id"],
            json.dumps(["Python Docker", "RAG evaluation"]),
            workflow["version"],
            "retrieve-flow",
        )
    )["workflow"]
    evidence = retrieved["checkpoint"]["evidence"]
    resume_id = next(item["evidence_id"] for item in evidence if item["source_type"] == "resume")
    jd_id = next(item["evidence_id"] for item in evidence if item["source_type"] == "jd")
    gap_checkpoint = {
        **retrieved["checkpoint"],
        "gaps": [
            {
                "competency": "Python and Docker",
                "status": "demonstrated",
                "rationale": "The resume evidence matches the JD requirement.",
                "evidence_ids": [resume_id, jd_id],
            },
            {
                "competency": "RAG evaluation",
                "status": "missing",
                "rationale": "The JD requires evaluation without resume evidence.",
                "evidence_ids": [jd_id],
            },
        ],
    }
    gap_ready = json.loads(
        await transition.execute(
            retrieved["workflow_id"],
            CareerWorkflowState.GAP_READY.value,
            json.dumps(gap_checkpoint),
            retrieved["version"],
            "gap-flow",
        )
    )
    plan_checkpoint = {
        **gap_ready["checkpoint"],
        "plan": [
                {
                    "item_id": "rag-eval",
                    "title": "Build a reproducible RAG evaluation set",
                    "priority": 1,
                    "addresses": ["RAG evaluation"],
                }
            ],
        }
    verifying = json.loads(
        await transition.execute(
            gap_ready["workflow_id"],
            CareerWorkflowState.PLAN_VERIFYING.value,
            json.dumps(plan_checkpoint),
            gap_ready["version"],
            "plan-flow",
        )
    )
    verifier = CareerWorkflowVerifyPlanTool(workspace=tmp_path, config=config)
    awaiting = json.loads(
        await verifier.execute(
            verifying["workflow_id"], verifying["version"], "verify-plan-flow"
        )
    )

    assert awaiting["state"] == CareerWorkflowState.AWAITING_CONFIRMATION
    assert awaiting["checkpoint"]["evidence"] == evidence
    assert awaiting["checkpoint"]["gaps"] == gap_checkpoint["gaps"]


async def test_bounded_replan_records_diff_and_binds_confirmation_revision(tmp_path) -> None:
    config = CareerToolsConfig(database_path="state/career.db")
    store = CareerWorkflowStore(tmp_path / config.database_path)
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="bounded-workflow",
    )
    checkpoint = CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ],
        gaps=[
            GapItem(
                competency="RAG evaluation",
                status=GapStatus.MISSING,
                rationale="Required by the JD.",
                evidence_ids=["K1"],
            )
        ],
        plan=[LearningPlanItem(item_id="generic", title="Read documentation")],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "bounded-evidence"),
        (CareerWorkflowState.GAP_READY, "bounded-gap"),
        (CareerWorkflowState.PLAN_VERIFYING, "bounded-plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=key,
        )

    verifier = CareerWorkflowVerifyPlanTool(workspace=tmp_path, config=config)
    rejected = json.loads(
        await verifier.execute(current.workflow_id, current.version, "bounded-reject")
    )
    assert rejected["state"] == CareerWorkflowState.REPLANNING
    assert any(
        "uncovered required competencies" in error
        for error in rejected["checkpoint"]["verification_errors"]
    )

    revised_plan = [
        {
            "item_id": "rag-eval",
            "title": "Build a RAG evaluation set",
            "description": "Measure Recall@K and MRR.",
            "priority": 1,
            "addresses": ["RAG evaluation"],
        }
    ]
    replan = CareerWorkflowReplanTool(workspace=tmp_path, config=config)
    revised = json.loads(
        await replan.execute(
            current.workflow_id,
            json.dumps(revised_plan),
            "uncovered-required-competency",
            "Replace an ungrounded task with an evidence-bound evaluation task.",
            rejected["version"],
            "bounded-replan-1",
        )
    )
    replay = json.loads(
        await replan.execute(
            current.workflow_id,
            json.dumps(revised_plan),
            "uncovered-required-competency",
            "Replace an ungrounded task with an evidence-bound evaluation task.",
            rejected["version"],
            "bounded-replan-1",
        )
    )
    assert replay == revised
    assert revised["checkpoint"]["plan_revision"] == 2
    assert revised["checkpoint"]["replan_count"] == 1
    diff = revised["checkpoint"]["replan_history"][0]
    assert diff["added_item_ids"] == ["rag-eval"]
    assert diff["removed_item_ids"] == ["generic"]
    assert diff["old_plan"][0]["title"] == "Read documentation"
    assert diff["new_plan"][0]["title"] == "Build a RAG evaluation set"

    accepted = json.loads(
        await verifier.execute(
            current.workflow_id, revised["version"], "bounded-accept-revision-2"
        )
    )
    assert accepted["state"] == CareerWorkflowState.AWAITING_CONFIRMATION
    confirm = CareerWorkflowConfirmTool(workspace=tmp_path, config=config)
    confirm.set_context(
        RequestContext(
            channel="feishu",
            chat_id="chat-1",
            original_user_text="确认创建学习任务",
        )
    )
    stale = await confirm.execute(
        current.workflow_id, revised["version"], "bounded-stale-confirmation"
    )
    assert "reload" in stale
    creating = json.loads(
        await confirm.execute(
            current.workflow_id, accepted["version"], "bounded-confirm-revision-2"
        )
    )
    assert creating["checkpoint"]["confirmed_plan_revision"] == 2


async def test_replanning_stops_after_two_rejected_revisions(tmp_path) -> None:
    config = CareerToolsConfig(database_path="state/career.db")
    store = CareerWorkflowStore(tmp_path / config.database_path)
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="exhausted-workflow",
    )
    checkpoint = CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ],
        gaps=[
            GapItem(
                competency="MCP reliability",
                status=GapStatus.MISSING,
                rationale="Required by the JD.",
                evidence_ids=["K1"],
            )
        ],
        plan=[LearningPlanItem(item_id="attempt", title="Attempt zero")],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "exhausted-evidence"),
        (CareerWorkflowState.GAP_READY, "exhausted-gap"),
        (CareerWorkflowState.PLAN_VERIFYING, "exhausted-plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=key,
        )
    verifier = CareerWorkflowVerifyPlanTool(workspace=tmp_path, config=config)
    replan = CareerWorkflowReplanTool(workspace=tmp_path, config=config)
    for attempt in (1, 2):
        rejected = json.loads(
            await verifier.execute(
                current.workflow_id, current.version, f"exhausted-reject-{attempt}"
            )
        )
        assert rejected["state"] == CareerWorkflowState.REPLANNING
        current = CareerWorkflow.model_validate(
            json.loads(
                await replan.execute(
                    current.workflow_id,
                    json.dumps(
                        [{"item_id": "attempt", "title": f"Attempt {attempt}"}]
                    ),
                    "missing-gap-binding",
                    f"Try revision {attempt} without changing accepted evidence.",
                    rejected["version"],
                    f"exhausted-replan-{attempt}",
                )
            )
        )
    exhausted = json.loads(
        await verifier.execute(current.workflow_id, current.version, "exhausted-final")
    )
    assert exhausted["state"] == CareerWorkflowState.FAILED
    assert exhausted["checkpoint"]["replan_count"] == 2
    assert exhausted["checkpoint"]["error"].startswith("REPLAN_EXHAUSTED:")


async def test_transition_reuses_authoritative_immutable_evidence(tmp_path) -> None:
    database_path = tmp_path / "state" / "career.db"
    store = CareerWorkflowStore(database_path)
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-canonical-evidence",
    )
    checkpoint = CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ]
    )
    current = store.transition(
        current.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.EVIDENCE_RETRIEVED,
            checkpoint=checkpoint,
        ),
        expected_version=current.version,
        idempotency_key="retrieved-canonical-evidence",
    )
    supplied = checkpoint.model_copy(
        update={
            "evidence": [
                EvidenceReference(
                    evidence_id="K1",
                    source_type="jd",
                    source_name="jd .md",
                    chunk_id="c1",
                )
            ],
            "gaps": [
                GapItem(
                    competency="RAG",
                    status=GapStatus.MISSING,
                    rationale="JD requires it.",
                    evidence_ids=["K1"],
                )
            ],
        }
    ).model_dump(mode="json")
    supplied.update(
        {
            "confirmed": True,
            "confirmed_plan_revision": 99,
            "plan_revision": 99,
            "replan_count": 2,
            "verification_errors": ["invented verifier result"],
            "task_ids": {"invented": "task"},
            "followup_job_id": "job",
        }
    )
    transition = CareerWorkflowTransitionTool(
        workspace=tmp_path,
        config=CareerToolsConfig(database_path="state/career.db"),
    )

    result = json.loads(
        await transition.execute(
            current.workflow_id,
            CareerWorkflowState.GAP_READY.value,
            json.dumps(supplied),
            current.version,
            "gap-canonical-evidence",
        )
    )

    assert result["state"] == CareerWorkflowState.GAP_READY
    assert result["checkpoint"]["evidence"][0]["source_name"] == "jd.md"
    assert result["checkpoint"]["confirmed"] is False
    assert result["checkpoint"]["confirmed_plan_revision"] is None
    assert result["checkpoint"]["plan_revision"] == 1
    assert result["checkpoint"]["replan_count"] == 0
    assert result["checkpoint"]["verification_errors"] == []
    assert result["checkpoint"]["task_ids"] == {}
    assert result["checkpoint"]["followup_job_id"] is None


async def test_generic_transition_cannot_forge_confirmation(tmp_path) -> None:
    config = CareerToolsConfig(database_path="state/career.db")
    transition = CareerWorkflowTransitionTool(workspace=tmp_path, config=config)

    result = await transition.execute(
        "made-up",
        CareerWorkflowState.TASKS_CREATING.value,
        '{"confirmed":true}',
        1,
        "forged-confirmation",
    )

    assert "protected state requires a dedicated tool" in result


async def test_confirmation_requires_runtime_bound_original_user_text(tmp_path) -> None:
    config = CareerToolsConfig(database_path="state/career.db")
    confirm = CareerWorkflowConfirmTool(workspace=tmp_path, config=config)

    missing = await confirm.execute("workflow", 1, "confirmation")
    confirm.set_context(
        RequestContext(
            channel="feishu",
            chat_id="chat-1",
            original_user_text="模型生成的普通回复",
        )
    )
    forged = await confirm.execute("workflow", 1, "confirmation")

    assert "explicit confirmation was not present" in missing
    assert "explicit confirmation was not present" in forged


def test_explicit_confirmation_accepts_natural_chinese_and_rejects_negation() -> None:
    assert CareerWorkflowConfirmTool._is_explicit_confirmation("我明确确认这个学习计划")
    assert CareerWorkflowConfirmTool._is_explicit_confirmation("确认创建学习任务")
    assert not CareerWorkflowConfirmTool._is_explicit_confirmation("我暂不确认这个学习计划")


async def test_explicit_user_confirmation_advances_displayed_plan(tmp_path) -> None:
    database_path = tmp_path / "state" / "career.db"
    store = CareerWorkflowStore(database_path)
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-1",
    )
    checkpoint = CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ],
        gaps=[
            GapItem(
                competency="RAG",
                status=GapStatus.MISSING,
                rationale="No resume evidence",
                evidence_ids=["K1"],
            )
        ],
        plan=[
            LearningPlanItem(
                item_id="rag", title="Learn RAG evaluation", addresses=["RAG"]
            )
        ],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "evidence"),
        (CareerWorkflowState.GAP_READY, "gap"),
        (CareerWorkflowState.PLAN_VERIFYING, "plan-verifying"),
        (CareerWorkflowState.AWAITING_CONFIRMATION, "plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=key,
        )

    confirm = CareerWorkflowConfirmTool(
        workspace=tmp_path,
        config=CareerToolsConfig(database_path="state/career.db"),
    )
    confirm.set_context(
        RequestContext(
            channel="feishu",
            chat_id="chat-1",
            original_user_text="我已检查计划，确认创建学习任务",
        )
    )
    result = json.loads(
        await confirm.execute(current.workflow_id, current.version, "confirmation-1")
    )

    assert result["state"] == CareerWorkflowState.TASKS_CREATING
    assert result["checkpoint"]["confirmed"] is True

    task_store = TaskStore(database_path.parent / "tasks.db")
    task_store.initialize()
    task = task_store.create(
        TaskCreate(
            title="Learn RAG evaluation",
            source=f"career:{current.workflow_id}:rag",
        ),
        idempotency_key=f"career:{current.workflow_id}:rag",
    )
    record = CareerWorkflowRecordTasksTool(
        workspace=tmp_path,
        config=CareerToolsConfig(
            database_path="state/career.db",
            task_database_path="state/tasks.db",
        ),
    )
    recorded = json.loads(
        await record.execute(
            current.workflow_id,
            json.dumps({"rag": task.task_id}),
            result["version"],
            "record-tasks-1",
        )
    )

    assert recorded["state"] == CareerWorkflowState.TASKS_CREATED
    assert recorded["checkpoint"]["task_ids"] == {"rag": task.task_id}


async def test_task_manifest_recovers_after_partial_mcp_creation(tmp_path) -> None:
    database_path = tmp_path / "state" / "career.db"
    store = CareerWorkflowStore(database_path)
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-partial",
    )
    checkpoint = CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ],
        gaps=[
            GapItem(
                competency="Agent reliability",
                status=GapStatus.MISSING,
                rationale="Required by the JD.",
                evidence_ids=["K1"],
            )
        ],
        plan=[
            LearningPlanItem(
                item_id="rag",
                title="Build RAG evaluation",
                priority=1,
                addresses=["Agent reliability"],
            ),
            LearningPlanItem(
                item_id="recovery",
                title="Test checkpoint recovery",
                priority=2,
                addresses=["Agent reliability"],
            ),
        ],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "partial-evidence"),
        (CareerWorkflowState.GAP_READY, "partial-gap"),
        (CareerWorkflowState.PLAN_VERIFYING, "partial-plan-verifying"),
        (CareerWorkflowState.AWAITING_CONFIRMATION, "partial-plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=key,
        )
    confirm = CareerWorkflowConfirmTool(
        workspace=tmp_path,
        config=CareerToolsConfig(database_path="state/career.db"),
    )
    confirm.set_context(
        RequestContext(
            channel="feishu",
            chat_id="chat-1",
            original_user_text="确认创建学习任务",
        )
    )
    creating = json.loads(
        await confirm.execute(current.workflow_id, current.version, "partial-confirm")
    )
    config = CareerToolsConfig(
        database_path="state/career.db", task_database_path="state/tasks.db"
    )
    manifest = CareerWorkflowTaskManifestTool(workspace=tmp_path, config=config)
    initial = json.loads(await manifest.execute(current.workflow_id))
    assert {call["plan_item_id"] for call in initial["pending_calls"]} == {
        "rag",
        "recovery",
    }

    task_store = TaskStore(database_path.parent / "tasks.db")
    task_store.initialize()
    first_call = initial["pending_calls"][0]
    first_args = first_call["arguments"]
    first_request = TaskCreate(
        title=first_args["title"],
        description=first_args["description"],
        priority=first_args["priority"],
        tags=first_args["tags"],
        source=first_args["source"],
    )
    first_task = task_store.create(
        first_request, idempotency_key=first_args["idempotency_key"]
    )

    recovered = json.loads(await manifest.execute(current.workflow_id))
    assert recovered["completed_task_ids"] == {first_call["plan_item_id"]: first_task.task_id}
    assert len(recovered["pending_calls"]) == 1
    assert task_store.create(
        first_request, idempotency_key=first_args["idempotency_key"]
    ).task_id == first_task.task_id

    second_call = recovered["pending_calls"][0]
    second_args = second_call["arguments"]
    second_task = task_store.create(
        TaskCreate(
            title=second_args["title"],
            description=second_args["description"],
            priority=second_args["priority"],
            tags=second_args["tags"],
            source=second_args["source"],
        ),
        idempotency_key=second_args["idempotency_key"],
    )
    complete = json.loads(await manifest.execute(current.workflow_id))
    assert complete["pending_calls"] == []

    record = CareerWorkflowRecordTasksTool(workspace=tmp_path, config=config)
    recorded = json.loads(
        await record.execute(
            current.workflow_id,
            json.dumps(
                {
                    first_call["plan_item_id"]: first_task.task_id,
                    second_call["plan_item_id"]: second_task.task_id,
                }
            ),
            creating["version"],
            "partial-record",
        )
    )
    assert recorded["state"] == CareerWorkflowState.TASKS_CREATED
    assert len(recorded["checkpoint"]["task_ids"]) == 2
    replay = json.loads(
        await record.execute(
            current.workflow_id,
            json.dumps(recorded["checkpoint"]["task_ids"]),
            creating["version"],
            "partial-record",
        )
    )
    assert replay == recorded


async def test_followup_schedule_survives_restart_and_completes_from_real_state(
    tmp_path, monkeypatch
) -> None:
    career_path = tmp_path / "state" / "career.db"
    task_path = tmp_path / "state" / "tasks.db"
    cron_path = tmp_path / "state" / "cron.json"
    workflow_store = CareerWorkflowStore(career_path)
    workflow_store.initialize()
    current = workflow_store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="scheduled-workflow",
    )
    task_store = TaskStore(task_path)
    task_store.initialize()
    task = task_store.create(
        TaskCreate(
            title="Build RAG evaluation",
            source=f"career:{current.workflow_id}:rag",
        ),
        idempotency_key=f"career:{current.workflow_id}:rag",
    )
    checkpoint = CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ],
        gaps=[
            GapItem(
                competency="RAG evaluation",
                status=GapStatus.MISSING,
                rationale="Required by the JD.",
                evidence_ids=["K1"],
            )
        ],
        plan=[
            LearningPlanItem(
                item_id="rag",
                title="Build RAG evaluation",
                addresses=["RAG evaluation"],
            )
        ],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "schedule-evidence"),
        (CareerWorkflowState.GAP_READY, "schedule-gap"),
        (CareerWorkflowState.PLAN_VERIFYING, "schedule-plan-verifying"),
        (CareerWorkflowState.AWAITING_CONFIRMATION, "schedule-plan"),
    ):
        current = workflow_store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=key,
        )
    confirmed = checkpoint.model_copy(
        update={"confirmed": True, "confirmed_plan_revision": checkpoint.plan_revision}
    )
    current = workflow_store.transition(
        current.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.TASKS_CREATING, checkpoint=confirmed
        ),
        expected_version=current.version,
        idempotency_key="schedule-confirm",
    )
    with_tasks = confirmed.model_copy(update={"task_ids": {"rag": task.task_id}})
    current = workflow_store.transition(
        current.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.TASKS_CREATED, checkpoint=with_tasks
        ),
        expected_version=current.version,
        idempotency_key="schedule-tasks",
    )

    config = CareerToolsConfig(
        database_path="state/career.db", task_database_path="state/tasks.db"
    )
    cron = CronService(cron_path)
    schedule = CareerWorkflowScheduleTool(
        workspace=tmp_path, config=config, cron_service=cron
    )
    origin = RequestContext(
        channel="feishu",
        chat_id="oc_interview",
        session_key=UNIFIED_SESSION_KEY,
        metadata={"message_id": "om_confirm"},
    )
    original_transition = schedule._store.transition

    def fail_checkpoint(*args, **kwargs):
        raise OSError("simulated checkpoint outage")

    monkeypatch.setattr(schedule._store, "transition", fail_checkpoint)
    with request_context(origin):
        interrupted = await schedule.execute(
            current.workflow_id, 3600, current.version, "schedule-followup"
        )
    assert "simulated checkpoint outage" in interrupted
    assert len(cron.list_jobs()) == 1
    monkeypatch.setattr(schedule._store, "transition", original_transition)
    recovery_cron = CronService(cron_path)
    schedule = CareerWorkflowScheduleTool(
        workspace=tmp_path, config=config, cron_service=recovery_cron
    )

    with request_context(origin):
        scheduled = json.loads(
            await schedule.execute(
                current.workflow_id, 3600, current.version, "schedule-followup"
            )
        )
        replay = json.loads(
            await schedule.execute(
                current.workflow_id, 3600, current.version, "schedule-followup"
            )
        )
    assert replay == scheduled
    assert len(recovery_cron.list_jobs()) == 1
    job = recovery_cron.list_jobs()[0]
    assert job.id == scheduled["checkpoint"]["followup_job_id"]
    assert job.payload.origin_channel == "feishu"
    assert job.payload.origin_chat_id == "oc_interview"

    restarted_cron = CronService(cron_path)
    complete = CareerWorkflowCompleteTool(
        workspace=tmp_path, config=config, cron_service=restarted_cron
    )
    incomplete = await complete.execute(
        current.workflow_id, scheduled["version"], "complete-followup"
    )
    assert "not all learning tasks are done" in incomplete

    task_store.update(
        task.task_id,
        TaskUpdate(status=TaskStatus.DONE),
        expected_version=task.version,
        idempotency_key="finish-rag-task",
    )
    completed = json.loads(
        await complete.execute(
            current.workflow_id, scheduled["version"], "complete-followup"
        )
    )
    assert completed["state"] == CareerWorkflowState.COMPLETED
    assert restarted_cron.get_job(job.id) is None
    completed_replay = json.loads(
        await complete.execute(
            current.workflow_id, scheduled["version"], "complete-followup"
        )
    )
    assert completed_replay == completed
