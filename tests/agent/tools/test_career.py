import json

from nanobot.agent.tools.career import (
    CareerToolsConfig,
    CareerWorkflowConfirmTool,
    CareerWorkflowGetTool,
    CareerWorkflowRecordTasksTool,
    CareerWorkflowRetrieveTool,
    CareerWorkflowStartTool,
    CareerWorkflowTaskManifestTool,
    CareerWorkflowTransitionTool,
)
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.knowledge import KnowledgeToolsConfig
from nanobot.career import (
    CareerCheckpoint,
    CareerWorkflowCreate,
    CareerWorkflowState,
    CareerWorkflowStore,
    CareerWorkflowTransition,
    EvidenceReference,
    GapItem,
    GapStatus,
    LearningPlanItem,
)
from nanobot.tasking import TaskCreate, TaskStore


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

    assert result.startswith("Career workflow creation failed:")


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

    assert result.startswith("Career workflow transition failed:")
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
            }
        ],
    }
    awaiting = json.loads(
        await transition.execute(
            gap_ready["workflow_id"],
            CareerWorkflowState.AWAITING_CONFIRMATION.value,
            json.dumps(plan_checkpoint),
            gap_ready["version"],
            "plan-flow",
        )
    )

    assert awaiting["state"] == CareerWorkflowState.AWAITING_CONFIRMATION
    assert awaiting["checkpoint"]["evidence"] == evidence
    assert awaiting["checkpoint"]["gaps"] == gap_checkpoint["gaps"]


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
        plan=[LearningPlanItem(item_id="rag", title="Learn RAG evaluation")],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "evidence"),
        (CareerWorkflowState.GAP_READY, "gap"),
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
            LearningPlanItem(item_id="rag", title="Build RAG evaluation", priority=1),
            LearningPlanItem(item_id="recovery", title="Test checkpoint recovery", priority=2),
        ],
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "partial-evidence"),
        (CareerWorkflowState.GAP_READY, "partial-gap"),
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
