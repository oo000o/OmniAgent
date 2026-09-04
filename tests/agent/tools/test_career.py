import json

from nanobot.agent.tools.career import (
    CareerToolsConfig,
    CareerWorkflowConfirmTool,
    CareerWorkflowGetTool,
    CareerWorkflowRecordTasksTool,
    CareerWorkflowStartTool,
    CareerWorkflowTransitionTool,
)
from nanobot.agent.tools.context import RequestContext
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
        """{"evidence":[{"evidence_id":"K1","source_name":"jd.md","chunk_id":"c1"}],
        "gaps":[{"competency":"RAG","status":"missing","rationale":"No evidence",
        "evidence_ids":["K1"]}]}""",
        1,
        "unsafe-skip",
    )

    assert result.startswith("Career workflow transition failed:")
    assert "invalid workflow transition" in result


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
        evidence=[EvidenceReference(evidence_id="K1", source_name="jd.md", chunk_id="c1")],
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
