import json

from nanobot.agent.tools.career import (
    CareerToolsConfig,
    CareerWorkflowConfirmTool,
    CareerWorkflowGetTool,
    CareerWorkflowReplanTool,
    CareerWorkflowScheduleTool,
    CareerWorkflowVerifyPlanTool,
)
from nanobot.agent.tools.context import RequestContext, request_context
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
from nanobot.cron.service import CronService
from nanobot.session.keys import UNIFIED_SESSION_KEY
from nanobot.tasking import TaskCreate, TaskStore


def _seed_plan_verifying(store: CareerWorkflowStore, *, key: str):
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key=key,
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
    for state, step in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "evidence"),
        (CareerWorkflowState.GAP_READY, "gap"),
        (CareerWorkflowState.PLAN_VERIFYING, "plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=f"{key}-{step}",
        )
    return current


async def test_task_recovery_three_times_does_not_duplicate(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    store.initialize()
    request = TaskCreate(title="Build RAG evaluation", source="career:fault:rag")
    first = store.create(request, idempotency_key="career:fault:rag")

    recovered = [
        store.create(request, idempotency_key="career:fault:rag").task_id for _ in range(3)
    ]

    assert recovered == [first.task_id, first.task_id, first.task_id]
    assert len(store.list()) == 1


async def test_cron_recovery_three_times_does_not_duplicate_jobs(tmp_path) -> None:
    career_path = tmp_path / "state" / "career.db"
    task_path = tmp_path / "state" / "tasks.db"
    cron_path = tmp_path / "state" / "cron.json"
    store = CareerWorkflowStore(career_path)
    store.initialize()
    current = _seed_plan_verifying(store, key="cron-thrice")
    config = CareerToolsConfig(
        database_path="state/career.db", task_database_path="state/tasks.db"
    )
    verifier = CareerWorkflowVerifyPlanTool(workspace=tmp_path, config=config)
    rejected = json.loads(
        await verifier.execute(current.workflow_id, current.version, "cron-thrice-reject")
    )
    revised = json.loads(
        await CareerWorkflowReplanTool(workspace=tmp_path, config=config).execute(
            current.workflow_id,
            json.dumps(
                [
                    {
                        "item_id": "rag-eval",
                        "title": "Build RAG evaluation",
                        "priority": 1,
                        "addresses": ["RAG evaluation"],
                    }
                ]
            ),
            "uncovered-required-competency",
            "Bind the missing competency.",
            rejected["version"],
            "cron-thrice-replan",
        )
    )
    awaiting = json.loads(
        await verifier.execute(current.workflow_id, revised["version"], "cron-thrice-verify")
    )
    confirm = CareerWorkflowConfirmTool(workspace=tmp_path, config=config)
    confirm.set_context(
        RequestContext(channel="feishu", chat_id="oc_thrice", original_user_text="确认创建学习任务")
    )
    creating = json.loads(
        await confirm.execute(current.workflow_id, awaiting["version"], "cron-thrice-confirm")
    )
    task_store = TaskStore(task_path)
    task_store.initialize()
    task = task_store.create(
        TaskCreate(
            title="Build RAG evaluation",
            source=f"career:{current.workflow_id}:rag-eval",
        ),
        idempotency_key=f"career:{current.workflow_id}:rag-eval",
    )
    created = store.transition(
        current.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.TASKS_CREATED,
            checkpoint=store.get(current.workflow_id).checkpoint.model_copy(
                update={"task_ids": {"rag-eval": task.task_id}}
            ),
        ),
        expected_version=creating["version"],
        idempotency_key="cron-thrice-record",
    )
    cron = CronService(cron_path)
    schedule = CareerWorkflowScheduleTool(workspace=tmp_path, config=config, cron_service=cron)
    origin = RequestContext(
        channel="feishu", chat_id="oc_thrice", session_key=UNIFIED_SESSION_KEY
    )
    original = schedule._store.transition

    def fail_checkpoint(*args, **kwargs):
        raise OSError("simulated checkpoint outage")

    schedule._store.transition = fail_checkpoint
    with request_context(origin):
        interrupted = await schedule.execute(
            current.workflow_id, 3600, created.version, "cron-thrice-schedule"
        )
    assert "CHECKPOINT_ERROR" in interrupted
    assert len(cron.list_jobs()) == 1
    schedule._store.transition = original
    job_ids = []
    with request_context(origin):
        for _ in range(3):
            recovered = json.loads(
                await schedule.execute(
                    current.workflow_id, 3600, created.version, "cron-thrice-schedule"
                )
            )
            job_ids.append(recovered["checkpoint"]["followup_job_id"])
    assert len(set(job_ids)) == 1
    assert len(CronService(cron_path).list_jobs()) == 1


async def test_replan_interrupt_before_checkpoint_keeps_old_revision(tmp_path) -> None:
    store = CareerWorkflowStore(tmp_path / "state" / "career.db")
    store.initialize()
    current = _seed_plan_verifying(store, key="replan-interrupt")
    config = CareerToolsConfig(database_path="state/career.db")
    verifier = CareerWorkflowVerifyPlanTool(workspace=tmp_path, config=config)
    rejected = json.loads(
        await verifier.execute(current.workflow_id, current.version, "replan-interrupt-reject")
    )
    replan = CareerWorkflowReplanTool(workspace=tmp_path, config=config)
    original = replan._store.transition

    def fail_checkpoint(*args, **kwargs):
        raise OSError("simulated replan checkpoint outage")

    replan._store.transition = fail_checkpoint
    interrupted = await replan.execute(
        current.workflow_id,
        json.dumps(
            [
                {
                    "item_id": "rag-eval",
                    "title": "Build RAG evaluation",
                    "priority": 1,
                    "addresses": ["RAG evaluation"],
                }
            ]
        ),
        "uncovered-required-competency",
        "Bind the missing competency.",
        rejected["version"],
        "replan-interrupt-1",
    )
    replan._store.transition = original
    loaded = json.loads(
        await CareerWorkflowGetTool(workspace=tmp_path, config=config).execute(current.workflow_id)
    )
    recovered = json.loads(
        await replan.execute(
            current.workflow_id,
            json.dumps(
                [
                    {
                        "item_id": "rag-eval",
                        "title": "Build RAG evaluation",
                        "priority": 1,
                        "addresses": ["RAG evaluation"],
                    }
                ]
            ),
            "uncovered-required-competency",
            "Bind the missing competency.",
            rejected["version"],
            "replan-interrupt-1",
        )
    )
    assert "CHECKPOINT_ERROR" in interrupted
    assert loaded["checkpoint"]["plan_revision"] == 1
    assert loaded["checkpoint"]["replan_count"] == 0
    assert recovered["checkpoint"]["plan_revision"] == 2
    assert recovered["checkpoint"]["replan_count"] == 1


async def test_confirming_v1_is_rejected_after_plan_v2(tmp_path) -> None:
    store = CareerWorkflowStore(tmp_path / "state" / "career.db")
    store.initialize()
    current = _seed_plan_verifying(store, key="stale-confirm")
    config = CareerToolsConfig(database_path="state/career.db")
    verifier = CareerWorkflowVerifyPlanTool(workspace=tmp_path, config=config)
    rejected = json.loads(
        await verifier.execute(current.workflow_id, current.version, "stale-reject")
    )
    seen_v1 = rejected["version"]
    revised = json.loads(
        await CareerWorkflowReplanTool(workspace=tmp_path, config=config).execute(
            current.workflow_id,
            json.dumps(
                [
                    {
                        "item_id": "rag-eval",
                        "title": "Build RAG evaluation",
                        "priority": 1,
                        "addresses": ["RAG evaluation"],
                    }
                ]
            ),
            "uncovered-required-competency",
            "Bind the missing competency.",
            rejected["version"],
            "stale-replan",
        )
    )
    awaiting = json.loads(
        await verifier.execute(current.workflow_id, revised["version"], "stale-verify")
    )
    confirm = CareerWorkflowConfirmTool(workspace=tmp_path, config=config)
    confirm.set_context(
        RequestContext(channel="feishu", chat_id="oc_stale", original_user_text="确认创建学习任务")
    )
    stale = await confirm.execute(current.workflow_id, seen_v1, "stale-confirm-exec")
    current_state = json.loads(
        await CareerWorkflowGetTool(workspace=tmp_path, config=config).execute(current.workflow_id)
    )
    assert "VERSION_CONFLICT" in stale
    assert awaiting["checkpoint"]["plan_revision"] == 2
    assert current_state["state"] == CareerWorkflowState.AWAITING_CONFIRMATION
    assert current_state["checkpoint"]["confirmed"] is False
