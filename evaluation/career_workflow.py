"""Deterministic end-to-end reliability scenarios for the career workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nanobot.agent.tools.career import (
    CareerToolsConfig,
    CareerWorkflowCompleteTool,
    CareerWorkflowConfirmTool,
    CareerWorkflowGetTool,
    CareerWorkflowRecordTasksTool,
    CareerWorkflowRetrieveTool,
    CareerWorkflowScheduleTool,
    CareerWorkflowStartTool,
    CareerWorkflowTaskManifestTool,
    CareerWorkflowTransitionTool,
)
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.knowledge import KnowledgeToolsConfig
from nanobot.cron.service import CronService
from nanobot.session.keys import UNIFIED_SESSION_KEY
from nanobot.tasking import TaskCreate, TaskStatus, TaskStore, TaskUpdate


@dataclass(frozen=True)
class CareerCaseResult:
    case_id: str
    group: str
    passed: bool
    detail: str


def _json_result(raw: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def evaluate_career_workflow(root: Path) -> dict[str, object]:
    """Run the real tool contracts against isolated SQLite and Cron stores."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "resume.md").write_text(
        "Built Python APIs with Docker and idempotent task processing.", encoding="utf-8"
    )
    (root / "jd.md").write_text(
        "The role requires Python, Docker, RAG evaluation, and recoverable Agent workflows.",
        encoding="utf-8",
    )
    config = CareerToolsConfig(
        database_path="state/career.db", task_database_path="state/tasks.db"
    )
    knowledge_config = KnowledgeToolsConfig(
        database_path="state/knowledge.db", retrieval_mode="lexical"
    )
    start = CareerWorkflowStartTool(workspace=root, config=config)
    retrieve = CareerWorkflowRetrieveTool(
        workspace=root, config=config, knowledge_config=knowledge_config
    )
    transition = CareerWorkflowTransitionTool(workspace=root, config=config)
    confirm = CareerWorkflowConfirmTool(workspace=root, config=config)
    manifest = CareerWorkflowTaskManifestTool(workspace=root, config=config)
    record = CareerWorkflowRecordTasksTool(workspace=root, config=config)
    get = CareerWorkflowGetTool(workspace=root, config=config)
    cases: list[CareerCaseResult] = []

    started = _json_result(await start.execute("resume.md", "jd.md", "eval-career"))
    cases.append(
        CareerCaseResult("career-start", "career_success", started is not None, "documents_ready")
    )
    assert started is not None
    workflow_id = str(started["workflow_id"])
    retrieved_result = _json_result(
        await retrieve.execute(
            workflow_id,
            json.dumps(["Python Docker", "RAG evaluation Agent recovery"]),
            int(started["version"]),
            "eval-retrieve",
        )
    )
    retrieved = retrieved_result["workflow"] if retrieved_result else None
    cases.append(
        CareerCaseResult(
            "career-retrieve", "career_success", isinstance(retrieved, dict), "real chunk IDs"
        )
    )
    assert isinstance(retrieved, dict)
    checkpoint = dict(retrieved["checkpoint"])
    evidence = list(checkpoint["evidence"])
    jd_id = next(
        str(item["evidence_id"])
        for item in evidence
        if isinstance(item, dict) and item.get("source_type") == "jd"
    )
    checkpoint["gaps"] = [
        {
            "competency": "RAG evaluation",
            "status": "missing",
            "rationale": "The JD requires evaluation and the resume does not demonstrate it.",
            "evidence_ids": [jd_id],
        }
    ]
    gap = _json_result(
        await transition.execute(
            workflow_id,
            "gap_ready",
            json.dumps(checkpoint),
            int(retrieved["version"]),
            "eval-gap",
        )
    )
    cases.append(CareerCaseResult("career-gap", "career_success", gap is not None, "cited gap"))
    assert gap is not None
    checkpoint = dict(gap["checkpoint"])
    checkpoint["plan"] = [
        {
            "item_id": "rag-eval",
            "title": "Build a reproducible RAG evaluation set",
            "description": "Measure Recall@K, MRR, and NDCG.",
            "priority": 1,
        }
    ]
    planned = _json_result(
        await transition.execute(
            workflow_id,
            "awaiting_confirmation",
            json.dumps(checkpoint),
            int(gap["version"]),
            "eval-plan",
        )
    )
    cases.append(
        CareerCaseResult("career-plan", "career_success", planned is not None, "awaiting confirmation")
    )
    assert planned is not None

    refused = await confirm.execute(workflow_id, int(planned["version"]), "eval-confirm-refused")
    cases.append(
        CareerCaseResult(
            "guard-confirmation", "career_guardrail", "explicit confirmation" in refused, refused
        )
    )
    confirm.set_context(
        RequestContext(
            channel="feishu",
            chat_id="oc_eval",
            session_key=UNIFIED_SESSION_KEY,
            original_user_text="确认创建学习任务",
        )
    )
    creating = _json_result(
        await confirm.execute(workflow_id, int(planned["version"]), "eval-confirm")
    )
    cases.append(
        CareerCaseResult(
            "career-confirm", "career_success", creating is not None, "runtime-bound confirmation"
        )
    )
    assert creating is not None
    initial_manifest = _json_result(await manifest.execute(workflow_id))
    assert initial_manifest is not None
    pending = list(initial_manifest["pending_calls"])
    cases.append(
        CareerCaseResult(
            "career-manifest", "career_success", len(pending) == 1, "one pending MCP call"
        )
    )

    task_store = TaskStore(root / config.task_database_path)
    task_store.initialize()
    call = pending[0]
    assert isinstance(call, dict)
    arguments = call["arguments"]
    assert isinstance(arguments, dict)
    request = TaskCreate(
        title=str(arguments["title"]),
        description=str(arguments["description"]),
        priority=int(arguments["priority"]),
        tags=list(arguments["tags"]),
        source=str(arguments["source"]),
    )
    task = task_store.create(request, idempotency_key=str(arguments["idempotency_key"]))
    replay = task_store.create(request, idempotency_key=str(arguments["idempotency_key"]))
    cases.append(
        CareerCaseResult(
            "recovery-task-replay",
            "career_recovery",
            replay.task_id == task.task_id and len(task_store.list()) == 1,
            "stable task receipt",
        )
    )
    recovered_manifest = _json_result(await manifest.execute(workflow_id))
    cases.append(
        CareerCaseResult(
            "recovery-task-manifest",
            "career_recovery",
            recovered_manifest is not None
            and recovered_manifest["pending_calls"] == []
            and recovered_manifest["completed_task_ids"] == {"rag-eval": task.task_id},
            "reconstructed from SQLite",
        )
    )
    invented = await record.execute(
        workflow_id,
        json.dumps({"rag-eval": "invented-task"}),
        int(creating["version"]),
        "eval-invented-task",
    )
    cases.append(
        CareerCaseResult(
            "guard-invented-task", "career_guardrail", "not found" in invented, invented
        )
    )
    recorded = _json_result(
        await record.execute(
            workflow_id,
            json.dumps({"rag-eval": task.task_id}),
            int(creating["version"]),
            "eval-record-task",
        )
    )
    cases.append(
        CareerCaseResult(
            "career-record-task", "career_success", recorded is not None, "verified MCP result"
        )
    )
    assert recorded is not None

    cron_path = root / "state" / "cron.json"
    cron = CronService(cron_path)
    schedule = CareerWorkflowScheduleTool(
        workspace=root, config=config, cron_service=cron
    )
    origin = RequestContext(
        channel="feishu", chat_id="oc_eval", session_key=UNIFIED_SESSION_KEY
    )
    with request_context(origin):
        scheduled = _json_result(
            await schedule.execute(
                workflow_id, 3600, int(recorded["version"]), "eval-schedule"
            )
        )
    cases.append(
        CareerCaseResult(
            "career-schedule",
            "career_success",
            scheduled is not None and len(cron.list_jobs()) == 1,
            "channel-bound Cron receipt",
        )
    )
    assert scheduled is not None
    complete = CareerWorkflowCompleteTool(
        workspace=root, config=config, cron_service=CronService(cron_path)
    )
    incomplete = await complete.execute(
        workflow_id, int(scheduled["version"]), "eval-complete"
    )
    cases.append(
        CareerCaseResult(
            "guard-incomplete-workflow",
            "career_guardrail",
            "not all learning tasks are done" in incomplete,
            incomplete,
        )
    )
    task_store.update(
        task.task_id,
        TaskUpdate(status=TaskStatus.DONE),
        expected_version=task.version,
        idempotency_key="eval-finish-task",
    )
    completed = _json_result(
        await complete.execute(workflow_id, int(scheduled["version"]), "eval-complete")
    )
    cases.append(
        CareerCaseResult(
            "career-complete",
            "career_success",
            completed is not None and completed["state"] == "completed",
            "all tasks verified done",
        )
    )
    loaded = _json_result(await get.execute(workflow_id))
    cases.append(
        CareerCaseResult(
            "recovery-workflow-restart",
            "career_recovery",
            loaded is not None and loaded["state"] == "completed",
            "completed checkpoint reopened",
        )
    )

    groups: dict[str, dict[str, int | float]] = {}
    for group in sorted({case.group for case in cases}):
        selected = [case for case in cases if case.group == group]
        passed = sum(case.passed for case in selected)
        groups[group] = {
            "passed": passed,
            "total": len(selected),
            "pass_rate": round(passed / len(selected), 4),
        }
    passed = sum(case.passed for case in cases)
    return {
        "fixture": "deterministic-career-workflow-v1",
        "scope": "offline reliability scenarios; not production traffic",
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 4),
        "groups": groups,
        "cases": [case.__dict__ for case in cases],
    }
