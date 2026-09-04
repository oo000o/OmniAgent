import pytest

from nanobot.career import (
    CareerCheckpoint,
    CareerWorkflowConflictError,
    CareerWorkflowCreate,
    CareerWorkflowState,
    CareerWorkflowStore,
    CareerWorkflowTransition,
    EvidenceReference,
    GapItem,
    GapStatus,
    LearningPlanItem,
)


def _checkpoint(*, confirmed: bool = False) -> CareerCheckpoint:
    return CareerCheckpoint(
        evidence=[EvidenceReference(evidence_id="K1", source_name="jd.md", chunk_id="c1")],
        gaps=[
            GapItem(
                competency="RAG evaluation",
                status=GapStatus.MISSING,
                rationale="The JD requires evaluation but the resume has no evidence.",
                evidence_ids=["K1"],
            )
        ],
        plan=[LearningPlanItem(item_id="rag-eval", title="Build a RAG evaluation set")],
        confirmed=confirmed,
    )


def test_create_is_idempotent_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "career.db"
    store = CareerWorkflowStore(path)
    store.initialize()
    request = CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md")

    first = store.create(request, idempotency_key="workflow-1")
    replay = store.create(request, idempotency_key="workflow-1")
    reopened = CareerWorkflowStore(path)
    reopened.initialize()

    assert replay.workflow_id == first.workflow_id
    assert reopened.get(first.workflow_id) == first


def test_workflow_rejects_skipped_state_and_stale_version(tmp_path) -> None:
    store = CareerWorkflowStore(tmp_path / "career.db")
    store.initialize()
    workflow = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-1",
    )

    with pytest.raises(CareerWorkflowConflictError, match="invalid workflow transition"):
        store.transition(
            workflow.workflow_id,
            CareerWorkflowTransition(
                target_state=CareerWorkflowState.AWAITING_CONFIRMATION,
                checkpoint=_checkpoint(),
            ),
            expected_version=workflow.version,
            idempotency_key="skip",
        )

    evidence = store.transition(
        workflow.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.EVIDENCE_RETRIEVED,
            checkpoint=_checkpoint(),
        ),
        expected_version=workflow.version,
        idempotency_key="evidence",
    )
    with pytest.raises(CareerWorkflowConflictError, match="reload"):
        store.transition(
            workflow.workflow_id,
            CareerWorkflowTransition(
                target_state=CareerWorkflowState.GAP_READY,
                checkpoint=_checkpoint(),
            ),
            expected_version=workflow.version,
            idempotency_key="stale",
        )
    assert evidence.version == 2


def test_task_creation_requires_confirmation_and_real_task_ids(tmp_path) -> None:
    store = CareerWorkflowStore(tmp_path / "career.db")
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-1",
    )
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, "evidence"),
        (CareerWorkflowState.GAP_READY, "gap"),
        (CareerWorkflowState.AWAITING_CONFIRMATION, "plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=_checkpoint()),
            expected_version=current.version,
            idempotency_key=key,
        )

    with pytest.raises(ValueError, match="confirmation"):
        store.transition(
            current.workflow_id,
            CareerWorkflowTransition(
                target_state=CareerWorkflowState.TASKS_CREATING,
                checkpoint=_checkpoint(),
            ),
            expected_version=current.version,
            idempotency_key="unconfirmed",
        )

    creating = store.transition(
        current.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.TASKS_CREATING,
            checkpoint=_checkpoint(confirmed=True),
        ),
        expected_version=current.version,
        idempotency_key="confirmed",
    )
    with pytest.raises(ValueError, match="persisted task ID"):
        store.transition(
            creating.workflow_id,
            CareerWorkflowTransition(
                target_state=CareerWorkflowState.TASKS_CREATED,
                checkpoint=_checkpoint(confirmed=True),
            ),
            expected_version=creating.version,
            idempotency_key="missing-tasks",
        )


def test_transition_replay_returns_original_checkpoint(tmp_path) -> None:
    store = CareerWorkflowStore(tmp_path / "career.db")
    store.initialize()
    workflow = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-1",
    )
    request = CareerWorkflowTransition(
        target_state=CareerWorkflowState.EVIDENCE_RETRIEVED,
        checkpoint=_checkpoint(),
    )

    first = store.transition(
        workflow.workflow_id,
        request,
        expected_version=1,
        idempotency_key="transition-1",
    )
    replay = store.transition(
        workflow.workflow_id,
        request,
        expected_version=1,
        idempotency_key="transition-1",
    )

    assert replay == first
    assert store.get(workflow.workflow_id).version == 2


def test_failed_workflow_only_resumes_recorded_state(tmp_path) -> None:
    store = CareerWorkflowStore(tmp_path / "career.db")
    store.initialize()
    workflow = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="workflow-1",
    )
    failed_checkpoint = CareerCheckpoint(
        error="embedding timeout",
        resume_state=CareerWorkflowState.DOCUMENTS_READY,
        retry_count=1,
    )
    failed = store.transition(
        workflow.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.FAILED,
            checkpoint=failed_checkpoint,
        ),
        expected_version=workflow.version,
        idempotency_key="fail-1",
    )

    with pytest.raises(CareerWorkflowConflictError, match="recorded state"):
        store.transition(
            failed.workflow_id,
            CareerWorkflowTransition(
                target_state=CareerWorkflowState.EVIDENCE_RETRIEVED,
                checkpoint=_checkpoint(),
            ),
            expected_version=failed.version,
            idempotency_key="wrong-resume",
        )

    resumed = store.transition(
        failed.workflow_id,
        CareerWorkflowTransition(
            target_state=CareerWorkflowState.DOCUMENTS_READY,
            checkpoint=CareerCheckpoint(retry_count=1),
        ),
        expected_version=failed.version,
        idempotency_key="resume-1",
    )
    assert resumed.state is CareerWorkflowState.DOCUMENTS_READY
    assert resumed.version == 3
