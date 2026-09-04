"""Validated models for the career-planning workflow."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CareerWorkflowState(StrEnum):
    DOCUMENTS_READY = "documents_ready"
    EVIDENCE_RETRIEVED = "evidence_retrieved"
    GAP_READY = "gap_ready"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    TASKS_CREATING = "tasks_creating"
    FOLLOWUP_SCHEDULED = "followup_scheduled"
    COMPLETED = "completed"
    FAILED = "failed"


class GapStatus(StrEnum):
    DEMONSTRATED = "demonstrated"
    WEAK_EVIDENCE = "weak_evidence"
    MISSING = "missing"


class EvidenceReference(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    evidence_id: str = Field(pattern=r"^K[1-9][0-9]*$")
    source_name: str = Field(min_length=1, max_length=500)
    chunk_id: str = Field(min_length=1, max_length=200)


class GapItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    competency: str = Field(min_length=1, max_length=200)
    status: GapStatus
    rationale: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class LearningPlanItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    item_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(default="", max_length=5_000)
    priority: int = Field(default=3, ge=1, le=5)


class CareerCheckpoint(BaseModel):
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=200)
    gaps: list[GapItem] = Field(default_factory=list, max_length=100)
    plan: list[LearningPlanItem] = Field(default_factory=list, max_length=100)
    task_ids: dict[str, str] = Field(default_factory=dict)
    confirmed: bool = False
    followup_job_id: str | None = Field(default=None, max_length=200)
    error: str | None = Field(default=None, max_length=4_000)
    resume_state: CareerWorkflowState | None = None
    retry_count: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def references_must_be_consistent(self) -> CareerCheckpoint:
        evidence_ids = {item.evidence_id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        unknown = {
            evidence_id
            for gap in self.gaps
            for evidence_id in gap.evidence_ids
            if evidence_id not in evidence_ids
        }
        if unknown:
            raise ValueError("gap items reference unknown evidence IDs")
        plan_ids = {item.item_id for item in self.plan}
        if len(plan_ids) != len(self.plan):
            raise ValueError("learning plan item IDs must be unique")
        if not set(self.task_ids).issubset(plan_ids):
            raise ValueError("task IDs must belong to learning plan items")
        if any(not task_id.strip() for task_id in self.task_ids.values()):
            raise ValueError("task IDs must not be empty")
        return self


class CareerWorkflowCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    resume_source: str = Field(min_length=1, max_length=1_000)
    jd_source: str = Field(min_length=1, max_length=1_000)


class CareerWorkflowTransition(BaseModel):
    target_state: CareerWorkflowState
    checkpoint: CareerCheckpoint


class CareerWorkflow(BaseModel):
    workflow_id: str
    resume_source: str
    jd_source: str
    state: CareerWorkflowState
    checkpoint: CareerCheckpoint
    created_at: datetime
    updated_at: datetime
    version: int
