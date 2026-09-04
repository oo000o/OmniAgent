"""Validated task-domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(timezone.utc)


class TaskCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=240)
    description: str = Field(default="", max_length=10_000)
    priority: int = Field(default=3, ge=1, le=5)
    due_at: datetime | None = None
    tags: list[str] = Field(default_factory=list, max_length=20)
    source: str = Field(default="manual", min_length=1, max_length=120)

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        return _normalize_datetime(value)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip().lower() for value in values if value.strip()]
        if any(len(value) > 60 for value in cleaned):
            raise ValueError("each tag must be at most 60 characters")
        return list(dict.fromkeys(cleaned))


class TaskUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=10_000)
    status: TaskStatus | None = None
    priority: int | None = Field(default=None, ge=1, le=5)
    due_at: datetime | None = None
    clear_due_at: bool = False
    tags: list[str] | None = Field(default=None, max_length=20)

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        return _normalize_datetime(value)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [value.strip().lower() for value in values if value.strip()]
        if any(len(value) > 60 for value in cleaned):
            raise ValueError("each tag must be at most 60 characters")
        return list(dict.fromkeys(cleaned))


class Task(BaseModel):
    task_id: str
    title: str
    description: str
    status: TaskStatus
    priority: int
    due_at: datetime | None
    tags: list[str]
    source: str
    created_at: datetime
    updated_at: datetime
    version: int
