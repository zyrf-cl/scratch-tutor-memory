"""Pydantic models for the memory module's request and response shapes.

Conventions:

* Five concrete event types live under a discriminated union ``Event``.
* Read-side state aggregates are rendered as ``StudentState``.
* Datetimes are always timezone-aware UTC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .concepts import is_valid_concept


# ===== Event types (write path) =====


class MasteryUpdateEvent(BaseModel):
    type: Literal["mastery_update"] = "mastery_update"
    concept_id: str
    score: float = Field(ge=0.0, le=1.0)
    evidence_note: Optional[str] = None

    @field_validator("concept_id")
    @classmethod
    def _valid_concept(cls, value: str) -> str:
        if not is_valid_concept(value):
            raise ValueError(f"unknown concept_id: {value}")
        return value


class SessionEndEvent(BaseModel):
    type: Literal["session_end"] = "session_end"
    current_project_id: Optional[str] = None
    current_topic: Optional[str] = None
    pending_task: Optional[str] = None


class ProjectSavedEvent(BaseModel):
    type: Literal["project_saved"] = "project_saved"
    project_id: str
    title: str
    structure: dict[str, Any] = Field(default_factory=dict)
    highlights: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class PreferenceSetEvent(BaseModel):
    type: Literal["preference_set"] = "preference_set"
    key: str
    value: str
    source: Literal["self", "inferred"] = "self"


class ErrorObservedEvent(BaseModel):
    type: Literal["error_observed"] = "error_observed"
    concept_id: Optional[str] = None
    description: str
    context: Optional[str] = None

    @field_validator("concept_id")
    @classmethod
    def _valid_optional_concept(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not is_valid_concept(value):
            raise ValueError(f"unknown concept_id: {value}")
        return value


Event = Union[
    MasteryUpdateEvent,
    SessionEndEvent,
    ProjectSavedEvent,
    PreferenceSetEvent,
    ErrorObservedEvent,
]


class EventEnvelope(BaseModel):
    event: Event = Field(discriminator="type")


# ===== Dialog =====


class DialogTurn(BaseModel):
    role: Literal["agent", "student"]
    content: str


class DialogBatch(BaseModel):
    turns: list[DialogTurn]


# ===== Read-side records =====


class MasteryRecord(BaseModel):
    concept_id: str
    score: float
    evidence_count: int
    last_updated: datetime


class SessionState(BaseModel):
    last_session_at: Optional[datetime] = None
    current_project_id: Optional[str] = None
    current_topic: Optional[str] = None
    pending_task: Optional[str] = None


class ProjectRecord(BaseModel):
    project_id: str
    title: str
    structure: dict[str, Any]
    highlights: list[str]
    issues: list[str]
    created_at: datetime


class PreferenceRecord(BaseModel):
    key: str
    value: str
    source: str
    updated_at: datetime


class DialogSummaryRecord(BaseModel):
    turn_id: int
    summary: str
    created_at: datetime


class MisconceptionRecord(BaseModel):
    pattern_id: str
    description: str
    occurrences: int
    last_seen: datetime


class StudentState(BaseModel):
    student_id: str
    mastery: list[MasteryRecord]
    session: SessionState
    projects: list[ProjectRecord]
    preferences: list[PreferenceRecord]
    misconceptions: list[MisconceptionRecord]


class RecallHit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["dialog", "misconception"]
    score: float
    payload: dict[str, Any]


# ===== Agent integration records =====


class AgentTurnWrite(BaseModel):
    """One persisted PracticeAgent diagnostic turn.

    This is intentionally separate from the A-F teaching memories. It
    stores the orchestration facts needed to reconstruct MemorySnapshot:
    recent feedback, repeated error types, and feedback level.
    """

    session_id: str = ""
    task_id: str = ""
    attempt_id: Optional[str] = None
    project_name: str = ""
    passed: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_error_type: str = ""
    concept_id: Optional[str] = None
    feedback_level: int = Field(default=1, ge=1, le=3)
    feedback_text: str = ""
    user_text: str = ""
    diff_summary: str = ""
    highlight_node_ids: list[str] = Field(default_factory=list)
    need_rag: bool = False
    need_repair_validation: bool = False

    @field_validator("concept_id")
    @classmethod
    def _valid_optional_concept(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not is_valid_concept(value):
            raise ValueError(f"unknown concept_id: {value}")
        return value


class AgentTurnRecord(AgentTurnWrite):
    id: int
    created_at: datetime


class AffectiveEventWrite(BaseModel):
    """A lightweight affective/learning-state signal.

    Scores are bounded and should be treated as strategy hints, not
    psychological labels.
    """

    session_id: str = ""
    task_id: str = ""
    source: Literal["text", "behavior", "agent_inferred"] = "agent_inferred"
    frustration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""


class AffectiveEventRecord(AffectiveEventWrite):
    id: int
    created_at: datetime


class AgentMemorySnapshot(BaseModel):
    recent_feedbacks: list[str] = Field(default_factory=list)
    recent_error_types: list[str] = Field(default_factory=list)
    recent_feedback_level: int = 1
    repeated_error_count: int = 0
    mastered_concepts: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    affective_state: dict[str, float] = Field(default_factory=dict)


__all__ = [
    "MasteryUpdateEvent",
    "SessionEndEvent",
    "ProjectSavedEvent",
    "PreferenceSetEvent",
    "ErrorObservedEvent",
    "Event",
    "EventEnvelope",
    "DialogTurn",
    "DialogBatch",
    "MasteryRecord",
    "SessionState",
    "ProjectRecord",
    "PreferenceRecord",
    "DialogSummaryRecord",
    "MisconceptionRecord",
    "StudentState",
    "RecallHit",
    "AgentTurnWrite",
    "AgentTurnRecord",
    "AffectiveEventWrite",
    "AffectiveEventRecord",
    "AgentMemorySnapshot",
]
