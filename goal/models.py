"""Các mô hình dữ liệu cho mục tiêu nghiên cứu tài chính (finance research goals)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GoalStatus(str, Enum):
    """Các trạng thái vòng đời của một mục tiêu nghiên cứu tài chính."""

    ACTIVE = "active"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    NEEDS_REFRESH = "needs_refresh"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    COMPLIANCE_BLOCKED = "compliance_blocked"
    BUDGET_LIMITED = "budget_limited"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class RiskTier(str, Enum):
    """Phân loại mức độ rủi ro cho các mục tiêu."""

    RESEARCH_GENERAL = "research_general"


class StaleGoalError(ValueError):
    """Ném ra khi một lượt của mô hình cố gắng chỉnh sửa một mục tiêu cũ hoặc đã bị thay thế."""


@dataclass(frozen=True)
class GoalRecord:
    """Bản ghi mục tiêu nghiên cứu tài chính đã lưu trữ."""

    goal_id: str
    session_id: str
    status: GoalStatus
    objective: str
    ui_summary: str
    source: str = "user"
    protocol: str = "research"
    risk_tier: RiskTier = RiskTier.RESEARCH_GENERAL
    token_budget: int | None = None
    tokens_used: int = 0
    turn_budget: int | None = None
    turns_used: int = 0
    time_budget_seconds: int | None = None
    time_used_seconds: int = 0
    budget_wrapup_sent: bool = False
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    recap: str | None = None


@dataclass(frozen=True)
class GoalClaim:
    """Một tuyên bố nghiên cứu được theo dõi bởi sổ cái mục tiêu."""

    claim_id: str
    goal_id: str
    session_id: str
    claim_type: str
    text: str
    status: str
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class GoalCriterion:
    """Một tiêu chí giao thức phải được đáp ứng trước khi hoàn thành."""

    criterion_id: str
    goal_id: str
    session_id: str
    text: str
    required: bool = True
    status: str = "pending"
    freshness_requirement: str | None = None
    protocol_step: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class EvidenceInput:
    """Đầu vào để đính kèm bằng chứng có thể truy xuất nguồn gốc vào một mục tiêu."""

    text: str
    criterion_id: str | None = None
    claim_id: str | None = None
    verification_status: str = "unverified"
    source_type: str = "observation"
    source_provider: str | None = None
    data_as_of: str | None = None
    raw_payload_ref: str | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    """Bản ghi bằng chứng nghiên cứu đã được lưu trữ."""

    evidence_id: str
    goal_id: str
    session_id: str
    criterion_id: str | None = None
    claim_id: str | None = None
    evidence_type: str = ""
    text: str = ""
    tool_call_id: str | None = None
    run_id: str | None = None
    source_provider: str | None = None
    source_type: str = "observation"
    source_uri: str | None = None
    symbol_universe: list[str] = field(default_factory=list)
    benchmark: list[str] = field(default_factory=list)
    timeframe: str | None = None
    method: str | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_hash: str | None = None
    retrieved_at: str = ""
    data_as_of: str | None = None
    freshness_status: str = "fresh"
    verification_status: str = "unverified"
    confidence: float | None = None
    caveat: str | None = None
    contradicts_claim_ids: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass(frozen=True)
class AuditRow:
    """Bản ghi kiểm toán tiêu chí khi hoàn thành mục tiêu."""

    criterion_id: str
    result: str
    evidence_ids: list[str] = field(default_factory=list)
    notes: str = ""

