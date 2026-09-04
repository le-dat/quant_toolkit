"""Các hàm hỗ trợ ngữ cảnh cho mục tiêu nghiên cứu (research goals) phạm vi phiên."""

from __future__ import annotations

from typing import Any

OPEN_CRITERION_STATUSES = {"", "pending", "open", "unsatisfied", "missing", "stale", "too_weak"}
CONTINUABLE_GOAL_STATUSES = {"active", "needs_refresh", "insufficient_evidence"}


DEFAULT_GOAL_CRITERIA = [
    "Formulate 16-Family Alpha Hypothesis",
]


def default_goal_criteria() -> list[str]:
    """Trả về danh sách tiêu chí nghiên cứu mặc định."""
    return list(DEFAULT_GOAL_CRITERIA)


def format_goal_context(snapshot: dict[str, Any]) -> str:
    """Định dạng một khối ngữ cảnh mục tiêu nghiên cứu gọn nhẹ cho agent prompt.

    Args:
        snapshot: Snapshot mục tiêu trả về từ GoalStore.

    Returns:
        Khối ngữ cảnh dạng XML tinh gọn được tiêm vào lượt tiếp theo của mô hình.
    """
    goal = snapshot["goal"]
    lines = [
        "<current-research-goal>",
        f"goal_id: {goal['goal_id']}",
        f"objective: {goal['objective']}",
        f"status: {goal['status']}",
        "</current-research-goal>",
    ]
    return "\n".join(lines)


def criterion_is_covered(snapshot: dict[str, Any], criterion: dict[str, Any]) -> bool:
    """Trả về liệu một tiêu chí đã được đáp ứng bởi trạng thái hoặc bằng chứng hay chưa."""
    status = str(criterion.get("status") or "").lower()
    if status not in OPEN_CRITERION_STATUSES:
        return True
    criterion_id = criterion.get("criterion_id")
    return any(item.get("criterion_id") == criterion_id for item in snapshot.get("evidence") or [])


def goal_progress_tuple(snapshot: dict[str, Any]) -> tuple[int, int]:
    """Trả về tiến độ so sánh được ở dạng ``(covered_criteria, evidence_count)``."""
    criteria = snapshot.get("criteria") or []
    covered = sum(1 for item in criteria if criterion_is_covered(snapshot, item))
    return covered, int(snapshot.get("evidence_count") or len(snapshot.get("evidence") or []))


def goal_needs_continuation(snapshot: dict[str, Any]) -> bool:
    """Trả về liệu runtime có cần tiếp tục thực hiện mục tiêu này hay không."""
    goal = snapshot.get("goal") or {}
    status = str(goal.get("status") or "").lower()
    if status not in CONTINUABLE_GOAL_STATUSES:
        return False
    return True


def format_goal_continuation_prompt(snapshot: dict[str, Any], previous_answer: str = "") -> str:
    """Xây dựng prompt tự động gọn nhẹ để tiếp tục một mục tiêu chưa hoàn thành."""
    goal = snapshot["goal"]
    return "\n".join(
        [
            "<goal-continuation>",
            f"goal_id: {goal['goal_id']}",
            f"objective: {goal['objective']}",
            f"status: {goal.get('status', 'active')}",
            "Instructions: Continue hypothesis formulation and diagnostic refinement towards this objective.",
            "</goal-continuation>",
        ]
    )


def get_current_goal_context(session_id: str) -> tuple[str, str | None]:
    """Trả về ngữ cảnh mục tiêu đang hoạt động và ID mục tiêu cho một phiên làm việc.

    Args:
        session_id: ID của chat/phiên hiện tại.

    Returns:
        Tuple gồm khối ngữ cảnh đã định dạng và ID mục tiêu đang hoạt động.
    """
    if not session_id.strip():
        return "", None
    from pseud.goal.store import GoalStore

    snapshot = GoalStore().get_current_snapshot(session_id)
    if snapshot is None:
        return "", None
    return format_goal_context(snapshot), str(snapshot["goal"]["goal_id"])
