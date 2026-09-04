"""Các hàm hỗ trợ serial hóa dùng chung cho các ranh giới đọc Swarm.

Nguồn sự thật duy nhất để chiếu một :class:`SwarmTask` thành dict JSON theo từng tác vụ
được trả về bởi các công cụ MCP (``run_swarm`` / ``get_swarm_status`` / ``get_run_result``)
và công cụ agent ``run_swarm`` trong tiến trình.
"""

from __future__ import annotations

from typing import Any

from pseud.tools.redaction import redact_internal_paths


def serialize_task(task: Any) -> dict:
    """Chuyển đổi một SwarmTask thành dict công khai theo từng tác vụ.

    ``error`` và ``iterations`` luôn được bao gồm để các tác vụ thất bại hoặc suy giảm
    có thể được chẩn đoán từ mọi đường dẫn đọc.
    """
    status = task.status.value if hasattr(task.status, "value") else str(task.status)
    return {
        "id": task.id,
        "agent_id": task.agent_id,
        "status": status,
        # `TaskStatus` làm phẳng bốn kết cục worker vào `failed`; trường này giữ giá trị
        # thô để phân biệt `incomplete` với `failed` ở mọi đường đọc (M-RS0 §1.2).
        "worker_status": getattr(task, "worker_status", "") or "",
        "summary": task.summary,
        "iterations": getattr(task, "worker_iterations", 0),
        "error": redact_internal_paths(getattr(task, "error", None)) or None,
        "started_at": getattr(task, "started_at", None),
        "completed_at": getattr(task, "completed_at", None),
        "depends_on": list(getattr(task, "depends_on", []) or []),
        "blocked_by": list(getattr(task, "blocked_by", []) or []),
    }


def run_level_error(run: Any) -> str | None:
    """Lỗi của tác vụ thất bại đầu tiên, dành cho trường ``error`` cấp cao nhất.

    Trả về ``None`` khi không có tác vụ nào chứa lỗi.
    """
    for task in getattr(run, "tasks", None) or []:
        err = getattr(task, "error", None)
        if err:
            return f"{task.id}/{task.agent_id}: {redact_internal_paths(err)}"
    return None
