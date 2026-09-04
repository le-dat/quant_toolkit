"""Workspace memory: trạng thái dùng chung giữa các lệnh gọi công cụ trong cùng một lần chạy.

Trạng thái thời gian chạy nhẹ nhàng — chỉ tồn tại trong một lần gọi AgentLoop.run().
Lưu trữ liên phiên làm việc được xử lý bởi pseud.src.memory.persistent.PersistentMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class WorkspaceMemory:
    """Trạng thái workspace dùng chung giữa các công cụ trong một lần chạy agent.

    Thuộc tính:
        run_dir: Đường dẫn thư mục của lượt chạy hiện tại.
        counters: Bộ đếm số lần gọi công cụ.
    """

    run_dir: Optional[str] = None
    counters: Dict[str, int] = field(default_factory=dict)

    def increment(self, key: str) -> int:
        """Tăng bộ đếm và trả về giá trị mới.

        Args:
            key: Khóa bộ đếm (thường là tên công cụ).

        Returns:
            Giá trị bộ đếm đã cập nhật.
        """
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def to_summary(self) -> str:
        """Tạo bản tóm tắt trạng thái cho LLM.

        Bao gồm run_dir và bộ đếm công cụ.
        Bản tóm tắt này tồn tại qua quá trình nén ngữ cảnh và giúp LLM
        ghi nhớ những gì nó đang xử lý.

        Returns:
            Văn bản tóm tắt trạng thái.
        """
        lines: list[str] = []
        if self.run_dir:
            lines.append(f"- run_dir: {self.run_dir}")
        if self.counters:
            counter_parts = [f"{k}={v}" for k, v in self.counters.items()]
            lines.append(f"- counters: {', '.join(counter_parts)}")
        return "\n".join(lines) if lines else "(empty state)"

