"""BaseTool + ToolRegistry: hạ tầng cho các công cụ (tools)."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Lớp cơ sở cho công cụ.

    Thuộc tính:
        name: Định danh duy nhất cho công cụ.
        description: Mô tả công cụ hiển thị cho LLM.
        parameters: Định nghĩa tham số theo định dạng JSON Schema.
        repeatable: Liệu công cụ có thể được gọi nhiều hơn một lần hay không.
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    repeatable: bool = False
    is_readonly: bool = True

    @classmethod
    def check_available(cls) -> bool:
        """Kiểm tra xem các phụ thuộc của công cụ này đã đáp ứng hay chưa.

        Ghi đè trong lớp con để kiểm tra API key, package, v.v.
        Các công cụ trả về False sẽ bị loại khỏi registry.
        """
        return True

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Thực thi công cụ và trả về một chuỗi JSON."""

    def to_openai_schema(self) -> Dict[str, Any]:
        """Chuyển đổi sang định dạng gọi hàm của OpenAI (OpenAI function calling format)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}, "required": []},
            },
        }


class ToolRegistry:
    """Registry lưu trữ và quản lý công cụ."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Đăng ký một công cụ."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Lấy công cụ theo tên."""
        return self._tools.get(name)

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Trả về tất cả công cụ theo định dạng OpenAI function calling."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, params: Dict[str, Any]) -> str:
        """Thực thi một công cụ và đảm bảo kết quả trả về là JSON hợp lệ."""
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"status": "error", "error": f"Tool '{name}' not found"}, ensure_ascii=False)
        try:
            return tool.execute(**params)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return json.dumps({
                "status": "error", "tool": name,
                "error": str(exc),
            }, ensure_ascii=False)

    def list_tools(self) -> List[BaseTool]:
        """Danh sách tất cả các công cụ đã đăng ký."""
        return list(self._tools.values())

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


