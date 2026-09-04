"""Công cụ `load_skill` — tiết lộ tiệm tiến cho 5 kỹ năng alpha của L2.6.

Vì sao tồn tại: system prompt chỉ tiêm mô tả một dòng mỗi kỹ năng (M-RS0 §1.1 #11).
Không có đường nạp toàn văn thì cơ chế đó vô nghĩa — agent biết kỹ năng tồn tại nhưng
không bao giờ đọc được nội dung. Đây là bản công cụ nạp kỹ năng tối giản, phục vụ thư mục `pseud/src/skills/`.

Đối lập là nhét cả 5 SKILL.md vào system prompt (~13 KB, ~3–4K token MỖI yêu cầu) —
đúng thứ tiết lộ tiệm tiến sinh ra để tránh.
"""

from __future__ import annotations

import json
from typing import Any

from pseud.agent.skills import SkillsLoader
from pseud.agent.tools import BaseTool


class LoadSkillTool(BaseTool):
    """Nạp toàn văn một kỹ năng theo tên."""

    name = "load_skill"
    description = (
        "Read the full documentation of one alpha-research skill. "
        "Call this BEFORE starting the work that skill governs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name, exactly as listed in the system prompt.",
            }
        },
        "required": ["name"],
    }
    is_readonly = True
    repeatable = True

    def __init__(self, skills_loader: SkillsLoader | None = None):
        self._loader = skills_loader or SkillsLoader()

    def execute(self, **kwargs: Any) -> str:
        ten = str(kwargs.get("name") or "").strip()
        if not ten:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Thiếu tham số 'name'.",
                    "available": [s.name for s in self._loader.skills],
                },
                ensure_ascii=False,
            )

        noi_dung = self._loader.get_content(ten)
        # `get_content` trả chuỗi lỗi bắt đầu bằng "Error:" khi không tìm thấy. Chuyển
        # thành `ok: false` để agent thấy đây là thất bại công cụ chứ không phải nội dung
        # kỹ năng — luật số 2 của system prompt dựa vào phân biệt này.
        if noi_dung.startswith("Error:"):
            return json.dumps(
                {
                    "ok": False,
                    "error": noi_dung,
                    "available": [s.name for s in self._loader.skills],
                },
                ensure_ascii=False,
            )

        return json.dumps({"ok": True, "name": ten, "content": noi_dung}, ensure_ascii=False)
