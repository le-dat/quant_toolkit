"""Bộ phân tích frontmatter kiểu YAML dùng chung cho các tệp skill và memory."""

from __future__ import annotations

import re
from typing import Any, Dict

# --- mở đầu, các dòng meta tùy chọn, --- kết thúc. Hàng rào kết thúc có thể nằm ở
# EOF (không có dòng mới phía sau) hoặc theo sau là phần thân; meta rỗng (---\n---) hợp lệ.
_FRONTMATTER_RE = re.compile(
    r"^---[ \t]*\r?\n(?:(.*?)\r?\n)?---[ \t]*(?:\r?\n(.*))?$",
    re.DOTALL,
)


def parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Phân tích frontmatter dạng YAML và phần thân từ tệp markdown.

    Hỗ trợ các giá trị kiểu chuỗi, danh sách (``[a, b]``), và giá trị luận lý (boolean).

    Args:
        text: Văn bản Markdown có phần frontmatter tùy chọn giới hạn bởi ``---``.

    Returns:
        Tuple gồm (dict metadata, văn bản phần thân).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    meta: Dict[str, Any] = {}
    for line in (match.group(1) or "").strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",")]
            meta[key] = [i for i in items if i]
        elif value.lower() in ("true", "false"):
            meta[key] = value.lower() == "true"
        else:
            meta[key] = value

    body = match.group(2) or ""
    return meta, body.strip()

