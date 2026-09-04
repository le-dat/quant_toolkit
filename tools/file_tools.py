"""Công cụ tệp phạm vi `run_dir` — `read_file` và `write_file`.

Vì sao tồn tại: hợp đồng sản phẩm đầu ra của swarm neo vào `artifact_dir/report.md`
([worker.py](../swarm/worker.py) `_report_written` / `_resolve_summary`), và cả 18 agent
trong `swarm/presets/*.yaml` đều khai `read_file` / `write_file` trong danh sách trắng.
Nhưng registry KHÔNG có hai công cụ đó, nên `_filter_registry` lặng lẽ loại chúng: worker
bị lệnh "MUST call write_file with path report.md" mà không có đường ghi ⇒ mọi worker kết
thúc `incomplete`. Đây đúng dạng hỏng M-RS0 §1.2 đã nêu — prompt hứa công cụ không tồn tại.

Phạm vi: mọi đường dẫn giải quyết qua `safe_path(p, run_dir)`, nên không thoát khỏi thư
mục chạy và không nhận đường dẫn UNC. L2.6 KHÔNG chạy mã sinh (đã gỡ `core/runner.py`),
nên ở đây không có công cụ shell nào — chỉ đọc/ghi tệp trong phạm vi run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pseud.agent.tools import BaseTool
from pseud.tools.path_utils import safe_path

# Trần đọc. Vượt trần thì cắt NHƯNG khai báo `truncated` + `omitted_bytes` — nguyên tắc
# M-RS2 §2: cắt im lặng là thứ khiến mô hình bịa ra phần đuôi.
MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 2_000_000


def _resolve(kwargs: dict[str, Any], path_value: str) -> tuple[Path | None, str | None]:
    """Giải quyết `path_value` bên trong `run_dir` được tiêm vào lời gọi công cụ.

    Args:
        kwargs: Đối số công cụ (worker/AgentLoop tiêm sẵn khóa ``run_dir``).
        path_value: Đường dẫn tương đối do mô hình cung cấp.

    Returns:
        Cặp ``(đường_dẫn, lỗi)`` — đúng một phần tử khác None.
    """
    run_dir = str(kwargs.get("run_dir") or "").strip()
    if not run_dir:
        return None, "run_dir is not available in this context"
    if not path_value:
        return None, "Missing required parameter 'path'"
    try:
        return safe_path(path_value, Path(run_dir)), None
    except ValueError as exc:
        return None, str(exc)


class ReadFileTool(BaseTool):
    """Đọc một tệp văn bản bên trong thư mục chạy."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file from the current run directory. Paths are relative to the run "
        "directory and may not escape it. Large files are truncated with an explicit "
        "truncated/omitted_bytes flag — never assume you saw the whole file without checking."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the run directory, e.g. 'report.md'.",
            }
        },
        "required": ["path"],
    }
    repeatable = True
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        path, error = _resolve(kwargs, str(kwargs.get("path") or "").strip())
        if error:
            return json.dumps({"status": "error", "error": error}, ensure_ascii=False)

        assert path is not None
        if not path.is_file():
            req_path = str(kwargs.get("path") or "").strip()
            clean_skill = req_path.replace("skills/", "").replace(".md", "").strip()
            if "skill" in req_path.lower() or clean_skill in (
                "tool-surface-and-limits",
                "alpha-hypothesis-writing",
                "anomaly-scan-protocol",
                "cemetery-cross-check",
                "ml-best-practices",
            ):
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            f"File not found: {req_path!r}. "
                            f"To read skill documentation, call `load_skill(name={clean_skill!r})` tool instead of `read_file`."
                        ),
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {"status": "error", "error": f"File not found: {kwargs.get('path')!r}"},
                ensure_ascii=False,
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

        truncated = len(raw) > MAX_READ_BYTES
        body = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        return json.dumps(
            {
                "status": "ok",
                "path": str(kwargs.get("path")),
                "content": body,
                "bytes": len(raw),
                "truncated": truncated,
                "omitted_bytes": len(raw) - MAX_READ_BYTES if truncated else 0,
            },
            ensure_ascii=False,
        )


class WriteFileTool(BaseTool):
    """Ghi một tệp văn bản bên trong thư mục chạy."""

    name = "write_file"
    description = (
        "Write a UTF-8 text file into the current run directory, creating parent directories as "
        "needed. Paths are relative to the run directory and may not escape it. Writing "
        "'report.md' is how a swarm worker delivers its final report."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the run directory, e.g. 'report.md'.",
            },
            "content": {
                "type": "string",
                "description": "Full file content. The file is overwritten, not appended to.",
            },
        },
        "required": ["path", "content"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        path, error = _resolve(kwargs, str(kwargs.get("path") or "").strip())
        if error:
            return json.dumps({"status": "error", "error": error}, ensure_ascii=False)

        content = kwargs.get("content")
        if content is None:
            return json.dumps(
                {"status": "error", "error": "Missing required parameter 'content'"},
                ensure_ascii=False,
            )
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        payload = content.encode("utf-8")
        if len(payload) > MAX_WRITE_BYTES:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Content is {len(payload)} bytes, over the {MAX_WRITE_BYTES} limit",
                },
                ensure_ascii=False,
            )

        assert path is not None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # `newline="\n"`: mặc định trên Windows dịch \n thành \r\n, khiến số byte báo cho
            # model lệch với số byte trên đĩa — và một sổ cái chống bịa số không nên tự phát ra
            # con số sai. Ghi LF nguyên vẹn để đọc lại đúng bằng cái đã ghi.
            path.write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

        return json.dumps(
            {"status": "ok", "path": str(kwargs.get("path")), "bytes": len(payload)},
            ensure_ascii=False,
        )
