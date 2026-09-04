"""Đăng ký công cụ (Tool registry): tự động phát hiện qua BaseTool.__subclasses__().

Thêm một công cụ mới:
  1. Tạo một tệp trong src/tools/ chứa lớp kế thừa từ BaseTool
  2. Hoàn tất. Công cụ sẽ tự động được phát hiện và đăng ký.

Các công cụ thiếu thư viện phụ thuộc có thể ghi đè check_available() -> False
để được bỏ qua âm thầm khỏi registry.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pseud.agent.tools import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from pseud.config.schema import AgentConfig
    from pseud.memory.persistent import PersistentMemory

logger = logging.getLogger(__name__)

_SUBCLASSES_CACHE: list[type[BaseTool]] | None = None
_SHELL_TOOL_NAMES = {"bash", "background_run", "cancel_background"}


def _discover_subclasses() -> list[type[BaseTool]]:
    """Import tất cả các module trong package này, sau đó thu thập các lớp con của BaseTool.

    Kết quả được lưu vào cache sau lần gọi đầu tiên.

    Returns:
        Danh sách các lớp con cụ thể của BaseTool có tên không rỗng.
    """
    global _SUBCLASSES_CACHE
    if _SUBCLASSES_CACHE is not None:
        return _SUBCLASSES_CACHE

    pkg_dir = str(Path(__file__).parent)
    for _, module_name, _ in pkgutil.walk_packages([pkg_dir], prefix="pseud.tools."):
        leaf_name = module_name.split(".")[-1]
        if leaf_name.startswith("_"):
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.warning("Bỏ qua %s: %s", module_name, exc)


    classes: list[type[BaseTool]] = []
    queue = deque(BaseTool.__subclasses__())
    while queue:
        cls = queue.popleft()
        if cls.name:
            classes.append(cls)
        queue.extend(cls.__subclasses__())

    _SUBCLASSES_CACHE = classes
    return classes


def build_registry(
    *,
    persistent_memory: "PersistentMemory | None" = None,
    include_shell_tools: bool = False,
    agent_config: "AgentConfig | None" = None,
    session_id: str | None = None,
    event_callback: Callable[[str, dict], None] | None = None,
    warn_callback: Callable[[str], None] | None = None,
    interactive: bool | None = None,
) -> ToolRegistry:
    """Xây dựng registry công cụ thông qua tự động phát hiện.

    Các công cụ cục bộ được phát hiện và đăng ký.

    Args:
        persistent_memory: Thể hiện PersistentMemory dùng chung.
        include_shell_tools: Có bao gồm các công cụ thực thi câu lệnh shell hay không.
        agent_config: Cấu hình agent tùy chọn.
        session_id: Mã phiên hiện tại injected vào các công cụ cục bộ.
        event_callback: Callback sự kiện tùy chọn cho các công cụ cục bộ.
        warn_callback: Callback cảnh báo cho vận hành viên.
        interactive: Liệu phiên làm việc có phải là TTY tương tác hay không.

    Returns:
        ToolRegistry chứa tất cả công cụ cục bộ khả dụng.
    """

    from pseud.tools.remember_tool import RememberTool

    registry = ToolRegistry()
    for cls in _discover_subclasses():
        try:
            if cls.name in _SHELL_TOOL_NAMES and not include_shell_tools:
                logger.info("Tool %s disabled by shell tool policy", cls.name)
                continue
            if not cls.check_available():
                logger.info("Tool %s unavailable, skipping", cls.name)
                continue
            if cls is RememberTool and persistent_memory is not None:
                registry.register(cls(memory=persistent_memory))
            else:
                registry.register(cls())
        except Exception as exc:
            logger.warning("Failed to register tool %s: %s", cls.name, exc)

    return registry


def build_filtered_registry(tool_names: list[str], *, include_shell_tools: bool = False) -> ToolRegistry:
    """Xây dựng ToolRegistry chỉ chứa các công cụ được chỉ định.

    Args:
        tool_names: Tên các công cụ cần bao gồm.
        include_shell_tools: Có bao gồm các công cụ thực thi shell đã được lọc hay không.

    Returns:
        ToolRegistry chỉ chứa các công cụ được yêu cầu.
    """
    full = build_registry(include_shell_tools=include_shell_tools)
    return _filter_registry(full, tool_names, include_shell_tools=include_shell_tools)


def build_swarm_registry(
    tool_names: list[str],
    *,
    agent_config: "AgentConfig | None" = None,
    include_shell_tools: bool = False,
) -> ToolRegistry:
    """Xây dựng registry cho từng worker Swarm dựa trên danh sách trắng.

    Worker Swarm nhận một danh sách trắng nghiêm ngặt (``agent_spec.tools``). Trình
    xây dựng này tôn trọng danh sách trắng đó để lọc ra các công cụ khả dụng.

    Args:
        tool_names: Danh sách trắng công cụ cho từng agent từ preset.
        agent_config: Cấu hình agent tùy chọn.
        include_shell_tools: Các công cụ thực thi shell có hợp lệ hay không.

    Returns:
        ToolRegistry chứa các công cụ cục bộ đã lọc theo danh sách trắng.
    """
    full = build_registry(
        agent_config=agent_config,
        include_shell_tools=include_shell_tools,
    )
    return _filter_registry(full, tool_names, include_shell_tools=include_shell_tools)


def _filter_registry(
    full: ToolRegistry,
    tool_names: list[str],
    *,
    include_shell_tools: bool,
) -> ToolRegistry:
    """Chiếu registry đầy đủ xuống danh sách trắng với nhật ký bỏ qua nhất quán."""
    filtered = ToolRegistry()
    for name in tool_names:
        tool = full.get(name)
        if tool:
            filtered.register(tool)
        else:
            logger.warning(
                "Requested tool %r is unavailable and was dropped from the "
                "filtered registry (include_shell_tools=%s); a preset that "
                "depends on it cannot execute it.",
                name, include_shell_tools,
            )
    return filtered


__all__ = ["build_registry", "build_filtered_registry", "build_swarm_registry"]
