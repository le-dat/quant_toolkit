"""Các tiện ích nạp cấu hình agent có cấu trúc."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from pseud.config.paths import get_config_path, get_runtime_root
from pseud.config.schema import AgentConfig, AgentConfigOverride, MCPServerConfig

logger = logging.getLogger(__name__)

_SWARM_AGENT_CONFIG_ENV_VAR = "KAIROS_SWARM_AGENT_CONFIG"
_SWARM_AGENT_CONFIG_FILENAME = "swarm-agent.json"
_MAIN_AGENT_FALLBACK_FILENAMES = ("agent.json", "agent.yaml", "agent.yml")

import yaml


def load_agent_config(config_path: Path | None = None) -> AgentConfig:
    """Nạp cấu hình agent từ đĩa với fallback an toàn.

    Args:
        config_path: Đường dẫn cấu hình tùy chọn. Khi bỏ qua, đường dẫn mặc định được sử dụng.

    Returns:
        Cấu hình agent đã qua xác minh. Tệp cấu hình không hợp lệ hoặc không đọc được sẽ fallback về ``AgentConfig()``.
    """
    path = get_config_path(config_path)

    if not path.exists():
        return AgentConfig()

    try:
        raw = _read_config_file(path)
        return AgentConfig.model_validate(raw)
    except (OSError, ValueError, ValidationError) as exc:
        logger.warning(
            "Failed to load agent config from %s: %s",
            path,
            type(exc).__name__,
        )
        logger.debug("Agent config load error details: %s", exc)
        return AgentConfig()


def merge_agent_config_overrides(
    config: AgentConfig,
    overrides: Mapping[str, Any] | None,
) -> AgentConfig:
    """Hợp nhất các ghi đè runtime lên trên cấu hình gốc.

    Args:
        config: Cấu hình agent gốc nạp từ đĩa hoặc mặc định.
        overrides: Các ghi đè runtime, thường lấy từ cấu hình cấp phiên.

    Returns:
        Cấu hình mới đã xác minh chứa kết quả đã hợp nhất.
    """
    if not overrides:
        return config

    try:
        override_model = AgentConfigOverride.model_validate(dict(overrides))
    except ValidationError as exc:
        logger.warning(
            "Ignoring invalid agent config overrides (%s): %s — using base config",
            type(exc).__name__,
            [str(e["loc"]) for e in exc.errors()],
        )
        return config

    merged = _merge_agent_config_dicts(
        config.model_dump(mode="json"),
        override_model.model_dump(mode="json", exclude_unset=True),
    )
    try:
        return AgentConfig.model_validate(merged)
    except ValidationError as exc:
        logger.warning(
            "Ignoring merged agent config overrides after validation failure (%s): %s — using base config",
            type(exc).__name__,
            [str(e["loc"]) for e in exc.errors()],
        )
        return config



# Các khóa trong ghi đè phiên làm việc mang định nghĩa tiến trình con và
# do đó yêu cầu quyền tin cậy cấp quản trị viên thay vì từ API caller.
_SESSION_RESTRICTED_KEYS: frozenset[str] = frozenset({"mcpServers", "mcp_servers"})


def sanitize_session_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Loại bỏ các khóa chỉ dành cho quản trị viên khỏi các ghi đè phiên do API caller cung cấp.

    ``mcpServers`` / ``mcp_servers`` định nghĩa các thuộc tính tiến trình con ``command``/``args``/``env``
    và do đó trao quyền thực thi. Chúng phải bắt nguồn từ tệp cấu hình trên đĩa
    được quản trị viên kiểm soát, không phải từ các API caller chưa xác thực hoặc bán tin cậy.
    Quản trị viên muốn cho phép tiêm MCP ở cấp phiên có thể đặt ``ALLOW_SESSION_MCP_SERVERS=1``.

    Args:
        overrides: Dict cấu hình phiên thô nhận từ API caller.

    Returns:
        Một dict mới đã loại bỏ các khóa bị hạn chế (hoặc dict gốc khi bật tùy chọn môi trường).
    """
    if os.environ.get("ALLOW_SESSION_MCP_SERVERS", "").strip().lower() in {"1", "true", "yes"}:
        return dict(overrides)

    restricted_present = _SESSION_RESTRICTED_KEYS & overrides.keys()
    if restricted_present:
        logger.warning(
            "Stripped %s from session config overrides: MCP server definitions "
            "require operator-level trust (disk config). "
            "Set ALLOW_SESSION_MCP_SERVERS=1 to allow session-level injection.",
            sorted(restricted_present),
        )
    return {k: v for k, v in overrides.items() if k not in _SESSION_RESTRICTED_KEYS}


def load_runtime_agent_config(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AgentConfig:
    """Nạp cấu hình từ đĩa và áp dụng các ghi đè runtime.

    Args:
        config_path: Đường dẫn tệp cấu hình chỉ định tùy chọn.
        overrides: Ánh xạ ghi đè runtime được áp dụng lên trên cấu hình tệp.

    Returns:
        Cấu hình runtime đã hợp nhất.
    """
    config = load_agent_config(config_path)
    return merge_agent_config_overrides(config, overrides)


def _resolve_swarm_agent_config_path(
    *,
    runtime_root: Path | None = None,
) -> Path | None:
    """Chọn tệp cấu hình mà swarm runtime nên khởi động cùng.

    Thứ tự phân giải (ưu tiên đầu tiên):

    1. Biến môi trường ``KAIROS_SWARM_AGENT_CONFIG`` — đường dẫn ghi đè tuyệt đối
       cho triển khai CI / sandbox nơi gốc runtime chỉ đọc.
    2. ``<runtime_root>/swarm-agent.json`` — danh sách cho phép dành riêng cho swarm.
    3. ``<runtime_root>/{agent.json,agent.yaml,agent.yml}`` — fallback về cấu hình agent chính.
    4. ``None`` khi không có tệp nào khớp — giữ nguyên hành vi chạy các công cụ cục bộ.

    Mô hình tin cậy: các caller gọi swarm không thể can thiệp vào đường dẫn này —
    phân giải cấu hình là thao tác cấp quản trị viên lúc khởi động.

    Args:
        runtime_root: Đường dẫn thư mục gốc runtime tùy chọn. Mặc định là ``~/.kairos``.

    Returns:
        Đường dẫn cấu hình được chọn, hoặc ``None`` khi không có ứng viên nào khả dụng.
    """
    env_value = os.environ.get(_SWARM_AGENT_CONFIG_ENV_VAR, "").strip()
    if env_value:
        return Path(env_value).expanduser()

    root = runtime_root if runtime_root is not None else get_runtime_root()
    swarm_specific = root / _SWARM_AGENT_CONFIG_FILENAME
    if swarm_specific.exists():
        return swarm_specific

    for fallback in _MAIN_AGENT_FALLBACK_FILENAMES:
        candidate = root / fallback
        if candidate.exists():
            return candidate

    return None


def load_swarm_agent_config(
    *,
    runtime_root: Path | None = None,
) -> AgentConfig:
    """Nạp AgentConfig phía swarm theo thứ tự phân giải khởi động.

    Hàm này hỗ trợ nạp cấu hình trước khi khởi tạo ``SwarmRuntime``.
    Nó trả về một :class:`AgentConfig` (không bao giờ là ``None``).
    Một cấu hình rỗng (``mcp_servers={}``) được xử lý tương đương ``agent_config=None``.

    Args:
        runtime_root: Ghi đè thư mục gốc runtime tùy chọn. Mặc định là ``~/.kairos``.

    Returns:
        Cấu hình agent swarm đã xác minh, hoặc :class:`AgentConfig` rỗng
        khi không tìm thấy tệp hoặc tệp bị lỗi phân tích.
    """
    path = _resolve_swarm_agent_config_path(runtime_root=runtime_root)
    if path is None:
        return AgentConfig()
    return load_agent_config(path)


def _read_config_file(path: Path) -> dict[str, Any]:
    """Đọc tệp cấu hình được hỗ trợ thành một dictionary.

    Args:
        path: Đường dẫn tệp cấu hình cần giải mã.

    Returns:
        Đối tượng cấu hình đã giải mã dưới dạng dict.

    Raises:
        ValueError: Nếu định dạng tệp không được hỗ trợ, thiếu thư viện PyYAML,
            hoặc nội dung giải mã không phải là đối tượng dictionary.
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ValueError("YAML config is not available because PyYAML is missing")
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported config file format: {suffix or '<none>'}")

    if not isinstance(data, dict):
        raise ValueError("Agent config must decode to a JSON/YAML object")
    return data


def _merge_agent_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Hợp nhất dữ liệu cấu hình agent cấp cao nhất với việc thay thế MCP server phù hợp."""
    non_mcp_override = {key: value for key, value in override.items() if key != "mcp_servers"}
    merged = _merge_dicts(base, non_mcp_override)

    override_servers = override.get("mcp_servers")
    if not isinstance(override_servers, dict):
        if "mcp_servers" in override:
            merged["mcp_servers"] = override_servers
        return merged

    merged_servers = dict(base.get("mcp_servers", {}))
    for server_name, server_override in override_servers.items():
        current_server = merged_servers.get(server_name)
        if isinstance(current_server, dict) and isinstance(server_override, dict):
            merged_servers[server_name] = _merge_mcp_server_dicts(current_server, server_override)
        else:
            merged_servers[server_name] = server_override

    merged["mcp_servers"] = merged_servers
    return merged


def _merge_mcp_server_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Hợp nhất dữ liệu một MCP server, reset các trường transport không tương thích khi cần."""
    if _override_switches_transport(base, override):
        return _merge_dicts(_default_mcp_server_payload(base), override)
    return _merge_dicts(base, override)


def _override_switches_transport(base: dict[str, Any], override: dict[str, Any]) -> bool:
    """Trả về liệu ghi đè một phần có làm thay đổi nhóm transport của server hay không."""
    override_transport = _resolve_override_transport(override)
    if override_transport is None:
        return False
    base_transport = MCPServerConfig.model_validate(base).resolved_transport()
    return override_transport != base_transport


def _resolve_override_transport(override: dict[str, Any]) -> str | None:
    """Suy luận ý định transport từ ghi đè một phần của MCP server."""
    explicit_type = override.get("type")
    if explicit_type in {"stdio", "sse", "streamableHttp"}:
        return str(explicit_type)
    if any(key in override for key in ("command", "args", "env")):
        return "stdio"
    return None


def _default_mcp_server_payload(base: dict[str, Any]) -> dict[str, Any]:
    """Trả về payload MCP server trung tính về transport nhưng bảo toàn các giá trị mặc định khác."""
    enabled_tools = base.get("enabled_tools")
    return {
        "type": None,
        "command": "",
        "args": [],
        "env": {},
        "url": "",
        "headers": {},
        "tool_timeout": base.get("tool_timeout", 30.0),
        "init_timeout": base.get("init_timeout"),
        "enabled_tools": list(enabled_tools) if isinstance(enabled_tools, list) else ["*"],
    }


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Hợp nhất đệ quy hai dictionary đơn thuần.

    Args:
        base: Dictionary gốc.
        override: Dictionary ghi đè áp dụng lên trên ``base``.

    Returns:
        Dictionary đã hợp nhất nơi các ánh xạ lồng nhau được hợp nhất đệ quy và
        các giá trị vô hướng từ ``override`` thay thế giá trị trong ``base``.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged
