"""Các hàm hỗ trợ đường dẫn cho cấu hình có cấu trúc ở cấp độ agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_FILENAMES = ("agent.json", "agent.yaml", "agent.yml")

_HOME_ENV_VAR = "KAIROS_HOME"
_LEGACY_HOME_ENV_VAR = "VIBE_TRADING_HOME"

RUNTIME_DIR_NAME = ".kairos"
_LEGACY_RUNTIME_DIR_NAME = ".vibe-trading"


def get_runtime_root(config_path: Path | None = None) -> Path:
    """Trả về thư mục gốc runtime chứa trạng thái agent cấp người dùng.

    Đây là NGUỒN DUY NHẤT của gốc runtime. Mọi module khác phải gọi hàm này thay vì
    tự ghép ``Path.home() / "..."`` — neo cứng vào home làm đặt biến môi trường gốc
    thì mọi thứ chuyển chỗ TRỪ chỗ neo cứng, và dữ liệu lặng lẽ tách làm hai nơi.

    Args:
        config_path: Đường dẫn tệp cấu hình chỉ định tùy chọn. Khi được cung cấp,
            gốc runtime được suy ra từ thư mục cha của tệp đó.

    Returns:
        Thư mục chứa tệp cấu hình khi được chỉ định rõ; nếu không là đường dẫn đè từ
        ``KAIROS_HOME``; nếu không là ``~/.kairos``. Khi ``~/.kairos`` chưa tồn tại mà
        thư mục cũ ``~/.vibe-trading`` còn dữ liệu, hàm trả về thư mục cũ kèm cảnh báo
        di trú — mất dữ liệu người dùng tệ hơn nhiều so với một cái tên cũ còn sót.
    """
    if config_path is not None:
        return config_path.expanduser().parent

    env_root = os.environ.get(_HOME_ENV_VAR, "").strip()
    if not env_root:
        env_root = os.environ.get(_LEGACY_HOME_ENV_VAR, "").strip()
        if env_root:
            logger.warning(
                "%s đã lỗi thời, hãy đổi sang %s.", _LEGACY_HOME_ENV_VAR, _HOME_ENV_VAR
            )
    if env_root:
        if env_root.startswith(("//", "\\\\")):
            raise ValueError(
                f"{_HOME_ENV_VAR} must not be a UNC path: {env_root!r}"
            )
        return Path(env_root).expanduser()

    home = Path.home()
    root = home / RUNTIME_DIR_NAME
    if not root.exists():
        legacy = home / _LEGACY_RUNTIME_DIR_NAME
        if legacy.exists():
            logger.warning(
                "Đang dùng thư mục runtime cũ %s. Đổi tên nó thành %s để hoàn tất di trú.",
                legacy, root,
            )
            return legacy
    return root


def legacy_runtime_roots() -> list[Path]:
    """Trả về gốc runtime cũ nếu nó CÒN tồn tại trên đĩa, ngược lại danh sách rỗng.

    Dùng cho các danh sách trắng đường dẫn: giữ quyền đọc tệp còn nằm ở thư mục cũ trong
    lúc di trú, và **tự co lại** khi người dùng xoá/đổi tên thư mục đó. Nhờ vậy không cần
    ai nhớ quay lại gỡ một hằng số gõ cứng.
    """
    legacy = Path.home() / _LEGACY_RUNTIME_DIR_NAME
    return [legacy] if legacy.exists() else []


def get_sessions_dir() -> Path:
    """Trả về thư mục cấp người dùng chứa các bản ghi phiên trò chuyện."""
    return get_runtime_root() / "sessions"


def get_runs_dir() -> Path:
    """Trả về thư mục cấp người dùng chứa các artifact lượt chạy."""
    return get_runtime_root() / "runs"


def get_swarm_runs_dir() -> Path:
    """Trả về thư mục cấp người dùng chứa bản ghi các lượt chạy swarm."""
    return get_runtime_root() / "swarm" / "runs"


def get_uploads_dir() -> Path:
    """Trả về thư mục cấp người dùng chứa các tệp đã tải lên."""
    return get_runtime_root() / "uploads"


def get_config_candidates(config_path: Path | None = None) -> list[Path]:
    """Trả về danh sách ứng viên đường dẫn cấu hình được hỗ trợ theo thứ tự tra cứu.

    Returns:
        Danh sách ứng viên đường dẫn cấu hình được sắp xếp theo mức độ ưu tiên. Khi một đường dẫn
        rõ ràng được cung cấp, chỉ đường dẫn đó được trả về.
    """
    if config_path is not None:
        return [config_path.expanduser()]
    root = get_runtime_root()
    return [root / filename for filename in _DEFAULT_FILENAMES]


def get_config_path(config_path: Path | None = None) -> Path:
    """Trả về đường dẫn tệp cấu hình đang hoạt động.

    Ưu tiên ứng viên đầu tiên đang tồn tại. Nếu chỉ định đường dẫn rõ ràng,
    trả về trực tiếp đường dẫn đó. Nếu chưa ứng viên nào tồn tại, trả về
    đường dẫn JSON mặc định được khuyến nghị.

    Args:
        config_path: Đường dẫn tệp cấu hình rõ ràng tùy chọn.

    Returns:
        Đường dẫn tệp cấu hình được chọn cho ngữ cảnh runtime hiện tại.
    """
    candidates = get_config_candidates(config_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_data_dir(config_path: Path | None = None) -> Path:
    """Trả về và tạo thư mục dữ liệu runtime suy ra từ đường dẫn cấu hình.

    Args:
        config_path: Đường dẫn tệp cấu hình rõ ràng tùy chọn.

    Returns:
        Thư mục chứa tệp cấu hình đang hoạt động. Thư mục được tạo
        nếu chưa tồn tại.
    """
    data_dir = get_config_path(config_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

