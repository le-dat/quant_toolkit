"""Accessor lazy singleton cho EnvConfig.

Sử dụng :func:`get_env_config` để lấy instance cấu hình được lưu trong cache. Lần gọi đầu tiên
tạo ra một :class:`~pseud.src.config.env_schema.EnvConfig` (đọc từ
``os.environ``) và lưu vào cache; các lần gọi tiếp theo sẽ trả về cùng một đối tượng.

Gọi :func:`reset_env_config` sau khi thay đổi ``os.environ`` lúc runtime
(ví dụ trong Settings API) để cuộc gọi :func:`get_env_config` tiếp theo nhận được
các giá trị mới.

An toàn đa luồng (Thread-safety) được đảm bảo bởi một :class:`threading.Lock` cấp mô-đun —
các swarm worker và agent loop đều chạy trong các luồng và có thể gọi
:func:`get_env_config` đồng thời.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pseud.config.env_schema import EnvConfig

__all__ = ["get_env_config", "reset_env_config", "_parse_bool"]

# ---------------------------------------------------------------------------
# Trạng thái singleton cấp mô-đun
# ---------------------------------------------------------------------------

_instance: EnvConfig | None = None
_lock = threading.RLock()

# ---------------------------------------------------------------------------
# Tập hợp chuỗi đúng/sai cho bộ phân tích kiểu boolean thống nhất
# ---------------------------------------------------------------------------

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_env_config() -> EnvConfig:
    """Trả về singleton :class:`EnvConfig` được lưu trong cache, khởi tạo khi truy cập lần đầu.

    An toàn luồng: yêu cầu khóa trước khi kiểm tra / khởi tạo instance để
    các caller đồng thời (swarm worker, agent loop) không nhìn thấy một cấu hình
    đang khởi tạo dở dang hoặc tạo các instance trùng lặp.

    Returns:
        Instance :class:`EnvConfig` dùng chung.
    """
    global _instance  # noqa: PLW0603
    if _instance is not None:
        return _instance
    with _lock:
        # Kiểm tra hai lần (Double-checked locking): một luồng khác có thể đã tạo
        # instance trong khi chúng ta đang chờ khóa.
        if _instance is not None:
            return _instance

        try:
            from pseud.providers.llm import _sync_provider_env
            _sync_provider_env()
        except Exception:
            pass

        # Import tại đây để tránh import vòng lặp lúc nạp mô-đun và
        # giữ cho Pydantic model nặng nằm ngoài đường dẫn import của
        # các caller chỉ cần ``_parse_bool`` hoặc ``get_env_or``.
        from pseud.config.env_schema import EnvConfig as _EnvConfig

        _instance = _EnvConfig()
    return _instance


def reset_env_config() -> None:
    """Xóa singleton :class:`EnvConfig` trong cache.

    Sử dụng cùng một khóa như :func:`get_env_config` để thao tác reset mang tính
    nguyên tử đối với các luồng đọc đồng thời. Sau cuộc gọi này, lần gọi :func:`get_env_config`
    tiếp theo sẽ tạo một instance mới phản ánh các thay đổi trong ``os.environ``.

    Caller điển hình: ``settings_routes.py`` sau khi ghi các giá trị mới vào
    ``agent/.env`` và cập nhật ``os.environ``.
    """
    global _instance  # noqa: PLW0603
    with _lock:
        _instance = None


def _parse_bool(value: str | None) -> bool:
    """Phân tích một chuỗi (hoặc ``None``) thành giá trị luận lý (boolean).

    Các giá trị True (không phân biệt chữ hoa/thường): ``"1"``, ``"true"``, ``"yes"``, ``"on"``.
    Mọi giá trị khác — bao gồm ``None``, ``""``, ``"0"``, ``"false"``,
    ``"no"``, ``"off"`` — trả về ``False``.

    Đây là nguồn sự thật duy nhất cho việc phân tích biến môi trường boolean trong toàn bộ
    mã nguồn, thay thế cho các đoạn code lặp lại.

    Args:
        value: Chuỗi thô (thường trích từ ``os.getenv``) hoặc ``None``.

    Returns:
        ``True`` khi *value* là một chuỗi mang ý nghĩa đúng; ``False`` trong các trường hợp còn lại.
    """
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY

