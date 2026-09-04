"""Kênh phát tiến độ cho các công cụ chạy lâu.

Hai cơ chế:
  * Heartbeat ở cấp công cụ: một bộ hẹn giờ chạy ngầm phát ra các sự kiện keepalive mỗi N
    giây trong khi một công cụ đang chạy, giúp giao diện UI không bao giờ bị đóng băng. Được điều khiển bởi
    agent loop, ẩn hoàn toàn đối với công cụ.
  * Tiến độ có cấu trúc: các công cụ có thể tham gia qua ``emit_progress()`` để xuất bản
    trạng thái định lượng (stage, current/total, message). Được định tuyến trở lại
    kênh sự kiện của agent loop thông qua một emitter thread-local được thiết lập trước khi
    công cụ thực thi.

Mô hình luồng (Thread model): các công cụ chạy đồng bộ bên trong ``ToolRegistry.execute`` từ
luồng worker (các đợt đọc) hoặc luồng loop chính (các đợt ghi). Một
khe ``threading.local`` giữ emitter theo từng luồng để tiến độ có cấu trúc
chảy về đúng instance AgentLoop ngay cả khi nhiều công cụ chỉ đọc chạy song song.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEvent:
    """Sự kiện tiến độ có cấu trúc được phát bởi công cụ trong quá trình thực thi.

    Thuộc tính:
        tool: Tên công cụ (do loop điền, công cụ không cần cung cấp).
        stage: Nhãn giai đoạn ngắn, ví dụ ``loading_data`` hoặc ``simulating``.
        current: Số lượng đơn vị hiện tại (ví dụ trang 23 trên 100). Tùy chọn.
        total: Tổng số lượng đơn vị. Tùy chọn.
        message: Chi tiết nội dung tự do cho con người đọc.
        elapsed_s: Số giây kể từ khi công cụ bắt đầu.
        ts: Dấu thời gian (wall-clock timestamp).
    """

    tool: str = ""
    stage: str = ""
    current: Optional[int] = None
    total: Optional[int] = None
    message: str = ""
    elapsed_s: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Trả về dict có thể serial hóa sang JSON cho payload SSE."""
        return {
            "tool": self.tool,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "elapsed_s": round(self.elapsed_s, 2),
            "ts": self.ts,
        }


# Khe emitter thread-local. Agent loop đặt ``_local.emit`` thành một callable
# trước khi gọi công cụ và dọn dẹp sau đó, để các công cụ chạy trên cùng
# luồng có thể phát tiến độ có cấu trúc mà không bị lặp import (circular import).
_local = threading.local()


def _set_emitter(emit: Optional[Callable[[ProgressEvent], None]]) -> None:
    """Cài đặt bộ phát tiến độ (progress emitter) đang hoạt động cho luồng hiện tại.

    Args:
        emit: Callable tiếp nhận một ``ProgressEvent``. Truyền ``None`` để xóa.
    """
    if emit is None:
        if hasattr(_local, "emit"):
            del _local.emit
        return
    _local.emit = emit


def _get_emitter() -> Optional[Callable[[ProgressEvent], None]]:
    """Trả về emitter đang hoạt động trên luồng hiện tại, nếu có."""
    return getattr(_local, "emit", None)


def emit_progress(
    stage: str = "",
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: str = "",
) -> None:
    """Xuất bản sự kiện tiến độ có cấu trúc từ một công cụ.

    Bỏ qua âm thầm khi được gọi bên ngoài ngữ cảnh công cụ đang hoạt động (ví dụ trong
    các unit test gọi công cụ trực tiếp). Không bao giờ ném lỗi.

    Args:
        stage: Nhãn giai đoạn ngắn.
        current: Số lượng đơn vị hiện tại.
        total: Tổng số lượng đơn vị.
        message: Chi tiết nội dung cho con người đọc.
    """
    emit = _get_emitter()
    if emit is None:
        return
    try:
        event = ProgressEvent(
            stage=stage,
            current=current,
            total=total,
            message=message,
        )
        emit(event)
    except Exception:
        # Việc phát tiến độ không bao giờ được làm gián đoạn công cụ.
        pass


class HeartbeatTimer:
    """Luồng chạy ngầm phát ra các nhịp keepalive trong khi công cụ đang chạy.

    Sử dụng làm context manager xung quanh một lần gọi công cụ:

        with HeartbeatTimer(tool_name="run_backtest", interval=3.0, emit=fn):
            result = registry.execute(...)

    Bộ hẹn giờ thức dậy mỗi khoảng ``interval`` giây và gọi ``emit`` với một
    dict chứa ``tool`` và ``elapsed_s``. Dừng sạch sẽ khi thoát context;
    ``join`` được giới hạn thời gian để emitter bị treo không làm nghẽn loop.
    """

    def __init__(
        self,
        tool_name: str,
        interval: float,
        emit: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Khởi tạo bộ hẹn giờ heartbeat (chưa chạy cho đến khi gọi ``__enter__``).

        Args:
            tool_name: Tên công cụ hiển thị trong mỗi payload nhịp tim.
            interval: Số giây giữa các nhịp. Giá trị <0.5 sẽ được giới hạn tối thiểu 0.5s.
            emit: Callback được gọi với mỗi payload nhịp tim.
        """
        self._tool_name = tool_name
        requested_interval = float(interval)
        self._interval = max(0.5, requested_interval)
        if requested_interval < 0.5:
            logger.warning(
                "HeartbeatTimer interval %s clamped to 0.5s", interval
            )
        self._emit = emit
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def __enter__(self) -> "HeartbeatTimer":
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{self._tool_name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        """Vòng lặp nhịp tim: chờ + phát cho đến khi sự kiện dừng được kích hoạt."""
        while not self._stop_event.wait(self._interval):
            elapsed = time.perf_counter() - self._t0
            try:
                self._emit({"tool": self._tool_name, "elapsed_s": round(elapsed, 2)})
            except Exception:
                # Callback bị lỗi không được làm chết luồng nhịp tim.
                pass

