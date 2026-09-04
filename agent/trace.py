"""TraceWriter: Ghi vết JSONL an toàn chống crash.

Mỗi dòng chứa một bản ghi JSON; mỗi lần ghi được xả bộ đệm (flush) và đồng bộ đĩa (fsync)
để bản ghi có thể sống sót sau một sự cố crash máy chủ đột ngột. Các trường dữ liệu lớn được ghi
ra các tệp phụ sidecar (ghi tạm -> fsync -> ``os.replace`` -> fsync thư mục) TRƯỚC KHI bản ghi
tham chiếu đến chúng được fsync, do đó một bản ghi bền vững không bao giờ trỏ tới một sidecar
bị thiếu hoặc bị cắt xén. Tệp sidecar giúp giữ vết đầy đủ mà không khiến mọi thao tác đọc CLI/lịch sử
trở thành một lần tải tệp khổng lồ.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Đọc biến môi trường kiểu số nguyên dương với giá trị mặc định an toàn.

    Args:
        name: Tên biến môi trường.
        default: Giá trị mặc định dự phòng.

    Returns:
        Số nguyên dương đã phân tích, hoặc ``default`` khi không đặt/không hợp lệ.
    """
    raw = os.getenv(name)  # noqa
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


TOOL_RESULT_OFFLOAD_THRESHOLD = _env_int(
    "KAIROS_TRACE_TOOL_RESULT_INLINE_LIMIT",
    50_000,
)
TRACE_TEXT_OFFLOAD_THRESHOLD = _env_int(
    "KAIROS_TRACE_TEXT_INLINE_LIMIT",
    50_000,
)
OFFLOAD_PREVIEW_CHARS = _env_int("KAIROS_TRACE_PREVIEW_CHARS", 500)

OPTIONAL_IRR_TRACE_FIELDS = frozenset({
    "artifact_refs",
    "data_audit_id",
    "policy_decision_id",
    "governance_overhead_ms",
    "warning_codes",
    "hard_failure_codes",
})


class TraceWriter:
    """Trình ghi vết JSONL, mỗi bản ghi một dòng, an toàn trước crash.

    Thuộc tính:
        dir_path: Thư mục chứa ``trace.jsonl`` và các blob sidecar.
        path: Đường dẫn tới tệp JSONL vết.
    """

    DEFAULT_FILENAME = "trace.jsonl"

    def __init__(self, dir_path: Path, filename: str | None = None) -> None:
        """Khởi tạo TraceWriter.

        Args:
            dir_path: Thư mục nơi các tệp trace được ghi.
            filename: Tên tệp JSONL. Mặc định ``trace.jsonl``. L2.6 dùng tham số này
                cho ``gate_ledger.jsonl`` (M-RS0 W2) để tái dùng nguyên kỷ luật fsync
                thay vì viết lại một bộ ghi thứ hai kém bền hơn.
        """
        self.dir_path = dir_path
        self.path = self.dir_path / (filename or self.DEFAULT_FILENAME)
        self._fsync_warned = False
        self.dir_path.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        self._file = open(self.path, "a", encoding="utf-8")
        try:
            if created:
                # Giúp mục thư mục trace.jsonl mới tạo ra tự nó trở nên bền vững.
                self._fsync_dir(self.dir_path)
        except Exception:
            self._file.close()
            raise

    def write(self, entry: Dict[str, Any]) -> None:
        """Ghi một bản ghi trace, xả bộ đệm và fsync để đảm bảo an toàn trước crash.

        Chỉ riêng ``flush`` chỉ chuyển các byte vào OS page cache; nếu hệ thống sập
        hoặc container bị kill thì các bản ghi cuối cùng sẽ bị mất. Dung lượng trace
        thường thấp (mỗi sự kiện LLM/tool một mục), do đó một lần fsync cho mỗi bản ghi là
        chi phí chấp nhận được để đảm bảo khả năng quan sát hậu sự cố. Nếu fsync thất bại
        (ví dụ hệ thống tệp không hỗ trợ), một cảnh báo sẽ được ghi nhật ký một lần và tiếp tục
        chỉ xả bộ đệm.

        Args:
            entry: Bản ghi trace; trường ``ts`` sẽ tự động được thêm nếu chưa có.
        """
        if "ts" not in entry:
            entry["ts"] = time.time()
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError as exc:
            self._warn_fsync_failure(exc, self.path)

    def write_text_entry(
        self,
        entry: Dict[str, Any],
        *,
        field: str,
        value: str,
        offload_kind: str,
        threshold: int | None = None,
    ) -> None:
        """Ghi một bản ghi chứa một trường văn bản có khả năng kích thước lớn.

        Args:
            entry: Bản ghi vết cơ sở. Sẽ bị thay đổi trước khi ghi.
            field: Tên trường văn bản, ví dụ ``"content"`` hoặc ``"prompt"``.
            value: Giá trị văn bản đầy đủ cần bảo tồn.
            offload_kind: Nhãn cố định dùng trong đầu vào hash tên tệp sidecar.
            threshold: Ngưỡng chèn inline đè tùy chọn.
        """
        self._attach_text_field(
            entry,
            field=field,
            value=value,
            offload_kind=offload_kind,
            threshold=threshold or TRACE_TEXT_OFFLOAD_THRESHOLD,
            offload_dir_name="trace-blobs",
        )
        self.write(entry)

    def write_tool_result(
        self,
        call_id: str,
        result: str,
        tool_name: str,
        status: str,
        elapsed_ms: int,
        iteration: int,
    ) -> None:
        """Ghi bản ghi tool_result, chuyển các kết quả lớn ra tệp phụ trên đĩa.

        Args:
            call_id: ID cuộc gọi tool của provider.
            result: Chuỗi kết quả thô của tool sau khi đã biên tập (redaction).
            tool_name: Tên công cụ.
            status: ``"ok"`` hoặc ``"error"``.
            elapsed_ms: Thời gian thực thi tính bằng miligiây.
            iteration: Số vòng lặp hiện tại.
        """
        entry: Dict[str, Any] = {
            "type": "tool_result",
            "iter": iteration,
            "tool": tool_name,
            "call_id": call_id,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "preview": result[:OFFLOAD_PREVIEW_CHARS],
        }
        self._attach_text_field(
            entry,
            field="result",
            value=result,
            offload_kind=f"tool-result-{tool_name}-{call_id}",
            threshold=TOOL_RESULT_OFFLOAD_THRESHOLD,
            offload_dir_name="tool-results",
            preview_field="result_preview",
        )
        self.write(entry)

    def close(self) -> None:
        """Đóng file handle."""
        self._file.close()

    def _attach_text_field(
        self,
        entry: Dict[str, Any],
        *,
        field: str,
        value: str,
        offload_kind: str,
        threshold: int,
        offload_dir_name: str,
        preview_field: str | None = None,
    ) -> None:
        """Gắn văn bản trực tiếp inline hoặc dưới dạng đường dẫn tệp sidecar.

        Tệp sidecar được ghi nguyên tử và bền vững tại đây, TRƯỚC KHI
        bên gọi ghi (và fsync) bản ghi tham chiếu đến nó, để một đợt crash
        không bao giờ để lại bản ghi bền vững trỏ vào sidecar bị thiếu hay bị cắt xén.
        """
        if len(value) <= threshold:
            entry[field] = value
            return

        offload_dir = self.dir_path / offload_dir_name
        if not offload_dir.is_dir():
            offload_dir.mkdir(parents=True, exist_ok=True)
            # Giúp mục thư mục sidecar mới trở nên bền vững.
            self._fsync_dir(self.dir_path)
        digest = hashlib.sha256(f"{offload_kind}\0{value}".encode("utf-8")).hexdigest()
        path = offload_dir / f"{digest[:24]}.txt"
        self._write_sidecar_durable(path, value)
        entry[f"{field}_path"] = f"{offload_dir_name}/{path.name}"
        entry[preview_field or f"{field}_preview"] = value[:OFFLOAD_PREVIEW_CHARS]
        entry[f"{field}_size"] = len(value)

    def _write_sidecar_durable(self, path: Path, value: str) -> None:
        """Ghi một blob sidecar theo cách nguyên tử và bền vững tối đa hệ thống tệp cho phép.

        Chuỗi ghi: tệp tạm trong cùng thư mục -> ghi đủ từng byte ->
        fsync -> ``os.replace`` -> fsync thư mục cha. Việc đổi tên giúp thao tác mang tính nguyên tử:
        một đợt crash sẽ chỉ để lại tệp sidecar hoàn chỉnh hoặc không có tệp nào,
        không bao giờ để lại tệp ghi dở dang dưới tên chính thức.
        """
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            payload = value.encode("utf-8")
            written = 0
            # os.write có thể ghi ít byte hơn yêu cầu; một cuộc gọi duy nhất có thể
            # bị cắt xén âm thầm với một sidecar lớn.
            while written < len(payload):
                written += os.write(fd, payload[written:])
            try:
                os.fsync(fd)
            except OSError as exc:
                self._warn_fsync_failure(exc, tmp)
        except BaseException:
            os.close(fd)
            # Không bao giờ để lại tệp tạm ghi dở dang cho lần chạy sau.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        else:
            os.close(fd)
        os.replace(tmp, path)
        self._fsync_dir(path.parent)

    def _fsync_dir(self, directory: Path) -> None:
        """Fsync một thư mục để các mục mới/được đổi tên sống sót sau sự cố crash.

        Args:
            directory: Thư mục có bảng mục từ cần được xả xuống đĩa.
        """
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError as exc:
            self._warn_fsync_failure(exc, directory)
        finally:
            os.close(dir_fd)

    def _warn_fsync_failure(self, exc: OSError, target: Path) -> None:
        """Ghi log cảnh báo thất bại fsync đầu tiên cho trình ghi này, sau đó giữ im lặng.

        Args:
            exc: Lỗi fsync.
            target: Tệp hoặc thư mục có fsync bị thất bại.
        """
        if self._fsync_warned:
            return
        self._fsync_warned = True
        logger.warning(
            "trace fsync failed on %s (%s); trace durability degraded to flush-only for this writer",
            target,
            exc,
        )

    @staticmethod
    def read(
        dir_path: Path,
        *,
        resolve_offloads: bool = False,
        resolve_fields: Optional[Iterable[str]] = None,
        filename: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Đọc các bản ghi trace.jsonl.

        Args:
            dir_path: Thư mục chứa ``trace.jsonl``.
            resolve_offloads: Liệu có đọc các tệp sidecar trở lại vào các bản ghi hay không.
            resolve_fields: Danh sách các trường cho phép đọc tùy chọn.
            filename: Tên tệp JSONL, mặc định ``trace.jsonl``. Phải khớp với tham số
                cùng tên đã truyền cho ``__init__``.

        Returns:
            Danh sách các bản ghi trace.
        """
        path = dir_path / (filename or TraceWriter.DEFAULT_FILENAME)
        if not path.exists():
            return []
        fields = set(resolve_fields) if resolve_fields is not None else None
        entries: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resolve_offloads:
                TraceWriter._resolve_entry_offloads(dir_path, entry, fields)
            entries.append(entry)
        return entries

    @staticmethod
    def _resolve_entry_offloads(
        dir_path: Path,
        entry: Dict[str, Any],
        fields: Optional[set[str]],
    ) -> None:
        """Phân giải các đường dẫn sidecar an toàn trở lại các trường ban đầu của chúng."""
        for key, rel_path in list(entry.items()):
            if not key.endswith("_path") or not isinstance(rel_path, str):
                continue
            field = key[:-5]
            if fields is not None and field not in fields:
                continue
            if field in entry:
                continue
            result_file = TraceWriter._safe_sidecar_path(dir_path, rel_path)
            if result_file is None or not result_file.exists():
                continue
            try:
                entry[field] = result_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

    @staticmethod
    def _safe_sidecar_path(dir_path: Path, rel_path: str) -> Path | None:
        """Trả về đường dẫn sidecar chỉ khi nó nằm hoàn toàn bên trong ``dir_path``."""
        root = dir_path.resolve()
        candidate = (dir_path / rel_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def find_trace_dir(
        run_id: str,
        runs_dir: Optional[Path] = None,
        sessions_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Tìm thư mục trace cho một run/session id.

        Args:
            run_id: ID lượt chạy hoặc phiên làm việc.
            runs_dir: Thư mục gốc chứa các lượt chạy.
            sessions_dir: Thư mục gốc chứa các phiên làm việc.

        Returns:
            Thư mục chứa ``trace.jsonl``, hoặc ``None`` nếu không tồn tại.
        """
        if sessions_dir is None:
            from pseud.config.paths import get_sessions_dir

            sessions_dir = get_sessions_dir()
        if runs_dir is None:
            from pseud.config.paths import get_runs_dir

            runs_dir = get_runs_dir()

        session_dir = sessions_dir / run_id
        if (session_dir / "trace.jsonl").exists():
            return session_dir

        run_dir = runs_dir / run_id
        if (run_dir / "trace.jsonl").exists():
            return run_dir

        return None

