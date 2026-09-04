"""Quản lý vòng đời ký ức: đánh giá chất lượng, suy giảm độ quan trọng, và dọn dẹp bộ nhớ (GC).

Cung cấp các cập nhật chất lượng theo phong cách học tăng cường, công thức suy giảm độ quan trọng
lấy cảm hứng từ đường cong Ebbinghaus, và dọn dẹp rác dựa trên dung lượng. Tất cả các thao tác ghi
đều được bảo vệ bởi khóa mức tệp (mô hình một người ghi).

Cờ tính năng (biến môi trường / cấu hình):
    VT_MEMORY_QUALITY  – bật đánh giá chất lượng / theo dõi truy cập
    VT_MEMORY_GC       – bật dọn dẹp rác bộ nhớ (GC)
    VT_MEMORY_DECAY    – bật công thức suy giảm độ quan trọng
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import MappingProxyType

from pseud.memory.persistent import (
    MemoryEntry,
    PersistentMemory,
    compute_importance,
    memory_lock,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cờ tính năng
# ---------------------------------------------------------------------------


def is_quality_enabled() -> bool:
    """Kiểm tra xem tính năng đánh giá chất lượng có được bật qua cấu hình hay không."""
    from pseud.config.accessor import get_env_config

    return get_env_config().memory.quality_enabled


def is_gc_enabled() -> bool:
    """Kiểm tra xem tính năng dọn dẹp rác (GC) có được bật qua cấu hình hay không."""
    from pseud.config.accessor import get_env_config

    return get_env_config().memory.gc_enabled


def is_decay_enabled() -> bool:
    """Kiểm tra xem tính năng suy giảm độ quan trọng có được bật qua cấu hình hay không."""
    from pseud.config.accessor import get_env_config

    return get_env_config().memory.decay_enabled


# ---------------------------------------------------------------------------
# Trợ lý
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Thời gian hiện tại ở dạng chuỗi ISO-8601."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


# ---------------------------------------------------------------------------
# Class MemoryLifecycle
# ---------------------------------------------------------------------------


class MemoryLifecycle:
    """Quản lý vòng đời cho bộ nhớ bền vững: chấm điểm chất lượng, suy giảm, GC.

    Bọc một instance PersistentMemory và cung cấp cơ chế củng cố, dọn dẹp rác,
    và theo dõi truy cập. Tất cả các thao tác ghi đều được bảo vệ bởi khóa cấp tệp.
    """

    _EVENT_DELTAS: MappingProxyType[str, float] = MappingProxyType(
        {
            "task_success": 0.1,
            "task_failure": -0.15,
            "user_confirm": 0.2,
            "user_reject": -0.3,
            "passive_decay": -0.05,
        }
    )

    # Giới hạn an toàn: mức điều chỉnh tối đa mỗi phiên cho từng ký ức
    _MAX_SESSION_DELTA = 0.5

    # Ngưỡng GC
    ARCHIVE_THRESHOLD = 0.15
    DELETE_THRESHOLD = 0.05
    MIN_AGE_DAYS = 7
    MAX_MEMORY_COUNT = 500
    ENABLE_DELETE = False  # Tier 1: chỉ lưu trữ

    def __init__(self, memory: PersistentMemory) -> None:
        self._memory = memory
        self._session_deltas: dict[str, float] = {}  # name -> delta lũy kế

    @property
    def memory_dir(self) -> Path:
        """Trả về thư mục bộ nhớ bên dưới."""
        return self._memory._dir

    # ------------------------------------------------------------------
    # Củng cố
    # ------------------------------------------------------------------

    def reinforce(self, name: str, event: str, source: str = "system") -> bool:
        """Cập nhật điểm chất lượng dựa trên phản hồi sử dụng.

        Tham số:
            name: Tên mục ký ức (khớp chính xác).
            event: Một trong các giá trị "task_success", "task_failure", "user_confirm",
                   "user_reject", "passive_decay".
            source: "user" (độ tin cậy tối đa) hoặc "system" (chiết khấu 0.7x).

        Trả về:
            True nếu củng cố thành công, False nếu bị bỏ qua.
        """
        if not is_quality_enabled():
            return False
        if event not in self._EVENT_DELTAS:
            logger.warning("reinforce: unknown event %r", event)
            return False

        delta = self._EVENT_DELTAS[event]
        if source == "system":
            delta *= 0.7

        # Kiểm tra giới hạn phiên
        current = self._session_deltas.get(name, 0.0)
        if abs(current + delta) > self._MAX_SESSION_DELTA:
            logger.info("reinforce(%s): session cap reached (%.2f)", name, current)
            return False

        entry = self._memory.find(name)
        if entry is None:
            logger.warning("reinforce(%s): not found", name)
            return False

        with memory_lock(self.memory_dir) as acquired:
            if not acquired:
                return False
            try:
                new_qs = max(0.0, min(1.0, entry.quality_score + delta))
                self._update_frontmatter_field(
                    entry.path, "quality_score", f"{new_qs:.2f}"
                )
                self._update_frontmatter_field(entry.path, "updated_at", _now_iso())
                self._session_deltas[name] = current + delta
                return True
            except (FileNotFoundError, IOError) as exc:
                logger.warning("reinforce(%s) skipped: %s", name, exc)
                return False

    # ------------------------------------------------------------------
    # Theo dõi truy cập
    # ------------------------------------------------------------------

    def track_access(self, entry: MemoryEntry) -> None:
        """Tăng access_count và cập nhật last_accessed cho một mục ký ức được truy xuất."""
        if not is_quality_enabled():
            return
        with memory_lock(self.memory_dir) as acquired:
            if not acquired:
                return
            try:
                self._update_frontmatter_field(
                    entry.path, "access_count", str(entry.access_count + 1)
                )
                self._update_frontmatter_field(entry.path, "last_accessed", _now_iso())
            except (FileNotFoundError, IOError) as exc:
                logger.warning("track_access(%s) skipped: %s", entry.title, exc)

    # ------------------------------------------------------------------
    # Dọn dẹp rác (GC)
    # ------------------------------------------------------------------

    def run_gc(self, dry_run: bool = True) -> list[dict]:
        """Chạy dọn dẹp rác trên kho bộ nhớ.

        Tham số:
            dry_run: Nếu là True (mặc định), chỉ ghi log các hành động mà không sửa đổi tệp.

        Trả về:
            Danh sách các bản ghi hành động [{name, action, importance, reason}].
        """
        if not is_gc_enabled():
            # Cảnh báo nếu bật nén nhưng tắt GC
            from pseud.config.accessor import get_env_config

            cfg = get_env_config().memory
            if cfg.compression_enabled:
                logger.warning(
                    "VT_MEMORY_COMPRESSION is enabled but VT_MEMORY_GC is disabled; "
                    "compression will not trigger. Enable GC or set VT_MEMORY=on/full."
                )
            return []

        entries = self._memory.list_entries()

        now = time.time()
        actions: list[dict] = []

        for entry in entries:
            age_days = (now - entry.created_at) / 86400.0
            if age_days < self.MIN_AGE_DAYS:
                continue

            days_since_access = (now - entry.last_accessed) / 86400.0
            imp = compute_importance(
                entry.quality_score, entry.access_count, days_since_access
            )

            action = None
            reason = ""
            if imp < self.DELETE_THRESHOLD and self.ENABLE_DELETE:
                action = "delete"
                reason = f"importance {imp:.3f} < delete threshold"
            elif imp < self.ARCHIVE_THRESHOLD:
                action = "archive"
                reason = f"importance {imp:.3f} < archive threshold"

            if action:
                record = {
                    "name": entry.title,
                    "action": action,
                    "importance": round(imp, 4),
                    "reason": reason,
                }
                actions.append(record)
                if not dry_run:
                    # Tier 1: bắt buộc chuyển thành archive dù bị phân loại là delete
                    effective = "archive" if not self.ENABLE_DELETE else action
                    self._execute_gc_action(entry, effective)

        self._append_gc_log(actions, dry_run)

        # Tier 2: Kích hoạt quy trình nén cho các mục có tuổi thọ cao
        from pseud.config.accessor import get_env_config
        if get_env_config().memory.compression_enabled:
            try:
                from pseud.memory.compression import CompressionPipeline
                pipeline = CompressionPipeline(self._memory._dir)
                now_ts = time.time()
                for entry in entries:
                    target = pipeline.should_compress(
                        compression_level=entry.compression_level,
                        last_accessed=entry.last_accessed,
                        now=now_ts,
                    )
                    if not target:
                        continue
                    compressed = pipeline.apply_compression(
                        entry_path=entry.path,
                        content=entry.body,
                        keywords=entry.keywords,
                        target_level=target,
                    )
                    if compressed is None:
                        continue
                    # Ghi lại: cập nhật compression_level trong frontmatter + thay thế body
                    self._write_compressed(entry, compressed, target)
            except Exception:
                logger.debug("compression cycle failed", exc_info=True)

        return actions

    def _execute_gc_action(self, entry: MemoryEntry, action: str) -> None:
        """Thực thi một hành động GC (lưu trữ hoặc xóa) trên một mục ký ức."""
        archive_dir = self.memory_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        with memory_lock(self.memory_dir) as acquired:
            if not acquired:
                return
            try:
                if action == "archive":
                    dest = archive_dir / entry.path.name
                    entry.path.rename(dest)
                elif action == "delete":
                    dest = archive_dir / entry.path.name
                    dest.write_text(
                        entry.path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    entry.path.unlink()
            except (OSError, IOError) as exc:
                logger.warning("GC action(%s, %s) failed: %s", entry.title, action, exc)
                return

        # Tái dựng chỉ mục sau khi xóa/chuyển
        self._memory._rebuild_index()

    def _append_gc_log(self, actions: list[dict], dry_run: bool) -> None:
        """Ghi thêm quyết định GC vào gc.log."""
        log_path = self.memory_dir / "gc.log"
        timestamp = _now_iso()
        mode = "dry_run" if dry_run else "execute"
        lines = [f"[{timestamp}] mode={mode} actions={len(actions)}"]
        for a in actions:
            lines.append(
                f"  {a['action']}: {a['name']} "
                f"(importance={a['importance']}, {a['reason']})"
            )
        lines.append("")

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError as exc:
            logger.warning("append_gc_log failed: %s", exc)

    # ------------------------------------------------------------------
    # Thao tác trên Frontmatter
    # ------------------------------------------------------------------

    def _write_compressed(
        self, entry: MemoryEntry, compressed_body: str, target_level: str
    ) -> None:
        """Ghi nội dung đã nén trở lại tệp ký ức một cách nguyên tử.

        Cập nhật compression_level và updated_at trong frontmatter, thay thế phần body.
        """
        with memory_lock(self.memory_dir) as acquired:
            if not acquired:
                logger.warning(
                    "_write_compressed(%s): lock timeout", entry.title
                )
                return
            try:
                text = entry.path.read_text(encoding="utf-8")
                lines = text.split("\n")
                if not lines or lines[0].strip() != "---":
                    return
                end_idx = None
                for i in range(1, len(lines)):
                    if lines[i].strip() == "---":
                        end_idx = i
                        break
                if end_idx is None:
                    return

                # Cập nhật compression_level trong frontmatter
                level_found = False
                updated_found = False
                now_iso = _now_iso()
                for i in range(1, end_idx):
                    if lines[i].startswith("compression_level:"):
                        lines[i] = f"compression_level: {target_level}"
                        level_found = True
                    elif lines[i].startswith("updated_at:"):
                        lines[i] = f"updated_at: {now_iso}"
                        updated_found = True
                if not level_found:
                    lines.insert(end_idx, f"compression_level: {target_level}")
                    end_idx += 1
                if not updated_found:
                    lines.insert(end_idx, f"updated_at: {now_iso}")
                    end_idx += 1

                # Thay thế body (mọi thứ sau --- đóng)
                new_lines = lines[: end_idx + 1]
                new_lines.append("")
                new_lines.append(compressed_body)

                tmp_path = entry.path.with_suffix(entry.path.suffix + ".tmp")
                tmp_path.write_text("\n".join(new_lines), encoding="utf-8")
                os.replace(tmp_path, entry.path)
            except (FileNotFoundError, IOError) as exc:
                logger.warning(
                    "_write_compressed(%s) failed: %s", entry.title, exc
                )

    def _update_frontmatter_field(self, path: Path, field: str, value: str) -> None:
        """Cập nhật một trường frontmatter đơn lẻ trong tệp ký ức.

        Sử dụng chiến lược ghi vào tệp tạm rồi đổi tên nguyên tử để tránh làm hỏng tệp
        nếu tiến trình gặp sự cố giữa chừng.
        """
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")

        if not lines or lines[0].strip() != "---":
            logger.warning(
                "_update_frontmatter_field(%s): no frontmatter delimiters in %s",
                field,
                path,
            )
            return
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is None:
            logger.warning(
                "_update_frontmatter_field(%s): no closing delimiter in %s",
                field,
                path,
            )
            return

        field_found = False
        for i in range(1, end_idx):
            if lines[i].startswith(f"{field}:"):
                lines[i] = f"{field}: {value}"
                field_found = True
                break
        if not field_found:
            lines.insert(end_idx, f"{field}: {value}")

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp_path, path)
