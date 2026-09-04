"""PersistentMemory: bộ nhớ liên phiên làm việc dựa trên tệp, không phụ thuộc bên ngoài."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from pseud.agent.frontmatter import parse_frontmatter as _parse_frontmatter
from pseud.config.paths import get_runtime_root

logger = logging.getLogger(__name__)

MEMORY_BASE = get_runtime_root() / "memory"
MAX_INDEX_LINES = 200
MAX_ENTRY_CHARS = 8000
MAX_RESULTS = 5
METADATA_WEIGHT = 2.0
# `research` là danh mục ký ức nghiên cứu của L2.6 (M-RS0 Phase 6.1). Bốn giá trị còn
# lại là taxonomy trợ lý đa dụng kế thừa; giữ để không vỡ ký ức đã ghi từ trước.
# Phải khớp với CATEGORIES trong hierarchy.py — lệch nhau thì mục ghi được nhưng
# không quét lại được, và ký ức biến mất trong im lặng.
MEMORY_TYPES = ("user", "feedback", "project", "reference", "research")

_LOCK_TIMEOUT_S = 5.0

# Cửa sổ trượt khử trùng lặp nội dung (tính bằng giây).
# Phát hiện các mục trùng lặp phát ra nhanh từ các vòng lặp thử lại hoặc các cuộc gọi agent song song.
DEDUP_WINDOW_SECONDS = 30.0


def content_hash(name: str, description: str, content: str = "") -> str:
    """Tạo mã hash xác định phục vụ khử trùng lặp."""
    payload = f"{name.strip().lower()}|{description.strip().lower()}|{content.strip().lower()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@contextmanager
def memory_lock(memory_dir: Path) -> Generator[bool, None, None]:
    """Yêu cầu khóa tệp độc quyền; trả về True nếu lấy được khóa, False khi hết thời gian chờ."""
    import filelock

    lock_path = memory_dir / ".lock"
    lock_path.touch(exist_ok=True)
    flock = filelock.FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S)
    try:
        flock.acquire(timeout=_LOCK_TIMEOUT_S)
        yield True
    except filelock.Timeout:
        logger.warning("memory_lock: timeout after %.1fs", _LOCK_TIMEOUT_S)
        yield False
    finally:
        if flock.is_locked:
            flock.release()



HALF_LIFE_DAYS = 14.0
_DECAY_LAMBDA = math.log(2) / HALF_LIFE_DAYS
_ACCESS_BOOST = 0.1


def compute_importance(
    quality_score: float, access_count: int, days_since_last_access: float
) -> float:
    """Tính toán tầm quan trọng thông qua công thức suy giảm lấy cảm hứng từ Ebbinghaus."""
    from pseud.config.accessor import get_env_config

    if not get_env_config().memory.decay_enabled:
        return quality_score
    retention = math.exp(-_DECAY_LAMBDA * max(0.0, days_since_last_access))
    access_bonus = min(0.3, access_count * _ACCESS_BOOST)
    raw = quality_score * (retention + access_bonus)
    return min(1.0, max(0.0, raw))


def _is_decay_enabled() -> bool:
    """Kiểm tra xem sự suy giảm mức độ quan trọng có được bật qua cấu hình hệ thống hay không."""
    from pseud.config.accessor import get_env_config

    return get_env_config().memory.decay_enabled


def _is_quality_enabled() -> bool:
    """Kiểm tra xem đánh giá chất lượng bộ nhớ có được bật qua cấu hình hệ thống hay không."""
    from pseud.config.accessor import get_env_config

    return get_env_config().memory.quality_enabled


# Phạm vi các bộ ký tự phi-Latinh phục vụ tách token và tạo slug.
_NON_LATIN_SCRIPT_RANGES = (
    "一-鿿"  # Chữ Hán CJK hợp nhất
    "㐀-䶿"  # CJK mở rộng A
    "฀-๿"  # Chữ Thái
    "ؠ-ي"  # Chữ Ả Rập
    "א-ת"  # Chữ Do Thái
    "Ѐ-ӿ"  # Chữ Cyrillic
)

_TOKEN_RE = re.compile(rf"[a-zA-Z0-9]{{3,}}|[{_NON_LATIN_SCRIPT_RANGES}]")
_SLUG_DISALLOWED_RE = re.compile(rf"[^a-z0-9_\-{_NON_LATIN_SCRIPT_RANGES}]")


@dataclass(frozen=True)
class MemoryEntry:
    """Một mục ký ức đơn lẻ lưu trữ trên đĩa."""

    path: Path
    title: str
    description: str
    memory_type: str
    body: str
    modified_at: float
    id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    keywords: tuple[str, ...] = ()
    quality_score: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    importance: float = 0.5
    related_memories: tuple[str, ...] = ()
    category: str = ""              # Phân loại thư mục H-MEM
    compression_level: str = "raw"  # Mức độ nén: raw/daily/digest


def _tokenize(text: str) -> set[str]:
    """Tách văn bản thành các token có thể tìm kiếm (ASCII >=3 ký tự + ký tự phi-Latinh)."""
    return set(_TOKEN_RE.findall(text.lower()))


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_TRUNCATION_MARKER = "\n\n[truncated at {limit} chars]\n"


def _sanitize_body(content: str) -> str:
    """Loại bỏ các byte điều khiển C0/C1 khỏi `content` nhưng giữ lại ``\\n`` và ``\\t``."""
    return _CONTROL_CHAR_RE.sub("", content)


def _truncate_body(content: str, limit: int = MAX_ENTRY_CHARS) -> str:
    """Cắt gọt `content` theo giới hạn `limit` ký tự, chừa chỗ cho nhãn cảnh báo."""
    if len(content) <= limit:
        return content
    marker = _TRUNCATION_MARKER.format(limit=limit)
    head_len = max(0, limit - len(marker))
    return content[:head_len] + marker


def _coerce_str(value: object, default: str = "") -> str:
    """Ép kiểu các giá trị frontmatter về chuỗi hiển thị."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _parse_timestamp(value: object, fallback: float) -> float:
    """Phân tích cú pháp mốc thời gian từ frontmatter. Trả về mốc thời gian dạng số thực epoch."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            pass
    return fallback


class PersistentMemory:
    """Bộ nhớ liên phiên dạng tệp, tồn tại qua các phiên làm việc."""

    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self._dir = memory_dir or MEMORY_BASE
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "MEMORY.md"
        self._snapshot: str = ""
        self._recent_hashes: dict[str, float] = {}  # hash -> mốc thời gian epoch
        self._load_snapshot()

    def _load_snapshot(self) -> None:
        """Nạp chỉ mục dạng ảnh chụp đóng băng. Được gọi một lần khi khởi tạo."""
        if self._index_path.exists():
            try:
                text = self._index_path.read_text(encoding="utf-8")
                lines = text.split("\n")[:MAX_INDEX_LINES]
                self._snapshot = "\n".join(lines)
            except OSError:
                self._snapshot = ""

    @property
    def snapshot(self) -> str:
        """Ảnh chụp chỉ mục ký ức đóng băng dùng cho việc tiêm vào prompt hệ thống."""
        return self._snapshot

    def _scan_entries(self) -> List[MemoryEntry]:
        """Quét tất cả các tệp .md (trừ MEMORY.md) và phân tích cú pháp frontmatter."""
        from pseud.config.accessor import get_env_config
        cfg = get_env_config().memory

        if cfg.hierarchy_enabled:
            from pseud.memory.hierarchy import MemoryHierarchy
            hierarchy = MemoryHierarchy(self._dir)
            md_files = hierarchy.scan_all()
        else:
            md_files = sorted(self._dir.glob("*.md"))

        entries: List[MemoryEntry] = []
        for path in md_files:
            if path.name == "MEMORY.md":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = _parse_frontmatter(text)
            mtime = path.stat().st_mtime
            raw_kw = meta.get("keywords", [])
            keywords = tuple(
                str(k)[:30] for k in (raw_kw if isinstance(raw_kw, list) else [])
            )[:5]
            raw_related = meta.get("related_memories", [])
            related = tuple(
                str(r)
                for r in (raw_related if isinstance(raw_related, list) else [])
                if isinstance(r, str)
                and len(r) == 6
                and all(c in "0123456789abcdef" for c in r.lower())
            )

            category = _coerce_str(meta.get("category"), default="")
            compression_level = _coerce_str(meta.get("compression_level"), default="raw")

            qs = meta.get("quality_score", 0.5)
            try:
                qs = max(0.0, min(1.0, float(qs)))
            except (TypeError, ValueError):
                qs = 0.5

            ac = meta.get("access_count", 0)
            try:
                ac = max(0, int(ac))
            except (TypeError, ValueError):
                ac = 0
            # Tạo ID nếu thiếu
            entry_id = str(meta.get("id", ""))
            if not entry_id or len(entry_id) != 6:
                entry_id = hashlib.sha256(
                    f"{meta.get('name', path.stem)}{mtime}".encode()
                ).hexdigest()[:6]

            # Mốc thời gian và độ quan trọng
            created = _parse_timestamp(meta.get("created_at"), mtime)
            updated = _parse_timestamp(meta.get("updated_at"), mtime)
            last_acc = _parse_timestamp(meta.get("last_accessed"), mtime)
            now = _time.time()
            days_since = max(0.0, (now - last_acc) / 86400.0)
            importance = compute_importance(qs, ac, days_since)

            entries.append(
                MemoryEntry(
                    path=path,
                    title=_coerce_str(meta.get("name"), default=path.stem),
                    description=_coerce_str(meta.get("description")),
                    memory_type=_coerce_str(meta.get("type"), default="project"),
                    body=body[:MAX_ENTRY_CHARS],
                    modified_at=mtime,
                    id=entry_id,
                    created_at=created,
                    updated_at=updated,
                    keywords=keywords,
                    quality_score=qs,
                    access_count=ac,
                    last_accessed=last_acc,
                    importance=importance,
                    related_memories=related,
                    category=category,
                    compression_level=compression_level,
                )
            )
        return entries

    def list_entries(self) -> List[MemoryEntry]:
        """Trả về toàn bộ các mục ký ức đã lưu trữ, sắp xếp theo tên tệp."""
        return self._scan_entries()

    def find(self, name: str) -> Optional[MemoryEntry]:
        """Tìm kiếm mục ký ức theo tên chính xác, sau đó theo tên tệp trên đĩa."""
        needle = name.strip()
        if not needle:
            return None
        entries = self._scan_entries()
        for entry in entries:
            if entry.title == needle:
                return entry
        for entry in entries:
            stem = entry.path.stem
            if stem == needle or stem.endswith(f"_{needle}"):
                return entry
        return None

    def remove_entry(self, entry: MemoryEntry) -> bool:
        """Xóa mục ký ức đã xác định mà không cần quét lại từ đầu."""
        from pseud.config.accessor import get_env_config

        with memory_lock(self._dir) as acquired:
            if not acquired:
                logger.warning("remove_entry(%s): lock timeout", entry.title)
            try:
                entry.path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove memory entry %s: %s", entry.path, exc)
                return False
            self._rebuild_index()

        if get_env_config().memory.fts_index_enabled:
            try:
                from pseud.memory.search_index import get_shared_index
                get_shared_index().remove_entry(entry.id)
            except Exception:
                logger.debug("FTS5 remove_entry failed", exc_info=True)

        if get_env_config().memory.links_enabled:
            try:
                from pseud.memory.semantic_links import SemanticLinker
                SemanticLinker(self._dir).remove_relations(entry.path)
            except Exception:
                logger.debug("Failed to remove relations for %s", entry.path, exc_info=True)

        return True

    def find_relevant(
        self, query: str, max_results: int = MAX_RESULTS
    ) -> List[MemoryEntry]:
        """Tìm kiếm theo từ khóa trên tất cả các mục, có trọng số theo tầm quan trọng."""
        from pseud.config.accessor import get_env_config
        cfg = get_env_config().memory

        if cfg.fts_index_enabled:
            try:
                from pseud.memory.search_index import get_shared_index
                index = get_shared_index()
                matches = index.search(query, max_results=max_results)

                # Tự động tái dựng chỉ mục ở lần tìm kiếm rỗng đầu tiên nếu có mục trên đĩa
                all_entries = None
                if not matches and not index._auto_rebuilt:
                    all_entries = self._scan_entries()
                    if all_entries:
                        entries_data = [
                            (e.id, e.title, e.description, " ".join(e.keywords), e.body)
                            for e in all_entries
                        ]
                        index.rebuild_all(entries_data)
                        index._auto_rebuilt = True
                        matches = index.search(query, max_results=max_results)

                if matches:
                    # Ánh xạ kết quả FTS về đầy đủ thông tin mục ký ức
                    if all_entries is None:
                        all_entries = self._scan_entries()
                    entry_map = {e.id: e for e in all_entries}
                    fts_results = [entry_map[m.entry_id] for m in matches if m.entry_id in entry_map]
                    if fts_results:
                        return fts_results[:max_results]
            except Exception:
                logger.debug("FTS5 search failed, falling back to scan", exc_info=True)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._scan_entries():
            meta_tokens = _tokenize(f"{entry.title} {entry.description}")
            body_tokens = _tokenize(entry.body)
            kw_tokens = _tokenize(" ".join(entry.keywords))
            token_score = (
                len(query_tokens & meta_tokens) * METADATA_WEIGHT
                + len(query_tokens & kw_tokens) * METADATA_WEIGHT
                + len(query_tokens & body_tokens)
            )
            if token_score > 0:
                final_score = token_score
                if _is_decay_enabled():
                    final_score *= 0.5 + 0.5 * entry.importance
                scored.append((final_score, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].modified_at))
        results = [entry for _, entry in scored[:max_results]]

        if cfg.links_enabled and results:
            try:
                from pseud.memory.semantic_links import SemanticLinker
                linker = SemanticLinker(self._dir)
                all_entries = self._scan_entries()
                linked_ids: set[str] = set()
                for r in results:
                    relations = linker.load_relations(r.path)
                    for target_file, _score in relations:
                        linked_ids.add(Path(target_file).stem)
                # Thêm các mục liên kết chưa có trong kết quả
                result_paths = {r.path for r in results}
                for entry in all_entries:
                    if entry.path.stem in linked_ids and entry.path not in result_paths:
                        results.append(entry)
                        if len(results) >= max_results:
                            break
            except Exception:
                logger.debug("semantic link expansion failed", exc_info=True)

        return results

    def is_duplicate(self, name: str, description: str, content: str = "") -> bool:
        """Kiểm tra xem ký ức có nội dung tương tự có vừa mới được ghi hay không.

        Sử dụng cửa sổ trượt 30 giây để bắt các mục trùng lặp phát ra liên tục
        từ các vòng lặp thử lại hoặc các cuộc gọi agent song song.
        """
        new_hash = content_hash(name, description, content)
        now = _time.time()
        self._cleanup_expired_hashes(now)
        if new_hash in self._recent_hashes:
            if now - self._recent_hashes[new_hash] < DEDUP_WINDOW_SECONDS:
                return True
        self._recent_hashes[new_hash] = now
        return False

    def _cleanup_expired_hashes(self, now: float) -> None:
        """Loại bỏ các mã hash đã hết hạn để giải phóng bộ nhớ."""
        threshold = now - DEDUP_WINDOW_SECONDS
        expired = [h for h, ts in self._recent_hashes.items() if ts < threshold]
        for h in expired:
            del self._recent_hashes[h]

    def add(
        self,
        name: str,
        content: str,
        memory_type: str = "project",
        description: str = "",
    ) -> Optional[Path]:
        """Lưu một mục ký ức mới và cập nhật chỉ mục."""
        if _is_quality_enabled() and self.is_duplicate(name, description, content):
            logger.debug(
                "Duplicate memory write blocked within %.0fs window: %s",
                DEDUP_WINDOW_SECONDS,
                name,
            )
            return None

        stripped_name = name.strip()
        if not stripped_name:
            raise ValueError("memory name must not be empty or whitespace-only")
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of: {', '.join(MEMORY_TYPES)}")

        slug = _SLUG_DISALLOWED_RE.sub("_", stripped_name.lower())[:60]
        if slug.strip("_") == "":
            digest = hashlib.sha256(stripped_name.encode("utf-8")).hexdigest()[:6]
            slug = f"{slug}_{digest}" if slug else digest

        from pseud.config.accessor import get_env_config
        if get_env_config().memory.hierarchy_enabled:
            from pseud.memory.hierarchy import MemoryHierarchy
            hierarchy = MemoryHierarchy(self._dir)
            path = hierarchy.route_entry(memory_type, slug)
        else:
            filename = f"{memory_type}_{slug}.md"
            path = self._dir / filename
        safe_name = stripped_name.replace("\n", " ").replace("\r", " ")
        safe_desc = (description or stripped_name).replace("\n", " ").replace("\r", " ")
        clean_content = _truncate_body(_sanitize_body(content))

        entry_id = hashlib.sha256(
            f"{stripped_name}{_time.time()}".encode()
        ).hexdigest()[:6]
        now_iso = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())

        frontmatter = (
            f"---\nname: {safe_name}\n"
            f"description: {safe_desc}\n"
            f"type: {memory_type}\n"
            f"id: {entry_id}\n"
            f"created_at: {now_iso}\n"
            f"updated_at: {now_iso}\n"
            f"keywords: []\n"
            f"quality_score: 0.5\n"
            f"access_count: 0\n"
            f"last_accessed: {now_iso}\n"
            f"importance: 0.5\n"
            f"related_memories: []\n"
            f"category: {memory_type}\n"
            f"compression_level: raw\n"
            f"---\n\n"
            f"{clean_content}"
        )
        with memory_lock(self._dir) as acquired:
            if not acquired:
                logger.warning(
                    "add(%s): lock timeout, best-effort write", stripped_name
                )
            path.write_text(frontmatter, encoding="utf-8")
            self._update_index(stripped_name, path.name, description or stripped_name)

            if get_env_config().memory.links_enabled:
                try:
                    from pseud.memory.semantic_links import SemanticLinker, _tokenize_for_bm25
                    linker = SemanticLinker(self._dir)
                    all_entries = self._scan_entries()
                    new_entry = next((e for e in all_entries if e.path == path), None)
                    if new_entry:
                        entry_tokens = _tokenize_for_bm25(
                            f"{new_entry.title} {new_entry.description} {new_entry.body}"
                        )
                        all_entries_data = [
                            (e.path.name, _tokenize_for_bm25(
                                f"{e.title} {e.description} {e.body}"
                            ))
                            for e in all_entries if e.path != path
                        ]
                        links = linker.discover_links(
                            entry_title=new_entry.path.name,
                            entry_tokens=entry_tokens,
                            all_entries_data=all_entries_data,
                        )
                        if links:
                            linker.save_relations(path, links)
                except Exception:
                    logger.debug("semantic link discovery failed", exc_info=True)

            if get_env_config().memory.fts_index_enabled:
                try:
                    from pseud.memory.search_index import get_shared_index
                    index = get_shared_index()
                    index.index_entry(
                        entry_id=entry_id,
                        title=safe_name,
                        description=safe_desc,
                        keywords="",
                        body=clean_content,
                    )
                except Exception:
                    logger.debug("FTS5 index_entry failed", exc_info=True)
        return path

    def remove(self, name: str) -> bool:
        """Xóa mục ký ức theo tên. Trả về True nếu tìm thấy và đã xóa."""
        from pseud.config.accessor import get_env_config

        for entry in self._scan_entries():
            if entry.title == name:
                with memory_lock(self._dir) as acquired:
                    if not acquired:
                        logger.warning("remove(%s): lock timeout", name)
                    entry.path.unlink(missing_ok=True)
                    self._rebuild_index()

                if get_env_config().memory.fts_index_enabled:
                    try:
                        from pseud.memory.search_index import get_shared_index
                        get_shared_index().remove_entry(entry.id)
                    except Exception:
                        logger.debug("FTS5 remove_entry failed", exc_info=True)

                if get_env_config().memory.links_enabled:
                    try:
                        from pseud.memory.semantic_links import SemanticLinker
                        SemanticLinker(self._dir).remove_relations(entry.path)
                    except Exception:
                        logger.debug("Failed to remove relations for %s", entry.path, exc_info=True)

                return True
        return False

    def _update_index(self, title: str, filename: str, description: str) -> None:
        """Thêm mới hoặc cập nhật một mục trong MEMORY.md."""
        new_line = f"- [{title}]({filename}) — {description}"

        if self._index_path.exists():
            lines = self._index_path.read_text(encoding="utf-8").split("\n")
            updated = False
            for i, line in enumerate(lines):
                if f"[{title}]" in line:
                    lines[i] = new_line
                    updated = True
                    break
            if not updated:
                lines.append(new_line)
            text = "\n".join(lines[:MAX_INDEX_LINES])
        else:
            text = new_line

        self._index_path.write_text(text, encoding="utf-8")

    def _rebuild_index(self) -> None:
        """Tái dựng lại MEMORY.md từ tất cả các tệp mục ký ức hiện có."""
        entries = self._scan_entries()
        lines = [f"- [{e.title}]({e.path.name}) — {e.description}" for e in entries]
        self._index_path.write_text(
            "\n".join(lines[:MAX_INDEX_LINES]), encoding="utf-8"
        )
