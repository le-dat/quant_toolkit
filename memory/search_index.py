"""Chỉ mục tìm kiếm toàn văn SQLite FTS5 cho ký ức bền vững (Tier 2).

Cung cấp khả năng tìm kiếm O(log n) thông qua inverted index, thay thế cho việc
quét tuần tự O(n) trong PersistentMemory.find_relevant().
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pseud.config.paths import get_runtime_root

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = get_runtime_root() / "memory_index.db"

# Giới hạn độ dài nội dung văn bản (ký tự)
_MAX_BODY_LEN = 50_000

# Phạm vi Unicode CJK
_CJK_RANGE_MAIN = (0x4E00, 0x9FFF)
_CJK_RANGE_EXT_A = (0x3400, 0x4DBF)


def _is_cjk_char(char: str) -> bool:
    """Kiểm tra xem ký tự có phải là chữ CJK hay không."""
    cp = ord(char)
    return (
        _CJK_RANGE_MAIN[0] <= cp <= _CJK_RANGE_MAIN[1]
        or _CJK_RANGE_EXT_A[0] <= cp <= _CJK_RANGE_EXT_A[1]
    )


def _expand_cjk_buffer(chars: list[str]) -> str:
    """Mở rộng chuỗi ký tự CJK thành unigram + bigram."""
    parts: list[str] = []
    # Unigrams
    parts.extend(chars)
    # Bigrams (cặp đè nhau)
    for i in range(len(chars) - 1):
        parts.append(chars[i] + chars[i + 1])
    return " ".join(parts)


def _cjk_query_tokens(chars: list[str]) -> list[str]:
    """Tạo token unigram + bigram từ các ký tự CJK liên tiếp cho truy vấn."""
    tokens: list[str] = list(chars)
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    return tokens


def _dedupe_cjk_runs(text: str) -> str:
    """Loại bỏ các chuỗi CJK bị trùng lặp sinh ra bởi mở rộng bigram."""
    _cjk_run_re = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]{2,}")

    def _shrink(match: re.Match) -> str:  # type: ignore[type-arg]
        run = match.group(0)
        total = len(run)
        n = (total + 2) // 3
        if 3 * n - 2 == total and n >= 2:
            candidate = run[:n]
            expected = candidate + "".join(
                candidate[i] + candidate[i + 1] for i in range(n - 1)
            )
            if expected == run:
                return candidate
        return run

    return _cjk_run_re.sub(_shrink, text)


@dataclass(frozen=True)
class MemoryMatch:
    """Một kết quả tìm kiếm FTS5 đơn lẻ.

    Thuộc tính:
        entry_id: Định danh hex 6 ký tự của mục ký ức.
        title: Tiêu đề ký ức.
        snippet: Đoạn trích FTS5 có làm nổi bật kết quả khớp (>>> <<<).
        rank: Mức độ liên quan FTS5 (giá trị nhỏ hơn là tốt hơn).
    """

    entry_id: str
    title: str
    snippet: str
    rank: float


class MemorySearchIndex:
    """Chỉ mục SQLite FTS5 phục vụ tìm kiếm toàn văn ký ức.

    Hỗ trợ:
        - Đánh chỉ mục từng mục ký ức khi được tạo hoặc cập nhật.
        - Tìm kiếm toàn văn với xếp hạng độ liên quan.
        - Tái dựng chỉ mục hàng loạt từ dữ liệu mục ký ức trong bộ nhớ.
        - Suy giảm mượt mà (graceful degradation) khi FTS5 không khả dụng.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Khởi tạo chỉ mục tìm kiếm.

        Tham số:
            db_path: Đường dẫn tới CSDL SQLite (mặc định: <gốc runtime>/memory_index.db).
        """
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._fts_available: bool = True
        self._auto_rebuilt: bool = False
        self._op_lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Lấy hoặc tạo kết nối SQLite ở chế độ WAL với tối ưu hóa mmap và bộ đệm bộ nhớ."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA mmap_size=268435456")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("PRAGMA cache_size=-64000")
        return self._conn

    def _init_db(self) -> None:
        """Tạo bảng dữ liệu + bảng ảo FTS5 + các trigger tự động đồng bộ."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT ''
            );
        """)

        # Bảng ảo FTS5 — try/except riêng biệt để suy giảm mượt mà
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(
                    title, description, keywords, body,
                    content=memories, content_rowid=rowid
                )
            """)
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 unavailable, search disabled: %s", exc)
            self._fts_available = False
            conn.commit()
            return

        # Trigger tự động đồng bộ
        for trigger_sql in [
            """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, title, description, keywords, body)
                VALUES (new.rowid, new.title, new.description, new.keywords, new.body);
            END""",
            """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, title, description, keywords, body)
                VALUES ('delete', old.rowid, old.title, old.description, old.keywords, old.body);
            END""",
            """CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, title, description, keywords, body)
                VALUES ('delete', old.rowid, old.title, old.description, old.keywords, old.body);
                INSERT INTO memories_fts(rowid, title, description, keywords, body)
                VALUES (new.rowid, new.title, new.description, new.keywords, new.body);
            END""",
        ]:
            try:
                conn.execute(trigger_sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def _is_index_populated(self) -> bool:
        """Kiểm tra xem chỉ mục FTS5 có chứa dữ liệu nào hay không."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0] > 0 if row else False
        except Exception:
            return False

    def index_entry(
        self,
        entry_id: str,
        title: str,
        description: str,
        keywords: str,
        body: str,
    ) -> None:
        """Thêm hoặc cập nhật một mục ký ức vào chỉ mục.

        Tham số:
            entry_id: Định danh hex 6 ký tự.
            title: Tiêu đề ký ức.
            description: Mô tả ngắn gọn một dòng.
            keywords: Các từ khóa nối bằng khoảng trắng.
            body: Toàn bộ nội dung văn bản (cắt gọt tối đa 50k ký tự).
        """
        with self._op_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO memories (id, title, description, keywords, body) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        entry_id,
                        self._prepare_cjk(title),
                        self._prepare_cjk(description),
                        self._prepare_cjk(keywords),
                        self._prepare_cjk(body[:_MAX_BODY_LEN]),
                    ),
                )
                conn.commit()
            except sqlite3.OperationalError as exc:
                logger.debug("index_entry failed for %s: %s", entry_id, exc)

    def remove_entry(self, entry_id: str) -> None:
        """Xóa một mục khỏi chỉ mục theo ID."""
        with self._op_lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
                conn.commit()
            except sqlite3.OperationalError as exc:
                logger.debug("remove_entry failed for %s: %s", entry_id, exc)

    def search(self, query: str, max_results: int = 5) -> List[MemoryMatch]:
        """Tìm kiếm toàn văn bằng câu lệnh MATCH của FTS5.

        Tham số:
            query: Chuỗi truy vấn tìm kiếm của người dùng.
            max_results: Số lượng kết quả tối đa trả về.

        Trả về:
            Danh sách MemoryMatch đã sắp xếp theo độ liên quan. Trả về danh sách rỗng nếu
            FTS5 không khả dụng hoặc truy vấn không đem lại kết quả.
        """
        if not self._fts_available:
            return []

        fts_query = self._sanitize_fts_query(query)
        if not fts_query or fts_query == '""':
            return []

        with self._op_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """
                    SELECT
                        m.id,
                        m.title,
                        snippet(memories_fts, 3, '>>>', '<<<', '...', 64) AS snippet,
                        rank
                    FROM memories_fts
                    JOIN memories m ON m.rowid = memories_fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, max_results),
                )
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 search failed: %s", exc)
                return []

            results: List[MemoryMatch] = []
            for row in cursor.fetchall():
                results.append(
                    MemoryMatch(
                        entry_id=row[0],
                        title=self._clean_cjk(row[1] or ""),
                        snippet=self._clean_cjk(row[2] or ""),
                        rank=row[3],
                    )
                )
            return results

    def rebuild_all(self, entries_data: List[tuple]) -> int:
        """Tái dựng lại toàn bộ chỉ mục từ danh sách các tuple (id, title, description, keywords, body).

        Xóa dữ liệu hiện tại và chèn lại toàn bộ các mục. Sử dụng cho việc đồng bộ
        hàng loạt từ kho bộ nhớ chuẩn PersistentMemory.

        Tham số:
            entries_data: Danh sách các tuple (id, title, description, keywords, body).

        Trả về:
            Số lượng mục đã được đánh chỉ mục.
        """
        with self._op_lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM memories")
                if self._fts_available:
                    conn.execute(
                        "INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')"
                    )
            except sqlite3.OperationalError as exc:
                logger.debug("rebuild_all clear failed: %s", exc)
            conn.commit()

            count = 0
            for entry in entries_data:
                if len(entry) < 5:
                    continue
                entry_id, title, description, keywords, body = (
                    entry[0],
                    entry[1],
                    entry[2],
                    entry[3],
                    entry[4],
                )
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO memories "
                        "(id, title, description, keywords, body) VALUES (?, ?, ?, ?, ?)",
                        (
                            entry_id,
                            self._prepare_cjk(title or ""),
                            self._prepare_cjk(description or ""),
                            self._prepare_cjk(keywords or ""),
                            self._prepare_cjk((body or "")[:_MAX_BODY_LEN]),
                        ),
                    )
                    count += 1
                except sqlite3.OperationalError as exc:
                    logger.debug("rebuild_all entry %s failed: %s", entry_id, exc)
            conn.commit()
            return count

    @staticmethod
    def _prepare_cjk(text: str) -> str:
        """Chèn khoảng trắng cho các ký tự CJK và tạo bigram để khớp tốt hơn.

        Ví dụ: "记忆系统" → "记 忆 系 统 记忆 忆系 系统"
        Cho phép khớp cả cụm từ đơn ký tự và hai ký tự trong FTS5.
        Văn bản phi-CJK được giữ nguyên.
        """
        result: list[str] = []
        cjk_buffer: list[str] = []
        text_buffer: list[str] = []

        for char in text:
            if _is_cjk_char(char):
                if text_buffer:
                    result.append("".join(text_buffer))
                    text_buffer = []
                cjk_buffer.append(char)
            else:
                if cjk_buffer:
                    result.append(_expand_cjk_buffer(cjk_buffer))
                    cjk_buffer = []
                text_buffer.append(char)

        if cjk_buffer:
            result.append(_expand_cjk_buffer(cjk_buffer))
        if text_buffer:
            result.append("".join(text_buffer))

        return " ".join(result)

    @staticmethod
    def _clean_cjk(text: str) -> str:
        """Thu gọn khoảng trắng thừa và xóa các bigram trùng lặp khỏi văn bản hiển thị.

        Xóa khoảng trắng giữa các ký tự CJK và lọc bỏ các token bigram là chuỗi con
        của các đoạn CJK hiện có, chuẩn hóa khoảng trắng.
        """
        # Đầu tiên xóa các token chỉ có bigram (hai ký tự CJK liền nhau được bao quanh bởi khoảng trắng)
        # bằng cách thu gọn khoảng trắng giữa các ký tự CJK.
        _cjk_space = re.compile(
            r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])"
        )
        prev = None
        while prev != text:
            prev = text
            text = _cjk_space.sub(r"\1\2", text)
        # Xóa các chuỗi CJK trùng lặp do mở rộng bigram
        text = _dedupe_cjk_runs(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Làm sạch truy vấn người dùng phù hợp với cú pháp MATCH của FTS5.

        Trích xuất các token chữ số (2+ ký tự) và ký tự CJK, tạo bigram cho các ký tự CJK
        liên tiếp, bọc dấu ngoặc kép cho từng token và nối bằng OR để ngăn chặn
        lỗi tiêm toán tử FTS5.

        Tham số:
            query: Chuỗi truy vấn thô của người dùng.

        Trả về:
            Biểu thức MATCH an toàn cho FTS5, hoặc chuỗi bọc ngoặc kép rỗng nếu không có token.
        """
        tokens: list[str] = []
        cjk_buffer: list[str] = []

        raw_tokens = re.findall(
            r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fff\u3400-\u4dbf]", query
        )
        for tok in raw_tokens:
            if len(tok) == 1 and _is_cjk_char(tok):
                cjk_buffer.append(tok)
            else:
                if cjk_buffer:
                    tokens.extend(_cjk_query_tokens(cjk_buffer))
                    cjk_buffer = []
                tokens.append(tok)

        if cjk_buffer:
            tokens.extend(_cjk_query_tokens(cjk_buffer))

        if not tokens:
            return '""'
        # Bọc ngoặc kép từng token và nối bằng OR để mở rộng khả năng khớp
        return " OR ".join(f'"{t}"' for t in tokens)

    def close(self) -> None:
        """Đóng kết nối cơ sở dữ liệu."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_shared_index: Optional[MemorySearchIndex] = None
_shared_lock = threading.Lock()


def get_shared_index(db_path: Optional[Path] = None) -> MemorySearchIndex:
    """Trả về instance Singleton MemorySearchIndex cho toàn bộ tiến trình.

    An toàn luồng (thread-safe) thông qua cơ chế khóa kiểm tra hai lần (double-checked locking).
    Được chia sẻ bởi các trình đánh chỉ mục và tìm kiếm ký ức để dùng chung một kết nối SQLite.

    Tham số:
        db_path: Tùy chọn ghi đè đường dẫn CSDL (chỉ dùng ở lần gọi đầu tiên khi tạo singleton).

    Trả về:
        Instance MemorySearchIndex dùng chung.
    """
    global _shared_index
    if _shared_index is None:
        with _shared_lock:
            if _shared_index is None:
                _shared_index = MemorySearchIndex(db_path=db_path)
    return _shared_index
