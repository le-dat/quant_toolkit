"""Liên kết ngữ nghĩa giữa các mục ký ức qua độ tương đồng BM25 (Tier 2).

Tự động khám phá và duy trì mối quan hệ giữa các ký ức sử dụng điểm tần suất thuật ngữ.
Các liên kết được lưu dưới dạng tệp sidecar .relations.json.

Cờ tính năng: VT_MEMORY_LINKS (mặc định tắt).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Tham số BM25
_BM25_K1 = 1.5
_BM25_B = 0.75

# Ngưỡng liên kết
_MIN_SCORE_THRESHOLD = 0.3
_MAX_OUTGOING_LINKS = 10

# Wikilink regex: [[6-char-hex-id]]
_WIKILINK_RE = re.compile(r"\[\[([a-f0-9]{6})\]\]")

# Tách từ (khớp với regex trong persistent.py)
_NON_LATIN_SCRIPT_RANGES = (
    "一-鿿"  # Chữ Hán CJK hợp nhất
    "㐀-䶿"  # CJK mở rộng A
    "฀-๿"  # Chữ Thái
    "ؠ-ي"  # Chữ Ả Rập
    "א-ת"  # Chữ Do Thái
    "Ѐ-ӿ"  # Chữ Cyrillic
)
_TOKEN_RE = re.compile(rf"[a-zA-Z0-9]{{3,}}|[{_NON_LATIN_SCRIPT_RANGES}]")

# Phiên bản schema tệp quan hệ
_RELATIONS_VERSION = 1


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tách từ văn bản để tính điểm BM25."""
    return _TOKEN_RE.findall(text.lower())


def compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """Tính điểm IDF từ một kho tài liệu các token.

    Công thức: IDF(t) = log((N - n + 0.5) / (n + 0.5) + 1)
    trong đó N = tổng số tài liệu, n = số tài liệu chứa thuật ngữ t.

    Tham số:
        corpus: Danh sách các tài liệu đã tách từ.

    Trả về:
        Dictionary ánh xạ mỗi thuật ngữ với điểm IDF của nó.
    """
    n_docs = len(corpus)
    if n_docs == 0:
        return {}

    doc_freq: Counter = Counter()
    for doc_tokens in corpus:
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            doc_freq[term] += 1

    idf_scores: Dict[str, float] = {}
    for term, freq in doc_freq.items():
        idf_scores[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    return idf_scores


def compute_bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    idf_scores: Dict[str, float],
    avg_dl: float,
) -> float:
    """Tính điểm BM25 cho một tài liệu đơn lẻ so với các token truy vấn.

    Công thức cho mỗi thuật ngữ t:
        score += IDF(t) * (tf(t) * (k1 + 1)) / (tf(t) + k1 * (1 - b + b * dl / avgdl))

    Tham số:
        query_tokens: Token truy vấn (mục ký ức nguồn).
        doc_tokens: Token tài liệu ứng viên.
        idf_scores: Ánh xạ điểm IDF đã tính sẵn.
        avg_dl: Độ dài tài liệu trung bình trên toàn bộ kho.

    Trả về:
        Điểm tương đồng BM25 (số thực không âm).
    """
    if not doc_tokens or avg_dl <= 0:
        return 0.0

    dl = len(doc_tokens)
    tf_map: Counter = Counter(doc_tokens)

    score = 0.0
    # Khử trùng lặp token truy vấn để chấm điểm
    seen_terms: Set[str] = set()
    for term in query_tokens:
        if term in seen_terms:
            continue
        seen_terms.add(term)

        idf = idf_scores.get(term, 0.0)
        if idf <= 0:
            continue

        tf = tf_map.get(term, 0)
        if tf == 0:
            continue

        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl)
        score += idf * numerator / denominator

    return score


# ---------------------------------------------------------------------------
# Class SemanticLinker
# ---------------------------------------------------------------------------


class SemanticLinker:
    """Quản lý liên kết ngữ nghĩa dựa trên BM25 giữa các mục ký ức.

    Chịu trách nhiệm khám phá các ký ức liên quan thông qua chấm điểm tương đồng BM25,
    lưu trữ dữ liệu liên kết thành tệp sidecar .relations.json, và giải mã
    các tham chiếu wikilink được chèn trong văn bản ký ức.
    """

    def __init__(self, memory_dir: Path) -> None:
        """Khởi tạo với đường dẫn thư mục ký ức.

        Tham số:
            memory_dir: Đường dẫn tới thư mục chứa các tệp ký ức .md.
        """
        self._memory_dir = memory_dir

    @property
    def memory_dir(self) -> Path:
        """Thư mục ký ức mà trình liên kết này thao tác."""
        return self._memory_dir

    def discover_links(
        self,
        entry_title: str,
        entry_tokens: List[str],
        all_entries_data: List[Tuple[str, List[str]]],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Tìm top-k mục tương đồng nhất qua BM25.

        Tham số:
            entry_title: Tiêu đề của mục nguồn (dùng để loại trừ chính nó).
            entry_tokens: Token của mục nguồn (truy vấn).
            all_entries_data: Danh sách (filename, tokens) cho tất cả các mục.
            top_k: Số lượng liên kết tối đa trả về.

        Trả về:
            Danh sách (target_filename, bm25_score) sắp xếp giảm dần theo điểm số,
            được lọc bởi _MIN_SCORE_THRESHOLD và giới hạn ở _MAX_OUTGOING_LINKS.
        """
        if not entry_tokens or not all_entries_data:
            return []

        # Xây dựng kho tài liệu phục vụ tính IDF (loại trừ chính nó)
        corpus: List[List[str]] = []
        filenames: List[str] = []
        for fname, tokens in all_entries_data:
            if fname == entry_title:
                continue
            corpus.append(tokens)
            filenames.append(fname)

        if not corpus:
            return []

        # Tính điểm IDF và độ dài tài liệu trung bình
        idf_scores = compute_idf(corpus)
        total_tokens = sum(len(doc) for doc in corpus)
        avg_dl = total_tokens / len(corpus) if corpus else 1.0

        # Tính điểm từng ứng viên
        scored: List[Tuple[str, float]] = []
        for idx, doc_tokens in enumerate(corpus):
            score = compute_bm25_score(entry_tokens, doc_tokens, idf_scores, avg_dl)
            if score >= _MIN_SCORE_THRESHOLD:
                scored.append((filenames[idx], score))

        # Sắp xếp giảm dần theo điểm số
        scored.sort(key=lambda x: x[1], reverse=True)

        # Áp dụng các giới hạn
        effective_k = min(top_k, _MAX_OUTGOING_LINKS)
        return scored[:effective_k]

    def save_relations(self, entry_path: Path, links: List[Tuple[str, float]]) -> None:
        """Ghi tệp sidecar .relations.json bên cạnh tệp ký ức.

        Sử dụng ghi nguyên tử (ghi vào tệp tạm + đổi tên) để đảm bảo an toàn luồng
        và ngăn ngừa việc ghi dở dang khi bị sự cố.

        Tham số:
            entry_path: Đường dẫn tới tệp ký ức nguồn.
            links: Danh sách các tuple (target_filename, score) cần lưu trữ.
        """
        rel_path = self.get_relation_path(entry_path)
        data = {
            "version": _RELATIONS_VERSION,
            "links": [
                {"target": target, "score": round(score, 4)}
                for target, score in links
            ],
            "updated_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        }

        content = json.dumps(data, indent=2, ensure_ascii=False)

        # Ghi nguyên tử: ghi vào tệp tạm trong cùng thư mục, sau đó đổi tên
        dir_path = rel_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix=".relations_", dir=str(dir_path)
            )
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(tmp_path, str(rel_path))
            tmp_path = None
        except OSError:
            logger.exception("Failed to write relations file: %s", rel_path)
            if fd is not None:
                os.close(fd)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def load_relations(self, entry_path: Path) -> List[Tuple[str, float]]:
        """Đọc các quan hệ từ tệp sidecar .relations.json.

        Tham số:
            entry_path: Đường dẫn tới tệp ký ức nguồn.

        Trả về:
            Danh sách các tuple (target_filename, score). Trả về [] nếu không tìm thấy tệp
            hoặc gặp lỗi phân tích cú pháp.
        """
        rel_path = self.get_relation_path(entry_path)
        if not rel_path.exists():
            return []

        try:
            raw = rel_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read relations file: %s", rel_path)
            return []

        if not isinstance(data, dict) or data.get("version") != _RELATIONS_VERSION:
            logger.warning("Unsupported relations format: %s", rel_path)
            return []

        links_raw = data.get("links", [])
        if not isinstance(links_raw, list):
            return []

        result: List[Tuple[str, float]] = []
        for item in links_raw:
            if isinstance(item, dict) and "target" in item and "score" in item:
                try:
                    result.append((str(item["target"]), float(item["score"])))
                except (TypeError, ValueError):
                    continue

        return result

    def resolve_wikilinks(self, body: str) -> List[str]:
        """Phân tích các tham chiếu [[6-char-hex-id]] từ văn bản nội dung.

        Wikilink cung cấp các liên kết chéo tường minh giữa các ký ức.
        Định dạng: [[abcdef]] trong đó abcdef là định danh hex 6 ký tự.

        Tham số:
            body: Văn bản nội dung ký ức cần phân tích.

        Trả về:
            Danh sách các ID hex duy nhất tìm thấy, theo thứ tự xuất hiện đầu tiên.
        """
        if not body:
            return []

        seen: Set[str] = set()
        result: List[str] = []
        for match in _WIKILINK_RE.finditer(body):
            hex_id = match.group(1)
            if hex_id not in seen:
                seen.add(hex_id)
                result.append(hex_id)

        return result

    def get_relation_path(self, entry_path: Path) -> Path:
        """Trả về đường dẫn .relations.json cho một tệp ký ức cho trước.

        Tệp sidecar được đặt cùng thư mục với tệp ký ức,
        có tên dạng {stem}.relations.json.

        Tham số:
            entry_path: Đường dẫn tới tệp ký ức.

        Trả về:
            Đường dẫn tới tệp .relations.json tương ứng.
        """
        return entry_path.parent / f"{entry_path.stem}.relations.json"

    def remove_relations(self, entry_path: Path) -> None:
        """Xóa tệp sidecar .relations.json nếu tồn tại.

        Tham số:
            entry_path: Đường dẫn tới tệp ký ức có các quan hệ cần xóa.
        """
        rel_path = self.get_relation_path(entry_path)
        if rel_path.exists():
            try:
                rel_path.unlink()
            except OSError:
                logger.warning("Failed to remove relations file: %s", rel_path)
