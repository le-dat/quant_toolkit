"""Quy trình nén bộ nhớ 3 cấp độ: Raw -> Daily -> Digest (Tier 2).

Sử dụng chấm điểm câu dựa trên TF-IDF để trích xuất thông tin cốt lõi trong khi vẫn giữ
tỷ lệ lưu giữ thông tin cao. Nội dung gốc sẽ được sao lưu lưu trữ trước khi nén.

Cờ tính năng: VT_MEMORY_COMPRESSION (mặc định tắt).
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ─── Hằng số cấp độ nén ──────────────────────────────────────────────────

LEVEL_RAW = "raw"
LEVEL_DAILY = "daily"
LEVEL_DIGEST = "digest"

# Ngưỡng kích hoạt (số ngày kể từ lần truy cập cuối)
DAILY_THRESHOLD_DAYS = 7
DIGEST_THRESHOLD_DAYS = 30

# Mục tiêu nén
DAILY_TOP_K_SENTENCES = 5  # Giữ top-5 câu + câu đầu/câu cuối
DIGEST_MAX_TOKENS = 50  # Số token tối đa trong digest
DIGEST_TOP_KEYWORDS = 15  # Top từ khóa cho danh sách gạch đầu dòng digest

# Số giây trong một ngày
_SECONDS_PER_DAY = 86400

# ─── Tách token ───────────────────────────────────────────────────────────

_NON_LATIN_RANGES = (
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Extension A
    "\u0e00-\u0e7f"  # Thai
    "\u0620-\u064a"  # Arabic letters
    "\u05d0-\u05ea"  # Hebrew letters
    "\u0400-\u04ff"  # Cyrillic
)
_TOKEN_RE = re.compile(rf"[a-zA-Z0-9]{{3,}}|[{_NON_LATIN_RANGES}]")

# Biểu thức chính quy ranh giới câu: tách theo . ! ? và các dấu ngắt câu CJK
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


# ─── Hàm công khai ────────────────────────────────────────────────────────


def _tokenize_for_tfidf(text: str) -> List[str]:
    """Tách token văn bản phục vụ tính điểm TF-IDF.

    Sử dụng biểu thức chính quy tương tự như trong persistent.py: Từ ASCII >= 3 ký tự
    hoặc từng ký tự phi-Latinh đơn lẻ.
    """
    return _TOKEN_RE.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    """Tách văn bản thành các câu đơn lẻ.

    Xử lý dấu chấm/chấm hỏi/chấm cảm tiếng Anh và dấu chấm câu tiếng Trung/CJK.
    Lọc bỏ các kết quả rỗng.
    """
    raw_parts = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in raw_parts if s.strip()]
    return sentences


def compute_tfidf(documents: List[str]) -> Dict[str, float]:
    """Tính điểm IDF trên tập hợp các tài liệu (câu).

    Tham số:
        documents: Danh sách văn bản tài liệu (thường là các câu).

    Trả về:
        Dict ánh xạ thuật ngữ -> điểm IDF. IDF = log(N / (1 + df(t))).
    """
    n = len(documents)
    if n == 0:
        return {}

    # Đếm tần suất tài liệu cho từng thuật ngữ
    df: Counter = Counter()
    for doc in documents:
        unique_terms = set(_tokenize_for_tfidf(doc))
        for term in unique_terms:
            df[term] += 1

    # Tính điểm IDF: log(N / (1 + df(term)))
    idf_scores: Dict[str, float] = {}
    for term, freq in df.items():
        idf_scores[term] = math.log(n / (1 + freq))

    return idf_scores


def _score_sentence(sentence: str, idf_scores: Dict[str, float]) -> float:
    """Tính điểm cho một câu bằng tổng trọng số IDF đã chuẩn hóa.

    Điểm = sum(idf(token) for token in sentence) / len(tokens).
    Trả về 0.0 đối với các câu rỗng.
    """
    tokens = _tokenize_for_tfidf(sentence)
    if not tokens:
        return 0.0
    total = sum(idf_scores.get(t, 0.0) for t in tokens)
    return total / len(tokens)


def extract_key_sentences(
    content: str, idf_scores: Dict[str, float], top_k: int = DAILY_TOP_K_SENTENCES
) -> str:
    """Trích xuất top-k câu có tổng trọng số TF-IDF cao nhất.

    Luôn bao gồm câu đầu tiên và câu cuối cùng để giữ ngữ cảnh.
    Trả về văn bản đã nối của các câu được chọn.
    """
    sentences = _split_sentences(content)
    if len(sentences) <= top_k + 2:
        # Đã đủ ngắn, giữ nguyên
        return content.strip()

    # Tính điểm tất cả các câu
    scored: List[Tuple[int, float, str]] = []
    for idx, sent in enumerate(sentences):
        score = _score_sentence(sent, idf_scores)
        scored.append((idx, score, sent))

    # Luôn giữ câu đầu và câu cuối
    first_idx = 0
    last_idx = len(sentences) - 1
    kept_indices: Set[int] = {first_idx, last_idx}

    # Chọn top-k các câu ở giữa theo điểm số
    middle = [(idx, sc, s) for idx, sc, s in scored if idx not in kept_indices]
    middle.sort(key=lambda x: x[1], reverse=True)

    for idx, _sc, _s in middle[:top_k]:
        kept_indices.add(idx)

    # Dựng lại theo thứ tự ban đầu
    selected = [sentences[i] for i in sorted(kept_indices)]
    return " ".join(selected)


# ─── CompressionPipeline ─────────────────────────────────────────────────────


class CompressionPipeline:
    """Quy trình nén 3 cấp độ dành cho các mục ký ức."""

    def __init__(self, memory_dir: Path) -> None:
        """Khởi tạo với thư mục ký ức phục vụ lưu trữ bản sao."""
        self._memory_dir = memory_dir
        self._archive_dir = memory_dir / "archive"

    def should_compress(
        self, compression_level: str, last_accessed: float, now: float = 0.0
    ) -> Optional[str]:
        """Xác định xem một mục có nên được nén hay không và nén đến cấp độ nào.

        Tham số:
            compression_level: Cấp độ nén hiện tại ("raw", "daily", "digest").
            last_accessed: Mốc thời gian epoch của lần truy cập cuối.
            now: Thời gian hiện tại (mặc định là time.time()).

        Trả về:
            Cấp độ nén mục tiêu ("daily" hoặc "digest") hoặc None.
        """
        if now <= 0.0:
            now = time.time()

        days_since_access = (now - last_accessed) / _SECONDS_PER_DAY

        if compression_level == LEVEL_RAW and days_since_access > DAILY_THRESHOLD_DAYS:
            return LEVEL_DAILY
        if compression_level == LEVEL_DAILY and days_since_access > DIGEST_THRESHOLD_DAYS:
            return LEVEL_DIGEST

        # Đã ở cấp độ digest hoặc chưa đủ thời gian trôi qua
        return None

    def compress_to_daily(self, content: str, keywords: tuple = ()) -> str:
        """Nén nội dung thô về cấp độ daily (~50% kích thước).

        Chiến lược: Trích xuất câu khóa bằng TF-IDF.
        - Tách thành các câu
        - Tính điểm cho từng câu bằng tổng trọng số IDF
        - Giữ top-5 câu + câu đầu tiên + câu cuối cùng
        - Thêm các từ khóa vào đầu làm tiêu đề ngữ cảnh
        """
        sentences = _split_sentences(content)
        if not sentences:
            return content

        # Sử dụng các câu làm tập hợp tài liệu để tính IDF
        idf_scores = compute_tfidf(sentences)

        # Trích xuất các câu khóa
        compressed = extract_key_sentences(content, idf_scores, DAILY_TOP_K_SENTENCES)

        # Thêm các từ khóa vào đầu làm tiêu đề ngữ cảnh nếu có
        if keywords:
            header = "Keywords: " + ", ".join(keywords)
            compressed = header + "\n\n" + compressed

        return compressed

    def compress_to_digest(self, daily_content: str, keywords: tuple = ()) -> str:
        """Nén nội dung cấp độ daily về cấp độ digest (~10-20% kích thước ban đầu).

        Chiến lược: Trích xuất khái niệm cốt lõi và từ khóa.
        - Trích xuất top-N thuật ngữ quan trọng nhất
        - Định dạng thành bản tóm tắt danh sách gạch đầu dòng
        """
        tokens = _tokenize_for_tfidf(daily_content)
        if not tokens:
            return daily_content

        # Đếm tần suất xuất hiện của thuật ngữ trong nội dung
        tf: Counter = Counter(tokens)

        # Sử dụng IDF cấp độ câu để tính trọng số tầm quan trọng
        sentences = _split_sentences(daily_content)
        idf_scores = compute_tfidf(sentences) if sentences else {}

        # Tính điểm cho từng thuật ngữ duy nhất: tf * idf
        term_scores: List[Tuple[str, float]] = []
        for term, freq in tf.items():
            idf = idf_scores.get(term, 1.0)
            term_scores.append((term, freq * idf))

        # Sắp xếp theo điểm số giảm dần, lấy các từ khóa hàng đầu
        term_scores.sort(key=lambda x: x[1], reverse=True)
        top_terms = [t for t, _s in term_scores[:DIGEST_TOP_KEYWORDS]]

        # Dựng bản tóm tắt danh sách gạch đầu dòng
        lines: List[str] = []
        if keywords:
            lines.append("Context: " + ", ".join(keywords))
        lines.append("Key concepts:")
        for term in top_terms:
            lines.append(f"  - {term}")

        return "\n".join(lines)

    def archive_original(self, entry_path: Path) -> Optional[Path]:
        """Sao lưu tệp gốc vào thư mục archive/ trước khi nén.

        Trả về đường dẫn lưu trữ hoặc None nếu thất bại. Tạo thư mục archive/
        khi có nhu cầu. Sử dụng thao tác sao chép nguyên tử (ghi tmp + đổi tên).
        """
        if not entry_path.exists():
            logger.warning("Cannot archive non-existent file: %s", entry_path)
            return None

        try:
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = self._archive_dir / entry_path.name
            tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")

            # Sao chép nguyên tử: ghi vào tệp tmp rồi đổi tên
            shutil.copy2(str(entry_path), str(tmp_path))
            os.replace(str(tmp_path), str(archive_path))

            logger.debug("Archived %s -> %s", entry_path.name, archive_path)
            return archive_path
        except OSError as exc:
            logger.error("Archive failed for %s: %s", entry_path, exc)
            # Dọn dẹp tệp tmp nếu tồn tại
            tmp_path = (self._archive_dir / entry_path.name).with_suffix(
                entry_path.suffix + ".tmp"
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return None

    def apply_compression(
        self, entry_path: Path, content: str, keywords: tuple, target_level: str
    ) -> Optional[str]:
        """Thực thi nén: lưu trữ bản gốc, tính toán nội dung đã nén.

        Tham số:
            entry_path: Đường dẫn tới tệp ký ức .md.
            content: Nội dung thân hiện tại.
            keywords: Các từ khóa của mục phục vụ ngữ cảnh.
            target_level: "daily" hoặc "digest".

        Trả về:
            Chuỗi nội dung đã nén, hoặc None nếu thất bại.
        """
        # Lưu trữ bản gốc trước khi sửa đổi
        archive_result = self.archive_original(entry_path)
        if archive_result is None and entry_path.exists():
            # Lưu trữ thất bại nhưng tệp tồn tại - hủy bỏ để tránh mất dữ liệu
            logger.error(
                "Compression aborted: archive failed for %s", entry_path
            )
            return None

        try:
            if target_level == LEVEL_DAILY:
                compressed = self.compress_to_daily(content, keywords)
            elif target_level == LEVEL_DIGEST:
                compressed = self.compress_to_digest(content, keywords)
            else:
                logger.error("Unknown compression target: %s", target_level)
                return None

            retention = self.estimate_retention(content, compressed)
            logger.info(
                "Compressed %s to %s (retention=%.2f)",
                entry_path.name,
                target_level,
                retention,
            )
            return compressed
        except Exception as exc:
            logger.error("Compression failed for %s: %s", entry_path, exc)
            return None

    def estimate_retention(self, original: str, compressed: str) -> float:
        """Ước tính tỷ lệ lưu giữ thông tin qua độ chồng lấp token Jaccard (0.0-1.0).

        Tính toán |giao| / |hợp| của các tập token giữa văn bản gốc và văn bản nén.
        """
        orig_tokens = set(_tokenize_for_tfidf(original))
        comp_tokens = set(_tokenize_for_tfidf(compressed))

        if not orig_tokens and not comp_tokens:
            return 1.0
        if not orig_tokens or not comp_tokens:
            return 0.0

        intersection = orig_tokens & comp_tokens
        union = orig_tokens | comp_tokens
        return len(intersection) / len(union)
