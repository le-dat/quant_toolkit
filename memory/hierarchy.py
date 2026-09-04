"""Định tuyến thư mục phân cấp cho các mục ký ức (Tier 2).

Sắp xếp tệp ký ức vào các thư mục con phân loại dựa trên memory_type,
cho phép tìm kiếm phạm vi O(category_size) thay vì quét phẳng O(n).

Cờ tính năng: VT_MEMORY_HIERARCHY (mặc định tắt).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Thư mục phân loại khớp với MEMORY_TYPES trong persistent.py
CATEGORIES = ("user", "feedback", "project", "reference", "research")

# Các tệp/thư mục cần bỏ qua khi quét
_SKIP_NAMES = frozenset({"MEMORY.md", ".hierarchy.yaml", ".lock", "archive", "gc.log"})


@dataclass
class CategorySummary:
    """Metadata tóm tắt cho một thư mục phân loại đơn lẻ."""

    count: int = 0
    keywords: List[str] = field(default_factory=list)


class MemoryHierarchy:
    """Quản lý cấu trúc thư mục phân cấp cho các mục ký ức.

    Cung cấp định tuyến truy cập O(category_size) tới tệp ký ức theo memory_type,
    đồng thời duy trì tính tương thích ngược với các mục lưu trữ phẳng ở thư mục gốc.
    """

    def __init__(self, base_dir: Path) -> None:
        """Khởi tạo với thư mục gốc ký ức.

        Tham số:
            base_dir: Thư mục gốc để lưu trữ ký ức.
        """
        self._base_dir = base_dir
        self._index_path = base_dir / ".hierarchy.yaml"

    @property
    def base_dir(self) -> Path:
        """Trả về thư mục gốc được cấu hình."""
        return self._base_dir

    def _ensure_category_dir(self, category: str) -> Path:
        """Tạo thư mục con phân loại nếu chưa tồn tại.

        Tham số:
            category: Một trong các CATEGORIES hoặc chuỗi bất kỳ.

        Trả về:
            Đường dẫn tới thư mục con phân loại.
        """
        cat_dir = self._base_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        return cat_dir

    def route_entry(self, memory_type: str, filename: str) -> Path:
        """Xác định đường dẫn lưu trữ: base_dir/{memory_type}/{filename}.

        Tạo thư mục phân loại khi có yêu cầu.
        Quay về base_dir đối với các loại chưa biết.

        Tham số:
            memory_type: Định danh phân loại (ví dụ: "user", "project").
            filename: Tên tệp .md cho mục ký ức.

        Trả về:
            Đường dẫn đầy đủ nơi mục ký ức sẽ được lưu trữ.
        """
        if memory_type in CATEGORIES:
            cat_dir = self._ensure_category_dir(memory_type)
            return cat_dir / filename
        # Loại không xác định sẽ quay về thư mục gốc
        logger.warning(
            "Unknown memory_type '%s', routing to base dir", memory_type
        )
        return self._base_dir / filename

    def scan_all(self) -> List[Path]:
        """Quét thư mục gốc + tất cả thư mục con phân loại cho các tệp *.md.

        Bỏ qua các mục trong _SKIP_NAMES và thư mục archive/.
        Bao gồm các mục lưu trữ phẳng ở thư mục gốc để đảm bảo tương thích ngược.

        Trả về:
            Danh sách đường dẫn tệp .md đã sắp xếp.
        """
        results: List[Path] = []

        # Quét thư mục gốc (các mục lưu trữ phẳng legacy)
        if self._base_dir.is_dir():
            for item in self._base_dir.iterdir():
                if item.name in _SKIP_NAMES:
                    continue
                if item.is_file() and item.suffix == ".md":
                    results.append(item)

        # Quét từng thư mục con phân loại đã biết
        for category in CATEGORIES:
            cat_dir = self._base_dir / category
            if cat_dir.is_dir():
                for item in cat_dir.iterdir():
                    if item.is_file() and item.suffix == ".md":
                        results.append(item)

        results.sort(key=lambda p: p.name)
        return results

    def scan_category(self, category: str) -> List[Path]:
        """Chỉ quét duy nhất một thư mục con phân loại cho các tệp *.md.

        Tham số:
            category: Tên danh mục cần quét.

        Trả về:
            Danh sách đường dẫn tệp .md đã sắp xếp trong thư mục danh mục đó.
        """
        cat_dir = self._base_dir / category
        results: List[Path] = []

        if not cat_dir.is_dir():
            return results

        for item in cat_dir.iterdir():
            if item.is_file() and item.suffix == ".md":
                results.append(item)

        results.sort(key=lambda p: p.name)
        return results

    def rebuild_index(self, entries: list) -> None:
        """Ghi tệp .hierarchy.yaml chứa tóm tắt theo từng danh mục.

        Định dạng:
            categories:
              <name>:
                count: <int>
                keywords: [kw1, kw2, ...]
            rebuilt_at: <ISO-8601>

        Sử dụng định dạng YAML thủ công (không phụ thuộc vào PyYAML).

        Tham số:
            entries: Danh sách các dict chứa tối thiểu khóa 'memory_type' và
                     tùy chọn 'keywords' (danh sách các chuỗi).
        """
        # Tổng hợp thống kê theo danh mục
        cat_data: Dict[str, CategorySummary] = {
            cat: CategorySummary() for cat in CATEGORIES
        }

        for entry in entries:
            mtype = entry.get("memory_type", "")
            if mtype not in cat_data:
                continue
            cat_data[mtype].count += 1
            # Thu thập từ khóa (khử trùng lặp sau)
            keywords = entry.get("keywords", [])
            if isinstance(keywords, list):
                cat_data[mtype].keywords.extend(keywords)

        # Khử trùng lặp và giới hạn số từ khóa cho mỗi danh mục
        max_keywords = 10  # giữ chỉ mục gọn nhẹ
        for summary in cat_data.values():
            seen: Set[str] = set()
            unique: List[str] = []
            for kw in summary.keywords:
                kw_lower = kw.lower().strip()
                if kw_lower and kw_lower not in seen:
                    seen.add(kw_lower)
                    unique.append(kw_lower)
            summary.keywords = unique[:max_keywords]

        # Tạo mốc thời gian ISO-8601
        rebuilt_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # Ghi YAML thủ công
        lines: List[str] = [
            "# Auto-generated memory hierarchy index",
            f'rebuilt_at: "{rebuilt_at}"',
            "categories:",
        ]
        for cat in CATEGORIES:
            summary = cat_data[cat]
            kw_list = ", ".join(summary.keywords)
            lines.append(f"  {cat}:")
            lines.append(f"    count: {summary.count}")
            lines.append(f"    keywords: [{kw_list}]")

        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.debug("Rebuilt hierarchy index at %s", self._index_path)

    def _parse_index_keywords(self) -> Dict[str, List[str]]:
        """Phân tích cú pháp .hierarchy.yaml để trích xuất từ khóa theo từng danh mục.

        Trả về:
            Dict ánh xạ tên danh mục với danh sách các từ khóa.
            Dict rỗng nếu tệp chỉ mục không tồn tại hoặc sai định dạng.
        """
        if not self._index_path.is_file():
            return {}

        result: Dict[str, List[str]] = {}
        current_cat: Optional[str] = None

        try:
            text = self._index_path.read_text(encoding="utf-8")
        except OSError:
            return {}

        for line in text.splitlines():
            stripped = line.strip()

            # Phát hiện tiêu đề danh mục (ví dụ: "  user:")
            if (
                stripped.endswith(":")
                and not stripped.startswith("#")
                and not stripped.startswith("categories")
                and not stripped.startswith("rebuilt_at")
            ):
                cat_name = stripped.rstrip(":")
                if cat_name in CATEGORIES:
                    current_cat = cat_name
                    if current_cat not in result:
                        result[current_cat] = []
                else:
                    current_cat = None

            # Phát hiện dòng từ khóa (ví dụ: "    keywords: [a, b, c]")
            elif stripped.startswith("keywords:") and current_cat:
                # Trích xuất nội dung giữa hai dấu ngoặc vuông
                bracket_start = stripped.find("[")
                bracket_end = stripped.find("]")
                if bracket_start != -1 and bracket_end != -1:
                    inner = stripped[bracket_start + 1 : bracket_end]
                    keywords = [
                        k.strip() for k in inner.split(",") if k.strip()
                    ]
                    result[current_cat] = keywords

        return result

    def prune_search_scope(
        self, query_tokens: Set[str], category_filter: str = ""
    ) -> List[Path]:
        """Thu hẹp phạm vi tìm kiếm dựa trên bộ lọc danh mục hoặc độ chồng lấp từ khóa.

        Nếu category_filter được thiết lập, chỉ quét duy nhất danh mục đó.
        Nếu không, quét tất cả các danh mục (bảo toàn tính tương thích).

        Khi không có category_filter và chỉ mục tồn tại, các danh mục
        được sắp xếp theo điểm chồng lấp từ khóa (liên quan nhất lên đầu), nhưng
        tất cả các tệp vẫn được trả về để đảm bảo đầy đủ.

        Tham số:
            query_tokens: Tập hợp các từ truy vấn chữ thường để chấm điểm độ liên quan.
            category_filter: Nếu không rỗng, giới hạn quét trong danh mục này.

        Trả về:
            Danh sách đường dẫn tệp .md theo thứ tự ưu tiên.
        """
        # Bộ lọc danh mục trực tiếp — đường đi nhanh
        if category_filter:
            if category_filter in CATEGORIES:
                return self.scan_category(category_filter)
            # Bộ lọc không xác định: quay về quét toàn bộ
            logger.warning(
                "Unknown category_filter '%s', falling back to scan_all",
                category_filter,
            )
            return self.scan_all()

        # Không có bộ lọc: dùng độ chồng lấp từ khóa để sắp xếp các danh mục
        index_keywords = self._parse_index_keywords()

        if not index_keywords:
            # Không có chỉ mục — quét toàn bộ
            return self.scan_all()

        # Tính điểm từng danh mục theo độ chồng lấp từ khóa với các token truy vấn
        scored: List[tuple] = []
        for cat in CATEGORIES:
            cat_kws = set(index_keywords.get(cat, []))
            overlap = len(query_tokens & cat_kws)
            scored.append((overlap, cat))

        # Sắp xếp giảm dần theo điểm số chồng lấp
        scored.sort(key=lambda x: x[0], reverse=True)

        # Thu thập các tệp: danh mục ưu tiên trước, sau đó tới thư mục gốc
        results: List[Path] = []
        seen: Set[Path] = set()

        for _score, cat in scored:
            for p in self.scan_category(cat):
                if p not in seen:
                    results.append(p)
                    seen.add(p)

        # Bao gồm các mục lưu trữ phẳng ở thư mục gốc để đảm bảo tương thích ngược
        if self._base_dir.is_dir():
            for item in self._base_dir.iterdir():
                if item.name in _SKIP_NAMES:
                    continue
                if item.is_file() and item.suffix == ".md" and item not in seen:
                    results.append(item)
                    seen.add(item)

        return results

    def migrate_flat_entry(
        self, file_path: Path, memory_type: str
    ) -> Optional[Path]:
        """Di chuyển một mục lưu trữ phẳng vào thư mục con danh mục tương ứng.

        Chỉ di chuyển nếu tệp hiện nằm ở base_dir (chưa ở trong thư mục con).
        Bảo toàn nguyên vẹn nội dung tệp.

        Tham số:
            file_path: Đường dẫn hiện tại của tệp ký ức .md.
            memory_type: Danh mục mục tiêu cho tệp.

        Trả về:
            Đường dẫn mới sau khi di chuyển, hoặc None nếu việc di chuyển bị bỏ qua
            (tệp không ở thư mục gốc, mục tiêu trùng hiện tại, hoặc có lỗi).
        """
        # Xác thực nguồn tồn tại và nằm trong thư mục gốc
        if not file_path.is_file():
            logger.warning("Cannot migrate non-existent file: %s", file_path)
            return None

        if file_path.parent != self._base_dir:
            logger.debug(
                "File %s not in base dir, skip migration", file_path.name
            )
            return None

        if memory_type not in CATEGORIES:
            logger.warning(
                "Cannot migrate to unknown category '%s'", memory_type
            )
            return None

        # Xác định đích đến
        dest_dir = self._ensure_category_dir(memory_type)
        dest_path = dest_dir / file_path.name

        if dest_path.exists():
            logger.warning(
                "Destination already exists, skip migration: %s", dest_path
            )
            return None

        try:
            file_path.rename(dest_path)
            logger.info(
                "Migrated %s -> %s/%s",
                file_path.name,
                memory_type,
                file_path.name,
            )
            return dest_path
        except OSError as exc:
            logger.error("Migration failed for %s: %s", file_path.name, exc)
            return None
