"""SkillsLoader: Nạp các tài liệu hướng dẫn kịch bản từ thư mục skills/.

Sử dụng cơ chế tiết lộ tiệm tiến (progressive disclosure):
- System prompt chỉ tiêm các bản tóm tắt 1 dòng (get_descriptions).
- Tài liệu đầy đủ được nạp theo yêu cầu (get_content, được gọi bởi công cụ load_skill).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pseud.agent.frontmatter import parse_frontmatter as _parse_frontmatter  # tiện ích dùng chung


@dataclass
class Skill:
    """Định nghĩa một kỹ năng (skill).

    Thuộc tính:
        name: Tên kỹ năng.
        description: Mô tả kỹ năng.
        category: Danh mục kỹ năng cho hiển thị theo nhóm.
        body: Nội dung văn bản SKILL.md.
        dir_path: Đường dẫn thư mục kỹ năng (dùng để nạp các tệp hỗ trợ khi cần).
        metadata: Metadata frontmatter đã phân tích.
    """

    name: str
    description: str = ""
    category: str = "other"
    body: str = ""
    dir_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def load_support_file(self, filename: str) -> Optional[str]:
        """Nạp một tệp hỗ trợ theo yêu cầu.

        Args:
            filename: Tên tệp (ví dụ examples.md).

        Returns:
            Nội dung tệp hoặc None.
        """
        if not self.dir_path:
            return None
        path = self.dir_path / filename
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None


def _load_skill_dir(dir_path: Path) -> Optional[Skill]:
    """Nạp một kỹ năng từ một thư mục.

    Args:
        dir_path: Đường dẫn thư mục kỹ năng (bắt buộc chứa tệp SKILL.md).

    Returns:
        Instance Skill hoặc None.
    """
    skill_file = dir_path / "SKILL.md"
    if not skill_file.exists():
        return None
    try:
        text = skill_file.read_text(encoding="utf-8")
    except Exception:
        return None

    meta, body = _parse_frontmatter(text)
    name = meta.get("name", dir_path.name)
    if not name:
        return None

    return Skill(
        name=name,
        description=meta.get("description", ""),
        category=meta.get("category", "other"),
        body=body,
        dir_path=dir_path,
        metadata=meta,
    )


def user_skills_dir_default() -> Path:
    """Trả về thư mục kỹ năng do người dùng tạo, bám theo gốc runtime cấu hình được.

    Hằng số cũ neo cứng vào `Path.home()`, nên nó là đường dẫn DUY NHẤT trong module
    bỏ qua biến môi trường gốc runtime — đặt biến đó rồi thì mọi thứ chuyển chỗ trừ
    skills, và kỹ năng người dùng lặng lẽ biến mất.
    """
    from pseud.config.paths import get_runtime_root

    return get_runtime_root() / "skills" / "user"


class SkillsLoader:
    """Nạp các kỹ năng từ thư mục skills/ đóng gói sẵn và thư mục skills người dùng.

    Thuộc tính:
        skills: Danh sách kỹ năng đã nạp (đóng gói sẵn + do người dùng tạo).
    """

    def __init__(self, skills_dir: Optional[Path] = None,
                 user_skills_dir: Optional[Path] = None) -> None:
        """Khởi tạo SkillsLoader.

        Args:
            skills_dir: Đường dẫn thư mục kỹ năng đóng gói sẵn; mặc định là pseud/src/skills/.
            user_skills_dir: Thư mục kỹ năng do người dùng tạo; mặc định `<gốc runtime>/skills/user/`.
        """
        self.skills_dir = skills_dir or Path(__file__).resolve().parents[1] / "skills"
        self._user_skills_dir = user_skills_dir or user_skills_dir_default()
        self.skills: List[Skill] = []
        self._load()

    def _load(self) -> None:
        """Nạp tất cả thư mục con kỹ năng từ thư mục người dùng và thư mục đóng gói sẵn.

        Kỹ năng người dùng được nạp trước để ghi đè lên kỹ năng đóng gói sẵn trùng tên
        (ví dụ sau khi patch_skill sao chép và sửa đổi một kỹ năng đóng gói sẵn).
        """
        seen_names: set[str] = set()
        for directory in (self._user_skills_dir, self.skills_dir):
            if not directory or not directory.exists():
                continue
            try:
                entries = sorted(directory.iterdir())
            except OSError:
                continue
            for path in entries:
                if path.is_dir() and (path / "SKILL.md").exists():
                    skill = _load_skill_dir(path)
                    if skill and skill.name not in seen_names:
                        self.skills.append(skill)
                        seen_names.add(skill.name)

    # Thứ tự hiển thị các danh mục (các danh mục không có trong danh sách xuất hiện ở cuối).
    _CATEGORY_ORDER = [
        "data-source", "strategy", "analysis", "asset-class",
        "crypto", "flow", "tool", "other",
    ]

    def get_descriptions(self) -> str:
        """Trả về danh sách kỹ năng nhóm theo danh mục phục vụ system prompt.

        Returns:
            Danh sách kỹ năng đã nhóm kèm tiêu đề danh mục.
        """
        if not self.skills:
            return "(no skills)"

        groups: Dict[str, List[Skill]] = {}
        for skill in self.skills:
            groups.setdefault(skill.category, []).append(skill)

        ordered_cats = [c for c in self._CATEGORY_ORDER if c in groups]
        ordered_cats += [c for c in sorted(groups) if c not in ordered_cats]

        lines: List[str] = []
        for cat in ordered_cats:
            lines.append(f"\n### {cat}")
            for skill in groups[cat]:
                lines.append(f"  - {skill.name}: {skill.description}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Trả về tài liệu đầy đủ cho một kỹ năng (được dùng bởi công cụ load_skill).

        Tự động tra cứu đĩa cho các kỹ năng người dùng được tạo giữa phiên làm việc.

        Args:
            name: Tên kỹ năng.

        Returns:
            Văn bản tài liệu kỹ năng đầy đủ bọc trong thẻ XML, hoặc thông báo lỗi.
        """
        for skill in self.skills:
            if skill.name == name:
                return f'<skill name="{name}">\n{skill.body}\n</skill>'

        # Tìm kiếm dự phòng: kiểm tra thư mục kỹ năng người dùng trên đĩa (kỹ năng tạo giữa phiên)
        if self._user_skills_dir:
            skill = _load_skill_dir(self._user_skills_dir / name)
            if skill:
                self.skills.append(skill)
                return f'<skill name="{name}">\n{skill.body}\n</skill>'

        available = ", ".join(s.name for s in self.skills)
        return f"Error: Unknown skill '{name}'. Available: {available}"

