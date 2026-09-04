"""Lưu trữ trạng thái lượt chạy: tạo các thư mục lượt chạy và ghi nhận trạng thái."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class RunStateStore:
    """Kho lưu trữ trạng thái lượt chạy: quản lý thư mục lượt chạy và trạng thái vòng đời của chúng."""

    def create_run_dir(self, workspace: Path) -> Path:
        """Tạo một thư mục lượt chạy duy nhất.

        Args:
            workspace: Thư mục cha (thường là runs/).

        Returns:
            Đường dẫn thư mục lượt chạy mới tạo.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
        suffix = uuid.uuid4().hex[:6]
        run_dir = workspace / f"{timestamp}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "code").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        return run_dir

    def save_request(self, run_dir: Path, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Lưu yêu cầu của người dùng.

        Args:
            run_dir: Thư mục lượt chạy.
            prompt: Prompt của người dùng.
            context: Metadata ngữ cảnh.

        Returns:
            Payload đã lưu.
        """
        payload = {"prompt": prompt, "context": context}
        self._write_json(run_dir / "req.json", payload)
        return payload

    def mark_success(self, run_dir: Path) -> None:
        """Đánh dấu lượt chạy thành công.

        Args:
            run_dir: Thư mục lượt chạy.
        """
        self._write_json(run_dir / "state.json", {"status": "success"})

    def mark_failure(self, run_dir: Path, reason: str) -> None:
        """Đánh dấu lượt chạy thất bại.

        Args:
            run_dir: Thư mục lượt chạy.
            reason: Lý do thất bại.
        """
        self._write_json(run_dir / "state.json", {"status": "failed", "reason": reason})

    def mark_cancelled(self, run_dir: Path, reason: str = "cancelled by user") -> None:
        """Đánh dấu lượt chạy đã bị người dùng hủy bỏ.

        Phân biệt rõ ràng với thất bại để artifact của lượt chạy khớp với
        transcript của phiên trò chuyện.

        Args:
            run_dir: Thư mục lượt chạy.
            reason: Lý do hủy bỏ.
        """
        self._write_json(run_dir / "state.json", {"status": "cancelled", "reason": reason})

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        # Ghi + fsync để sự cố crash không làm tệp state.json bị rỗng/cắt xén.
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        with open(path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

