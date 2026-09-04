"""Tiện ích an toàn đường dẫn được sử dụng bởi các công cụ truy cập tệp.

Ba trợ thủ, ba mô hình đe dọa:
* `safe_path(p, workdir)` — Sandbox kiểm soát bởi công cụ.
* `safe_user_path(p)` — Tệp do người dùng cung cấp.
* `safe_document_path(p)` — Đầu vào cho bộ đọc tài liệu.
"""

from __future__ import annotations

from pathlib import Path

from pseud.config.accessor import get_env_config

_ALLOWED_FILE_ROOTS_ENV = "KAIROS_ALLOWED_FILE_ROOTS"
_ALLOWED_RUN_ROOTS_ENV = "KAIROS_ALLOWED_RUN_ROOTS"


def _rejects_unc(p: str) -> None:
    """Ném lỗi ValueError nếu `p` bắt đầu bằng tiền tố chia sẻ UNC."""
    if p.startswith("\\\\") or p.startswith("//"):
        raise ValueError(f"UNC paths are not allowed: {p!r}")


def safe_path(p: str, workdir: Path) -> Path:
    """Giải quyết `p` dưới `workdir` và đảm bảo nó nằm bên trong.

    Args:
        p: Đường dẫn do người dùng cung cấp (tương đối hoặc tuyệt đối).
        workdir: Gốc không gian làm việc.

    Returns:
        Đường dẫn tuyệt đối đã giải quyết bên trong `workdir`.

    Raises:
        ValueError: Nếu `p` sử dụng chia sẻ UNC, hoặc thoát khỏi `workdir`.
    """
    _rejects_unc(p)
    base = Path(workdir).resolve()
    expanded = Path(p).expanduser()
    if expanded.is_absolute():
        resolved = expanded.resolve()
    else:
        resolved = (base / p).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Path {p!r} escapes the workspace root") from exc
    return resolved


def _agent_root() -> Path:
    """Trả về thư mục gốc của gói agent."""
    return Path(__file__).resolve().parents[2]


def _configured_file_roots() -> list[Path]:
    """Trả về các thư mục gốc của tệp được cấu hình qua biến môi trường."""
    raw = get_env_config().api.kairos_allowed_file_roots
    roots: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        _rejects_unc(item)
        roots.append(Path(item).expanduser().resolve())
    return roots


def _default_file_roots() -> list[Path]:
    """Trả về các thư mục gốc mặc định cho các tệp người dùng tải lên/nhập vào."""
    from pseud.config.paths import get_runtime_root, legacy_runtime_roots

    cwd = Path.cwd().resolve()
    agent_root = _agent_root()
    runtime_root = get_runtime_root()
    roots = [
        agent_root / "uploads",
        agent_root / "runs",
        cwd / "uploads",
        cwd / "data",
        runtime_root / "uploads",
        runtime_root / "runs",
        runtime_root / "imports",
    ]
    for legacy in legacy_runtime_roots():
        roots += [legacy / "uploads", legacy / "imports"]
    return roots


def _default_run_roots() -> list[Path]:
    """Trả về thư mục gốc mặc định cho các thư mục chạy backtest/tool được tạo."""
    from pseud.config.paths import get_runtime_root, legacy_runtime_roots
    from pseud.swarm.store import swarm_runs_root

    cwd = Path.cwd().resolve()
    agent_root = _agent_root()
    runtime_root = get_runtime_root()
    roots = [
        agent_root / "runs",
        agent_root / ".swarm" / "runs",  # un-migrated legacy swarm runs
        swarm_runs_root(),
        cwd / "runs",
        runtime_root / "shadow_runs",
        runtime_root / "runs",
    ]
    for legacy in legacy_runtime_roots():
        roots += [legacy / "shadow_runs", legacy / "runs"]
    return roots


def allowed_file_roots() -> list[Path]:
    """Trả về tất cả các thư mục gốc được phép đọc tài liệu và tệp môi giới."""
    roots: list[Path] = []
    for root in [*_default_file_roots(), *_configured_file_roots()]:
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


_ALLOWED_WRITE_ROOTS_ENV = "KAIROS_ALLOWED_WRITE_ROOTS"


def allowed_write_roots() -> list[Path]:
    """Trả về tất cả các thư mục gốc được phép ghi và sửa tệp."""
    raw = get_env_config().api.kairos_allowed_write_roots
    configured: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        _rejects_unc(item)
        configured.append(Path(item).expanduser().resolve())

    from pseud.config.paths import get_runtime_root, legacy_runtime_roots

    cwd = Path.cwd().resolve()
    agent_root = _agent_root()
    runtime_root = get_runtime_root()
    defaults = [
        agent_root / "uploads",
        agent_root / "runs",
        cwd / "uploads",
        runtime_root / "uploads",
        runtime_root / "runs",
    ]
    for legacy in legacy_runtime_roots():
        defaults += [legacy / "uploads", legacy / "runs"]

    roots: list[Path] = []
    for root in [*defaults, *configured]:
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def resolve_safe_path(
    file_path: str,
    run_dir: str | None,
    allowed_roots: list[Path],
    *,
    purpose: str = "workspace",
) -> Path:
    """Phân giải đường dẫn tệp theo run_dir với cơ chế fallback về các thư mục gốc cho phép.

    Args:
        file_path: Đường dẫn tương đối hoặc tuyệt đối.
        run_dir: Thư mục lượt chạy tùy chọn để phân giải dựa theo.
        allowed_roots: Danh sách thư mục gốc cho phép mặc định.
        purpose: Tên ngữ cảnh cho thông báo lỗi.

    Returns:
        Đường dẫn Path tuyệt đối đã giải quyết.

    Raises:
        ValueError: Nếu đường dẫn không thể phân giải hoặc vượt quá phạm vi cho phép.
    """
    _rejects_unc(file_path)

    # Thử phân giải theo run_dir nếu được cung cấp
    if run_dir:
        try:
            run_root = safe_run_dir(run_dir)
        except ValueError as exc:
            # Nếu safe_run_dir thất bại, kiểm tra xem đường dẫn có nằm trong allowed_roots trước không
            candidate = Path(file_path).expanduser().resolve()
            for root in allowed_roots:
                if candidate.is_relative_to(root):
                    return candidate
            raise exc

        try:
            return safe_path(file_path, run_root)
        except ValueError as exc:
            # Fallback về allowed roots nếu kiểm tra safe_path thất bại
            candidate = Path(file_path).expanduser().resolve()
            for root in allowed_roots:
                if candidate.is_relative_to(root):
                    return candidate
            raise ValueError(
                f"Path {file_path!r} escapes run_dir {run_dir!r} and is not in allowed {purpose} roots."
            ) from exc

    # Nếu không có run_dir, đường dẫn bắt buộc phải nằm trong một trong các thư mục gốc cho phép
    candidate = Path(file_path).expanduser().resolve()
    for root in allowed_roots:
        if candidate.is_relative_to(root):
            return candidate

    raise ValueError(
        f"run_dir is required to write/edit {file_path!r}, or the path must resolve inside allowed {purpose} roots."
    )


def _allowed_run_roots() -> list[Path]:
    """Trả về tất cả thư mục gốc được phép cho công cụ dựa trên run_dir."""
    raw = get_env_config().api.kairos_allowed_run_roots
    configured: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        _rejects_unc(item)
        configured.append(Path(item).expanduser().resolve())

    roots: list[Path] = []
    for root in [*_default_run_roots(), *configured]:
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _import_candidate(p: str) -> Path:
    """Trả về đối tượng đường dẫn thực tế trên hệ thống tệp cho một đường dẫn nhập vào."""
    from pseud.config.paths import get_uploads_dir

    candidate = Path(p).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    parts = candidate.parts
    if parts and parts[0] == "uploads":
        return (get_uploads_dir() / Path(*parts[1:])).resolve()
    if len(parts) >= 2 and parts[0] == "agent" and parts[1] == "uploads":
        return (get_uploads_dir() / Path(*parts[2:])).resolve()
    return (Path.cwd() / candidate).resolve()


def _safe_import_path(p: str, *, purpose: str) -> Path:
    """Xác thực đường dẫn do người dùng cung cấp so với các thư mục gốc nhập cho phép.

    Args:
        p: Đường dẫn do người dùng cung cấp.
        purpose: Mục đích hiển thị trong thông báo lỗi.

    Returns:
        Đường dẫn tuyệt đối hợp lệ trong thư mục gốc cho phép.

    Raises:
        ValueError: Nếu đường dẫn là chia sẻ UNC hoặc vượt ngoài phạm vi cho phép.
    """
    _rejects_unc(p)
    resolved = _import_candidate(p)

    for root in allowed_file_roots():
        if resolved.is_relative_to(root):
            return resolved

    raise ValueError(
        f"Path {p!r} is outside allowed {purpose} roots. "
        f"Set {_ALLOWED_FILE_ROOTS_ENV} to add an import directory."
    )


def safe_user_path(p: str) -> Path:
    """Xác thực đường dẫn tệp tệp người dùng/môi giới.

    Args:
        p: Đường dẫn do người dùng cung cấp.

    Returns:
        Đường dẫn tuyệt đối nằm trong thư mục gốc cho phép.
    """
    return _safe_import_path(p, purpose="user-file")


def safe_document_path(p: str) -> Path:
    """Xác thực đường dẫn tệp cho bộ đọc tài liệu.

    Args:
        p: Đường dẫn tài liệu.

    Returns:
        Đường dẫn tuyệt đối nằm trong thư mục gốc cho phép.
    """
    return _safe_import_path(p, purpose="document")


def safe_run_dir(p: str) -> Path:
    """Xác thực thư mục lượt chạy dùng bởi các công cụ sinh mã.

    Args:
        p: Thư mục lượt chạy do người dùng/LLM cung cấp.

    Returns:
        Đường dẫn tuyệt đối nằm trong thư mục gốc lượt chạy cho phép.
    """
    _rejects_unc(p)
    resolved = Path(p).expanduser().resolve()

    for root in _allowed_run_roots():
        if resolved.is_relative_to(root):
            return resolved

    raise ValueError(
        f"run_dir {p!r} is outside allowed run roots. "
        f"Set {_ALLOWED_RUN_ROOTS_ENV} to add a run directory."
    )


def safe_run_id(run_id: str) -> Path:
    """Phân giải run_id thành thư mục lượt chạy hợp lệ hiện có.

    Args:
        run_id: Tên thư mục lượt chạy thuần túy.

    Returns:
        Đường dẫn thư mục lượt chạy hiện có dưới các thư mục gốc cho phép.
    """
    _rejects_unc(run_id)
    candidate = Path(run_id)
    if (
        not run_id.strip()
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"run_id {run_id!r} must be a bare run directory name")

    for root in _allowed_run_roots():
        resolved = (root / candidate.name).resolve()
        if resolved.is_relative_to(root) and resolved.is_dir():
            return resolved

    raise ValueError(f"run_id {run_id!r} was not found under allowed run roots")

