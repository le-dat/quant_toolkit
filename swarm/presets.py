"""Bộ nạp preset YAML cho Swarm.

Đọc các tệp preset YAML từ thư mục ``presets/`` đóng gói sẵn trong mô-đun này
và phân tích thành các mô hình dữ liệu SwarmRun / SwarmAgentSpec / SwarmTask.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter

import yaml

from pseud.config.paths import get_runtime_root
from pseud.swarm.models import RunStatus, SwarmAgentSpec, SwarmRun, SwarmTask, TaskStatus
from pseud.swarm.task_store import topological_layers, validate_dag

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
#: Các preset do người dùng tạo; được tìm kiếm trước thư mục đóng gói sẵn để tệp người dùng
#: có thể thêm vào và ghi đè các preset có sẵn theo tên.
USER_PRESETS_DIR = get_runtime_root() / "swarm" / "presets"
_INTERNAL_TEMPLATE_VARS = {"upstream_context"}


def _redact_home(path: Path) -> str:
    """Hiển thị ``path`` với tiền tố home được thu gọn thành ``~`` để các lỗi
    dành cho người dùng không bị rò rỉ đường dẫn tuyệt đối (CWE-209)."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _validate_preset_name(name: str) -> str:
    """Từ chối các tên rỗng hoặc có thể thoát khỏi thư mục presets.

    Returns:
        Tên đã loại bỏ khoảng trắng dư thừa.

    Raises:
        ValueError: Tên rỗng, chứa ký tự phân cách đường dẫn, hoặc tham chiếu thư mục cha.
    """
    cleaned = (name or "").strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"invalid preset name: {name!r}")
    return cleaned


def _preset_search_dirs() -> tuple[Path, ...]:
    """Các thư mục được tìm kiếm theo thứ tự ưu tiên từ cao xuống thấp."""
    return (USER_PRESETS_DIR, PRESETS_DIR)


def resolve_preset_path(name: str) -> Path | None:
    """Trả về đường dẫn tệp YAML cho *name* (thư mục người dùng trước), hoặc ``None``."""
    cleaned = _validate_preset_name(name)
    for directory in _preset_search_dirs():
        path = directory / f"{cleaned}.yaml"
        if path.is_file():
            return path
    return None


def load_preset(name: str) -> dict:
    """Nạp một preset YAML theo tên (thư mục người dùng trước, sau đó tới thư mục đóng gói).

    Args:
        name: Tên preset (không kèm phần mở rộng .yaml).

    Returns:
        Dict đại diện cho YAML đã phân tích.

    Raises:
        ValueError: Nếu tên rỗng hoặc chứa ký tự phân cách đường dẫn.
        FileNotFoundError: Nếu tệp preset không tồn tại ở cả thư mục người dùng
            và thư mục đóng gói.
    """
    path = resolve_preset_path(name)
    if path is None:
        available = sorted({
            p.stem
            for directory in _preset_search_dirs()

            if directory.exists()
            for p in directory.glob("*.yaml")
        })
        raise FileNotFoundError(
            f"Preset {name!r} not found in {_redact_home(USER_PRESETS_DIR)} or "
            f"the bundled presets. Available: {available}"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def list_presets() -> list[dict]:
    """Trả về thông tin tóm tắt cho tất cả các preset khả dụng, sắp xếp theo tên.

    Các preset của người dùng (``<gốc runtime>/swarm/presets/``) được liệt kê bên cạnh
    danh sách mặc định đóng gói sẵn; khi cả hai thư mục chứa cùng một tên tệp, preset của người dùng
    sẽ được ưu tiên — cùng quy tắc ghi đè như các kỹ năng (skills) của người dùng.

    Returns:
        Danh sách dict chứa các khóa: name, title, description, agent_count,
        variables, source (``"user"`` hoặc ``"bundled"``).
    """
    by_stem: dict[str, dict] = {}
    for directory, source in ((PRESETS_DIR, "bundled"), (USER_PRESETS_DIR, "user")):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            # Lần lặp sau (user) cố ý ghi đè tên tệp đóng gói sẵn.
            by_stem[path.stem] = {
                "name": data.get("name", path.stem),
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "agent_count": len(data.get("agents", [])),
                "variables": data.get("variables", []),
                "keywords": data.get("keywords", []),
                "weight_boost": data.get("weight_boost", 1.0),
                "default_variables": data.get("default_variables", {}),
                "source": source,
            }

    return sorted(by_stem.values(), key=lambda entry: entry["name"])


def _declared_variable_names(raw_variables: list) -> set[str]:
    """Trích xuất tên các biến từ phần variables của tệp YAML."""
    names: set[str] = set()
    for item in raw_variables:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = str(item)
        if name:
            names.add(str(name))
    return names


def _template_variables(template: str) -> set[str]:
    """Trả về các trường định dạng Python được tham chiếu bởi một mẫu prompt."""
    variables: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template or ""):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root and root not in _INTERNAL_TEMPLATE_VARS:
            variables.add(root)
    return variables


def inspect_preset(name: str) -> dict:
    """Xác thực một preset swarm và trả về kế hoạch thực thi thử nghiệm (dry-run).

    Hàm này không khởi chạy worker hay gọi LLM. Nó phát hiện các sai sót
    YAML/DAG phổ biến từ sớm và hiển thị các lớp tác vụ tô-pô được sử dụng bởi runtime.
    """
    data = load_preset(name)
    run = build_run_from_preset(name, {})

    errors: list[str] = []
    warnings: list[str] = []

    agent_ids = [agent.id for agent in run.agents]
    task_ids = [task.id for task in run.tasks]
    agent_id_set = set(agent_ids)
    task_id_set = set(task_ids)

    for duplicate in sorted(item for item, count in Counter(agent_ids).items() if count > 1):
        errors.append(f"Duplicate agent id: {duplicate}")
    for duplicate in sorted(item for item, count in Counter(task_ids).items() if count > 1):
        errors.append(f"Duplicate task id: {duplicate}")

    for task in run.tasks:
        if task.agent_id not in agent_id_set:
            errors.append(f"Task '{task.id}' references unknown agent '{task.agent_id}'")
        for _, upstream_task_id in task.input_from.items():
            if upstream_task_id not in task_id_set:
                errors.append(
                    f"Task '{task.id}' input_from references unknown task '{upstream_task_id}'"
                )

    layers: list[list[str]] = []
    try:
        validate_dag(run.tasks)
        layers = topological_layers(run.tasks)
    except ValueError as exc:
        errors.append(str(exc))

    dependents: dict[str, list[str]] = defaultdict(list)
    for task in run.tasks:
        for dep in task.depends_on:
            dependents[dep].append(task.id)

    def is_upstream(candidate: str, task_id: str) -> bool:
        """Trả về việc candidate có thể tiếp cận task_id qua các cạnh phụ thuộc hay không."""
        seen: set[str] = set()
        stack = [candidate]
        while stack:
            current = stack.pop()
            if current == task_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(dependents.get(current, []))
        return False

    for task in run.tasks:
        for key, upstream_task_id in task.input_from.items():
            if upstream_task_id in task_id_set and not is_upstream(upstream_task_id, task.id):
                warnings.append(
                    f"Task '{task.id}' input_from '{key}' references '{upstream_task_id}', "
                    "which is not upstream in the DAG"
                )

    declared_variables = _declared_variable_names(data.get("variables", []))
    used_variables: set[str] = set()
    for task in data.get("tasks", []):
        try:
            used_variables.update(_template_variables(task.get("prompt_template", "")))
        except ValueError as exc:
            errors.append(f"Task '{task.get('id', '?')}' has invalid prompt template: {exc}")

    missing_declarations = sorted(used_variables - declared_variables)
    unused_declarations = sorted(declared_variables - used_variables)
    if missing_declarations:
        warnings.append(
            "Prompt templates use undeclared variables: " + ", ".join(missing_declarations)
        )
    if unused_declarations:
        warnings.append(
            "Declared variables are not used by task prompt templates: "
            + ", ".join(unused_declarations)
        )

    task_agent = {task.id: task.agent_id for task in run.tasks}
    return {
        "name": data.get("name", name),
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "variables": sorted(declared_variables),
        "used_variables": sorted(used_variables),
        "agents": [
            {"id": agent.id, "role": agent.role, "tools": agent.tools, "skills": agent.skills}
            for agent in run.agents
        ],
        "tasks": [
            {
                "id": task.id,
                "agent_id": task.agent_id,
                "depends_on": task.depends_on,
                "input_from": task.input_from,
            }
            for task in run.tasks
        ],
        "layers": [
            [{"task_id": task_id, "agent_id": task_agent.get(task_id, "")} for task_id in layer]
            for layer in layers
        ],
    }


def build_run_from_preset(preset_name: str, user_vars: dict[str, str]) -> SwarmRun:
    """Tạo một SwarmRun từ một preset với các biến người dùng được áp dụng.

    Các bước:
        1. Nạp preset YAML
        2. Tạo danh sách SwarmAgentSpec từ phần agents
        3. Tạo danh sách SwarmTask từ phần tasks
        4. Tạo run_id: f"swarm-{datetime}-{uuid[:8]}"
        5. Trả về SwarmRun với đầy đủ các trường dữ liệu

    Args:
        preset_name: Tên của preset cần nạp.
        user_vars: Các biến do người dùng cung cấp để render template prompt.

    Returns:
        Instance SwarmRun hoàn chỉnh (trạng thái ban đầu pending).

    Raises:
        FileNotFoundError: Nếu preset không tồn tại.
        ValueError: Nếu YAML của preset không hợp lệ.
    """
    data = load_preset(preset_name)

    # Phân tích danh sách agent
    agents: list[SwarmAgentSpec] = []
    for agent_data in data.get("agents", []):
        agents.append(SwarmAgentSpec(
            id=agent_data["id"],
            role=agent_data.get("role", ""),
            system_prompt=agent_data.get("system_prompt", ""),
            tools=agent_data.get("tools", []),
            skills=agent_data.get("skills", []),
            max_iterations=agent_data.get("max_iterations", 25),
            timeout_seconds=agent_data.get("timeout_seconds", 300),
            model_name=agent_data.get("model_name"),
            max_retries=agent_data.get("max_retries", 2),
        ))

    # Phân tích tác vụ, khởi tạo blocked_by từ depends_on
    tasks: list[SwarmTask] = []
    for task_data in data.get("tasks", []):
        depends_on = task_data.get("depends_on", [])
        status = TaskStatus.blocked if depends_on else TaskStatus.pending
        tasks.append(SwarmTask(
            id=task_data["id"],
            agent_id=task_data["agent_id"],
            prompt_template=task_data.get("prompt_template", ""),
            depends_on=depends_on,
            blocked_by=list(depends_on),
            input_from=task_data.get("input_from", {}),
            status=status,
        ))

    # Tạo ID lượt chạy
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    run_id = f"swarm-{ts}-{short_uuid}"

    return SwarmRun(
        id=run_id,
        preset_name=preset_name,
        status=RunStatus.pending,
        user_vars=user_vars,
        agents=agents,
        tasks=tasks,
        created_at=now.isoformat(),
    )
