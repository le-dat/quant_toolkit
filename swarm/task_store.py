"""Hệ thống đa agent Swarm — Lưu trữ tác vụ và các thuật toán DAG.

Mỗi tác vụ được lưu trữ độc lập dưới dạng tasks/task-{id}.json với hỗ trợ đầy đủ CRUD.
Cung cấp các thuật toán DAG: giải quyết phụ thuộc, phát hiện chu trình, và phân lớp tô-pô.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from pseud.swarm.models import SwarmTask, TaskStatus

_TRANSIENT_WINERRORS = (5, 32)
_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF = (0.025, 0.05, 0.1, 0.2, 0.4)


def _replace_with_retry(tmp: Path, target: Path) -> None:
    """Atomic replace with backoff retry for Windows transient file locks."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in _TRANSIENT_WINERRORS or attempt == _REPLACE_ATTEMPTS - 1:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                raise
            time.sleep(_REPLACE_BACKOFF[attempt])


class TaskStore:
    """Lớp lưu trữ dựa trên tệp cho các tác vụ.

    Mỗi tác vụ được lưu tại run_dir/tasks/task-{id}.json.

    Thuộc tính:
        run_dir: Thư mục gốc của lượt chạy hiện tại.
    """

    def __init__(self, run_dir: Path) -> None:
        """Khởi tạo TaskStore.

        Args:
            run_dir: Đường dẫn đến thư mục .swarm/runs/{run_id}/.
        """
        self.run_dir = run_dir
        self._tasks_dir = run_dir / "tasks"
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _task_path(self, task_id: str) -> Path:
        """Trả về đường dẫn tệp cho một tác vụ.

        Args:
            task_id: ID tác vụ.

        Returns:
            Đường dẫn đến tệp JSON tác vụ.
        """
        return self._tasks_dir / f"task-{task_id}.json"

    def save_task(self, task: SwarmTask) -> None:
        """Lưu hoặc ghi đè trạng thái tác vụ.

        Args:
            task: Instance SwarmTask.
        """
        path = self._task_path(task.id)
        tmp_path = path.with_suffix(".tmp")
        with self._lock:
            tmp_path.write_text(task.model_dump_json(indent=2), encoding="utf-8")
            _replace_with_retry(tmp_path, path)

    def load_task(self, task_id: str) -> SwarmTask:
        """Nạp một tác vụ theo ID.

        Args:
            task_id: ID tác vụ.

        Returns:
            Instance SwarmTask.

        Raises:
            FileNotFoundError: Nếu tệp tác vụ không tồn tại.
        """
        path = self._task_path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"Task not found: {path.name}")
        return SwarmTask.model_validate_json(path.read_text(encoding="utf-8"))

    def load_all(self) -> list[SwarmTask]:
        """Nạp tất cả các tác vụ cho lượt chạy hiện tại.

        Returns:
            Danh sách SwarmTask được sắp xếp theo ID.
        """
        tasks: list[SwarmTask] = []
        for path in sorted(self._tasks_dir.glob("task-*.json")):
            tasks.append(
                SwarmTask.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return tasks

    def update_status(
        self, task_id: str, status: TaskStatus, **kwargs: str | int | list[str] | None
    ) -> SwarmTask:
        """Cập nhật trạng thái tác vụ và các trường bổ sung.

        Args:
            task_id: ID tác vụ.
            status: Trạng thái mới.

        Returns:
            Instance SwarmTask đã cập nhật.
        """

        task = self.load_task(task_id)
        updated_data = task.model_dump()
        updated_data["status"] = status
        for key, value in kwargs.items():
            if key in updated_data:
                updated_data[key] = value
        updated_task = SwarmTask.model_validate(updated_data)
        self.save_task(updated_task)
        return updated_task


def resolve_dependencies(
    tasks_dir: Path, completed_task_id: str, lock: threading.Lock | None = None
) -> list[str]:
    """Loại bỏ completed_task_id khỏi danh sách blocked_by trong tất cả các tác vụ cấp dưới.

    Quét tất cả các tệp tác vụ trong tasks_dir và loại bỏ completed_task_id khỏi danh sách
    blocked_by của từng tác vụ. Nếu danh sách blocked_by của tác vụ trở nên rỗng và trạng thái
    của nó là blocked, nó được đánh dấu là mở khóa mới (pending).

    Args:
        tasks_dir: Đường dẫn tới thư mục tasks/.
        completed_task_id: ID của tác vụ vừa hoàn thành.
        lock: Khóa threading tùy chọn để đồng bộ hóa việc ghi tệp.

    Returns:
        Danh sách các ID tác vụ mới được mở khóa.
    """
    newly_unblocked: list[str] = []

    for path in tasks_dir.glob("task-*.json"):
        try:
            task = SwarmTask.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if completed_task_id not in task.blocked_by:
            continue

        new_blocked_by = [tid for tid in task.blocked_by if tid != completed_task_id]
        updated_data = task.model_dump()
        updated_data["blocked_by"] = new_blocked_by

        if not new_blocked_by and task.status == TaskStatus.blocked:
            updated_data["status"] = TaskStatus.pending
            newly_unblocked.append(task.id)

        updated_task = SwarmTask.model_validate(updated_data)
        tmp_path = path.with_suffix(".tmp")
        if lock is not None:
            with lock:
                tmp_path.write_text(updated_task.model_dump_json(indent=2), encoding="utf-8")
                _replace_with_retry(tmp_path, path)
        else:
            tmp_path.write_text(updated_task.model_dump_json(indent=2), encoding="utf-8")
            _replace_with_retry(tmp_path, path)

    return newly_unblocked


def validate_dag(tasks: list[SwarmTask]) -> None:
    """Duyệt DFS phát hiện chu trình để đảm bảo đồ thị tác vụ DAG không có chu trình.

    Args:
        tasks: Danh sách các SwarmTask.

    Raises:
        ValueError: Nếu phát hiện chu trình; thông báo chứa đường đi của chu trình.
    """
    graph: dict[str, list[str]] = {t.id: list(t.depends_on) for t in tasks}
    all_ids = {t.id for t in tasks}

    for task in tasks:
        for dep in task.depends_on:
            if dep not in all_ids:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown task '{dep}'"
                )

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {tid: WHITE for tid in all_ids}
    path: list[str] = []

    def dfs(node: str) -> None:
        """Duyệt DFS để phát hiện các cạnh ngược (back edges).

        Args:
            node: ID nút hiện tại.

        Raises:
            ValueError: Nếu phát hiện chu trình.
        """
        color[node] = GRAY
        path.append(node)

        for neighbor in graph.get(node, []):
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                raise ValueError(
                    f"Cycle detected in task DAG: {' -> '.join(cycle)}"
                )
            if color[neighbor] == WHITE:
                dfs(neighbor)

        path.pop()
        color[node] = BLACK

    for tid in all_ids:
        if color[tid] == WHITE:
            dfs(tid)


def topological_layers(tasks: list[SwarmTask]) -> list[list[str]]:
    """Phân tầng tô-pô theo thuật toán Kahn; các tác vụ trong cùng một tầng có thể chạy song song.

    Args:
        tasks: Danh sách SwarmTask (bắt buộc là đồ thị DAG hợp lệ không chu trình).

    Returns:
        Danh sách các tầng theo thứ tự thực thi; mỗi tầng chứa các ID tác vụ có thể chạy song song.

    Raises:
        ValueError: Nếu DAG chứa chu trình (không thể hoàn thành sắp xếp tô-pô).
    """
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    dependents: dict[str, list[str]] = defaultdict(list)

    for task in tasks:
        in_degree[task.id] = len(task.depends_on)
        for dep in task.depends_on:
            dependents[dep].append(task.id)

    queue: deque[str] = deque(
        tid for tid, deg in in_degree.items() if deg == 0
    )

    layers: list[list[str]] = []
    processed = 0

    while queue:
        layer: list[str] = list(queue)
        queue.clear()
        layers.append(layer)
        processed += len(layer)

        for tid in layer:
            for downstream in dependents[tid]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

    if processed != len(tasks):
        raise ValueError(
            f"DAG contains a cycle: processed {processed}/{len(tasks)} tasks"
        )

    return layers
