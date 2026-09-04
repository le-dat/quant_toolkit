"""Runtime điều phối DAG cho Swarm.

Bộ điều phối cốt lõi: lên lịch cho các worker theo từng lớp tô-pô (topological layer),
chạy song song trong cùng một lớp và chạy nối tiếp giữa các lớp. Quá trình thực thi chạy trong một luồng daemon nền với hỗ trợ hủy bỏ và callback sự kiện.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pseud.config.accessor import get_env_config
from pseud.config.schema import AgentConfig
from pseud.swarm.models import (
    RunStatus,
    SwarmAgentSpec,
    SwarmEvent,
    SwarmRun,
    SwarmTask,
    TaskStatus,
    WorkerResult,
    WorkerStatus,
)
from pseud.swarm.presets import build_run_from_preset
from pseud.swarm.store import SwarmStore
from pseud.swarm.task_store import (
    TaskStore,
    resolve_dependencies,
    topological_layers,
    validate_dag,
)
from pseud.tools.redaction import redact_internal_paths
from pseud.swarm.worker import run_worker

logger = logging.getLogger(__name__)


class SwarmRuntime:
    """Động cơ điều phối DAG cho Swarm.

    Quản lý toàn bộ vòng đời của một lượt chạy swarm: khởi tạo, lên lịch, thực thi và hủy bỏ. 
    Mỗi lượt chạy thực hiện trong một luồng daemon nền độc lập; các tác vụ trong một lớp 
    chạy song song qua ThreadPoolExecutor.

    Thuộc tính:
        _store: Lớp lưu trữ bền vững SwarmStore.
        _max_workers: Số worker tối đa chạy đồng thời trong ThreadPoolExecutor.
    """

    def __init__(
        self,
        store: SwarmStore,
        max_workers: int = 4,
        agent_config: AgentConfig | None = None,
    ) -> None:
        """Khởi tạo SwarmRuntime.

        Args:
            store: Instance SwarmStore để lưu trữ lượt chạy.
            max_workers: Số lượng thread worker tối đa.
            agent_config: Cấu hình agent tùy chọn mang các định nghĩa MCP server từ xa.
                Được tin cậy ở thời điểm khởi chạy/bởi người vận hành; không bao giờ 
                được lấy từ người gọi swarm. Chuyển tiếp đến mọi worker trên mỗi lượt 
                chạy để worker có thể tập hợp registry bao gồm các công cụ MCP từ xa. 
                ``None`` (mặc định) bảo toàn hành vi chỉ sử dụng công cụ cục bộ.
        """
        self._store = store
        self._max_workers = max_workers
        self._agent_config = agent_config
        self._cancel_events: dict[str, threading.Event] = {}
        self._live_callbacks: dict[str, Callable] = {}
        self._lock = threading.Lock()

    def start_run(
        self,
        preset_name: str,
        user_vars: dict[str, str],
        live_callback: Callable | None = None,
        include_shell_tools: bool = False,
    ) -> SwarmRun:
        """Bắt đầu một lượt chạy swarm. Trả về ngay lập tức, việc thực thi diễn ra ở chế độ nền.

        Args:
            preset_name: Tên preset YAML cần thực thi.
            user_vars: Các biến do người dùng cung cấp cho template prompt.
            live_callback: Callback tùy chọn được gọi cho mỗi sự kiện theo thời gian thực.
            include_shell_tools: Liệu các worker có được phép đăng ký công cụ shell hay không.

        Returns:
            Instance SwarmRun vừa tạo (trạng thái ban đầu là pending).

        Raises:
            FileNotFoundError: Nếu tệp preset không tồn tại.
            ValueError: Nếu kiểm tra DAG thất bại.
        """
        # Reap any previously running runs whose host process died without
        # finalizing them. Threshold is computed per-run from agent timeouts +
        # heartbeat interval (see SwarmStore.compute_stale_threshold), so a
        # legitimately slow long-running task is not killed.
        try:
            reaped = self._store.reap_stale_running_runs()
            if reaped:
                logger.info("Reaped %d stale swarm run(s): %s", len(reaped), reaped)
        except Exception:
            logger.warning("Stale-run reaper failed", exc_info=True)

        run = build_run_from_preset(preset_name, user_vars)
        validate_dag(run.tasks)

        # Capture which provider/model the run was launched against so the
        # serialized run.json carries enough context for cost audits and
        # post-hoc debugging. Read directly from the same env vars the
        # provider layer uses (src/providers/llm.py:136,195) — that way an
        # override applied via os.environ still shows up. Per-agent overrides
        # remain visible on SwarmAgentSpec.model_name.
        _cfg = get_env_config()
        run.provider = _cfg.llm.langchain_provider.strip().lower() or None
        run.model = _cfg.llm.langchain_model_name.strip() or None

        self._store.create_run(run)

        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[run.id] = cancel_event
            if live_callback is not None:
                self._live_callbacks[run.id] = live_callback

        thread = threading.Thread(
            target=self._execute_run,
            args=(run, cancel_event, include_shell_tools),
            name=f"swarm-{run.id}",
            daemon=True,
        )
        thread.start()

        return run

    def cancel_run(self, run_id: str) -> bool:
        """Phát tín hiệu hủy bỏ cho một lượt chạy swarm đang diễn ra.

        Args:
            run_id: ID của lượt chạy cần hủy.

        Returns:
            True nếu tín hiệu hủy đã được phát, False nếu không tìm thấy lượt chạy.
        """
        with self._lock:
            cancel_event = self._cancel_events.get(run_id)
        if cancel_event is None:
            return False
        cancel_event.set()
        return True

    def _emit_event(self, run_id: str, event: SwarmEvent) -> None:
        """Lưu trữ một sự kiện và chuyển tiếp đến live callback nếu được đăng ký.

        Args:
            run_id: Định danh lượt chạy.
            event: Sự kiện cần lưu trữ.
        """
        try:
            self._store.append_event(run_id, event)
        except Exception:
            logger.warning("Failed to persist event for run %s", run_id, exc_info=True)
        with self._lock:
            cb = self._live_callbacks.get(run_id)
        if cb is not None:
            try:
                cb(event)
            except Exception:
                logger.warning("Live callback failed for run %s", run_id, exc_info=True)

    def _make_event(
        self,
        event_type: str,
        agent_id: str | None = None,
        task_id: str | None = None,
        data: dict | None = None,
    ) -> SwarmEvent:
        """Tạo một SwarmEvent với dấu thời gian hiện tại.

        Args:
            event_type: Chuỗi loại sự kiện.
            agent_id: Định danh agent tùy chọn.
            task_id: Định danh tác vụ tùy chọn.
            data: Dữ liệu bổ sung tùy chọn.

        Returns:
            Instance SwarmEvent.
        """
        return SwarmEvent(
            type=event_type,
            agent_id=agent_id,
            task_id=task_id,
            data=data or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _execute_run(
        self,
        run: SwarmRun,
        cancel_event: threading.Event,
        include_shell_tools: bool = False,
    ) -> None:
        """Vòng lặp điều phối cốt lõi (chạy trong luồng nền).

        Các bước:
            1. Cập nhật trạng thái lượt chạy thành running
            2. Khởi tạo TaskStore, lưu tất cả tác vụ
            3. Tính toán các lớp tô-pô (topological layers)
            4. Cho từng lớp:
               a. Kiểm tra việc hủy bỏ
               b. Gửi tất cả tác vụ vào ThreadPoolExecutor
               c. Thu thập kết quả, giải quyết phụ thuộc, cập nhật store
            5. Cập nhật trạng thái lượt chạy thành completed/failed

        Args:
            run: SwarmRun cần thực thi.
            cancel_event: Sự kiện threading cho tín hiệu hủy bỏ.
            include_shell_tools: Liệu worker có được phép đăng ký công cụ shell hay không.
        """
        run_id = run.id
        run_dir = self._store.run_dir(run_id)

        # Đánh dấu là đang chạy
        run.status = RunStatus.running
        self._store.update_run(run)
        self._emit_event(run_id, self._make_event("run_started"))

        self._prefetch_grounding_data(run)

        # Khởi tạo task store
        task_store = TaskStore(run_dir)
        for task in run.tasks:
            task_store.save_task(task)

        # Xây dựng ánh xạ tra cứu agent
        agent_map: dict[str, SwarmAgentSpec] = {a.id: a for a in run.agents}

        # Render khối grounding một lần và chuyển tới mọi worker
        grounding_block = ""

        # Tính toán các lớp thực thi
        layers = topological_layers(run.tasks)
        task_summaries: dict[str, str] = {}
        all_succeeded = True

        try:
            for layer_idx, layer_task_ids in enumerate(layers):
                # Kiểm tra hủy bỏ giữa các lớp
                if cancel_event.is_set():
                    logger.info("Run %s cancelled at layer %d", run_id, layer_idx)
                    self._cancel_remaining_tasks(task_store, layer_task_ids, run.tasks)
                    all_succeeded = False
                    break

                self._emit_event(
                    run_id,
                    self._make_event(
                        "layer_started",
                        data={"layer": layer_idx, "tasks": layer_task_ids},
                    ),
                )

                # Thực thi song song tất cả các tác vụ trong lớp này
                layer_results = self._execute_layer(
                    run=run,
                    task_store=task_store,
                    agent_map=agent_map,
                    layer_task_ids=layer_task_ids,
                    task_summaries=task_summaries,
                    run_dir=run_dir,
                    cancel_event=cancel_event,
                    include_shell_tools=include_shell_tools,
                    grounding_block=grounding_block,
                )

                # Xử lý kết quả
                for tid, result in layer_results.items():
                    # Tích lũy số lượng token vào tổng lượt chạy
                    run.total_input_tokens += result.input_tokens
                    run.total_output_tokens += result.output_tokens

                    if result.status == "completed":
                        task_summaries[tid] = result.summary
                        now_iso = datetime.now(timezone.utc).isoformat()
                        task_store.update_status(
                            tid,
                            TaskStatus.completed,
                            summary=result.summary,
                            completed_at=now_iso,
                            artifacts=result.artifact_paths,
                            worker_iterations=result.iterations,
                        )
                        resolve_dependencies(run_dir / "tasks", tid)
                        self._emit_event(
                            run_id,
                            self._make_event(
                                "task_completed",
                                task_id=tid,
                                data={
                                    "status": result.status,
                                    "iterations": result.iterations,
                                    "input_tokens": result.input_tokens,
                                    "output_tokens": result.output_tokens,
                                },
                            ),
                        )
                    else:
                        all_succeeded = False
                        # `TaskStatus` không có ô cho `incomplete`, nhưng giá trị thô
                        # PHẢI sống sót: gộp "worker chạy xong mà không có sản phẩm" vào
                        # "worker chết" là xoá đúng thứ WorkerStatus.incomplete sinh ra để
                        # phân biệt (M-RS0 §1.1 #9, §1.2). Tầng trên đọc `worker_status`
                        # trên bản ghi và trong luồng sự kiện, không đọc `TaskStatus`.
                        worker_status = str(
                            getattr(result.status, "value", result.status) or ""
                        )
                        la_incomplete = worker_status == WorkerStatus.incomplete.value
                        task_store.update_status(
                            tid,
                            TaskStatus.failed,
                            error=redact_internal_paths(result.error)
                            or f"worker did not complete (status={worker_status})",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                            worker_iterations=result.iterations,
                            worker_status=worker_status,
                        )
                        self._emit_event(
                            run_id,
                            self._make_event(
                                "task_incomplete" if la_incomplete else "task_failed",
                                task_id=tid,
                                data={
                                    "error": redact_internal_paths(result.error),
                                    "worker_status": worker_status,
                                    "input_tokens": result.input_tokens,
                                    "output_tokens": result.output_tokens,
                                },
                            ),
                        )

                for tid in layer_task_ids:
                    if tid not in layer_results:
                        all_succeeded = False

                # Snapshot run.json tại ranh giới lớp
                self._sync_run_tasks_snapshot(run, task_store)

        except Exception as exc:
            logger.error("Run %s failed with exception", run_id, exc_info=True)
            all_succeeded = False
            self._emit_event(
                run_id,
                self._make_event("run_error", data={"error": redact_internal_paths(str(exc))}),
            )

        # Hoàn tất lượt chạy
        final_status = (
            RunStatus.cancelled if cancel_event.is_set() else RunStatus.completed if all_succeeded else RunStatus.failed
        )
        run.status = final_status
        run.completed_at = datetime.now(timezone.utc).isoformat()

        # Đồng bộ các tác vụ trở lại model lượt chạy
        run.tasks = task_store.load_all()

        # Đặt báo cáo cuối cùng từ tác vụ tổng hợp
        if task_summaries:
            last_layer = layers[-1] if layers else []
            for tid in last_layer:
                if tid in task_summaries:
                    run.final_report = task_summaries[tid]
                    break

        self._store.update_run(run)
        self._emit_event(run_id, self._make_event("run_completed", data={"status": final_status.value}))

        # Dọn dẹp sự kiện hủy bỏ và callback
        with self._lock:
            self._cancel_events.pop(run_id, None)
            self._live_callbacks.pop(run_id, None)

    def _sync_run_tasks_snapshot(self, run: SwarmRun, task_store: TaskStore) -> None:
        """Đồng bộ ``tasks/*.json`` trực tiếp trở lại ``run.json`` tại thời điểm an toàn."""
        try:
            run.tasks = task_store.load_all()
            self._store.update_run(run)
        except Exception:
            logger.warning("Layer-boundary run.json sync failed", exc_info=True)

    def _prefetch_grounding_data(self, run: SwarmRun) -> None:
        """Tải trước dữ liệu grounding ở cấp lượt chạy mà không làm nghẽn ``start_run``."""
        return

    def _execute_layer(
        self,
        run: SwarmRun,
        task_store: TaskStore,
        agent_map: dict[str, SwarmAgentSpec],
        layer_task_ids: list[str],
        task_summaries: dict[str, str],
        run_dir: Path,
        cancel_event: threading.Event,
        include_shell_tools: bool = False,
        grounding_block: str = "",
    ) -> dict[str, WorkerResult]:
        """Thực thi song song tất cả các tác vụ trong một lớp đơn lẻ, có thử lại khi thất bại.

        Args:
            run: SwarmRun đang được thực thi.
            task_store: TaskStore để lưu trữ tác vụ.
            agent_map: Thông số kỹ thuật agent theo agent_id.
            layer_task_ids: Danh sách ID tác vụ trong lớp này.
            task_summaries: Tóm tắt tác vụ tích lũy từ các lớp trước.
            run_dir: Đường dẫn thư mục lượt chạy.
            cancel_event: Sự kiện hủy bỏ.
            include_shell_tools: Liệu worker có được phép đăng ký công cụ shell hay không.
            grounding_block: Markdown "Ground Truth" đã được chuẩn bị sẵn cho các worker.

        Returns:
            Ánh xạ task_id -> WorkerResult cho tất cả tác vụ trong lớp này.
        """
        results: dict[str, WorkerResult] = {}

        def _event_callback(event: SwarmEvent) -> None:
            self._emit_event(run.id, event)

        executor = ThreadPoolExecutor(max_workers=self._max_workers)
        futures: dict[Future[WorkerResult], str] = {}
        layer_budget = 0
        try:
            for tid in layer_task_ids:
                task = task_store.load_task(tid)

                blocked_upstreams: list[tuple[str, str]] = []
                for dep_id in task.depends_on:
                    try:
                        dep_task = task_store.load_task(dep_id)
                    except FileNotFoundError:
                        blocked_upstreams.append((dep_id, "missing"))
                        continue
                    if dep_task.status != TaskStatus.completed:
                        blocked_upstreams.append((dep_id, dep_task.status.value))

                if blocked_upstreams:
                    reason = ", ".join(f"{d}={s}" for d, s in blocked_upstreams)
                    blocked_by_ids = [d for d, _ in blocked_upstreams]
                    task_store.update_status(
                        tid,
                        TaskStatus.blocked,
                        error=f"Blocked: upstream not completed ({reason})",
                        blocked_by=blocked_by_ids,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    self._emit_event(
                        run.id,
                        self._make_event(
                            "task_blocked",
                            agent_id=task.agent_id,
                            task_id=tid,
                            data={"blocked_by": blocked_by_ids, "reason": reason},
                        ),
                    )
                    continue

                agent_spec = agent_map.get(task.agent_id)
                if agent_spec is None:
                    results[tid] = WorkerResult(
                        status="failed",
                        summary="",
                        error=f"Agent '{task.agent_id}' not found in preset",
                    )
                    continue

                # Đánh dấu tác vụ là in_progress
                task_store.update_status(
                    tid,
                    TaskStatus.in_progress,
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
                self._emit_event(
                    run.id,
                    self._make_event("task_started", agent_id=agent_spec.id, task_id=tid),
                )

                # Xây dựng tóm tắt cấp trên từ ánh xạ input_from
                upstream: dict[str, str] = {}
                for context_key, source_task_id in task.input_from.items():
                    if source_task_id in task_summaries:
                        upstream[context_key] = task_summaries[source_task_id]

                future = executor.submit(
                    self._run_worker_with_retries,
                    agent_spec=agent_spec,
                    task=task,
                    upstream_summaries=upstream,
                    user_vars=run.user_vars,
                    run_dir=run_dir,
                    event_callback=_event_callback,
                    run_id=run.id,
                    include_shell_tools=include_shell_tools,
                    grounding_block=grounding_block,
                )
                futures[future] = tid
                per_task_budget = agent_spec.timeout_seconds * (agent_spec.max_retries + 1)
                layer_budget = max(layer_budget, per_task_budget)

            deadline_buffer = 60
            layer_deadline = layer_budget + deadline_buffer if layer_budget else None

            try:
                for future in as_completed(futures, timeout=layer_deadline):
                    tid = futures[future]
                    try:
                        results[tid] = future.result()
                    except Exception as exc:
                        logger.error("Worker for task %s raised exception", tid, exc_info=True)
                        results[tid] = WorkerResult(
                            status="failed",
                            summary="",
                            error=str(exc),
                        )
            except FuturesTimeoutError:
                for pending, tid in futures.items():
                    if tid in results:
                        continue
                    pending.cancel()
                    logger.error(
                        "Worker for task %s exceeded layer deadline (%ds)",
                        tid,
                        layer_deadline,
                    )
                    results[tid] = WorkerResult(
                        status="timeout",
                        summary="",
                        error=f"Worker exceeded layer deadline of {layer_deadline}s",
                    )
        except KeyboardInterrupt:
            cancel_event.set()
            logger.warning("Swarm layer interrupted — cancelling pending workers")
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return results

    def _run_worker_with_retries(
        self,
        agent_spec: SwarmAgentSpec,
        task: SwarmTask,
        upstream_summaries: dict[str, str],
        user_vars: dict[str, str],
        run_dir: Path,
        event_callback: Callable[[SwarmEvent], None] | None,
        run_id: str,
        include_shell_tools: bool = False,
        grounding_block: str = "",
    ) -> WorkerResult:
        """Chạy một worker với cơ chế tự động thử lại khi gặp lỗi.

        Args:
            agent_spec: Đặc tả vai trò agent.
            task: Tác vụ cần thực thi.
            upstream_summaries: Tóm tắt từ các tác vụ cấp trên.
            user_vars: Biến template do người dùng cung cấp.
            run_dir: Đường dẫn thư mục lượt chạy.
            event_callback: Callback sự kiện tùy chọn.
            run_id: Định danh lượt chạy để phát sự kiện.
            include_shell_tools: Liệu worker có được phép đăng ký công cụ shell hay không.
            grounding_block: Markdown "Ground Truth" được ghép vào system prompt của worker.

        Returns:
            WorkerResult từ lần thử cuối cùng.
        """
        max_retries = agent_spec.max_retries
        cumulative_input_tokens = 0
        cumulative_output_tokens = 0
        result: WorkerResult | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                self._emit_event(
                    run_id,
                    self._make_event(
                        "task_retry",
                        agent_id=agent_spec.id,
                        task_id=task.id,
                        data={
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "previous_error": result.error if result else None,
                        },
                    ),
                )
                logger.info(
                    "Retrying task %s (attempt %d/%d)",
                    task.id,
                    attempt + 1,
                    max_retries + 1,
                )

            result = run_worker(
                agent_spec=agent_spec,
                task=task,
                upstream_summaries=upstream_summaries,
                user_vars=user_vars,
                run_dir=run_dir,
                event_callback=event_callback,
                include_shell_tools=include_shell_tools,
                grounding_block=grounding_block,
                agent_config=self._agent_config,
            )

            cumulative_input_tokens += result.input_tokens
            cumulative_output_tokens += result.output_tokens

            if result.status != "failed":
                result = result.model_copy(
                    update={
                        "input_tokens": cumulative_input_tokens,
                        "output_tokens": cumulative_output_tokens,
                    }
                )
                return result

        if result is not None:
            result = result.model_copy(
                update={
                    "input_tokens": cumulative_input_tokens,
                    "output_tokens": cumulative_output_tokens,
                }
            )
        return result  # type: ignore[return-value]

    def _cancel_remaining_tasks(
        self,
        task_store: TaskStore,
        current_layer_ids: list[str],
        all_tasks: list[SwarmTask],
    ) -> None:
        """Đánh dấu tất cả các tác vụ chưa hoàn thành là bị hủy.

        Args:
            task_store: TaskStore lưu trữ dữ liệu.
            current_layer_ids: Danh sách ID tác vụ trong lớp hiện tại (bị ngắt câu).
            all_tasks: Tất cả các tác vụ trong lượt chạy.
        """
        for task in all_tasks:
            if task.status not in (TaskStatus.completed, TaskStatus.failed):
                try:
                    task_store.update_status(task.id, TaskStatus.cancelled)
                except Exception:
                    logger.warning("Failed to cancel task %s", task.id, exc_info=True)

