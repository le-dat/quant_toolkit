"""Swarm Worker: Động cơ thực thi worker độc lập với vòng lặp ReAct tinh gọn.

Sử dụng ChatLLM.chat + vòng lặp trực tiếp (không khởi tạo AgentLoop),
giữ cho worker tự chứa và phần lõi agent không thay đổi.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pseud.agent.context import ContextBuilder
from pseud.agent.progress import HeartbeatTimer
from pseud.agent.skills import SkillsLoader
from pseud.agent.tools import ToolRegistry
from pseud.config.schema import AgentConfig
from pseud.providers.chat import ChatLLM, ChatResponse, ProviderStreamError
from pseud.providers.content_filter import (
    CONTENT_FILTER_SKIP_MESSAGE,
    MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS,
    compute_content_filter_warnings,
)
from pseud.swarm.models import (
    SwarmAgentSpec,
    SwarmEvent,
    SwarmTask,
    WorkerResult,
)
from pseud.tools import build_swarm_registry
from pseud.tools.redaction import is_sensitive_arg, redact_payload, redact_tool_result

logger = logging.getLogger(__name__)


def _default_max_iterations() -> int:
    """Trả về số vòng lặp tối đa mặc định cho swarm worker từ cấu hình môi trường."""
    from pseud.config.accessor import get_env_config
    return get_env_config().swarm.swarm_worker_max_iter


def _default_timeout_seconds() -> int:
    """Trả về thời gian chờ (timeout) tính bằng giây mặc định cho swarm worker."""
    from pseud.config.accessor import get_env_config
    return get_env_config().swarm.swarm_worker_timeout


def _heartbeat_interval_s() -> float:
    """Xác định khoảng thời gian gửi heartbeat (tính bằng giây) từ cấu hình môi trường."""
    from pseud.config.accessor import get_env_config
    return get_env_config().swarm.swarm_heartbeat_interval_s


def _stream_retry_delay_s() -> float:
    """Xác định thời gian chờ trước khi thử lại cuộc gọi stream.

    Returns:
        Số giây tạm dừng giữa các lần thử lại cuộc gọi stream_chat bị lỗi.
    """
    from pseud.config.accessor import get_env_config
    return get_env_config().swarm.swarm_stream_retry_delay_s


_HEARTBEAT_INTERVAL_S = _heartbeat_interval_s()
_STREAM_RETRY_DELAY_S = _stream_retry_delay_s()
_MAX_TOKEN_ESTIMATE = 60_000


def _microcompact(messages: list) -> None:
    """Tắt âm thầm các kết quả công cụ cũ khi gặp áp lực token (giữ lại 3 kết quả mới nhất)."""
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    if len(tool_msgs) <= 3:
        return
    for msg in tool_msgs[:-3]:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 100:
            msg["content"] = "[cleared]"


def _emit(
    callback: Callable[[SwarmEvent], None] | None,
    event_type: str,
    agent_id: str,
    task_id: str,
    data: dict | None = None,
) -> None:
    """Phát một sự kiện swarm thông qua callback nếu được cung cấp.

    Args:
        callback: Hàm callback nhận sự kiện tùy chọn.
        event_type: Tên chuỗi loại sự kiện.
        agent_id: Định danh của agent.
        task_id: Định danh của tác vụ.
        data: Dữ liệu bổ sung đi kèm sự kiện.
    """
    if callback is None:
        return
    event = SwarmEvent(
        type=event_type,
        agent_id=agent_id,
        task_id=task_id,
        data=data or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    try:
        callback(event)
    except Exception:
        logger.warning("Event callback failed for %s", event_type, exc_info=True)


def _filter_skill_descriptions(loader: SkillsLoader, skill_names: list[str]) -> str:
    """Trả về mô tả kỹ năng đã được lọc theo danh sách trắng (whitelist).

    Args:
        loader: Đối tượng SkillsLoader chứa tất cả kỹ năng hiện có.
        skill_names: Danh sách tên các kỹ năng cần bao gồm.

    Returns:
        Chuỗi mô tả các kỹ năng đã định dạng.
    """
    if not skill_names:
        return loader.get_descriptions()
    lines: list[str] = []
    for skill in loader.skills:
        if skill.name in skill_names:
            lines.append(f"  - {skill.name}: {skill.description}")
    return "\n".join(lines) if lines else "(no matching skills)"


def _estimate_tokens(
    messages: list[dict],
    response: object,
) -> tuple[int, int]:
    """Ước tính số lượng token tiêu tốn cho một lần gọi LLM đơn lẻ.

    Args:
        messages: Danh sách các tin nhắn gửi tới LLM.
        response: Phản hồi trả về từ ChatLLM.chat hoặc ChatLLM.stream_chat.

    Returns:
        Tuple chứa (input_tokens, output_tokens).
    """
    from pseud.providers.chat import ChatResponse

    if isinstance(response, ChatResponse) and response.usage_metadata:
        usage = response.usage_metadata
        real_input = int(usage.get("input_tokens") or 0)
        real_output = int(usage.get("output_tokens") or 0)
        if real_input or real_output:
            return real_input, real_output

    try:
        input_tokens = len(json.dumps(messages, ensure_ascii=False)) // 4
    except Exception:
        input_tokens = 0

    if isinstance(response, ChatResponse):
        output_tokens = len(response.content or "") // 4
    else:
        output_tokens = 0

    return input_tokens, output_tokens


def build_worker_prompt(
    agent_spec: SwarmAgentSpec,
    upstream_summaries: dict[str, str],
    skill_descriptions: str,
    grounding_block: str = "",
) -> str:
    """Xây dựng system prompt tổng quát cho worker với vai trò, ngữ cảnh cấp trên và kỹ năng.

    Args:
        agent_spec: Đặc tả vai trò và cấu hình của agent.
        upstream_summaries: Ánh xạ context_key -> tóm tắt tác vụ cấp trên.
        skill_descriptions: Văn bản mô tả danh sách kỹ năng đã lọc.
        grounding_block: Khối thông tin "Ground Truth" định dạng markdown (tùy chọn).

    Returns:
        Chuỗi system prompt hoàn chỉnh cho worker LLM.
    """

    upstream_block = ""
    if upstream_summaries:
        sections = []
        for key, summary in upstream_summaries.items():
            sections.append(f"### {key}\n{summary}")
        upstream_block = (
            "## Upstream Context (from previous agents)\n\n"
            + "\n\n".join(sections)
        )

    prompt_parts = [
        f"## Role\n\n{agent_spec.role}",
        agent_spec.system_prompt.replace("{upstream_context}", upstream_block),
    ]

    if skill_descriptions and skill_descriptions != "(no matching skills)":
        prompt_parts.append(
            f"## Available Skills (use load_skill to access full documentation)\n\n{skill_descriptions}"
        )

    if grounding_block:
        # Đặt trước Execution Rules để nằm trong phạm vi khi worker lập kế hoạch gọi tool đầu tiên.
        prompt_parts.append(grounding_block)

    # Quy tắc chống bịa đặt dữ liệu toàn cục. Khối grounding_block mang chỉ dẫn
    # tương tự nhưng chỉ hiển thị khi user_vars cung cấp các biến định danh rõ ràng.
    # Các prompt định dạng tự do nếu không có rào chắn sẽ khiến worker
    # dẫn tới việc trích dẫn thông tin số liệu từ dữ liệu huấn luyện. Khối này áp dụng
    # quy tắc vô điều kiện — bao gồm cả các agent tổng hợp / biên tập không có công cụ dữ liệu.
    prompt_parts.append(
        "## Data Citation Discipline (HARD RULE)\n\n"
        "Every specific number you cite in your output — values, percentages, "
        "statistics, metrics, measurements — MUST be cited by evidence cell id in the "
        "form [[cell_id]] (e.g. [[e:run1:call1:data.ic]]) whenever fetched from a tool.\n\n"
        "Bare uncited numbers are forbidden (except years like 19xx/20xx or values "
        "explicitly flagged approximate with `~`). Never recall a number from model "
        "memory. Every cited number MUST be traceable to one of:\n"
        "  (a) a tool call result obtained in THIS run via [[cell_id]],\n"
        "  (b) the Ground Truth block above (if present),\n"
        "  (c) the Upstream Context above (if present and the upstream agent "
        "itself sourced it from (a) or (b)).\n\n"
        "If you cannot back a number with (a), (b), or (c), you have two "
        "choices:\n"
        "  - call a data tool to fetch it (preferred), or\n"
        "  - omit the number and qualify the statement (e.g. \"directional "
        "only — not verified against live data\").\n\n"
        "This rule applies equally to synthesis / aggregator / editor roles "
        "that lack data tools. If upstream did not provide a specific number, "
        "do NOT introduce one from training data — say the upstream omitted "
        "it and proceed without."
    )

    max_iters = getattr(agent_spec, "max_iterations", 25)
    timeout_s = getattr(agent_spec, "timeout_seconds", 300)
    exec_iters = max(1, max_iters - 3)
    # Lời khuyên `load_skill` chỉ được xuất hiện khi agent THẬT SỰ có công cụ đó. Khuyên một
    # agent gọi công cụ ngoài danh sách trắng của nó là đúng dạng hỏng M-RS0 §1.2 đã nêu:
    # agent gọi, nhận lỗi, lặp tới hết `max_iterations` mà không sinh sản phẩm nào.
    dong_skill = (
        "- `load_skill` first when a skill covers the work; the skills carry the measured numbers.\n"
        if "load_skill" in (agent_spec.tools or [])
        else ""
    )
    prompt_parts.append(
        "## Execution Rules\n\n"
        f"You have a HARD LIMIT of {max_iters} tool calls (timeout {timeout_s}s). After that you will be cut off. Work efficiently.\n\n"
        "**Phase 1 — Plan (0 tool calls):** State plan in 3-5 bullet points focusing on clear objectives, methodology, and tools needed.\n\n"
        f"**Phase 2 — Execute (≤{exec_iters} tool calls):\n"
        "- Execute step-by-step using the tools listed above. Those are the ONLY tools you have.\n"
        "- Do NOT attempt to run code, shell commands, or scripts. This layer does not execute generated code.\n"
        "- Do NOT fetch external data. Every figure must come from a tool call made in THIS run.\n"
        + dong_skill +
        "- A tool answering `\"status\": \"ok\"` has not necessarily measured anything. Some tools return\n"
        "  fixed placeholder constants. Check `tool-surface-and-limits` before citing any tool output,\n"
        "  and treat a top-level `status: error` / `ok: false` as a real failure.\n"
        "- If a tool call fails, read the error and adjust. Do not retry the same call unchanged.\n\n"
        "**Phase 3 — Summarize (MUST use write_file):**\n"
        "- You MUST call `write_file` with path `report.md` to save your final report.\n"
        "- This is REQUIRED, not optional. Your final response MUST include a write_file call for report.md.\n"
        "- Paths are relative to your run directory; `report.md` is correct, absolute paths are not.\n"
        "- The report must include clear findings, methodology, and actionable conclusions.\n"
        "- Reporting that you found nothing, with the reason, is a valid and valued result.\n"
        "  Inventing a finding to fill the report is a far more serious failure.\n"
        "- After writing report.md, output a brief 2-3 sentence summary in your text response.\n"
        "- Write report.md and your text response in English, whatever language the task prompt uses."
    )

    now = datetime.now(timezone.utc)
    prompt_parts.append(
        f"## Current Date & Time\n\n"
        f"Today is {now.strftime('%A, %B %d, %Y %H:%M UTC')}"
    )

    return "\n\n".join(prompt_parts)


def run_worker(
    agent_spec: SwarmAgentSpec,
    task: SwarmTask,
    upstream_summaries: dict[str, str],
    user_vars: dict[str, str],
    run_dir: Path,
    event_callback: Callable[[SwarmEvent], None] | None = None,
    include_shell_tools: bool = False,
    grounding_block: str = "",
    agent_config: AgentConfig | None = None,
) -> WorkerResult:
    """Thực thi một tác vụ worker đơn lẻ bằng vòng lặp ReAct tinh gọn.

    Các bước thực hiện:
      1. Khởi tạo ToolRegistry đã lọc từ agent_spec.tools
      2. Khởi tạo ChatLLM với agent_spec.model_name
      3. Xây dựng system prompt kết hợp vai trò, tóm tắt cấp trên và kỹ năng đã lọc
      4. Định dạng task.prompt_template bằng các biến user_vars
      5. Chạy vòng lặp ReAct (với số vòng lặp tối đa max_iterations)
      6. Ghi báo cáo tóm tắt vào artifacts/{agent_id}/summary.md
      7. Trả về kết quả WorkerResult

    Args:
        agent_spec: Đặc tả vai trò của agent kèm cấu hình công cụ/kỹ năng/mô hình.
        task: Tác vụ cần thực thi, bao gồm mẫu prompt.
        upstream_summaries: Tóm tắt từ các tác vụ cấp trên theo khóa input_from.
        user_vars: Các biến do người dùng cung cấp để render mẫu prompt.
        run_dir: Đường dẫn đến thư mục thực thi .swarm/runs/{run_id}/.
        event_callback: Callback tùy chọn để nhận sự kiện swarm.
        include_shell_tools: Cho phép worker đăng ký các công cụ shell hay không.
        grounding_block: Khối markdown "Ground Truth" tùy chọn giúp tân trang dữ liệu thực tế.
        agent_config: Cấu hình agent tùy chọn chứa các định nghĩa MCP từ xa.

    Returns:
        Đối tượng WorkerResult chứa trạng thái, tóm tắt, danh sách artifacts và số vòng lặp.
    """
    agent_id = agent_spec.id
    task_id = task.id
    max_iterations = agent_spec.max_iterations or _default_max_iterations()
    timeout = agent_spec.timeout_seconds or _default_timeout_seconds()

    _emit(event_callback, "worker_started", agent_id, task_id)

    # 1. Dựng tool registry cho từng worker — từ pool cục bộ và các công cụ MCP từ xa,
    #    được chiếu theo danh sách trắng (whitelist) của agent.
    registry = build_swarm_registry(
        agent_spec.tools,
        agent_config=agent_config,
        include_shell_tools=include_shell_tools,
    )

    # 2. Khởi tạo LLM
    llm = ChatLLM(model_name=agent_spec.model_name)

    # 3. Xây dựng system prompt với danh sách kỹ năng đã được lọc
    skills_loader = SkillsLoader()
    skill_desc = _filter_skill_descriptions(skills_loader, agent_spec.skills)
    system_prompt = build_worker_prompt(
        agent_spec, upstream_summaries, skill_desc, grounding_block=grounding_block,
    )

    # 4. Giải mã prompt template với các biến người dùng truyền vào (biến thiếu -> LLM tự suy luận)
    class _FallbackDict(dict):
        """Dict hỗ trợ tự động gợi ý LLM điền các biến mẫu prompt còn thiếu."""
        def __missing__(self, key: str) -> str:
            return f"(determine the appropriate {key} based on the objective)"

    template_vars = _FallbackDict(user_vars)

    try:
        user_prompt = task.prompt_template.format_map(_FallbackDict(template_vars))
    except (KeyError, ValueError) as exc:
        error_msg = f"Failed to render prompt template: {exc}"
        _emit(event_callback, "worker_failed", agent_id, task_id, {"error": error_msg})
        return WorkerResult(
            status="failed", summary="", iterations=0, error=error_msg,
            input_tokens=0, output_tokens=0,
        )

    # 5. Dựng danh sách tin nhắn ban đầu
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 6. Vòng lặp ReAct
    artifact_dir = run_dir / "artifacts" / agent_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    iteration = 0
    summary = ""
    total_input_tokens = 0
    total_output_tokens = 0

    # Ngưỡng phát tin nhắn nhắc nhở "hoàn thiện" (80% ngân sách vòng lặp)
    wrap_up_at = max(1, int(max_iterations * 0.8))
    last_assistant_content = ""

    _KEEP_RECENT_TOOLS = 3
    data_tool_calls = 0
    content_filter_count = 0
    consecutive_content_filter_count = 0

    for iteration in range(max_iterations):
        # Microcompact: xóa bớt kết quả tool cũ để tránh phình token
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) > _KEEP_RECENT_TOOLS:
            for msg in tool_msgs[:-_KEEP_RECENT_TOOLS]:
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 100:
                    msg["content"] = "[cleared]"

        # Kiểm tra thời gian chờ (timeout)
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            summary = _best_summary(messages, last_assistant_content) or f"Worker timed out after {elapsed:.0f}s ({iteration} iterations)"
            summary = _resolve_summary(artifact_dir, summary)
            _emit(event_callback, "worker_timeout", agent_id, task_id, {"elapsed": elapsed})
            _write_summary(artifact_dir, summary)
            _persist_messages(artifact_dir, messages)
            return WorkerResult(
                status="timeout",
                summary=summary,
                artifact_paths=_collect_artifacts(artifact_dir),
                iterations=iteration,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                content_filter_warnings=compute_content_filter_warnings(
                    content_filter_count, iteration + 1,
                ),
            )

        # Kiểm tra ước tính token
        token_estimate = len(json.dumps(messages, ensure_ascii=False)) // 4
        if token_estimate > _MAX_TOKEN_ESTIMATE:
            summary = last_assistant_content or f"Worker context too large (~{token_estimate} tokens, {iteration} iterations)"
            summary = _resolve_summary(artifact_dir, summary)
            _emit(event_callback, "worker_token_limit", agent_id, task_id, {"tokens": token_estimate})
            _write_summary(artifact_dir, summary)
            return WorkerResult(
                status="token_limit",
                summary=summary,
                artifact_paths=_collect_artifacts(artifact_dir),
                iterations=iteration,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                content_filter_warnings=compute_content_filter_warnings(
                    content_filter_count, iteration + 1,
                ),
            )

        # Nhắc nhở hoàn thiện công việc khi sắp đến giới hạn số vòng lặp
        if iteration == wrap_up_at:
            remaining = max_iterations - iteration
            messages.append({
                "role": "user",
                "content": (
                    f"[SYSTEM] You have {remaining} iterations remaining. "
                    "If report.md is not written yet, make one final write_file call for report.md. "
                    "Otherwise stop calling tools and output your final analysis summary as plain text."
                ),
            })

        # Tại lượt chạy cuối cùng, gọi LLM không có định nghĩa tool để ép trả ra kết quả dạng văn bản
        is_last_iteration = iteration == max_iterations - 1
        tool_defs = None if is_last_iteration else registry.get_definitions()

        # Stream LLM - hỗ trợ hiển thị tiến độ thời gian thực trên dashboard
        try:
            def _on_text_chunk(delta: str) -> None:
                _emit(event_callback, "worker_text", agent_id, task_id,
                      {"content": delta, "iteration": iteration})

            def _on_llm_heartbeat(payload: dict) -> None:
                _emit(
                    event_callback,
                    "task_heartbeat",
                    agent_id,
                    task_id,
                    {**payload, "iteration": iteration, "phase": "llm"},
                )

            def _stream_once() -> ChatResponse:
                """Thực thi một lần gọi streaming LLM được bọc trong HeartbeatTimer.

                Tính toán lại thời gian chờ còn lại tại thời điểm gọi để đảm bảo
                lần thử lại sau khi stream thất bại không dùng lại timeout cũ.

                Returns:
                    Đối tượng ChatResponse được parse từ ChatLLM.stream_chat.

                Raises:
                    ProviderStreamError: Khi luồng stream từ nhà cung cấp gặp lỗi.
                """
                with HeartbeatTimer(
                    tool_name=f"llm:{agent_spec.model_name or 'default'}",
                    interval=_HEARTBEAT_INTERVAL_S,
                    emit=_on_llm_heartbeat,
                ):
                    return llm.stream_chat(
                        messages,
                        tools=tool_defs,
                        on_text_chunk=_on_text_chunk,
                    )

            try:
                response = _stream_once()
            except ProviderStreamError as stream_exc:
                if not stream_exc.retryable:
                    raise
                logger.warning(
                    "Provider stream failed for agent=%s task=%s iteration=%d "
                    "(provider=%s model=%s); retrying once: %s",
                    agent_id,
                    task_id,
                    iteration,
                    stream_exc.provider,
                    stream_exc.model,
                    stream_exc,
                )
                exc_str = str(stream_exc).lower()
                if (
                    getattr(stream_exc, "status_code", None) == 413
                    or any(k in exc_str for k in ("413", "payload too large", "request too large", "rate_limit", "tpm"))
                ):
                    _microcompact(messages)
                time.sleep(_STREAM_RETRY_DELAY_S)
                response = _stream_once()
        except Exception as exc:
            error_msg = f"LLM call failed at iteration {iteration}: {exc}"
            logger.warning(error_msg)
            _emit(event_callback, "worker_failed", agent_id, task_id, {"error": error_msg})
            return WorkerResult(
                status="failed",
                summary=_resolve_summary(artifact_dir, last_assistant_content or ""),
                artifact_paths=_collect_artifacts(artifact_dir),
                iterations=iteration,
                error=error_msg,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                content_filter_warnings=compute_content_filter_warnings(
                    content_filter_count, iteration + 1,
                ),
            )

        # Tích lũy số lượng token
        iter_in, iter_out = _estimate_tokens(messages, response)
        total_input_tokens += iter_in
        total_output_tokens += iter_out

        # Theo dõi nội dung phản hồi ý nghĩa gần nhất của assistant
        if response.content and len(response.content.strip()) > 20:
            last_assistant_content = response.content

        # Bỏ qua do bộ lọc nội dung (content filter)
        if response.content_filter_triggered:
            content_filter_count += 1
            consecutive_content_filter_count += 1
            if consecutive_content_filter_count >= MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS:
                _emit(
                    event_callback,
                    "content_filter_circuit_breaker",
                    agent_id,
                    task_id,
                    {"count": content_filter_count},
                )
                summary = _resolve_summary(artifact_dir, last_assistant_content or "")
                _write_summary(artifact_dir, summary)
                return WorkerResult(
                    status="failed",
                    summary=summary,
                    artifact_paths=_collect_artifacts(artifact_dir),
                    iterations=iteration + 1,
                    error=(
                        f"content_filter_circuit_breaker: "
                        f"{MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS} consecutive "
                        "LLM responses were blocked by content moderation"
                    ),
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    content_filter_warnings=compute_content_filter_warnings(
                        content_filter_count, iteration + 1,
                    ),
                )
            _emit(
                event_callback,
                "content_filter_skipped",
                agent_id,
                task_id,
                {"iteration": iteration, "content_filter_count": content_filter_count},
            )
            messages.append({
                "role": "system",
                "content": CONTENT_FILTER_SKIP_MESSAGE,
            })
            continue

        consecutive_content_filter_count = 0

        # Nếu không có yêu cầu gọi công cụ, đây là phản hồi cuối cùng
        if not response.has_tool_calls:
            summary = response.content or last_assistant_content or "(no summary)"
            summary = _resolve_summary(artifact_dir, summary)
            _write_summary(artifact_dir, summary)
            reason = _classify_deliverable(
                summary,
                is_data_agent=_is_data_agent(agent_spec),
                report_written=_report_written(artifact_dir),
                data_tool_calls=data_tool_calls,
            )
            if reason:
                _emit(event_callback, "worker_incomplete", agent_id, task_id,
                      {"iterations": iteration + 1, "reason": reason})
                return WorkerResult(
                    status="incomplete",
                    summary=summary,
                    artifact_paths=_collect_artifacts(artifact_dir),
                    iterations=iteration + 1,
                    error=f"output contract not met: {reason}",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    content_filter_warnings=compute_content_filter_warnings(
                        content_filter_count, iteration + 1,
                    ),
                )
            _emit(event_callback, "worker_completed", agent_id, task_id, {"iterations": iteration + 1})
            return WorkerResult(
                status="completed",
                summary=summary,
                artifact_paths=_collect_artifacts(artifact_dir),
                iterations=iteration + 1,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                content_filter_warnings=compute_content_filter_warnings(
                    content_filter_count, iteration + 1,
                ),
            )

        # Thêm thông điệp assistant kèm các yêu cầu gọi công cụ vào lịch sử
        messages.append(
            ContextBuilder.format_assistant_tool_calls(
                response.tool_calls,
                content=response.content,
                reasoning_content=response.reasoning_content,
            )
        )

        # Thực thi từng lệnh gọi công cụ — truyền run_dir để các công cụ ghi tệp vào artifact_dir
        for tc in response.tool_calls:
            mcp_meta = _remote_tool_metadata(registry, tc.name)
            _emit(
                event_callback, "tool_call", agent_id, task_id,
                {"tool": tc.name, "iteration": iteration,
                 "call_id": tc.id,
                 "arguments": _preview_tool_arguments(tc.arguments),
                 **mcp_meta},
            )
            tc_start = time.monotonic()
            args = {**tc.arguments, "run_dir": str(artifact_dir)}

            def _on_heartbeat(payload: dict) -> None:
                _emit(
                    event_callback,
                    "task_heartbeat",
                    agent_id,
                    task_id,
                    {**payload, "iteration": iteration, "phase": "tool"},
                )

            with HeartbeatTimer(
                tool_name=tc.name,
                interval=_HEARTBEAT_INTERVAL_S,
                emit=_on_heartbeat,
            ):
                result = registry.execute(tc.name, args)
            result_is_error = _is_error_result(result)
            if tc.name != "load_skill" and not result_is_error:
                data_tool_calls += 1
            tc_elapsed = time.monotonic() - tc_start
            _emit(
                event_callback,
                "tool_result",
                agent_id,
                task_id,
                {
                    "tool": tc.name,
                    "call_id": tc.id,
                    "elapsed_ms": int(tc_elapsed * 1000),
                    "status": "error" if result_is_error else "ok",
                    "iteration": iteration,
                    "result_preview": _preview_tool_result(result),
                    **mcp_meta,
                },
            )
            canonical_str = redact_tool_result(result)
            try:
                canonical_obj = json.loads(canonical_str)
            except Exception:
                canonical_obj = None
            if isinstance(canonical_obj, (dict, list)):
                view_str = json.dumps(canonical_obj, ensure_ascii=False)[:10_000]
            else:
                view_str = canonical_str[:10_000]

            messages.append(
                ContextBuilder.format_tool_result(tc.id, tc.name, view_str)
            )

    # Theo dõi cảnh báo từ bộ lọc nội dung
    content_filter_warnings = compute_content_filter_warnings(
        content_filter_count, iteration + 1,
    )

    # Chạm giới hạn vòng lặp — sử dụng nội dung ý nghĩa gần nhất làm tóm tắt
    summary = _best_summary(messages, last_assistant_content) or f"Worker hit iteration limit ({max_iterations} iterations)"
    summary = _resolve_summary(artifact_dir, summary)
    _write_summary(artifact_dir, summary)
    _persist_messages(artifact_dir, messages)
    reason = _classify_deliverable(
        summary,
        is_data_agent=_is_data_agent(agent_spec),
        report_written=_report_written(artifact_dir),
        data_tool_calls=data_tool_calls,
    )
    if reason:
        _emit(event_callback, "worker_incomplete", agent_id, task_id,
              {"iterations": max_iterations, "reason": f"iteration limit; {reason}"})
        return WorkerResult(
            status="incomplete",
            summary=summary,
            artifact_paths=_collect_artifacts(artifact_dir),
            iterations=max_iterations,
            error=f"hit iteration limit without a valid deliverable: {reason}",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            content_filter_warnings=content_filter_warnings,
        )
    _emit(event_callback, "worker_iteration_limit", agent_id, task_id)
    return WorkerResult(
        status="completed",
        summary=summary,
        artifact_paths=_collect_artifacts(artifact_dir),
        iterations=max_iterations,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        content_filter_warnings=content_filter_warnings,
    )


def _best_summary(messages: list[dict], fallback: str) -> str:
    """Trích xuất bản tóm tắt tốt nhất từ tất cả các tin nhắn của assistant."""
    texts = [
        m["content"] for m in messages
        if m.get("role") == "assistant" and m.get("content")
        and len(m["content"].strip()) > 100
    ]
    if texts:
        return max(texts, key=len)
    return fallback


def _remote_tool_metadata(registry: ToolRegistry, tool_name: str) -> dict[str, str]:
    """Trả về siêu dữ liệu định tuyến cho ``tool_name``."""
    return {}


def _preview_tool_arguments(arguments: dict) -> dict[str, str]:
    """Trả về bản xem trước ngắn gọn đã ẩn thông tin nhạy cảm của các tham số công cụ."""
    preview: dict[str, str] = {}
    for key, value in arguments.items():
        if key == "run_dir":
            continue
        if is_sensitive_arg(key):
            preview[key] = "[redacted]"
            continue
        preview[key] = _truncate_preview(redact_payload(value))
    return preview


def _preview_tool_result(result: str) -> str:
    """Trả về bản xem trước ngắn gọn đã ẩn thông tin nhạy cảm của kết quả gọi công cụ."""
    return _truncate_preview(redact_tool_result(result))


def _truncate_preview(value: Any, *, limit: int = 200) -> str:
    """Chuyển đổi thành chuỗi và cắt ngắn dữ liệu xem trước của sự kiện."""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


# Các công cụ chung không tự truy xuất/tính toán dữ liệu chuyên biệt.
_GENERIC_TOOLS = {"bash", "read_file", "write_file", "load_skill", "edit_file"}

_UNPARSED_TOOL_MARKERS = (
    "<\uff5ctool\u2581calls\u2581begin\uff5c>",
    "<tool_calls_begin>",
    "<tool_call_begin>",
    "<tool_sep>",
    "tool\u2581sep",
)
_FABRICATION_MARKERS = ("mock data", "without actual data", "fabricated data", "placeholder data")
_PLAN_PREFIXES = (
    "# phase 1", "## phase 1", "### phase 1",
    "phase 1 \u2014 plan", "phase 1 - plan", "phase 1: plan",
    "# plan", "## plan", "### plan", "**plan**",
)
_HANDOFF_TAILS = (
    "execute", "execute.", "execute:", "skills.", "skills", "proceed?",
    "proceed.", "without writing files.", "let me adjust the approach",
    "let me adjust the approach.", "stand by for final synthesis.",
)


def _report_written(artifact_dir: Path) -> bool:
    """Trả về True nếu tệp report.md hợp lệ và không rỗng được worker tạo ra."""
    try:
        p = artifact_dir / "report.md"
        return p.is_file() and bool(p.read_text(encoding="utf-8").strip())
    except Exception:
        return False


def _is_data_agent(agent_spec: SwarmAgentSpec) -> bool:
    """Kiểm tra xem agent có sở hữu ít nhất một công cụ xử lý dữ liệu ngoài bộ công cụ chung hay không."""
    return bool(set(agent_spec.tools or []) - _GENERIC_TOOLS)


def _is_error_result(result: str) -> bool:
    """Kiểm tra xem kết quả gọi công cụ có trả về envelope lỗi cấp cao nhất hay không."""
    text = (result or "").strip()
    if not text or not text.startswith("{"):
        return False
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        head = text[:160].lower()
        return '"status": "error"' in head or '"status":"error"' in head
    return isinstance(parsed, dict) and parsed.get("status") == "error"


def _classify_deliverable(
    summary: str,
    *,
    is_data_agent: bool,
    report_written: bool,
    data_tool_calls: int,
) -> str | None:
    """Hợp đồng kiểm tra sản phẩm đầu ra (Hybrid Output Contract).

    Trả về chuỗi lý do ngắn gọn khi worker KHÔNG tạo ra sản phẩm hoàn chỉnh,
    ngược lại trả về ``None``.
    """
    text = (summary or "").strip()
    if not text:
        return "empty deliverable"
    low = text.lower()
    if any(m in low for m in _UNPARSED_TOOL_MARKERS):
        return "unparsed tool-call markup (provider did not parse tool calls)"
    if any(m in low for m in _FABRICATION_MARKERS):
        return "explicitly fabricated / mock data"
    if text.startswith("{") and '"status"' in text[:40] and (
        '"content"' in text[:300] or '"ok"' in text[:40]
    ):
        return "raw tool-result envelope, not analysis"
    if low.startswith(_PLAN_PREFIXES):
        tail = low.rsplit("phase 2", 1)[-1].strip() if "phase 2" in low else ""
        if len(text) < 600 or low.rstrip().endswith(_HANDOFF_TAILS) or (
            "phase 2" in low and len(tail) < 80
        ):
            return "plan-only stub (no executed analysis / conclusion)"
    if is_data_agent and not report_written and data_tool_calls == 0:
        return "data agent produced no tool calls and no report.md"
    return None


def _resolve_summary(artifact_dir: Path, fallback: str) -> str:
    """Trả về nội dung tệp report.md nếu tồn tại, ngược lại dùng văn bản fallback."""
    report_path = artifact_dir / "report.md"
    try:
        if report_path.is_file():
            content = report_path.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:
        logger.warning("Failed to read report.md from %s", artifact_dir, exc_info=True)
    return fallback


def _persist_messages(artifact_dir: Path, messages: list[dict]) -> None:
    """Lưu lịch sử tin nhắn xuống đĩa để phục vụ phân tích nghiệm thu."""
    try:
        path = artifact_dir / "messages.json"
        path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to persist messages to %s", artifact_dir, exc_info=True)


def _write_summary(artifact_dir: Path, summary: str) -> None:
    """Ghi tóm tắt kết quả của worker vào thư mục sản phẩm (artifacts).

    Args:
        artifact_dir: Đường dẫn tới thư mục artifacts/{agent_id}/.
        summary: Nội dung văn bản tóm tắt cần ghi.
    """
    try:
        summary_path = artifact_dir / "summary.md"
        summary_path.write_text(summary, encoding="utf-8")
    except Exception:
        logger.warning("Failed to write summary to %s", artifact_dir, exc_info=True)


def _collect_artifacts(artifact_dir: Path) -> list[str]:
    """Thu thập các tệp sản phẩm (artifacts) thông thường dưới dạng đường dẫn tương đối.

    Args:
        artifact_dir: Đường dẫn tới thư mục artifacts/{agent_id}/.

    Returns:
        Danh sách đường dẫn POSIX đã sắp xếp tương đối so với thư mục swarm run.
    """
    if not artifact_dir.exists():
        return []

    run_dir = artifact_dir.parent.parent.resolve()
    artifact_root = artifact_dir.resolve()
    if not artifact_root.is_relative_to(run_dir):
        return []

    artifacts: list[str] = []
    for path in artifact_dir.rglob("*"):
        try:
            if path.is_symlink():
                continue
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(artifact_root) or not resolved.is_file():
                continue
            artifacts.append(resolved.relative_to(run_dir).as_posix())
        except (OSError, RuntimeError, ValueError):
            continue
    return sorted(artifacts)
