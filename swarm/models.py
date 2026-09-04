"""Hệ thống đa agent Swarm — các mô hình dữ liệu (data models).

Tất cả các Pydantic models được định nghĩa tại đây, dùng chung cho store / task_store / worker / runtime.
Các Enum sử dụng str+Enum để đảm bảo khả năng tương thích serial hóa JSON.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Trạng thái vòng đời của SwarmTask.

    Chuyển trạng thái:
        pending -> blocked -> in_progress -> completed | failed | cancelled
    """

    pending = "pending"
    blocked = "blocked"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class RunStatus(str, Enum):
    """Trạng thái vòng đời của SwarmRun.

    Chuyển trạng thái:
        pending -> running -> completed | failed | cancelled
    """

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class WorkerStatus(str, Enum):
    """Trạng thái kết thúc mà một worker trả về.

    ``incomplete`` phân biệt rõ với ``failed``: worker chạy không phát sinh ngoại lệ
    nhưng không tạo ra sản phẩm hoàn chỉnh thực sự.
    """

    completed = "completed"
    failed = "failed"
    timeout = "timeout"
    token_limit = "token_limit"
    incomplete = "incomplete"


class SwarmAgentSpec(BaseModel):
    """Định nghĩa vai trò cho một agent trong Swarm.

    Được phân tích từ các file preset YAML, mô tả danh tính, công cụ khả dụng và giới hạn của agent.

    Thuộc tính:
        id: Định danh duy nhất, ví dụ "macro_analyst".
        role: Mô tả vai trò.
        system_prompt: System prompt tiêm vào LLM.
        tools: Danh sách các tên công cụ được phép.
        skills: Danh sách các tên kỹ năng được phép.
        max_iterations: Số vòng lặp ReAct tối đa.
        timeout_seconds: Thời gian hết hạn của Worker tính bằng giây.
        model_name: Ghi đè mô hình mặc định; None dùng cấu hình toàn cục.
        max_retries: Số lần thử lại tối đa khi thất bại.
    """

    id: str
    role: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    max_iterations: int = 25
    timeout_seconds: int = 300
    model_name: str | None = None
    max_retries: int = 2


class SwarmTask(BaseModel):
    """Một nút tác vụ (task node) trong DAG của Swarm.

    Mỗi tác vụ được gắn với một agent. Phụ thuộc được khai báo qua depends_on.

    Thuộc tính:
        id: Định danh duy nhất, ví dụ "analyze_macro".
        agent_id: ID của agent thực thi tác vụ này.
        prompt_template: Template prompt người dùng hỗ trợ các giữ chỗ {var}.
        depends_on: Danh sách ID tác vụ cấp trên trong DAG (bất biến).
        blocked_by: Danh sách ID tác vụ cấp trên chưa hoàn thành.
        input_from: Ánh xạ để lấy tóm tắt từ các tác vụ cấp trên.
        status: Trạng thái hiện tại của tác vụ.
        summary: Văn bản tóm tắt sau khi hoàn thành.
        artifacts: Danh sách đường dẫn tệp đầu ra.
        error: Thông báo lỗi khi thất bại.
        started_at: Thời gian bắt đầu theo định dạng ISO.
        completed_at: Thời gian hoàn thành theo định dạng ISO.
        worker_iterations: Số vòng lặp ReAct thực tế được thực thi bởi worker.
        worker_status: `WorkerStatus` thô mà worker trả về, giữ nguyên không làm phẳng.
    """


    id: str
    agent_id: str
    prompt_template: str
    depends_on: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    input_from: dict[str, str] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.pending
    summary: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    worker_iterations: int = 0

    # `TaskStatus` chỉ có `failed`, nên runtime phải dồn cả `failed`/`timeout`/
    # `token_limit`/`incomplete` vào một ô. Trường này giữ lại giá trị gốc để phân biệt
    # "worker chạy xong mà không đẻ ra sản phẩm" với "worker chết" — mất phân biệt đó thì
    # I7 (thiếu dữ liệu ≠ hỏng) không thi hành được ở tầng swarm (M-RS0 §1.2).
    worker_status: str = ""


class SwarmEvent(BaseModel):
    """Nhật ký sự kiện Swarm (event log entry).

    Được ghi nối tiếp vào events.jsonl; hỗ trợ SSE streaming và kiểm tra lại sau khi lượt chạy hoàn tất.

    Thuộc tính:
        type: Loại sự kiện, ví dụ "run_started", "task_completed", "task_failed",
            "task_blocked" (cấp trên chưa hoàn thành -> cấp dưới bị bỏ qua).
        agent_id: ID agent liên quan (tùy chọn).
        task_id: ID tác vụ liên quan (tùy chọn).
        data: Dữ liệu bổ sung tùy ý.
        timestamp: Dấu thời gian định dạng ISO.
    """

    type: str
    agent_id: str | None = None
    task_id: str | None = None
    data: dict = Field(default_factory=dict)
    timestamp: str


class SwarmRun(BaseModel):
    """Trạng thái hoàn chỉnh của một lượt chạy Swarm preset.

    Lưu trữ bền vững tại .swarm/runs/{id}/run.json; đây là mô hình gốc tổng hợp cấp cao nhất.

    Thuộc tính:
        id: ID lượt chạy duy nhất (UUID).
        preset_name: Tên preset được dùng, ví dụ "research_team".
        status: Trạng thái lượt chạy.
        user_vars: Các biến do người dùng cung cấp để render template.
        agents: Danh sách định nghĩa các agent tham gia.
        tasks: Tất cả các mục tác vụ trong DAG.
        created_at: Thời gian tạo định dạng ISO.
        completed_at: Thời gian hoàn thành định dạng ISO.
        final_report: Văn bản báo cáo tổng hợp cuối cùng.
        total_input_tokens: Tổng lượng input token tích lũy qua tất cả worker.
        total_output_tokens: Tổng lượng output token tích lũy qua tất cả worker.
        provider: Tên nhà cung cấp LLM có hiệu lực khi khởi tạo lượt chạy.
        model: Tên mô hình LLM có hiệu lực khi khởi tạo lượt chạy.
        grounding_data: Dữ liệu nến OHLCV tải trước cho các mã tài sản trong user_vars.
    """

    id: str
    preset_name: str
    status: RunStatus = RunStatus.pending
    user_vars: dict[str, str] = Field(default_factory=dict)
    agents: list[SwarmAgentSpec] = Field(default_factory=list)
    tasks: list[SwarmTask] = Field(default_factory=list)
    created_at: str
    completed_at: str | None = None
    final_report: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    provider: str | None = None
    model: str | None = None
    grounding_data: dict[str, list[dict]] | None = None


class WorkerResult(BaseModel):
    """Giá trị trả về sau khi worker kết thúc thực thi.

    Thuộc tính:
        status: WorkerStatus — completed|failed|timeout|token_limit|incomplete.
        summary: Tóm tắt kết quả thực thi.
        artifact_paths: Danh sách đường dẫn tệp sản phẩm được tạo ra.
        iterations: Số vòng lặp ReAct thực tế đã thực thi.
        error: Thông báo lỗi khi thất bại.
        input_tokens: Số token đầu vào tích lũy (chính xác hoặc ước tính).
        output_tokens: Số token đầu ra tích lũy (chính xác hoặc ước tính).
        content_filter_warnings: Danh sách cảnh báo bộ lọc nội dung.
    """

    status: WorkerStatus
    summary: str
    artifact_paths: list[str] = Field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    content_filter_warnings: list[str] = Field(default_factory=list)
