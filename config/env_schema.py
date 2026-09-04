"""Nguồn sự thật cho tất cả các giá trị mặc định của biến môi trường Kairos v3."""

from __future__ import annotations

import os
from typing import Annotated, Any
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

__all__ = [
    "EnvConfig",
    "LLMConfig",
    "DataConfig",
    "APIConfig",
    "SwarmConfig",
    "AgentTuningConfig",
    "PathConfig",
    "MemoryConfig",
]


def _parse_env_bool(v: Any) -> Any:
    """Ép chuỗi giá trị môi trường sang kiểu boolean."""
    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off", ""}:
            return False
    return v


EnvBool = Annotated[bool, BeforeValidator(_parse_env_bool)]


class _EnvBase(BaseModel):
    """Lớp cơ sở cho các sub-model cấu hình từ biến môi trường."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, data: Any) -> Any:
        """Điền các trường bị thiếu từ os.environ bằng alias của trường."""
        if not isinstance(data, dict):
            return data
        result = dict(data)
        for field_name, field_info in cls.model_fields.items():
            alias = field_info.alias
            if field_name in result or (alias and alias in result):
                continue
            if alias and alias in os.environ:
                env_val = os.environ[alias]
                annotation = field_info.annotation
                if annotation is int and isinstance(env_val, str):
                    try:
                        env_val = int(env_val)
                    except ValueError:
                        continue
                elif annotation is float and isinstance(env_val, str):
                    try:
                        env_val = float(env_val)
                    except ValueError:
                        continue
                result[alias] = env_val
        return result


class LLMConfig(_EnvBase):
    """Tham số cấu hình nhà cung cấp LLM."""

    langchain_provider: str = Field(alias="LANGCHAIN_PROVIDER", default="openai")
    langchain_model_name: str = Field(alias="LANGCHAIN_MODEL_NAME", default="")
    langchain_temperature: float = Field(alias="LANGCHAIN_TEMPERATURE", default=0.0)
    # Điểm cuối và khoá cho backend tương thích OpenAI. Rỗng ⇒ dùng điểm cuối mặc định
    # của nhà cung cấp. Đặt base_url là cách trỏ sang DeepSeek/OpenRouter/vLLM nội bộ
    # mà không phải đổi mã.
    langchain_base_url: str = Field(alias="LANGCHAIN_BASE_URL", default="")
    langchain_api_key: str = Field(alias="LANGCHAIN_API_KEY", default="")
    anthropic_max_tokens: int | None = Field(alias="ANTHROPIC_MAX_TOKENS", default=None, gt=0)
    timeout_seconds: int = Field(alias="TIMEOUT_SECONDS", default=120)
    max_retries: int = Field(alias="MAX_RETRIES", default=2)
    langchain_reasoning_effort: str = Field(alias="LANGCHAIN_REASONING_EFFORT", default="")


class DataConfig(_EnvBase):
    """Thông tin xác thực nguồn dữ liệu thị trường Crypto."""

    ccxt_exchange: str = Field(alias="CCXT_EXCHANGE", default="binance")
    ccxt_timeout_ms: int = Field(alias="CCXT_TIMEOUT_MS", default=15000)
    ccxt_fetch_budget_s: float = Field(alias="CCXT_FETCH_BUDGET_S", default=60.0)


class APIConfig(_EnvBase):
    """Cấu hình API server, CORS và bảo mật."""

    api_auth_key: str = Field(alias="API_AUTH_KEY", default="")
    kairos_api_key: str = Field(alias="KAIROS_API_KEY", default="")
    cors_origins: str = Field(alias="CORS_ORIGINS", default="")
    api_allowed_hosts: str = Field(alias="API_ALLOWED_HOSTS", default="")
    enable_session_runtime: EnvBool = Field(alias="ENABLE_SESSION_RUNTIME", default=True)

    # Ba danh sách gốc thư mục (phân tách bằng dấu phẩy) mà `tools/path_utils.py` ĐỌC
    # nhưng lược đồ cũ không KHAI ⇒ `allowed_file_roots()`/`allowed_write_roots()`/
    # `_allowed_run_roots()` ném AttributeError. Rỗng nghĩa là "chỉ dùng gốc mặc định",
    # KHÔNG phải "cho phép mọi nơi" — mã đọc nối chúng vào danh sách mặc định.
    kairos_allowed_file_roots: str = Field(alias="KAIROS_ALLOWED_FILE_ROOTS", default="")
    kairos_allowed_write_roots: str = Field(alias="KAIROS_ALLOWED_WRITE_ROOTS", default="")
    kairos_allowed_run_roots: str = Field(alias="KAIROS_ALLOWED_RUN_ROOTS", default="")


class SwarmConfig(_EnvBase):
    """Các tham số thực thi Swarm đa agent."""

    swarm_worker_timeout: int = Field(alias="SWARM_WORKER_TIMEOUT", default=300)
    swarm_worker_max_iter: int = Field(alias="SWARM_WORKER_MAX_ITER", default=50)
    swarm_max_workers: int = Field(alias="SWARM_MAX_WORKERS", default=4)
    swarm_timeout: int = Field(alias="SWARM_TIMEOUT", default=1800)
    swarm_heartbeat_interval_s: float = Field(alias="SWARM_HEARTBEAT_INTERVAL_S", default=3.0)
    swarm_stream_retry_delay_s: float = Field(alias="SWARM_STREAM_RETRY_DELAY_S", default=2.0)
    swarm_grounding_max_symbols: int = Field(alias="SWARM_GROUNDING_MAX_SYMBOLS", default=8)


class AgentTuningConfig(_EnvBase):
    """Tinh chỉnh vòng lặp agent và bộ lập lịch."""

    token_threshold: int = Field(alias="TOKEN_THRESHOLD", default=40000)
    vt_heartbeat_interval_s: float = Field(alias="VT_HEARTBEAT_INTERVAL_S", default=3.0)
    kairos_tool_timeout_seconds: float = Field(
        alias="KAIROS_TOOL_TIMEOUT_SECONDS", default=1800.0,
    )

    # Bốn trường dưới đây được `agent/loop.py` và `preflight.py` ĐỌC nhưng lược đồ cũ
    # không KHAI ⇒ `AgentLoop` ném AttributeError ngay vòng lặp đầu. Cùng loại lỗi với
    # bốn cờ Tier-2 của MemoryConfig: module bóc sang đầy đủ, lược đồ config thì bị cắt.
    vt_reasoning_delta_min_interval_s: float = Field(
        alias="VT_REASONING_DELTA_MIN_INTERVAL_S", default=0.5,
    )
    # Khớp mặc định của `SwarmConfig.swarm_stream_retry_delay_s` — hai đường đi
    # (AgentLoop và swarm worker) dùng cùng chính sách thử lại một lần.
    vt_stream_retry_delay_s: float = Field(alias="VT_STREAM_RETRY_DELAY_S", default=2.0)
    kairos_goal_max_continuations: int = Field(
        alias="KAIROS_GOAL_MAX_CONTINUATIONS", default=3, ge=0,
    )
    content_filter_warning_threshold: float = Field(
        alias="CONTENT_FILTER_WARNING_THRESHOLD", default=0.2, ge=0.0, le=1.0,
    )


class PathConfig(_EnvBase):
    """Đường dẫn hệ thống tệp cho agent."""

    kairos_hypotheses_path: str = Field(alias="KAIROS_HYPOTHESES_PATH", default="")
    allow_session_mcp_servers: EnvBool = Field(alias="ALLOW_SESSION_MCP_SERVERS", default=False)
    # `goal/store.py::_default_db_path()` ĐỌC trường này; rỗng ⇒ dùng _DEFAULT_DB_PATH.
    kairos_goal_db_path: str = Field(alias="KAIROS_GOAL_DB_PATH", default="")


class MemoryConfig(_EnvBase):
    """Các cờ tính năng bộ nhớ bền vững."""

    preset: str = Field(default="on", alias="VT_MEMORY")
    quality_enabled: EnvBool = Field(default=True, alias="VT_MEMORY_QUALITY")
    gc_enabled: EnvBool = Field(default=True, alias="VT_MEMORY_GC")
    decay_enabled: EnvBool = Field(default=True, alias="VT_MEMORY_DECAY")

    # Bốn cờ Tier-2 dưới đây được `memory/` ĐỌC nhưng lược đồ cũ không KHAI, nên mọi
    # lần ghi ký ức ném AttributeError — cả tầng hỏng ở đường nóng. Mặc định False:
    # bốn module tương ứng (hierarchy, compression, search_index, semantic_links)
    # chưa nằm trong đường đi nào của L2.6, bật sẵn là kích hoạt mã chưa được kiểm.
    hierarchy_enabled: EnvBool = Field(default=False, alias="VT_MEMORY_HIERARCHY")
    compression_enabled: EnvBool = Field(default=False, alias="VT_MEMORY_COMPRESSION")
    fts_index_enabled: EnvBool = Field(default=False, alias="VT_MEMORY_FTS")
    links_enabled: EnvBool = Field(default=False, alias="VT_MEMORY_LINKS")


class EnvConfig(_EnvBase):
    """Mô hình cấu hình gốc hợp thành tất cả nhóm biến môi trường."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    agent_tuning: AgentTuningConfig = Field(default_factory=AgentTuningConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @model_validator(mode="after")
    def _resolve_api_key_alias(self) -> "EnvConfig":
        if self.api.kairos_api_key and not self.api.api_auth_key:
            self.api.api_auth_key = self.api.kairos_api_key
        return self
