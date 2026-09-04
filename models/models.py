# pseud/src/models/models.py
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GATE_VERSION = "gate_v1_ohlc"


class KetCucChang(str, Enum):
    """Kết quả một chặng ở L2.6. Phân biệt 'alpha dở' với 'không đủ dữ liệu để phán'.

    ★ TÊN KHÔNG ĐƯỢC LÀ `StageOutcome`. L2.5 đã có một lớp tên `StageOutcome`
    (factory_types.py:97) với tập giá trị KHÁC (PASS|KILL|BORDERLINE|ERROR|SKIPPED).
    `dieu_phoi.py` import cả hai; trùng tên là lỗi im lặng.
    """

    PASS = "pass"
    KILL = "kill"  # đã chấm, trượt cổng
    HARD_BLOCK = "hard_block"  # M-RS4 T-A1/T-A2/T-B — CHƯA từng được chấm
    SOFT_BLOCK = "soft_block"  # M-RS4 T-C — ĐÃ dispatch, có ghi nhận nghi ngờ trùng
    INSUFFICIENT_DATA = "insufficient_data"  # KHÔNG tính vào phễu KILL
    SKIPPED = "skipped"  # tiền đề không thỏa (vd L2 không có schema)
    ERROR = "error"  # fail-closed ⇒ như KILL nhưng gắn cờ điều tra


class DataTier(str, Enum):
    """Tầng dữ liệu M-RS1 được phép chạm (OHLCV nến 1-phút)."""

    BAR = "bar"  # da_xu_ly/BINANCE/{SYM}/year=/month=


class RunHealth(str, Enum):
    """Sức khỏe HẠ TẦNG của một lượt gọi L2.5 — TÁCH HẲN khỏi phán xét alpha."""

    OK = "ok"
    INFRA_ABORTED = "infra_aborted"  # breaker mở, pool sập, manual stop, status lạ


class BriefStatus(str, Enum):
    """Kết cục một BRIEF."""

    DONE = "done"
    BUDGET_LIMITED = "budget_limited"
    INFRA_ABORTED = "infra_aborted"


class ResearchBrief(BaseModel):
    """Đầu vào một đợt nghiên cứu."""

    model_config = ConfigDict(frozen=True)

    brief_id: str
    target_universe: list[str]
    timeframe: str = "1h"
    lookback_days: int = Field(default=365, ge=1)
    max_horizon_minutes: int = Field(default=120, ge=1)
    pit_override: list[str] | None = None
    allowed_tiers: list[DataTier] = Field(default_factory=lambda: [DataTier.BAR])
    family_hint: str | None = None
    max_hypotheses: int = Field(default=8, ge=1)
    enable_debate: bool = False
    token_budget: int | None = None
    time_budget_s: int | None = None


class AnomalySignal(BaseModel):
    """Một phát hiện dị thường của M-RS1. Bắt buộc kèm nguồn gốc kiểm chứng được."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    symbol: str
    tier: DataTier
    raw_columns: list[str]
    feature_names: list[str]
    stat: str
    value: float
    effect_size: float
    effect_confirm: float
    n_effective: int = Field(ge=0)
    p_value: float | None = None
    tests_performed: int = Field(ge=0)
    tests_independent: int = Field(ge=0)
    segment_id: int
    event_cluster_id: str
    co_fires_with: list[str] = Field(default_factory=list)
    window: str
    observed_at: str
    source_paths: list[str]
    content_fingerprint: str
    byte_sha256: str
    by_fdr_threshold: float | None = None
    tau_effective: float | None = None
    config_git_sha: str
    pit_source: str
    data_provenance: str


class CemeteryVerdict(BaseModel):
    """Phán quyết cổng nghĩa trang. Thay cemetery_overlap: float của bản cũ."""

    model_config = ConfigDict(frozen=True)

    tier: Literal["T-A1", "T-A2", "T-B", "T-C", "T-D", "NONE"]
    outcome: Literal["HARD_BLOCK", "SOFT_BLOCK", "WARNING", "PASS"]
    score: float | None = None
    hamming: int | None = None
    matched_alpha_ids: list[str] = Field(default_factory=list)
    fingerprint_status: Literal["OK", "UNVERIFIED"] = "OK"
    stem_version: str = ""
    kill_reason_class: Literal["EVIDENTIAL", "REGIME_DEPENDENT", "INFRASTRUCTURAL", "NONE"] = "NONE"


class HypothesisSpec(BaseModel):
    """Payload duy nhất đẩy sang L2.5. Hợp đồng một chiều."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    title: str
    mechanism_columns: list[str]
    predicted_sign: Literal[-1, 1]
    predicted_horizon_band: list[str]
    conditional_prediction: dict[str, str]
    target_regime: str | None
    target_universes: list[str]
    expansion_budget: int = Field(ge=1)
    vocab_version: str
    resolution_mode: Literal["ANCHORED", "PARTIAL", "UNANCHORED"]
    anchor_coverage: float = Field(ge=0.0, le=1.0)
    resolved_template_ids: list[str]
    resolved_families: list[str]
    horizon_clamped: bool = False
    budget_truncated: bool = False
    cemetery_verdict: CemeteryVerdict
    evidence_signal_ids: list[str]
    evidence_hash: str
    spec_fingerprint: str
    tests_independent: int = Field(ge=0)
    gate_token: str
    #: Hai trường KHÔNG trang trí: `gate_token` là HMAC trên
    #: `spec_fingerprint ‖ plan_id ‖ gate_version`, nên thiếu chúng thì token là chuỗi
    #: KHÔNG AI XÁC THỰC ĐƯỢC — kể cả chính L2.6. `execute()` bên L2.5 chỉ kiểm được cổng
    #: chặng 1 khi bản ghi mang đủ cả ba.
    plan_id: str = ""
    gate_version: str = GATE_VERSION
    narrative: str = ""
    hypothesis_text: str = ""
    generated_by: str = "ai_autonomous_layer2.6"
    created_at: str


class MutationSpec(BaseModel):
    """Đầu ra vòng chẩn đoán hậu-KILL của M-RS3. Đầu vào M-RS4 lần hai."""

    model_config = ConfigDict(frozen=True)

    decision: Literal["MUTATE", "BURY", "INSUFFICIENT_EVIDENCE"]
    parent_variant_id: str
    dimension: Literal["lookback_window", "decay_factor", "clip_threshold", "regime_mask", "operator"]
    from_value: str
    to_value: str
    mutated_operator_expr: str
    ast_logic_hash: str
    variant_id: str
    decisive_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class StageRecord(BaseModel):
    """Bản ghi bền vững/stage — nền tảng của resume idempotent."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    stage_id: str
    outcome: KetCucChang
    trace_seq: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    elapsed_ms: int = Field(default=0, ge=0)
    finished_at: str = ""

