"""Động cơ kiểm chứng thực thể và bằng chứng số học tổng quát (Generic Fact & Numeric Grounding Engine) cho Agent Loop.

Mô hình ngôn ngữ chịu trách nhiệm nghiên cứu và giải thích, nhưng hai quy tắc
sau đây mang tính cấu trúc bắt buộc chứ không chỉ mang tính cố vấn:

* Bộ tiêu thụ dữ liệu chỉ được sử dụng định danh thực thể đã được khóa trước khi
  danh sách gọi công cụ (tool call) hiện tại của trợ lý bắt đầu; và
* Tuyên bố số liệu cuối cùng không được mâu thuẫn với kết quả thực tế (chưa bị cắt gọt) của công cụ.

Mô-đun này cố ý không chứa phụ thuộc vào nhà cung cấp dịch vụ hay thư viện công cụ cụ thể
để máy trạng thái và các kiểm tra câu trả lời cuối cùng luôn mang tính xác định và có thể kiểm thử.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


GROUNDING_ARTIFACT = "grounding_evidence.json"

_RESOLVER_TOOL = "search"
_SYMBOL_ARGUMENT_KEYS = {
    "code",
    "codes",
    "symbol",
    "symbols",
    "ticker",
    "tickers",
    "underlying",
    "underlyings",
    "entity",
    "entities",
    "id",
    "ids",
}
_NUMERIC_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price",
    "value",
    "val",
    "amount",
    "count",
    "zscore",
    "volatility",
    "ratio",
    "mean",
    "std",
    "min",
    "max",
}
_PRICE_FIELDS = _NUMERIC_FIELDS  # Bí danh tương thích ngược
_TIMESTAMP_FIELDS = ("trade_date", "date", "datetime", "timestamp", "time", "index")
_MAX_GENERIC_EVIDENCE = 2_000

# Biểu thức chính quy bắt thực thể/mã định danh tổng quát (Generic Entity/Code Regex).
# Bắt các mã có phân cách (AAPL.US, BTC-USDT, 600519.SH) hoặc mã viết hoa (NVDA, TSLA).
#
# ⚠️ ĐÂY LÀ BỘ DỰ PHÒNG, KHÔNG CÒN LÀ MẶC ĐỊNH. Trên vốn từ Kairos nó bắt sai hàng loạt:
# `ZSCORE`, `RATIO`, `EMA`, `VOLATILITY`, `MEAN`, `STD` đều khớp `[A-Z0-9]{2,15}` và bị đọc thành "mã tài
# sản", khiến agent đòi phân giải danh tính cho một chỉ báo thống kê (M-RS0 §1.2, M-RS2 §5).
# Mặc định giờ là whitelist lấy từ AssetRegistry — dữ liệu thật thay vì đoán bằng chữ hoa.
_GENERIC_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?=[A-Z0-9.-]*[A-Z])(?:[A-Z0-9]{2,15}(?:[-./][A-Z0-9]{1,10})+|[A-Z0-9]{2,15})(?![A-Za-z0-9_])"
)

# Thuật ngữ chỉ số và kỹ thuật định lượng OHLCV không phải mã tài sản
_THUAT_NGU_DINH_LUONG = frozenset({
    "OHLCV", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "ZSCORE", "VOLATILITY",
    "RATIO", "MEAN", "STD", "MIN", "MAX", "SKEW", "KURT", "VOL",
    "NAN", "NA", "UTC", "USD", "USDT", "API", "CSV", "JSON", "SQL", "ID",
    "AND", "OR", "NOT", "IF", "THE", "FOR", "ALL", "ANY", "SUM", "ABS", "LOG", "EXP",
    "TODO", "OK", "PASS", "KILL", "ERROR",
})

logger = logging.getLogger(__name__)

# Sàn/loại công cụ mặc định khi nạp whitelist. Lakehouse hiện chỉ có BINANCE (M-RS1 §1).
# `PERP` là giá trị THẬT trong asset_registry.db — không phải `PERPETUAL`; `all_active`
# so khớp chính xác nên viết sai một chữ là whitelist rỗng và im lặng lùi về regex.
_SAN_MAC_DINH = "BINANCE"
_LOAI_CONG_CU_MAC_DINH = "PERP"


def nap_ma_tai_san_tu_registry(
    exchange: str = _SAN_MAC_DINH,
    instrument_type: str = _LOAI_CONG_CU_MAC_DINH,
    db_path: Any = None,
) -> set[str]:
    """Trả tập mã tài sản đang hoạt động từ thư mục data/ hoặc danh sách mặc định."""
    ma: set[str] = set()
    data_dir = Path("data")
    if data_dir.is_dir():
        for p in data_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                ma.add(p.name.strip().upper())
    if not ma:
        ma = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}
    return ma


def dung_regex_whitelist(ma_tai_san: set[str]) -> re.Pattern | None:
    """Dựng regex chỉ khớp ĐÚNG các mã trong whitelist.

    Sắp xếp theo độ dài giảm dần để `BTCUSDT` được thử trước `BTC` — nếu không, phép
    thay thế của regex sẽ khớp tiền tố ngắn rồi cắt đôi mã dài.
    """
    if not ma_tai_san:
        return None
    nhanh = "|".join(re.escape(m) for m in sorted(ma_tai_san, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{nhanh})(?![A-Za-z0-9_])", re.IGNORECASE)


class _RegexDuPhongLocThuatNgu:
    """Bọc regex tổng quát và loại bỏ thuật ngữ định lượng khỏi kết quả.

    Dùng khi AssetRegistry không sẵn sàng. Vẫn tốt hơn regex trần: nó không còn biến
    `ZSCORE`/`VOLATILITY`/`RATIO` thành mã tài sản, tức là bỏ đúng lớp dương tính giả mà §1.2 nêu.
    """

    def __init__(self, goc: re.Pattern, loai_tru: frozenset[str] = _THUAT_NGU_DINH_LUONG):
        self._goc = goc
        self._loai_tru = loai_tru

    def finditer(self, text: str):
        for m in self._goc.finditer(text or ""):
            if m.group(0).upper() not in self._loai_tru:
                yield m

    def search(self, text: str):
        for m in self.finditer(text):
            return m
        return None


def entity_matcher_mac_dinh(db_path: Any = None):
    """Trả bộ khớp thực thể mặc định cho L2.6.

    Ưu tiên whitelist AssetRegistry; registry rỗng/hỏng thì lùi về regex tổng quát đã
    lọc thuật ngữ định lượng, kèm cảnh báo — không im lặng quay lại hành vi cũ.
    """
    ma = nap_ma_tai_san_tu_registry(db_path=db_path)
    whitelist = dung_regex_whitelist(ma)
    if whitelist is not None:
        return whitelist
    logger.warning(
        "AssetRegistry rỗng hoặc không đọc được — dùng regex dự phòng đã lọc %d thuật ngữ "
        "định lượng. Phát hiện thực thể sẽ kém chính xác hơn whitelist.",
        len(_THUAT_NGU_DINH_LUONG),
    )
    return _RegexDuPhongLocThuatNgu(_GENERIC_ENTITY_RE)

# Biểu thức chính quy cho từ khóa hành động/truy vấn tổng quát (Generic Actionable Query Regex).
_GENERIC_ACTIONABLE_RE = re.compile(
    r"(?:\bquery\b|\bsearch\b|\bfetch\b|\bverify\b|\bcheck\b|\blookup\b|\bfind\b|\bcalculate\b|\bget\b)",
    re.IGNORECASE,
)

# Biểu thức chính quy cho ngữ cảnh số học tổng quát (Generic Numeric Context Regex).
_NUMERIC_CONTEXT_RE = re.compile(
    r"(?:\b(?:opening|open|high|low|closing|close|price|quote|value|val|amount|total|count|score|rate|quantity|metric|result)\b)",
    re.IGNORECASE,
)
_PRICE_CONTEXT_RE = _NUMERIC_CONTEXT_RE  # Bí danh tương thích ngược

_DERIVATION_RE = re.compile(
    r"(?:\bderived\b|\bcalculated\b|\bformula\b|\bbased on\b|\bdẫn xuất\b|\btính toán\b|\bcông thức\b|\bdựa trên\b)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?![A-Za-z0-9_])"
)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")

_TABLE_FIELD_ALIASES = {
    "open": "open",
    "opening": "open",
    "opening price": "open",
    "mở cửa": "open",
    "giá mở cửa": "open",
    "high": "high",
    "highest": "high",
    "cao nhất": "high",
    "giá cao nhất": "high",
    "low": "low",
    "lowest": "low",
    "thấp nhất": "low",
    "giá thấp nhất": "low",
    "close": "close",
    "closing": "close",
    "closing price": "close",
    "đóng cửa": "close",
    "giá đóng cửa": "close",
    "value": "value",
    "val": "value",
    "giá trị": "value",
    "amount": "amount",
    "score": "score",
    "count": "count",
    "số lượng": "count",
    "rate": "rate",
    "tỷ lệ": "rate",
    "result": "result",
    "kết quả": "result",
}
_DATE_HEADERS = {"date", "datetime", "trade date", "timestamp", "time", "ngày", "thời gian"}
_SYMBOL_HEADERS = {"symbol", "ticker", "code", "entity", "id", "mã", "thực thể"}


def _utc_now() -> str:
    """Trả về nhãn thời gian UTC ISO-8601 phục vụ kiểm toán."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_symbol(value: Any) -> str:
    """Chuẩn hóa mã thực thể (symbol/code) để so sánh danh tính chính xác."""
    return str(value or "").strip().upper()


def _query_key(value: Any) -> str:
    """Chuẩn hóa truy vấn bộ phân giải thành khóa máy trạng thái ổn định."""
    return " ".join(str(value or "").casefold().split())


def _json_object(value: Any) -> dict[str, Any] | None:
    """Phân tích đối tượng JSON từ kết quả công cụ khi có thể."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_number(value: Any) -> bool:
    """Kiểm tra một giá trị có phải là số kiểu JSON hữu hạn (không bao gồm boolean) hay không."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _infer_domain(symbol: str) -> str | None:
    """Suy luận phạm vi/phân vùng thực thể từ mã định danh tổng quát."""
    upper = _normalize_symbol(symbol)
    if "-" in upper or "/" in upper or "." in upper:
        return "structured_entity"
    return "generic_entity"


_infer_venue = _infer_domain  # Bí danh tương thích ngược


def _infer_unit(symbol: str) -> str | None:
    """Suy luận đơn vị đo lường hoặc định danh phụ từ mã thực thể."""
    upper = _normalize_symbol(symbol)
    for separator in ("-", "/", "."):
        if separator in upper:
            parts = upper.rsplit(separator, 1)
            if len(parts) == 2 and 1 <= len(parts[1]) <= 10:
                return parts[1]
    return None


_infer_currency = _infer_unit  # Bí danh tương thích ngược


def _infer_entity_type(symbol: str, candidate_type: Any = None) -> str:
    """Chuẩn hóa loại thực thể tổng quát theo hợp đồng danh tính."""
    if candidate_type:
        return str(candidate_type).strip().casefold()
    return "generic"


_infer_instrument_type = _infer_entity_type  # Bí danh tương thích ngược


def _is_entity_alias_conflict(existing_symbol: str, new_symbol: str) -> bool:
    """Kiểm tra xem hai mã thực thể có xung đột bí danh hay không."""
    s1 = _normalize_symbol(existing_symbol)
    s2 = _normalize_symbol(new_symbol)
    if s1 == s2:
        return False
    base1 = re.split(r"[-/.=]", s1)[0]
    base2 = re.split(r"[-/.=]", s2)[0]
    return base1 == base2 and s1 != s2


_is_exchange_alias_conflict = _is_entity_alias_conflict  # Bí danh tương thích ngược


@dataclass(frozen=True)
class IdentityRecord:
    """Bản ghi phân giải danh tính thực thể sang công cụ có phiên bản."""

    query: str
    status: str
    symbol: str | None = None
    venue: str | None = None
    instrument_type: str | None = None
    currency: str | None = None
    source_tool_call_id: str | None = None
    source: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    updated_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True)
class EvidenceRecord:
    """Bản ghi mục bằng chứng số học được quan sát, không khả dụng, hoặc dẫn xuất."""

    call_id: str
    tool: str
    symbol: str | None
    source: str
    timestamp: str | None
    field: str
    value: int | float | None
    status: str
    currency: str | None = None
    venue: str | None = None
    currency_conversion: str | None = None


@dataclass(frozen=True)
class ToolAuthorization:
    """Quyết định ủy quyền công cụ mang tính xác định được đưa ra trước khi công cụ bắt đầu."""

    allowed: bool
    error_code: str | None = None
    message: str | None = None
    symbols: tuple[str, ...] = ()

    def error_payload(self, tool_name: str, identity: Mapping[str, Any]) -> str:
        """Hiển thị cuộc gọi công cụ bị chặn dưới dạng kết quả lỗi có cấu trúc thông thường."""
        return json.dumps(
            {
                "status": "error",
                "error_code": self.error_code or "identity_gate_blocked",
                "tool": tool_name,
                "message": self.message or "Tool call blocked by identity gate",
                "symbols": list(self.symbols),
                "identity": dict(identity),
                "required_action": (
                    "Call search in a separate assistant tool turn, wait for "
                    "its result, then reuse the exact locked entity symbol and domain."
                ),
            },
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class ValidationResult:
    """Kết quả kiểm chứng neo ngữ cảnh (grounding) cho câu trả lời cuối cùng."""

    valid: bool
    issues: list[dict[str, Any]] = field(default_factory=list)


class GroundingLedger:
    """Máy trạng thái kiểm chứng danh tính thực thể và sổ nhật ký bằng chứng số học tổng quát."""

    def __init__(
        self,
        *,
        run_dir: Path,
        user_message: str,
        history: Sequence[Mapping[str, Any]] | None = None,
        entity_re: re.Pattern[str] | str | None = None,
        actionable_re: re.Pattern[str] | str | None = None,
        numeric_context_re: re.Pattern[str] | str | None = None,
    ) -> None:
        """Khởi tạo sổ nhật ký kiểm chứng danh tính và nạp các thực thể ban đầu từ yêu cầu người dùng.

        Args:
            run_dir: Thư mục lượt chạy đang hoạt động.
            user_message: Yêu cầu hiện tại của người dùng.
            history: Lịch sử tin nhắn trước đó (tùy chọn).
            entity_re: Regex tùy chỉnh để phát hiện thực thể/mã định danh.
            actionable_re: Regex tùy chỉnh để phát hiện câu hỏi/hành động cần kiểm chứng.
            numeric_context_re: Regex tùy chỉnh cho ngữ cảnh số học.
        """
        self.run_dir = Path(run_dir)
        self.user_message = user_message

        # Cấu hình các biểu thức chính quy (có thể nạp từ tham số hoặc dùng mặc định)
        if isinstance(entity_re, str):
            self._entity_re = re.compile(entity_re)
        elif isinstance(entity_re, re.Pattern):
            self._entity_re = entity_re
        else:
            # Mặc định là whitelist AssetRegistry, KHÔNG phải regex chữ hoa (M-RS2 §5).
            self._entity_re = entity_matcher_mac_dinh()

        if isinstance(actionable_re, str):
            self._actionable_re = re.compile(actionable_re, re.IGNORECASE)
        elif isinstance(actionable_re, re.Pattern):
            self._actionable_re = actionable_re
        else:
            self._actionable_re = _GENERIC_ACTIONABLE_RE

        if isinstance(numeric_context_re, str):
            self._numeric_context_re = re.compile(numeric_context_re, re.IGNORECASE)
        elif isinstance(numeric_context_re, re.Pattern):
            self._numeric_context_re = numeric_context_re
        else:
            self._numeric_context_re = _NUMERIC_CONTEXT_RE

        self._identities: dict[str, IdentityRecord] = {}
        self._evidence: list[EvidenceRecord] = []
        self._tool_failures: list[dict[str, Any]] = []
        self._validations: list[dict[str, Any]] = []
        self._identity_required = bool(self._actionable_re.search(user_message))
        self._buffer_output = self._identity_required

        self.ledger_dir = self.run_dir / "ledger"

        self._seed_symbols(user_message, source="user_message")
        self.persist()

    @property
    def authorized_symbols(self) -> set[str]:
        """Trả về các mã thực thể chính xác đã được khóa trước đợt gọi công cụ tiếp theo."""
        return {
            record.symbol
            for record in self._identities.values()
            if record.status == "locked" and record.symbol
        }

    @property
    def identity_status(self) -> str:
        """Trả về trạng thái danh tính hợp thành cấp cao nhất."""
        records = list(self._identities.values())
        if not records:
            return "unresolved" if self._identity_required else "not_required"
        statuses = {record.status for record in records}
        for blocking in ("conflicting", "ambiguous", "invalidated", "unresolved"):
            if blocking in statuses:
                return blocking
        if "locked" in statuses:
            return "locked"
        if statuses == {"not_found"}:
            return "not_found"
        return "unresolved"

    @property
    def should_buffer_output(self) -> bool:
        """Trả về liệu văn bản nháp chưa được kiểm chứng của mô hình có cần ẩn khỏi output trực tiếp hay không."""
        return self._buffer_output or bool(self._evidence)

    @property
    def validation_count(self) -> int:
        """Trả về số lượng bản nháp cuối cùng đã được kiểm tra."""
        return len(self._validations)

    def identity_summary(self) -> dict[str, Any]:
        """Trả về tóm tắt ngắn gọn trạng thái danh tính thực thể cho vết (trace) và lỗi công cụ."""
        return {
            "status": self.identity_status,
            "authorized_symbols": sorted(self.authorized_symbols),
            "records": [asdict(record) for record in self._identities.values()],
        }

    def authorize_tool_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        batch_authorized_symbols: Iterable[str],
        call_id: str,
        batch_identity_status: str | None = None,
    ) -> ToolAuthorization:
        """Ủy quyền dựa trên trạng thái danh tính đã được khóa trước đợt gọi LLM."""
        if tool_name == _RESOLVER_TOOL:
            self._identity_required = True
            self._buffer_output = True
            self._begin_resolution(str(arguments.get("query") or ""), call_id)
            return ToolAuthorization(allowed=True)

        if tool_name == "load_skill" and self._identity_required:
            authorized = {
                _normalize_symbol(item) for item in batch_authorized_symbols
            }
            frozen_status = batch_identity_status or self.identity_status
            if frozen_status != "locked" or not authorized:
                return ToolAuthorization(
                    allowed=False,
                    error_code="identity_required",
                    message=(
                        "Sensitive workflow selection is blocked until identity "
                        "was locked in an earlier assistant tool turn."
                    ),
                )
            return ToolAuthorization(allowed=True)

        symbols = tuple(self._extract_symbol_arguments(arguments))
        if not symbols:
            return ToolAuthorization(allowed=True)

        self._identity_required = True
        self._buffer_output = True
        authorized = {_normalize_symbol(item) for item in batch_authorized_symbols}
        frozen_status = batch_identity_status or self.identity_status
        if frozen_status != "locked" or not authorized:
            return ToolAuthorization(
                allowed=False,
                error_code=(
                    "identity_conflict"
                    if frozen_status in {"ambiguous", "conflicting", "invalidated"}
                    else "identity_required"
                ),
                message=(
                    "A canonical, non-conflicting identity was not locked before this "
                    "assistant tool-call batch started. A resolver result from this same "
                    "batch cannot be consumed."
                ),
                symbols=symbols,
            )

        mismatched = tuple(
            symbol
            for symbol in symbols
            if self._match_authorized_symbol(tool_name, symbol, authorized) is None
        )
        if mismatched:
            return ToolAuthorization(
                allowed=False,
                error_code="identity_mismatch",
                message=(
                    "Consumer symbol/domain differs from the locked resolver identity; "
                    "silent suffix or entity rewrites are forbidden."
                ),
                symbols=mismatched,
            )
        return ToolAuthorization(allowed=True, symbols=symbols)

    @staticmethod
    def _match_authorized_symbol(
        tool_name: str,
        requested_symbol: str,
        authorized_symbols: Iterable[str],
    ) -> str | None:
        """Ánh xạ đối số phía tiêu thụ sang symbol/mã thực thể chuẩn duy nhất đã được khóa."""
        requested = _normalize_symbol(requested_symbol)
        authorized = {_normalize_symbol(item) for item in authorized_symbols}
        if requested in authorized:
            return requested
        return None

    def ingest_tool_result(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: str,
        call_id: str,
        success: bool,
        batch_authorized_symbols: Sequence[str] | None = None,
    ) -> None:
        """Tiêu thụ kết quả công cụ đầy đủ chưa bị cắt gọt và lưu trữ bằng chứng số học (M-RS2 v2)."""
        payload = _json_object(result)

        if tool_name == _RESOLVER_TOOL:
            if not success:
                self._finish_failed_resolution(arguments, call_id)
            elif payload is not None:
                self._ingest_resolution(arguments, payload, call_id)

        if success and payload is not None:
            self._ingest_generic_numeric(tool_name, arguments, payload, call_id)
        elif not success:
            self._record_tool_failure(tool_name, call_id, result)

        self.persist()

    def validate_final_answer(self, content: str) -> ValidationResult:
        """Kiểm chứng các tuyên bố thực thể và khẳng định số liệu số học trong câu trả lời cuối cùng."""
        issues: list[dict[str, Any]] = []
        issues.extend(self._validate_identity(content))

        result = ValidationResult(valid=not issues, issues=issues)
        self._validations.append(
            {
                "attempt": len(self._validations) + 1,
                "checked_at": _utc_now(),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "valid": result.valid,
                "issues": issues,
            }
        )
        self.persist()
        return result

    def correction_prompt(self, validation: ValidationResult) -> str:
        """Xây dựng phản hồi có giới hạn cho một bản nháp mô hình bị từ chối."""
        lines = [
            "[GROUNDING GATE] The previous draft was rejected and was not released to the user.",
            "Correct every issue using the existing structured identity and tool evidence:",
        ]
        for issue in validation.issues[:12]:
            lines.append(f"- {issue.get('message', issue.get('reason', issue.get('code', 'grounding error')))}")
        lines.extend(
            [
                "REQUIREMENT: Every numeric claim MUST cite an observed cell via [[cell_id]] format (e.g. [[e:run1:call1:data.ic]]).",
                "Bare un-cited numbers are forbidden (except for years like 19xx/20xx or numbers marked with ~).",
                "Reuse the exact locked symbol and venue/domain.",
                "If evidence is unavailable or conflicting, say so and ask for clarification; do not guess.",
            ]
        )
        return "\n".join(lines)

    def safe_fallback(self) -> str:
        """Trả về câu trả lời dự phòng an toàn (fail-closed) mang tính xác định sau khi bị từ chối liên tiếp (Spec §4.3)."""
        if self._evidence_cells:
            cell_ids = [f"`{cid}` ({c.metric}: {c.status})" for cid, c in sorted(self._evidence_cells.items())[:10]]
            cell_summary = ", ".join(cell_ids)
            return (
                "I rejected the previous draft because numeric citations could not be verified against tool evidence. "
                f"Recorded evidence cell IDs and statuses: [{cell_summary}]. "
                "No unverified numeric claims will be generated without explicit visible derivation or refreshed evidence."
            )
        return (
            "I could not safely lock the entity identity or numeric evidence, so I did not "
            "produce a final conclusion. Please confirm the candidate entity and venue."
        )

    def persist(self) -> None:
        """Lưu trữ nguyên tử sổ nhật ký có cấu trúc hiện tại xuống đĩa."""
        artifact_dir = self.run_dir / "artifacts"
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / GROUNDING_ARTIFACT
            temp = path.with_suffix(path.suffix + ".tmp")
            payload = {
                "schema_version": 1,
                "updated_at": _utc_now(),
                "identity": self.identity_summary(),
                "evidence": [asdict(record) for record in self._evidence],
                "tool_failures": list(self._tool_failures),
                "validations": list(self._validations),
            }
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp.replace(path)
        except OSError:
            # Các quyết định grounding vẫn nằm trong bộ nhớ RAM; lỗi I/O không được làm hỏng luồng chạy.
            return

    def _seed_symbols(self, text: str, *, source: str) -> None:
        """Khóa các thực thể/mã định danh do người dùng cung cấp rõ ràng."""
        for match in self._entity_re.finditer(text or ""):
            symbol = _normalize_symbol(match.group(0))
            key = f"explicit:{symbol}"
            existing = self._identities.get(key)
            version = existing.version + 1 if existing else 1
            self._identities[key] = IdentityRecord(
                query=symbol,
                status="locked",
                symbol=symbol,
                venue=_infer_domain(symbol),
                instrument_type=_infer_entity_type(symbol),
                currency=_infer_unit(symbol),
                source_tool_call_id=source,
                source=[source],
                version=version,
            )
            self._identity_required = True
            self._buffer_output = True

    def _begin_resolution(self, query: str, call_id: str) -> None:
        """Chuyển sang trạng thái chưa phân giải trước khi bộ phân giải thực thi."""
        key = _query_key(query) or f"call:{call_id}"
        existing = self._identities.get(key)
        self._identities[key] = IdentityRecord(
            query=query,
            status="unresolved",
            source_tool_call_id=call_id,
            version=(existing.version + 1) if existing else 1,
        )
        self.persist()

    def _finish_failed_resolution(
        self,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> None:
        """Đánh dấu thất bại phân giải là vô hiệu."""
        query = str(arguments.get("query") or "")
        key = _query_key(query) or f"call:{call_id}"
        existing = self._identities.get(key)
        self._identities[key] = IdentityRecord(
            query=query,
            status="invalidated",
            source_tool_call_id=call_id,
            version=(existing.version + 1) if existing else 1,
        )

    def _ingest_resolution(
        self,
        arguments: Mapping[str, Any],
        payload: dict[str, Any] | None,
        call_id: str,
    ) -> None:
        """Cập nhật danh tính chưa phân giải từ kết quả bộ phân giải có cấu trúc."""
        data = payload.get("data") if isinstance(payload, dict) else None
        data = data if isinstance(data, dict) else {}
        query = str(data.get("query") or arguments.get("query") or "")
        key = _query_key(query) or f"call:{call_id}"
        existing = self._identities.get(key)
        version = (existing.version + 1) if existing else 1

        if not isinstance(payload, dict) or payload.get("ok") is False:
            self._identities[key] = IdentityRecord(
                query=query,
                status="invalidated",
                source_tool_call_id=call_id,
                version=version,
            )
            return

        raw_candidates = data.get("candidates")
        candidates = [dict(item) for item in raw_candidates if isinstance(item, dict)] if isinstance(raw_candidates, list) else []
        sources = data.get("sources") if isinstance(data.get("sources"), dict) else {}
        if not candidates:
            clean_sources = [
                str(name)
                for name, value in sources.items()
                if str(value).casefold() == "ok"
            ]
            self._identities[key] = IdentityRecord(
                query=query,
                status="not_found" if len(clean_sources) >= 2 else "invalidated",
                source_tool_call_id=call_id,
                source=clean_sources,
                candidates=[],
                version=version,
            )
            return

        chosen = self._choose_candidate(query, candidates)
        if chosen is None:
            self._identities[key] = IdentityRecord(
                query=query,
                status="ambiguous",
                source_tool_call_id=call_id,
                candidates=candidates,
                version=version,
            )
            return

        symbol = _normalize_symbol(chosen.get("symbol"))
        if not symbol:
            self._identities[key] = IdentityRecord(
                query=query,
                status="invalidated",
                source_tool_call_id=call_id,
                candidates=candidates,
                version=version,
            )
            return

        alias_conflicts = [
            record
            for record in self._identities.values()
            if record.status == "locked"
            and record.symbol
            and _is_entity_alias_conflict(record.symbol, symbol)
        ]
        if alias_conflicts:
            conflicting = list(candidates)
            conflicting.extend(
                {"symbol": record.symbol, "source": record.source}
                for record in alias_conflicts
            )
            self._identities[key] = IdentityRecord(
                query=query,
                status="conflicting",
                source_tool_call_id=call_id,
                candidates=conflicting,
                version=version,
            )
            return

        if existing and existing.status == "locked" and existing.symbol != symbol:
            conflicting = list(candidates)
            conflicting.insert(0, {"symbol": existing.symbol, "source": existing.source})
            self._identities[key] = IdentityRecord(
                query=query,
                status="conflicting",
                source_tool_call_id=call_id,
                candidates=conflicting,
                version=version,
            )
            return

        source_names = []
        for value in [chosen.get("source"), *(chosen.get("also_from") or [])]:
            name = str(value or "").strip()
            if name and name not in source_names:
                source_names.append(name)
        venue = str(chosen.get("exchange") or chosen.get("market") or chosen.get("domain") or "").strip() or _infer_domain(symbol)
        self._identities[key] = IdentityRecord(
            query=query,
            status="locked",
            symbol=symbol,
            venue=venue,
            instrument_type=_infer_entity_type(symbol, chosen.get("type")),
            currency=_infer_unit(symbol),
            source_tool_call_id=call_id,
            source=source_names,
            candidates=candidates,
            version=version,
        )

    @staticmethod
    def _choose_candidate(
        query: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Chỉ chọn ứng viên bộ phân giải duy nhất hoặc được đối soát mạnh mẽ."""
        if len(candidates) == 1:
            return candidates[0]
        normalized_query = re.sub(r"[^a-z0-9\u3400-\u9fff]", "", query.casefold())
        exact: list[dict[str, Any]] = []
        strong: list[dict[str, Any]] = []
        for candidate in candidates:
            symbol = _normalize_symbol(candidate.get("symbol"))
            base = symbol.split(".", 1)[0].split("-", 1)[0].split("/", 1)[0]
            name = str(candidate.get("name") or "")
            comparable = {
                re.sub(r"[^a-z0-9\u3400-\u9fff]", "", base.casefold()),
                re.sub(r"[^a-z0-9\u3400-\u9fff]", "", name.casefold()),
                re.sub(r"[^a-z0-9\u3400-\u9fff]", "", symbol.casefold()),
            }
            if normalized_query and normalized_query in comparable:
                exact.append(candidate)
            if candidate.get("also_from") or candidate.get("cik") or candidate.get("id"):
                strong.append(candidate)
        if len(exact) == 1:
            return exact[0]
        if len(strong) == 1:
            return strong[0]
        return None

    @staticmethod
    def _extract_symbol_arguments(arguments: Mapping[str, Any]) -> list[str]:
        """Trích xuất các danh tính/mã thực thể từ các khóa đối số đã biết."""
        symbols: list[str] = []
        for key, value in arguments.items():
            if str(key).casefold() not in _SYMBOL_ARGUMENT_KEYS:
                continue
            values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
            for item in values:
                if not isinstance(item, (str, int)):
                    continue
                symbol = _normalize_symbol(item)
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
        return symbols

    def _record_tool_failure(self, tool_name: str, call_id: str, result: str) -> None:
        """Lưu trữ bằng chứng không khả dụng có cấu trúc cho các công cụ bị lỗi."""
        payload = _json_object(result) or {}
        self._tool_failures.append(
            {
                "call_id": call_id,
                "tool": tool_name,
                "status": "unavailable",
                "error_code": payload.get("error_code"),
                "message": str(payload.get("error") or payload.get("message") or "tool failed")[:500],
                "recorded_at": _utc_now(),
            }
        )

    def _ingest_generic_numeric(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        payload: dict[str, Any],
        call_id: str,
    ) -> None:
        """Phẳng hóa các lá số học có giới hạn từ kết quả công cụ."""
        symbols = self._extract_symbol_arguments(arguments)
        symbol = symbols[0] if len(symbols) == 1 else None
        if symbol:
            symbol = self._match_authorized_symbol(
                tool_name,
                symbol,
                self.authorized_symbols,
            ) or symbol
        source = str(payload.get("source") or tool_name)
        remaining = _MAX_GENERIC_EVIDENCE

        def visit(value: Any, path: str) -> None:
            nonlocal remaining
            if remaining <= 0:
                return
            if _is_number(value):
                self._evidence.append(
                    EvidenceRecord(
                        call_id=call_id,
                        tool=tool_name,
                        symbol=symbol,
                        source=source,
                        timestamp=None,
                        field=path or "value",
                        value=value,
                        status="observed",
                        currency=_infer_unit(symbol or ""),
                        venue=_infer_domain(symbol or ""),
                    )
                )
                remaining -= 1
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}.{key}" if path else str(key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(payload, "")

    def _validate_identity(self, content: str) -> list[dict[str, Any]]:
        """Kiểm chứng trạng thái tổng hợp của danh tính thực thể."""
        issues: list[dict[str, Any]] = []
        status = self.identity_status
        if self._identity_required and status in {
            "unresolved",
            "ambiguous",
            "conflicting",
            "invalidated",
        }:
            issues.append(
                {
                    "code": "identity_not_locked",
                    "status": status,
                    "message": f"Entity identity is {status}; a final conclusion requires locked identity.",
                }
            )
        return issues

    def _validate_price_claims(self, content: str) -> list[dict[str, Any]]:
        """Kiểm tra các bảng Markdown và văn bản tuyên bố số liệu so với bản ghi đã quan sát."""
        issues, table_lines = self._validate_price_tables(content)
        records = self._price_records()
        has_price_claim = any(
            self._numbers_without_dates_or_percent(line)
            for index, line in enumerate(content.splitlines())
            if index in table_lines
        )
        for index, line in enumerate(content.splitlines()):
            if index in table_lines or "|" in line:
                continue
            line_symbol = self._symbol_for_claim(line, records)
            for segment in re.split(r"[,，;；。\n]", line):
                if not self._numeric_context_re.search(segment):
                    continue
                values = self._numbers_without_dates_or_percent(segment)
                if not values:
                    continue
                has_price_claim = True
                symbol = self._symbol_for_claim(segment, records) or line_symbol
                if self._is_explicit_derivation(segment, records, symbol):
                    continue
                for value in values:
                    issue = self._compare_price_claim(
                        value=value,
                        records=records,
                        field_name=None,
                        date_value=None,
                        symbol=symbol,
                        claim=segment.strip(),
                    )
                    if issue:
                        issues.append(issue)
        if has_price_claim and records:
            issues.extend(self._validate_price_provenance(content, records))
        return self._dedupe_issues(issues)

    def _symbol_for_claim(
        self,
        content: str,
        records: Sequence[EvidenceRecord],
    ) -> str | None:
        """Trả về một mã thực thể bằng chứng chuẩn được nêu rõ ràng trong một tuyên bố."""
        known = {record.symbol for record in records if record.symbol}
        matches = {
            _normalize_symbol(match.group(0))
            for match in self._entity_re.finditer(content)
            if _normalize_symbol(match.group(0)) in known
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def _validate_price_provenance(
        self,
        content: str,
        records: Sequence[EvidenceRecord],
    ) -> list[dict[str, Any]]:
        """Yêu cầu mã thực thể chuẩn, nguồn dữ liệu thực tế và đơn vị trong đầu ra."""
        issues: list[dict[str, Any]] = []
        folded = content.casefold()
        symbols = sorted({record.symbol for record in records if record.symbol})
        mentioned = [symbol for symbol in symbols if symbol.casefold() in folded]
        if not mentioned:
            issues.append(
                {
                    "code": "canonical_symbol_not_surfaced",
                    "symbols": symbols,
                    "message": (
                        "A numeric claim must surface its locked canonical symbol or domain."
                    ),
                }
            )
        target_symbols = set(mentioned or (symbols if len(symbols) == 1 else []))
        target_records = [
            record
            for record in records
            if not target_symbols or record.symbol in target_symbols
        ]

        sources = sorted(
            {
                record.source
                for record in target_records
                if record.source and record.source.casefold() not in {"auto", "unknown"}
            }
        )
        missing_sources = [source for source in sources if source.casefold() not in folded]
        if missing_sources:
            issues.append(
                {
                    "code": "data_source_not_surfaced",
                    "sources": missing_sources,
                    "message": (
                        "Numeric claims must name the actual data source: "
                        + ", ".join(missing_sources)
                        + "."
                    ),
                }
            )

        currencies = sorted(
            {record.currency for record in target_records if record.currency}
        )
        missing_currencies = [
            currency
            for currency in currencies
            if not self._currency_is_surfaced(currency, content)
        ]
        if missing_currencies:
            issues.append(
                {
                    "code": "currency_not_surfaced",
                    "currencies": missing_currencies,
                    "message": (
                        "Numeric claims must surface their unit or secondary identifier: "
                        + ", ".join(missing_currencies)
                        + "."
                    ),
                }
            )
        return issues

    @staticmethod
    def _currency_is_surfaced(currency: str, content: str) -> bool:
        """Kiểm tra xem đơn vị hoặc bí danh của thực thể có xuất hiện trong nội dung hay không."""
        if not currency:
            return True
        folded = content.casefold()
        return currency.casefold() in folded

    def _validate_price_tables(
        self,
        content: str,
    ) -> tuple[list[dict[str, Any]], set[int]]:
        """Kiểm chứng các tuyên bố theo trường/ngày cụ thể trong các bảng Markdown."""
        lines = content.splitlines()
        issues: list[dict[str, Any]] = []
        consumed: set[int] = set()
        index = 0
        records = self._price_records()
        while index + 1 < len(lines):
            header = self._table_cells(lines[index])
            separator = self._table_cells(lines[index + 1])
            if not header or not separator or len(header) != len(separator):
                index += 1
                continue
            if not all(_TABLE_SEPARATOR_RE.match(cell.replace(" ", "")) for cell in separator):
                index += 1
                continue
            field_columns = {
                position: _TABLE_FIELD_ALIASES[cell.strip().casefold()]
                for position, cell in enumerate(header)
                if cell.strip().casefold() in _TABLE_FIELD_ALIASES
            }
            if not field_columns:
                index += 1
                continue
            date_column = next(
                (position for position, cell in enumerate(header) if cell.strip().casefold() in _DATE_HEADERS),
                None,
            )
            symbol_column = next(
                (position for position, cell in enumerate(header) if cell.strip().casefold() in _SYMBOL_HEADERS),
                None,
            )
            consumed.update({index, index + 1})
            row_index = index + 2
            while row_index < len(lines):
                row = self._table_cells(lines[row_index])
                if not row or len(row) != len(header):
                    break
                consumed.add(row_index)
                date_value = row[date_column].strip() if date_column is not None else None
                symbol = _normalize_symbol(row[symbol_column]) if symbol_column is not None else None
                for position, field_name in field_columns.items():
                    values = self._numbers_without_dates_or_percent(row[position])
                    if len(values) != 1:
                        continue
                    issue = self._compare_price_claim(
                        value=values[0],
                        records=records,
                        field_name=field_name,
                        date_value=date_value,
                        symbol=symbol,
                        claim=row[position].strip(),
                    )
                    if issue:
                        issues.append(issue)
                row_index += 1
            index = max(row_index, index + 1)
        return issues, consumed

    @staticmethod
    def _table_cells(line: str) -> list[str]:
        """Tách các ô trong một dòng bảng Markdown."""
        if "|" not in line:
            return []
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def _compare_price_claim(
        self,
        *,
        value: float,
        records: list[EvidenceRecord],
        field_name: str | None,
        date_value: str | None,
        symbol: str | None,
        claim: str,
    ) -> dict[str, Any] | None:
        """So sánh một tuyên bố số liệu chưa dán nhãn với giá trị bằng chứng gần nhất."""
        candidates = records
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        symbols = sorted({record.symbol for record in candidates if record.symbol})
        if not symbol and len(symbols) == 1:
            symbol = symbols[0]
        elif not symbol and len(symbols) > 1:
            return {
                "code": "numeric_claim_ambiguous_symbol",
                "claim": claim,
                "value": value,
                "symbols": symbols,
                "message": (
                    f"Numeric claim {value:g} is ambiguous across multiple evidence symbols; "
                    "name the canonical symbol explicitly."
                ),
            }
        if field_name:
            candidates = [record for record in candidates if record.field == field_name]
        if date_value:
            candidates = [
                record
                for record in candidates
                if record.timestamp and record.timestamp.startswith(date_value)
            ]
        if not candidates:
            return {
                "code": "numeric_claim_unavailable",
                "claim": claim,
                "value": value,
                "symbol": symbol,
                "field": field_name,
                "date": date_value,
                "message": f"Numeric claim {value:g} has no matching observed tool evidence.",
            }
        observed = [float(record.value) for record in candidates if record.value is not None]
        if any(abs(value - item) <= max(abs(item) * 0.005, 1e-9) for item in observed):
            return None
        return {
            "code": "numeric_claim_conflict",
            "claim": claim,
            "value": value,
            "symbol": symbol,
            "field": field_name,
            "date": date_value,
            "observed_min": min(observed),
            "observed_max": max(observed),
            "source_tool_call_ids": sorted({record.call_id for record in candidates}),
            "message": (
                f"Numeric claim {value:g} conflicts with observed {field_name or 'numerical'} "
                f"evidence {min(observed):g}–{max(observed):g}."
            ),
        }

    def _price_records(self) -> list[EvidenceRecord]:
        """Trả về danh sách các bản ghi bằng chứng số học đã quan sát (M-RS2 §1 P5 fix)."""
        return [
            record
            for record in self._evidence
            if record.status == "observed"
            and record.value is not None
            and _is_number(record.value)
        ]

    @staticmethod
    def _numbers_without_dates_or_percent(text: str) -> list[float]:
        """Trích xuất các giá trị số trong khi loại trừ ngày tháng và phần trăm."""
        without_dates = _DATE_RE.sub(" ", text)
        values: list[float] = []
        for match in _NUMBER_RE.finditer(without_dates):
            tail = without_dates[match.end() :].lstrip()
            if tail.startswith(("%", "％")):
                continue
            try:
                values.append(float(match.group(0).replace(",", "")))
            except ValueError:
                continue
        return values

    def _is_explicit_derivation(
        self,
        text: str,
        records: Sequence[EvidenceRecord],
        symbol: str | None,
    ) -> bool:
        """Kiểm tra tính hợp lệ của công thức số học dẫn xuất dựa trên bằng chứng đã quan sát."""
        if not _DERIVATION_RE.search(text):
            return False
        candidates = list(records)
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        candidate_symbols = {record.symbol for record in candidates if record.symbol}
        if not symbol and len(candidate_symbols) > 1:
            return False
        observed = [
            float(record.value) for record in candidates if record.value is not None
        ]
        if not observed:
            return False

        for equals in re.finditer(r"=", text):
            left = re.search(r"([0-9.,+\-*/×÷()\s]+)$", text[: equals.start()])
            right = re.match(
                r"\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
                text[equals.end() :],
            )
            if not left or not right:
                continue
            evaluated = self._evaluate_formula(left.group(1))
            if evaluated is None:
                continue
            computed, inputs = evaluated
            try:
                claimed = float(right.group(1).replace(",", ""))
            except ValueError:
                continue
            if not any(
                abs(item - value) <= max(abs(value) * 0.005, 1e-9)
                for item in inputs
                for value in observed
            ):
                continue
            if abs(computed - claimed) <= max(abs(computed) * 0.005, 1e-9):
                return True
        return False

    @staticmethod
    def _evaluate_formula(expression: str) -> tuple[float, list[float]] | None:
        """Tính toán biểu thức số học mà không thực thi mã nguồn ngẫu nhiên (sử dụng AST)."""
        normalized = expression.replace("×", "*").replace("÷", "/").replace(",", "").strip()
        try:
            tree = ast.parse(normalized, mode="eval")
        except (SyntaxError, ValueError):
            return None
        inputs: list[float] = []

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and _is_number(node.value):
                value = float(node.value)
                inputs.append(value)
                return value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div),
            ):
                left_value = visit(node.left)
                right_value = visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left_value + right_value
                if isinstance(node.op, ast.Sub):
                    return left_value - right_value
                if isinstance(node.op, ast.Mult):
                    return left_value * right_value
                if right_value == 0:
                    raise ValueError("division by zero")
                return left_value / right_value
            raise ValueError("unsupported formula")

        try:
            value = visit(tree)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        if len(inputs) < 2 or not math.isfinite(value):
            return None
        return value, inputs

    @staticmethod
    def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Loại bỏ các phát hiện trùng lặp của trình kiểm chứng trong khi bảo toàn thứ tự."""
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for issue in issues:
            key = json.dumps(issue, sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            unique.append(issue)
        return unique


# Lớp bí danh hỗ trợ tổng quát hóa và khả năng tương thích ngược
GroundingVerifier = GroundingLedger

__all__ = [
    "GROUNDING_ARTIFACT",
    "EvidenceRecord",
    "GroundingLedger",
    "GroundingVerifier",
    "IdentityRecord",
    "ToolAuthorization",
    "ValidationResult",
]
