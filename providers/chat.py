"""ChatLLM — hợp đồng nhà cung cấp LLM tối thiểu cho L2.6 (M-RS0 §9).

Hợp đồng dưới đây **suy ra từ chỗ gọi thật trong mã**, không chép từ văn bản đặc tả:
§9 ghi `stream(...)` nhưng [loop.py:761](../agent/loop.py) và [:794] gọi `stream_chat(...)`.
Khi hai nguồn lệch nhau thì mã là nguồn sự thật, vì mã là thứ sẽ chạy.

Bề mặt mà `AgentLoop` và `swarm/worker.py` trông đợi:

* `ChatLLM.model_name: str`
* `ChatLLM.runtime_snapshot: LLMRuntimeSnapshot`
* `ChatLLM.chat(messages) -> ChatResponse`                          (dùng bởi `_auto_compact`)
* `ChatLLM.stream_chat(messages, tools, on_text_chunk,
                       on_reasoning_chunk, should_cancel) -> ChatResponse`
* `ChatResponse.content / .has_tool_calls / .tool_calls /
   .reasoning_content / .response_model / .usage_metadata /
   .content_filter_triggered`
* `ProviderStreamError.retryable / .provider / .model`

**Vì sao dùng `urllib` của thư viện chuẩn thay vì SDK.** Bất biến I1 cấm `requests`,
`httpx`, `ccxt` trong `pseud/src/`, và test `test_I1_cam_import_http_rest_trong_pseud`
quét AST toàn thư mục nên vi phạm là CI đỏ ngay. Ngoài ra máy này chưa cài SDK LLM nào
(`openai`, `langchain`, `anthropic` đều thiếu), nên chọn `urllib` giữ cho tầng này chạy
được mà không thêm phụ thuộc.

**Backend cắm rời.** Bộ điều hợp mặc định nói giao thức tương thích OpenAI — cùng một
giao thức mà OpenAI, DeepSeek, OpenRouter, Together, Groq, Azure và vLLM/Ollama nội bộ
đều nói. Đổi nhà cung cấp trong nhóm đó chỉ cần đổi `LANGCHAIN_BASE_URL`, không đụng mã.
Nhà cung cấp có giao thức khác (ví dụ Anthropic Messages API) thì đăng ký thêm một bộ
điều hợp vào `_BACKENDS` — xem `_khong_ho_tro()`.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Điểm cuối mặc định theo tên nhà cung cấp, dùng khi LANGCHAIN_BASE_URL rỗng.
_BASE_URL_MAC_DINH = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.ai/v1",
    "kimi-coding": "https://api.kimi.com/coding/v1",
    "minimax": "https://api.minimax.io/v1",
    "mimo": "https://api.xiaomimimo.com/v1",
    "spark": "https://spark-api-open.xf-yun.com/v1",
    "zai": "https://api.z.ai/api/coding/paas/v4",
    "requesty": "https://router.requesty.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}

# Nhà cung cấp nói giao thức tương thích OpenAI.
_TUONG_THICH_OPENAI = frozenset(_BASE_URL_MAC_DINH) | {"openai_compatible", "vllm", "azure"}

# Mã HTTP đáng thử lại: quá tải, giới hạn nhịp, lỗi phía máy chủ.
_HTTP_TAM_THOI = frozenset({408, 409, 413, 425, 429, 500, 502, 503, 504})


class ProviderStreamError(Exception):
    """Lỗi khi gọi/stream nhà cung cấp.

    `retryable` là thứ `AgentLoop` đọc để quyết định thử lại một lần
    ([loop.py:768](../agent/loop.py)). Phân biệt tạm/vĩnh viễn là bắt buộc: thử lại
    một lỗi 400 chỉ đốt gấp đôi token rồi vẫn hỏng.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.status_code = status_code


@dataclass(frozen=True)
class LLMRuntimeSnapshot:
    """Ảnh chụp cấu hình nhà cung cấp tại thời điểm dựng `AgentLoop`.

    `AgentLoop.__init__` tự dựng bản dự phòng khi `llm` không mang sẵn
    ([loop.py:542-555](../agent/loop.py)), nên ba trường này là hợp đồng cứng.
    """

    provider: str
    configured_model: str
    reasoning_effort: str = ""


@dataclass
class ToolCall:
    """Một lời gọi công cụ do mô hình phát ra."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    extra_content: dict[str, Any] | None = None



@dataclass
class ChatResponse:
    """Kết quả một lượt gọi LLM.

    `usage_metadata` phải là **dict** với ba khoá `input_tokens`/`output_tokens`/
    `total_tokens`: `_normalize_llm_usage` gọi `usage.get(...)` trực tiếp
    ([loop.py:151-153](../agent/loop.py)). Trả object có thuộc tính sẽ làm nó im lặng
    bỏ qua toàn bộ số liệu, và kiểm soát ngân sách (Nguyên tắc 6) mất nền.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    response_model: str = ""
    usage_metadata: dict[str, int] | None = None
    content_filter_triggered: bool = False
    finish_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _cau_hinh():
    from pseud.config.accessor import get_env_config

    return get_env_config().llm


def _giai_quyet_base_url(provider: str, cau_hinh_base: str) -> str:
    """Trả điểm cuối cuối cùng: cấu hình đè mặc định theo nhà cung cấp."""
    base = (cau_hinh_base or "").strip().rstrip("/")
    if base:
        return base
    return _BASE_URL_MAC_DINH.get(provider, "")


def _khong_ho_tro(provider: str):
    raise ProviderStreamError(
        f"Chưa có bộ điều hợp cho nhà cung cấp '{provider}'. Hai lựa chọn: "
        f"(1) đặt LANGCHAIN_BASE_URL trỏ tới một điểm cuối tương thích OpenAI, hoặc "
        f"(2) thêm một bộ điều hợp vào _BACKENDS trong pseud/src/providers/chat.py. "
        f"Các giá trị đã hỗ trợ: {', '.join(sorted(_TUONG_THICH_OPENAI))}.",
        retryable=False,
        provider=provider,
    )


class ChatLLM:
    """Client LLM cắm rời theo `LANGCHAIN_PROVIDER`.

    Khởi tạo KHÔNG gọi mạng — dựng được cả khi chưa cấu hình khoá, để `preflight` và
    test đường ống tất định chạy được mà không cần nhà cung cấp. Thiếu khoá chỉ nổ khi
    thật sự gọi, và nổ với thông báo chỉ đúng biến môi trường phải đặt.
    """

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
    ):
        try:
            from pseud.providers.llm import _sync_provider_env
            _sync_provider_env()
        except Exception:
            pass

        cfg = _cau_hinh()
        self.provider = (provider or cfg.langchain_provider or "openai").strip().lower()
        self.model_name = (model_name or cfg.langchain_model_name or "").strip()
        self._api_key = (api_key if api_key is not None else cfg.langchain_api_key).strip()
        self.base_url = _giai_quyet_base_url(
            self.provider, base_url if base_url is not None else cfg.langchain_base_url
        )
        self.timeout_seconds = timeout_seconds or cfg.timeout_seconds
        self.max_retries = cfg.max_retries if max_retries is None else max_retries
        self.temperature = cfg.langchain_temperature if temperature is None else temperature

        self.runtime_snapshot = LLMRuntimeSnapshot(
            provider=self.provider,
            configured_model=self.model_name,
            reasoning_effort=(cfg.langchain_reasoning_effort or "").strip().lower(),
        )

        if self.provider not in _TUONG_THICH_OPENAI and not (base_url or cfg.langchain_base_url):
            self._backend = None  # nổ ở lần gọi đầu, không nổ lúc dựng
        else:
            self._backend = "openai_compatible"

    # ── Bề mặt công khai ────────────────────────────────────────────────────

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        """Gọi không streaming. Dùng bởi `_auto_compact` ([loop.py:1809](../agent/loop.py))."""
        payload = self._dung_payload(messages, tools, stream=False)
        du_lieu = self._goi_co_thu_lai(payload, stream=False)
        return self._doc_phan_hoi_day_du(du_lieu)

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatResponse:
        """Gọi có streaming, phát từng mẩu qua callback rồi trả phản hồi đã gộp.

        `should_cancel` được hỏi giữa các mẩu. Hủy giữa chừng trả về phần đã nhận thay
        vì ném lỗi — `AgentLoop` kiểm `_cancel_event` ngay sau khi gọi và tự bỏ lượt đó
        ([loop.py:801-803](../agent/loop.py)); ném lỗi ở đây sẽ biến "người dùng bấm hủy"
        thành "nhà cung cấp hỏng".
        """
        payload = self._dung_payload(messages, tools, stream=True)
        return self._goi_stream_co_thu_lai(
            payload, on_text_chunk, on_reasoning_chunk, should_cancel
        )

    # ── Dựng yêu cầu ────────────────────────────────────────────────────────

    def _dung_payload(
        self, messages: list[dict], tools: list[dict] | None, stream: bool
    ) -> dict[str, Any]:
        if self.provider not in _TUONG_THICH_OPENAI and not self.base_url:
            _khong_ho_tro(self.provider)
        if not self.model_name:
            raise ProviderStreamError(
                "Chưa đặt LANGCHAIN_MODEL_NAME trong .env",
                retryable=False,
                provider=self.provider,
            )
        cleaned_messages = []
        for m in messages:
            if isinstance(m, dict):
                clean_m = {k: v for k, v in m.items() if k not in ("extra_content", "additional_kwargs")}
                if "tool_calls" in clean_m and isinstance(clean_m["tool_calls"], list) and clean_m["tool_calls"]:
                    clean_tcs = []
                    for tc in clean_m["tool_calls"]:
                        if isinstance(tc, dict):
                            tc_copy = {k: v for k, v in tc.items() if k not in ("additional_kwargs",)}
                            extra = tc_copy.get("extra_content")
                            if not isinstance(extra, dict):
                                extra = {}
                            sig = extra.get("thought_signature") or "thought_sig_preserved"
                            extra["thought_signature"] = sig
                            google_obj = extra.get("google")
                            if not isinstance(google_obj, dict):
                                google_obj = {}
                            google_obj["thought_signature"] = sig
                            extra["google"] = google_obj
                            tc_copy["extra_content"] = extra
                            clean_tcs.append(tc_copy)
                        else:
                            clean_tcs.append(tc)
                    clean_m["tool_calls"] = clean_tcs
                cleaned_messages.append(clean_m)
            else:
                cleaned_messages.append(m)

        max_tokens_val = int(os.environ.get("LANGCHAIN_MAX_TOKENS", "4096"))
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": cleaned_messages,
            "stream": stream,
            "max_tokens": max_tokens_val,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            # Không có cờ này thì bản tin stream KHÔNG mang usage, và llm_usage.json
            # rỗng ⇒ BudgetTracker không có gì để trừ.
            payload["stream_options"] = {"include_usage": True}
        effort = self.runtime_snapshot.reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    def _mo_ket_noi(self, payload: dict[str, Any]):
        if self._backend is None:
            _khong_ho_tro(self.provider)

        if not self._api_key:
            raise ProviderStreamError(
                "Chưa đặt LANGCHAIN_API_KEY trong .env",
                retryable=False,
                provider=self.provider,
                model=self.model_name,
            )

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(
                payload,
                ensure_ascii=False
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key.strip()}",
                "Accept": (
                    "text/event-stream"
                    if payload.get("stream")
                    else "application/json"
                ),
                "User-Agent": "Kairos-v3/1.0",
            },
            method="POST",
        )

        return urllib.request.urlopen(
            req,
            timeout=self.timeout_seconds,
            context=ssl.create_default_context(),
        )

    # ── Thử lại ─────────────────────────────────────────────────────────────

    def _bao_loi(self, exc: Exception) -> ProviderStreamError:
        """Quy lỗi tầng vận chuyển về ProviderStreamError, có phân loại tạm/vĩnh viễn."""
        if isinstance(exc, urllib.error.HTTPError):
            than = ""
            try:
                than = exc.read().decode("utf-8", errors="ignore")[:400]
            except Exception:
                pass
            lower_than = than.lower()
            is_rate_limit_or_payload = (
                exc.code in (413, 429)
                or "rate_limit" in lower_than
                or "tpm" in lower_than
                or "rpm" in lower_than
                or "tokens per minute" in lower_than
                or "payload too large" in lower_than
                or "request too large" in lower_than
            )
            retryable = (exc.code in _HTTP_TAM_THOI) or is_rate_limit_or_payload
            return ProviderStreamError(
                f"HTTP {exc.code} từ {self.provider}: {than or exc.reason}",
                retryable=retryable,
                provider=self.provider,
                model=self.model_name,
                status_code=exc.code,
            )
        # URLError / timeout / ket noi dut — deu la loi tam thoi.
        return ProviderStreamError(
            f"Lỗi kết nối tới {self.provider}: {exc}",
            retryable=True,
            provider=self.provider,
            model=self.model_name,
        )

    def _cho_truoc_lan_thu(self, lan: int, status_code: int | None = None) -> None:
        if status_code in (413, 429):
            delay = min(5.0 * (2.0 ** lan), 45.0)
        else:
            delay = min(2.0 ** lan, 30.0)
        time.sleep(delay)


    def _goi_co_thu_lai(self, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
        loi_cuoi: ProviderStreamError | None = None
        for lan in range(self.max_retries + 1):
            try:
                with self._mo_ket_noi(payload) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except ProviderStreamError:
                raise
            except Exception as exc:
                loi_cuoi = self._bao_loi(exc)
                if loi_cuoi.status_code == 413 or not loi_cuoi.retryable or lan == self.max_retries:
                    raise loi_cuoi
                logger.warning("Gọi %s thất bại (lần %s), thử lại: %s", self.provider, lan + 1, loi_cuoi)
                self._cho_truoc_lan_thu(lan, loi_cuoi.status_code)
        if loi_cuoi is not None:
            raise loi_cuoi
        raise ProviderStreamError("Thất bại không xác định", provider=self.provider, model=self.model_name)

    def _goi_stream_co_thu_lai(
        self,
        payload: dict[str, Any],
        on_text_chunk: Callable[[str], None] | None,
        on_reasoning_chunk: Callable[[str], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> ChatResponse:
        loi_cuoi: ProviderStreamError | None = None
        for lan in range(self.max_retries + 1):
            try:
                with self._mo_ket_noi(payload) as resp:
                    return self._doc_stream(resp, on_text_chunk, on_reasoning_chunk, should_cancel)
            except ProviderStreamError:
                raise
            except Exception as exc:
                loi_cuoi = self._bao_loi(exc)
                if loi_cuoi.status_code == 413 or not loi_cuoi.retryable or lan == self.max_retries:
                    raise loi_cuoi
                logger.warning("Stream %s thất bại (lần %s), thử lại: %s", self.provider, lan + 1, loi_cuoi)
                self._cho_truoc_lan_thu(lan, loi_cuoi.status_code)
        if loi_cuoi is not None:
            raise loi_cuoi
        raise ProviderStreamError("Stream thất bại không xác định", provider=self.provider, model=self.model_name)



    # ── Phân giải phản hồi ──────────────────────────────────────────────────

    @staticmethod
    def _chuan_hoa_usage(usage: dict | None) -> dict[str, int] | None:
        """Đổi tên trường usage của giao thức OpenAI sang khoá mà loop.py đọc."""
        if not usage:
            return None
        vao = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ra = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tong = int(usage.get("total_tokens") or 0) or (vao + ra)
        if not (vao or ra or tong):
            return None
        return {"input_tokens": vao, "output_tokens": ra, "total_tokens": tong}

    @staticmethod
    def _doc_tool_calls(raw: Iterable[dict] | None) -> list[ToolCall]:
        ket_qua: list[ToolCall] = []
        for tc in raw or []:
            fn = tc.get("function") or {}
            tham_so = fn.get("arguments")
            if isinstance(tham_so, str):
                try:
                    tham_so = json.loads(tham_so) if tham_so.strip() else {}
                except json.JSONDecodeError:
                    # Mô hình đôi khi phát JSON hỏng. Giữ nguyên văn để tầng trên báo lỗi
                    # công cụ tử tế, thay vì làm sập cả lượt chạy tại đây.
                    tham_so = {"_raw_arguments": tham_so}
            extra_content = tc.get("extra_content")
            ket_qua.append(
                ToolCall(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=tham_so or {},
                    extra_content=extra_content if isinstance(extra_content, dict) else None,
                )
            )
        return ket_qua

    def _doc_phan_hoi_day_du(self, du_lieu: dict[str, Any]) -> ChatResponse:
        lua_chon = (du_lieu.get("choices") or [{}])[0]
        msg = lua_chon.get("message") or {}
        ly_do = lua_chon.get("finish_reason") or ""
        return ChatResponse(
            content=msg.get("content") or "",
            tool_calls=self._doc_tool_calls(msg.get("tool_calls")),
            reasoning_content=msg.get("reasoning_content") or msg.get("reasoning") or "",
            response_model=du_lieu.get("model") or self.model_name,
            usage_metadata=self._chuan_hoa_usage(du_lieu.get("usage")),
            content_filter_triggered=(ly_do == "content_filter"),
            finish_reason=ly_do,
        )

    def _doc_stream(
        self,
        resp,
        on_text_chunk: Callable[[str], None] | None,
        on_reasoning_chunk: Callable[[str], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> ChatResponse:
        """Gộp các sự kiện SSE thành một ChatResponse hoàn chỉnh."""
        van_ban: list[str] = []
        suy_luan: list[str] = []
        tool_dang_gom: dict[int, dict] = {}
        usage = None
        model = self.model_name
        ly_do = ""

        for dong_raw in resp:
            if should_cancel is not None and should_cancel():
                break

            dong = dong_raw.decode("utf-8", errors="ignore").strip()
            if not dong or not dong.startswith("data:"):
                continue
            than = dong[5:].strip()
            if than == "[DONE]":
                break
            try:
                goi = json.loads(than)
            except json.JSONDecodeError:
                continue

            if goi.get("model"):
                model = goi["model"]
            if goi.get("usage"):
                usage = goi["usage"]

            for lua_chon in goi.get("choices") or []:
                delta = lua_chon.get("delta") or {}
                if lua_chon.get("finish_reason"):
                    ly_do = lua_chon["finish_reason"]

                mau_van = delta.get("content")
                if mau_van:
                    van_ban.append(mau_van)
                    if on_text_chunk:
                        on_text_chunk(mau_van)

                mau_suy_luan = delta.get("reasoning_content") or delta.get("reasoning")
                if mau_suy_luan:
                    suy_luan.append(mau_suy_luan)
                    if on_reasoning_chunk:
                        on_reasoning_chunk(mau_suy_luan)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    cho = tool_dang_gom.setdefault(
                        idx, {"id": "", "function": {"name": "", "arguments": ""}}
                    )
                    if tc.get("id"):
                        cho["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        cho["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        # Tham số tool tới theo từng mẩu chuỗi, phải NỐI chứ không gán đè.
                        cho["function"]["arguments"] += fn["arguments"]
                    extra = tc.get("extra_content") or tc.get("extra")
                    if isinstance(extra, dict):
                        cho_extra = cho.setdefault("extra_content", {})
                        cho_extra.update(extra)


        return ChatResponse(
            content="".join(van_ban),
            tool_calls=self._doc_tool_calls(
                [tool_dang_gom[i] for i in sorted(tool_dang_gom)]
            ),
            reasoning_content="".join(suy_luan),
            response_model=model,
            usage_metadata=self._chuan_hoa_usage(usage),
            content_filter_triggered=(ly_do == "content_filter"),
            finish_reason=ly_do,
        )


_BACKENDS = {"openai_compatible": ChatLLM}
