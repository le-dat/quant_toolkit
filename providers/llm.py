"""Nối biến môi trường nhà cung cấp LLM và chẩn đoán cấu hình.

`preflight.py` import đúng ba hàm ở đây (`_ensure_dotenv`, `_sync_provider_env`,
`provider_diagnostics`) trong khối `try/except ImportError` rồi rơi về stub. Stub trả
`base_url=""`, `timeout=0`, nên preflight báo "ready" cho một cấu hình chưa hề tồn tại —
đúng kiểu fail-open mà Nguyên tắc 3 cấm. Module này thay stub bằng số liệu thật.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Khoá tương thích. Nguồn sự thật của L2.6 vẫn là LANGCHAIN_*; các biến còn lại là bí danh vào.
# Bí danh CHUNG — thử sau bí danh riêng theo nhà cung cấp.
_BI_DANH_KHOA = ("LANGCHAIN_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
_BI_DANH_BASE = ("LANGCHAIN_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL")

# Bí danh RIÊNG theo nhà cung cấp. Người dùng đặt `OPENROUTER_API_KEY` là quy ước tự nhiên
# nhất khi `LANGCHAIN_PROVIDER=openrouter`; không phủ tên này thì hệ thống báo "chưa cấu
# hình" trong khi khoá nằm ngay đó — một lỗi không có triệu chứng nào chỉ đúng chỗ.
_BI_DANH_KHOA_THEO_NCC = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": ("OLLAMA_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "zhipu": ("ZHIPU_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "kimi-coding": ("KIMI_CODING_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "mimo": ("MIMO_API_KEY",),
    "spark": ("SPARK_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "requesty": ("REQUESTY_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
}

_BI_DANH_BASE_THEO_NCC = {
    "gemini": ("GEMINI_BASE_URL",),
    "openrouter": ("OPENROUTER_BASE_URL",),
    "deepseek": ("DEEPSEEK_BASE_URL",),
    "groq": ("GROQ_BASE_URL",),
    "together": ("TOGETHER_BASE_URL",),
    "openai": ("OPENAI_BASE_URL",),
    "azure": ("AZURE_OPENAI_BASE_URL",),
    "dashscope": ("DASHSCOPE_BASE_URL",),
    "zhipu": ("ZHIPU_BASE_URL",),
    "moonshot": ("MOONSHOT_BASE_URL",),
    "kimi-coding": ("KIMI_CODING_BASE_URL",),
    "minimax": ("MINIMAX_BASE_URL",),
    "mimo": ("MIMO_BASE_URL",),
    "spark": ("SPARK_BASE_URL",),
    "zai": ("ZAI_BASE_URL",),
    "requesty": ("REQUESTY_BASE_URL",),
    "nvidia": ("NVIDIA_BASE_URL",),
}


_da_nap_dotenv = False


def _ensure_dotenv(env_path: str | Path | None = None) -> bool:
    """Nạp `.env` vào môi trường tiến trình, chỉ một lần.

    Returns:
        True nếu đã nạp được một tệp `.env`, False nếu không tìm thấy.
    """
    global _da_nap_dotenv
    if _da_nap_dotenv and env_path is None:
        return True

    duong_dan = Path(env_path) if env_path else _tim_dotenv()
    if duong_dan is None or not duong_dan.exists():
        return False

    from dotenv import load_dotenv

    load_dotenv(duong_dan, override=True)

    _da_nap_dotenv = True
    return True


def _tim_dotenv() -> Path | None:
    """Tìm `.env` ngược lên từ thư mục gói tới gốc kho mã hoặc qua biến môi trường."""
    for env_var in ("KAIROS_ENV_FILE", "DOTENV_PATH", "ENV_FILE"):
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            p = Path(env_val).expanduser().resolve()
            if p.exists():
                return p

    for thu_muc in [Path.cwd(), *Path(__file__).resolve().parents]:
        for ten_tuc in (".env", "agent/.env", ".env.local"):
            ung_vien = thu_muc / ten_tuc
            if ung_vien.exists():
                return ung_vien
    return None


def _nap_dotenv_toi_gian(duong_dan: Path) -> None:
    """Bộ đọc `.env` dự phòng khi thiếu `python-dotenv`."""
    try:
        noi_dung = duong_dan.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.debug("Không đọc được %s: %s", duong_dan, exc)
        return

    for dong in noi_dung.splitlines():
        dong = dong.strip()
        if not dong or dong.startswith("#") or "=" not in dong:
            continue
        khoa, gia_tri = dong.split("=", 1)
        khoa = khoa.strip()
        gia_tri = gia_tri.strip().strip("'\"")
        if khoa:
            os.environ[khoa] = gia_tri


def _sync_provider_env() -> None:
    """Đồng bộ bí danh khoá/điểm cuối về biến chuẩn `LANGCHAIN_*`.

    Thứ tự thử khoá: `LANGCHAIN_API_KEY` → bí danh RIÊNG của nhà cung cấp đang chọn
    (`OPENROUTER_API_KEY` khi provider là `openrouter`…) → bí danh chung.

    Bí danh riêng phải đứng trước bí danh chung: máy có sẵn `OPENAI_API_KEY` từ việc khác
    mà đang chạy `LANGCHAIN_PROVIDER=openrouter` sẽ gửi nhầm khoá OpenAI tới OpenRouter và
    nhận 401 — lỗi trông như "khoá sai" trong khi thật ra là "lấy nhầm khoá".
    """
    _ensure_dotenv()

    provider = os.environ.get("LANGCHAIN_PROVIDER", "").strip().lower()

    if not provider:
        for p_name, keys in _BI_DANH_KHOA_THEO_NCC.items():
            for k in keys:
                if os.environ.get(k, "").strip():
                    provider = p_name
                    os.environ["LANGCHAIN_PROVIDER"] = provider
                    break
            if provider:
                break

    rieng_khoa = _BI_DANH_KHOA_THEO_NCC.get(provider, ())
    rieng_base = _BI_DANH_BASE_THEO_NCC.get(provider, ())

    for nhom, chuan in (
        (rieng_khoa + _BI_DANH_KHOA, "LANGCHAIN_API_KEY"),
        (rieng_base + _BI_DANH_BASE, "LANGCHAIN_BASE_URL"),
    ):
        if os.environ.get(chuan, "").strip():
            continue
        for bi_danh in nhom:
            gia_tri = os.environ.get(bi_danh, "").strip()
            if gia_tri:
                os.environ[chuan] = gia_tri
                break

    try:
        from pseud.config.accessor import reset_env_config
        reset_env_config()
    except Exception:
        pass


def provider_diagnostics() -> dict:
    """Trả thông tin cấu hình nhà cung cấp cho preflight (không lộ khoá).

    Returns:
        dict có `base_url`, `timeout_seconds`, `max_retries`, `proxy`, `api_key_set`.
    """
    from pseud.config.accessor import get_env_config, reset_env_config
    from pseud.providers.chat import _giai_quyet_base_url

    # Tự đồng bộ rồi mới nạp lại cấu hình. Không làm thế thì kết quả phụ thuộc việc bên
    # gọi có nhớ gọi `_sync_provider_env()` trước hay không, và `api_key_set` sẽ báo False
    # cho một máy đã cấu hình đầy đủ — chẩn đoán nói dối là chẩn đoán tệ hơn không có.
    _sync_provider_env()
    reset_env_config()

    cfg = get_env_config().llm
    provider = (cfg.langchain_provider or "openai").strip().lower()

    proxy = {
        ten: os.environ.get(ten, "")
        for ten in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
        if os.environ.get(ten, "")
    }

    return {
        "provider": provider,
        "base_url": _giai_quyet_base_url(provider, cfg.langchain_base_url),
        "timeout_seconds": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
        "proxy": proxy,
        # Chỉ báo CÓ hay KHÔNG, tuyệt đối không trả giá trị khoá.
        "api_key_set": bool((cfg.langchain_api_key or "").strip()),
    }
