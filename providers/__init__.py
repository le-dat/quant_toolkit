"""Tầng nhà cung cấp LLM cho Layer 2.6 (M-RS0 §9).

`agent/loop.py` và `swarm/worker.py` import gói này trong khối `try/except ImportError`
và rơi về stub khi thiếu. Stub khiến agent chạy được nhưng câm — nên gói này tồn tại để
biến "câm im lặng" thành "chạy thật, hoặc báo lỗi rõ ràng".
"""

from pseud.providers.chat import (
    ChatLLM,
    ChatResponse,
    LLMRuntimeSnapshot,
    ProviderStreamError,
    ToolCall,
)
from pseud.providers.content_filter import (
    CONTENT_FILTER_SKIP_MESSAGE,
    MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS,
    compute_content_filter_warnings,
)

__all__ = [
    "ChatLLM",
    "ChatResponse",
    "LLMRuntimeSnapshot",
    "ProviderStreamError",
    "ToolCall",
    "CONTENT_FILTER_SKIP_MESSAGE",
    "MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS",
    "compute_content_filter_warnings",
]
