"""ContextBuilder: Xây dựng ngữ cảnh tin nhắn cho LLM phục vụ ReAct AgentLoop."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pseud.agent.memory import WorkspaceMemory
from pseud.agent.skills import SkillsLoader
from pseud.agent.tools import ToolRegistry

if TYPE_CHECKING:
    from pseud.memory.persistent import PersistentMemory

logger = logging.getLogger(__name__)

# ── System prompt Layer 2.6 ──────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are the autonomous quantitative research agent of Kairos Layer 2.6.

Your job: find real anomalies in the internal lakehouse, build testable alpha hypotheses, and hand
them to Layer 2.5 for backtesting. You do NOT run backtests, you do NOT place orders, and you do NOT
call external APIs.

## Tools ({tool_count})

{tool_descriptions}

## Skills ({skill_count}) — call `load_skill` to read one in full

{skill_descriptions}

## Session state

{memory_summary}

## Four rules you may never break

1. **Missing data is NOT a bad alpha.** An absent column, an empty partition, or a feature that is not
   live-ready means the conclusion is `INSUFFICIENT_DATA`. Never turn that into "this idea is weak".
   Confusing the two blocks the path to good alphas and the damage only surfaces months later.
2. **Every number needs an evidence citation (HARD RULE).** Every numeric value in your final answer
   MUST be cited by evidence cell id in the form [[cell_id]] — for example [[e:run1:call1:data.stat]].
   Bare uncited numbers are forbidden, except years (19xx/20xx) and values explicitly flagged
   approximate with `~`. Never recall a number from model memory. The CitationGate REJECTS any draft
   containing a bare number outside a citation span.
3. **No conclusion before you touch data.** Do not claim "this factor has alpha" before you have a
   real statistical anomaly value or statistic. A plausible economic story is not evidence — a language model can write a
   plausible story for any formula whatsoever.
4. **A cemetery collision is a hard block.** Look up `doc_nghia_trang` before proposing a hypothesis.
   A dead idea proposed again is stopped by the gate at dispatch, not here.

## Workflow

1. **Scan for anomalies** — `scan_lakehouse_anomalies`. Read the returned `tests_independent`
   carefully: it is the multiple-testing multiplicity and it feeds the anomaly verification hurdle directly.
   Report it; never drop it.
2. **Record evidence and cite it** — route every material figure through `evidence_ledger`. The final
   answer MUST cite via [[cell_id]] tags; the renderer substitutes the value together with the metric
   name.
3. **Check for duplicates** — `doc_nghia_trang` and `doc_so_dang_ky_alpha`.
4. **Build the hypothesis** — follow `alpha-hypothesis-writer` and `alpha-grammar-contract`.
   `predicted_sign` and `conditional_prediction` must be declared up front. They are pre-registered
   commitments, hashed into the fingerprint BEFORE anything runs; choosing them after seeing results
   is cheating yourself.
5. **Diagnose failures** — follow `alpha-diagnostic-loop`. A mutated hypothesis must pass the cemetery
   gate again; mutations fall back into dead territory very easily.

## Which skill to load, and when

- Before naming any column, symbol, or market fact → `lakehouse-ground-truth`
- Before interpreting a scan result or a `SKIPPED` / `INSUFFICIENT_DATA` outcome → `anomaly-scan-protocol`
- Before writing any number into an answer → `evidence-citation-protocol`
- Before naming an alpha family or a template key → `alpha-family-taxonomy`
- Before emitting a `HypothesisSpec` → `alpha-hypothesis-writer`
- Before assuming a gate blocked something → `cemetery-containment-guard`
- Before handing anything to Layer 2.5 → `alpha-grammar-contract`
- After a KILL, before proposing a mutation → `alpha-diagnostic-loop`
- Before treating any tool output as a measurement → `tool-surface-and-limits`

Load the skill first, then act. The skills carry the measured numbers; your memory does not.

## Tool trust

A tool returning `"status": "ok"` does not mean it measured anything. Several tools on the debate path
currently return hardcoded constants. Read `tool-surface-and-limits` before citing any tool output,
and never turn a placeholder constant into an evidence cell. Treat a top-level `ok: false`,
`success: false`, or `status: error` as a genuine failure, not as an empty-but-valid result.

## How to answer

- Write your answer in English, including when the user writes in another language.
- Present multi-row data as markdown tables (`| col | col |` plus `|---|---|`).
- If critical information is missing (symbol, timeframe, alpha family), ASK — do not guess.
- Do not use `---` as a horizontal rule.
- State explicitly what you could NOT verify instead of passing over it in silence.
{memory_section}
## Current date and time

Today is {current_datetime}.
"""

_MEMORY_SECTION = """
## Ký ức liên phiên

{snapshot}

"""


class ContextBuilder:
    """Xây dựng ngữ cảnh tin nhắn cho AgentLoop.

    Thuộc tính:
        registry: Tool registry.
        memory: Workspace memory.
        skills_loader: Skills loader.
    """

    def __init__(self, registry: ToolRegistry, memory: WorkspaceMemory,
                 skills_loader: Optional[SkillsLoader] = None,
                 persistent_memory: Optional[PersistentMemory] = None) -> None:
        """Khởi tạo ContextBuilder.

        Args:
            registry: Tool registry.
            memory: Workspace memory.
            skills_loader: Bộ nạp kỹ năng (tự động tạo nếu không được cung cấp).
            persistent_memory: Instance PersistentMemory dùng cho ghi nhớ liên phiên.
        """
        self.registry = registry
        self.memory = memory
        self.skills_loader = skills_loader or SkillsLoader()
        self._persistent_memory = persistent_memory

    def build_system_prompt(self, user_message: str = "") -> str:
        """Xây dựng system prompt.

        Tiêm các tóm tắt kỹ năng 1 dòng qua get_descriptions; tài liệu đầy đủ được nạp theo yêu cầu bằng load_skill.
        Snapshot PersistentMemory được đóng đóng cố định khi bắt đầu phiên (giữ cache prompt).

        Args:
            user_message: Tin nhắn của người dùng (giữ lại để tương thích API).

        Returns:
            Văn bản system prompt.
        """
        now = datetime.now(timezone.utc)

        # Xây dựng phần memory chỉ khi có lưu trữ ký ức
        memory_section = ""
        if self._persistent_memory and self._persistent_memory.snapshot:
            memory_section = _MEMORY_SECTION.format(
                snapshot=self._persistent_memory.snapshot,
            )

        return _SYSTEM_PROMPT.format(
            tool_count=len(self.registry._tools),
            skill_count=len(self.skills_loader.skills),
            tool_descriptions=self._format_tool_descriptions(),
            skill_descriptions=self.skills_loader.get_descriptions(),
            memory_summary=self.memory.to_summary(),
            memory_section=memory_section,
            current_datetime=now.strftime("%A, %B %d, %Y %H:%M UTC"),
        )

    # `_count_data_sources()` ĐÃ GỠ. Docstring của nó nói "trích xuất từ VALID_SOURCES của
    # loader registry (nguồn sự thật duy nhất)", nhưng thân hàm dựng set gõ cứng ngay tại
    # chỗ rồi đếm — một con số tự tin mà không đo gì (M-RS0 §1.2). Prompt L2.6 không còn
    # quảng cáo số nguồn dữ liệu, vì lakehouse hiện chỉ có BINANCE (M-RS1 §1).

    def build_messages(self, user_message: str, history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Xây dựng danh sách tin nhắn đầy đủ.

        Tự động truy xuất các ký ức liên quan và tiêm vào tin nhắn của người dùng như ngữ cảnh.
        Điều này giữ cho system prompt ổn định (có thể cache) trong khi cung cấp các ký ức
        phù hợp theo từng truy vấn.

        Args:
            user_message: Tin nhắn của người dùng.
            history: Lịch sử cuộc trò chuyện trước đó.

        Returns:
            Danh sách tin nhắn theo định dạng OpenAI.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt(user_message)},
        ]
        if history:
            messages.extend(history)

        # Tự động truy xuất: tiêm các ký ức liên quan vào tin nhắn người dùng
        enriched = user_message
        if self._persistent_memory:
            try:
                recalls = self._persistent_memory.find_relevant(user_message, max_results=3)
                if recalls:
                    lines = [f"- **{r.title}** ({r.memory_type}): {r.body[:500]}" for r in recalls]
                    recall_block = "\n".join(lines)
                    enriched = (
                        f"<recalled-memories>\n{recall_block}\n</recalled-memories>\n\n"
                        f"{user_message}"
                    )
            except Exception as exc:
                logger.debug("Auto-recall failed: %s", exc)

        messages.append({"role": "user", "content": enriched})
        return messages

    def _format_tool_descriptions(self) -> str:
        """Định dạng mô tả công cụ."""
        lines = []
        for tool in self.registry._tools.values():
            params = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            param_parts = []
            for pname, pschema in params.items():
                req = " (required)" if pname in required else ""
                param_parts.append(f"    - {pname}: {pschema.get('description', pschema.get('type', ''))}{req}")
            param_text = "\n".join(param_parts) if param_parts else "    (no params)"
            lines.append(f"### {tool.name}\n{tool.description}\n  Params:\n{param_text}")
        return "\n\n".join(lines)

    @staticmethod
    def format_tool_result(tool_call_id: str, tool_name: str, result: str) -> Dict[str, Any]:
        """Định dạng kết quả thực thi công cụ thành một tin nhắn."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }

    @staticmethod
    def format_assistant_tool_calls(
        tool_calls: list,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Định dạng tin nhắn tool_calls của assistant, bảo toàn nội dung suy nghĩ (thinking text).

        Args:
            tool_calls: Danh sách các đối tượng gọi công cụ (tool call).
            content: Văn bản cuối cùng của assistant (có thể chứa chuỗi suy nghĩ inline).
            reasoning_content: Trường suy nghĩ đặc thù của nhà cung cấp (Kimi K2.5,
                DeepSeek reasoner, Qwen thinking). Chỉ gắn vào tin nhắn đầu ra khi khác None.

        Returns:
            Tin nhắn assistant theo định dạng OpenAI.
        """
        formatted_tool_calls = []
        has_extra_content = False
        for tc in tool_calls:
            tool_call = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            extra_content = getattr(tc, "extra_content", None)
            if extra_content:
                tool_call["extra_content"] = dict(extra_content)
                has_extra_content = True
            formatted_tool_calls.append(tool_call)

        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": formatted_tool_calls,
        }
        if has_extra_content:
            message["additional_kwargs"] = {
                "tool_calls": copy.deepcopy(formatted_tool_calls),
            }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        return message

