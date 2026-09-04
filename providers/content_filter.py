"""Xử lý sự kiện bộ lọc nội dung của nhà cung cấp LLM (M-RS0 §9).

Nhà cung cấp có thể từ chối trả lời một lượt vì bộ lọc an toàn của họ. Với L2.6 đây
KHÔNG phải lỗi hạ tầng và cũng KHÔNG phải phán xét alpha — nó là một lượt bị mất.
Ba thứ mà `agent/loop.py` và `swarm/worker.py` cần từ đây:

* `CONTENT_FILTER_SKIP_MESSAGE` — nội dung chèn vào lịch sử hội thoại thay cho lượt bị
  chặn. Phải khác rỗng: chuỗi rỗng làm hỏng cặp tool_call/tool_result và nhiều nhà cung
  cấp từ chối nguyên request khi có message rỗng.
* `MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS` — cầu dao. Bị chặn liên tiếp đủ số lần thì
  dừng hẳn thay vì quay vòng đốt token.
* `compute_content_filter_warnings()` — cảnh báo hậu kiểm khi tỉ lệ bị chặn vượt ngưỡng.

Module thuần logic, không chạm mạng và không phụ thuộc nhà cung cấp nào.
"""

from __future__ import annotations

CONTENT_FILTER_SKIP_MESSAGE = (
    "[Lượt này bị bộ lọc nội dung của nhà cung cấp chặn nên không có phản hồi. "
    "Hãy diễn đạt lại bằng thuật ngữ định lượng trung tính rồi tiếp tục.]"
)

# Ba lần liên tiếp là đủ để kết luận đây không phải trục trặc thoáng qua.
MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS = 3


def compute_content_filter_warnings(
    content_filter_count: int,
    total_iterations: int,
    threshold: float | None = None,
) -> list[str]:
    """Trả danh sách cảnh báo khi tỉ lệ lượt bị bộ lọc chặn quá cao.

    Args:
        content_filter_count: Số lượt bị chặn trong lượt chạy.
        total_iterations: Tổng số vòng lặp đã thực hiện (bên gọi bảo đảm >= 1).
        threshold: Ngưỡng tỉ lệ. Mặc định lấy từ
            `AgentTuningConfig.content_filter_warning_threshold`.

    Returns:
        Danh sách chuỗi cảnh báo; rỗng nghĩa là không có gì bất thường.

    Vì sao trả list chứ không raise: bị lọc nội dung không làm kết quả nghiên cứu sai,
    nó chỉ làm kết quả *mỏng đi*. Ném lỗi ở đây sẽ biến một lượt chạy dùng được thành
    thất bại, đúng kiểu "hết ngân sách ⇒ ERROR ⇒ KILL" mà §6.1 cấm.
    """
    if content_filter_count <= 0:
        return []

    if total_iterations <= 0:
        total_iterations = 1

    if threshold is None:
        from pseud.config.accessor import get_env_config

        threshold = get_env_config().agent_tuning.content_filter_warning_threshold

    ti_le = content_filter_count / total_iterations
    canh_bao: list[str] = []

    if ti_le >= threshold:
        canh_bao.append(
            f"Bộ lọc nội dung chặn {content_filter_count}/{total_iterations} lượt "
            f"({ti_le:.0%} ≥ ngưỡng {threshold:.0%}). Kết quả lượt chạy này thiếu dữ liệu "
            f"so với bình thường — đừng đọc nó như một kết luận đầy đủ."
        )

    if content_filter_count >= MAX_CONSECUTIVE_CONTENT_FILTER_SKIPS:
        canh_bao.append(
            f"Đã chạm {content_filter_count} lượt bị chặn. Nếu chúng liên tiếp thì cầu dao "
            f"đã ngắt lượt chạy; hãy xem lại cách diễn đạt prompt trước khi chạy lại."
        )

    return canh_bao
