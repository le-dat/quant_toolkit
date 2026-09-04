"""Hàm hỗ trợ chính sách cho mục tiêu nghiên cứu tài chính (Kairos v3)."""

from __future__ import annotations


def normalize_required_text(value: str, field_name: str) -> str:
    """Loại bỏ khoảng trắng dư thừa và xác minh trường văn bản bắt buộc.

    Args:
        value: Văn bản do người dùng cung cấp.
        field_name: Tên trường để đưa vào thông báo lỗi.

    Returns:
        Giá trị đã chuẩn hóa.

    Raises:
        ValueError: Nếu văn bản bị rỗng sau khi loại bỏ khoảng trắng.
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


# Ý định giao dịch thật. L2.6 chỉ nghiên cứu và đẩy `HypothesisSpec` sang L2.5; không có
# đường nào từ đây tới sàn. Mục tiêu mang ý định đặt lệnh phải bị chặn ở khâu nhận, chứ
# không phải chặn bằng việc hy vọng không ai viết ra.
_Y_DINH_GIAO_DICH_THAT = (
    "đặt lệnh", "vào lệnh", "khớp lệnh", "giao dịch thật", "tiền thật", "tài khoản thật",
    "chuyển tiền", "rút tiền", "nạp tiền", "đòn bẩy thật", "lệnh thị trường", "lệnh giới hạn",
    "place order", "submit order", "execute trade", "live trading", "real money",
    "real account", "withdraw", "deposit", "market order", "limit order", "send order",
)


def reject_live_execution_objective(objective: str) -> None:
    """Từ chối mục tiêu mang ý định giao dịch thật.

    Args:
        objective: Văn bản mục tiêu do người dùng đặt.

    Raises:
        ValueError: Khi văn bản rỗng, hoặc chứa ý định đặt lệnh/chuyển tiền thật.

    Bản trước chỉ gọi `normalize_required_text` rồi trả về — tên hàm hứa chặn lệnh chạy
    thật nhưng thân hàm chỉ kiểm chuỗi rỗng. Đó là guard fail-open: nó tạo cảm giác đã có
    rào chắn nên không ai dựng rào thật, cùng loại với `in_pit()` trả True khi PiT rỗng.
    """
    text = normalize_required_text(objective, "objective")
    thap = text.lower()

    for cum in _Y_DINH_GIAO_DICH_THAT:
        if cum in thap:
            raise ValueError(
                f"Mục tiêu chứa ý định giao dịch thật ({cum!r}). Layer 2.6 chỉ nghiên cứu "
                f"và bàn giao HypothesisSpec sang Layer 2.5; nó không có đường nào tới sàn. "
                f"Hãy viết lại mục tiêu dưới dạng câu hỏi nghiên cứu kiểm được."
            )
