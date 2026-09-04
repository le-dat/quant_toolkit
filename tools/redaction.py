"""Tiện ích ẩn dữ liệu nhạy cảm (Shared redaction helpers - CWE-209/497, P10).

Ba chức năng độc lập ở đây:
1. ``redact_internal_paths`` — Thay thế các tiền tố đường dẫn nội bộ bằng sentinel ``'<redacted>'``.
2. ``redact_payload`` / ``is_sensitive_arg`` — Khử đệ quy các khóa nhạy cảm khỏi payload công cụ.
3. ``redact_text`` / ``redact_tool_result`` — Khử theo mẫu cho các bề mặt văn bản tự do.
"""

from __future__ import annotations

import json
import re
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path
from typing import Any

_SENTINEL = "<redacted>"

_REDACTED = "[redacted]"

#: Các định danh Sink cho :func:`is_sensitive_arg` / :func:`redact_payload`.
ARGUMENTS_SINK = "arguments"
RESULT_SINK = "result"

#: Các tên trường PII / định danh tài khoản khớp chính xác.
_PII_EXACT_KEYS = {
    # Các định danh tài khoản môi giới chung.
    "account_number",
    "account_id",
    "account_no",
    "account_num",
    # Mã định danh chính phủ / thuế.
    "ssn",
    "social_security_number",
    "tax_id",
    "taxpayer_id",
    "tin",
    "routing_number",
    "bank_account_number",
}

_SENSITIVE_ARG_KEYS = {
    "api_key",
    "authorization",
    "content",
    "env",
    "headers",
    "passphrase",
    "password",
    "secret",
    "token",
} | _PII_EXACT_KEYS

_SENSITIVE_ARG_MARKERS = ("api_key", "authorization", "password", "secret", "token")


#: Các khóa vẫn nhạy cảm trong sink ARGUMENTS / kiểm toán nhưng được giải phóng trong
#: sink tool-RESULT.
#:
#: ``content`` là trường payload của các bao bọc kết quả ``read_file`` / ``read_document`` /
#: ``read_url`` / ``load_skill``, do đó ẩn nó theo tên sẽ làm trống chính văn bản mà
#: trace tồn tại để giải thích — đó là sự ẩn quá mức thực tế mà sự nới lỏng này khắc phục.
#: Trong sink đối số, cùng một khóa mang *đầu vào* của tool: ``write_file(content=…)`` là
#: toàn bộ tài liệu người dùng, thông tin xác thực được tạo, hoặc phần thân skill riêng tư,
#: do đó nó vẫn được ẩn ở đó và trong sổ nhật ký kiểm toán trực tiếp.
#:
#: ``env`` cố ý KHÔNG có ở đây: các giá trị của nó là bí mật theo cả hai chiều
#: (dữ liệu env bị rò rỉ trong kết quả giống hệt như một đối số env).
_RESULT_SAFE_KEYS = {"content"}


def _fold_key(name: str) -> str:
    """Gập một key về lõi chữ-số (chữ thường, loại bỏ ký tự phân cách).

    Thu gọn biến thể snake_case, camelCase, kebab-case và khoảng trắng về một dạng
    để ``account_number``, ``accountNumber``, ``account-number`` và
    ``Account Number`` đều khớp với cùng một mục được tuyển chọn — mà không cần dùng
    chuỗi con rộng ``"account"`` gây ẩn quá mức các trường vô hại. Không dùng regex
    (không có rủi ro ReDoS), nhất quán với phần còn lại của mô-đun này.

    Args:
        name: Tên key thô.

    Returns:
        Dạng gập chữ-số viết thường của ``name``.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


#: Dạng gập ký tự phân cách của các key nhạy cảm + nhãn đánh dấu, khớp với
#: key ứng viên dạng gập để bắt được cả biến thể camelCase/kebab.
_SENSITIVE_ARG_KEYS_FOLDED = frozenset(_fold_key(k) for k in _SENSITIVE_ARG_KEYS)
_SENSITIVE_ARG_MARKERS_FOLDED = tuple(_fold_key(m) for m in _SENSITIVE_ARG_MARKERS)
#: Dạng gập của :data:`_RESULT_SAFE_KEYS`. Chỉ khớp chính xác dạng gập, để
#: ``secret_content`` / ``content_token`` tiếp tục bị ẩn ở mọi nơi.
_RESULT_SAFE_KEYS_FOLDED = frozenset(_fold_key(k) for k in _RESULT_SAFE_KEYS)


@lru_cache(maxsize=8)
def _internal_roots_for_cwd(cwd: str) -> tuple[str, ...]:
    """Tính toán các tiền tố thư mục gốc nội bộ cho một thư mục làm việc.

    Args:
        cwd: Thư mục làm việc đưa vào danh sách thư mục gốc (khóa cache).

    Returns:
        Tiền tố thư mục gốc, dài nhất trước, theo cả hai kiểu phân cách đường dẫn.
    """
    # AGENT_DIR được truy xuất giống như pseud/src/providers/llm.py:90
    # (Path(__file__).resolve().parents[2]); redaction.py nằm ở pseud/src/tools/
    # nên parents[2] == thư mục agent/. Mốc lấy theo cấu trúc layout, không hardcode.
    agent_dir = Path(__file__).resolve().parents[2]
    cands = [
        Path.home(),
        Path(cwd),
        agent_dir,
        agent_dir.parent,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sysconfig.get_paths().get("purelib", "")),
        Path(sysconfig.get_paths().get("platlib", "")),
    ]
    roots: set[str] = set()
    for c in cands:
        s = str(c)
        if len(s) > 3 and s not in (".", "/", "\\"):
            roots.add(s)
            roots.add(s.replace("\\", "/"))
            roots.add(s.replace("/", "\\"))
    return tuple(sorted(roots, key=len, reverse=True))


def _internal_roots() -> list[str]:
    """Trả về các tiền tố thư mục gốc nội bộ cần ẩn, dài nhất trước.

    Ghi nhớ (memoized) theo từng thư mục làm việc thay vì toàn cục: một tập hợp được cache ở lần
    gọi đầu tiên sẽ bắt lấy ``Path.cwd()`` tại thời điểm đó và dừng ẩn âm thầm
    sau một lệnh ``os.chdir``, trong khi việc tính toán lại không điều kiện sẽ
    chạy lại ``sysconfig.get_paths()`` cho mỗi chuỗi cần ẩn trên đường dẫn nóng kết quả công cụ.
    Lấy key cho cache theo cwd đang hoạt động giúp duy trì cả hai đặc tính.

    Returns:
        Tiền tố gốc được sắp xếp từ dài nhất đến ngắn nhất, để thư mục con được thay thế trước
        thư mục cha.
    """
    return list(_internal_roots_for_cwd(str(Path.cwd())))


def redact_internal_paths(text: object) -> str:
    """Thay thế tiền tố thư mục gốc nội bộ bằng '<redacted>', giữ nguyên phần đuôi tương đối.

    Mọi sự xuất hiện của một gốc nội bộ được thay thế khi nó kết thúc tại ranh giới đường dẫn thực:
    dấu phân cách (``/Users/me/x`` → ``<redacted>/x``), phần cuối của
    chuỗi (khớp chính xác thư mục gốc — đường dẫn cwd hoặc home đơn lẻ cũng là một
    rò rỉ cấu trúc), hoặc bất kỳ ký tự nào khác không thể tiếp tục một phần tử
    đường dẫn (dấu ngoặc kép, dấu phẩy, khoảng trắng, dấu ngoặc đóng …). Một gốc
    chỉ là tiền tố *tên* của một phần tử dài hơn (``/Users/metest/x``) sẽ được giữ nguyên
    nguyên vẹn; việc ẩn quá mức đó là điều mà kiểm tra ranh giới khắc phục.

    Args:
        text: Giá trị cần ẩn. ``None`` trở thành chuỗi rỗng; giá trị không phải chuỗi
            được ép kiểu qua :func:`str`.

    Returns:
        Chuỗi đã được làm sạch. Có tính idempotent — chuỗi đánh dấu không chứa gốc, nên
        chạy lại không bao giờ lồng các sự thay thế.
    """
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    if not s:
        return s
    for root in _internal_roots():
        if root not in s:
            continue
        out: list[str] = []
        idx = 0
        while (pos := s.find(root, idx)) >= 0:
            end = pos + len(root)
            nxt = s[end : end + 1]
            after = s[end + 1 : end + 2]
            # Một phần tử đường dẫn tiếp tục trên một ký tự từ (word character), hoặc trên
            # ``.``/``-`` mà chính nó được theo sau bởi một ký tự từ (``/Users/me.bak/x``
            # là một thư mục khác, trong khi dấu vết ``/Users/me.`` ở
            # cuối câu vẫn là thư mục gốc và phải được ẩn).
            if nxt and (
                nxt.isalnum()
                or nxt == "_"
                or (nxt in "-." and (after.isalnum() or after == "_"))
            ):
                out.append(s[idx:end])
            else:
                out.append(s[idx:pos])
                out.append(_SENTINEL)
            idx = end
        out.append(s[idx:])
        s = "".join(out)
    return s


def is_sensitive_arg(name: str, *, sink: str = ARGUMENTS_SINK) -> bool:
    """Trả về việc liệu tên đối số công cụ / key payload có nên được ẩn hay không.

    Một key là nhạy cảm khi dạng chuẩn hóa (stripped, lower-cased) của nó hoặc

    * khớp chính xác với một key chứng thư đã biết (``api_key`` …) hoặc một trường
      tài khoản/PII được tuyển chọn (``account_number``, ``ssn``, ``routing_number`` … xem
      :data:`_PII_EXACT_KEYS`), hoặc
    * chứa chuỗi con đánh dấu chứng thư (nên ``api_token``,
      ``access_token``, ``x-authorization`` đều được ẩn).

    Việc khớp tài khoản/PII cố ý chỉ khớp chính xác — không bao giờ dùng chuỗi con rộng
    ``"account"`` — để các trường vô hại không bị ẩn quá mức và trường nguồn gốc
    ``account_ref`` mờ đục của bản ghi kiểm toán (chuỗi trách nhiệm giải trình mandate→consent, SPEC §5) được bảo toàn.

    Args:
        name: Tên đối số hoặc key payload cần phân loại.
        sink: Nơi giá trị sẽ đến. :data:`ARGUMENTS_SINK` (mặc định)
            là tập hợp nghiêm ngặt dùng cho đối số gọi công cụ và sổ nhật ký kiểm toán live;
            :data:`RESULT_SINK` bổ sung thêm việc giải phóng các vỏ bọc kết quả
            :data:`_RESULT_SAFE_KEYS`. Bất kỳ giá trị nào khác đều được
            xử lý như :data:`ARGUMENTS_SINK` (thất bại ở trạng thái đóng/fail closed).

    Returns:
        ``True`` khi key chứa chứng thư, số tài khoản hoặc giá trị PII khác
        và giá trị của nó phải được thay thế bằng ``'[redacted]'`` trước
        khi hiển thị.
    """
    normalized = name.strip().lower()
    folded = _fold_key(name)
    if sink == RESULT_SINK and folded in _RESULT_SAFE_KEYS_FOLDED:
        return False
    if normalized in _SENSITIVE_ARG_KEYS or any(
        marker in normalized for marker in _SENSITIVE_ARG_MARKERS
    ):
        return True
    # Khớp gập dấu phân cách bắt các biến thể camelCase / kebab / khoảng trắng
    # (accountNumber, account-number, socialSecurityNumber) mà tập hợp exact
    # snake_case sẽ bỏ sót, mà không làm ẩn quá mức chuỗi con rộng.
    return folded in _SENSITIVE_ARG_KEYS_FOLDED or any(
        marker in folded for marker in _SENSITIVE_ARG_MARKERS_FOLDED
    )


def redact_payload(obj: Any, *, sink: str = ARGUMENTS_SINK) -> Any:
    """Ẩn đệ quy các key nhạy cảm trong một payload có cấu trúc.

    Duyệt dicts và lists; bất kỳ giá trị dict nào có key thỏa :func:`is_sensitive_arg`
    sẽ được thay thế bằng chuỗi đánh dấu ``'[redacted]'``. Các giá trị không phải container giữ nguyên.
    Được sử dụng trước khi stringify bản xem trước sự kiện và trước
    khi ghi payload yêu cầu/phản hồi broker vào sổ nhật ký kiểm toán live.

    Làm sạch hai lớp key (xem :func:`is_sensitive_arg`): key chứng thư
    (OAuth tokens, ``api_key``, ``authorization``, ``password``/``secret``,
    và bất kỳ ký hiệu dạng ``*token*`` nào) và một tập hợp các tên trường
    tài khoản/PII chính xác (``account_number``, ``ssn``, ``routing_number`` …). Nó
    KHÔNG ẩn trường nguồn gốc ``account_ref`` mờ đục của bản ghi kiểm toán,
    vốn được giữ lại cho chuỗi trách nhiệm giải trình mandate→consent (SPEC §5).

    Args:
        obj: Payload tùy ý (dict / list / scalar) cần làm sạch.
        sink: Nơi payload sẽ đến; được chuyển tiếp đến
            :func:`is_sensitive_arg`. Trong :data:`RESULT_SINK`, các lá chuỗi còn lại
            được làm sạch thêm theo pattern với
            :func:`redact_text`, vì kết quả công cụ JSON thường chứa
            văn bản tự do (shell ``stdout``, thân lỗi) dưới một key vô hại mà
            không có quy tắc dựa trên key nào có thể phân loại.

    Returns:
        Cấu trúc mới có cùng hình dạng với các giá trị nhạy cảm đã được thay thế. Đầu vào không bao giờ bị đột biến.
    """
    if isinstance(obj, dict):
        return {
            key: _REDACTED
            if is_sensitive_arg(str(key), sink=sink)
            else redact_payload(item, sink=sink)
            for key, item in obj.items()
        }
    if isinstance(obj, list):
        return [redact_payload(item, sink=sink) for item in obj]
    if isinstance(obj, str):
        # Cả hai sink: phân loại dựa trên key không thể thấy chứng thư nhúng
        # bên trong một giá trị vốn vô hại, ví dụ token bearer trong đối số
        # lệnh shell hoặc trong chuỗi lỗi broker gửi đến sổ nhật ký.
        return redact_text(obj)
    return obj


#: Tên key có dạng thông tin xác thực được công nhận trong VĂN BẢN TỰ DO. Các lựa chọn dài hơn
#: đứng trước để ``access_token`` không bị cắt thành ``token`` và
#: ``secret_key`` không bị cắt thành ``secret``. Danh sách được cố ý
#: giới hạn trong tên trường thông tin xác thực để số vô hại trong đầu ra shell (giá,
#: thời lượng, nhãn thời gian) không bao giờ bị biến dạng.
#:
#: Tất cả tên đều ở số ÍT và được tân mốc ``\b``: số nhiều là bộ đếm hoặc
#: tập hợp, không bao giờ là thông tin xác thực, và việc khớp nó làm hỏng đầu ra thông thường
#: (``tokens: 1204 in / 318 out`` từ dòng sử dụng LLM, ``api_keys: 3``).
_TEXT_CREDENTIAL_KEYS = (
    r"api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token"
    r"|refresh[_-]?token|bearer[_-]?token|id[_-]?token|session[_-]?token"
    r"|client[_-]?secret|secret[_-]?key|private[_-]?key|app[_-]?secret"
    r"|passphrase|password|passwd|authorization|secret|token"
)

#: Dạng giá trị: chuỗi nằm trong dấu ngoặc kép, ngoặc đơn, hoặc token trần.
#: Nhánh trần dừng lại ở khoảng trắng, dấu câu danh sách/dict/hàm
#: và ngoặc đơn/kép, nên nó không thể nuốt khóa JSON tiếp theo cũng như không — do
#: ``[`` và ``]`` bị loại trừ — khớp lại một chuỗi ``[redacted]`` đã được chèn.
_TEXT_VALUE = r"\"[^\"\n]*\"|'[^'\n]*'|[^\s,;{}\[\]()\"']+"

#: Các cặp thông tin xác thực ``key=value`` / ``key: value`` / ``"key": "value"``. Khóa
#: có thể ở trong dấu ngoặc (JSON) miễn là cặp ngoặc đối xứng, được khớp bằng
#: tham chiếu ngược (backreference). Phép lookahead phủ định giữ cho việc thay thế mang tính
#: idempotent (giá trị đã ẩn sẽ bị bỏ qua) và nhường ``Bearer <jwt>`` cho
#: :data:`_TEXT_BEARER_PATTERN`, nơi mà nếu không có nó sẽ bị nuốt dưới dạng giá trị
#: và để lộ phần thân token.
#:
#: Mọi bộ định lượng áp dụng cho một lớp ký tự có giới hạn, không lồng nhau và
#: không chồng lấp các phương án thay thế, để việc khớp giữ tính tuyến tính (không có ReDoS).
_TEXT_KV_PATTERN = re.compile(
    r"(?P<q>[\"']?)\b(?P<name>" + _TEXT_CREDENTIAL_KEYS + r")\b(?P=q)"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?![\"']?" + re.escape(_REDACTED) + r"[\"']?|bearer\b)"
    r"(?P<value>" + _TEXT_VALUE + r")",
    re.IGNORECASE,
)

#: Các giá trị header ``Authorization: Bearer <token>`` và ``bearer <token>`` trần.
#: Áp dụng trước :data:`_TEXT_KV_PATTERN`; lớp giá trị loại trừ ``[`` để
#: lượt quét thứ hai không thể khớp với ``Bearer [redacted]``.
_TEXT_BEARER_PATTERN = re.compile(
    r"\b(?P<scheme>bearer)\s+(?P<value>[A-Za-z0-9._~+/-]+=*)",
    re.IGNORECASE,
)


def _sub_credential_pair(match: re.Match[str]) -> str:
    """Tái xây dựng một kết quả khớp ``key=value`` với giá trị đã được ẩn.

    Dấu ngoặc kép của key và value được bảo toàn, để đoạn JSON đã ẩn vẫn phân tích được
    (``{"api_key": "[redacted]"}``) và lượt chạy thứ hai tạo ra chuỗi đồng nhất.

    Args:
        match: Kết quả khớp được tạo bởi :data:`_TEXT_KV_PATTERN`.

    Returns:
        Đoạn ``key``/phân cách/sentinel đã được xây dựng lại.
    """
    value = match.group("value")
    quote = value[0] if value[:1] in ("'", '"') else ""
    key_quote = match.group("q")
    return (
        f"{key_quote}{match.group('name')}{key_quote}{match.group('sep')}"
        f"{quote}{_REDACTED}{quote}"
    )


#: Định dạng thông tin xác thực được nhận diện KHÔNG có nhãn khóa. Khớp dựa trên khóa và
#: ``key=value`` đều bỏ sót token trần được dán vào đầu ra shell hoặc
#: chuỗi lỗi, vì vậy các tiền tố phát hành nổi tiếng được khớp theo dạng cấu trúc. Mỗi
#: phương án thay thế được neo và giới hạn độ dài để văn xuôi thông thường không bị khớp nhầm.
_TEXT_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])("
    r"sk-[A-Za-z0-9_-]{20,}"                 # OpenAI / Anthropic style
    r"|gh[pousr]_[A-Za-z0-9]{30,}"           # GitHub PAT family
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"         # Slack
    r"|AKIA[0-9A-Z]{16}"                     # AWS access key id
    r"|pypi-[A-Za-z0-9_-]{16,}"              # PyPI upload token
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"  # JWT
    r")(?![A-Za-z0-9_-])"
)


def redact_text(text: object) -> str:
    """Tẩy sạch các giá trị có dạng thông tin xác thực khỏi một chuỗi định dạng tự do.

    Được sử dụng cho kết quả tool dạng văn bản thuần, đầu ra shell, thông điệp lỗi và các dòng log
    nơi :func:`redact_payload` là thao tác rỗng (nó chỉ duyệt key có cấu trúc).
    Giá trị header Bearer và các cặp thông tin xác thực ``key=value`` / ``key: value`` /
    ``"key": "value"`` được thay thế bằng ``'[redacted]'``
    trong khi văn bản xung quanh — nhãn, đường dẫn, ngữ cảnh lỗi — vẫn giữ
    đọc được.

    Args:
        text: Chuỗi định dạng tự do (hoặc bất kỳ đối tượng nào được ép kiểu qua :func:`str`).
            ``None`` trở thành chuỗi rỗng để khớp với
            :func:`redact_internal_paths`.

    Returns:
        Chuỗi đã được ẩn. Đẳng sâm (idempotent): ẩn một chuỗi đã ẩn
        sẽ trả về chuỗi đó không đổi. Không bao giờ ném ngoại lệ trên đầu vào không phải chuỗi.
    """
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    if not s:
        return s
    s = _TEXT_BEARER_PATTERN.sub(r"\g<scheme> " + _REDACTED, s)
    s = _TEXT_KV_PATTERN.sub(_sub_credential_pair, s)
    # Token trần không có nhãn khóa phía trước.
    return _TEXT_TOKEN_PATTERN.sub(_REDACTED, s)


def redact_tool_result(result: object) -> str:
    """Ẩn một kết quả tool cho việc lưu trữ vết (trace) và xem trước sự kiện.

    Điểm tập trung duy nhất cho chuỗi kết quả tool: bên gọi thực hiện ẩn một lần và
    gửi cùng một giá trị tới mọi bên đăng ký (bản ghi trace được lưu trữ, bản xem trước
    SSE, vết react) thay vì ẩn riêng cho từng bên đăng ký.

    Kết quả JSON đi qua :func:`redact_payload` trong
    :data:`RESULT_SINK`: các key nhạy cảm được thay thế theo tên, các vỏ bọc
    ``content`` vẫn giữ đọc được, và các lá chuỗi văn bản tự do còn lại được
    làm sạch theo mẫu — dạng phổ biến là bao bọc JSON có trường ``stdout`` /
    ``error`` chứa đầu ra thô mà không quy tắc dựa trên key nào phân loại được. Kết quả
    không phải JSON được làm sạch theo mẫu trực tiếp. Mỗi byte được xử lý bởi
    chính xác một trong hai cơ chế; không có gì bị ẩn hai lần.

    Args:
        result: Kết quả tool thô (chuỗi, hoặc đối tượng ép kiểu qua
            :func:`str`). ``None`` trở thành chuỗi rỗng.

    Returns:
        Kết quả đã ẩn dưới dạng chuỗi. Đầu vào JSON được tuần tự hóa lại
        với ``ensure_ascii=False`` và các ký tự phân cách mặc định, để bên tiêu thụ bản xem trước
        grep dạng JSON (``"audit_id": "la_…"``) tiếp tục khớp.
    """
    if result is None:
        return ""
    text = result if isinstance(result, str) else str(result)
    if not text:
        return text
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return redact_text(text)
    return json.dumps(redact_payload(payload, sink=RESULT_SINK), ensure_ascii=False)
