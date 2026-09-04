# 🧬 pseud — Quantitative OHLCV Anomaly Researcher

> **Hệ thống AI nghiên cứu dị biệt thị trường tài chính từ dữ liệu nến 1-phút (OHLCV)
> và sinh Giả thuyết Kinh tế Định lượng qua đội ngũ đa Agent Swarm.**

[ 🇻🇳 Tiếng Việt ] | [ 🇺🇸 English ](README.md)

---

## 🎯 Tổng quan

**pseud** là một bộ công cụ mã nguồn mở cho phép:

1. **Quét bất thường thống kê** trên hàng triệu dòng dữ liệu nến 1-phút (Open, High, Low, Close, Volume).
2. **Sinh giả thuyết kinh tế định lượng** bằng AI (LLM — Gemini / OpenAI).
3. **Thẩm định & phản biện đa góc nhìn** bằng Swarm Multi-Agent Review Tribunal.

### Phạm vi dữ liệu

Hệ thống **chỉ hoạt động trên 5 cột OHLCV** — không sử dụng Order Book, Funding Rate,
Open Interest, Liquidation hay bất kỳ dữ liệu L2/L3 nào.

| Cột dữ liệu | Mô tả |
|:--|:--|
| `open` | Giá mở cửa nến 1 phút |
| `high` | Giá cao nhất trong 1 phút |
| `low` | Giá thấp nhất trong 1 phút |
| `close` | Giá đóng cửa nến 1 phút |
| `volume` | Khối lượng giao dịch trong 1 phút |

---

## 🏗️ Kiến trúc: Màng lọc Định lượng → AI Suy luận

Đây là điểm mạnh **cốt lõi** của hệ thống: **không đẩy trực tiếp dữ liệu thô vào LLM**,
mà đi qua một **màng lọc thống kê (Quantitative Filter)** trước khi chuyển cho AI suy luận.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  DỮ LIỆU THÔ PARQUET (OHLCV)                        │
│              Hàng triệu dòng nến 1-phút × 5 cặp giao dịch           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG 1 · MÀNG LỌC ĐỊNH LƯỢNG (Polars / Rust Engine)                │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  A1' Moment Shift    — Skewness & Kurtosis phân phối giá      │  │
│  │  A2' Volatility Jump — Garman-Klass & Parkinson Volatility    │  │
│  │  A3' Volume-Price    — Phân kỳ khối lượng & giá               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  • Robust Z-Score (Median / MAD) — miễn nhiễm với ngoại lai         │
│  • Ngưỡng lọc: chỉ trích xuất các điểm dị biệt |Z| > 2.0            │
│  • Kết quả: nén từ ~1,000,000 dòng → 10-40 tín hiệu dị biệt         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  TẦNG 2 · AI SUY LUẬN KINH TẾ (LLM — Gemini / OpenAI)                │
│                                                                      │
│  Nhận đầu vào đã được lọc:                                           │
│    • Z-Score, Thời điểm, Cột dữ liệu (Close, Volume...)              │
│    • Không nhận dữ liệu thô — chỉ nhận tín hiệu thống kê             │
│                                                                      │
│  Sinh ra:                                                            │
│    • Luận điểm kinh tế vi mô (momentum overextension, mean-reversion)│
│    • Hướng dự đoán (BUY / SELL) + Chế độ thị trường + Khung thời gian│
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG 3 · SWARM MULTI-AGENT REVIEW TRIBUNAL                         │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ QUANT        │  │ MARKET       │  │ RISK OFFICER             │   │
│  │ ADVOCATE (+) │  │ CRITIC (-)   │  │ (Final Verdict)          │   │
│  │              │  │              │  │                          │   │
│  │ Bảo vệ luận  │  │ Phản biện    │  │ Phán quyết đồng thuận:   │   │
│  │ điểm kinh tế │  │ bẫy giá giả, │  │ APPROVED /               │   │
│  │ hành vi giá  │  │ false signal │  │ APPROVED_WITH_CAUTION /  │   │
│  │ & khối lượng │  │ warnings     │  │ REJECTED_NOISE           │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Tại sao mô hình này vượt trội?

| Tiêu chí | Đẩy thẳng dữ liệu vào LLM ❌ | Qua Màng lọc rồi mới vào LLM ✅ |
|:--|:--|:--|
| **Chi phí Token** | Hàng triệu tokens/lần gọi (~$5-50/lần) | 200-500 tokens/lần gọi (~$0.001) |
| **Độ chính xác số liệu** | LLM hay "bốc phét" số | Z-Score tính bằng Polars/Rust — chính xác 100% |
| **Tốc độ quét** | Phụ thuộc vào context window LLM | Polars quét hàng triệu dòng trong ~2 giây |
| **Khả năng hallucination** | Cao — LLM tự bịa Z-Score | Không — số liệu do engine tính, LLM chỉ suy luận |
| **Phân vai** | LLM làm tất cả (kém) | Polars = số học, LLM = suy luận kinh tế (tốt) |

---

## 💎 Điểm mạnh nổi bật

### 1. 🪙 Tiết kiệm Token cực đại (Token Economy)

Thay vì nhồi hàng triệu dòng nến thô vào context window LLM (dễ gây tràn bộ nhớ, chi phí
$5-50 mỗi lần gọi API), hệ thống **nén dữ liệu qua Màng lọc Thống kê**:

- **Đầu vào:** ~1,000,000 dòng Parquet (hàng GB)
- **Đầu ra màng lọc:** 10-40 tín hiệu dị biệt (mỗi tín hiệu chỉ ~50 tokens)
- **Chi phí LLM thực tế:** ~200-500 tokens/giả thuyết ≈ **$0.001/lần gọi**
- **Tiết kiệm:** **99.99% tokens** so với cách đẩy thẳng dữ liệu

### 2. 🛡️ Chống Overfit triệt để (Anti-Overfitting by Design)

Hệ thống được thiết kế chống overfit ở **3 tầng phòng thủ**:

| Tầng | Cơ chế chống Overfit |
|:--|:--|
| **Tầng 1 — Robust Statistics** | Sử dụng Median/MAD thay vì Mean/Std. Miễn nhiễm với ngoại lai (outliers). Ngưỡng Z > 2.0 loại bỏ nhiễu thống kê. |
| **Tầng 2 — LLM không thấy dữ liệu thô** | LLM chỉ nhận Z-Score đã tính sẵn → không thể "nhìn trước" dữ liệu để bịa ra kết luận phù hợp. |
| **Tầng 3 — Swarm Phản biện** | Market Critic chủ động tìm bẫy giá giả, false momentum trap, và cảnh báo rủi ro regime breakdown. |

**So sánh với cách tiếp cận truyền thống:**
- ❌ **Cách cũ:** Cho LLM xem toàn bộ dữ liệu → LLM "học thuộc" pattern → Overfit 100%
- ✅ **Cách này:** LLM chỉ thấy tóm tắt thống kê → Không thể memorize data → Chống overfit tự nhiên

### 3. 🚫 Chống Hallucination (Anti-Fabrication)

LLM **nổi tiếng** là hay bịa số liệu. Trong hệ thống này:

- **Mọi con số** (Z-Score, Effect Size, Timestamp) đều do **Polars/Rust Engine** tính toán
  bằng công thức toán học chính xác — không phải do LLM suy đoán
- LLM **chỉ được phép suy luận định tính** (giải thích cơ chế kinh tế) —
  không được phép tự tính hay sáng tạo số liệu
- Prompt gửi cho LLM có **ràng buộc cứng**: cấm nhắc tới Order Book, L2 Data,
  hoặc bất kỳ dữ liệu nào ngoài OHLCV

### 4. ⚡ Tốc độ xử lý siêu nhanh (Performance)

| Thành phần | Tốc độ | Lý do |
|:--|:--|:--|
| Quét ~1 triệu dòng Parquet | **~2 giây** | Polars chạy trên Rust Engine (multi-threaded, zero-copy) |
| Tính Z-Score + Garman-Klass | **~0.5 giây** | NumPy vectorized operations |
| Sinh 1 giả thuyết AI | **~3-5 giây** | 1 lần gọi LLM API (~300 tokens) |
| Swarm Review 1 giả thuyết | **~3-5 giây** | 1 lần gọi LLM API (~400 tokens) |

**Tổng thời gian end-to-end:** Quét 5 cặp giao dịch + Sinh 3 giả thuyết + Swarm Review ≈ **30-45 giây**

### 5. 🔬 Tách biệt Vai trò rõ ràng (Separation of Concerns)

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│    POLARS ENGINE     │     │    LLM REASONING     │     │    SWARM TRIBUNAL    │
│    (Toán học thuần)  │     │   (Suy luận kinh tế) │     │   (Phản biện rủi ro) │
│                      │     │                      │     │                      │
│  Giỏi: Số học        │     │  Giỏi: Ngôn ngữ      │     │  Giỏi: Đa góc nhìn   │
│  Giỏi: Ma trận       │ ──> │  Giỏi: Suy luận      │ ──> │  Giỏi: Phản biện     │
│  Kém: Giải thích     │     │  Kém: Tính toán      │     │  Kém: Số liệu gốc    │
│                      │     │                      │     │                      │
│  → Để Polars tính    │     │  → Để LLM giải thích │     │  → Để Swarm thẩm định│
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### 6. 🔄 Lọc Đôi — Double Filtering Mechanism

Dữ liệu đi qua **2 lần sàng lọc** trước khi thành giả thuyết cuối cùng:

1. **Lọc 1 (Polars Scanner):** Loại bỏ 99.9% dữ liệu bình thường / nhiễu —
   chỉ giữ lại các tín hiệu dị biệt thống kê có ý nghĩa
2. **Lọc 2 (Swarm Review Tribunal):** Đánh giá lại từng giả thuyết đã sinh —
   phát hiện bẫy giá giả, false signal, và gán nhãn rủi ro (`REJECTED_NOISE`)

---

## 🧩 Tính mở rộng (Extensibility)

Kiến trúc của **pseud** được thiết kế theo nguyên tắc **Plugin-First** — mở rộng chức năng
mới mà **không cần sửa mã nguồn lõi**. Bạn chỉ cần thêm tệp vào đúng thư mục.

### Thêm Tool mới (Tự động phát hiện)

Tạo một tệp Python trong `tools/` kế thừa `BaseTool` — hệ thống **tự động phát hiện và
đăng ký** vào Tool Registry mà không cần sửa bất kỳ tệp nào khác:

```python
# tools/my_custom_scanner.py
from pseud.agent.tools import BaseTool

class MyCustomScanner(BaseTool):
    name = "my_custom_scanner"
    description = "Quét tín hiệu tùy chỉnh trên dữ liệu OHLCV."
    parameters = { ... }

    def execute(self, **kwargs):
        # Logic quét dữ liệu của bạn
        return json.dumps({"status": "ok", "signals": [...]})
```

**Xong.** Tool mới sẽ tự động xuất hiện trong Agent loop và Swarm worker.

### Thêm Skill mới (Tri thức nghiên cứu)

Tạo thư mục trong `skills/` với tệp `SKILL.md`:

```
skills/
└── my-new-strategy/
    └── SKILL.md          ← Tri thức & quy trình nghiên cứu
```

Agent sẽ tự động nạp và sử dụng tri thức này khi suy luận.

### Thêm Preset Swarm mới (Đội nhóm AI)

Tạo tệp YAML trong `swarm/presets/` để định nghĩa đội nhóm AI chuyên biệt:

```yaml
# swarm/presets/my_review_team.yaml
name: my_review_team
description: Đội phản biện chiến lược tùy chỉnh
agents:
  - role: Strategist
    system_prompt: "Bạn là chiến lược gia..."
    tools: [my_custom_scanner, read_file]
  - role: Risk_Auditor
    system_prompt: "Bạn là kiểm soát viên rủi ro..."
    tools: [read_file, write_file]
```

### Áp dụng làm khung kiến trúc cho các dự án Agent khác

Kiến trúc của `pseud` hoàn toàn **tách biệt giữa lõi (core) và nghiệp vụ (domain)**,
nếu bạn muốn mang sang dự án khác, bạn có thể tái sử dụng làm nền tảng cho nhiều dự án AI Agent khác nhau:

| Dự án Agent | Thay đổi cần làm |
|:--|:--|
| **Agent phân tích Forex** | Thêm Tool quét dữ liệu Forex + Skill phân tích cặp tiền tệ |
| **Agent nghiên cứu cổ phiếu** | Thêm Tool đọc dữ liệu chứng khoán + Preset đội phân tích cơ bản |
| **Agent kiểm soát rủi ro** | Thêm Tool tính VaR/CVaR + Skill đánh giá rủi ro danh mục |
| **Agent phân tích dữ liệu tổng quát** | Thêm Tool đọc CSV/DB + Skill phân tích thống kê |
| **Agent nghiên cứu học thuật** | Thêm Tool đọc paper PDF + Skill tổng hợp literature review |

---

## 🚀 Khả năng Tái sử dụng & Mở rộng sang các Ngành khác (Domain Agnostic Extensibility)

Kiến trúc của `pseud` được thiết kế theo dạng **Module hóa tách biệt hoàn toàn giữa Khung tác vụ AI (Core Agent / Swarm Framework) và Tầng Nghiệp vụ Dữ liệu (Domain Layer)**. Bộ mã nguồn này có thể **tái sử dụng 80–90%** để xây dựng các hệ thống Multi-Agent AI cho bất kỳ ngành nghề nào ngoài tài chính.

### 🧩 1. Các Module Khung có thể Re-use trực tiếp (`pseud/`):

- **`pseud/swarm/`**: Động cơ Swarm điều phối Multi-Agent (Advocate, Critic, Risk Officer) chạy song song, tranh luận đối kháng, xử lý lỗi tự động (Operational Fallback) và quản lý tiến trình.
- **`pseud/agent/`**: Vòng lặp ReAct Agent, quản lý Tool/Skill động, và động cơ **Grounding Guardrail** giúp chống bịa đặt dữ liệu (Hallucination Prevention).
- **`pseud/memory/`**: Hệ thống bộ nhớ đa cấp tích hợp SQLite FTS5 Full-Text Search, nén ngữ cảnh (Memory Compression), và liên kết ngữ nghĩa (Semantic Links).
- **`pseud/providers/`**: Tầng kết nối LLM linh hoạt (OpenAI, Gemini, Anthropic, Ollama/vLLM local).
- **`pseud/tools/` & `pseud/config/`**: Quản lý công cụ, cấu hình Thread-safe và bảo vệ dữ liệu nhạy cảm (Data Redaction).

### 💡 2. Ứng dụng thực tế cho các Ngành khác:

Khi mở rộng sang lĩnh vực khác, nhà phát triển **chỉ cần thay đổi công cụ quét dữ liệu (Data Scanner)** và **Prompt hệ thống**, giữ nguyên toàn bộ bộ khung bên dưới:

| Ngành nghề | Đầu vào Dị biệt (Anomaly Input) | Giả thuyết AI Sinh ra | Swarm Tribunal Thẩm định |
| :--- | :--- | :--- | :--- |
| **Y tế & Chẩn đoán** | Nhịp tim, huyết áp, chỉ số xét nghiệm bất thường | **Giả thuyết Bệnh lý & Chẩn đoán** | Bác sĩ Chẩn đoán (+) vs Bác sĩ Phản bác (-) vs Hội đồng Y đức (🛡️) |
| **An ninh mạng (SOC)** | Log mạng, lưu lượng IP nghi vấn, truy cập lạ | **Giả thuyết Tấn công (APT/Malware)** | Chuyên gia Giám sát (+) vs Analyst Phản bác (-) vs CISO Giám sát Rủi ro (🛡️) |
| **IoT & Bảo trì Công nghiệp** | Độ rung, nhiệt độ, áp suất máy móc vượt ngưỡng | **Giả thuyết Hỏng hóc Thiết bị** | Kỹ sư Bảo trì (+) vs Kỹ sư Vận hành (-) vs Quản lý An toàn (🛡️) |
| **Thương mại Điện tử & Fraud** | Chuỗi giao dịch thanh toán bất thường | **Giả thuyết Hành vi Gian lận (Fraud)** | Risk Analyst (+) vs Business Critic (-) vs Compliance Officer (🛡️) |

---

## 🚀 Cài đặt & Sử dụng

### Yêu cầu

- Python ≥ 3.10
- Các thư viện: `polars`, `numpy`, `scipy`, `pyyaml`
- API Key: `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` (trong tệp `.env`)

### Cấu trúc dữ liệu

```
data/
├── BTCUSDT/
│   ├── year=2024/month=01/*.parquet
│   ├── year=2024/month=02/*.parquet
│   └── ...
├── ETHUSDT/
│   └── ...
└── SOLUSDT/
    └── ...
```

Mỗi tệp Parquet chứa dữ liệu nến 1-phút với các cột: `event_time_ns`, `open`, `high`, `low`, `close`, `volume`.

### Chạy công cụ

```bash
# Chạy với Swarm Multi-Agent Review (mặc định BẬT)
python run_ohlc_anomaly_hypothesis.py --symbols BTCUSDT ETHUSDT --max-hypotheses 3

# Chạy tắt Swarm để tăng tốc
python run_ohlc_anomaly_hypothesis.py --symbols BTCUSDT ETHUSDT --max-hypotheses 3 --disable-swarm

# Chạy toàn bộ 5 cặp giao dịch
python run_ohlc_anomaly_hypothesis.py --max-hypotheses 5
```

### Tham số CLI

| Tham số | Mặc định | Mô tả |
|:--|:--|:--|
| `--symbols` | `BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT` | Danh sách cặp giao dịch cần quét |
| `--lookback-days` | `730` | Cửa sổ quét dữ liệu (ngày) |
| `--max-hypotheses` | `5` | Số lượng giả thuyết AI tối đa |
| `--enable-swarm` | `True` | Bật Swarm Multi-Agent Review Tribunal |
| `--disable-swarm` | — | Tắt Swarm Review |
| `--output` | `hypotheses_report.json` | Tệp JSON kết quả |

---

## 📊 Kết quả mẫu

<img width="1920" height="942" alt="1786366738627_2073241689277404219_3458779731416677886_a3e15f36aed2b41710ffed147738a70a" src="https://github.com/user-attachments/assets/f9f569ae-a5eb-443b-8bbd-e214af6587a7" />

## 📁 Cấu trúc dự án

```
pseud/
├── run_ohlc_anomaly_hypothesis.py   # CLI launcher chính
├── tools/
│   ├── ohlc_anomaly_hypothesis_tool.py  # Máy quét OHLCV + AI Synthesizer + Swarm Reviewer
│   ├── file_tools.py                    # Công cụ đọc/ghi tệp
│   ├── skill_tool.py                    # Nạp tri thức Skill
│   └── remember_tool.py                 # Ghi nhớ persistent memory
├── swarm/
│   ├── runtime.py                       # Động cơ điều phối đa Agent
│   ├── worker.py                        # Worker thực thi tác vụ
│   └── presets/                         # Cấu hình đội nhóm AI đóng gói sẵn
│       ├── general_research_team.yaml
│       ├── alpha_autopsy_desk.yaml
│       └── overfit_tribunal.yaml
├── skills/                              # Tri thức nghiên cứu (100% OHLCV)
│   ├── alpha-hypothesis-writer/
│   ├── anomaly-scan-protocol/
│   ├── evidence-citation-protocol/
│   ├── lakehouse-ground-truth/
│   └── tool-surface-and-limits/
├── providers/                           # LLM Provider (Gemini, OpenAI, Ollama)
├── agent/                               # Agent loop & grounding
├── config/                              # Cấu hình hệ thống
├── data/                                # Dữ liệu Parquet OHLCV 1-phút
└── .env                                 # API keys (GEMINI_API_KEY, v.v.)
```

---

## 🔬 Bộ quét dị biệt OHLCV

### A1' — Moment Regime Shift (Dịch chuyển phân phối giá)
- **Dữ liệu:** `close`
- **Phương pháp:** Robust Z-Score trên rolling skewness & kurtosis (24h / 30d baseline)
- **Phát hiện:** Sự đứt gãy cấu trúc trong phân phối lợi suất (giá bị kéo giãn quá mức)

### A2' — Volatility Jump (Biến động đột biến)
- **Dữ liệu:** `open`, `high`, `low`, `close`
- **Phương pháp:** Garman-Klass & Parkinson Volatility estimators
- **Phát hiện:** Sự kiện biến động nhảy vọt (khi tỷ số GK/Parkinson > 2.0)

### A3' — Volume-Price Decoupling (Phân kỳ khối lượng - giá)
- **Dữ liệu:** `close`, `volume`
- **Phương pháp:** Rolling correlation & volume ratio Z-Score
- **Phát hiện:** Khối lượng bùng nổ nhưng giá đứng yên (dấu hiệu hấp thụ / bão hòa động lượng)

---

## 🤖 Swarm Multi-Agent Review Tribunal

Sau khi AI sinh giả thuyết, hệ thống **Swarm** kích hoạt đội ngũ 3 Agent tranh luận:

| Vai trò | Nhiệm vụ |
|:--|:--|
| **Quant Advocate (+)** | Lập luận bảo vệ tính hợp lý của mối quan hệ hành vi giá và khối lượng |
| **Market Critic (-)** | Tìm bẫy giá giả (false momentum trap), cảnh báo rủi ro đảo chiều thất bại |
| **Risk Officer (🛡️)** | Đưa ra phán quyết cuối cùng: `APPROVED`, `APPROVED_WITH_CAUTION`, hoặc `REJECTED_NOISE` |

---

## 📜 Giấy phép & Ghi nhận

Dự án mã nguồn mở được phát hành theo giấy phép [**MIT License**](LICENSE).

Hệ thống được phát triển dựa trên nền tảng kiến trúc từ
[**Vibe Trading**](https://github.com/HKUDS/Vibe-Trading).

Một phần mã nguồn cơ sở hạ tầng (Agent loop, Swarm runtime, Provider abstraction,
Config system) được tham khảo và điều chỉnh từ dự án Vibe Trading.

---