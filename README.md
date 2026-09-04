# 🧬 pseud — Quantitative OHLCV Anomaly Researcher

> **An AI system for quantitative financial market research from 1-minute candle (OHLCV) data, generating Quantitative Economic Alpha Hypotheses via a Multi-Agent Swarm Tribunal.**

[ 🇺🇸 English ] | [ 🇻🇳 Tiếng Việt ](README_VI.md)

---

## 🎯 Overview

**pseud** is an open-source research workspace that enables:

1. **Statistical Anomaly Scanning** across millions of 1-minute candle rows (Open, High, Low, Close, Volume).
2. **Quantitative Economic Hypothesis Generation** powered by AI (LLM — Gemini / OpenAI).
3. **Multi-Perspective Adversarial Review** executed by a Swarm Multi-Agent Review Tribunal.

### Data Scope

The system **operates strictly on 5 OHLCV columns** — it does NOT use Order Books, Funding Rates, Open Interest, Liquidations, or any L2/L3 market data.

| Data Column | Description |
|:--|:--|
| `open` | 1-minute bar open price |
| `high` | 1-minute bar highest price |
| `low` | 1-minute bar lowest price |
| `close` | 1-minute bar close price |
| `volume` | 1-minute bar trading volume |

---

## 🏗️ Architecture: Quantitative Filter → AI Reasoning

This is the **core advantage** of the architecture: **never feed raw data directly into the LLM**. Instead, data passes through a high-performance **Quantitative Filter** before being handed to AI for economic reasoning.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   RAW PARQUET DATASET (OHLCV)                       │
│           Millions of 1-minute bar rows × 5 trading pairs           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 · QUANTITATIVE FILTER (Polars / Rust Engine)               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  A1' Moment Shift    — Return distribution Skewness & Kurtosis│  │
│  │  A2' Volatility Jump — Garman-Klass & Parkinson Volatility    │  │
│  │  A3' Volume-Price    — Volume-Price decoupling & divergence   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  • Robust Z-Score (Median / MAD) — immune to statistical outliers   │
│  • Anomaly threshold: extracts only signals with |Z| > 2.0          │
│  • Result: compresses ~1,000,000 rows → 10-40 anomaly signals       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 2 · AI ECONOMIC REASONING (LLM — Gemini / OpenAI)             │
│                                                                      │
│  Receives filtered inputs:                                           │
│    • Z-Score, Timestamps, Target Column (Close, Volume...)           │
│    • No raw candle dumps — statistical signals only                  │
│                                                                      │
│  Generates:                                                          │
│    • Microeconomic thesis (momentum overextension, mean-reversion)   │
│    • Directional prediction (BUY / SELL) + Market Regime + Horizon   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 · SWARM MULTI-AGENT REVIEW TRIBUNAL                        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ QUANT        │  │ MARKET       │  │ RISK OFFICER             │   │
│  │ ADVOCATE (+) │  │ CRITIC (-)   │  │ (Final Verdict)          │   │
│  │              │  │              │  │                          │   │
│  │ Defends price│  │ Challenges   │  │ Consensus Verdict:       │   │
│  │ action &     │  │ false traps, │  │ APPROVED /               │   │
│  │ volume thesis│  │ fake breaks  │  │ APPROVED_WITH_CAUTION /  │   │
│  │              │  │              │  │ REJECTED_NOISE           │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why is this approach superior?

| Criteria | Direct Raw Data to LLM ❌ | Filtered Signals to LLM ✅ |
|:--|:--|:--|
| **Token Cost** | Millions of tokens/call (~$5-50/call) | 200-500 tokens/call (~$0.001) |
| **Numeric Accuracy** | LLM hallucinates numbers | Polars/Rust computes Z-Scores — 100% exact |
| **Scan Speed** | Constrained by LLM context window | Polars scans millions of rows in ~2 seconds |
| **Hallucination Risk** | High — LLM invents metrics | Zero — metrics computed by engine, LLM only reasons |
| **Separation of Roles** | LLM does everything (poorly) | Polars = math, LLM = economic reasoning (optimal) |

---

## 💎 Core Highlights

### 1. 🪙 Maximum Token Economy

Instead of stuffing millions of raw candle lines into the LLM context window (causing memory overflows and costing $5–50 per API call), the system **compresses data via Statistical Filtering**:

- **Input:** ~1,000,000 Parquet rows (gigabytes of raw data)
- **Filtered Output:** 10–40 anomaly signals (~50 tokens per signal)
- **Actual LLM Cost:** ~200–500 tokens/hypothesis ≈ **$0.001/call**
- **Token Savings:** **99.99% reduction** compared to raw data ingestion

### 2. 🛡️ Anti-Overfitting by Design

The system enforces anti-overfitting at **3 defense layers**:

| Layer | Defense Mechanism |
|:--|:--|
| **Layer 1 — Robust Statistics** | Uses Median/MAD instead of Mean/Std. Outlier-proof. Threshold |Z| > 2.0 filters noise. |
| **Layer 2 — Blinded LLM** | LLM receives pre-computed Z-Scores → cannot "look ahead" to fit arbitrary patterns. |
| **Layer 3 — Swarm Adversary** | Market Critic actively searches for false momentum traps, fake wicks, and regime breaks. |

### 3. 🚫 Anti-Hallucination (Anti-Fabrication)

LLMs are prone to inventing numbers. In `pseud`:

- **All numeric metrics** (Z-Score, Effect Size, Timestamps) are computed by **Polars/Rust Engine** using exact formulas.
- LLM is **restricted to qualitative economic reasoning** (explaining market mechanics) — it cannot invent numeric data.
- Prompts enforce **strict negative constraints**: prohibiting mentions of Order Books, L2 data, or non-OHLCV fields.

### 4. ⚡ High-Speed Execution (Performance)

| Component | Execution Time | Driver |
|:--|:--|:--|
| Scan ~1M Parquet rows | **~2 seconds** | Polars Rust Engine (multi-threaded, zero-copy) |
| Compute Z-Scores & Volatility | **~0.5 seconds** | Vectorized NumPy operations |
| Generate 1 AI Hypothesis | **~3–5 seconds** | Single LLM API call (~300 tokens) |
| Swarm Review per Hypothesis | **~3–5 seconds** | Single LLM API call (~400 tokens) |

**Total End-to-End Runtime:** Scan 5 symbols + Generate 3 Hypotheses + Swarm Review ≈ **30–45 seconds**.

---

## 🧩 Extensibility

`pseud` is architected with a **Plugin-First** design — extend functionality **without modifying core code**.

### Add a Custom Tool (Auto-Discovery)

Create a Python file in `tools/` inheriting from `BaseTool`. The framework **automatically detects and registers** it into the Tool Registry:

```python
# tools/my_custom_scanner.py
from pseud.agent.tools import BaseTool

class MyCustomScanner(BaseTool):
    name = "my_custom_scanner"
    description = "Scan custom signals on OHLCV data."
    parameters = { ... }

    def execute(self, **kwargs):
        # Your custom scanner logic
        return json.dumps({"status": "ok", "signals": [...]})
```

### Add a Custom Skill

Add a directory in `skills/` containing a `SKILL.md` file:

```
skills/
└── my-new-strategy/
    └── SKILL.md          ← Research protocol & guidelines
```

---

## 🚀 Domain Agnostic Extensibility

The architecture of `pseud` cleanly decouples the **Core AI Framework (Agent / Swarm Engine)** from the **Domain Layer (OHLCV Scanners)**. **80–90% of this codebase** can be reused directly to build Multi-Agent AI systems for non-financial industries.

### 🧩 1. Reusable Core Modules (`pseud/`):

- **`pseud/swarm/`**: Multi-Agent orchestration engine (Advocate, Critic, Risk Officer) managing parallel execution, adversarial debate, operational fallbacks, and task graphs.
- **`pseud/agent/`**: ReAct Agent loop, dynamic Tool/Skill loading, and **Grounding Guardrails** preventing data fabrication.
- **`pseud/memory/`**: Hierarchical memory engine with SQLite FTS5 Full-Text Search, context compression, and semantic links.
- **`pseud/providers/`**: Flexible LLM client supporting OpenAI, Gemini, Anthropic, and local Ollama/vLLM.
- **`pseud/tools/` & `pseud/config/`**: Thread-safe config loader and data redaction security tools.

### 💡 2. Real-World Industry Applications:

To adapt to another domain, **simply replace the Data Scanner Tool** and **System Prompts**, keeping the entire underlying core intact:

| Industry Domain | Anomaly Input | AI Generated Hypothesis | Swarm Review Tribunal |
| :--- | :--- | :--- | :--- |
| **Healthcare & Diagnostics** | Abnormal vitals, ECG spikes, lab report outliers | **Diagnostic / Pathology Thesis** | Diagnostic Physician (+) vs Specialist Critic (-) vs Ethics Board (🛡️) |
| **Cybersecurity (SOC)** | Network log spikes, suspicious IP traffic, unusual logins | **Threat / APT Attack Thesis** | SecOps Advocate (+) vs Threat Analyst (-) vs CISO Risk Officer (🛡️) |
| **Industrial IoT & Maintenance** | Sensor telemetry (vibration, heat, pressure spikes) | **Equipment Failure Thesis** | Maintenance Tech (+) vs Operations Lead (-) vs Safety Manager (🛡️) |
| **E-commerce & Fraud** | Transaction velocity spikes, unusual cart behavior | **Fraud Pattern Thesis** | Risk Analyst (+) vs Business Critic (-) vs Compliance Officer (🛡️) |

---

## 🚀 Setup & Usage

### Requirements

- Python ≥ 3.10
- Dependencies: `polars`, `numpy`, `scipy`, `pyyaml`, `pydantic`, `python-dotenv`, `rich`
- API Key: `GEMINI_API_KEY` or `OPENAI_API_KEY` (configured in `.env`)

### Dataset Layout

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

Each Parquet file contains 1-minute bars with columns: `event_time_ns`, `open`, `high`, `low`, `close`, `volume`.

### Running the CLI

```bash
# Run with Swarm Multi-Agent Review Tribunal (Enabled by default)
python run_ohlc_anomaly_hypothesis.py --symbols BTCUSDT ETHUSDT --max-hypotheses 3

# Disable Swarm Tribunal for faster execution
python run_ohlc_anomaly_hypothesis.py --symbols BTCUSDT ETHUSDT --max-hypotheses 3 --disable-swarm

# Run across all top 5 symbols
python run_ohlc_anomaly_hypothesis.py --max-hypotheses 5
```

---

## 📊 Sample Output

<img width="1920" height="942" alt="1786366738627_2073241689277404219_3458779731416677886_a3e15f36aed2b41710ffed147738a70a" src="https://github.com/user-attachments/assets/588020d3-2de3-4412-b2a8-0219019e4493" />

---

## 📁 Directory Structure

```
pseud/
├── run_ohlc_anomaly_hypothesis.py   # Main CLI launcher
├── tools/
│   ├── ohlc_anomaly_hypothesis_tool.py  # OHLCV Scanner + AI Synthesizer + Swarm Reviewer
│   ├── file_tools.py                    # File I/O tools
│   ├── skill_tool.py                    # Skill loader tool
│   └── remember_tool.py                 # Persistent memory tool
├── swarm/
│   ├── runtime.py                       # Swarm orchestration engine
│   ├── worker.py                        # Task execution worker
│   └── presets/                         # Pre-configured AI team presets
│       ├── general_research_team.yaml
│       ├── alpha_autopsy_desk.yaml
│       └── overfit_tribunal.yaml
├── skills/                              # Quantitative research skills
│   ├── alpha-hypothesis-writer/
│   ├── anomaly-scan-protocol/
│   ├── evidence-citation-protocol/
│   ├── lakehouse-ground-truth/
│   └── tool-surface-and-limits/
├── providers/                           # LLM Provider abstraction (Gemini, OpenAI, Ollama)
├── agent/                               # Agent ReAct loop & grounding guardrails
├── config/                              # System configuration
├── data/                                # 1-minute Parquet OHLCV dataset
└── .env                                 # API Keys & environment variables
```

---

## 📜 License & Acknowledgements

This open-source project is licensed under the [**MIT License**](LICENSE).

Infrastructure architecture inspired and adapted from [**Vibe Trading**](https://github.com/HKUDS/Vibe-Trading).

---

## 👤 Author

- **Author:** PVinh
- **Email:** ppvinh1513@gmail.com
- **Project:** `pseud` (Quantitative OHLCV Anomaly Researcher)
