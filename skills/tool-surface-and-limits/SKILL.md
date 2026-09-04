# Tool Surface and Its Limits

> **Phạm vi hoạt động (OHLC Data & Hypothesis Tools):**
> Hướng dẫn này liệt kê các công cụ hỗ trợ cho việc **nghiên cứu dữ liệu OHLC, phát hiện dị biệt và tổng hợp giả thuyết kinh tế**.

## Operating core

**Use when** — before using pseud tools or interpreting output.
**Settles** — active tools for OHLC data research and hypothesis generation.

**Procedure**
1. Use `scan_lakehouse_anomalies` to detect statistical anomalies in 1m OHLCV data.
2. Use `synthesize_hypothesis` to cluster signals and generate structured `HypothesisSpec` payloads.
3. Use `evidence_ledger` to record and validate citation anchors.
4. Use `read_file` / `write_file` for artifact storage.

---

## 1. Active Researcher Tools

| Tool | Parameters | Functionality |
| :-- | :-- | :-- |
| `scan_lakehouse_anomalies` | `target_universe`, `lookback_days` | Runs M-RS1 scanners (`A1'` Moment, `A2'` Volatility Jump, `A3'` Volume-Price Decoupling) over OHLCV parquet data in `ho_du_lieu/`. |
| `synthesize_hypothesis` | `signals`, `plan_id`, `predicted_horizon_band` | Synthesizes `AnomalySignal` objects into formal `HypothesisSpec` payloads with HMAC gate tokens (M-RS5). |
| `evidence_ledger` | `action` (`digest` \| `validate_citation`), `payload` | Structural digest for citable cells (`e:run:call:jsonpath`) and V1–V4 citation validation. |
| `read_file` | `path` | Reads file contents relative to run directory. |
| `write_file` | `path`, `content` | Writes UTF-8 file contents into run directory. |
| `remember` | `action`, `title`, `content` | Cross-session persistent memory storage. |

---

## 2. Best Practices & Invariants

- **Read-only tools:** `scan_lakehouse_anomalies` and `evidence_ledger` are read-only and execute without mutating source parquet data.
- **Evidence Citation:** Every claim should be supported by citable `EvidenceCell` anchors.
- **Fail-Closed Handling:** If data is missing or incomplete, report `INSUFFICIENT_DATA` cleanly instead of inventing dummy observations.
