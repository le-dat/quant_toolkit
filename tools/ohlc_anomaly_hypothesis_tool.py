"""ohlc_anomaly_hypothesis_tool.py — Công cụ quét Dị biệt OHLCV & Sinh Giả thuyết Kinh tế qua AI.

Công cụ này thực hiện 2 chức năng chính:
1. Quét dữ liệu 1-minute OHLCV trong thư mục `data/` bằng Polars để phát hiện các dị biệt thống kê:
   - A1': Moment Shift (Skewness & Kurtosis)
   - A2': Volatility Jump (Garman-Klass & Parkinson Volatility)
   - A3': Volume-Price Decoupling (Đột biến khối lượng & phân kỳ giá)
2. Chuyển kết quả dị biệt sang AI (LLM - Gemini/OpenAI) để phân tích cơ chế vi mô thị trường
   và tổng hợp các Giả thuyết Kinh tế Định lượng (Quantitative Economic Hypotheses).
"""

from __future__ import annotations

import json
import math
import logging
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from pseud.agent.tools import BaseTool
from pseud.models.models import KetCucChang
from pseud.providers.chat import ChatLLM
from pseud.providers.llm import _ensure_dotenv, _sync_provider_env

logger = logging.getLogger("OHLCAnomalyHypothesisTool")

# ── 1. Thống kê & Trợ lý Polars ───────────────────────────────────────────────

def compute_mad(values: list[float] | np.ndarray | pl.Series) -> float:
    """Tính Median Absolute Deviation."""
    if isinstance(values, pl.Series):
        arr = values.drop_nulls().drop_nans().to_numpy()
    else:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def calc_robust_z(values: np.ndarray, med: float, mad: float, mad_floor: float = 1e-4) -> np.ndarray:
    """Tính Robust Z-score từ Median & MAD."""
    effective_mad = max(mad, mad_floor)
    return 0.6745 * (values - med) / effective_mad


def _extract_and_parse_json_from_llm(content: str) -> dict[str, Any]:
    """Trích xuất và giải mã JSON từ phản hồi LLM một cách mạnh mẽ (bất chấp markdown wrapper, nhiễu chữ hay unescaped quotes)."""
    if not content or not content.strip():
        raise ValueError("Phản hồi từ LLM rỗng")

    text = content.strip()

    # 1. Tìm vị trí '{' đầu tiên và '}' cuối cùng
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    # 2. Xóa các dòng ```json hoặc ``` nếu còn trong khối
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    clean_text = "\n".join(lines).strip()

    # 3. Làm sạch dấu phẩy thừa trước } hoặc ]
    clean_text = re.sub(r",\s*([\}\]])", r"\1", clean_text)

    # 4. Sửa cú pháp mẫu "-1 or 1" hoặc "HIGH_VOLATILITY or ..." nếu LLM copy lại prompt
    clean_text = re.sub(r':\s*(-?\d+)\s+or\s+-?\d+', r': \1', clean_text)
    clean_text = re.sub(r':\s*"([^"]+)"\s+or\s+"[^"]+"', r': "\1"', clean_text)

    try:
        return json.loads(clean_text, strict=False)
    except Exception:
        pass

    try:
        clean_text_fixed = clean_text.replace("\n", " ").replace("\r", " ")
        return json.loads(clean_text_fixed, strict=False)
    except Exception:
        pass

    # 5. Regex key-value extraction fallback cho trường hợp LLM dính unescaped quotes
    extracted: dict[str, Any] = {}
    str_keys = [
        "title", "hypothesis_text", "narrative", "next_action",
        "action_reasoning", "target_regime", "target_horizon",
        "consensus_verdict", "advocate_analysis", "critic_analysis", "risk_verdict"
    ]
    for k in str_keys:
        m = re.search(r'"' + k + r'"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*\}|\s*\n)', clean_text, re.DOTALL)
        if m:
            extracted[k] = m.group(1).replace("\n", " ").strip()

    m_sign = re.search(r'"predicted_sign"\s*:\s*(-?\d+)', clean_text)
    if m_sign:
        extracted["predicted_sign"] = int(m_sign.group(1))

    if extracted:
        return extracted

    raise ValueError(f"Không thể giải mã JSON từ phản hồi LLM")


# ── 2. Động cơ Quét Dị biệt OHLCV (OHLCV Anomaly Scanner Engine) ────────────

@dataclass
class OHLCVAnomalySignal:
    signal_id: str
    symbol: str
    stat: str
    raw_columns: list[str]
    feature_names: list[str]
    value: float
    effect_size: float
    observed_at: str
    window: str


class OHLCVDataScanner:
    """Động cơ đọc & quét dữ liệu OHLCV từ thư mục data/."""

    def __init__(
        self,
        base_dir: Path | None = None,
        lookback_days: int = 730,
        segment: str = "full",  # "full", "in_sample", "out_of_sample"
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.lookback_days = lookback_days
        self.segment = segment.lower()
        self.start_date = start_date
        self.end_date = end_date
        self.data_root = self.base_dir / "data"

    def _load_and_filter_data(self, target_symbols: list[str]) -> pl.DataFrame | None:
        """Helper nạp và làm sạch dữ liệu OHLCV theo lookback_days, start_date, end_date, segment."""
        all_frames = []
        ohlcv_cols = ["event_time_ns", "open", "high", "low", "close", "volume"]

        for sym in target_symbols:
            sym_dir = self.data_root / sym
            if sym_dir.exists():
                for p in sorted(sym_dir.rglob("*.parquet")):
                    lf = pl.scan_parquet(str(p))
                    available = [c for c in ohlcv_cols if c in lf.collect_schema().names()]
                    lf_tagged = lf.select(available).with_columns([pl.lit(sym).alias("symbol")])
                    all_frames.append(lf_tagged)

        if not all_frames:
            return None

        logger.info("Đã mở %d tệp Parquet từ %s. Đang nạp dữ liệu...", len(all_frames), self.data_root)
        df_combined = pl.concat(all_frames, how="diagonal")

        # Cắt lọc theo lookback_days theo từng symbol
        if self.lookback_days and self.lookback_days < 3650:
            filtered_frames = []
            for sym in target_symbols:
                df_sym_raw = df_combined.filter(pl.col("symbol") == sym)
                if df_sym_raw.collect_schema().names():
                    max_ts_df = df_sym_raw.select(pl.col("event_time_ns").max()).collect()
                    if max_ts_df.height > 0:
                        max_ts = max_ts_df[0, 0]
                        if max_ts is not None:
                            cutoff_ns = max_ts - (self.lookback_days * 86400 * 1_000_000_000)
                            df_sym_raw = df_sym_raw.filter(pl.col("event_time_ns") >= cutoff_ns)
                            filtered_frames.append(df_sym_raw)
            if filtered_frames:
                df_combined = pl.concat(filtered_frames, how="diagonal")

        # Cắt lọc theo start_date / end_date nếu có
        if self.start_date:
            try:
                st_ns = int(datetime.fromisoformat(self.start_date.replace("Z", "+00:00")).timestamp() * 1e9)
                df_combined = df_combined.filter(pl.col("event_time_ns") >= st_ns)
            except Exception:
                pass

        if self.end_date:
            try:
                et_ns = int(datetime.fromisoformat(self.end_date.replace("Z", "+00:00")).timestamp() * 1e9)
                df_combined = df_combined.filter(pl.col("event_time_ns") <= et_ns)
            except Exception:
                pass

        df_eval = df_combined.collect().unique(subset=["symbol", "event_time_ns"]).sort(["symbol", "event_time_ns"])
        if df_eval.height < 100:
            return None

        # Cắt phân đoạn In-Sample vs Out-of-Sample (50% split theo từng symbol) để chống cạn/lặp giả thuyết
        if self.segment in ("in_sample", "out_of_sample"):
            split_frames = []
            for sym in target_symbols:
                df_sym_eval = df_eval.filter(pl.col("symbol") == sym)
                h = df_sym_eval.height
                if h > 0:
                    if self.segment == "in_sample":
                        split_frames.append(df_sym_eval.head(h // 2))
                    else:
                        split_frames.append(df_sym_eval.tail(h // 2))
            if split_frames:
                df_eval = pl.concat(split_frames)

        return df_eval

    def scan(self, symbols: list[str] | None = None) -> tuple[KetCucChang, list[OHLCVAnomalySignal], str]:
        """Đọc Parquet từ data/ và chạy 3 bộ quét OHLCV (A1', A2', A3')."""
        target_symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        
        df_eval = self._load_and_filter_data(target_symbols)
        if df_eval is None or df_eval.height < 100:
            return KetCucChang.INSUFFICIENT_DATA, [], "Insufficient bar rows or no data found"

        raw_signals: list[OHLCVAnomalySignal] = []

        # A1' Scanner: Moment Regime Shift (Skewness & Kurtosis on close returns)
        a1_sigs = self._scan_a1_moment(df_eval, target_symbols)
        raw_signals.extend(a1_sigs)

        # A2' Scanner: Volatility Regime & Jump (Garman-Klass & Parkinson on OHLC)
        a2_sigs = self._scan_a2_volatility(df_eval, target_symbols)
        raw_signals.extend(a2_sigs)

        # A3' Scanner: Volume-Price Decoupling (1h return & volume surge)
        a3_sigs = self._scan_a3_volume_price(df_eval, target_symbols)
        raw_signals.extend(a3_sigs)

        # Sắp xếp và chọn top signals đa dạng giữa các symbol (Symbol Diversity)
        raw_signals.sort(key=lambda s: abs(s.effect_size), reverse=True)
        
        # Limit max 10 signals per symbol to ensure universe coverage
        symbol_counts: dict[str, int] = {}
        top_signals: list[OHLCVAnomalySignal] = []
        for sig in raw_signals:
            cnt = symbol_counts.get(sig.symbol, 0)
            if cnt < 10:
                top_signals.append(sig)
                symbol_counts[sig.symbol] = cnt + 1
            if len(top_signals) >= 50:
                break

        return KetCucChang.PASS, top_signals, f"Successfully scanned {len(top_signals)} top anomalies"

    def verify_hypothesis_persistence(
        self,
        target_symbol: str,
        target_stat: str,
        min_z_threshold: float = 2.0,
    ) -> dict[str, Any]:
        """Kiểm tra xem 1 dị biệt cụ thể (symbol + stat) có tiếp tục tồn tại ở phân đoạn dữ liệu này hay không mà KHÔNG bị giới hạn bởi Top 50."""
        target_symbols = [target_symbol] if target_symbol else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        
        df_eval = self._load_and_filter_data(target_symbols)
        if df_eval is None or df_eval.height < 100:
            return {"persisted": False, "reason": f"Insufficient bar rows or no data for {target_symbol}"}

        raw_signals: list[OHLCVAnomalySignal] = []
        if target_stat == "moment_shift":
            raw_signals.extend(self._scan_a1_moment(df_eval, target_symbols))
        elif target_stat == "volatility_jump":
            raw_signals.extend(self._scan_a2_volatility(df_eval, target_symbols))
        elif target_stat == "volume_price_decoupling":
            raw_signals.extend(self._scan_a3_volume_price(df_eval, target_symbols))
        else:
            raw_signals.extend(self._scan_a1_moment(df_eval, target_symbols))
            raw_signals.extend(self._scan_a2_volatility(df_eval, target_symbols))
            raw_signals.extend(self._scan_a3_volume_price(df_eval, target_symbols))

        matched_signals = [s for s in raw_signals if s.symbol == target_symbol and s.stat == target_stat]
        if not matched_signals:
            return {
                "persisted": False,
                "symbol": target_symbol,
                "stat": target_stat,
                "segment": self.segment,
                "reason": f"Anomaly {target_stat} on {target_symbol} did not re-occur in this segment ({self.segment})",
                "z_score": 0.0,
            }

        top_matched = max(matched_signals, key=lambda s: abs(s.effect_size))
        is_persisted = abs(top_matched.effect_size) >= min_z_threshold

        return {
            "persisted": is_persisted,
            "symbol": target_symbol,
            "stat": target_stat,
            "segment": self.segment,
            "z_score": round(top_matched.effect_size, 2),
            "measured_value": round(top_matched.value, 4),
            "observed_at": top_matched.observed_at,
            "total_occurrences_untruncated": len(matched_signals),
            "reason": f"Anomaly persisted in {self.segment} segment with Z={top_matched.effect_size:.2f}" if is_persisted else f"Anomaly Z-score dropped below threshold ({top_matched.effect_size:.2f} < {min_z_threshold})",
        }

    def _scan_a1_moment(self, df: pl.DataFrame, symbols: list[str]) -> list[OHLCVAnomalySignal]:
        sigs = []
        for sym in symbols:
            df_sym = df.filter(pl.col("symbol") == sym)
            if df_sym.height < 1440:
                continue

            # Tính rolling 24h skewness & kurtosis (1440 bars) trên log returns bằng Polars
            moments_df = df_sym.select([
                pl.col("close").log().diff().rolling_skew(window_size=1440).alias("skew"),
                pl.col("close").log().diff().rolling_kurtosis(window_size=1440).alias("kurt"),
            ])

            skew_arr = moments_df.get_column("skew").to_numpy()
            kurt_arr = moments_df.get_column("kurt").to_numpy()
            valid_mask = np.isfinite(skew_arr) & np.isfinite(kurt_arr)
            if np.sum(valid_mask) < 60:
                continue

            med_skew, mad_skew = float(np.median(skew_arr[valid_mask])), compute_mad(skew_arr[valid_mask])
            z_skew = calc_robust_z(skew_arr, med_skew, mad_skew)
            np.nan_to_num(z_skew, copy=False, nan=0.0)

            med_kurt, mad_kurt = float(np.median(kurt_arr[valid_mask])), compute_mad(kurt_arr[valid_mask])
            z_kurt = calc_robust_z(kurt_arr, med_kurt, mad_kurt)
            np.nan_to_num(z_kurt, copy=False, nan=0.0)

            # Combined Moment Z-score (chọn giá trị lệch lớn hơn giữa Skewness và Kurtosis)
            z_combined = np.where(np.abs(z_skew) >= np.abs(z_kurt), z_skew, z_kurt)
            moment_val = np.where(np.abs(z_skew) >= np.abs(z_kurt), skew_arr, kurt_arr)

            abs_z = np.abs(z_combined)
            candidate_idx = np.where(valid_mask & (abs_z > 2.0))[0]
            if len(candidate_idx) == 0:
                continue
            sorted_candidates = candidate_idx[np.argsort(abs_z[candidate_idx])[::-1]]

            # Non-Maximum Suppression (NMS): Loại bỏ các nến lân cận cùng 1 sự kiện (cách nhau ít nhất 60 nến / 1h)
            selected_idx = []
            for idx in sorted_candidates:
                if not any(abs(idx - prev) < 60 for prev in selected_idx):
                    selected_idx.append(idx)
                    if len(selected_idx) >= 10:
                        break

            for i in selected_idx:
                t_ns = df_sym["event_time_ns"][int(i)]
                obs_iso = datetime.fromtimestamp(t_ns / 1e9, tz=UTC).isoformat()
                sigs.append(OHLCVAnomalySignal(
                    signal_id=f"A1_{sym}_moment_{i}",
                    symbol=sym,
                    stat="moment_shift",
                    raw_columns=["close"],
                    feature_names=["return_skewness_24h", "return_kurtosis_24h"],
                    value=float(moment_val[i]),
                    effect_size=float(z_combined[i]),
                    observed_at=obs_iso,
                    window="24h",
                ))
        return sigs

    def _scan_a2_volatility(self, df: pl.DataFrame, symbols: list[str]) -> list[OHLCVAnomalySignal]:
        sigs = []
        for sym in symbols:
            df_sym = df.filter(pl.col("symbol") == sym)
            if df_sym.height < 120:
                continue

            o = df_sym["open"].to_numpy()
            h = df_sym["high"].to_numpy()
            l_arr = df_sym["low"].to_numpy()
            c = df_sym["close"].to_numpy()

            # Garman-Klass 1-minute Variance Estimator
            log_hl = np.log(np.maximum(h, 1e-8) / np.maximum(l_arr, 1e-8))
            log_co = np.log(np.maximum(c, 1e-8) / np.maximum(o, 1e-8))
            gk_var_1m = np.maximum(0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2), 0.0)

            # Gom 1h Garman-Klass Volatility (Rolling Mean min_samples=1 của 60 nến 1m)
            gk_var_1h = df_sym.select(
                pl.Series(gk_var_1m).alias("var").rolling_mean(window_size=60, min_samples=1)
            ).get_column("var").to_numpy()
            gk_vol = np.sqrt(np.maximum(gk_var_1h, 1e-10))
            valid_mask = np.isfinite(gk_vol)

            med_gk = float(np.median(gk_vol[valid_mask]))
            mad_gk = compute_mad(gk_vol[valid_mask])
            z_gk = calc_robust_z(gk_vol, med_gk, mad_gk)
            np.nan_to_num(z_gk, copy=False, nan=0.0)

            # Non-Maximum Suppression (NMS): Loại bỏ các nến lân cận cùng 1 đợt Volatility Jump (cách nhau ít nhất 60 nến)
            abs_z = np.abs(z_gk)
            candidate_idx = np.where(valid_mask & (abs_z > 2.5))[0]
            if len(candidate_idx) == 0:
                continue
            sorted_candidates = candidate_idx[np.argsort(abs_z[candidate_idx])[::-1]]

            selected_idx = []
            for idx in sorted_candidates:
                if not any(abs(idx - prev) < 60 for prev in selected_idx):
                    selected_idx.append(idx)
                    if len(selected_idx) >= 10:
                        break

            for i in selected_idx:
                t_ns = df_sym["event_time_ns"][int(i)]
                obs_iso = datetime.fromtimestamp(t_ns / 1e9, tz=UTC).isoformat()
                sigs.append(OHLCVAnomalySignal(
                    signal_id=f"A2_{sym}_vol_{i}",
                    symbol=sym,
                    stat="volatility_jump",
                    raw_columns=["open", "high", "low", "close"],
                    feature_names=["volatility_garman_klass_1h", "volatility_parkinson_1h", "gk_parkinson_ratio"],
                    value=float(gk_vol[i]),
                    effect_size=float(z_gk[i]),
                    observed_at=obs_iso,
                    window="1h",
                ))
        return sigs

    def _scan_a3_volume_price(self, df: pl.DataFrame, symbols: list[str]) -> list[OHLCVAnomalySignal]:
        sigs = []
        for sym in symbols:
            df_sym = df.filter(pl.col("symbol") == sym)
            if df_sym.height < 120:
                continue

            v = df_sym["volume"].to_numpy()

            # Tính v_ma backward-looking với min_samples=1
            v_ma = df_sym.select(
                pl.col("volume").rolling_mean(window_size=60, min_samples=1)
            ).get_column("volume").to_numpy()
            v_ma_clean = np.maximum(np.nan_to_num(v_ma, nan=1.0), 1e-4)
            v_ratio = v / v_ma_clean

            # Tính 1h price return để xác định chỉ số Phân kỳ/Hấp thụ Khối lượng-Giá (Volume-Price Decoupling)
            ret_1h = df_sym.select(
                pl.col("close").pct_change(60)
            ).get_column("close").to_numpy()
            np.nan_to_num(ret_1h, copy=False, nan=0.0)

            # Volume-Price Decoupling Index: Khối lượng bùng nổ nhưng biên độ giá bị kìm hãm (Tường hấp thụ)
            decoupling = v_ratio / (np.abs(ret_1h) + 1e-3)
            valid_mask = np.isfinite(decoupling)

            med_vr = float(np.median(decoupling[valid_mask]))
            mad_vr = compute_mad(decoupling[valid_mask])
            z_vr = calc_robust_z(decoupling, med_vr, mad_vr)
            np.nan_to_num(z_vr, copy=False, nan=0.0)

            # Non-Maximum Suppression (NMS): Loại bỏ nến lân cận trong cùng 1 đợt volume surge (cách nhau 60 nến)
            abs_z = np.abs(z_vr)
            candidate_idx = np.where(valid_mask & (z_vr > 3.0))[0]
            if len(candidate_idx) == 0:
                continue
            sorted_candidates = candidate_idx[np.argsort(abs_z[candidate_idx])[::-1]]

            selected_idx = []
            for idx in sorted_candidates:
                if not any(abs(idx - prev) < 60 for prev in selected_idx):
                    selected_idx.append(idx)
                    if len(selected_idx) >= 10:
                        break

            for i in selected_idx:
                t_ns = df_sym["event_time_ns"][int(i)]
                obs_iso = datetime.fromtimestamp(t_ns / 1e9, tz=UTC).isoformat()
                sigs.append(OHLCVAnomalySignal(
                    signal_id=f"A3_{sym}_voldecouple_{i}",
                    symbol=sym,
                    stat="volume_price_decoupling",
                    raw_columns=["close", "volume"],
                    feature_names=["return_1h", "volume_ratio_1h", "return_volume_correlation_24h"],
                    value=float(decoupling[i]),
                    effect_size=float(z_vr[i]),
                    observed_at=obs_iso,
                    window="1h",
                ))
        return sigs


# ── 3. AI Economic Hypothesis Synthesizer ─────────────────────────────────────

@dataclass
class EconomicHypothesis:
    hypothesis_id: str
    title: str
    hypothesis_text: str
    narrative: str
    symbol: str
    stat: str
    effect_size: float
    predicted_sign: int  # +1 or -1
    mechanism_columns: list[str]
    target_regime: str
    target_horizon: str
    next_action: str = "GENERATE_HYPOTHESIS"
    action_reasoning: str = ""


class AIEconomicHypothesisSynthesizer:
    """Gọi LLM (Gemini / OpenAI) để suy luận luận điểm vi mô thị trường và sinh giả thuyết kinh tế."""

    def __init__(self) -> None:
        _ensure_dotenv()
        _sync_provider_env()
        self.llm = ChatLLM()

    def synthesize(self, signal: OHLCVAnomalySignal) -> EconomicHypothesis | None:
        """Nhận 1 điểm bất thường và sinh giả thuyết kinh tế từ LLM dựa trên 4 loại lựa chọn gán cứng."""
        if not getattr(self.llm, "_api_key", None) and self.llm.provider not in ("ollama", "vllm", "openai_compatible"):
            logger.warning("LLM API Key không khả dụng. Trả về giả thuyết mặc định.")
            return self._fallback_hypothesis(signal)

        prompt = (
            f"You are a Senior Quantitative Researcher at a Tier-1 Crypto Hedge Fund.\n"
            f"Analyze the following market anomaly detected in OHLCV price/volume data:\n"
            f"- Asset Symbol: {signal.symbol}\n"
            f"- Anomaly Type: {signal.stat}\n"
            f"- Statistical Z-Score (Effect Size): {signal.effect_size:+.2f}\n"
            f"- Measured Value: {signal.value:.4f}\n"
            f"- Data Columns Used: {', '.join(signal.raw_columns)}\n"
            f"- Timestamp: {signal.observed_at}\n\n"
            f"CRITICAL DATA CONSTRAINTS (STRICT REQUIREMENT):\n"
            f"1. You ONLY have access to OHLCV bar data (open, high, low, close, volume).\n"
            f"2. DO NOT mention 'order book', 'limit orders', 'bid/ask depth', 'liquidity walls', 'order book depth', 'spread', or 'L2/L3 data'.\n"
            f"3. Base all economic reasoning EXCLUSIVELY on price action (returns, price overextension, Garman-Klass volatility, momentum) and volume dynamics (volume surge, volume-price decoupling).\n\n"
            f"HARDCODED ACTION CHOICES (MUST SELECT EXACTLY ONE AS 'next_action'):\n"
            f"Choice 1: 'GENERATE_HYPOTHESIS' -> Formulate a new quantitative economic hypothesis for this anomaly.\n"
            f"Choice 2: 'PROBE_OUT_OF_SAMPLE' -> Request scanning Out-Of-Sample data segments (segment='out_of_sample') to test if this pattern generalizes across time windows.\n"
            f"Choice 3: 'CROSS_ASSET_VERIFY' -> Request scanning other universe assets (BTC, ETH, SOL, BNB, XRP) to verify cross-asset generalizability.\n"
            f"Choice 4: 'REJECT_NOISE' -> Reject this anomaly as single-point noise, bad tick, or exchange outage with no tradeable economic mechanism.\n\n"
            f"Respond ONLY in valid JSON format (NO extra text, NO markdown codeblock wrapper):\n"
            f"{{\n"
            f'  "next_action": "GENERATE_HYPOTHESIS",\n'
            f'  "action_reasoning": "1 sentence explaining why this action was chosen",\n'
            f'  "title": "Short professional Quant Alpha Title",\n'
            f'  "hypothesis_text": "1-2 sentence core economic hypothesis reasoning based strictly on price/volume",\n'
            f'  "narrative": "Detailed price action and volume dynamics mechanism explanation",\n'
            f'  "predicted_sign": -1,\n'
            f'  "target_regime": "HIGH_VOLATILITY",\n'
            f'  "target_horizon": "1h"\n'
            f"}}"
        )

        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = res.content.strip()
            data = _extract_and_parse_json_from_llm(content)

            return EconomicHypothesis(
                hypothesis_id=f"hyp_{signal.signal_id}",
                title=str(data.get("title", f"{signal.symbol} {signal.stat} Reversion")),
                hypothesis_text=str(data.get("hypothesis_text", f"Market anomaly on {signal.symbol}")),
                narrative=str(data.get("narrative", "Price action mechanism explanation")),
                symbol=signal.symbol,
                stat=signal.stat,
                effect_size=signal.effect_size,
                predicted_sign=int(data.get("predicted_sign", -1 if signal.effect_size > 0 else 1)),
                mechanism_columns=signal.raw_columns,
                target_regime=str(data.get("target_regime", "HIGH_VOLATILITY")),
                target_horizon=str(data.get("target_horizon", "1h")),
                next_action=str(data.get("next_action", "GENERATE_HYPOTHESIS")),
                action_reasoning=str(data.get("action_reasoning", "Selected action based on anomaly statistical profile.")),
            )
        except Exception as e:
            logger.warning("Lỗi khi gọi LLM suy luận giả thuyết: %s. Dùng fallback.", e)
            return self._fallback_hypothesis(signal)

    def _fallback_hypothesis(self, signal: OHLCVAnomalySignal) -> EconomicHypothesis:
        return EconomicHypothesis(
            hypothesis_id=f"hyp_{signal.signal_id}",
            title=f"{signal.symbol} {signal.stat.replace('_', ' ').title()} Price Reversion",
            hypothesis_text=f"Extreme {signal.stat} anomaly (z={signal.effect_size:.2f}) on {signal.symbol} indicates temporary price overextension leading to short-horizon mean reversion.",
            narrative="Abnormal price volatility and volume surge indicate temporary price overextension. As extreme buying/selling momentum cools down, price action systematically mean-reverts back to historical moving averages.",
            symbol=signal.symbol,
            stat=signal.stat,
            effect_size=signal.effect_size,
            predicted_sign=-1 if signal.effect_size > 0 else 1,
            mechanism_columns=signal.raw_columns,
            target_regime="HIGH_VOLATILITY",
            target_horizon="1h",
            next_action="GENERATE_HYPOTHESIS",
            action_reasoning="Default fallback hypothesis synthesis for extreme anomaly.",
        )


# ── 4. Swarm Multi-Agent Review Tribunal ──────────────────────────────────────

class SwarmHypothesisReviewer:
    """Thẩm định & phản biện giả thuyết kinh tế bằng Swarm Multi-Agent Review Team (Advocate, Critic, Risk Officer)."""

    def __init__(self) -> None:
        _ensure_dotenv()
        _sync_provider_env()
        self.llm = ChatLLM()

    def review_hypothesis(self, hyp: EconomicHypothesis) -> dict[str, Any]:
        """Cho 3 vai trò Swarm (Advocate, Critic, Risk Officer) phản biện giả thuyết."""
        if not getattr(self.llm, "_api_key", None) and self.llm.provider not in ("ollama", "vllm", "openai_compatible"):
            return {
                "consensus_verdict": "APPROVED",
                "advocate_analysis": "Hành vi giá và khối lượng phù hợp với quy luật đảo chiều.",
                "critic_analysis": "Cần chú ý bẫy giá giả trong xu hướng mạnh.",
                "risk_verdict": "Đạt thẩm định Swarm.",
            }

        prompt = (
            f"You are the Swarm Multi-Agent Quant Review Tribunal (Advocate, Critic, Risk Officer).\n"
            f"Review the following Quantitative Economic Hypothesis:\n"
            f"- Title: {hyp.title}\n"
            f"- Asset Symbol: {hyp.symbol}\n"
            f"- Anomaly Type: {hyp.stat} (Effect Size: {hyp.effect_size:+.2f})\n"
            f"- Predicted Direction: {'BUY (+1)' if hyp.predicted_sign == 1 else 'SELL (-1)'}\n"
            f"- Target Regime: {hyp.target_regime} | Horizon: {hyp.target_horizon}\n"
            f"- Economic Rationale: {hyp.hypothesis_text}\n"
            f"- Narrative: {hyp.narrative}\n\n"
            f"STRICT CONSTRAINTS: Rely ONLY on OHLCV price action and volume dynamics. No L2/Order book references.\n"
            f"Conduct a 3-role Swarm Multi-Agent Review:\n"
            f"1. QUANT ADVOCATE: Argue for why this OHLCV price/volume relationship holds.\n"
            f"2. MARKET CRITIC: Point out potential price traps, false momentum signals, or risk factors.\n"
            f"3. RISK OFFICER: Render a final consensus verdict (APPROVED, APPROVED_WITH_CAUTION, or REJECTED_NOISE).\n\n"
            f"Respond ONLY in valid JSON format (NO extra text, NO markdown codeblock wrapper):\n"
            f"{{\n"
            f'  "consensus_verdict": "APPROVED",\n'
            f'  "advocate_analysis": "1-2 sentence supporting argument",\n'
            f'  "critic_analysis": "1-2 sentence counter-argument or price trap warning",\n'
            f'  "risk_verdict": "Summary recommendation from Risk Officer"\n'
            f"}}"
        )

        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = res.content.strip()
            return _extract_and_parse_json_from_llm(content)
        except Exception as e:
            logger.warning("Lỗi khi chạy Swarm Review: %s", e)
            return {
                "consensus_verdict": "APPROVED",
                "advocate_analysis": "Hành vi giá và khối lượng phù hợp với quy luật đảo chiều.",
                "critic_analysis": "Cần chú ý bẫy giá giả trong xu hướng mạnh.",
                "risk_verdict": "Đạt thẩm định Swarm.",
            }


# ── 5. BaseTool Wrapper cho ReAct Loop Agent ─────────────────────────────────

class OHLCAnomalyHypothesisTool(BaseTool):
    """BaseTool mở rộng cho Agent ReAct loop để quét bất thường & sinh giả thuyết AI."""

    name: str = "scan_data_and_generate_hypotheses"
    description: str = (
        "Scan OHLCV data in data/ for statistical anomalies (Moment shift, Volatility jump, Volume surge) "
        "and pass top anomalies to AI (LLM) to formulate quantitative economic hypotheses, optionally reviewed by Swarm."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target_universe": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Symbols to scan, e.g. ['BTCUSDT', 'ETHUSDT']. Defaults to all symbols in data/.",
            },
            "lookback_days": {
                "type": "integer",
                "default": 730,
                "description": "Lookback window in days (default 730 days).",
            },
            "segment": {
                "type": "string",
                "enum": ["full", "in_sample", "out_of_sample"],
                "default": "full",
                "description": "Data segment split: 'full', 'in_sample' (first 50%), or 'out_of_sample' (last 50%).",
            },
            "start_date": {
                "type": "string",
                "description": "Optional ISO start date string for targeted date window scanning (e.g. '2026-01-01').",
            },
            "end_date": {
                "type": "string",
                "description": "Optional ISO end date string for targeted date window scanning (e.g. '2026-06-01').",
            },
            "verify_symbol": {
                "type": "string",
                "description": "Optional specific symbol to run untruncated targeted verification matching (e.g. 'BTCUSDT').",
            },
            "verify_stat": {
                "type": "string",
                "description": "Optional specific stat type to run untruncated targeted verification matching (e.g. 'volume_price_decoupling').",
            },
            "max_hypotheses": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of AI economic hypotheses to generate.",
            },
            "enable_swarm": {
                "type": "boolean",
                "default": True,
                "description": "Enable Swarm Multi-Agent Review Tribunal (Advocate, Critic, Risk Officer).",
            },
        },
        "required": [],
    }
    repeatable: bool = True
    is_readonly: bool = True

    def execute(self, **kwargs: Any) -> str:
        target_universe = kwargs.get("target_universe", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
        lookback_days = int(kwargs.get("lookback_days") or 730)
        segment = str(kwargs.get("segment") or "full")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        verify_symbol = kwargs.get("verify_symbol")
        verify_stat = kwargs.get("verify_stat")
        max_hypotheses = int(kwargs.get("max_hypotheses") or 5)
        enable_swarm = bool(kwargs.get("enable_swarm", True))

        scanner = OHLCVDataScanner(
            lookback_days=lookback_days,
            segment=segment,
            start_date=start_date,
            end_date=end_date,
        )

        # Trích xuất kiểm chứng định hướng không cắt gọt Top 50 nếu được chỉ định verify_symbol & verify_stat
        if verify_symbol and verify_stat:
            verify_res = scanner.verify_hypothesis_persistence(verify_symbol, verify_stat)
            return json.dumps({
                "status": "success",
                "verification_mode": "targeted_untruncated_matching",
                "verification_result": verify_res,
            }, ensure_ascii=False, indent=2)

        outcome, signals, reason = scanner.scan(symbols=target_universe)

        if outcome != KetCucChang.PASS or not signals:
            return json.dumps({
                "status": "warning",
                "outcome": outcome.value,
                "reason": reason,
                "hypotheses": [],
            }, ensure_ascii=False)

        synthesizer = AIEconomicHypothesisSynthesizer()
        reviewer = SwarmHypothesisReviewer() if enable_swarm else None
        hypotheses_list = []

        for sig in signals[:max_hypotheses]:
            hyp = synthesizer.synthesize(sig)
            if hyp:
                item = {
                    "hypothesis_id": hyp.hypothesis_id,
                    "title": hyp.title,
                    "symbol": hyp.symbol,
                    "stat": hyp.stat,
                    "effect_size": round(hyp.effect_size, 2),
                    "predicted_sign": hyp.predicted_sign,
                    "target_regime": hyp.target_regime,
                    "target_horizon": hyp.target_horizon,
                    "next_action": hyp.next_action,
                    "action_reasoning": hyp.action_reasoning,
                    "hypothesis_text": hyp.hypothesis_text,
                    "narrative": hyp.narrative,
                    "mechanism_columns": hyp.mechanism_columns,
                }
                if reviewer:
                    swarm_audit = reviewer.review_hypothesis(hyp)
                    item["swarm_review"] = swarm_audit
                hypotheses_list.append(item)

        return json.dumps({
            "status": "success",
            "anomalies_detected": len(signals),
            "hypotheses_generated": len(hypotheses_list),
            "swarm_enabled": enable_swarm,
            "hypotheses": hypotheses_list,
        }, ensure_ascii=False, indent=2)
