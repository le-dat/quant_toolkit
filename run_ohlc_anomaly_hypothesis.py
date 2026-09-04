"""run_ohlc_anomaly_hypothesis.py — Script chạy trực tiếp công cụ quét dị biệt data/ & sinh giả thuyết AI.

Sử dụng:
    python run_ohlc_anomaly_hypothesis.py [--symbols BTCUSDT ETHUSDT] [--max-hypotheses 5]
"""

import sys
import json
import argparse
from pathlib import Path

# Thêm thư mục chứa gói pseud vào sys.path
root = Path(__file__).resolve().parent
if str(root.parent) not in sys.path:
    sys.path.insert(0, str(root.parent))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pseud.tools.ohlc_anomaly_hypothesis_tool import OHLCAnomalyHypothesisTool  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Quét bất thường dữ liệu OHLCV trong data/ và truyền cho AI sinh Giả thuyết Kinh tế."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        help="Danh sách các cặp giao dịch cần quét (mặc định: top 5 symbols)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Cửa sổ quét dữ liệu theo ngày (mặc định: 730 ngày / 2 năm)",
    )
    parser.add_argument(
        "--max-hypotheses",
        type=int,
        default=5,
        help="Số lượng Giả thuyết Kinh tế AI tối đa cần tạo (mặc định: 5)",
    )
    parser.add_argument(
        "--enable-swarm",
        action="store_true",
        default=True,
        help="Bật Swarm Multi-Agent Review Tribunal (Advocate, Critic, Risk Officer) (mặc định: True)",
    )
    parser.add_argument(
        "--disable-swarm",
        dest="enable_swarm",
        action="store_false",
        help="Tắt Swarm Multi-Agent Review Tribunal",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hypotheses_report.json",
        help="Tệp JSON lưu kết quả giả thuyết (mặc định: hypotheses_report.json)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MÁY QUÉT DỊ BIỆT DỮ LIỆU OHLCV & AI SINH GIẢ THUYẾT KINH TẾ (SWARM REVIEW)")
    print("=" * 70)
    print(f" Target Symbols: {args.symbols}")
    print(f" Lookback Window: {args.lookback_days} days")
    print(f" Max Hypotheses: {args.max_hypotheses}")
    print(f" Swarm Review Team: {'ĐÃ BẬT (Advocate, Critic, Risk Officer)' if args.enable_swarm else 'TẮT'}")
    print("-" * 70)

    tool = OHLCAnomalyHypothesisTool()
    res_str = tool.execute(
        target_universe=args.symbols,
        lookback_days=args.lookback_days,
        max_hypotheses=args.max_hypotheses,
        enable_swarm=args.enable_swarm,
    )

    result = json.loads(res_str)

    if result.get("status") != "success":
        print(f"❌ Thông báo: {result.get('reason')}")
        return

    anomalies_count = result.get("anomalies_detected", 0)
    hypotheses = result.get("hypotheses", [])

    print(f"✅ Đã phát hiện {anomalies_count} tín hiệu dị biệt OHLCV.")
    print(f"✅ AI & Swarm Review đã hoàn tất {len(hypotheses)} Giả thuyết Kinh tế Định lượng:\n")

    for idx, hyp in enumerate(hypotheses, 1):
        sign_str = "BUY (+1)" if hyp["predicted_sign"] == 1 else "SELL (-1)"
        print(f"📌 [{idx}] {hyp['title']} ({hyp['symbol']})")
        print(f"   • Dị biệt phát hiện: {hyp['stat']} (Z-Score: {hyp['effect_size']:+.2f})")
        print(f"   • Hướng dự đoán: {sign_str} | Chế độ: {hyp['target_regime']} | Khung: {hyp['target_horizon']}")
        print(f"   • Cột dữ liệu: {', '.join(hyp['mechanism_columns'])}")
        print(f"   • Luận điểm kinh tế: {hyp['hypothesis_text']}")
        print(f"   • Cơ lý thị trường: {hyp['narrative']}")

        if "swarm_review" in hyp:
            s_review = hyp["swarm_review"]
            print("\n   👥 SWARM MULTI-AGENT REVIEW TRIBUNAL:")
            print(f"      • Kết luận Swarm: {s_review.get('consensus_verdict')}")
            print(f"      • Quant Advocate (+): {s_review.get('advocate_analysis')}")
            print(f"      • Market Critic (-): {s_review.get('critic_analysis')}")
            print(f"      • Risk Officer (🛡️): {s_review.get('risk_verdict')}")
        print("-" * 70)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Báo cáo chi tiết Swarm đã lưu vào tệp: {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
