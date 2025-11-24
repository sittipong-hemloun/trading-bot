"""
Monthly Trading Strategy Module
Contains MonthlyTradingStrategy class for monthly position trading
"""

import pandas as pd
import pandas_ta as ta
from datetime import timedelta
from typing import Optional

from trading.base_strategy import BaseStrategy


class MonthlyTradingStrategy(BaseStrategy):
    """
    Strategy สำหรับ Trade รอบละ 1 เดือน

    Timeframes analyzed:
    - Monthly: Primary trend direction
    - Weekly: Confirmation
    - Daily: Fine-tuning entries

    Inherits from BaseStrategy for common functionality.
    """

    def __init__(self, symbol: str = "BTCUSDT", leverage: int = 3):
        """
        Initialize Monthly Trading Strategy

        Args:
            symbol: Trading pair (default: BTCUSDT)
            leverage: Leverage multiplier (default: 3x for position trading)
        """
        super().__init__(
            symbol=symbol,
            leverage=leverage,
            timeframes={"monthly": "1M", "weekly": "1w", "daily": "1d"}
        )

    def _get_timeframe_weights(self) -> dict[str, int]:
        """Get weights for each timeframe (Monthly > Weekly > Daily)"""
        return {"monthly": 4, "weekly": 2, "daily": 1}

    def analyze_multi_timeframe(self) -> Optional[bool]:
        """Fetch and analyze data across Monthly, Weekly, Daily timeframes"""
        print("📊 กำลังดึงข้อมูล Monthly...")
        monthly_data = self.fetch_data(self.timeframes["monthly"], 60)
        weekly_data = self.fetch_data(self.timeframes["weekly"], 104)
        daily_data = self.fetch_data(self.timeframes["daily"], 200)

        if monthly_data is None or weekly_data is None or daily_data is None:
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return None

        if monthly_data.empty or weekly_data.empty or daily_data.empty:
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return None

        # Use monthly-specific indicator calculation
        self.data["monthly"] = self._calculate_monthly_indicators(monthly_data)
        self.data["weekly"] = self._calculate_monthly_indicators(weekly_data)
        self.data["daily"] = self._calculate_monthly_indicators(daily_data)

        return True

    # === ABSTRACT METHOD IMPLEMENTATIONS ===

    def get_signal(self) -> tuple[dict, dict]:
        """Generate trading signals (implements abstract method)"""
        return self.get_monthly_signal()

    def get_recommendation(self, balance: float) -> None:
        """Display trading recommendation (implements abstract method)"""
        self.get_monthly_recommendation(balance)

    # Note: The following methods are inherited from BaseStrategy:
    # - fetch_data, detect_market_regime, analyze_historical_performance
    # - calculate_risk_score, calculate_volatility_adjusted_risk
    # - check_divergence, get_confidence_level

    # === MONTHLY-SPECIFIC METHODS ===

    def _calculate_monthly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """คำนวณตัวชี้วัดสำหรับ Monthly Strategy (ใช้ EMA 12/26)"""

        # === MOVING AVERAGES (Monthly uses EMA 12/26 instead of 9/21) ===
        df["EMA_12"] = ta.ema(df["close"], length=12)
        df["EMA_26"] = ta.ema(df["close"], length=26)
        df["EMA_50"] = ta.ema(df["close"], length=50)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_200"] = ta.sma(df["close"], length=200)

        # Also calculate EMA 9/21 for compatibility with base methods
        df["EMA_9"] = ta.ema(df["close"], length=9)
        df["EMA_21"] = ta.ema(df["close"], length=21)

        # === RSI ===
        df["RSI"] = ta.rsi(df["close"], length=14)

        # === MACD ===
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_histogram"] = macd["MACDh_12_26_9"]

        # === Stochastic RSI ===
        stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        df["STOCHRSI_K"] = stochrsi["STOCHRSIk_14_14_3_3"]
        df["STOCHRSI_D"] = stochrsi["STOCHRSId_14_14_3_3"]

        # === Stochastic ===
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        df["STOCH_K"] = stoch["STOCHk_14_3_3"]
        df["STOCH_D"] = stoch["STOCHd_14_3_3"]

        # === Bollinger Bands ===
        bbands = ta.bbands(df["close"], length=20, std=2.0)  # type: ignore[arg-type]
        df["BB_upper"] = bbands["BBU_20_2.0_2.0"]
        df["BB_middle"] = bbands["BBM_20_2.0_2.0"]
        df["BB_lower"] = bbands["BBL_20_2.0_2.0"]

        # === ADX ===
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["ADX"] = adx["ADX_14"]
        df["DI_plus"] = adx["DMP_14"]
        df["DI_minus"] = adx["DMN_14"]

        # === ATR ===
        df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ATR_percent"] = df["ATR"] / df["close"] * 100

        # === Volume ===
        df["Volume_MA"] = df["volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["volume"] / df["Volume_MA"]

        # === OBV ===
        df["OBV"] = ta.obv(df["close"], df["volume"])
        df["OBV_EMA"] = ta.ema(df["OBV"], length=21)

        # === MFI ===
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

        # === Supertrend ===
        supertrend = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        df["SUPERTREND"] = supertrend["SUPERT_10_3.0"]
        df["SUPERTREND_DIR"] = supertrend["SUPERTd_10_3.0"]

        return df

    def get_monthly_signal(self) -> tuple[dict, dict]:
        """วิเคราะห์สัญญาณ Monthly แบบปรับปรุง พร้อมข้อมูลย้อนหลัง"""

        monthly = self.data["monthly"].iloc[-1]
        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]

        monthly_prev = self.data["monthly"].iloc[-2]
        weekly_prev = self.data["weekly"].iloc[-2]

        # วิเคราะห์ Market Regime และ Historical Performance
        market_regime = self.detect_market_regime(self.data["weekly"])
        historical_perf = self.analyze_historical_performance(self.data["weekly"])

        signals = {"long": 0, "short": 0, "neutral": 0}
        reasons = {"long": [], "short": [], "neutral": []}

        # เพิ่มข้อมูล Market Context
        regime_text = market_regime["regime"].replace("_", " ")
        reasons["neutral"].append(f"📈 Market Regime: {regime_text} ({market_regime['confidence']:.0f}%)")

        if historical_perf["total_signals"] > 0:
            reasons["neutral"].append(
                f"📊 Historical: Win Rate {historical_perf['win_rate']:.1f}%, "
                f"Avg Return {historical_perf['avg_return']:.2f}%"
            )

        # === MONTHLY TIMEFRAME ANALYSIS ===
        if monthly["EMA_12"] > monthly["EMA_26"]:
            signals["long"] += 4
            reasons["long"].append("📈 Monthly Uptrend: EMA 12 > 26")
        elif monthly["EMA_12"] < monthly["EMA_26"]:
            signals["short"] += 4
            reasons["short"].append("📉 Monthly Downtrend: EMA 12 < 26")

        if monthly_prev["EMA_12"] <= monthly_prev["EMA_26"] and monthly["EMA_12"] > monthly["EMA_26"]:
            signals["long"] += 5
            reasons["long"].append("🔥 Monthly Golden Cross!")
        elif monthly_prev["EMA_12"] >= monthly_prev["EMA_26"] and monthly["EMA_12"] < monthly["EMA_26"]:
            signals["short"] += 5
            reasons["short"].append("🔥 Monthly Death Cross!")

        if monthly["RSI"] < 30:
            signals["long"] += 3
            reasons["long"].append(f"💪 Monthly RSI Oversold: {monthly['RSI']:.1f}")
        elif monthly["RSI"] > 70:
            signals["short"] += 3
            reasons["short"].append(f"⚠️ Monthly RSI Overbought: {monthly['RSI']:.1f}")
        elif 45 < monthly["RSI"] < 55:
            signals["neutral"] += 1
            reasons["neutral"].append(f"😐 Monthly RSI Neutral: {monthly['RSI']:.1f}")

        if monthly["MACD"] > monthly["MACD_signal"] and monthly["MACD_histogram"] > 0:
            signals["long"] += 3
            reasons["long"].append("📊 Monthly MACD Bullish")
        elif monthly["MACD"] < monthly["MACD_signal"] and monthly["MACD_histogram"] < 0:
            signals["short"] += 3
            reasons["short"].append("📊 Monthly MACD Bearish")

        if pd.notna(monthly.get("SUPERTREND_DIR")):
            if monthly["SUPERTREND_DIR"] == 1:
                signals["long"] += 2
                reasons["long"].append("🚀 Monthly Supertrend Bullish")
            else:
                signals["short"] += 2
                reasons["short"].append("🔻 Monthly Supertrend Bearish")

        # === WEEKLY TIMEFRAME CONFIRMATION ===
        if weekly["EMA_12"] > weekly["EMA_26"]:
            signals["long"] += 2
            reasons["long"].append("📈 Weekly Uptrend")
        elif weekly["EMA_12"] < weekly["EMA_26"]:
            signals["short"] += 2
            reasons["short"].append("📉 Weekly Downtrend")

        if weekly["RSI"] < 35:
            signals["long"] += 2
            reasons["long"].append(f"💪 Weekly RSI: {weekly['RSI']:.1f}")
        elif weekly["RSI"] > 65:
            signals["short"] += 2
            reasons["short"].append(f"⚠️ Weekly RSI: {weekly['RSI']:.1f}")

        if weekly_prev["MACD"] <= weekly_prev["MACD_signal"] and weekly["MACD"] > weekly["MACD_signal"]:
            signals["long"] += 2
            reasons["long"].append("✅ Weekly MACD Cross Up")
        elif weekly_prev["MACD"] >= weekly_prev["MACD_signal"] and weekly["MACD"] < weekly["MACD_signal"]:
            signals["short"] += 2
            reasons["short"].append("❌ Weekly MACD Cross Down")

        # === DAILY TIMEFRAME ===
        if daily["EMA_12"] > daily["EMA_26"]:
            signals["long"] += 1
            reasons["long"].append("📊 Daily Aligned Bullish")
        elif daily["EMA_12"] < daily["EMA_26"]:
            signals["short"] += 1
            reasons["short"].append("📊 Daily Aligned Bearish")

        # Divergence Detection แบบปรับปรุง
        daily_divergence, div_strength = self.check_divergence(self.data["daily"], "RSI")
        if daily_divergence == "bullish" and div_strength > 0:
            div_score = 2 if div_strength < 30 else 3 if div_strength < 60 else 4
            signals["long"] += div_score
            reasons["long"].append(f"🔄 Daily Bullish Divergence (Strength: {div_strength:.0f})")
        elif daily_divergence == "bearish" and div_strength > 0:
            div_score = 2 if div_strength < 30 else 3 if div_strength < 60 else 4
            signals["short"] += div_score
            reasons["short"].append(f"🔄 Daily Bearish Divergence (Strength: {div_strength:.0f})")

        # Weekly Divergence
        weekly_divergence, weekly_div_strength = self.check_divergence(self.data["weekly"], "RSI", lookback=20)
        if weekly_divergence == "bullish" and weekly_div_strength > 20:
            signals["long"] += 3
            reasons["long"].append("🔄 Weekly Bullish Divergence")
        elif weekly_divergence == "bearish" and weekly_div_strength > 20:
            signals["short"] += 3
            reasons["short"].append("🔄 Weekly Bearish Divergence")

        # === TREND STRENGTH ===
        if monthly["ADX"] > 25:
            if monthly["DI_plus"] > monthly["DI_minus"]:
                signals["long"] += 2
                reasons["long"].append(f"💪 Strong Uptrend (ADX: {monthly['ADX']:.1f})")
            else:
                signals["short"] += 2
                reasons["short"].append(f"💪 Strong Downtrend (ADX: {monthly['ADX']:.1f})")
        else:
            signals["neutral"] += 2
            reasons["neutral"].append(f"🌊 Weak Trend (ADX: {monthly['ADX']:.1f})")

        if daily["Volume_Ratio"] > 1.5:
            if daily["close"] > daily["open"]:
                signals["long"] += 1
                reasons["long"].append("📊 High Volume Bullish")
            else:
                signals["short"] += 1
                reasons["short"].append("📊 High Volume Bearish")

        return signals, reasons

    def calculate_position_management(self, current_price: float, signal_type: str) -> dict:
        """คำนวณการจัดการ Position สำหรับ Monthly แบบปรับปรุง"""

        monthly_df = self.data["monthly"]
        weekly_df = self.data["weekly"]
        monthly = monthly_df.iloc[-1]

        atr_monthly = monthly["ATR"]
        atr_percent = monthly["ATR_percent"]
        sr = self.calculate_support_resistance(monthly_df)
        fib_levels, fib_trend = self.calculate_fibonacci_levels(monthly_df)

        # Market Regime และ Volatility Analysis
        market_regime = self.detect_market_regime(weekly_df)
        vol_risk = self.calculate_volatility_adjusted_risk(weekly_df)

        # ปรับ Multipliers ตาม Market Regime
        regime = market_regime["regime"]
        if regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
            sl_multiplier = 2.5
            tp_multiplier = [4, 6, 10]
        elif regime == "RANGING":
            sl_multiplier = 1.5
            tp_multiplier = [2, 3, 5]
        else:
            sl_multiplier = 2.0
            tp_multiplier = [3, 5, 8]

        if signal_type == "LONG":
            stop_loss_support = sr["main_support"]
            stop_loss_atr = current_price - (atr_monthly * sl_multiplier)
            stop_loss = max(stop_loss_support, stop_loss_atr)

            tp1 = current_price + (atr_monthly * tp_multiplier[0])
            tp2 = current_price + (atr_monthly * tp_multiplier[1])
            tp3 = current_price + (atr_monthly * tp_multiplier[2])

            # ใช้ Resistance เป็น TP ถ้าใกล้กว่า
            if sr["main_resistance"] < tp2 and sr["main_resistance"] > current_price:
                tp2 = sr["main_resistance"]

            for level, price in fib_levels.items():
                if price > current_price and "1.272" in level:
                    tp3 = max(tp3, price)
        else:
            stop_loss_resistance = sr["main_resistance"]
            stop_loss_atr = current_price + (atr_monthly * sl_multiplier)
            stop_loss = min(stop_loss_resistance, stop_loss_atr)

            tp1 = current_price - (atr_monthly * tp_multiplier[0])
            tp2 = current_price - (atr_monthly * tp_multiplier[1])
            tp3 = current_price - (atr_monthly * tp_multiplier[2])

            # ใช้ Support เป็น TP ถ้าใกล้กว่า
            if sr["main_support"] > tp2 and sr["main_support"] < current_price:
                tp2 = sr["main_support"]

            for level, price in fib_levels.items():
                if price < current_price and "1.272" in level:
                    tp3 = min(tp3, price)

        return {
            "entry": current_price,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "atr": atr_monthly,
            "atr_percent": atr_percent,
            "support_resistance": sr,
            "fibonacci": fib_levels,
            "fib_trend": fib_trend,
            "market_regime": market_regime,
            "volatility_risk": vol_risk,
        }

    def get_monthly_recommendation(self, balance: float = 10000) -> None:
        """แสดงคำแนะนำ Monthly Trading"""

        if not self.analyze_multi_timeframe():
            return

        signals, reasons = self.get_monthly_signal()

        monthly = self.data["monthly"].iloc[-1]
        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]

        current_price = daily["close"]

        print("=" * 100)
        print(f"🌙 MONTHLY TRADING STRATEGY - {self.symbol}")
        print(f"💰 Leverage: {self.leverage}x | 📅 Hold Period: ~1 เดือน")
        print(f"⏰ วันที่: {daily['timestamp'].strftime('%Y-%m-%d')}")
        print("=" * 100)

        print(f"\n💵 ราคาปัจจุบัน: ${current_price:,.2f}")

        print("\n📊 MULTI-TIMEFRAME ANALYSIS:")
        print("\n🌙 Monthly Indicators:")
        print(f"  • EMA 12/26: ${monthly['EMA_12']:,.2f} / ${monthly['EMA_26']:,.2f}")
        print(f"  • RSI: {monthly['RSI']:.2f}")
        print(f"  • MACD: {monthly['MACD']:.2f}")
        print(f"  • ADX: {monthly['ADX']:.2f}")
        print(f"  • ATR: ${monthly['ATR']:,.2f} ({monthly['ATR_percent']:.2f}%)")
        if pd.notna(monthly.get("SUPERTREND_DIR")):
            st_dir = "Bullish 🟢" if monthly["SUPERTREND_DIR"] == 1 else "Bearish 🔴"
            print(f"  • Supertrend: {st_dir}")

        print("\n📅 Weekly Indicators:")
        print(f"  • EMA 12/26: ${weekly['EMA_12']:,.2f} / ${weekly['EMA_26']:,.2f}")
        print(f"  • RSI: {weekly['RSI']:.2f}")
        print(f"  • MACD: {weekly['MACD']:.2f}")

        print("\n📈 Daily Indicators:")
        print(f"  • EMA 12/26: ${daily['EMA_12']:,.2f} / ${daily['EMA_26']:,.2f}")
        print(f"  • RSI: {daily['RSI']:.2f}")
        print(f"  • Volume Ratio: {daily['Volume_Ratio']:.2f}x")

        total = signals["long"] + signals["short"] + signals["neutral"]
        long_pct = (signals["long"] / total * 100) if total > 0 else 0
        short_pct = (signals["short"] / total * 100) if total > 0 else 0
        neutral_pct = (signals["neutral"] / total * 100) if total > 0 else 0

        print("\n" + "=" * 100)
        print("📊 SIGNAL ANALYSIS")
        print(f"📈 Total Score: {total} points (🟢 {long_pct:.1f}% / 🔴 {short_pct:.1f}% / ⚪ {neutral_pct:.1f}%)")
        print("=" * 100)

        print(f"\n🟢 LONG Signals: {signals['long']} ({long_pct:.1f}%)")
        for reason in reasons["long"]:
            print(f"  {reason}")

        print(f"\n🔴 SHORT Signals: {signals['short']} ({short_pct:.1f}%)")
        for reason in reasons["short"]:
            print(f"  {reason}")

        print(f"\n⚪ NEUTRAL Signals: {signals['neutral']} ({neutral_pct:.1f}%)")
        for reason in reasons["neutral"]:
            print(f"  {reason}")

        print("\n" + "=" * 100)
        print("🎯 MONTHLY RECOMMENDATION")
        print("=" * 100)

        recommendation, confidence = self.get_confidence_level(signals)

        if abs(long_pct - short_pct) < 15:
            print("\n⚠️ WARNING: Mixed signals detected - proceed with caution!")

        if recommendation in ["STRONG_LONG", "LONG"]:
            signal_type = "LONG"
            position_mgmt = self.calculate_position_management(current_price, signal_type)

            conf_text = "STRONG" if recommendation == "STRONG_LONG" else "MODERATE"
            print(f"\n✅ {conf_text} LONG SIGNAL ({confidence:.1f}%)")
            print("💡 แนะนำ: เปิด Long Position และ Hold 1 เดือน")

            self._print_trade_setup(position_mgmt, signal_type, balance, current_price)

        elif recommendation in ["STRONG_SHORT", "SHORT"]:
            signal_type = "SHORT"
            position_mgmt = self.calculate_position_management(current_price, signal_type)

            conf_text = "STRONG" if recommendation == "STRONG_SHORT" else "MODERATE"
            print(f"\n❌ {conf_text} SHORT SIGNAL ({confidence:.1f}%)")
            print("💡 แนะนำ: เปิด Short Position และ Hold 1 เดือน")

            self._print_trade_setup(position_mgmt, signal_type, balance, current_price)

        else:
            print(f"\n⏸️ WAIT - ไม่มีสัญญาณชัดเจน ({confidence:.1f}%)")
            print("💡 แนะนำ: รอสัญญาณที่ชัดเจนกว่านี้")
            print("📌 ตรวจสอบใหม่อีกครั้งในอีก 1-2 สัปดาห์")

            sr = self.calculate_support_resistance(self.data["monthly"])
            print("\n📊 LEVELS TO WATCH:")
            print(f"  🛡️ Support: ${sr['main_support']:,.2f}")
            print(f"  🔒 Resistance: ${sr['main_resistance']:,.2f}")

        print("\n" + "=" * 100)
        print("📅 NEXT REVIEW DATE: " + (daily["timestamp"] + timedelta(days=7)).strftime("%Y-%m-%d"))
        print("⚠️ คำเตือน: ตรวจสอบสถานะทุก 1 สัปดาห์ แต่ไม่ต้อง Trade บ่อย")
        print("💰 ใช้ Leverage ต่ำ (2-5x) เพื่อความปลอดภัยในระยะยาว")
        print("🎯 Patience is Key - ให้เวลากับ Position ทำงาน")
        print("=" * 100)

    def _print_trade_setup(self, position_mgmt: dict, signal_type: str, balance: float, current_price: float) -> None:
        """พิมพ์ Trade Setup พร้อมข้อมูล Advanced Analysis"""
        entry = position_mgmt["entry"]
        sl = position_mgmt["stop_loss"]
        tp1 = position_mgmt["tp1"]
        tp2 = position_mgmt["tp2"]
        tp3 = position_mgmt["tp3"]

        if signal_type == "LONG":
            sl_pct = ((entry - sl) / entry) * 100
            tp1_pct = ((tp1 - entry) / entry) * 100
            tp2_pct = ((tp2 - entry) / entry) * 100
            tp3_pct = ((tp3 - entry) / entry) * 100
        else:
            sl_pct = ((sl - entry) / entry) * 100
            tp1_pct = ((entry - tp1) / entry) * 100
            tp2_pct = ((entry - tp2) / entry) * 100
            tp3_pct = ((entry - tp3) / entry) * 100

        # Market Regime Info
        if "market_regime" in position_mgmt:
            regime = position_mgmt["market_regime"]
            regime_text = regime["regime"].replace("_", " ")
            print("\n🌍 MARKET CONTEXT:")
            print(f"  • Regime: {regime_text} ({regime['confidence']:.0f}% confidence)")
            print(f"  • ADX: {regime['adx']:.1f}")
            print(f"  • Price Range: {regime['price_range_pct']:.1f}%")

        # Volatility Info
        if "volatility_risk" in position_mgmt:
            vol = position_mgmt["volatility_risk"]
            vol_status = "🔴" if vol["volatility_ratio"] > 1.3 else "🟢" if vol["volatility_ratio"] < 0.8 else "🟡"
            print("\n📊 VOLATILITY ANALYSIS:")
            print(f"  • Current ATR: {vol['current_atr_pct']:.2f}% | Avg: {vol['avg_atr_pct']:.2f}%")
            print(f"  • Volatility Ratio: {vol_status} {vol['volatility_ratio']:.2f}x")
            print(f"  • Risk Adjustment: {vol['risk_note']}")
            adjusted_risk = vol["adjusted_risk_pct"]
        else:
            adjusted_risk = 2.0

        print(f"\n📊 ATR: ${position_mgmt['atr']:,.2f} ({position_mgmt['atr_percent']:.2f}%)")

        print("\n💼 TRADE SETUP:")
        print(f"  🎯 Entry: ${entry:,.2f}")
        print(f"  🛡️ Stop Loss: ${sl:,.2f} ({sl_pct:+.2f}% = {sl_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP1 (33%): ${tp1:,.2f} ({tp1_pct:+.2f}% = {tp1_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP2 (33%): ${tp2:,.2f} ({tp2_pct:+.2f}% = {tp2_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP3 (34%): ${tp3:,.2f} ({tp3_pct:+.2f}% = {tp3_pct * self.leverage:+.1f}% margin)")

        # ใช้ Volatility-adjusted Risk
        risk_pct = adjusted_risk
        risk_amount = balance * (risk_pct / 100)
        position_size = (risk_amount / (abs(sl_pct) / 100)) * self.leverage
        margin_required = position_size / self.leverage

        print(f"\n💰 POSITION MANAGEMENT (Balance: ${balance:,.2f}):")
        print(f"  📊 Leverage: {self.leverage}x")
        print(f"  📊 Risk per Trade: {risk_pct:.1f}% (${risk_amount:,.2f}) - Volatility Adjusted")
        print(f"  📊 Margin Required: ${margin_required:,.2f}")
        print(f"  📊 Position Size: ${position_size:,.2f}")

        print("\n📊 Risk/Reward Ratio:")
        print(f"  • TP1: 1:{abs(tp1_pct/sl_pct):.2f}")
        print(f"  • TP2: 1:{abs(tp2_pct/sl_pct):.2f}")
        print(f"  • TP3: 1:{abs(tp3_pct/sl_pct):.2f}")

        sr = position_mgmt["support_resistance"]
        print("\n🛡️ SUPPORT LEVELS:")
        for i, support in enumerate(sr["support"], 1):
            print(f"  S{i}: ${support:,.2f}")

        print("\n🔒 RESISTANCE LEVELS:")
        for i, resistance in enumerate(sr["resistance"], 1):
            print(f"  R{i}: ${resistance:,.2f}")

        fib = position_mgmt["fibonacci"]
        print(f"\n🎯 FIBONACCI LEVELS ({position_mgmt['fib_trend'].upper()}):")
        for level, price in fib.items():
            marker = "👉" if abs(price - current_price) / current_price < 0.02 else "  "
            print(f"  {marker} {level}: ${price:,.2f}")

        action = "Long" if signal_type == "LONG" else "Short"
        print("\n📅 MONTHLY STRATEGY:")
        print(f"  1️⃣ เปิด {action} ที่ราคาปัจจุบัน ${entry:,.2f}")
        print(f"  2️⃣ ตั้ง Stop Loss ที่ ${sl:,.2f}")
        print("  3️⃣ ปิด 33% ที่ TP1, 33% ที่ TP2, 34% ที่ TP3")
        print("  4️⃣ ถ้าถึง TP1 → ขยับ SL ไปที่ Entry (Break Even)")
        print("  5️⃣ Review ทุก 1 สัปดาห์")
        print("  6️⃣ Hold จนกว่าจะถึง TP หรือ SL หรือสัญญาณกลับตัว")
