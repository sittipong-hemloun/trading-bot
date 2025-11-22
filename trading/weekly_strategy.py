"""
Weekly Trading Strategy Module
Contains WeeklyTradingStrategy class for weekly swing trading
"""

import pandas as pd
from datetime import timedelta
from typing import Literal, Optional

from trading.base_strategy import BaseStrategy


class WeeklyTradingStrategy(BaseStrategy):
    """
    Strategy สำหรับ Trade รอบละ 1 สัปดาห์

    Timeframes analyzed:
    - Weekly: Primary trend direction
    - Daily: Entry timing and confirmation
    - 4H: Fine-tuning entries

    Inherits from BaseStrategy for common functionality.
    """

    def __init__(self, symbol: str = "BTCUSDT", leverage: int = 5):
        """
        Initialize Weekly Trading Strategy

        Args:
            symbol: Trading pair (default: BTCUSDT)
            leverage: Leverage multiplier (default: 5x for swing trading)
        """
        super().__init__(
            symbol=symbol,
            leverage=leverage,
            timeframes={"weekly": "1w", "daily": "1d", "h4": "4h"}
        )

    def _get_timeframe_weights(self) -> dict[str, int]:
        """Get weights for each timeframe (Weekly > Daily > 4H)"""
        return {"weekly": 3, "daily": 2, "h4": 1}

    def analyze_multi_timeframe(self) -> Optional[bool]:
        """Fetch and analyze data across Weekly, Daily, 4H timeframes"""
        print("📊 กำลังดึงข้อมูล...")
        weekly_data = self.fetch_data(self.timeframes["weekly"], 52)
        daily_data = self.fetch_data(self.timeframes["daily"], 100)
        h4_data = self.fetch_data(self.timeframes["h4"], 200)

        if weekly_data is None or daily_data is None or h4_data is None:
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return None

        if weekly_data.empty or daily_data.empty or h4_data.empty:
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return None

        self.data["weekly"] = self.calculate_indicators(weekly_data)
        self.data["daily"] = self.calculate_indicators(daily_data)
        self.data["h4"] = self.calculate_indicators(h4_data)

        return True

    # Note: The following methods are inherited from BaseStrategy:
    # - fetch_data, calculate_indicators
    # - get_multi_indicator_confirmation, get_volume_confirmation
    # - find_confluence_zones, get_dynamic_thresholds
    # - check_divergence, detect_market_regime, analyze_historical_performance
    # - calculate_risk_score, calculate_volatility_adjusted_risk
    # - calculate_support_resistance, calculate_fibonacci_levels
    # - get_trend_strength, check_trend_consistency, get_confidence_level

    # === ABSTRACT METHOD IMPLEMENTATIONS ===

    def get_signal(self) -> tuple[dict, dict]:
        """Generate trading signals (implements abstract method)"""
        return self.get_weekly_signal()

    def get_recommendation(self, balance: float) -> None:
        """Display trading recommendation (implements abstract method)"""
        self.get_weekly_recommendation(balance)

    # === WEEKLY-SPECIFIC METHODS ===

    def get_weighted_signal_score(
        self,
        base_score: int,
        timeframe: Literal["weekly", "daily", "h4"],
        market_regime: dict,
        historical_perf: dict
    ) -> float:
        """คำนวณคะแนนสัญญาณแบบถ่วงน้ำหนัก"""
        # น้ำหนักตาม Timeframe
        tf_weights = {"weekly": 1.5, "daily": 1.2, "h4": 1.0}
        weight = tf_weights.get(timeframe, 1.0)

        # ปรับตาม Market Regime
        regime = market_regime.get("regime", "RANGING")
        if regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
            weight *= 1.3  # Trend-following signals มีน้ำหนักมากขึ้น
        elif regime == "HIGH_VOLATILITY":
            weight *= 0.7  # ลดน้ำหนักในช่วง Volatile
        elif regime == "CONSOLIDATION":
            weight *= 0.8  # ลดน้ำหนักในช่วง Sideways

        # ปรับตาม Historical Performance
        win_rate = historical_perf.get("win_rate", 50)
        if win_rate >= 60:
            weight *= 1.2
        elif win_rate < 40:
            weight *= 0.8

        return base_score * weight

    def get_weekly_signal(self):
        """วิเคราะห์สัญญาณ Weekly แบบปรับปรุง พร้อม Weighted Scoring System"""

        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]
        h4 = self.data["h4"].iloc[-1]

        weekly_prev = self.data["weekly"].iloc[-2]
        daily_prev = self.data["daily"].iloc[-2]

        # วิเคราะห์ Market Regime และ Historical Performance
        market_regime = self.detect_market_regime(self.data["daily"])
        historical_perf = self.analyze_historical_performance(self.data["daily"])
        trend_consistency = self.check_trend_consistency()

        # === Advanced Analysis ===
        dynamic_thresholds = self.get_dynamic_thresholds(self.data["daily"])
        multi_indicator = self.get_multi_indicator_confirmation(self.data["daily"])
        volume_confirm = self.get_volume_confirmation(self.data["daily"])
        candlestick_signals = self.get_candlestick_signals(self.data["daily"])
        current_price = daily["close"]
        confluence_zones = self.find_confluence_zones(self.data["daily"], current_price)

        # === WEIGHT CONFIGURATION ===
        # Higher timeframe = Higher weight (Weekly > Daily > 4H)
        # Stronger signals = Higher weight
        WEIGHTS = {
            # Timeframe weights
            "weekly_trend": 5,        # Weekly EMA trend (most important)
            "weekly_cross": 8,        # Golden/Death cross (very strong)
            "weekly_rsi_extreme": 4,  # RSI < 30 or > 70
            "weekly_rsi_moderate": 2, # RSI 30-40 or 60-70
            "weekly_macd": 3,         # MACD signal
            "weekly_macd_momentum": 2,# MACD histogram increasing
            "weekly_stochrsi": 3,     # StochRSI extreme

            "daily_trend": 3,         # Daily EMA trend
            "daily_rsi_extreme": 4,   # RSI extreme with dynamic threshold
            "daily_macd_cross": 3,    # MACD crossover
            "daily_divergence_strong": 5,  # Strong divergence
            "daily_divergence_moderate": 3,# Moderate divergence
            "daily_divergence_weak": 2,    # Weak divergence

            "h4_trend": 1,            # 4H alignment
            "h4_supertrend": 2,       # Supertrend

            # Confirmation weights
            "trend_consistency": 4,   # Multi-timeframe alignment
            "multi_indicator_strong": 5,  # 4+ indicators confirm
            "multi_indicator_moderate": 3,# 3 indicators confirm
            "volume_confirmed": 4,    # Volume + OBV confirmation
            "candlestick_strong": 4,  # Strong candlestick pattern
            "candlestick_moderate": 2,# Moderate candlestick pattern
            "confluence_zone": 3,     # Near support/resistance confluence

            # New indicators weights
            "squeeze_breakout": 4,    # Squeeze release (potential big move)
            "tsi_signal": 3,          # TSI crossover
            "cmf_strong": 3,          # CMF > 0.2 or < -0.2
            "cmf_moderate": 2,        # CMF confirmation
            "vwap_position": 2,       # Price vs VWAP

            # Trend strength
            "adx_strong": 3,          # ADX > 25
            "adx_weak": -2,           # ADX < 20 (reduces confidence)

            # Risk factors (negative weights)
            "counter_trend": -2,      # Trading against higher timeframe
            "low_volume": -2,         # Below average volume
            "mixed_signals": -1,      # Conflicting indicators

            # Oversold/Overbought in Strong Trend (penalty for counter-trend signals)
            "oversold_in_downtrend_penalty": 0.3,  # Reduce oversold weight by 70%
            "overbought_in_uptrend_penalty": 0.3,  # Reduce overbought weight by 70%
        }

        signals = {"long": 0, "short": 0, "neutral": 0}
        reasons = {"long": [], "short": [], "neutral": []}

        # === DYNAMIC WEIGHT ADJUSTMENT BASED ON ADX ===
        # เมื่อ ADX สูง (trend แรง): เพิ่ม weight ให้ trend signals, ลด weight ให้ oscillators
        # เมื่อ ADX ต่ำ (sideways): ลด weight ให้ trend signals, เพิ่ม weight ให้ oscillators
        adx_value = daily["ADX"] if pd.notna(daily.get("ADX")) else 20

        if adx_value >= 40:
            # Very strong trend - heavily favor trend-following
            trend_multiplier = 1.5      # Boost trend signals by 50%
            oscillator_multiplier = 0.5  # Reduce oscillator signals by 50%
            weight_note = "Very Strong Trend (ADX≥40): Trend signals boosted, oscillators reduced"
        elif adx_value >= 25:
            # Strong trend - moderately favor trend-following
            trend_multiplier = 1.2      # Boost trend signals by 20%
            oscillator_multiplier = 0.8  # Reduce oscillator signals by 20%
            weight_note = "Strong Trend (ADX≥25): Trend signals slightly boosted"
        elif adx_value < 20:
            # Weak trend/sideways - favor mean reversion (oscillators)
            trend_multiplier = 0.8      # Reduce trend signals by 20%
            oscillator_multiplier = 1.2  # Boost oscillator signals by 20%
            weight_note = "Weak Trend (ADX<20): Oscillator signals boosted for mean reversion"
        else:
            # Normal - no adjustment
            trend_multiplier = 1.0
            oscillator_multiplier = 1.0
            weight_note = "Normal Trend (20≤ADX<25): Balanced weights"

        # === DETECT STRONG TREND FOR OVERSOLD/OVERBOUGHT PENALTY ===
        # ใน Strong Downtrend: Oversold อาจเป็น "falling knife" ไม่ใช่โอกาสซื้อ
        # ใน Strong Uptrend: Overbought อาจเป็น "momentum" ไม่ใช่โอกาสขาย
        is_strong_downtrend = (
            market_regime["regime"] == "STRONG_DOWNTREND" or
            (trend_consistency["consistent"] and
             trend_consistency["direction"] == "bearish" and
             trend_consistency["score"] >= 80)
        )
        is_strong_uptrend = (
            market_regime["regime"] == "STRONG_UPTREND" or
            (trend_consistency["consistent"] and
             trend_consistency["direction"] == "bullish" and
             trend_consistency["score"] >= 80)
        )

        # Multiplier สำหรับ oversold/overbought signals
        oversold_multiplier = WEIGHTS["oversold_in_downtrend_penalty"] if is_strong_downtrend else 1.0
        overbought_multiplier = WEIGHTS["overbought_in_uptrend_penalty"] if is_strong_uptrend else 1.0

        if is_strong_downtrend:
            reasons["neutral"].append("⚠️ Strong Downtrend: Oversold signals discounted (falling knife risk)")
        if is_strong_uptrend:
            reasons["neutral"].append("⚠️ Strong Uptrend: Overbought signals discounted (momentum may continue)")

        # เพิ่มข้อมูล Weight Adjustment
        reasons["neutral"].append(f"⚖️ {weight_note}")

        # เพิ่มข้อมูล Market Context
        regime_text = market_regime["regime"].replace("_", " ")
        reasons["neutral"].append(f"📈 Market Regime: {regime_text} ({market_regime['confidence']:.0f}%)")

        if historical_perf["total_signals"] > 0:
            pf = historical_perf.get("profit_factor", 0)
            pf_str = f", PF {pf:.2f}" if pf > 0 else ""
            reasons["neutral"].append(
                f"📊 Backtest ({historical_perf['total_signals']} trades): "
                f"Win {historical_perf['win_rate']:.0f}%, "
                f"Avg {historical_perf['avg_return']:.1f}%{pf_str}"
            )

        # === TREND CONSISTENCY (Multi-timeframe alignment) - TREND SIGNAL ===
        if trend_consistency["consistent"]:
            direction = trend_consistency["direction"]
            base_weight = WEIGHTS["trend_consistency"]
            adjusted_weight = int(base_weight * trend_multiplier)
            if direction == "bullish":
                signals["long"] += adjusted_weight
                reasons["long"].append(f"✅ Trend Consistency: Strong Bullish ({trend_consistency['score']:.0f}%)")
            elif direction == "bearish":
                signals["short"] += adjusted_weight
                reasons["short"].append(f"✅ Trend Consistency: Strong Bearish ({trend_consistency['score']:.0f}%)")
        else:
            signals["neutral"] += 1
            reasons["neutral"].append(f"⚠️ Mixed Trend ({trend_consistency['score']:.0f}%)")

        # === MULTI-INDICATOR CONFIRMATION ===
        if multi_indicator["confirmations"] >= 4:
            weight = WEIGHTS["multi_indicator_strong"]
            if multi_indicator["direction"] == "bullish":
                signals["long"] += weight
                reasons["long"].append(f"🎯 Multi-Indicator Confirmed Bullish ({multi_indicator['confirmations']}/6)")
            elif multi_indicator["direction"] == "bearish":
                signals["short"] += weight
                reasons["short"].append(f"🎯 Multi-Indicator Confirmed Bearish ({multi_indicator['confirmations']}/6)")
        elif multi_indicator["confirmations"] >= 3:
            weight = WEIGHTS["multi_indicator_moderate"]
            if multi_indicator["direction"] == "bullish":
                signals["long"] += weight
                reasons["long"].append(f"📊 Multi-Indicator Bullish ({multi_indicator['confirmations']}/6)")
            elif multi_indicator["direction"] == "bearish":
                signals["short"] += weight
                reasons["short"].append(f"📊 Multi-Indicator Bearish ({multi_indicator['confirmations']}/6)")

        # === VOLUME CONFIRMATION (Enhanced with CMF) ===
        if volume_confirm["confirmed"]:
            if volume_confirm["obv_trend"] == "bullish":
                signals["long"] += WEIGHTS["volume_confirmed"]
                reasons["long"].append(f"📈 Volume Confirmed Bullish ({volume_confirm['volume_ratio']:.1f}x)")
            elif volume_confirm["obv_trend"] == "bearish":
                signals["short"] += WEIGHTS["volume_confirmed"]
                reasons["short"].append(f"📉 Volume Confirmed Bearish ({volume_confirm['volume_ratio']:.1f}x)")
        elif volume_confirm["volume_ratio"] < 0.5:
            # Low volume warning
            signals["neutral"] += abs(WEIGHTS["low_volume"])
            reasons["neutral"].append(f"⚠️ Low Volume Warning ({volume_confirm['volume_ratio']:.1f}x)")

        # === CMF (Chaikin Money Flow) - New indicator ===
        if pd.notna(daily.get("CMF")):
            cmf = daily["CMF"]
            if cmf > 0.2:
                signals["long"] += WEIGHTS["cmf_strong"]
                reasons["long"].append(f"💰 CMF Strong Bullish: {cmf:.2f}")
            elif cmf > 0.05:
                signals["long"] += WEIGHTS["cmf_moderate"]
                reasons["long"].append(f"💰 CMF Bullish: {cmf:.2f}")
            elif cmf < -0.2:
                signals["short"] += WEIGHTS["cmf_strong"]
                reasons["short"].append(f"💰 CMF Strong Bearish: {cmf:.2f}")
            elif cmf < -0.05:
                signals["short"] += WEIGHTS["cmf_moderate"]
                reasons["short"].append(f"💰 CMF Bearish: {cmf:.2f}")

        # === SQUEEZE DETECTION (Keltner inside BB) ===
        if pd.notna(daily.get("SQUEEZE")) and pd.notna(daily.get("SQUEEZE_OFF")):
            # Squeeze just released = potential big move
            squeeze_prev = self.data["daily"].iloc[-2].get("SQUEEZE", False)
            if squeeze_prev and daily["SQUEEZE_OFF"]:
                # Squeeze just released - determine direction from momentum
                if pd.notna(daily.get("MACD_hist_change")) and daily["MACD_hist_change"] > 0:
                    signals["long"] += WEIGHTS["squeeze_breakout"]
                    reasons["long"].append("🔥 Squeeze Breakout - Bullish Momentum")
                elif pd.notna(daily.get("MACD_hist_change")) and daily["MACD_hist_change"] < 0:
                    signals["short"] += WEIGHTS["squeeze_breakout"]
                    reasons["short"].append("🔥 Squeeze Breakout - Bearish Momentum")
            elif daily["SQUEEZE"]:
                reasons["neutral"].append("⏳ In Squeeze - Wait for Breakout")

        # === TSI (True Strength Index) ===
        if pd.notna(daily.get("TSI")) and pd.notna(daily.get("TSI_signal")):
            tsi = daily["TSI"]
            tsi_signal = daily["TSI_signal"]
            tsi_prev = self.data["daily"].iloc[-2].get("TSI", 0)
            tsi_signal_prev = self.data["daily"].iloc[-2].get("TSI_signal", 0)

            # TSI crossover
            if tsi_prev <= tsi_signal_prev and tsi > tsi_signal:
                signals["long"] += WEIGHTS["tsi_signal"]
                reasons["long"].append(f"📈 TSI Bullish Cross: {tsi:.1f}")
            elif tsi_prev >= tsi_signal_prev and tsi < tsi_signal:
                signals["short"] += WEIGHTS["tsi_signal"]
                reasons["short"].append(f"📉 TSI Bearish Cross: {tsi:.1f}")
            # TSI extreme levels - OSCILLATOR SIGNAL (apply both multipliers)
            elif tsi < -25:
                weight = int(2 * oscillator_multiplier * oversold_multiplier)
                signals["long"] += weight
                discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["long"].append(f"💪 TSI Oversold: {tsi:.1f}{discount_note}")
            elif tsi > 25:
                weight = int(2 * oscillator_multiplier * overbought_multiplier)
                signals["short"] += weight
                discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["short"].append(f"⚠️ TSI Overbought: {tsi:.1f}{discount_note}")

        # === VWAP Position ===
        if pd.notna(daily.get("VWAP")):
            vwap = daily["VWAP"]
            if current_price > vwap * 1.02:  # 2% above VWAP
                signals["long"] += WEIGHTS["vwap_position"]
                reasons["long"].append(f"📈 Price Above VWAP: ${vwap:,.0f}")
            elif current_price < vwap * 0.98:  # 2% below VWAP
                signals["short"] += WEIGHTS["vwap_position"]
                reasons["short"].append(f"📉 Price Below VWAP: ${vwap:,.0f}")

        # === CANDLESTICK PATTERNS ===
        total_patterns = len(candlestick_signals["bullish"]) + len(candlestick_signals["bearish"])
        if total_patterns > 0:
            cs_score = candlestick_signals["score"]
            if cs_score >= 3:
                signals["long"] += WEIGHTS["candlestick_strong"]
                patterns_str = ", ".join(candlestick_signals["bullish"][:2])
                reasons["long"].append(f"🕯️ Strong Bullish Patterns: {patterns_str}")
            elif cs_score >= 1:
                signals["long"] += WEIGHTS["candlestick_moderate"]
                patterns_str = ", ".join(candlestick_signals["bullish"][:1])
                reasons["long"].append(f"🕯️ Bullish Pattern: {patterns_str}")
            elif cs_score <= -3:
                signals["short"] += WEIGHTS["candlestick_strong"]
                patterns_str = ", ".join(candlestick_signals["bearish"][:2])
                reasons["short"].append(f"🕯️ Strong Bearish Patterns: {patterns_str}")
            elif cs_score <= -1:
                signals["short"] += WEIGHTS["candlestick_moderate"]
                patterns_str = ", ".join(candlestick_signals["bearish"][:1])
                reasons["short"].append(f"🕯️ Bearish Pattern: {patterns_str}")

        # === CONFLUENCE ZONES ===
        if confluence_zones["support"]:
            nearest_support = confluence_zones["support"][0]
            support_distance_pct = (current_price - nearest_support["price"]) / current_price * 100
            if support_distance_pct < 2:
                signals["long"] += WEIGHTS["confluence_zone"]
                reasons["long"].append(f"🎯 Near Confluence Support (Strength: {nearest_support['strength']})")

        if confluence_zones["resistance"]:
            nearest_resistance = confluence_zones["resistance"][0]
            resist_distance_pct = (nearest_resistance["price"] - current_price) / current_price * 100
            if resist_distance_pct < 2:
                signals["short"] += WEIGHTS["confluence_zone"]
                reasons["short"].append(f"🎯 Near Confluence Resistance (Strength: {nearest_resistance['strength']})")

        # === WEEKLY TIMEFRAME ANALYSIS (Highest Weight) - TREND SIGNALS ===
        if weekly["EMA_9"] > weekly["EMA_21"]:
            weight = int(WEIGHTS["weekly_trend"] * trend_multiplier)
            signals["long"] += weight
            reasons["long"].append("📈 Weekly Uptrend: EMA 9 > 21")
        elif weekly["EMA_9"] < weekly["EMA_21"]:
            weight = int(WEIGHTS["weekly_trend"] * trend_multiplier)
            signals["short"] += weight
            reasons["short"].append("📉 Weekly Downtrend: EMA 9 < 21")

        # Golden/Death Cross (strongest signal) - TREND SIGNAL
        if weekly_prev["EMA_9"] <= weekly_prev["EMA_21"] and weekly["EMA_9"] > weekly["EMA_21"]:
            weight = int(WEIGHTS["weekly_cross"] * trend_multiplier)
            signals["long"] += weight
            reasons["long"].append("🔥 Weekly Golden Cross!")
        elif weekly_prev["EMA_9"] >= weekly_prev["EMA_21"] and weekly["EMA_9"] < weekly["EMA_21"]:
            weight = int(WEIGHTS["weekly_cross"] * trend_multiplier)
            signals["short"] += weight
            reasons["short"].append("🔥 Weekly Death Cross!")

        # Weekly RSI - OSCILLATOR SIGNAL with CONFIRMATION
        # Oversold/Overbought ต้องมี reversal confirmation (RSI เริ่มกลับตัว) จึงจะให้คะแนนเต็ม
        weekly_rsi_rising = weekly["RSI"] > weekly_prev["RSI"]
        weekly_rsi_falling = weekly["RSI"] < weekly_prev["RSI"]

        if weekly["RSI"] < 30:
            if weekly_rsi_rising:
                # Confirmed reversal from oversold
                weight = int(WEIGHTS["weekly_rsi_extreme"] * oscillator_multiplier * oversold_multiplier)
                signals["long"] += weight
                discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["long"].append(f"💪 Weekly RSI Oversold + Reversal: {weekly['RSI']:.1f}{discount_note}")
            else:
                # No confirmation - just note it, no points (still falling)
                reasons["neutral"].append(f"⏳ Weekly RSI Oversold ({weekly['RSI']:.1f}) - Awaiting reversal")
        elif weekly["RSI"] < 40:
            if weekly_rsi_rising:
                weight = int(WEIGHTS["weekly_rsi_moderate"] * oscillator_multiplier * oversold_multiplier)
                signals["long"] += weight
                discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["long"].append(f"📊 Weekly RSI Low + Rising: {weekly['RSI']:.1f}{discount_note}")
            else:
                reasons["neutral"].append(f"⏳ Weekly RSI Low ({weekly['RSI']:.1f}) - Awaiting reversal")
        elif weekly["RSI"] > 70:
            if weekly_rsi_falling:
                # Confirmed reversal from overbought
                weight = int(WEIGHTS["weekly_rsi_extreme"] * oscillator_multiplier * overbought_multiplier)
                signals["short"] += weight
                discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["short"].append(f"⚠️ Weekly RSI Overbought + Reversal: {weekly['RSI']:.1f}{discount_note}")
            else:
                reasons["neutral"].append(f"⏳ Weekly RSI Overbought ({weekly['RSI']:.1f}) - Awaiting reversal")
        elif weekly["RSI"] > 60:
            if weekly_rsi_falling:
                weight = int(WEIGHTS["weekly_rsi_moderate"] * oscillator_multiplier * overbought_multiplier)
                signals["short"] += weight
                discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["short"].append(f"📊 Weekly RSI High + Falling: {weekly['RSI']:.1f}{discount_note}")
            else:
                reasons["neutral"].append(f"⏳ Weekly RSI High ({weekly['RSI']:.1f}) - Awaiting reversal")
        elif 45 < weekly["RSI"] < 55:
            signals["neutral"] += 1
            reasons["neutral"].append(f"😐 Weekly RSI Neutral: {weekly['RSI']:.1f}")

        # Weekly MACD
        if weekly["MACD"] > weekly["MACD_signal"] and weekly["MACD_histogram"] > 0:
            signals["long"] += WEIGHTS["weekly_macd"]
            reasons["long"].append("📊 Weekly MACD Bullish")
            if weekly["MACD_histogram"] > weekly_prev["MACD_histogram"]:
                signals["long"] += WEIGHTS["weekly_macd_momentum"]
                reasons["long"].append("📈 Weekly MACD Momentum Increasing")
        elif weekly["MACD"] < weekly["MACD_signal"] and weekly["MACD_histogram"] < 0:
            signals["short"] += WEIGHTS["weekly_macd"]
            reasons["short"].append("📊 Weekly MACD Bearish")
            if weekly["MACD_histogram"] < weekly_prev["MACD_histogram"]:
                signals["short"] += WEIGHTS["weekly_macd_momentum"]
                reasons["short"].append("📉 Weekly MACD Momentum Decreasing")

        # Weekly StochRSI - OSCILLATOR SIGNAL with CROSSOVER CONFIRMATION
        # Oversold: K < 20 AND K crosses above D (bullish crossover)
        # Overbought: K > 80 AND K crosses below D (bearish crossover)
        if pd.notna(weekly.get("STOCHRSI_K")) and pd.notna(weekly_prev.get("STOCHRSI_K")):
            stoch_k = weekly["STOCHRSI_K"]
            stoch_d = weekly["STOCHRSI_D"]
            stoch_k_prev = weekly_prev["STOCHRSI_K"]
            stoch_d_prev = weekly_prev["STOCHRSI_D"]

            # Bullish crossover: K crosses above D
            bullish_cross = stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d
            # Bearish crossover: K crosses below D
            bearish_cross = stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d
            # K rising
            k_rising = stoch_k > stoch_k_prev

            if stoch_k < 20 and stoch_d < 20:
                if bullish_cross or k_rising:
                    # Confirmed: oversold with bullish crossover or K rising
                    weight = int(WEIGHTS["weekly_stochrsi"] * oscillator_multiplier * oversold_multiplier)
                    signals["long"] += weight
                    discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                    confirm_type = "Cross" if bullish_cross else "Rising"
                    reasons["long"].append(f"💪 Weekly StochRSI Oversold + {confirm_type}: {stoch_k:.1f}{discount_note}")
                else:
                    reasons["neutral"].append(f"⏳ Weekly StochRSI Oversold ({stoch_k:.1f}) - Awaiting crossover")
            elif stoch_k > 80 and stoch_d > 80:
                if bearish_cross or stoch_k < stoch_k_prev:
                    # Confirmed: overbought with bearish crossover or K falling
                    weight = int(WEIGHTS["weekly_stochrsi"] * oscillator_multiplier * overbought_multiplier)
                    signals["short"] += weight
                    discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                    confirm_type = "Cross" if bearish_cross else "Falling"
                    reasons["short"].append(f"⚠️ Weekly StochRSI Overbought + {confirm_type}: {stoch_k:.1f}{discount_note}")
                else:
                    reasons["neutral"].append(f"⏳ Weekly StochRSI Overbought ({stoch_k:.1f}) - Awaiting crossover")

        # === DAILY TIMEFRAME CONFIRMATION - TREND SIGNAL ===
        if daily["EMA_9"] > daily["EMA_21"]:
            weight = int(WEIGHTS["daily_trend"] * trend_multiplier)
            signals["long"] += weight
            reasons["long"].append("📈 Daily Uptrend")
            # Check for counter-trend warning
            if weekly["EMA_9"] < weekly["EMA_21"]:
                signals["neutral"] += abs(WEIGHTS["counter_trend"])
                reasons["neutral"].append("⚠️ Daily vs Weekly conflict")
        elif daily["EMA_9"] < daily["EMA_21"]:
            weight = int(WEIGHTS["daily_trend"] * trend_multiplier)
            signals["short"] += weight
            reasons["short"].append("📉 Daily Downtrend")
            if weekly["EMA_9"] > weekly["EMA_21"]:
                signals["neutral"] += abs(WEIGHTS["counter_trend"])
                reasons["neutral"].append("⚠️ Daily vs Weekly conflict")

        # Daily RSI - OSCILLATOR SIGNAL with CONFIRMATION
        rsi_oversold = dynamic_thresholds["rsi_oversold"]
        rsi_overbought = dynamic_thresholds["rsi_overbought"]
        daily_rsi_rising = daily["RSI"] > daily_prev["RSI"]
        daily_rsi_falling = daily["RSI"] < daily_prev["RSI"]

        if daily["RSI"] < rsi_oversold:
            if daily_rsi_rising:
                # Confirmed reversal from oversold
                weight = int(WEIGHTS["daily_rsi_extreme"] * oscillator_multiplier * oversold_multiplier)
                signals["long"] += weight
                discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["long"].append(f"💪 Daily RSI Oversold + Reversal: {daily['RSI']:.1f} (< {rsi_oversold:.0f}){discount_note}")
            else:
                reasons["neutral"].append(f"⏳ Daily RSI Oversold ({daily['RSI']:.1f}) - Awaiting reversal")
        elif daily["RSI"] > rsi_overbought:
            if daily_rsi_falling:
                # Confirmed reversal from overbought
                weight = int(WEIGHTS["daily_rsi_extreme"] * oscillator_multiplier * overbought_multiplier)
                signals["short"] += weight
                discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["short"].append(f"⚠️ Daily RSI Overbought + Reversal: {daily['RSI']:.1f} (> {rsi_overbought:.0f}){discount_note}")
            else:
                reasons["neutral"].append(f"⏳ Daily RSI Overbought ({daily['RSI']:.1f}) - Awaiting reversal")

        # Divergence Detection (weighted by strength)
        daily_divergence, div_strength = self.check_divergence(self.data["daily"], "RSI")
        if daily_divergence == "bullish" and div_strength > 0:
            if div_strength >= 60:
                signals["long"] += WEIGHTS["daily_divergence_strong"]
                reasons["long"].append(f"🔄 Strong Bullish Divergence ({div_strength:.0f})")
            elif div_strength >= 30:
                signals["long"] += WEIGHTS["daily_divergence_moderate"]
                reasons["long"].append(f"🔄 Bullish Divergence ({div_strength:.0f})")
            else:
                signals["long"] += WEIGHTS["daily_divergence_weak"]
                reasons["long"].append(f"🔄 Weak Bullish Divergence ({div_strength:.0f})")
        elif daily_divergence == "bearish" and div_strength > 0:
            if div_strength >= 60:
                signals["short"] += WEIGHTS["daily_divergence_strong"]
                reasons["short"].append(f"🔄 Strong Bearish Divergence ({div_strength:.0f})")
            elif div_strength >= 30:
                signals["short"] += WEIGHTS["daily_divergence_moderate"]
                reasons["short"].append(f"🔄 Bearish Divergence ({div_strength:.0f})")
            else:
                signals["short"] += WEIGHTS["daily_divergence_weak"]
                reasons["short"].append(f"🔄 Weak Bearish Divergence ({div_strength:.0f})")

        # MACD Divergence
        macd_divergence, macd_div_strength = self.check_divergence(self.data["daily"], "MACD", lookback=20)
        if macd_divergence == "bullish" and macd_div_strength > 20:
            signals["long"] += WEIGHTS["daily_divergence_weak"]
            reasons["long"].append("🔄 MACD Bullish Divergence")
        elif macd_divergence == "bearish" and macd_div_strength > 20:
            signals["short"] += WEIGHTS["daily_divergence_weak"]
            reasons["short"].append("🔄 MACD Bearish Divergence")

        # Daily MACD Cross
        if daily_prev["MACD"] <= daily_prev["MACD_signal"] and daily["MACD"] > daily["MACD_signal"]:
            signals["long"] += WEIGHTS["daily_macd_cross"]
            reasons["long"].append("✅ Daily MACD Cross Up")
        elif daily_prev["MACD"] >= daily_prev["MACD_signal"] and daily["MACD"] < daily["MACD_signal"]:
            signals["short"] += WEIGHTS["daily_macd_cross"]
            reasons["short"].append("❌ Daily MACD Cross Down")

        # MFI - OSCILLATOR SIGNAL (apply both oscillator_multiplier and oversold/overbought multiplier)
        if pd.notna(daily.get("MFI")):
            if daily["MFI"] < 20:
                weight = int(2 * oscillator_multiplier * oversold_multiplier)
                signals["long"] += weight
                discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["long"].append(f"💰 Daily MFI Oversold: {daily['MFI']:.1f}{discount_note}")
            elif daily["MFI"] > 80:
                weight = int(2 * oscillator_multiplier * overbought_multiplier)
                signals["short"] += weight
                discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["short"].append(f"💰 Daily MFI Overbought: {daily['MFI']:.1f}{discount_note}")

        # CCI - OSCILLATOR SIGNAL (apply both oscillator_multiplier and oversold/overbought multiplier)
        if pd.notna(daily.get("CCI")):
            if daily["CCI"] < -100:
                weight = int(1 * oscillator_multiplier * oversold_multiplier)
                signals["long"] += weight
                discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["long"].append(f"📊 Daily CCI Oversold: {daily['CCI']:.1f}{discount_note}")
            elif daily["CCI"] > 100:
                weight = int(1 * oscillator_multiplier * overbought_multiplier)
                signals["short"] += weight
                discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
                reasons["short"].append(f"📊 Daily CCI Overbought: {daily['CCI']:.1f}{discount_note}")

        # === 4H TIMEFRAME (Lowest Weight) - TREND SIGNALS ===
        if h4["EMA_9"] > h4["EMA_21"]:
            weight = int(WEIGHTS["h4_trend"] * trend_multiplier)
            signals["long"] += weight
            reasons["long"].append("📊 4H Aligned Bullish")
        elif h4["EMA_9"] < h4["EMA_21"]:
            weight = int(WEIGHTS["h4_trend"] * trend_multiplier)
            signals["short"] += weight
            reasons["short"].append("📊 4H Aligned Bearish")

        if pd.notna(h4.get("SUPERTREND_DIR")):
            if h4["SUPERTREND_DIR"] == 1:
                weight = int(WEIGHTS["h4_supertrend"] * trend_multiplier)
                signals["long"] += weight
                reasons["long"].append("🚀 4H Supertrend Bullish")
            else:
                weight = int(WEIGHTS["h4_supertrend"] * trend_multiplier)
                signals["short"] += weight
                reasons["short"].append("🔻 4H Supertrend Bearish")

        # === ADX TREND STRENGTH ===
        if daily["ADX"] > 25:
            if daily["DI_plus"] > daily["DI_minus"]:
                signals["long"] += WEIGHTS["adx_strong"]
                reasons["long"].append(f"💪 Strong Uptrend (ADX: {daily['ADX']:.1f})")
            else:
                signals["short"] += WEIGHTS["adx_strong"]
                reasons["short"].append(f"💪 Strong Downtrend (ADX: {daily['ADX']:.1f})")
        elif daily["ADX"] < 20:
            # ADX weak reduces confidence
            signals["neutral"] += abs(WEIGHTS["adx_weak"])
            reasons["neutral"].append(f"🌊 Weak Trend - Low Confidence (ADX: {daily['ADX']:.1f})")

        # === VOLUME ANALYSIS ===
        if daily["Volume_Ratio"] > 1.5:
            if daily["IS_BULLISH"]:
                signals["long"] += 2
                reasons["long"].append(f"📊 High Volume Bullish: {daily['Volume_Ratio']:.1f}x")
            else:
                signals["short"] += 2
                reasons["short"].append(f"📊 High Volume Bearish: {daily['Volume_Ratio']:.1f}x")

        if pd.notna(daily.get("OBV")) and pd.notna(daily.get("OBV_EMA")):
            if daily["OBV"] > daily["OBV_EMA"]:
                signals["long"] += 1
                reasons["long"].append("📈 OBV Accumulation")
            else:
                signals["short"] += 1
                reasons["short"].append("📉 OBV Distribution")

        # === BOLLINGER BANDS - OSCILLATOR SIGNAL (apply both multipliers) ===
        if daily["close"] < daily["BB_lower"]:
            weight = int(2 * oscillator_multiplier * oversold_multiplier)
            signals["long"] += weight
            discount_note = " [discounted]" if oversold_multiplier < 1 or oscillator_multiplier < 1 else ""
            reasons["long"].append(f"📉 Price below BB Lower (Oversold){discount_note}")
        elif daily["close"] > daily["BB_upper"]:
            weight = int(2 * oscillator_multiplier * overbought_multiplier)
            signals["short"] += weight
            discount_note = " [discounted]" if overbought_multiplier < 1 or oscillator_multiplier < 1 else ""
            reasons["short"].append(f"📈 Price above BB Upper (Overbought){discount_note}")

        # === ICHIMOKU (weight 2 for cloud, 1 for TK cross) ===
        if pd.notna(daily.get("ICHI_TENKAN")) and pd.notna(daily.get("ICHI_KIJUN")):
            if daily["close"] > daily["ICHI_SENKOU_A"] and daily["close"] > daily["ICHI_SENKOU_B"]:
                signals["long"] += 2
                reasons["long"].append("☁️ Price Above Ichimoku Cloud")
            elif daily["close"] < daily["ICHI_SENKOU_A"] and daily["close"] < daily["ICHI_SENKOU_B"]:
                signals["short"] += 2
                reasons["short"].append("☁️ Price Below Ichimoku Cloud")

            if daily["ICHI_TENKAN"] > daily["ICHI_KIJUN"]:
                signals["long"] += 1
                reasons["long"].append("📊 Ichimoku TK Cross Bullish")
            elif daily["ICHI_TENKAN"] < daily["ICHI_KIJUN"]:
                signals["short"] += 1
                reasons["short"].append("📊 Ichimoku TK Cross Bearish")

        # === NEW: Add Advanced Analysis Summary ===
        # Calculate overall signal direction for risk score
        signal_type = "LONG" if signals["long"] > signals["short"] else "SHORT" if signals["short"] > signals["long"] else "NEUTRAL"
        risk_score = self.calculate_risk_score(self.data["daily"], signal_type)

        # Add analysis summary to reasons
        reasons["neutral"].append(f"📊 Dynamic RSI Thresholds: Oversold < {dynamic_thresholds['rsi_oversold']:.0f}, Overbought > {dynamic_thresholds['rsi_overbought']:.0f}")
        reasons["neutral"].append(f"⚠️ Trade Risk Score: {risk_score['score']:.0f}/100 ({risk_score['level']})")

        # Store additional analysis data for position management
        self._last_analysis = {
            "multi_indicator": multi_indicator,
            "volume_confirm": volume_confirm,
            "candlestick_signals": candlestick_signals,
            "confluence_zones": confluence_zones,
            "dynamic_thresholds": dynamic_thresholds,
            "risk_score": risk_score
        }

        return signals, reasons

    def calculate_position_management(self, current_price, signal_type):
        """คำนวณการจัดการ Position สำหรับ Weekly แบบปรับปรุง"""

        daily_df = self.data["daily"]
        daily = daily_df.iloc[-1]

        atr_daily = daily["ATR"]
        sr = self.calculate_support_resistance(daily_df)
        fib_levels, fib_trend = self.calculate_fibonacci_levels(daily_df)

        # Market Regime สำหรับปรับ Strategy
        market_regime = self.detect_market_regime(daily_df)

        # Volatility-adjusted Risk
        vol_risk = self.calculate_volatility_adjusted_risk(daily_df)

        atr_percent = daily["ATR_percent"]

        # ปรับ Multipliers ตาม Market Regime
        regime = market_regime["regime"]
        if regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
            # Trending market - wider stops, bigger targets
            if atr_percent > 5:
                sl_multiplier = 2.5
                tp_multiplier = [3, 5, 8]
            elif atr_percent > 3:
                sl_multiplier = 2.0
                tp_multiplier = [2.5, 4, 6]
            else:
                sl_multiplier = 1.5
                tp_multiplier = [2, 3, 5]
        elif regime == "HIGH_VOLATILITY":
            # High volatility - wider stops
            if atr_percent > 5:
                sl_multiplier = 3.0
                tp_multiplier = [2, 3, 4]
            else:
                sl_multiplier = 2.5
                tp_multiplier = [1.5, 2.5, 3.5]
        elif regime == "CONSOLIDATION":
            # Consolidation - tighter stops
            sl_multiplier = 1.0
            tp_multiplier = [1.2, 2, 2.5]
        else:
            # Default
            if atr_percent > 5:
                sl_multiplier = 2.0
                tp_multiplier = [2.5, 4, 6]
            elif atr_percent > 3:
                sl_multiplier = 1.5
                tp_multiplier = [2, 3, 4]
            else:
                sl_multiplier = 1.2
                tp_multiplier = [1.5, 2.5, 3.5]

        # Extract Fibonacci extension levels for targets
        fib_ext_127 = fib_levels.get("127.2%")
        fib_ext_161 = fib_levels.get("161.8%")
        fib_ext_200 = fib_levels.get("200.0%")
        fib_ext_261 = fib_levels.get("261.8%")

        if signal_type == "LONG":
            stop_loss_support = sr["main_support"]
            stop_loss_atr = current_price - (atr_daily * sl_multiplier)
            stop_loss = max(stop_loss_support, stop_loss_atr)

            # TP1: ATR-based or Fib 127.2% (whichever is closer but profitable)
            tp1_atr = current_price + (atr_daily * tp_multiplier[0])
            tp1 = tp1_atr
            if fib_ext_127 and fib_ext_127 > current_price:
                tp1 = min(tp1_atr, fib_ext_127)  # Use closer target

            # TP2: ATR-based or Fib 161.8%
            tp2_atr = current_price + (atr_daily * tp_multiplier[1])
            tp2 = tp2_atr
            if fib_ext_161 and fib_ext_161 > current_price:
                tp2 = max(tp2_atr, fib_ext_161)  # Use Fib if higher

            # TP3: Fib 200% or 261.8% extension (aggressive target)
            tp3 = sr["main_resistance"]
            if fib_ext_200 and fib_ext_200 > current_price:
                tp3 = max(tp3, fib_ext_200)
            if fib_ext_261 and fib_ext_261 > current_price and regime in ["STRONG_UPTREND"]:
                tp3 = fib_ext_261  # Very aggressive in strong uptrend

        else:  # SHORT
            stop_loss_resistance = sr["main_resistance"]
            stop_loss_atr = current_price + (atr_daily * sl_multiplier)
            stop_loss = min(stop_loss_resistance, stop_loss_atr)

            # TP1: ATR-based or Fib 127.2% (whichever is closer but profitable)
            tp1_atr = current_price - (atr_daily * tp_multiplier[0])
            tp1 = tp1_atr
            if fib_ext_127 and fib_ext_127 < current_price:
                tp1 = max(tp1_atr, fib_ext_127)  # Use closer target

            # TP2: ATR-based or Fib 161.8%
            tp2_atr = current_price - (atr_daily * tp_multiplier[1])
            tp2 = tp2_atr
            if fib_ext_161 and fib_ext_161 < current_price:
                tp2 = min(tp2_atr, fib_ext_161)  # Use Fib if lower

            # TP3: Fib 200% or 261.8% extension (aggressive target)
            tp3 = sr["main_support"]
            if fib_ext_200 and fib_ext_200 < current_price:
                tp3 = min(tp3, fib_ext_200)
            if fib_ext_261 and fib_ext_261 < current_price and regime in ["STRONG_DOWNTREND"]:
                tp3 = fib_ext_261  # Very aggressive in strong downtrend

        return {
            "entry": current_price,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "atr": atr_daily,
            "atr_percent": atr_percent,
            "support_resistance": sr,
            "fibonacci": fib_levels,
            "fib_trend": fib_trend,
            "market_regime": market_regime,
            "volatility_risk": vol_risk,
        }

    def get_weekly_recommendation(self, balance: float = 10000) -> None:
        """แสดงคำแนะนำ Weekly Trading แบบปรับปรุง"""

        if not self.analyze_multi_timeframe():
            return

        signals, reasons = self.get_weekly_signal()

        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]
        h4 = self.data["h4"].iloc[-1]

        current_price = h4["close"]

        weekly_trend, _ = self.get_trend_strength(self.data["weekly"])
        daily_trend, _ = self.get_trend_strength(self.data["daily"])

        print("=" * 100)
        print(f"📅 WEEKLY TRADING STRATEGY - {self.symbol}")
        print(f"💰 Leverage: {self.leverage}x | 📅 Hold Period: ~1 สัปดาห์")
        print(f"⏰ วันที่: {h4['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        print("=" * 100)

        print(f"\n💵 ราคาปัจจุบัน: ${current_price:,.2f}")

        print("\n📈 TREND ANALYSIS:")
        trend_emoji = "🟢" if weekly_trend > 0 else "🔴" if weekly_trend < 0 else "🟡"
        print(f"  Weekly Trend Score: {trend_emoji} {weekly_trend:+d}")
        trend_emoji = "🟢" if daily_trend > 0 else "🔴" if daily_trend < 0 else "🟡"
        print(f"  Daily Trend Score: {trend_emoji} {daily_trend:+d}")

        print("\n📊 MULTI-TIMEFRAME ANALYSIS:")
        print("\n📅 Weekly Indicators:")
        print(f"  • EMA 9/21: ${weekly['EMA_9']:,.2f} / ${weekly['EMA_21']:,.2f}")
        print(f"  • RSI: {weekly['RSI']:.2f}")
        print(f"  • MACD: {weekly['MACD']:.2f} (Signal: {weekly['MACD_signal']:.2f})")
        if pd.notna(weekly.get("STOCHRSI_K")):
            print(f"  • StochRSI: {weekly['STOCHRSI_K']:.2f}")

        print("\n📈 Daily Indicators:")
        print(f"  • EMA 9/21: ${daily['EMA_9']:,.2f} / ${daily['EMA_21']:,.2f}")
        print(f"  • RSI: {daily['RSI']:.2f}")
        print(f"  • ADX: {daily['ADX']:.2f} (DI+: {daily['DI_plus']:.1f}, DI-: {daily['DI_minus']:.1f})")
        print(f"  • ATR: ${daily['ATR']:,.2f} ({daily['ATR_percent']:.2f}%)")
        if pd.notna(daily.get("MFI")):
            print(f"  • MFI: {daily['MFI']:.2f}")
        if pd.notna(daily.get("CCI")):
            print(f"  • CCI: {daily['CCI']:.2f}")

        print("\n⏰ 4H Indicators:")
        print(f"  • EMA 9/21: ${h4['EMA_9']:,.2f} / ${h4['EMA_21']:,.2f}")
        print(f"  • RSI: {h4['RSI']:.2f}")
        if pd.notna(h4.get("SUPERTREND_DIR")):
            st_dir = "Bullish 🟢" if h4["SUPERTREND_DIR"] == 1 else "Bearish 🔴"
            print(f"  • Supertrend: {st_dir}")

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
        print("🎯 WEEKLY RECOMMENDATION")
        print("=" * 100)

        recommendation, confidence = self.get_confidence_level(signals)

        if abs(long_pct - short_pct) < 15:
            print("\n⚠️ WARNING: Mixed signals detected - proceed with caution!")

        if recommendation in ["STRONG_LONG", "LONG"]:
            signal_type = "LONG"
            position_mgmt = self.calculate_position_management(current_price, signal_type)

            conf_text = "STRONG" if recommendation == "STRONG_LONG" else "MODERATE"
            print(f"\n✅ {conf_text} LONG SIGNAL ({confidence:.1f}%)")
            print("💡 แนะนำ: เปิด Long Position และ Hold 1 สัปดาห์")

            self._print_trade_setup(position_mgmt, signal_type, balance, current_price)

        elif recommendation in ["STRONG_SHORT", "SHORT"]:
            signal_type = "SHORT"
            position_mgmt = self.calculate_position_management(current_price, signal_type)

            conf_text = "STRONG" if recommendation == "STRONG_SHORT" else "MODERATE"
            print(f"\n❌ {conf_text} SHORT SIGNAL ({confidence:.1f}%)")
            print("💡 แนะนำ: เปิด Short Position และ Hold 1 สัปดาห์")

            self._print_trade_setup(position_mgmt, signal_type, balance, current_price)

        else:
            print(f"\n⏸️ WAIT - ไม่มีสัญญาณชัดเจน ({confidence:.1f}%)")
            print("💡 แนะนำ: รอสัญญาณที่ชัดเจนกว่านี้")
            print("📌 ตรวจสอบใหม่อีกครั้งใน 1-2 วัน")

            sr = self.calculate_support_resistance(self.data["daily"])
            print("\n📊 LEVELS TO WATCH:")
            print(f"  🛡️ Support: ${sr['main_support']:,.2f}")
            print(f"  🔒 Resistance: ${sr['main_resistance']:,.2f}")

        print("\n" + "=" * 100)
        print("📅 NEXT REVIEW DATE: " + (h4["timestamp"] + timedelta(days=1)).strftime("%Y-%m-%d"))
        print("⚠️ คำเตือน: Review ทุกวัน แต่อย่า Overtrade")
        print("💰 ใช้ Leverage 5-10x สำหรับ Swing Trade")
        print("🎯 ตั้ง SL/TP แล้วปล่อยให้ระบบทำงาน")
        print("=" * 100)

    def _print_trade_setup(self, position_mgmt, signal_type, balance, current_price):
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
            print(f"  • ADX: {regime['adx']:.1f} | BB Width: {regime['bb_width']:.2f}%")
            print(f"  • Price Range (20d): {regime['price_range_pct']:.1f}%")

        # Advanced Analysis from last signal
        if hasattr(self, "_last_analysis") and self._last_analysis:
            analysis = self._last_analysis

            # Multi-Indicator Confirmation
            multi_ind = analysis.get("multi_indicator", {})
            if multi_ind:
                confirm_pct = (multi_ind.get("confirmations", 0) / 6) * 100
                direction = multi_ind.get("direction", "neutral").upper()
                print("\n🎯 MULTI-INDICATOR CONFIRMATION:")
                print(f"  • Direction: {direction} ({multi_ind.get('confirmations', 0)}/6 indicators)")
                print(f"  • Confirmation: {confirm_pct:.0f}% | Strength: {multi_ind.get('strength', 0):.0f}%")
                if multi_ind.get("details"):
                    for detail in multi_ind["details"][:3]:
                        print(f"    ✓ {detail}")

            # Risk Score
            risk_score = analysis.get("risk_score", {})
            if risk_score:
                risk_level = risk_score.get("level", "Unknown")
                risk_emoji = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
                print("\n⚠️ TRADE RISK ASSESSMENT:")
                print(f"  • Risk Score: {risk_emoji} {risk_score.get('score', 0):.0f}/100 ({risk_level})")
                factors = risk_score.get("factors", [])
                if factors:
                    print("  • Risk Factors:")
                    for factor in factors[:4]:
                        print(f"    - {factor}")

            # Confluence Zones
            confluence = analysis.get("confluence_zones", {})
            if confluence:
                supports = confluence.get("support", [])
                resistances = confluence.get("resistance", [])
                if supports or resistances:
                    print("\n🎯 CONFLUENCE ZONES:")
                    if supports:
                        for i, zone in enumerate(supports[:2], 1):
                            print(f"  • Support Zone {i}: ${zone['price']:,.0f} (Strength: {zone['strength']} levels)")
                    if resistances:
                        for i, zone in enumerate(resistances[:2], 1):
                            print(f"  • Resistance Zone {i}: ${zone['price']:,.0f} (Strength: {zone['strength']} levels)")

            # Candlestick Patterns
            candle_signals = analysis.get("candlestick_signals", {})
            total_cs_patterns = len(candle_signals.get("bullish", [])) + len(candle_signals.get("bearish", []))
            if candle_signals and total_cs_patterns > 0:
                print("\n🕯️ CANDLESTICK PATTERNS:")
                if candle_signals.get("bullish"):
                    print(f"  • Bullish: {', '.join(candle_signals['bullish'][:3])}")
                if candle_signals.get("bearish"):
                    print(f"  • Bearish: {', '.join(candle_signals['bearish'][:3])}")
                print(f"  • Net Score: {candle_signals.get('score', 0):+d}")

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
        print(f"  🎁 TP1 (40%): ${tp1:,.2f} ({tp1_pct:+.2f}% = {tp1_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP2 (30%): ${tp2:,.2f} ({tp2_pct:+.2f}% = {tp2_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP3 (30%): ${tp3:,.2f} ({tp3_pct:+.2f}% = {tp3_pct * self.leverage:+.1f}% margin)")

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
        print("\n📅 WEEKLY STRATEGY:")
        print(f"  1️⃣ เปิด {action} ที่ราคาปัจจุบัน ${entry:,.2f}")
        print(f"  2️⃣ ตั้ง Stop Loss ที่ ${sl:,.2f}")
        print("  3️⃣ ปิด 40% ที่ TP1, 30% ที่ TP2, 30% ที่ TP3")
        print("  4️⃣ ถ้าถึง TP1 → ขยับ SL ไปที่ Entry (Break Even)")
        print("  5️⃣ Review ทุกวัน แต่ไม่ต้อง Trade บ่อย")
        print("  6️⃣ Hold จนกว่าจะถึง TP หรือ SL หรือครบ 1 สัปดาห์")


