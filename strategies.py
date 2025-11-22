"""
Trading Strategies Module
Contains WeeklyTradingStrategy and MonthlyTradingStrategy classes
"""

import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import timedelta
from typing import Literal


class WeeklyTradingStrategy:
    """Strategy สำหรับ Trade รอบละ 1 สัปดาห์"""

    def __init__(self, symbol="BTCUSDT", leverage=5):
        self.symbol = symbol
        self.leverage = leverage
        self.timeframes = {"weekly": "1w", "daily": "1d", "h4": "4h"}
        self.data = {}

    def fetch_data(self, timeframe, limit=100):
        """ดึงข้อมูลจาก Binance"""
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": self.symbol, "interval": timeframe, "limit": limit}

        try:
            response = requests.get(url, params=params)
            data = response.json()

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_indicators(self, df):
        """คำนวณตัวชี้วัดแบบครบถ้วน"""

        # === MOVING AVERAGES ===
        df["EMA_9"] = ta.ema(df["close"], length=9)
        df["EMA_21"] = ta.ema(df["close"], length=21)
        df["EMA_50"] = ta.ema(df["close"], length=50)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_200"] = ta.sma(df["close"], length=200)

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
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"] * 100
        df["BB_percent"] = (df["close"] - df["BB_lower"]) / (
            df["BB_upper"] - df["BB_lower"]
        )

        # === ADX (Trend Strength) ===
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["ADX"] = adx["ADX_14"]
        df["DI_plus"] = adx["DMP_14"]
        df["DI_minus"] = adx["DMN_14"]

        # === ATR ===
        df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ATR_percent"] = df["ATR"] / df["close"] * 100

        # === Volume Analysis ===
        df["Volume_MA"] = df["volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["volume"] / df["Volume_MA"]

        # === OBV (On-Balance Volume) ===
        df["OBV"] = ta.obv(df["close"], df["volume"])
        df["OBV_EMA"] = ta.ema(df["OBV"], length=21)

        # === MFI (Money Flow Index) ===
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

        # === CCI (Commodity Channel Index) ===
        # Manual calculation because pandas_ta CCI has issues with high-priced assets
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(20).mean()
        mean_dev = abs(typical_price - sma_tp).rolling(20).mean()
        df["CCI"] = (typical_price - sma_tp) / (0.015 * mean_dev)

        # === Williams %R ===
        df["WILLR"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        # === Ichimoku Cloud ===
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if ichimoku is not None and len(ichimoku) >= 2 and ichimoku[0] is not None:
                df["ICHI_TENKAN"] = ichimoku[0]["ITS_9"]
                df["ICHI_KIJUN"] = ichimoku[0]["IKS_26"]
                df["ICHI_SENKOU_A"] = ichimoku[0]["ISA_9"]
                df["ICHI_SENKOU_B"] = ichimoku[0]["ISB_26"]
        except (TypeError, KeyError):
            # Ichimoku ต้องการข้อมูลจำนวนมาก ถ้าไม่พอก็ข้าม
            pass

        # === VWAP (Volume Weighted Average Price) ===
        # VWAP requires DatetimeIndex, so we calculate manually
        typical_price_vwap = (df["high"] + df["low"] + df["close"]) / 3
        df["VWAP"] = (typical_price_vwap * df["volume"]).cumsum() / df["volume"].cumsum()

        # === Supertrend ===
        supertrend = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        df["SUPERTREND"] = supertrend["SUPERT_10_3.0"]
        df["SUPERTREND_DIR"] = supertrend["SUPERTd_10_3.0"]

        # === Pivot Points ===
        df["PIVOT"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["R1"] = 2 * df["PIVOT"] - df["low"].shift(1)
        df["S1"] = 2 * df["PIVOT"] - df["high"].shift(1)
        df["R2"] = df["PIVOT"] + (df["high"].shift(1) - df["low"].shift(1))
        df["S2"] = df["PIVOT"] - (df["high"].shift(1) - df["low"].shift(1))

        # === Price Action Patterns ===
        df["HIGHER_HIGH"] = df["high"] > df["high"].shift(1)
        df["LOWER_LOW"] = df["low"] < df["low"].shift(1)
        df["HIGHER_LOW"] = df["low"] > df["low"].shift(1)
        df["LOWER_HIGH"] = df["high"] < df["high"].shift(1)

        # === Candle Patterns ===
        df["BODY"] = abs(df["close"] - df["open"])
        df["RANGE"] = df["high"] - df["low"]
        df["BODY_PERCENT"] = df["BODY"] / df["RANGE"] * 100
        df["IS_BULLISH"] = df["close"] > df["open"]
        df["IS_BEARISH"] = df["close"] < df["open"]

        # === Momentum ===
        df["ROC"] = ta.roc(df["close"], length=10)
        df["MOM"] = ta.mom(df["close"], length=10)

        return df

    def calculate_support_resistance(self, df, lookback=20):
        """คำนวณ Support & Resistance แบบปรับปรุง"""
        recent_data = df.tail(lookback)

        highs = []
        lows = []

        for i in range(2, len(recent_data) - 2):
            if (
                recent_data.iloc[i]["high"] > recent_data.iloc[i - 1]["high"]
                and recent_data.iloc[i]["high"] > recent_data.iloc[i - 2]["high"]
                and recent_data.iloc[i]["high"] > recent_data.iloc[i + 1]["high"]
                and recent_data.iloc[i]["high"] > recent_data.iloc[i + 2]["high"]
            ):
                highs.append(recent_data.iloc[i]["high"])

            if (
                recent_data.iloc[i]["low"] < recent_data.iloc[i - 1]["low"]
                and recent_data.iloc[i]["low"] < recent_data.iloc[i - 2]["low"]
                and recent_data.iloc[i]["low"] < recent_data.iloc[i + 1]["low"]
                and recent_data.iloc[i]["low"] < recent_data.iloc[i + 2]["low"]
            ):
                lows.append(recent_data.iloc[i]["low"])

        if len(highs) < 3:
            highs = list(recent_data.nlargest(5, "high")["high"].values)
        if len(lows) < 3:
            lows = list(recent_data.nsmallest(5, "low")["low"].values)

        resistance_levels = sorted(highs, reverse=True)[:3]
        support_levels = sorted(lows)[:3]

        return {
            "resistance": resistance_levels,
            "support": support_levels,
            "main_resistance": resistance_levels[0] if resistance_levels else df["high"].max(),
            "main_support": support_levels[0] if support_levels else df["low"].min(),
        }

    def calculate_fibonacci_levels(self, df, lookback=50):
        """คำนวณ Fibonacci Retracement"""
        recent_data = df.tail(lookback)

        high = recent_data["high"].max()
        low = recent_data["low"].min()
        diff = high - low

        current_price = df.iloc[-1]["close"]

        if current_price > (high + low) / 2:
            fib_levels = {
                "0.0 (Low)": low,
                "0.236": low + (diff * 0.236),
                "0.382": low + (diff * 0.382),
                "0.5": low + (diff * 0.5),
                "0.618": low + (diff * 0.618),
                "0.786": low + (diff * 0.786),
                "1.0 (High)": high,
                "1.272": high + (diff * 0.272),
                "1.618": high + (diff * 0.618),
            }
            trend = "uptrend"
        else:
            fib_levels = {
                "0.0 (High)": high,
                "0.236": high - (diff * 0.236),
                "0.382": high - (diff * 0.382),
                "0.5": high - (diff * 0.5),
                "0.618": high - (diff * 0.618),
                "0.786": high - (diff * 0.786),
                "1.0 (Low)": low,
                "1.272": low - (diff * 0.272),
                "1.618": low - (diff * 0.618),
            }
            trend = "downtrend"

        return fib_levels, trend

    def analyze_multi_timeframe(self):
        """วิเคราะห์หลาย Timeframe"""

        print("📊 กำลังดึงข้อมูล...")
        self.data["weekly"] = self.fetch_data(self.timeframes["weekly"], 52)
        self.data["daily"] = self.fetch_data(self.timeframes["daily"], 100)
        self.data["h4"] = self.fetch_data(self.timeframes["h4"], 200)

        if any(df is None or df.empty for df in self.data.values()):
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return None

        for timeframe in self.data:
            self.data[timeframe] = self.calculate_indicators(self.data[timeframe])

        return True

    def check_divergence(self, df, indicator="RSI", lookback=14):
        """ตรวจสอบ Divergence แบบปรับปรุง - หา Swing Points จริงๆ"""
        if len(df) < lookback + 5:
            return None, 0

        recent = df.tail(lookback + 5).copy()
        recent = recent.reset_index(drop=True)

        # หา Swing Lows และ Swing Highs (อย่างน้อย 2 แท่งซ้ายขวา)
        swing_lows = []
        swing_highs = []

        for i in range(2, len(recent) - 2):
            # Swing Low
            if (recent.iloc[i]["low"] < recent.iloc[i-1]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i-2]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i+1]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i+2]["low"]):
                swing_lows.append({
                    "idx": i,
                    "price": recent.iloc[i]["low"],
                    "indicator": recent.iloc[i][indicator] if pd.notna(recent.iloc[i].get(indicator)) else None
                })

            # Swing High
            if (recent.iloc[i]["high"] > recent.iloc[i-1]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i-2]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i+1]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i+2]["high"]):
                swing_highs.append({
                    "idx": i,
                    "price": recent.iloc[i]["high"],
                    "indicator": recent.iloc[i][indicator] if pd.notna(recent.iloc[i].get(indicator)) else None
                })

        # ตรวจสอบ Bullish Divergence (ราคาทำ Lower Low แต่ Indicator ทำ Higher Low)
        if len(swing_lows) >= 2:
            last_two_lows = swing_lows[-2:]
            if (last_two_lows[1]["price"] < last_two_lows[0]["price"] and
                last_two_lows[1]["indicator"] is not None and
                last_two_lows[0]["indicator"] is not None and
                last_two_lows[1]["indicator"] > last_two_lows[0]["indicator"]):
                # คำนวณความแข็งแกร่งของ Divergence
                price_diff = abs(last_two_lows[1]["price"] - last_two_lows[0]["price"]) / last_two_lows[0]["price"] * 100
                ind_diff = abs(last_two_lows[1]["indicator"] - last_two_lows[0]["indicator"])
                strength = min(100, price_diff * 10 + ind_diff)
                return "bullish", strength

        # ตรวจสอบ Bearish Divergence (ราคาทำ Higher High แต่ Indicator ทำ Lower High)
        if len(swing_highs) >= 2:
            last_two_highs = swing_highs[-2:]
            if (last_two_highs[1]["price"] > last_two_highs[0]["price"] and
                last_two_highs[1]["indicator"] is not None and
                last_two_highs[0]["indicator"] is not None and
                last_two_highs[1]["indicator"] < last_two_highs[0]["indicator"]):
                price_diff = abs(last_two_highs[1]["price"] - last_two_highs[0]["price"]) / last_two_highs[0]["price"] * 100
                ind_diff = abs(last_two_highs[1]["indicator"] - last_two_highs[0]["indicator"])
                strength = min(100, price_diff * 10 + ind_diff)
                return "bearish", strength

        return None, 0

    def detect_market_regime(self, df):
        """ตรวจจับสภาวะตลาด: Trending, Ranging, หรือ Volatile"""
        latest = df.iloc[-1]
        lookback = min(20, len(df) - 1)

        # วิเคราะห์ ADX สำหรับ Trend Strength
        adx = latest["ADX"] if pd.notna(latest.get("ADX")) else 20

        # วิเคราะห์ Bollinger Band Width สำหรับ Volatility
        bb_width = latest["BB_width"] if pd.notna(latest.get("BB_width")) else 5
        avg_bb_width = df["BB_width"].tail(lookback).mean() if "BB_width" in df.columns else 5

        # วิเคราะห์ ATR Percent
        atr_pct = latest["ATR_percent"] if pd.notna(latest.get("ATR_percent")) else 3
        avg_atr_pct = df["ATR_percent"].tail(lookback).mean() if "ATR_percent" in df.columns else 3

        # วิเคราะห์ Price Range
        recent_high = df["high"].tail(lookback).max()
        recent_low = df["low"].tail(lookback).min()
        price_range_pct = (recent_high - recent_low) / recent_low * 100

        # ตัดสินสภาวะตลาด
        if adx > 25 and (latest["DI_plus"] - latest["DI_minus"]) > 5:
            regime = "STRONG_UPTREND"
            confidence = min(100, adx + 20)
        elif adx > 25 and (latest["DI_minus"] - latest["DI_plus"]) > 5:
            regime = "STRONG_DOWNTREND"
            confidence = min(100, adx + 20)
        elif adx < 20 and bb_width < avg_bb_width * 0.8:
            regime = "CONSOLIDATION"  # ตลาด Sideways แคบๆ
            confidence = 80 - adx
        elif bb_width > avg_bb_width * 1.5 or atr_pct > avg_atr_pct * 1.5:
            regime = "HIGH_VOLATILITY"
            confidence = min(100, bb_width / avg_bb_width * 50)
        elif adx >= 20 and adx <= 25:
            regime = "WEAK_TREND"
            confidence = 50
        else:
            regime = "RANGING"
            confidence = 60

        return {
            "regime": regime,
            "confidence": confidence,
            "adx": adx,
            "bb_width": bb_width,
            "atr_percent": atr_pct,
            "price_range_pct": price_range_pct
        }

    def analyze_historical_performance(self, df, lookback=50):
        """วิเคราะห์ประสิทธิภาพของสัญญาณในอดีต"""
        if len(df) < lookback + 20:
            return {"win_rate": 50, "avg_return": 0, "max_drawdown": 0, "sharpe": 0}

        historical = df.tail(lookback + 20).copy()
        historical = historical.reset_index(drop=True)

        # จำลองสัญญาณในอดีต
        signals = []
        for i in range(20, len(historical) - 5):
            row = historical.iloc[i]
            prev = historical.iloc[i-1]

            # สัญญาณ Long เบื้องต้น
            if (row["EMA_9"] > row["EMA_21"] and prev["EMA_9"] <= prev["EMA_21"]):
                signals.append({"idx": i, "type": "long", "entry": row["close"]})
            # สัญญาณ Short เบื้องต้น
            elif (row["EMA_9"] < row["EMA_21"] and prev["EMA_9"] >= prev["EMA_21"]):
                signals.append({"idx": i, "type": "short", "entry": row["close"]})

        # คำนวณผลลัพธ์ของแต่ละสัญญาณ (ดู 5 แท่งข้างหน้า)
        wins = 0
        losses = 0
        returns = []

        for signal in signals:
            if signal["idx"] + 5 >= len(historical):
                continue

            future_price = historical.iloc[signal["idx"] + 5]["close"]
            entry = signal["entry"]

            if signal["type"] == "long":
                ret = (future_price - entry) / entry * 100
            else:
                ret = (entry - future_price) / entry * 100

            returns.append(ret)
            if ret > 0:
                wins += 1
            else:
                losses += 1

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 50
        avg_return = np.mean(returns) if returns else 0

        # คำนวณ Max Drawdown
        cumulative = np.cumsum(returns) if returns else [0]
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # คำนวณ Sharpe Ratio (simplified)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = avg_return / np.std(returns) * np.sqrt(52)  # Annualized สำหรับ Weekly
        else:
            sharpe = 0

        return {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "total_signals": total
        }

    def check_trend_consistency(self, lookback=10):
        """ตรวจสอบความสอดคล้องของ Trend ในหลาย Timeframes"""
        if not self.data:
            return {"consistent": False, "direction": "neutral", "score": 0}

        scores = {"bullish": 0, "bearish": 0}
        weights = {"weekly": 3, "daily": 2, "h4": 1}

        for tf, weight in weights.items():
            if tf not in self.data or self.data[tf] is None:
                continue

            df = self.data[tf]
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)
            latest = df.iloc[-1]

            # EMA Trend
            if latest["EMA_9"] > latest["EMA_21"]:
                scores["bullish"] += weight * 2
            else:
                scores["bearish"] += weight * 2

            # MACD
            if latest["MACD"] > latest["MACD_signal"]:
                scores["bullish"] += weight
            else:
                scores["bearish"] += weight

            # Price vs EMA
            if latest["close"] > latest["EMA_21"]:
                scores["bullish"] += weight
            else:
                scores["bearish"] += weight

            # Trend ต่อเนื่อง (Higher Highs/Higher Lows หรือ Lower Highs/Lower Lows)
            higher_highs = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["high"] > recent.iloc[i-1]["high"])
            lower_lows = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["low"] < recent.iloc[i-1]["low"])

            if higher_highs > len(recent) * 0.6:
                scores["bullish"] += weight
            if lower_lows > len(recent) * 0.6:
                scores["bearish"] += weight

        total_score = scores["bullish"] + scores["bearish"]
        if total_score == 0:
            return {"consistent": False, "direction": "neutral", "score": 0}

        bullish_pct = scores["bullish"] / total_score * 100
        bearish_pct = scores["bearish"] / total_score * 100

        if bullish_pct >= 70:
            return {"consistent": True, "direction": "bullish", "score": bullish_pct}
        elif bearish_pct >= 70:
            return {"consistent": True, "direction": "bearish", "score": bearish_pct}
        else:
            return {"consistent": False, "direction": "mixed", "score": max(bullish_pct, bearish_pct)}

    def get_trend_strength(self, df):
        """วิเคราะห์ความแข็งแกร่งของ Trend"""
        latest = df.iloc[-1]

        score = 0
        max_score = 10

        if latest["EMA_9"] > latest["EMA_21"] > latest["EMA_50"]:
            score += 2
        elif latest["EMA_9"] < latest["EMA_21"] < latest["EMA_50"]:
            score -= 2

        if latest["close"] > latest["EMA_9"] > latest["EMA_21"]:
            score += 1
        elif latest["close"] < latest["EMA_9"] < latest["EMA_21"]:
            score -= 1

        if latest["ADX"] > 25:
            if latest["DI_plus"] > latest["DI_minus"]:
                score += 2
            else:
                score -= 2
        elif latest["ADX"] < 20:
            score = score * 0.5

        if pd.notna(latest.get("SUPERTREND_DIR")):
            if latest["SUPERTREND_DIR"] == 1:
                score += 1
            else:
                score -= 1

        if latest["MACD"] > latest["MACD_signal"] and latest["MACD_histogram"] > 0:
            score += 1
        elif latest["MACD"] < latest["MACD_signal"] and latest["MACD_histogram"] < 0:
            score -= 1

        return score, max_score

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
        """วิเคราะห์สัญญาณ Weekly แบบปรับปรุง พร้อมข้อมูลย้อนหลัง"""

        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]
        h4 = self.data["h4"].iloc[-1]

        weekly_prev = self.data["weekly"].iloc[-2]
        daily_prev = self.data["daily"].iloc[-2]

        # วิเคราะห์ Market Regime และ Historical Performance
        market_regime = self.detect_market_regime(self.data["daily"])
        historical_perf = self.analyze_historical_performance(self.data["daily"])
        trend_consistency = self.check_trend_consistency()

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

        if trend_consistency["consistent"]:
            direction = trend_consistency["direction"]
            if direction == "bullish":
                signals["long"] += 3
                reasons["long"].append(f"✅ Trend Consistency: Strong Bullish ({trend_consistency['score']:.0f}%)")
            elif direction == "bearish":
                signals["short"] += 3
                reasons["short"].append(f"✅ Trend Consistency: Strong Bearish ({trend_consistency['score']:.0f}%)")
        else:
            signals["neutral"] += 1
            reasons["neutral"].append(f"⚠️ Mixed Trend ({trend_consistency['score']:.0f}%)")

        # === WEEKLY TIMEFRAME ANALYSIS ===
        if weekly["EMA_9"] > weekly["EMA_21"]:
            signals["long"] += 3
            reasons["long"].append("📈 Weekly Uptrend: EMA 9 > 21")
        elif weekly["EMA_9"] < weekly["EMA_21"]:
            signals["short"] += 3
            reasons["short"].append("📉 Weekly Downtrend: EMA 9 < 21")

        if weekly_prev["EMA_9"] <= weekly_prev["EMA_21"] and weekly["EMA_9"] > weekly["EMA_21"]:
            signals["long"] += 5
            reasons["long"].append("🔥 Weekly Golden Cross!")
        elif weekly_prev["EMA_9"] >= weekly_prev["EMA_21"] and weekly["EMA_9"] < weekly["EMA_21"]:
            signals["short"] += 5
            reasons["short"].append("🔥 Weekly Death Cross!")

        if weekly["RSI"] < 30:
            signals["long"] += 3
            reasons["long"].append(f"💪 Weekly RSI Oversold: {weekly['RSI']:.1f}")
        elif weekly["RSI"] < 40:
            signals["long"] += 2
            reasons["long"].append(f"📊 Weekly RSI Low: {weekly['RSI']:.1f}")
        elif weekly["RSI"] > 70:
            signals["short"] += 3
            reasons["short"].append(f"⚠️ Weekly RSI Overbought: {weekly['RSI']:.1f}")
        elif weekly["RSI"] > 60:
            signals["short"] += 2
            reasons["short"].append(f"📊 Weekly RSI High: {weekly['RSI']:.1f}")
        elif 45 < weekly["RSI"] < 55:
            signals["neutral"] += 1
            reasons["neutral"].append(f"😐 Weekly RSI Neutral: {weekly['RSI']:.1f}")

        if weekly["MACD"] > weekly["MACD_signal"] and weekly["MACD_histogram"] > 0:
            signals["long"] += 2
            reasons["long"].append("📊 Weekly MACD Bullish")
            if weekly["MACD_histogram"] > weekly_prev["MACD_histogram"]:
                signals["long"] += 1
                reasons["long"].append("📈 Weekly MACD Momentum Increasing")
        elif weekly["MACD"] < weekly["MACD_signal"] and weekly["MACD_histogram"] < 0:
            signals["short"] += 2
            reasons["short"].append("📊 Weekly MACD Bearish")
            if weekly["MACD_histogram"] < weekly_prev["MACD_histogram"]:
                signals["short"] += 1
                reasons["short"].append("📉 Weekly MACD Momentum Decreasing")

        if pd.notna(weekly.get("STOCHRSI_K")):
            if weekly["STOCHRSI_K"] < 20 and weekly["STOCHRSI_D"] < 20:
                signals["long"] += 2
                reasons["long"].append(f"💪 Weekly StochRSI Oversold: {weekly['STOCHRSI_K']:.1f}")
            elif weekly["STOCHRSI_K"] > 80 and weekly["STOCHRSI_D"] > 80:
                signals["short"] += 2
                reasons["short"].append(f"⚠️ Weekly StochRSI Overbought: {weekly['STOCHRSI_K']:.1f}")

        # === DAILY TIMEFRAME CONFIRMATION ===
        if daily["EMA_9"] > daily["EMA_21"]:
            signals["long"] += 2
            reasons["long"].append("📈 Daily Uptrend")
        elif daily["EMA_9"] < daily["EMA_21"]:
            signals["short"] += 2
            reasons["short"].append("📉 Daily Downtrend")

        # RSI Analysis with Divergence
        if daily["RSI"] < 30:
            signals["long"] += 3
            reasons["long"].append(f"💪 Daily RSI Oversold: {daily['RSI']:.1f}")
        elif daily["RSI"] > 70:
            signals["short"] += 3
            reasons["short"].append(f"⚠️ Daily RSI Overbought: {daily['RSI']:.1f}")

        # Divergence Detection แบบปรับปรุง
        daily_divergence, div_strength = self.check_divergence(self.data["daily"], "RSI")
        if daily_divergence == "bullish" and div_strength > 0:
            # คะแนนตามความแข็งแกร่งของ Divergence
            div_score = 2 if div_strength < 30 else 3 if div_strength < 60 else 4
            signals["long"] += div_score
            reasons["long"].append(f"🔄 Daily Bullish Divergence (Strength: {div_strength:.0f})")
        elif daily_divergence == "bearish" and div_strength > 0:
            div_score = 2 if div_strength < 30 else 3 if div_strength < 60 else 4
            signals["short"] += div_score
            reasons["short"].append(f"🔄 Daily Bearish Divergence (Strength: {div_strength:.0f})")

        # ตรวจสอบ MACD Divergence ด้วย
        macd_divergence, macd_div_strength = self.check_divergence(self.data["daily"], "MACD", lookback=20)
        if macd_divergence == "bullish" and macd_div_strength > 20:
            signals["long"] += 2
            reasons["long"].append("🔄 MACD Bullish Divergence")
        elif macd_divergence == "bearish" and macd_div_strength > 20:
            signals["short"] += 2
            reasons["short"].append("🔄 MACD Bearish Divergence")

        if daily_prev["MACD"] <= daily_prev["MACD_signal"] and daily["MACD"] > daily["MACD_signal"]:
            signals["long"] += 2
            reasons["long"].append("✅ Daily MACD Cross Up")
        elif daily_prev["MACD"] >= daily_prev["MACD_signal"] and daily["MACD"] < daily["MACD_signal"]:
            signals["short"] += 2
            reasons["short"].append("❌ Daily MACD Cross Down")

        if pd.notna(daily.get("MFI")):
            if daily["MFI"] < 20:
                signals["long"] += 2
                reasons["long"].append(f"💰 Daily MFI Oversold: {daily['MFI']:.1f}")
            elif daily["MFI"] > 80:
                signals["short"] += 2
                reasons["short"].append(f"💰 Daily MFI Overbought: {daily['MFI']:.1f}")

        if pd.notna(daily.get("CCI")):
            if daily["CCI"] < -100:
                signals["long"] += 1
                reasons["long"].append(f"📊 Daily CCI Oversold: {daily['CCI']:.1f}")
            elif daily["CCI"] > 100:
                signals["short"] += 1
                reasons["short"].append(f"📊 Daily CCI Overbought: {daily['CCI']:.1f}")

        # === 4H TIMEFRAME ===
        if h4["EMA_9"] > h4["EMA_21"]:
            signals["long"] += 1
            reasons["long"].append("📊 4H Aligned Bullish")
        elif h4["EMA_9"] < h4["EMA_21"]:
            signals["short"] += 1
            reasons["short"].append("📊 4H Aligned Bearish")

        if pd.notna(h4.get("SUPERTREND_DIR")):
            if h4["SUPERTREND_DIR"] == 1:
                signals["long"] += 2
                reasons["long"].append("🚀 4H Supertrend Bullish")
            else:
                signals["short"] += 2
                reasons["short"].append("🔻 4H Supertrend Bearish")

        # === TREND STRENGTH ===
        if daily["ADX"] > 25:
            if daily["DI_plus"] > daily["DI_minus"]:
                signals["long"] += 2
                reasons["long"].append(f"💪 Strong Uptrend (ADX: {daily['ADX']:.1f})")
            else:
                signals["short"] += 2
                reasons["short"].append(f"💪 Strong Downtrend (ADX: {daily['ADX']:.1f})")
        elif daily["ADX"] < 20:
            signals["neutral"] += 2
            reasons["neutral"].append(f"🌊 Weak Trend (ADX: {daily['ADX']:.1f})")

        # === VOLUME CONFIRMATION ===
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
                reasons["long"].append("📈 OBV Above Average (Accumulation)")
            else:
                signals["short"] += 1
                reasons["short"].append("📉 OBV Below Average (Distribution)")

        # === BOLLINGER BANDS ===
        if daily["close"] < daily["BB_lower"]:
            signals["long"] += 1
            reasons["long"].append("📉 Price below BB Lower (Oversold)")
        elif daily["close"] > daily["BB_upper"]:
            signals["short"] += 1
            reasons["short"].append("📈 Price above BB Upper (Overbought)")

        # === ICHIMOKU ===
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

        return signals, reasons

    def calculate_volatility_adjusted_risk(self, df, base_risk_pct=2.0):
        """คำนวณ Risk ที่ปรับตาม Volatility"""
        latest = df.iloc[-1]
        lookback = min(20, len(df) - 1)

        # Current vs Average Volatility
        current_atr_pct = latest["ATR_percent"] if pd.notna(latest.get("ATR_percent")) else 3
        avg_atr_pct = df["ATR_percent"].tail(lookback).mean() if "ATR_percent" in df.columns else 3

        volatility_ratio = current_atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1

        # ปรับ Risk ตาม Volatility
        if volatility_ratio > 1.5:
            # Volatility สูงกว่าปกติ - ลด Risk
            adjusted_risk = base_risk_pct * 0.6
            risk_note = "High Volatility - Reduced Risk"
        elif volatility_ratio > 1.2:
            adjusted_risk = base_risk_pct * 0.8
            risk_note = "Elevated Volatility - Slightly Reduced Risk"
        elif volatility_ratio < 0.7:
            # Volatility ต่ำกว่าปกติ - เพิ่ม Risk ได้นิดหน่อย
            adjusted_risk = base_risk_pct * 1.2
            risk_note = "Low Volatility - Slightly Increased Risk"
        else:
            adjusted_risk = base_risk_pct
            risk_note = "Normal Volatility"

        return {
            "adjusted_risk_pct": min(adjusted_risk, 3.0),  # Cap at 3%
            "volatility_ratio": volatility_ratio,
            "current_atr_pct": current_atr_pct,
            "avg_atr_pct": avg_atr_pct,
            "risk_note": risk_note
        }

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

        if signal_type == "LONG":
            stop_loss_support = sr["main_support"]
            stop_loss_atr = current_price - (atr_daily * sl_multiplier)
            stop_loss = max(stop_loss_support, stop_loss_atr)

            tp1 = current_price + (atr_daily * tp_multiplier[0])
            tp2 = current_price + (atr_daily * tp_multiplier[1])
            tp3 = sr["main_resistance"]

            for level, price in fib_levels.items():
                if price > current_price and "1.272" in level:
                    tp3 = max(tp3, price)

        else:
            stop_loss_resistance = sr["main_resistance"]
            stop_loss_atr = current_price + (atr_daily * sl_multiplier)
            stop_loss = min(stop_loss_resistance, stop_loss_atr)

            tp1 = current_price - (atr_daily * tp_multiplier[0])
            tp2 = current_price - (atr_daily * tp_multiplier[1])
            tp3 = sr["main_support"]

            for level, price in fib_levels.items():
                if price < current_price and "1.272" in level:
                    tp3 = min(tp3, price)

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

    def get_confidence_level(self, signals):
        """คำนวณระดับความมั่นใจ"""
        total = signals["long"] + signals["short"] + signals["neutral"]
        if total == 0:
            return "WAIT", 0

        long_pct = signals["long"] / total * 100
        short_pct = signals["short"] / total * 100

        if long_pct >= 70:
            return "STRONG_LONG", long_pct
        elif long_pct >= 55:
            return "LONG", long_pct
        elif short_pct >= 70:
            return "STRONG_SHORT", short_pct
        elif short_pct >= 55:
            return "SHORT", short_pct
        else:
            return "WAIT", max(long_pct, short_pct)

    def get_weekly_recommendation(self, balance=10000):
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


class MonthlyTradingStrategy:
    """Strategy สำหรับ Trade รอบละ 1 เดือน"""

    def __init__(self, symbol="BTCUSDT", leverage=3):
        self.symbol = symbol
        self.leverage = leverage
        self.timeframes = {"monthly": "1M", "weekly": "1w", "daily": "1d"}
        self.data = {}

    def fetch_data(self, timeframe, limit=100):
        """ดึงข้อมูลจาก Binance"""
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": self.symbol, "interval": timeframe, "limit": limit}

        try:
            response = requests.get(url, params=params)
            data = response.json()

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_indicators(self, df):
        """คำนวณตัวชี้วัดแบบครบถ้วน"""

        # === MOVING AVERAGES ===
        df["EMA_12"] = ta.ema(df["close"], length=12)
        df["EMA_26"] = ta.ema(df["close"], length=26)
        df["EMA_50"] = ta.ema(df["close"], length=50)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_200"] = ta.sma(df["close"], length=200)

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

    def calculate_support_resistance(self, df, lookback=50):
        """คำนวณ Support & Resistance"""
        recent_data = df.tail(lookback)

        highs = recent_data.nlargest(5, "high")["high"].values
        lows = recent_data.nsmallest(5, "low")["low"].values

        resistance_levels = sorted(highs, reverse=True)[:3]
        support_levels = sorted(lows)[:3]

        return {
            "resistance": resistance_levels,
            "support": support_levels,
            "main_resistance": resistance_levels[0],
            "main_support": support_levels[0],
        }

    def calculate_fibonacci_levels(self, df, lookback=100):
        """คำนวณ Fibonacci Retracement"""
        recent_data = df.tail(lookback)

        high = recent_data["high"].max()
        low = recent_data["low"].min()
        diff = high - low
        current_price = df.iloc[-1]["close"]

        if current_price > (high + low) / 2:
            fib_levels = {
                "0.0 (Low)": low,
                "0.236": low + (diff * 0.236),
                "0.382": low + (diff * 0.382),
                "0.5": low + (diff * 0.5),
                "0.618": low + (diff * 0.618),
                "0.786": low + (diff * 0.786),
                "1.0 (High)": high,
                "1.272": high + (diff * 0.272),
                "1.618": high + (diff * 0.618),
            }
            trend = "uptrend"
        else:
            fib_levels = {
                "0.0 (High)": high,
                "0.236": high - (diff * 0.236),
                "0.382": high - (diff * 0.382),
                "0.5": high - (diff * 0.5),
                "0.618": high - (diff * 0.618),
                "0.786": high - (diff * 0.786),
                "1.0 (Low)": low,
                "1.272": low - (diff * 0.272),
                "1.618": low - (diff * 0.618),
            }
            trend = "downtrend"

        return fib_levels, trend

    def analyze_multi_timeframe(self):
        """วิเคราะห์หลาย Timeframe"""
        print("📊 กำลังดึงข้อมูล Monthly...")
        self.data["monthly"] = self.fetch_data(self.timeframes["monthly"], 60)
        self.data["weekly"] = self.fetch_data(self.timeframes["weekly"], 104)
        self.data["daily"] = self.fetch_data(self.timeframes["daily"], 200)

        if any(df is None or df.empty for df in self.data.values()):
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return None

        for timeframe in self.data:
            self.data[timeframe] = self.calculate_indicators(self.data[timeframe])

        return True

    def check_divergence(self, df, indicator="RSI", lookback=14):
        """ตรวจสอบ Divergence แบบปรับปรุง - หา Swing Points จริงๆ"""
        if len(df) < lookback + 5:
            return None, 0

        recent = df.tail(lookback + 5).copy()
        recent = recent.reset_index(drop=True)

        swing_lows = []
        swing_highs = []

        for i in range(2, len(recent) - 2):
            if (recent.iloc[i]["low"] < recent.iloc[i-1]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i-2]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i+1]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i+2]["low"]):
                swing_lows.append({
                    "idx": i,
                    "price": recent.iloc[i]["low"],
                    "indicator": recent.iloc[i][indicator] if pd.notna(recent.iloc[i].get(indicator)) else None
                })

            if (recent.iloc[i]["high"] > recent.iloc[i-1]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i-2]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i+1]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i+2]["high"]):
                swing_highs.append({
                    "idx": i,
                    "price": recent.iloc[i]["high"],
                    "indicator": recent.iloc[i][indicator] if pd.notna(recent.iloc[i].get(indicator)) else None
                })

        if len(swing_lows) >= 2:
            last_two_lows = swing_lows[-2:]
            if (last_two_lows[1]["price"] < last_two_lows[0]["price"] and
                last_two_lows[1]["indicator"] is not None and
                last_two_lows[0]["indicator"] is not None and
                last_two_lows[1]["indicator"] > last_two_lows[0]["indicator"]):
                price_diff = abs(last_two_lows[1]["price"] - last_two_lows[0]["price"]) / last_two_lows[0]["price"] * 100
                ind_diff = abs(last_two_lows[1]["indicator"] - last_two_lows[0]["indicator"])
                strength = min(100, price_diff * 10 + ind_diff)
                return "bullish", strength

        if len(swing_highs) >= 2:
            last_two_highs = swing_highs[-2:]
            if (last_two_highs[1]["price"] > last_two_highs[0]["price"] and
                last_two_highs[1]["indicator"] is not None and
                last_two_highs[0]["indicator"] is not None and
                last_two_highs[1]["indicator"] < last_two_highs[0]["indicator"]):
                price_diff = abs(last_two_highs[1]["price"] - last_two_highs[0]["price"]) / last_two_highs[0]["price"] * 100
                ind_diff = abs(last_two_highs[1]["indicator"] - last_two_highs[0]["indicator"])
                strength = min(100, price_diff * 10 + ind_diff)
                return "bearish", strength

        return None, 0

    def detect_market_regime(self, df):
        """ตรวจจับสภาวะตลาด"""
        latest = df.iloc[-1]
        lookback = min(20, len(df) - 1)

        adx = latest["ADX"] if pd.notna(latest.get("ADX")) else 20
        atr_pct = latest["ATR_percent"] if pd.notna(latest.get("ATR_percent")) else 3

        recent_high = df["high"].tail(lookback).max()
        recent_low = df["low"].tail(lookback).min()
        price_range_pct = (recent_high - recent_low) / recent_low * 100

        if adx > 25 and (latest["DI_plus"] - latest["DI_minus"]) > 5:
            regime = "STRONG_UPTREND"
            confidence = min(100, adx + 20)
        elif adx > 25 and (latest["DI_minus"] - latest["DI_plus"]) > 5:
            regime = "STRONG_DOWNTREND"
            confidence = min(100, adx + 20)
        elif adx < 20:
            regime = "RANGING"
            confidence = 60
        else:
            regime = "WEAK_TREND"
            confidence = 50

        return {
            "regime": regime,
            "confidence": confidence,
            "adx": adx,
            "atr_percent": atr_pct,
            "price_range_pct": price_range_pct
        }

    def analyze_historical_performance(self, df, lookback=50):
        """วิเคราะห์ประสิทธิภาพของสัญญาณในอดีต"""
        if len(df) < lookback + 20:
            return {"win_rate": 50, "avg_return": 0, "max_drawdown": 0, "total_signals": 0}

        historical = df.tail(lookback + 20).copy()
        historical = historical.reset_index(drop=True)

        signals = []
        for i in range(20, len(historical) - 5):
            row = historical.iloc[i]
            prev = historical.iloc[i-1]

            if (row["EMA_12"] > row["EMA_26"] and prev["EMA_12"] <= prev["EMA_26"]):
                signals.append({"idx": i, "type": "long", "entry": row["close"]})
            elif (row["EMA_12"] < row["EMA_26"] and prev["EMA_12"] >= prev["EMA_26"]):
                signals.append({"idx": i, "type": "short", "entry": row["close"]})

        wins = 0
        returns = []

        for signal in signals:
            if signal["idx"] + 5 >= len(historical):
                continue

            future_price = historical.iloc[signal["idx"] + 5]["close"]
            entry = signal["entry"]

            if signal["type"] == "long":
                ret = (future_price - entry) / entry * 100
            else:
                ret = (entry - future_price) / entry * 100

            returns.append(ret)
            if ret > 0:
                wins += 1

        total = len(returns)
        win_rate = (wins / total * 100) if total > 0 else 50
        avg_return = np.mean(returns) if returns else 0

        return {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_signals": total
        }

    def calculate_volatility_adjusted_risk(self, df, base_risk_pct=2.0):
        """คำนวณ Risk ที่ปรับตาม Volatility"""
        latest = df.iloc[-1]
        lookback = min(20, len(df) - 1)

        current_atr_pct = latest["ATR_percent"] if pd.notna(latest.get("ATR_percent")) else 3
        avg_atr_pct = df["ATR_percent"].tail(lookback).mean() if "ATR_percent" in df.columns else 3

        volatility_ratio = current_atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1

        if volatility_ratio > 1.5:
            adjusted_risk = base_risk_pct * 0.6
            risk_note = "High Volatility - Reduced Risk"
        elif volatility_ratio > 1.2:
            adjusted_risk = base_risk_pct * 0.8
            risk_note = "Elevated Volatility"
        elif volatility_ratio < 0.7:
            adjusted_risk = base_risk_pct * 1.2
            risk_note = "Low Volatility"
        else:
            adjusted_risk = base_risk_pct
            risk_note = "Normal Volatility"

        return {
            "adjusted_risk_pct": min(adjusted_risk, 3.0),
            "volatility_ratio": volatility_ratio,
            "current_atr_pct": current_atr_pct,
            "avg_atr_pct": avg_atr_pct,
            "risk_note": risk_note
        }

    def get_monthly_signal(self):
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

    def calculate_position_management(self, current_price, signal_type):
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

    def get_confidence_level(self, signals):
        """คำนวณระดับความมั่นใจ"""
        total = signals["long"] + signals["short"] + signals["neutral"]
        if total == 0:
            return "WAIT", 0

        long_pct = signals["long"] / total * 100
        short_pct = signals["short"] / total * 100

        if long_pct >= 65:
            return "STRONG_LONG", long_pct
        elif long_pct >= 55:
            return "LONG", long_pct
        elif short_pct >= 65:
            return "STRONG_SHORT", short_pct
        elif short_pct >= 55:
            return "SHORT", short_pct
        else:
            return "WAIT", max(long_pct, short_pct)

    def get_monthly_recommendation(self, balance=10000):
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
