"""
Trading Strategies Module
Contains WeeklyTradingStrategy and MonthlyTradingStrategy classes
"""

import requests
import pandas as pd
import pandas_ta as ta
from datetime import timedelta


def fetch_from_coingecko(symbol, days=100):
    """Fetch OHLC data from CoinGecko market_chart (no geo-restrictions)"""
    # Map symbol to CoinGecko ID
    symbol_map = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "BNBUSDT": "binancecoin",
        "SOLUSDT": "solana",
        "XRPUSDT": "ripple",
        "ADAUSDT": "cardano",
        "DOGEUSDT": "dogecoin",
    }

    coin_id = symbol_map.get(symbol.upper(), "bitcoin")

    # Use market_chart endpoint for more data points
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"   CoinGecko: HTTP {response.status_code}")
            return None

        data = response.json()
        if not data or "prices" not in data:
            print("   CoinGecko: No data returned")
            return None

        # market_chart returns prices, market_caps, total_volumes
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if len(prices) < 30:
            print(f"   CoinGecko: Insufficient data ({len(prices)} points)")
            return None

        # Create DataFrame from prices
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["close"].astype(float)

        # Add volume
        if volumes and len(volumes) == len(prices):
            df["volume"] = [v[1] for v in volumes]
        else:
            df["volume"] = 0.0

        # Estimate OHLC from close prices (since we only have close)
        # Use close as base and add small variations for open/high/low
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * 1.005  # +0.5%
        df["low"] = df[["open", "close"]].min(axis=1) * 0.995   # -0.5%

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        print(f"   CoinGecko: Got {len(df)} data points")
        return df

    except Exception as e:
        print(f"   CoinGecko error: {type(e).__name__}: {e}")
        return None


def fetch_from_binance(symbol, interval, limit=100):
    """Fetch data from Binance with multiple endpoint fallback"""
    api_endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api4.binance.com/api/v3/klines",
    ]

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    for url in api_endpoints:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 451:
                # Geo-blocked, skip all Binance endpoints
                return None

            if response.status_code != 200:
                continue

            data = response.json()
            if isinstance(data, dict) and "code" in data:
                continue

            if not data:
                continue

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

        except Exception:
            continue

    return None


class WeeklyTradingStrategy:
    """Strategy สำหรับ Trade รอบละ 1 สัปดาห์"""

    def __init__(self, symbol="BTCUSDT", leverage=5):
        self.symbol = symbol
        self.leverage = leverage
        self.timeframes = {"weekly": "1w", "daily": "1d", "h4": "4h"}
        self.data = {}

    def fetch_data(self, timeframe, limit=100):
        """ดึงข้อมูลจาก Binance with CoinGecko fallback"""
        # Try Binance first
        df = fetch_from_binance(self.symbol, timeframe, limit)
        if df is not None:
            return df

        # Fallback to CoinGecko if Binance is blocked (HTTP 451)
        print("   Binance blocked, trying CoinGecko...")
        days = 90 if timeframe in ["1w", "1M"] else 30
        df = fetch_from_coingecko(self.symbol, days=days)
        return df

    def calculate_indicators(self, df):
        """คำนวณตัวชี้วัดแบบครบถ้วน (with safety checks)"""

        # === MOVING AVERAGES ===
        df["EMA_9"] = ta.ema(df["close"], length=9)
        df["EMA_21"] = ta.ema(df["close"], length=21)
        df["EMA_50"] = ta.ema(df["close"], length=50)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_200"] = ta.sma(df["close"], length=200)

        # === RSI ===
        df["RSI"] = ta.rsi(df["close"], length=14)

        # === MACD (with safety check) ===
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_histogram"] = macd["MACDh_12_26_9"]
        else:
            df["MACD"] = 0.0
            df["MACD_signal"] = 0.0
            df["MACD_histogram"] = 0.0

        # === Stochastic RSI (with safety check) ===
        stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi is not None and not stochrsi.empty:
            df["STOCHRSI_K"] = stochrsi["STOCHRSIk_14_14_3_3"]
            df["STOCHRSI_D"] = stochrsi["STOCHRSId_14_14_3_3"]
        else:
            df["STOCHRSI_K"] = 50.0
            df["STOCHRSI_D"] = 50.0

        # === Stochastic (with safety check) ===
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df["STOCH_K"] = stoch["STOCHk_14_3_3"]
            df["STOCH_D"] = stoch["STOCHd_14_3_3"]
        else:
            df["STOCH_K"] = 50.0
            df["STOCH_D"] = 50.0

        # === Bollinger Bands (with safety check) ===
        bbands = ta.bbands(df["close"], length=20, std=2.0)  # type: ignore[arg-type]
        if bbands is not None and not bbands.empty:
            df["BB_upper"] = bbands["BBU_20_2.0_2.0"]
            df["BB_middle"] = bbands["BBM_20_2.0_2.0"]
            df["BB_lower"] = bbands["BBL_20_2.0_2.0"]
            df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"] * 100
            df["BB_percent"] = (df["close"] - df["BB_lower"]) / (
                df["BB_upper"] - df["BB_lower"]
            )
        else:
            df["BB_upper"] = df["close"] * 1.02
            df["BB_middle"] = df["close"]
            df["BB_lower"] = df["close"] * 0.98
            df["BB_width"] = 4.0
            df["BB_percent"] = 0.5

        # === ADX (Trend Strength) (with safety check) ===
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            df["ADX"] = adx["ADX_14"]
            df["DI_plus"] = adx["DMP_14"]
            df["DI_minus"] = adx["DMN_14"]
        else:
            df["ADX"] = 25.0
            df["DI_plus"] = 25.0
            df["DI_minus"] = 25.0

        # === ATR ===
        df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        if df["ATR"].isna().all():
            df["ATR"] = df["close"] * 0.02  # Fallback: 2% of price
        df["ATR_percent"] = df["ATR"] / df["close"] * 100

        # === Volume Analysis ===
        df["Volume_MA"] = df["volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["volume"] / df["Volume_MA"].replace(0, 1)

        # === OBV (On-Balance Volume) ===
        df["OBV"] = ta.obv(df["close"], df["volume"])
        if df["OBV"].isna().all():
            df["OBV"] = 0.0
        df["OBV_EMA"] = ta.ema(df["OBV"], length=21)

        # === MFI (Money Flow Index) ===
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        if df["MFI"].isna().all():
            df["MFI"] = 50.0

        # === CCI (Commodity Channel Index) ===
        # Manual calculation because pandas_ta CCI has issues with high-priced assets
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(20).mean()
        mean_dev = abs(typical_price - sma_tp).rolling(20).mean()
        df["CCI"] = (typical_price - sma_tp) / (0.015 * mean_dev)

        # === Williams %R ===
        df["WILLR"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        # === Ichimoku Cloud ===
        ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
        if ichimoku is not None and len(ichimoku) >= 2:
            df["ICHI_TENKAN"] = ichimoku[0]["ITS_9"]
            df["ICHI_KIJUN"] = ichimoku[0]["IKS_26"]
            df["ICHI_SENKOU_A"] = ichimoku[0]["ISA_9"]
            df["ICHI_SENKOU_B"] = ichimoku[0]["ISB_26"]

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
            return False

        for timeframe in self.data:
            self.data[timeframe] = self.calculate_indicators(self.data[timeframe])

        return True

    def check_divergence(self, df, indicator="RSI", lookback=14):
        """ตรวจสอบ Divergence (Bullish/Bearish)"""
        price = df["close"].tail(lookback)
        ind = df[indicator].tail(lookback)

        price_higher_high = price.iloc[-1] > price.iloc[0]
        price_lower_low = price.iloc[-1] < price.iloc[0]
        ind_higher_high = ind.iloc[-1] > ind.iloc[0]
        ind_lower_low = ind.iloc[-1] < ind.iloc[0]

        if price_lower_low and not ind_lower_low:
            return "bullish"

        if price_higher_high and not ind_higher_high:
            return "bearish"

        return None

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

    def get_weekly_signal(self):
        """วิเคราะห์สัญญาณ Weekly แบบปรับปรุง"""

        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]
        h4 = self.data["h4"].iloc[-1]

        weekly_prev = self.data["weekly"].iloc[-2]
        daily_prev = self.data["daily"].iloc[-2]

        signals = {"long": 0, "short": 0, "neutral": 0}
        reasons = {"long": [], "short": [], "neutral": []}

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

        daily_divergence = self.check_divergence(self.data["daily"], "RSI")
        if daily["RSI"] < 30:
            signals["long"] += 3
            reasons["long"].append(f"💪 Daily RSI Oversold: {daily['RSI']:.1f}")
        elif daily["RSI"] > 70:
            signals["short"] += 3
            reasons["short"].append(f"⚠️ Daily RSI Overbought: {daily['RSI']:.1f}")

        if daily_divergence == "bullish":
            signals["long"] += 2
            reasons["long"].append("🔄 Daily Bullish Divergence")
        elif daily_divergence == "bearish":
            signals["short"] += 2
            reasons["short"].append("🔄 Daily Bearish Divergence")

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

    def calculate_position_management(self, current_price, signal_type):
        """คำนวณการจัดการ Position สำหรับ Weekly แบบปรับปรุง"""

        daily_df = self.data["daily"]
        daily = daily_df.iloc[-1]

        atr_daily = daily["ATR"]
        sr = self.calculate_support_resistance(daily_df)
        fib_levels, fib_trend = self.calculate_fibonacci_levels(daily_df)

        atr_percent = daily["ATR_percent"]
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
            return False

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

        return True

    def _print_trade_setup(self, position_mgmt, signal_type, balance, current_price):
        """พิมพ์ Trade Setup"""
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

        print(f"\n📊 Volatility: {position_mgmt['atr_percent']:.2f}% (ATR: ${position_mgmt['atr']:,.2f})")

        print("\n💼 TRADE SETUP:")
        print(f"  🎯 Entry: ${entry:,.2f}")
        print(f"  🛡️ Stop Loss: ${sl:,.2f} ({sl_pct:+.2f}% = {sl_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP1 (40%): ${tp1:,.2f} ({tp1_pct:+.2f}% = {tp1_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP2 (30%): ${tp2:,.2f} ({tp2_pct:+.2f}% = {tp2_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP3 (30%): ${tp3:,.2f} ({tp3_pct:+.2f}% = {tp3_pct * self.leverage:+.1f}% margin)")

        risk_pct = 2
        risk_amount = balance * (risk_pct / 100)
        position_size = (risk_amount / (abs(sl_pct) / 100)) * self.leverage
        margin_required = position_size / self.leverage

        print(f"\n💰 POSITION MANAGEMENT (Balance: ${balance:,.2f}):")
        print(f"  📊 Leverage: {self.leverage}x")
        print(f"  📊 Risk per Trade: {risk_pct}% (${risk_amount:,.2f})")
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
        """ดึงข้อมูลจาก Binance with CoinGecko fallback"""
        # Try Binance first
        df = fetch_from_binance(self.symbol, timeframe, limit)
        if df is not None:
            return df

        # Fallback to CoinGecko if Binance is blocked (HTTP 451)
        print("   Binance blocked, trying CoinGecko...")
        days = 180 if timeframe == "1M" else 90
        df = fetch_from_coingecko(self.symbol, days=days)
        return df

    def calculate_indicators(self, df):
        """คำนวณตัวชี้วัดแบบครบถ้วน (with safety checks)"""

        # === MOVING AVERAGES ===
        df["EMA_12"] = ta.ema(df["close"], length=12)
        df["EMA_26"] = ta.ema(df["close"], length=26)
        df["EMA_50"] = ta.ema(df["close"], length=50)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_200"] = ta.sma(df["close"], length=200)

        # === RSI ===
        df["RSI"] = ta.rsi(df["close"], length=14)

        # === MACD (with safety check) ===
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_histogram"] = macd["MACDh_12_26_9"]
        else:
            df["MACD"] = 0.0
            df["MACD_signal"] = 0.0
            df["MACD_histogram"] = 0.0

        # === Stochastic RSI (with safety check) ===
        stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi is not None and not stochrsi.empty:
            df["STOCHRSI_K"] = stochrsi["STOCHRSIk_14_14_3_3"]
            df["STOCHRSI_D"] = stochrsi["STOCHRSId_14_14_3_3"]
        else:
            df["STOCHRSI_K"] = 50.0
            df["STOCHRSI_D"] = 50.0

        # === Stochastic (with safety check) ===
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df["STOCH_K"] = stoch["STOCHk_14_3_3"]
            df["STOCH_D"] = stoch["STOCHd_14_3_3"]
        else:
            df["STOCH_K"] = 50.0
            df["STOCH_D"] = 50.0

        # === Bollinger Bands (with safety check) ===
        bbands = ta.bbands(df["close"], length=20, std=2.0)  # type: ignore[arg-type]
        if bbands is not None and not bbands.empty:
            df["BB_upper"] = bbands["BBU_20_2.0_2.0"]
            df["BB_middle"] = bbands["BBM_20_2.0_2.0"]
            df["BB_lower"] = bbands["BBL_20_2.0_2.0"]
        else:
            df["BB_upper"] = df["close"] * 1.02
            df["BB_middle"] = df["close"]
            df["BB_lower"] = df["close"] * 0.98

        # === ADX (with safety check) ===
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            df["ADX"] = adx["ADX_14"]
            df["DI_plus"] = adx["DMP_14"]
            df["DI_minus"] = adx["DMN_14"]
        else:
            df["ADX"] = 25.0
            df["DI_plus"] = 25.0
            df["DI_minus"] = 25.0

        # === ATR ===
        df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        if df["ATR"].isna().all():
            df["ATR"] = df["close"] * 0.02
        df["ATR_percent"] = df["ATR"] / df["close"] * 100

        # === Volume ===
        df["Volume_MA"] = df["volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["volume"] / df["Volume_MA"].replace(0, 1)

        # === OBV ===
        df["OBV"] = ta.obv(df["close"], df["volume"])
        if df["OBV"].isna().all():
            df["OBV"] = 0.0
        df["OBV_EMA"] = ta.ema(df["OBV"], length=21)

        # === MFI ===
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        if df["MFI"].isna().all():
            df["MFI"] = 50.0

        # === Supertrend (with safety check) ===
        supertrend = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        if supertrend is not None and not supertrend.empty:
            df["SUPERTREND"] = supertrend["SUPERT_10_3.0"]
            df["SUPERTREND_DIR"] = supertrend["SUPERTd_10_3.0"]
        else:
            df["SUPERTREND"] = df["close"]
            df["SUPERTREND_DIR"] = 1

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
            return False

        for timeframe in self.data:
            self.data[timeframe] = self.calculate_indicators(self.data[timeframe])

        return True

    def check_divergence(self, df, indicator="RSI", lookback=14):
        """ตรวจสอบ Divergence"""
        price = df["close"].tail(lookback)
        ind = df[indicator].tail(lookback)

        price_higher_high = price.iloc[-1] > price.iloc[0]
        price_lower_low = price.iloc[-1] < price.iloc[0]
        ind_higher_high = ind.iloc[-1] > ind.iloc[0]
        ind_lower_low = ind.iloc[-1] < ind.iloc[0]

        if price_lower_low and not ind_lower_low:
            return "bullish"
        if price_higher_high and not ind_higher_high:
            return "bearish"
        return None

    def get_monthly_signal(self):
        """วิเคราะห์สัญญาณ Monthly แบบปรับปรุง"""

        monthly = self.data["monthly"].iloc[-1]
        weekly = self.data["weekly"].iloc[-1]
        daily = self.data["daily"].iloc[-1]

        monthly_prev = self.data["monthly"].iloc[-2]
        weekly_prev = self.data["weekly"].iloc[-2]

        signals = {"long": 0, "short": 0, "neutral": 0}
        reasons = {"long": [], "short": [], "neutral": []}

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

        daily_divergence = self.check_divergence(self.data["daily"], "RSI")
        if daily_divergence == "bullish":
            signals["long"] += 2
            reasons["long"].append("🔄 Daily Bullish Divergence")
        elif daily_divergence == "bearish":
            signals["short"] += 2
            reasons["short"].append("🔄 Daily Bearish Divergence")

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
        """คำนวณการจัดการ Position สำหรับ Monthly"""

        monthly_df = self.data["monthly"]
        monthly = monthly_df.iloc[-1]

        atr_monthly = monthly["ATR"]
        atr_percent = monthly["ATR_percent"]
        sr = self.calculate_support_resistance(monthly_df)
        fib_levels, fib_trend = self.calculate_fibonacci_levels(monthly_df)

        if signal_type == "LONG":
            stop_loss_support = sr["main_support"]
            stop_loss_atr = current_price - (atr_monthly * 2)
            stop_loss = max(stop_loss_support, stop_loss_atr)

            tp1 = current_price + (atr_monthly * 3)
            tp2 = sr["main_resistance"]
            tp3 = current_price + (atr_monthly * 6)

            for level, price in fib_levels.items():
                if price > current_price and "1.272" in level:
                    tp3 = max(tp3, price)
        else:
            stop_loss_resistance = sr["main_resistance"]
            stop_loss_atr = current_price + (atr_monthly * 2)
            stop_loss = min(stop_loss_resistance, stop_loss_atr)

            tp1 = current_price - (atr_monthly * 3)
            tp2 = sr["main_support"]
            tp3 = current_price - (atr_monthly * 6)

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
            return False

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

        return True

    def _print_trade_setup(self, position_mgmt, signal_type, balance, current_price):
        """พิมพ์ Trade Setup"""
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

        print(f"\n📊 Volatility: {position_mgmt['atr_percent']:.2f}% (ATR: ${position_mgmt['atr']:,.2f})")

        print("\n💼 TRADE SETUP:")
        print(f"  🎯 Entry: ${entry:,.2f}")
        print(f"  🛡️ Stop Loss: ${sl:,.2f} ({sl_pct:+.2f}% = {sl_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP1 (33%): ${tp1:,.2f} ({tp1_pct:+.2f}% = {tp1_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP2 (33%): ${tp2:,.2f} ({tp2_pct:+.2f}% = {tp2_pct * self.leverage:+.1f}% margin)")
        print(f"  🎁 TP3 (34%): ${tp3:,.2f} ({tp3_pct:+.2f}% = {tp3_pct * self.leverage:+.1f}% margin)")

        risk_pct = 2
        risk_amount = balance * (risk_pct / 100)
        position_size = (risk_amount / (abs(sl_pct) / 100)) * self.leverage
        margin_required = position_size / self.leverage

        print(f"\n💰 POSITION MANAGEMENT (Balance: ${balance:,.2f}):")
        print(f"  📊 Leverage: {self.leverage}x")
        print(f"  📊 Risk per Trade: {risk_pct}% (${risk_amount:,.2f})")
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
