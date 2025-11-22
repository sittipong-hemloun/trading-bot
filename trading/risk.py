"""
Risk Management Module
Contains functions for risk calculation, position sizing, and volatility analysis
"""

import pandas as pd


def calculate_risk_score(df: pd.DataFrame, signal_type: str) -> dict:
    """คำนวณ Risk Score สำหรับ Trade (0-100, lower is better)"""
    latest = df.iloc[-1]
    risk_score = 50  # Start at neutral
    risk_factors = []

    # 1. Volatility Risk
    atr_pct = latest["ATR_percent"] if pd.notna(latest.get("ATR_percent")) else 3
    if atr_pct > 5:
        risk_score += 15
        risk_factors.append(f"High Volatility ({atr_pct:.1f}%)")
    elif atr_pct > 3:
        risk_score += 5
        risk_factors.append(f"Moderate Volatility ({atr_pct:.1f}%)")
    else:
        risk_score -= 5
        risk_factors.append(f"Low Volatility ({atr_pct:.1f}%)")

    # 2. Trend Alignment Risk
    if signal_type == "LONG":
        if latest["EMA_9"] < latest["EMA_21"]:
            risk_score += 10
            risk_factors.append("Counter-trend: EMA bearish")
        if latest["close"] < latest["EMA_50"]:
            risk_score += 5
            risk_factors.append("Price below EMA50")
    else:  # SHORT
        if latest["EMA_9"] > latest["EMA_21"]:
            risk_score += 10
            risk_factors.append("Counter-trend: EMA bullish")
        if latest["close"] > latest["EMA_50"]:
            risk_score += 5
            risk_factors.append("Price above EMA50")

    # 3. RSI Risk
    rsi = latest["RSI"] if pd.notna(latest.get("RSI")) else 50
    if signal_type == "LONG" and rsi > 70:
        risk_score += 10
        risk_factors.append("RSI Overbought for Long")
    elif signal_type == "SHORT" and rsi < 30:
        risk_score += 10
        risk_factors.append("RSI Oversold for Short")

    # 4. ADX Risk (weak trend)
    adx = latest["ADX"] if pd.notna(latest.get("ADX")) else 20
    if adx < 20:
        risk_score += 10
        risk_factors.append(f"Weak Trend (ADX: {adx:.0f})")
    elif adx > 40:
        risk_score -= 10
        risk_factors.append(f"Strong Trend (ADX: {adx:.0f})")

    # 5. Volume Risk
    vol_ratio = latest["Volume_Ratio"] if pd.notna(latest.get("Volume_Ratio")) else 1
    if vol_ratio < 0.5:
        risk_score += 10
        risk_factors.append("Low Volume")
    elif vol_ratio > 2:
        risk_score -= 5
        risk_factors.append("High Volume Confirmation")

    # 6. BB Position Risk
    bb_pct = latest["BB_percent"] if pd.notna(latest.get("BB_percent")) else 0.5
    if signal_type == "LONG" and bb_pct > 0.9:
        risk_score += 10
        risk_factors.append("Price at BB Upper")
    elif signal_type == "SHORT" and bb_pct < 0.1:
        risk_score += 10
        risk_factors.append("Price at BB Lower")

    # Clamp between 0-100
    risk_score = max(0, min(100, risk_score))

    risk_level = "LOW" if risk_score < 40 else "MEDIUM" if risk_score < 60 else "HIGH"

    return {
        "score": risk_score,
        "level": risk_level,
        "factors": risk_factors
    }


def calculate_volatility_adjusted_risk(df: pd.DataFrame, base_risk_pct: float = 2.0) -> dict:
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


def calculate_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """คำนวณระดับ Support และ Resistance"""
    if lookback > len(df):
        lookback = len(df)
    if lookback < 1:
        lookback = 1

    recent = df.tail(lookback)

    # หา High/Low ที่สำคัญ
    highs = recent["high"].nlargest(3).tolist()
    lows = recent["low"].nsmallest(3).tolist()

    # Pivot Points
    pivot = (recent["high"].iloc[-1] + recent["low"].iloc[-1] + recent["close"].iloc[-1]) / 3

    return {
        "support": sorted(lows, reverse=True),
        "resistance": sorted(highs),
        "main_support": lows[0] if lows else recent["low"].min(),
        "main_resistance": highs[0] if highs else recent["high"].max(),
        "pivot": pivot,
    }


def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> tuple:
    """คำนวณ Fibonacci Retracement และ Extension Levels"""
    if lookback > len(df):
        lookback = len(df)

    recent = df.tail(lookback)
    high = recent["high"].max()
    low = recent["low"].min()

    # Determine trend
    first_close = recent["close"].iloc[0]
    last_close = recent["close"].iloc[-1]
    is_uptrend = last_close > first_close

    diff = high - low

    if is_uptrend:
        # Uptrend: retracement from high, extension above high
        levels = {
            # Retracement levels (support zones for pullback)
            "0.0%": low,
            "23.6%": low + diff * 0.236,
            "38.2%": low + diff * 0.382,
            "50.0%": low + diff * 0.5,
            "61.8%": low + diff * 0.618,
            "78.6%": low + diff * 0.786,
            "100%": high,
            # Extension levels (profit targets above high)
            "127.2%": high + diff * 0.272,
            "161.8%": high + diff * 0.618,
            "200.0%": high + diff * 1.0,
            "261.8%": high + diff * 1.618,
        }
        trend = "uptrend"
    else:
        # Downtrend: retracement from low, extension below low
        levels = {
            # Retracement levels (resistance zones for pullback)
            "0.0%": high,
            "23.6%": high - diff * 0.236,
            "38.2%": high - diff * 0.382,
            "50.0%": high - diff * 0.5,
            "61.8%": high - diff * 0.618,
            "78.6%": high - diff * 0.786,
            "100%": low,
            # Extension levels (profit targets below low)
            "127.2%": low - diff * 0.272,
            "161.8%": low - diff * 0.618,
            "200.0%": low - diff * 1.0,
            "261.8%": low - diff * 1.618,
        }
        trend = "downtrend"

    return levels, trend
