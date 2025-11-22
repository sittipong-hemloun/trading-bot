"""
Candlestick Patterns Module
Contains functions for detecting candlestick patterns
"""

import pandas as pd
import numpy as np


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """ตรวจจับ Candlestick Patterns"""
    # Calculate candle properties
    body = abs(df["close"] - df["open"])
    upper_shadow = df["high"] - df[["close", "open"]].max(axis=1)
    lower_shadow = df[["close", "open"]].min(axis=1) - df["low"]
    candle_range = df["high"] - df["low"]

    # Avoid division by zero
    candle_range = candle_range.replace(0, np.nan)

    # Doji - small body relative to range
    df["DOJI"] = (body / candle_range) < 0.1

    # Hammer - small body at top, long lower shadow
    df["HAMMER"] = (
        (lower_shadow > body * 2) &
        (upper_shadow < body * 0.5) &
        (df["close"] > df["open"])  # Bullish
    )

    # Inverted Hammer
    df["INVERTED_HAMMER"] = (
        (upper_shadow > body * 2) &
        (lower_shadow < body * 0.5) &
        (df["close"] > df["open"])
    )

    # Shooting Star - small body at bottom, long upper shadow (bearish)
    df["SHOOTING_STAR"] = (
        (upper_shadow > body * 2) &
        (lower_shadow < body * 0.5) &
        (df["close"] < df["open"])
    )

    # Hanging Man - like hammer but at top of uptrend (bearish)
    df["HANGING_MAN"] = (
        (lower_shadow > body * 2) &
        (upper_shadow < body * 0.5) &
        (df["close"] < df["open"])
    )

    # Engulfing patterns
    prev_body = body.shift(1)
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    # Bullish Engulfing
    df["BULLISH_ENGULFING"] = (
        (df["close"] > df["open"]) &  # Current is bullish
        (prev_close < prev_open) &  # Previous is bearish
        (df["open"] < prev_close) &  # Open below previous close
        (df["close"] > prev_open) &  # Close above previous open
        (body > prev_body)  # Current body larger
    )

    # Bearish Engulfing
    df["BEARISH_ENGULFING"] = (
        (df["close"] < df["open"]) &  # Current is bearish
        (prev_close > prev_open) &  # Previous is bullish
        (df["open"] > prev_close) &  # Open above previous close
        (df["close"] < prev_open) &  # Close below previous open
        (body > prev_body)  # Current body larger
    )

    # Morning Star (3 candle bullish reversal)
    df["MORNING_STAR"] = (
        (df["close"].shift(2) < df["open"].shift(2)) &  # First candle bearish
        (body.shift(1) < body.shift(2) * 0.3) &  # Second candle small
        (df["close"] > df["open"]) &  # Third candle bullish
        (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)  # Close above midpoint of first
    )

    # Evening Star (3 candle bearish reversal)
    df["EVENING_STAR"] = (
        (df["close"].shift(2) > df["open"].shift(2)) &  # First candle bullish
        (body.shift(1) < body.shift(2) * 0.3) &  # Second candle small
        (df["close"] < df["open"]) &  # Third candle bearish
        (df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2)  # Close below midpoint of first
    )

    # Three White Soldiers (strong bullish)
    df["THREE_WHITE_SOLDIERS"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["close"].shift(2) > df["open"].shift(2)) &
        (df["close"] > df["close"].shift(1)) &
        (df["close"].shift(1) > df["close"].shift(2))
    )

    # Three Black Crows (strong bearish)
    df["THREE_BLACK_CROWS"] = (
        (df["close"] < df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"].shift(2) < df["open"].shift(2)) &
        (df["close"] < df["close"].shift(1)) &
        (df["close"].shift(1) < df["close"].shift(2))
    )

    return df


def get_candlestick_signals(df: pd.DataFrame) -> dict:
    """รวบรวมสัญญาณจาก Candlestick Patterns"""
    latest = df.iloc[-1]
    signals = {"bullish": [], "bearish": [], "score": 0}

    # Bullish patterns
    bullish_patterns = [
        ("HAMMER", "Hammer", 2),
        ("INVERTED_HAMMER", "Inverted Hammer", 1),
        ("BULLISH_ENGULFING", "Bullish Engulfing", 3),
        ("MORNING_STAR", "Morning Star", 3),
        ("THREE_WHITE_SOLDIERS", "Three White Soldiers", 4),
    ]

    for col, name, score in bullish_patterns:
        if pd.notna(latest.get(col)) and latest[col]:
            signals["bullish"].append(name)
            signals["score"] += score

    # Bearish patterns
    bearish_patterns = [
        ("SHOOTING_STAR", "Shooting Star", 2),
        ("HANGING_MAN", "Hanging Man", 1),
        ("BEARISH_ENGULFING", "Bearish Engulfing", 3),
        ("EVENING_STAR", "Evening Star", 3),
        ("THREE_BLACK_CROWS", "Three Black Crows", 4),
    ]

    for col, name, score in bearish_patterns:
        if pd.notna(latest.get(col)) and latest[col]:
            signals["bearish"].append(name)
            signals["score"] -= score

    return signals
