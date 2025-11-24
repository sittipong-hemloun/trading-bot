"""
Technical Indicators Module
Contains functions for calculating technical indicators
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


def calculate_indicators(
    df: pd.DataFrame,
    timeframe: str = "daily"
) -> pd.DataFrame:
    """
    คำนวณตัวชี้วัดแบบครบถ้วน พร้อมปรับ parameters ตาม timeframe

    Args:
        df: DataFrame with OHLCV data
        timeframe: "weekly", "daily", or "h4" - affects indicator parameters

    Weekly uses shorter periods because each candle = 1 week:
    - RSI: 7 (vs 14 for daily) = ~7 weeks lookback
    - MACD: 8/17/9 (vs 12/26/9) = faster response
    - StochRSI: 7 (vs 14)
    """
    from trading.patterns import detect_candlestick_patterns

    # === PARAMETER CONFIGURATION BY TIMEFRAME ===
    if timeframe == "weekly":
        # Weekly: shorter periods (each candle = 1 week)
        rsi_length = 7
        macd_fast, macd_slow, macd_signal = 8, 17, 9
        stochrsi_length = 7
        stoch_k = 7
        bb_length = 10
        adx_length = 7
        atr_length = 7
    elif timeframe == "h4":
        # 4H: standard periods
        rsi_length = 14
        macd_fast, macd_slow, macd_signal = 12, 26, 9
        stochrsi_length = 14
        stoch_k = 14
        bb_length = 20
        adx_length = 14
        atr_length = 14
    else:
        # Daily: standard periods (default)
        rsi_length = 14
        macd_fast, macd_slow, macd_signal = 12, 26, 9
        stochrsi_length = 14
        stoch_k = 14
        bb_length = 20
        adx_length = 14
        atr_length = 14

    # === MOVING AVERAGES (same for all timeframes) ===
    df["EMA_9"] = ta.ema(df["close"], length=9)
    df["EMA_21"] = ta.ema(df["close"], length=21)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)

    # === RSI (adjusted by timeframe) ===
    df["RSI"] = ta.rsi(df["close"], length=rsi_length)
    # RSI Smoothed (EMA of RSI) - ลด noise
    df["RSI_smoothed"] = ta.ema(df["RSI"], length=3)

    # === MACD (adjusted by timeframe) ===
    macd = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    df["MACD"] = macd[macd_col]
    df["MACD_signal"] = macd[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
    df["MACD_histogram"] = macd[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]
    # MACD Histogram change - ตรวจจับ momentum shift
    df["MACD_hist_change"] = df["MACD_histogram"] - df["MACD_histogram"].shift(1)

    # === Stochastic RSI (adjusted by timeframe) ===
    stochrsi = ta.stochrsi(df["close"], length=stochrsi_length, rsi_length=stochrsi_length, k=3, d=3)
    stochrsi_col = f"STOCHRSIk_{stochrsi_length}_{stochrsi_length}_3_3"
    df["STOCHRSI_K"] = stochrsi[stochrsi_col]
    df["STOCHRSI_D"] = stochrsi[f"STOCHRSId_{stochrsi_length}_{stochrsi_length}_3_3"]

    # === Stochastic (adjusted by timeframe) ===
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=stoch_k, d=3)
    df["STOCH_K"] = stoch[f"STOCHk_{stoch_k}_3_3"]
    df["STOCH_D"] = stoch[f"STOCHd_{stoch_k}_3_3"]

    # === Bollinger Bands (adjusted by timeframe) ===
    bbands = ta.bbands(df["close"], length=bb_length, std=2.0)  # type: ignore[arg-type]
    df["BB_upper"] = bbands[f"BBU_{bb_length}_2.0_2.0"]
    df["BB_middle"] = bbands[f"BBM_{bb_length}_2.0_2.0"]
    df["BB_lower"] = bbands[f"BBL_{bb_length}_2.0_2.0"]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"] * 100
    # ป้องกัน division by zero
    bb_range = df["BB_upper"] - df["BB_lower"]
    df["BB_percent"] = np.where(
        bb_range > 0,
        (df["close"] - df["BB_lower"]) / bb_range,
        0.5  # กลางกรณี bands แคบมาก
    )

    # === ADX (Trend Strength) - adjusted by timeframe ===
    adx = ta.adx(df["high"], df["low"], df["close"], length=adx_length)
    df["ADX"] = adx[f"ADX_{adx_length}"]
    df["DI_plus"] = adx[f"DMP_{adx_length}"]
    df["DI_minus"] = adx[f"DMN_{adx_length}"]

    # === ATR - adjusted by timeframe ===
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=atr_length)
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
    # Rolling VWAP (20 periods) - แม่นยำกว่า cumulative สำหรับ swing trading
    typical_price_vwap = (df["high"] + df["low"] + df["close"]) / 3
    df["VWAP"] = (
        (typical_price_vwap * df["volume"]).rolling(window=20).sum() /
        df["volume"].rolling(window=20).sum()
    )
    # VWAP Bands (standard deviation) สำหรับหา overbought/oversold
    vwap_std = typical_price_vwap.rolling(window=20).std()
    df["VWAP_upper"] = df["VWAP"] + (vwap_std * 2)
    df["VWAP_lower"] = df["VWAP"] - (vwap_std * 2)

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
    # ป้องกัน division by zero
    df["BODY_PERCENT"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"] * 100,
        0
    )
    df["IS_BULLISH"] = df["close"] > df["open"]
    df["IS_BEARISH"] = df["close"] < df["open"]

    # === Momentum ===
    df["ROC"] = ta.roc(df["close"], length=10)
    df["MOM"] = ta.mom(df["close"], length=10)

    # === Keltner Channel (สำหรับ Squeeze Detection) ===
    kc_ema = ta.ema(df["close"], length=20)
    kc_atr = ta.atr(df["high"], df["low"], df["close"], length=10)
    df["KC_upper"] = kc_ema + (kc_atr * 1.5)
    df["KC_lower"] = kc_ema - (kc_atr * 1.5)
    # Squeeze Detection: BB inside KC = low volatility, potential breakout
    df["SQUEEZE"] = (df["BB_lower"] > df["KC_lower"]) & (df["BB_upper"] < df["KC_upper"])
    df["SQUEEZE_OFF"] = ~df["SQUEEZE"]

    # === CMF (Chaikin Money Flow) - Volume confirmation ===
    mf_multiplier = np.where(
        (df["high"] - df["low"]) > 0,
        ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]),
        0
    )
    mf_volume = pd.Series(mf_multiplier, index=df.index) * df["volume"]
    df["CMF"] = mf_volume.rolling(window=20).sum() / df["volume"].rolling(window=20).sum()

    # === Volume Profile Approximation (VAH, VAL) ===
    # Value Area High/Low - price zones with most trading activity
    df["VAH"] = df["high"].rolling(window=20).apply(
        lambda x: np.percentile(x, 70), raw=True
    )  # Value Area High (70th percentile)
    df["VAL"] = df["low"].rolling(window=20).apply(
        lambda x: np.percentile(x, 30), raw=True
    )  # Value Area Low (30th percentile)

    # === True Strength Index (TSI) - Better momentum indicator ===
    price_change = df["close"].diff()
    double_smoothed_pc = ta.ema(ta.ema(price_change, length=25), length=13)
    double_smoothed_abs_pc = ta.ema(ta.ema(abs(price_change), length=25), length=13)
    df["TSI"] = np.where(
        double_smoothed_abs_pc != 0,
        100 * (double_smoothed_pc / double_smoothed_abs_pc),
        0
    )
    df["TSI_signal"] = ta.ema(df["TSI"], length=7)

    # === Candlestick Patterns ===
    df = detect_candlestick_patterns(df)

    return df
