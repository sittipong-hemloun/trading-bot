"""
Technical Indicators Module
Contains functions for calculating technical indicators
"""

import pandas as pd
import pandas_ta as ta


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณตัวชี้วัดแบบครบถ้วน"""
    from trading.patterns import detect_candlestick_patterns

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

    # === Candlestick Patterns ===
    df = detect_candlestick_patterns(df)

    return df
