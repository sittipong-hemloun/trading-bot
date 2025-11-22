"""
Market Analysis Module
Contains functions for market analysis, divergence detection, and regime detection
"""

import pandas as pd
import numpy as np


def get_multi_indicator_confirmation(df: pd.DataFrame) -> dict:
    """ตรวจสอบการยืนยันจากหลาย Indicators พร้อมกัน"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_confirmations = 0
    bearish_confirmations = 0
    confirmation_details = {"bullish": [], "bearish": []}

    # 1. Trend Confirmation (EMA alignment)
    if latest["EMA_9"] > latest["EMA_21"] > latest["EMA_50"]:
        bullish_confirmations += 1
        confirmation_details["bullish"].append("EMA Aligned Bullish")
    elif latest["EMA_9"] < latest["EMA_21"] < latest["EMA_50"]:
        bearish_confirmations += 1
        confirmation_details["bearish"].append("EMA Aligned Bearish")

    # 2. Momentum Confirmation (MACD + RSI)
    macd_bullish = latest["MACD"] > latest["MACD_signal"] and latest["MACD_histogram"] > 0
    macd_bearish = latest["MACD"] < latest["MACD_signal"] and latest["MACD_histogram"] < 0
    rsi_bullish = 30 < latest["RSI"] < 70 and latest["RSI"] > prev["RSI"]
    rsi_bearish = 30 < latest["RSI"] < 70 and latest["RSI"] < prev["RSI"]

    if macd_bullish and rsi_bullish:
        bullish_confirmations += 1
        confirmation_details["bullish"].append("MACD+RSI Momentum Bullish")
    elif macd_bearish and rsi_bearish:
        bearish_confirmations += 1
        confirmation_details["bearish"].append("MACD+RSI Momentum Bearish")

    # 3. Trend Strength Confirmation (ADX + DI)
    if pd.notna(latest.get("ADX")) and latest["ADX"] > 25:
        if latest["DI_plus"] > latest["DI_minus"]:
            bullish_confirmations += 1
            confirmation_details["bullish"].append(f"ADX Strong Uptrend ({latest['ADX']:.0f})")
        else:
            bearish_confirmations += 1
            confirmation_details["bearish"].append(f"ADX Strong Downtrend ({latest['ADX']:.0f})")

    # 4. Supertrend Confirmation
    if pd.notna(latest.get("SUPERTREND_DIR")):
        if latest["SUPERTREND_DIR"] == 1:
            bullish_confirmations += 1
            confirmation_details["bullish"].append("Supertrend Bullish")
        else:
            bearish_confirmations += 1
            confirmation_details["bearish"].append("Supertrend Bearish")

    # 5. Price vs Cloud (Ichimoku)
    if pd.notna(latest.get("ICHI_SENKOU_A")) and pd.notna(latest.get("ICHI_SENKOU_B")):
        cloud_top = max(latest["ICHI_SENKOU_A"], latest["ICHI_SENKOU_B"])
        cloud_bottom = min(latest["ICHI_SENKOU_A"], latest["ICHI_SENKOU_B"])
        if latest["close"] > cloud_top:
            bullish_confirmations += 1
            confirmation_details["bullish"].append("Above Ichimoku Cloud")
        elif latest["close"] < cloud_bottom:
            bearish_confirmations += 1
            confirmation_details["bearish"].append("Below Ichimoku Cloud")

    # 6. Bollinger Band Position
    if pd.notna(latest.get("BB_percent")):
        if latest["BB_percent"] < 0.2:  # Near lower band
            bullish_confirmations += 1
            confirmation_details["bullish"].append("Near BB Lower (Oversold)")
        elif latest["BB_percent"] > 0.8:  # Near upper band
            bearish_confirmations += 1
            confirmation_details["bearish"].append("Near BB Upper (Overbought)")

    total = bullish_confirmations + bearish_confirmations
    if total == 0:
        return {"direction": "neutral", "strength": 0, "confirmations": 0, "details": []}

    if bullish_confirmations > bearish_confirmations:
        return {
            "direction": "bullish",
            "strength": bullish_confirmations / 6 * 100,
            "confirmations": bullish_confirmations,
            "details": confirmation_details["bullish"]
        }
    elif bearish_confirmations > bullish_confirmations:
        return {
            "direction": "bearish",
            "strength": bearish_confirmations / 6 * 100,
            "confirmations": bearish_confirmations,
            "details": confirmation_details["bearish"]
        }
    else:
        return {"direction": "mixed", "strength": 50, "confirmations": total, "details": []}


def get_volume_confirmation(df: pd.DataFrame, lookback: int = 20) -> dict:
    """วิเคราะห์ Volume เพื่อยืนยันสัญญาณ"""
    latest = df.iloc[-1]
    recent = df.tail(lookback)

    result = {
        "confirmed": False,
        "volume_trend": "neutral",
        "volume_ratio": latest["Volume_Ratio"] if pd.notna(latest.get("Volume_Ratio")) else 1,
        "obv_trend": "neutral",
        "details": []
    }

    # 1. Volume Breakout
    if result["volume_ratio"] > 2.0:
        result["details"].append(f"High Volume Breakout ({result['volume_ratio']:.1f}x)")
        result["confirmed"] = True
    elif result["volume_ratio"] > 1.5:
        result["details"].append(f"Above Average Volume ({result['volume_ratio']:.1f}x)")
        result["confirmed"] = True

    # 2. Volume Trend (increasing or decreasing)
    vol_sma_5 = recent["volume"].tail(5).mean()
    vol_sma_10 = recent["volume"].tail(10).mean()
    if vol_sma_5 > vol_sma_10 * 1.1:
        result["volume_trend"] = "increasing"
        result["details"].append("Volume Trend Increasing")
    elif vol_sma_5 < vol_sma_10 * 0.9:
        result["volume_trend"] = "decreasing"
        result["details"].append("Volume Trend Decreasing")

    # 3. OBV Analysis
    if pd.notna(latest.get("OBV")) and pd.notna(latest.get("OBV_EMA")):
        if latest["OBV"] > latest["OBV_EMA"]:
            result["obv_trend"] = "bullish"
            result["details"].append("OBV Above Average (Accumulation)")
        else:
            result["obv_trend"] = "bearish"
            result["details"].append("OBV Below Average (Distribution)")

    # 4. Price-Volume Divergence
    price_up = latest["close"] > df.iloc[-2]["close"]
    volume_up = latest["volume"] > df.iloc[-2]["volume"]

    if price_up and not volume_up:
        result["details"].append("Warning: Price Up on Lower Volume")
    elif not price_up and volume_up:
        result["details"].append("Warning: Price Down on Higher Volume")

    # 5. MFI Analysis
    if pd.notna(latest.get("MFI")):
        if latest["MFI"] < 20:
            result["details"].append(f"MFI Oversold ({latest['MFI']:.0f})")
        elif latest["MFI"] > 80:
            result["details"].append(f"MFI Overbought ({latest['MFI']:.0f})")

    return result


def find_confluence_zones(df: pd.DataFrame, current_price: float) -> dict:
    """หา Confluence Zones - บริเวณที่มีหลาย levels รวมกัน"""
    zones = {"support": [], "resistance": []}
    tolerance = current_price * 0.02  # 2% tolerance

    # Collect all significant levels
    levels = []

    latest = df.iloc[-1]

    # EMAs
    for ema_col in ["EMA_9", "EMA_21", "EMA_50"]:
        if pd.notna(latest.get(ema_col)):
            levels.append({"price": latest[ema_col], "type": "EMA", "name": ema_col})

    # SMAs
    for sma_col in ["SMA_50", "SMA_200"]:
        if pd.notna(latest.get(sma_col)):
            levels.append({"price": latest[sma_col], "type": "SMA", "name": sma_col})

    # Bollinger Bands
    for bb_col in ["BB_upper", "BB_middle", "BB_lower"]:
        if pd.notna(latest.get(bb_col)):
            levels.append({"price": latest[bb_col], "type": "BB", "name": bb_col})

    # Pivot Points
    for pivot_col in ["PIVOT", "R1", "R2", "S1", "S2"]:
        if pd.notna(latest.get(pivot_col)):
            levels.append({"price": latest[pivot_col], "type": "Pivot", "name": pivot_col})

    # VWAP
    if pd.notna(latest.get("VWAP")):
        levels.append({"price": latest["VWAP"], "type": "VWAP", "name": "VWAP"})

    # Supertrend
    if pd.notna(latest.get("SUPERTREND")):
        levels.append({"price": latest["SUPERTREND"], "type": "Supertrend", "name": "Supertrend"})

    # Ichimoku levels
    for ichi_col in ["ICHI_TENKAN", "ICHI_KIJUN", "ICHI_SENKOU_A", "ICHI_SENKOU_B"]:
        if pd.notna(latest.get(ichi_col)):
            levels.append({"price": latest[ichi_col], "type": "Ichimoku", "name": ichi_col})

    # Find confluence - levels that are close to each other
    confluence_zones = []

    for i, level1 in enumerate(levels):
        confluence_count = 1
        confluence_levels = [level1["name"]]
        avg_price = level1["price"]

        for j, level2 in enumerate(levels):
            if i != j and abs(level1["price"] - level2["price"]) < tolerance:
                confluence_count += 1
                confluence_levels.append(level2["name"])
                avg_price = (avg_price + level2["price"]) / 2

        if confluence_count >= 2:  # At least 2 levels together
            confluence_zones.append({
                "price": avg_price,
                "strength": confluence_count,
                "levels": confluence_levels
            })

    # Remove duplicates and sort
    seen_prices = set()
    for zone in sorted(confluence_zones, key=lambda x: -x["strength"]):
        price_key = round(zone["price"] / tolerance) * tolerance
        if price_key not in seen_prices:
            seen_prices.add(price_key)
            if zone["price"] < current_price:
                zones["support"].append(zone)
            else:
                zones["resistance"].append(zone)

    # Sort by distance from current price
    zones["support"] = sorted(zones["support"], key=lambda x: -x["price"])[:3]
    zones["resistance"] = sorted(zones["resistance"], key=lambda x: x["price"])[:3]

    return zones


def get_dynamic_thresholds(df: pd.DataFrame) -> dict:
    """คำนวณ Thresholds แบบ Dynamic ตาม Volatility และ Market Regime"""
    latest = df.iloc[-1]
    lookback = min(20, len(df) - 1)

    # Current volatility
    current_atr_pct = latest["ATR_percent"] if pd.notna(latest.get("ATR_percent")) else 3
    avg_atr_pct = df["ATR_percent"].tail(lookback).mean() if "ATR_percent" in df.columns else 3
    volatility_ratio = current_atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1

    # Base thresholds
    base_rsi_oversold = 30
    base_rsi_overbought = 70
    base_volume_threshold = 1.5

    # Adjust based on volatility
    if volatility_ratio > 1.5:  # High volatility
        rsi_oversold = base_rsi_oversold - 5  # More extreme = 25
        rsi_overbought = base_rsi_overbought + 5  # = 75
        volume_threshold = base_volume_threshold + 0.5  # = 2.0
    elif volatility_ratio < 0.7:  # Low volatility
        rsi_oversold = base_rsi_oversold + 5  # = 35
        rsi_overbought = base_rsi_overbought - 5  # = 65
        volume_threshold = base_volume_threshold - 0.3  # = 1.2
    else:
        rsi_oversold = base_rsi_oversold
        rsi_overbought = base_rsi_overbought
        volume_threshold = base_volume_threshold

    return {
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "volume_threshold": volume_threshold,
        "volatility_ratio": volatility_ratio,
        "sl_multiplier": 1.5 + (volatility_ratio - 1) * 0.5,  # Wider SL in high vol
        "tp_multiplier": 2.0 + (volatility_ratio - 1) * 0.5,  # Wider TP in high vol
    }


def check_divergence(df: pd.DataFrame, indicator: str = "RSI", lookback: int = 14) -> tuple:
    """ตรวจสอบ Divergence ระหว่างราคาและ Indicator โดยใช้ Swing Points"""
    if len(df) < lookback:
        return None, 0

    recent_df = df.tail(lookback)
    prices = recent_df["close"].values
    indicator_values = recent_df[indicator].values

    # หา Swing Points (Local Highs/Lows)
    price_highs = []
    price_lows = []
    ind_highs = []
    ind_lows = []

    for i in range(1, len(prices) - 1):
        # Swing High
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            price_highs.append((i, prices[i]))
            ind_highs.append((i, indicator_values[i]))
        # Swing Low
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            price_lows.append((i, prices[i]))
            ind_lows.append((i, indicator_values[i]))

    # Check for Bullish Divergence (price lower lows, indicator higher lows)
    if len(price_lows) >= 2:
        last_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1]
        last_ind_low = ind_lows[-1][1]
        prev_ind_low = ind_lows[-2][1]

        if last_price_low < prev_price_low and last_ind_low > prev_ind_low:
            strength = abs(last_ind_low - prev_ind_low) / max(prev_ind_low, 1) * 100
            return "bullish", min(strength, 100)

    # Check for Bearish Divergence (price higher highs, indicator lower highs)
    if len(price_highs) >= 2:
        last_price_high = price_highs[-1][1]
        prev_price_high = price_highs[-2][1]
        last_ind_high = ind_highs[-1][1]
        prev_ind_high = ind_highs[-2][1]

        if last_price_high > prev_price_high and last_ind_high < prev_ind_high:
            strength = abs(last_ind_high - prev_ind_high) / max(prev_ind_high, 1) * 100
            return "bearish", min(strength, 100)

    return None, 0


def detect_market_regime(df: pd.DataFrame) -> dict:
    """ตรวจจับสภาวะตลาดปัจจุบัน"""
    if len(df) < 20:
        return {"regime": "UNKNOWN", "confidence": 0, "adx": 0, "bb_width": 0, "price_range_pct": 0}

    latest = df.iloc[-1]
    recent = df.tail(20)

    # ADX for trend strength
    adx = latest["ADX"] if pd.notna(latest.get("ADX")) else 20

    # BB Width for volatility
    bb_width = latest["BB_width"] if pd.notna(latest.get("BB_width")) else 5

    # Price range over period
    price_high = recent["high"].max()
    price_low = recent["low"].min()
    price_range_pct = (price_high - price_low) / price_low * 100

    # DI for trend direction
    di_plus = latest["DI_plus"] if pd.notna(latest.get("DI_plus")) else 20
    di_minus = latest["DI_minus"] if pd.notna(latest.get("DI_minus")) else 20

    # Determine regime
    if adx > 40:
        if di_plus > di_minus:
            regime = "STRONG_UPTREND"
            confidence = min(90, adx + 20)
        else:
            regime = "STRONG_DOWNTREND"
            confidence = min(90, adx + 20)
    elif adx > 25:
        if bb_width > 8:  # High volatility
            regime = "HIGH_VOLATILITY"
            confidence = min(80, adx + bb_width)
        elif di_plus > di_minus:
            regime = "WEAK_TREND"
            confidence = min(70, adx + 30)
        else:
            regime = "WEAK_TREND"
            confidence = min(70, adx + 30)
    else:  # ADX < 25
        if bb_width < 3:
            regime = "CONSOLIDATION"
            confidence = min(80, 100 - adx)
        else:
            regime = "RANGING"
            confidence = min(60, 100 - adx)

    return {
        "regime": regime,
        "confidence": confidence,
        "adx": adx,
        "bb_width": bb_width,
        "price_range_pct": price_range_pct
    }


def analyze_historical_performance(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    วิเคราะห์ Performance ย้อนหลังของสัญญาณแบบ Multi-Indicator

    Entry Conditions (Long):
    - RSI < 35 AND RSI rising (confirmation)
    - MACD histogram turning positive OR
    - Price above EMA9

    Exit Conditions:
    - RSI > 65 OR
    - MACD histogram turning negative OR
    - Stop loss: -3% OR Take profit: +5%
    """
    if len(df) < lookback:
        lookback = len(df) - 1

    if lookback < 15:
        return {
            "total_signals": 0, "win_rate": 0, "avg_return": 0,
            "max_drawdown": 0, "sharpe": 0, "profit_factor": 0,
            "long_signals": 0, "short_signals": 0
        }

    recent_df = df.tail(lookback).copy()
    recent_df = recent_df.reset_index(drop=True)

    trades = []
    position = None

    for i in range(2, len(recent_df)):
        row = recent_df.iloc[i]
        prev_row = recent_df.iloc[i - 1]

        # Skip if essential indicators are missing
        if pd.isna(row.get("RSI")) or pd.isna(row.get("MACD_histogram")):
            continue

        rsi = row["RSI"]
        rsi_prev = prev_row["RSI"]
        rsi_rising = rsi > rsi_prev
        rsi_falling = rsi < rsi_prev

        macd_hist = row["MACD_histogram"]
        macd_hist_prev = prev_row["MACD_histogram"]
        macd_turning_up = macd_hist > macd_hist_prev
        macd_turning_down = macd_hist < macd_hist_prev

        ema_bullish = row["close"] > row["EMA_9"] > row["EMA_21"]
        ema_bearish = row["close"] < row["EMA_9"] < row["EMA_21"]

        # ADX for trend strength
        adx = row.get("ADX", 20)
        strong_trend = adx > 25

        if position is None:
            # === LONG ENTRY ===
            # Condition: RSI oversold + rising + (MACD turning up OR EMA bullish)
            long_signal = (
                rsi < 35 and
                rsi_rising and
                (macd_turning_up or ema_bullish)
            )

            # === SHORT ENTRY ===
            # Condition: RSI overbought + falling + (MACD turning down OR EMA bearish)
            short_signal = (
                rsi > 65 and
                rsi_falling and
                (macd_turning_down or ema_bearish)
            )

            if long_signal:
                position = {
                    "entry": row["close"],
                    "type": "long",
                    "entry_idx": i,
                    "stop_loss": row["close"] * 0.97,  # -3%
                    "take_profit": row["close"] * 1.05  # +5%
                }
            elif short_signal and strong_trend:
                position = {
                    "entry": row["close"],
                    "type": "short",
                    "entry_idx": i,
                    "stop_loss": row["close"] * 1.03,  # +3%
                    "take_profit": row["close"] * 0.95  # -5%
                }

        else:
            # === EXIT CONDITIONS ===
            current_price = row["close"]
            entry_price = position["entry"]

            if position["type"] == "long":
                # Long exit conditions
                pnl_pct = (current_price - entry_price) / entry_price * 100

                exit_signal = (
                    current_price <= position["stop_loss"] or  # Stop loss
                    current_price >= position["take_profit"] or  # Take profit
                    (rsi > 65 and rsi_falling) or  # RSI overbought + falling
                    (macd_turning_down and macd_hist < 0)  # MACD bearish
                )

                if exit_signal:
                    trades.append({
                        "type": "long",
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl_pct": pnl_pct,
                        "bars_held": i - position["entry_idx"]
                    })
                    position = None

            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price * 100

                exit_signal = (
                    current_price >= position["stop_loss"] or  # Stop loss
                    current_price <= position["take_profit"] or  # Take profit
                    (rsi < 35 and rsi_rising) or  # RSI oversold + rising
                    (macd_turning_up and macd_hist > 0)  # MACD bullish
                )

                if exit_signal:
                    trades.append({
                        "type": "short",
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl_pct": pnl_pct,
                        "bars_held": i - position["entry_idx"]
                    })
                    position = None

    # Calculate statistics
    if not trades:
        return {
            "total_signals": 0, "win_rate": 0, "avg_return": 0,
            "max_drawdown": 0, "sharpe": 0, "profit_factor": 0,
            "long_signals": 0, "short_signals": 0
        }

    returns = [t["pnl_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_return = np.mean(returns) if returns else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Sharpe ratio (simplified, assuming 0 risk-free rate)
    sharpe = avg_return / np.std(returns) if np.std(returns) > 0 else 0

    # Count by type
    long_trades = [t for t in trades if t["type"] == "long"]
    short_trades = [t for t in trades if t["type"] == "short"]

    return {
        "total_signals": len(trades),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "long_signals": len(long_trades),
        "short_signals": len(short_trades),
        "avg_bars_held": np.mean([t["bars_held"] for t in trades]) if trades else 0,
        "best_trade": max(returns) if returns else 0,
        "worst_trade": min(returns) if returns else 0
    }
