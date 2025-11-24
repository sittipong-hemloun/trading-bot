"""
DeepSeek AI Analyzer Module
ใช้ DeepSeek API วิเคราะห์ข้อมูล Bitcoin และให้คำแนะนำ
"""

import os
import json
from datetime import datetime
from typing import Optional
import requests
import pandas as pd


class DeepSeekAnalyzer:
    """วิเคราะห์ข้อมูล Bitcoin ด้วย DeepSeek AI"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = "deepseek-reasoner"

    def is_configured(self) -> bool:
        """ตรวจสอบว่ามี API key หรือไม่"""
        return bool(self.api_key)

    def fetch_binance_data(
        self, symbol: str = "BTCUSDT", timeframe: str = "4h", limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """ดึงข้อมูลจาก Binance API โดยตรง"""
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": timeframe, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
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
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(float)
            df["trades"] = df["trades"].astype(int)

            return df

        except Exception as e:
            print(f"❌ Error fetching Binance data: {e}")
            return None

    def fetch_multi_timeframe_data(self, symbol: str = "BTCUSDT") -> dict:
        """ดึงข้อมูลหลาย timeframe จาก Binance"""
        timeframes = {
            "1h": 50,  # 50 ชั่วโมง
            "4h": 50,  # 200 ชั่วโมง (~8 วัน)
            "1d": 30,  # 30 วัน
            "1w": 12,  # 12 สัปดาห์ (~3 เดือน)
        }

        data = {}
        for tf, limit in timeframes.items():
            df = self.fetch_binance_data(symbol, tf, limit)
            if df is not None:
                data[tf] = df
        return data

    def calculate_basic_indicators(self, df: pd.DataFrame) -> dict:
        """คำนวณ indicators พื้นฐาน"""
        if df is None or df.empty:
            return {}

        indicators = {}

        # Current price
        indicators["current_price"] = float(df["close"].iloc[-1])
        indicators["timestamp"] = df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M")

        # Price change
        if len(df) > 1:
            indicators["price_change_24h"] = round(
                (df["close"].iloc[-1] - df["close"].iloc[-2])
                / df["close"].iloc[-2]
                * 100,
                2,
            )

        # High/Low
        indicators["high_24h"] = float(df["high"].iloc[-1])
        indicators["low_24h"] = float(df["low"].iloc[-1])

        # Volume
        indicators["volume"] = float(df["volume"].iloc[-1])
        indicators["avg_volume"] = float(df["volume"].mean())

        # Simple Moving Averages
        if len(df) >= 20:
            indicators["SMA_20"] = round(float(df["close"].tail(20).mean()), 2)
        if len(df) >= 50:
            indicators["SMA_50"] = round(float(df["close"].tail(50).mean()), 2)

        # EMA
        if len(df) >= 21:
            indicators["EMA_9"] = round(
                float(df["close"].ewm(span=9).mean().iloc[-1]), 2
            )
            indicators["EMA_21"] = round(
                float(df["close"].ewm(span=21).mean().iloc[-1]), 2
            )

        # RSI
        if len(df) >= 14:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()  # type: ignore
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # type: ignore
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators["RSI"] = round(float(rsi.iloc[-1]), 2)

        # MACD
        if len(df) >= 26:
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            indicators["MACD"] = round(float(macd.iloc[-1]), 2)
            indicators["MACD_signal"] = round(float(signal.iloc[-1]), 2)
            indicators["MACD_histogram"] = round(
                float(macd.iloc[-1] - signal.iloc[-1]), 2
            )

        # Bollinger Bands
        if len(df) >= 20:
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            indicators["BB_upper"] = round(
                float(sma20.iloc[-1] + 2 * std20.iloc[-1]), 2
            )
            indicators["BB_middle"] = round(float(sma20.iloc[-1]), 2)
            indicators["BB_lower"] = round(
                float(sma20.iloc[-1] - 2 * std20.iloc[-1]), 2
            )

        # ATR (Average True Range)
        if len(df) >= 14:
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            indicators["ATR"] = round(float(atr.iloc[-1]), 2)
            indicators["ATR_percent"] = round(
                float(atr.iloc[-1] / df["close"].iloc[-1] * 100), 2
            )

        # Support/Resistance (simple pivots)
        if len(df) >= 5:
            recent = df.tail(20)
            indicators["recent_high"] = float(recent["high"].max())
            indicators["recent_low"] = float(recent["low"].min())

        # Trend detection
        if len(df) >= 21:
            ema9 = df["close"].ewm(span=9).mean().iloc[-1]
            ema21 = df["close"].ewm(span=21).mean().iloc[-1]
            price = df["close"].iloc[-1]
            if price > ema9 > ema21:
                indicators["trend"] = "UPTREND"
            elif price < ema9 < ema21:
                indicators["trend"] = "DOWNTREND"
            else:
                indicators["trend"] = "SIDEWAYS"

        return indicators

    def prepare_standalone_market_data(self, symbol: str = "BTCUSDT") -> dict:
        """เตรียมข้อมูลตลาดจาก Binance โดยตรง (ไม่พึ่ง Strategy)"""
        print("   📊 Fetching data from Binance API...")
        multi_tf_data = self.fetch_multi_timeframe_data(symbol)

        if not multi_tf_data:
            return {}

        market_data = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timeframes": {},
        }

        for tf, df in multi_tf_data.items():
            indicators = self.calculate_basic_indicators(df)
            if indicators:
                market_data["timeframes"][tf] = indicators

                # Get OHLCV summary
                market_data["timeframes"][tf]["ohlcv_last_5"] = []
                for i in range(-5, 0):
                    if len(df) >= abs(i):
                        row = df.iloc[i]
                        market_data["timeframes"][tf]["ohlcv_last_5"].append(
                            {
                                "time": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                                "open": round(float(row["open"]), 2),
                                "high": round(float(row["high"]), 2),
                                "low": round(float(row["low"]), 2),
                                "close": round(float(row["close"]), 2),
                                "volume": round(float(row["volume"]), 2),
                            }
                        )

        # Get current price from 1h data
        if "1h" in market_data["timeframes"]:
            market_data["current_price"] = market_data["timeframes"]["1h"].get(
                "current_price", 0
            )

        return market_data

    def create_standalone_prompt(self, market_data: dict) -> str:
        """สร้าง prompt สำหรับ DeepSeek วิเคราะห์ข้อมูลจาก Binance โดยตรง"""
        prompt = f"""คุณเป็นนักวิเคราะห์ Crypto ระดับมืออาชีพ กรุณาวิเคราะห์ข้อมูล {market_data.get('symbol', 'BTCUSDT')} ต่อไปนี้:

## ข้อมูลตลาดจาก Binance API ณ {market_data.get('timestamp', 'N/A')}
## ราคาปัจจุบัน: ${market_data.get('current_price', 0):,.2f}

### ข้อมูล Multi-Timeframe:
{json.dumps(market_data.get('timeframes', {}), indent=2, ensure_ascii=False)}

---

กรุณาวิเคราะห์ข้อมูลด้านบนและให้คำแนะนำในรูปแบบต่อไปนี้:

## 1. สรุปสถานการณ์ตลาด (Market Overview)
- ราคาปัจจุบันและแนวโน้ม
- สภาพตลาดโดยรวม (Bullish/Bearish/Sideways)
- ระดับความผันผวน (ATR%)

## 2. การวิเคราะห์ทางเทคนิค (Technical Analysis)
- วิเคราะห์ RSI, MACD, EMA ในแต่ละ Timeframe
- สัญญาณที่สอดคล้องกัน/ขัดแย้งกัน
- Bollinger Bands position

## 3. ระดับราคาสำคัญ (Key Levels)
- Support levels ที่ควรจับตา
- Resistance levels ที่ควรจับตา
- Fibonacci levels (ถ้าเหมาะสม)

## 4. คำแนะนำการเทรด (Trading Recommendation)
### สำหรับ Swing Trade (2-10 วัน):
- แนะนำ: LONG / SHORT / WAIT
- ระดับความมั่นใจ (1-10)
- Entry Zone
- Stop Loss
- Take Profit (TP1, TP2, TP3)
- Risk/Reward Ratio

### สำหรับ Position Trade (1+ เดือน):
- แนะนำ: LONG / SHORT / WAIT
- มุมมองระยะยาว

## 5. ความเสี่ยง (Risk Assessment)
- ความเสี่ยงหลักที่ควรระวัง
- สถานการณ์ที่อาจทำให้การวิเคราะห์ผิดพลาด
- Position sizing แนะนำ (% of portfolio)

## 6. สรุป (Key Takeaways)
- 3-5 ประเด็นสำคัญ
- Action Items สำหรับนักเทรด

ตอบเป็นภาษาไทย ใช้ภาษาที่เข้าใจง่าย กระชับ ได้ใจความ
ให้ข้อมูลเชิงตัวเลขที่ชัดเจน (ราคา Entry, SL, TP)"""

        return prompt

    def analyze_standalone(self, symbol: str = "BTCUSDT") -> Optional[dict]:
        """วิเคราะห์ข้อมูลจาก Binance โดยตรงด้วย DeepSeek AI"""
        if not self.is_configured():
            print("❌ DeepSeek API key not configured")
            return None

        # Fetch market data from Binance
        market_data = self.prepare_standalone_market_data(symbol)
        if not market_data:
            print("❌ Failed to fetch market data from Binance")
            return None

        prompt = self.create_standalone_prompt(market_data)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "คุณเป็นนักวิเคราะห์ Cryptocurrency มืออาชีพ "
                        "ที่มีความเชี่ยวชาญด้าน Technical Analysis และ Risk Management. "
                        "คุณวิเคราะห์ข้อมูลจาก Binance API โดยตรงและให้คำแนะนำที่เป็นรูปธรรม",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 4000,
            }

            print("   🤖 Calling DeepSeek API...")
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=90
            )
            response.raise_for_status()

            result = response.json()
            analysis_text = result["choices"][0]["message"]["content"]

            return {
                "success": True,
                "analysis": analysis_text,
                "model": self.model,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tokens_used": result.get("usage", {}),
                "market_data": market_data,
            }

        except requests.exceptions.Timeout:
            print("❌ DeepSeek API timeout")
            return {"success": False, "error": "API timeout"}
        except requests.exceptions.RequestException as e:
            print(f"❌ DeepSeek API error: {e}")
            return {"success": False, "error": str(e)}
        except (KeyError, IndexError) as e:
            print(f"❌ DeepSeek response parsing error: {e}")
            return {"success": False, "error": f"Response parsing error: {e}"}

    def prepare_market_data(
        self,
        swing_data: dict,
        monthly_data: dict,
    ) -> dict:
        """เตรียมข้อมูลตลาดสำหรับส่งให้ DeepSeek วิเคราะห์"""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "swing_trading": swing_data,
            "monthly_trading": monthly_data,
        }

    def create_analysis_prompt(self, market_data: dict) -> str:
        """สร้าง prompt สำหรับให้ DeepSeek วิเคราะห์"""
        prompt = f"""คุณเป็นนักวิเคราะห์ Crypto ระดับมืออาชีพ กรุณาวิเคราะห์ข้อมูล Bitcoin ต่อไปนี้:

## ข้อมูลตลาด ณ {market_data['timestamp']}

### Swing Trading Analysis (2-10 วัน):
{json.dumps(market_data['swing_trading'], indent=2, ensure_ascii=False)}

### Monthly Trading Analysis:
{json.dumps(market_data['monthly_trading'], indent=2, ensure_ascii=False)}

---

กรุณาวิเคราะห์และให้ข้อมูลในรูปแบบต่อไปนี้:

## 1. สรุปสถานการณ์ตลาด (Market Overview)
- สภาพตลาดโดยรวม
- แนวโน้มหลัก (Trend)
- ระดับความผันผวน

## 2. การวิเคราะห์ทางเทคนิค (Technical Analysis)
- สัญญาณสำคัญที่พบ
- ระดับ Support/Resistance ที่ควรจับตา
- Indicators ที่น่าสนใจ

## 3. ความเสี่ยง (Risk Assessment)
- ความเสี่ยงหลักที่ควรระวัง
- ปัจจัยที่อาจกระทบราคา
- สถานการณ์ที่ควรหลีกเลี่ยง

## 4. คำแนะนำการเทรด (Trading Recommendation)
- แนะนำ: LONG / SHORT / WAIT
- ระดับความมั่นใจ (1-10)
- Entry Zone ที่แนะนำ
- Stop Loss ที่แนะนำ
- Take Profit Targets

## 5. มุมมองระยะสั้น vs ระยะยาว
- Swing Trade (2-10 วัน): ควรทำอย่างไร
- Position Trade (1+ เดือน): ควรทำอย่างไร

## 6. สรุป (Key Takeaways)
- 3 ประเด็นสำคัญที่ต้องจำ
- Action Items สำหรับนักเทรด

ตอบเป็นภาษาไทย ใช้ภาษาที่เข้าใจง่าย กระชับ ได้ใจความ"""

        return prompt

    def analyze(self, market_data: dict) -> Optional[dict]:
        """เรียก DeepSeek API เพื่อวิเคราะห์ข้อมูล"""
        if not self.is_configured():
            print("❌ DeepSeek API key not configured")
            return None

        prompt = self.create_analysis_prompt(market_data)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "คุณเป็นนักวิเคราะห์ Cryptocurrency มืออาชีพ "
                        "ที่มีความเชี่ยวชาญด้าน Technical Analysis และ Risk Management",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 3000,
            }

            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()

            result = response.json()
            analysis_text = result["choices"][0]["message"]["content"]

            return {
                "success": True,
                "analysis": analysis_text,
                "model": self.model,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tokens_used": result.get("usage", {}),
            }

        except requests.exceptions.Timeout:
            print("❌ DeepSeek API timeout")
            return {"success": False, "error": "API timeout"}
        except requests.exceptions.RequestException as e:
            print(f"❌ DeepSeek API error: {e}")
            return {"success": False, "error": str(e)}
        except (KeyError, IndexError) as e:
            print(f"❌ DeepSeek response parsing error: {e}")
            return {"success": False, "error": f"Response parsing error: {e}"}


def extract_trading_data(strategy, strategy_type: str = "swing") -> dict:
    """ดึงข้อมูลสำคัญจาก strategy object"""
    data = {}

    try:
        # ดึง timeframe data ที่เหมาะสมกับ strategy type
        if strategy_type == "swing":
            primary_tf = "h4"
        else:  # monthly
            primary_tf = "daily"

        if primary_tf in strategy.data and not strategy.data[primary_tf].empty:
            df = strategy.data[primary_tf]
            latest = df.iloc[-1]

            data["current_price"] = float(latest["close"])
            data["timestamp"] = latest["timestamp"].strftime("%Y-%m-%d %H:%M")

            # Price data
            data["price"] = {
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": float(latest["volume"]),
            }

            # Indicators
            data["indicators"] = {}

            # RSI
            if "RSI" in latest and not _is_nan(latest["RSI"]):
                data["indicators"]["RSI"] = round(float(latest["RSI"]), 2)

            # MACD
            if "MACD" in latest and not _is_nan(latest["MACD"]):
                data["indicators"]["MACD"] = {
                    "value": round(float(latest["MACD"]), 2),
                    "signal": (
                        round(float(latest["MACD_signal"]), 2)
                        if not _is_nan(latest.get("MACD_signal"))
                        else None
                    ),
                    "histogram": (
                        round(float(latest["MACD_histogram"]), 2)
                        if not _is_nan(latest.get("MACD_histogram"))
                        else None
                    ),
                }

            # EMA
            if "EMA_9" in latest and not _is_nan(latest["EMA_9"]):
                data["indicators"]["EMA"] = {
                    "EMA_9": round(float(latest["EMA_9"]), 2),
                    "EMA_21": (
                        round(float(latest["EMA_21"]), 2)
                        if not _is_nan(latest.get("EMA_21"))
                        else None
                    ),
                }

            # ADX
            if "ADX" in latest and not _is_nan(latest["ADX"]):
                data["indicators"]["ADX"] = {
                    "value": round(float(latest["ADX"]), 2),
                    "DI_plus": (
                        round(float(latest["DI_plus"]), 2)
                        if not _is_nan(latest.get("DI_plus"))
                        else None
                    ),
                    "DI_minus": (
                        round(float(latest["DI_minus"]), 2)
                        if not _is_nan(latest.get("DI_minus"))
                        else None
                    ),
                }

            # ATR
            if "ATR" in latest and not _is_nan(latest["ATR"]):
                data["indicators"]["ATR"] = {
                    "value": round(float(latest["ATR"]), 2),
                    "percent": (
                        round(float(latest["ATR_percent"]), 2)
                        if not _is_nan(latest.get("ATR_percent"))
                        else None
                    ),
                }

            # Bollinger Bands
            if "BB_upper" in latest and not _is_nan(latest["BB_upper"]):
                data["indicators"]["BB"] = {
                    "upper": round(float(latest["BB_upper"]), 2),
                    "middle": (
                        round(float(latest["BB_middle"]), 2)
                        if not _is_nan(latest.get("BB_middle"))
                        else None
                    ),
                    "lower": (
                        round(float(latest["BB_lower"]), 2)
                        if not _is_nan(latest.get("BB_lower"))
                        else None
                    ),
                }

            # Volume Ratio
            if "Volume_Ratio" in latest and not _is_nan(latest["Volume_Ratio"]):
                data["indicators"]["Volume_Ratio"] = round(
                    float(latest["Volume_Ratio"]), 2
                )

            # StochRSI
            if "STOCHRSI_K" in latest and not _is_nan(latest["STOCHRSI_K"]):
                data["indicators"]["StochRSI"] = {
                    "K": round(float(latest["STOCHRSI_K"]), 2),
                    "D": (
                        round(float(latest["STOCHRSI_D"]), 2)
                        if not _is_nan(latest.get("STOCHRSI_D"))
                        else None
                    ),
                }

            # Support/Resistance
            try:
                sr = strategy.calculate_support_resistance(df)
                data["support_resistance"] = {
                    "main_support": round(sr["main_support"], 2),
                    "main_resistance": round(sr["main_resistance"], 2),
                    "supports": [round(s, 2) for s in sr.get("support", [])[:3]],
                    "resistances": [round(r, 2) for r in sr.get("resistance", [])[:3]],
                }
            except Exception:
                pass

            # Market Regime
            try:
                regime = strategy.detect_market_regime(df)
                data["market_regime"] = {
                    "regime": regime["regime"],
                    "confidence": round(regime["confidence"], 1),
                    "adx": round(regime["adx"], 1),
                }
            except Exception:
                pass

            # Trend
            try:
                trend_score, _ = strategy.get_trend_strength(df)
                data["trend_score"] = trend_score
            except Exception:
                pass

        # Get signals if available
        try:
            signals, reasons = strategy.get_signal()
            total = signals["long"] + signals["short"] + signals["neutral"]
            data["signals"] = {
                "long": signals["long"],
                "short": signals["short"],
                "neutral": signals["neutral"],
                "long_pct": (
                    round((signals["long"] / total * 100), 1) if total > 0 else 0
                ),
                "short_pct": (
                    round((signals["short"] / total * 100), 1) if total > 0 else 0
                ),
            }
            # Include top reasons
            data["signal_reasons"] = {
                "long": reasons["long"][:5],
                "short": reasons["short"][:5],
                "neutral": reasons["neutral"][:3],
            }
        except Exception:
            pass

    except Exception as e:
        data["error"] = str(e)

    return data


def _is_nan(value) -> bool:
    """ตรวจสอบว่าเป็น NaN หรือไม่"""
    try:
        import math

        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True
