"""
Data Fetching Module
Contains functions for fetching market data from exchanges
"""

import requests
import pandas as pd
from typing import Optional


def fetch_binance_data(symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """ดึงข้อมูลจาก Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": timeframe, "limit": limit}

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
