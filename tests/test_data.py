"""
Unit tests for data fetching functionality
"""

import pytest
import pandas as pd
from unittest.mock import patch

from trading.weekly_strategy import WeeklyTradingStrategy


@pytest.fixture
def mock_binance_response():
    """Create mock Binance API response"""
    n = 100
    base_price = 50000
    data = []

    for i in range(n):
        timestamp = 1704067200000 + (i * 86400000)  # Daily timestamps
        price = base_price + (i * 100)
        data.append([
            timestamp,
            str(price),  # open
            str(price * 1.02),  # high
            str(price * 0.98),  # low
            str(price * 1.01),  # close
            str(1000 + i * 10),  # volume
            timestamp + 86400000,  # close_time
            str(50000000),  # quote_volume
            str(1000),  # trades
            str(500),  # taker_buy_base
            str(25000000),  # taker_buy_quote
            "0",  # ignore
        ])

    return data


class TestDataFetching:
    """Test data fetching functionality"""

    @pytest.fixture
    def strategy(self):
        """Create a WeeklyTradingStrategy instance"""
        return WeeklyTradingStrategy(symbol="BTCUSDT", leverage=5)

    def test_fetch_data_returns_dataframe(self, strategy, mock_binance_response):
        """Test that fetch_data returns a valid DataFrame"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 100

    def test_fetch_data_has_required_columns(self, strategy, mock_binance_response):
        """Test that fetched data has all required columns"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"

    def test_fetch_data_no_null_values_in_ohlcv(self, strategy, mock_binance_response):
        """Test that OHLCV data has no null values"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_columns:
                assert df[col].isna().sum() == 0, f"Column {col} has null values"

    def test_fetch_data_no_zero_prices(self, strategy, mock_binance_response):
        """Test that price data has no zero values"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                assert (df[col] == 0).sum() == 0, f"Column {col} has zero values"
                assert (df[col] > 0).all(), f"Column {col} should have positive values"

    def test_fetch_data_no_negative_volume(self, strategy, mock_binance_response):
        """Test that volume has no negative values"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            assert (df["volume"] >= 0).all(), "Volume should not be negative"

    def test_fetch_data_high_greater_than_low(self, strategy, mock_binance_response):
        """Test that high is always >= low"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            assert (df["high"] >= df["low"]).all(), "High should always be >= Low"

    def test_fetch_data_handles_api_error(self, strategy):
        """Test that fetch_data handles API errors gracefully"""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("API Error")

            df = strategy.fetch_data("1d", limit=100)

            assert df is None

    def test_fetch_data_numeric_types(self, strategy, mock_binance_response):
        """Test that OHLCV columns are numeric types"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)

            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
