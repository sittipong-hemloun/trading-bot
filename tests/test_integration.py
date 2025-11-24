"""
Integration tests for full trading strategy workflow
"""

import pytest
import pandas as pd
import numpy as np
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


class TestFullWeeklyWorkflow:
    """Integration tests for complete weekly strategy workflow"""

    def test_full_weekly_workflow(self, mock_binance_response):
        """Test complete weekly strategy workflow"""
        strategy = WeeklyTradingStrategy(symbol="BTCUSDT", leverage=5)

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            # Fetch data
            df = strategy.fetch_data("1d", limit=100)
            assert df is not None

            # Calculate indicators
            df = strategy.calculate_indicators(df)
            assert "RSI" in df.columns

            # Check divergence
            div_type, strength = strategy.check_divergence(df, "RSI")
            assert div_type is None or div_type in ["bullish", "bearish"]

            # Detect market regime
            regime = strategy.detect_market_regime(df)
            assert "regime" in regime

            # Calculate support/resistance
            sr = strategy.calculate_support_resistance(df)
            assert sr["main_support"] < sr["main_resistance"]

    def test_data_consistency_across_timeframes(self, mock_binance_response):
        """Test that data is consistent across different timeframes"""
        strategy = WeeklyTradingStrategy(symbol="BTCUSDT", leverage=5)

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            for tf in ["1d", "4h", "1w"]:
                df = strategy.fetch_data(tf, limit=50)
                assert df is not None
                # Mock returns 100 rows regardless of limit, just check we got data
                assert len(df) > 0
                assert (df["high"] >= df["low"]).all()

    def test_indicator_pipeline(self, mock_binance_response):
        """Test complete indicator calculation pipeline"""
        strategy = WeeklyTradingStrategy()

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)
            df = strategy.calculate_indicators(df)

            # Verify all major indicator groups are calculated
            assert "EMA_9" in df.columns
            assert "RSI" in df.columns
            assert "MACD" in df.columns
            assert "BB_upper" in df.columns
            assert "ADX" in df.columns
            assert "ATR" in df.columns
            assert "Volume_Ratio" in df.columns

    def test_signal_generation_pipeline(self, mock_binance_response):
        """Test complete signal generation pipeline"""
        strategy = WeeklyTradingStrategy()

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)
            df = strategy.calculate_indicators(df)

            # Test trend strength
            score, max_score = strategy.get_trend_strength(df)
            assert max_score > 0

            # Test multi-indicator confirmation
            confirmation = strategy.get_multi_indicator_confirmation(df)
            assert confirmation["direction"] in ["bullish", "bearish", "neutral"]

            # Test volume confirmation
            volume_conf = strategy.get_volume_confirmation(df)
            assert isinstance(volume_conf["confirmed"], bool)

    def test_risk_management_pipeline(self, mock_binance_response):
        """Test complete risk management pipeline"""
        strategy = WeeklyTradingStrategy()

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = strategy.fetch_data("1d", limit=100)
            df = strategy.calculate_indicators(df)

            # Test risk score
            risk = strategy.calculate_risk_score(df, "LONG")
            assert 0 <= risk["score"] <= 100

            # Test volatility adjusted risk
            vol_risk = strategy.calculate_volatility_adjusted_risk(df)
            assert vol_risk["adjusted_risk_pct"] <= 3.0

            # Test fibonacci levels
            fib_levels, trend = strategy.calculate_fibonacci_levels(df)
            assert trend in ["uptrend", "downtrend"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame"""
        strategy = WeeklyTradingStrategy()
        empty_df = pd.DataFrame()

        with pytest.raises((IndexError, KeyError, ValueError)):
            strategy.calculate_indicators(empty_df)

    def test_single_row_dataframe(self):
        """Test handling of single row DataFrame"""
        strategy = WeeklyTradingStrategy()
        single_row_df = pd.DataFrame({
            "open": [100],
            "high": [105],
            "low": [95],
            "close": [102],
            "volume": [1000],
        })

        # Single row may cause issues with indicators that need more data
        with pytest.raises((TypeError, KeyError, ValueError)):
            strategy.calculate_indicators(single_row_df)

    def test_negative_lookback(self):
        """Test handling of negative lookback values"""
        strategy = WeeklyTradingStrategy()

        # Create sample data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(110, 120, n),
            "low": np.random.uniform(90, 100, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 10000, n),
        })
        df = strategy.calculate_indicators(df)

        # Should handle gracefully
        result = strategy.calculate_support_resistance(df, lookback=-1)
        assert isinstance(result, dict)

    def test_very_large_lookback(self):
        """Test handling of lookback larger than data"""
        strategy = WeeklyTradingStrategy()

        # Create sample data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(110, 120, n),
            "low": np.random.uniform(90, 100, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 10000, n),
        })
        df = strategy.calculate_indicators(df)

        # Should handle gracefully
        result = strategy.calculate_support_resistance(df, lookback=10000)
        assert isinstance(result, dict)

    def test_divergence_with_insufficient_data(self):
        """Test divergence check with insufficient data"""
        strategy = WeeklyTradingStrategy()
        small_df = pd.DataFrame({
            "close": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [98, 99, 100],
            "RSI": [50, 51, 52],
        })

        div_type, strength = strategy.check_divergence(small_df, "RSI", lookback=14)

        assert div_type is None
        assert strength == 0

    def test_historical_performance_with_short_data(self):
        """Test historical performance with insufficient data"""
        strategy = WeeklyTradingStrategy()
        small_df = pd.DataFrame({
            "close": list(range(10)),
            "EMA_9": list(range(10)),
            "EMA_21": list(range(10)),
        })

        result = strategy.analyze_historical_performance(small_df, lookback=50)

        # With short data, function should still return valid result
        assert "win_rate" in result
        assert 0 <= result["win_rate"] <= 100


class TestMultiTimeframeAnalysis:
    """Test multi-timeframe analysis functionality"""

    def test_trend_consistency_with_no_data(self):
        """Test trend consistency when no data is loaded"""
        strategy = WeeklyTradingStrategy()
        strategy.data = {}

        result = strategy.check_trend_consistency()

        assert isinstance(result, dict)
        assert "consistent" in result
        assert "direction" in result

    def test_confidence_level_with_zero_signals(self):
        """Test confidence level with zero signals"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 0, "short": 0, "neutral": 0}

        level, conf = strategy.get_confidence_level(signals)

        assert level == "WAIT"
        assert conf == 0

    def test_confidence_level_various_scenarios(self):
        """Test confidence level with various signal combinations"""
        strategy = WeeklyTradingStrategy()

        # Strong Long
        signals = {"long": 70, "short": 20, "neutral": 10}
        level, conf = strategy.get_confidence_level(signals)
        assert level in ["STRONG_LONG", "LONG"]

        # Strong Short
        signals = {"long": 20, "short": 70, "neutral": 10}
        level, conf = strategy.get_confidence_level(signals)
        assert level in ["STRONG_SHORT", "SHORT"]

        # Wait (mixed signals)
        signals = {"long": 40, "short": 40, "neutral": 20}
        level, conf = strategy.get_confidence_level(signals)
        assert level == "WAIT"
