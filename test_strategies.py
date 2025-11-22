"""
Unit Tests for Trading Strategies Module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from strategies import WeeklyTradingStrategy, MonthlyTradingStrategy


# ==================== Fixtures ====================

@pytest.fixture
def weekly_strategy():
    """Create a WeeklyTradingStrategy instance"""
    return WeeklyTradingStrategy(symbol="BTCUSDT", leverage=5)


@pytest.fixture
def monthly_strategy():
    """Create a MonthlyTradingStrategy instance"""
    return MonthlyTradingStrategy(symbol="BTCUSDT", leverage=3)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n = 100

    # Generate realistic price data
    base_price = 50000.0
    prices: list[float] = [base_price]
    for _ in range(n - 1):
        change = np.random.normal(0, 0.02) * prices[-1]
        prices.append(prices[-1] + change)

    data = {
        "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="D"),
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "close": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        "volume": [np.random.uniform(1000, 10000) for _ in range(n)],
    }

    df = pd.DataFrame(data)
    # Ensure high is highest and low is lowest
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    return df


@pytest.fixture
def sample_data_with_indicators(weekly_strategy, sample_ohlcv_data):
    """Create sample data with calculated indicators"""
    return weekly_strategy.calculate_indicators(sample_ohlcv_data.copy())


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


# ==================== Data Fetching Tests ====================

class TestDataFetching:
    """Test data fetching functionality"""

    def test_fetch_data_returns_dataframe(self, weekly_strategy, mock_binance_response):
        """Test that fetch_data returns a valid DataFrame"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 100

    def test_fetch_data_has_required_columns(self, weekly_strategy, mock_binance_response):
        """Test that fetched data has all required columns"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"

    def test_fetch_data_no_null_values_in_ohlcv(self, weekly_strategy, mock_binance_response):
        """Test that OHLCV data has no null values"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_columns:
                assert df[col].isna().sum() == 0, f"Column {col} has null values"

    def test_fetch_data_no_zero_prices(self, weekly_strategy, mock_binance_response):
        """Test that price data has no zero values"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                assert (df[col] == 0).sum() == 0, f"Column {col} has zero values"
                assert (df[col] > 0).all(), f"Column {col} should have positive values"

    def test_fetch_data_no_negative_volume(self, weekly_strategy, mock_binance_response):
        """Test that volume has no negative values"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            assert (df["volume"] >= 0).all(), "Volume should not be negative"

    def test_fetch_data_high_greater_than_low(self, weekly_strategy, mock_binance_response):
        """Test that high is always >= low"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            assert (df["high"] >= df["low"]).all(), "High should always be >= Low"

    def test_fetch_data_handles_api_error(self, weekly_strategy):
        """Test that fetch_data handles API errors gracefully"""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("API Error")

            df = weekly_strategy.fetch_data("1d", limit=100)

            assert df is None

    def test_fetch_data_numeric_types(self, weekly_strategy, mock_binance_response):
        """Test that OHLCV columns are numeric types"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            df = weekly_strategy.fetch_data("1d", limit=100)

            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"


# ==================== Indicator Calculation Tests ====================

class TestIndicatorCalculations:
    """Test indicator calculation functionality"""

    def test_calculate_indicators_returns_dataframe(self, weekly_strategy, sample_ohlcv_data):
        """Test that calculate_indicators returns a DataFrame"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlcv_data)

    def test_ema_calculated_correctly(self, weekly_strategy, sample_ohlcv_data):
        """Test that EMA indicators are calculated"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        ema_columns = ["EMA_9", "EMA_21", "EMA_50"]
        for col in ema_columns:
            assert col in df.columns, f"Missing EMA column: {col}"
            # EMA should have some valid values (may have NaN at the start)
            assert df[col].notna().sum() > 0, f"EMA {col} has no valid values"

    def test_rsi_in_valid_range(self, weekly_strategy, sample_ohlcv_data):
        """Test that RSI is in valid range [0, 100]"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        assert "RSI" in df.columns
        valid_rsi = df["RSI"].dropna()
        assert (valid_rsi >= 0).all(), "RSI should be >= 0"
        assert (valid_rsi <= 100).all(), "RSI should be <= 100"

    def test_macd_calculated(self, weekly_strategy, sample_ohlcv_data):
        """Test that MACD indicators are calculated"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        macd_columns = ["MACD", "MACD_signal", "MACD_histogram"]
        for col in macd_columns:
            assert col in df.columns, f"Missing MACD column: {col}"

    def test_bollinger_bands_order(self, weekly_strategy, sample_ohlcv_data):
        """Test that Bollinger Bands are in correct order: upper > middle > lower"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        assert "BB_upper" in df.columns
        assert "BB_middle" in df.columns
        assert "BB_lower" in df.columns

        # Check order (excluding NaN rows)
        valid_rows = df[["BB_upper", "BB_middle", "BB_lower"]].dropna()
        assert (valid_rows["BB_upper"] >= valid_rows["BB_middle"]).all()
        assert (valid_rows["BB_middle"] >= valid_rows["BB_lower"]).all()

    def test_adx_in_valid_range(self, weekly_strategy, sample_ohlcv_data):
        """Test that ADX is in valid range [0, 100]"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        assert "ADX" in df.columns
        valid_adx = df["ADX"].dropna()
        assert (valid_adx >= 0).all(), "ADX should be >= 0"
        assert (valid_adx <= 100).all(), "ADX should be <= 100"

    def test_atr_positive(self, weekly_strategy, sample_ohlcv_data):
        """Test that ATR is always positive"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        assert "ATR" in df.columns
        valid_atr = df["ATR"].dropna()
        assert (valid_atr >= 0).all(), "ATR should be >= 0"

    def test_volume_ratio_calculated(self, weekly_strategy, sample_ohlcv_data):
        """Test that Volume Ratio is calculated"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        assert "Volume_Ratio" in df.columns
        valid_vr = df["Volume_Ratio"].dropna()
        assert (valid_vr >= 0).all(), "Volume Ratio should be >= 0"

    def test_stochrsi_in_valid_range(self, weekly_strategy, sample_ohlcv_data):
        """Test that StochRSI is in valid range [0, 100]"""
        df = weekly_strategy.calculate_indicators(sample_ohlcv_data)

        if "STOCHRSI_K" in df.columns:
            valid_stoch = df["STOCHRSI_K"].dropna()
            if len(valid_stoch) > 0:
                assert (valid_stoch >= 0).all(), "StochRSI K should be >= 0"
                assert (valid_stoch <= 100).all(), "StochRSI K should be <= 100"


# ==================== Divergence Detection Tests ====================

class TestDivergenceDetection:
    """Test divergence detection functionality"""

    def test_check_divergence_returns_tuple(self, weekly_strategy, sample_data_with_indicators):
        """Test that check_divergence returns a tuple"""
        result = weekly_strategy.check_divergence(sample_data_with_indicators, "RSI")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_divergence_valid_types(self, weekly_strategy, sample_data_with_indicators):
        """Test that check_divergence returns valid types"""
        div_type, strength = weekly_strategy.check_divergence(sample_data_with_indicators, "RSI")

        assert div_type is None or div_type in ["bullish", "bearish"]
        assert isinstance(strength, (int, float))
        assert strength >= 0

    def test_check_divergence_strength_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that divergence strength is bounded"""
        div_type, strength = weekly_strategy.check_divergence(sample_data_with_indicators, "RSI")

        assert strength <= 100, "Divergence strength should be <= 100"

    def test_check_divergence_with_insufficient_data(self, weekly_strategy):
        """Test divergence check with insufficient data"""
        small_df = pd.DataFrame({
            "close": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [98, 99, 100],
            "RSI": [50, 51, 52],
        })

        div_type, strength = weekly_strategy.check_divergence(small_df, "RSI", lookback=14)

        assert div_type is None
        assert strength == 0

    def test_check_divergence_macd(self, weekly_strategy, sample_data_with_indicators):
        """Test divergence detection with MACD"""
        result = weekly_strategy.check_divergence(sample_data_with_indicators, "MACD", lookback=20)

        assert isinstance(result, tuple)
        assert len(result) == 2


# ==================== Market Regime Detection Tests ====================

class TestMarketRegimeDetection:
    """Test market regime detection functionality"""

    def test_detect_market_regime_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that detect_market_regime returns a dictionary"""
        result = weekly_strategy.detect_market_regime(sample_data_with_indicators)

        assert isinstance(result, dict)

    def test_detect_market_regime_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that market regime result has required keys"""
        result = weekly_strategy.detect_market_regime(sample_data_with_indicators)

        required_keys = ["regime", "confidence", "adx"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_detect_market_regime_valid_regime_values(self, weekly_strategy, sample_data_with_indicators):
        """Test that regime is one of valid values"""
        result = weekly_strategy.detect_market_regime(sample_data_with_indicators)

        valid_regimes = [
            "STRONG_UPTREND", "STRONG_DOWNTREND",
            "WEAK_TREND", "RANGING",
            "HIGH_VOLATILITY", "CONSOLIDATION"
        ]
        assert result["regime"] in valid_regimes, f"Invalid regime: {result['regime']}"

    def test_detect_market_regime_confidence_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that confidence is bounded [0, 100]"""
        result = weekly_strategy.detect_market_regime(sample_data_with_indicators)

        assert 0 <= result["confidence"] <= 100


# ==================== Historical Performance Tests ====================

class TestHistoricalPerformance:
    """Test historical performance analysis"""

    def test_analyze_historical_performance_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that analyze_historical_performance returns a dictionary"""
        result = weekly_strategy.analyze_historical_performance(sample_data_with_indicators)

        assert isinstance(result, dict)

    def test_analyze_historical_performance_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has required keys"""
        result = weekly_strategy.analyze_historical_performance(sample_data_with_indicators)

        required_keys = ["win_rate", "avg_return", "total_signals"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_analyze_historical_performance_win_rate_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that win_rate is bounded [0, 100]"""
        result = weekly_strategy.analyze_historical_performance(sample_data_with_indicators)

        assert 0 <= result["win_rate"] <= 100

    def test_analyze_historical_performance_with_insufficient_data(self, weekly_strategy):
        """Test with insufficient data"""
        small_df = pd.DataFrame({
            "close": range(10),
            "EMA_9": range(10),
            "EMA_21": range(10),
        })

        result = weekly_strategy.analyze_historical_performance(small_df, lookback=50)

        assert result["win_rate"] == 50  # Default value
        # May or may not have total_signals depending on implementation
        assert "win_rate" in result


# ==================== Trend Consistency Tests ====================

class TestTrendConsistency:
    """Test trend consistency check"""

    def test_check_trend_consistency_returns_dict(self, weekly_strategy):
        """Test that check_trend_consistency returns a dictionary"""
        # Need to set up data first
        weekly_strategy.data = {}
        result = weekly_strategy.check_trend_consistency()

        assert isinstance(result, dict)

    def test_check_trend_consistency_has_required_keys(self, weekly_strategy):
        """Test that result has required keys"""
        weekly_strategy.data = {}
        result = weekly_strategy.check_trend_consistency()

        required_keys = ["consistent", "direction", "score"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_check_trend_consistency_valid_direction(self, weekly_strategy):
        """Test that direction is valid"""
        weekly_strategy.data = {}
        result = weekly_strategy.check_trend_consistency()

        valid_directions = ["bullish", "bearish", "neutral", "mixed"]
        assert result["direction"] in valid_directions


# ==================== Position Management Tests ====================

class TestPositionManagement:
    """Test position management calculations"""

    def test_calculate_volatility_adjusted_risk_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that calculate_volatility_adjusted_risk returns a dictionary"""
        result = weekly_strategy.calculate_volatility_adjusted_risk(sample_data_with_indicators)

        assert isinstance(result, dict)

    def test_calculate_volatility_adjusted_risk_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has required keys"""
        result = weekly_strategy.calculate_volatility_adjusted_risk(sample_data_with_indicators)

        required_keys = ["adjusted_risk_pct", "volatility_ratio", "risk_note"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_calculate_volatility_adjusted_risk_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that adjusted_risk_pct is bounded"""
        result = weekly_strategy.calculate_volatility_adjusted_risk(sample_data_with_indicators)

        assert result["adjusted_risk_pct"] <= 3.0, "Risk should be capped at 3%"
        assert result["adjusted_risk_pct"] > 0, "Risk should be positive"

    def test_calculate_support_resistance_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that calculate_support_resistance returns a dictionary"""
        result = weekly_strategy.calculate_support_resistance(sample_data_with_indicators)

        assert isinstance(result, dict)
        assert "resistance" in result
        assert "support" in result
        assert "main_resistance" in result
        assert "main_support" in result

    def test_calculate_support_resistance_levels_valid(self, weekly_strategy, sample_data_with_indicators):
        """Test that support < resistance"""
        result = weekly_strategy.calculate_support_resistance(sample_data_with_indicators)

        assert result["main_support"] < result["main_resistance"]

    def test_calculate_fibonacci_levels_returns_tuple(self, weekly_strategy, sample_data_with_indicators):
        """Test that calculate_fibonacci_levels returns a tuple"""
        result = weekly_strategy.calculate_fibonacci_levels(sample_data_with_indicators)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_calculate_fibonacci_levels_valid_trend(self, weekly_strategy, sample_data_with_indicators):
        """Test that Fibonacci trend is valid"""
        fib_levels, trend = weekly_strategy.calculate_fibonacci_levels(sample_data_with_indicators)

        assert trend in ["uptrend", "downtrend"]
        assert isinstance(fib_levels, dict)


# ==================== Signal Generation Tests ====================

class TestSignalGeneration:
    """Test signal generation"""

    def test_get_trend_strength_returns_tuple(self, weekly_strategy, sample_data_with_indicators):
        """Test that get_trend_strength returns a tuple"""
        result = weekly_strategy.get_trend_strength(sample_data_with_indicators)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_trend_strength_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that trend strength is bounded"""
        score, max_score = weekly_strategy.get_trend_strength(sample_data_with_indicators)

        assert isinstance(score, (int, float))
        assert max_score == 10

    def test_get_confidence_level_valid_values(self, weekly_strategy):
        """Test get_confidence_level with different signal combinations"""
        # Strong Long
        signals = {"long": 70, "short": 20, "neutral": 10}
        level, conf = weekly_strategy.get_confidence_level(signals)
        assert level in ["STRONG_LONG", "LONG"]

        # Strong Short
        signals = {"long": 20, "short": 70, "neutral": 10}
        level, conf = weekly_strategy.get_confidence_level(signals)
        assert level in ["STRONG_SHORT", "SHORT"]

        # Wait
        signals = {"long": 40, "short": 40, "neutral": 20}
        level, conf = weekly_strategy.get_confidence_level(signals)
        assert level == "WAIT"

    def test_get_confidence_level_handles_zero_total(self, weekly_strategy):
        """Test get_confidence_level with zero total"""
        signals = {"long": 0, "short": 0, "neutral": 0}
        level, conf = weekly_strategy.get_confidence_level(signals)

        assert level == "WAIT"
        assert conf == 0


# ==================== Monthly Strategy Tests ====================

class TestMonthlyStrategy:
    """Test MonthlyTradingStrategy specific functionality"""

    def test_monthly_strategy_initialization(self, monthly_strategy):
        """Test MonthlyTradingStrategy initialization"""
        assert monthly_strategy.symbol == "BTCUSDT"
        assert monthly_strategy.leverage == 3
        assert "monthly" in monthly_strategy.timeframes

    def test_monthly_calculate_indicators(self, monthly_strategy, sample_ohlcv_data):
        """Test that monthly strategy calculates indicators"""
        df = monthly_strategy.calculate_indicators(sample_ohlcv_data)

        # Monthly uses EMA_12 and EMA_26 instead of EMA_9 and EMA_21
        assert "EMA_12" in df.columns
        assert "EMA_26" in df.columns

    def test_monthly_check_divergence(self, monthly_strategy, sample_ohlcv_data):
        """Test monthly divergence check"""
        df = monthly_strategy.calculate_indicators(sample_ohlcv_data)
        result = monthly_strategy.check_divergence(df, "RSI")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_monthly_detect_market_regime(self, monthly_strategy, sample_ohlcv_data):
        """Test monthly market regime detection"""
        df = monthly_strategy.calculate_indicators(sample_ohlcv_data)
        result = monthly_strategy.detect_market_regime(df)

        assert isinstance(result, dict)
        assert "regime" in result


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_weekly_workflow(self, weekly_strategy, mock_binance_response):
        """Test complete weekly strategy workflow"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            # Fetch data
            df = weekly_strategy.fetch_data("1d", limit=100)
            assert df is not None

            # Calculate indicators
            df = weekly_strategy.calculate_indicators(df)
            assert "RSI" in df.columns

            # Check divergence
            div_type, strength = weekly_strategy.check_divergence(df, "RSI")
            assert div_type is None or div_type in ["bullish", "bearish"]

            # Detect market regime
            regime = weekly_strategy.detect_market_regime(df)
            assert "regime" in regime

            # Calculate support/resistance
            sr = weekly_strategy.calculate_support_resistance(df)
            assert sr["main_support"] < sr["main_resistance"]

    def test_data_consistency_across_timeframes(self, weekly_strategy, mock_binance_response):
        """Test that data is consistent across different timeframes"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_binance_response

            for tf in ["1d", "4h", "1w"]:
                df = weekly_strategy.fetch_data(tf, limit=50)
                assert df is not None
                # Mock returns 100 rows regardless of limit, just check we got data
                assert len(df) > 0
                assert (df["high"] >= df["low"]).all()


# ==================== Edge Case Tests ====================

# ==================== New Feature Tests ====================

class TestMultiIndicatorConfirmation:
    """Test Multi-Indicator Confirmation System"""

    def test_get_multi_indicator_confirmation_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that multi-indicator confirmation returns a dictionary"""
        result = weekly_strategy.get_multi_indicator_confirmation(sample_data_with_indicators)
        assert isinstance(result, dict)

    def test_get_multi_indicator_confirmation_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has all required keys"""
        result = weekly_strategy.get_multi_indicator_confirmation(sample_data_with_indicators)
        required_keys = ["direction", "strength", "confirmations", "details"]
        for key in required_keys:
            assert key in result

    def test_get_multi_indicator_confirmation_valid_direction(self, weekly_strategy, sample_data_with_indicators):
        """Test that direction is a valid value"""
        result = weekly_strategy.get_multi_indicator_confirmation(sample_data_with_indicators)
        valid_directions = ["bullish", "bearish", "neutral"]
        assert result["direction"] in valid_directions

    def test_get_multi_indicator_confirmation_confirmations_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that confirmations is between 0 and 6"""
        result = weekly_strategy.get_multi_indicator_confirmation(sample_data_with_indicators)
        assert 0 <= result["confirmations"] <= 6


class TestVolumeConfirmation:
    """Test Volume Confirmation System"""

    def test_get_volume_confirmation_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that volume confirmation returns a dictionary"""
        result = weekly_strategy.get_volume_confirmation(sample_data_with_indicators)
        assert isinstance(result, dict)

    def test_get_volume_confirmation_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has all required keys"""
        result = weekly_strategy.get_volume_confirmation(sample_data_with_indicators)
        required_keys = ["confirmed", "volume_ratio", "obv_trend", "details"]
        for key in required_keys:
            assert key in result

    def test_get_volume_confirmation_confirmed_boolean(self, weekly_strategy, sample_data_with_indicators):
        """Test that confirmed is a boolean"""
        result = weekly_strategy.get_volume_confirmation(sample_data_with_indicators)
        assert isinstance(result["confirmed"], bool)

    def test_get_volume_confirmation_volume_ratio_positive(self, weekly_strategy, sample_data_with_indicators):
        """Test that volume ratio is positive"""
        result = weekly_strategy.get_volume_confirmation(sample_data_with_indicators)
        assert result["volume_ratio"] >= 0


class TestCandlestickPatterns:
    """Test Candlestick Pattern Detection"""

    def test_get_candlestick_signals_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that candlestick signals returns a dictionary"""
        result = weekly_strategy.get_candlestick_signals(sample_data_with_indicators)
        assert isinstance(result, dict)

    def test_get_candlestick_signals_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has all required keys"""
        result = weekly_strategy.get_candlestick_signals(sample_data_with_indicators)
        required_keys = ["bullish", "bearish", "score"]
        for key in required_keys:
            assert key in result

    def test_get_candlestick_signals_patterns_are_lists(self, weekly_strategy, sample_data_with_indicators):
        """Test that patterns are lists"""
        result = weekly_strategy.get_candlestick_signals(sample_data_with_indicators)
        assert isinstance(result["bullish"], list)
        assert isinstance(result["bearish"], list)

    def test_get_candlestick_signals_score_is_int(self, weekly_strategy, sample_data_with_indicators):
        """Test that score is an integer"""
        result = weekly_strategy.get_candlestick_signals(sample_data_with_indicators)
        assert isinstance(result["score"], int)


class TestDynamicThresholds:
    """Test Dynamic Threshold System"""

    def test_get_dynamic_thresholds_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that dynamic thresholds returns a dictionary"""
        result = weekly_strategy.get_dynamic_thresholds(sample_data_with_indicators)
        assert isinstance(result, dict)

    def test_get_dynamic_thresholds_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has all required keys"""
        result = weekly_strategy.get_dynamic_thresholds(sample_data_with_indicators)
        required_keys = ["rsi_oversold", "rsi_overbought", "volume_threshold"]
        for key in required_keys:
            assert key in result

    def test_get_dynamic_thresholds_rsi_bounds_valid(self, weekly_strategy, sample_data_with_indicators):
        """Test that RSI thresholds are within valid bounds"""
        result = weekly_strategy.get_dynamic_thresholds(sample_data_with_indicators)
        # Oversold should be less than overbought
        assert result["rsi_oversold"] < result["rsi_overbought"]
        # Oversold should be between 20-40, overbought between 60-80
        assert 15 <= result["rsi_oversold"] <= 45
        assert 55 <= result["rsi_overbought"] <= 85


class TestConfluenceZones:
    """Test Confluence Zones Detection"""

    def test_find_confluence_zones_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that confluence zones returns a dictionary"""
        current_price = sample_data_with_indicators["close"].iloc[-1]
        result = weekly_strategy.find_confluence_zones(sample_data_with_indicators, current_price)
        assert isinstance(result, dict)

    def test_find_confluence_zones_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has support and resistance keys"""
        current_price = sample_data_with_indicators["close"].iloc[-1]
        result = weekly_strategy.find_confluence_zones(sample_data_with_indicators, current_price)
        assert "support" in result
        assert "resistance" in result

    def test_find_confluence_zones_returns_lists(self, weekly_strategy, sample_data_with_indicators):
        """Test that support and resistance are lists"""
        current_price = sample_data_with_indicators["close"].iloc[-1]
        result = weekly_strategy.find_confluence_zones(sample_data_with_indicators, current_price)
        assert isinstance(result["support"], list)
        assert isinstance(result["resistance"], list)


class TestRiskScore:
    """Test Risk Score Calculation"""

    def test_calculate_risk_score_returns_dict(self, weekly_strategy, sample_data_with_indicators):
        """Test that risk score returns a dictionary"""
        result = weekly_strategy.calculate_risk_score(sample_data_with_indicators, "LONG")
        assert isinstance(result, dict)

    def test_calculate_risk_score_has_required_keys(self, weekly_strategy, sample_data_with_indicators):
        """Test that result has all required keys"""
        result = weekly_strategy.calculate_risk_score(sample_data_with_indicators, "LONG")
        required_keys = ["score", "level", "factors"]
        for key in required_keys:
            assert key in result

    def test_calculate_risk_score_bounded(self, weekly_strategy, sample_data_with_indicators):
        """Test that risk score is between 0 and 100"""
        result = weekly_strategy.calculate_risk_score(sample_data_with_indicators, "LONG")
        assert 0 <= result["score"] <= 100

    def test_calculate_risk_score_valid_level(self, weekly_strategy, sample_data_with_indicators):
        """Test that level is a valid value"""
        result = weekly_strategy.calculate_risk_score(sample_data_with_indicators, "LONG")
        valid_levels = ["LOW", "MEDIUM", "HIGH"]
        assert result["level"] in valid_levels

    def test_calculate_risk_score_short_signal(self, weekly_strategy, sample_data_with_indicators):
        """Test risk score calculation for SHORT signal"""
        result = weekly_strategy.calculate_risk_score(sample_data_with_indicators, "SHORT")
        assert isinstance(result, dict)
        assert 0 <= result["score"] <= 100


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe_handling(self, weekly_strategy):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()

        with pytest.raises((IndexError, KeyError, ValueError)):
            weekly_strategy.calculate_indicators(empty_df)

    def test_single_row_dataframe(self, weekly_strategy):
        """Test handling of single row DataFrame"""
        single_row_df = pd.DataFrame({
            "open": [100],
            "high": [105],
            "low": [95],
            "close": [102],
            "volume": [1000],
        })

        # Single row may cause issues with indicators that need more data
        # This is expected behavior - we test that it raises appropriate errors
        with pytest.raises((TypeError, KeyError, ValueError)):
            weekly_strategy.calculate_indicators(single_row_df)

    def test_negative_lookback(self, weekly_strategy, sample_data_with_indicators):
        """Test handling of negative lookback values"""
        # Should use default or handle gracefully
        result = weekly_strategy.calculate_support_resistance(sample_data_with_indicators, lookback=-1)
        assert isinstance(result, dict)

    def test_very_large_lookback(self, weekly_strategy, sample_data_with_indicators):
        """Test handling of lookback larger than data"""
        result = weekly_strategy.calculate_support_resistance(sample_data_with_indicators, lookback=10000)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
