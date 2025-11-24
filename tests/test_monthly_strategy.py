"""
Unit tests for trading/monthly_strategy.py
"""

import pytest
import pandas as pd
import numpy as np

from trading.monthly_strategy import MonthlyTradingStrategy


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n_periods = 100

    base_price = 50000
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = base_price * np.cumprod(1 + returns)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_periods)),
        'high': prices * (1 + np.random.uniform(0.005, 0.03, n_periods)),
        'low': prices * (1 - np.random.uniform(0.005, 0.03, n_periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_periods) * 1e6
    })

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestMonthlyTradingStrategyInit:
    """Tests for MonthlyTradingStrategy initialization"""

    def test_default_initialization(self):
        """Test default initialization"""
        strategy = MonthlyTradingStrategy()
        assert strategy.symbol == "BTCUSDT"
        assert strategy.leverage == 3

    def test_custom_symbol(self):
        """Test initialization with custom symbol"""
        strategy = MonthlyTradingStrategy(symbol="ETHUSDT")
        assert strategy.symbol == "ETHUSDT"

    def test_custom_leverage(self):
        """Test initialization with custom leverage"""
        strategy = MonthlyTradingStrategy(leverage=5)
        assert strategy.leverage == 5

    def test_timeframes_set(self):
        """Test that timeframes are properly set"""
        strategy = MonthlyTradingStrategy()
        assert "monthly" in strategy.timeframes
        assert "weekly" in strategy.timeframes
        assert "daily" in strategy.timeframes


class TestMonthlyIndicators:
    """Tests for monthly strategy indicator calculation"""

    def test_calculate_indicators_returns_dataframe(self, sample_ohlcv_data):
        """Test that calculate_indicators returns a DataFrame"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        assert isinstance(df, pd.DataFrame)

    def test_monthly_has_ema_columns(self, sample_ohlcv_data):
        """Test that monthly strategy calculates EMA indicators"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        # Monthly strategy may use standard EMAs or custom EMAs
        # depending on implementation
        ema_cols = [c for c in df.columns if c.startswith("EMA_")]
        assert len(ema_cols) > 0, "EMA columns should be present"

    def test_monthly_contains_rsi(self, sample_ohlcv_data):
        """Test that RSI is calculated"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        assert "RSI" in df.columns

    def test_monthly_contains_macd(self, sample_ohlcv_data):
        """Test that MACD indicators are calculated"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        assert "MACD" in df.columns
        assert "MACD_signal" in df.columns


class TestMonthlyDivergence:
    """Tests for monthly divergence check"""

    def test_check_divergence_returns_tuple(self, sample_ohlcv_data):
        """Test that check_divergence returns a tuple"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.check_divergence(df, "RSI")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_divergence_valid_types(self, sample_ohlcv_data):
        """Test that divergence returns valid types"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        div_type, strength = strategy.check_divergence(df, "RSI")
        assert div_type is None or div_type in ["bullish", "bearish"]
        assert isinstance(strength, (int, float))


class TestMonthlyMarketRegime:
    """Tests for monthly market regime detection"""

    def test_detect_market_regime_returns_dict(self, sample_ohlcv_data):
        """Test that detect_market_regime returns a dictionary"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.detect_market_regime(df)
        assert isinstance(result, dict)

    def test_detect_market_regime_has_required_keys(self, sample_ohlcv_data):
        """Test that market regime result has required keys"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.detect_market_regime(df)
        assert "regime" in result
        assert "confidence" in result

    def test_detect_market_regime_valid_regime(self, sample_ohlcv_data):
        """Test that regime is one of valid values"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.detect_market_regime(df)
        valid_regimes = [
            "STRONG_UPTREND", "STRONG_DOWNTREND",
            "WEAK_TREND", "RANGING",
            "HIGH_VOLATILITY", "CONSOLIDATION", "UNKNOWN"
        ]
        assert result["regime"] in valid_regimes


class TestMonthlyPositionManagement:
    """Tests for monthly position management"""

    def test_calculate_support_resistance(self, sample_ohlcv_data):
        """Test that support/resistance is calculated"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.calculate_support_resistance(df)
        assert isinstance(result, dict)
        assert "main_support" in result
        assert "main_resistance" in result

    def test_support_below_resistance(self, sample_ohlcv_data):
        """Test that support is below resistance"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.calculate_support_resistance(df)
        assert result["main_support"] < result["main_resistance"]

    def test_calculate_fibonacci_levels(self, sample_ohlcv_data):
        """Test that Fibonacci levels are calculated"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        fib_levels, trend = strategy.calculate_fibonacci_levels(df)
        assert isinstance(fib_levels, dict)
        assert trend in ["uptrend", "downtrend"]


class TestMonthlyRiskManagement:
    """Tests for monthly risk management"""

    def test_calculate_volatility_adjusted_risk(self, sample_ohlcv_data):
        """Test volatility adjusted risk calculation"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.calculate_volatility_adjusted_risk(df)
        assert isinstance(result, dict)
        assert "adjusted_risk_pct" in result
        assert result["adjusted_risk_pct"] <= 3.0

    def test_calculate_risk_score(self, sample_ohlcv_data):
        """Test risk score calculation"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.calculate_risk_score(df, "LONG")
        assert isinstance(result, dict)
        assert 0 <= result["score"] <= 100
        assert result["level"] in ["LOW", "MEDIUM", "HIGH"]


class TestMonthlySignalGeneration:
    """Tests for monthly signal generation"""

    def test_get_trend_strength(self, sample_ohlcv_data):
        """Test trend strength calculation"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        score, max_score = strategy.get_trend_strength(df)
        assert isinstance(score, (int, float))
        assert max_score > 0

    def test_get_confidence_level(self):
        """Test confidence level calculation"""
        strategy = MonthlyTradingStrategy()
        signals = {"long": 60, "short": 30, "neutral": 10}
        level, confidence = strategy.get_confidence_level(signals)
        assert level in ["STRONG_LONG", "LONG", "STRONG_SHORT", "SHORT", "WAIT"]
        assert 0 <= confidence <= 100

    def test_get_multi_indicator_confirmation(self, sample_ohlcv_data):
        """Test multi-indicator confirmation"""
        strategy = MonthlyTradingStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.get_multi_indicator_confirmation(df)
        assert isinstance(result, dict)
        assert result["direction"] in ["bullish", "bearish", "neutral"]
