"""
Unit tests for trading/weekly_strategy.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from trading.weekly_strategy import WeeklyTradingStrategy


class TestWeeklyTradingStrategyInit:
    """Tests for WeeklyTradingStrategy initialization"""

    def test_default_initialization(self):
        """Test default initialization"""
        strategy = WeeklyTradingStrategy()
        assert strategy.symbol == "BTCUSDT"
        assert strategy.leverage == 5

    def test_custom_symbol(self):
        """Test initialization with custom symbol"""
        strategy = WeeklyTradingStrategy(symbol="ETHUSDT")
        assert strategy.symbol == "ETHUSDT"

    def test_custom_leverage(self):
        """Test initialization with custom leverage"""
        strategy = WeeklyTradingStrategy(leverage=10)
        assert strategy.leverage == 10

    def test_timeframes_set(self):
        """Test that timeframes are properly set"""
        strategy = WeeklyTradingStrategy()
        assert "weekly" in strategy.timeframes
        assert "daily" in strategy.timeframes
        assert "h4" in strategy.timeframes
        assert strategy.timeframes["weekly"] == "1w"
        assert strategy.timeframes["daily"] == "1d"
        assert strategy.timeframes["h4"] == "4h"


class TestTimeframeWeights:
    """Tests for timeframe weights"""

    def test_get_timeframe_weights(self):
        """Test that timeframe weights are returned"""
        strategy = WeeklyTradingStrategy()
        weights = strategy._get_timeframe_weights()
        assert isinstance(weights, dict)
        assert "weekly" in weights
        assert "daily" in weights
        assert "h4" in weights

    def test_weekly_highest_weight(self):
        """Test that weekly has highest weight"""
        strategy = WeeklyTradingStrategy()
        weights = strategy._get_timeframe_weights()
        assert weights["weekly"] > weights["daily"]
        assert weights["daily"] > weights["h4"]


class TestGetWeightedSignalScore:
    """Tests for get_weighted_signal_score method"""

    def test_returns_float(self):
        """Test that function returns a float"""
        strategy = WeeklyTradingStrategy()
        market_regime = {"regime": "RANGING", "confidence": 50}
        historical_perf = {"win_rate": 50}

        result = strategy.get_weighted_signal_score(
            base_score=10,
            timeframe="daily",
            market_regime=market_regime,
            historical_perf=historical_perf
        )
        assert isinstance(result, float)

    def test_weekly_higher_weight(self):
        """Test that weekly gets higher weight than daily"""
        strategy = WeeklyTradingStrategy()
        market_regime = {"regime": "RANGING", "confidence": 50}
        historical_perf = {"win_rate": 50}

        weekly_score = strategy.get_weighted_signal_score(
            base_score=10, timeframe="weekly",
            market_regime=market_regime, historical_perf=historical_perf
        )
        daily_score = strategy.get_weighted_signal_score(
            base_score=10, timeframe="daily",
            market_regime=market_regime, historical_perf=historical_perf
        )

        assert weekly_score > daily_score

    def test_strong_trend_increases_weight(self):
        """Test that strong trend increases weight"""
        strategy = WeeklyTradingStrategy()
        historical_perf = {"win_rate": 50}

        ranging_score = strategy.get_weighted_signal_score(
            base_score=10, timeframe="daily",
            market_regime={"regime": "RANGING", "confidence": 50},
            historical_perf=historical_perf
        )
        trending_score = strategy.get_weighted_signal_score(
            base_score=10, timeframe="daily",
            market_regime={"regime": "STRONG_UPTREND", "confidence": 80},
            historical_perf=historical_perf
        )

        assert trending_score > ranging_score

    def test_high_win_rate_increases_weight(self):
        """Test that high historical win rate increases weight"""
        strategy = WeeklyTradingStrategy()
        market_regime = {"regime": "RANGING", "confidence": 50}

        low_wr_score = strategy.get_weighted_signal_score(
            base_score=10, timeframe="daily",
            market_regime=market_regime,
            historical_perf={"win_rate": 35}
        )
        high_wr_score = strategy.get_weighted_signal_score(
            base_score=10, timeframe="daily",
            market_regime=market_regime,
            historical_perf={"win_rate": 65}
        )

        assert high_wr_score > low_wr_score


class TestCalculatePositionManagement:
    """Tests for calculate_position_management method"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with mock data"""
        strategy = WeeklyTradingStrategy()

        # Calculate indicators
        from trading.indicators import calculate_indicators
        df = calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")

        strategy.data = {
            "weekly": df.copy(),
            "daily": df.copy(),
            "h4": df.copy()
        }
        return strategy

    def test_returns_dict(self, strategy_with_data):
        """Test that function returns a dictionary"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "LONG")
        assert isinstance(result, dict)

    def test_contains_required_keys(self, strategy_with_data):
        """Test that result contains required keys"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "LONG")

        required_keys = ["entry", "stop_loss", "tp1", "tp2", "tp3", "atr", "fibonacci"]
        for key in required_keys:
            assert key in result

    def test_long_stop_loss_below_entry(self, strategy_with_data):
        """Test that LONG stop loss is below entry"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "LONG")
        assert result["stop_loss"] < result["entry"]

    def test_long_tp_above_entry(self, strategy_with_data):
        """Test that LONG take profits are above entry"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "LONG")
        assert result["tp1"] > result["entry"]
        assert result["tp2"] > result["tp1"]

    def test_short_stop_loss_above_entry(self, strategy_with_data):
        """Test that SHORT stop loss is above entry"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "SHORT")
        assert result["stop_loss"] > result["entry"]

    def test_short_tp_below_entry(self, strategy_with_data):
        """Test that SHORT take profits are below entry"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "SHORT")
        assert result["tp1"] < result["entry"]
        assert result["tp2"] < result["tp1"]

    def test_fibonacci_included(self, strategy_with_data):
        """Test that Fibonacci levels are included"""
        current_price = strategy_with_data.data["daily"].iloc[-1]["close"]
        result = strategy_with_data.calculate_position_management(current_price, "LONG")
        assert "fibonacci" in result
        assert "127.2%" in result["fibonacci"]
        assert "161.8%" in result["fibonacci"]


class TestGetWeeklySignal:
    """Tests for get_weekly_signal method"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with mock data"""
        strategy = WeeklyTradingStrategy()

        from trading.indicators import calculate_indicators
        df = calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")

        strategy.data = {
            "weekly": calculate_indicators(sample_ohlcv_data.copy(), timeframe="weekly"),
            "daily": df.copy(),
            "h4": calculate_indicators(sample_ohlcv_data.copy(), timeframe="h4")
        }
        return strategy

    def test_returns_tuple(self, strategy_with_data):
        """Test that function returns a tuple"""
        result = strategy_with_data.get_weekly_signal()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_signals_is_dict(self, strategy_with_data):
        """Test that signals is a dictionary"""
        signals, reasons = strategy_with_data.get_weekly_signal()
        assert isinstance(signals, dict)

    def test_reasons_is_dict(self, strategy_with_data):
        """Test that reasons is a dictionary"""
        signals, reasons = strategy_with_data.get_weekly_signal()
        assert isinstance(reasons, dict)

    def test_signals_has_required_keys(self, strategy_with_data):
        """Test that signals has required keys"""
        signals, _ = strategy_with_data.get_weekly_signal()
        assert "long" in signals
        assert "short" in signals
        assert "neutral" in signals

    def test_reasons_has_required_keys(self, strategy_with_data):
        """Test that reasons has required keys"""
        _, reasons = strategy_with_data.get_weekly_signal()
        assert "long" in reasons
        assert "short" in reasons
        assert "neutral" in reasons

    def test_signals_are_non_negative(self, strategy_with_data):
        """Test that signal scores are non-negative"""
        signals, _ = strategy_with_data.get_weekly_signal()
        assert signals["long"] >= 0
        assert signals["short"] >= 0
        assert signals["neutral"] >= 0

    def test_reasons_are_lists(self, strategy_with_data):
        """Test that reasons are lists"""
        _, reasons = strategy_with_data.get_weekly_signal()
        assert isinstance(reasons["long"], list)
        assert isinstance(reasons["short"], list)
        assert isinstance(reasons["neutral"], list)


class TestGetTrendStrength:
    """Tests for get_trend_strength method"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with mock data"""
        strategy = WeeklyTradingStrategy()
        from trading.indicators import calculate_indicators
        strategy.data = {
            "daily": calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")
        }
        return strategy

    def test_returns_tuple(self, strategy_with_data):
        """Test that function returns a tuple"""
        result = strategy_with_data.get_trend_strength(strategy_with_data.data["daily"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_and_max_score(self, strategy_with_data):
        """Test that score and max_score are returned"""
        score, max_score = strategy_with_data.get_trend_strength(strategy_with_data.data["daily"])
        assert isinstance(score, int)
        assert isinstance(max_score, int)

    def test_max_score_positive(self, strategy_with_data):
        """Test that max_score is positive"""
        _, max_score = strategy_with_data.get_trend_strength(strategy_with_data.data["daily"])
        assert max_score > 0


class TestCheckTrendConsistency:
    """Tests for check_trend_consistency method"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with multi-timeframe data"""
        strategy = WeeklyTradingStrategy()
        from trading.indicators import calculate_indicators

        strategy.data = {
            "weekly": calculate_indicators(sample_ohlcv_data.copy(), timeframe="weekly"),
            "daily": calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily"),
            "h4": calculate_indicators(sample_ohlcv_data.copy(), timeframe="h4")
        }
        return strategy

    def test_returns_dict(self, strategy_with_data):
        """Test that function returns a dictionary"""
        result = strategy_with_data.check_trend_consistency()
        assert isinstance(result, dict)

    def test_contains_required_keys(self, strategy_with_data):
        """Test that result contains required keys"""
        result = strategy_with_data.check_trend_consistency()
        assert "consistent" in result
        assert "direction" in result
        assert "score" in result

    def test_consistent_is_boolean(self, strategy_with_data):
        """Test that consistent is boolean"""
        result = strategy_with_data.check_trend_consistency()
        assert isinstance(result["consistent"], bool)

    def test_direction_is_valid(self, strategy_with_data):
        """Test that direction is valid"""
        result = strategy_with_data.check_trend_consistency()
        assert result["direction"] in ["bullish", "bearish", "neutral", "mixed"]

    def test_score_range(self, strategy_with_data):
        """Test that score is within valid range"""
        result = strategy_with_data.check_trend_consistency()
        assert 0 <= result["score"] <= 100


class TestGetConfidenceLevel:
    """Tests for get_confidence_level method"""

    def test_returns_tuple(self):
        """Test that function returns a tuple"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 10, "short": 5, "neutral": 2}
        result = strategy.get_confidence_level(signals)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_recommendation_and_confidence(self):
        """Test that recommendation and confidence are returned"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 10, "short": 5, "neutral": 2}
        recommendation, confidence = strategy.get_confidence_level(signals)
        assert isinstance(recommendation, str)
        assert isinstance(confidence, float)

    def test_strong_long_signal(self):
        """Test strong long signal detection"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 80, "short": 10, "neutral": 10}
        recommendation, confidence = strategy.get_confidence_level(signals)
        assert recommendation == "STRONG_LONG"
        assert confidence >= 70

    def test_strong_short_signal(self):
        """Test strong short signal detection"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 10, "short": 80, "neutral": 10}
        recommendation, confidence = strategy.get_confidence_level(signals)
        assert recommendation == "STRONG_SHORT"
        assert confidence >= 70

    def test_wait_signal(self):
        """Test wait signal detection"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 40, "short": 40, "neutral": 20}
        recommendation, _ = strategy.get_confidence_level(signals)
        assert recommendation == "WAIT"

    def test_zero_signals_returns_wait(self):
        """Test that zero signals returns WAIT"""
        strategy = WeeklyTradingStrategy()
        signals = {"long": 0, "short": 0, "neutral": 0}
        recommendation, confidence = strategy.get_confidence_level(signals)
        assert recommendation == "WAIT"
        assert confidence == 0


class TestUtilityMethods:
    """Tests for utility methods"""

    def test_format_price_large(self):
        """Test price formatting for large prices"""
        strategy = WeeklyTradingStrategy()
        result = strategy._format_price(50000)
        assert "$" in result
        assert "50,000" in result

    def test_format_price_small(self):
        """Test price formatting for small prices"""
        strategy = WeeklyTradingStrategy()
        result = strategy._format_price(0.5)
        assert "$" in result

    def test_calculate_position_size(self):
        """Test position size calculation"""
        strategy = WeeklyTradingStrategy()
        result = strategy._calculate_position_size(
            balance=10000,
            risk_pct=2.0,
            stop_loss_pct=5.0
        )

        assert "risk_amount" in result
        assert "position_size" in result
        assert "margin_required" in result
        assert result["risk_amount"] == 200  # 2% of 10000


class TestCandlestickPatterns:
    """Tests for candlestick pattern detection"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with mock data"""
        strategy = WeeklyTradingStrategy()
        from trading.indicators import calculate_indicators
        strategy.data = {
            "daily": calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")
        }
        return strategy

    def test_get_candlestick_signals_returns_dict(self, strategy_with_data):
        """Test that candlestick signals returns a dictionary"""
        result = strategy_with_data.get_candlestick_signals(strategy_with_data.data["daily"])
        assert isinstance(result, dict)

    def test_get_candlestick_signals_has_required_keys(self, strategy_with_data):
        """Test that result has all required keys"""
        result = strategy_with_data.get_candlestick_signals(strategy_with_data.data["daily"])
        required_keys = ["bullish", "bearish", "score"]
        for key in required_keys:
            assert key in result

    def test_get_candlestick_signals_patterns_are_lists(self, strategy_with_data):
        """Test that patterns are lists"""
        result = strategy_with_data.get_candlestick_signals(strategy_with_data.data["daily"])
        assert isinstance(result["bullish"], list)
        assert isinstance(result["bearish"], list)

    def test_get_candlestick_signals_score_is_int(self, strategy_with_data):
        """Test that score is an integer"""
        result = strategy_with_data.get_candlestick_signals(strategy_with_data.data["daily"])
        assert isinstance(result["score"], int)


class TestDynamicThresholds:
    """Tests for dynamic threshold system"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with mock data"""
        strategy = WeeklyTradingStrategy()
        from trading.indicators import calculate_indicators
        strategy.data = {
            "daily": calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")
        }
        return strategy

    def test_get_dynamic_thresholds_returns_dict(self, strategy_with_data):
        """Test that dynamic thresholds returns a dictionary"""
        result = strategy_with_data.get_dynamic_thresholds(strategy_with_data.data["daily"])
        assert isinstance(result, dict)

    def test_get_dynamic_thresholds_has_required_keys(self, strategy_with_data):
        """Test that result has all required keys"""
        result = strategy_with_data.get_dynamic_thresholds(strategy_with_data.data["daily"])
        required_keys = ["rsi_oversold", "rsi_overbought", "volume_threshold"]
        for key in required_keys:
            assert key in result

    def test_get_dynamic_thresholds_rsi_bounds_valid(self, strategy_with_data):
        """Test that RSI thresholds are within valid bounds"""
        result = strategy_with_data.get_dynamic_thresholds(strategy_with_data.data["daily"])
        # Oversold should be less than overbought
        assert result["rsi_oversold"] < result["rsi_overbought"]
        # Oversold should be between 15-45, overbought between 55-85
        assert 15 <= result["rsi_oversold"] <= 45
        assert 55 <= result["rsi_overbought"] <= 85


class TestConfluenceZonesStrategy:
    """Tests for confluence zones detection in strategy"""

    @pytest.fixture
    def strategy_with_data(self, sample_ohlcv_data):
        """Create strategy with mock data"""
        strategy = WeeklyTradingStrategy()
        from trading.indicators import calculate_indicators
        strategy.data = {
            "daily": calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")
        }
        return strategy

    def test_find_confluence_zones_returns_dict(self, strategy_with_data):
        """Test that confluence zones returns a dictionary"""
        current_price = strategy_with_data.data["daily"]["close"].iloc[-1]
        result = strategy_with_data.find_confluence_zones(
            strategy_with_data.data["daily"], current_price
        )
        assert isinstance(result, dict)

    def test_find_confluence_zones_has_required_keys(self, strategy_with_data):
        """Test that result has support and resistance keys"""
        current_price = strategy_with_data.data["daily"]["close"].iloc[-1]
        result = strategy_with_data.find_confluence_zones(
            strategy_with_data.data["daily"], current_price
        )
        assert "support" in result
        assert "resistance" in result

    def test_find_confluence_zones_returns_lists(self, strategy_with_data):
        """Test that support and resistance are lists"""
        current_price = strategy_with_data.data["daily"]["close"].iloc[-1]
        result = strategy_with_data.find_confluence_zones(
            strategy_with_data.data["daily"], current_price
        )
        assert isinstance(result["support"], list)
        assert isinstance(result["resistance"], list)
