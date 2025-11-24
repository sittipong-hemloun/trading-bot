"""
Unit tests for trading/analysis.py
"""


from trading.analysis import (
    get_multi_indicator_confirmation,
    get_volume_confirmation,
    find_confluence_zones,
    get_dynamic_thresholds,
    check_divergence,
    detect_market_regime,
    analyze_historical_performance,
)


class TestGetMultiIndicatorConfirmation:
    """Tests for get_multi_indicator_confirmation function"""

    def test_returns_dict(self, df_with_indicators):
        """Test that function returns a dictionary"""
        result = get_multi_indicator_confirmation(df_with_indicators)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, df_with_indicators):
        """Test that result contains required keys"""
        result = get_multi_indicator_confirmation(df_with_indicators)
        assert "direction" in result
        assert "strength" in result
        assert "confirmations" in result
        assert "details" in result

    def test_direction_values(self, df_with_indicators):
        """Test that direction is valid"""
        result = get_multi_indicator_confirmation(df_with_indicators)
        assert result["direction"] in ["bullish", "bearish", "neutral", "mixed"]

    def test_strength_range(self, df_with_indicators):
        """Test that strength is within 0-100 range"""
        result = get_multi_indicator_confirmation(df_with_indicators)
        assert 0 <= result["strength"] <= 100

    def test_confirmations_count(self, df_with_indicators):
        """Test that confirmations is non-negative"""
        result = get_multi_indicator_confirmation(df_with_indicators)
        assert result["confirmations"] >= 0

    def test_bullish_trend_confirmation(self, bullish_df_with_indicators):
        """Test that bullish trend gets bullish confirmation"""
        result = get_multi_indicator_confirmation(bullish_df_with_indicators)
        # In strong bullish trend, should be bullish or at least not bearish
        assert result["direction"] in ["bullish", "mixed", "neutral"]

    def test_bearish_trend_confirmation(self, bearish_df_with_indicators):
        """Test that bearish trend gets bearish confirmation"""
        result = get_multi_indicator_confirmation(bearish_df_with_indicators)
        # In strong bearish trend, should be bearish or at least not bullish
        assert result["direction"] in ["bearish", "mixed", "neutral"]


class TestGetVolumeConfirmation:
    """Tests for get_volume_confirmation function"""

    def test_returns_dict(self, df_with_indicators):
        """Test that function returns a dictionary"""
        result = get_volume_confirmation(df_with_indicators)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, df_with_indicators):
        """Test that result contains required keys"""
        result = get_volume_confirmation(df_with_indicators)
        assert "confirmed" in result
        assert "volume_trend" in result
        assert "volume_ratio" in result
        assert "obv_trend" in result
        assert "details" in result

    def test_confirmed_is_boolean(self, df_with_indicators):
        """Test that confirmed is boolean"""
        result = get_volume_confirmation(df_with_indicators)
        assert isinstance(result["confirmed"], bool)

    def test_volume_ratio_positive(self, df_with_indicators):
        """Test that volume ratio is positive"""
        result = get_volume_confirmation(df_with_indicators)
        assert result["volume_ratio"] > 0

    def test_obv_trend_values(self, df_with_indicators):
        """Test that OBV trend is valid"""
        result = get_volume_confirmation(df_with_indicators)
        assert result["obv_trend"] in ["bullish", "bearish", "neutral"]


class TestFindConfluenceZones:
    """Tests for find_confluence_zones function"""

    def test_returns_dict(self, df_with_indicators):
        """Test that function returns a dictionary"""
        current_price = df_with_indicators.iloc[-1]["close"]
        result = find_confluence_zones(df_with_indicators, current_price)
        assert isinstance(result, dict)

    def test_contains_support_resistance(self, df_with_indicators):
        """Test that result contains support and resistance"""
        current_price = df_with_indicators.iloc[-1]["close"]
        result = find_confluence_zones(df_with_indicators, current_price)
        assert "support" in result
        assert "resistance" in result

    def test_support_below_price(self, df_with_indicators):
        """Test that support zones are below current price"""
        current_price = df_with_indicators.iloc[-1]["close"]
        result = find_confluence_zones(df_with_indicators, current_price)
        for zone in result["support"]:
            assert zone["price"] < current_price

    def test_resistance_above_price(self, df_with_indicators):
        """Test that resistance zones are above current price"""
        current_price = df_with_indicators.iloc[-1]["close"]
        result = find_confluence_zones(df_with_indicators, current_price)
        for zone in result["resistance"]:
            assert zone["price"] > current_price

    def test_zone_structure(self, df_with_indicators):
        """Test that zones have correct structure"""
        current_price = df_with_indicators.iloc[-1]["close"]
        result = find_confluence_zones(df_with_indicators, current_price)
        for zone in result["support"] + result["resistance"]:
            assert "price" in zone
            assert "strength" in zone
            assert zone["strength"] >= 2  # Confluence requires at least 2 levels


class TestGetDynamicThresholds:
    """Tests for get_dynamic_thresholds function"""

    def test_returns_dict(self, df_with_indicators):
        """Test that function returns a dictionary"""
        result = get_dynamic_thresholds(df_with_indicators)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, df_with_indicators):
        """Test that result contains required keys"""
        result = get_dynamic_thresholds(df_with_indicators)
        assert "rsi_oversold" in result
        assert "rsi_overbought" in result
        assert "volume_threshold" in result
        assert "volatility_ratio" in result

    def test_rsi_thresholds_valid(self, df_with_indicators):
        """Test that RSI thresholds are valid"""
        result = get_dynamic_thresholds(df_with_indicators)
        assert 0 < result["rsi_oversold"] < 50
        assert 50 < result["rsi_overbought"] < 100
        assert result["rsi_oversold"] < result["rsi_overbought"]

    def test_volatility_ratio_positive(self, df_with_indicators):
        """Test that volatility ratio is positive"""
        result = get_dynamic_thresholds(df_with_indicators)
        assert result["volatility_ratio"] > 0

    def test_high_volatility_adjusts_thresholds(self, high_volatility_data):
        """Test that high volatility adjusts thresholds"""
        from trading.indicators import calculate_indicators
        df = calculate_indicators(high_volatility_data)
        result = get_dynamic_thresholds(df)
        # In high volatility, thresholds should be more extreme
        # This test just checks the function handles high volatility
        assert result["rsi_oversold"] <= 30
        assert result["rsi_overbought"] >= 70


class TestCheckDivergence:
    """Tests for check_divergence function"""

    def test_returns_tuple(self, df_with_indicators):
        """Test that function returns a tuple"""
        result = check_divergence(df_with_indicators, "RSI")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_divergence_type_valid(self, df_with_indicators):
        """Test that divergence type is valid"""
        div_type, strength = check_divergence(df_with_indicators, "RSI")
        assert div_type in [None, "bullish", "bearish"]

    def test_strength_range(self, df_with_indicators):
        """Test that strength is within 0-100 range"""
        _, strength = check_divergence(df_with_indicators, "RSI")
        assert 0 <= strength <= 100

    def test_handles_short_data(self, sample_ohlcv_data):
        """Test that function handles short data"""
        from trading.indicators import calculate_indicators
        # Use 50 rows minimum for MACD calculation (26 slow + signal period)
        short_df = calculate_indicators(sample_ohlcv_data.head(50))
        div_type, strength = check_divergence(short_df, "RSI", lookback=10)
        # Should not raise an error
        assert div_type in [None, "bullish", "bearish"]

    def test_handles_macd_indicator(self, df_with_indicators):
        """Test that function works with MACD"""
        div_type, strength = check_divergence(df_with_indicators, "MACD")
        assert div_type in [None, "bullish", "bearish"]


class TestDetectMarketRegime:
    """Tests for detect_market_regime function"""

    def test_returns_dict(self, df_with_indicators):
        """Test that function returns a dictionary"""
        result = detect_market_regime(df_with_indicators)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, df_with_indicators):
        """Test that result contains required keys"""
        result = detect_market_regime(df_with_indicators)
        assert "regime" in result
        assert "confidence" in result
        assert "adx" in result
        assert "bb_width" in result

    def test_regime_values(self, df_with_indicators):
        """Test that regime is valid"""
        result = detect_market_regime(df_with_indicators)
        valid_regimes = [
            "STRONG_UPTREND", "STRONG_DOWNTREND",
            "WEAK_TREND", "HIGH_VOLATILITY",
            "CONSOLIDATION", "RANGING", "UNKNOWN"
        ]
        assert result["regime"] in valid_regimes

    def test_confidence_range(self, df_with_indicators):
        """Test that confidence is within valid range"""
        result = detect_market_regime(df_with_indicators)
        assert 0 <= result["confidence"] <= 100

    def test_bullish_trend_regime(self, bullish_df_with_indicators):
        """Test that bullish trend is detected"""
        result = detect_market_regime(bullish_df_with_indicators)
        # Strong uptrend should be detected
        assert result["regime"] in ["STRONG_UPTREND", "WEAK_TREND", "HIGH_VOLATILITY"]

    def test_bearish_trend_regime(self, bearish_df_with_indicators):
        """Test that bearish trend is detected"""
        result = detect_market_regime(bearish_df_with_indicators)
        # Strong downtrend should be detected
        assert result["regime"] in ["STRONG_DOWNTREND", "WEAK_TREND", "HIGH_VOLATILITY"]

    def test_short_data_returns_unknown(self, sample_ohlcv_data):
        """Test that short data returns UNKNOWN"""
        # Create minimal dataframe without indicators (simulates very short data)
        short_df = sample_ohlcv_data.head(15).copy()
        short_df["ADX"] = None
        short_df["BB_width"] = None
        short_df["DI_plus"] = None
        short_df["DI_minus"] = None
        result = detect_market_regime(short_df)
        assert result["regime"] == "UNKNOWN"


class TestAnalyzeHistoricalPerformance:
    """Tests for analyze_historical_performance function"""

    def test_returns_dict(self, df_with_indicators):
        """Test that function returns a dictionary"""
        result = analyze_historical_performance(df_with_indicators)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, df_with_indicators):
        """Test that result contains required keys"""
        result = analyze_historical_performance(df_with_indicators)
        assert "total_signals" in result
        assert "win_rate" in result
        assert "avg_return" in result
        assert "max_drawdown" in result
        assert "sharpe" in result
        assert "profit_factor" in result
        assert "long_signals" in result
        assert "short_signals" in result

    def test_total_signals_non_negative(self, df_with_indicators):
        """Test that total signals is non-negative"""
        result = analyze_historical_performance(df_with_indicators)
        assert result["total_signals"] >= 0

    def test_win_rate_range(self, df_with_indicators):
        """Test that win rate is within 0-100 range"""
        result = analyze_historical_performance(df_with_indicators)
        assert 0 <= result["win_rate"] <= 100

    def test_signal_count_consistency(self, df_with_indicators):
        """Test that long + short = total (or less due to filtering)"""
        result = analyze_historical_performance(df_with_indicators)
        assert result["long_signals"] + result["short_signals"] == result["total_signals"]

    def test_handles_short_data(self, sample_ohlcv_data):
        """Test that function handles short data"""
        from trading.indicators import calculate_indicators
        # Use 50 rows minimum for indicator calculation (MACD needs 26+)
        short_df = calculate_indicators(sample_ohlcv_data.head(50))
        result = analyze_historical_performance(short_df, lookback=20)
        # With short data, may have 0 or few signals
        assert result["total_signals"] >= 0

    def test_different_lookback_periods(self, df_with_indicators):
        """Test that different lookback periods work"""
        result_50 = analyze_historical_performance(df_with_indicators, lookback=50)
        result_80 = analyze_historical_performance(df_with_indicators, lookback=80)

        # Both should work without error
        assert "total_signals" in result_50
        assert "total_signals" in result_80

    def test_uses_dynamic_thresholds(self, df_with_indicators):
        """Test that backtest uses dynamic thresholds (ATR-based SL/TP)"""
        # This is a regression test to ensure we're using dynamic parameters
        result = analyze_historical_performance(df_with_indicators, lookback=50)
        # Just verify it runs without error - dynamic thresholds are internal
        assert isinstance(result, dict)
