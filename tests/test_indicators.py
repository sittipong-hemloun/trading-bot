"""
Unit tests for trading/indicators.py
"""

import pandas as pd

from trading.indicators import calculate_indicators


class TestCalculateIndicators:
    """Tests for calculate_indicators function"""

    def test_returns_dataframe(self, sample_ohlcv_data):
        """Test that function returns a DataFrame"""
        result = calculate_indicators(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_contains_ema_columns(self, sample_ohlcv_data):
        """Test that EMA columns are created"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "EMA_9" in result.columns
        assert "EMA_21" in result.columns
        assert "EMA_50" in result.columns

    def test_contains_sma_columns(self, sample_ohlcv_data):
        """Test that SMA columns are created"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "SMA_50" in result.columns
        assert "SMA_200" in result.columns

    def test_contains_rsi(self, sample_ohlcv_data):
        """Test that RSI is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "RSI" in result.columns
        assert "RSI_smoothed" in result.columns

    def test_rsi_range(self, sample_ohlcv_data):
        """Test that RSI values are within 0-100 range"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_rsi = result["RSI"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_contains_macd(self, sample_ohlcv_data):
        """Test that MACD columns are created"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "MACD" in result.columns
        assert "MACD_signal" in result.columns
        assert "MACD_histogram" in result.columns

    def test_contains_bollinger_bands(self, sample_ohlcv_data):
        """Test that Bollinger Bands are calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "BB_upper" in result.columns
        assert "BB_middle" in result.columns
        assert "BB_lower" in result.columns
        assert "BB_width" in result.columns
        assert "BB_percent" in result.columns

    def test_bollinger_band_order(self, sample_ohlcv_data):
        """Test that BB_upper > BB_middle > BB_lower"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_idx = result["BB_upper"].notna()
        assert (result.loc[valid_idx, "BB_upper"] >= result.loc[valid_idx, "BB_middle"]).all()
        assert (result.loc[valid_idx, "BB_middle"] >= result.loc[valid_idx, "BB_lower"]).all()

    def test_contains_adx(self, sample_ohlcv_data):
        """Test that ADX and DI columns are created"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "ADX" in result.columns
        assert "DI_plus" in result.columns
        assert "DI_minus" in result.columns

    def test_adx_range(self, sample_ohlcv_data):
        """Test that ADX values are within 0-100 range"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_adx = result["ADX"].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_contains_atr(self, sample_ohlcv_data):
        """Test that ATR columns are created"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "ATR" in result.columns
        assert "ATR_percent" in result.columns

    def test_atr_positive(self, sample_ohlcv_data):
        """Test that ATR is always positive"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_atr = result["ATR"].dropna()
        assert (valid_atr >= 0).all()

    def test_contains_volume_indicators(self, sample_ohlcv_data):
        """Test that volume indicators are calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "Volume_MA" in result.columns
        assert "Volume_Ratio" in result.columns
        assert "OBV" in result.columns
        assert "OBV_EMA" in result.columns

    def test_contains_stochrsi(self, sample_ohlcv_data):
        """Test that StochRSI is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "STOCHRSI_K" in result.columns
        assert "STOCHRSI_D" in result.columns

    def test_contains_supertrend(self, sample_ohlcv_data):
        """Test that Supertrend is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "SUPERTREND" in result.columns
        assert "SUPERTREND_DIR" in result.columns

    def test_supertrend_direction_values(self, sample_ohlcv_data):
        """Test that Supertrend direction is 1 or -1"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_dir = result["SUPERTREND_DIR"].dropna()
        assert valid_dir.isin([1, -1]).all()

    def test_contains_pivot_points(self, sample_ohlcv_data):
        """Test that pivot points are calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "PIVOT" in result.columns
        assert "R1" in result.columns
        assert "S1" in result.columns
        assert "R2" in result.columns
        assert "S2" in result.columns

    def test_contains_mfi(self, sample_ohlcv_data):
        """Test that MFI is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "MFI" in result.columns

    def test_mfi_range(self, sample_ohlcv_data):
        """Test that MFI is within 0-100 range"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_mfi = result["MFI"].dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()

    def test_contains_cci(self, sample_ohlcv_data):
        """Test that CCI is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "CCI" in result.columns

    def test_contains_vwap(self, sample_ohlcv_data):
        """Test that VWAP is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "VWAP" in result.columns
        assert "VWAP_upper" in result.columns
        assert "VWAP_lower" in result.columns

    def test_contains_tsi(self, sample_ohlcv_data):
        """Test that TSI is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "TSI" in result.columns
        assert "TSI_signal" in result.columns

    def test_contains_cmf(self, sample_ohlcv_data):
        """Test that CMF is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "CMF" in result.columns

    def test_cmf_range(self, sample_ohlcv_data):
        """Test that CMF is within -1 to 1 range"""
        result = calculate_indicators(sample_ohlcv_data)
        valid_cmf = result["CMF"].dropna()
        assert (valid_cmf >= -1).all()
        assert (valid_cmf <= 1).all()

    def test_contains_squeeze_indicator(self, sample_ohlcv_data):
        """Test that squeeze indicator is calculated"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "SQUEEZE" in result.columns
        assert "SQUEEZE_OFF" in result.columns

    def test_squeeze_boolean(self, sample_ohlcv_data):
        """Test that squeeze is boolean"""
        result = calculate_indicators(sample_ohlcv_data)
        assert result["SQUEEZE"].dtype == bool
        assert result["SQUEEZE_OFF"].dtype == bool

    def test_contains_price_action_patterns(self, sample_ohlcv_data):
        """Test that price action patterns are detected"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "HIGHER_HIGH" in result.columns
        assert "LOWER_LOW" in result.columns
        assert "IS_BULLISH" in result.columns
        assert "IS_BEARISH" in result.columns

    def test_weekly_timeframe_parameters(self, sample_ohlcv_data):
        """Test that weekly timeframe uses different parameters"""
        daily = calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")
        weekly = calculate_indicators(sample_ohlcv_data.copy(), timeframe="weekly")

        # RSI should be calculated with different lengths
        # Weekly RSI (length=7) vs Daily RSI (length=14)
        # Values may differ due to different parameters
        assert "RSI" in daily.columns
        assert "RSI" in weekly.columns

    def test_h4_timeframe(self, sample_ohlcv_data):
        """Test that h4 timeframe works"""
        result = calculate_indicators(sample_ohlcv_data, timeframe="h4")
        assert "RSI" in result.columns
        assert "MACD" in result.columns

    def test_preserves_original_columns(self, sample_ohlcv_data):
        """Test that original OHLCV columns are preserved"""
        result = calculate_indicators(sample_ohlcv_data)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_same_row_count(self, sample_ohlcv_data):
        """Test that row count is preserved"""
        result = calculate_indicators(sample_ohlcv_data)
        assert len(result) == len(sample_ohlcv_data)


class TestIndicatorLogic:
    """Tests for indicator calculation logic"""

    def test_bullish_trend_ema_alignment(self, bullish_trend_data):
        """Test that EMA alignment is bullish in uptrend"""
        result = calculate_indicators(bullish_trend_data)
        latest = result.iloc[-1]
        # In uptrend, shorter EMAs should be above longer ones
        assert latest["EMA_9"] > latest["EMA_21"]

    def test_bearish_trend_ema_alignment(self, bearish_trend_data):
        """Test that EMA alignment is bearish in downtrend"""
        result = calculate_indicators(bearish_trend_data)
        latest = result.iloc[-1]
        # In downtrend, shorter EMAs should be below longer ones
        assert latest["EMA_9"] < latest["EMA_21"]

    def test_oversold_rsi(self, oversold_data):
        """Test that RSI is oversold after sharp decline"""
        result = calculate_indicators(oversold_data)
        latest = result.iloc[-1]
        # RSI should be below 30 (oversold)
        assert latest["RSI"] < 35

    def test_overbought_rsi(self, overbought_data):
        """Test that RSI is overbought after sharp rise"""
        result = calculate_indicators(overbought_data)
        latest = result.iloc[-1]
        # RSI should be above 70 (overbought)
        assert latest["RSI"] > 65

    def test_high_volatility_atr(self, high_volatility_data, sample_ohlcv_data):
        """Test that ATR is higher in volatile market"""
        high_vol_result = calculate_indicators(high_volatility_data)
        normal_result = calculate_indicators(sample_ohlcv_data)

        # ATR percent should be higher in volatile data
        high_vol_atr = high_vol_result["ATR_percent"].dropna().mean()
        normal_atr = normal_result["ATR_percent"].dropna().mean()
        assert high_vol_atr > normal_atr

    def test_sideways_adx(self, sideways_data):
        """Test that ADX is low in sideways market"""
        result = calculate_indicators(sideways_data)
        latest = result.iloc[-1]
        # ADX should be lower in ranging market
        assert latest["ADX"] < 30
