"""
Pytest fixtures and mock data for trading bot tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n_periods = 100

    # Generate realistic price data with trend
    base_price = 50000
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_periods)),
        'high': prices * (1 + np.random.uniform(0.005, 0.03, n_periods)),
        'low': prices * (1 - np.random.uniform(0.005, 0.03, n_periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_periods) * 1e6
    })

    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def bullish_trend_data():
    """Create data with clear bullish trend"""
    np.random.seed(123)
    n_periods = 100

    # Strong uptrend
    base_price = 40000
    trend = np.linspace(0, 0.5, n_periods)  # 50% increase
    noise = np.random.normal(0, 0.01, n_periods)
    prices = base_price * (1 + trend + noise)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 - 0.005),
        'high': prices * (1 + np.random.uniform(0.01, 0.02, n_periods)),
        'low': prices * (1 - np.random.uniform(0.005, 0.015, n_periods)),
        'close': prices,
        'volume': np.random.uniform(2000, 8000, n_periods) * 1e6
    })

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def bearish_trend_data():
    """Create data with clear bearish trend"""
    np.random.seed(456)
    n_periods = 100

    # Strong downtrend
    base_price = 60000
    trend = np.linspace(0, -0.4, n_periods)  # 40% decrease
    noise = np.random.normal(0, 0.01, n_periods)
    prices = base_price * (1 + trend + noise)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + 0.005),
        'high': prices * (1 + np.random.uniform(0.005, 0.015, n_periods)),
        'low': prices * (1 - np.random.uniform(0.01, 0.02, n_periods)),
        'close': prices,
        'volume': np.random.uniform(2000, 8000, n_periods) * 1e6
    })

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sideways_data():
    """Create data with sideways/ranging market"""
    np.random.seed(789)
    n_periods = 100

    # Sideways with small oscillations
    base_price = 50000
    oscillation = np.sin(np.linspace(0, 8 * np.pi, n_periods)) * 0.05
    noise = np.random.normal(0, 0.005, n_periods)
    prices = base_price * (1 + oscillation + noise)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.003, 0.003, n_periods)),
        'high': prices * (1 + np.random.uniform(0.003, 0.01, n_periods)),
        'low': prices * (1 - np.random.uniform(0.003, 0.01, n_periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n_periods) * 1e6
    })

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def high_volatility_data():
    """Create data with high volatility"""
    np.random.seed(321)
    n_periods = 100

    base_price = 50000
    returns = np.random.normal(0, 0.05, n_periods)  # High volatility
    prices = base_price * np.cumprod(1 + returns)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.02, 0.02, n_periods)),
        'high': prices * (1 + np.random.uniform(0.02, 0.06, n_periods)),
        'low': prices * (1 - np.random.uniform(0.02, 0.06, n_periods)),
        'close': prices,
        'volume': np.random.uniform(3000, 15000, n_periods) * 1e6
    })

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def oversold_data():
    """Create data where RSI should be oversold"""
    np.random.seed(111)
    n_periods = 50

    # Sharp decline to create oversold conditions
    base_price = 60000
    decline = np.linspace(0, -0.35, n_periods)
    prices = base_price * (1 + decline)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 1.005,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(2000, 8000, n_periods) * 1e6
    })

    return df


@pytest.fixture
def overbought_data():
    """Create data where RSI should be overbought"""
    np.random.seed(222)
    n_periods = 50

    # Sharp rise to create overbought conditions
    base_price = 40000
    rise = np.linspace(0, 0.5, n_periods)
    prices = base_price * (1 + rise)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.995,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(2000, 8000, n_periods) * 1e6
    })

    return df


@pytest.fixture
def df_with_indicators(sample_ohlcv_data):
    """Sample data with indicators already calculated"""
    from trading.indicators import calculate_indicators
    return calculate_indicators(sample_ohlcv_data.copy(), timeframe="daily")


@pytest.fixture
def bullish_df_with_indicators(bullish_trend_data):
    """Bullish trend data with indicators"""
    from trading.indicators import calculate_indicators
    return calculate_indicators(bullish_trend_data.copy(), timeframe="daily")


@pytest.fixture
def bearish_df_with_indicators(bearish_trend_data):
    """Bearish trend data with indicators"""
    from trading.indicators import calculate_indicators
    return calculate_indicators(bearish_trend_data.copy(), timeframe="daily")
